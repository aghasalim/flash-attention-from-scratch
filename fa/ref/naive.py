"""Unfused reference attentions: the strawmen the fused kernel has to beat.

Three implementations, deliberately *not* fused:

* :func:`naive_attention`  -- materialises the whole ``S = QK^T / sqrt(d)``.
* :func:`chunked_attention` -- tiles over K/V, still materialises each tile's scores.
* :func:`sdpa_attention`   -- ``F.scaled_dot_product_attention`` with a forced backend.

``naive`` vs ``chunked`` isolates *tiling* (a memory fix) from *fusion* (a bandwidth
fix). They do the same arithmetic in the same order-of-accumulation sense and move
the same score bytes through memory; only the peak residency differs. The gap
between chunked and a fused kernel is exactly what FlashAttention buys.

Shapes are ``(B, H, N, D)``, contiguous, matching what the Triton kernel will take.

Numerics (repo rule 5): inputs may be fp16/bf16, but every accumulator -- the softmax
running max ``m_i``, the running denominator ``l_i`` and the output accumulator -- is
fp32. The matmuls run in the input dtype, which is what makes ``naive_attention`` a
fair "naive fp16" error baseline for the relative correctness bar.

Backend honesty: on some devices ``torch.nn.attention.sdpa_kernel`` is not honoured at
all (MPS in torch 2.13 runs even with an *empty* backend list). :func:`sdpa_report`
probes that per device so a benchmark never labels an unknown kernel "FLASH".
Run ``python -m fa.ref.naive`` to print the probe for this machine.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

__all__ = [
    "naive_attention",
    "chunked_attention",
    "sdpa_attention",
    "sdpa_report",
    "SdpaUnavailable",
    "BACKEND_NAMES",
]

BACKEND_NAMES = ("MATH", "EFFICIENT_ATTENTION", "FLASH_ATTENTION", "CUDNN_ATTENTION")


class SdpaUnavailable(RuntimeError):
    """Raised instead of silently falling through to whatever backend torch picks."""


def _causal_mask(n_q: int, n_k: int, device: torch.device, col_offset: int = 0) -> torch.Tensor:
    """True where a query may *not* attend (strictly upper triangular, aligned right).

    ``col_offset`` is the absolute column index of the first key in this tile, so the
    same helper works for the full matrix and for a K/V chunk.
    """
    q_idx = torch.arange(n_q, device=device).unsqueeze(1)
    k_idx = torch.arange(col_offset, col_offset + n_k, device=device).unsqueeze(0)
    return k_idx > q_idx


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """Materialise the full ``B*H*N*N`` score matrix. The strawman.

    Peak memory is O(N^2) per head and every score is written to memory and read back
    twice (once by the softmax, once by the PV matmul). That round trip is the whole
    problem.
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = q @ k.transpose(-2, -1)  # (B, H, N, N) in the input dtype -- the whole problem
    s.mul_(scale)  # in place: a strawman that also copies S would be unfair to itself
    if causal:
        s.masked_fill_(_causal_mask(q.shape[-2], k.shape[-2], q.device), float("-inf"))
    # dtype=float32 makes the softmax max/sum accumulate in fp32 (rule 5) whatever S is.
    p = torch.softmax(s, dim=-1, dtype=torch.float32)
    return p.to(q.dtype) @ v


def chunked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk: int = 1024,
    causal: bool = False,
) -> torch.Tensor:
    """Tile over K/V in a Python loop, materialising one ``N x chunk`` tile at a time.

    Peak score residency drops from ``B*H*N^2`` to ``B*H*N*chunk`` -- the memory problem
    is fixed. The bandwidth problem is not: every tile is still a real tensor that gets
    written to memory and read back by the softmax and the PV matmul, and Q is re-read
    once per tile. See ``notes/00-roofline.md``.

    The running-max rescaling below is the standard numerically-safe way to combine
    tiles. ``fa/ref/online_softmax.py`` (task 02) is where that algorithm is studied;
    it is reproduced here only so this file stands alone.
    """
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    b, h, n_q, d = q.shape
    n_k = k.shape[-2]
    scale = 1.0 / math.sqrt(d)

    # rule 5: every accumulator is fp32 regardless of the input dtype.
    acc = torch.zeros((b, h, n_q, d), device=q.device, dtype=torch.float32)
    m_i = torch.full((b, h, n_q, 1), float("-inf"), device=q.device, dtype=torch.float32)
    l_i = torch.zeros((b, h, n_q, 1), device=q.device, dtype=torch.float32)

    for start in range(0, n_k, chunk):
        stop = min(start + chunk, n_k)
        s = q @ k[:, :, start:stop, :].transpose(-2, -1)  # (B, H, N, chunk)
        s.mul_(scale)
        if causal:
            s.masked_fill_(_causal_mask(n_q, stop - start, q.device, start), float("-inf"))
        s = s.float()
        # The first tile starts at column 0, so under a causal mask every row has at
        # least one unmasked entry there and m_i is finite from the first iteration on.
        # Without that, exp(-inf - (-inf)) would be NaN.
        m_new = torch.maximum(m_i, s.amax(dim=-1, keepdim=True))
        correction = torch.exp(m_i - m_new)
        p = s.sub_(m_new).exp_()  # in place on the fp32 tile
        l_i = l_i * correction + p.sum(dim=-1, keepdim=True)
        acc = acc * correction + (p.to(q.dtype) @ v[:, :, start:stop, :]).float()
        m_i = m_new

    return (acc / l_i).to(q.dtype)


@dataclass(frozen=True)
class _Report:
    device: str
    honored: bool
    reason: str
    backends: dict = field(default_factory=dict)

    @property
    def usable(self) -> tuple[str, ...]:
        return tuple(n for n, info in self.backends.items() if info["ok"])

    def as_dict(self) -> dict:
        return {
            "device": self.device,
            "selection_honored": self.honored,
            "honored_reason": self.reason,
            "backends": dict(self.backends),
        }


@functools.lru_cache(maxsize=None)
def sdpa_report(device: str, dtype: torch.dtype = torch.float16) -> _Report:
    """Probe which SDPA backends actually run here, and whether the choice is honoured.

    The honesty check is ``sdpa_kernel([])``: with *no* backend enabled, sdpa must raise.
    If it runs anyway, the context manager is not gating anything on this device and any
    row labelled "FLASH" would be a fabrication.
    """
    dev = torch.device(device)
    q = torch.randn(1, 2, 64, 64, device=dev, dtype=dtype)

    try:
        with sdpa_kernel([]):
            F.scaled_dot_product_attention(q, q, q)
        honored, reason = False, "sdpa ran with an empty backend list -- sdpa_kernel is a no-op on this device"
    except RuntimeError as exc:
        honored, reason = True, f"empty backend list correctly raised: {type(exc).__name__}"

    backends = {}
    for name in BACKEND_NAMES:
        try:
            with sdpa_kernel(getattr(SDPBackend, name)):
                F.scaled_dot_product_attention(q, q, q)
            backends[name] = {"ok": True, "error": None}
        except Exception as exc:  # noqa: BLE001 - any failure means "not usable here"
            backends[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return _Report(device=str(dev.type), honored=honored, reason=reason, backends=backends)


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    backend: str = "MATH",
    require_honored: bool = True,
) -> torch.Tensor:
    """``F.scaled_dot_product_attention`` with ``backend`` forced -- or a loud failure.

    ``is_causal=True`` is used for the causal case on purpose: an additive mask disables
    the fused path and would produce a misleadingly slow "flash" number.

    Raises :class:`SdpaUnavailable` if the backend does not run on this device, or if
    the device ignores backend selection entirely (pass ``require_honored=False`` to run
    anyway -- then the result is "whatever kernel this device picked", not ``backend``).
    """
    if backend not in BACKEND_NAMES:
        raise ValueError(f"unknown backend {backend!r}, expected one of {BACKEND_NAMES}")
    report = sdpa_report(str(q.device.type), q.dtype)
    info = report.backends[backend]
    if not info["ok"]:
        raise SdpaUnavailable(f"SDPA backend {backend} does not run on {report.device}: {info['error']}")
    if require_honored and not report.honored:
        raise SdpaUnavailable(
            f"SDPA backend selection is not honoured on {report.device} ({report.reason}); "
            f"a result labelled {backend} would be a guess"
        )
    with sdpa_kernel(getattr(SDPBackend, backend)):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


if __name__ == "__main__":
    for dev, dt in (("cpu", torch.float32), ("mps", torch.float16)):
        if dev == "mps" and not torch.backends.mps.is_available():
            print(f"{dev}: unavailable")
            continue
        r = sdpa_report(dev, dt)
        print(f"{dev} ({dt}): selection honored = {r.honored} -- {r.reason}")
        for name, info in r.backends.items():
            print(f"    {name:<20} {'runs' if info['ok'] else 'UNAVAILABLE: ' + info['error'][:90]}")
