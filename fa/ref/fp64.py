"""Ground truth: attention in float64, written to be read, not to be fast.

Everything else in this repo is measured against this file. It therefore does the
dumbest possible thing at every step: materialize the full N x N score matrix,
subtract the row max by hand, exponentiate, normalize, multiply by V. No fusion,
no tiling, no online softmax, no `F.scaled_dot_product_attention` (that *is*
FlashAttention -- using it as ground truth would hide any bug the kernel and it
happen to share).

Shapes are (B, H, N, D) throughout, matching the Triton kernel's layout.

Memory: the score matrix is B*H*N*N*8 bytes. At B=4, H=32, N=4096 that is 17 GB,
so callers who want a big-N reference should use B=H=1 and loop.
"""

from __future__ import annotations

import math

import torch

__all__ = ["attention_fp64", "causal_mask"]


def causal_mask(n_q: int, n_k: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """True where a query row may NOT attend, for the bottom-right aligned causal mask.

    With n_q == n_k this is the usual "query i sees keys 0..i". When n_q < n_k the
    query block is aligned to the *end* of the key sequence, which is the convention
    every decoding path in this repo uses.
    """
    q_idx = torch.arange(n_k - n_q, n_k, device=device).unsqueeze(1)
    k_idx = torch.arange(n_k, device=device).unsqueeze(0)
    return k_idx > q_idx


def attention_fp64(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """softmax(QK^T * sm_scale) @ V, entirely in float64.

    Args:
        q, k, v: (B, H, N, D) -- any dtype; each is upcast to float64 first.
        causal:  mask out keys that come after the query position.
        sm_scale: defaults to 1/sqrt(D).

    Returns:
        (B, H, N_q, D) float64.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"expected (B, H, N, D) tensors, got {q.shape}, {k.shape}, {v.shape}")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"head dim mismatch: q {q.shape[-1]} vs k {k.shape[-1]}")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError(f"key/value length mismatch: k {k.shape[-2]} vs v {v.shape[-2]}")

    qf, kf, vf = q.double(), k.double(), v.double()
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    # S = QK^T / sqrt(d), full N x N, deliberately materialized.
    s = torch.matmul(qf, kf.transpose(-2, -1)) * sm_scale

    if causal:
        mask = causal_mask(q.shape[-2], k.shape[-2], device=s.device)
        s = s.masked_fill(mask, float("-inf"))

    # Safe softmax, spelled out. torch.softmax would do the same thing; writing it
    # by hand is the point of a reference -- the max subtraction is the one step
    # every fast implementation has to reproduce and is where kernels go wrong.
    m = s.max(dim=-1, keepdim=True).values
    m = torch.nan_to_num(m, neginf=0.0)  # a fully masked row: all -inf, shift by 0
    p = torch.exp(s - m)
    l = p.sum(dim=-1, keepdim=True)  # noqa: E741 -- l_i is the name in the paper
    p = p / torch.where(l == 0, torch.ones_like(l), l)

    return torch.matmul(p, vf)


if __name__ == "__main__":
    # Self-check: uniform scores must give the mean of V, and a single unmasked key
    # must give that key's V row exactly.
    torch.manual_seed(0)
    v = torch.randn(1, 2, 8, 4, dtype=torch.float64)
    zeros = torch.zeros(1, 2, 8, 4, dtype=torch.float64)
    out = attention_fp64(zeros, zeros, v)
    assert torch.allclose(out, v.mean(dim=-2, keepdim=True).expand_as(out)), "uniform case"

    q = torch.randn(1, 2, 8, 4, dtype=torch.float64)
    k = torch.randn(1, 2, 8, 4, dtype=torch.float64)
    out = attention_fp64(q, k, v, causal=True)
    assert torch.equal(out[:, :, 0, :], v[:, :, 0, :]), "causal row 0 is V row 0"
    print("fa/ref/fp64.py self-check OK")
