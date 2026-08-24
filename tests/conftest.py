"""Shared fixtures, tolerances, and the relative correctness bar.

Read this file first. The single most important thing in it is
`assert_no_worse_than_naive`: every correctness claim in this repo is *relative* to
naive attention in the same input dtype, both measured against the float64
reference in `fa/ref/fp64.py`. An absolute tolerance on fp16 attention output is
either loose enough to pass a broken kernel or tight enough to fail a correct one,
because the error grows with N and with score magnitude.

Everything runs on CPU. There is no CUDA device on the machine this was developed
on (Apple M4), and CPU is deterministic, which is what a correctness harness wants.

`python tests/conftest.py` re-measures the naive-vs-fp64 error table that TOLERANCES
below is derived from.
"""

from __future__ import annotations

import functools
import importlib
import math
from collections.abc import Callable

import pytest
import torch

SEED = 0
DEVICE = torch.device("cpu")

# Absolute tolerances, measured -- not guessed. Source: the worst |naive - fp64|
# over the sweep printed by `python tests/conftest.py` (B=1, H=2, D in
# {16,32,64,128}, N in {128,512,1000,2048,4096}, causal in {F,T}, N(0,1) inputs,
# naive = fa.ref.naive.naive_attention), run 2026-08-24:
#
#     fp16 2.117e-03    bf16 1.259e-02    fp32 2.300e-06
#
# The entries below are that worst case x2, rounded up to one significant figure,
# so an unluckier seed does not turn into a tolerance edit. bf16 gets its own
# entry because it has 8 mantissa bits to fp16's 11; the measured ratio between
# the two on this sweep is 5.9x, and a single shared number would be wrong in one
# direction or the other.
#
# These are a *secondary* guard, used by the invariant tests, where both sides of
# the identity come from the same implementation and there is no naive baseline to
# be relative to. The primary bar is relative: assert_no_worse_than_naive.
TOLERANCES: dict[torch.dtype, float | None] = {
    torch.float16: 5e-3,
    torch.bfloat16: 3e-2,
    torch.float32: 5e-6,
}

# How much worse than naive a candidate may be before the test fails. 2.0 is the
# number in the task spec; it is a factor, not a fudge -- if a kernel needs 3x it
# is wrong, and raising this constant to make a test pass is rule 1 in AGENTS.md.
BAR_FACTOR = 2.0


# --------------------------------------------------------------------------------
# pytest plumbing
# --------------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="run the slow sweeps (N >= 4096); the default run stays under 60s",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: needs --slow (large N; minutes, not seconds)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--slow"):
        return
    skip = pytest.mark.skip(reason="needs --slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def seed() -> int:
    """Seed every RNG that could touch a test, and hand back the seed."""
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    return SEED


@pytest.fixture
def skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


@pytest.fixture
def skip_if_no_mps() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS device")


# --------------------------------------------------------------------------------
# implementations under test
# --------------------------------------------------------------------------------
#
# Tasks 01, 02 and 03 own their own files and may not have landed yet (01 and 02 run
# in parallel with this one; 03 is a whole wave later, and on an Apple machine it can
# never run at all -- Triton has no macOS wheel and there is no CUDA device). Every
# import below is therefore soft: the suite degrades to "skipped", never to a
# collection error.


def _optional(module: str, name: str) -> Callable | None:
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        return None


def external_naive() -> Callable | None:
    """`fa.ref.naive.naive_attention` (task 01), or None if it has not landed."""
    return _optional("fa.ref.naive", "naive_attention")


def external_chunked() -> Callable | None:
    """`fa.ref.naive.chunked_attention` (task 01), or None."""
    return _optional("fa.ref.naive", "chunked_attention")


def external_online() -> Callable | None:
    """`fa.ref.online_softmax.online_attention` (task 02, NumPy), or None."""
    return _optional("fa.ref.online_softmax", "online_attention")


def kernel_attention() -> Callable:
    """`fa.ops.attention.attention` (task 03). Raises ImportError until it exists."""
    from fa.ops.attention import attention

    return attention


def local_naive(q, k, v, causal=False, sm_scale=None):
    """Textbook attention in the *input* dtype: the strawman the bar is set by.

    Deliberately unfused and deliberately low precision -- scores, softmax and PV
    all in q.dtype. This is the thing a kernel has to be no worse than. It is a
    fallback for `fa/ref/naive.py` (task 01) so that this harness can be run and
    the bar measured before that file exists; when it exists it is preferred.
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    s = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    if causal:
        from fa.ref.fp64 import causal_mask

        s = s.masked_fill(causal_mask(q.shape[-2], k.shape[-2], s.device), float("-inf"))
    return torch.matmul(torch.softmax(s, dim=-1), v)


def naive_attention(q, k, v, causal=False, sm_scale=None):
    """The bar. Prefers task 01's implementation, falls back to `local_naive`."""
    ext = external_naive()
    if ext is not None and sm_scale is None:
        return ext(q, k, v, causal=causal)
    return local_naive(q, k, v, causal=causal, sm_scale=sm_scale)


def naive_is_external() -> bool:
    return external_naive() is not None


# --------------------------------------------------------------------------------
# inputs and the bar
# --------------------------------------------------------------------------------


def make_qkv(
    b: int,
    h: int,
    n: int,
    d: int,
    dtype: torch.dtype,
    std: float = 1.0,
    seed: int = SEED,
    n_kv: int | None = None,
):
    """Contiguous (B, H, N, D) inputs on CPU, same values for the same arguments."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_kv = n if n_kv is None else n_kv
    q = torch.randn(b, h, n, d, generator=g, dtype=torch.float32) * std
    k = torch.randn(b, h, n_kv, d, generator=g, dtype=torch.float32) * std
    v = torch.randn(b, h, n_kv, d, generator=g, dtype=torch.float32) * std
    return q.to(dtype), k.to(dtype), v.to(dtype)


@functools.lru_cache(maxsize=64)
def reference_bundle(b, h, n, d, dtype, causal, std=1.0, seed=SEED):
    """(q, k, v, out_fp64, out_naive) -- cached, because fp64 and fp16 CPU attention
    are the expensive part of this suite and every candidate reuses the same pair."""
    from fa.ref.fp64 import attention_fp64

    q, k, v = make_qkv(b, h, n, d, dtype, std=std, seed=seed)
    ref = attention_fp64(q, k, v, causal=causal)
    nai = naive_attention(q, k, v, causal=causal)
    return q, k, v, ref, nai


def max_abs_err(out: torch.Tensor, ref_fp64: torch.Tensor) -> float:
    return (out.double() - ref_fp64).abs().max().item()


def assert_no_worse_than_naive(out, ref_fp64, naive_fp16, factor: float = BAR_FACTOR) -> None:
    """The bar, exactly as the spec states it.

    err_kernel <= factor * err_naive, both against the same float64 reference.
    Do not loosen `factor` to make a test pass -- if the candidate is more than 2x
    worse than the strawman it is wrong (AGENTS.md rule 1).
    """
    err_kernel = (out.double() - ref_fp64).abs().max()
    err_naive = (naive_fp16.double() - ref_fp64).abs().max()
    assert torch.isfinite(out.double()).all(), "candidate produced NaN or Inf"
    # err_naive can be exactly 0 for degenerate cases (N=1, uniform scores), where
    # the strawman is exact. Then the candidate has to be exact too, up to one ulp
    # of the input dtype, which is what the epsilon covers.
    eps = torch.finfo(torch.float16).eps * float(ref_fp64.abs().max().clamp(min=1.0))
    assert err_kernel <= factor * err_naive + eps, (
        f"candidate error {err_kernel:.3e} > {factor} x naive error {err_naive:.3e} "
        f"(both vs the fp64 reference; +{eps:.1e} slack for the exact-naive case)"
    )


# --------------------------------------------------------------------------------
# the candidate registry
# --------------------------------------------------------------------------------
#
# Every test takes an `impl` name and calls `resolve_impl(name)`, which hands back a
# uniform `(q, k, v, causal=False, sm_scale=None) -> Tensor` callable:
#
#   fp64     fa.ref.fp64.attention_fp64          -- the ground truth, always present
#   naive    fa.ref.naive.naive_attention        -- task 01; local_naive until it lands
#   local    tests' own textbook attention       -- always present, takes sm_scale
#   chunked  fa.ref.naive.chunked_attention      -- task 01, tiled but unfused
#   kernel   fa.ops.attention.attention          -- task 03, the Triton kernel
#
# `kernel` raises ImportError until task 03 lands, which is why every parameter that
# uses it carries an xfail mark (see KERNEL_PARAM). Missing task-01 files skip.

KERNEL_XFAIL_REASON = (
    "task 03: fa/ops/attention.py does not exist yet -- and on this machine it can "
    "never run (no CUDA device, no Triton wheel for macOS; developed on Apple M4)"
)
KERNEL_PARAM = pytest.param("kernel", marks=pytest.mark.xfail(reason=KERNEL_XFAIL_REASON))

# chunked_attention's tile size. 256 does not divide 1000, 129, 7 or 997, which is
# deliberate: ragged final tiles are where tiled attention gets masking wrong.
CHUNK = 256


def resolve_impl(name: str) -> Callable:
    """Name -> `(q, k, v, causal=False, sm_scale=None)` callable. Skips if absent."""
    if name == "fp64":
        from fa.ref.fp64 import attention_fp64

        return attention_fp64
    if name == "local":
        return local_naive
    if name == "naive":
        return _adapt(naive_attention, name)
    if name == "chunked":
        fn = external_chunked()
        if fn is None:
            pytest.skip("fa/ref/naive.py::chunked_attention not present yet (task 01)")
        return _adapt(functools.partial(fn, chunk=CHUNK), name)
    if name == "kernel":
        return kernel_attention()  # ImportError here is what the xfail mark expects
    raise ValueError(f"unknown implementation {name!r}")


def _adapt(fn: Callable, name: str) -> Callable:
    """Give an implementation that has no sm_scale argument the common signature."""

    def call(q, k, v, causal=False, sm_scale=None):
        if sm_scale is not None:
            pytest.skip(f"{name} takes no sm_scale argument")
        return fn(q, k, v, causal=causal)

    return call


if __name__ == "__main__":
    # Where TOLERANCES comes from. Re-run with: python tests/conftest.py
    from fa.ref.fp64 import attention_fp64

    print(f"naive ({'fa.ref.naive' if naive_is_external() else 'local_naive'}) vs fp64, B=1 H=2")
    print(f"{'dtype':>9} {'N':>6} {'D':>5} {'causal':>7} {'max abs err':>12}")
    worst: dict[torch.dtype, float] = {}
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for n in (128, 512, 1000, 2048, 4096):
            for d in (16, 32, 64, 128):
                for causal in (False, True):
                    q, k, v = make_qkv(1, 2, n, d, dtype)
                    err = max_abs_err(
                        naive_attention(q, k, v, causal=causal),
                        attention_fp64(q, k, v, causal=causal),
                    )
                    worst[dtype] = max(worst.get(dtype, 0.0), err)
                    print(f"{str(dtype):>9} {n:>6} {d:>5} {str(causal):>7} {err:>12.3e}")
    for dtype, err in worst.items():
        print(f"worst {dtype}: {err:.3e}")
