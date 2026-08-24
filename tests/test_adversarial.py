"""The inputs that actually catch bugs.

Random N(0,1) tensors pass on almost anything. These do not: they are the cases
where a wrong max-subtraction, a wrong boundary mask, or a masked row handled as
zero instead of -inf shows up as NaN or as a silently wrong number.

Each case asserts a *property* that holds for any correct implementation (the
output is exactly V's row, or the mean of V, or finite), not just agreement with a
reference -- a property assertion still means something when the reference and the
candidate share a bug.

`kernel` is xfail (task 03, and no CUDA device here); `naive` and `chunked` run.
"""

from __future__ import annotations

import math

import pytest
import torch
from conftest import (
    KERNEL_PARAM,
    TOLERANCES,
    assert_no_worse_than_naive,
    make_qkv,
    naive_attention,
    resolve_impl,
)

from fa.ref.fp64 import attention_fp64

IMPLS = ["naive", "chunked", KERNEL_PARAM]
DTYPES = [torch.float16, torch.bfloat16]
DTYPE_IDS = ["fp16", "bf16"]


def assert_close_to(out: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
    """Absolute check against a known-exact answer, at the measured tolerance."""
    assert torch.isfinite(out.double()).all(), "output has NaN or Inf"
    err = (out.double() - expected.double()).abs().max().item()
    assert err <= TOLERANCES[dtype], f"max abs error {err:.3e} > {TOLERANCES[dtype]:.1e}"


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("impl", IMPLS)
def test_one_hot_scores(impl, dtype):
    """One score at +100, every other at -100: the overflow probe.

    exp(100) is inf in fp32 and long past inf in fp16, so any implementation that
    forgets to subtract the row max returns NaN here. The correct answer is V's
    row 3, because exp(-200) underflows to exactly 0 next to it.
    """
    fn = resolve_impl(impl)
    b, h, n, d = 1, 2, 16, 16
    hot = 3
    sm_scale = 1.0 / math.sqrt(d)
    q = torch.zeros(b, h, n, d, dtype=dtype)
    k = torch.zeros(b, h, n, d, dtype=dtype)
    q[..., 0] = 1.0
    k[..., 0] = -100.0 / sm_scale
    k[:, :, hot, 0] = 100.0 / sm_scale
    _, _, v = make_qkv(b, h, n, d, dtype)

    out = fn(q, k, v, causal=False)
    expected = v[:, :, hot, :].unsqueeze(-2).expand_as(out)
    assert_close_to(out, expected, dtype)


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("impl", IMPLS)
def test_uniform_scores(impl, dtype, causal):
    """All scores identical, so l_i is exactly N and the output is the mean of V.

    Q = K = 0 makes every score 0. Non-causal: the mean over all of V. Causal: the
    running mean, row i being the mean of V[0..i] -- which is also the cheapest
    off-by-one detector in the suite, since row i of a kernel with the diagonal
    wrong averages i or i+2 rows instead of i+1.
    """
    fn = resolve_impl(impl)
    b, h, n, d = 1, 2, 64, 32
    q = torch.zeros(b, h, n, d, dtype=dtype)
    k = torch.zeros(b, h, n, d, dtype=dtype)
    _, _, v = make_qkv(b, h, n, d, dtype)

    out = fn(q, k, v, causal=causal)
    vd = v.double()
    if causal:
        counts = torch.arange(1, n + 1, dtype=torch.float64).view(1, 1, n, 1)
        expected = vd.cumsum(dim=-2) / counts
    else:
        expected = vd.mean(dim=-2, keepdim=True).expand_as(vd)
    assert_close_to(out, expected, dtype)


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("impl", IMPLS)
def test_single_token(impl, dtype, causal):
    """N=1: softmax over one element is exactly 1.0, so the output is V, exactly."""
    fn = resolve_impl(impl)
    q, k, v = make_qkv(1, 2, 1, 32, dtype)
    out = fn(q, k, v, causal=causal)
    assert torch.equal(out, v), f"N=1 must return V unchanged, max diff {(out - v).abs().max()}"


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("impl", IMPLS)
def test_head_dim_one(impl, dtype, causal):
    """D=1. Degenerate for any tiling scheme whose tile is 16 wide in the D axis."""
    fn = resolve_impl(impl)
    q, k, v = make_qkv(1, 2, 64, 1, dtype)
    out = fn(q, k, v, causal=causal)
    assert_no_worse_than_naive(
        out, attention_fp64(q, k, v, causal=causal), naive_attention(q, k, v, causal=causal)
    )


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("impl", IMPLS)
def test_large_magnitude_inputs(impl, dtype, causal):
    """Q, K, V ~ N(0, 100), i.e. std 10, which puts scores in the hundreds.

    exp() of a raw score overflows long before this; only the running-max tracking
    keeps it finite. Kept at D=16 so the fp16 score matrix itself stays inside
    65504 -- this is a test of max tracking, not of matmul overflow.
    """
    fn = resolve_impl(impl)
    q, k, v = make_qkv(1, 2, 128, 16, dtype, std=10.0)
    ref = attention_fp64(q, k, v, causal=causal)
    assert ref.abs().max() > 1.0, "test inputs are not actually large"
    out = fn(q, k, v, causal=causal)
    assert torch.isfinite(out.double()).all(), "large scores produced NaN or Inf"
    assert_no_worse_than_naive(out, ref, naive_attention(q, k, v, causal=causal))


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("impl", IMPLS)
def test_causal_row_zero_is_identity_on_v(impl, dtype):
    """Under causal masking, row 0 sees exactly one key: itself.

    Every other column is -inf. Handle that with a zero instead of a -inf and the
    row max is wrong; handle the row as "fully masked" and you get 0/0 = NaN. The
    correct answer is V's row 0, bit for bit, because softmax over one element is
    exactly 1.0.
    """
    fn = resolve_impl(impl)
    q, k, v = make_qkv(1, 2, 128, 32, dtype)
    out = fn(q, k, v, causal=True)
    assert torch.isfinite(out.double()).all(), "causal attention produced NaN or Inf"
    assert torch.equal(out[:, :, 0, :], v[:, :, 0, :]), (
        f"row 0 must be V's row 0 exactly, max diff "
        f"{(out[:, :, 0, :] - v[:, :, 0, :]).abs().max()}"
    )


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("impl", IMPLS)
def test_non_contiguous_inputs(impl, dtype, causal):
    """Transposed views. Either handle them or reject them loudly -- never guess.

    A kernel that takes raw pointers and assumes contiguity reads the right bytes
    in the wrong order and returns plausible garbage. Both outcomes below are a
    pass; silently wrong output is not.
    """
    fn = resolve_impl(impl)
    b, h, n, d = 1, 2, 64, 32
    # (B, N, H, D) allocation viewed as (B, H, N, D): correct shape, wrong strides.
    q, k, v = (t.transpose(1, 2) for t in make_qkv(b, n, h, d, dtype))
    assert not q.is_contiguous()

    try:
        out = fn(q, k, v, causal=causal)
    except (RuntimeError, ValueError, AssertionError, NotImplementedError, TypeError) as exc:
        assert str(exc).strip(), f"{impl} rejected non-contiguous input with an empty message"
        return
    ref = attention_fp64(q, k, v, causal=causal)
    naive = naive_attention(q.contiguous(), k.contiguous(), v.contiguous(), causal=causal)
    assert_no_worse_than_naive(out, ref, naive)


@pytest.mark.parametrize("n", [997, 1009], ids=lambda n: f"N{n}")
@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("impl", IMPLS)
def test_prime_sequence_length(impl, dtype, causal, n):
    """N prime: no tile size divides it, so every boundary mask is exercised.

    If a candidate is right at N=1024 and wrong here, it is the boundary mask.
    Every time.
    """
    fn = resolve_impl(impl)
    q, k, v = make_qkv(1, 1, n, 64, dtype)
    out = fn(q, k, v, causal=causal)
    assert out.shape == q.shape
    assert_no_worse_than_naive(
        out, attention_fp64(q, k, v, causal=causal), naive_attention(q, k, v, causal=causal)
    )
