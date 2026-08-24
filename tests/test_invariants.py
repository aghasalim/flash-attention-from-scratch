"""Properties that hold for *any* correct attention, reference or kernel.

These are the tests that make the harness trustworthy before a kernel exists: they
need no ground truth at all, only an implementation and an identity it must
satisfy. They run today against `fa/ref/fp64.py` and task 01's references, which
is what proves the harness itself is right rather than merely self-consistent.

Four identities:

* permutation -- jointly permuting K and V along N leaves non-causal output alone
* scale       -- attention(cQ, K, V, s/c) == attention(Q, K, V, s)
* shift       -- adding a constant to every score changes nothing
* containment -- output at row i does not depend on K/V beyond column i

The last one is the strong causal test. It compares an implementation against
itself, so no reference can hide an off-by-one in the diagonal from it: if row i
moves when K[i+1:] changes, the mask is wrong, full stop.
"""

from __future__ import annotations

import math

import pytest
import torch
from conftest import KERNEL_PARAM, TOLERANCES, make_qkv, resolve_impl

IMPLS = ["fp64", "naive", "chunked", KERNEL_PARAM]
# Identities that need an explicit sm_scale can only be posed to implementations
# that take one. `local` is the harness's own textbook attention (see conftest).
SCALED_IMPLS = ["fp64", "local", KERNEL_PARAM]
DTYPES = [torch.float16, torch.bfloat16]
DTYPE_IDS = ["fp16", "bf16"]


def tol(impl: str, dtype: torch.dtype) -> float:
    """fp64 arithmetic is held to fp64 standards; everything else to the measured bar."""
    return 1e-12 if impl == "fp64" else TOLERANCES[dtype]


def assert_same(a: torch.Tensor, b: torch.Tensor, atol: float, what: str) -> None:
    err = (a.double() - b.double()).abs().max().item()
    assert err <= atol, f"{what}: max abs difference {err:.3e} > {atol:.1e}"


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("impl", IMPLS)
def test_permutation_invariance(impl, dtype):
    """Non-causal attention is a weighted sum over keys, and sums do not care about order."""
    fn = resolve_impl(impl)
    b, h, n, d = 1, 2, 256, 64
    q, k, v = make_qkv(b, h, n, d, dtype)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(1))

    out = fn(q, k, v, causal=False)
    out_perm = fn(q, k[:, :, perm, :], v[:, :, perm, :], causal=False)
    assert_same(out, out_perm, tol(impl, dtype), "permuting K and V changed the output")


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("impl", SCALED_IMPLS)
def test_scale_invariance(impl, dtype, causal):
    """Only the product sm_scale * (Q . K) matters, so c and 1/c must cancel.

    c = 2 is a power of two on purpose: doubling a float and halving the scale are
    both exact, so a correct implementation returns bit-identical output and any
    difference here is a real bug rather than rounding.
    """
    fn = resolve_impl(impl)
    b, h, n, d = 1, 2, 128, 64
    q, k, v = make_qkv(b, h, n, d, dtype)
    sm_scale = 1.0 / math.sqrt(d)
    c = 2.0

    out = fn(q, k, v, causal=causal, sm_scale=sm_scale)
    out_scaled = fn((q * c).to(dtype), k, v, causal=causal, sm_scale=sm_scale / c)
    assert_same(out, out_scaled, tol(impl, dtype), "scaling Q by 2 and sm_scale by 1/2")


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("dtype", [torch.float32], ids=["fp32"])
@pytest.mark.parametrize("impl", ["fp64", "local"])
def test_shift_invariance(impl, dtype, causal):
    """Softmax is invariant to a constant added to every score in a row.

    There is no API for "add c to the scores", so the constant is smuggled in
    through an extra head dimension: append 1 to every Q row and c/sm_scale to
    every K row, and every score gains exactly c. Two consequences:

    * The head dim becomes D+1, so this identity is only posed to the references.
      A kernel with HEAD_DIM as a constexpr in {16,32,64,128} cannot be handed 65
      columns, and the property under test belongs to softmax, not to the kernel.
    * The shift travels *through* the score matmul, so it costs mantissa bits in
      whatever dtype that matmul runs in. In fp16 a score near 200 has an ulp of
      0.125; measured on 2026-08-24, this identity comes apart at 5.9e-03 (fp16)
      and 6.5e-02 (bf16) with shift=25 -- an artefact of the test's own
      construction, not of the implementation, which is why it is posed in fp32
      (measured 1.0e-06 for `local`, 2.2e-15 for the fp64 reference). Overflow of
      un-shifted scores is covered in test_adversarial.py, in the dtype it matters in.

    Both calls pass sm_scale explicitly: the default 1/sqrt(D) would change under
    the augmentation and the comparison would no longer be a pure shift.
    """
    fn = resolve_impl(impl)
    b, h, n, d = 1, 2, 128, 64
    q, k, v = make_qkv(b, h, n, d, dtype)
    sm_scale = 1.0 / math.sqrt(d)
    shift = 25.0  # exp(25) = 7.2e10, already inf in fp16: only the max subtraction saves it

    ones = torch.ones(b, h, n, 1, dtype=dtype)
    q_aug = torch.cat([q, ones], dim=-1)
    k_aug = torch.cat([k, ones * (shift / sm_scale)], dim=-1)

    out = fn(q, k, v, causal=causal, sm_scale=sm_scale)
    out_shifted = fn(q_aug, k_aug, v, causal=causal, sm_scale=sm_scale)
    assert_same(out, out_shifted, tol(impl, dtype), f"adding {shift} to every score")


@pytest.mark.parametrize("i", [0, 1, 255, 256, 400], ids=lambda i: f"row{i}")
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("impl", IMPLS)
def test_causal_containment(impl, dtype, i):
    """Rows 0..i must be bit-identical when K/V beyond column i are replaced.

    Masked contributions enter as exp(-inf) = 0 exactly and rescale by
    exp(m - m) = 1 exactly, so a correct causal implementation is not merely close
    here, it is unchanged -- which is why this asserts an exact match. A diagonal
    that is off by one shifts rows 0..i and fails immediately, with no reference
    involved. Rows 255 and 256 straddle the tile boundary of `chunked`.
    """
    fn = resolve_impl(impl)
    b, h, n, d = 1, 2, 512, 64
    q, k, v = make_qkv(b, h, n, d, dtype)

    g = torch.Generator().manual_seed(2)
    k2, v2 = k.clone(), v.clone()
    tail = (n - i - 1, d)
    k2[:, :, i + 1 :, :] = torch.randn(b, h, *tail, generator=g).to(dtype)
    v2[:, :, i + 1 :, :] = torch.randn(b, h, *tail, generator=g).to(dtype)

    out = fn(q, k, v, causal=True)[:, :, : i + 1, :]
    out_tail_changed = fn(q, k2, v2, causal=True)[:, :, : i + 1, :]
    diff = (out.double() - out_tail_changed.double()).abs().max().item()
    assert diff == 0.0, (
        f"rows 0..{i} moved by {diff:.3e} when K/V after column {i} changed -- "
        "the causal mask is wrong (most likely off by one on the diagonal)"
    )
