"""Backward pass: gradcheck against the fp64 reference, plus per-tensor finite differences.

Two layers, on purpose:

* `gradcheck` is the thorough one, but when it fails it says "Jacobian mismatch"
  and leaves you to work out which of dQ, dK, dV is wrong.
* The finite-difference checks below probe dQ, dK and dV *separately*, so a broken
  gradient is localised to one tensor before you start reading kernel code.

The reference-side tests run today and must pass: they are what says the harness
is right. The kernel-side ones are xfail until task 05 -- and on this machine they
can never run at all (no CUDA device, no Triton wheel for macOS).

gradcheck needs float64 leaves with requires_grad=True; run it on an fp32 or fp16
implementation and it reports a failure that is gradcheck being right about
precision, not the implementation being wrong.
"""

from __future__ import annotations

import pytest
import torch
from conftest import (
    KERNEL_PARAM,
    assert_no_worse_than_naive,
    make_qkv,
    naive_attention,
    resolve_impl,
)

from fa.ref.fp64 import attention_fp64

B, H, N, D = 1, 2, 16, 8
TENSORS = ["q", "k", "v"]


def leaves(dtype=torch.float64, requires_grad=True):
    q, k, v = make_qkv(B, H, N, D, dtype, seed=7)
    return [t.clone().requires_grad_(requires_grad) for t in (q, k, v)]


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
def test_gradcheck_reference(causal):
    """The fp64 reference is differentiable and its autograd graph is correct.

    This is the test that licenses using reference autograd as ground truth for
    dQ/dK/dV everywhere else in the file.
    """
    q, k, v = leaves()
    assert torch.autograd.gradcheck(
        lambda a, b, c: attention_fp64(a, b, c, causal=causal), (q, k, v), fast_mode=True
    )


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("impl", [KERNEL_PARAM])
def test_gradcheck_kernel(impl, causal):
    """Task 05. Kept here so the day the backward kernel lands, the test already exists."""
    fn = resolve_impl(impl)
    q, k, v = leaves()
    assert torch.autograd.gradcheck(
        lambda a, b, c: fn(a, b, c, causal=causal), (q, k, v), fast_mode=True
    )


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("wrt", TENSORS, ids=lambda t: f"d{t.upper()}")
def test_finite_difference_reference(wrt, causal):
    """Central differences on the fp64 reference, one tensor at a time.

    Perturb 24 random entries of exactly one of Q, K, V by +-eps, re-run the
    forward, and compare (L(+eps) - L(-eps)) / 2*eps against autograd's gradient
    for that entry. In fp64 with eps=1e-6 the central difference is good to ~1e-9
    relative, so a real gradient bug is orders of magnitude clear of the noise.
    """
    q, k, v = leaves()
    tensors = {"q": q, "k": k, "v": v}
    target = tensors[wrt]

    gen = torch.Generator().manual_seed(3)
    weights = torch.randn(B, H, N, D, dtype=torch.float64, generator=gen)

    def loss(a, b, c):
        return (attention_fp64(a, b, c, causal=causal) * weights).sum()

    loss(q, k, v).backward()
    analytic = target.grad.detach().clone()
    assert torch.isfinite(analytic).all(), f"d{wrt.upper()} has NaN or Inf"

    g = torch.Generator().manual_seed(4)
    flat_idx = torch.randperm(target.numel(), generator=g)[:24]
    eps = 1e-6
    with torch.no_grad():
        for idx in flat_idx.tolist():
            base = target.view(-1)[idx].item()
            target.view(-1)[idx] = base + eps
            lp = loss(q, k, v).item()
            target.view(-1)[idx] = base - eps
            lm = loss(q, k, v).item()
            target.view(-1)[idx] = base
            numeric = (lp - lm) / (2 * eps)
            got = analytic.view(-1)[idx].item()
            assert abs(numeric - got) <= 1e-5 * max(1.0, abs(numeric)), (
                f"d{wrt.upper()}[{idx}]: autograd {got:.9f} vs finite difference {numeric:.9f}"
            )


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("wrt", TENSORS, ids=lambda t: f"d{t.upper()}")
@pytest.mark.parametrize("dtype", [torch.float16], ids=["fp16"])
@pytest.mark.parametrize("impl", ["naive", "chunked", KERNEL_PARAM])
def test_grad_no_worse_than_naive(impl, dtype, wrt, causal):
    """Per-tensor version of the forward bar: dQ, dK and dV, each judged separately.

    Ground truth is the fp64 reference's own autograd gradient (licensed by
    test_gradcheck_reference above); the bar is naive fp16 attention's gradient
    error against it. Reporting per tensor is the whole point -- "gradcheck
    failed" does not tell you which matmul in the backward pass is wrong.

    With impl="naive" the candidate *is* the bar, so this only checks that the
    plumbing produces finite gradients of the right shape. It earns its keep when
    task 05's backward kernel arrives; it is written now so that it cannot be
    written to match that kernel's bugs.
    """
    fn = resolve_impl(impl)

    def grads(f, dtype_, **kw):
        q, k, v = (t.clone().requires_grad_(True) for t in make_qkv(B, H, N, D, dtype_, seed=7))
        w = torch.randn(B, H, N, D, generator=torch.Generator().manual_seed(3)).to(dtype_)
        (f(q, k, v, causal=causal, **kw) * w).sum().backward()
        return {"q": q.grad, "k": k.grad, "v": v.grad}

    ref = grads(attention_fp64, torch.float64)[wrt]
    naive = grads(naive_attention, dtype)[wrt]
    try:
        out = grads(fn, dtype)[wrt]
    except RuntimeError as exc:
        # Some references are forward-only by construction. fa/ref/naive.py's
        # chunked_attention rescales its score tile in place (`s.sub_(m_new).exp_()`),
        # which autograd refuses to differentiate. That is a property of that file,
        # not a failure of this test, and it is not this task's file to change --
        # so say so out loud rather than swallowing it.
        if "inplace" in str(exc) or "version" in str(exc):
            pytest.skip(f"{impl} is forward-only, autograd rejects it: {exc}")
        raise
    assert_no_worse_than_naive(out, ref, naive)
