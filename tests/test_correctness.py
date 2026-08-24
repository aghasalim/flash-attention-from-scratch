"""The main correctness sweep: every candidate, against fp64, relative to naive.

The bar is `conftest.assert_no_worse_than_naive`, and it is relative on purpose:
a candidate's max error vs. the float64 reference must be no worse than 2x naive
attention's own error vs. the same reference. An absolute threshold on fp16
attention would have to be loose enough to pass a broken kernel at N=128 or tight
enough to fail a correct one at N=4096, because the error grows with N.

The `kernel` parameters are xfail: `fa/ops/attention.py` is task 03, and on the
machine this suite was written on it can never run at all -- there is no CUDA
device and Triton publishes no macOS wheel. `chunked` is task 01's tiled reference
and is the candidate that actually exercises the harness today; it skips until
that file lands.

Everything runs on CPU, at B=1 H=2 for the N/D sweep (B and H get their own sweep
below), because the fp64 reference materialises a B*H*N*N float64 score matrix.
N >= 4096 is marked slow: `pytest tests/` skips it, `pytest tests/ --slow` runs it.
"""

from __future__ import annotations

import pytest
import torch
from conftest import (
    KERNEL_PARAM,
    assert_no_worse_than_naive,
    external_online,
    reference_bundle,
    resolve_impl,
)

IMPLS = ["chunked", KERNEL_PARAM]
DTYPES = [torch.float16, torch.bfloat16]
DTYPE_IDS = ["fp16", "bf16"]

# 1 and 7 are degenerate, 129 and 1000 and 2048+1 are the non-power-of-two cases that
# catch boundary-mask bugs. 4096 is the size the LOGBOOK bar is quoted at.
N_VALUES = [
    pytest.param(1, id="N1"),
    pytest.param(7, id="N7"),
    pytest.param(128, id="N128"),
    pytest.param(129, id="N129"),
    pytest.param(512, id="N512"),
    pytest.param(1000, id="N1000"),
    pytest.param(2048, id="N2048"),
    pytest.param(4096, id="N4096", marks=pytest.mark.slow),
]
D_VALUES = [16, 32, 64, 128]


@pytest.mark.parametrize("d", D_VALUES, ids=lambda d: f"D{d}")
@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("impl", IMPLS)
def test_correctness(impl, dtype, causal, n, d):
    fn = resolve_impl(impl)
    q, k, v, ref_fp64, naive = reference_bundle(1, 2, n, d, dtype, causal)
    out = fn(q, k, v, causal=causal)
    assert out.shape == q.shape, f"expected {tuple(q.shape)}, got {tuple(out.shape)}"
    assert out.dtype == dtype, f"output dtype {out.dtype} != input dtype {dtype}"
    assert_no_worse_than_naive(out, ref_fp64, naive)


@pytest.mark.parametrize("h", [1, 8, 32], ids=lambda h: f"H{h}")
@pytest.mark.parametrize("b", [1, 4], ids=lambda b: f"B{b}")
@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("impl", IMPLS)
def test_correctness_batch_heads(impl, dtype, causal, b, h):
    """Same bar, sweeping the batch and head axes at a small N.

    Separate from the N/D sweep on purpose: the full cartesian product of
    B x H x N x D x dtype x causal is 1536 configurations and hours of fp64 CPU
    attention. Batching and head-striding are independent of sequence length, so
    sweeping them independently costs 24 tests instead of 1500.
    """
    fn = resolve_impl(impl)
    q, k, v, ref_fp64, naive = reference_bundle(b, h, 128, 64, dtype, causal)
    out = fn(q, k, v, causal=causal)
    assert out.shape == q.shape
    assert_no_worse_than_naive(out, ref_fp64, naive)


@pytest.mark.parametrize("block", [(128, 64), (64, 32)], ids=lambda b: f"BM{b[0]}xBN{b[1]}")
@pytest.mark.parametrize("n", [129, 1000], ids=lambda n: f"N{n}")
@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("dtype", [torch.float16], ids=["fp16"])
def test_online_softmax_reference(dtype, causal, n, block):
    """Task 02's NumPy tiled reference, held to the same bar as any kernel.

    `fa/ref/online_softmax.py::online_attention` is the algorithm task 03 ports
    into Triton loop for loop, so it is worth checking here and not only in that
    task's own self-test: neither block size divides either N, which is where a
    tiled softmax gets its boundary masking wrong.

    It takes a single (N, D) head, hence B=H=1, and fp16 only -- NumPy has no
    bfloat16 dtype to convert to.
    """
    online = external_online()
    if online is None:
        pytest.skip("fa/ref/online_softmax.py not present yet (task 02)")
    block_m, block_n = block
    q, k, v, ref_fp64, naive = reference_bundle(1, 1, n, 64, dtype, causal)
    out = online(q[0, 0].numpy(), k[0, 0].numpy(), v[0, 0].numpy(), block_m, block_n, causal=causal)
    assert_no_worse_than_naive(torch.from_numpy(out), ref_fp64[0, 0], naive[0, 0])
