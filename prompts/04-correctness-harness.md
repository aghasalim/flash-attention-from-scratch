# Task 04 — Correctness harness

**Wave:** 1 (parallel with 01 and 02)
**OWNS:** `tests/`, `fa/ref/fp64.py`
**READS:** `fa/ref/`, `scripts/`

## Context

Build the test suite *before* the kernel exists, against the reference implementations. This is deliberate: a harness written after a kernel tends to encode that kernel's bugs as expected behavior. Tests that need `fa/triton/fwd.py` should be written now and marked `pytest.mark.xfail(reason="task 03")`.

## Task

**1. `fa/ref/fp64.py`** — the ground truth. Plain PyTorch attention in float64, no fusion, no tricks, deliberately slow and obviously correct. Everything in the repo is measured against this.

**2. `tests/test_correctness.py`** — parametrized over `B ∈ {1,4}`, `H ∈ {1,8,32}`, `N ∈ {1,7,128,129,512,1000,2048,4096}`, `D ∈ {16,32,64,128}`, dtype ∈ {fp16,bf16}, causal ∈ {T,F}.

The comparison must be **relative**, and this is the most important design decision in the file:

```python
def assert_no_worse_than_naive(out, ref_fp64, naive_fp16):
    err_kernel = (out.double()   - ref_fp64).abs().max()
    err_naive  = (naive_fp16.double() - ref_fp64).abs().max()
    assert err_kernel <= 2.0 * err_naive
```

An absolute tolerance is either so loose it passes broken kernels or so tight it fails correct ones, because attention's numerical error grows with `N` and with score magnitude. "No worse than the naive implementation in the same precision" is the bar the FlashAttention paper uses and the only one that's meaningful.

**3. `tests/test_adversarial.py`** — the cases that actually catch bugs:

- One score at +100, rest at −100 (overflow probe)
- All scores identical (uniform attention; `l_i` becomes exactly `N`)
- `N=1` (single token, degenerate softmax)
- `D=1`
- Q, K, V drawn from `N(0, 100)` — large magnitudes stress the max-tracking
- One row entirely masked under causal at position 0 (must be the identity on V, not NaN)
- Non-contiguous input tensors (transposed views) — must either work or raise clearly, never silently produce garbage
- `N` prime (997, 1009) — catches every boundary-mask bug

**4. `tests/test_gradients.py`** — `torch.autograd.gradcheck` in fp64 against the reference, small shapes (`B=1,H=2,N=16,D=8`). Marked xfail until task 05. Also a finite-difference sanity check on dQ/dK/dV separately, so a broken gradient is localized to one tensor rather than reported as "gradcheck failed."

**5. `tests/test_invariants.py`** — properties that must hold regardless of implementation:

- **Permutation:** jointly permuting K and V along `N` leaves non-causal output unchanged
- **Scale:** `attention(cQ, K, V)` with `sm_scale/c` equals `attention(Q, K, V)`
- **Shift:** adding a constant to every score changes nothing (softmax shift-invariance)
- **Causal containment:** output at position `i` is unchanged by modifying K/V at positions `> i`. This is a strong causal-correctness test that doesn't require a reference implementation, and it catches off-by-one diagonal errors that comparison tests miss.

**6. `tests/conftest.py`** — fill in the `TOLERANCES` dict left as `None` by task 00. Seed fixture, `skip_if_no_cuda`, and a `--slow` flag gating the `N ≥ 4096` sweeps so the default run stays under 60 seconds.

## Acceptance criteria

- `pytest tests/ -v` runs green with everything kernel-dependent marked xfail
- `pytest tests/ -m "not slow"` finishes in under 60s
- The invariant tests pass against `fa/ref/naive.py`, proving the tests themselves are correct before any kernel exists
- Test IDs are readable: `test_correctness[fp16-causal-N4096-D64]`, not `test_correctness[0-1-2-3]`

## Gotchas

- `gradcheck` needs fp64 and `requires_grad=True` on leaves. It will report failure on a correct fp32 implementation — that's gradcheck being right, not your code being wrong.
- bf16 has 8 mantissa bits vs fp16's 11 — wider dynamic range, less precision. It needs its own tolerance entry. A single tolerance for both is wrong in one direction or the other.
- Don't compare against `F.scaled_dot_product_attention` as ground truth. It *is* FlashAttention. If your kernel and SDPA share a bug you'll never see it. fp64 naive is the only reference.

## Finish by

Adding a LOGBOOK entry with the test count and the naive-fp16-vs-fp64 error at `N=4096`. That number is the bar every kernel in this repo has to clear.
