# Task 03 — The Triton forward kernel

**Wave:** 2 (serial — run alone; every later task extends this file)
**OWNS:** `fa/triton/fwd.py`, `fa/ops/attention.py`
**READS:** `fa/ref/`, `tests/`, `notes/01-online-softmax.md`, `HARDWARE.md`

## Context

This is the core of the project. `fa/ref/online_softmax.py::online_attention` is the NumPy version of exactly this algorithm — port it, loop for loop. Read `notes/01-online-softmax.md` first; the recurrence and the fp32-accumulator finding are both there and both matter.

Non-causal only. Task 06 adds masking. Resist the urge to do both at once — a wrong causal kernel and a wrong forward kernel look identical from the outside and you will not be able to tell which you have.

## Task

**`fa/triton/fwd.py`** — `_attn_fwd` Triton kernel plus a Python launcher.

**Grid:** `(triton.cdiv(N, BLOCK_M), B * H)`. Each program owns one block of `BLOCK_M` query rows for one (batch, head) pair.

**Per-program state, all fp32, all in registers:**
- `acc` : `(BLOCK_M, HEAD_DIM)` output accumulator
- `m_i` : `(BLOCK_M,)` running row max, init `-inf`
- `l_i` : `(BLOCK_M,)` running row sum, init `0.0`

**Inner loop** over `BLOCK_N`-sized KV blocks:

```
qk       = tl.dot(q, k) * sm_scale          # fp32 accumulate
m_new    = tl.maximum(m_i, tl.max(qk, 1))
alpha    = tl.exp2((m_i - m_new) * LOG2E)   # rescale factor, always ≤ 1
p        = tl.exp2((qk - m_new[:, None]) * LOG2E)
l_i      = l_i * alpha + tl.sum(p, 1)
acc      = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
m_i      = m_new
```

Then `acc = acc / l_i[:, None]`, store `O`, and store `L = m_i + tl.log(l_i)` — the per-row logsumexp that task 05's backward pass needs.

**Required details:**

- **Use `tl.exp2`, not `tl.exp`.** Fold `log2(e)` into `sm_scale` up front so the multiply is free. `exp2` maps to a single `ex2.approx.f32` SASS instruction; `exp` becomes a multiply plus `exp2`. Measure both and record the delta — on most cards it's a few percent, which is worth knowing precisely because it's the kind of thing people claim without checking.
- `q` is loaded once and stays in registers for the whole inner loop. That's the entire point of the outer-Q/inner-KV ordering (this is the FlashAttention-2 arrangement; v1 had the loops the other way and had to re-read and rescale the output from HBM every step).
- Boundary handling for `N % BLOCK_M != 0` and `N % BLOCK_N != 0`: `tl.load(..., mask=..., other=0.0)` for K and V, but the score mask must add `-inf` to out-of-range columns **before** the max. Zeros are not `-inf`; getting this wrong produces a kernel that's correct only for power-of-two sequence lengths, which is the single most common bug in hand-rolled attention kernels.
- `HEAD_DIM` as `tl.constexpr`. Support 16, 32, 64, 128. Above 128, register pressure will spill — detect and raise a clear error rather than silently producing a slow kernel.
- Strides passed explicitly per tensor; do not assume contiguity, assert it in the launcher.
- Start with fixed `BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=3`. Task 07 autotunes. Do not autotune now — you cannot debug correctness while the config is moving.

**`fa/ops/attention.py`** — the public API:

```python
def attention(q, k, v, causal=False, sm_scale=None) -> Tensor
```

Validates shapes/dtype/contiguity, defaults `sm_scale = 1/sqrt(D)`, dispatches to the kernel, raises `NotImplementedError` for `causal=True` with a pointer to task 06.

## Acceptance criteria

- Matches `fa/ref/fp64.py` reference with **max relative error no greater than naive fp16 attention's own error vs. the same fp64 reference.** This relative bar is the one from the paper and the only fair one — absolute thresholds either pass broken kernels or fail correct ones.
- Correct for `N ∈ {1, 7, 128, 129, 1000, 2048, 4096}` — the non-power-of-two cases are the point of the list
- Correct for `D ∈ {16, 32, 64, 128}`, dtype ∈ {fp16, bf16}
- Correct for `B ∈ {1, 4}`, `H ∈ {1, 8, 32}`
- Peak memory scales O(N), not O(N²) — assert this by measuring at N=2048 and N=8192 and checking the ratio is ~4, not ~16
- Runs at `N=32768` without OOM, which naive cannot do at all
- Faster than `sdpa(MATH)` at `N ≥ 1024`. Being slower than `sdpa(FLASH)` right now is fine and expected — that's a hand-tuned CUDA kernel and you have an untuned Triton one.

## Gotchas

- **All three accumulators in fp32.** `fa/ref/online_softmax.py`'s experiment shows what fp16 accumulation does at long sequence length. If your max relative error sits stubbornly around 1e-2 and won't improve, this is why — check `acc`, `m_i`, and `l_i` before you check anything else.
- `tl.dot` needs both operands ≥16 in every dimension. Small `BLOCK_N` or `HEAD_DIM=8` will fail with an unhelpful compiler error.
- Cast `p` to the V dtype before `tl.dot(p, v)`, but keep the *accumulator* fp32. `tl.dot` accumulates in fp32 by default — verify that's actually happening on your card rather than assuming.
- If the kernel is correct at `N=2048` and wrong at `N=2049`, it's the boundary mask. Every time.
- Triton caches compiled kernels aggressively. When you change a `constexpr` and behavior doesn't change, clear `~/.triton/cache`.

## Finish by

Adding a LOGBOOK entry with: max relative error vs. the fp64 reference, the naive-fp16 error for comparison, latency at `N=4096`, the `exp2`-vs-`exp` delta, and peak memory at `N=32768`.
