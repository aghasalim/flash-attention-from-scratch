# Task 01 — Naive baseline and roofline analysis

**Wave:** 1 (parallel with 02 and 04)
**OWNS:** `fa/ref/naive.py`, `bench/roofline.py`, `notes/00-roofline.md`
**READS:** `scripts/`, `HARDWARE.md`, `hardware.json`

## Context

The premise of this whole project is that attention is memory-bandwidth-bound, not compute-bound. I don't want to take that on faith. Your job is to prove it on my specific GPU, with numbers.

## Task

**1. `fa/ref/naive.py`** — three reference implementations, all correct, all deliberately unfused:

- `naive_attention(q, k, v, causal=False)` — materializes the full `S = QK^T / sqrt(d)` score matrix, applies mask, `softmax`, `P @ V`. This is the strawman.
- `chunked_attention(q, k, v, chunk=1024, causal=False)` — loops over K/V chunks in Python, still materializes each chunk's scores. Memory-bounded but not fused. This isolates *tiling* from *kernel fusion*, which matters for the ablation later.
- `sdpa_attention(q, k, v, causal=False, backend=...)` — wraps `F.scaled_dot_product_attention` with `torch.nn.attention.sdpa_kernel` forcing each of `MATH`, `EFFICIENT`, `FLASH` separately. Skip backends the GPU doesn't support and say so rather than silently falling through.

Shapes: `(B, H, N, D)` throughout, contiguous, matching what the Triton kernel will take.

**2. `bench/roofline.py`** — for `B=4, H=32, D=64`, sweep `N ∈ {512, 1024, 2048, 4096, 8192, 16384}`, fp16, causal ∈ {False, True}:

- Latency (CUDA events, 20 warmup, median of 100)
- Peak memory (`torch.cuda.reset_peak_memory_stats()` then `max_memory_allocated()`)
- **Analytic FLOPs.** Forward: `4 * B * H * N² * D` (two matmuls, `2*M*N*K` each). Halve for causal. Backward with recomputation: five matmuls of the same shape → `10 * B * H * N² * D`, i.e. 2.5× forward. Derive these in the notes, don't just assert them.
- **Analytic HBM bytes.** Naive: `Q,K,V,O` are `4*B*H*N*D*2` bytes, but `S` and `P` are `2 * B*H*N² * 2` bytes written and read again. For `N=4096, B=4, H=32` that's the dominant term by two orders of magnitude — show this explicitly.
- **Arithmetic intensity** = FLOPs / HBM bytes, and plot it against the machine balance point (measured TFLOP/s ÷ measured GB/s from `hardware.json`). Anything left of the ridge point is memory-bound.
- Catch OOM per configuration, record `OOM`, and continue the sweep — do not let one OOM kill the run.

Write `results/roofline.csv` and `results/roofline.png` (log-log, ridge point marked).

**3. `notes/00-roofline.md`** — the write-up. Must contain:

- The FLOP and byte derivations, shown step by step
- The measured ridge point for this GPU in FLOP/byte
- The exact `N` at which naive attention crosses from compute-bound to memory-bound on this card
- The exact `N` at which naive attention OOMs, and the closed-form prediction of that `N` from available VRAM (`B*H*N²*2 bytes * 2 tensors` ≈ VRAM), showing prediction vs. reality
- One paragraph, in your own words, on why `chunked_attention` fixes the memory problem but not the bandwidth problem — that gap is exactly what kernel fusion buys and it's the single most important idea in the repo

## Acceptance criteria

- `python -m bench.roofline` produces the CSV and PNG
- `chunked_attention` and `naive_attention` agree with `sdpa_attention(backend=MATH)` to fp16 tolerance
- The write-up predicts the OOM threshold within 15% of the measured one
- The measured latency ratio between naive and `sdpa(FLASH)` at `N=4096` is stated, and it is a real number from your CSV

## Gotchas

- `max_memory_allocated()` reports PyTorch's caching allocator, not true device usage. Call `torch.cuda.empty_cache()` and `reset_peak_memory_stats()` between configs or the numbers are cumulative garbage.
- `F.scaled_dot_product_attention` silently picks a backend. If you don't force it with `sdpa_kernel`, you'll benchmark FlashAttention and label it "math."
- Causal `sdpa` uses `is_causal=True`, not an additive mask. Passing a mask disables the fast path and you'll get a misleadingly slow "flash" number.

## Finish by

Adding a LOGBOOK entry with the naive-vs-flash latency ratio at `N=4096` and the OOM threshold. These are the two numbers the whole project is measured against.
