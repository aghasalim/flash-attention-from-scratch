# Task 09 — Flash-Decoding and paged KV cache

**Wave:** 4 (parallel with 08 and 10)
**OWNS:** `fa/triton/decode.py`, `fa/ops/paged.py`
**READS:** `fa/triton/`, `bench/`

## Context

Autoregressive decoding is a completely different problem from training, and the kernel you built for training is *bad* at it. At batch 1 with one new query token against a 100k-token cache, the grid is `(1, B·H)` — a few dozen programs on a GPU with 100+ SMs. Most of the machine sits idle while a handful of programs stream the entire KV cache through memory.

Flash-Decoding fixes this by splitting the KV dimension across many programs and reducing their partial results. It's the same online-softmax trick applied one level up: partial softmax statistics combine associatively, so you can compute them in parallel and merge.

## Task

**1. `fa/triton/decode.py` — split-KV attention.**

Phase 1 kernel: grid `(B·H, num_splits)`. Split `k` handles KV positions `[k·chunk, (k+1)·chunk)`. Each program computes a partial output `O_k` and partial statistics `(m_k, l_k)` and writes them to scratch.

Phase 2 kernel: reduce across splits using exactly the recurrence from `notes/01-online-softmax.md`:

```
m = max_k(m_k)
l = Σ_k exp(m_k - m) · l_k
O = Σ_k exp(m_k - m) · l_k · O_k / l
```

State clearly in `notes/04-decoding.md` that this is the *same* associativity that makes the tiled forward pass work, now exploited for parallelism instead of for memory. That connection is the insight worth writing down.

Choose `num_splits` adaptively: enough to fill the SMs (`num_splits ≈ ceil(num_SMs / (B·H))`, read SM count from `hardware.json`), capped so each split is at least ~256 positions. Reduction overhead eats the win past that point — find the actual crossover empirically and plot it.

**2. `fa/ops/paged.py` — paged KV cache.**

The cache lives in fixed-size blocks (16 or 32 tokens) in a flat pool. A `block_table` of shape `(batch, max_blocks)` maps logical position to physical block. Sequences grow by allocating new blocks; no reallocation, no copying, no fragmentation. This is vLLM's PagedAttention.

- Kernel reads K/V through the block table: logical `j` → `block_table[b, j // BLOCK] * BLOCK + (j % BLOCK)`
- Support ragged batches — different sequences at different lengths in one launch
- Handle the partial last block
- Implement a minimal allocator: `allocate(seq_id, n_tokens)`, `free(seq_id)`, `append(seq_id, k, v)`
- Optional but a strong addition: copy-on-write block sharing for a common prompt prefix across a batch. It's what makes beam search and parallel sampling cheap, and it demonstrates you understand *why* paging is the right abstraction rather than just that it works

**3. Benchmark the regime that matters:**

- Batch 1, `N ∈ {1k, 4k, 16k, 64k, 128k}`, one query token: standard kernel vs. Flash-Decoding. The gap should be large and should grow with `N`
- Batch ∈ {1, 8, 64}: show that the advantage *shrinks* as batch grows and explain why — at large batch there's already enough parallelism and splitting only adds reduction overhead. Find the crossover batch size on this GPU
- Achieved DRAM bandwidth for both. Decoding is purely memory-bound; the correct target is ~100% of measured peak bandwidth, not a TFLOP/s number. If you're at 40% of bandwidth, that's the bug to chase
- Paged vs. contiguous cache overhead — should be a few percent; if it's more, the indirection isn't being hidden

## Acceptance criteria

- Split-KV matches the non-split kernel to fp16 tolerance for all split counts including `num_splits=1`
- Reduction is numerically stable when split maxima differ by large margins — construct that case deliberately
- ≥3× faster than the training kernel at batch 1, `N=32768`. Larger at `N=128k`
- Achieved bandwidth is >70% of measured peak in the batch-1 long-context case
- Paged and contiguous produce identical output
- The allocator handles fragmentation: allocate 10 sequences, free the even ones, allocate 5 more, verify correctness
- The `num_splits` sweep and the batch-size crossover are both plotted

## Gotchas

- Phase 2's reduction must use the max-shifted form. Naively summing `l_k · O_k` overflows when one split has much larger scores than the others — and long contexts make that common, not rare.
- Scratch buffers for partial results are `(B, H, num_splits, D)` fp32. That's real memory; account for it and include it in the memory table, don't quietly leave it out.
- Block table indirection adds a dependent load per KV block. Prefetch or it becomes the bottleneck — and you'll only see this in the profiler, not the latency number.
- At batch 1 the *kernel launch overhead* becomes measurable relative to the kernel itself. Consider CUDA graphs, and measure launch overhead separately so you know what fraction of the number it is.

## Finish by

Adding a LOGBOOK entry: speedup at batch 1 / N=32768, achieved bandwidth as % of peak, the optimal `num_splits` curve, and the batch size at which splitting stops helping.
