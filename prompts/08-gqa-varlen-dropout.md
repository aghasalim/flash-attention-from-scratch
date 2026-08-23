# Task 08 — GQA, variable-length batches, dropout

**Wave:** 4 (parallel with 09 and 10)
**OWNS:** `fa/triton/gqa.py`, `fa/triton/varlen.py`, `fa/triton/dropout.py`
**READS:** `fa/triton/`, `tests/`

## Context

This is where the kernel stops being a demo. No production model uses plain MHA with uniform sequence lengths and no dropout. Each of these three features is individually unglamorous and collectively they're the difference between "I implemented the paper" and "I implemented something you could actually train with."

## Task

**1. `fa/triton/gqa.py` — grouped-query attention.**

`Q` has `H_q` heads, `K`/`V` have `H_kv` heads, `H_q % H_kv == 0`, `G = H_q / H_kv` queries share each KV head. MQA is `H_kv = 1`.

The naive approach — `repeat_interleave` on K and V — allocates `G×` the KV memory and defeats the entire purpose of GQA. Do it with **index arithmetic in the kernel**: map program ID to `(batch, q_head)`, compute `kv_head = q_head // G`, offset the K/V pointers accordingly. Zero extra memory.

The backward pass is the interesting part: `dK` and `dV` for one KV head accumulate contributions from `G` query heads. Handle this in the block decomposition (loop over the group inside the kernel), not with atomics. Explain the choice in `notes/03-gqa.md`.

Benchmark KV-cache memory: MHA vs. GQA(8) vs. MQA at `N=32768`. The ratio is the reason GQA exists and it's a good table to have. Note the connection: this is the same memory wall that Multi-head Latent Attention attacks from a completely different angle — GQA shares KV heads, MLA compresses them into a low-rank latent. Worth a paragraph in the notes because it's the bridge to the next project.

**2. `fa/triton/varlen.py` — packed variable-length batches.**

Real batches have sequences of different lengths. Padding to the max wastes compute quadratically. The standard solution: concatenate all sequences into one flat `(total_tokens, H, D)` tensor plus a `cu_seqlens` int32 array of cumulative offsets, shape `(batch+1,)`.

- Grid becomes 1-D over (sequence, q_block) pairs, computed from `cu_seqlens`
- Each program looks up its sequence's start/end and clamps its KV loop
- No attention across sequence boundaries — ever. This is the correctness property that matters most and the one that's easiest to get subtly wrong.
- Handle empty sequences (`cu_seqlens[i] == cu_seqlens[i+1]`) without launching degenerate programs
- Support per-sequence causal masking with offsets

Benchmark against padded attention on a realistic length distribution — sample lengths from a lognormal with mean 512, max 4096, batch 32. The speedup is roughly `mean_len/max_len` and it's usually larger than people expect.

**3. `fa/triton/dropout.py` — attention dropout.**

Dropout is applied to `P` after softmax, before `P @ V`. The hard constraint: **forward and backward must generate identical masks** without materializing or storing them.

Use Triton's Philox counter-based RNG (`tl.rand` with an explicit seed and offset). The offset must be a deterministic function of `(batch, head, q_block_idx, kv_block_idx, position_within_block)` so backward regenerates exactly the same values. Storing the mask defeats the memory savings and is not an acceptable shortcut.

Scale surviving values by `1/(1-p)`. Verify the scaling by checking that `E[output]` is unchanged over many seeds.

## Acceptance criteria

- GQA matches a `repeat_interleave` reference exactly, forward and backward, for `G ∈ {1, 2, 4, 8, 32}`
- GQA allocates no more KV memory than the un-repeated tensors — assert on `max_memory_allocated`
- Varlen matches per-sequence separate attention calls exactly
- **No cross-sequence attention:** construct a test where sequence 2's V is pure garbage and assert sequence 1's output is bit-identical to running it alone. This is the test that matters
- Varlen handles empty and length-1 sequences
- Dropout masks are identical between fwd and bwd — assert directly by instrumenting the kernel to dump masks in a debug build
- Dropout with `p=0` is bit-identical to no dropout
- Gradients still pass `gradcheck` with dropout disabled

## Gotchas

- GQA backward accumulating `dK`/`dV` across `G` query heads is the one genuinely tricky part of this task. Get the forward working and tested before touching it.
- `cu_seqlens` is `int32` and must be on the GPU. A silent host/device mismatch here produces garbage indices and reads out of bounds — sometimes without crashing, which is worse.
- Philox offset collisions across blocks produce *correlated* dropout masks, which won't fail any test you'd naively write but will quietly degrade training. Verify statistical independence across blocks explicitly.
- With varlen, `max_seqlen` must be passed separately for grid sizing — it can't be derived inside the kernel.

## Finish by

Adding a LOGBOOK entry: KV memory MHA vs GQA(8) vs MQA at N=32768; varlen speedup over padded on the lognormal distribution; confirmation that fwd/bwd dropout masks match.
