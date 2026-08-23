# Task 06 — Causal masking, sliding window, ALiBi, attention sinks

**Wave:** 3 (parallel with 05 and 07)
**OWNS:** `fa/triton/masks.py`, `fa/triton/fwd_causal.py`
**READS:** `fa/triton/fwd.py`

## Context

Causal masking is not "add `-inf` above the diagonal." Done properly it's a *performance* feature: roughly half the KV blocks never need to be visited at all, so causal attention should run near 2× faster than non-causal, not the same speed. If your causal kernel isn't meaningfully faster than your dense one, you're computing and discarding.

## Task

**Three-zone block classification.** For Q block `i` and KV block `j` under causal masking:

- `j·BLOCK_N + BLOCK_N - 1 < i·BLOCK_M` → **fully visible**, no mask, dense path
- `j·BLOCK_N > i·BLOCK_M + BLOCK_M - 1` → **fully hidden**, skip entirely, never load K/V
- otherwise → **diagonal**, apply element-level mask

Implement as two loops with different bodies, not one loop with a branch. Triton will generate a mask-free inner loop for the dense zone, which is the whole point. A single loop with `if` inside costs you the win.

**1. `fa/triton/masks.py`** — composable score modifiers applied to `qk` before the max:

- `causal(offs_m, offs_n)` — the diagonal-block element mask
- `sliding_window(offs_m, offs_n, window)` — Mistral-style local attention; keys outside `[i-window, i]` masked. Note this makes *some* blocks fully hidden on both sides, so the block classification becomes a band rather than a triangle. Extend the loop bounds accordingly.
- `alibi(offs_m, offs_n, slope)` — additive linear bias `-slope·(i-j)`, applied to `qk` before the max. Slopes are a geometric sequence over heads: head `h` of `H` gets `2^(-8(h+1)/H)`.
- `attention_sink(n_sink)` — first `n_sink` keys always visible regardless of window. Needed for streaming inference; interacts with sliding window in a way worth testing explicitly.
- `custom_bias(bias_ptr, ...)` — arbitrary additive `(B, H, N, N)` or broadcast bias. Slow path, but needed for relative position embeddings and worth having for completeness.

Design them so they compose (causal + sliding window + sink is a real configuration used in production models) and so unused modifiers are `tl.constexpr`-gated to zero cost.

**2. `fa/triton/fwd_causal.py`** — the causal forward kernel with block skipping. Also extend task 05's backward if it's merged; if not, leave a clearly marked TODO and coordinate — do not edit `bwd.py`, it isn't yours this wave.

**3. Benchmark the skipping.** Measure causal vs. non-causal latency at `N ∈ {1024, 4096, 16384}`. Expected ratio approaches 2× as `N` grows and is worse at small `N` where the diagonal blocks are a large fraction of the total. Plot the ratio against `N` and explain the curve — the asymptote and the shape of the approach are both derivable, so derive them.

## Acceptance criteria

- Causal output matches the fp64 reference under the task 04 relative bar
- The causal-containment invariant from `tests/test_invariants.py` passes: modifying K/V at positions `> i` must not change output at position `i`
- Causal is ≥1.6× faster than non-causal at `N=4096`. If it isn't, blocks aren't being skipped — check the loop bounds, not the mask
- Sliding window with `window=256` at `N=16384` is dramatically faster than full causal, and the ratio roughly matches the fraction of blocks visited
- ALiBi matches a reference implementation that adds the bias to a materialized score matrix
- Sink + window composition is tested and correct
- Correct for `N` not divisible by `BLOCK_M` or `BLOCK_N` — the diagonal-block logic is where this breaks

## Gotchas

- **The off-by-one.** Position `i` attends to positions `j ≤ i` (inclusive). Whether your loop bound is `<` or `<=` decides whether a token can see itself. Both variants produce plausible-looking models; only one is right. The containment invariant test catches it, which is why it exists.
- With `BLOCK_M ≠ BLOCK_N` the diagonal isn't aligned to block boundaries and the classification arithmetic gets fiddly. Get it right for equal blocks first, then generalize, and test with `BLOCK_M=128, BLOCK_N=32` specifically.
- ALiBi bias is added to `qk` **before** the running max. Adding it after breaks the max-tracking and produces silently wrong results with no NaNs to tip you off.
- With sliding window, a query near position 0 may have *fewer* visible keys than the window size. Don't let the loop bound go negative.
- A fully-masked row (possible with pathological window/sink configs) gives `l_i = 0` and a division by zero. Guard it and decide explicitly what the output should be — zeros, not NaN.

## Finish by

Adding a LOGBOOK entry: causal/non-causal speedup at N ∈ {1024, 4096, 16384}, sliding-window speedup at N=16384, and confirmation that the containment invariant passes.
