# Task 02 — Online softmax: derivation, reference implementation, numerical analysis

**Wave:** 1 (parallel with 01 and 04)
**OWNS:** `fa/ref/online_softmax.py`, `notes/01-online-softmax.md`
**READS:** `scripts/`

## Context

Everything in FlashAttention rests on one fact: you can compute softmax exactly in a single streaming pass, never holding the full row in memory, by tracking a running max and a running sum and rescaling what you've already accumulated. If I don't understand this cold, the kernel is cargo cult. This task is the math, in NumPy, before any GPU code exists.

Do not write Triton here. Do not import torch. NumPy and paper only.

## Task

**1. `notes/01-online-softmax.md`** — the derivation, written out properly.

Start from the safe softmax: `softmax(x)_i = exp(x_i - m) / Σ_j exp(x_j - m)` where `m = max(x)`, and state why the shift is needed (fp16 max is 65504; `exp(12)` already overflows fp16 and score magnitudes routinely exceed that at large `d`).

Then derive the streaming recurrence. Processing block `j` with block max `m̃_j` and block sum `l̃_j`:

```
m_j     = max(m_{j-1}, m̃_j)
l_j     = exp(m_{j-1} - m_j) · l_{j-1}  +  exp(m̃_j - m_j) · l̃_j
O_j     = exp(m_{j-1} - m_j) · O_{j-1}  +  exp(m̃_j - m_j) · (P̃_j V_j)
```

**Prove that after the final block, `O_T / l_T` equals the true attention output exactly in exact arithmetic.** Induction on the number of blocks. This proof is short and it is the thing to be able to reproduce on a whiteboard.

Then state the IO-complexity result: standard attention needs Θ(N·d + N²) HBM accesses; tiled attention with SRAM of size `M` needs Θ(N²d²/M). Explain where the `d²/M` comes from (number of KV blocks × passes over Q) and why this is an *improvement* even though the FLOP count is unchanged — and note that it's an improvement precisely when `M >> d²`, which is the regime real GPUs are in.

Note the rescale-factor subtlety: `exp(m_{j-1} - m_j) ≤ 1` always, because `m_j ≥ m_{j-1}` by construction. The correction only ever shrinks. That's why the recurrence can't overflow, and it's worth one sentence of explanation.

**2. `fa/ref/online_softmax.py`** — NumPy reference:

- `online_softmax(x, block_size)` → the running-statistics version
- `online_attention(q, k, v, block_m, block_n, causal=False)` → the full tiled algorithm in NumPy, structured *exactly* as the eventual Triton kernel will be (outer loop over Q blocks, inner loop over KV blocks, fp32 accumulators). Task 03 will port this line by line, so make the structure explicit and comment which loop becomes the Triton grid axis.
- `logsumexp_rows(q, k, causal)` → returns `L = m + log(l)` per row. The backward pass needs this and only this to reconstruct `P` — one float per row instead of the whole N×N matrix. Make sure the write-up says why.

**3. Numerical experiments** in the notes, with actual numbers:

- Max absolute deviation between `online_softmax` and `scipy.special.softmax` in fp64, across 1000 random rows — should be ≲1e-15
- Same in simulated fp32 and fp16 accumulation. Show the fp16-accumulator version degrading badly at `N=8192` and the fp32-accumulator version holding. **This is the experiment that justifies the "all accumulators are fp32" rule** and later tasks will cite it.
- Adversarial input: a row where one score is +100 and the rest are −100. Show naive softmax overflowing and the online version surviving.
- Block-order invariance: for non-causal attention, permuting the KV block order must not change the result beyond floating-point noise. Assert it.

## Acceptance criteria

- `pytest tests/` (whatever exists) plus a self-test in the module: online vs. scipy agreement < 1e-14 in fp64
- The induction proof is written out, not gestured at
- The fp16-vs-fp32 accumulator experiment shows a concrete error gap with real numbers in a table
- `online_attention` matches a direct NumPy `softmax(QK^T)V` to 1e-12 in fp64, for causal and non-causal, for several block sizes including ones that don't divide `N` evenly

## Gotchas

- The uneven-block case is where people get this wrong. If `N % block_size != 0`, the last block is short. Masking it with zeros is wrong for the max computation — masked positions must contribute `-inf` to the max and `0` to the sum, which is not the same thing.
- For causal masking, blocks entirely above the diagonal are skipped, blocks entirely below are dense, and only the diagonal blocks need element-level masking. Structure the reference this way now; task 06 depends on it.

## Finish by

Adding a LOGBOOK entry with the fp16-vs-fp32 accumulator error gap at `N=8192`. Task 03 will hit this exact bug and will need the number to recognize it.
