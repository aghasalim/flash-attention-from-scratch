# Logbook

Dated entries, in order. Every kernel change that moves a number gets one. Three lines, not an essay.

This is the highest-value file in the repo and the one people don't expect. It's the difference between "I built a Flash Attention kernel" and being able to answer, two months later, *why* `num_stages=3` beat `num_stages=4` on this specific card. It is also — if you're ever asked to talk about this work — the record that makes the story concrete instead of general.

**Format:**

```markdown
## YYYY-MM-DD — <one-line title>
**Tried:** what I changed
**Measured:** the number, before → after
**Concluded:** what it means, or what I still don't understand
```

Write the "Concluded" line even when the answer is "no idea why." Especially then. The unexplained results are the ones you come back to.

---

## Format examples — NOT my results

These are illustrative, from the task-spec pack this repo is built on. They are
here to show the grain of a good entry: specific, numeric, unembellished. The
`2025-XX-XX` dates are placeholders. Every one of them gets deleted the moment I
have a real entry to put in its place, and nothing here should be read as a
measurement I made.

## 2025-XX-XX — fp16 accumulator, 4096 seq len
**Tried:** kept `l_i` in fp16 to save registers
**Measured:** max rel error vs fp64 went 3.1e-3 → 2.7e-2. Latency unchanged.
**Concluded:** no speed win, 10× the error. Accumulating thousands of terms in 11 mantissa bits. All accumulators fp32, permanently. Matches the NumPy experiment in notes/01.

## 2025-XX-XX — exp2 vs exp
**Tried:** folded log2(e) into sm_scale, swapped `tl.exp` → `tl.exp2`
**Measured:** N=4096 causal fp16: 1.84ms → 1.79ms (2.7%)
**Concluded:** real but small. SASS confirms one `ex2.approx.f32` instead of a multiply + ex2. Worth having; not worth the two hours I spent convincing myself it wasn't measurement noise. Should have run the variance check first.

## 2025-XX-XX — backward with atomics
**Tried:** single backward kernel, `tl.atomic_add` into dK/dV
**Measured:** 14.2ms vs 4.1ms for the split-kernel version at N=4096
**Concluded:** 3.5× slower. Every Q block contends on the same KV block. Recomputing S twice is far cheaper than serializing on atomics. Keeping the atomic version behind a flag for the ablation table — the gap is the interesting part.

## 2025-XX-XX — N=2049 fails, N=2048 passes
**Tried:** nothing, found it in the prime-length test from task 04
**Measured:** max rel error 0.4 at N=2049, 3e-3 at N=2048
**Concluded:** boundary mask. Loading K with `other=0.0` means masked columns score 0, which is *higher* than most real scores, so they win the max and get weight. Masked columns need `-inf` before the max, not 0. The test caught it; without the prime-length case I'd have shipped it.
