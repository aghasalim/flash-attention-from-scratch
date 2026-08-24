# Online softmax

Everything in FlashAttention rests on one fact: softmax can be computed exactly in a single
streaming pass, holding a running max and a running sum instead of the whole row. This note is
the derivation, the proof that it is exact, the IO-complexity argument for why it is worth doing,
and four experiments with real numbers.

Reference implementation: `fa/ref/online_softmax.py`. NumPy only, no torch, no Triton.
Everything below was produced by

```
.venv/bin/python -m fa.ref.online_softmax
```

which runs the self-check and prints all five experiment blocks. Seeded, deterministic
(`np.random.default_rng(1234)`); two consecutive runs are byte-identical. The raw output is in
the appendix at the bottom.

---

## 1. Safe softmax, and why the shift is not optional

Softmax of a row `x ∈ R^N`:

```
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

Numerator and denominator both overflow long before the ratio does. The ratio is invariant to a
constant shift, because `exp(x_i - c) / Σ_j exp(x_j - c) = e^{-c}exp(x_i) / (e^{-c}Σ_j exp(x_j))`,
so with `m = max(x)`:

```
softmax(x)_i = exp(x_i - m) / Σ_j exp(x_j - m)
```

Now every exponent is `≤ 0`, so every `exp` is in `(0, 1]`, and the denominator is in `[1, N]`.
Nothing can overflow, in any float type, ever.

The range this buys matters more than it sounds. fp16's largest finite value is 65504, and
`log(65504) = 11.0899`, so `exp(x)` overflows fp16 at `x > 11.09` — `exp(12)` is already `inf`.
Attention scores are `q·k/√d`, and with `d = 64..128` and unnormalised activations they routinely
run into the tens. fp32 buys more headroom (`log(3.403e38) = 88.7`) but not unlimited headroom,
and experiment (c) below is a row that kills fp32 too.

The cost of the shift is that you need `max(x)` before you can exponentiate anything, and that
looks like it forces a pass over the whole row. The whole point of the online algorithm is that
it does not.

---

## 2. The streaming recurrence

Split the row into `T` blocks `x^(1), …, x^(T)`, block `j` having `n_j` entries, and split `V`
conformally into `V_j ∈ R^{n_j × d}` (`V` is the value matrix; for plain softmax read `V = I`).
Per-block statistics, computed from block `j` alone:

```
m̃_j = max(x^(j))
P̃_j = exp(x^(j) - m̃_j)          (a row vector of length n_j)
l̃_j = Σ_a P̃_j[a]
```

Running state, initialised `m_0 = -∞`, `l_0 = 0`, `O_0 = 0 ∈ R^d`:

```
m_j = max(m_{j-1}, m̃_j)
l_j = exp(m_{j-1} - m_j) · l_{j-1}  +  exp(m̃_j - m_j) · l̃_j
O_j = exp(m_{j-1} - m_j) · O_{j-1}  +  exp(m̃_j - m_j) · (P̃_j V_j)
```

Output after the last block: `O_T / l_T`. One division, at the end, once.

The state is `(m_j, l_j, O_j)` — two scalars and one `d`-vector per query row. It does not grow
with `N`. That is the entire trick: the `N×N` score matrix is never materialised, and neither is
the `N`-long probability row.

---

## 3. Proof that this is exact

**Claim.** For every `j ∈ {0, …, T}`, writing `B_j` for the set of indices in the first `j` blocks:

```
(i)    m_j = max_{a ∈ B_j} x_a
(ii)   l_j = Σ_{a ∈ B_j} exp(x_a - m_j)
(iii)  O_j = Σ_{a ∈ B_j} exp(x_a - m_j) · v_a
```

**Base case, `j = 0`.** `B_0 = ∅`. With the usual conventions `max ∅ = -∞` and empty sums `= 0`,
the initialisation `m_0 = -∞`, `l_0 = 0`, `O_0 = 0` satisfies (i)–(iii).

If the `-∞` bothers you, check `j = 1` directly instead: `m_1 = max(-∞, m̃_1) = m̃_1`,
`exp(m_0 - m_1) = exp(-∞) = 0` kills the stale term, `exp(m̃_1 - m_1) = 1`, so `l_1 = l̃_1` and
`O_1 = P̃_1 V_1`, which are (ii) and (iii) over `B_1`.

**Inductive hypothesis.** Assume (i)–(iii) hold at step `j-1`.

**Inductive step.** Take step `j`.

*(i)* `m_j = max(m_{j-1}, m̃_j) = max(max_{a ∈ B_{j-1}} x_a, max_{a ∈ block j} x_a) = max_{a ∈ B_j} x_a`,
since `B_j = B_{j-1} ⊔ block j`.

*(ii)* Substitute the hypothesis into the recurrence:

```
l_j = exp(m_{j-1} - m_j) · Σ_{a ∈ B_{j-1}} exp(x_a - m_{j-1})
    + exp(m̃_j   - m_j) · Σ_{a ∈ block j} exp(x_a - m̃_j)

    = Σ_{a ∈ B_{j-1}} exp(x_a - m_{j-1} + m_{j-1} - m_j)
    + Σ_{a ∈ block j} exp(x_a - m̃_j   + m̃_j   - m_j)

    = Σ_{a ∈ B_{j-1}} exp(x_a - m_j) + Σ_{a ∈ block j} exp(x_a - m_j)

    = Σ_{a ∈ B_j} exp(x_a - m_j)
```

The only identity used is `e^α · e^β = e^{α+β}`, which is exact in `R`. The old shift `m_{j-1}`
cancels against the correction factor and the new shift `m_j` is left. That cancellation is the
whole proof; everything else is bookkeeping.

*(iii)* Identical, with `v_a` carried along. `P̃_j V_j = Σ_{a ∈ block j} exp(x_a - m̃_j) v_a` by
definition of the matrix product, and scalar multiplication distributes over the sum, so the same
two lines go through with each term multiplied by `v_a`.

So (i)–(iii) hold at `j`, and by induction at every `j` up to `T`.

**Conclusion.** At `j = T`, `B_T` is the whole row, so `m_T = max(x) = m`, and

```
l_T = Σ_a exp(x_a - m)            (the safe-softmax denominator)
O_T = Σ_a exp(x_a - m) · v_a
```

Therefore

```
O_T / l_T = Σ_a [ exp(x_a - m) / Σ_b exp(x_b - m) ] · v_a = Σ_a softmax(x)_a · v_a
```

which is the attention output for that row. Exactly, in exact arithmetic — not an approximation,
not a bound, not "converges to". ∎

**Two things the proof did not use.** It never used that the blocks are contiguous, and it never
used their order. Any partition of the row, visited in any order, gives the same `(m_T, l_T, O_T)`.
That is why experiment (d) below holds, and it is also why the outer loop over query blocks is
embarrassingly parallel: each query block owns its own `(m_i, l_i, acc)` and never has to talk to
another block.

---

## 4. Why the backward pass only needs `L = m + log l`

The forward pass stores, per query row, one fp32 number:

```
L_i = m_i + log(l_i)
```

That single number reconstructs the entire probability row:
`P_ij = exp(S_ij - L_i)`, since `exp(S_ij - m_i - log l_i) = exp(S_ij - m_i)/l_i`, which is the
safe softmax. So the backward pass recomputes `S = QK^T` from `Q` and `K` (cheap: it is FLOPs, and
attention is not FLOP-bound) and gets `P` for free, instead of reading an `N×N` matrix back out of
HBM. `N` floats per head instead of `N²`. At `N = 8192` that is 8192 floats versus 67 million.

`logsumexp_rows(q, k, causal)` in the reference returns exactly this. It is the same loop nest as
`online_attention` with a width-1 zero `V`, because `V` does not enter `L` at all.

---

## 5. The rescale factor can only shrink

`m_j = max(m_{j-1}, m̃_j)`, so `m_j ≥ m_{j-1}` and `m_j ≥ m̃_j` by construction. Therefore

```
exp(m_{j-1} - m_j) ≤ 1     and     exp(m̃_j - m_j) ≤ 1
```

Both correction factors are in `(0, 1]`. The running max is monotone non-decreasing, so the
correction applied to what is already accumulated only ever scales it *down*. Combined with
`exp(x_a - m_j) ≤ 1` for every element of the current block, every quantity that gets
exponentiated anywhere in the recurrence has a non-positive argument. The recurrence cannot
overflow in any float type. The bound on `l` is `l_j ≤ |B_j| ≤ N`, so even fp16 has the *range*
for it up to `N = 65504` — fp16's problem is precision, not range, which is what experiment (b)
measures.

Underflow is possible and harmless: `exp(m̃_j - m_j)` flushing to zero means that block's
contribution is genuinely negligible next to the new max. In fp16 that happens at a score gap of
only about 17.3 (smallest fp16 subnormal is 6e-8), which is why fp16 accumulators lose the small
probabilities entirely — the `fp16 rel` column in experiment (b) hits exactly 1.000 when a
probability that should be nonzero comes back as 0.

---

## 6. Two implementation traps

**Short trailing blocks.** If `N % block_size != 0` the last block is short. The masked-out lanes
must contribute `-∞` to the max and `0` to the sum. Those are the same thing only if you use `-∞`:
padding the score tile with `0` makes `0` a candidate for the block max, and if the real scores are
negative — which they routinely are — the pad *becomes* the max and the whole row is wrong.
Experiment (e) measures it: `7.033e-02` max error and probabilities summing to `0.430484` instead
of 1, versus `3.469e-18` for the correct short-slice version. In NumPy the fix is to slice a
genuinely shorter block. In Triton it will be `tl.load(..., mask=..., other=-float('inf'))` on the
score tile — the `other` value is the thing to get right, and it is not zero.

**Causal masking is three zones, not one.** For query block rows `[q_start, q_end)` and KV block
cols `[kv_start, kv_end)` with mask `col ≤ row`:

| zone | condition | work |
|---|---|---|
| skip | `kv_start > q_end - 1` | entirely above the diagonal: do nothing, and break, since later blocks are too |
| dense | `kv_end - 1 ≤ q_start` | entirely below: full tile, no mask, no predication |
| diagonal | otherwise | straddles the diagonal: element-level mask, the only zone that pays for it |

Only `O(N/BLOCK)` blocks are diagonal, so masking cost is linear in the number of blocks while the
dense zone is quadratic. The classification lives in `causal_zone()` in the reference so task 06
can split the kernel along the same line.

---

## 7. IO complexity

Take `M` = size of SRAM in elements, `N` = sequence length, `d` = head dimension, and count HBM
accesses (`Θ(·)`, one head).

**Standard attention.** Read `Q, K, V`: `Θ(Nd)`. Compute `S = QK^T` and write it: `Θ(N²)`. Read `S`,
softmax it, write `P`: `Θ(N²)`. Read `P` and `V`, write `O`: `Θ(N² + Nd)`. Total:

```
Θ(N·d + N²)
```

The `N²` term dominates whenever `N > d`, which is always. Every one of those `N²` accesses exists
only because an intermediate was written to HBM and read straight back.

**Tiled attention.** The block sizes are chosen so a working set fits in SRAM: one `Q` block, one
`K` block, one `V` block and the accumulator together are `Θ(M)` elements, which puts the block
row-count at `B = Θ(M/d)`. The reference (and FlashAttention-2, and this project's kernel) loops
outer over `Q` blocks and inner over `KV` blocks:

```
number of Q blocks     = N / B          = Θ(N d / M)
HBM traffic per Q block = Θ(N d)                       (it streams all of K and V once)
```

so

```
Θ(N d / M) · Θ(N d) = Θ(N² d² / M)
```

That is where `d²/M` comes from: **one factor of `d` from the block count** (blocks hold `M/d` rows,
so there are `Nd/M` of them) **and one from the pass over K and V that each block performs**
(`Nd` elements). FlashAttention-1 loops the other way round — outer over KV blocks, inner over `Q`,
re-reading `Q` and `O` on every KV block — and lands on the same `Θ(N²d²/M)`; the loop order changes
which operand gets re-read and how much rescaling work happens, not the asymptotics.

**Why this is an improvement even though the FLOPs are identical.** Both versions do `2N²d` FLOPs
for `QK^T` and `2N²d` for `PV`. Not one multiply is saved. What changes is the traffic:
`N²d²/M` versus `N²`, a factor of `M/d²`. Attention at these shapes is memory-bound, not
compute-bound, so the wall-clock is set by the traffic term and not the FLOP term — the fused kernel
wins by never writing `S` and `P` to HBM at all.

**The condition.** `N²d²/M < N²` exactly when `M > d²`. For `d = 64`, `d² = 4096` elements = 16 KB in
fp32; for `d = 128`, 64 KB. Shared memory per SM on real datacenter GPUs is in the 100–228 KB range,
so `M >> d²` comfortably holds and the inequality is not close. It is worth noticing that it *is* a
condition and not a law: at very large head dimension the tiling stops paying, which is the same
reason the kernel needs `BLOCK_M`/`BLOCK_N` tuned per `d` rather than one config for everything.

Everything in this section is an operation count, not a measurement. The actual speedup of a fused
kernel over an unfused one on a real GPU is
**not measured on this hardware (no CUDA device; developed on Apple M4)** — that number belongs to
tasks 01 and 07, on a machine that has tensor cores.

---

## 8. Numerical experiments

All four were run; every number below is copied from the run whose full output is in the appendix.

### (a) Online vs. `scipy.special.softmax`, fp64

1000 rows × N=1024, `x ~ 4·N(0,1)`, fp64 throughout, max over all 1024000 entries:

| block_size | divides 1024 | max abs deviation |
|---|---|---|
| 64 | yes | 5.551e-16 |
| 128 | yes | 4.441e-16 |
| 100 | no | 5.551e-16 |
| 333 | no | 5.551e-16 |

At fp64 the agreement is a few `ulp` of 1.0 (`eps = 2.22e-16`), and the block size — including block
sizes that leave a short trailing block — makes no difference. That is the proof in section 3
showing up as a number. The module self-check asserts `< 1e-14` for block sizes 1, 7, 64, 128, 333,
1000 and 4096 and reports `0.000e+00`.

### (b) fp32 vs fp16 accumulators — why rule 5 exists

Input is fp16 in both arms, so the input quantisation error is identical and cancels out of the
comparison; the reference is the fp64 softmax **of that same fp16 input**. The only difference
between arms is the dtype of `m`, `l` and the `exp` arithmetic. 64 rows per `N`, `block_size = 128`,
`x ~ 2·N(0,1)`.

`abs` = max `|p̂ - p|`. `rel` = max `|p̂ - p| / p`. `|sum-1|` = max over rows of `|Σ p̂ - 1|`, which is
the relative error of the denominator `l` and is the sharpest look at the accumulator itself.

| N | fp32 abs | fp16 abs | fp32 rel | fp16 rel | fp32 \|sum-1\| | fp16 \|sum-1\| | abs gap | sum gap |
|---|---|---|---|---|---|---|---|---|
| 128 | 9.552e-08 | 3.442e-04 | 2.611e-07 | 3.995e-02 | 1.627e-07 | 5.172e-04 | 3603x | 3178x |
| 512 | 2.874e-08 | 2.547e-04 | 2.782e-07 | 1.907e-01 | 1.444e-07 | 7.994e-04 | 8864x | 5536x |
| 1024 | 1.544e-08 | 1.739e-04 | 2.862e-07 | 9.990e-01 | 1.500e-07 | 7.674e-04 | 11260x | 5116x |
| 2048 | 5.869e-08 | 1.666e-04 | 6.230e-07 | 1.000e+00 | 2.431e-07 | 1.209e-03 | 2839x | 4974x |
| 4096 | 2.270e-08 | 1.389e-04 | 5.893e-07 | 1.052e+00 | 2.592e-07 | 1.997e-03 | 6120x | 7706x |
| 8192 | 4.755e-08 | 1.568e-04 | 6.544e-07 | 1.268e+00 | 2.805e-07 | 2.301e-03 | 3297x | 8202x |

**The number to remember: at N=8192 the fp16 accumulator is wrong by 1.568e-04 absolute against
4.755e-08 for fp32, a gap of 3297x. On the denominator it is 2.301e-03 against 2.805e-07, a gap of
8202x.**

Reading the columns:

- **fp32 holds.** Every fp32 column is flat in `N`, at a few times `fp32 eps = 1.19e-07`. Growing the
  row 64x does not degrade it measurably.
- **fp16 degrades, and the denominator degrades monotonically.** `|sum-1|` for fp16 climbs
  5.172e-04 → 2.301e-03 from N=128 to N=8192 while fp32 sits at ~2e-07. Each term added into `l`
  costs one fp16 rounding (`fp16 eps = 9.77e-04`); at N=8192 there are 8192 of them, partly
  cancelling, and the residue is 2.3e-03.
- **`rel` hits 1.000 and then exceeds it.** A relative error of exactly 1 means a probability that
  should be nonzero came back as 0 — the fp16 `exp` underflowed. Past 1 means it came back with the
  wrong magnitude entirely. From N=1024 upward, fp16 is not merely imprecise on the small
  probabilities, it has lost them.
- **The `abs` column is the least informative one** and I nearly reported only it. These score
  distributions stay peaked (max probability is 0.66 at N=128 and 0.27 at N=8192), so the absolute
  error does not grow with `N` even while the accumulator gets steadily worse. If you only look at
  max-abs-error you will conclude fp16 accumulation is fine at long context. It is not.

This is the experiment behind repo rule 5: **inputs may be fp16/bf16, but `acc`, `m_i`, `l_i` and `D`
are fp32 always.** If a future kernel's error jumps by three to four orders of magnitude, an fp16
accumulator is the first thing to check — a `tl.zeros(..., dtype=tl.float16)` or a missing
`.to(tl.float32)` on the accumulator update.

### (c) Adversarial row: one score +100, the rest -100

`N = 1024`, `x[17] = +100`, all other entries `-100`. Exact answer: `p[17] = 1 - O(1e-84)`,
everything else `~1e-87`.

| arm | dtype | result |
|---|---|---|
| naive `exp(x)/Σexp(x)` | fp32 | `Σ exp(x) = inf`, peak prob `nan`, 1 non-finite entry of 1024 |
| naive `exp(x)/Σexp(x)` | fp16 | `Σ exp(x) = inf`, peak prob `nan`, 1 non-finite entry of 1024 |
| online (max-shifted) | fp32 | peak prob `1.0`, max err vs fp64 exact `1.384e-87` |
| online (max-shifted) | fp16 | peak prob `1.0`, max err vs fp64 exact `1.384e-87` |

The failure mode is worth being precise about: the naive version does not return a merely
inaccurate answer, it returns `nan` **at the one index that carries all the probability mass**.
Everything else comes back as a clean `0` and the row looks almost right. In fp64 the same row is
fine (`exp(100) = 2.688e+43`, well inside fp64's range), so this bug is invisible until you switch
to the dtype you actually want to run in. The online version is exact in both, because after the
shift the largest exponent is `exp(0) = 1`.

### (d) Block-order invariance

N=512, d=64, `BLOCK_M=128`, `BLOCK_N=64`, so 8 KV blocks; non-causal; fp64. Permuting whole KV
blocks of `K` and `V` identically is the same thing as visiting the KV blocks in a different order,
since block `j` of the permuted tensors is block `perm[j]` of the originals. 20 random permutations:

```
max |permuted - in-order| = 2.220e-16
```

which is one `ulp` of 1.0. Asserted `< 1e-12` in the module. This is the empirical face of the
"the proof never used block order" remark in section 3, and it is the property that makes the
grid-axis loop safe to run in any order the GPU happens to schedule it.

### (e) Uneven blocks: `-∞` versus zero-fill

N=1000, `block_size=128`, so the last block is 104 wide. Scores drawn in `[-12.46, 0.81]` — all
negative except the tail, which is the realistic case.

| handling of the short block | max abs deviation from `scipy.softmax` | `Σ p` |
|---|---|---|
| short slice (correct) | 3.469e-18 | 1.000000 |
| padded with 0.0 (the bug) | 7.033e-02 | 0.430484 |

Zero-padding does not produce a slightly wrong answer, it produces an answer that is wrong by 7e-02
and does not even sum to 1, because the pad value `0` became the row max and rescaled everything
real by `exp(x_a - 0)` instead of `exp(x_a - 0.81)`. This is the gotcha in the spec and it is now a
number rather than a warning.

---

## Appendix: raw run output

```
$ .venv/bin/python -m fa.ref.online_softmax
[ok] online_softmax vs scipy, fp64, block sizes 1..4096: max err 0.000e+00
[ok] online_attention vs direct softmax(QK^T)V, fp64, N in (133,256,512), 6 block shapes incl. non-dividing, causal+non-causal: max err 7.216e-16
[ok] logsumexp_rows vs scipy.special.logsumexp, causal+non-causal: max err 8.882e-16

(a) online vs scipy.special.softmax, fp64, 1000 rows x N=1024
  block_size   64                           max |online - scipy| = 5.551e-16
  block_size  128                           max |online - scipy| = 4.441e-16
  block_size  100  (does not divide 1024)   max |online - scipy| = 5.551e-16
  block_size  333  (does not divide 1024)   max |online - scipy| = 5.551e-16

(b) fp32 vs fp16 accumulators, input fp16, reference = fp64 softmax of the same fp16 input
    64 rows per N, block_size=128
             N |     fp32 abs |     fp16 abs |     fp32 rel |     fp16 rel | fp32 |sum-1| | fp16 |sum-1| |      abs gap |      sum gap
  -------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+-------------
           128 |    9.552e-08 |    3.442e-04 |    2.611e-07 |    3.995e-02 |    1.627e-07 |    5.172e-04 |        3603x |        3178x
           512 |    2.874e-08 |    2.547e-04 |    2.782e-07 |    1.907e-01 |    1.444e-07 |    7.994e-04 |        8864x |        5536x
          1024 |    1.544e-08 |    1.739e-04 |    2.862e-07 |    9.990e-01 |    1.500e-07 |    7.674e-04 |       11260x |        5116x
          2048 |    5.869e-08 |    1.666e-04 |    6.230e-07 |    1.000e+00 |    2.431e-07 |    1.209e-03 |        2839x |        4974x
          4096 |    2.270e-08 |    1.389e-04 |    5.893e-07 |    1.052e+00 |    2.592e-07 |    1.997e-03 |        6120x |        7706x
          8192 |    4.755e-08 |    1.568e-04 |    6.544e-07 |    1.268e+00 |    2.805e-07 |    2.301e-03 |        3297x |        8202x

(c) adversarial row: naive softmax overflows, the max-shift does not
  row: one score +100, 1023 scores -100
  float32: naive exp(x).sum() = inf, naive peak prob = nan, non-finite entries = 1/1024
  float32: online peak prob = np.float64(1.0), max |online - fp64 exact| = 1.384e-87
  float16: naive exp(x).sum() = inf, naive peak prob = nan, non-finite entries = 1/1024
  float16: online peak prob = np.float64(1.0), max |online - fp64 exact| = 1.384e-87
  fp64 naive does not overflow: exp(100) = 2.688e+43 (fp32 max 3.403e+38, fp16 max 6.550e+04, so fp16 overflows above x = 11.0899)

(d) block-order invariance, non-causal
  20 random permutations of the 8 KV blocks, N=512 d=64 BLOCK_M=128 BLOCK_N=64, fp64: max |permuted - in-order| = 2.220e-16

(e) uneven blocks: zero-fill vs -inf
  N=1000, block_size=128 (last block 104 wide), scores in [-12.46, 0.81]
  short slice (correct):        max |online - scipy| = 3.469e-18
  zero-padded last block (bug): max |online - scipy| = 7.033e-02, probabilities sum to 0.430484 instead of 1
```

Machine: Apple M4, macOS, Python 3.14.4, NumPy 2.5.2, SciPy 1.18.1. This task is CPU-only by
design (the spec forbids GPU code here), so nothing in this note is hardware-limited — every
number above is reproducible anywhere NumPy runs.
