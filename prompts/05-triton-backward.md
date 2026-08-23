# Task 05 — The Triton backward pass

**Wave:** 3 (parallel with 06 and 07)
**OWNS:** `fa/triton/bwd.py`, `fa/ops/autograd.py`
**READS:** `fa/triton/fwd.py`, `tests/`, `notes/01-online-softmax.md`

## Context

The backward pass is harder than the forward pass and it's where most from-scratch attempts stall. The forward saved only `O` and the row logsumexp `L` — not `S`, not `P`. So backward has to *recompute* the scores from Q and K, reconstruct `P = exp(S - L)` (one exp, no max-tracking needed, because `L` already contains it), and then do five matmuls.

The parallelization is the crux. Read the "wrong version first" note below before writing any code — it will save you a week.

## The math

Given `dO`, with `D = rowsum(dO ∘ O)` precomputed:

```
S  = Q Kᵀ · scale
P  = exp(S - L)              # L from forward, per row; no running max needed
dV = Pᵀ dO
dP = dO Vᵀ
dS = P ∘ (dP - D)            # the softmax Jacobian, collapsed
dQ = dS K · scale
dK = dSᵀ Q · scale
```

Derive `dS = P ∘ (dP - D)` in `notes/02-backward.md` from the softmax Jacobian `∂p_i/∂s_j = p_i(δ_ij - p_j)`, and show why the full `N×N` Jacobian collapses to an elementwise operation plus one row-reduction. That collapse is the entire reason the backward pass is tractable and it's the derivation to be able to reproduce.

## Task

**Three kernels, not one.**

**1. `_attn_bwd_preprocess`** — computes `D = rowsum(dO ∘ O)`, shape `(B, H, N)`. Trivially parallel over rows. Separate kernel because both main kernels need it.

**2. `_attn_bwd_dkdv`** — grid over **KV** blocks, inner loop over Q blocks. Each program owns one `BLOCK_N` slice of K and V, accumulates `dK` and `dV` in fp32 registers, writes once at the end.

**3. `_attn_bwd_dq`** — grid over **Q** blocks, inner loop over KV blocks. Each program owns one `BLOCK_M` slice of Q, accumulates `dQ`, writes once.

**Why two kernels instead of one.** A single kernel parallelized over Q blocks must accumulate into `dK` and `dV`, which every Q block touches. That means `tl.atomic_add`, which means serialization on contended addresses, which means a backward pass several times slower than it should be. Splitting by which output tensor each kernel owns eliminates atomics entirely at the cost of recomputing `S` twice. Recomputation is cheap; atomic contention is not. **Write the atomic version first and measure it** — the number is worth having, it goes in the ablation table in task 11, and it's the kind of thing that makes a project look like real engineering rather than a transcription of a paper.

**4. `fa/ops/autograd.py`** — `torch.autograd.Function` subclass. `forward` saves `q, k, v, o, L` and the `sm_scale` / `causal` flags. `backward` calls the three kernels. Rewire `fa/ops/attention.py` to route through it.

## Acceptance criteria

- `torch.autograd.gradcheck` passes in fp64 for small shapes — un-xfail `tests/test_gradients.py`
- dQ, dK, dV each match the fp64 reference no worse than naive fp16 backward does
- No `tl.atomic_add` in the final kernels
- Backward is between 2× and 4× forward latency. Analytically it's 2.5× the FLOPs plus recomputation; outside that band, something is wrong
- Memory still O(N)
- Correct under causal masking for all the shapes in the task 04 sweep
- The atomic-vs-split timing comparison is recorded in the logbook and in `notes/02-backward.md`

## Gotchas

- `dQ` and `dK` both need the `sm_scale` factor; `dV` does not. This asymmetry is easy to get wrong and produces gradients that are correct up to a constant — which passes eyeball inspection and fails gradcheck.
- Under causal masking the recomputed `S` must be masked *identically* to the forward pass. A different masking convention between fwd and bwd gives you a kernel that trains and slowly diverges, which is the worst possible failure mode.
- `D = rowsum(dO ∘ O)` uses the **final normalized** `O`, not the unnormalized accumulator. Using the pre-division value gives a subtly wrong `dS`.
- If gradcheck fails on exactly one of the three tensors, you've localized the bug — that's why task 04 tests them separately.
- fp32 accumulators for `dK` and `dV`. Same rule as forward, same failure mode.

## Finish by

Adding a LOGBOOK entry: gradcheck status, backward/forward latency ratio, atomic vs. split-kernel timings, and per-tensor max error for dQ/dK/dV.
