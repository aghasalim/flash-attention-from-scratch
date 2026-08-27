# Attention is a memory-bandwidth problem: measuring the premise before building the kernel

**Aghasalim Mustafazada** · [github.com/aghasalim](https://github.com/aghasalim) · August 2026

---

## Abstract

This is a from-scratch reimplementation of IO-aware attention (FlashAttention),
built in waves against a fixed set of task specs. This write-up covers what is
built and measured: the online-softmax derivation and its proof of exactness, a
NumPy reference structured as the eventual kernel, a 500-test correctness harness
written before any kernel exists, and a roofline analysis of the unfused
baselines. All measurement was done on an **Apple M4 (10 GPU cores, 25.77 GB
unified, no CUDA device)**.

The headline result is more careful than the one I expected to write. The unfused
baselines sit at an arithmetic intensity of 31.51 (naive) and 29.47 (chunked)
against a median ridge point of 30.91 FLOP/byte, which is to say they sit *on*
the balance point, not clearly to the memory side of it. Thermal variance puts the
ridge anywhere in 20.08 to 40.55, and 31.51 is inside that band, so **this hardware
cannot decisively call naive attention memory-bound** (§4.3). What survives the
noise by a wide margin is the comparison that actually matters: a fused
implementation that never writes`S` and`P` to memory sits at 2048 FLOP/byte, 66×
the ridge, and the memory cliff at N=4096 is a 37.95× latency inversion with an
OOM 1024 tokens later. The online-softmax recurrence is exact,
verified to 7.216e-16 against a direct fp64 computation, so any error a kernel
later shows is arithmetic rather than algorithmic. Keeping accumulators in fp32
rather than fp16 is worth a factor of **3297×** in error at N=8192.

The Triton and CUDA C++ kernels are **not implemented**. Triton publishes no
macOS wheel and there is no NVIDIA GPU here, so tasks 03 and 05 to 10 of the plan
are blocked on hardware rather than on effort. Nothing in this document is
extrapolated to hardware that was not measured.

## 1. Background: why anyone fuses attention

Standard attention computes`S = QKᵀ/√d`,`P = softmax(S)`,`O = PV`. The two
matmuls are`4·B·H·N²·D` FLOPs. The problem is not the FLOPs, it is that`S` and
`P` are each`B·H·N²` elements that get written to memory and read back, purely as
intermediates the caller never wants.

The ratio of score traffic to parameter traffic is`N/D`. At`D=64` the scores
pass everything else at`N=64`, and by`N=4096` they are 32× the traffic of`Q`,
`K`,`V` and`O` combined (§4.2). Fusing softmax into the matmul so that`S` and
`P` never leave the chip removes that term entirely.

This is a well-known argument. The point of §4 is that I did not want to take it
on faith, and it is measurable on hardware I own.

## 2. Method

### 2.1 Online softmax

Safe softmax subtracts the row max:`softmax(x)_i = exp(x_i − m)/Σ_j exp(x_j − m)`,
`m = max(x)`. The shift is not optional, fp16 tops out at 65504 and`exp(12)`
already overflows it, while attention scores routinely exceed that magnitude at
large`d`.

The streaming form processes one block at a time, carrying a running max`m` and
running sum`l`. For block`j` with block statistics`m̃_j`,`l̃_j`:

```
m_j = max(m_{j-1}, m̃_j)
l_j = exp(m_{j-1} − m_j)·l_{j-1} + exp(m̃_j − m_j)·l̃_j
O_j = exp(m_{j-1} − m_j)·O_{j-1} + exp(m̃_j − m_j)·(P̃_j V_j)
```

`O_T / l_T` equals the true attention output **exactly** in exact arithmetic; the
induction proof is in [`notes/01-online-softmax.md`](01-online-softmax.md). The
correction factor`exp(m_{j-1} − m_j) ≤ 1` always, since`m_j ≥ m_{j-1}` by
construction, the rescale only ever shrinks, so the recurrence cannot overflow.

This matters more than it looks. Because the algorithm is exact, any discrepancy a
kernel shows against a reference is an arithmetic defect, precision, ordering,
masking, and never the algorithm itself. That collapses the search space when
debugging a kernel.

### 2.2 IO complexity

Standard attention needs`Θ(N·d + N²)` HBM accesses. Tiled attention with SRAM of
size`M` needs`Θ(N²d²/M)`. The`d²/M` arises from the number of KV blocks times
the passes over`Q`. The FLOP count is unchanged; only the traffic moves. It is an
improvement exactly when`M ≫ d²`, which is the regime real accelerators are in.

### 2.3 Backward, derived but not implemented

Forward saves only`O` and the per-row logsumexp`L = m + log(l)`, one float per
row instead of an`N×N` matrix. Backward recomputes`S` from`Q`,`K` and
reconstructs`P = exp(S − L)`; no running max is needed because`L` already
contains it. With`D = rowsum(dO ∘ O)`:

```
dV = Pᵀ dO      dP = dO Vᵀ      dS = P ∘ (dP − D)
dQ = dS K·scale                 dK = dSᵀ Q·scale
```

The softmax Jacobian`∂p_i/∂s_j = p_i(δ_ij − p_j)` is`N×N` per row, but
contracting it against`dP` collapses to an elementwise product and one row
reduction, which is the only reason the backward pass is tractable at all.`dQ`
and`dK` carry the`sm_scale` factor and`dV` does not; that asymmetry produces
gradients wrong by a constant if missed, which passes inspection and fails
gradcheck.

Five matmuls of the same shape gives`10·B·H·N²·D`, i.e. **2.5× forward**. This is
the analytic target a backward implementation has to land near. It is not measured
here:`not measured on this hardware (no CUDA device; developed on Apple M4)`.

## 3. Implementation

### 3.1 What exists

| component | file | status |
|---|---|---|
| hardware fingerprint |`scripts/env.py` | full NVIDIA path + Apple path; run |
| fp64 ground truth |`fa/ref/fp64.py` | run |
| naive / chunked / sdpa baselines |`fa/ref/naive.py` | run |
| online-softmax reference (NumPy) |`fa/ref/online_softmax.py` | run |
| correctness harness |`tests/` (500 tests) | run |
| roofline harness |`bench/roofline.py` | run |

`fa/ref/online_softmax.py` is deliberately structured as the eventual Triton
kernel, outer loop over Q blocks (the grid axis), inner sequential loop over KV
blocks, fp32 accumulators, causal handled as three block zones (skip / dense /
diagonal-masked) rather than one masked loop. Task 03 ports it rather than
inventing it.

### 3.2 The harness came first, on purpose

The test suite was written against the references *before* any kernel existed. A
harness written after a kernel tends to encode that kernel's bugs as expected
behaviour. Everything kernel-dependent is`xfail`, 192 of the 500 tests, and
those are xfail for lack of a GPU, not for lack of a test.

The comparison bar is **relative**, not absolute:

```python
err_kernel = (out.double() - ref_fp64).abs().max()
err_naive  = (naive_fp16.double() - ref_fp64).abs().max()
assert err_kernel <= 2.0 * err_naive
```

An absolute tolerance is either loose enough to pass broken kernels or tight
enough to fail correct ones, because attention's error grows with`N` and with
score magnitude. "No worse than the naive implementation at the same precision" is
the only bar that survives that. The measured starting values, what any kernel
must clear, are fp16 4.753e-04 and bf16 2.039e-03 at N=4096, D=64, non-causal.
bf16 is 5.9× worse than fp16 on identical inputs (8 mantissa bits vs 11), which is
why it gets its own tolerance rather than sharing one.

Ground truth is fp64, never`F.scaled_dot_product_attention`. SDPA *is*
FlashAttention; if the kernel and SDPA shared a bug it would be invisible.

### 3.3 What does not exist

Tasks 03 (Triton forward), 05 (backward), 06 (causal/masks), 07 (autotune), 08
(GQA/varlen/dropout), 09 (Flash-Decoding/paged), 10 (CUDA C++) are not
implemented.`pip install triton` on macOS returns *No matching distribution
found*, there is no darwin wheel, and there is no NVIDIA GPU, nvcc or`ncu`
here. Writing kernels that cannot be compiled or tested would produce six files
that look finished and are unverified, which the project's own rule 3 (*measured
or absent*) forbids.

## 4. Evaluation

Config throughout:`B=4, H=32, D=64`. MPS rows are fp16, CPU rows are fp32 (arm64
torch has no fast half GEMM: 3.3 GFLOP/s vs 1652, a 500× gap that would measure
the dispatch path). 92 rows in`results/roofline.csv`, 834.6 s sweep.

### 4.1 The machine

| quantity | MPS | CPU |
|---|---:|---:|
| copy-kernel bandwidth | 95.86 GB/s | 101.29 GB/s |
| fp16 matmul | 2963.5 GFLOP/s | 3.3 GFLOP/s |
| fp32 matmul | 2542.8 GFLOP/s | 1738.3 GFLOP/s |
| **ridge point (median)** | **30.91 FLOP/byte** | **17.16 FLOP/byte** |
| **ridge point (range)** | **20.08 to 40.55** |, |

The range matters and is the reason §4.3 is hedged. Run-to-run spread on the same
fp16 matmul is large and thermal: within a single fingerprint run the min, max was
1937 to 3793 GFLOP/s, and separate runs an hour apart returned medians of 3142.6 and
2963.5. Bandwidth is stable by comparison (93.5 to 96.5 GB/s), so essentially all the
ridge-point uncertainty comes from peak compute.

CUDA equivalents, HBM bandwidth, tensor-core TFLOP/s, SM count, measured-vs-
advertised ratio: `not measured on this hardware (no CUDA device; developed on
Apple M4)`.

### 4.2 The byte argument, arithmetically

fp16,`B=4 H=32 D=64`:

| N | Q,K,V,O | S,P | ratio |
|---:|---:|---:|---:|
| 1024 | 0.062 GiB | 0.500 GiB | 8× |
| 4096 | 0.250 GiB | 8.000 GiB | **32×** |
| 16384 | 1.000 GiB | 128.000 GiB | 128× |

### 4.3 Arithmetic intensity vs the ridge

Measured at N=4096, MPS fp16, non-causal, against a ridge point of 30.91
(band 20.08 to 40.55):

| implementation | AI (FLOP/byte) | position |
|---|---:|---|
| naive | 31.51 | **inside the ridge band, indeterminate** |
| chunked | 29.47 | inside the ridge band, indeterminate |
| fused ideal | 2048.00 | **66× the ridge, decisively compute-bound** |

**This is not the result I expected and it is worth stating plainly.** I set out to
confirm that attention is memory-bound, and on this machine the roofline does not
support that claim for the unfused baselines: 31.51 against a ridge of 30.91 is a
2% difference, and the ridge's own uncertainty is ±35%. Naive attention here is
*balanced*, sitting on the knee of the roofline, and which side of it you land on
depends on how warm the laptop is.

Two things do survive. The fused point at 2048 is 66× the ridge, no amount of
thermal drift moves that. And the roofline is not the only evidence: §4.4's 37.95×
latency inversion and §4.5's OOM are direct measurements that do not depend on a
ridge point at all.

Naive's AI is essentially constant in`N` (31.51 → 31.75 → 31.88 across a 4× range)
and asymptotes at`4D/2e = 32`. Longer sequences do not make naive attention more
compute-dense; they only make it move more bytes. That asymptote is a property of
the algorithm and`D`, not of the machine, and it is why the comparison against a
*fused* implementation is the load-bearing one:`D=64` fixes unfused attention near
32 FLOP/byte on any hardware, so whether that is memory-bound is entirely a
question about the machine's balance point.

### 4.4 Latency and the memory cliff

MPS fp16, non-causal, median:

| N | naive | chunked | naive ÷ chunked |
|---:|---:|---:|---:|
| 512 | 11.2 ms | 20.1 ms | 0.56× |
| 1024 | 43.1 ms | 74.7 ms | 0.58× |
| 2048 | 174.2 ms | 294.6 ms | 0.59× |
| 4096 | 46545.5 ms | 1226.6 ms | **37.95×** |
| 8192 | OOM | 4785.5 ms |, |
| 16384 | OOM | 32669.5 ms |, |

Latency tracks`N²` exactly while it fits (ratios 3.84×, 4.04× for successive
doublings), then naive goes from 174 ms to 46.5 s, a 267× jump for 4× the work,
as 8 GiB of score traffic stops fitting the 17.76 GiB working set and the run goes
to swap. Peak for that config is 21.7 GB.

### 4.5 OOM threshold: the prediction was wrong, and usefully so

Ladder in steps of 256: last OK`N=4864`, first OOM`N=5120`.

The textbook model counts two`N²` tensors (`S`,`P`) against the budget and
predicts`N≈6103`, **+19.2%, outside the 15% acceptance bar**. Rather than widen
the bar: one fp16`N²` tensor at N=5120 is 6.25 GiB, and the allocator reported
19.75 GiB in flight at failure, implying **~3.16 concurrent`N²` tensors**.

| model | predicted N | vs first OOM |
|---|---:|---:|
| 2 tensors | 6103 | +19.2% |
| **3 tensors** | **4983** | **−2.7%** |
| 4 tensors | 4315 | −15.7% |

The third tensor is the`masked_fill` allocation, held alongside`S` and`P` in
the autograd graph. The textbook byte model under-counts any real implementation
by exactly the intermediates the framework materialises.

## 5. Ablations

Every cell is measured on this machine or explicitly absent. Command for each is
in`notes/LOGBOOK.md`.

| ablation | config | result | note |
|---|---|---|---|
| **fp32 vs fp16 accumulator** | N=8192, fp16 in | **3297× abs error, 8202× on the denominator** | 4.755e-08 → 1.568e-04. fp16 rel error hits 1.000 from N=1024, i.e. small probabilities return as literal 0 |
| **masked-block fill: −inf vs 0** | N=1000, block 128 | **7.033e-02 error vs 3.469e-18** | zero-fill gives probabilities summing to 0.430484 |
| **tiling (chunked vs naive)** | N≤2048 | **0.56 to 0.59×, slower** | AI 29.47 vs 31.51: tiling moves intensity the *wrong* way |
| **tiling (chunked vs naive)** | N=4096 | **37.95× faster** | the same change, past the memory wall |
| **causal block skipping** | N=16384 | **2.02× (sdpa) vs 0.98× (chunked)** | implementations that skip blocks approach the theoretical 2×; ones that mask a dense N² get *slower* |
| **causal block skipping** | N=4096 | 1.89× (sdpa) vs 0.93× (chunked) | same shape at 4× shorter context |
| block-order invariance | N=512, 20 perms | 2.220e-16 | confirms the recurrence is order-independent |
| backward: atomic vs split kernels |, |`not measured on this hardware (no CUDA device; developed on Apple M4)` | task 05 |
|`exp2` vs`exp` |, |`not measured on this hardware (no CUDA device; developed on Apple M4)` | task 03 |
| BLOCK_M / BLOCK_N / num_stages / num_warps sweeps |, |`not measured on this hardware (no CUDA device; developed on Apple M4)` | task 07 |
| smem swizzling,`cp.async` |, |`not measured on this hardware (no CUDA device; developed on Apple M4)` | task 10 |
| Flash-Decoding vs standard |, |`not measured on this hardware (no CUDA device; developed on Apple M4)` | task 09 |
| GQA vs MHA KV memory |, |`not measured on this hardware (no CUDA device; developed on Apple M4)` | task 08 |
| varlen vs padded |, |`not measured on this hardware (no CUDA device; developed on Apple M4)` | task 08 |

### 5.1 Where the causal result comes from

The interesting row is causal skipping, because it is measurable *without* writing
a kernel, by comparing implementations that skip against ones that do not.
`naive` and`chunked` compute the full`N²` and mask afterwards; under causal they
run at **0.91 to 0.98×**, i.e. slightly slower, since masking is extra work for
identical traffic. The SDPA path, which classifies blocks and skips the
fully-hidden ones, runs at **1.69 to 2.02×**, approaching the theoretical 2× and
getting closer as`N` grows and the diagonal becomes a smaller fraction of the
triangle. That gap *is* the value of block skipping, isolated.

## 6. Limitations

- **No kernel.** The central artifact of the project does not exist. Everything
  above is baselines, references, harness and analysis.
- **Every measurement is Apple-silicon.** MPS unified memory has a
  compute-to-bandwidth ratio unlike any discrete GPU. The *shape* of the results
  (cliff at the working-set limit, fusion moves AI by two orders of magnitude)
  should transfer; not one number will. In particular the roofline verdict for the
  unfused baselines is indeterminate here (§4.3) and would likely be decisive on a
  discrete GPU, where the ridge point is far higher, an A100 at roughly 312
  TFLOP/s over 1.5 TB/s sits near 200 FLOP/byte, which would put unfused attention
  at 31.5 unambiguously in the memory-bound region. That arithmetic is a
  spec-sheet inference, not a measurement, and is offered as motivation for
  re-running this on real hardware rather than as a result.
- **The ridge point is noisy enough to change a conclusion.** ±35% on peak
  compute, from thermal throttling. Any single-run roofline verdict near the knee
  on this machine should be distrusted.
- **`sdpa_kernel` is a no-op on MPS.** Forcing a backend silently does nothing`sdpa_backend_honored=False` in the CSV. The "sdpa" rows are whatever kernel the
  device chose, and are labelled`NOT HONORED` rather than reported as a MATH
  measurement. On CPU the same probe correctly raises.
- **No per-config peak memory on CPU.** torch exposes no resettable per-device
  peak allocator stat there.
- **Naive causal at N=4096 is unstable.** 4372 ms against 46545 ms non-causal
  a 10.65× "speedup" that is really thrash variance near the memory limit, not a
  causal saving. Recorded rather than quietly dropped.
- **Backward is analytic only.** Derived in §2.3, never executed.
- **fp16 not swept on CPU**; 500× slower than fp32 there.

## 7. Related work

- **Milakov & Gimelshein (2018)**, *Online normalizer calculation for softmax*
  the two-page result the whole thing rests on. §2.1 is this.
- **Rabe & Staats (2021)**, *Self-attention Does Not Need O(n²) Memory*, the
  memory result without the IO framing. This repo's`chunked_attention` is
  essentially their construction, and §4.4 shows it buys memory and not bandwidth.
- **Dao et al. (2022)**, *FlashAttention*, adds IO-awareness: the point is not
  just avoiding the`N²` allocation but avoiding the`N²` *traffic*. §4.3 is the
  measurement of that distinction.
- **Dao (2023)**, *FlashAttention-2*, work partitioning, and the outer-Q /
  inner-KV loop order that`fa/ref/online_softmax.py` is written in.
- **Shah et al. (2024)**, *FlashAttention-3*, warp specialisation, FP8, Hopper.
  Out of reach without an H100.
- **Kwon et al. (2023)**, *PagedAttention/vLLM*, task 09's block-table design.

## 8. What I would do differently

**The task order was right, and I would defend it.** Building the harness (04) and
the math (02) before the kernel (03) meant that when the kernel becomes possible
there is a known-good oracle waiting. The alternative, kernel first, tests after
produces tests that agree with whatever the kernel does.

**I should have checked hardware feasibility before scoping.** The plan assumed a
CUDA box throughout. Discovering at task 00 that 8 of 12 tasks were unrunnable
should have happened before the specs were written, not after.

**I trusted`sdpa_kernel` without verifying it was honored.** It failed silently
and plausibly. Four columns of the CSV would have carried "MATH backend" numbers
that were nothing of the sort. The general lesson, verify that a knob you turned
actually did something, is the same one behind the fresh-clone check that later
found an undeclared scipy dependency and a`make lint` target that disagreed with
CI.

**I had the value of tiling backwards.** I expected chunking to be the win and it
is 0.56× at every size that fits. The win is fusion; tiling is its prerequisite.
That is obvious in hindsight and was not obvious to me before §4.4.

## 9. Honest self-assessment

Of what is in this repo: the online-softmax derivation, the exactness proof, the
IO-complexity argument, the correctness-bar design and the roofline analysis are
things I understand and could reproduce on a whiteboard. The backward derivation
in §2.3 I have worked through on paper but never debugged in code, which is a
weaker form of knowing. The kernel engineering, shared-memory layout, bank
conflicts,`cp.async` pipelining, register pressure, I have read about and not
done, and I would not claim otherwise.

The next thing to build is task 03 on rented hardware, against the 192 tests that
are already written and waiting.
