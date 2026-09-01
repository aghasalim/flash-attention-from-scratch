# 00, Roofline: is attention actually memory-bound?

The premise of this repo is that attention is memory-bandwidth-bound rather than
compute-bound. This note tries to establish that on the hardware I actually have,
with numbers, instead of taking it from the paper.

**Hardware caveat, up front.** There is no NVIDIA GPU here. Everything measured
below ran on an Apple M4, MPS (the Apple GPU) in fp16, or the CPU in fp32. Those
are real measurements of real hardware, but they are *not* stand-ins for CUDA
numbers, and nothing here should be read as a FlashAttention result. Every
CUDA-specific quantity in the task spec is marked
`not measured on this hardware (no CUDA device; developed on Apple M4)`.

Reproduce with:

```bash
.venv/bin/python -m scripts.env        # writes HARDWARE.md, hardware.json
.venv/bin/python -m bench.roofline     # writes results/roofline.csv
.venv/bin/python -m bench.figures      # draws the figures from that CSV
```

Config throughout: `B=4, H=32, D=64`. Sweep wall clock 834.6 s, 92 rows in
`results/roofline.csv`.

---

## 1. FLOPs, derived

Attention is two matmuls. A matmul of `(M×K) @ (K×N)` is `2·M·N·K` FLOPs, one
multiply and one add per inner-product term.

**Forward.** Per (batch, head):

| step | shape | FLOPs |
|---|---|---|
| `S = Q Kᵀ` | `(N×D) @ (D×N)` | `2·N·N·D` |
| `O = P V` | `(N×N) @ (N×D)` | `2·N·N·D` |

So `4·N²·D` per head, and over the whole batch:

```
FLOPs_fwd = 4 · B · H · N² · D
```

The softmax itself is `Θ(B·H·N²)`, one exp, one add, one divide per score. It is
lower-order in `D` and, more to the point, it is not a matmul, so it does not run
on the tensor cores. FlashAttention-2's headline optimisation is precisely
reducing these non-matmul FLOPs. Excluded from the count above; noted because it
is not free.

Causal masking halves the score entries that need computing (the strict lower
triangle plus the diagonal, `N(N+1)/2` of `N²`), so `FLOPs_fwd_causal ≈ 2·B·H·N²·D`.
"≈" because the diagonal blocks are computed densely and then masked, the saving
is real but is not exactly one half at finite block size. Task 06 measures what
fraction is actually recovered.

**Backward.** With recomputation, five matmuls of the same shape:

| step | FLOPs |
|---|---|
| recompute `S = Q Kᵀ` | `2·N²·D` |
| `dV = Pᵀ dO` | `2·N²·D` |
| `dP = dO Vᵀ` | `2·N²·D` |
| `dQ = dS K` | `2·N²·D` |
| `dK = dSᵀ Q` | `2·N²·D` |

```
FLOPs_bwd = 10 · B · H · N² · D   =   2.5 × forward
```

That 2.5× is the analytic target task 05 has to land near. Outside roughly 2 to 4×
of forward wall-clock, something is wrong.

## 2. Bytes, derived, and why the score matrix is the whole problem

Let `e` be bytes per element (2 for fp16/bf16). Per (batch, head), `Q, K, V, O` are
each `N·D` elements. The score matrices `S` and `P` are each `N²`.

```
bytes(Q,K,V,O) = 4 · B · H · N · D · e
bytes(S,P)     = 4 · B · H · N² · e      # each of S and P written once and read once
```

The ratio of the second term to the first is `N/D`. At `D=64` that means the score
traffic passes the parameter traffic at `N=64` and never looks back. Concretely,
at `B=4, H=32, D=64`, fp16:

| N | Q,K,V,O | S,P | ratio |
|---:|---:|---:|---:|
| 512 | 0.031 GiB | 0.250 GiB | 8× |
| 1024 | 0.062 GiB | 1.000 GiB | 16× |
| 2048 | 0.125 GiB | 4.000 GiB | 32× |
| **4096** | **0.250 GiB** | **16.000 GiB** | **64×** |
| 8192 | 0.500 GiB | 64.000 GiB | 128× |
| 16384 | 1.000 GiB | 256.000 GiB | 256× |

At `N=4096` the score matrix is 64× the traffic of all four real tensors combined.
This is the number that makes the argument: the kernel spends its time moving a
matrix that exists only as an intermediate and is never wanted by the caller.
Fusing softmax into the matmul means `S` and `P` never reach memory at all, which
takes the byte count back down to `4·B·H·N·D·e`.

## 3. Arithmetic intensity and the ridge point

Machine balance = measured compute ÷ measured bandwidth. Both measured by
`scripts/env.py` on this machine (copy-kernel bandwidth, `2·M·N·K` matmul):

| device | compute | bandwidth | ridge point |
|---|---:|---:|---:|
| MPS fp16 | 2963.5 GFLOP/s | 95.86 GB/s | **30.91 FLOP/byte** |
| CPU fp32 | 1738.3 GFLOP/s | 101.29 GB/s | **17.16 FLOP/byte** |

The MPS ridge point is **not stable**. Peak compute swings 1937 to 3793 GFLOP/s with
thermal state, which puts the ridge anywhere in **20.08 to 40.55 FLOP/byte**. That band
is wide enough to change the verdict below, so read the next table with it in mind.
| CUDA | `not measured on this hardware (no CUDA device; developed on Apple M4)` | same | same |

Measured arithmetic intensity at `N=4096`, MPS fp16, non-causal, from
`results/roofline.csv`:

| implementation | AI (FLOP/byte) | vs ridge (30.91, band 20.08 to 40.55) |
|---|---:|---|
| naive | 31.51 | inside the band → **indeterminate** |
| chunked | 29.47 | inside the band → **indeterminate** |
| fused ideal (`S`, `P` never stored) | 2048.00 | 66× the ridge → **decisively compute-bound** |

That is the roofline result, and it is weaker than I wanted. Both unfused
implementations sit essentially *on* the knee, 31.51 against a ridge of 30.91 is a
2% margin against a quantity with ±35% uncertainty, so this hardware cannot settle
whether naive attention is memory-bound. It is balanced, and which side it lands on
depends on how warm the machine is.

What the noise cannot touch is the fused point: 2048 FLOP/byte is 66× the ridge, and
no thermal drift moves that. Fusion relocates the problem to the compute side by two
orders of magnitude. The direct evidence in §4 and §5, the 37.95× latency inversion
and the OOM, is also independent of any ridge point.

Note the naive AI barely moves with `N` (31.51 at 4096, 31.75 at 8192, 31.88 at
16384) and asymptotes at `4·D/(2·e) = 32`. It is a constant, independent of
sequence length. Making `N` larger does not make naive attention more
compute-dense; it just makes it move more bytes.

## 4. Where naive falls off the cliff

MPS, fp16, non-causal, median latency from `results/roofline.csv`:

| N | naive | chunked | naive ÷ chunked |
|---:|---:|---:|---:|
| 512 | 11.2 ms | 20.1 ms | 0.56× |
| 1024 | 43.1 ms | 74.7 ms | 0.58× |
| 2048 | 174.2 ms | 294.6 ms | 0.59× |
| **4096** | **46545.5 ms** | **1226.6 ms** | **37.95×** |
| 8192 | OOM | 4785.5 ms |, |
| 16384 | OOM | 32669.5 ms |, |

Two things worth reading carefully.

**Up to N=2048, chunked is *slower*.** 0.56 to 0.59×, consistently. Tiling costs a
Python-level loop and re-reads `Q` once per tile, and it buys nothing while
everything still fits comfortably. If you only benchmarked short sequences you
would conclude tiling was a pessimisation.

**At N=4096 it inverts by a factor of 38.** Naive goes from 174 ms to 46.5
seconds, a 267× jump for a 4× increase in work. Nothing about the arithmetic
changed; the 8 GiB of score matrices stopped fitting in the 17.76 GiB MPS working
set alongside everything else, and the run went to swap. The measured peak for
that config is 21.7 GB. This is the memory wall, and it does not arrive gently.

Latency scales as `N²` exactly while it fits: 11.2 → 43.1 → 174.2 ms is
3.84×, 4.04× for successive doublings.

## 5. OOM threshold: prediction vs. reality

`bench/roofline.py` walks `N` upward in steps of 256 (`oom_ladder` rows) to find
where naive dies:

| N | result | sampled peak |
|---:|---|---:|
| 4096 | ok | 21.74 GB |
| 4352 | ok | 24.53 GB |
| 4608 | ok | 27.41 GB |
| 4864 | ok | 30.52 GB |
| **5120** | **OOM** |, (allocator reported 19.75 GiB allocated at failure) |

The closed-form prediction in the task spec assumes two concurrent `N²` tensors
(`S` and `P`) against the memory budget. `torch.mps.recommended_max_memory()` is
17.76 GiB, so:

```
2 · B·H·N²·e = 17.76 GiB   →   N ≈ 6103
```

Measured first failure is 5120. That is **+19.2% error, outside the 15% bar the
spec asks for.** The two-tensor model is wrong, and it is worth saying why rather
than widening the bar.

Solving for the tensor count that the observed failure implies: one fp16 `N²`
tensor at `N=5120` is 6.25 GiB, and 19.75 GiB ÷ 6.25 GiB ≈ **3.16 concurrent
`N²` tensors**, not 2. Re-running the prediction with three:

```
3 · B·H·N²·e = 17.76 GiB   →   N ≈ 4983
```

| model | predicted N | vs first OOM (5120) | vs last OK (4864) |
|---|---:|---:|---:|
| 2 tensors (`S`, `P`) | 6103 | +19.2% | +25.5% |
| **3 tensors** | **4983** | **−2.7%** | **+2.4%** |
| 4 tensors | 4315 | −15.7% | −11.3% |

The third tensor is the masked scores. `naive_attention` computes `S`, then
`S.masked_fill(...)`, a new allocation, then softmax over that, and the
autograd graph holds them concurrently. The textbook `S`-and-`P` count under-counts
any real implementation by exactly the intermediates the framework materialises.
With the correct count the prediction lands within 3%.

## 6. Why chunking fixes memory but not bandwidth

This is the single most important idea in the repo, so, plainly:

`chunked_attention` never holds more than one `BLOCK×N` tile of scores, so peak
memory drops from `Θ(N²)` to `Θ(N·BLOCK)` and it survives to `N=16384` where naive
died at 5120. That is a real and large win, and it is the whole reason it is
still running in the last two rows of the table above.

But it moves *the same number of bytes*. Each score tile is still computed,
written out to memory, read back for the softmax, written again, and read again
for the `P V` matmul, just in pieces rather than all at once. The measured
arithmetic intensity says so directly: 29.47 FLOP/byte for chunked against 31.51
for naive. Chunking made it slightly *worse*, because tiling adds re-reads of `Q`
and of the fp32 accumulator, once per tile.

So tiling alone converts an out-of-memory error into a slow program. The bytes
are unchanged and the ridge point is unchanged, so the roofline position is
unchanged.

What fusion adds is that the tile never leaves the chip. Compute the score tile in
SRAM, run the online-softmax update on it in registers (see
[`notes/01-online-softmax.md`](01-online-softmax.md): the recurrence is exact, so
this costs no accuracy), accumulate into the output, and discard it. `S` and `P`
are never written to HBM at all. That is the step from AI ≈ 30 to AI = 2048, and
it is the step that moves the workload across the ridge point. Tiling is a
prerequisite for it, not a substitute for it.

## 7. What could not be measured here

Per rule 6, stated rather than estimated:

- **Naive-vs-FlashAttention latency ratio at N=4096.** The task's headline number.
`SDPBackend.FLASH_ATTENTION` on this machine is not the FlashAttention-2 CUDA
  kernel. `not measured on this hardware (no CUDA device; developed on Apple M4)`
- **CUDA ridge point, HBM bandwidth, tensor-core TFLOP/s, measured-vs-advertised
  ratio.** `not measured on this hardware (no CUDA device; developed on Apple M4)`
- **Backend-forced SDPA comparison on MPS.** `torch.nn.attention.sdpa_kernel` is a
  **no-op on MPS**: the probe row in the CSV records
`sdpa_backend_honored=False`, and every MPS sdpa row is labelled
`MATH requested / NOT HONORED: whatever kernel this device picked`. Those rows
  are *not* a MATH-backend measurement and must not be read as one. On CPU the
  same probe returns honored=True (an empty backend list correctly raises), and
  there `EFFICIENT_ATTENTION` and `CUDNN_ATTENTION` are simply unavailable.
- **Per-config peak memory on CPU.** torch exposes no resettable per-device peak
  allocator stat for CPU; `ru_maxrss` is a monotonic process high-water mark and
  cannot be reset between configs, so those cells say
`not measured: torch has no per-device peak allocator stat for CPU` rather than
  carrying a misleading number.
- **CPU beyond N=4096.** Capped by `CPU_N_MAX`; the fp32 score matrix is ≥34 GB at
  N=8192 and a swap-thrashing run is not a measurement. The 16 dropped points are
  still emitted as rows with `status=skipped` and a reason, nothing is silently
  truncated.
- **fp16 on CPU.** Not swept. arm64 torch has no fast half GEMM: `scripts/env.py`
  measures 3.3 GFLOP/s against 1652 GFLOP/s for fp32, a 500× gap that would
  measure the dispatch path rather than the algorithm. CPU rows are fp32 and the
`dtype` column says so.
- **Backward pass.** Analytic only (§1). Not measured; there is no backward
  implementation until task 05.

## 8. Correctness

`bench/roofline.py` asserts before it times: `naive_attention` and
`chunked_attention` agree with `sdpa_attention(backend=MATH)`, and every
implementation is within 2× the naive-in-that-dtype error against the same fp64
reference. That check passes, a benchmark of a wrong implementation is worse than
no benchmark.
