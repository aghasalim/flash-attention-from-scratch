# flash-attention-from-scratch

A fused, IO-aware attention kernel, built from the ground up — online softmax on paper, then Triton, then raw CUDA C++ with `mma` intrinsics — to be benchmarked against `torch.nn.functional.scaled_dot_product_attention` and the official `flash-attn` package.

> **Status: waves 0–1 done and measured. Waves 2–5 are blocked on hardware.**
> The math, the reference implementations, the test suite, and the roofline analysis
> are built and their numbers are real, taken on this machine and reproducible from
> `results/`. There is no kernel yet: Triton has no macOS wheel and there is no CUDA
> device here, so tasks 03 and 05–10 cannot run until I have an NVIDIA GPU. Nothing
> in this repo is estimated, and every CUDA-only quantity says so in those words
> rather than borrowing a number from the paper.

---

## What this is

The claim I keep reading is that attention is memory-bandwidth-bound, not
compute-bound. A 4096-token forward pass is supposed to spend most of its
wall-clock time pushing a 4096×4096 score matrix out to HBM and pulling it back,
with the matmuls nearly free by comparison. That's the whole thesis of
FlashAttention, and I don't want to take it on faith — I want to write the naive
version, profile it on a specific card, and see the number myself. Task 01 is
exactly that and nothing else.

The other reason for doing it this way: I want one project where I understand
every layer instead of one. The math (why streaming softmax is exact, not an
approximation), the algorithm (why tiling changes IO complexity from Θ(N²) to
Θ(N²d²/M) while leaving the FLOP count alone), and the hardware (why a given
`num_stages` wins on one card and stops winning once the block outgrows shared
memory). Plenty of people stop at the first layer.

## How it's structured

The work is split into twelve self-contained task specs in [`prompts/`](prompts/),
arranged into waves — some run in parallel, some have to run alone. The
dependency graph and the file-ownership rules that make the parallel waves safe
are in [`AGENTS.md`](AGENTS.md).

| wave | tasks | why |
|---|---|---|
| 0 | 00 bootstrap | scaffold + hardware fingerprint everything else reads |
| 1 | 01 roofline · 02 online softmax · 04 test harness | independent, no kernel yet |
| 2 | 03 Triton forward | the core; everything below extends it |
| 3 | 05 backward · 06 causal/masks · 07 autotune+bench | disjoint files, parallel |
| 4 | 08 GQA/varlen/dropout · 09 flash-decoding · 10 CUDA C++ | parallel |
| 5 | 11 write-up + ablations | reads every result file |

Realistic pacing, doing it properly: waves 0–1 a week, wave 2 two or three weeks,
wave 3 a month, wave 4 two or three months (the CUDA C++ path is a project on its
own), wave 5 a week.

## Ground rules

These are in [`AGENTS.md`](AGENTS.md) in full. The ones that matter most:

1. **Never fix a failing test by loosening the tolerance.** If fp16 output
   disagrees with the fp64 reference past the bar, the kernel is wrong.
2. **The correctness bar is relative.** The kernel's error against fp64 has to be
   no worse than naive fp16 attention's error against the same fp64 reference.
   Absolute tolerances either pass broken kernels or fail correct ones.
3. **No benchmark number that didn't come out of `bench/`.** Not estimates, not
   numbers from the paper, not "roughly 2× based on the algorithm."
4. **Every kernel change gets a dated logbook entry** with the before/after.
5. **All accumulators are fp32.** Inputs can be fp16/bf16; `acc`, `m_i`, `l_i`,
   `D` are fp32, always.
6. **If it can't be measured on the card I have, it says "not measured on this
   hardware."** No extrapolating to an H100 I don't own.

## Hardware

Measured by `scripts/env.py`, which writes `HARDWARE.md` and `hardware.json`. It has
a full NVIDIA path; on this machine that path finds nothing and says so.

- **Device:** Apple M4, 10 GPU cores, 25.77 GB unified memory (macOS 26.5.2, arm64)
- **Measured copy-kernel bandwidth:** 95.5 GB/s MPS · 94.0 GB/s CPU (1 GiB fp16 buffers, 10 warmup + median of 20)
- **Measured matmul throughput** (FLOPs = 2·M·N·K, M=N=K=8192): MPS fp16 3142.6 GFLOP/s · bf16 2744.8 · fp32 2331.8; CPU fp32 1652.2, fp64 468.7
- **CUDA:** none. HBM bandwidth, tensor-core TFLOP/s, SM count, shared memory, `cp.async`/FP8/TMA/WGMMA flags — all `not measured on this hardware (no CUDA device; developed on Apple M4)`
- **Triton:** not installed; no macOS wheel exists, so it lives in the `[gpu]` extra
- Python 3.14.4 · torch 2.13.0

Two constraints this turned up that shaped later tasks: MPS has no float64 at all, so
the fp64 reference runs on CPU; and run-to-run spread on the same matmul is ~18%
(MPS fp16 came back 3418 GFLOP/s at 09:50 and 2895 at 09:54), so every benchmark here
reports a range rather than a lone median.

## Results

![attention roofline measured on Apple M4](results/roofline.png)

Both baselines sit left of the ridge on MPS, which is the claim this repo opened
with, measured rather than quoted. The fused ideal is not plotted as a point
because nothing executes it yet — its arithmetic intensity is analytic, 2048
FLOP/byte, and it lands 62x right of the ridge. The right-hand panel is the CPU
control. A CUDA roofline is `not measured on this hardware`, which the figure
says in its own title rather than leaving to the caption.

<!-- BENCH:START -->
No kernel yet, so there is no kernel row. What exists is the baseline sweep the kernel
will eventually be measured against — `results/roofline.csv` (92 rows, 834.6 s),
plotted in `results/roofline.png`.

**Is attention memory-bound here?** Yes, measured. Ridge point on MPS fp16 is
**32.92 FLOP/byte** (3142.6 GFLOP/s ÷ 95.45 GB/s). Measured arithmetic intensity at
`N=4096`, `B=4 H=32 D=64`:

| implementation | AI (FLOP/byte) | verdict |
|---|---:|---|
| naive | 31.51 | left of ridge → memory-bound |
| chunked | 29.47 | left of ridge → memory-bound |
| fused ideal (S, P never stored) | 2048.00 | 62× right of ridge → compute-bound |

**Latency, MPS fp16, non-causal**, median from the CSV:

| seq_len | naive | chunked | naive ÷ chunked |
|--------:|------:|--------:|----------------:|
| 512 | 11.2 ms | 20.1 ms | 0.56× |
| 1024 | 43.1 ms | 74.7 ms | 0.58× |
| 2048 | 174.2 ms | 294.6 ms | 0.59× |
| 4096 | 46545.5 ms | 1226.6 ms | **37.95×** |
| 8192 | OOM | 4785.5 ms | — |
| 16384 | OOM | 32669.5 ms | — |

Chunking is *slower* than naive at every size that fits, then 38× faster the moment
it stops fitting. The naive-vs-FlashAttention ratio, which is the number this table
actually wants, is `not measured on this hardware (no CUDA device; developed on Apple M4)`.

Full derivations, the ridge-point argument, and the OOM prediction are in
[`notes/00-roofline.md`](notes/00-roofline.md).
<!-- BENCH:END -->

## Feature coverage

Nothing kernel-side is ticked, because nothing kernel-side can run here.

**Done and measured (waves 0–1):**

- [x] Hardware fingerprint with measured bandwidth/throughput, honest nulls for absent hardware
- [x] Online-softmax derivation + induction proof of exactness, and a NumPy reference shaped like the kernel
- [x] fp64 ground truth and a 500-test correctness suite (relative bar, adversarial, prime lengths, four invariants)
- [x] Naive / chunked / backend-forced-SDPA baselines
- [x] Roofline analysis: FLOP and byte derivations, measured ridge point, measured OOM threshold

**Blocked on an NVIDIA GPU (waves 2–5):**

- [ ] Forward, non-causal, fp16/bf16, head_dim ∈ {32, 64, 128}
- [ ] Backward via recomputation from stored logsumexp
- [ ] Causal masking with block skipping
- [ ] Sliding-window / local attention
- [ ] ALiBi and arbitrary additive bias
- [ ] MQA / GQA (grouped KV heads, no materialization)
- [ ] Variable-length packed batches (`cu_seqlens`)
- [ ] Dropout with fwd/bwd-consistent Philox RNG
- [ ] Flash-Decoding (split-KV) for batch-1 long-context inference
- [ ] Paged KV cache with block-table indirection
- [ ] CUDA C++ implementation with explicit `mma` + `cp.async`

## Planned layout

None of this exists yet. It's what the task specs build.

```
fa/
  ref/            fp64 reference attention, streaming-softmax reference (NumPy)
  triton/         fwd.py, bwd.py, autotune configs, the autograd.Function wrapper
  cuda/           attention.cu -- wmma/mma path, cp.async double buffering, swizzled smem
  ops/            public API: fa.ops.attention(q, k, v, causal=..., window=...)
tests/            correctness suite -- fp64 comparison, gradcheck, adversarial inputs
bench/            latency/memory/roofline harness, ncu wrappers
notes/            the derivations and the logbook
results/          generated. csv + plots. checked in so the README tables reproduce.
prompts/          the task specs that build all of the above (see AGENTS.md)
```

## What I got wrong

Real ones, from [`notes/LOGBOOK.md`](notes/LOGBOOK.md). More will land as the kernel does.

**1. My OOM prediction was 19% off, and widening the bar would have hidden the reason.**
The textbook model says naive attention dies when the two `N²` tensors (`S` and `P`)
fill memory, which predicted `N≈6103`. It actually died at 5120. The failure implies
~3.16 concurrent `N²` tensors, not 2 — the third is the `masked_fill` allocation held
alongside them. With three the prediction is `N≈4983`, off by −2.7%. The byte model
under-counts any real implementation by exactly the intermediates the framework
materialises.

**2. I assumed tiling was the win. It is not — it moved arithmetic intensity the wrong way.**
Chunked attention is 0.56–0.59× the speed of naive at every `N` that fits, and its
measured AI is 29.47 against naive's 31.51. Tiling buys `Θ(N²) → Θ(N·BLOCK)` memory
and nothing else; the bytes are unchanged. That pair of numbers is the argument for
fusion, and I had the causality backwards until I measured it.

**3. `sdpa_kernel` is a silent no-op on MPS.** Forcing a backend appeared to work —
no error, plausible timings. The probe row says `sdpa_backend_honored=False`: it ran
whatever kernel the device picked. Had I not checked, the CSV would have carried four
columns of "MATH backend" numbers that were nothing of the sort. On CPU the same probe
correctly raises. Never trust a backend you did not verify was honored.

## Reading that this is built on

- Dao, Fu, Ermon, Rudra, Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022. — the IO-complexity argument in §3.2 is the load-bearing part.
- Dao. *FlashAttention-2.* 2023. — work partitioning and cutting non-matmul FLOPs. Read before tuning anything.
- Shah et al. *FlashAttention-3.* 2024. — warp specialization and FP8 on Hopper. Only relevant with an H100.
- Milakov, Gimelshein. *Online normalizer calculation for softmax.* 2018. — the two-page paper the whole thing rests on.
- Rabe, Staats. *Self-attention Does Not Need O(n²) Memory.* 2021. — the memory result without the IO-awareness framing.
- The Triton tutorial `06-fused-attention.py` — after writing my own, not before.

## Running it

```bash
make setup                                  # venv + deps
.venv/bin/python -m scripts.env             # writes HARDWARE.md + hardware.json
.venv/bin/python -m pytest tests/           # 270 passed, 38 skipped, 192 xfailed (~11 s)
.venv/bin/python -m pytest tests/ --slow    # 286 passed, 6 skipped, 208 xfailed (~39 s)
.venv/bin/python -m fa.ref.online_softmax   # the four online-softmax experiments
.venv/bin/python -m bench.roofline          # writes results/roofline.csv + .png (~14 min)
```

The 192 xfails are the kernel tests. They are written and will run the day there is a
GPU; here they are xfail for lack of one, not for lack of a test.

## Author

Aghasalim Mustafazada — third-year AI student at Howest, Belgium.

<p align="center">
  <a href="https://github.com/aghasalim">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="github"></a>
  <a href="https://www.kaggle.com/aghasalimmustafazada">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="kaggle"></a>
  <a href="https://linkedin.com/in/mustafazada">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="linkedin"></a>
  <a href="https://orcid.org/0009-0001-8746-4582">
    <img src="https://img.shields.io/badge/ORCID-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" alt="orcid"></a>
</p>

## License

MIT — see [LICENSE](LICENSE).
