# flash-attention-from-scratch

[![ci](https://github.com/aghasalim/flash-attention-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/aghasalim/flash-attention-from-scratch/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![results](https://img.shields.io/badge/results-reproducible-1a9850.svg)](results/)

A from-scratch implementation of IO-aware attention, built to check whether the
standard claim about it, that attention is bound by memory bandwidth rather than
arithmetic, actually holds on hardware I can measure.

**Current state.** The mathematics, the reference implementations, the test suite
and the empirical analysis are complete and reproducible. The Triton and CUDA
kernels are not written. Triton publishes no macOS wheel and this machine has no
NVIDIA GPU, so six of the twelve planned tasks are blocked on hardware rather than
on effort. I have not written kernels I cannot compile or test; the repository
holds no unverified kernel code, and every quantity that could not be measured
here says so explicitly rather than being estimated or taken from the literature.

---

## 1. The question

Attention computes `S = QKᵀ/√d`, `P = softmax(S)`, `O = PV`. The cost is supposed to
be dominated not by the two matmuls but by writing the `B·H·N²` score matrix to
memory and reading it back, an intermediate the caller never asked for.

The ratio of score traffic to parameter traffic is `N/D`, so at `D = 64` the scores
overtake `Q`, `K`, `V` and `O` combined at `N = 64`:

| N | Q,K,V,O | S,P | ratio |
|---:|---:|---:|---:|
| 1024 | 0.062 GiB | 0.500 GiB | 8× |
| 4096 | 0.250 GiB | 8.000 GiB | 32× |
| 16384 | 1.000 GiB | 128.000 GiB | 128× |

fp16, `B=4 H=32 D=64`. The argument is clearly right in the limit; what I wanted was
a number for how much it is worth on a machine I own.

![HBM traffic and how each configuration ended](results/memory.png)

*Left: analytic traffic. Naive and chunked lie on top of each other, since chunking
changes when the bytes move, not how many. Right: naive is the only implementation
that fails outright, on 4 of 24 configurations.*

## 2. What I found

**Fusion is worth roughly 3× on this hardware, and it takes achieved throughput
from 22% of the CPU's measured fp32 peak to roughly 67%.** This is the central result
and it is measured rather than modelled. Triton will not install here, but
`torch.compile`'s inductor backend performs genuine kernel fusion on the CPU, which
exercises the same mechanism, keeping intermediates off the memory bus, at a
different scale.

| N | eager | fused (`torch.compile`) | speedup | across repeats |
|---:|---:|---:|---:|---|
| 256 | 0.58 ms | 0.35 ms | 1.70× | 1.60 to 1.74 |
| 512 | 2.92 ms | 1.18 ms | 2.48× | 2.46 to 2.50 |
| 1024 | 12.44 ms | 4.21 ms | 3.01× | 2.27 to 3.94 |
| 2048 | 44.88 ms | 14.52 ms | 3.09× | 2.88 to 3.19 |
| 4096 | 176.05 ms | 60.20 ms | 2.92× | 2.39 to 3.39 |

CPU, fp32, `B=2 H=8 D=64`, five independent repeats with the variants interleaved;
outputs agree with eager to 5.5e-07. Source: `results/fusion.csv`.

![CPU fusion: latency and speedup with ranges](results/fusion.png)

I first ran this once per configuration and read a speedup climbing steadily to
4.47×, the shape the bandwidth argument predicts and the one I was pleased to see.
It does not survive repetition. With five repeats the ratio reaches about 3× by
`N = 512` and then flattens, and the `N = 1024` interval alone spans 2.27 to 3.94. The
benefit is real; the trend I wanted to read into it was noise.

What does hold is the level shift in throughput, which is large and stable at every
size:

![Achieved throughput as a share of measured fp32 peak](results/throughput.png)

*Eager attention is pinned near 20 to 26% of peak regardless of sequence length, the
signature of a workload waiting on memory. Fusing lifts it to between 45% and 68%.*

**Tiling on its own buys nothing, and I had assumed the opposite.** Chunked
attention, looping over key/value blocks without fusing, is *not* faster than the
naive implementation at any sequence length, on either device I tested. It is
0.56 to 0.59× on the GPU and level with naive on the CPU. Its measured arithmetic
intensity is 29.47 against naive's 31.51, so tiling moves intensity in the wrong
direction, because it re-reads `Q` and the accumulator once per tile while moving
the same score traffic. Tiling converts an out-of-memory failure into a slow
program; it is a prerequisite for fusion, not a substitute for it. That distinction
is the single most useful thing I learned here.

**The memory wall is a cliff, not a slope.** Naive attention tracks `N²` exactly
while it fits, with successive doublings costing 3.84× and 4.04×, and then at `N = 4096`
it goes from 174 ms to 46.5 s, a 267× jump for 4× the work, as 8 GiB of score
traffic stops fitting the 17.76 GiB working set. At that point chunked attention,
slower at every smaller size, is 37.95× faster, and naive fails outright 1024
tokens later. Chunked, on the same sweep, reaches `N = 16384`.

![Latency scaling on MPS and CPU](results/latency-scaling.png)

Stepping `N` in increments of 256 locates the failure precisely, and shows why the
textbook prediction missed it:

![OOM ladder](results/oom-ladder.png)

![Roofline](results/roofline.png)

**The roofline does not settle the memory-bound question on this machine, and I
initially claimed that it did.** The ridge point is 30.91 FLOP/byte at the median,
but peak compute swings between 1937 and 3793 GFLOP/s with thermal state, which
places the ridge anywhere in 20.08 to 40.55. Naive attention's arithmetic intensity is
31.51, which is inside that band. It sits on the knee, and which side of it you measure
depends on how warm the laptop is. The first version of this file asserted
"memory-bound, measured" on the strength of a 4% margin against a quantity whose own
uncertainty is ±35%.

What survives the noise is the comparison that was always the important one: a fused
implementation sits at 2048 FLOP/byte, 66× the ridge, and no amount of thermal drift
moves that. The direct measurements above, the fusion speedup, the cliff and the OOM,
do not depend on a ridge point at all.

**Causal masking is only a speedup if the hidden blocks are actually skipped.** This
is measurable without writing a kernel, by comparing implementations that skip
against ones that mask a dense `N×N`. The ones that mask run at 0.91 to 0.98× under
causal, slightly *slower*, since masking is extra work over identical traffic,
while the SDPA path, which classifies and skips blocks, reaches 2.02× and approaches
the theoretical ceiling as `N` grows and the diagonal becomes a smaller share of the
triangle. That gap is the entire value of block skipping, isolated.

![Causal block skipping](results/causal-skipping.png)

**The online-softmax recurrence is exact, and I proved it rather than assuming it.**
Verified to 7.216e-16 against direct computation in fp64, across block sizes that do
not evenly divide the sequence, causal and non-causal. Because the algorithm is
exact, any discrepancy a kernel later shows must be an arithmetic defect, never the
algorithm, which considerably narrows the search when debugging one.

**Keeping accumulators in fp32 is worth a factor of 3297.** At `N = 8192`, fp16
accumulation gives 1.568e-04 maximum absolute error against 4.755e-08 for fp32, and
the denominator is 8202× worse. The relative error reaches 1.000 from `N = 1024`
upward, meaning small probabilities come back as literal zero.

![fp32 vs fp16 accumulators](results/accumulator.png)

*Only the accumulator dtype differs; inputs are fp16 in both arms. The left panel is
why max-absolute-error alone is a poor diagnostic, since it is flat in `N` because
`max(p) ~ 1/N`, and the right panel is where the damage actually shows.*

Separately, zero-filling a short trailing block instead of masking it with −∞ gives
probabilities summing to 0.430484 rather than 1. Both traps are now numbers in the
test suite rather than warnings in a comment.

## 3. What is measured, and what is not

Hardware fingerprint via `scripts/env.py`, which has a full NVIDIA code path that
finds nothing here and says so.

| | |
|---|---|
| Device | Apple M4, 10 GPU cores, 25.77 GB unified, macOS 26.5.2 |
| Copy-kernel bandwidth | 95.86 GB/s (MPS) · 101.29 GB/s (CPU) |
| Matmul throughput | 2963.5 GFLOP/s fp16 · 3014.7 bf16 · 2542.8 fp32 (MPS); 1738.3 fp32 · 463.0 fp64 (CPU) |
| Ridge point | 30.91 FLOP/byte median, 20.08 to 40.55 across thermal state |
| CUDA | none. HBM bandwidth, tensor-core throughput, SM count, `cp.async`/FP8/TMA all `not measured on this hardware` |
| Triton | not installable; no darwin wheel exists |

Two constraints this surfaced: MPS provides no float64 at all, so the fp64 reference
runs on the CPU; and run-to-run spread on an identical matmul is large enough to move
a conclusion, so every figure here is a median reported with its range.

Quantities requiring an NVIDIA GPU, namely the naive-versus-FlashAttention ratio, `exp2`
against `exp`, block-size and pipeline-depth sweeps, shared-memory swizzling,
`cp.async`, Flash-Decoding and GQA cache savings, are absent rather than estimated.
The full ablation table, including every cell marked unmeasurable, is in
[`notes/paper.md`](notes/paper.md).

## 4. Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

```bash
python -m scripts.env          # hardware fingerprint -> HARDWARE.md, hardware.json
python -m pytest tests/        # 270 passed, 38 skipped, 192 xfailed (~11 s)
python -m fa.ref.online_softmax  # exactness proof and the accumulator experiments
python -m bench.fusion         # CPU fusion measurement -> results/fusion.csv (~5 min)
python -m bench.roofline       # full sweep -> results/roofline.csv, .png (~14 min)
python scripts/check_numbers.py  # every figure above, re-derived from source data
```

The 192 expected failures are the kernel tests. They are written and will run
against a Triton implementation the day there is a GPU; they are marked `xfail` for
want of hardware, not for want of a test.

`scripts/check_numbers.py` re-derives 23 of the figures in this file from
`hardware.json` and `results/*.csv` and fails if they disagree. It runs in CI on
every push, because prose goes stale when the underlying data is regenerated rather
than when the prose is edited, which is precisely how the ridge-point error above
survived for several hours.

## 5. Method and structure

The work is organised into waves, each one verifiable on its own before the next
depends on it. The rules every wave follows are in [`METHODOLOGY.md`](METHODOLOGY.md).

```
fa/ref/        fp64 ground truth; naive, chunked and backend-forced SDPA baselines;
               the NumPy online-softmax reference, written in the shape the eventual
               Triton kernel takes (outer loop over query blocks, inner loop over
               key/value blocks, fp32 accumulators, causal split into three zones)
fa/triton/     empty, task 03 and 05-10
fa/cuda/       empty, task 10
tests/         500 tests; 192 xfail pending a GPU
bench/         roofline sweep and CPU fusion measurement
notes/         derivations, the write-up, and the logbook
results/       generated CSVs and figures, committed so the tables above reproduce
```

Two decisions are worth defending. First, the test suite was written against the
references *before* any kernel existed; a harness written afterwards tends to encode
the kernel's own bugs as expected behaviour. Second, the correctness criterion is
relative rather than absolute: a kernel's error against the fp64 reference must be
no worse than twice the naive implementation's error against the same reference.
Attention's numerical error grows with sequence length and score magnitude, so any
fixed tolerance is either loose enough to admit broken kernels or tight enough to
reject correct ones. Ground truth is fp64 throughout and never
`scaled_dot_product_attention`, which is itself FlashAttention and would conceal any
bug the two implementations shared.

[`notes/LOGBOOK.md`](notes/LOGBOOK.md) is the running record: what was tried, the
number before and after, and what I concluded, including the entries where the
conclusion was wrong.

## 6. Limitations

The kernel, the object the project is named after, does not exist. Everything here
is baselines, references, harness and analysis.

Every measurement is Apple silicon. Unified memory has a compute-to-bandwidth ratio
unlike any discrete GPU, and while the shape of these results should transfer, no
individual number will. In particular the roofline verdict for the unfused baselines
is indeterminate here and would likely be decisive on a discrete card, where the
ridge point sits far higher.

The fusion measurement in §2 is inductor's C++ code generation, not FlashAttention.
There is no tiling, no online softmax, no explicit shared-memory blocking, and it
still allocates `O(N²)`, so it does not address the memory wall, only the traffic.
It establishes that the mechanism is real and worth measuring; it is not a
substitute for the kernel and should not be quoted as one.

`sdpa_kernel` is a silent no-op on MPS. Forcing a backend there does nothing and
raises no error; those rows are labelled `NOT HONORED` in the CSV rather than
reported as a MATH-backend measurement. On the CPU the same probe behaves correctly.

The backward pass is derived on paper and never executed.

## 7. Errors worth recording

Five, in descending order of how much they should have embarrassed me.

I twice reported a trend from single-run measurements and twice had to withdraw it.
First the ridge point, below. Then the fusion speedup, where one pass per size gave
a clean monotone climb to 4.47× and five passes gave a plateau near 3× with a
±25% interval at one of the sizes. Both times the number was not wrong so much as
reported without an error bar, and both times the error bar was wide enough to
delete the claim. `bench/fusion.py` now interleaves repeats and reports the range.

I claimed attention was memory-bound here before checking the error bar on the
quantity I was claiming it from. Re-running the fingerprint an hour later moved the
ridge point across naive's arithmetic intensity and inverted the conclusion. The
measurement was fine; treating one median as a fact was not.

I expected tiling to be the win. It is 0.56× at every size that fits. The win is
fusion, and tiling is its prerequisite. Obvious in hindsight, and not obvious to me
beforehand.

I predicted the out-of-memory threshold with the textbook two-tensor model and was
19.2% out, beyond the 15% bar I had set. Rather than widen the bar I solved for the
implied tensor count: 3.16, not 2. The third is the `masked_fill` intermediate held
alongside `S` and `P`. With three the prediction lands at −2.7%. The textbook byte
model under-counts any real implementation by exactly the tensors the framework
happens to materialise.

I trusted `sdpa_kernel` without verifying it was honoured. It failed silently and
plausibly, and four columns of results would have carried a backend label that was
simply false. The same instinct, check that the knob you turned did something,
later caught an undeclared scipy dependency and a `make lint` target that contradicted
a passing CI, both found by cloning the repository fresh and running it as a stranger
would.

## 8. References

Milakov and Gimelshein, *Online normalizer calculation for softmax* (2018), is the
two-page result the whole construction rests on. Rabe and Staats, *Self-attention
Does Not Need O(n²) Memory* (2021), gives the memory result without the IO framing;
`chunked_attention` here is essentially their construction, and §2 is the measurement
of why that is not sufficient. Dao, Fu, Ermon, Rudra and Ré, *FlashAttention* (2022),
adds the IO-awareness that makes the difference, with the complexity argument in §3.2.
Dao, *FlashAttention-2* (2023), covers work partitioning and the loop ordering the
NumPy reference is written in. Shah et al., *FlashAttention-3* (2024), addresses
Hopper-specific warp specialisation and FP8. Kwon et al., *PagedAttention* (2023), is
the basis for the paged-cache task.

## Author

Aghasalim Mustafazada, third year AI student at Howest, Belgium.

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

MIT, see [LICENSE](LICENSE).
