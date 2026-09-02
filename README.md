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
| 1024 | 0.062 GiB | 1.000 GiB | 16× |
| 4096 | 0.250 GiB | 16.000 GiB | 64× |
| 16384 | 1.000 GiB | 256.000 GiB | 256× |

fp16, `B=4 H=32 D=64`. The argument is clearly right in the limit; what I wanted was
a number for how much it is worth on a machine I own.

![HBM traffic and how each configuration ended](results/memory.png)

*Left: analytic traffic. Naive and chunked lie on top of each other, since chunking
changes when the bytes move, not how many. Right: naive is the only implementation
that fails outright, on 4 of 24 configurations.*

The way out is to never build `S` at all. The scores are computed one tile at a time,
and the only state carried from one tile to the next is a running row max `m` and a
running row sum `l`, two floats per query row. That is the whole trick, and it is
easier to watch than to read:

![Blockwise tiling with the running online softmax statistics](results/online-softmax-tiling.gif)

*Schematic of the algorithm, not a measurement. Non-causal, N=64, D=16, tile 16 by 16,
seed 0, traced through the reference in
[fa/ref/online_softmax.py](fa/ref/online_softmax.py). Only the coloured block of scores
exists at each step: grey blocks were computed and freed, white ones have not been
touched. Every other figure on this page is measured data.*

## 2. What I found
**Fusion is worth roughly 3× on this hardware, and it takes achieved throughput from 22% of the CPU's measured fp32 peak to roughly 67%.** This is the central result and it is measured rather than modelled.

Tiling on its own buys nothing. Chunked attention runs at 0.56 to 0.59× naive on the
GPU, and its arithmetic intensity is 29.47 against naive's 31.51, so looping over
key blocks without fusing moves intensity the wrong way. The memory wall is a cliff
rather than a slope: naive attention follows `N²` up to 2048, where it takes 174 ms,
and then takes 46.5 s at `N = 4096`, a 267× jump for 4× the work against a fitted
`N²` trend of 679 ms, while chunked attention is 37.95× faster at that size and
still runs at 16384. Causal masking pays only where blocks are skipped, 2.02× for
the SDPA path that skips them against 0.91 to 0.98× for the implementations that
mask a dense `N×N`, and fp32 accumulators are worth a factor of 3297 in maximum
absolute error at `N = 8192`.

![CPU fusion: latency and speedup with ranges](results/fusion.png)
![Achieved throughput as a share of measured fp32 peak](results/throughput.png)
![Latency scaling on MPS and CPU](results/latency-scaling.png)
![OOM ladder](results/oom-ladder.png)
![Roofline](results/roofline.png)
![Causal block skipping](results/causal-skipping.png)
![fp32 vs fp16 accumulators](results/accumulator.png)

Full detail in [notes/METHODS.md](notes/METHODS.md#2-what-i-found).
## 3. What is measured, and what is not
Hardware fingerprint via `scripts/env.py`, which has a full NVIDIA code path that finds nothing here and says so.

Every number on this page comes off an Apple M4 with 10 GPU cores and 25.77 GB of
unified memory: 95.86 GB/s copy bandwidth on MPS, 101.29 GB/s on the CPU, and matmul
peaks of 2963.5 GFLOP/s fp16 on MPS and 1738.3 GFLOP/s fp32 on the CPU. There is no
CUDA device and no macOS Triton wheel, so HBM bandwidth, tensor-core throughput, SM
count, `cp.async`, FP8 and TMA are all recorded as not measured on this hardware
instead of being filled in from a spec sheet. MPS has no float64 either, which is why
the fp64 reference runs on the CPU. Run-to-run spread on an identical matmul is wide
enough to move a conclusion, so every figure here is a median reported with its range.

Full detail in [notes/METHODS.md](notes/METHODS.md#3-what-is-measured-and-what-is-not).
## 4. Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

```bash
python -m scripts.env          # hardware fingerprint -> HARDWARE.md, hardware.json
python -m pytest tests/        # 297 passed, 38 skipped, 192 xfailed (~12 s)
python -m fa.ref.online_softmax  # exactness proof and the accumulator experiments
python -m bench.fusion         # CPU fusion measurement -> results/fusion.csv (~5 min)
python -m bench.roofline       # full sweep -> results/roofline.csv (~14 min)
python -m bench.figures        # every figure above, drawn from the committed CSVs
python scripts/check_numbers.py  # every figure above, re-derived from source data
```

The 192 expected failures are the kernel tests. They are written and will run
against a Triton implementation the day there is a GPU; they are marked `xfail` for
want of hardware, not for want of a test.

`scripts/check_numbers.py` re-derives 34 quoted figures from `hardware.json` and
`results/*.csv` and fails if the prose and the data disagree. It reads this file and
the notes together, since the detail lives in `notes/METHODS.md` now. It runs in CI on
every push, because prose goes stale when the underlying data is regenerated rather
than when the prose is edited, which is precisely how the ridge-point error above
survived for several hours. Independently of that, `verify/` recomputes every
published quantity from the rawest form of it in the repository, by another
route, and CI fails if any of them disagrees.

## 5. Method and structure
The work is organised into waves, each one verifiable on its own before the next depends on it.

`fa/ref/` holds the fp64 ground truth, the naive, chunked and backend-forced SDPA
baselines, and the NumPy online-softmax reference written in the shape the Triton
kernel will take; `fa/triton/` and `fa/cuda/` are empty and waiting on hardware. The
suite is 527 tests, 192 of them xfail pending a GPU, and it was written against the
references before any kernel existed, since a harness written afterwards tends to
encode the kernel's own bugs as expected behaviour. Correctness is a relative bar
rather than a fixed tolerance: a kernel's error against the fp64 reference must be no
worse than twice the naive implementation's error against that same reference.

Full detail in [notes/METHODS.md](notes/METHODS.md#5-method-and-structure).
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

Twice I reported a trend from single runs and twice had to withdraw it. The fusion
speedup looked like a clean climb to 4.47× until five repeats flattened it near 3×,
and I called attention memory-bound off one ridge point before noticing that its band,
20.08 to 40.55, straddles naive's arithmetic intensity of 31.51. My out-of-memory
prediction was 19.2% out because the textbook model counts two `N²` tensors and the
real one holds 3.16, the third being the `masked_fill` intermediate; with three the
prediction lands 2.7% under the measured failure. I also expected tiling to be the
win when it is 0.56×, and I trusted `sdpa_kernel` without checking it was honoured,
which on MPS it is not.

Two more were found by the checks in `verify/` rather than by me: the §1 traffic table
counted half the score traffic it said it counted, and §2 called a measurement a
trend prediction.

Full detail in [notes/METHODS.md](notes/METHODS.md#7-errors-worth-recording).
## 8. References

Each paper is listed because the implementation follows it, not as background reading.

- **Milakov, Gimelshein. Online normalizer calculation for softmax. 2018.** [arXiv:1805.02867](https://arxiv.org/abs/1805.02867) The two page result the whole construction rests on.
- **Rabe, Staats. Self-attention Does Not Need O(n^2) Memory. 2021.** [arXiv:2112.05682](https://arxiv.org/abs/2112.05682) The memory result without the IO framing. `chunked_attention` here is essentially their construction, and section 2 measures why that is not sufficient on its own.
- **Dao, Fu, Ermon, Rudra, Ré. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022.** [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) The IO-awareness that makes the difference, with the complexity argument in its section 3.2.
- **Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. 2023.** [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) Work partitioning, and the loop ordering the NumPy reference is written in.
- **Shah, Bikshandi, Zhang et al. FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. NeurIPS 2024.** [arXiv:2407.08608](https://arxiv.org/abs/2407.08608) Hopper warp specialisation and FP8. Out of reach without that hardware, and not measured here.
- **Kwon, Li, Zhuang et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023.** [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) The block table design behind the paged cache task.

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
