# flash-attention-from-scratch

A fused, IO-aware attention kernel, built from the ground up — online softmax on paper, then Triton, then raw CUDA C++ with `mma` intrinsics — to be benchmarked against `torch.nn.functional.scaled_dot_product_attention` and the official `flash-attn` package.

> **Status: planned, not built.** Right now this repo contains the task specs and
> the research log, and nothing else. There is no kernel, no benchmark, and no
> results table yet. Every number below is a `TODO` because nothing has been
> measured. I'd rather have an empty table than a borrowed one.
>
> I also don't have the hardware yet — this is a Mac, so there's no CUDA device
> to run any of it on. Sorting that out is step zero.

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

Nothing measured yet — `scripts/env.py` (task 00) writes this section, and it
needs an NVIDIA GPU that this machine doesn't have.

- **GPU:** TODO
- **Measured FP16 tensor-core throughput:** TODO TFLOP/s
- **Measured HBM bandwidth:** TODO GB/s (measured with a copy kernel, not the spec sheet)
- **Shared memory per SM:** TODO KB
- CUDA TODO · PyTorch TODO · Triton TODO

Everything in `results/` will be specific to whatever card this ends up running
on. Those numbers won't transfer to a different architecture and I'm not going to
pretend they do.

## Results

<!-- BENCH:START -->
Nothing measured yet. `make bench` writes `results/bench.csv`, and `make report`
regenerates this section from that file — so a number can't appear here without
first existing in a CSV.
<!-- BENCH:END -->

## Feature coverage

Nothing ticked yet. An honest partial checklist beats a complete one.

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

Empty on purpose. [`notes/LOGBOOK.md`](notes/LOGBOOK.md) collects these as they
happen — dated, with the number that moved and what I concluded. This section
gets filled in from real entries at the end (task 11), not from a template.

The logbook is the file I expect to get the most out of. Two months after tuning
a kernel, the difference between "I built a FlashAttention kernel" and being able
to say *why* one config beat another on one specific card is whether you wrote it
down at the time.

## Reading that this is built on

- Dao, Fu, Ermon, Rudra, Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022. — the IO-complexity argument in §3.2 is the load-bearing part.
- Dao. *FlashAttention-2.* 2023. — work partitioning and cutting non-matmul FLOPs. Read before tuning anything.
- Shah et al. *FlashAttention-3.* 2024. — warp specialization and FP8 on Hopper. Only relevant with an H100.
- Milakov, Gimelshein. *Online normalizer calculation for softmax.* 2018. — the two-page paper the whole thing rests on.
- Rabe, Staats. *Self-attention Does Not Need O(n²) Memory.* 2021. — the memory result without the IO-awareness framing.
- The Triton tutorial `06-fused-attention.py` — after writing my own, not before.

## Running it

Once task 00 lands:

```bash
make setup       # venv + deps, prints the hardware table
make test        # correctness suite (needs a GPU)
make bench       # writes results/bench.csv and regenerates the README tables
make profile     # ncu on the forward kernel, occupancy + stall reasons
```

## License

MIT — see [LICENSE](LICENSE).
