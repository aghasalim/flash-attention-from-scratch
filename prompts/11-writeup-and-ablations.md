# Task 11 — The write-up and the ablation study

**Wave:** 5 (serial — run last, alone)
**OWNS:** `notes/paper.md`, `README.md`
**READS:** everything

## Context

The code is done. This task produces the artifact that makes the work legible to someone who won't read the code — which is most people, including most interviewers. A serious write-up with honest ablations is what separates a project from a repo.

## Task

**1. `notes/paper.md` — a paper-style write-up. Target 3000–5000 words.**

Sections:

- **Abstract** — what was built, on what hardware, what the headline numbers are
- **Background** — attention as a memory-bandwidth problem. Reuse the roofline analysis from task 01, with the plot
- **Method** — the online softmax derivation (task 02), the tiled algorithm, the IO-complexity argument, the backward recomputation scheme (task 05). Include the induction proof and the `dS = P ∘ (dP - D)` derivation
- **Implementation** — Triton kernel design, the two-kernel backward split and why atomics lost, causal block skipping, the CUDA C++ path and what it revealed
- **Evaluation** — the full benchmark suite. Latency, memory, TFLOP/s, % of peak, achieved bandwidth. Every plot from `results/`
- **Ablations** — see below
- **Limitations** — what doesn't work, what wasn't measured, where the implementation is behind the official one and by how much. Be specific
- **Related work** — FlashAttention 1/2/3, Rabe & Staats, Milakov & Gimelshein, PagedAttention, and where this sits relative to each

**2. The ablation table.** This is the most valuable single artifact in the repo. Each row is one design decision, isolated, measured:

| Ablation | Config | Latency | Δ | Notes |
|---|---|---|---|---|
| `exp2` vs `exp` | N=4096 | | | |
| fp32 vs fp16 accumulator | N=4096 | | | *and* the max error, which is the real point |
| BLOCK_M sweep {32,64,128,256} | N=4096 | | | |
| BLOCK_N sweep {32,64,128} | N=4096 | | | |
| num_stages {1..5} | N=4096 | | | where the pipeline stops helping |
| num_warps {2,4,8} | N=4096 | | | |
| causal block skipping on/off | N=4096 | | | |
| backward: atomic vs split kernels | N=4096 | | | from task 05 |
| smem swizzling on/off | N=4096 | | | bank conflicts, from task 10 |
| `cp.async` on/off | N=4096 | | | from task 10 |
| Flash-Decoding vs standard | B=1, N=32768 | | | from task 09 |
| GQA(8) vs MHA — KV memory | N=32768 | | | |
| varlen vs padded | lognormal lengths | | | |

Every cell from `results/`. Any cell you couldn't measure gets "not measured," not an estimate.

**3. Three interpretive sections that are the actual payoff.**

- **"Where did the speedup come from?"** Decompose the total gain over naive attention into: reduced HBM traffic, kernel fusion (fewer launches), causal skipping, and tuning. Attribute with numbers from the ablations. Most people can quote "FlashAttention is 2–4× faster"; almost nobody can say *which fraction came from which mechanism on their hardware*. That decomposition is the thing worth having.
- **"What the profiler said."** Occupancy, stall reasons, bandwidth utilization for the final kernel. What's the current bottleneck, and what would you do next? A specific, evidence-backed answer to "what would you optimize next" is one of the most common deep-dive interview questions and one of the least commonly answered well.
- **"What I'd do differently."** Honest retrospective. Which task order was wrong, which optimization wasn't worth it, what you'd cut.

**4. Rewrite `README.md`.** Fill every `TODO` from `results/bench.csv` via `bench/report.py`. Fill the hardware section from `HARDWARE.md`. Tick the feature checkboxes that are actually done and leave the rest unticked — an honest partial checklist is more credible than a complete one. Replace the placeholder "three things I got wrong" with the real ones from `notes/LOGBOOK.md`; the logbook has been accumulating them the whole time and they'll be better than the placeholders.

**5. Verify the whole repo.** Fresh clone, `make setup && make test && make bench`. Every number in the README traceable to a CSV row. Every claim in `paper.md` traceable to a measurement or a citation.

## Acceptance criteria

- `notes/paper.md` complete, every section populated
- Ablation table fully populated from real measurements, with "not measured" where applicable
- README has zero `TODO` markers
- Fresh-clone reproduction works end to end
- Every performance claim traceable to `results/`
- The limitations section is specific — "the CUDA backward is 1.4× slower than the official implementation at D=128, likely because of X" beats "some optimizations remain"

## Gotchas

- Don't overclaim. The official FlashAttention is years of expert CUDA work. Being within 1.5× is a strong result for a from-scratch implementation and saying so plainly is more impressive than implying parity.
- Don't hide the negative results. "I tried X, it was 8% slower, here's why" is signal. Ablation tables with only wins in them are not believable, and experienced readers know it.
- Include failed experiments in the limitations section. They're evidence you explored the space rather than implementing the paper's happy path.

## Finish by

A final LOGBOOK entry: what you'd build next, and — honestly — what fraction of the project you'd say you actually understand versus transcribed. That last one is the question you'll get asked, in some form, in every technical conversation about this work.
