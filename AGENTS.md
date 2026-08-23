# AGENTS.md — how to run these prompts

Every file in `prompts/` is a self-contained task spec. Hand one to an agent (Claude Code, Cursor, Codex, whatever you use) as the whole prompt. Do not paste two at once.

## Running them on your machine

```bash
# one agent, one task
claude "$(cat prompts/03-triton-forward.md)"

# or inside an interactive session
> Read prompts/03-triton-forward.md and execute it end to end.
```

## Parallelism: use git worktrees, not one directory

Multiple agents editing the same checkout will destroy each other's work. Give each concurrent agent its own worktree:

```bash
git worktree add ../fa-05-bwd    -b task/05-bwd
git worktree add ../fa-06-causal -b task/06-causal
git worktree add ../fa-07-bench  -b task/07-bench
# three terminals, three agents, three directories, one repo
```

Merge back to `main` in wave order. Resolve conflicts yourself — do not let an agent resolve a merge it wasn't scoped for.

## Dependency graph

```
                        ┌──────────────────┐
                        │ 00 bootstrap     │   run alone, first, no exceptions
                        └────────┬─────────┘
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
        ┌────────────────┐ ┌───────────┐ ┌──────────────┐
  W1 →  │ 01 baseline    │ │ 02 online │ │ 04 test      │   ← 3 agents in parallel
        │    + roofline  │ │  softmax  │ │   harness    │
        └────────┬───────┘ └─────┬─────┘ └──────┬───────┘
                 └───────────────┼──────────────┘
                                 ▼
                        ┌──────────────────┐
  W2 →                  │ 03 triton fwd    │   run alone — everything below needs it
                        └────────┬─────────┘
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
        ┌────────────────┐ ┌───────────┐ ┌──────────────┐
  W3 →  │ 05 triton bwd  │ │ 06 causal │ │ 07 autotune  │   ← 3 in parallel
        │                │ │  + masks  │ │   + bench    │
        └────────┬───────┘ └─────┬─────┘ └──────┬───────┘
                 └───────────────┼──────────────┘
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
        ┌────────────────┐ ┌───────────┐ ┌──────────────┐
  W4 →  │ 08 gqa/varlen  │ │ 09 flash- │ │ 10 cuda c++  │   ← 3 in parallel
        │    /dropout    │ │  decoding │ │    wmma      │
        └────────┬───────┘ └─────┬─────┘ └──────┬───────┘
                 └───────────────┼──────────────┘
                                 ▼
                        ┌──────────────────┐
  W5 →                  │ 11 writeup       │   run alone, last
                        └──────────────────┘
```

**Waves 1, 3, and 4 are the parallel ones.** Waves 0, 2, and 5 are serial by nature — 00 creates the scaffold everything imports, 03 is the kernel everything else extends, 11 reads every result file in the repo.

Realistic pacing if you're doing this properly rather than speedrunning: W0–W1 in a week, W2 in two to three weeks (the forward kernel is where you actually learn), W3 in a month, W4 is two to three months because the CUDA C++ path alone is a project, W5 is a week. That's the shape of a year.

## File ownership contract

Every prompt declares `OWNS` (may create/edit) and `READS` (may read, must not edit). Agents in the same wave have disjoint `OWNS` sets. This is the whole reason parallel execution is safe here.

| Task | OWNS | READS |
|---|---|---|
| 00 | `scripts/`, `Makefile`, `pyproject.toml`, `fa/__init__.py`, `HARDWARE.md` | — |
| 01 | `fa/ref/naive.py`, `bench/roofline.py`, `notes/00-roofline.md` | `scripts/`, `HARDWARE.md` |
| 02 | `fa/ref/online_softmax.py`, `notes/01-online-softmax.md` | `scripts/` |
| 04 | `tests/`, `fa/ref/fp64.py` | `fa/ref/`, `scripts/` |
| 03 | `fa/triton/fwd.py`, `fa/ops/attention.py` | `fa/ref/`, `tests/`, `notes/01-*` |
| 05 | `fa/triton/bwd.py`, `fa/ops/autograd.py` | `fa/triton/fwd.py`, `tests/` |
| 06 | `fa/triton/masks.py`, `fa/triton/fwd_causal.py` | `fa/triton/fwd.py` |
| 07 | `fa/triton/configs.py`, `bench/`, `results/` | `fa/triton/`, `HARDWARE.md` |
| 08 | `fa/triton/gqa.py`, `fa/triton/varlen.py`, `fa/triton/dropout.py` | `fa/triton/`, `tests/` |
| 09 | `fa/triton/decode.py`, `fa/ops/paged.py` | `fa/triton/`, `bench/` |
| 10 | `fa/cuda/`, `setup.py` | `fa/triton/`, `tests/` |
| 11 | `notes/paper.md`, `README.md` | everything |

## Non-negotiable rules for every agent

Copy these into any prompt you write yourself for this repo.

1. **Never skip a failing test by loosening the tolerance.** If fp16 output disagrees with the fp64 reference by more than the bar in `tests/conftest.py`, the kernel is wrong. Fix the kernel.
2. **The correctness bar is relative, not absolute.** The kernel's error vs. fp64 must be *no worse than* naive fp16 attention's error vs. fp64. That's the bar the FlashAttention paper uses and it's the only fair one.
3. **Never report a benchmark number that didn't come from `bench/`.** No estimates, no numbers from the paper, no "roughly 2× based on the algorithm." Measured or absent.
4. **Every kernel change gets a dated entry in `notes/LOGBOOK.md`** with the before/after number and what you concluded. One entry, three lines, not an essay.
5. **All accumulators are fp32.** Inputs can be fp16/bf16. `acc`, `m_i`, `l_i`, and `D` are fp32 always.
6. **If a claim can't be measured on this GPU, write "not measured on this hardware."** Do not extrapolate to H100.

## Extending to the other seven concepts

The skeleton is deliberately reusable — `prompts/` + `AGENTS.md` + wave graph + ownership table + logbook. When you're ready for concept 2, the same structure holds:

- **MLA (DeepSeek-V2):** waves become `KV-cache profiling → low-rank projection math → naive MLA → decoupled RoPE → cache-size ablation vs MHA/MQA/GQA → fold up-projections into W_Q/W_O at inference`. The interesting hard part is that RoPE does *not* commute with the absorption, which is the entire reason decoupled RoPE exists.
- **Rectified flow:** `interpolant + velocity target → CFM training loop → ODE samplers → reflow → NFE-vs-FID ablation`.
- **Flash-Decoding here in W4 is the bridge** into the MLA project — same memory wall, different floor of the building.
