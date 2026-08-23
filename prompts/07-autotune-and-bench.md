# Task 07 — Autotuning and the benchmark suite

**Wave:** 3 (parallel with 05 and 06)
**OWNS:** `fa/triton/configs.py`, `bench/`, `results/`
**READS:** `fa/triton/`, `HARDWARE.md`, `hardware.json`

## Context

The README's results table gets generated from this task's output. Every number I ever quote about this project comes from `results/bench.csv`. Make it trustworthy — that matters more than making it fast.

## Task

**1. `fa/triton/configs.py`** — `triton.autotune` config space:

- `BLOCK_M ∈ {32, 64, 128, 256}`
- `BLOCK_N ∈ {32, 64, 128}`
- `num_warps ∈ {2, 4, 8}`
- `num_stages ∈ {1, 2, 3, 4, 5}`

Prune configs that can't fit: shared memory needed is roughly `(BLOCK_N · HEAD_DIM · 2 bytes · 2 tensors · num_stages)`. Read the real limit from `hardware.json` and filter before compilation rather than catching the out-of-resources error after. Key on `(N, HEAD_DIM, causal, dtype)` so the tuner doesn't re-tune on every shape change.

Cache tuned configs to `results/autotune_cache_<gpu_name>.json` and load on import. Retuning on every process start is unbearable and destroys benchmark reproducibility.

**2. `bench/bench.py`** — the main harness:

- Implementations: `naive`, `chunked`, `sdpa(MATH)`, `sdpa(EFFICIENT)`, `sdpa(FLASH)`, `ours-triton`, `ours-cuda` (skip if not built), `flash-attn` (skip if not installed)
- Sweep: `N ∈ {512, 1024, 2048, 4096, 8192, 16384, 32768}`, `D ∈ {64, 128}`, `B·H` held at 128, causal ∈ {T, F}, dtype ∈ {fp16, bf16}
- Modes: forward-only, and forward+backward
- Per config record: median/p10/p90 latency, achieved TFLOP/s, % of measured peak from `hardware.json`, peak memory, achieved DRAM GB/s
- CUDA events, 25 warmup, 100 timed, `torch.cuda.empty_cache()` between configs
- Catch OOM and CUDA errors per-config, record them, keep going
- Write `results/bench.csv`

**3. `bench/plots.py`** — latency vs. N (log-log), TFLOP/s vs. N, memory vs. N, and a % -of-peak bar chart. Save to `results/`.

**4. `bench/report.py`** — reads `bench.csv`, emits the markdown tables, and **rewrites the README results section in place** between `<!-- BENCH:START -->` / `<!-- BENCH:END -->` markers. Add those markers to the README. This is the mechanism that makes it structurally impossible for the README to contain a number that didn't come from a measurement.

**5. `bench/profile.py`** — `ncu` wrapper collecting: achieved occupancy, DRAM read/write throughput, L2 hit rate, shared memory bank conflicts, warp stall reasons, tensor-core utilization. Parse the CSV output into a readable table. Skip gracefully with a clear message if `ncu` isn't available (it needs elevated permissions on many systems — say so in the message rather than failing cryptically).

## Acceptance criteria

- `make bench` produces `results/bench.csv` and the plots
- `make report` rewrites the README tables from the CSV
- Autotuning is cached; second run of the same shape doesn't retune
- Run-to-run variance on the same config is under 5% (if not, you have thermal throttling or another process on the GPU — check `nvidia-smi` and note it in the logbook rather than ignoring it)
- The tuned Triton kernel is within 2× of `sdpa(FLASH)` at `N=4096, D=64`. Within 1.3× is a good result for hand-written Triton
- `bench/profile.py` reports achieved occupancy and at least one identified bottleneck

## Gotchas

- **Autotuning inside a benchmark contaminates the first measurement.** Warm up until the config is locked in, then time.
- GPU clocks drift under sustained load. Consumer cards throttle hard. Either lock clocks with `nvidia-smi -lgc` or interleave implementations rather than running all of one then all of another — otherwise you're measuring thermal state, not kernels.
- Report the **median**, not the mean. One preemption spike ruins a mean.
- `torch.cuda.max_memory_allocated` reports the caching allocator. Reset it between configs and be explicit in the README that it's allocator-reported, not device-reported.
- Include `sdpa(MATH)` even though it OOMs early. The OOM row is informative — it's the memory result made visible.

## Finish by

Adding a LOGBOOK entry with the best config found for `N=4096, D=64, causal`, the achieved % of peak TFLOP/s, the ratio to `sdpa(FLASH)`, and whatever the profiler says the bottleneck is.
