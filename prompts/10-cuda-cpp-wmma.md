# Task 10 — CUDA C++ implementation with tensor-core intrinsics

**Wave:** 4 (parallel with 08 and 09)
**OWNS:** `fa/cuda/`, `setup.py`
**READS:** `fa/triton/`, `tests/`

## Context

Triton hides the hard parts: shared memory layout, bank conflicts, the async pipeline, register allocation, warp-level data movement. Writing the same kernel in CUDA C++ is how you find out what Triton was doing for you and where it left performance behind.

This is the largest task in the repo. Budget weeks, not days. It's also the one that most clearly demonstrates depth — anyone can call a Triton tutorial; far fewer people have hand-written a double-buffered tensor-core attention kernel and profiled its bank conflicts.

## Task

**1. `fa/cuda/attention_fwd.cu` — forward kernel.**

Structure: one thread block per Q block, `BLOCK_M × HEAD_DIM` in shared memory for Q, double-buffered `BLOCK_N × HEAD_DIM` for K and V.

Required techniques, in order of implementation:

- **Tensor cores.** Start with the `wmma` API (`nvcuda::wmma`, 16×16×16 fragments) because it's readable. Once correct, port the inner matmuls to inline PTX `mma.sync.aligned.m16n8k16` for finer control over fragment layout. Keep both behind a compile flag and benchmark the difference — that comparison is a real result.
- **Async copy.** On SM80+, `cp.async.cg.shared.global` to load the next K/V block while computing on the current one. Guard with `__pipeline_commit()` / `__pipeline_wait_prior(N)`. This is what Triton's `num_stages` was doing, made explicit.
- **Swizzled shared memory.** The naive row-major layout causes 32-way bank conflicts when warps read K columns for the transposed matmul. Implement XOR swizzling (`col ^= (row & mask)`) and **measure bank conflicts before and after with `ncu`.** The before/after number is one of the most convincing artifacts you can produce for this project — record it prominently.
- **Register-resident softmax state.** `m_i`, `l_i`, `acc` stay in registers. Warp-level max and sum reductions via `__shfl_xor_sync`, not shared memory round-trips.
- **`__expf` / `exp2f`** for the exponential, matching the Triton kernel's choice so the comparison is fair.

**2. `fa/cuda/attention_bwd.cu`** — same split-kernel structure as task 05 (preprocess, dK/dV, dQ). Same techniques.

**3. `setup.py`** — `torch.utils.cpp_extension.CUDAExtension`. Set `-arch` from the detected compute capability in `hardware.json`. `-O3`, `--use_fast_math`, `-lineinfo` (needed for `ncu` source correlation). Build must fail loudly with a readable message if the toolkit is missing, not produce a broken import.

**4. `fa/cuda/bindings.cpp`** — pybind11 bindings. Register as a custom op so it composes with `torch.compile`.

**5. Profile and write up `notes/05-cuda-vs-triton.md`.** With `ncu`, collect for both implementations:

- Achieved occupancy, and the limiter (registers, shared memory, or block size)
- Shared memory bank conflicts per request — before and after swizzling
- Warp stall reason breakdown (`stall_long_scoreboard` dominating means memory-bound; `stall_math_pipe_throttle` means compute-bound)
- Tensor-core pipe utilization
- Achieved DRAM throughput vs. measured peak
- Register count per thread and whether any spilling occurs (check the `-Xptxas -v` output)

Then answer, with evidence: **where does Triton leave performance on the table, and where does it beat hand-written code?** An honest answer of "Triton was faster and here's the profiler evidence for why" is a much better result than a hand-tuned kernel that wins by 3% because you tuned it harder. Say which one you got.

## Acceptance criteria

- Passes the full task 04 test suite, identical tolerances, no exceptions carved out
- Builds cleanly on the detected architecture
- Within 1.5× of the Triton kernel at `N=4096, D=64` — beating it is a genuinely good result, but the profiler write-up matters more than the number
- `ncu` numbers collected for both, presented side by side
- Bank conflicts measurably reduced by swizzling, with the before/after in the notes
- No register spilling at `BLOCK_M=128, D=64` (verify with `-Xptxas -v`); if there is spilling, document what you cut to fix it
- The `wmma` vs. inline-PTX `mma` comparison is measured

## Gotchas

- `wmma` fragment layouts are opaque and architecture-specific. Do not assume a mapping from fragment element index to matrix position — it changed between Volta, Ampere, and Hopper. Test on tiny matrices where you can verify by hand.
- `cp.async` requires 4/8/16-byte alignment. Misaligned addresses fail silently or corrupt data rather than erroring.
- Shared memory over 48KB per block needs an opt-in `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`. Without it you get a cryptic launch failure.
- `--use_fast_math` changes `expf` accuracy. Verify the test suite still passes with it on; if it doesn't, use `__expf` explicitly rather than the global flag so the trade is deliberate and documented.
- `ncu` needs elevated permissions on most systems. If it won't run, that's a system config issue — document the workaround in the README rather than skipping the profiling, because the profiling is half the value of this task.

## Finish by

Adding a LOGBOOK entry: CUDA vs Triton latency, occupancy for both, bank conflicts before/after swizzling, dominant stall reason, and a one-sentence honest verdict on which implementation won and why.
