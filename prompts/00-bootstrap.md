# Task 00 — Bootstrap the repo and fingerprint the hardware

**Wave:** 0 (serial — run alone, before everything else)
**OWNS:** `scripts/`, `Makefile`, `pyproject.toml`, `fa/__init__.py`, `HARDWARE.md`, `.gitignore`
**READS:** nothing

## Context

I'm building a fused, IO-aware attention kernel (FlashAttention) from scratch — Triton first, then CUDA C++. Every later task depends on the scaffold and the hardware facts you produce here. Later tasks will read `HARDWARE.md` to decide block sizes, whether to use `cp.async`, and whether FP8 is even available. Get the numbers right; they are load-bearing.

## Task

Create the project skeleton and a hardware fingerprinting script.

**1. Directory scaffold**

```
fa/{__init__,ops/__init__,ref/__init__,triton/__init__,cuda/.gitkeep}
tests/conftest.py
bench/__init__.py
notes/LOGBOOK.md
results/.gitkeep
scripts/env.py
```

**2. `pyproject.toml`** — Python ≥3.10. Deps: `torch`, `triton`, `numpy`, `pytest`, `pandas`, `matplotlib`, `tabulate`. Dev: `ruff`, `mypy`. Do not pin torch/triton versions; record the resolved versions in `HARDWARE.md` instead.

**3. `scripts/env.py`** — must print AND write `HARDWARE.md` containing:

- GPU name, compute capability, SM count
- Total HBM, and **measured** achievable HBM bandwidth (write a small copy-kernel microbenchmark: allocate ~1GB, `x.copy_(y)`, time with CUDA events, report GB/s — do not read the spec sheet number, measure it)
- Shared memory per SM and max shared memory per block (`torch.cuda.get_device_properties` + `cudaDeviceGetAttribute` via `torch.cuda` where available)
- Registers per SM, max threads per block, warp size
- **Measured** peak FP16 tensor-core TFLOP/s: large square `torch.matmul` in fp16 (M=N=K=8192), FLOPs = `2*M*N*K`, median of 50 runs
- Same measurement for BF16 and FP32/TF32
- CUDA version, driver version, PyTorch version, Triton version
- Boolean capability flags: `has_bf16` (SM80+), `has_cp_async` (SM80+), `has_fp8` (SM89+), `has_tma` (SM90+), `has_wgmma` (SM90+)

Also emit `hardware.json` with the same data for programmatic use by `fa/triton/configs.py` later.

**4. `Makefile`** with targets: `setup`, `test`, `bench`, `profile`, `lint`, `clean`. `profile` should shell out to `ncu` if present and print a clear message if not.

**5. `tests/conftest.py`** — a `skip_if_no_cuda` fixture, a global `TOLERANCES` dict keyed by dtype, and a fixed-seed fixture. Leave the tolerance values as `None` with a comment saying task 04 sets them.

**6. `notes/LOGBOOK.md`** — seed it with the format every later task must follow:

```markdown
## YYYY-MM-DD — <one-line title>
**Tried:** ...
**Measured:** ...
**Concluded:** ...
```

**7. `.gitignore`** — venv, `__pycache__`, `*.so`, `build/`, `.ncu-rep`, but **do not ignore `results/`** — the benchmark CSVs are checked in on purpose so the README tables are reproducible.

## Acceptance criteria

- `make setup && python -m scripts.env` runs clean and prints a readable table
- `HARDWARE.md` and `hardware.json` exist and every field is populated from a real query or a real measurement — no placeholders, no spec-sheet copying
- Measured FP16 TFLOP/s is within a plausible fraction (50–90%) of the card's advertised peak. If it's above advertised peak, your timing is wrong — you forgot `torch.cuda.synchronize()` or you're timing kernel launches instead of kernel execution
- `make test` runs and collects zero tests without erroring
- `pytest --collect-only` exits 0

## Gotchas

- Always `torch.cuda.synchronize()` before and after timing, or use `torch.cuda.Event(enable_timing=True)`. Every naive GPU benchmark that looks impossibly fast is measuring launch latency.
- Warm up ≥10 iterations before timing anything. First-call cuBLAS autotuning and lazy module init will poison your first measurement.
- On consumer cards (RTX 40xx) the FP16-with-FP32-accumulate rate is often *half* the FP16-with-FP16-accumulate rate. Measure the fp32-accumulate path, because that's what the attention kernel will actually use.

## Finish by

Adding a LOGBOOK entry recording the measured bandwidth and TFLOP/s numbers and the ratio of measured to advertised peak. That ratio is your realistic ceiling for the rest of the project — write it down.
