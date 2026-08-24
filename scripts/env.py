"""Fingerprint the machine this repo is being built on.

Writes HARDWARE.md (human) and hardware.json (machine, for fa/triton/configs.py).

Two paths, picked at runtime:

  * CUDA present -> full NVIDIA fingerprint: device props, SM count, shared memory,
                    measured HBM bandwidth, measured fp16/bf16/fp32/tf32 matmul
                    throughput, SM-version capability flags.
  * no CUDA      -> fingerprint what IS here (Apple GPU via MPS, and the CPU), and
                    set every NVIDIA-only field to null in the JSON, which renders
                    in HARDWARE.md as the exact string in NOT_MEASURED below.

Every number in the output is measured by this script or read from a real device
query. Nothing is copied from a spec sheet and nothing is extrapolated to a GPU
this machine does not have.

Re-run with:  python -m scripts.env
"""

from __future__ import annotations

import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

# AGENTS.md rule 6, verbatim. Any field that would need an NVIDIA GPU to fill in
# gets this string in the Markdown and `null` in the JSON -- never a guess, never
# a number borrowed from the FlashAttention paper, never an MPS number wearing a
# CUDA label.
NOT_MEASURED = "not measured on this hardware (no CUDA device; developed on Apple M4)"

NO_CUDA_NOTE = (
    "Built on an Apple M4 (arm64, macOS). torch.cuda.is_available() is False and Triton "
    "publishes no macOS wheel, so every NVIDIA-only field in this file is null and renders "
    "in HARDWARE.md as: '" + NOT_MEASURED + "'. The mps and cpu numbers here are real "
    "measurements taken by scripts/env.py on this machine; they are Apple GPU / CPU numbers "
    "and are never a stand-in for a CUDA result. Re-run with: python -m scripts.env"
)

# Bandwidth microbenchmark: ~1 GiB per buffer, so copy_ moves ~2 GiB (read + write).
BW_BYTES = 1 << 30
BW_WARMUP = 10
BW_RUNS = 20

# Matmul throughput: square M=N=K, FLOPs = 2*M*N*K.
MM_MAX_N = 8192  # what the task spec asks for; used wherever it fits the budget
MM_MIN_N = 1024
MM_BUDGET_S = 1.5  # per-matmul wall-clock ceiling used to pick the size
MM_WARMUP = 10
MM_RUNS = 20

# Why a field is empty, when the reason is "this is an NVIDIA concept".
CUDA_ONLY_REASON = {
    "compute_capability": "NVIDIA SM version; nothing to query without a CUDA device",
    "sm_count": "streaming multiprocessors are an NVIDIA concept",
    "shared_memory_per_sm_bytes": "CUDA shared memory; Metal threadgroup memory is a "
    "different thing and is not queried here",
    "shared_memory_per_block_bytes": "same; CUDA-only device attribute",
    "regs_per_sm": "CUDA register file per SM; no equivalent query on Metal",
    "max_threads_per_block": "CUDA launch geometry limit",
    "warp_size": "CUDA warp; Metal has 32-wide SIMD groups but that is not the same "
    "attribute and is not queried here",
    "l2_cache_bytes": "reported by cudaDeviceProp only",
    "total_memory_bytes": "dedicated HBM/GDDR; this machine has unified memory instead "
    "(recorded above)",
}


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #


def _progress(msg: str) -> None:
    print(f"  [env] {msg}", file=sys.stderr, flush=True)


def _sync_for(device: str) -> Callable[[], None]:
    """Return the barrier that makes wall-clock timing honest on `device`.

    Without this, every GPU timing here would be measuring kernel launch latency.
    perf_counter around a full sync measures the same thing CUDA events do at this
    granularity (hundreds of ms per iteration), and it is one code path for cuda,
    mps and cpu instead of three.
    """
    if device == "cuda":
        return torch.cuda.synchronize
    if device == "mps":
        return torch.mps.synchronize
    return lambda: None


def _time_once(fn: Callable[[], Any], sync: Callable[[], None]) -> float:
    sync()
    t0 = time.perf_counter()
    fn()
    sync()
    return time.perf_counter() - t0


def _timed(fn: Callable[[], Any], device: str, warmup: int, runs: int) -> dict[str, Any]:
    """Time `runs` iterations after `warmup`, and report the spread, not just the median.

    This is a fanless laptop: back-to-back runs of the same 8192^3 matmul have come out
    25% apart depending on how warm the machine already was. A lone median hides that,
    so min/max/stdev travel with every number.
    """
    sync = _sync_for(device)
    for _ in range(warmup):
        fn()
    sync()
    samples = [_time_once(fn, sync) for _ in range(runs)]
    return {
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "warmup_iters": warmup,
        "timed_iters": runs,
    }


# --------------------------------------------------------------------------- #
# measurements
# --------------------------------------------------------------------------- #


def measure_bandwidth(device: str) -> dict[str, Any]:
    """Copy-kernel bandwidth. Allocates 2 x ~1 GiB and times x.copy_(y)."""
    n = BW_BYTES // 2  # fp16 elements
    try:
        x = torch.empty(n, dtype=torch.float16, device=device)
        y = torch.randn(n, dtype=torch.float16, device=device)
    except Exception as exc:  # noqa: BLE001 - report it, don't kill the fingerprint
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    moved = 2 * x.numel() * x.element_size()  # one read + one write
    t = _timed(lambda: x.copy_(y), device, BW_WARMUP, BW_RUNS)
    return {
        "ok": True,
        "gb_per_s": moved / t["median_s"] / 1e9,
        "gb_per_s_best": moved / t["min_s"] / 1e9,
        "gb_per_s_worst": moved / t["max_s"] / 1e9,
        "buffer_bytes": BW_BYTES,
        "bytes_moved_per_iter": moved,
        "method": "x.copy_(y), fp16, median of timed wall-clock runs, device-synced each run",
        **t,
    }


def _pick_size(device: str, dtype: torch.dtype) -> int:
    """Largest square size up to MM_MAX_N whose matmul stays under MM_BUDGET_S.

    Climbs 1024 -> 8192 measuring at each rung instead of extrapolating, because the
    n^3 extrapolation is off by 2.5x on this CPU (bigger matmuls amortise better).
    The ladder runs double as warmup. Without this, CPU fp16 at 8192 would take ~5
    minutes per iteration on arm64, where torch has no fast half-precision GEMM.
    """
    sync = _sync_for(device)
    n = MM_MIN_N
    while n < MM_MAX_N:
        a = torch.randn(n, n, dtype=torch.float32, device=device).to(dtype)
        t = _time_once(lambda a=a: torch.matmul(a, a), sync)
        if t * 8 > MM_BUDGET_S:  # doubling n multiplies work by 8
            return n
        n *= 2
    return MM_MAX_N


def measure_matmul(device: str, dtype: torch.dtype) -> dict[str, Any]:
    """Square matmul throughput. FLOPs = 2*M*N*K, median of MM_RUNS after MM_WARMUP."""
    try:
        n = _pick_size(device, dtype)
        a = torch.randn(n, n, dtype=torch.float32, device=device).to(dtype)
        b = torch.randn(n, n, dtype=torch.float32, device=device).to(dtype)
        t = _timed(lambda: torch.matmul(a, b), device, MM_WARMUP, MM_RUNS)  # noqa: B023
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    flops = 2.0 * n * n * n
    return {
        "ok": True,
        "gflop_s": flops / t["median_s"] / 1e9,
        "gflop_s_best": flops / t["min_s"] / 1e9,
        "gflop_s_worst": flops / t["max_s"] / 1e9,
        "m_n_k": n,
        "flops_per_matmul": flops,
        "size_note": (
            f"M=N=K={n}"
            + (
                ""
                if n == MM_MAX_N
                else f" (not {MM_MAX_N}: one matmul there exceeds the "
                f"{MM_BUDGET_S}s per-iteration budget on this device/dtype)"
            )
        ),
        "accumulate": "torch.matmul default accumulation for this dtype/backend",
        **t,
    }


def probe_dtype(device: str, dtype: torch.dtype) -> dict[str, Any]:
    """Does this dtype actually work on this device? Probe, never assume."""
    try:
        a = torch.randn(64, 64, device=device, dtype=torch.float32).to(dtype)
        out = (a @ a).float()
        if not torch.isfinite(out).all():
            return {"ok": False, "error": "matmul produced non-finite values"}
        return {"ok": True, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _bench_dtypes(devices: list[str], dtypes: list[tuple[str, torch.dtype]]) -> dict[str, Any]:
    """{dtype_name: {device: {supported, probe_error, matmul}}} for every combination."""
    out: dict[str, Any] = {}
    for name, dt in dtypes:
        per_dev: dict[str, Any] = {}
        for dev in devices:
            probe = probe_dtype(dev, dt)
            rec: dict[str, Any] = {
                "supported": probe["ok"],
                "probe_error": probe["error"],
                "matmul": None,
            }
            if probe["ok"]:
                _progress(f"matmul {name} @ {dev} ...")
                rec["matmul"] = measure_matmul(dev, dt)
            per_dev[dev] = rec
        out[name] = per_dev
    return out


# --------------------------------------------------------------------------- #
# CUDA path (untested here by definition -- there is no CUDA device on this machine)
# --------------------------------------------------------------------------- #


def _nvidia_smi(query: str) -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        return out.stdout.strip().splitlines()[0].strip()
    except Exception:  # noqa: BLE001
        return "unavailable (nvidia-smi did not answer)"


def _triton_version() -> str:
    try:
        import triton  # noqa: PLC0415

        return triton.__version__
    except Exception:  # noqa: BLE001
        return "not installed (no macOS wheels; see the [gpu] extra in pyproject.toml)"


def _prop(props: Any, attr: str) -> Any:
    """Device property, or an explicit 'torch did not expose it' -- never a guess."""
    v = getattr(props, attr, None)
    return v if v is not None else f"unavailable (torch.cuda.get_device_properties has no {attr})"


def cuda_fingerprint() -> dict[str, Any]:
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    cc = props.major * 10 + props.minor

    tf32_before = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    dtypes = _bench_dtypes(
        ["cuda"],
        [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)],
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    _progress("matmul tf32 @ cuda ...")
    dtypes["tf32"] = {
        "cuda": {
            "supported": cc >= 80,
            "probe_error": None if cc >= 80 else f"TF32 needs sm_80+; this is sm_{cc}",
            "matmul": measure_matmul("cuda", torch.float32) if cc >= 80 else None,
            "note": "fp32 inputs with torch.backends.cuda.matmul.allow_tf32 = True",
        }
    }
    torch.backends.cuda.matmul.allow_tf32 = tf32_before

    _progress("bandwidth @ cuda ...")
    return {
        "cuda_available": True,
        "notes": "Measured on a real CUDA device by scripts/env.py.",
        "device_class": "cuda",
        "has_mps": bool(torch.backends.mps.is_available()),
        "device": {
            "name": props.name,
            "gpu_backend": "CUDA",
            "compute_capability": f"{props.major}.{props.minor}",
            "sm_count": props.multi_processor_count,
            "total_memory_bytes": props.total_memory,
            "memory_model": "discrete device memory",
            "warp_size": _prop(props, "warp_size"),
            "max_threads_per_block": _prop(props, "max_threads_per_multi_processor"),
            "shared_memory_per_block_bytes": _prop(props, "shared_memory_per_block"),
            "shared_memory_per_sm_bytes": _prop(props, "shared_memory_per_multiprocessor"),
            "regs_per_sm": _prop(props, "regs_per_multiprocessor"),
            "l2_cache_bytes": _prop(props, "L2_cache_size"),
        },
        "capabilities": {
            "has_bf16": {
                "value": bool(torch.cuda.is_bf16_supported()),
                "reason": f"torch.cuda.is_bf16_supported() on sm_{cc}",
            },
            "has_cp_async": {"value": cc >= 80, "reason": f"sm_{cc}; cp.async needs sm_80+"},
            "has_fp8": {"value": cc >= 89, "reason": f"sm_{cc}; FP8 tensor cores need sm_89+"},
            "has_tma": {"value": cc >= 90, "reason": f"sm_{cc}; TMA needs sm_90+"},
            "has_wgmma": {"value": cc >= 90, "reason": f"sm_{cc}; wgmma needs sm_90+"},
            "has_triton": {
                "value": _triton_version()[0].isdigit(),
                "reason": f"import triton -> {_triton_version()}",
            },
        },
        "bandwidth": {"cuda": measure_bandwidth("cuda")},
        "dtypes": dtypes,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "triton": _triton_version(),
            "cuda_runtime": torch.version.cuda,
            "cudnn": str(torch.backends.cudnn.version()),
            "driver": _nvidia_smi("driver_version"),
        },
        "profiler": {"ncu": shutil.which("ncu") or "not on PATH"},
        "platform": _platform_block(),
    }


# --------------------------------------------------------------------------- #
# Apple / CPU path (what this machine actually is)
# --------------------------------------------------------------------------- #


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _apple_gpu_cores() -> int | None:
    try:
        out = subprocess.run(
            ["system_profiler", "-json", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        for gpu in json.loads(out.stdout).get("SPDisplaysDataType", []):
            cores = gpu.get("sppci_cores")
            if cores:
                return int(cores)
    except Exception:  # noqa: BLE001
        return None
    return None


def _platform_block() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": _sysctl("machdep.cpu.brand_string") or platform.processor(),
    }


def apple_fingerprint() -> dict[str, Any]:
    has_mps = bool(torch.backends.mps.is_available())
    devices = (["mps"] if has_mps else []) + ["cpu"]

    dtypes = _bench_dtypes(
        devices,
        [
            ("fp16", torch.float16),
            ("bf16", torch.bfloat16),
            ("fp32", torch.float32),
            ("fp64", torch.float64),
        ],
    )

    bandwidth = {}
    for dev in devices:
        _progress(f"bandwidth @ {dev} ...")
        bandwidth[dev] = measure_bandwidth(dev)
    bandwidth["cuda"] = None  # null, not a number borrowed from somewhere else

    bf16_mps = dtypes["bf16"].get("mps", {}).get("supported", False)
    fp64_mps = dtypes["fp64"].get("mps", {}).get("supported", False)
    memsize = _sysctl("hw.memsize")

    return {
        "cuda_available": False,
        "notes": NO_CUDA_NOTE,
        "device_class": "apple-mps" if has_mps else "cpu",
        "has_mps": has_mps,
        "device": {
            "name": _sysctl("machdep.cpu.brand_string") or platform.processor(),
            "gpu_backend": "Apple Metal via torch MPS" if has_mps else "none",
            "gpu_cores": _apple_gpu_cores(),
            "cpu_logical_cores": int(_sysctl("hw.ncpu") or 0) or None,
            "cpu_performance_cores": int(_sysctl("hw.perflevel0.logicalcpu") or 0) or None,
            "cpu_efficiency_cores": int(_sysctl("hw.perflevel1.logicalcpu") or 0) or None,
            "unified_memory_bytes": int(memsize) if memsize else None,
            "memory_model": "unified (CPU and GPU share physical memory)",
            # Every field below is null on purpose: it needs an NVIDIA GPU to answer.
            "compute_capability": None,
            "sm_count": None,
            "total_memory_bytes": None,
            "shared_memory_per_sm_bytes": None,
            "shared_memory_per_block_bytes": None,
            "regs_per_sm": None,
            "max_threads_per_block": None,
            "warp_size": None,
            "l2_cache_bytes": None,
        },
        "capabilities": {
            "has_bf16": {
                "value": bool(bf16_mps),
                "reason": "probed: 64x64 bf16 matmul on device='mps'",
            },
            "has_cp_async": {
                "value": False,
                "reason": "CUDA sm_80+ PTX instruction; no CUDA device",
            },
            "has_fp8": {"value": False, "reason": "CUDA sm_89+ tensor-core format; no CUDA device"},
            "has_tma": {"value": False, "reason": "CUDA sm_90+ feature; no CUDA device"},
            "has_wgmma": {"value": False, "reason": "CUDA sm_90+ feature; no CUDA device"},
            "has_triton": {
                "value": False,
                "reason": f"import triton -> {_triton_version()}",
            },
            "fp64_on_mps": {
                "value": bool(fp64_mps),
                "reason": "probed: MPS has no float64, so fp64 references must run on the CPU",
            },
        },
        "bandwidth": bandwidth,
        "dtypes": dtypes,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "triton": _triton_version(),
            "cuda_runtime": None,
            "cudnn": None,
            "driver": None,
            "macos": " ".join(
                filter(None, [_sysctl("kern.osproductversion"), _sysctl("kern.osversion")])
            )
            or None,
        },
        "profiler": {
            "ncu": shutil.which("ncu"),
            "ncu_note": "Nsight Compute is NVIDIA-only and is not installed here; "
            "`make profile` says so instead of pretending",
            "alternative": "Metal System Trace / Xcode Instruments (not wired up by task 00)",
        },
        "platform": _platform_block(),
    }


def fingerprint() -> dict[str, Any]:
    fp = cuda_fingerprint() if torch.cuda.is_available() else apple_fingerprint()
    fp["measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    fp["command"] = "python -m scripts.env"
    return fp


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def _val(v: Any) -> str:
    """null means 'would need an NVIDIA GPU'. Say so in the words rule 6 demands."""
    return NOT_MEASURED if v is None else str(v)


def _fmt_bytes(n: Any) -> str:
    if not isinstance(n, int):
        return _val(n)
    return f"{n / 1e9:.2f} GB ({n} bytes)"


def _table(rows: list[tuple[str, str]], title: str) -> str:
    width = max((len(k) for k, _ in rows), default=10)
    lines = [f"  {title}", "  " + "-" * (width + 40)]
    lines += [f"  {k.ljust(width)}  {v}" for k, v in rows]
    return "\n".join(lines)


def _bw_cell(bw: Any) -> tuple[str, str]:
    """(value, how) for one bandwidth record."""
    if bw is None:
        return NOT_MEASURED, "no CUDA device to run a copy kernel on"
    if not bw.get("ok"):
        return "measurement failed", str(bw.get("error"))
    how = (
        f"`x.copy_(y)` fp16, {bw['buffer_bytes'] / 2**30:.0f} GiB per buffer, "
        f"{bw['bytes_moved_per_iter'] / 2**30:.0f} GiB moved per iter (read+write), "
        f"{bw['warmup_iters']} warmup + median of {bw['timed_iters']} timed runs, "
        f"device-synced each run"
    )
    return (
        f"{bw['gb_per_s']:.1f} GB/s median [{bw['gb_per_s_worst']:.1f}-{bw['gb_per_s_best']:.1f}]",
        how,
    )


def _mm_cell(entry: dict[str, Any]) -> tuple[str, str]:
    """(value, note) for one dtype/device matmul record."""
    if not entry.get("supported"):
        return "dtype unsupported here", str(entry.get("probe_error"))
    mm = entry.get("matmul")
    if mm is None:
        return "not benchmarked", str(entry.get("note", "no measurement taken"))
    if not mm.get("ok"):
        return "measurement failed", str(mm.get("error"))
    return (
        f"{mm['gflop_s']:.1f} GFLOP/s median [{mm['gflop_s_worst']:.1f}-{mm['gflop_s_best']:.1f}]",
        f"{mm['size_note']}, median {mm['median_s'] * 1e3:.1f} ms, "
        f"{mm['warmup_iters']} warmup + {mm['timed_iters']} timed, {mm['accumulate']}",
    )


def render_report(fp: dict[str, Any]) -> str:
    dev = fp["device"]
    out = [
        _table(
            [
                ("device_class", fp["device_class"]),
                ("cuda_available", str(fp["cuda_available"])),
                ("has_mps", str(fp["has_mps"])),
                ("name", _val(dev.get("name"))),
                ("gpu_backend", _val(dev.get("gpu_backend"))),
                ("gpu cores / SMs", _val(dev.get("gpu_cores", dev.get("sm_count")))),
                ("cpu cores", _val(dev.get("cpu_logical_cores", "n/a"))),
                (
                    "memory",
                    _fmt_bytes(dev.get("unified_memory_bytes") or dev.get("total_memory_bytes")),
                ),
                ("compute_capability", _val(dev.get("compute_capability"))),
                ("sm_count", _val(dev.get("sm_count"))),
                ("shared_mem/SM", _fmt_bytes(dev.get("shared_memory_per_sm_bytes"))),
                ("regs/SM", _val(dev.get("regs_per_sm"))),
                ("warp_size", _val(dev.get("warp_size"))),
            ],
            "DEVICE",
        ),
        _table(
            [(name, _bw_cell(bw)[0]) for name, bw in fp["bandwidth"].items()],
            "MEASURED MEMORY BANDWIDTH (copy kernel)",
        ),
        _table(
            [
                (f"{dname} @ {devname}", _mm_cell(entry)[0])
                for dname, per_dev in fp["dtypes"].items()
                for devname, entry in per_dev.items()
            ],
            "MEASURED MATMUL THROUGHPUT (FLOPs = 2*M*N*K)",
        ),
        _table(
            [(k, f"{v['value']}  -- {v['reason']}") for k, v in fp["capabilities"].items()],
            "CAPABILITY FLAGS",
        ),
        _table([(k, _val(v)) for k, v in fp["versions"].items()], "VERSIONS"),
        _table([(k, _val(v)) for k, v in fp["platform"].items()], "PLATFORM"),
    ]
    return "\n\n".join(out)


def render_markdown(fp: dict[str, Any]) -> str:
    dev = fp["device"]
    L: list[str] = [
        "# HARDWARE",
        "",
        f"Generated by `{fp['command']}` at **{fp['measured_at']}**. Every number below was "
        "measured by that script on this machine or read from a real device query. Nothing is "
        "copied from a spec sheet, and nothing is extrapolated to a GPU I do not own.",
        "",
        f"> **{'CUDA device present.' if fp['cuda_available'] else 'No CUDA device.'}** "
        f"{fp['notes']}",
        "",
        "## Summary",
        "",
        "| field | value |",
        "|---|---|",
    ]
    for k, v in [
        ("device_class", fp["device_class"]),
        ("cuda_available", fp["cuda_available"]),
        ("has_mps", fp["has_mps"]),
        ("name", dev.get("name")),
        ("gpu backend", dev.get("gpu_backend")),
        ("GPU cores / SM count", dev.get("gpu_cores", dev.get("sm_count"))),
        ("CPU cores (logical)", dev.get("cpu_logical_cores", "n/a")),
        (
            "CPU perf / efficiency cores",
            f"{dev.get('cpu_performance_cores')} / {dev.get('cpu_efficiency_cores')}",
        ),
        ("memory", _fmt_bytes(dev.get("unified_memory_bytes") or dev.get("total_memory_bytes"))),
        ("memory model", dev.get("memory_model")),
    ]:
        L.append(f"| {k} | {_val(v)} |")

    L += ["", "## NVIDIA device fields", "", "| field | value | why |", "|---|---|---|"]
    for k in (
        "compute_capability",
        "sm_count",
        "total_memory_bytes",
        "shared_memory_per_sm_bytes",
        "shared_memory_per_block_bytes",
        "regs_per_sm",
        "max_threads_per_block",
        "warp_size",
        "l2_cache_bytes",
    ):
        v = dev.get(k)
        why = "" if v is not None else CUDA_ONLY_REASON.get(k, "needs an NVIDIA GPU to query")
        L.append(f"| {k} | {_fmt_bytes(v) if isinstance(v, int) else _val(v)} | {why} |")

    L += [
        "",
        "## Measured memory bandwidth",
        "",
        "| device | achieved | how |",
        "|---|---|---|",
    ]
    for name, bw in fp["bandwidth"].items():
        value, how = _bw_cell(bw)
        L.append(f"| {name} | **{value}** | {how} |")

    L += [
        "",
        "## Measured matmul throughput",
        "",
        "FLOPs = `2*M*N*K`, square matrices. Median over the timed runs with the full min-max",
        "range beside it. This is a fanless laptop: the same matmul has measured 25% apart",
        "across two back-to-back runs depending on how warm the machine already was, so read",
        "the range, not the median, as the honest statement of what this hardware does.",
        "",
        "These are **Apple GPU (MPS) and CPU** numbers. They are not tensor-core numbers and",
        "they are not a stand-in for any CUDA measurement.",
        "",
        "| dtype | device | GFLOP/s | detail |",
        "|---|---|---|---|",
    ]
    for dname, per_dev in fp["dtypes"].items():
        for devname, entry in per_dev.items():
            value, note = _mm_cell(entry)
            L.append(f"| {dname} | {devname} | **{value}** | {note} |")
    if not fp["cuda_available"]:
        L.append(
            f"| fp16/bf16/tf32 tensor core | cuda | {NOT_MEASURED} | "
            "there are no tensor cores on this machine |"
        )

    L += ["", "## Capability flags", "", "| flag | value | why |", "|---|---|---|"]
    for k, v in fp["capabilities"].items():
        L.append(f"| `{k}` | {v['value']} | {v['reason']} |")

    L += ["", "## Versions", "", "| what | version |", "|---|---|"]
    for k, v in fp["versions"].items():
        L.append(f"| {k} | {_val(v)} |")

    L += ["", "## Profiling", "", "| tool | status |", "|---|---|"]
    for k, v in fp.get("profiler", {}).items():
        L.append(f"| {k} | {_val(v)} |")

    L += ["", "## Platform", "", "| field | value |", "|---|---|"]
    for k, v in fp["platform"].items():
        L.append(f"| {k} | {_val(v)} |")

    if not fp["cuda_available"]:
        L += [
            "",
            "## What this means for the rest of the project",
            "",
            "- No NVIDIA GPU here: no Triton (no macOS wheel), no CUDA C++, no `ncu`. Tasks 03",
            "  and 05-10 are written against CUDA and need a rented GPU before they can run.",
            "- `fp64` does not exist on MPS, so every fp64 reference in `fa/ref/` runs on the CPU.",
            "- Memory is unified: there is no host-to-device copy to hide, and CPU and GPU contend",
            "  for the same bandwidth. A roofline drawn here is not a roofline for an A100.",
            "- The CUDA ceiling for this project is unknown. It stays unknown until the kernels",
            "  run on a real NVIDIA card, and no number in this repo will pretend otherwise.",
        ]
    L.append("")
    return "\n".join(L)


def main() -> int:
    fp = fingerprint()
    print(render_report(fp))
    (REPO_ROOT / "HARDWARE.md").write_text(render_markdown(fp))
    (REPO_ROOT / "hardware.json").write_text(json.dumps(fp, indent=2) + "\n")
    print(f"\n  wrote {REPO_ROOT / 'HARDWARE.md'}")
    print(f"  wrote {REPO_ROOT / 'hardware.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
