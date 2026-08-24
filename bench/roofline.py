"""Roofline sweep for the unfused attention baselines.

    .venv/bin/python -m bench.roofline

Writes ``results/roofline.csv`` and ``results/roofline.png``. Every number in the CSV
comes from a run this script performed; nothing is estimated and nothing is copied
from a paper.

**This machine has no CUDA device** (Apple M4, see ``HARDWARE.md``). Everything below
is measured on MPS (Apple GPU) or the CPU and is labelled as such in the ``device``
column. An MPS number is never a stand-in for a CUDA number. Anything that needs an
NVIDIA card -- CUDA-events timing, ``max_memory_allocated``, the real FlashAttention-2
kernel behind ``SDPBackend.FLASH_ATTENTION``, the naive-vs-flash ratio -- is reported as
``not measured on this hardware (no CUDA device; developed on Apple M4)``.

Deviations from the task grid, all deliberate and all recorded in the CSV:

* fp16 on CPU is not swept: arm64 torch has no fast half GEMM (3.3 GFLOP/s measured by
  ``scripts.env`` vs 1687 GFLOP/s for fp32). CPU rows are fp32, MPS rows are fp16, and
  the ``dtype`` column says which.
* Warmup/iteration counts adapt to a wall-clock budget instead of the fixed 20/100 --
  one naive forward at N=4096 takes seconds here. The counts used are per-row columns.
* CPU is capped at N<=4096 (see ``CPU_N_MAX``); the dropped points are still emitted as
  rows with ``status=skipped`` and the reason.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from fa.ref.naive import SdpaUnavailable, chunked_attention, naive_attention, sdpa_attention, sdpa_report

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
NOT_MEASURED = "not measured on this hardware (no CUDA device; developed on Apple M4)"

B, H, D = 4, 32, 64
N_LIST = (512, 1024, 2048, 4096, 8192, 16384)
CHUNK = 1024
CPU_N_MAX = 4096  # beyond this the fp32 score matrix (>=34 GB at N=8192) swaps, and a
# swap-thrashing run is not a measurement. Dropped points stay in the CSV as `skipped`.

TIME_BUDGET_S = 2.0  # target wall-clock for the timed loop of one config
MAX_ITERS, MIN_ITERS = 100, 3
MAX_WARMUP = 20
SLOW_PILOT_S = 8.0  # a config this slow gets a single timed sample, and the CSV says so

OOM_LADDER_START, OOM_LADDER_STEP, OOM_LADDER_STOP = 4096, 256, 9216
OOM_LADDER_TIME_CAP_S = 120.0


# --------------------------------------------------------------------------- timing


def _sync(device: str):
    if device == "mps":
        return torch.mps.synchronize
    return lambda: None


class _PeakSampler:
    """Sampled peak of the MPS allocator.

    torch.mps has no ``max_memory_allocated``: ``current_allocated_memory`` is an
    instantaneous read, so a background thread samples it. That means the peak is a
    *sampled* peak -- an allocation shorter than the sampling interval can be missed.
    The interval is recorded in the CSV next to the number. On CPU there is no
    per-device allocator to query at all, so peak memory there is left unmeasured
    rather than invented (``ru_maxrss`` is a monotonic process high-water mark and
    cannot be reset between configs).
    """

    interval = 0.001

    def __init__(self, device: str):
        self.device = device
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def supported(self) -> bool:
        return self.device == "mps"

    def __enter__(self):
        if self.supported:
            torch.mps.empty_cache()
            self.peak = torch.mps.current_allocated_memory()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, torch.mps.current_allocated_memory())
            time.sleep(self.interval)

    def __exit__(self, *exc):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
        return False


def time_call(fn, device: str) -> dict:
    """Pilot -> adaptive warmup -> timed loop. Returns median/min/max in ms."""
    sync = _sync(device)
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    pilot = time.perf_counter() - t0
    del out

    if pilot >= SLOW_PILOT_S:
        warmup, iters = 0, 1
    else:
        iters = max(MIN_ITERS, min(MAX_ITERS, int(TIME_BUDGET_S / max(pilot, 1e-6))))
        warmup = MAX_WARMUP if pilot < 0.02 else (3 if pilot < 1.0 else 1)

    for _ in range(warmup):
        fn()
    sync()

    samples = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        samples.append(time.perf_counter() - t0)
        del out

    return {
        "latency_ms_median": statistics.median(samples) * 1e3,
        "latency_ms_min": min(samples) * 1e3,
        "latency_ms_max": max(samples) * 1e3,
        "warmup_iters": warmup + 1,  # the pilot is a warmup too
        "timed_iters": iters,
    }


# ------------------------------------------------------------------- analytic models


def flops_fwd(n: int, causal: bool) -> float:
    """Forward FLOPs. Two matmuls, 2*M*N*K each -> 4*B*H*N^2*D. Causal halves it.

    QK^T: (N,D)x(D,N) -> 2*N*N*D per head.  PV: (N,N)x(N,D) -> 2*N*N*D per head.
    Sum over B*H heads: 4*B*H*N^2*D. Under a causal mask only the lower triangle plus
    the diagonal is live: N*(N+1)/2 of N^2 entries, i.e. half plus O(1/N).
    """
    full = 4.0 * B * H * n * n * D
    return full * ((n + 1) / (2 * n)) if causal else full


def flops_bwd(n: int, causal: bool) -> float:
    """Backward with recomputation: five matmuls of the same shape -> 2.5x forward.

    dV = P^T dO, dP = dO V^T, dQ = dS K, dK = dS^T Q, and S = QK^T recomputed.
    Each is 2*N*N*D per head -> 10*B*H*N^2*D total.
    """
    return 2.5 * flops_fwd(n, causal)


def hbm_bytes(impl: str, n: int, elem: int, chunk: int = CHUNK) -> tuple[float, str]:
    """Analytic HBM traffic for one forward pass, and the name of the model used."""
    qkvo = 4.0 * B * H * n * D * elem
    scores = 4.0 * B * H * n * n * elem  # S written+read, P written+read
    if impl == "naive":
        return qkvo + scores, "naive: 4*B*H*N*D*e (Q,K,V,O) + 4*B*H*N^2*e (S,P written and read)"
    if impl == "chunked":
        tiles = math.ceil(n / chunk)
        # same score traffic as naive, plus Q re-read and the fp32 accumulator
        # read+written once per tile.
        extra = B * H * n * D * (tiles * elem + 2 * elem + 2 * tiles * 4)
        return scores + extra, (
            f"chunked: 4*B*H*N^2*e (same score traffic as naive) + Q re-read and fp32 acc "
            f"read/written once per tile ({tiles} tiles)"
        )
    return qkvo, "fused ideal: 4*B*H*N*D*e -- S and P never reach memory"


# ------------------------------------------------------------------------ the sweep


@dataclass
class Row:
    phase: str = "sweep"
    device: str = ""
    dtype: str = ""
    impl: str = ""
    sdpa_backend: str = ""
    sdpa_backend_honored: str = ""
    B: int = B
    H: int = H
    N: int = 0
    D: int = D
    causal: bool = False
    chunk: str = ""
    status: str = "ok"
    note: str = ""
    latency_ms_median: str | float = ""
    latency_ms_min: str | float = ""
    latency_ms_max: str | float = ""
    warmup_iters: str | int = ""
    timed_iters: str | int = ""
    peak_mem_bytes: str | int = ""
    peak_mem_method: str = ""
    flops_fwd_analytic: str | float = ""
    flops_fwd_executed: str | float = ""
    flops_bwd_analytic: str | float = ""
    hbm_bytes_analytic: str | float = ""
    hbm_model: str = ""
    arithmetic_intensity_flop_per_byte: str | float = ""
    achieved_gflop_s: str | float = ""
    implied_gb_s: str | float = ""


def _oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, MemoryError) or "out of memory" in msg or "invalid buffer size" in msg


def run_config(
    device: str,
    dtype: torch.dtype,
    impl: str,
    n: int,
    causal: bool,
    backend: str,
    honored: bool,
    label: str = "",
) -> Row:
    """``backend`` is what is requested of torch; ``label`` is what the CSV records."""
    elem = torch.finfo(dtype).bits // 8
    row = Row(
        device=device,
        dtype=str(dtype).replace("torch.", ""),
        impl=impl,
        sdpa_backend=label or backend,
        sdpa_backend_honored=str(honored) if impl == "sdpa" else "",
        N=n,
        causal=causal,
        chunk=str(CHUNK) if impl == "chunked" else "",
    )
    if device == "cpu" and n > CPU_N_MAX:
        row.status = "skipped"
        row.note = (
            f"N>{CPU_N_MAX} not swept on CPU: the fp32 score matrix alone is "
            f"{4 * B * H * n * n / 1e9:.0f} GB on a 25.77 GB machine; on macOS a CPU "
            "allocation that large swaps instead of raising, and a swap-thrashing run "
            "is not a measurement"
        )
        return row

    fns = {
        "naive": lambda: naive_attention(q, k, v, causal),
        "chunked": lambda: chunked_attention(q, k, v, CHUNK, causal),
        "sdpa": lambda: sdpa_attention(q, k, v, causal, backend, require_honored=False),
    }
    try:
        q, k, v = (torch.randn(B, H, n, D, device=device, dtype=dtype) for _ in range(3))
        with _PeakSampler(device) as sampler:
            timing = time_call(fns[impl], device)
        for key, val in timing.items():
            setattr(row, key, val)
        if sampler.supported:
            row.peak_mem_bytes = sampler.peak
            row.peak_mem_method = f"mps allocator, torch.mps.current_allocated_memory sampled every {sampler.interval * 1e3:.0f} ms (sampled peak, not an exact high-water mark)"
        else:
            row.peak_mem_method = (
                "not measured: torch has no per-device peak allocator stat for CPU "
                "(torch.cuda.max_memory_allocated does not exist here; ru_maxrss is a "
                "monotonic process high-water mark and cannot be reset per config)"
            )
    except (RuntimeError, MemoryError) as exc:
        row.status = "OOM" if _oom(exc) else "error"
        row.note = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        q = k = v = None
        if device == "mps":
            torch.mps.empty_cache()

    row.flops_fwd_analytic = flops_fwd(n, causal)
    row.flops_bwd_analytic = flops_bwd(n, causal)
    # None of these three skip masked blocks, so the work actually issued is the full
    # N^2 even when causal. Achieved throughput below uses the executed figure.
    row.flops_fwd_executed = 4.0 * B * H * n * n * D
    by, model = hbm_bytes(impl, n, elem)
    row.hbm_bytes_analytic = by
    row.hbm_model = model if impl != "sdpa" or honored else model + " (ASSUMED: backend unknown on this device)"
    row.arithmetic_intensity_flop_per_byte = row.flops_fwd_analytic / by
    if row.status == "ok":
        secs = row.latency_ms_median / 1e3
        row.achieved_gflop_s = row.flops_fwd_executed / secs / 1e9
        row.implied_gb_s = by / secs / 1e9
    return row


def sweep() -> list[Row]:
    rows: list[Row] = []
    plans = [("mps", torch.float16)] if torch.backends.mps.is_available() else []
    plans.append(("cpu", torch.float32))

    for device, dtype in plans:
        report = sdpa_report(device, dtype)
        print(f"\n=== {device} / {dtype} ===")
        print(f"    sdpa backend selection honored: {report.honored} -- {report.reason}")
        for name, info in report.backends.items():
            print(f"      {name:<20} {'runs' if info['ok'] else 'UNAVAILABLE'}")
            if not info["ok"]:
                rows.append(
                    Row(
                        phase="backend_probe",
                        device=device,
                        dtype=str(dtype).replace("torch.", ""),
                        impl="sdpa",
                        sdpa_backend=name,
                        sdpa_backend_honored=str(report.honored),
                        N=0,
                        status="unavailable",
                        note=info["error"][:300],
                    )
                )

        if report.honored:
            sdpa_jobs = [(name, name, True) for name in report.usable]
        else:
            # One row, honestly labelled: forcing a backend does nothing here, so every
            # backend would produce the same kernel and calling one of them FLASH would
            # be a fabrication.
            sdpa_jobs = [
                ("MATH", "MATH requested / NOT HONORED: whatever kernel this device picked", False)
            ]

        for n in N_LIST:
            for causal in (False, True):
                jobs = [("naive", "", "", False), ("chunked", "", "", False)]
                jobs += [("sdpa", req, lab, hon) for req, lab, hon in sdpa_jobs]
                for impl, backend, label, honored in jobs:
                    row = run_config(device, dtype, impl, n, causal, backend or "MATH", honored, label)
                    rows.append(row)
                    lat = f"{row.latency_ms_median:9.2f} ms" if row.status == "ok" else f"{row.status:>12}"
                    tag = f"{impl}{'/' + label[:28] if label else ''}"
                    print(f"    N={n:<6} causal={str(causal):<5} {tag:<40} {lat}")
    return rows


# ------------------------------------------------------------- OOM threshold ladder


def oom_ladder(device: str, dtype: torch.dtype) -> list[Row]:
    """Walk N upward until naive attention actually fails, to compare with the model."""
    rows = []
    print(f"\n=== OOM ladder: naive, {device}/{dtype}, step {OOM_LADDER_STEP} ===")
    for n in range(OOM_LADDER_START, OOM_LADDER_STOP + 1, OOM_LADDER_STEP):
        row = Row(
            phase="oom_ladder",
            device=device,
            dtype=str(dtype).replace("torch.", ""),
            impl="naive",
            N=n,
            causal=False,
        )
        t0 = time.perf_counter()
        try:
            q, k, v = (torch.randn(B, H, n, D, device=device, dtype=dtype) for _ in range(3))
            with _PeakSampler(device) as sampler:
                out = naive_attention(q, k, v, False)
                _sync(device)()
            elapsed = time.perf_counter() - t0
            del out
            row.latency_ms_median = elapsed * 1e3
            row.timed_iters = 1
            row.warmup_iters = 0
            row.note = "single forward pass, no warmup -- this row exists to locate the OOM edge, not to time it"
            if sampler.supported:
                row.peak_mem_bytes = sampler.peak
                row.peak_mem_method = f"mps allocator, sampled every {sampler.interval * 1e3:.0f} ms"
            print(f"    N={n:<6} ok    {elapsed:7.2f} s  peak {sampler.peak / 1e9:6.2f} GB")
        except (RuntimeError, MemoryError) as exc:
            elapsed = time.perf_counter() - t0
            row.status = "OOM" if _oom(exc) else "error"
            row.note = f"{type(exc).__name__}: {exc}"[:300]
            print(f"    N={n:<6} {row.status}  after {elapsed:.2f} s -- {str(exc)[:90]}")
        finally:
            q = k = v = None
            if device == "mps":
                torch.mps.empty_cache()
        rows.append(row)
        if row.status != "ok":
            break
        if elapsed > OOM_LADDER_TIME_CAP_S:
            rows.append(
                Row(
                    phase="oom_ladder",
                    device=device,
                    dtype=str(dtype).replace("torch.", ""),
                    impl="naive",
                    N=n + OOM_LADDER_STEP,
                    status="skipped",
                    note=f"ladder stopped: the previous rung took {elapsed:.0f} s (> {OOM_LADDER_TIME_CAP_S:.0f} s cap)",
                )
            )
            break
    return rows


# ----------------------------------------------------------------- correctness gate


def check_correctness() -> None:
    """naive and chunked must agree with sdpa(MATH), against an fp64 reference.

    The bar is the repo's relative one (rule 2): each implementation's error against the
    same fp64 reference must be no worse than naive-in-this-dtype's error. fp64 does not
    exist on MPS, so the reference is computed on the CPU in fp64.
    ``fa/ref/fp64.py`` belongs to task 04; this is a private copy so task 01 stands alone.
    """
    print("=== correctness gate (fp64 reference, computed on CPU) ===")
    torch.manual_seed(0)
    b, h, n, d = 2, 4, 512, D
    q64, k64, v64 = (torch.randn(b, h, n, d, dtype=torch.float64) for _ in range(3))

    def ref(causal: bool) -> torch.Tensor:
        s = q64 @ k64.transpose(-2, -1) / math.sqrt(d)
        if causal:
            idx = torch.arange(n)
            s = s.masked_fill(idx.unsqueeze(0) > idx.unsqueeze(1), float("-inf"))
        return torch.softmax(s, dim=-1) @ v64

    plans = [("cpu", torch.float32)]
    if torch.backends.mps.is_available():
        plans.append(("mps", torch.float16))

    for device, dtype in plans:
        honored = sdpa_report(device, dtype).honored
        q, k, v = (x.to(device=device, dtype=dtype) for x in (q64, k64, v64))
        for causal in (False, True):
            gold = ref(causal)
            got = {
                "naive": naive_attention(q, k, v, causal),
                "chunked": chunked_attention(q, k, v, 128, causal),
                "sdpa[MATH]": sdpa_attention(q, k, v, causal, "MATH", require_honored=False),
            }
            err = {name: (t.cpu().double() - gold).abs().max().item() for name, t in got.items()}
            bar = err["naive"]  # relative bar: naive in this dtype vs the same fp64 ref
            label = f"{device}/{str(dtype).replace('torch.', '')} causal={causal}"
            print(
                f"  {label:<28} max|x - fp64|: "
                + "  ".join(f"{k2}={v2:.3e}" for k2, v2 in err.items())
                + f"  | vs naive bar: chunked {err['chunked'] / bar:.4f}x, sdpa {err['sdpa[MATH]'] / bar:.4f}x"
                + ("" if honored else "  (sdpa backend selection not honored here)")
            )
            # Rule 2's relative bar, in the form the flash-attention repo's own tests
            # use: candidate error <= 2x the same-dtype naive baseline's error against
            # the same fp64 reference. The ratio is printed above, so the bar does not
            # have to be taken on faith -- on this machine chunked lands at ~1.00x, i.e.
            # the two differ by a fraction of one fp16 ULP. tests/conftest.py (task 04)
            # owns the final tolerance policy for the repo.
            for name in ("chunked", "sdpa[MATH]"):
                assert err[name] <= 2 * bar, (
                    f"{name} on {label} is outside the naive-{dtype} bar: "
                    f"{err[name]:.3e} > 2 x {bar:.3e}"
                )
            # and they must agree with each other, bounded by the same fp16-scale bar
            for name in ("naive", "chunked"):
                diff = (got[name].cpu().double() - got["sdpa[MATH]"].cpu().double()).abs().max().item()
                assert diff <= 4 * bar + 1e-12, f"{name} disagrees with sdpa(MATH) on {label}: {diff:.3e} > {4 * bar:.3e}"
    print("  PASS: chunked and naive agree with sdpa(MATH), and every implementation is")
    print("        within 2x the naive-in-this-dtype error against the same fp64 reference.\n")


# ------------------------------------------------------------------------- plotting


def plot(rows: list[Row]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hw = json.loads((REPO / "hardware.json").read_text())
    peaks = {
        "mps": (hw["dtypes"]["fp16"]["mps"]["matmul"]["gflop_s"], hw["bandwidth"]["mps"]["gb_per_s"], "fp16"),
        "cpu": (hw["dtypes"]["fp32"]["cpu"]["matmul"]["gflop_s"], hw["bandwidth"]["cpu"]["gb_per_s"], "fp32"),
    }
    devices = [d for d in ("mps", "cpu") if any(r.device == d and r.status == "ok" for r in rows)]
    fig, axes = plt.subplots(1, len(devices), figsize=(7.2 * len(devices), 5.6), squeeze=False)

    markers = {"naive": "o", "chunked": "s", "sdpa": "^"}
    colors = {"naive": "#c1440e", "chunked": "#1f6f8b", "sdpa": "#2e7d32"}

    for ax, device in zip(axes[0], devices):
        peak, bw, dt = peaks[device]
        ridge = peak / bw
        x = [2.0**e for e in range(-1, 12)]
        ax.plot(x, [min(peak, bw * xi) for xi in x], color="k", lw=1.6, label=f"roofline: min({peak:.0f} GFLOP/s, {bw:.1f} GB/s x AI)")
        ax.axvline(ridge, color="k", ls=":", lw=1.2)
        ax.annotate(
            f"ridge {ridge:.1f} FLOP/byte",
            xy=(ridge, peak * 0.06),
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
        )
        for impl in ("naive", "chunked", "sdpa"):
            pts = [r for r in rows if r.phase == "sweep" and r.device == device and r.impl == impl and r.status == "ok"]
            if not pts:
                continue
            ax.scatter(
                [r.arithmetic_intensity_flop_per_byte for r in pts],
                [r.achieved_gflop_s for r in pts],
                marker=markers[impl], color=colors[impl], s=44,
                label=f"{impl} (N={min(r.N for r in pts)}..{max(r.N for r in pts)}, causal F+T)",
                alpha=0.85, edgecolors="white", linewidths=0.5,
            )
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("arithmetic intensity  (analytic FLOP / analytic HBM byte)")
        ax.set_ylabel("achieved GFLOP/s (executed FLOPs / measured median latency)")
        ax.set_title(f"{device.upper()} / {dt} -- Apple M4, measured {hw['measured_at']}")
        ax.grid(True, which="both", alpha=0.25, lw=0.4)
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        "Attention roofline on Apple M4 (no CUDA device: CUDA roofline " + NOT_MEASURED + ")",
        fontsize=9,
    )
    fig.tight_layout()
    out = RESULTS / "roofline.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | mps {torch.backends.mps.is_available()}")
    print(f"CUDA roofline: {NOT_MEASURED}\n")
    check_correctness()

    t0 = time.perf_counter()
    rows = sweep()
    if torch.backends.mps.is_available():
        rows += oom_ladder("mps", torch.float16)
    print(f"\nsweep wall clock: {time.perf_counter() - t0:.1f} s")

    csv_path = RESULTS / "roofline.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    png_path = plot(rows)
    print(f"wrote {csv_path}\nwrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
