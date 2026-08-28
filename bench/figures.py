"""Draw the README figures from the committed result CSVs.

Every figure in the README is drawn here: the roofline, how latency scales, where
naive runs out of memory, what the OOM ladder found, what fusion buys on the CPU,
what fp16 accumulators cost, and how much causal block-skipping is worth. One
animation is included as well, and it is a schematic of the algorithm rather than a
measurement; it is labelled as such in the frame and in its caption.

Reads saved CSVs only, nothing is re-measured here, so the figures always match the
numbers the write-ups quote. The one exception is the animation, which runs the
repo's own NumPy reference on a seeded input and writes no CSV. Nothing needs CUDA.

    python -m bench.figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch, Rectangle

from bench.style import PALETTE, titled

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

# These three colours mean the same thing in every figure and in the README:
# red is the implementation that materialises the N x N matrix and eventually
# fails, blue is the chunked one, green is the fused one. Taken from PALETTE so
# they match the rest of the plots, but the mapping itself carries meaning.
NAIVE, CHUNKED, FUSED = PALETTE[1], PALETTE[0], PALETTE[2]
IMPL_COLOURS = {"naive": NAIVE, "chunked": CHUNKED, "sdpa": FUSED}
NEUTRAL = PALETTE[5]

# Read from the same hardware fingerprint scripts/check_numbers.py reads, so the
# "% of peak" axis cannot drift away from the file it is derived from.
CPU_FP32_PEAK_GFLOPS = json.loads(
    (REPO / "hardware.json").read_text())["dtypes"]["fp32"]["cpu"]["matmul"]["gflop_s"]


def sweep() -> pd.DataFrame:
    table = pd.read_csv(RESULTS / "roofline.csv")
    return table[table.phase == "sweep"]


def roofline(out: Path) -> Path:
    """Where each implementation lands against this machine's roof.

    The measurement lives in bench/roofline.py. This reads the CSV that script
    wrote and the peaks scripts/env.py measured, so the picture cannot drift away
    from either. The roof is min(matmul peak, bandwidth x arithmetic intensity),
    and the ridge is where those two meet.
    """
    table = sweep()
    table = table[table.status == "ok"]
    hardware = json.loads((REPO / "hardware.json").read_text())
    peaks = {
        "mps": (hardware["dtypes"]["fp16"]["mps"]["matmul"]["gflop_s"],
                hardware["bandwidth"]["mps"]["gb_per_s"], "fp16"),
        "cpu": (hardware["dtypes"]["fp32"]["cpu"]["matmul"]["gflop_s"],
                hardware["bandwidth"]["cpu"]["gb_per_s"], "fp32"),
    }
    devices = [d for d in ("mps", "cpu") if (table.device == d).any()]
    markers = {"naive": "o", "chunked": "s", "sdpa": "^"}

    figure, axes = plt.subplots(1, len(devices), figsize=(6.5 * len(devices), 5.2),
                                squeeze=False)
    for axis, device in zip(axes[0], devices, strict=True):
        peak, bandwidth, dtype = peaks[device]
        ridge = peak / bandwidth
        # Span the roof over the intensities this device actually reached, so the
        # panel is not mostly empty axis on the CPU side, where the sweep stops at
        # N=4096 and the intensities only reach a fraction of the MPS range.
        intensity = table[table.device == device].arithmetic_intensity_flop_per_byte
        xs = np.geomspace(intensity.min() / 6, intensity.max() * 6, 200)
        axis.plot(xs, [min(peak, bandwidth * x) for x in xs], color="#333333", lw=1.6,
                  label=f"roof: min({peak:.0f} GFLOP/s, {bandwidth:.1f} GB/s x AI)")
        axis.axvline(ridge, color="#999999", ls=":", lw=1.2)
        # Blended transform so the label sits on the axis floor whatever the data
        # does, instead of landing on top of a cluster of points.
        axis.text(ridge * 1.15, 0.02, f"ridge {ridge:.1f} FLOP/byte", rotation=90,
                  transform=axis.get_xaxis_transform(), ha="left", va="bottom",
                  fontsize=8.6, color="#5a5a5a")
        for impl, colour in IMPL_COLOURS.items():
            rows = table[(table.device == device) & (table.impl == impl)]
            if rows.empty:
                continue
            axis.scatter(rows.arithmetic_intensity_flop_per_byte, rows.achieved_gflop_s,
                         marker=markers[impl], color=colour, s=44, alpha=0.85,
                         edgecolors="white", linewidths=0.5,
                         label=f"{impl}, N={int(rows.N.min())} to {int(rows.N.max())}")
        axis.set_xscale("log")
        axis.set_yscale("log")
        # Open a band under the lowest point so the legend has somewhere to sit that
        # is not on top of the data. The legend has no frame, so an overlap here does
        # not look like an overlap, it looks like a label attached to a point.
        floor = min(float(table[table.device == device].achieved_gflop_s.min()),
                    bandwidth * xs[0]) / 5
        axis.set_ylim(bottom=floor)
        axis.set_xlabel("arithmetic intensity (analytic FLOP per analytic HBM byte)")
        axis.set_ylabel("achieved GFLOP/s (executed FLOPs / median latency)")
        titled(axis, f"{device.upper()}, {dtype}",
               "one point per sweep configuration, causal and non-causal, B=4 H=32 D=64")
        axis.legend(loc="lower right", fontsize=8.6)

    figure.tight_layout(rect=(0, 0.055, 1, 1))
    figure.text(0.5, 0.018,
                "Apple M4. There is no CUDA device on this machine, so no CUDA roofline "
                "is measured. The roof comes from scripts/env.py on this machine.",
                ha="center", fontsize=8.6, color="#5a5a5a")
    figure.savefig(out)
    plt.close(figure)
    return out


def latency_scaling(out: Path) -> Path:
    """Latency against sequence length, with the point where naive stops fitting.

    The crossover is the whole argument for tiling. Chunked attention is slower
    than naive at every size that fits in memory, and then 38x faster at the first
    size that does not, because naive is no longer running, it is swapping.

    Restricted to the non-causal rows. An earlier version took the median over the
    causal and non-causal runs together, which is two different configurations
    averaged into one line and gave a 4096 ratio of about 20x where the write-ups
    quote 37.95x. The figure now plots the same rows the prose quotes.
    """
    table = sweep()
    table = table[(table.status == "ok") & (table.device == "mps")
                  & (~table.causal.astype(bool))]

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for impl, colour in IMPL_COLOURS.items():
        rows = table[table.impl == impl].groupby("N").latency_ms_median.median()
        if rows.empty:
            continue
        left.plot(rows.index, rows.values, "o-", color=colour, label=impl)
    left.set_xscale("log", base=2)
    left.set_yscale("log")
    left.set_xlabel("sequence length N (tokens)")
    left.set_ylabel("median latency (ms)")
    left.text(0.97, 0.05, "naive fails to allocate at N=8192 and above",
              transform=left.transAxes, fontsize=9, color=NAIVE, ha="right")
    titled(left, "Naive is fastest until it cannot run at all",
           "MPS fp16, non-causal, B=4 H=32 D=64, median of the timed iterations")
    left.legend(loc="upper left")

    naive = table[table.impl == "naive"].groupby("N").latency_ms_median.median()
    chunked = table[table.impl == "chunked"].groupby("N").latency_ms_median.median()
    shared = sorted(set(naive.index) & set(chunked.index))
    ratio = [naive[n] / chunked[n] for n in shared]
    labels = [str(n) for n in shared]
    # Stems from 1.0 rather than bars from the axis floor: on a log scale a bar
    # starts wherever the axis happens to end, which reads as a magnitude it is not.
    # The band below 1.0 is shaded and every stem is labelled, because on a log axis
    # a 0.57x stub is tiny beside a 37.9x one and would otherwise read as nothing
    # happening. Colour carries the winner: red is naive, blue is chunked, the same
    # meaning they have in the left panel.
    right.set_yscale("log")
    right.set_ylim(0.38, 90)
    right.axhspan(0.38, 1.0, color="#f4f4f4", zorder=0)
    for x, r in zip(labels, ratio, strict=True):
        colour = CHUNKED if r > 1 else NAIVE
        right.vlines(x, 1.0, r, color=colour, lw=9, alpha=0.9)
        right.plot([x], [r], "o", color=colour, markersize=8)
        above = r > 1
        right.annotate(f"{r:.2f}x", xy=(x, r * (1.22 if above else 0.86)),
                       ha="center", va="bottom" if above else "top",
                       fontsize=9.5, color=colour)
    right.axhline(1.0, color="0.35", ls="--", lw=1.2)
    right.set_xlabel("sequence length N (tokens)")
    right.set_ylabel("naive latency / chunked latency")
    right.text(0.44, 0.195, "chunked faster", transform=right.transAxes,
               fontsize=9, color=CHUNKED, va="bottom")
    right.text(0.44, 0.160, "naive faster", transform=right.transAxes,
               fontsize=9, color=NAIVE, va="top")
    right.annotate("and only because the naive run\nat this size is swapping, not computing",
                   xy=(labels[-1], ratio[-1] * 0.55), xytext=(0.26, 0.62),
                   textcoords="axes fraction", fontsize=9, color="#333333",
                   arrowprops=dict(arrowstyle="->", color="#333333", lw=1.1))
    titled(right, "Chunking wins only where naive falls over",
           "same runs as the left panel; above the line, chunking is faster")

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def memory(out: Path) -> Path:
    """Analytic HBM traffic per implementation, and where each one OOMs.

    The score matrix is N^2 elements. Naive materialises it; chunked never does.
    That is a memory-model claim before it is a speed claim, and the OOM column is
    where the model becomes visible.
    """
    table = sweep()
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    ok = table[table.status == "ok"]
    # Naive and chunked lie on top of each other to within a line width. Dashing
    # the chunked line is the only way both are visible; solid on solid just hides
    # one of them and makes the reader think a curve is missing.
    styles = {"naive": "-", "chunked": "--", "sdpa": "-"}
    for impl, colour in IMPL_COLOURS.items():
        rows = ok[(ok.impl == impl) & (ok.device == "mps")]
        rows = rows.groupby("N").hbm_bytes_analytic.median() / 1e9
        if rows.empty:
            continue
        left.plot(rows.index, rows.values, marker="o", ls=styles[impl],
                  color=colour, label=impl)
    left.set_xscale("log", base=2)
    left.set_yscale("log")
    left.set_xlabel("sequence length N (tokens)")
    left.set_ylabel("analytic HBM traffic per forward pass (GB)")
    titled(left, "Chunking moves the same bytes, it just moves them later",
           "naive and chunked coincide, so chunked is dashed; only the fused path moves less")
    left.legend(loc="lower right")

    statuses = ["ok", "OOM", "skipped", "unavailable"]
    present = [s for s in statuses if (table.status == s).any()]
    impls = sorted(table.impl.unique())
    bottom = [0] * len(impls)
    colours = {"ok": FUSED, "OOM": NAIVE, "skipped": "#c8c8c8", "unavailable": NEUTRAL}
    for status in present:
        counts = [
            len(table[(table.impl == i) & (table.status == status)]) for i in impls
        ]
        right.bar(impls, counts, 0.55, bottom=bottom, label=status,
                  color=colours[status], edgecolor="white", lw=0.8)
        bottom = [b + c for b, c in zip(bottom, counts, strict=True)]
    right.set_ylabel("sweep configurations (count)")
    right.set_ylim(0, max(bottom) * 1.22)
    right.grid(axis="x", visible=False)
    titled(right, "Naive is the only implementation that fails outright",
           "how every configuration in the sweep ended, counted per implementation")
    right.legend(loc="upper left", ncols=3)

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def oom_ladder(out: Path) -> Path:
    """The exact sequence length at which the naive implementation stops fitting.

    Walks N upward in steps of 256 until the allocation fails. Plots the sampled
    allocator peak, which is the column the ladder rows actually carry: the
    analytic traffic column is empty for these rows, and an earlier version of this
    function plotted it and produced an empty axis with one zero-height bar.
    """
    table = pd.read_csv(RESULTS / "roofline.csv")
    ladder = table[table.phase == "oom_ladder"].sort_values("N")
    if ladder.empty:
        figure, ax = plt.subplots(figsize=(9, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "no OOM ladder rows in this run",
                ha="center", va="center", fontsize=12, color="0.4")
        figure.savefig(out)
        plt.close(figure)
        return out

    ok = ladder[ladder.status == "ok"]
    failed = ladder[ladder.status != "ok"]
    labels = [str(int(n)) for n in ladder.N]
    top = float(ok.peak_mem_bytes.max()) / 1e9 * 1.28

    figure, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.bar([str(int(n)) for n in ok.N], ok.peak_mem_bytes / 1e9, 0.6,
           color=CHUNKED, edgecolor="white", lw=0.8, label="ran, sampled peak")
    for n, b in zip(ok.N, ok.peak_mem_bytes, strict=True):
        ax.text(str(int(n)), b / 1e9 + top * 0.02, f"{b / 1e9:.1f}",
                ha="center", fontsize=9, color="#333333")
    if not failed.empty:
        # No bar height exists for these rows: the allocation never completed, so
        # there is no peak to report. An earlier version drew a hatched box to the
        # top of the axis, which reads as a measured 39 GB until you notice the
        # hatch in the legend. A refusal is not a measurement, so it gets a cross on
        # the axis and a gap where the bar would be.
        refused = [str(int(n)) for n in failed.N]
        ax.plot(refused, [0] * len(refused), marker="x", ls="none", color=NAIVE,
                markersize=13, markeredgewidth=2.6, clip_on=False, zorder=5,
                label="allocation refused, no peak to report")
        for x in refused:
            ax.annotate("allocation refused\nnothing to measure", xy=(x, top * 0.045),
                        ha="center", va="bottom", fontsize=9, color=NAIVE)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, top)
    ax.set_xlabel("sequence length N (tokens)")
    ax.set_ylabel("sampled allocator peak (GB)")
    ax.grid(axis="x", visible=False)
    first = int(failed.N.min()) if not failed.empty else None
    titled(ax,
           f"Naive stops fitting between N={int(ok.N.max())} and N={first}"
           if first else "Naive fit at every rung of the ladder",
           "single forward pass, MPS fp16, B=4 H=32 D=64, sampled every 1 ms by the MPS allocator")
    ax.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def fusion(out: Path) -> Path:
    """What removing the score round-trip actually buys, with the spread.

    Two panels: absolute latency, and the speedup with its across-repeat range.
    The error bars are the point of this figure. A single run of this benchmark
    showed a monotone climb to 4.47x that did not reproduce, and the interval at
    N=1024 is wide enough to swallow any trend between neighbouring sizes.
    """
    path = RESULTS / "fusion.csv"
    if not path.exists():
        return out
    table = pd.read_csv(path)
    lat = table[table["impl"].isin(["naive-eager", "chunked-eager", "naive-compiled"])]
    spd = table[table["impl"] == "fusion-speedup"].sort_values("N")

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.8, 5.0))

    styles = {"naive-eager": (NAIVE, "o", "naive, eager"),
              "chunked-eager": (CHUNKED, "s", "chunked, eager"),
              "naive-compiled": (FUSED, "^", "naive, torch.compile (fused)")}
    for impl, (colour, marker, label) in styles.items():
        sub = lat[lat["impl"] == impl].sort_values("N")
        left.plot(sub["N"], sub["latency_ms_median"], marker=marker, color=colour,
                  label=label)
        left.fill_between(sub["N"], sub["latency_ms_min"], sub["latency_ms_max"],
                          color=colour, alpha=0.18, linewidth=0)
    left.set_xscale("log", base=2)
    left.set_yscale("log")
    left.set_xlabel("sequence length N (tokens)")
    left.set_ylabel("latency (ms, median of 5 repeats)")
    titled(left, "Chunking does not pay on the CPU, fusion does",
           "CPU fp32, B=2 H=8 D=64, shaded band spans min to max over the 5 repeats")
    left.legend(loc="upper left")

    lo = spd["speedup_median"] - spd["speedup_min"]
    hi = spd["speedup_max"] - spd["speedup_median"]
    right.errorbar(spd["N"], spd["speedup_median"], yerr=[lo, hi], marker="o",
                   color=FUSED, capsize=5, markersize=7)
    right.axhline(1.0, color="0.45", linestyle=":", linewidth=1.2)
    right.text(spd["N"].iloc[0], 1.06, "no benefit", fontsize=9, color="#5a5a5a")
    right.set_xscale("log", base=2)
    right.set_xlabel("sequence length N (tokens)")
    right.set_ylabel("fusion speedup (eager latency / compiled latency)")
    titled(right, "The speedup settles near 3x from N=1024 onward",
           "bars span min to max of the per-repeat ratio, so N=1024 is the unstable point")
    right.set_ylim(0, max(spd["speedup_max"]) * 1.18)

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def throughput(out: Path) -> Path:
    """Achieved throughput as a fraction of this CPU's measured fp32 peak.

    The stable half of the fusion result. The speedup ratio moves around; this
    level shift, roughly 22% of peak to roughly 67%, does not.
    """
    path = RESULTS / "fusion.csv"
    if not path.exists():
        return out
    table = pd.read_csv(path)
    table = table[table["impl"].isin(["naive-eager", "naive-compiled"])]

    figure, axis = plt.subplots(figsize=(9.4, 5.0))
    width = 0.38
    sizes = sorted(table["N"].unique())
    positions = range(len(sizes))

    for offset, (impl, colour, label) in enumerate([
            ("naive-eager", NAIVE, "eager, S and P go to memory"),
            ("naive-compiled", FUSED, "fused, S and P stay on chip")]):
        vals = [table[(table["N"] == n) & (table["impl"] == impl)]["achieved_gflop_s"].iloc[0]
                for n in sizes]
        pct = [v / CPU_FP32_PEAK_GFLOPS * 100 for v in vals]
        bars = axis.bar([p + offset * width for p in positions], pct, width,
                        color=colour, edgecolor="white", lw=0.8, label=label)
        for rect, raw, share in zip(bars, vals, pct, strict=True):
            axis.text(rect.get_x() + rect.get_width() / 2, share + 1.5,
                      f"{share:.0f}%\n{raw:.0f}", ha="center", fontsize=8.5,
                      color="#333333", linespacing=1.25)

    axis.set_xticks([p + width / 2 for p in positions])
    axis.set_xticklabels(sizes)
    axis.set_xlabel("sequence length N (tokens)")
    axis.set_ylabel(f"% of measured fp32 peak ({CPU_FP32_PEAK_GFLOPS:.0f} GFLOP/s)")
    axis.set_ylim(0, 100)
    axis.grid(axis="x", visible=False)
    titled(axis, "Fusion moves attention from memory-starved to compute-limited",
           "labels give the share of peak and the absolute GFLOP/s behind it")
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def causal(out: Path) -> Path:
    """Causal masking is a performance feature only if you skip the hidden blocks.

    Isolated without writing a kernel, by comparing implementations that skip
    against ones that mask a dense N-by-N.
    """
    table = sweep()
    table = table[(table["device"] == "mps") & (table["status"] == "ok")]

    figure, axis = plt.subplots(figsize=(9.6, 5.2))
    top = 2.4
    off_scale = []
    for impl, colour, label in [
            ("naive", NAIVE, "naive, masks a dense N x N"),
            ("chunked", CHUNKED, "chunked, masks a dense N x N"),
            ("sdpa", FUSED, "sdpa, skips the hidden blocks")]:
        xs, ys = [], []
        for n in sorted(table["N"].unique()):
            rows = table[(table["impl"] == impl) & (table["N"] == n)]
            f = rows[~rows["causal"].astype(bool)]["latency_ms_median"]
            c = rows[rows["causal"].astype(bool)]["latency_ms_median"]
            if len(f) and len(c) and float(c.iloc[0]) > 0:
                xs.append(n)
                ys.append(float(f.iloc[0]) / float(c.iloc[0]))
        if not xs:
            continue
        # Clip rather than rescale. naive at N=4096 reads 10.65x, which is NOT a
        # causal saving: its non-causal run was thrashing against the memory limit
        # (46.5 s against 4.4 s), so the ratio measures swap variance. On a linear
        # axis that one point dominates the plot and implies naive benefits most,
        # which is the opposite of the finding.
        inside = [(x, y) for x, y in zip(xs, ys, strict=True) if y <= top]
        axis.plot([x for x, _ in inside], [y for _, y in inside], marker="o",
                  color=colour, label=label)
        off_scale += [(x, y, colour) for x, y in zip(xs, ys, strict=True) if y > top]

    for x, y, colour in off_scale:
        axis.plot([x], [top - 0.02], marker="^", color=colour, markersize=9,
                  clip_on=False)
        axis.annotate(f"{y:.1f}x, off scale: this run was swapping",
                      xy=(x, top - 0.05), xytext=(0.30, 0.86),
                      textcoords="axes fraction", fontsize=9, color=colour,
                      arrowprops=dict(arrowstyle="->", color=colour, lw=1.1,
                                      connectionstyle="arc3,rad=0.18"))

    axis.axhline(2.0, color="0.35", linestyle="--", linewidth=1.2)
    axis.text(512, 2.03, "ceiling: half the blocks are hidden", fontsize=9,
              color="#333333", ha="left")
    axis.axhline(1.0, color="0.55", linestyle=":", linewidth=1.2)
    axis.text(16384, 1.03, "no benefit, masking costs what it saves", fontsize=9,
              color="#5a5a5a", ha="right")

    axis.set_xscale("log", base=2)
    axis.set_ylim(0.6, top)
    axis.set_xlabel("sequence length N (tokens)")
    axis.set_ylabel("non-causal latency / causal latency")
    titled(axis, "Causal masking only pays if the hidden blocks are skipped",
           "MPS fp16, B=4 H=32 D=64, the gap between the green and blue lines is the whole feature")
    axis.legend(loc="center left")
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def accumulator(out: Path) -> Path:
    """What fp16 accumulators cost, read from results/accumulator.csv.

    Written by `python -m fa.ref.online_softmax`. This function does not re-run the
    experiment: an earlier version did, forgot to pass acc_dtype, and drew two
    identical curves asserting a 1x gap where the real one is 3297x.
    """
    path = RESULTS / "accumulator.csv"
    if not path.exists():
        print("   (skipped accumulator.png, run `python -m fa.ref.online_softmax` first)")
        return out
    table = pd.read_csv(path)

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.8, 5.0))
    panels = (
        (left, "fp32 abs", "fp16 abs",
         "An fp16 accumulator costs four orders of magnitude",
         "max |computed - exact fp64|",
         "inputs are fp16 in both arms, only the accumulator dtype differs", "abs gap"),
        (right, "fp32 |sum-1|", "fp16 |sum-1|",
         "The softmax denominator stops summing to one",
         "max |sum(p) - 1|",
         "the same runs, measuring the normaliser rather than the output", "sum gap"),
    )
    for axis, col32, col16, title, ylabel, subtitle, gapcol in panels:
        axis.plot(table["N"], table[col32], marker="o", color=FUSED,
                  label="fp32 accumulator")
        axis.plot(table["N"], table[col16], marker="s", color=NAIVE,
                  label="fp16 accumulator")
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("sequence length N (tokens)")
        axis.set_ylabel(ylabel)
        titled(axis, title, subtitle)
        gap = float(table[gapcol].iloc[-1])
        n_last = int(table["N"].iloc[-1])
        # The middle of both panels is empty: red sits at the top, green at the
        # bottom. Legend and annotation both live there so neither covers a curve.
        axis.annotate(f"{gap:,.0f}x apart at N={n_last}",
                      xy=(n_last, float(table[col16].iloc[-1])),
                      xytext=(0.46, 0.42), textcoords="axes fraction", fontsize=9.5,
                      arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2))
        axis.legend(loc="center left")

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


# --------------------------------------------------------------------------- #
# animation
# --------------------------------------------------------------------------- #

ANIM_SEED = 0
ANIM_N = 64        # small enough that 16 tiles fit on screen and read clearly
ANIM_D = 16
ANIM_BLOCK = 16
ANIM_HOLD = 6      # frames each algorithm step is held for
ANIM_FPS = 14


def _tiling_trace():
    """Run the tiled forward pass once at ANIM_SEED and record every tile step.

    The loop nest here is the one in ``fa.ref.online_softmax._attention_core``,
    unrolled so the running statistics can be snapshotted after each KV block. The
    assert at the bottom is the check that it stayed the same algorithm: the output
    this loop builds has to match what the repo's own reference returns.
    """
    from fa.ref.online_softmax import online_attention

    rng = np.random.default_rng(ANIM_SEED)
    n, d, b = ANIM_N, ANIM_D, ANIM_BLOCK
    q = rng.standard_normal((n, d)).astype(np.float32)
    k = rng.standard_normal((n, d)).astype(np.float32)
    v = rng.standard_normal((n, d)).astype(np.float32)
    scale = np.float32(1.0 / np.sqrt(d))

    steps = []
    o = np.empty((n, d), dtype=np.float32)
    computed = 0
    for q_start in range(0, n, b):
        q_blk = q[q_start:q_start + b]
        m_i = np.full(b, -np.inf, dtype=np.float32)
        l_i = np.zeros(b, dtype=np.float32)
        acc = np.zeros((b, d), dtype=np.float32)
        for kv_start in range(0, n, b):
            s = (q_blk @ k[kv_start:kv_start + b].T) * scale
            m_new = np.maximum(m_i, s.max(axis=1))
            corr = np.exp(m_i - m_new)
            p = np.exp(s - m_new[:, None])
            l_i = l_i * corr + p.sum(axis=1)
            acc = acc * corr[:, None] + p @ v[kv_start:kv_start + b]
            m_i = m_new
            computed += b * b
            steps.append({"q_start": q_start, "kv_start": kv_start, "s": s.copy(),
                          "m": m_i.copy(), "l": l_i.copy(), "computed": computed})
        o[q_start:q_start + b] = acc / l_i[:, None]

    reference = online_attention(q, k, v, b, b)
    assert np.abs(o - reference).max() < 1e-5, "trace diverged from fa.ref"
    return steps


def tiling_animation(out: Path) -> Path:
    """Schematic: the blockwise sweep with the running online-softmax statistics.

    This is the one figure here that is not a measurement. It illustrates why the
    N by N score matrix never has to exist: one tile is resident at a time, and the
    only state carried between tiles is two floats per query row. The frame says
    SCHEMATIC so it cannot be mistaken for data.
    """
    steps = _tiling_trace()
    n, b = ANIM_N, ANIM_BLOCK
    m_lo = min(float(s["m"].min()) for s in steps)
    m_hi = max(float(s["m"].max()) for s in steps)
    l_hi = max(float(s["l"].max()) for s in steps)
    v_lo = min(float(s["s"].min()) for s in steps)
    v_hi = max(float(s["s"].max()) for s in steps)

    figure = plt.figure(figsize=(9.6, 4.7))
    spec = figure.add_gridspec(2, 2, width_ratios=[1.05, 1.0],
                               left=0.075, right=0.985, top=0.775, bottom=0.275,
                               wspace=0.26, hspace=0.75)
    grid = figure.add_subplot(spec[:, 0])
    ax_m = figure.add_subplot(spec[0, 1])
    ax_l = figure.add_subplot(spec[1, 1])

    figure.text(0.5, 0.085,
                "SCHEMATIC of the algorithm, not a measurement. "
                f"N={n}, D={ANIM_D}, tile {b} by {b}, seed {ANIM_SEED}, "
                "statistics from the reference in fa/ref/online_softmax.py.",
                ha="center", fontsize=8.4, color="#5a5a5a")
    figure.text(0.5, 0.030,
                "The coloured square is the only block of scores that exists at that "
                "moment. The pale line on the right is the same statistic one tile earlier.",
                ha="center", fontsize=8.4, color="#5a5a5a")

    key = [Patch(facecolor="#9ec4e0", edgecolor=CHUNKED, label="live tile"),
           Patch(facecolor="#dcdcdc", edgecolor="#9a9a9a", label="computed, then freed"),
           Patch(facecolor="white", edgecolor="#9a9a9a", label="never computed")]

    def draw(frame: int) -> None:
        index = frame // ANIM_HOLD
        step = steps[index]
        q_start, kv_start = step["q_start"], step["kv_start"]
        previous = steps[index - 1] if kv_start else None
        for axis in (grid, ax_m, ax_l):
            axis.clear()

        grid.set_xlim(0, n)
        grid.set_ylim(n, 0)
        grid.set_aspect("equal", adjustable="box")
        grid.grid(visible=False)
        for qs in range(0, n, b):
            for ks in range(0, n, b):
                freed = qs < q_start or (qs == q_start and ks < kv_start)
                grid.add_patch(Rectangle(
                    (ks, qs), b, b, facecolor="#dcdcdc" if freed else "white",
                    edgecolor="#9a9a9a", lw=1.0))
        grid.axhspan(q_start, q_start + b, color=CHUNKED, alpha=0.12, zorder=2)
        grid.imshow(step["s"], cmap="Blues", vmin=v_lo, vmax=v_hi,
                    extent=(kv_start, kv_start + b, q_start + b, q_start),
                    aspect="auto", zorder=3)
        grid.add_patch(Rectangle((kv_start, q_start), b, b, facecolor="none",
                                 edgecolor=CHUNKED, lw=2.4, zorder=4))
        grid.set_xticks(range(0, n + 1, b))
        grid.set_yticks(range(0, n + 1, b))
        grid.set_xlabel("key index j")
        grid.set_ylabel("query index i")
        titled(grid, "The N by N score matrix never exists",
               f"{b * b} entries live, {step['computed']:,} of {n * n:,} computed so far")
        grid.legend(handles=key, loc="upper center", bbox_to_anchor=(0.5, -0.19),
                    ncols=3, fontsize=8.4)

        rows = np.arange(b)
        for axis, field, colour, ylabel, title in (
                (ax_m, "m", CHUNKED, "m", "m: the running row max"),
                (ax_l, "l", FUSED, "l", "l: the running row sum")):
            if previous is not None:
                axis.step(rows, previous[field], where="mid", color="#c4c4c4", lw=1.4)
            axis.step(rows, step[field], where="mid", color=colour)
            axis.set_ylabel(ylabel)
            axis.set_xlim(-0.5, b - 0.5)
            titled(axis, title)
        ax_m.set_ylim(m_lo - 0.25, m_hi + 0.25)
        ax_m.set_xticklabels([])
        ax_l.set_ylim(0, l_hi * 1.1)
        ax_l.set_xlabel("row within the Q block")

    frames = len(steps) * ANIM_HOLD
    anim = FuncAnimation(figure, draw, frames=frames, interval=1000 / ANIM_FPS)
    anim.save(out, writer=PillowWriter(fps=ANIM_FPS), dpi=88)
    plt.close(figure)
    return out


def main() -> None:
    for path in (
        roofline(RESULTS / "roofline.png"),
        latency_scaling(RESULTS / "latency-scaling.png"),
        memory(RESULTS / "memory.png"),
        oom_ladder(RESULTS / "oom-ladder.png"),
        fusion(RESULTS / "fusion.png"),
        throughput(RESULTS / "throughput.png"),
        causal(RESULTS / "causal-skipping.png"),
        accumulator(RESULTS / "accumulator.png"),
        tiling_animation(RESULTS / "online-softmax-tiling.gif"),
    ):
        size = path.stat().st_size / 1e6 if path.exists() else 0.0
        print(f"-> {path.relative_to(REPO)}  ({size:.2f} MB)")


if __name__ == "__main__":
    main()
