"""Draw the README figures from the committed result CSVs.

``bench.roofline`` writes ``results/roofline.png`` itself. These cover what that
plot cannot show: how latency scales, where naive runs out of memory, what the OOM
ladder found, what fusion buys on the CPU, what fp16 accumulators cost, and how much
causal block-skipping is worth.

Reads saved CSVs only -- nothing is re-measured here, so the figures always match
the numbers the write-ups quote. Nothing needs CUDA.

    python -m bench.figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

IMPL_COLOURS = {"naive": "#b2182b", "chunked": "#2166ac", "sdpa": "#1a9850"}

# Measured by scripts/env.py on this machine; used to draw the "% of peak" axis.
CPU_FP32_PEAK_GFLOPS = 1738.3


def sweep() -> pd.DataFrame:
    table = pd.read_csv(RESULTS / "roofline.csv")
    return table[table.phase == "sweep"]


def latency_scaling(out: Path) -> Path:
    """Latency against sequence length, with the point where naive stops fitting.

    The crossover is the whole argument for tiling. Chunked attention is slower
    than naive at every size that fits in memory, and then 38x faster at the first
    size that does not, because naive is no longer running -- it is swapping.
    """
    table = sweep()
    table = table[(table.status == "ok") & (table.device == "mps")]

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for impl, colour in IMPL_COLOURS.items():
        rows = table[table.impl == impl].groupby("N").latency_ms_median.median()
        if rows.empty:
            continue
        left.plot(rows.index, rows.values, "o-", color=colour, lw=2, label=impl)
    left.set_xscale("log", base=2)
    left.set_yscale("log")
    left.set_xlabel("sequence length N")
    left.set_ylabel("median latency (ms)")
    left.set_title("MPS, fp16", fontsize=10)
    left.legend(frameon=False, fontsize=9)
    left.spines[["top", "right"]].set_visible(False)

    naive = table[table.impl == "naive"].groupby("N").latency_ms_median.median()
    chunked = table[table.impl == "chunked"].groupby("N").latency_ms_median.median()
    shared = sorted(set(naive.index) & set(chunked.index))
    ratio = [naive[n] / chunked[n] for n in shared]
    right.bar([str(n) for n in shared], ratio,
              color=["#1a9850" if r > 1 else "#b2182b" for r in ratio],
              edgecolor="0.3", lw=0.5)
    right.axhline(1.0, color="0.25", ls="--", lw=1.3)
    right.set_xlabel("sequence length N")
    right.set_ylabel("naive ÷ chunked")
    right.set_yscale("log")
    right.set_title(
        "above 1 means chunking wins; it only does at the size\n"
        "where naive stops fitting",
        fontsize=10,
    )
    right.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def memory(out: Path) -> Path:
    """Analytic HBM traffic per implementation, and where each one OOMs.

    The score matrix is N^2 elements. Naive materialises it; chunked never does.
    That is a memory-model claim before it is a speed claim, and the OOM column is
    where the model becomes visible.
    """
    table = sweep()
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ok = table[table.status == "ok"]
    for impl, colour in IMPL_COLOURS.items():
        rows = ok[(ok.impl == impl) & (ok.device == "mps")]
        rows = rows.groupby("N").hbm_bytes_analytic.median() / 1e9
        if rows.empty:
            continue
        left.plot(rows.index, rows.values, "o-", color=colour, lw=2, label=impl)
    left.set_xscale("log", base=2)
    left.set_yscale("log")
    left.set_xlabel("sequence length N")
    left.set_ylabel("analytic HBM traffic (GB)")
    left.set_title("what each implementation has to move", fontsize=10)
    left.legend(frameon=False, fontsize=9)
    left.spines[["top", "right"]].set_visible(False)

    statuses = ["ok", "OOM", "skipped", "unavailable"]
    present = [s for s in statuses if (table.status == s).any()]
    impls = sorted(table.impl.unique())
    bottom = [0] * len(impls)
    colours = {"ok": "#1a9850", "OOM": "#b2182b", "skipped": "#bdbdbd",
               "unavailable": "#7f7f7f"}
    for status in present:
        counts = [
            len(table[(table.impl == i) & (table.status == status)]) for i in impls
        ]
        right.bar(impls, counts, 0.55, bottom=bottom, label=status,
                  color=colours[status], edgecolor="0.3", lw=0.4)
        bottom = [b + c for b, c in zip(bottom, counts, strict=True)]
    right.set_ylabel("sweep configurations")
    right.set_title("how each configuration ended", fontsize=10)
    right.legend(frameon=False, fontsize=8)
    right.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def oom_ladder(out: Path) -> Path:
    """The exact sequence length at which the naive implementation stops fitting.

    Predicted from the analytic score-matrix size and then measured by walking N
    upward until allocation fails. The point of the ladder is that the prediction
    is checkable rather than asserted.
    """
    table = pd.read_csv(RESULTS / "roofline.csv")
    ladder = table[table.phase == "oom_ladder"].sort_values("N")
    if ladder.empty:
        figure, ax = plt.subplots(figsize=(9, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "no OOM ladder rows in this run",
                ha="center", va="center", fontsize=12, color="0.4")
        figure.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(figure)
        return out

    figure, ax = plt.subplots(figsize=(10, 4.4))
    colours = ["#1a9850" if s == "ok" else "#b2182b" for s in ladder.status]
    ax.bar(ladder.N.astype(str), ladder.hbm_bytes_analytic / 1e9, 0.6,
           color=colours, edgecolor="0.3", lw=0.4)
    ax.set_xlabel("sequence length N")
    ax.set_ylabel("analytic score-matrix traffic (GB)")
    failed = ladder[ladder.status != "ok"]
    first = int(failed.N.min()) if not failed.empty else None
    ax.set_title(
        "Green ran, red failed to allocate."
        + (f" First failure at N={first}." if first else ""),
        fontsize=10,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def fusion(out: Path) -> Path:
    """What removing the score round-trip actually buys, with the spread.

    Two panels: absolute latency, and the speedup with its across-repeat range.
    The error bars are the point of this figure -- a single run of this benchmark
    showed a monotone climb to 4.47x that did not reproduce, and the interval at
    N=1024 is wide enough to swallow any trend between neighbouring sizes.
    """
    path = RESULTS / "fusion.csv"
    if not path.exists():
        return out
    table = pd.read_csv(path)
    lat = table[table["impl"].isin(["naive-eager", "chunked-eager", "naive-compiled"])]
    spd = table[table["impl"] == "fusion-speedup"].sort_values("N")

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5))

    styles = {"naive-eager": ("#b2182b", "o", "naive, eager"),
              "chunked-eager": ("#2166ac", "s", "chunked, eager"),
              "naive-compiled": ("#1a9850", "^", "naive, torch.compile (fused)")}
    for impl, (colour, marker, label) in styles.items():
        sub = lat[lat["impl"] == impl].sort_values("N")
        left.plot(sub["N"], sub["latency_ms_median"], marker=marker, color=colour,
                  label=label, linewidth=1.8, markersize=6)
        left.fill_between(sub["N"], sub["latency_ms_min"], sub["latency_ms_max"],
                          color=colour, alpha=0.18, linewidth=0)
    left.set_xscale("log", base=2)
    left.set_yscale("log")
    left.set_xlabel("sequence length N")
    left.set_ylabel("latency (ms, median of 5 repeats)")
    left.set_title("CPU fp32, B=2 H=8 D=64\nshaded = min-max across repeats")
    left.grid(alpha=0.3, which="both")
    left.legend(frameon=False, fontsize=9)

    lo = spd["speedup_median"] - spd["speedup_min"]
    hi = spd["speedup_max"] - spd["speedup_median"]
    right.errorbar(spd["N"], spd["speedup_median"], yerr=[lo, hi], marker="o",
                   color="#1a9850", capsize=5, linewidth=1.8, markersize=7)
    right.axhline(1.0, color="#666666", linestyle=":", linewidth=1)
    right.text(spd["N"].iloc[0], 1.02, "no benefit", fontsize=8, color="#666666")
    right.set_xscale("log", base=2)
    right.set_xlabel("sequence length N")
    right.set_ylabel("fusion speedup (eager / compiled)")
    right.set_title("Rises to ~3x by N=512, then flat\nbars = range over 5 repeats")
    right.grid(alpha=0.3)
    right.set_ylim(0, max(spd["speedup_max"]) * 1.15)

    figure.suptitle("Kernel fusion on the CPU: measured, not modelled", fontsize=13)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def throughput(out: Path) -> Path:
    """Achieved throughput as a fraction of this CPU's measured fp32 peak.

    The stable half of the fusion result. The speedup ratio moves around; this
    level shift -- roughly 22% of peak to roughly 67% -- does not.
    """
    path = RESULTS / "fusion.csv"
    if not path.exists():
        return out
    table = pd.read_csv(path)
    table = table[table["impl"].isin(["naive-eager", "naive-compiled"])]

    figure, axis = plt.subplots(figsize=(9, 5))
    width = 0.38
    sizes = sorted(table["N"].unique())
    positions = range(len(sizes))

    for offset, (impl, colour, label) in enumerate([
            ("naive-eager", "#b2182b", "eager (S and P go to memory)"),
            ("naive-compiled", "#1a9850", "fused (kept on chip)")]):
        vals = [table[(table["N"] == n) & (table["impl"] == impl)]["achieved_gflop_s"].iloc[0]
                for n in sizes]
        pct = [v / CPU_FP32_PEAK_GFLOPS * 100 for v in vals]
        bars = axis.bar([p + offset * width for p in positions], pct, width,
                        color=colour, label=label)
        for rect, raw, share in zip(bars, vals, pct):
            axis.text(rect.get_x() + rect.get_width() / 2, share + 1.2,
                      f"{share:.0f}%\n{raw:.0f}", ha="center", fontsize=8)

    axis.set_xticks([p + width / 2 for p in positions])
    axis.set_xticklabels(sizes)
    axis.set_xlabel("sequence length N")
    axis.set_ylabel(f"% of measured fp32 peak ({CPU_FP32_PEAK_GFLOPS:.0f} GFLOP/s)")
    axis.set_title("Fusion moves attention from memory-starved to compute-limited\n"
                   "labels show % of peak and absolute GFLOP/s")
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.3, axis="y")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def causal(out: Path) -> Path:
    """Causal masking is a performance feature only if you skip the hidden blocks.

    Isolated without writing a kernel, by comparing implementations that skip
    against ones that mask a dense N-by-N.
    """
    table = sweep()
    table = table[(table["device"] == "mps") & (table["status"] == "ok")]

    figure, axis = plt.subplots(figsize=(9, 5))
    for impl, colour, label in [
            ("naive", "#b2182b", "naive (masks a dense N x N)"),
            ("chunked", "#2166ac", "chunked (masks a dense N x N)"),
            ("sdpa", "#1a9850", "sdpa (skips hidden blocks)")]:
        xs, ys = [], []
        for n in sorted(table["N"].unique()):
            rows = table[(table["impl"] == impl) & (table["N"] == n)]
            f = rows[~rows["causal"].astype(bool)]["latency_ms_median"]
            c = rows[rows["causal"].astype(bool)]["latency_ms_median"]
            if len(f) and len(c) and float(c.iloc[0]) > 0:
                xs.append(n)
                ys.append(float(f.iloc[0]) / float(c.iloc[0]))
        if xs:
            axis.plot(xs, ys, marker="o", color=colour, label=label,
                      linewidth=1.8, markersize=6)

    axis.axhline(2.0, color="#333333", linestyle="--", linewidth=1.2)
    axis.text(560, 2.04, "theoretical ceiling: half the blocks are hidden",
              fontsize=8.5, color="#333333")
    axis.axhline(1.0, color="#666666", linestyle=":", linewidth=1)
    axis.text(560, 0.90, "no benefit -- masking costs more than it saves",
              fontsize=8.5, color="#666666")

    # Clip the y-axis. naive at N=4096 reads 10.65x, which is NOT a causal saving:
    # the non-causal run at that size was thrashing against the memory limit
    # (46.5 s vs 4.4 s), so the ratio measures swap variance. Left on a linear
    # axis it dominates the plot and implies naive benefits most, which is the
    # opposite of the finding. Clipped, and labelled as the artefact it is.
    axis.set_ylim(0.6, 2.35)
    axis.annotate("naive at N=4096 is 10.65x, off-scale:\nits non-causal run was swapping "
                  "(46.5 s vs 4.4 s),\nso that ratio is thrash variance, not a saving",
                  xy=(4096, 2.28), xytext=(0.04, 0.30), textcoords="axes fraction",
                  fontsize=8.5, color="#b2182b", va="bottom",
                  bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#b2182b", alpha=0.9),
                  arrowprops=dict(arrowstyle="->", color="#b2182b", lw=1.1,
                                  connectionstyle="arc3,rad=-0.2"))

    axis.set_xscale("log", base=2)
    axis.set_xlabel("sequence length N")
    axis.set_ylabel("non-causal latency / causal latency")
    axis.set_title("What block skipping is worth, MPS fp16\n"
                   "the gap between the green and blue lines is the whole feature")
    axis.grid(alpha=0.3)
    axis.legend(frameon=False, loc="lower right", fontsize=9)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
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
        print("   (skipped accumulator.png -- run `python -m fa.ref.online_softmax` first)")
        return out
    table = pd.read_csv(path)

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    panels = (
        (left, "fp32 abs", "fp16 abs", "Max absolute error vs exact fp64",
         "max |computed - exact|", "abs gap"),
        (right, "fp32 |sum-1|", "fp16 |sum-1|", "Denominator error: |sum(p) - 1|",
         "max |sum(p) - 1|", "sum gap"),
    )
    for axis, col32, col16, title, ylabel, gapcol in panels:
        axis.plot(table["N"], table[col32], marker="o", color="#1a9850",
                  label="fp32 accumulator", linewidth=1.8, markersize=6)
        axis.plot(table["N"], table[col16], marker="s", color="#b2182b",
                  label="fp16 accumulator", linewidth=1.8, markersize=6)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("sequence length N")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.3, which="both")
        axis.legend(frameon=False)
        gap = float(table[gapcol].iloc[-1])
        n_last = int(table["N"].iloc[-1])
        axis.annotate(f"{gap:,.0f}x apart\nat N={n_last}",
                      xy=(n_last, float(table[col16].iloc[-1])),
                      xytext=(0.42, 0.55), textcoords="axes fraction", fontsize=9.5,
                      arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2))

    figure.suptitle("Why every accumulator in this project is fp32 "
                    "(inputs are fp16 in both arms; only the accumulator differs)",
                    fontsize=13)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def main() -> None:
    for path in (
        latency_scaling(RESULTS / "latency-scaling.png"),
        memory(RESULTS / "memory.png"),
        oom_ladder(RESULTS / "oom-ladder.png"),
        fusion(RESULTS / "fusion.png"),
        throughput(RESULTS / "throughput.png"),
        causal(RESULTS / "causal-skipping.png"),
        accumulator(RESULTS / "accumulator.png"),
    ):
        print(f"-> {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
