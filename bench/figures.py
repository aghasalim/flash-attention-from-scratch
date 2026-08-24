"""Draw the additional README figures from results/roofline.csv.

``bench.roofline`` already writes ``results/roofline.png``. These cover the parts
of that sweep the roofline plot cannot show: how latency scales, where the naive
implementation runs out of memory, and what the OOM ladder found.

Reads the saved CSV only -- nothing is re-measured, and nothing here needs CUDA.

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


def main() -> None:
    for path in (
        latency_scaling(RESULTS / "latency-scaling.png"),
        memory(RESULTS / "memory.png"),
        oom_ladder(RESULTS / "oom-ladder.png"),
    ):
        print(f"-> {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
