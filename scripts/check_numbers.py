"""Fail if a number quoted in the prose no longer matches the data it came from.

Why this exists: `hardware.json` was regenerated after README.md was written, and
nothing noticed. The MPS fp16 matmul figure moved 3142.6 -> 2963.5 GFLOP/s, which
moved the ridge point 32.92 -> 30.91, which flipped the roofline verdict for naive
attention from memory-bound to compute-bound. The README went on asserting the old
conclusion for hours. See notes/LOGBOOK.md, 2026-08-24.

Prose drifts because the *data* is regenerated, not because the prose is edited, so
this runs in CI on every push rather than only when a doc changes.

What it checks: every derived figure the write-ups quote, recomputed from
hardware.json and results/roofline.csv. Not prose, not claims that need judgement --
just the numbers, against their source.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "notes/paper.md", "notes/00-roofline.md"]


def load():
    hw = json.loads((ROOT / "hardware.json").read_text())
    rows = list(csv.DictReader((ROOT / "results" / "roofline.csv").open()))
    return hw, rows


def sweep(rows, device, impl, n, causal="False", field="latency_ms_median"):
    for r in rows:
        if (r["phase"] == "sweep" and r["device"] == device and r["impl"] == impl
                and r["N"] == str(n) and r["causal"] == causal and r["status"] == "ok"):
            return float(r[field])
    return None


def claims(hw, rows):
    """(label, value, format, files-that-must-agree). Recomputed from source."""
    bw = hw["bandwidth"]
    dt = hw["dtypes"]
    mps_fp16 = dt["fp16"]["mps"]["matmul"]["gflop_s"]
    mps_bw = bw["mps"]["gb_per_s"]
    ridge = mps_fp16 / mps_bw
    ridge_lo = dt["fp16"]["mps"]["matmul"]["gflop_s_worst"] / bw["mps"]["gb_per_s_best"]
    ridge_hi = dt["fp16"]["mps"]["matmul"]["gflop_s_best"] / bw["mps"]["gb_per_s_worst"]
    ai_naive = sweep(rows, "mps", "naive", 4096, field="arithmetic_intensity_flop_per_byte")
    ai_chunk = sweep(rows, "mps", "chunked", 4096, field="arithmetic_intensity_flop_per_byte")
    ai_fused = sweep(rows, "mps", "sdpa", 4096, field="arithmetic_intensity_flop_per_byte")
    naive4 = sweep(rows, "mps", "naive", 4096)
    chunk4 = sweep(rows, "mps", "chunked", 4096)

    out = [
        ("MPS fp16 matmul GFLOP/s", mps_fp16, f"{mps_fp16:.1f}", DOCS),
        ("MPS bandwidth GB/s", mps_bw, f"{mps_bw:.2f}", ["notes/paper.md"]),
        ("ridge point (median)", ridge, f"{ridge:.2f}", DOCS),
        ("ridge band low", ridge_lo, f"{ridge_lo:.2f}", ["README.md", "notes/paper.md"]),
        ("ridge band high", ridge_hi, f"{ridge_hi:.2f}", ["README.md", "notes/paper.md"]),
        ("AI naive @4096", ai_naive, f"{ai_naive:.2f}", DOCS),
        ("AI chunked @4096", ai_chunk, f"{ai_chunk:.2f}", DOCS),
        ("AI fused @4096", ai_fused, f"{ai_fused:.2f}", DOCS),
        ("naive/chunked @4096", naive4 / chunk4, f"{naive4 / chunk4:.2f}", DOCS),
    ]
    return [c for c in out if c[1] is not None]


# Values that were true once, are quoted nowhere any more, and must never come back
# without the data changing to match. Each is a real stale figure this check caught.
STALE = {
    "32.92": "old ridge point, superseded when hardware.json was regenerated",
    "3142.6": "old MPS fp16 matmul median",
    "1652.2": "old CPU fp32 matmul median",
}


def main() -> int:
    hw, rows = load()
    failures = []
    checked = 0

    for label, _value, text, files in claims(hw, rows):
        for f in files:
            body = (ROOT / f).read_text()
            checked += 1
            if text not in body:
                failures.append(f"{f}: {label} should read {text}, not found")

    # A stale value is only a failure if it is presented as current. The logbook and
    # the 'what I got wrong' sections quote old numbers on purpose, as history.
    # Paragraph-level, not line-level: prose wraps, so "medians of 3142.6 and 2963.5 an
    # / hour apart" splits a historical marker across two lines and a line-based check
    # flags it. Blocks are separated by blank lines; a markdown table is one block.
    for f in ["README.md", "notes/paper.md", "notes/00-roofline.md"]:
        for block in (ROOT / f).read_text().split("\n\n"):
            low = block.lower()
            if any(w in low for w in ("superseded", "first version", "hour apart",
                                      "never asked", "read the next table",
                                      "went 3142.6", "→", "->")):
                continue
            for bad, why in STALE.items():
                if bad in block:
                    snip = next(ln for ln in block.splitlines() if bad in ln)
                    failures.append(f"{f}: stale value {bad} presented as current ({why})\n      {snip.strip()[:100]}")

    print(f"checked {checked} quoted figures against hardware.json + results/roofline.csv")
    if failures:
        print("\nDRIFT DETECTED:")
        for x in failures:
            print(f"  - {x}")
        print("\nThe prose and its source data disagree. Regenerate the prose, or say why\n"
              "the figure is legitimately historical.")
        return 1
    print("no drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
