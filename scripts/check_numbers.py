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
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# notes/METHODS.md holds the detail moved out of the README. A figure quoted
# there is still a quoted figure and still has to match its source.
DOCS = ["README.md", "notes/METHODS.md", "notes/paper.md", "notes/00-roofline.md"]


def load():
    hw = json.loads((ROOT / "hardware.json").read_text())
    rows = list(csv.DictReader((ROOT / "results" / "roofline.csv").open()))
    fus_path = ROOT / "results" / "fusion.csv"
    fus = list(csv.DictReader(fus_path.open())) if fus_path.exists() else []
    return hw, rows, fus


def fusion(fus, impl, n, field="latency_ms_median"):
    for r in fus:
        if r["impl"] == impl and r["N"] == str(n) and r["status"] == "ok":
            return float(r[field])
    return None


def sweep(rows, device, impl, n, causal="False", field="latency_ms_median"):
    for r in rows:
        if (r["phase"] == "sweep" and r["device"] == device and r["impl"] == impl
                and r["N"] == str(n) and r["causal"] == causal and r["status"] == "ok"):
            return float(r[field])
    return None


def claims(hw, rows, fus):
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
        ("ridge band low", ridge_lo, f"{ridge_lo:.2f}", ["README.md", "notes/METHODS.md", "notes/paper.md"]),
        ("ridge band high", ridge_hi, f"{ridge_hi:.2f}", ["README.md", "notes/METHODS.md", "notes/paper.md"]),
        ("AI naive @4096", ai_naive, f"{ai_naive:.2f}", DOCS),
        ("AI chunked @4096", ai_chunk, f"{ai_chunk:.2f}", DOCS),
        ("AI fused @4096", ai_fused, f"{ai_fused:.2f}", DOCS),
        ("naive/chunked @4096", naive4 / chunk4, f"{naive4 / chunk4:.2f}", DOCS),
    ]

    # bench/fusion.py -- the CPU fusion measurement. README quotes the table and
    # the speedups; those are the numbers most likely to be misread as a
    # FlashAttention result, so they get checked hardest.
    for n in (256, 512, 1024, 2048, 4096):
        e = fusion(fus, "naive-eager", n)
        f_ = fusion(fus, "naive-compiled", n)
        if e is None or f_ is None:
            continue
        out += [
            (f"fusion eager N={n}", e, f"{e:.2f}", ["README.md", "notes/METHODS.md"]),
            (f"fusion compiled N={n}", f_, f"{f_:.2f}", ["README.md", "notes/METHODS.md"]),
        ]
        # Read the speedup off the CSV rather than recomputing it as a ratio of
        # medians. bench/fusion.py takes the ratio within each repeat and then the
        # median of those, which is the honest statistic: it reflects how unstable
        # the ratio itself is. The two differ (2.95 vs 3.01 at N=1024) and the
        # per-repeat form is the one the README quotes.
        sp = next((r for r in fus if r["impl"] == "fusion-speedup"
                   and r["N"] == str(n) and r["status"] == "ok"), None)
        if sp:
            for key, lbl in (("speedup_median", "speedup"),
                             ("speedup_min", "speedup low"),
                             ("speedup_max", "speedup high")):
                val = float(sp[key])
                out.append((f"fusion {lbl} N={n}", val, f"{val:.2f}", ["README.md", "notes/METHODS.md"]))
    return [c for c in out if c[1] is not None]


# Values that were true once, are quoted nowhere any more, and must never come back
# without the data changing to match. Each is a real stale figure this check caught.
STALE = {
    "32.92": "old ridge point, superseded when hardware.json was regenerated",
    "3142.6": "old MPS fp16 matmul median",
    "1652.2": "old CPU fp32 matmul median",
}


def main() -> int:
    hw, rows, fus = load()
    failures = []
    checked = 0

    for label, _value, text, files in claims(hw, rows, fus):
        # Accept any rendering that reads back to the same number -- prose writes
        # "2048 FLOP/byte" where a table writes "2048.00" -- but ONLY forms that
        # keep at least 3 significant characters, and only on a number boundary.
        #
        # An earlier version also generated the %.0f form, which for 9.99 is "10",
        # and "10" occurs inside "1024" and "10 GPU cores". That made the check
        # pass against deliberately corrupted data. Caught by negative-testing the
        # checker itself, which is the only reason to write a negative test.
        forms = {text, text.rstrip("0").rstrip(".")}
        forms = {x for x in forms if len(x.replace(".", "").replace("-", "")) >= 3}
        if not forms:
            forms = {text}
        # The figure must appear in at least ONE of the listed documents, not in
        # every one of them. Detail moved out of the README into notes/METHODS.md,
        # so requiring every file to carry every figure would fail on numbers that
        # simply live in the other document now.
        present = []
        for f in files:
            path = ROOT / f
            if not path.exists():
                continue
            body = path.read_text()
            # (?<![\d.]) / (?![\d]) so 3.86 does not match inside 13.865
            if any(re.search(r"(?<![\d.])" + re.escape(x) + r"(?!\d)", body)
                   for x in sorted(forms)):
                present.append(f)
        checked += 1
        if not present:
            failures.append(
                f"{label} should read {text} (or an equivalent form), "
                f"not found in any of {', '.join(files)}")

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
    # What this does and does not cover, so the green line is not read as more
    # than it is: each figure is recomputed from results/ and looked for in the
    # prose. It cannot catch a wrong number that happens to appear somewhere,
    # it does not check claims written in words (ratios, multiples, ranges),
    # and it does not read notes/LOGBOOK.md.
    print("this checks quoted figures against results/, not claims written in words")
    return 0


if __name__ == "__main__":
    sys.exit(main())
