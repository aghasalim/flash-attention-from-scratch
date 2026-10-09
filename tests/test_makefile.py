"""`make bench` must write the CSVs before it draws them."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("make") is None, reason="no make")
def test_bench_runs_figures_last():
    out = subprocess.run(["make", "-n", "bench", "PY=py"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    order = [m for m in ("bench.roofline", "bench.fusion", "bench.figures") if m in out]
    assert order == ["bench.roofline", "bench.fusion", "bench.figures"]
    assert out.index("bench.figures") > max(out.index("bench.roofline"), out.index("bench.fusion"))
