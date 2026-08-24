"""Measure kernel fusion on the CPU, since Triton cannot run on this machine.

Why this exists: the roofline in `bench/roofline.py` places a "fused ideal" point at
2048 FLOP/byte, and that point is *analytic* -- it assumes S and P never reach
memory and computes the byte count that would follow. It is a model, not a
measurement, and the CSV labels it ASSUMED.

Triton has no macOS wheel, so the planned Triton kernel cannot supply the real
number. But `torch.compile` with the inductor `cpp` backend does perform genuine
kernel fusion on the CPU: it generates a C++ kernel that keeps intermediates in
registers/cache instead of materialising them. That is the same *mechanism* the
FlashAttention argument rests on, at a different scale and without tiling or an
online softmax.

So this measures the one thing the roofline could only assume: does removing the
score-matrix round trip actually buy time on this hardware? It is not a
FlashAttention result and must not be quoted as one. It is an empirical floor under
a claim that was otherwise purely analytic.

    .venv/bin/python -m bench.fusion     # writes results/fusion.csv

CPU and fp32 throughout: arm64 torch has no fast half GEMM (3.3 GFLOP/s vs 1738 for
fp32, measured by scripts/env.py), and inductor's CPU path is the only fusion
backend available here.
"""
from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

import torch

from fa.ref.naive import chunked_attention, naive_attention

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
NOT_MEASURED = "not measured on this hardware (no CUDA device; developed on Apple M4)"

B, H, D = 2, 8, 64
N_LIST = (256, 512, 1024, 2048, 4096)
DTYPE = torch.float32
WARMUP, ITERS = 2, 5
# Independent repeats, interleaved across variants. A single round gave speedups of
# 2.19x and 1.63x for N=512 on two consecutive runs -- a 34% swing -- so a point
# estimate here would repeat the mistake documented in notes/LOGBOOK.md, where a
# single-median ridge point flipped a conclusion. Report the range.
REPEATS = 5


def timeit(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3, min(ts) * 1e3, max(ts) * 1e3


def analytic(n: int, elem: int = 4):
    """FLOPs, and HBM bytes under two models: materialised vs fused."""
    flops = 4 * B * H * n * n * D
    qkvo = 4 * B * H * n * D * elem
    scores = 4 * B * H * n * n * elem          # S and P, each written and read
    return flops, qkvo + scores, qkvo


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    torch.manual_seed(0)
    rows = []

    print(f"CPU fusion sweep: B={B} H={H} D={D} dtype={DTYPE}")
    print(f"inductor cpu backend = {torch._inductor.config.cpu_backend}\n")

    compiled = torch.compile(naive_attention, backend="inductor")

    for n in N_LIST:
        q, k, v = (torch.randn(B, H, n, D, dtype=DTYPE) for _ in range(3))
        ref = naive_attention(q, k, v)
        flops, bytes_mat, bytes_fused = analytic(n)

        # Bind q/k/v as defaults rather than closing over the loop variables.
        # These are only called inside this iteration so late binding would not
        # actually bite, but a closure over a rebound tensor is the kind of thing
        # that becomes a silent wrong-shape benchmark the moment the loop changes.
        variants = [
            ("naive-eager", lambda q=q, k=k, v=v: naive_attention(q, k, v), bytes_mat),
            ("chunked-eager", lambda q=q, k=k, v=v: chunked_attention(q, k, v), bytes_mat),
            ("naive-compiled", lambda q=q, k=k, v=v: compiled(q, k, v), bytes_fused),
        ]

        # Interleave variants within each repeat. Running all of one and then all of
        # the other would fold thermal drift straight into the comparison.
        per_repeat: dict[str, list[float]] = {name: [] for name, _, _ in variants}
        errs: dict[str, float] = {}
        for _ in range(REPEATS):
            for name, fn, _ in variants:
                out = fn()
                errs[name] = max(errs.get(name, 0.0),
                                 (out.double() - ref.double()).abs().max().item())
                med, _lo, _hi = timeit(fn)
                per_repeat[name].append(med)

        for name, _fn, model_bytes in variants:
            samples = sorted(per_repeat[name])
            med, lo, hi = statistics.median(samples), samples[0], samples[-1]
            gflops = flops / (med * 1e-3) / 1e9
            ai = flops / model_bytes
            rows.append(dict(
                N=n, impl=name, status="ok", dtype="float32", device="cpu",
                latency_ms_median=med, latency_ms_min=lo, latency_ms_max=hi,
                max_abs_err_vs_eager_naive=errs[name],
                flops_analytic=flops, hbm_bytes_model=model_bytes,
                arithmetic_intensity=ai, achieved_gflop_s=gflops,
                byte_model=("materialised: Q,K,V,O + S,P written and read"
                            if model_bytes == bytes_mat else
                            "fused: Q,K,V,O only -- S,P assumed never to reach memory"),
                note="", warmup=WARMUP, iters=ITERS, repeats=REPEATS,
            ))
            print(f"  N={n:5} {name:16} {med:9.2f} ms [{lo:8.2f}-{hi:8.2f}]  "
                  f"AI={ai:8.2f}  {gflops:7.1f} GFLOP/s  max|err|={errs[name]:.2e}")

        # Speedup per repeat, then the spread across repeats -- not the ratio of
        # medians, which hides how unstable the ratio itself is.
        ups = sorted(e / c for e, c in zip(per_repeat["naive-eager"],
                                           per_repeat["naive-compiled"]))
        rows.append(dict(N=n, impl="fusion-speedup", status="ok", device="cpu",
                         dtype="float32", latency_ms_median="", repeats=REPEATS,
                         speedup_median=statistics.median(ups),
                         speedup_min=ups[0], speedup_max=ups[-1],
                         note="naive-eager / naive-compiled, computed per repeat"))
        print(f"  {'':5} {'-> fusion speedup':16} {statistics.median(ups):9.2f}x "
              f"[{ups[0]:.2f}-{ups[-1]:.2f}]\n")

    fields = sorted({k for r in rows for k in r})
    out = RESULTS / "fusion.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")
    print(f"\nGPU fusion (Triton/CUDA): {NOT_MEASURED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
