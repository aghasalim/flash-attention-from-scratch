"""Write the reference attention outputs to a binary file the C and Rust checks read.

The golden outputs come from fa/ref/fp64.py, which materialises the whole N x N
score matrix and normalises it in one pass. The implementations that read this
file are tiled: they never build S, they carry a running max and a running sum
across key blocks, and they divide once at the end. So the comparison is between
two different algorithms, not between two spellings of the same one, and a
mistake in the tiling would have to be a mistake in the dense reference too to
go unnoticed.

The log-sum-exp column is here because the backward pass needs it and nothing
else in the repository checks it against an independent computation.

Format, little-endian throughout:

    char[8]  "FAGOLD01"
    int32    n_cases
    per case:
      int32   n_q, n_k, d, causal, block_m, block_n
      float64 sm_scale
      float32 q[n_q*d], k[n_k*d], v[n_k*d]
      float64 o[n_q*d], lse[n_q]

Inputs are float32 so both readers get the exact bits torch used; outputs are
float64 because that is the precision the reference computed them in.

Two modes:

    python verify/export_golden.py           rewrite the golden file
    python verify/export_golden.py --check    re-derive the stored outputs

The check mode is the one CI runs. It reads the committed file, recomputes O and
the log-sum-exp from the q, k and v that are stored *in that file*, and requires
the stored outputs to match. That is what makes the golden file trustworthy
without re-running the export: it does not depend on any random seed, so it holds
across torch versions and platforms, and a golden file that had been edited to
agree with a broken kernel would fail it.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import torch

# The two sides are the same fp64 computation, so this is rounding only.
TOL = 1e-12

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fa.ref.fp64 import attention_fp64, causal_mask  # noqa: E402

OUT = ROOT / "verify" / "golden" / "attention_golden.bin"

# (n_q, n_k, d, causal, block_m, block_n). Block shapes that do not divide the
# sequence length are deliberate: the short trailing block is where a tiled
# implementation is most likely to be wrong, and padding it with zeros instead
# of -inf is the classic version of that bug.
CASES = [
    (64, 64, 16, False, 16, 16),
    (133, 133, 32, False, 32, 64),
    (133, 133, 32, True, 32, 64),
    (256, 256, 32, False, 128, 128),
    (256, 256, 32, True, 64, 32),
    (100, 100, 8, True, 100, 100),
    (48, 137, 24, False, 16, 48),
]


def lse_rows(q, k, causal, sm_scale):
    """m + log(l) per query row, from the dense scores, in float64."""
    s = (q.double() @ k.double().transpose(-2, -1)) * sm_scale
    if causal:
        s = s.masked_fill(causal_mask(q.shape[-2], k.shape[-2], s.device), float("-inf"))
    return torch.logsumexp(s, dim=-1)


def read_cases(path: Path):
    """Parse the golden file back into a list of cases, strictly."""
    blob = path.read_bytes()
    if blob[:8] != b"FAGOLD01":
        raise SystemExit(f"{path} is not a golden file (bad magic)")
    (n_cases,) = struct.unpack_from("<i", blob, 8)
    at = 12
    cases = []
    for _ in range(n_cases):
        n_q, n_k, d, causal, bm, bn = struct.unpack_from("<6i", blob, at)
        at += 24
        (sm_scale,) = struct.unpack_from("<d", blob, at)
        at += 8

        def take(count, dtype, width, _at=lambda: at):
            nonlocal at
            arr = np.frombuffer(blob, dtype=dtype, count=count, offset=at)
            at += count * width
            return arr

        q = take(n_q * d, "<f4", 4).reshape(n_q, d)
        k = take(n_k * d, "<f4", 4).reshape(n_k, d)
        v = take(n_k * d, "<f4", 4).reshape(n_k, d)
        o = take(n_q * d, "<f8", 8).reshape(n_q, d)
        lse = take(n_q, "<f8", 8)
        cases.append((n_q, n_k, d, bool(causal), bm, bn, sm_scale, q, k, v, o, lse))
    if at != len(blob):
        raise SystemExit(f"{path}: {len(blob) - at} trailing bytes after {n_cases} cases")
    return cases


def check() -> int:
    """Recompute every stored output from the stored inputs."""
    if not OUT.exists():
        print(f"{OUT} is missing. Run this script without --check first.")
        return 2
    cases = read_cases(OUT)
    worst_o = worst_lse = 0.0
    failures = 0
    print(f"re-deriving {len(cases)} cases from the inputs stored in "
          f"{OUT.relative_to(ROOT)}")
    for n_q, n_k, d, causal, bm, bn, sm_scale, q, k, v, o, lse in cases:
        qt = torch.from_numpy(q.copy())[None, None]
        kt = torch.from_numpy(k.copy())[None, None]
        vt = torch.from_numpy(v.copy())[None, None]
        got_o = attention_fp64(qt, kt, vt, causal=causal, sm_scale=sm_scale)[0, 0].numpy()
        got_lse = lse_rows(qt[0, 0], kt[0, 0], causal, sm_scale).numpy()
        eo = float(abs(got_o - o).max())
        el = float(abs(got_lse - lse).max())
        worst_o, worst_lse = max(worst_o, eo), max(worst_lse, el)
        bad = not (eo <= TOL and el <= TOL)
        failures += bad
        print(f"  n_q={n_q:>4} n_k={n_k:>4} d={d:>3} causal={str(causal):<5} "
              f"block {bm}x{bn}  |O| {eo:.2e}  |lse| {el:.2e}  "
              f"{'FAIL' if bad else 'ok'}")
    print(f"\nworst |O| {worst_o:.2e}, worst |lse| {worst_lse:.2e} "
          f"(tolerance {TOL:.0e})")
    if failures:
        print(f"{failures} cases do not match their own inputs")
        return 1
    print("every stored output is what the fp64 reference computes for its inputs")
    return 0


def main() -> int:
    torch.manual_seed(20260901)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    blob = bytearray()
    blob += b"FAGOLD01"
    blob += struct.pack("<i", len(CASES))

    for n_q, n_k, d, causal, bm, bn in CASES:
        q = torch.randn(1, 1, n_q, d, dtype=torch.float32)
        k = torch.randn(1, 1, n_k, d, dtype=torch.float32)
        v = torch.randn(1, 1, n_k, d, dtype=torch.float32)
        sm_scale = 1.0 / d ** 0.5

        o = attention_fp64(q, k, v, causal=causal, sm_scale=sm_scale)[0, 0]
        lse = lse_rows(q[0, 0], k[0, 0], causal, sm_scale)

        blob += struct.pack("<6i", n_q, n_k, d, int(causal), bm, bn)
        blob += struct.pack("<d", sm_scale)
        for t in (q, k, v):
            blob += t.flatten().numpy().astype("<f4").tobytes()
        blob += o.flatten().numpy().astype("<f8").tobytes()
        blob += lse.flatten().numpy().astype("<f8").tobytes()

        print(f"  n_q={n_q:>4} n_k={n_k:>4} d={d:>3} causal={str(causal):<5} "
              f"block {bm}x{bn}  |o| max {o.abs().max():.6f}  lse[0] {lse[0]:+.9f}")

    OUT.write_bytes(bytes(blob))
    print(f"wrote {OUT.relative_to(ROOT)}, {len(CASES)} cases, {len(blob)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(check() if "--check" in sys.argv[1:] else main())
