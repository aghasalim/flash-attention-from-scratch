"""Online (streaming) softmax and tiled attention — NumPy reference.

Paper math only: no torch, no Triton, no GPU. This is the version the kernel is
ported *from*, so the structure matters as much as the numbers:

  * `online_attention` has the exact loop nest the Triton kernel will have —
    outer loop over Q blocks (the grid axis), inner sequential loop over KV
    blocks, running `m_i` / `l_i` / `acc` carried across the inner loop.
  * Accumulators are fp32 at minimum (repo rule 5). `acc_dtype=np.float16` is
    exposed only so the experiments can measure how badly that rule bites.
  * Short trailing blocks are handled by *slicing*, never by zero-padding.
    A masked score must be -inf (contributes -inf to the max, exp(-inf)=0 to
    the sum). Zero-filling silently corrupts the max — see `_experiments`.

Run the self-check and the four numerical experiments:

    .venv/bin/python -m fa.ref.online_softmax
"""

from __future__ import annotations

import numpy as np

__all__ = ["online_softmax", "online_attention", "logsumexp_rows", "causal_zone"]


def _acc_dtype(x, acc_dtype):
    """Rule 5: accumulators are fp32 at minimum; fp64 inputs keep fp64."""
    if acc_dtype is not None:
        return np.dtype(acc_dtype)
    return np.promote_types(np.asarray(x).dtype, np.float32)


def online_softmax(x, block_size, acc_dtype=None):
    """softmax over the last axis in one streaming pass of `block_size` chunks.

    Keeps a running max `m` and running sum `l`. When a block pushes the max up,
    everything already accumulated is rescaled by exp(m_old - m_new), which is
    <= 1 by construction — the correction only ever shrinks, so it cannot blow up.

    The output of softmax is N numbers, so this function does hold N of them;
    what streaming buys you is in `online_attention`, where the accumulator is
    (BLOCK_M, d) instead of the (BLOCK_M, N) probability matrix.
    """
    x = np.asarray(x)
    one_d = x.ndim == 1
    xm = x[None, :] if one_d else x
    n = xm.shape[-1]
    acc = _acc_dtype(xm, acc_dtype)
    flat = xm.reshape(-1, n)

    m_i = np.full(flat.shape[0], -np.inf, dtype=acc)  # running max
    l_i = np.zeros(flat.shape[0], dtype=acc)          # running sum of exp
    p = np.empty(flat.shape, dtype=acc)               # unnormalised, rescaled in place

    for s in range(0, n, block_size):
        # short trailing block: a genuinely shorter slice, not a padded one
        blk = flat[:, s : s + block_size].astype(acc)
        m_blk = blk.max(axis=1)
        m_new = np.maximum(m_i, m_blk)
        corr = np.exp(m_i - m_new)[:, None]           # <= 1 always
        p[:, :s] *= corr                              # the O(N) rescale; O(d) in the kernel
        p[:, s : s + block_size] = np.exp(blk - m_new[:, None])
        l_i = l_i * corr[:, 0] + p[:, s : s + block_size].sum(axis=1)
        m_i = m_new

    out = (p / l_i[:, None]).reshape(xm.shape)
    return out[0] if one_d else out


def causal_zone(q_start, q_end, kv_start, kv_end):
    """Which of the three causal zones a (Q block, KV block) pair falls in.

    Rows are [q_start, q_end), cols are [kv_start, kv_end), mask is col <= row.
    Task 06 splits the kernel along exactly this classification, so it lives in
    one named place rather than inline in the loop.

      'skip'     — every col > every row: contributes nothing, do no work at all
      'dense'    — every col <= every row: no mask, full tile
      'diagonal' — straddles the diagonal: element-level mask, the only zone
                   that pays for masking
    """
    if kv_start > q_end - 1:
        return "skip"
    if kv_end - 1 <= q_start:
        return "dense"
    return "diagonal"


def _attention_core(q, k, v, block_m, block_n, causal, sm_scale, acc_dtype):
    """Tiled attention forward, shaped like the Triton kernel it becomes.

    OUTER loop over Q blocks  -> the Triton GRID AXIS (`pid = tl.program_id(0)`).
        Every iteration is an independent program: it reads Q once, streams all
        of K/V, and writes its own rows of O and L. No cross-program communication,
        no global sync — that is the whole point of the online recurrence.
    INNER loop over KV blocks -> stays a real sequential loop inside the kernel.
        `m_i`, `l_i`, `acc` live in registers across it and are fp32 (rule 5).

    Returns (O, L) with L = m + log(l), one fp32 per row.
    """
    q, k, v = np.asarray(q), np.asarray(k), np.asarray(v)
    n_q, d = q.shape
    n_k, d_v = k.shape[0], v.shape[1]
    if causal and n_q != n_k:
        raise ValueError("causal reference assumes n_q == n_k")
    acc_t = _acc_dtype(q, acc_dtype)
    if sm_scale is None:
        sm_scale = 1.0 / np.sqrt(d)
    sm_scale = acc_t.type(sm_scale)

    o = np.empty((n_q, d_v), dtype=acc_t)
    lse = np.empty(n_q, dtype=acc_t)

    for q_start in range(0, n_q, block_m):                 # <-- Triton grid axis 0
        q_end = min(q_start + block_m, n_q)                # short block: slice, never pad
        bm = q_end - q_start
        q_blk = q[q_start:q_end].astype(acc_t)
        rows = np.arange(q_start, q_end)

        m_i = np.full(bm, -np.inf, dtype=acc_t)            # fp32 accumulators
        l_i = np.zeros(bm, dtype=acc_t)
        acc = np.zeros((bm, d_v), dtype=acc_t)

        for kv_start in range(0, n_k, block_n):            # <-- sequential loop in-kernel
            kv_end = min(kv_start + block_n, n_k)
            zone = causal_zone(q_start, q_end, kv_start, kv_end) if causal else "dense"
            if zone == "skip":
                break   # kv_start only grows, so every later block is above the diagonal too

            s = (q_blk @ k[kv_start:kv_end].astype(acc_t).T) * sm_scale
            if zone == "diagonal":
                # masked -> -inf: contributes -inf to the max and exp(-inf)=0 to the
                # sum. Filling with 0 instead would make the max wrong, which is the
                # classic bug (see the zero-fill experiment).
                cols = np.arange(kv_start, kv_end)
                s = np.where(cols[None, :] <= rows[:, None], s, -np.inf)

            m_blk = s.max(axis=1)
            m_new = np.maximum(m_i, m_blk)
            corr = np.exp(m_i - m_new)                     # <= 1 always
            p = np.exp(s - m_new[:, None])
            l_i = l_i * corr + p.sum(axis=1)
            acc = acc * corr[:, None] + p @ v[kv_start:kv_end].astype(acc_t)
            m_i = m_new

        o[q_start:q_end] = acc / l_i[:, None]              # the single divide, at the end
        lse[q_start:q_end] = m_i + np.log(l_i)
    return o, lse


def online_attention(q, k, v, block_m, block_n, causal=False, sm_scale=None, acc_dtype=None):
    """softmax(QK^T * sm_scale) V, tiled, never materialising the N x N matrix."""
    return _attention_core(q, k, v, block_m, block_n, causal, sm_scale, acc_dtype)[0]


def logsumexp_rows(q, k, causal=False, block_m=128, block_n=128, sm_scale=None, acc_dtype=None):
    """L = m + log(l) per query row — the one float the backward pass needs.

    With L saved, P = exp(S - L) row-wise, so the backward pass recomputes the
    scores it needs from Q and K instead of reading an N x N probability matrix
    back from HBM: N floats instead of N^2.

    V does not enter L at all, so this reuses the same loop nest with a width-1
    zero V rather than keeping a second copy of it.
    """
    v = np.zeros((np.asarray(k).shape[0], 1), dtype=np.asarray(q).dtype)
    return _attention_core(q, k, v, block_m, block_n, causal, sm_scale, acc_dtype)[1]


# --------------------------------------------------------------------------- #
# self-check + numerical experiments
# --------------------------------------------------------------------------- #

def _direct_attention(q, k, v, causal, sm_scale):
    """Textbook softmax(QK^T)V in fp64 — the thing the tiled version must match."""
    from scipy.special import softmax

    s = (q @ k.T) * sm_scale
    if causal:
        cols, rows = np.arange(k.shape[0])[None, :], np.arange(q.shape[0])[:, None]
        s = np.where(cols <= rows, s, -np.inf)
    return softmax(s, axis=-1) @ v


def _self_check():
    from scipy.special import logsumexp, softmax

    rng = np.random.default_rng(0)

    # 1. online softmax vs scipy, fp64, dividing and non-dividing block sizes
    x = rng.standard_normal((64, 1000)) * 3.0
    for bs in (1, 7, 64, 128, 333, 1000, 4096):
        err = np.abs(online_softmax(x, bs) - softmax(x, axis=-1)).max()
        assert err < 1e-14, (bs, err)
    print(f"[ok] online_softmax vs scipy, fp64, block sizes 1..4096: max err {err:.3e}")

    # 2. tiled attention vs direct softmax(QK^T)V, causal and not, uneven blocks
    worst = 0.0
    for n, d in ((133, 32), (256, 64), (512, 16)):
        q, k, v = (rng.standard_normal((n, d)) for _ in range(3))
        scale = 1.0 / np.sqrt(d)
        for bm, bn in ((16, 16), (32, 64), (64, 16), (100, 100), (128, 128), (n, n)):
            for causal in (False, True):
                got = online_attention(q, k, v, bm, bn, causal=causal, sm_scale=scale)
                ref = _direct_attention(q, k, v, causal, scale)
                err = np.abs(got - ref).max()
                assert err < 1e-12, (n, d, bm, bn, causal, err)
                worst = max(worst, err)
    print(f"[ok] online_attention vs direct softmax(QK^T)V, fp64, "
          f"N in (133,256,512), 6 block shapes incl. non-dividing, causal+non-causal: "
          f"max err {worst:.3e}")

    # 3. logsumexp_rows vs scipy.special.logsumexp
    n, d = 200, 32
    q, k = rng.standard_normal((n, d)), rng.standard_normal((n, d))
    scale = 1.0 / np.sqrt(d)
    worst = 0.0
    for causal in (False, True):
        s = (q @ k.T) * scale
        if causal:
            s = np.where(np.arange(n)[None, :] <= np.arange(n)[:, None], s, -np.inf)
        got = logsumexp_rows(q, k, causal=causal, block_m=64, block_n=48, sm_scale=scale)
        err = np.abs(got - logsumexp(s, axis=-1)).max()
        assert err < 1e-12, (causal, err)
        worst = max(worst, err)
    print(f"[ok] logsumexp_rows vs scipy.special.logsumexp, causal+non-causal: max err {worst:.3e}")


def _exp_a(rng):
    from scipy.special import softmax

    x = rng.standard_normal((1000, 1024)) * 4.0
    rows = []
    for bs in (64, 128, 100, 333):
        err = np.abs(online_softmax(x, bs) - softmax(x, axis=-1)).max()
        rows.append((bs, err))
        print(f"  block_size {bs:>4}{'  (does not divide 1024)' if 1024 % bs else '':<26} "
              f"max |online - scipy| = {err:.3e}")
    return rows


def _exp_b(rng):
    """fp32 vs fp16 accumulators. Three metrics, because they fail differently:

      abs  — max |p_hat - p|. Shrinks with N on its own because max(p) ~ 1/N,
             so it is the least informative of the three.
      rel  — max |p_hat - p| / p. Catches small probabilities flushed to zero.
      sum  — |sum(p_hat) - 1|. The denominator l is the accumulator under test,
             and this is exactly its relative error: sum(p_hat) = l_true / l_hat
             up to the error in the numerators.
    """
    from scipy.special import softmax

    hdr = ("N", "fp32 abs", "fp16 abs", "fp32 rel", "fp16 rel", "fp32 |sum-1|",
           "fp16 |sum-1|", "abs gap", "sum gap")
    print("  " + " | ".join(f"{h:>12}" for h in hdr))
    print("  " + "-+-".join("-" * 12 for _ in hdr))
    out = []
    for n in (128, 512, 1024, 2048, 4096, 8192):
        x16 = (rng.standard_normal((64, n)) * 2.0).astype(np.float16)
        ref = softmax(x16.astype(np.float64), axis=-1)      # same input, fp64 arithmetic
        res = {}
        for dt in (np.float32, np.float16):
            got = online_softmax(x16, 128, acc_dtype=dt).astype(np.float64)
            res[dt] = (np.abs(got - ref).max(),
                       (np.abs(got - ref) / ref).max(),
                       np.abs(got.sum(axis=1) - 1.0).max())
        a32, r32, s32 = res[np.float32]
        a16, r16, s16 = res[np.float16]
        out.append((n, a32, a16, r32, r16, s32, s16, a16 / a32, s16 / s32))
        print(f"  {n:>12} | {a32:>12.3e} | {a16:>12.3e} | {r32:>12.3e} | {r16:>12.3e} | "
              f"{s32:>12.3e} | {s16:>12.3e} | {a16/a32:>11.0f}x | {s16/s32:>11.0f}x")
    return out


def _exp_c(rng):
    from scipy.special import softmax

    n = 1024
    x = np.full(n, -100.0)
    x[17] = 100.0
    ref = softmax(x)                                    # fp64: exp(100) is finite here
    print(f"  row: one score +100, {n - 1} scores -100")
    for dt in (np.float32, np.float16):
        xd = x.astype(dt)
        with np.errstate(over="ignore", invalid="ignore"):
            e = np.exp(xd)
            naive = e / e.sum()
        got = online_softmax(xd, 128, acc_dtype=dt).astype(np.float64)
        bad = (~np.isfinite(naive)).sum()
        print(f"  {np.dtype(dt).name}: naive exp(x).sum() = {e.sum()}, "
              f"naive peak prob = {naive[17]}, non-finite entries = {bad}/{n}")
        print(f"  {np.dtype(dt).name}: online peak prob = {got[17]!r}, "
              f"max |online - fp64 exact| = {np.abs(got - ref).max():.3e}")
    print(f"  fp64 naive does not overflow: exp(100) = {np.exp(100.0):.3e} "
          f"(fp32 max 3.403e+38, fp16 max 6.550e+04, so fp16 overflows above x = "
          f"{np.log(65504.0):.4f})")


def _exp_d(rng):
    n, d, bm, bn = 512, 64, 128, 64
    q, k, v = (rng.standard_normal((n, d)) for _ in range(3))
    scale = 1.0 / np.sqrt(d)
    base = online_attention(q, k, v, bm, bn, sm_scale=scale)
    nb = n // bn
    worst = 0.0
    for _ in range(20):
        perm = rng.permutation(nb)
        idx = np.concatenate([np.arange(p * bn, (p + 1) * bn) for p in perm])
        # permuting KV *rows* by whole blocks == visiting the KV blocks in a
        # different order, since block j of the permuted K/V is block perm[j].
        got = online_attention(q, k[idx], v[idx], bm, bn, sm_scale=scale)
        worst = max(worst, np.abs(got - base).max())
    assert worst < 1e-12, worst
    print(f"  20 random permutations of the {nb} KV blocks, N={n} d={d} "
          f"BLOCK_M={bm} BLOCK_N={bn}, fp64: max |permuted - in-order| = {worst:.3e}")
    return worst


def _exp_zero_fill(rng):
    """The uneven-block trap: pad the short block with 0 instead of -inf."""
    from scipy.special import softmax

    n, bs = 1000, 128                      # 1000 = 7*128 + 104, last block is short
    x = rng.standard_normal(n) * 2.0 - 6.0  # scores below 0, so a 0 pad becomes the max
    ref = softmax(x)
    good = online_softmax(x, bs)
    pad = np.concatenate([x, np.zeros(bs - n % bs)])     # the wrong fix
    bad = online_softmax(pad, bs)[:n]
    print(f"  N={n}, block_size={bs} (last block {n % bs} wide), scores in "
          f"[{x.min():.2f}, {x.max():.2f}]")
    print(f"  short slice (correct):        max |online - scipy| = {np.abs(good - ref).max():.3e}")
    print(f"  zero-padded last block (bug): max |online - scipy| = {np.abs(bad - ref).max():.3e}, "
          f"probabilities sum to {bad.sum():.6f} instead of 1")
    return np.abs(bad - ref).max()


def _experiments():
    rng = np.random.default_rng(1234)
    print("\n(a) online vs scipy.special.softmax, fp64, 1000 rows x N=1024")
    _exp_a(rng)
    print("\n(b) fp32 vs fp16 accumulators, input fp16, reference = fp64 softmax of the "
          "same fp16 input\n    64 rows per N, block_size=128")
    _exp_b(rng)
    print("\n(c) adversarial row: naive softmax overflows, the max-shift does not")
    _exp_c(rng)
    print("\n(d) block-order invariance, non-causal")
    _exp_d(rng)
    print("\n(e) uneven blocks: zero-fill vs -inf")
    _exp_zero_fill(rng)


if __name__ == "__main__":
    _self_check()
    _experiments()
