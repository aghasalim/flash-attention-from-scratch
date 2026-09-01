//! Is the tiled attention answer independent of the tile shape? Exhaustively.
//!
//! The whole argument for the online softmax rescaling is that the result does
//! not depend on how the key sequence is cut up: whenever a block pushes the
//! running max up, everything accumulated so far is multiplied by
//! exp(m_old - m_new) and the answer comes out the same. fa/ref/online_softmax.py
//! tests that on six block shapes, and its block-order experiment uses twenty
//! random permutations, because it is NumPy and every extra shape costs a second.
//!
//! This runs the kernel over every block shape in a wide grid, for every case in
//! verify/golden/attention_golden.bin, and requires all of them to agree with the
//! PyTorch reference. Hundreds of tilings per case instead of six. A rescale that
//! were subtly wrong -- correct for tiles that divide the sequence, wrong for the
//! short trailing one, or wrong only when a later block raises the max -- would
//! survive six shapes and cannot survive this.
//!
//! No crates: a small binary reader and the kernel, nothing else.
//!
//! Run: cargo run --release --quiet -- <repo root>

use std::env;
use std::fs;
use std::process::exit;

/// Absolute tolerance against the float64 PyTorch reference. Both sides are
/// double precision and differ only in summation order.
const TOL: f64 = 1e-12;

/// Block edges to try. Powers of two, primes, values just above and below a
/// power of two, and awkward numbers that divide nothing. Any candidate larger
/// than the sequence is dropped, and the sequence length itself is always
/// included so the untiled case is covered too.
const CANDIDATES: [usize; 19] = [
    1, 2, 3, 5, 7, 8, 11, 13, 16, 17, 31, 32, 33, 48, 64, 100, 127, 128, 129,
];

struct Case {
    n_q: usize,
    n_k: usize,
    d: usize,
    causal: bool,
    sm_scale: f64,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    o: Vec<f64>,
    lse: Vec<f64>,
}

struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        if self.at + n > self.buf.len() {
            eprintln!("golden file is short: wanted {} more bytes", n);
            exit(2);
        }
        let s = &self.buf[self.at..self.at + n];
        self.at += n;
        s
    }
    fn i32(&mut self) -> i32 {
        i32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn f64(&mut self) -> f64 {
        f64::from_le_bytes(self.take(8).try_into().unwrap())
    }
    fn f32s(&mut self, n: usize) -> Vec<f32> {
        self.take(4 * n)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
    fn f64s(&mut self, n: usize) -> Vec<f64> {
        self.take(8 * n)
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}

/// Tiled forward pass: outer loop over query blocks, inner sequential loop over
/// key blocks, a running max and a running sum carried across the inner loop, and
/// one divide at the very end. The N x N score matrix is never built.
fn tiled(c: &Case, bm: usize, bn: usize) -> (Vec<f64>, Vec<f64>) {
    let (n_q, n_k, d) = (c.n_q, c.n_k, c.d);
    let mut o = vec![0.0f64; n_q * d];
    let mut lse = vec![0.0f64; n_q];
    let mut s = vec![0.0f64; bn];

    let mut q_start = 0;
    while q_start < n_q {
        let q_end = (q_start + bm).min(n_q); // short trailing block: a real slice
        for i in q_start..q_end {
            let mut m_i = f64::NEG_INFINITY;
            let mut l_i = 0.0f64;
            let mut acc = vec![0.0f64; d];

            let mut kv_start = 0;
            while kv_start < n_k {
                // Above the diagonal the whole block contributes nothing, and
                // kv_start only grows, so every later block is above it too.
                if c.causal && kv_start > q_end - 1 {
                    break;
                }
                let kv_end = (kv_start + bn).min(n_k);
                let mut m_blk = f64::NEG_INFINITY;
                for j in kv_start..kv_end {
                    let mut dot = 0.0f64;
                    for e in 0..d {
                        dot += c.q[i * d + e] as f64 * c.k[j * d + e] as f64;
                    }
                    // A masked score is -inf, never zero: -inf loses the max and
                    // exp(-inf) is zero in the sum. Zero corrupts the max.
                    let val = if c.causal && j > i {
                        f64::NEG_INFINITY
                    } else {
                        dot * c.sm_scale
                    };
                    s[j - kv_start] = val;
                    if val > m_blk {
                        m_blk = val;
                    }
                }

                let m_new = if m_blk > m_i { m_blk } else { m_i };
                // Both -inf means nothing has been seen yet and nothing is in
                // this block either; exp(-inf - -inf) is NaN, so say zero.
                let corr = if m_i == f64::NEG_INFINITY && m_new == f64::NEG_INFINITY {
                    0.0
                } else {
                    (m_i - m_new).exp()
                };
                let mut sum = 0.0f64;
                for j in 0..(kv_end - kv_start) {
                    let p = (s[j] - m_new).exp();
                    s[j] = p;
                    sum += p;
                }
                l_i = l_i * corr + sum;
                for e in 0..d {
                    let mut pv = 0.0f64;
                    for j in 0..(kv_end - kv_start) {
                        pv += s[j] * c.v[(kv_start + j) * d + e] as f64;
                    }
                    acc[e] = acc[e] * corr + pv;
                }
                m_i = m_new;
                kv_start += bn;
            }

            for e in 0..d {
                o[i * d + e] = acc[e] / l_i; // the single divide
            }
            lse[i] = m_i + l_i.ln();
        }
        q_start += bm;
    }
    (o, lse)
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    let mut worst = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        if d.is_nan() {
            return f64::INFINITY;
        }
        if d > worst {
            worst = d;
        }
    }
    worst
}

fn edges(n: usize) -> Vec<usize> {
    let mut v: Vec<usize> = CANDIDATES.iter().copied().filter(|&c| c < n).collect();
    v.push(n);
    v
}

fn main() {
    let root = env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let path = format!("{}/verify/golden/attention_golden.bin", root);
    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("cannot read {}: {}", path, e);
            exit(2);
        }
    };
    let mut r = Reader { buf: &bytes, at: 0 };
    if r.take(8) != b"FAGOLD01" {
        eprintln!("{} is not a golden file (bad magic)", path);
        exit(2);
    }
    let n_cases = r.i32();
    if n_cases <= 0 || n_cases > 1000 {
        eprintln!("implausible case count {}", n_cases);
        exit(2);
    }

    let mut total = 0usize;
    let mut failures = 0usize;
    let mut worst_o = 0.0f64;
    let mut worst_lse = 0.0f64;
    let mut worst_spread = 0.0f64;

    println!("{} cases, every block shape in the grid", n_cases);
    for _ in 0..n_cases {
        let n_q = r.i32() as usize;
        let n_k = r.i32() as usize;
        let d = r.i32() as usize;
        let causal = r.i32() != 0;
        let _bm = r.i32();
        let _bn = r.i32();
        let sm_scale = r.f64();
        let q = r.f32s(n_q * d);
        let k = r.f32s(n_k * d);
        let v = r.f32s(n_k * d);
        let o = r.f64s(n_q * d);
        let lse = r.f64s(n_q);
        if causal && n_q != n_k {
            eprintln!("causal case with n_q {} != n_k {} is not defined here", n_q, n_k);
            exit(2);
        }
        let c = Case { n_q, n_k, d, causal, sm_scale, q, k, v, o, lse };

        // The untiled run is the baseline the other shapes are spread against.
        let (base_o, _) = tiled(&c, n_q, n_k);
        let (mut case_o, mut case_lse, mut case_spread) = (0.0f64, 0.0f64, 0.0f64);
        let mut shapes = 0usize;

        for &bm in &edges(n_q) {
            for &bn in &edges(n_k) {
                let (got_o, got_lse) = tiled(&c, bm, bn);
                let eo = max_abs_diff(&got_o, &c.o);
                let el = max_abs_diff(&got_lse, &c.lse);
                let sp = max_abs_diff(&got_o, &base_o);
                if !(eo <= TOL) || !(el <= TOL) {
                    failures += 1;
                    if failures <= 5 {
                        println!(
                            "  FAIL n_q {} n_k {} d {} causal {} block {}x{}: \
                             |O - torch| {:.2e}  |lse - torch| {:.2e}",
                            n_q, n_k, d, causal, bm, bn, eo, el
                        );
                    }
                }
                case_o = case_o.max(eo);
                case_lse = case_lse.max(el);
                case_spread = case_spread.max(sp);
                shapes += 1;
            }
        }
        total += shapes;
        worst_o = worst_o.max(case_o);
        worst_lse = worst_lse.max(case_lse);
        worst_spread = worst_spread.max(case_spread);
        println!(
            "  n_q {:>4}  n_k {:>4}  d {:>3}  causal {}  {:>4} block shapes  \
             max |O - torch| {:.2e}  max |lse - torch| {:.2e}  spread over shapes {:.2e}",
            n_q, n_k, d, causal as u8, shapes, case_o, case_lse, case_spread
        );
    }

    println!(
        "\n{} tilings checked, worst |O - torch| {:.2e}, worst |lse - torch| {:.2e}, \
         worst spread across block shapes {:.2e} (tolerance {:.0e})",
        total, worst_o, worst_lse, worst_spread, TOL
    );
    if failures > 0 {
        println!("{} tilings disagree with the PyTorch reference", failures);
        exit(1);
    }
    println!("the rescaling is exact for every block shape tried");
}
