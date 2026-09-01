// Two things nothing else in the repository recomputes.
//
// 1. The traffic table in README section 1. It is arithmetic on tensor shapes,
//    not a measurement, so no CSV backs it and scripts/check_numbers.py cannot
//    reach it. It is also the table the whole memory-wall argument opens with.
//    This counts the bytes again from B, H, N, D and the element size and reads
//    the published table straight out of README.md.
//
// 2. The share of the machine's measured fp32 peak that the fusion benchmark
//    achieves. results/fusion.csv holds GFLOP/s, hardware.json holds the peak,
//    and the percentages in the prose were worked out by hand from the two. The
//    achieved column is recomputed from the analytic FLOP count and the latency
//    first, so a wrong GFLOP/s could not quietly produce a right percentage.
//
// Run: node verify/traffic.js <repo root>

const fs = require("fs");
const path = require("path");

const root = process.argv[2] || ".";
const GIB = 1024 ** 3;

let failures = 0;
function report(label, got, want, ok) {
  if (!ok) failures++;
  console.log(
    "  " + label.padEnd(42) + String(got).padEnd(14) +
    "want " + String(want).padEnd(16) + (ok ? "ok" : "FAIL"));
}

// ---------------------------------------------------------------- traffic ---
// fp16, B=4 H=32 D=64, stated under the table in README section 1.
const B = 4, H = 32, D = 64, ELEM = 2;

// The formula the tables say they are instances of, stated in
// notes/00-roofline.md section 2 and used by bench/roofline.py for the
// hbm_bytes_analytic column of results/roofline.csv:
//
//   bytes(Q,K,V,O) = 4 * B * H * N * D * e
//   bytes(S,P)     = 4 * B * H * N^2 * e     each of S and P written once, read once
//
// Four accesses in each case, which is why the ratio is N/D and not N/2D.
const params = (N) => 4 * B * H * N * D * ELEM;
const scores = (N) => 4 * B * H * N * N * ELEM;

const DOCS = ["README.md", "notes/00-roofline.md", "notes/paper.md"];
// | 1024 | 0.062 GiB | 1.000 GiB | 16x |   (the multiplication sign varies)
const rowRe = /^\|\s*(\d+)\s*\|\s*([\d.]+) GiB\s*\|\s*([\d.]+) GiB\s*\|\s*(\d+)/;
const published = [];
for (const doc of DOCS) {
  const text = fs.readFileSync(path.join(root, doc), "utf8");
  for (const raw of text.split("\n")) {
    // Strip markdown emphasis first: one row of the notes table is bolded, and
    // bolding a number must not read as the number having gone missing.
    const m = rowRe.exec(raw.replace(/\*/g, "").trim());
    if (m) published.push({
      doc: doc, n: +m[1], params: m[2], scores: m[3], ratio: +m[4],
    });
  }
}
console.log("traffic table, recounted from B=" + B + " H=" + H + " D=" + D +
            " fp16, in " + DOCS.length + " documents");
report("table rows found", published.length, 12, published.length === 12);

// A printed figure agrees if it is a correctly rounded form of the computed one.
// Half of the last printed decimal is the furthest rounding can move it, and the
// slack is there because that bound is itself computed in binary floating point.
function agrees(text, value) {
  const decimals = (text.split(".")[1] || "").length;
  return Math.abs(parseFloat(text) - value) <= 0.5 * Math.pow(10, -decimals) * (1 + 1e-9);
}

for (const row of published) {
  const p = params(row.n) / GIB, s = scores(row.n) / GIB;
  const where = row.doc.replace("notes/", "") + " N=" + row.n;
  report(where + " Q,K,V,O GiB", row.params, p.toFixed(3), agrees(row.params, p));
  report(where + " S,P GiB", row.scores, s.toFixed(3), agrees(row.scores, s));
  // The ratio is N/D exactly, which is the sentence above every one of these
  // tables: 4*B*H*N^2*e over 4*B*H*N*D*e.
  report(where + " ratio", row.ratio, row.n / D, row.ratio === row.n / D);
}

// ------------------------------------------------------------- throughput ---
function readCsv(file) {
  const text = fs.readFileSync(file, "utf8").trim();
  const rows = [];
  // The benchmark CSVs quote any field containing a comma, so a plain split
  // would tear the note columns apart and shift everything after them.
  for (const line of text.split("\n")) {
    const cells = [];
    let cur = "", quoted = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (quoted) {
        if (ch === '"' && line[i + 1] === '"') { cur += '"'; i++; }
        else if (ch === '"') quoted = false;
        else cur += ch;
      } else if (ch === '"') quoted = true;
      else if (ch === ",") { cells.push(cur); cur = ""; }
      else cur += ch;
    }
    cells.push(cur);
    rows.push(cells);
  }
  const header = rows[0];
  return rows.slice(1).map((r) => Object.fromEntries(header.map((h, i) => [h, r[i]])));
}

const hw = JSON.parse(fs.readFileSync(path.join(root, "hardware.json"), "utf8"));
const peak = hw.dtypes.fp32.cpu.matmul.gflop_s;
const fusion = readCsv(path.join(root, "results", "fusion.csv"))
  .filter((r) => r.status === "ok" && r.latency_ms_median !== "");

console.log("\nachieved throughput against the measured CPU fp32 peak of " +
            peak.toFixed(1) + " GFLOP/s");

const shares = { "naive-eager": [], "naive-compiled": [] };
let worstRel = 0;
for (const r of fusion) {
  if (!(r.impl in shares)) continue;
  const flops = parseFloat(r.flops_analytic);
  const lat = parseFloat(r.latency_ms_median);
  const recomputed = flops / (lat * 1e6);          // GFLOP/s
  const publishedG = parseFloat(r.achieved_gflop_s);
  worstRel = Math.max(worstRel, Math.abs(recomputed - publishedG) / publishedG);
  shares[r.impl].push({ n: +r.N, share: 100 * recomputed / peak });
}
report("achieved GFLOP/s = flops/latency", worstRel.toExponential(1), "<= 1e-12",
       worstRel <= 1e-12);

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
for (const impl of Object.keys(shares)) shares[impl].sort((a, b) => a.n - b.n);
const eager = shares["naive-eager"].map((x) => x.share);
const fused = shares["naive-compiled"].map((x) => x.share);
for (const [impl, xs] of [["eager", eager], ["fused", fused]]) {
  console.log("  " + impl.padEnd(8) + xs.map((x) => x.toFixed(1) + "%").join("  "));
}

// notes/METHODS.md: "pinned near 20 to 26% of peak" and "Fusing lifts it to
// between 45% and 68%".
report("eager share, low", Math.round(Math.min(...eager)), 20,
       Math.round(Math.min(...eager)) === 20);
report("eager share, high", Math.round(Math.max(...eager)), 26,
       Math.round(Math.max(...eager)) === 26);
report("fused share, low", Math.round(Math.min(...fused)), 45,
       Math.round(Math.min(...fused)) === 45);
report("fused share, high", Math.round(Math.max(...fused)), 68,
       Math.round(Math.max(...fused)) === 68);

// README section 2: "from 22% of the CPU's measured fp32 peak to roughly 67%".
// 22 is the median eager share across the five sizes. 67 is the mean of the two
// largest sizes, which is what "roughly 67%" has to mean: no single size gives
// 67 and the median fused share is 59.
const eagerMedian = Math.round(median(eager));
const big = shares["naive-compiled"].slice(-2).map((x) => x.share);
const fusedBig = Math.round((big[0] + big[1]) / 2);
report("eager share, median over sizes", eagerMedian, 22, eagerMedian === 22);
report("fused share, mean of two largest N", fusedBig, 67, fusedBig === 67);

if (failures > 0) {
  console.log("\n" + failures + " checks failed");
  process.exit(1);
}
console.log("\nthe traffic table and the throughput shares both recompute");
