-- Recompute the derived columns and the headline ratios of results/roofline.csv
-- and results/fusion.csv, in SQL, from the raw columns of the same files.
--
-- bench/roofline.py writes arithmetic_intensity_flop_per_byte, achieved_gflop_s
-- and implied_gb_s at the same time as the latency they are derived from, so
-- nothing downstream ever checked that the derivation was right: the figures and
-- the prose both read the derived column. This recomputes each of them from
-- flops_fwd_analytic, hbm_bytes_analytic and latency_ms_median, and recomputes
-- the ratios the README quotes by hand, with nothing but SQL.
--
-- Every row is printed as  check|got|want|status  and verify/verify.sh fails if
-- any status is not ok.
--
-- Run: sqlite3 -init verify/roofline.sql :memory: "" < /dev/null

.mode list
.separator |
.headers off
.import --csv results/roofline.csv sweep
.import --csv results/fusion.csv fusion

-- Only timed rows carry a latency. The OOM ladder rows deliberately have none.
CREATE TEMP VIEW ok_rows AS
    SELECT device, impl, CAST(N AS INT) AS n, causal,
           CAST(latency_ms_median AS REAL)            AS lat_ms,
           CAST(flops_fwd_analytic AS REAL)           AS flops,
           CAST(hbm_bytes_analytic AS REAL)           AS bytes,
           CAST(arithmetic_intensity_flop_per_byte AS REAL) AS ai_published,
           CAST(achieved_gflop_s AS REAL)             AS gflops_published,
           CAST(implied_gb_s AS REAL)                 AS gbs_published
    FROM sweep
    WHERE phase = 'sweep' AND status = 'ok' AND latency_ms_median <> '';

-- A derived column is wrong if it differs from its own inputs by more than
-- double precision rounding. 1e-12 relative is far above that and far below any
-- real mistake.
CREATE TEMP VIEW derived AS
    SELECT
        SUM(abs(ai_published - flops / bytes) > 1e-12 * abs(flops / bytes))       AS bad_ai,
        SUM(abs(gflops_published - flops / (lat_ms * 1e6))
            > 1e-12 * abs(flops / (lat_ms * 1e6)))                               AS bad_gflops,
        SUM(abs(gbs_published - bytes / (lat_ms * 1e6))
            > 1e-12 * abs(bytes / (lat_ms * 1e6)))                               AS bad_gbs,
        COUNT(*)                                                                 AS n_rows
    FROM ok_rows;

SELECT 'roofline rows recomputed', n_rows, n_rows,
       CASE WHEN n_rows >= 30 THEN 'ok' ELSE 'FAIL' END FROM derived;
SELECT 'arithmetic intensity = flops/bytes', bad_ai, 0,
       CASE WHEN bad_ai = 0 THEN 'ok' ELSE 'FAIL' END FROM derived;
SELECT 'achieved GFLOP/s = flops/latency', bad_gflops, 0,
       CASE WHEN bad_gflops = 0 THEN 'ok' ELSE 'FAIL' END FROM derived;
SELECT 'implied GB/s = bytes/latency', bad_gbs, 0,
       CASE WHEN bad_gbs = 0 THEN 'ok' ELSE 'FAIL' END FROM derived;

-- Everything below re-derives a number the README states in words.
CREATE TEMP VIEW mps AS
    SELECT impl, n, causal, lat_ms, ai_published FROM ok_rows
    WHERE device = 'mps';

-- "its arithmetic intensity is 29.47 against naive's 31.51"
SELECT 'AI naive @4096',
       printf('%.2f', (SELECT ai_published FROM mps WHERE impl='naive' AND n=4096 AND causal='False')),
       '31.51',
       CASE WHEN printf('%.2f', (SELECT ai_published FROM mps WHERE impl='naive' AND n=4096 AND causal='False')) = '31.51'
            THEN 'ok' ELSE 'FAIL' END;
SELECT 'AI chunked @4096',
       printf('%.2f', (SELECT ai_published FROM mps WHERE impl='chunked' AND n=4096 AND causal='False')),
       '29.47',
       CASE WHEN printf('%.2f', (SELECT ai_published FROM mps WHERE impl='chunked' AND n=4096 AND causal='False')) = '29.47'
            THEN 'ok' ELSE 'FAIL' END;

-- "Chunked attention runs at 0.56 to 0.59x naive on the GPU". Naive only reaches
-- 4096, where it has already fallen off the cliff, so the comparable sizes are
-- the ones where both are on the N^2 trend.
CREATE TEMP VIEW chunk_ratio AS
    SELECT a.n AS n, a.lat_ms / b.lat_ms AS ratio
    FROM mps a JOIN mps b ON a.n = b.n AND a.causal = b.causal
    WHERE a.impl = 'naive' AND b.impl = 'chunked' AND a.causal = 'False' AND a.n <= 2048;
SELECT 'chunked/naive low',  printf('%.2f', MIN(ratio)), '0.56',
       CASE WHEN printf('%.2f', MIN(ratio)) = '0.56' THEN 'ok' ELSE 'FAIL' END FROM chunk_ratio;
SELECT 'chunked/naive high', printf('%.2f', MAX(ratio)), '0.59',
       CASE WHEN printf('%.2f', MAX(ratio)) = '0.59' THEN 'ok' ELSE 'FAIL' END FROM chunk_ratio;

-- "chunked attention is 37.95x faster at that size"
SELECT 'chunked over naive @4096',
       printf('%.2f', (SELECT lat_ms FROM mps WHERE impl='naive'   AND n=4096 AND causal='False')
                    / (SELECT lat_ms FROM mps WHERE impl='chunked' AND n=4096 AND causal='False')),
       '37.95',
       CASE WHEN printf('%.2f', (SELECT lat_ms FROM mps WHERE impl='naive'   AND n=4096 AND causal='False')
                              / (SELECT lat_ms FROM mps WHERE impl='chunked' AND n=4096 AND causal='False')) = '37.95'
            THEN 'ok' ELSE 'FAIL' END;

-- "a 267x jump for 4x the work", naive from N=2048 to N=4096
SELECT 'naive 4096 over 2048',
       printf('%.0f', (SELECT lat_ms FROM mps WHERE impl='naive' AND n=4096 AND causal='False')
                    / (SELECT lat_ms FROM mps WHERE impl='naive' AND n=2048 AND causal='False')),
       '267',
       CASE WHEN printf('%.0f', (SELECT lat_ms FROM mps WHERE impl='naive' AND n=4096 AND causal='False')
                              / (SELECT lat_ms FROM mps WHERE impl='naive' AND n=2048 AND causal='False')) = '267'
            THEN 'ok' ELSE 'FAIL' END;

-- "the SDPA path, which classifies and skips blocks, reaches 2.02x" causal, and
-- "the implementations that mask a dense NxN" run at 0.91 to 0.98x. Both are the
-- MPS sweep; the causal figure in the README is the GPU one.
CREATE TEMP VIEW causal_gain AS
    SELECT a.impl AS impl, a.n AS n, a.lat_ms / b.lat_ms AS gain
    FROM mps a JOIN mps b ON a.impl = b.impl AND a.n = b.n
    WHERE a.causal = 'False' AND b.causal = 'True';
SELECT 'causal gain, sdpa, best', printf('%.2f', MAX(gain)), '2.02',
       CASE WHEN printf('%.2f', MAX(gain)) = '2.02' THEN 'ok' ELSE 'FAIL' END
    FROM causal_gain WHERE impl = 'sdpa';
-- naive at N=4096 is the one row past the OOM cliff, where the non-causal run
-- takes 46.5 s and the causal one still fits, so its ratio is 10.65 and measures
-- the cliff rather than the cost of masking. Every other masking row is here.
CREATE TEMP VIEW masking AS
    SELECT gain FROM causal_gain
    WHERE impl IN ('naive', 'chunked') AND NOT (impl = 'naive' AND n = 4096);
SELECT 'causal gain, masking impls, low', printf('%.2f', MIN(gain)), '0.91',
       CASE WHEN printf('%.2f', MIN(gain)) = '0.91' THEN 'ok' ELSE 'FAIL' END FROM masking;
SELECT 'causal gain, masking impls, high', printf('%.2f', MAX(gain)), '0.98',
       CASE WHEN printf('%.2f', MAX(gain)) = '0.98' THEN 'ok' ELSE 'FAIL' END FROM masking;

-- results/fusion.csv. The speedup column is the median of the per-repeat ratios,
-- not the ratio of the medians, so the two are allowed to differ. They are still
-- the same measurement seen two ways and cannot drift far apart; 0.10 absolute is
-- wide enough for the sampling difference and narrow enough to catch a swapped
-- or truncated column.
CREATE TEMP VIEW fus AS
    SELECT CAST(N AS INT) AS n, impl,
           CAST(latency_ms_median AS REAL) AS lat_ms,
           CAST(speedup_median AS REAL)    AS speedup
    FROM fusion WHERE status = 'ok';
CREATE TEMP VIEW fusion_check AS
    SELECT e.n AS n,
           e.lat_ms / c.lat_ms AS ratio_of_medians,
           s.speedup           AS published
    FROM fus e JOIN fus c ON e.n = c.n JOIN fus s ON s.n = e.n
    WHERE e.impl = 'naive-eager' AND c.impl = 'naive-compiled' AND s.impl = 'fusion-speedup';
SELECT 'fusion speedup, ratio of medians vs published',
       printf('%.3f', MAX(abs(ratio_of_medians - published))), '<= 0.100',
       CASE WHEN MAX(abs(ratio_of_medians - published)) <= 0.10 THEN 'ok' ELSE 'FAIL' END
    FROM fusion_check;
SELECT 'fusion sizes compared', COUNT(*), 5,
       CASE WHEN COUNT(*) = 5 THEN 'ok' ELSE 'FAIL' END FROM fusion_check;
