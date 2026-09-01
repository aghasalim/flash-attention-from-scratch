# Does naive attention really follow N^2 until it falls off the cliff?
#
# The README and notes/METHODS.md make a shape claim, not just a list of
# timings: latency tracks N^2 while the score matrix fits, and then stops. That
# claim was read off a plot. Nothing fitted it. This fits it, in base R, from
# results/roofline.csv, and re-derives the three numbers the prose quotes with it:
# the successive doubling factors, the 267x jump from N=2048 to N=4096, and what
# the N^2 trend actually predicts at N=4096.
#
# No packages, so CI needs nothing beyond the R already on the runner.

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."

rows <- read.csv(file.path(root, "results", "roofline.csv"),
                 stringsAsFactors = FALSE)
rows <- rows[rows$phase == "sweep" & rows$status == "ok" &
             rows$device == "mps" & rows$causal == "False" &
             !is.na(rows$latency_ms_median), ]

lat <- function(impl, n) {
    v <- rows$latency_ms_median[rows$impl == impl & rows$N == n]
    if (length(v) != 1) stop(sprintf("expected one %s row at N=%d, found %d",
                                     impl, n, length(v)))
    v
}

failures <- 0
report <- function(label, got, want, ok) {
    failures <<- failures + !ok
    cat(sprintf("  %-42s %-12s want %-12s %s\n", label, got, want,
                if (ok) "ok" else "FAIL"))
}

# The sizes where naive still fits. 4096 is the cliff and is deliberately out of
# the fit, since fitting the point you are testing against would prove nothing.
fit_n <- c(512, 1024, 2048)
naive <- sapply(fit_n, function(n) lat("naive", n))
chunked <- sapply(fit_n, function(n) lat("chunked", n))

cat("log-log fit of median latency against N, MPS, non-causal, N <= 2048\n")
slope <- function(y) unname(coef(lm(log(y) ~ log(fit_n)))[2])
s_naive <- slope(naive)
s_chunk <- slope(chunked)
# "naive attention follows N^2 up to 2048". A quadratic is a slope of 2 on a
# log-log fit; 1.9 to 2.1 is the band that claim can survive in.
report("naive exponent", sprintf("%.3f", s_naive), "2.0 +/- 0.1",
       abs(s_naive - 2) <= 0.1)
report("chunked exponent", sprintf("%.3f", s_chunk), "2.0 +/- 0.1",
       abs(s_chunk - 2) <= 0.1)

# "successive doublings costing 3.84x and 4.04x" -- notes/METHODS.md section 2.
d1 <- naive[2] / naive[1]
d2 <- naive[3] / naive[2]
report("naive 1024/512", sprintf("%.2f", d1), "3.84", sprintf("%.2f", d1) == "3.84")
report("naive 2048/1024", sprintf("%.2f", d2), "4.04", sprintf("%.2f", d2) == "4.04")

# The cliff. The measured jump is what the prose quotes; the fitted prediction is
# what the same trend would have said, and the gap between them is the finding.
measured_4096 <- lat("naive", 4096)
jump <- measured_4096 / naive[3]
report("naive 4096/2048, measured", sprintf("%.0f", jump), "267",
       sprintf("%.0f", jump) == "267")

model <- lm(log(naive) ~ log(fit_n))
predicted <- exp(unname(predict(model, data.frame(fit_n = 4096))))
cat(sprintf("\n  N=2048 measured  %8.1f ms\n", naive[3]))
cat(sprintf("  N=4096 predicted by the N^2 fit  %8.1f ms\n", predicted))
cat(sprintf("  N=4096 measured  %8.1f ms, %.0fx over the fitted trend\n",
            measured_4096, measured_4096 / predicted))
# The cliff has to be a cliff: an order of magnitude over the trend, not a bend.
report("measured over fitted trend at 4096",
       sprintf("%.0f", measured_4096 / predicted), ">= 10",
       measured_4096 / predicted >= 10)
# README section 2 quotes this prediction, so it is a published figure now.
report("fitted N^2 trend at 4096, ms", sprintf("%.0f", predicted), "679",
       sprintf("%.0f", predicted) == "679")

# Chunked stays on the trend through the size where naive leaves it. That is the
# other half of the claim: the cliff is naive's, not the sweep's.
chunk_4096 <- lat("chunked", 4096)
chunk_pred <- exp(unname(predict(lm(log(chunked) ~ log(fit_n)),
                                 data.frame(fit_n = 4096))))
report("chunked 4096, measured over fitted", sprintf("%.2f", chunk_4096 / chunk_pred),
       "within 2x", abs(log2(chunk_4096 / chunk_pred)) <= 1)
report("chunked over naive at 4096", sprintf("%.2f", measured_4096 / chunk_4096),
       "37.95", sprintf("%.2f", measured_4096 / chunk_4096) == "37.95")

if (failures > 0) {
    cat(sprintf("\n%d checks failed\n", failures))
    quit(status = 1)
}
cat("\nthe N^2 shape, the doubling factors and the cliff all reproduce in R\n")
