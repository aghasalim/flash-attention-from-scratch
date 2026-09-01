/* Tiled attention in C, checked against the PyTorch reference outputs.
 *
 * fa/ref/fp64.py builds the whole N x N score matrix, takes the row max,
 * exponentiates, normalises, and multiplies by V. verify/export_golden.py runs
 * that and writes the results to verify/golden/attention_golden.bin. This file
 * computes the same outputs the other way: one key block at a time, carrying a
 * running max m, a running sum l and an unnormalised accumulator, rescaling by
 * exp(m_old - m_new) whenever a block pushes the max up, and dividing once at
 * the very end. The N x N matrix is never built.
 *
 * That makes this a check on the algorithm, not a translation of it. Two
 * different loop structures, two languages, two arithmetic orderings, and the
 * answers have to agree.
 *
 * It also checks the log-sum-exp column, m + log(l), which is the one float per
 * row the backward pass needs and which nothing else here recomputes
 * independently.
 *
 * Build and run:
 *   cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror -o attention verify/attention.c -lm
 *   ./attention .
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Absolute tolerance on O and on the log-sum-exp. Both sides are double
 * precision but sum in a different order, so the difference is rounding only.
 * The measured worst case over the committed golden file is printed at the end
 * and is orders of magnitude under this. */
#define TOL 1e-12

typedef struct {
    int32_t n_q, n_k, d, causal, block_m, block_n;
    double sm_scale;
    float *q, *k, *v;
    double *o, *lse;
} Case;

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "out of memory asking for %zu bytes\n", n);
        exit(2);
    }
    return p;
}

static void must_read(void *dst, size_t n, FILE *f, const char *what)
{
    if (fread(dst, 1, n, f) != n) {
        fprintf(stderr, "golden file is short while reading %s\n", what);
        exit(2);
    }
}

/* Tiled forward pass. Outer loop over query blocks, inner sequential loop over
 * key blocks, exactly the nest fa/ref/online_softmax.py describes. */
static void tiled_attention(const Case *c, double *o, double *lse)
{
    const int n_q = c->n_q, n_k = c->n_k, d = c->d;
    const int bm = c->block_m, bn = c->block_n;

    double *s = xmalloc((size_t)bm * bn * sizeof(double));
    double *acc = xmalloc((size_t)bm * d * sizeof(double));
    double *m_i = xmalloc((size_t)bm * sizeof(double));
    double *l_i = xmalloc((size_t)bm * sizeof(double));

    for (int q_start = 0; q_start < n_q; q_start += bm) {
        const int q_end = q_start + bm < n_q ? q_start + bm : n_q;
        const int rows = q_end - q_start;   /* short trailing block: a real slice */

        for (int i = 0; i < rows; i++) {
            m_i[i] = -INFINITY;
            l_i[i] = 0.0;
            for (int e = 0; e < d; e++)
                acc[(size_t)i * d + e] = 0.0;
        }

        for (int kv_start = 0; kv_start < n_k; kv_start += bn) {
            const int kv_end = kv_start + bn < n_k ? kv_start + bn : n_k;
            const int cols = kv_end - kv_start;

            /* Causal blocks entirely above the diagonal contribute nothing and
             * kv_start only grows, so every later block is above it too. This is
             * the block skipping the causal section of the README measures. */
            if (c->causal && kv_start > q_end - 1)
                break;

            for (int i = 0; i < rows; i++) {
                const float *qrow = c->q + (size_t)(q_start + i) * d;
                double m_blk = -INFINITY;
                for (int j = 0; j < cols; j++) {
                    const float *krow = c->k + (size_t)(kv_start + j) * d;
                    double dot = 0.0;
                    for (int e = 0; e < d; e++)
                        dot += (double)qrow[e] * (double)krow[e];
                    double val = dot * c->sm_scale;
                    /* Masked entries are -inf, never zero: -inf loses the max and
                     * exp(-inf) is 0 in the sum. Zero-filling corrupts the max. */
                    if (c->causal && kv_start + j > q_start + i)
                        val = -INFINITY;
                    s[(size_t)i * bn + j] = val;
                    if (val > m_blk)
                        m_blk = val;
                }

                const double m_new = m_blk > m_i[i] ? m_blk : m_i[i];
                const double corr = (m_i[i] == -INFINITY && m_new == -INFINITY)
                                        ? 0.0 : exp(m_i[i] - m_new);
                double sum = 0.0;
                for (int j = 0; j < cols; j++) {
                    const double p = exp(s[(size_t)i * bn + j] - m_new);
                    s[(size_t)i * bn + j] = p;
                    sum += p;
                }
                l_i[i] = l_i[i] * corr + sum;
                for (int e = 0; e < d; e++) {
                    double pv = 0.0;
                    for (int j = 0; j < cols; j++)
                        pv += s[(size_t)i * bn + j] * (double)c->v[(size_t)(kv_start + j) * d + e];
                    acc[(size_t)i * d + e] = acc[(size_t)i * d + e] * corr + pv;
                }
                m_i[i] = m_new;
            }
        }

        for (int i = 0; i < rows; i++) {
            for (int e = 0; e < d; e++)                 /* the single divide */
                o[(size_t)(q_start + i) * d + e] = acc[(size_t)i * d + e] / l_i[i];
            lse[q_start + i] = m_i[i] + log(l_i[i]);
        }
    }

    free(s); free(acc); free(m_i); free(l_i);
}

int main(int argc, char **argv)
{
    const char *root = argc > 1 ? argv[1] : ".";
    char path[1024];
    snprintf(path, sizeof path, "%s/verify/golden/attention_golden.bin", root);

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s. Run verify/export_golden.py first.\n", path);
        return 2;
    }
    char magic[8];
    must_read(magic, sizeof magic, f, "magic");
    if (memcmp(magic, "FAGOLD01", 8) != 0) {
        fprintf(stderr, "%s is not a golden file (bad magic)\n", path);
        fclose(f);
        return 2;
    }
    int32_t n_cases = 0;
    must_read(&n_cases, sizeof n_cases, f, "case count");
    if (n_cases <= 0 || n_cases > 1000) {
        fprintf(stderr, "implausible case count %d\n", (int)n_cases);
        fclose(f);
        return 2;
    }
    printf("%d cases from verify/golden/attention_golden.bin\n", (int)n_cases);

    double worst_o = 0.0, worst_lse = 0.0;
    int failures = 0;

    for (int ci = 0; ci < n_cases; ci++) {
        Case c;
        int32_t hdr[6];
        must_read(hdr, sizeof hdr, f, "case header");
        c.n_q = hdr[0]; c.n_k = hdr[1]; c.d = hdr[2];
        c.causal = hdr[3]; c.block_m = hdr[4]; c.block_n = hdr[5];
        must_read(&c.sm_scale, sizeof c.sm_scale, f, "sm_scale");
        if (c.n_q <= 0 || c.n_k <= 0 || c.d <= 0 || c.block_m <= 0 || c.block_n <= 0) {
            fprintf(stderr, "case %d has a non-positive dimension\n", ci);
            fclose(f);
            return 2;
        }
        /* The mask here is col <= row. fa/ref/fp64.py aligns a shorter query
         * block to the END of the key sequence, so the two only agree when the
         * lengths match. Refuse rather than compare two different masks. */
        if (c.causal && c.n_q != c.n_k) {
            fprintf(stderr, "case %d is causal with n_q %d != n_k %d\n",
                    ci, c.n_q, c.n_k);
            fclose(f);
            return 2;
        }

        c.q = xmalloc((size_t)c.n_q * c.d * sizeof(float));
        c.k = xmalloc((size_t)c.n_k * c.d * sizeof(float));
        c.v = xmalloc((size_t)c.n_k * c.d * sizeof(float));
        c.o = xmalloc((size_t)c.n_q * c.d * sizeof(double));
        c.lse = xmalloc((size_t)c.n_q * sizeof(double));
        must_read(c.q, (size_t)c.n_q * c.d * sizeof(float), f, "q");
        must_read(c.k, (size_t)c.n_k * c.d * sizeof(float), f, "k");
        must_read(c.v, (size_t)c.n_k * c.d * sizeof(float), f, "v");
        must_read(c.o, (size_t)c.n_q * c.d * sizeof(double), f, "o");
        must_read(c.lse, (size_t)c.n_q * sizeof(double), f, "lse");

        double *got_o = xmalloc((size_t)c.n_q * c.d * sizeof(double));
        double *got_lse = xmalloc((size_t)c.n_q * sizeof(double));
        tiled_attention(&c, got_o, got_lse);

        double eo = 0.0, el = 0.0;
        for (size_t i = 0; i < (size_t)c.n_q * c.d; i++) {
            const double dd = fabs(got_o[i] - c.o[i]);
            if (!(dd == dd)) { eo = INFINITY; break; }   /* NaN anywhere is a failure */
            if (dd > eo) eo = dd;
        }
        for (int i = 0; i < c.n_q; i++) {
            const double dd = fabs(got_lse[i] - c.lse[i]);
            if (!(dd == dd)) { el = INFINITY; break; }
            if (dd > el) el = dd;
        }
        if (eo > worst_o) worst_o = eo;
        if (el > worst_lse) worst_lse = el;

        const int bad = !(eo <= TOL) || !(el <= TOL);
        failures += bad;
        printf("  n_q %4d  n_k %4d  d %3d  causal %d  block %3dx%-3d  "
               "max |O - torch| %.2e   max |lse - torch| %.2e   %s\n",
               c.n_q, c.n_k, c.d, c.causal, c.block_m, c.block_n, eo, el,
               bad ? "FAIL" : "ok");

        free(c.q); free(c.k); free(c.v); free(c.o); free(c.lse);
        free(got_o); free(got_lse);
    }
    fclose(f);

    printf("\nworst over all cases: O %.2e, lse %.2e (tolerance %.0e)\n",
           worst_o, worst_lse, TOL);
    if (failures) {
        printf("%d of %d cases disagree with the PyTorch reference\n",
               failures, (int)n_cases);
        return 1;
    }
    printf("the C tiled kernel reproduces the PyTorch reference on every case\n");
    return 0;
}
