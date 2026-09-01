// Structural validation of every CSV under results/, plus an independent
// recomputation of the accumulator table.
//
// The CSVs in results/ are the evidence for every figure in the README and every
// plot in bench/figures.py. Nothing checked that they are well formed. A write
// interrupted halfway, a column that drifted when a benchmark grew a field, or a
// NaN that leaked out of a division would all be invisible until someone read the
// table and believed it.
//
// The recompute is the accumulator experiment. fa/ref/online_softmax.py writes
// "abs gap" and "sum gap" into results/accumulator.csv at the same moment it
// writes the errors they are ratios of, so the ratio was never checked against
// anything. The README quotes one of them, the factor of 3297 at N = 8192, as a
// headline number. This divides the columns again, in Go.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const relTol = 1e-12

// The sequence lengths the accumulator experiment sweeps. The README quotes the
// last of them by name, so a silently shortened file has to be an error.
var wantN = []int{128, 512, 1024, 2048, 4096, 8192}

func readCSV(path string) ([]string, [][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	r.FieldsPerRecord = 0 // a ragged file is an error, which is the point
	rows, err := r.ReadAll()
	if err != nil {
		return nil, nil, err
	}
	if len(rows) < 2 {
		return nil, nil, fmt.Errorf("only %d rows", len(rows))
	}
	return rows[0], rows[1:], nil
}

func col(header []string, name string) int {
	for i, h := range header {
		if h == name {
			return i
		}
	}
	return -1
}

// validate reports every structural problem in a file rather than the first, so
// one run diagnoses a broken benchmark completely.
func validate(path string) []string {
	var problems []string
	header, rows, err := readCSV(path)
	if err != nil {
		return []string{fmt.Sprintf("unreadable: %v", err)}
	}

	seen := map[string]bool{}
	for _, h := range header {
		if strings.TrimSpace(h) == "" {
			problems = append(problems, "a column has an empty name")
		}
		if seen[h] {
			problems = append(problems, fmt.Sprintf("duplicate column %q", h))
		}
		seen[h] = true
	}

	for i, row := range rows {
		for j, cell := range row {
			s := strings.TrimSpace(cell)
			if s == "" {
				continue
			}
			// Only fields that are meant to be numbers are judged as numbers.
			// results/roofline.csv carries prose in `note` and `hbm_model`, and
			// prose is not a failed float.
			low := strings.ToLower(s)
			if low == "nan" || low == "inf" || low == "-inf" || low == "infinity" {
				problems = append(problems, fmt.Sprintf(
					"row %d column %q is %s", i+2, header[j], s))
				continue
			}
			if v, err := strconv.ParseFloat(s, 64); err == nil {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					problems = append(problems, fmt.Sprintf(
						"row %d column %q is not finite: %s", i+2, header[j], s))
				}
			}
		}
	}
	return problems
}

// accumulator re-divides the error columns of results/accumulator.csv and checks
// the published ratio columns, the direction of the claim, and the one value the
// README quotes.
func accumulator(path string) []string {
	var problems []string
	header, rows, err := readCSV(path)
	if err != nil {
		return []string{fmt.Sprintf("unreadable: %v", err)}
	}
	idx := map[string]int{}
	for _, name := range []string{"N", "fp32 abs", "fp16 abs", "fp32 |sum-1|",
		"fp16 |sum-1|", "abs gap", "sum gap"} {
		i := col(header, name)
		if i < 0 {
			return []string{fmt.Sprintf("no column %q", name)}
		}
		idx[name] = i
	}

	var gotN []int
	worst := 0.0
	at8192 := math.NaN()
	for r, row := range rows {
		num := func(name string) float64 {
			v, e := strconv.ParseFloat(strings.TrimSpace(row[idx[name]]), 64)
			if e != nil {
				problems = append(problems, fmt.Sprintf("row %d: %q is not a number", r+2, name))
			}
			return v
		}
		n := int(num("N"))
		gotN = append(gotN, n)

		for _, p := range []struct{ gap, hi, lo string }{
			{"abs gap", "fp16 abs", "fp32 abs"},
			{"sum gap", "fp16 |sum-1|", "fp32 |sum-1|"},
		} {
			hi, lo, published := num(p.hi), num(p.lo), num(p.gap)
			if lo <= 0 {
				problems = append(problems, fmt.Sprintf("N=%d: %s is not positive", n, p.lo))
				continue
			}
			// The claim is directional before it is numerical: fp16 accumulators
			// must be worse than fp32 ones, not merely different.
			if !(hi > lo) {
				problems = append(problems, fmt.Sprintf(
					"N=%d: %s (%g) is not worse than %s (%g)", n, p.hi, hi, p.lo, lo))
			}
			rel := math.Abs(published-hi/lo) / math.Abs(hi/lo)
			if rel > relTol {
				problems = append(problems, fmt.Sprintf(
					"N=%d: %s published %.6f, recomputed %.6f, rel %.1e",
					n, p.gap, published, hi/lo, rel))
			}
			if rel > worst {
				worst = rel
			}
		}
		if n == 8192 {
			at8192 = num("fp16 abs") / num("fp32 abs")
		}
	}

	sort.Ints(gotN)
	if fmt.Sprint(gotN) != fmt.Sprint(wantN) {
		problems = append(problems, fmt.Sprintf(
			"sequence lengths are %v, expected %v", gotN, wantN))
	}
	// "fp32 accumulators are worth a factor of 3297 in maximum absolute error at
	// N = 8192" -- README section 2.
	if math.IsNaN(at8192) {
		problems = append(problems, "no N=8192 row to check the quoted factor against")
	} else if got := fmt.Sprintf("%.0f", at8192); got != "3297" {
		problems = append(problems, fmt.Sprintf(
			"README quotes a factor of 3297 at N=8192; the file gives %s", got))
	} else {
		fmt.Printf("  accumulator gap at N=8192: %.4f, README says 3297, ok\n", at8192)
	}
	fmt.Printf("  ratio columns re-divided for %d sequence lengths, worst relative "+
		"difference %.1e\n", len(rows), worst)
	return problems
}

func main() {
	root := flag.String("root", ".", "repository root")
	flag.Parse()

	paths, err := filepath.Glob(filepath.Join(*root, "results", "*.csv"))
	if err != nil || len(paths) == 0 {
		fmt.Fprintf(os.Stderr, "no CSVs under %s/results\n", *root)
		os.Exit(2)
	}
	sort.Strings(paths)

	failures := 0
	fmt.Printf("structural validation of %d files under results/\n", len(paths))
	for _, p := range paths {
		rel, _ := filepath.Rel(*root, p)
		problems := validate(p)
		if len(problems) == 0 {
			_, rows, _ := readCSV(p)
			fmt.Printf("  %-24s %3d data rows, well formed\n", rel, len(rows))
			continue
		}
		failures += len(problems)
		fmt.Printf("  %-24s FAIL\n", rel)
		for _, x := range problems {
			fmt.Printf("      %s\n", x)
		}
	}

	fmt.Println("\nrecomputing the derived columns of results/accumulator.csv")
	problems := accumulator(filepath.Join(*root, "results", "accumulator.csv"))
	for _, x := range problems {
		fmt.Printf("  FAIL: %s\n", x)
	}
	failures += len(problems)

	if failures > 0 {
		fmt.Printf("\n%d problems\n", failures)
		os.Exit(1)
	}
	fmt.Println("\nevery results CSV is well formed and the accumulator ratios recompute")
}
