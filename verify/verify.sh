#!/usr/bin/env bash
# Recompute what this repository publishes, in languages that are not Python.
#
# Every figure in the README came out of one implementation. The kernel is
# NumPy and PyTorch, the tables are pandas, the plots read the same CSVs the
# tables do, and scripts/check_numbers.py compares the prose against those CSVs
# in Python again. Nothing in that chain is independent of anything else in it:
# an error in the derivation would be reproduced by everything downstream,
# because everything downstream reads its output.
#
# So each check here recomputes a published quantity from the rawest form of it
# in the repository, in a different language, by a different route. A mistake
# would have to be made identically in C, Rust, Go, SQL, R and JavaScript to
# survive.
#
# Each is skipped with a clear message if its toolchain is missing, so this runs
# on a laptop with only some of them installed. CI has all of them.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

pass=0 fail=0 skip=0
tmp="${TMPDIR:-/tmp}"

run () {
    local name="$1" tool="$2"; shift 2
    printf '\n=== %s ===\n' "$name"
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'skipped: %s is not installed\n' "$tool"
        skip=$((skip + 1)); return
    fi
    "$@"
    case $? in
        0)  pass=$((pass + 1)) ;;
        77) skip=$((skip + 1)) ;;   # ran, but its own prerequisite was missing
        *)  fail=$((fail + 1)) ;;
    esac
}

# sqlite3 reads stdin, and inside a script stdin is the script, so it would eat
# the rest of this file and return nothing while looking like it had worked.
# Redirect it. Its CSV output is also CRLF, hence the tr.
check_sql () {
    local out
    out=$(sqlite3 -init verify/roofline.sql :memory: "" < /dev/null 2>&1 | tr -d '\r')
    printf '%s\n' "$out" | awk -F'|' '{printf "  %-46s %-14s %s\n", $1, $2, $4}'
    if printf '%s\n' "$out" | grep -q 'FAIL'; then
        echo "SQL disagrees with the published figures"
        return 1
    fi
    if [ "$(printf '%s\n' "$out" | grep -c '|ok$')" -lt 10 ]; then
        echo "SQL produced too few checks to be doing anything"
        return 1
    fi
    return 0
}

check_c () {
    cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror \
       -o "$tmp/fa_attention" verify/attention.c -lm || return 1
    "$tmp/fa_attention" "$root"
}

check_go () { ( cd verify/gocheck && go run . -root "$root" ); }

check_rust () { ( cd verify/tiling && cargo run --release --quiet -- "$root" ); }

# The golden file is the input to the C and Rust checks, so something has to
# check the golden file itself. This re-derives its outputs from its own stored
# inputs, which needs torch but no random seed.
check_golden () {
    local py=.venv/bin/python
    [ -x "$py" ] || py=python3
    if ! "$py" -c "import torch, numpy" >/dev/null 2>&1; then
        echo "skipped: $py has no torch, which the fp64 reference needs"
        return 77
    fi
    "$py" verify/export_golden.py --check
}

run "C, tiled attention against the fp64 reference"   cc      check_c
run "Rust, every block shape"                         cargo   check_rust
run "Go, structure of results/ and the accumulator"   go      check_go
run "SQL, derived columns and the headline ratios"    sqlite3 check_sql
run "R, the N^2 fit and the cliff"                    Rscript Rscript verify/scaling.R "$root"
run "JavaScript, byte counts and share of peak"       node    node verify/traffic.js "$root"
run "Python, the golden file against its own inputs"  python3 check_golden

printf '\n%s\n' "----------------------------------------"
printf '%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
[ "$pass" -gt 0 ] || { echo "nothing ran"; exit 1; }
