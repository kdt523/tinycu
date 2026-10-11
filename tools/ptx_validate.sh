#!/usr/bin/env bash
# Compile every valid test program (golden, warnings, kernels) at -O0 and -O1 and check that
# ptxas accepts the PTX.   tools/ptx_validate.sh [path/to/tinycu] [sm_75]
set -uo pipefail
cd "$(dirname "$0")/.."
TINYCU=${1:-build-debug/tinycu}
SM=${2:-sm_75}
out=$(mktemp -d)
total=0; failed=0
for tc in tests/golden/*.tc tests/warnings/*.tc tests/kernels/*.tc; do
  [ -e "$tc" ] || continue
  for opt in 0 1; do
    total=$((total + 1))
    ptx="$out/$(basename "${tc%.tc}")-O$opt.ptx"
    if ! "$TINYCU" "$tc" -O$opt --sm="$SM" -o "$ptx" 2>"$out/err"; then
      echo "FAIL compile: $tc -O$opt"; cat "$out/err"; failed=$((failed + 1)); continue
    fi
    if ! ptxas -arch="$SM" "$ptx" -o "${ptx%.ptx}.cubin" 2>"$out/err"; then
      echo "FAIL ptxas: $tc -O$opt"; cat "$out/err"; failed=$((failed + 1))
    fi
  done
done
echo "ptxas accepted $((total - failed))/$total compilations (target $SM)"
rm -rf "$out"
[ "$failed" -eq 0 ]
