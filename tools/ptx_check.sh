#!/usr/bin/env bash
# Assemble one PTX file with ptxas and print its resource usage.
#   tools/ptx_check.sh k.ptx [sm_75]
# Prints one line per kernel: "<kernel> regs=N spill_stores=B spill_loads=B smem=B";
# exits non-zero if ptxas rejects the file.
set -euo pipefail
ptx=$1
sm=${2:-sm_75}
cubin=${ptx%.ptx}.cubin
log=$(ptxas -arch="$sm" -v "$ptx" -o "$cubin" 2>&1) || { echo "$log" >&2; exit 1; }
echo "$log" | awk '
  /Compiling entry function/ { gsub(/[\x27]/, "", $0); k = $(NF-2) }
  /bytes spill stores/ { ss = $5; sl = $9 }
  /Used [0-9]+ registers/ {
    regs = $0; sub(/.*Used /, "", regs); sub(/ registers.*/, "", regs)
    smem = 0; if (match($0, /[0-9]+ bytes smem/)) { smem = substr($0, RSTART, RLENGTH); sub(/ bytes smem/, "", smem) }
    printf "%s regs=%s spill_stores=%s spill_loads=%s smem=%s\n", k, regs, ss, sl, smem
  }'
