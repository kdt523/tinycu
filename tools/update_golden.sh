#!/usr/bin/env bash
# Regenerate golden snapshots. REVIEW THE DIFF (git diff tests/golden) before committing!
#   tools/update_golden.sh [build-dir] [ast|ir|ptx ...]
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${1:-build-debug}
shift || true
KINDS=${*:-ast}
for tc in tests/golden/*.tc; do
  for k in $KINDS; do
    "$BUILD/tinycu" "$tc" --emit="$k" > "${tc%.tc}.$k"
  done
  # Pass dumps: regenerate every existing <name>.dump-<pass> snapshot.
  for dump in "${tc%.tc}".dump-*; do
    [ -e "$dump" ] || continue
    "$BUILD/tinycu" "$tc" -O1 --dump-after="${dump##*.dump-}" --emit=ir 2> "$dump" > /dev/null
  done
done
git --no-pager diff --stat -- tests/golden || true
