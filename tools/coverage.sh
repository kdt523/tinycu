#!/usr/bin/env bash
# Line coverage of the compiler: unit + golden + error tests (ctest) and a fuzz run.
#   tools/coverage.sh [fuzz-programs=1000]
# Writes results/coverage.md and results/coverage.json (needs gcovr).
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-1000}
cmake -S . -B build-cov -G Ninja -DCMAKE_BUILD_TYPE=Debug -DTINYCU_COVERAGE=ON \
      -DTINYCU_BUILD_RUNTIME=OFF > /dev/null
cmake --build build-cov
find build-cov -name '*.gcda' -delete
ctest --test-dir build-cov -j"$(nproc)" > /dev/null
. .venv/bin/activate 2>/dev/null || true
TINYCU=$PWD/build-cov/tinycu python -m tcharness.fuzz --n "$N" --jobs "$(nproc)" \
  --out "$(mktemp -d)" --float-ratio 0.3 > /dev/null || true
gcovr -r . --filter 'src/' --exclude 'src/runtime/' --object-directory build-cov \
  --json-summary -o results/coverage.json > /dev/null
python3 tools/coverage_md.py results/coverage.json "$N" > results/coverage.md
cat results/coverage.md
