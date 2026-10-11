#!/usr/bin/env bash
# Full local check: Debug+ASan/UBSan and Release builds, ctest, pytest.
set -euo pipefail
cd "$(dirname "$0")/.."
build() {
  local dir=$1; shift
  cmake -S . -B "$dir" -G Ninja "$@" >/dev/null
  cmake --build "$dir"
  ctest --test-dir "$dir" --output-on-failure
}
build build-debug -DCMAKE_BUILD_TYPE=Debug -DTINYCU_ENABLE_SANITIZERS=ON
build build-release -DCMAKE_BUILD_TYPE=Release
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q -e harness
pytest -q
