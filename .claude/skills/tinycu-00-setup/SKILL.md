---
name: tinycu-00-setup
description: Phase 0 of TinyCU — create repo skeleton, CMake build, test frameworks, CLI stub and CI. Use first, before any other tinycu phase.
---

# Phase 0 — Project setup

## Goal
An empty-but-working project: `tinycu --version` builds, one C++ test and one Python test
pass locally and in GitHub Actions.

## Tasks
1. Create the layout from CLAUDE.md §4 (empty files are fine, add `.gitkeep`).
2. **CMakeLists.txt**
   - `project(tinycu LANGUAGES CXX)`, `CMAKE_CXX_STANDARD 17`, export compile_commands.
   - Static lib `tinycu_core` (all of `src/` except driver/runtime) + executable `tinycu`.
   - Warnings: `-Wall -Wextra -Werror -pedantic`; add `-ffp-contract=off` globally.
   - Options: `TINYCU_ENABLE_SANITIZERS` (ASan+UBSan, ON in Debug CI), `TINYCU_BUILD_RUNTIME`
     (AUTO: ON if `find_package(CUDAToolkit)` succeeds; links `CUDA::cuda_driver`).
   - GoogleTest via `FetchContent` (tag v1.14.0); `enable_testing()`; `gtest_discover_tests`.
3. **CLI** (`src/driver/main.cpp`) — hand-rolled arg parsing, no deps:
   ```
   tinycu <input.tc> [-o out.ptx] [-O0|-O1] [--emit=ast|ir|ptx] [--dump-after=<pass>]
          [--interp --grid N --block M --args args.json] [--sm=sm_75] [--version] [--help]
   ```
   Exit codes: 0 ok, 1 compile error, 2 usage error, 3 internal error.
   For now, print "not implemented" for every stage.
4. **Python harness skeleton** `harness/tcharness/` with `pyproject.toml`
   (deps: numpy, pytest, hypothesis optional; `cupy` as extra `[gpu]`).
   `pytest.ini`: marker `gpu`; default addopts `-m "not gpu"`.
5. `.clang-format` (LLVM, ColumnLimit 100), `.gitignore`, MIT `LICENSE`, `README.md` stub,
   `docs/STATUS.md` (phase checklist table).
6. **CI** `.github/workflows/ci.yml`: ubuntu-latest; jobs `build-test` (Debug+sanitizers,
   Release) → `cmake -G Ninja`, build, `ctest --output-on-failure`, `pip install -e harness`,
   `pytest`. Add a placeholder job `ptx-validate` (filled in phase 4) that installs CUDA
   toolkit with `Jimver/cuda-toolkit` (sub-packages: nvcc) — no GPU needed for ptxas.
7. Smoke tests: `tests/unit/test_smoke.cpp` (1+1), `harness/tests/test_smoke.py`.

## Definition of Done
- [ ] `cmake -B build -G Ninja && cmake --build build && ctest --test-dir build` green
- [ ] `./build/tinycu --version` prints `tinycu 0.0.1`
- [ ] `pytest` green; CI green on push
- [ ] STATUS.md shows phase 0 done; tag `v0.0`

## Interview notes to learn in this phase
How a compiler driver works (gcc driver vs cc1 vs as vs ld); what `nvcc` does internally:
`.cu → cudafe++ → cicc (NVVM/LLVM) → PTX → ptxas → cubin (SASS) → fatbinary → host link`.
