# TinyCU — a mini GPU compiler targeting NVIDIA PTX

> Master brief for Claude Code. Read this file fully, then execute the phase skills in
> `.claude/skills/` **in order** (00 → 08). Do not skip a phase's "Definition of Done".

## 1. What we are building
TinyCU compiles a small CUDA-like kernel language (`.tc` files) into **NVIDIA PTX**,
which is assembled by `ptxas` and executed on a real NVIDIA GPU. It ships with:
- a C++17 compiler (`tinycu`): lexer → parser → AST → semantic analysis → 3-address IR
  → optimizer → PTX emitter
- an **IR interpreter** (CPU reference oracle that emulates the GPU thread grid)
- a **runner** that loads PTX on a GPU (C++ CUDA Driver API + Python/CuPy)
- a **verification suite**: golden tests, differential tests vs `nvcc`/NVRTC,
  a random-kernel fuzzer (`-O0` vs `-O1` vs interpreter), and a test-case reducer
- CI (GitHub Actions) + docs + metrics report

Purpose: portfolio project for NVIDIA *Compiler Verification* and *PTX Compiler (PTXAS)*
System Software Intern roles. Code quality, tests and honest metrics matter more than features.

## 2. Non-goals (keep scope small — do NOT build these)
- No LLVM/MLIR dependency. No full C. No structs, no user function calls, no recursion.
- No SSA form, no register allocator (PTX has unlimited virtual registers; ptxas allocates).
- No 2D/3D thread indices (x dimension only). No float atomics. No templates.

## 3. Environment
- Developer machine: Windows 11 → use **WSL2 Ubuntu 22.04/24.04** for all dev.
- Build: CMake ≥ 3.20, g++ ≥ 11 or clang ≥ 14, Ninja. Python ≥ 3.10.
- `ptxas` / `nvcc` / `cuobjdump`: install CUDA Toolkit in WSL (works **without a GPU**,
  used for PTX validation + register stats).
- GPU execution: **Google Colab (T4, sm_75)** via `notebooks/run_on_colab.ipynb`.
- Everything except actual GPU execution must be runnable & testable without a GPU
  (the interpreter is the oracle). GPU-only tests are marked `@pytest.mark.gpu`.

## 4. Repository layout (target)
```
tinycu/
  CMakeLists.txt
  include/tinycu/   (public headers: token.h ast.h types.h diag.h ir.h ...)
  src/
    driver/main.cpp            CLI
    frontend/lexer.cpp parser.cpp
    sema/sema.cpp
    ir/lower.cpp ir.cpp ir_printer.cpp verifier.cpp
    interp/interp.cpp
    opt/constfold.cpp dce.cpp simplifycfg.cpp copyprop.cpp passmanager.cpp
    backend/ptx_emitter.cpp
    runtime/tinycu_run.cpp     (CUDA Driver API runner; built only if CUDA found)
  tests/
    unit/          GoogleTest C++ unit tests
    golden/        *.tc + expected *.ast/*.ir/*.ptx snapshots
    errors/        *.tc that must fail with an expected diagnostic
    kernels/       *.tc + *.cu twin + *.json test spec (sizes, inputs, launch cfg)
  harness/         Python package: tcharness (run, diff, fuzz, reduce, metrics)
  notebooks/run_on_colab.ipynb
  docs/  LANGUAGE.md  ARCHITECTURE.md  IR.md  VERIFICATION.md  INTERVIEW_NOTES.md
  results/metrics.md (generated, never hand-edited)
  .github/workflows/ci.yml
```

## 5. Global engineering rules
1. C++17, `-Wall -Wextra -Werror -pedantic`, no exceptions for control flow in hot paths
   (diagnostics are collected, not thrown). Format with clang-format (LLVM style, 100 cols).
2. Every phase adds tests **before** being called done. Run `ctest` + `pytest -m "not gpu"`
   after every change; never commit red.
3. One git commit per meaningful step, message `phaseN: <what>`. Tag `v0.N` at phase end.
4. Diagnostics format: `file.tc:LINE:COL: error: message` (+ caret line). Stable for tests.
5. Determinism: same input + flags ⇒ byte-identical output (needed for golden tests).
6. **Honest metrics only.** All numbers in README/resume come from `results/metrics.md`,
   produced by `python -m tcharness.metrics`. Never invent numbers.
7. Float semantics: emit `.rn` rounding on f32 add/sub/mul (prevents ptxas FMA contraction),
   compile the interpreter with `-ffp-contract=off`. Goal: bit-exact GPU vs interpreter.
8. Int semantics: 32-bit two's-complement **wrapping**. Division/modulo by zero and
   `INT_MIN / -1` are undefined; the fuzzer must never generate them.
9. If a phase needs a GPU and none is available, finish everything else, mark GPU steps as
   "pending Colab run" in `docs/STATUS.md`, and continue.

## 6. Phase order (each = one skill)
| # | Skill | Output | Est. |
|---|-------|--------|------|
| 00 | tinycu-00-setup | repo, CMake, CI skeleton, CLI stub | 0.5 d |
| 01 | tinycu-01-frontend | lexer, parser, AST, `--dump-ast` | 2.5 d |
| 02 | tinycu-02-sema | types, scopes, diagnostics | 1.5 d |
| 03 | tinycu-03-ir-interp | 3-address IR, lowering, verifier, interpreter | 3 d |
| 04 | tinycu-04-ptx-backend | PTX emitter, ptxas validation | 3 d |
| 05 | tinycu-05-gpu-runtime | C++ & Python runners, Colab notebook, kernel suite | 2 d |
| 06 | tinycu-06-optimizer | const fold/prop, copy prop, DCE, simplify-CFG, `-O1` | 2.5 d |
| 07 | tinycu-07-verification | differential tests, fuzzer, reducer, metrics | 3 d |
| 08 | tinycu-08-release | docs, CI polish, README, demo, resume bullets | 1.5 d |

Total ≈ 19–20 working days.

## 7. How to run a phase
"Read CLAUDE.md, then execute skill `tinycu-0N-...`. Follow its Tasks in order, satisfy every
item of its Definition of Done, update docs/STATUS.md, commit, tag."
