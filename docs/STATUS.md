# TinyCU status

| Phase | Skill | State | Tag |
|---|---|---|---|
| 00 | setup | done | v0.0 |
| 01 | frontend | done | v0.1 |
| 02 | sema | done | v0.2 |
| 03 | ir-interp | done | v0.3 |
| 04 | ptx-backend | todo | |
| 05 | gpu-runtime | todo | |
| 06 | optimizer | todo | |
| 07 | verification | todo | |
| 08 | release | todo | |

## Pending kd (needs the owner)
- [x] GitHub repo `kdt523/tinycu` created, first push done, CI green (2026-10-11)
- [ ] Colab GPU runs (phases 5 and 7)
- [ ] Codecov token (phase 8)
- [ ] Demo recording (phase 8)
- [ ] Pin repo on GitHub profile (phase 8)

## Environment
- WSL2 Ubuntu 24.04, g++ 13.3, clang, CMake 3.28, Ninja 1.11, Python 3.12
- CUDA toolkit 12.6 (`nvcc`, `ptxas` at /usr/local/cuda/bin) — no GPU locally
- Local full check: `tools/check.sh` (Debug+ASan/UBSan, Release, pytest)

## Phase log
- **Phase 0**: CMake + GoogleTest + pytest skeleton, CLI stub with exit codes, CI workflow.
- **Phase 1**: lexer, recursive-descent parser with panic-mode recovery, AST + `--emit=ast`.
  Tests: 19 lexer, 21 parser, 4 robustness-fuzz unit tests (3,000 random inputs under ASan/UBSan),
  13 golden programs, 18 syntax-error cases.
- **Phase 1 follow-up**: optional CUDA-style `.x` on builtin variables (`threadIdx.x`).
- **Phase 2**: types (interned), scopes/symbols with per-kernel ids, implicit int→float casts,
  lvalue/pointer/shared-array rules, typed `--emit=ast`. Tests: 15 sema unit tests, sema fuzz
  (1,000 token substitutions), 29 sema error cases, 6 warning cases, all 14 golden programs
  re-snapshotted with types.
- **Phase 3**: non-SSA three-address IR, lowering (short-circuit, loops, single exit), verifier,
  CFG utilities, printer (`--emit=ir`), interpreter oracle (`--interp`, JSON I/O, UB detection,
  barrier lock-step), Python `tcharness.tinycu` wrapper. Tests: 18 IR/verifier/CFG unit tests,
  23 interpreter unit tests, 20 pytest NumPy bit-exact checks (13 kernels + 1 runtime-error check), 14 IR goldens.
  Decisions: uninitialized locals are 0; f32 `!=` is true for NaN (PTX must use `setp.neu`);
  exit code 4 = kernel runtime error.
