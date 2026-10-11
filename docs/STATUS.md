# TinyCU status

| Phase | Skill | State | Tag |
|---|---|---|---|
| 00 | setup | done | v0.0 |
| 01 | frontend | done | v0.1 |
| 02 | sema | todo | |
| 03 | ir-interp | todo | |
| 04 | ptx-backend | todo | |
| 05 | gpu-runtime | todo | |
| 06 | optimizer | todo | |
| 07 | verification | todo | |
| 08 | release | todo | |

## Pending kd (needs the owner)
- [ ] Create GitHub repo `kdt523/tinycu`, `gh auth login`, first push (CI green on push)
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
