# TinyCU status

| Phase | Skill | State | Tag |
|---|---|---|---|
| 00 | setup | done | v0.0 |
| 01 | frontend | done | v0.1 |
| 02 | sema | done | v0.2 |
| 03 | ir-interp | done | v0.3 |
| 04 | ptx-backend | done | v0.4 |
| 05 | gpu-runtime | done | v0.5 |
| 06 | optimizer | done | v0.6 |
| 07 | verification | done | v0.7 |
| 08 | release | todo | |

## Pending kd (needs the owner)
- [x] GitHub repo `kdt523/tinycu` created, first push done, CI green (2026-10-11)
- [x] Colab T4 run of `notebooks/run_on_colab.ipynb` (2026-10-11): Tesla T4, 108/108 cases bit-exact on numpy == interp == gpu == nvrtc; `results/*_colab.md`
- [ ] Colab GPU fuzz run (phase 7): optional, cell 8b of the notebook; ≥1,000 programs already ran on the local RTX 3050 (results/fuzz_summary_gpu.json)
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
- **Phase 4**: PTX emitter (ISA 7.0, register classes, `.rn` floats, `setp.neu` for f32 `!=`,
  hex f32 immediates, shared memory, barriers, fall-through branches, variable-name comments),
  `tools/ptx_check.sh`, `tools/ptx_validate.sh` (CI job), `python -m tcharness.ptxstats`.
  ptxas sm_75 accepts every valid test program at -O0 and -O1. Tests: 14 PTX unit tests,
  15 PTX golden snapshots, new golden `imm_operands.tc`.
- **Phase 5**: 22-kernel suite (`tests/kernels/*.{tc,cu,json}`), NumPy references, `tcharness.suite`
  (interp / gpu via CuPy / nvrtc twin / tinycu-run), C++ Driver API runner `tinycu-run` with JIT
  error log and cuEvent timing, `tcharness.bench`, Colab notebook.
  **A local GPU exists** (RTX 3050 Laptop, sm_86, via WSL2): 108/108 cases bit-exact on
  numpy == interp == gpu == nvrtc (results/kernel_suite.md); 130 GPU pytest + 153 non-GPU pytest.
  Pending kd: running the notebook on Colab's T4 (sm_75).
- **Phase 6**: pass manager (`-O1`, `--passes`, `--disable-pass`, `--dump-after`, `--stats`,
  `--verify-each`), constfold (shared `evalPure` with the interpreter) + algebraic simplification,
  copyprop, dce, simplifycfg; IR text parser for tests; `tcharness.optstats` →
  `results/opt_impact.md`. Tests: 44 optimizer unit tests, 5 pass-dump goldens, all 22 kernels
  at -O1 == -O0 bit-exact on interp + GPU + NVRTC (108/108), ptxas accepts every program at -O1.
- **Phase 7**: UB-free random kernel generator with CUDA twins (`tcharness.gen`), 5-path
  differential oracle (`tcharness.diff`), parallel fuzzer (`tcharness.fuzz`), AST reducer + pass
  bisection (`tcharness.reduce`), 4 planted bugs (`--inject-bug`) with `tcharness.bug_injection`,
  coverage (`tools/coverage.sh`), metrics aggregator (`tcharness.metrics` → results/metrics.md),
  CI fuzz-smoke job (300 programs), docs/VERIFICATION.md, docs/BUGLOG.md.
