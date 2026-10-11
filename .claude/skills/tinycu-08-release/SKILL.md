---
name: tinycu-08-release
description: Phase 8 of TinyCU — production polish: README with architecture diagram, docs, CI badges, release build, demo, interview notes and resume bullets generated from real metrics.
---

# Phase 8 — Polish & release

## Tasks
1. **README.md** (this is what recruiters see — top 30 lines matter most):
   - One-line pitch + badges (CI, coverage, license).
   - 10-second demo: saxpy `.tc` → `tinycu saxpy.tc --emit=ptx` snippet → "runs on T4, matches nvcc".
   - Pipeline diagram (Mermaid): Lexer → Parser → Sema → IR → Opt → PTX → ptxas → GPU,
     with Interpreter + Fuzzer + Reducer box beside it.
   - Feature list, Quickstart (WSL + Colab), Language tour, Results table (from metrics.md),
     "How TinyCU is verified" section, Limitations & future work (SSA, LLVM backend,
     warp shuffles, loop unrolling), links to docs.
2. Docs complete: LANGUAGE.md, ARCHITECTURE.md, IR.md, VERIFICATION.md, BUGLOG.md,
   INTERVIEW_NOTES.md (compiler phases, PTX, CUDA model, testing strategies, Q&A).
3. Code polish: clang-format all, clang-tidy (modernize, bugprone) clean, no TODOs left in
   core, Doxygen-style comments on public headers.
4. CI final: matrix gcc/clang, sanitizers, ptx-validate, fuzz-smoke (300), coverage upload
   (Codecov), release workflow building `tinycu` Linux binary on tag.
5. `examples/` folder: 6 showcase kernels with expected output.
6. Demo: 2-min screen recording (asciinema or video) — compile, show PTX, run on Colab,
   show fuzzer catching an injected bug and reducing it.
7. **Resume entry** — generate `docs/RESUME.md` from results/metrics.md, template:
   **TinyCU — Mini GPU Compiler & Verification Suite targeting NVIDIA PTX** | C++17, CUDA, PTX, Python, CMake, GitHub Actions
   - Built a C++17 compiler for a CUDA-like kernel language (hand-written lexer, recursive-descent parser, typed AST, three-address IR) that emits PTX validated by ptxas and executed on an NVIDIA T4 GPU via the CUDA Driver API.
   - Implemented an -O1 pipeline (constant folding, copy propagation, DCE, CFG simplification) cutting PTX instructions by [X]% across [N] kernels; verified bit-exact against nvcc-compiled CUDA twins.
   - Built verification infrastructure: UB-free random kernel generator, 5-way differential testing (interpreter, -O0/-O1, GPU, NVRTC) and delta-debugging reducer; ran [K] programs, found and fixed [B] optimizer bugs, detected [D]/[T] injected miscompiles with [R]% average repro reduction.
   - [T] unit/golden tests with [C]% line coverage; CI runs sanitizers, ptxas validation and fuzz smoke tests on every push.
   Fill brackets ONLY from results/metrics.md.
8. Final checklist in STATUS.md; tag `v1.0.0`; GitHub release notes.

## Definition of Done
- [ ] Fresh clone → build → tests green in < 5 min on Linux; Colab notebook green
- [ ] README results table matches results/metrics.md exactly
- [ ] v1.0.0 released; pinned on GitHub profile
