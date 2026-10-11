---
name: tinycu-07-verification
description: Phase 7 of TinyCU — compiler verification infrastructure: random UB-free kernel generator, differential testing (O0 vs O1 vs interpreter vs GPU vs nvcc), automatic test-case reducer, bug triage log, coverage and metrics. The Compiler-Verification showcase.
---

# Phase 7 — Verification infrastructure (the "break the compiler" part)

## Goal
An automated system that generates thousands of valid kernels, runs them through multiple
independent paths, flags any disagreement, shrinks it to a minimal repro, and logs it.

## 1. Random program generator `tcharness/gen.py` (seeded, reproducible)
- Generates a single kernel: params (`int* out`, `int* in0`, `float* fin`, `int n`, scalars),
  body = guard `if (i < n) { ... }` with random depth-limited statements: vardecls,
  assignments, if/else, bounded `for (int k = 0; k < C; k++)` (C ≤ 8), min/max/abs, casts.
- **UB-free by construction**: divisors for `/` and `%` are constant non-zero literals in
  [1,16] (or `(abs(e % 1000) % 13) + 1`, which is always in [1,13]; `e % 1000` first avoids
  abs(INT_MIN)). Indices only `i` (guarded by `i < n`). Never emit INT_MIN / -1.
- Two modes: `--int-only` (bit-exact, default) and `--float` (no div of near-zero; compare bit-exact
  between our own paths; ULP-tolerant only vs NVRTC).
- Size knobs: max stmts, max expr depth, var count. Output `.tc` + `.json` spec.
- Also generate the **CUDA twin** (`.cu`) by pretty-printing the same AST in CUDA syntax
  (wrap int arithmetic via `unsigned` casts to match wrapping semantics).

## 2. Differential oracle `tcharness/diff.py`
Paths per program: (A) interp -O0, (B) interp -O1, (C) GPU -O0, (D) GPU -O1, (E) NVRTC twin.
No GPU → A vs B only (still catches optimizer bugs; runs in CI).
Classify result: `OK`, `MISMATCH(pair)`, `CRASH(tinycu exit 3)`, `PTXAS_REJECT`,
`TIMEOUT`, `INTERP_ERROR` (generator bug!). Store artifacts under `fuzz_out/<seed>/`.

## 3. Fuzz driver `python -m tcharness.fuzz --n 5000 --jobs 8 [--gpu]`
Parallel (multiprocessing), prints throughput (programs/min) and a summary table; writes
`results/fuzz_summary.json`. Nightly-style CI job: 300 programs A vs B + ptxas acceptance.

## 4. Reducer `python -m tcharness.reduce fuzz_out/<seed>`
Delta-debugging on the **AST** (via `tinycu --emit=ast` round-trip or generator’s own AST):
try deleting statements, replacing expressions with sub-expressions/constants, unwrapping
if/for, shrinking constants; keep a change only if the *same* failure class persists and
program stays valid (sema passes, interpreter no error). Report lines before/after.
Plus **pass bisection**: rerun with `--disable-pass=` each pass to name the culprit pass.

## 5. Planted-bug validation (proves the fuzzer works — great interview story)
Add `--inject-bug=<id>` hidden flag with 3–4 deliberate miscompiles (e.g. constfold uses
signed overflow without wrap, copyprop ignores redefinition, DCE removes a store,
simplifycfg swaps condbr targets). Measure: % detected, programs-to-first-detection,
reduced size. results/bug_injection.md.

## 6. Real bug log `docs/BUGLOG.md`
Every genuine bug found in TinyCU by the fuzzer: seed, reduced repro, root cause, fix commit.
(Do not claim nvcc bugs unless truly confirmed; NVRTC mismatches are usually our semantics.)

## 7. Coverage
Build `-DTINYCU_COVERAGE=ON` (`--coverage`), run unit+golden+fuzz(1000), `gcovr` HTML +
summary → results/coverage.md (line % for src/opt, src/backend).

## 8. Metrics aggregator `python -m tcharness.metrics` → results/metrics.md
Pulls: #kernels in suite, #tests (ctest + pytest), fuzz programs run & throughput,
mismatches found/fixed, injected-bug detection rate, avg reduction ratio, coverage %,
O1 PTX instruction reduction %, register deltas. **Resume numbers come only from here.**

## Definition of Done
- [ ] 5,000+ generated programs run A vs B with 0 open mismatches (all found ones fixed + logged)
- [ ] GPU fuzz run (≥1,000) on Colab: C/D/E agree, or differences explained in BUGLOG
- [ ] Injected bugs detected and auto-reduced; reducer shrinks repros ≥80% on average
- [ ] results/metrics.md generated; Tag `v0.7`

## Interview notes
Ways to test a compiler: unit, golden/regression, spec conformance suites, differential
testing (Csmith, YARPGen, CUDAsmith), EMI (equivalence modulo inputs), metamorphic testing,
test-case reduction (delta debugging, C-Reduce), sanitizers, coverage-guided fuzzing,
performance regression tracking. The UB problem in random testing and how we avoid it.
