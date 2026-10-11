# How TinyCU is verified

TinyCU is tested at every level, from single tokens up to random whole programs running on a
GPU. All numbers quoted anywhere come from `results/metrics.md` (`python -m tcharness.metrics`).

| Layer | What | Where |
|---|---|---|
| Unit tests | lexer, parser, sema, IR/verifier/CFG, interpreter, every optimizer pass, PTX emitter | `tests/unit/*.cpp` (GoogleTest) |
| Golden tests | `--emit=ast/ir/ptx` and `--dump-after=<pass>` snapshots, byte-for-byte | `tests/golden/` |
| Diagnostics | one file per error or warning, checking `file:L:C: error: message` | `tests/errors/`, `tests/warnings/` |
| Robustness | 4,000 random byte strings / token shuffles / substitutions through the lexer, parser and sema under ASan + UBSan | `tests/unit/test_frontend_fuzz.cpp` |
| IR verifier | after lowering and after every pass (`--verify-each`) | `src/ir/verifier.cpp` |
| Oracle | interpreter with exact semantics; every UB is reported, never executed | `src/interp/` |
| Kernel suite | 22 kernels × sizes: NumPy == interpreter == GPU == nvcc twin, bit-exact, at -O0 and -O1 | `tests/kernels/`, `tcharness.suite` |
| ptxas acceptance | every valid program at -O0 and -O1 | `tools/ptx_validate.sh` (CI) |
| Random differential testing | UB-free random kernels through 5 paths | `tcharness.gen/diff/fuzz` |
| Test-case reduction | delta debugging on the program AST plus pass bisection | `tcharness.reduce` |
| Planted bugs | 4 deliberate miscompiles must be found and reduced | `--inject-bug`, `tcharness.bug_injection` |
| Coverage | gcov line coverage of ctest + fuzzing | `tools/coverage.sh` |

## The oracle problem and UB
Random testing needs an answer key. TinyCU has three:
- the **interpreter** (`src/interp`), written for obviousness and checking every undefined
  behaviour: out-of-bounds access, division by zero, `INT_MIN / -1`, out-of-range float→int,
  reading uninitialized shared memory, barrier divergence, endless loops;
- **NumPy references** for the fixed kernel suite;
- **nvcc**, via the CUDA twin of every program.

A program with UB has no defined result, so a disagreement on it means nothing. The
generator is therefore **UB-free by construction**, and the interpreter *proves* it on every
run: if the reference path reports UB, the result is `INTERP_ERROR`, a generator bug, never
a compiler bug.

## Random program generator (`tcharness/gen.py`)
Seeded and reproducible (`python -m tcharness.gen --seed 42 [--float]`). Every program has the
signature `kernel fuzz(int* out, float* fout, int* in0, float* fin, int n, int s0, float f0)`
and a guarded body `if (i < n) { … }` containing random declarations, assignments, compound
assignments, variable copies and swaps, nested `if`/`else`, bounded `for` loops (≤ 8 iterations),
`min`/`max`/`abs`/`sqrt`, casts, short-circuit conditions, and selects (hoisted into `if`s for
TinyCU, `?:` in CUDA).

How each source of UB is avoided:

| UB | Prevention |
|---|---|
| division / remainder by 0, `INT_MIN / -1` | divisor is a literal in [1,16] or `(abs(e % 1000) % 13) + 1` ∈ [1,13] |
| `abs(INT_MIN)` in the twin | TinyCU defines it; the twin uses a wrapping helper |
| signed overflow in the CUDA twin | int `+ - *` go through `unsigned` helpers (`tc_add`…) |
| float → int out of range / NaN | `(int)min(max(e, -1e6), 1e6)`; PTX `min`/`max` drop a NaN operand |
| out-of-bounds | memory is only indexed with the guarded `i` |
| uninitialized variables | every declaration has an initializer |
| unsigned `threadIdx.x` in CUDA | the twin casts it: `(int)threadIdx.x` |
| endless loops | constant trip counts; the loop variable is read-only |

**Float mode** (`--float`, 30% of programs by default, 50% in GPU runs) adds float arithmetic and
deliberately produces **NaN and ±inf from run-time values**: `sqrt` of negative inputs, division
by zero inputs, and huge edge values. The input arrays are rich in edge values: `INT_MIN`,
`INT_MAX`, 0, −0.0, ±1e30, subnormals (±1e-40), 3.4e38.

**Checksum.** The last statement stores the sum of *every* top-level variable (as Csmith does),
so a wrong value anywhere becomes visible in the output. Adding this raised detection of planted
bug 1 from 9 to 31 per 2,000 programs, and bug 2 from 0 to 55.

## Differential oracle (`tcharness/diff.py`)
| Path | Meaning |
|---|---|
| A | interpreter, `-O0`: the reference |
| B | interpreter, `-O1`: catches optimizer bugs without a GPU, runs in CI |
| C | TinyCU PTX on the GPU, `-O0`: backend + ptxas + hardware |
| D | TinyCU PTX on the GPU, `-O1` |
| E | the CUDA twin compiled by NVRTC (`-fmad=false`, no flush-to-zero) |

Classes: `OK`, `MISMATCH(A-X)`, `CRASH` (tinycu exit 3), `PTXAS_REJECT`, `TIMEOUT`, and two
generator-bug classes, `INTERP_ERROR` and `COMPILE_ERROR`. Failing programs keep their directory
(`fuzz_out/<seed>/`: `k.tc`, `k.cu`, `k.json`, `ast.json`, `result.json`).

**Comparison policy:** bit-exact, except that any two NaNs are equal. NaN payloads are not part of
TinyCU's semantics: x86 produces `0xFFC00000` for `0/0`, NVIDIA GPUs the canonical `0x7FFFFFFF`.
`+0.0` vs `−0.0` *is* a mismatch.

## Fuzz driver (`tcharness/fuzz.py`)
`python -m tcharness.fuzz --n 5000 --jobs 8 [--gpu]` runs the CPU paths in a process pool and the
GPU paths in the main process (one CUDA context). It prints throughput and a summary, and writes
`results/fuzz_summary*.json`. CI runs 300 programs (A vs B + ptxas) on every push.

## Reducer (`tcharness/reduce.py`)
`python -m tcharness.reduce fuzz_out/<seed>` does greedy delta debugging on the generator's AST:
1. delete statement chunks (halves down to single statements);
2. replace `if` with a branch and `for` with one iteration, and shrink trip counts;
3. replace expressions with a same-typed sub-expression or a constant;
4. shrink literals.

A candidate is kept only if it **still fails with the same label** and **strictly decreases** the
measure (AST size, sum of |literals|); the measure prevents cycles. Every candidate goes through
the full oracle, so invalid programs are rejected automatically. A cursor continues after an
accepted change instead of restarting.

**Pass bisection** then re-runs the reduced program with `--disable-pass=<p>` for each `-O1`
pass. Passes whose removal makes the failure disappear are the culprits.

## Planted-bug validation (`tcharness/bug_injection.py`)
A fuzzer that never finds anything proves nothing. A hidden flag, `tinycu --inject-bug=N`, plants
one of four realistic miscompiles:

| N | Pass | Defect |
|---|---|---|
| 1 | constfold | int `+ - *` folded with saturation instead of wrap-around |
| 2 | copyprop | copy not invalidated when its *source* is redefined (the "swap problem") |
| 3 | dce | stores of constant values deleted |
| 4 | simplifycfg | condbr targets swapped while threading a jump |

The script fuzzes each bug on the same seeds, reduces the first failures and checks that
bisection names the right pass. Results are in `results/bug_injection.md`.

Bug 2 was originally **never** detected, because copy propagation chains back to never-redefined
temporaries, so the stale fact is harmless within one block. It only matters when a variable
defined in an earlier block is swapped inside an `if` or loop. Adding copies and swaps to the
generator, plus the checksum, made it detectable. A planted bug that isn't found is a coverage
gap in the generator, and here it was fixed there.

## Coverage
`tools/coverage.sh` builds with `--coverage`, runs every ctest test plus 1,000 fuzz programs, and
writes `results/coverage.md` / `.json` with line coverage per source directory (gcovr).

## What is *not* covered
- Barriers and shared memory are not generated randomly; they are covered by the kernel suite
  and unit tests.
- The generator never produces programs with UB, so UB *handling* is covered by interpreter unit
  tests, not by fuzzing.
- The interpreter and the constant folder share `evalPure` by design. A bug there would affect
  paths A and B identically, and only the GPU paths C–E could catch it.
