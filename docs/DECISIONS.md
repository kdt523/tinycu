# Design decisions

One line per decision with the reason. Newest at the bottom of each section.

## Language
- **Builtins without `.x`, `.x` optional** (`threadIdx` == `threadIdx.x`; `.y`/`.z` rejected): TinyCU is 1-D; the optional `.x` makes code read like CUDA.
- **`-2147483648` cannot be written as a literal**: `-` is a unary operator on `2147483648`, which overflows int32; write `-2147483647 - 1`. Keeps the lexer simple and literal overflow always an error.
- **Float literals that overflow to inf, or non-zero ones that underflow to 0, are lexer errors**: silent changes of a constant's value are a classic source of surprises; inf can never reach codegen as an immediate.
- **`.5` and `1.` are not float literals**: one unambiguous form (`0.5`, `1.0`) keeps the lexer and the spec small.
- **No C truthiness** (`if (n)` is an error; conditions must be bool): removes a whole class of int/bool confusion and makes the type rules (and the fuzzer) simpler.
- **`bool` is internal only** (no keyword, cannot be stored, `bool == bool` rejected): bools only exist as comparison/logic results, which maps 1:1 to PTX predicates.
- **Only int→float is implicit; float→int needs `(int)`**: no silent precision loss.
- **Pointers can only be indexed** (no arithmetic, comparison, copies, local pointer variables): every memory access is `param[i]` or `shared[i]`, which makes bounds checking in the interpreter exact and the backend trivial.
- **Shadowing is a warning, redeclaration in the same scope is an error**: as specified; shadowing is legal but usually a mistake.
- **Scalar locals are zero-initialized**: makes use-before-assignment defined (sema still warns), so the interpreter and the GPU agree; `-O1` may delete the `mov 0` only when the variable is provably written before every read.
- **Shared arrays are not initialized; reading an unwritten element is a run-time error** (`uninit_shared`): the GPU does not zero shared memory, so any such read would make GPU and interpreter disagree.
- **Float `!=` is true when an operand is NaN (C semantics)**: PTX must use the unordered `setp.neu.f32`; `==`, `<`, `<=`, `>`, `>=` are ordered (`setp.eq/lt/le/gt/ge.f32`).
- **`min`/`max` with mixed int/float promote to float**: consistent with arithmetic promotion.
- **Kernel names must be unique per file**: one PTX module holds all kernels; `--kernel` selects one for `--interp`.

## Compiler
- **Diagnostics are collected, never thrown; capped at 20 errors; printed in source order**: robust recovery, deterministic output for tests.
- **Parser recovery skips a balanced `{…}` body**: otherwise `if (x {…}` makes the kernel's `}` look unmatched and produces cascades.
- **Nesting depth limit 200 in the parser**: random inputs like `((((…` must not overflow the stack.
- **Sema uses an `<error>` type**: one mistake → one diagnostic, no cascades.
- **Implicit conversions are real AST nodes inserted by sema**: lowering never re-derives conversion rules.
- **IR is non-SSA (variables are named vregs, assignment = `mov`)**: simpler lowering and passes; PTX allows re-assignment and ptxas builds SSA itself. SSA is future work.
- **Short-circuit `&&`/`||` lowered as control flow**: required for `i < n && x[i] > 0` to be safe.
- **Single `exit` block with the only `ret`**: simpler CFG invariants.
- **Labels `<kind>.<counter>`, blocks appended in source order, f32 immediates printed with `%.9g`**: byte-for-byte deterministic output for golden tests.
- **Verifier after lowering and after every pass (`--verify-each`, default on in Debug)**: catch malformed IR at the pass that created it.

## Backend
- **PTX ISA 7.0, `.target` from `--sm` (default sm_75)**: works with every CUDA ≥ 11 driver JIT, including Colab's T4.
- **One PTX register per IR vreg, no register allocator**: PTX has unlimited virtual registers; ptxas allocates (and reports spills).
- **`.rn` on every f32 add/sub/mul/div/sqrt**: forbids ptxas FMA contraction; verified in SASS (FMUL+FADD, no FFMA). Needed for bit-exact GPU vs interpreter.
- **f32 immediates as hex bit patterns (`0f3F800000`)**: no decimal round-trip errors; subnormals and −0 stay exact.
- **`bra.uni` only for unconditional jumps; conditional branches use `@%p bra`**: `bra.uni` promises non-divergence, which only holds when every thread jumps.
- **Fall-through branches are omitted, `@!%p bra F` when the true target falls through**: shorter, nvcc-like PTX.
- **Shared arrays are declared inside the kernel as `__shared_<name>`**: per-kernel scope, no clashes between kernels in one module.
- **The `mul.wide` index is always a register** (immediates are moved first): uniform code; ptxas folds it anyway.
- **SASS instruction counts exclude NOPs and the trailing self-`BRA`**: those are alignment padding, not code; counting them would inflate every number.

## Optimizer
- **Interpreter and constant folder share one function, `ir::evalPure`**: folding can never disagree with execution. Trade-off: a bug in evalPure would affect both identically; the GPU and nvcc-twin paths are the independent check.
- **Constant folding never folds UB or doubtful cases**: div/rem by 0, `INT_MIN / -1`, out-of-range float→int, sqrt of a negative number; the run time reports them instead.
- **Never rewrite `!(a < b)` into `a >= b`** (wrong for floats with NaN); TinyCU does no comparison rewriting at all (comment in constfold.cpp, test `ConstFold.FloatCompareIsNeverInverted`).
- **Float identities limited to exact ones** (`x + -0.0`, `x - 0.0`, `x * 1.0`, `x / 1.0`); `x + 0.0`, `x * 0.0`, `x - x` are int-only: NaN, inf and −0 break them.
- **Local (per-block) constant and copy propagation**: obviously correct on non-SSA IR (a redefinition kills the fact); global versions need reaching definitions and are future work.
- **DCE = "no use anywhere"**: the zero-initialization `mov %x, 0` is removed only when %x is never read, which satisfies "keep it unless provably written before every read".
- **Pipeline iterates to a fixpoint (max 10)**: one pass's output enables another's (fold → constant branch → unreachable block → dead definitions).
- **The PTX emitter declares only registers that still occur in the code**: DCE leaves unused entries in the vreg table; declaring them inflated the register count.
- **IR text parser exists only for tests**: optimizer unit tests are written as readable before/after IR text.
- **Optimizer impact is reported at PTX and SASS level as measured** (−4.2% PTX instructions, −1.5% SASS, ptxas registers unchanged at first measurement): ptxas performs the same cleanups, which is itself the interesting result.

## Runtime & kernel suite
- **Driver API for the C++ runner** (not the Runtime API): PTX produced at run time must be loaded as a module; the JIT log is available via `CU_JIT_ERROR_LOG_BUFFER`.
- **One JSON format for interpreter input/output and the GPU runner**: every execution path can be diffed with the same code.
- **CUDA twins use `unsigned` for wrap-around arithmetic**: signed overflow is UB in CUDA C++ but defined (wrapping) in TinyCU.
- **NVRTC twins compiled with `-fmad=false` for correctness tests, default flags (`-O3`, FMA) for the benchmark**: correctness needs bit-exactness; the benchmark compares against what users would actually get.
- **Kernel inputs are seeded from (kernel name, n)**: every run of the suite is reproducible.
- **NumPy references reproduce the kernel's operation order** (reduction tree, matmul k-loop, dot chunks): float results can be compared bit-exactly instead of with a tolerance.
- **GPU work runs locally on the RTX 3050 Laptop GPU (sm_86, WSL2 GPU passthrough) as well as on Colab**: the plan assumed no local GPU; the Colab notebook still exists for the T4 (sm_75) target.
- **Benchmark summaries are computed from the measured numbers, never written in advance**: the first local run showed TinyCU's naive matmul *faster* than NVRTC's, the opposite of the expectation; it stays reported as measured until explained from the SASS.

## Verification
- **The interpreter is the oracle and reports every UB as an error**: random tests must be UB-free, and a UB report means a generator bug, not a miscompile.
- **Interpreter run-time errors exit with code 4, still print the JSON, and include `error.kind`**: the fuzz harness sorts failures (oob, div0, div_overflow, bad_cvt, barrier_divergence, step_limit, uninit_shared, bad_args) without parsing text.
- **Float data crosses process boundaries as bit patterns**: comparisons are bit-exact; NaN/−0/subnormals survive JSON.
- **Float fuzz mode produces NaN/inf from run-time values** (`sqrt` of negative inputs, division by zero inputs, ±1e30 / subnormal edge inputs): exercises the NaN comparison and min/max paths end to end.
- **Any two NaNs compare equal in differential testing; everything else (including −0.0) is bit-exact**: NaN payloads are not part of TinyCU's semantics (x86: 0xFFC00000, NVIDIA: 0x7FFFFFFF).
- **Generator is UB-free by construction and the interpreter proves it per run**: a UB report from the reference path is classified as a generator bug (`INTERP_ERROR`), never as a miscompile.
- **CUDA twins wrap int `+ - *` through `unsigned` helpers and cast `threadIdx.x` to int**: signed overflow is UB in CUDA C++; CUDA's builtin indices are unsigned.
- **Csmith-style checksum of all top-level variables as the final store**: any miscompiled value becomes observable (detection of planted bug 1 rose 9 → 31, bug 2 0 → 55 per 2,000 programs).
- **Twins compiled with NVRTC directly (`cupy.cuda.nvrtc`), not `cupy.RawKernel`**: CuPy appends `-ftz=true` to every NVRTC compilation (found by fuzzing, BUGLOG H2).
- **Reducer works on the generator AST, accepts only strictly smaller programs (AST size, |literals|), and keeps the same failure label**: valid candidates by construction, no cycles (BUGLOG H3).
- **Reduction is reported in source lines (includes the fixed 7-line kernel wrapper) and in AST nodes**: both measured; lines are the primary number.
- **Planted bugs live behind a hidden `--inject-bug=N` flag in the real passes**: validates the fuzzer against realistic defects without forking the compiler.
- **GPU fuzzing runs CPU paths in a process pool and GPU paths in the main process**: one CUDA context; no fork-after-CUDA problems.

## Tooling
- **nlohmann/json 3.11.3 via FetchContent from the single-header release tarball, pinned by SHA256**: no full-repo clone, reproducible builds.
- **`--kernel <name>`; without it `--interp` on a multi-kernel file fails and lists the kernel names**: PTX emission still outputs all kernels in one module.
- **`--no-verify-each`**: lets you time the compiler without the verifier.
- **All edits to repo text files are done inside WSL (UTF-8)**: Windows Python defaulted to cp1252 and corrupted a file once.
