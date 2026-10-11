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

## Optimizer (to be honoured in phase 6)
- **Constant folding uses exactly the interpreter's semantics**: wrapping i32 via `uint32_t`, no folding of div/rem by 0 or `INT_MIN / -1`, f32 with `float` math, `!=` true on NaN. A unit test checks NaN `!=` folding against the interpreter.
- **Never rewrite `!(a < b)` into `a >= b` for floats** (they differ on NaN); fine for ints. Needs a comment in the code and a test.
- **No `x * 0 → 0` / `x - x → 0` for floats**: NaN, inf and −0 break these identities.

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
- **Phase 7 float fuzz mode must produce NaNs from run-time values** (e.g. `sqrt(-x)`, `z / z` with `z = 0.0` loaded from memory): exercises the NaN comparison path end to end.

## Tooling
- **nlohmann/json 3.11.3 via FetchContent from the single-header release tarball, pinned by SHA256**: no full-repo clone, reproducible builds.
- **`--kernel <name>`; without it `--interp` on a multi-kernel file fails and lists the kernel names**: PTX emission still outputs all kernels in one module.
- **`--no-verify-each`**: lets you time the compiler without the verifier.
- **All edits to repo text files are done inside WSL (UTF-8)**: Windows Python defaulted to cp1252 and corrupted a file once.
