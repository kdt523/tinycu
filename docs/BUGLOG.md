# Bug log

Every genuine bug found by the verification infrastructure: how it was found, a minimal
reproducer, the root cause, and the fix. Entries under "Compiler" are bugs in TinyCU itself
(counted in `results/metrics.md`); "Harness" entries are bugs in the test infrastructure.
NVRTC/nvcc differences are not claimed as nvcc bugs; so far every one was explained by our
own setup.

## Compiler

No compiler bug has been found by the fuzzer so far (5,000+ programs on paths A–B with ptxas,
2,000 programs on paths A–E on a GPU; see `results/fuzz_summary*.json`).

Bugs found during development by other means (tests, measurement) are listed for
completeness but not counted as fuzzer findings:

- **PTX register declarations included DCE'd vregs** (found by `tcharness.optstats`, phase 6).
  The emitter numbered every entry of the vreg table, so `-O1` PTX declared registers for values
  the optimizer had deleted, inflating the register count. Fix: number only vregs that occur
  in the code (`src/backend/ptx_emitter.cpp`).
- **Panic-mode recovery cascaded on `if (x {`** (found by error tests, phase 1). Recovery
  stopped at the body's `}`, so the kernel's `}` was reported as unmatched. Fix: skip a balanced
  `{…}` while synchronizing.

## Harness

### H1 — NVRTC twin cache keyed by kernel name
- **Found by:** the first GPU fuzz run. 299/300 programs were `MISMATCH(A-E)`, but paths A–D
  all agreed.
- **Root cause:** `suite.run_nvrtc` cached the compiled twin by kernel name, and every
  generated program's kernel is called `fuzz`, so all programs ran the first program's twin.
- **Fix:** key the cache by (source, name, options). The same fix was applied to the PTX cache.
- **Lesson:** a fuzz harness must assume every program shares names.

### H2 — CuPy silently forces `-ftz=true` on NVRTC
- **Found by:** GPU fuzzing in float mode. 77/1000 programs were `MISMATCH(A-E)`, for example
  `fout[5]: 1e-40 (TinyCU, interpreter) vs 0.0 (twin)`.
- **Minimal reproducer:** a twin with `fout[i] = -fin[i];` and `fin = {-1e-40}`. TinyCU's PTX
  and nvcc (`-fmad=false`) give `1e-40`; the CuPy-compiled twin gives `0.0`.
- **Root cause:** `cupy/cuda/compiler.py` appends `-ftz=true` to every NVRTC compilation (flush
  subnormals to zero), and that can't be overridden through `RawKernel` (a second `-ftz` is
  rejected). The twin therefore had different float semantics from both nvcc's defaults and
  TinyCU. The kernel suite never noticed because its inputs contain no subnormals; the
  fuzzer's edge-value inputs do.
- **Fix:** compile twins with NVRTC directly (`cupy.cuda.nvrtc`, explicit options) and load the
  PTX as a module (`suite.nvrtc_kernel`). Result: 1000/1000 programs agree on A–E.
- **Lesson:** "the reference compiler" includes its flags. Check what a wrapper library adds.

### H3 — reducer cycled without shrinking
- **Found by:** `reduce.json` statistics (2,985 of 3,004 candidates "accepted", 18% reduction).
- **Root cause:** constant replacement could turn `0` into `1` and back. Each step was accepted
  as "still failing" but nothing got smaller.
- **Fix:** accept a candidate only if it strictly decreases (AST size, sum of |literals|).
