---
name: tinycu-05-gpu-runtime
description: Phase 5 of TinyCU — execute generated PTX on a real NVIDIA GPU via a C++ CUDA Driver API runner and a Python/CuPy harness, build the kernel test suite with CUDA twins, and the Colab notebook.
---

# Phase 5 — GPU runtime, kernel suite, Colab

## Goal
`pytest -m gpu` on Colab T4 runs ~20 kernels end-to-end (tinycu → PTX → GPU) and every
result matches the interpreter and NumPy.

## 1. C++ runner `tinycu-run` (`src/runtime/tinycu_run.cpp`, only if CUDA found)
```
tinycu-run k.ptx --kernel saxpy --grid G --block B --args args.json --out out.json [--time]
```
Driver API flow: `cuInit → cuDeviceGet → cuCtxCreate (or primary ctx) → cuModuleLoadDataEx`
(capture JIT error log via `CU_JIT_ERROR_LOG_BUFFER`) `→ cuModuleGetFunction → cuMemAlloc /
cuMemcpyHtoD → cuLaunchKernel → cuCtxSynchronize → cuMemcpyDtoH`. Every call wrapped in
`CHECK_CU()` printing `cuGetErrorName`. `--time` uses cuEvent for kernel ms (median of 20).
Minimal JSON reader/writer (hand-written or nlohmann/json via FetchContent).

## 2. Python harness `tcharness`
- `compile(tc_path, opt) -> ptx_str` (subprocess `tinycu`).
- `interp(tc_path, spec) -> outputs` (subprocess `tinycu --interp`).
- `run_gpu(ptx, spec) -> outputs` using **CuPy**: `cupy.RawModule(path=ptx_file)` →
  `get_function(name)` → launch with `(grid,), (block,), args` (np.int32/np.float32 scalars,
  cupy arrays). Fallback backend: `tinycu-run` subprocess.
- `run_nvcc_twin(cu_path, spec)` → `cupy.RawKernel(code, name, options=("-fmad=false",))`
  (NVRTC) for differential testing.
- Comparison: ints exact; floats bit-exact by default (`.rn` + `-fmad=false` + no contraction),
  fallback ULP tolerance flag for `sqrt`/`div` vs NVRTC only if needed — record any case needing it.

## 3. Kernel suite `tests/kernels/` (each: `name.tc`, `name.cu`, `name.json`)
spec json: `{"kernel":"saxpy","grid":"ceil(n/256)","block":256,"n":[1,255,256,1000,100003],
"params":[...generators: "rand_f32", "rand_i32(-100,100)", "zeros", "scalar:2.5", "n"],
"outputs":["y"]}`
Required kernels (≥20): vecadd_i32, vecadd_f32, saxpy, scale, relu, leaky_relu, clamp,
abs_diff, square, polynomial (horner), int_hash (wraparound), branchy (nested if/else),
loop_sum (for), while_collatz (bounded), break_continue, grid_stride_loop, dot_partial,
matmul_naive (one thread per output), softmax_row_max_sum (per-row loops), transpose,
**reduce_shared** (shared + `__syncthreads`, stretch), stencil_1d.
Sizes include non-multiples of block size (tests the `if (i < n)` guard).

## 4. Colab notebook `notebooks/run_on_colab.ipynb`
Cells: check `nvidia-smi`; `git clone`; install cmake/ninja; build (CUDA toolkit already on
Colab); `pip install -e harness[gpu]`; `pytest -m gpu -v`; run `tcharness.ptxstats` +
`tcharness.metrics`; display PTX of saxpy side by side with nvcc’s PTX; save results/ to Drive.

## 5. Performance sanity (not a goal, but a talking point)
`python -m tcharness.bench`: tinycu-O1 vs NVRTC(-O3) kernel time for vecadd/saxpy/matmul
at n=2^24 — report honestly in results (expect NVRTC faster on matmul; explain why:
no loop unrolling, no `ld.global.nc`, no FMA).

## Definition of Done
- [ ] ≥20 kernels: GPU == interpreter == numpy == nvcc twin (logged in results/)
- [ ] Colab notebook runs top-to-bottom without edits
- [ ] C++ runner works with JIT error log printing
- [ ] Tag `v0.5`

## Interview notes
CUDA execution model (grid/block/warp of 32/SIMT); memory hierarchy (global, shared, regs,
L1/L2); coalescing; why `__syncthreads` must not be in divergent code; driver API vs runtime
API; module loading & JIT; what ptxas does (register allocation, scheduling, SASS).
