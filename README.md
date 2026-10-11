# TinyCU

A mini GPU compiler: a small CUDA-like kernel language (`.tc`) compiled to NVIDIA PTX,
with an IR interpreter oracle and a differential-testing / fuzzing verification suite.

```c
kernel saxpy(float a, float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}
```

## Quickstart (WSL2 Ubuntu or Linux)
```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build
python3 -m venv .venv && . .venv/bin/activate && pip install -e "harness[gpu]"
build/tinycu tests/kernels/saxpy.tc -o saxpy.ptx            # compile
python -m tcharness.suite --gpu                             # every kernel: numpy/interp/GPU/nvcc twin
```
No local GPU? Open `notebooks/run_on_colab.ipynb` in Colab (T4) and **Run all**.

## Exit codes
| Code | Meaning |
|---|---|
| 0 | success |
| 1 | compile error (diagnostics on stderr) |
| 2 | usage error (bad command line, unreadable file, bad args JSON) |
| 3 | internal compiler error (e.g. IR verifier failure) |
| 4 | kernel run-time error in the interpreter (`--interp`); the result JSON is still printed |

## Dependencies
- C++17 compiler, CMake ≥ 3.20, Ninja, Python ≥ 3.10 (NumPy, pytest)
- Fetched automatically by CMake: [nlohmann/json](https://github.com/nlohmann/json) 3.11.3
  (single-header release tarball, pinned by SHA256) and GoogleTest 1.14.0 (tests only)
- Optional: CUDA toolkit (`ptxas` for validation; GPU runner)

> Work in progress — see [docs/STATUS.md](docs/STATUS.md).

```sh
cmake -B build -G Ninja && cmake --build build && ctest --test-dir build
python3 -m venv .venv && . .venv/bin/activate && pip install -e harness && pytest
```
