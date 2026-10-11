# TinyCU

A mini GPU compiler: a small CUDA-like kernel language (`.tc`) compiled to NVIDIA PTX,
with an IR interpreter oracle and a differential-testing / fuzzing verification suite.

> Work in progress — see [docs/STATUS.md](docs/STATUS.md).

```sh
cmake -B build -G Ninja && cmake --build build && ctest --test-dir build
python3 -m venv .venv && . .venv/bin/activate && pip install -e harness && pytest
```
