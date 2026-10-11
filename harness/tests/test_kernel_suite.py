"""The kernel suite (tests/kernels): interpreter, GPU, NVRTC twin and C++ runner vs NumPy.

Non-GPU tests run everywhere; `-m gpu` runs the GPU paths (Colab T4 or any CUDA machine).
"""
import shutil
import subprocess

import pytest

from tcharness import suite

SPECS = suite.load_specs()
CASES = [(s, n) for s in SPECS for n in s.sizes]
IDS = [f"{s.name}-n{n}" for s, n in CASES]
OPT_CASES = [(s, n, o) for s, n in CASES for o in (0, 1)]
OPT_IDS = [f"{s.name}-n{n}-O{o}" for s, n, o in OPT_CASES]

REQUIRED = {
    "vecadd_i32", "vecadd_f32", "saxpy", "scale", "relu", "leaky_relu", "clamp", "abs_diff",
    "square", "polynomial", "int_hash", "branchy", "loop_sum", "while_collatz", "break_continue",
    "grid_stride_loop", "dot_partial", "matmul_naive", "softmax_row_max_sum", "transpose",
    "reduce_shared", "stencil_1d",
}


def test_suite_is_complete():
    names = {s.name for s in SPECS}
    assert len(names) >= 20
    assert REQUIRED <= names, REQUIRED - names
    for s in SPECS:
        assert s.cu.exists(), f"missing CUDA twin {s.cu}"
        assert s.name in suite.REFERENCES, f"missing NumPy reference for {s.name}"
        # Sizes must include a non-multiple of the block size (tests the i < n guard).
        assert any(n % s.block for n in s.sizes), s.name


@pytest.mark.parametrize("spec,n,opt", OPT_CASES, ids=OPT_IDS)
def test_interp_matches_numpy(spec, n, opt):
    # -O1 must not change any result: the optimizer is checked against the same oracle.
    r = suite.run_case(spec, n, ["interp"], opt)
    assert r.paths["interp"] == "ok", r.paths["interp"]


@pytest.mark.skipif(shutil.which("nvcc") is None and not
                    __import__("pathlib").Path("/usr/local/cuda/bin/nvcc").exists(),
                    reason="nvcc not installed")
@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_cuda_twin_compiles(spec, tmp_path):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    r = subprocess.run([nvcc, "-ptx", "-arch=sm_75", "-fmad=false", str(spec.cu), "-o",
                        str(tmp_path / "twin.ptx")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.gpu
@pytest.mark.parametrize("spec,n,opt", OPT_CASES, ids=OPT_IDS)
def test_gpu_and_nvrtc_match(spec, n, opt):
    r = suite.run_case(spec, n, ["interp", "gpu", "nvrtc"], opt)
    assert all(v == "ok" for v in r.paths.values()), r.paths


@pytest.mark.gpu
@pytest.mark.skipif(suite.find_runner() is None, reason="tinycu-run not built")
@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_cpp_runner_matches(spec):
    n = max(spec.sizes[:-1] or spec.sizes)  # a mid-size case keeps this quick
    args = suite.make_inputs(spec, n)
    diff = suite.compare(suite.run_gpu_runner(spec, n, args), suite.run_numpy(spec, n, args))
    assert diff is None, diff
