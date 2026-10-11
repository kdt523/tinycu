"""Kernel suite runner: every kernel in tests/kernels through every available execution path.

    python -m tcharness.suite [--gpu] [--opt 0|1] [--kernels a,b] [--out results/kernel_suite.md]

Paths (all compared bit-exactly against the NumPy reference):
  numpy   independent reference (tcharness.references)       always
  interp  tinycu --interp (the oracle)                         always
  gpu     tinycu PTX launched with CuPy (or tinycu-run)        --gpu
  nvrtc   the .cu twin compiled by NVRTC with -fmad=false      --gpu

Spec format (tests/kernels/<name>.json):
  {"kernel": "saxpy", "grid": "ceil(n/256)", "block": 256, "n": [1, 255, ...],
   "params": ["scalar:2.5", "rand_f32", "rand_f32", "n"], "outputs": ["y"]}
Parameter generators (one per kernel parameter, typed by the .tc signature):
  rand_f32[(lo,hi)] / rand_i32[(lo,hi)]   random array (default size n)
  zeros / iota                            array of the parameter's element type
  fuzz_i32 / fuzz_f32                     random arrays rich in edge values (INT_MIN/MAX, 0, -0.0,
                                          huge, subnormal), used by the fuzzer
  <gen>[expr]                             array size from an expression in n, e.g. zeros[n*n]
  scalar:V                                scalar of the parameter's type
  n                                       the problem size (int)
Expressions may use n, ceil, min, max.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tcharness.references import REFERENCES
from tcharness.tinycu import REPO_ROOT, compile as tc_compile, interp as tc_interp

KERNEL_DIR = REPO_ROOT / "tests" / "kernels"
GEN_RE = re.compile(r"^(rand_f32|rand_i32|zeros|iota|fuzz_i32|fuzz_f32)(?:\(([^)]*)\))?(?:\[(.+)\])?$")
SIG_RE = re.compile(r"kernel\s+(\w+)\s*\(([^)]*)\)")


# --------------------------------------------------------------------------- specs
@dataclass
class Param:
    type: str  # int, float, int*, float*
    name: str

    @property
    def is_ptr(self) -> bool:
        return self.type.endswith("*")

    @property
    def dtype(self):
        return np.float32 if self.type.startswith("float") else np.int32


@dataclass
class Spec:
    name: str
    grid: str
    block: int
    sizes: list[int]
    gens: list[str]
    outputs: list[str]
    params: list[Param]
    tc: Path
    cu: Path

    def grid_for(self, n: int) -> int:
        return max(1, int(evaluate(self.grid, n)))


def evaluate(expr: str, n: int) -> float:
    return eval(expr, {"__builtins__": {}}, {"n": n, "ceil": math.ceil, "min": min, "max": max})


def parse_signature(tc_text: str, kernel: str) -> list[Param]:
    for m in SIG_RE.finditer(tc_text):
        if m.group(1) == kernel:
            params = []
            for p in m.group(2).split(","):
                p = p.strip()
                if p:
                    t, name = p.rsplit(None, 1)
                    params.append(Param(t.replace(" ", ""), name))
            return params
    raise ValueError(f"kernel {kernel} not found")


def load_specs(names: list[str] | None = None) -> list[Spec]:
    specs = []
    for js in sorted(KERNEL_DIR.glob("*.json")):
        d = json.loads(js.read_text())
        if names and d["kernel"] not in names:
            continue
        tc = js.with_suffix(".tc")
        params = parse_signature(tc.read_text(), d["kernel"])
        if len(params) != len(d["params"]):
            raise ValueError(f"{js.name}: {len(d['params'])} generators for {len(params)} params")
        specs.append(Spec(d["kernel"], str(d["grid"]), int(d["block"]), d["n"], d["params"],
                          d["outputs"], params, tc, js.with_suffix(".cu")))
    return specs


# --------------------------------------------------------------------------- inputs
def make_inputs(spec: Spec, n: int) -> list:
    """Deterministic inputs: the seed depends only on (kernel, n)."""
    rng = np.random.default_rng(zlib.crc32(spec.name.encode()) + n)
    args = []
    for gen, p in zip(spec.gens, spec.params):
        if gen == "n":
            args.append(np.int32(n))
        elif gen.startswith("scalar:"):
            v = gen.split(":", 1)[1]
            args.append(np.float32(float(v)) if p.dtype == np.float32 else np.int32(int(v)))
        else:
            m = GEN_RE.match(gen)
            if not m or not p.is_ptr:
                raise ValueError(f"{spec.name}: bad generator {gen!r} for {p.type} {p.name}")
            kind, rng_args, size_expr = m.groups()
            size = int(evaluate(size_expr, n)) if size_expr else n
            lo, hi = ((float(x) for x in rng_args.split(",")) if rng_args else (-10.0, 10.0))
            if kind == "rand_f32":
                arr = rng.uniform(lo, hi, size).astype(np.float32)
            elif kind == "rand_i32":
                arr = rng.integers(int(lo), int(hi), size, dtype=np.int64).astype(np.int32)
            elif kind == "fuzz_i32":
                arr = fuzz_ints(rng, size)
            elif kind == "fuzz_f32":
                arr = fuzz_floats(rng, size)
            elif kind == "zeros":
                arr = np.zeros(size, p.dtype)
            else:
                arr = np.arange(size).astype(p.dtype)
            args.append(arr)
    return args


INT_EDGES = np.array([0, 1, -1, 2, 7, 1000, -1000, 2**31 - 1, -2**31, 2**31 - 2, -2**31 + 1, 65536],
                     dtype=np.int64)
FLOAT_EDGES = np.array([0.0, -0.0, 1.0, -1.0, 0.5, 1e30, -1e30, 1e-40, -1e-40, 3.4e38, 1e-3],
                       dtype=np.float32)


def fuzz_ints(rng, size: int) -> np.ndarray:
    small = rng.integers(-1000, 1000, size)
    full = rng.integers(-2**31, 2**31, size)
    edge = INT_EDGES[rng.integers(0, len(INT_EDGES), size)]
    pick = rng.random(size)
    return np.where(pick < 0.6, small, np.where(pick < 0.8, full, edge)).astype(np.int32)


def fuzz_floats(rng, size: int) -> np.ndarray:
    normal = rng.uniform(-100, 100, size).astype(np.float32)
    wide = rng.uniform(-1e6, 1e6, size).astype(np.float32)
    edge = FLOAT_EDGES[rng.integers(0, len(FLOAT_EDGES), size)]
    pick = rng.random(size)
    return np.where(pick < 0.6, normal, np.where(pick < 0.75, wide, edge)).astype(np.float32)


def load_spec(js: Path) -> Spec:
    d = json.loads(js.read_text())
    tc = js.with_suffix(".tc")
    params = parse_signature(tc.read_text(), d["kernel"])
    if len(params) != len(d["params"]):
        raise ValueError(f"{js.name}: {len(d['params'])} generators for {len(params)} params")
    return Spec(d["kernel"], str(d["grid"]), int(d["block"]), d["n"], d["params"], d["outputs"],
                params, tc, js.with_suffix(".cu"))


def outputs_of(spec: Spec, values: list) -> dict[str, np.ndarray]:
    index = {p.name: k for k, p in enumerate(spec.params)}
    return {o: np.asarray(values[index[o]]) for o in spec.outputs}


# --------------------------------------------------------------------------- paths
def run_numpy(spec: Spec, n: int, args: list) -> dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in REFERENCES[spec.name](*[np.copy(a) for a in args]).items()}


def run_interp(spec: Spec, n: int, args: list, opt: int = 0) -> dict[str, np.ndarray]:
    r = tc_interp(spec.tc, args, spec.grid_for(n), spec.block, kernel=spec.name, opt=opt)
    if not r.ok:
        raise RuntimeError(f"interpreter error ({r.error_kind}): {r.error}")
    return outputs_of(spec, r.outputs)


def _cupy():
    import cupy  # noqa: PLC0415  (optional dependency: only for GPU runs)
    return cupy


def _launch_cupy(kernel, spec: Spec, n: int, args: list) -> dict[str, np.ndarray]:
    cp = _cupy()
    dev = [cp.asarray(a) if isinstance(a, np.ndarray) else a for a in args]
    kernel((spec.grid_for(n),), (spec.block,), tuple(dev))
    cp.cuda.Device().synchronize()
    host = [cp.asnumpy(d) if isinstance(d, cp.ndarray) else d for d in dev]
    return outputs_of(spec, host)


_ptx_cache: dict[tuple[str, str, int], object] = {}


def run_gpu(spec: Spec, n: int, args: list, opt: int = 0) -> dict[str, np.ndarray]:
    """tinycu PTX on the GPU via CuPy (falls back to the C++ tinycu-run runner)."""
    try:
        cp = _cupy()
    except ImportError:
        return run_gpu_runner(spec, n, args, opt)
    key = (str(spec.tc), spec.tc.read_text(), opt)
    if key not in _ptx_cache:
        ptx = tc_compile(spec.tc, emit="ptx", opt=opt)
        d = Path(tempfile.mkdtemp())
        (d / "k.ptx").write_text(ptx)
        _ptx_cache[key] = cp.RawModule(path=str(d / "k.ptx")).get_function(spec.name)
    return _launch_cupy(_ptx_cache[key], spec, n, args)


_nvrtc_cache: dict[tuple, object] = {}


def nvrtc_kernel(src: str, name: str, options: tuple[str, ...]):
    """Compile CUDA source with NVRTC *directly* and load the PTX.

    Not via cupy.RawKernel: CuPy always appends `-ftz=true` (flush subnormals to zero), which
    would silently give the twin different float semantics from nvcc's defaults and TinyCU.
    Cached by (source, options): every fuzz program is called `fuzz`.
    """
    key = (src, name, options)
    if key not in _nvrtc_cache:
        cp = _cupy()
        from cupy.cuda import nvrtc  # noqa: PLC0415
        arch = f"--gpu-architecture=compute_{cp.cuda.Device().compute_capability}"
        prog = nvrtc.createProgram(src, "twin.cu", [], [])
        try:
            nvrtc.compileProgram(prog, [arch, *options])
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"NVRTC failed: {e}\n{nvrtc.getProgramLog(prog)}") from e
        ptx = nvrtc.getPTX(prog)
        nvrtc.destroyProgram(prog)
        ptx = ptx.decode() if isinstance(ptx, bytes) else ptx
        d = Path(tempfile.mkdtemp())
        (d / "twin.ptx").write_text(ptx.rstrip("\x00"))  # NVRTC returns a NUL-terminated string
        _nvrtc_cache[key] = cp.RawModule(path=str(d / "twin.ptx")).get_function(name)
    return _nvrtc_cache[key]


def run_nvrtc(spec: Spec, n: int, args: list) -> dict[str, np.ndarray]:
    """The CUDA twin compiled by NVRTC with -fmad=false (no FMA contraction) and otherwise
    default (IEEE) float semantics: no flush-to-zero, precise division and sqrt."""
    kernel = nvrtc_kernel(spec.cu.read_text(), spec.name, ("-fmad=false",))
    return _launch_cupy(kernel, spec, n, args)


def find_runner() -> Path | None:
    for d in ("build", "build-release", "build-debug"):
        p = REPO_ROOT / d / "tinycu-run"
        if p.exists():
            return p
    return None


def run_gpu_runner(spec: Spec, n: int, args: list, opt: int = 0) -> dict[str, np.ndarray]:
    """Fallback GPU path: the C++ Driver API runner (tinycu-run)."""
    from tcharness.tinycu import _arg_json  # noqa: PLC0415
    runner = find_runner()
    if runner is None:
        raise RuntimeError("no GPU backend: install cupy or build tinycu-run")
    with tempfile.TemporaryDirectory() as d:
        ptx, aj, oj = Path(d) / "k.ptx", Path(d) / "args.json", Path(d) / "out.json"
        ptx.write_text(tc_compile(spec.tc, emit="ptx", opt=opt))
        aj.write_text(json.dumps({"params": [_arg_json(a) for a in args]}))
        subprocess.run([str(runner), str(ptx), "--kernel", spec.name, "--grid", str(spec.grid_for(n)),
                        "--block", str(spec.block), "--args", str(aj), "--out", str(oj)], check=True)
        doc = json.loads(oj.read_text())
    values = []
    for p in doc["params"]:
        if p["type"] == "float*":
            values.append(np.array(p["bits"], np.uint32).view(np.float32))
        elif p["type"] == "int*":
            values.append(np.array(p["data"], np.int32))
        else:
            values.append(None)
    return outputs_of(spec, values)


# --------------------------------------------------------------------------- comparison
def compare(got: dict[str, np.ndarray], want: dict[str, np.ndarray]) -> str | None:
    """None if bit-identical, else a description of the first difference."""
    for name, w in want.items():
        g = got.get(name)
        if g is None:
            return f"{name}: missing"
        w = np.asarray(w)
        g = np.asarray(g)
        if g.shape != w.shape or g.dtype != w.dtype:
            return f"{name}: shape/dtype {g.shape}/{g.dtype} != {w.shape}/{w.dtype}"
        gb = g.view(np.uint32) if g.dtype == np.float32 else g
        wb = w.view(np.uint32) if w.dtype == np.float32 else w
        diff = np.nonzero(gb != wb)[0]
        if diff.size:
            k = int(diff[0])
            return f"{name}[{k}]: got {g[k]!r}, want {w[k]!r} ({diff.size} differing elements)"
    return None


@dataclass
class CaseResult:
    kernel: str
    n: int
    paths: dict[str, str] = field(default_factory=dict)  # path -> "ok" | "MISMATCH ..." | "ERROR ..."


def run_case(spec: Spec, n: int, paths: list[str], opt: int = 0) -> CaseResult:
    args = make_inputs(spec, n)
    want = run_numpy(spec, n, args)
    res = CaseResult(spec.name, n)
    runners = {"interp": lambda: run_interp(spec, n, args, opt),
               "gpu": lambda: run_gpu(spec, n, args, opt),
               "nvrtc": lambda: run_nvrtc(spec, n, args)}
    for path in paths:
        try:
            diff = compare(runners[path](), want)
            res.paths[path] = "ok" if diff is None else f"MISMATCH {diff}"
        except Exception as e:  # noqa: BLE001  (report every failure in the table)
            res.paths[path] = f"ERROR {type(e).__name__}: {str(e).splitlines()[0][:200]}"
    return res


def gpu_name() -> str:
    try:
        cp = _cupy()
        return cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:  # noqa: BLE001
        if shutil.which("nvidia-smi"):
            return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.strip()
        return "none"


def to_markdown(results: list[CaseResult], paths: list[str], opt: int, seconds: float) -> str:
    kernels = sorted({r.kernel for r in results})
    ok_cases = sum(all(v == "ok" for v in r.paths.values()) for r in results)
    lines = [f"# Kernel suite results (-O{opt})", "",
             "Generated by `python -m tcharness.suite`; do not edit by hand.", "",
             f"- GPU: {gpu_name()}",
             f"- paths compared bit-exactly against the NumPy reference: numpy, {', '.join(paths)}",
             f"- {len(kernels)} kernels, {len(results)} cases, **{ok_cases}/{len(results)} cases "
             f"agree on every path**, {seconds:.1f} s", "",
             "| kernel | n | " + " | ".join(paths) + " |",
             "|---|---:|" + "---|" * len(paths)]
    for r in results:
        cells = [r.paths[p] if r.paths[p] == "ok" else f"**{r.paths[p]}**" for p in paths]
        lines.append(f"| {r.kernel} | {r.n} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", action="store_true", help="also run the gpu and nvrtc paths")
    ap.add_argument("--opt", type=int, default=0, choices=(0, 1))
    ap.add_argument("--kernels", help="comma-separated subset")
    ap.add_argument("--out", type=Path, help="write the markdown table here")
    args = ap.parse_args(argv)
    paths = ["interp"] + (["gpu", "nvrtc"] if args.gpu else [])
    t0 = time.time()
    results = []
    for spec in load_specs(args.kernels.split(",") if args.kernels else None):
        for n in spec.sizes:
            r = run_case(spec, n, paths, args.opt)
            results.append(r)
            status = "ok" if all(v == "ok" for v in r.paths.values()) else "FAIL"
            print(f"{status:4} {spec.name:22} n={n:<7} " +
                  " ".join(f"{p}={v.split()[0]}" for p, v in r.paths.items()), flush=True)
    md = to_markdown(results, paths, args.opt, time.time() - t0)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md)
    print(md)
    return 0 if all(all(v == "ok" for v in r.paths.values()) for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
