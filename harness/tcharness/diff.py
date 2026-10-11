"""Differential oracle for one generated program.

Paths:  A = interpreter -O0 (reference)   B = interpreter -O1
        C = GPU -O0 (TinyCU PTX)          D = GPU -O1          E = NVRTC twin (-fmad=false)
Without a GPU only A and B run (plus ptxas acceptance when ptxas is installed).

Result classes:
  OK               every path agrees with A
  MISMATCH(A-X)    path X differs from A (or fails where A succeeded) -> compiler bug candidate
  CRASH            tinycu exited with code 3 (internal error, e.g. verifier failure)
  PTXAS_REJECT     ptxas refused TinyCU's PTX
  TIMEOUT          step limit / wall-clock timeout
  INTERP_ERROR     the reference itself hit UB -> a *generator* bug
  COMPILE_ERROR    tinycu rejected the program -> a *generator* bug

Float comparison: bit-exact, except that any two NaNs compare equal (NaN payloads are not
specified by TinyCU; x86 produces 0xFFC00000, NVIDIA GPUs the canonical 0x7FFFFFFF).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tcharness import suite
from tcharness.tinycu import find_tinycu

TIMEOUT_S = 60


@dataclass
class DiffResult:
    status: str  # OK | MISMATCH | CRASH | PTXAS_REJECT | TIMEOUT | INTERP_ERROR | COMPILE_ERROR
    pair: str = ""  # e.g. "A-B" for MISMATCH
    detail: str = ""
    paths: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"MISMATCH({self.pair})" if self.status == "MISMATCH" else self.status

    def to_json(self) -> dict:
        return {"status": self.status, "label": self.label, "pair": self.pair, "detail": self.detail,
                "paths": self.paths}


def same(a: np.ndarray, b: np.ndarray) -> str | None:
    """None if equal under the comparison policy, else a description of the first difference."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return f"shape/dtype {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"
    if a.dtype == np.float32:
        diff = (a.view(np.uint32) != b.view(np.uint32)) & ~(np.isnan(a) & np.isnan(b))
    else:
        diff = a != b
    idx = np.nonzero(diff)[0]
    if idx.size:
        k = int(idx[0])
        return f"[{k}] {a[k]!r} vs {b[k]!r} ({idx.size} elements differ)"
    return None


def compare_outputs(ref: dict, got: dict) -> str | None:
    for name, r in ref.items():
        d = same(r, got[name])
        if d:
            return f"{name}{d}"
    return None


def find_ptxas() -> str | None:
    return shutil.which("ptxas") or ("/usr/local/cuda/bin/ptxas"
                                     if Path("/usr/local/cuda/bin/ptxas").exists() else None)


def _tinycu(args: list[str], timeout: int = TIMEOUT_S) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run([str(find_tinycu()), *args], capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def _interp(spec: suite.Spec, n: int, args: list, opt: int, extra: list[str], tmp: Path):
    """Returns (outputs | None, error_kind, message)."""
    from tcharness.tinycu import _arg_json  # noqa: PLC0415
    aj = tmp / f"args{opt}.json"
    aj.write_text(json.dumps({"params": [_arg_json(a) for a in args]}))
    r = _tinycu([str(spec.tc), "--interp", "--grid", str(spec.grid_for(n)), "--block", str(spec.block),
                 "--args", str(aj), f"-O{opt}", *extra])
    if r is None:
        return None, "timeout", "wall-clock timeout"
    if r.returncode == 3:
        return None, "crash", r.stderr.strip()[-500:]
    if r.returncode not in (0, 4):
        return None, "compile", r.stderr.strip()[-500:]
    doc = json.loads(r.stdout)
    if doc["status"] != "ok":
        return None, doc["error"]["kind"], doc["error"]["text"]
    values = []
    for p in doc["params"]:
        if p["type"] == "float*":
            values.append(np.array(p["bits"], np.uint32).view(np.float32))
        elif p["type"] == "int*":
            values.append(np.array(p["data"], np.int32))
        else:
            values.append(None)
    return suite.outputs_of(spec, values), "", ""


def check_cpu(prog_dir: Path, extra: list[str] | None = None, ptxas: bool = True) -> tuple[DiffResult, dict]:
    """Paths A and B (+ ptxas acceptance). Returns the result and A's outputs."""
    extra = extra or []
    spec = suite.load_spec(prog_dir / "k.json")
    n = spec.sizes[0]
    args = suite.make_inputs(spec, n)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # Compile both levels first: internal errors and ptxas rejections.
        ptx_tool = find_ptxas() if ptxas else None
        for opt in (0, 1):
            r = _tinycu([str(spec.tc), f"-O{opt}", "-o", str(tmp / f"k{opt}.ptx"), *extra])
            if r is None:
                return DiffResult("TIMEOUT", detail=f"compile -O{opt} timed out"), {}
            if r.returncode == 3:
                return DiffResult("CRASH", detail=f"-O{opt}: {r.stderr.strip()[-800:]}"), {}
            if r.returncode != 0:
                return DiffResult("COMPILE_ERROR", detail=f"-O{opt}: {r.stderr.strip()[-800:]}"), {}
            if ptx_tool:
                p = subprocess.run([ptx_tool, "-arch=sm_75", str(tmp / f"k{opt}.ptx"), "-o",
                                    str(tmp / f"k{opt}.cubin")], capture_output=True, text=True)
                if p.returncode != 0:
                    return DiffResult("PTXAS_REJECT", detail=f"-O{opt}: {p.stderr.strip()[-800:]}"), {}
        a, ka, ma = _interp(spec, n, args, 0, extra, tmp)
        if a is None:
            if ka in ("step_limit", "timeout"):
                return DiffResult("TIMEOUT", detail=ma), {}
            if ka == "crash":
                return DiffResult("CRASH", detail=ma), {}
            return DiffResult("INTERP_ERROR", detail=f"{ka}: {ma}"), {}
        b, kb, mb = _interp(spec, n, args, 1, extra, tmp)
        if b is None:
            if kb == "crash":
                return DiffResult("CRASH", detail=mb), {}
            return DiffResult("MISMATCH", "A-B", f"B failed where A succeeded: {kb}: {mb}", ["A", "B"]), a
        d = compare_outputs(a, b)
        if d:
            return DiffResult("MISMATCH", "A-B", d, ["A", "B"]), a
        return DiffResult("OK", paths=["A", "B"] + (["ptxas"] if ptx_tool else [])), a


def check_gpu(prog_dir: Path, ref: dict, extra: list[str] | None = None) -> DiffResult:
    """Paths C, D, E against A's outputs `ref`. Run in a single process (CUDA context)."""
    spec = suite.load_spec(prog_dir / "k.json")
    n = spec.sizes[0]
    args = suite.make_inputs(spec, n)
    runs = {"C": lambda: _gpu(spec, n, args, 0, extra), "D": lambda: _gpu(spec, n, args, 1, extra),
            "E": lambda: suite.run_nvrtc(spec, n, args)}
    for path, fn in runs.items():
        try:
            got = fn()
        except Exception as e:  # noqa: BLE001
            return DiffResult("MISMATCH", f"A-{path}", f"{path} failed: {type(e).__name__}: {e}"[:800],
                              ["A", path])
        d = compare_outputs(ref, got)
        if d:
            return DiffResult("MISMATCH", f"A-{path}", d, ["A", path])
    return DiffResult("OK", paths=["A", "B", "C", "D", "E"])


def _gpu(spec, n, args, opt, extra):
    import cupy as cp  # noqa: PLC0415
    from tcharness.tinycu import compile as tc_compile  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as d:
        ptx = Path(d) / "k.ptx"
        ptx.write_text(tc_compile(spec.tc, emit="ptx", opt=opt, extra=extra or []))
        kernel = cp.RawModule(path=str(ptx)).get_function(spec.name)
        return suite._launch_cupy(kernel, spec, n, args)
