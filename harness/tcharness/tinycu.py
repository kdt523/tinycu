"""Thin Python wrapper around the `tinycu` binary (compile / interpret).

Float data crosses the process boundary as raw bit patterns, so results are bit-exact.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


class TinycuError(RuntimeError):
    """Raised when tinycu fails; `returncode` tells which stage (see tinycu --help)."""

    def __init__(self, msg: str, returncode: int, stderr: str):
        super().__init__(msg)
        self.returncode = returncode
        self.stderr = stderr


def find_tinycu() -> Path:
    """$TINYCU, else the first existing build/{,-debug,-release}/tinycu in the repo."""
    env = os.environ.get("TINYCU")
    if env:
        return Path(env)
    for d in ("build", "build-debug", "build-release"):
        p = REPO_ROOT / d / "tinycu"
        if p.exists():
            return p
    raise FileNotFoundError("tinycu binary not found; build it or set $TINYCU")


def _source_path(src: str | Path, tmp: Path) -> Path:
    """Accept a path to a .tc file or the source text itself."""
    if isinstance(src, Path) or (isinstance(src, str) and src.endswith(".tc") and "\n" not in src):
        return Path(src)
    p = tmp / "kernel.tc"
    p.write_text(src)
    return p


def compile(src: str | Path, emit: str = "ptx", opt: int = 0, extra: list[str] | None = None) -> str:
    """Run tinycu and return its output text (--emit=ast|ir|ptx)."""
    with tempfile.TemporaryDirectory() as d:
        path = _source_path(src, Path(d))
        cmd = [str(find_tinycu()), str(path), f"--emit={emit}", f"-O{opt}", *(extra or [])]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise TinycuError(f"tinycu failed ({r.returncode}): {r.stderr}", r.returncode, r.stderr)
        return r.stdout


def _arg_json(a) -> dict:
    if isinstance(a, np.ndarray):
        if a.dtype == np.float32:
            return {"type": "float*", "bits": a.view(np.uint32).tolist()}
        if a.dtype == np.int32:
            return {"type": "int*", "data": a.tolist()}
        raise TypeError(f"unsupported array dtype {a.dtype} (use float32 or int32)")
    if isinstance(a, (np.float32, float)):
        return {"type": "float", "bits": int(np.float32(a).view(np.uint32))}
    if isinstance(a, (np.int32, int)):
        return {"type": "int", "value": int(a)}
    raise TypeError(f"unsupported argument {a!r}")


@dataclass
class InterpResult:
    ok: bool
    error: str  # human-readable message ("" when ok)
    error_kind: str  # oob | div0 | div_overflow | bad_cvt | barrier_divergence | step_limit |
    #                  uninit_shared | bad_args ("" when ok)
    outputs: list  # per parameter: np.ndarray for pointers, scalar otherwise
    steps: int


def interp(src: str | Path, args: list, grid: int, block: int, kernel: str | None = None,
           opt: int = 0) -> InterpResult:
    """Run a kernel in the interpreter. Arrays in `args` are not modified; results come back
    in `InterpResult.outputs` (same order as the parameters)."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        path = _source_path(src, tmp)
        args_path = tmp / "args.json"
        args_path.write_text(json.dumps({"params": [_arg_json(a) for a in args]}))
        cmd = [str(find_tinycu()), str(path), "--interp", "--grid", str(grid), "--block",
               str(block), "--args", str(args_path), f"-O{opt}"]
        if kernel:
            cmd += ["--kernel", kernel]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode not in (0, 4):  # 4 = kernel runtime error (still has JSON)
            raise TinycuError(f"tinycu failed ({r.returncode}): {r.stderr}", r.returncode, r.stderr)
        doc = json.loads(r.stdout)
    outputs = []
    for p in doc["params"]:
        if p["type"] == "float*":
            outputs.append(np.array(p["bits"], dtype=np.uint32).view(np.float32))
        elif p["type"] == "int*":
            outputs.append(np.array(p["data"], dtype=np.int32))
        elif p["type"] == "float":
            outputs.append(np.uint32(p["bits"]).view(np.float32))
        else:
            outputs.append(np.int32(p["value"]))
    err = doc.get("error") or {}
    return InterpResult(doc["status"] == "ok", err.get("text", ""), err.get("kind", ""), outputs,
                        doc["steps"])
