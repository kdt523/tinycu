"""Aggregates every measured number into results/metrics.md (the ONLY source for README/resume).

    python -m tcharness.metrics [--out results/metrics.md] [--no-count-tests]

Inputs (each produced by its own tool; a missing input is reported as "not measured"):
  tests/kernels/*.json            kernel-suite size
  ctest -N / pytest --collect-only test counts (live)
  results/kernel_suite*.md        suite results (tcharness.suite)
  results/fuzz_summary.json       fuzzing (tcharness.fuzz --summary)
  results/fuzz_summary_gpu.json   GPU fuzzing (optional)
  results/bug_injection.json      planted bugs (tcharness.bug_injection)
  results/coverage.json           line coverage (tools/coverage.sh)
  results/opt_impact.json         -O1 effect (tcharness.optstats)
  docs/BUGLOG.md                  real bugs found (entries starting with "## ")
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from tcharness.tinycu import REPO_ROOT

RES = REPO_ROOT / "results"


def load(name: str):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def count_ctest() -> int | None:
    for d in ("build", "build-release", "build-debug"):
        if (REPO_ROOT / d / "CTestTestfile.cmake").exists():
            r = subprocess.run(["ctest", "--test-dir", str(REPO_ROOT / d), "-N"], capture_output=True, text=True)
            m = re.search(r"Total Tests: (\d+)", r.stdout)
            if m:
                return int(m.group(1))
    return None


def count_pytest() -> int | None:
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", ""],
                       capture_output=True, text=True, cwd=REPO_ROOT)
    m = re.search(r"(\d+) tests? collected", r.stdout) or re.search(r"(\d+) tests?", r.stdout.splitlines()[-1] if r.stdout else "")
    return int(m.group(1)) if m else None


def suite_line(name: str) -> str | None:
    p = RES / name
    if not p.exists():
        return None
    text = p.read_text()
    gpu = re.search(r"- GPU: (.*)", text)
    agree = re.search(r"\*\*(\d+)/(\d+) cases agree on every path\*\*", text)
    paths = re.search(r"paths compared bit-exactly against the NumPy reference: (.*)", text)
    if not agree:
        return None
    return f"{agree.group(1)}/{agree.group(2)} cases bit-exact on {paths.group(1) if paths else '?'} ({gpu.group(1) if gpu else '?'})"


def buglog_entries() -> tuple[int, int]:
    p = REPO_ROOT / "docs" / "BUGLOG.md"
    if not p.exists():
        return 0, 0
    text = p.read_text()
    compiler = text.split("## Harness")[0] if "## Harness" in text else text
    real = len(re.findall(r"^### ", compiler, re.M))
    fixed = len(re.findall(r"^\*\*Status:\*\* fixed", compiler, re.M))
    return real, fixed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=RES / "metrics.md")
    ap.add_argument("--no-count-tests", action="store_true")
    a = ap.parse_args(argv)

    kernels = len(list((REPO_ROOT / "tests" / "kernels").glob("*.json")))
    ctest = None if a.no_count_tests else count_ctest()
    pytest_n = None if a.no_count_tests else count_pytest()
    fuzz = load("fuzz_summary.json")
    fuzz_gpu = load("fuzz_summary_gpu.json")
    inj = load("bug_injection.json")
    cov = load("coverage.json")
    opt = load("opt_impact.json")
    bugs_found, bugs_fixed = buglog_entries()
    nm = "not measured"

    rows = [("Kernels in the GPU suite (each with a CUDA twin)", str(kernels))]
    rows.append(("Tests: ctest (unit + golden + error/warning)", str(ctest) if ctest is not None else nm))
    rows.append(("Tests: pytest (incl. GPU-marked)", str(pytest_n) if pytest_n is not None else nm))
    for name, label in (("kernel_suite.md", "Kernel suite -O0 (local GPU)"), ("kernel_suite_O1.md", "Kernel suite -O1 (local GPU)"),
                        ("kernel_suite_colab.md", "Kernel suite -O0 (Colab)")):
        line = suite_line(name)
        rows.append((label, line or nm))
    if fuzz:
        bad = {k: v for k, v in fuzz["counts"].items() if k != "OK"}
        rows.append(("Fuzz programs run (A vs B + ptxas)", f"{fuzz['programs']}"))
        rows.append(("Fuzz throughput", f"{fuzz['programs_per_min']:.0f} programs/min ({fuzz['jobs']} jobs)"))
        rows.append(("Open fuzz failures", ", ".join(f"{k}: {v}" for k, v in bad.items()) or "0"))
    else:
        rows.append(("Fuzz programs run", nm))
    if fuzz_gpu:
        bad = {k: v for k, v in fuzz_gpu["counts"].items() if k != "OK"}
        rows.append(("GPU fuzz programs (paths A–E)", f"{fuzz_gpu['programs']} "
                     f"({', '.join(f'{k}: {v}' for k, v in bad.items()) or '0 mismatches'})"))
    rows.append(("Compiler bugs found by the fuzzer / fixed (docs/BUGLOG.md)", f"{bugs_found} / {bugs_fixed}"))
    if inj:
        rows.append(("Injected miscompiles detected", f"{inj['detected']}/{inj['bugs']}"))
        rows.append(("Average reduction of failing programs (source lines)", f"{inj['avg_reduction'] * 100:.0f}% ({inj['reduced_total']} reductions)"))
        rows.append(("Average reduction of failing programs (AST nodes)", f"{inj['avg_ast_reduction'] * 100:.0f}%"))
        rows.append(("Pass bisection named the culprit", f"{inj['culprit_correct']}/{inj['reduced_total']}"))
    else:
        rows.append(("Injected miscompiles detected", nm))
    if cov:
        rows.append(("Line coverage (src/, ctest + fuzz)", f"{cov['line_percent']:.1f}%"))
        per = {}
        for f in cov["files"]:
            d = "/".join(f["filename"].split("/")[:2])
            c, t = per.get(d, (0, 0))
            per[d] = (c + f["line_covered"], t + f["line_total"])
        for d in ("src/opt", "src/backend"):
            if d in per and per[d][1]:
                rows.append((f"Line coverage {d}", f"{per[d][0] / per[d][1] * 100:.1f}%"))
    else:
        rows.append(("Line coverage", nm))
    if opt:
        t = opt["totals"]
        def pct(k):
            b, a2 = t[k + "0"], t[k + "1"]
            return f"{b} → {a2} ({(a2 - b) / b * 100:+.1f}%)" if b else "n/a"
        rows.append(("-O1 effect: PTX instructions", pct("p")))
        rows.append(("-O1 effect: PTX virtual registers", pct("v")))
        rows.append(("-O1 effect: ptxas registers (sum)", pct("r")))
        rows.append(("-O1 effect: SASS instructions", pct("s")))
    else:
        rows.append(("-O1 effect", nm))

    md = ["# TinyCU metrics", "",
          "Generated by `python -m tcharness.metrics`; **do not edit by hand**. Every number in the "
          "README and resume must come from this file.", "",
          "| metric | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rows]
    a.out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
