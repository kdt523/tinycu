"""PTX / SASS statistics for every test program.

    python -m tcharness.ptxstats [--opt 0|1] [--sm sm_75] [--out results/ptx_stats.md] [files...]

For each kernel: PTX instruction count (from tinycu), registers / spills / shared memory
(from `ptxas -v`), SASS instruction count (from `cuobjdump -sass`), and the ELF sections of
the cubin (`cuobjdump -elf`). Needs the CUDA toolkit (no GPU).
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from tcharness.tinycu import REPO_ROOT, compile as tc_compile

DECL_RE = re.compile(r"^\s*(\.reg|\.param|\.shared|\.visible|\.version|\.target|\.address_size)")


@dataclass
class KernelStats:
    file: str
    kernel: str
    ptx_instrs: int
    regs: int
    spill_stores: int
    spill_loads: int
    smem: int
    sass_instrs: int


def count_ptx_instructions(ptx: str) -> dict[str, int]:
    """Instructions per kernel: lines ending in ';' that are not declarations."""
    counts: dict[str, int] = {}
    kernel = None
    for line in ptx.splitlines():
        m = re.match(r"\.visible \.entry (\w+)\(", line)
        if m:
            kernel = m.group(1)
            counts[kernel] = 0
            continue
        code = line.split("//", 1)[0].strip()
        if kernel and code.endswith(";") and not DECL_RE.match(code):
            counts[kernel] += 1
    return counts


def find_tool(name: str) -> str:
    path = shutil.which(name) or (f"/usr/local/cuda/bin/{name}"
                                  if Path(f"/usr/local/cuda/bin/{name}").exists() else None)
    if not path:
        sys.exit(f"error: {name} not found (install the CUDA toolkit or add it to PATH)")
    return path


def ptxas_stats(ptx_path: Path, cubin: Path, sm: str) -> dict[str, dict[str, int]]:
    r = subprocess.run([find_tool("ptxas"), f"-arch={sm}", "-v", str(ptx_path), "-o", str(cubin)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ptxas rejected {ptx_path}:\n{r.stderr}")
    stats: dict[str, dict[str, int]] = {}
    kernel = None
    for line in r.stderr.splitlines():
        m = re.search(r"Compiling entry function '(\w+)'", line)
        if m:
            kernel = m.group(1)
            stats[kernel] = {"regs": 0, "spill_stores": 0, "spill_loads": 0, "smem": 0}
        m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m and kernel:
            stats[kernel]["spill_stores"] = int(m.group(1))
            stats[kernel]["spill_loads"] = int(m.group(2))
        m = re.search(r"Used (\d+) registers", line)
        if m and kernel:
            stats[kernel]["regs"] = int(m.group(1))
            s = re.search(r"(\d+) bytes smem", line)
            stats[kernel]["smem"] = int(s.group(1)) if s else 0
    return stats


def run_tool(args: list[str]) -> str:
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed ({r.returncode}):
{r.stderr}")
    return r.stdout


def sass_counts(cubin: Path) -> dict[str, int]:
    # cuobjdump -sass needs nvdisasm from the CUDA toolkit.
    out = run_tool([find_tool("cuobjdump"), "-sass", str(cubin)])
    counts: dict[str, int] = {}
    kernel = None
    for line in out.splitlines():
        m = re.search(r"Function : (\w+)", line)
        if m:
            kernel = m.group(1)
            counts[kernel] = 0
            continue
        m = re.match(r"^\s*/\*([0-9a-f]{4})\*/\s+(.*?);", line)
        if not (kernel and m):
            continue
        addr, text = int(m.group(1), 16), m.group(2)
        # Skip alignment padding: NOPs and the trailing `BRA <own address>` trap.
        if re.search(r"\bNOP\b", text):
            continue
        b = re.search(r"\bBRA\s+0x([0-9a-f]+)", text)
        if b and int(b.group(1), 16) == addr:
            continue
        counts[kernel] += 1
    return counts


def elf_sections(cubin: Path) -> list[str]:
    out = run_tool([find_tool("cuobjdump"), "-elf", str(cubin)])
    names = re.findall(r"(\.(?:nv\.\S+|text\.\S+|shstrtab|strtab|symtab))", out)
    seen: list[str] = []
    for n in names:
        if n not in seen:
            seen.append(n)
    return seen


def default_files() -> list[Path]:
    files = []
    for d in ("tests/golden", "tests/kernels"):
        files += sorted((REPO_ROOT / d).glob("*.tc"))
    return files


def collect(files: list[Path], opt: int, sm: str) -> tuple[list[KernelStats], dict[str, list[str]]]:
    rows: list[KernelStats] = []
    sections: dict[str, list[str]] = {}
    with tempfile.TemporaryDirectory() as d:
        for tc in files:
            ptx = tc_compile(tc, emit="ptx", opt=opt, extra=[f"--sm={sm}"])
            ptx_path = Path(d) / (tc.stem + ".ptx")
            cubin = ptx_path.with_suffix(".cubin")
            ptx_path.write_text(ptx)
            pstats = ptxas_stats(ptx_path, cubin, sm)
            sass = sass_counts(cubin)
            sections[tc.name] = elf_sections(cubin)
            for kernel, n in count_ptx_instructions(ptx).items():
                s = pstats.get(kernel, {})
                rows.append(KernelStats(tc.name, kernel, n, s.get("regs", 0), s.get("spill_stores", 0),
                                        s.get("spill_loads", 0), s.get("smem", 0), sass.get(kernel, 0)))
    return rows, sections


def to_markdown(rows: list[KernelStats], opt: int, sm: str, sections: dict[str, list[str]]) -> str:
    out = [f"# PTX / SASS statistics (-O{opt}, {sm})", "",
           "Generated by `python -m tcharness.ptxstats`; do not edit by hand.", "",
           "| file | kernel | PTX instrs | regs | spill st/ld (B) | smem (B) | SASS instrs |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        out.append(f"| {r.file} | {r.kernel} | {r.ptx_instrs} | {r.regs} | "
                   f"{r.spill_stores}/{r.spill_loads} | {r.smem} | {r.sass_instrs} |")
    total_ptx = sum(r.ptx_instrs for r in rows)
    total_sass = sum(r.sass_instrs for r in rows)
    out += ["", f"Totals: {len(rows)} kernels, {total_ptx} PTX instructions, "
                f"{total_sass} SASS instructions, max {max((r.regs for r in rows), default=0)} registers.",
            ""]
    if sections:
        first = next(iter(sections))
        out += [f"cubin ELF sections (`cuobjdump -elf`, {first}): " +
                ", ".join(f"`{s}`" for s in sections[first]), ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--opt", type=int, default=0, choices=(0, 1))
    ap.add_argument("--sm", default="sm_75")
    ap.add_argument("--out", type=Path, help="also write the markdown table to this file")
    args = ap.parse_args(argv)
    rows, sections = collect(args.files or default_files(), args.opt, args.sm)
    md = to_markdown(rows, args.opt, args.sm, sections)
    print(md)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
