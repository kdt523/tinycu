"""Automatic test-case reduction + optimizer pass bisection for a failing fuzz program.

    python -m tcharness.reduce fuzz_out/<seed> [--budget 3000] [--gpu]

Reads <dir>/ast.json and <dir>/result.json (written by tcharness.fuzz), then greedily applies
AST transformations (delta debugging), keeping a change only if the program still fails with
the *same* result label (e.g. MISMATCH(A-B)). Every candidate goes through the full oracle, so
invalid programs (sema errors, UB in the reference) are rejected automatically.

Transformations, tried largest-first:
  * delete chunks of statements (halves, quarters, ... single statements)
  * replace `if` with its then/else body, `for` with one copy of its body (`int k = 0;` first)
  * replace an expression by a same-typed sub-expression or a constant (0, 1, 0.0, 1.0)
  * shrink integer literals (0, 1, halve)
Then *pass bisection*: re-run the reduced program with --disable-pass=<p> for every -O1 pass;
a pass whose removal makes the failure disappear is reported as the culprit.

Writes <dir>/reduced/ (k.tc, k.cu, k.json, ast.json) and <dir>/reduce.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

from tcharness import diff, gen

PASSES = ["constfold", "copyprop", "simplifycfg", "dce"]


# --------------------------------------------------------------------------- AST helpers
def expr_type(e: list) -> str:
    k = e[0]
    if k in ("int", "bvar"):
        return "int"
    if k == "float":
        return "float"
    if k in ("cmp", "and", "or", "not"):
        return "bool"
    if k == "cast" or k == "sel":
        return e[1]
    return e[2]  # var, load, bin, un, call


def constants(ty: str) -> list[list]:
    if ty == "int":
        return [["int", 0], ["int", 1]]
    if ty == "float":
        return [["float", 0.0], ["float", 1.0]]
    return [["cmp", "==", "int", ["int", 0], ["int", 0]], ["cmp", "!=", "int", ["int", 0], ["int", 0]]]


def child_exprs(e: list) -> list[list]:
    k = e[0]
    if k == "bin":
        return [e[3], e[4]]
    if k == "un":
        return [e[3]]
    if k == "call":
        return list(e[3])
    if k == "cast":
        return [e[2]]
    if k == "sel":
        return [e[2], e[3], e[4]]
    if k in ("and", "or"):
        return [e[1], e[2]]
    if k == "not":
        return [e[1]]
    if k == "cmp":
        return [e[3], e[4]]
    return []


def size(node) -> int:
    return 1 + sum(size(x) for x in node if isinstance(x, list)) if isinstance(node, list) else 0


def measure(prog: dict) -> tuple[int, float]:
    """Strictly decreasing measure for accepted changes: (AST size, sum of |literals|).
    Without it the reducer can cycle forever (e.g. replacing 0 by 1 and 1 by 0)."""
    total = 0.0

    def lits(node):
        nonlocal total
        if isinstance(node, list):
            if node and node[0] in ("int", "float") and len(node) == 2:
                total += abs(node[1])
            for x in node:
                lits(x)

    def loops(node):
        nonlocal total
        if isinstance(node, list):
            if node and node[0] == "for":
                total += node[2]  # trip counts count as "literal size"
            for x in node:
                loops(x)

    lits(prog["stmts"])
    loops(prog["stmts"])
    return size(prog["stmts"]), total


def lines(prog: dict) -> int:
    return sum(1 for line in gen.to_tc(prog).splitlines() if line.strip())


# Locations: a statement list is addressed by a path of keys into the program dict/lists.
def stmt_lists(prog: dict):
    """Yields (container, key) pairs such that container[key] is a statement list."""
    def walk(lst_owner, key):
        yield lst_owner, key
        for s in lst_owner[key]:
            if s[0] == "if":
                yield from walk(s, 2)
                yield from walk(s, 3)
            elif s[0] == "for":
                yield from walk(s, 3)
    yield from walk(prog, "stmts")


def expr_slots(prog: dict):
    """Yields (container, index) pairs such that container[index] is an expression."""
    def in_expr(e):
        k = e[0]
        idxs = {"bin": [3, 4], "un": [3], "cast": [2], "sel": [2, 3, 4], "and": [1, 2], "not": [1],
                "or": [1, 2], "cmp": [3, 4]}.get(k, [])
        for i in idxs:
            yield e, i
            yield from in_expr(e[i])
        if k == "call":
            for i in range(len(e[3])):
                yield e[3], i
                yield from in_expr(e[3][i])

    for owner, key in stmt_lists(prog):
        for s in owner[key]:
            pos = {"decl": [3], "assign": [4], "store": [3], "if": [1]}.get(s[0], [])
            for i in pos:
                yield s, i
                yield from in_expr(s[i])


# --------------------------------------------------------------------------- candidates
def candidates(prog: dict):
    """Yields reduced copies of `prog`, biggest reductions first."""
    # 1. statement chunks
    for li, (owner, key) in enumerate(list(stmt_lists(prog))):
        n = len(owner[key])
        chunk = n
        while chunk >= 1:
            for start in range(0, n, chunk):
                c = copy.deepcopy(prog)
                tgt = list(stmt_lists(c))[li]
                del tgt[0][tgt[1]][start:start + chunk]
                yield c
            chunk //= 2
    # 2. unwrap if / for
    for li, (owner, key) in enumerate(list(stmt_lists(prog))):
        for si, s in enumerate(owner[key]):
            if s[0] == "if":
                for branch in (2, 3):
                    c = copy.deepcopy(prog)
                    o, k = list(stmt_lists(c))[li]
                    o[k][si:si + 1] = copy.deepcopy(s[branch])
                    yield c
            elif s[0] == "for":
                c = copy.deepcopy(prog)
                o, k = list(stmt_lists(c))[li]
                o[k][si:si + 1] = [["decl", "int", s[1], ["int", 0]]] + copy.deepcopy(s[3])
                yield c
                for count in sorted({0, 1, 2, s[2] // 2} - {s[2]}):  # fewer iterations
                    if count < s[2]:
                        c = copy.deepcopy(prog)
                        o, k = list(stmt_lists(c))[li]
                        o[k][si][2] = count
                        yield c
    # 3. expression simplification
    slots = list(expr_slots(prog))
    for si, (owner, idx) in enumerate(slots):
        e = owner[idx]
        ty = expr_type(e)
        repl = [ch for ch in child_exprs(e) if expr_type(ch) == ty] + \
               [k for k in constants(ty) if k != e]
        if e[0] == "int" and e[1] not in (0, 1):
            repl.append(["int", e[1] // 2])
        for r in repl:
            if size(r) >= size(e) and r[0] not in ("int", "float", "cmp"):
                continue
            c = copy.deepcopy(prog)
            o, i = list(expr_slots(c))[si]
            o[i] = copy.deepcopy(r)
            yield c


# --------------------------------------------------------------------------- oracle
class Oracle:
    def __init__(self, seed: int, label: str, extra: list[str], gpu: bool):
        self.seed, self.label, self.extra, self.gpu = seed, label, extra, gpu
        self.tests = 0
        self.outcomes: dict[str, int] = {}  # result label of every candidate tried

    def result(self, prog: dict, extra: list[str] | None = None) -> diff.DiffResult:
        self.tests += 1
        with tempfile.TemporaryDirectory() as d:
            gen.write(prog, self.seed, Path(d))
            res, ref = diff.check_cpu(Path(d), extra if extra is not None else self.extra,
                                      ptxas=self.label == "PTXAS_REJECT")
            if res.status == "OK" and self.gpu:
                res = diff.check_gpu(Path(d), ref, extra if extra is not None else self.extra)
            return res

    def interesting(self, prog: dict) -> bool:
        label = self.result(prog).label
        self.outcomes[label] = self.outcomes.get(label, 0) + 1
        return label == self.label


def reduce_program(prog: dict, oracle: Oracle, budget: int) -> dict:
    """Greedy reduction with a cursor: after an accepted change the search continues at the
    same candidate position instead of restarting (earlier candidates just failed). Rounds
    repeat from the start until a whole round makes no progress."""
    current = prog
    while oracle.tests < budget:
        progress = False
        cursor = 0
        while oracle.tests < budget:
            accepted = False
            m = measure(current)
            for j, cand in enumerate(candidates(current)):
                if j < cursor or measure(cand) >= m:
                    continue
                if oracle.tests >= budget:
                    break
                if oracle.interesting(cand):
                    current, cursor, accepted, progress = cand, j, True, True
                    break
            if not accepted:
                break
        if not progress:
            break
    return current


def bisect_passes(prog: dict, oracle: Oracle) -> list[str]:
    culprits = []
    if "B" not in oracle.label and "D" not in oracle.label:
        return culprits  # not an optimizer (-O1) failure
    for p in PASSES:
        if oracle.result(prog, oracle.extra + [f"--disable-pass={p}"]).status == "OK":
            culprits.append(p)
    return culprits


def run(prog_dir: Path, budget: int = 3000, gpu: bool = False, quiet: bool = False) -> dict:
    ast = json.loads((prog_dir / "ast.json").read_text())
    res = json.loads((prog_dir / "result.json").read_text())
    seed = ast.pop("seed")
    oracle = Oracle(seed, res["label"], res.get("extra", []), gpu)
    if not oracle.interesting(ast):
        raise SystemExit(f"{prog_dir}: the failure ({res['label']}) does not reproduce")
    before = lines(ast)
    nodes_before = size(ast["stmts"])
    reduced = reduce_program(ast, oracle, budget)
    after = lines(reduced)
    nodes_after = size(reduced["stmts"])
    culprits = bisect_passes(reduced, oracle)
    gen.write(reduced, seed, prog_dir / "reduced")
    summary = {"seed": seed, "label": res["label"], "lines_before": before, "lines_after": after,
               "reduction": round(1 - after / before, 3) if before else 0.0,
               "ast_nodes_before": nodes_before, "ast_nodes_after": nodes_after,
               "ast_reduction": round(1 - nodes_after / nodes_before, 3) if nodes_before else 0.0,
               "tests": oracle.tests,
               "culprit_passes": culprits, "candidate_outcomes": oracle.outcomes}
    (prog_dir / "reduce.json").write_text(json.dumps(summary, indent=1) + "\n")
    if not quiet:
        print(gen.to_tc(reduced))
        print(f"{res['label']}: {before} -> {after} lines ({summary['reduction'] * 100:.0f}% smaller) "
              f"in {oracle.tests} oracle runs; culprit pass(es): {', '.join(culprits) or 'n/a'}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--budget", type=int, default=3000, help="max oracle runs")
    ap.add_argument("--gpu", action="store_true", help="include GPU paths in the oracle")
    a = ap.parse_args(argv)
    run(a.dir, a.budget, a.gpu)
    return 0


if __name__ == "__main__":
    sys.exit(main())
