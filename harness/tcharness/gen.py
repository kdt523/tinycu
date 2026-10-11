"""Random UB-free TinyCU kernel generator (seeded, reproducible).

    python -m tcharness.gen --seed 42 [--float] [--out dir]     # writes k.tc, k.cu, k.json, ast.json

Every program has the same signature

    kernel fuzz(int* out, float* fout, int* in0, float* fin, int n, int s0, float f0)

and the body `int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) { ...random... }`.

UB-free by construction (the interpreter would report any UB as a generator bug):
  * `/` and `%` divisors are literals in [1, 16] or `(abs(e % 1000) % 13) + 1` (always 1..13;
    `e % 1000` first so abs never sees INT_MIN); INT_MIN / -1 is therefore impossible.
  * Memory is only indexed with the guarded `i`.
  * float -> int casts are clamped first: (int)min(max(e, -1000000.0), 1000000.0); PTX min/max
    return the non-NaN operand, so NaN and +-inf map into range too.
  * Every variable is initialized; for loops have constant trip counts (<= 8), and their loop
    variable is never assigned in the body.
Int-only mode (default) is bit-exact on every path. --float mode adds float arithmetic,
including NaN/inf from run-time values (sqrt of negatives, division by zero inputs).

The program is a small JSON-serializable AST (nested lists), so the reducer can transform it.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

PARAMS = "int* out, float* fout, int* in0, float* fin, int n, int s0, float f0"
INT_OPS = ["+", "-", "*", "/", "%"]
FLOAT_OPS = ["+", "-", "*", "/"]
CMP_OPS = ["<", "<=", ">", ">=", "==", "!="]


@dataclass
class Knobs:
    max_stmts: int = 8
    max_depth: int = 3  # nesting of if/for
    max_expr_depth: int = 4
    max_vars: int = 6
    float_mode: bool = False


class Gen:
    def __init__(self, seed: int, knobs: Knobs):
        self.r = random.Random(seed)
        self.k = knobs
        self.counter = 0
        self.scopes: list[list[tuple[str, str]]] = []  # (name, type)
        self.readonly: set[str] = set()

    # ----------------------------------------------------------------- helpers
    def fresh(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}{self.counter}"

    def visible(self, ty: str, writable: bool = False) -> list[str]:
        out = []
        for scope in self.scopes:
            out += [n for n, t in scope if t == ty and not (writable and n in self.readonly)]
        return out

    def int_lit(self) -> list:
        c = self.r.random()
        if c < 0.5:
            v = self.r.randint(0, 20)
        elif c < 0.8:
            v = self.r.randint(0, 100000)
        else:  # boundary values (never 2147483648: not a valid literal)
            v = self.r.choice([2147483647, 2147483646, 65536, 46341, 1103515245])
        return ["int", v] if self.r.random() < 0.8 else ["un", "-", "int", ["int", v]]

    def float_lit(self) -> list:
        v = self.r.choice([0.0, 0.5, 1.0, 2.0, 0.25, 3.0, 10.0, 100.0, 1e-3, 1.5, 7.25, 1e30])
        return ["float", v] if self.r.random() < 0.8 else ["un", "-", "float", ["float", v]]

    # ----------------------------------------------------------------- expressions
    def expr(self, ty: str, depth: int) -> list:
        if ty == "bool":
            return self.cond(depth)
        leaf = depth <= 0 or self.r.random() < 0.25
        if leaf:
            return self.leaf(ty)
        c = self.r.random()
        if ty == "int":
            if c < 0.45:
                op = self.r.choice(INT_OPS)
                if op in ("/", "%"):
                    return ["bin", op, "int", self.expr("int", depth - 1), self.divisor(depth - 1)]
                return ["bin", op, "int", self.expr("int", depth - 1), self.expr("int", depth - 1)]
            if c < 0.55:
                return ["un", "-", "int", self.expr("int", depth - 1)]
            if c < 0.75:
                fn = self.r.choice(["min", "max", "abs"])
                args = [self.expr("int", depth - 1)] + ([self.expr("int", depth - 1)] if fn != "abs" else [])
                return ["call", fn, "int", args]
            if c < 0.85 and self.k.float_mode:
                return self.safe_f2i(self.expr("float", depth - 1))
            return ["sel", "int", self.cond(depth - 1), self.expr("int", depth - 1), self.expr("int", depth - 1)]
        # float
        if c < 0.45:
            op = self.r.choice(FLOAT_OPS)
            return ["bin", op, "float", self.expr("float", depth - 1), self.expr("float", depth - 1)]
        if c < 0.55:
            return ["un", "-", "float", self.expr("float", depth - 1)]
        if c < 0.8:
            fn = self.r.choice(["min", "max", "abs", "sqrt"])
            args = [self.expr("float", depth - 1)] + ([self.expr("float", depth - 1)] if fn in ("min", "max") else [])
            return ["call", fn, "float", args]
        return ["cast", "float", self.expr("int", depth - 1)]

    def leaf(self, ty: str) -> list:
        c = self.r.random()
        if ty == "int":
            vars_ = self.visible("int")
            if c < 0.45 and vars_:
                return ["var", self.r.choice(vars_), "int"]
            if c < 0.6:
                return ["load", "in0", "int"]
            if c < 0.67:
                return ["var", self.r.choice(["s0", "i", "n"]), "int"]
            if c < 0.72:
                return ["bvar", self.r.choice(["threadIdx", "blockIdx", "blockDim", "gridDim"])]
            return self.int_lit()
        vars_ = self.visible("float")
        if c < 0.45 and vars_:
            return ["var", self.r.choice(vars_), "float"]
        if c < 0.65:
            return ["load", "fin", "float"]
        if c < 0.72:
            return ["var", "f0", "float"]
        return self.float_lit()

    def divisor(self, depth: int) -> list:
        if depth <= 0 or self.r.random() < 0.6:
            return ["int", self.r.randint(1, 16)]
        # (abs(e % 1000) % 13) + 1  in [1, 13]
        e = self.expr("int", depth - 1)
        return ["bin", "+", "int",
                ["bin", "%", "int", ["call", "abs", "int", [["bin", "%", "int", e, ["int", 1000]]]], ["int", 13]],
                ["int", 1]]

    def safe_f2i(self, e: list) -> list:
        clamped = ["call", "min", "float", [["call", "max", "float", [e, ["float", -1000000.0]]],
                                             ["float", 1000000.0]]]
        return ["cast", "int", clamped]

    def cond(self, depth: int) -> list:
        c = self.r.random()
        if depth > 0 and c < 0.25:
            return [self.r.choice(["and", "or"]), self.cond(depth - 1), self.cond(depth - 1)]
        if depth > 0 and c < 0.35:
            return ["not", self.cond(depth - 1)]
        ty = "float" if self.k.float_mode and self.r.random() < 0.4 else "int"
        return ["cmp", self.r.choice(CMP_OPS), ty, self.expr(ty, max(0, depth - 1)), self.expr(ty, max(0, depth - 1))]

    # ----------------------------------------------------------------- statements
    def block(self, depth: int, n: int) -> list:
        self.scopes.append([])
        stmts: list = []
        for _ in range(n):
            s = self.stmt(depth)
            stmts += s[1] if s[0] == "seq" else [s]  # "seq" = several statements in a row
        self.scopes.pop()
        return stmts

    def stmt(self, depth: int) -> list:
        c = self.r.random()
        nvars = sum(len(s) for s in self.scopes)
        ed = self.k.max_expr_depth
        if c < 0.3 and nvars < self.k.max_vars + 2:
            ty = "float" if self.k.float_mode and self.r.random() < 0.4 else "int"
            name = self.fresh("v" if ty == "int" else "f")
            e = self.expr(ty, self.r.randint(1, ed))
            self.scopes[-1].append((name, ty))
            return ["decl", ty, name, e]
        if c < 0.55:
            ty = "float" if self.k.float_mode and self.r.random() < 0.4 else "int"
            targets = self.visible(ty, writable=True)
            # Plain copies and swaps (`t = a; a = b; b = t;`): the patterns that break copy
            # propagation when a copy is not invalidated after its source is redefined.
            if len(targets) >= 2 and self.r.random() < 0.35:
                a, b = self.r.sample(targets, 2)
                if self.r.random() < 0.5:
                    return ["assign", a, "=", ty, ["var", b, ty]]
                tmp = self.fresh("t")
                self.scopes[-1].append((tmp, ty))
                return ["seq", [["decl", ty, tmp, ["var", a, ty]],
                                ["assign", a, "=", ty, ["var", b, ty]],
                                ["assign", b, "=", ty, ["var", tmp, ty]]]]
            if targets:
                op = self.r.choice(["=", "=", "+=", "-=", "*="])
                return ["assign", self.r.choice(targets), op, ty, self.expr(ty, self.r.randint(1, ed))]
        if c < 0.68 and depth < self.k.max_depth:
            then = self.block(depth + 1, self.r.randint(1, 3))
            els = self.block(depth + 1, self.r.randint(1, 3)) if self.r.random() < 0.5 else []
            return ["if", self.cond(2), then, els]
        if c < 0.78 and depth < self.k.max_depth:
            var = self.fresh("k")
            count = self.r.randint(0, 8)
            self.scopes.append([(var, "int")])
            self.readonly.add(var)
            body = self.block(depth + 1, self.r.randint(1, 3))
            self.scopes.pop()
            return ["for", var, count, body]
        if c < 0.9 or not self.k.float_mode:
            return ["store", "out", "int", self.expr("int", self.r.randint(1, ed))]
        return ["store", "fout", "float", self.expr("float", self.r.randint(1, ed))]

    def program(self) -> dict:
        self.scopes = [[("i", "int")], []]
        self.readonly = {"i"}
        stmts: list = []
        for _ in range(self.r.randint(2, self.k.max_stmts)):
            s = self.stmt(0)
            stmts += s[1] if s[0] == "seq" else [s]
        # Csmith-style checksum: the final stores combine *every* top-level variable, so a
        # miscompiled value anywhere is observable in the output.
        top = self.scopes[-1]
        int_sum: list = self.expr("int", 2)
        for name, ty in top:
            if ty == "int":
                int_sum = ["bin", "+", "int", int_sum, ["var", name, "int"]]
        stmts.append(["store", "out", "int", int_sum])
        if self.k.float_mode:
            f_sum: list = self.expr("float", 2)
            for name, ty in top:
                if ty == "float":
                    f_sum = ["bin", "+", "float", f_sum, ["var", name, "float"]]
            stmts.append(["store", "fout", "float", f_sum])
        self.scopes = []
        return {"float_mode": self.k.float_mode, "stmts": stmts}


# ===================================================================== printers
def _lit_f(v: float) -> str:
    s = repr(float(v))
    if "e" in s and "." not in s.split("e")[0]:
        s = s.replace("e", ".0e")
    return s


def tc_expr(e: list) -> str:
    k = e[0]
    if k == "int":
        return str(e[1])
    if k == "float":
        return _lit_f(e[1])
    if k == "var":
        return e[1]
    if k == "bvar":
        return e[1] + ".x"
    if k == "load":
        return f"{e[1]}[i]"
    if k == "bin":
        return f"({tc_expr(e[3])} {e[1]} {tc_expr(e[4])})"
    if k == "un":
        return f"(-{tc_expr(e[3])})"
    if k == "call":
        return f"{e[1]}({', '.join(tc_expr(a) for a in e[3])})"
    if k == "cast":
        return f"(({e[1]}){tc_expr(e[2])})"
    if k == "cmp":
        return f"({tc_expr(e[3])} {e[1]} {tc_expr(e[4])})"
    if k in ("and", "or"):
        return f"({tc_expr(e[1])} {'&&' if k == 'and' else '||'} {tc_expr(e[2])})"
    if k == "not":
        return f"(!{tc_expr(e[1])})"
    if k == "sel":  # TinyCU has no ?: -- printed as a statement by tc_stmt via a temp
        raise ValueError("sel must be lowered by the statement printer")
    raise ValueError(k)


class TcPrinter:
    """TinyCU text. `sel` (c ? a : b) has no expression form, so it is hoisted into
    `int t = b; if (c) { t = a; }` before the statement that uses it."""

    def __init__(self):
        self.tmp = 0

    def hoist(self, e: list, pre: list[str], ind: str) -> list:
        if not isinstance(e, list):
            return e
        if e[0] == "sel":
            self.tmp += 1
            name = f"sel{self.tmp}"
            c = self.hoist(e[2], pre, ind)
            a = self.hoist(e[3], pre, ind)
            b = self.hoist(e[4], pre, ind)
            pre.append(f"{ind}{e[1]} {name} = {tc_expr(b)};")
            pre.append(f"{ind}if ({tc_expr(c)}) {{ {name} = {tc_expr(a)}; }}")
            return ["var", name, e[1]]
        return [self.hoist(x, pre, ind) if isinstance(x, list) else x for x in e]

    def stmts(self, ss: list, ind: str) -> list[str]:
        out = []
        for s in ss:
            out += self.stmt(s, ind)
        return out

    def stmt(self, s: list, ind: str) -> list[str]:
        pre: list[str] = []
        k = s[0]
        if k == "decl":
            e = self.hoist(s[3], pre, ind)
            return pre + [f"{ind}{s[1]} {s[2]} = {tc_expr(e)};"]
        if k == "assign":
            e = self.hoist(s[4], pre, ind)
            return pre + [f"{ind}{s[1]} {s[2]} {tc_expr(e)};"]
        if k == "store":
            e = self.hoist(s[3], pre, ind)
            return pre + [f"{ind}{s[1]}[i] = {tc_expr(e)};"]
        if k == "if":
            c = self.hoist(s[1], pre, ind)
            lines = pre + [f"{ind}if ({tc_expr(c)}) {{"] + self.stmts(s[2], ind + "  ")
            if s[3]:
                lines += [f"{ind}}} else {{"] + self.stmts(s[3], ind + "  ")
            return lines + [f"{ind}}}"]
        if k == "for":
            return ([f"{ind}for (int {s[1]} = 0; {s[1]} < {s[2]}; {s[1]}++) {{"]
                    + self.stmts(s[3], ind + "  ") + [f"{ind}}}"])
        raise ValueError(k)


def to_tc(prog: dict) -> str:
    p = TcPrinter()
    body = p.stmts(prog["stmts"], "    ")
    return ("// generated by tcharness.gen\n"
            f"kernel fuzz({PARAMS}) {{\n"
            "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "  if (i < n) {\n" + "\n".join(body) + "\n  }\n}\n")


# CUDA twin: same AST, int arithmetic through unsigned helpers (signed overflow is UB in C++).
CUDA_PRELUDE = """// generated by tcharness.gen (CUDA twin)
__device__ __forceinline__ int tc_add(int a, int b) { return (int)((unsigned)a + (unsigned)b); }
__device__ __forceinline__ int tc_sub(int a, int b) { return (int)((unsigned)a - (unsigned)b); }
__device__ __forceinline__ int tc_mul(int a, int b) { return (int)((unsigned)a * (unsigned)b); }
__device__ __forceinline__ int tc_neg(int a) { return (int)(0u - (unsigned)a); }
__device__ __forceinline__ int tc_abs(int a) { return a < 0 ? tc_neg(a) : a; }
"""


def cu_expr(e: list) -> str:
    k = e[0]
    if k == "int":
        return str(e[1])
    if k == "float":
        return _lit_f(e[1]) + "f"
    if k == "var":
        return e[1]
    if k == "bvar":  # CUDA's threadIdx.x etc. are unsigned; TinyCU's are int
        return f"((int){e[1]}.x)"
    if k == "load":
        return f"{e[1]}[i]"
    if k == "bin":
        a, b = cu_expr(e[3]), cu_expr(e[4])
        if e[2] == "int" and e[1] in ("+", "-", "*"):
            helper = {"+": "tc_add", "-": "tc_sub", "*": "tc_mul"}[e[1]]
            return f"{helper}({a}, {b})"
        return f"({a} {e[1]} {b})"
    if k == "un":
        return f"tc_neg({cu_expr(e[3])})" if e[2] == "int" else f"(-{cu_expr(e[3])})"
    if k == "call":
        args = [cu_expr(a) for a in e[3]]
        if e[2] == "float":
            fn = {"min": "fminf", "max": "fmaxf", "abs": "fabsf", "sqrt": "sqrtf"}[e[1]]
        else:
            fn = {"min": "min", "max": "max", "abs": "tc_abs"}[e[1]]
        return f"{fn}({', '.join(args)})"
    if k == "cast":
        return f"(({e[1]}){cu_expr(e[2])})"
    if k == "cmp":
        return f"({cu_expr(e[3])} {e[1]} {cu_expr(e[4])})"
    if k in ("and", "or"):
        return f"({cu_expr(e[1])} {'&&' if k == 'and' else '||'} {cu_expr(e[2])})"
    if k == "not":
        return f"(!{cu_expr(e[1])})"
    if k == "sel":
        return f"({cu_expr(e[2])} ? {cu_expr(e[3])} : {cu_expr(e[4])})"
    raise ValueError(k)


def cu_stmt(s: list, ind: str) -> list[str]:
    k = s[0]
    if k == "decl":
        return [f"{ind}{s[1]} {s[2]} = {cu_expr(s[3])};"]
    if k == "assign":
        name, op, ty, e = s[1], s[2], s[3], cu_expr(s[4])
        if ty == "int" and op != "=":
            fn = {"+=": "tc_add", "-=": "tc_sub", "*=": "tc_mul"}[op]
            return [f"{ind}{name} = {fn}({name}, {e});"]
        return [f"{ind}{name} {op} {e};"]
    if k == "store":
        return [f"{ind}{s[1]}[i] = {cu_expr(s[3])};"]
    if k == "if":
        lines = [f"{ind}if ({cu_expr(s[1])}) {{"] + [x for t in s[2] for x in cu_stmt(t, ind + "  ")]
        if s[3]:
            lines += [f"{ind}}} else {{"] + [x for t in s[3] for x in cu_stmt(t, ind + "  ")]
        return lines + [f"{ind}}}"]
    if k == "for":
        return ([f"{ind}for (int {s[1]} = 0; {s[1]} < {s[2]}; {s[1]}++) {{"]
                + [x for t in s[3] for x in cu_stmt(t, ind + "  ")] + [f"{ind}}}"])
    raise ValueError(k)


def to_cu(prog: dict) -> str:
    body = [x for s in prog["stmts"] for x in cu_stmt(s, "    ")]
    return (CUDA_PRELUDE +
            'extern "C" __global__ void fuzz(int* out, float* fout, const int* in0, const float* fin, '
            "int n, int s0, float f0) {\n"
            "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "  if (i < n) {\n" + "\n".join(body) + "\n  }\n}\n")


def spec_for(seed: int, prog: dict) -> dict:
    r = random.Random(seed ^ 0x5EED)
    n = r.choice([1, 7, 31, 64, 100, 257])
    s0 = r.choice([0, 1, 3, -5, 1000, 2147483647])
    return {"kernel": "fuzz", "grid": "ceil(n/64)", "block": 64, "n": [n],
            "params": ["zeros", "zeros", "fuzz_i32", "fuzz_f32", "n", f"scalar:{s0}",
                       f"scalar:{r.choice([0.0, 1.5, -2.0, 0.1])}"],
            "outputs": ["out", "fout"] if prog["float_mode"] else ["out"]}


def generate(seed: int, float_mode: bool = False, knobs: Knobs | None = None) -> dict:
    k = knobs or Knobs()
    k.float_mode = float_mode
    return Gen(seed, k).program()


def write(prog: dict, seed: int, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "k.tc").write_text(to_tc(prog))
    (out / "k.cu").write_text(to_cu(prog))
    (out / "k.json").write_text(json.dumps(spec_for(seed, prog), indent=1) + "\n")
    (out / "ast.json").write_text(json.dumps({"seed": seed, **prog}) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--float", action="store_true", dest="float_mode")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    prog = generate(a.seed, a.float_mode)
    if a.out:
        write(prog, a.seed, a.out)
    print(to_tc(prog), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
