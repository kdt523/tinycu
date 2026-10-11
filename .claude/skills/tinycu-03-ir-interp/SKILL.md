---
name: tinycu-03-ir-interp
description: Phase 3 of TinyCU — design the three-address IR, lower the typed AST into basic blocks/CFG, IR verifier, printer, and a CPU interpreter that emulates the GPU grid (the reference oracle).
---

# Phase 3 — Three-address IR, lowering, verifier, interpreter

## Goal
A typed, non-SSA three-address IR organised in basic blocks; a CPU interpreter that runs it
over a simulated grid and produces the **reference output** used by every later test.

## 1. IR design (document in docs/IR.md)
```
Function  { name, params[], blocks[] (entry = blocks[0]), vreg types }
BasicBlock{ label, insts[], terminator }  — last inst is exactly one terminator
Value     = VReg(id, type) | ImmI32(v) | ImmF32(v) | ImmBool(v)
```
Types: i1 (pred), i32, f32, ptr.global, ptr.shared, i64 (addresses).

Instruction set (`enum class Op`):
| Group | Ops |
|---|---|
| Arith | add sub mul div rem neg  (typed i32 / f32) |
| Logic | and.i1 or.i1 not.i1 |
| Compare | cmp.{eq,ne,lt,le,gt,ge} → i1 (i32 / f32) |
| Convert | cvt.f32.i32, cvt.i32.f32 (truncate toward zero) |
| Builtins | min max abs sqrt |
| Move | mov dst, src |
| Memory | addr dst, base, idx, elemSize (→ i64 address); load.{global,shared}; store.{global,shared} |
| Special | sreg dst, {tid,ctaid,ntid,nctaid}; param dst, index; barrier |
| Terminators | br L; condbr p, Lt, Lf; ret |

Text format (`--emit=ir`, deterministic):
```
kernel saxpy(f32 %a, ptr %x, ptr %y, i32 %n) {
entry:
  %0:i32 = sreg ctaid
  %1:i32 = sreg ntid
  %2:i32 = mul %0, %1
  ...
  %5:i1 = cmp.lt %i, %n
  condbr %5, if.then.1, if.end.1
...
}
```

## 2. Lowering (`src/ir/lower.cpp`)
- Each local/param Symbol → one fixed named vreg (`%i`, `%a`); temporaries numbered.
  Assignment = `mov`. (Non-SSA by design: simple, and PTX accepts it.)
- **Short-circuit** `&&`/`||` with control flow (needed for `if (i < n && x[i] > 0)` safety).
- `if/else`, `while`, `for` → standard block patterns (`for.cond`, `for.body`, `for.inc`,
  `for.end`); `break`→end, `continue`→inc (for) / cond (while). Keep a loop stack.
- `return` → `br exit`; single `exit` block with `ret`.
- Index `p[i]` → `addr t, p, i, 4` then load/store.
- Compound assign `a[i] += e` evaluates the address once.
- Unique, deterministic labels: `<kind>.<counter>`.

## 3. Verifier (`src/ir/verifier.cpp`) — run after lowering and after every pass
Every block ends with exactly one terminator and none in the middle; branch targets exist;
operand types match op; every vreg has one declared type; entry has no predecessors;
all used vregs are defined on some path (simple check). Failure = internal error with dump.
CLI: `--verify-each` (default ON in Debug).

## 4. CFG utilities
predecessors/successors, reverse post-order, reachable set. Unit-test them.

## 5. Interpreter (`src/interp/interp.cpp`) — THE ORACLE
- `tinycu k.tc --interp --grid G --block B --args args.json` → writes outputs JSON.
- args.json: `{"params":[{"type":"float","value":2.0},{"type":"float*","data":[...]},...]}`
- Memory model: each pointer param = separate buffer; out-of-bounds access → **error
  report** (kernel, thread id, index) not UB. Shared memory per block.
- Execution: for each block, run threads. Without barriers: run each thread to completion
  sequentially. With `barrier`: run all threads of the block in lockstep **phases** (each
  thread runs until barrier/ret; then next phase). Detect barrier divergence → error.
- Exact semantics: i32 wraps (compute in uint32_t); div/rem by 0 → error; f32 ops in `float`
  with `-ffp-contract=off`; `cvt.i32.f32` of NaN/out-of-range → error (defined-behaviour
  check); sqrt via `std::sqrt(float)`.
- Step limit (default 10M instructions per thread) → "possible infinite loop" error.

## Tests
- Golden `*.ir` for all programs.
- Interpreter unit tests: vecadd, saxpy, relu, loop sum, nested if, break/continue,
  short-circuit guarding OOB, wrap-around `2147483647 + 1`, shared-memory reduction.
- Verifier negative tests (hand-built bad IR must be rejected).

## Definition of Done
- [ ] `--emit=ir` works for all valid programs; verifier passes on all
- [ ] Interpreter gives correct results for ≥10 kernels (checked against numpy in pytest)
- [ ] Tag `v0.3`

## Interview notes
Three-address code, quadruples; basic blocks & leaders algorithm; CFG; why SSA exists
(and why we skipped it); short-circuit evaluation as control flow; interpreter as test oracle.
