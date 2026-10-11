# TinyCU IR

A typed, **non-SSA** three-address IR organised in basic blocks. It is produced by lowering
the type-checked AST (`src/ir/lower.cpp`), checked by the verifier, executed by the
interpreter (the test oracle), optimized at `-O1`, and translated to PTX.

```
Function   { name, params[], shared[], blocks[] (entry = blocks[0]), vregs[] }
BasicBlock { label, insts[] }  — the last inst is the one and only terminator
Value      = VReg(id) | ImmI32 | ImmF32 | ImmBool | Shared(@array)
```

## Types
| IR type | Meaning |
|---|---|
| `i1` | predicate (comparison / logic result) |
| `i32` | int, two's-complement, wrapping |
| `f32` | IEEE binary32 |
| `ptr` | pointer parameter into global memory |
| `ptr.shared` | base of a shared array (`@name`) |
| `i64` | a computed byte address (`addr` result) |

Each vreg has exactly one type (stored in `Function::vregs`). Source variables map to one
**named** vreg (`%i`; duplicates are uniqued as `%i.1`, `%i.2`). Temporaries are numbered by their
vreg id (`%7`). Assignment is `mov`, so a named vreg may be defined many times. That is the
non-SSA part. PTX accepts this (it has unlimited virtual registers, and `ptxas` builds SSA
internally), so we don't need phi nodes or out-of-SSA translation.

## Instructions
| Group | Syntax | Notes |
|---|---|---|
| Arith | `%d:T = add/sub/mul/div/rem a, b`, `%d:T = neg a` | T = i32 or f32; `rem` is i32 only |
| Logic | `%d:i1 = and/or a, b`, `%d:i1 = not a` | |
| Compare | `%d:i1 = cmp.{eq,ne,lt,le,gt,ge} a, b` | operands i32 or f32 (`Inst::type`) |
| Convert | `%d:f32 = cvt.f32.i32 a`, `%d:i32 = cvt.i32.f32 a` | round-to-nearest-even / truncate toward zero |
| Builtins | `min max` (a, b), `abs` (a), `sqrt` (a, f32) | |
| Move | `%d:T = mov a` | |
| Memory | `%d:i64 = addr base, idx, 4`; `%d:T = load.{global,shared} addr`; `store.{global,shared} addr, v` | `base` is a `ptr` vreg or `@shared` |
| Special | `%d:i32 = sreg {tid,ctaid,ntid,nctaid}`; `%d:T = param N`; `barrier` | |
| Terminators | `br L`; `condbr p, Lt, Lf`; `ret` | |

## Text format (`--emit=ir`)
Deterministic: same input gives byte-identical output. f32 immediates print with `%.9g`, which
round-trips exactly, plus an `f` suffix (`0.100000001f`). Non-finite constants print as
`f32(0x7fc00000)`.

```
kernel saxpy(f32 %a, ptr %x, ptr %y, i32 %n) {
entry:
  %a:f32 = param 0
  ...
  %5:i32 = sreg ctaid
  %6:i32 = sreg ntid
  %7:i32 = mul %5, %6
  %8:i32 = sreg tid
  %9:i32 = add %7, %8
  %i:i32 = mov %9
  %10:i1 = cmp.lt %i, %n
  condbr %10, if.then.1, if.end.1
if.then.1:
  %11:i64 = addr %y, %i, 4
  ...
  store.global %11, %17
  br if.end.1
if.end.1:
  br exit
exit:
  ret
}
```

## Lowering rules
- **Parameters:** `param N` into the parameter's named vreg at the start of `entry`.
- **Locals** without an initializer start at **0** (`mov %x, 0`). TinyCU defines this; sema still
  warns about reading an unassigned variable.
- **Short-circuit `&&` / `||`** use control flow, so the right operand is evaluated only when
  needed (`if (i < n && x[i] > 0)` never loads out of bounds):
  ```
    %r = mov <lhs>
    condbr <lhs>, and.rhs.N, and.end.N      ; || swaps the targets
  and.rhs.N:
    %r = mov <rhs>
    br and.end.N
  and.end.N:
  ```
- **if / while / for** use blocks `if.then.N if.else.N if.end.N`, `while.cond.N while.body.N
  while.end.N`, `for.cond.N for.body.N for.inc.N for.end.N`. `break` jumps to the end block.
  `continue` jumps to `for.inc` (for) or `while.cond` (while).
- **`return`** jumps to the single `exit` block, which holds the only `ret`.
- Code after `return`/`break`/`continue` is still lowered, into an unreachable `dead.N` block
  (SimplifyCFG removes it at `-O1`).
- **`p[i]`** becomes `addr t, p, i, 4` followed by `load`/`store`. **`a[i] op= e`** computes the
  address once: `addr`, `load`, op, `store`.
- Labels are `<kind>.<counter>`. The counter is per kernel, and blocks are appended in source
  order.

## Verifier (`src/ir/verifier.cpp`)
Runs after lowering and after every pass with `--verify-each` (default ON in Debug builds). It
checks:
- every block is non-empty and ends with exactly one terminator, with none in the middle;
- labels are unique, branch targets belong to the function, and nothing branches to `entry`;
- operand and result types match the opcode, `mov` source and destination types agree, and
  `addr` takes a pointer base and an i32 index;
- side-effect-only ops (`store`, `barrier`, terminators) have no destination;
- `param` indices are in range;
- every used vreg is defined somewhere (a simple check, not dominance).

A failure is an internal compiler error (exit code 3) and prints the offending function.

## CFG utilities (`src/ir/cfg.cpp`)
`successors`, `predecessors`, `reversePostOrder` (iterative DFS from entry) and `reachable`.

## Interpreter semantics (`src/interp/interp.cpp`)
`tinycu k.tc --interp --grid G --block B --args args.json [--kernel name]` prints a result
JSON. Exit code 4 means the kernel hit a runtime error; the JSON is still printed, with
`"status": "error"`.

- **Memory:** each pointer parameter is its own buffer, and each shared array is a fresh
  buffer per block. The interpreter tracks a *written* bit for every shared element, and a load
  of an element no thread of the current block has stored yet is an error (`uninit_shared`),
  because the GPU does not zero shared memory. Accesses are bounds-checked. *Computing* an out-of-range address
  is allowed; *accessing* it is an error naming the array, index, size, block and thread.
- **Execution:** blocks run one after another. Without a barrier, each thread runs to completion
  in order. With `barrier`, the threads of a block run in lock-step phases: each runs until the
  barrier or exit. If some threads wait while others exited, or threads wait at different
  barriers, that is reported as **barrier divergence**.
- **i32:** wraps (computed in `uint32_t`). Division and remainder truncate toward zero. Division
  or remainder by 0, and `INT_MIN / -1`, are errors. `abs(INT_MIN) = INT_MIN`, matching
  `abs.s32`.
- **f32:** the host `float` operations, compiled with `-ffp-contract=off` so no FMA is formed.
  `sqrt` is `std::sqrt(float)`, which is correctly rounded.
  - **Comparisons follow C:** `NaN != x` is true, and every other comparison with NaN is false.
  - `min`/`max` follow PTX `min.f32`/`max.f32`: a NaN operand returns the other one, and
    `-0 < +0`.
- **Conversions:** `cvt.i32.f32` of NaN or of a value outside [-2^31, 2^31) is an error.
- **Step limit:** 10,000,000 instructions per thread, then "possible infinite loop".

Float arguments and results move as raw bit patterns (`"bits"`), so NaN, -0 and subnormals
survive the JSON round trip and comparisons can be bit-exact.

On a run-time error the result JSON has `"status": "error"` and a machine-readable error
object, so the fuzz harness can classify failures without parsing text:

```json
"error": {"kind": "oob", "block": 3, "thread": 7,
          "msg": "out-of-bounds load from 'x' at index 1000 (size 1000)",
          "text": "runtime error in kernel 'k' (block 3, thread 7): out-of-bounds ..."}
```

| kind | meaning |
|---|---|
| `oob` | out-of-bounds (or invalid) load/store |
| `div0` | integer division or remainder by zero |
| `div_overflow` | `INT_MIN / -1` or `INT_MIN % -1` |
| `bad_cvt` | `(int)` of NaN or of a value outside the int32 range |
| `barrier_divergence` | threads of a block disagree about a `__syncthreads()` |
| `step_limit` | a thread exceeded the step limit (possible infinite loop) |
| `uninit_shared` | shared element read before any thread of the block wrote it |
| `bad_args` | argument list does not match the kernel (`block`/`thread` are null) |

### Float comparisons (C semantics, matched exactly by the PTX backend)
| TinyCU | IR | PTX | NaN operand |
|---|---|---|---|
| `==` | `cmp.eq` | `setp.eq.f32` (ordered) | false |
| `!=` | `cmp.ne` | `setp.neu.f32` (**unordered**) | true |
| `<` `<=` `>` `>=` | `cmp.lt/le/gt/ge` | `setp.lt/le/gt/ge.f32` (ordered) | false |

Consequence for the optimizer: `!(a < b)` is **not** `a >= b` for floats (they differ when an
operand is NaN), so that rewrite is only legal for ints.
