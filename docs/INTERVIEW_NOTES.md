# Interview notes

Study notes collected while building TinyCU, one section per phase.

## Phase 0 — compiler drivers

**gcc is a driver.** `gcc foo.c` runs several programs in sequence: `cc1` (preprocess + compile
to assembly), `as` (assemble to an object file), `collect2`/`ld` (link). `gcc -v` shows each
step. TinyCU's `tinycu` binary is also a driver: it runs the stages in-process and stops early
for `--emit=ast|ir`.

**What `nvcc` does with a `.cu` file:**
`.cu → cudafe++` splits host and device code → device code goes to **`cicc`** (the NVVM/LLVM-based
front/middle end) → **PTX** (virtual ISA text) → **`ptxas`** (register allocation, scheduling,
SASS generation) → **cubin** (ELF with SASS for one `sm_XX`) → **`fatbinary`** bundles cubins
plus PTX (for JIT on future GPUs) → embedded into the host object → host compiler + link.
`nvcc --dryrun` lists every step. TinyCU replaces the `cicc` part: `.tc → PTX`, then uses the
real `ptxas`.

## Phase 1 — lexing and parsing

**Lexeme vs token vs pattern.** A *pattern* is the rule (e.g. `[A-Za-z_][A-Za-z0-9_]*`). A
*lexeme* is the actual character sequence matched in the source (`saxpy`). A *token* is the
pair (kind, attributes) the parser consumes: `(Ident, "saxpy", 1:8)`. TinyCU's `Token` carries
kind, a `string_view` lexeme, the location, and the decoded literal value.

**Maximal munch.** The lexer always takes the longest operator that matches: `a+++b` →
`a ++ + b`, and `<=` is never lexed as `<` `=`.

**Why recursive descent fits.** The grammar is LL(1) except for one place: after `(` we need
one extra token to tell a cast `(int)x` from a parenthesized expression `(x)` (LL(2) there).
Each rule becomes one function, so the code mirrors the spec. That makes it easy to review for
conformance and easy to give precise errors.

**Left recursion removal.** `additive := additive "+" mult | mult` would recurse forever in a
top-down parser. Rewriting it as `additive := mult ("+" mult)*` and building the tree in a loop
keeps left associativity (`a-b-c` = `(a-b)-c`).

**Precedence climbing.** The alternative to one function per precedence level: a single
`parseBinary(minPrec)` loop driven by an operator table. TinyCU uses the per-level functions
because the grammar is written that way. The two approaches give identical trees.

**FIRST / FOLLOW for two TinyCU rules**

`unary := ("-"|"!") unary | "(" type ")" unary | postfix`
- FIRST(unary) = { `-`, `!`, `(`, INTLIT, FLOATLIT, `true`, `false`, IDENT, `threadIdx`,
  `blockIdx`, `blockDim`, `gridDim`, `min`, `max`, `abs`, `sqrt` }
- `(` appears in FIRST of both the cast alternative and `postfix` (via `"(" expr ")"`), so this
  rule is **not LL(1)**. The parser peeks one token further: `int`/`float` means cast.
- FOLLOW(unary) = { `*`, `/`, `%` } ∪ FOLLOW(mult) = { `* / % + - < <= > >= == != && ||`, `)`,
  `]`, `,`, `;` }

`stmt` (alternatives are selected by their first token, so this part is LL(1)):
- FIRST(stmt) = { `shared`, `int`, `float`, IDENT, `if`, `for`, `while`, `return`, `break`,
  `continue`, `{`, `__syncthreads` }
- FOLLOW(stmt) = FIRST(stmt) ∪ { `}`, `else` }. `else` is in FOLLOW because of `ifstmt`: this is
  the dangling-else conflict, resolved by greedily binding `else` to the nearest `if`.

**Parse tree vs AST.** A parse tree has a node for every grammar rule application (including
`logic_or → logic_and → … → primary` chains for a single identifier). The AST keeps only the
semantics: `x` is just a `VarRef`. Sugar is removed (`x++` becomes `x += 1`), and parentheses
disappear because the tree shape encodes them.

**Panic-mode error recovery.** On a syntax error, report it, then discard tokens until a
*synchronizing token* (`;`, `}`, or the start of the next `kernel`), and resume. TinyCU also
skips a balanced `{…}` body while synchronizing, so `if (x {…}` doesn't make the kernel's `}`
look unmatched. Errors are capped at 20, and errors at tokens the lexer already reported are
suppressed so one bad character doesn't cascade.

**Robustness.** The parser has a nesting-depth limit (200), so `((((…` cannot overflow the
stack. A unit test feeds 3,000 random byte strings and token shuffles through the frontend under
ASan/UBSan.

## Phase 2 — semantic analysis

**Symbol tables and scope chains.** Each scope is a hash map name → Symbol, kept on a stack.
Lookup walks from the innermost scope outward, which is the static scope chain. Declaring a name
checks only the innermost map (redeclaration), but looks outward to warn about shadowing. Symbols
get a per-kernel `id` that IR lowering reuses for register names, so the mapping from source
variable to virtual register is stable and printable.

**Static vs dynamic typing.** TinyCU is statically typed: every expression's type is known at
compile time, and sema stores it in `Expr::type`. Later phases never inspect values to decide
types. A GPU compiler needs this, because the same `+` becomes `add.s32` or `add.rn.f32` in PTX.

**Coercion is explicit in the tree.** Sema inserts `CastExpr(implicit)` nodes for int→float, so
lowering never has to re-derive conversions. That's one source of truth for the conversion rules.
Only widening int→float is implicit. Narrowing float→int needs a cast, the "no silent precision
loss" rule that C++ style guides enforce with `-Wconversion`.

**Attribute grammars.** The type of an expression is a *synthesized* attribute, computed
bottom-up from the children's types (`int + float → float`). The scope/environment is an
*inherited* attribute, passed top-down (a block inherits its parent's symbols). The flags "inside
a loop" (legal `break`) and "at kernel top level" (legal `shared`) are inherited attributes too.

**Error type to avoid cascades.** An ill-typed sub-expression gets the special `<error>` type,
and any operator with an `<error>` operand silently produces `<error>`. One mistake gives one
diagnostic, the same technique clang uses.

**Why specs matter for verification.** A compiler can only be "correct" relative to a spec.
LANGUAGE.md §5 is that spec. Every rule has an error test, and the fuzzer (phase 7) generates
only programs the spec says are valid and UB-free. Ambiguities found while writing sema (e.g. is
`bool == bool` allowed? No) were settled in the spec first, then implemented.

## Phase 3 — IR, CFG and the interpreter oracle

**Three-address code.** Every instruction has at most one operator and three "addresses":
`%d = op a, b`. Nested expressions are flattened into temporaries. Stored as *quadruples*
(op, a, b, dst); TinyCU's `Inst` is a quadruple plus branch targets and an `aux` field.

**Basic blocks and the leaders algorithm.** A *leader* is the first instruction of the
program, any branch target, and any instruction right after a branch. A basic block runs from
one leader up to the next. TinyCU creates blocks directly during lowering (one per
`if.then`, `for.cond`, ...), so leaders never need to be recomputed. The verifier enforces the
result: exactly one terminator, at the end.

**CFG.** Nodes are blocks and edges are branches. *Reverse post-order* is the classic iteration
order for forward dataflow: every block comes before its successors, except along loop back
edges. Unreachable blocks, like code after `return`, simply don't appear in the RPO.

**Why SSA exists, and why TinyCU skips it.** In SSA every variable is assigned exactly once, and
phi nodes merge values at join points. That makes def-use chains trivial and many optimizations
(constant propagation, GVN, register allocation by graph coloring on chordal graphs) simpler and
faster. The costs are building it (dominance frontiers) and leaving it (phi elimination). TinyCU
stays non-SSA:
- its optimizations are local or simple global ones;
- PTX accepts repeated assignments to a virtual register;
- `ptxas` builds its own SSA form anyway.

This is a deliberate scope choice, and "add SSA" is listed as future work.

**Short-circuit evaluation as control flow.** `a && b` must not evaluate `b` when `a` is false,
because `b` might index out of bounds. Lowering emits a conditional branch around `b`'s code,
not an `and` instruction. On a GPU this introduces divergence. A later optimization could turn a
side-effect-free `b` back into predicated code (`and.pred`).

**An interpreter as the test oracle.** A second, much simpler implementation of the language
semantics. It is written for obviousness, not speed, and every undefined behavior is checked and
reported. Every compiled result (`-O0`, `-O1`, the GPU, nvcc twins) is compared against it. This
works because the interpreter is independent of the backend and the optimizer: a bug has to be
made twice, the same way, to go unnoticed. The interpreter is also what makes random testing
safe. A generated program that triggers UB is caught as a generator bug, instead of becoming a
false miscompile report.

**SIMT emulation detail.** Without barriers, running threads one after another is equivalent to
any interleaving for race-free kernels. With `__syncthreads`, the interpreter runs each thread up
to the barrier, then releases all of them. That is exactly the guarantee CUDA gives. Threads that
skip a barrier others are waiting at are *barrier divergence*: UB in CUDA, an error here.

## Phase 4 — PTX backend

**Reading nvcc's PTX for vecadd** (`nvcc -ptx -arch=sm_75 vecadd.cu`, CUDA 12.6):
```
.visible .entry vecadd(.param .u64 vecadd_param_0, ..., .param .u32 vecadd_param_3)
ld.param.u64    %rd1, [vecadd_param_0];   // kernel args live in the .param state space
mov.u32         %r3, %ctaid.x;            // special registers: %tid, %ctaid, %ntid, %nctaid
mad.lo.s32      %r1, %r3, %r4, %r5;       // blockIdx*blockDim + threadIdx (int mad: exact)
setp.ge.s32     %p1, %r1, %r2;            // compare -> predicate register
@%p1 bra        $L__BB0_2;                // predicated branch (guard i < n)
cvta.to.global.u64 %rd4, %rd1;            // generic address -> global window
mul.wide.s32    %rd5, %r1, 4;             // 32x32 -> 64-bit multiply: byte offset
add.s64         %rd6, %rd4, %rd5;
ld.global.u32   %r6, [%rd8];              // state space is part of the opcode
st.global.u32   [%rd10], %r8;
```
- **State spaces:** `.reg` (registers), `.param` (kernel arguments, read via `ld.param`),
  `.global` (device memory), `.shared` (per-block on-chip memory), plus `.local`, `.const` and
  `.tex`. A pointer is just a 64-bit integer. Which memory it points into is decided by the
  state space of the load or store.
- **`.entry`** is a kernel (callable from the host); `.func` would be a device function.
  `.visible` makes the symbol exported, so the driver can find it by name.

**PTX vs SASS.** PTX is a *virtual* ISA: stable, documented, with unlimited typed virtual
registers, and forward-compatible. SASS is the real machine code of one GPU generation
(`sm_75` = Turing). `ptxas` (or the driver's JIT) translates PTX to SASS. It does register
allocation, instruction scheduling, and its own optimizations. That is why TinyCU doesn't need a
register allocator: it emits one virtual register per IR value and lets ptxas map them onto 255
physical registers, spilling to local memory if needed (`ptxas -v` reports spill bytes).

**Predication vs branching.** `setp` writes a predicate, and any instruction can be guarded
`@%p` / `@!%p`. Short `if`s can be fully predicated (no branch). TinyCU always branches and
leaves if-conversion to ptxas, which does it. In the saxpy SASS, the guard became `@P0 EXIT`.

**Warp divergence and `bra.uni`.** A warp runs 32 threads in lockstep (SIMT). When a
conditional branch splits a warp, both paths execute serially with threads masked off, and they
reconverge after. `bra.uni` *promises* the branch is non-divergent (all active threads go the
same way), which lets the hardware skip the divergence handling. That holds for unconditional
jumps, so TinyCU uses `bra.uni` only there and plain `@%p bra` for conditional branches.

**What `.rn` means.** It is the IEEE round-to-nearest-even rounding mode. Specifying it
explicitly on `add/sub/mul.f32` also forbids ptxas from fusing a multiply and an add into an FMA,
which rounds once instead of twice and would break bit-exactness with the interpreter.
`div.rn.f32` and `sqrt.rn.f32` are the IEEE-correct versions; plain `div.f32` is not valid PTX
without `.approx` or `.full`.

**Float compare ordering.** `setp.lt.f32` is *ordered*: false if either operand is NaN.
`setp.neu.f32` is *unordered*: true if either operand is NaN. C's `!=` is true for NaN, so it
maps to `neu`. Using `setp.ne.f32` would silently differ from the interpreter on NaN inputs.

**Generic vs global addressing.** A kernel pointer argument is a *generic* address that can point
into global, shared or local memory. `cvta.to.global` converts it to a global-window address, so
the code can use `ld.global`/`st.global`, which skip the hardware's generic-to-specific address
check.

**cubin = ELF.** `cuobjdump -elf` on a TinyCU cubin shows:
- `.text.<kernel>`: the SASS;
- `.nv.info` and `.nv.info.<kernel>`: attributes such as register count, parameter layout and
  barrier usage;
- `.nv.constant0.<kernel>`: the kernel parameters are passed in constant bank 0, so SASS reads
  `c[0x0][0x160]` and so on;
- `.nv.shared.<kernel>`: a NOBITS section sized to the shared arrays;
- `.symtab`, `.strtab`, `.shstrtab`, `.nv.callgraph` and `.nv.rel.action`.

The PTXAS team's output is exactly this object file.

**JIT and forward compatibility.** A fatbinary carries cubins for known GPUs plus PTX. On a newer
GPU with no matching cubin, the driver JIT-compiles the PTX with its built-in ptxas. TinyCU
targets PTX ISA 7.0, which any CUDA 11+ driver accepts, so the same `.ptx` loads on Colab's T4
through `cuModuleLoadDataEx` without an offline ptxas step.
