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
