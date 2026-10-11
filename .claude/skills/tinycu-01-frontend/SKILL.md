---
name: tinycu-01-frontend
description: Phase 1 of TinyCU — language spec, hand-written lexer, recursive-descent parser, AST and --emit=ast pretty printer with golden and unit tests.
---

# Phase 1 — Language spec, Lexer, Parser, AST

## Goal
Parse every valid `.tc` program into an AST, print it deterministically, and give precise
syntax errors with line:col.

## 1. Language spec (write it into docs/LANGUAGE.md verbatim, then implement)
```ebnf
program     := kernel+ EOF
kernel      := "kernel" IDENT "(" [param ("," param)*] ")" block
param       := type IDENT
type        := ("int" | "float") ["*"]
block       := "{" stmt* "}"
stmt        := vardecl | assign ";" | ifstmt | forstmt | whilestmt
             | "return" ";" | "break" ";" | "continue" ";" | block | "__syncthreads" "(" ")" ";"
vardecl     := ["shared"] type IDENT ["[" INTLIT "]"] ["=" expr] ";"
assign      := lvalue ("=" | "+=" | "-=" | "*=" | "/=") expr | lvalue ("++" | "--")
lvalue      := IDENT | IDENT "[" expr "]"
ifstmt      := "if" "(" expr ")" stmt ["else" stmt]
forstmt     := "for" "(" [vardecl_noSemi | assign] ";" [expr] ";" [assign] ")" stmt
whilestmt   := "while" "(" expr ")" stmt
expr        := logic_or
logic_or    := logic_and ("||" logic_and)*
logic_and   := equality ("&&" equality)*
equality    := relational (("=="|"!=") relational)*
relational  := additive (("<"|"<="|">"|">=") additive)*
additive    := mult (("+"|"-") mult)*
mult        := unary (("*"|"/"|"%") unary)*
unary       := ("-"|"!") unary | "(" type ")" unary | postfix
postfix     := primary ["[" expr "]"]
primary     := INTLIT | FLOATLIT | "true" | "false" | IDENT | builtin_var
             | builtin_fn "(" [expr ("," expr)*] ")" | "(" expr ")"
builtin_var := "threadIdx" | "blockIdx" | "blockDim" | "gridDim"     // x-dimension only
builtin_fn  := "min" | "max" | "abs" | "sqrt"
```
- Comments: `//` and `/* */`. INTLIT: decimal, fits in int32. FLOATLIT: `1.5`, `1.0f`, `2e-3`.
- `shared` arrays only at kernel top level, size is a constant (used in phase 5 stretch).
- Example kernels to include in docs: vecadd, saxpy, relu, clamp, reduce_shared (stretch).

## 2. Lexer (`src/frontend/lexer.cpp`)
- `struct SourceLoc { uint32_t line, col; }`; `struct Token { TokKind kind; std::string_view text; SourceLoc loc; }`.
- Keywords table; maximal munch for `<= >= == != && || += -= *= /= ++ --`.
- Errors: unknown char, unterminated block comment, int literal overflow → diagnostic, emit
  `TokKind::Error`, keep going.
- `DiagEngine` (include/tinycu/diag.h): collects `{severity, loc, msg}`, prints
  `file:line:col: error: msg` + source line + caret. Shared by all later phases.

## 3. AST (`include/tinycu/ast.h`)
- Node kinds via `enum class` + `std::unique_ptr`, or `std::variant` — pick unique_ptr
  class hierarchy with `kind()` + `dyn_cast<>` helper (LLVM-style RTTI, good interview talk).
- Expr: IntLit, FloatLit, BoolLit, VarRef, Index, Unary, Binary, Cast, BuiltinVar, Call.
- Stmt: VarDecl, Assign (op), IfStmt, ForStmt, WhileStmt, Return, Break, Continue, Block, SyncThreads.
- Every node stores SourceLoc. Expr has a `Type* type` slot filled by sema (phase 2).
- Desugar in the parser: `x++` → `x += 1`; `for` stays a node (lowered in phase 3).

## 4. Parser (`src/frontend/parser.cpp`)
- Recursive descent exactly following the grammar (one function per rule).
- Cast disambiguation: `(` followed by `int`/`float` keyword ⇒ cast.
- Error recovery: on error, report, then **synchronize** to next `;` or `}`; cap at 20 errors.
- Messages like `expected ';' after expression`, `expected ')' to match '(' at 3:14`.

## 5. AST printer (`--emit=ast`)
S-expression style, deterministic, e.g.
```
(kernel saxpy (params (float a) (float* x) (float* y) (int n))
  (block (vardecl int i (+ (* blockIdx blockDim) threadIdx))
         (if (< i n) (assign = (index y i) (+ (* a (index x i)) (index y i))))))
```

## 6. Tests
- Unit (GoogleTest): lexer token streams (≥15 cases incl. edge cases), precedence/associativity
  (`a-b-c`, `a+b*c`, `-a*b`, `!a&&b||c`), cast parsing, dangling else.
- Golden: `tests/golden/*.tc` → `*.ast`; ctest target compares output; script
  `tools/update_golden.sh` regenerates (review diffs!). ≥10 programs.
- Errors: `tests/errors/*.tc` with first line `// EXPECT: 3:9: expected ';'`; runner checks
  substring. ≥10 cases.
- Fuzz-robustness: feed 1,000 random byte strings / token shuffles → must never crash
  (run under ASan).

## Definition of Done
- [ ] All grammar features parse; golden + error + unit tests green under ASan/UBSan
- [ ] docs/LANGUAGE.md complete with examples
- [ ] Tag `v0.1`

## Interview notes
Lexeme vs token vs pattern; why recursive descent fits LL(1); left recursion removal;
operator precedence climbing; FIRST/FOLLOW sets for 2 rules of this grammar (write them in
docs/INTERVIEW_NOTES.md); parse tree vs AST; panic-mode error recovery.
