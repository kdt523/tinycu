---
name: tinycu-02-sema
description: Phase 2 of TinyCU — semantic analysis: types, scopes/symbol table, implicit conversions, lvalue rules and diagnostics.
---

# Phase 2 — Semantic analysis

## Goal
Reject every ill-typed program with a clear diagnostic; annotate every Expr with its type
and insert implicit casts so later phases never guess.

## Type system (`include/tinycu/types.h`)
- Types: `int` (s32), `float` (f32), `bool` (internal: result of comparisons/logic),
  `int*`, `float*` (global memory), `shared int[N]`, `shared float[N]`, `void`.
- Interned singletons (`TypeContext`) so types compare by pointer.

## Rules (put them in docs/LANGUAGE.md §Semantics)
1. Scopes: block-scoped; shadowing an outer variable = **warning**; redeclare in same scope = error.
   Kernel params are in the outermost scope. Builtin vars are read-only int.
2. Arithmetic `+ - * / %`: int⊕int→int; float⊕float→float; int⊕float → int implicitly
   converted to float (insert `ImplicitCast` node). `%` int only.
3. Comparisons → bool; `&& || !` require bool (int in condition is an **error**: no C truthiness —
   simpler and safer). `if/while/for` conditions must be bool.
4. Assignment: target must be lvalue (var or index); int→float implicit allowed; float→int
   requires explicit cast (error otherwise: "implicit conversion from float to int loses precision; use (int)").
5. Indexing: base must be pointer or shared array; index must be int. Result is element type.
6. Pointers cannot be assigned, compared, or used in arithmetic (only indexed) — simplifies backend.
7. Builtins: `min/max(int,int)|(float,float)`, `abs(int|float)`, `sqrt(float)` (int arg → implicit cast).
8. `break/continue` only inside loops; `__syncthreads()` allowed anywhere (document divergence risk).
9. Constant-size shared arrays: 1 ≤ N ≤ 12288 floats (48 KB) — error otherwise.
10. Unused variable → warning. Use-before-init of scalar → warning (simple: declared without init and read before any assignment in straight-line order).

## Implementation
- `class Sema : ASTVisitor` with `ScopeStack` (vector of unordered_map<string, Symbol*>).
- `Symbol { name, Type*, kind (Param|Local|Shared), SourceLoc, unique id }` — the id is
  reused by IR lowering for virtual register naming.
- `--emit=ast` now prints types: `(+:float (cast:float i) x)`.

## Tests
- Errors (≥20 cases): undeclared var, redeclared, float→int assign, `if (x)` with int, pointer
  arithmetic, index with float, `%` on float, break outside loop, wrong builtin arity, assign
  to `threadIdx`, shared array size 0, etc.
- Golden typed-AST for all phase 1 programs.
- Warnings tested separately (`// EXPECT-WARN:`).

## Definition of Done
- [ ] All valid golden programs type-check; all error cases produce expected message
- [ ] No crash on any phase-1 fuzz input (sema runs only if parse succeeded)
- [ ] Tag `v0.2`

## Interview notes
Symbol tables & scope chains; static vs dynamic typing; type coercion; attribute grammars
(synthesized vs inherited attributes — type is synthesized); why language specs matter for
verification ("good understanding of programming languages and their specification" in the JD).
