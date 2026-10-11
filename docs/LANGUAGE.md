# The TinyCU kernel language

TinyCU (`.tc`) is a small CUDA-like language for writing GPU kernels. A file holds one or
more `kernel`s. Each kernel is launched over a 1-D grid of 1-D thread blocks (x dimension only).

## 1. Grammar (EBNF)

```ebnf
program        := kernel+ EOF
kernel         := "kernel" IDENT "(" [param ("," param)*] ")" block
param          := type IDENT
type           := ("int" | "float") ["*"]
block          := "{" stmt* "}"
stmt           := vardecl | assign ";" | ifstmt | forstmt | whilestmt
                | "return" ";" | "break" ";" | "continue" ";" | block | "__syncthreads" "(" ")" ";"
vardecl        := vardecl_noSemi ";"
vardecl_noSemi := ["shared"] type IDENT ["[" INTLIT "]"] ["=" expr]
assign         := lvalue ("=" | "+=" | "-=" | "*=" | "/=") expr | lvalue ("++" | "--")
lvalue         := IDENT | IDENT "[" expr "]"
ifstmt         := "if" "(" expr ")" stmt ["else" stmt]
forstmt        := "for" "(" [vardecl_noSemi | assign] ";" [expr] ";" [assign] ")" stmt
whilestmt      := "while" "(" expr ")" stmt
expr           := logic_or
logic_or       := logic_and ("||" logic_and)*
logic_and      := equality ("&&" equality)*
equality       := relational (("=="|"!=") relational)*
relational     := additive (("<"|"<="|">"|">=") additive)*
additive       := mult (("+"|"-") mult)*
mult           := unary (("*"|"/"|"%") unary)*
unary          := ("-"|"!") unary | "(" type ")" unary | postfix
postfix        := primary ["[" expr "]"]
primary        := INTLIT | FLOATLIT | "true" | "false" | IDENT | builtin_var
                | builtin_fn "(" [expr ("," expr)*] ")" | "(" expr ")"
builtin_var    := ("threadIdx" | "blockIdx" | "blockDim" | "gridDim") ["." "x"]   // x only
builtin_fn     := "min" | "max" | "abs" | "sqrt"
```

`vardecl_noSemi` is `vardecl` without the trailing `;`; it is used by the `for` initializer.

### Notes on the grammar
- **Precedence** (lowest → highest): `||`, `&&`, `== !=`, `< <= > >=`, `+ -`, `* / %`,
  unary `- !` and casts, indexing. All binary operators are left-associative.
- **Dangling else** binds to the nearest `if`.
- **Casts**: `(` followed by the keyword `int` or `float` starts a cast, e.g. `(float)i`.
- **Builtin variables** may be written CUDA-style (`threadIdx.x`) or plain (`threadIdx`); both
  mean the same thing and `--emit=ast` always prints the plain form. Only the x dimension exists:
  `.y` / `.z` give `error: only the x dimension is supported (TinyCU is 1-D)`, and any other
  member (`.w`) is a syntax error. The builtin names, and `min max abs sqrt`, are keywords and
  cannot be used as variable names.
- `x++` / `x--` are statements (not expressions) and are desugared by the parser to
  `x += 1` / `x -= 1`.
- Only one level of indexing: `a[i]` is valid, `a[i][j]` is not.
- `shared` declarations are only meaningful at kernel top level with a constant size (checked by
  semantic analysis).

## 2. Lexical structure
- Whitespace: space, tab, CR, LF, FF, VT. CRLF line endings are accepted.
- Comments: `// …` to end of line, and `/* … */` (not nested). An unterminated block comment is
  an error.
- Identifiers: `[A-Za-z_][A-Za-z0-9_]*`, excluding keywords.
- Keywords: `kernel int float shared if else for while return break continue true false
  __syncthreads threadIdx blockIdx blockDim gridDim min max abs sqrt`.
- **INTLIT**: decimal digits; the value must fit in int32 (≤ 2147483647).
- **FLOATLIT**: `digits "." digits [exp] ["f"|"F"]` or `digits exp ["f"|"F"]`, where
  `exp := ("e"|"E") ["+"|"-"] digits`. Examples: `1.5`, `1.0f`, `2e-3`, `3E+2f`.
  Forms without digits on both sides of the dot (`1.`, `.5`) are not accepted.
  The value is rounded to the nearest f32.
- A letter or `_` directly after a number (`12abc`, `1.5x`) is an error.
- Operators use maximal munch: `a+++b` lexes as `a ++ + b`.
- `.` is a token only directly after one of the four builtin variable names (the `.x` member);
  anywhere else it is an unexpected character. Float literals such as `1.5` are unaffected.

### Literal limits (known limitations)
1. **`-2147483648` cannot be written as a literal.** `-` is a unary operator applied to the
   literal `2147483648`, which does not fit in int32 and is rejected by the lexer. Write
   `-2147483647 - 1` instead. The random program generator never emits this literal.
2. **Float literals must be finite and must not silently become zero.** A literal that
   overflows to infinity (`1e39`) or a literal with a non-zero mantissa that underflows to
   zero (`1e-50`) is a lexer error. Subnormal results (`1e-40`) and true zeros (`0.0`, `0e-50`)
   are accepted. So `inf` can never appear as a float immediate in later phases.

## 3. Diagnostics
Errors are printed as

```
file.tc:LINE:COL: error: message
<source line>
    ^
```

in source order. The parser recovers from a syntax error by skipping to the next `;` (or over a
balanced `{ … }` statement body) and continues, up to 20 errors. Exit code 1 means a compile
error.

## 4. Example kernels

### vecadd
```c
kernel vecadd(int* a, int* b, int* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}
```

### saxpy
```c
kernel saxpy(float a, float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}
```

### relu
```c
kernel relu(float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    if (v < 0.0f) {
      v = 0.0;
    }
    y[i] = v;
  }
}
```

### clamp
```c
kernel clamp(float* x, float lo, float hi, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  x[i] = min(max(x[i], lo), hi);
}
```

### reduce_shared (stretch: shared memory + barrier)
```c
kernel reduce_shared(float* in, float* out, int n) {
  shared float buf[256];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = 0.0f;
  if (i < n) {
    v = in[i];
  }
  buf[tid] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s /= 2) {
    if (tid < s) {
      buf[tid] += buf[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[blockIdx.x] = buf[0];
  }
}
```

## 5. Semantics

### 5.1 Types
| Type | Meaning |
|---|---|
| `int` | 32-bit two's-complement integer; arithmetic **wraps** on overflow |
| `float` | IEEE-754 binary32 |
| `bool` | internal only: the result of comparisons and `&&`, `\|\|`, `!`. There is no `bool` keyword, so a bool can't be stored in a variable or array |
| `int*`, `float*` | pointer to global memory; **kernel parameters only** |
| `shared int[N]`, `shared float[N]` | per-block shared-memory array |
| `void` | internal |

### 5.2 Rules
1. **Scopes.** Every block, every `if`/loop body, and every `for` statement (for its initializer)
   opens a scope. Kernel parameters are in the outermost scope; the kernel body is the next one.
   Redeclaring a name in the same scope is an **error**. Shadowing an outer name, including a
   parameter, is a **warning**. A variable's initializer is checked before the name is declared,
   so in `int x = x + 1;` the `x` on the right is the outer `x`. The builtin variables are
   read-only `int`s.
2. **Arithmetic** `+ - * /`: int⊕int→int, float⊕float→float. For int⊕float, the int operand is
   implicitly converted to float and sema inserts an implicit cast node. `%` requires two ints.
   Unary `-` works on int or float.
3. **Comparisons** `< <= > >= == !=` take int/float operands (mixed operands are promoted like
   arithmetic) and produce `bool`. Bools can't be compared. `&&`, `||` and `!` require `bool`
   operands. Conditions of `if`, `while` and `for` must be `bool`. There is **no C truthiness**:
   `if (n)` is an error, so write `if (n != 0)`.
4. **Assignment.** The target must be a scalar variable or an element `a[i]`. int→float is
   implicit. float→int is an error ("implicit conversion from float to int loses precision; use
   (int)"). `x op= e` is checked as `x = x op e`, so `int x; x += 1.5;` is an error.
5. **Indexing.** The base must be a pointer parameter or a shared array, and the index must be
   `int`. The result has the element type.
6. **Pointers** can only be indexed. They can't be assigned, compared, copied, cast, or used in
   arithmetic, and local pointer variables don't exist. A shared array name likewise can only be
   indexed.
7. **Builtins:** `min(x,y)` and `max(x,y)` take (int,int) or (float,float), and mixed arguments
   are promoted to float. `abs(x)` takes int or float and returns the same type. `sqrt(x)` takes
   float; an int argument is implicitly converted.
8. **`break` / `continue`** are only allowed inside a `for` or `while`. `__syncthreads()` is
   allowed anywhere. **Divergence risk:** calling it where only some threads of a block reach it
   (for example inside `if (threadIdx.x < 16)`) is undefined behavior on the GPU, and the
   interpreter reports it as an error at run time. Sema doesn't try to prove convergence.
9. **Shared arrays** must be declared at kernel top level (not inside a block, branch, loop, or
   `for` initializer). They need a constant size 1 ≤ N ≤ 12288 (48 KB of 4-byte elements) and
   can't have an initializer. The kernel's total shared memory must not exceed 48 KB. Local
   (non-shared) arrays aren't supported.
10. **Warnings:** unused local variable; variable used before any assignment. The uninitialized
    check is deliberately simple: it uses straight-line source order, and any earlier assignment,
    even in a branch, counts as initialization.
11. **Casts** `(int)e` / `(float)e` take int or float. `(int)` truncates toward zero. Casting a
    `bool`, or casting to a pointer type, is an error.
12. Kernel names must be unique within a file.

### 5.3 Typed AST
After sema, `--emit=ast` annotates every operator node with its type and shows the implicit
conversions, e.g. `(+:float (cast:float i) x)`. Leaves (names, literals) are printed bare. An
explicit cast and an implicit one print the same way (`cast:float`).

Errors stop compilation (exit code 1). Warnings are printed but don't affect the exit code. Sema
runs only if parsing succeeded.
