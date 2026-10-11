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
_Filled in by phase 2 (types, scopes, conversions, lvalues)._
