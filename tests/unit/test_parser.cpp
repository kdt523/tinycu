#include "tinycu/parser.h"

#include <gtest/gtest.h>

#include <string>

using namespace tinycu;

namespace {

// Parses `kernel k(<params>) { <body> }` and returns the AST dump ("" on error).
std::string parseOk(const std::string &body, const std::string &params = "int a, int b, int c") {
  std::string src = "kernel k(" + params + ") {\n" + body + "\n}\n";
  DiagEngine diag("t.tc", src);
  auto prog = parseSource(src, diag);
  if (diag.hasErrors())
    return "";
  return printAST(*prog);
}

// Returns the init expression of the first statement `int r = <expr>;`.
std::string exprOf(const std::string &expr) {
  std::string src = "kernel k(int a, int b, int c, float f) {\nint r = " + expr + ";\n}\n";
  DiagEngine diag("t.tc", src);
  auto prog = parseSource(src, diag);
  EXPECT_FALSE(diag.hasErrors()) << expr;
  if (diag.hasErrors())
    return "";
  auto *d = dyn_cast<VarDeclStmt>(prog->kernels[0]->body->stmts[0].get());
  return d && d->init ? printExpr(*d->init) : "";
}

unsigned errorsFor(const std::string &src) {
  DiagEngine diag("t.tc", src);
  parseSource(src, diag);
  return diag.errorCount();
}

} // namespace

TEST(Parser, LeftAssociativeSubtraction) { EXPECT_EQ(exprOf("a-b-c"), "(- (- a b) c)"); }

TEST(Parser, MulBindsTighterThanAdd) {
  EXPECT_EQ(exprOf("a+b*c"), "(+ a (* b c))");
  EXPECT_EQ(exprOf("a*b+c"), "(+ (* a b) c)");
}

TEST(Parser, UnaryBindsTighterThanMul) { EXPECT_EQ(exprOf("-a*b"), "(* (- a) b)"); }

TEST(Parser, LogicalPrecedence) {
  EXPECT_EQ(exprOf("!a&&b||c"), "(|| (&& (! a) b) c)");
  EXPECT_EQ(exprOf("a||b&&c"), "(|| a (&& b c))");
}

TEST(Parser, ComparisonPrecedence) {
  EXPECT_EQ(exprOf("a+1<b*2==c>=a"), "(== (< (+ a 1) (* b 2)) (>= c a))");
}

TEST(Parser, ParenthesesOverride) { EXPECT_EQ(exprOf("(a+b)*c"), "(* (+ a b) c)"); }

TEST(Parser, Remainder) { EXPECT_EQ(exprOf("a%b/c"), "(/ (% a b) c)"); }

TEST(Parser, Casts) {
  EXPECT_EQ(exprOf("(int)f"), "(cast int f)");
  EXPECT_EQ(exprOf("(float)a*b"), "(* (cast float a) b)");
  EXPECT_EQ(exprOf("(int)(f+1.0)"), "(cast int (+ f 1.0))");
  EXPECT_EQ(exprOf("-(int)-f"), "(- (cast int (- f)))");
}

TEST(Parser, ParenthesizedVarIsNotCast) { EXPECT_EQ(exprOf("(a)-b"), "(- a b)"); }

TEST(Parser, BuiltinsAndIndex) {
  EXPECT_EQ(exprOf("min(a, max(b, c)) + abs(-a)"),
            "(+ (call min a (call max b c)) (call abs (- a)))");
  EXPECT_EQ(exprOf("blockIdx*blockDim+threadIdx"), "(+ (* blockIdx blockDim) threadIdx)");
  EXPECT_EQ(exprOf("sqrt()"), "(call sqrt)"); // arity is checked by sema
}

TEST(Parser, DanglingElseBindsToNearestIf) {
  std::string ast = parseOk("if (a > 0) if (b > 0) c = 1; else c = 2;");
  EXPECT_NE(
      ast.find("(if (> a 0)\n      (if (> b 0)\n        (assign = c 1)\n        (assign = c 2)))"),
      std::string::npos)
      << ast;
}

TEST(Parser, IncrementDesugars) {
  std::string ast = parseOk("a++; b--;");
  EXPECT_NE(ast.find("(assign += a 1)"), std::string::npos) << ast;
  EXPECT_NE(ast.find("(assign -= b 1)"), std::string::npos) << ast;
}

TEST(Parser, ForVariants) {
  EXPECT_NE(parseOk("for (int i = 0; i < a; i++) { }")
                .find("(for (vardecl int i 0) (< i a) (assign += i 1)"),
            std::string::npos);
  EXPECT_NE(parseOk("for (;;) { break; }").find("(for () () ()"), std::string::npos);
  EXPECT_NE(parseOk("for (a = 0; a < 3; a += 2) continue;").find("(for (assign = a 0)"),
            std::string::npos);
}

TEST(Parser, SharedArrayDecl) {
  EXPECT_NE(parseOk("shared float buf[256]; __syncthreads();")
                .find("(vardecl shared float buf[256])\n    (syncthreads)"),
            std::string::npos);
}

TEST(Parser, PointerParams) {
  EXPECT_NE(parseOk("x[a] = 1.0;", "float* x, int a").find("(params (float* x) (int a))"),
            std::string::npos);
}

TEST(Parser, EmptyParams) { EXPECT_NE(parseOk("return;", ""), ""); }

TEST(Parser, ErrorRecoveryReportsMultipleErrors) {
  // Each bad statement is reported; recovery resyncs at ';'.
  EXPECT_EQ(errorsFor("kernel k(int a) {\n a = ;\n a = 1 +;\n a = (1;\n}\n"), 3u);
}

TEST(Parser, ErrorCapAt20) {
  std::string src = "kernel k(int a) {\n";
  for (int i = 0; i < 50; ++i)
    src += "a = ;\n";
  src += "}\n";
  EXPECT_EQ(errorsFor(src), 20u);
}

TEST(Parser, DeepNestingDoesNotCrash) {
  std::string deep(100000, '(');
  EXPECT_GT(errorsFor("kernel k(int a) { a = " + deep + "1; }"), 0u);
  std::string blocks(100000, '{');
  EXPECT_GT(errorsFor("kernel k(int a) " + blocks), 0u);
}

TEST(Parser, ChainedIndexIsError) {
  EXPECT_GT(errorsFor("kernel k(int* a) { a[0] = a[a[0]][1]; }"), 0u);
}
