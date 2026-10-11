#include "tinycu/parser.h"
#include "tinycu/sema.h"

#include <gtest/gtest.h>

#include <sstream>
#include <string>

using namespace tinycu;

namespace {

struct Checked {
  std::unique_ptr<Program> prog;
  std::string ast;
  std::string diags;
  unsigned errors = 0;
  unsigned warnings = 0;
};

Checked sema(const std::string &src) {
  static std::string keep;
  keep = src;
  DiagEngine diag("t.tc", keep);
  Checked c;
  c.prog = parseSource(keep, diag);
  EXPECT_FALSE(diag.hasErrors()) << "parse error in test source";
  Sema(diag).run(*c.prog);
  c.ast = printAST(*c.prog);
  std::ostringstream os;
  diag.print(os);
  c.diags = os.str();
  c.errors = diag.errorCount();
  for (const Diagnostic &d : diag.diagnostics())
    c.warnings += d.severity == Severity::Warning;
  return c;
}

// Statements inside a kernel with handy parameters.
Checked body(const std::string &stmts) {
  return sema("kernel k(int a, int b, float f, float g, int* p, float* q) {\n" + stmts + "\n}\n");
}

bool contains(const std::string &hay, const std::string &needle) {
  return hay.find(needle) != std::string::npos;
}

} // namespace

TEST(Sema, TypesAreInterned) {
  TypeContext &tc = TypeContext::get();
  EXPECT_EQ(tc.ptrTo(tc.intTy()), tc.ptrTo(tc.intTy()));
  EXPECT_NE(tc.ptrTo(tc.intTy()), tc.ptrTo(tc.floatTy()));
  EXPECT_EQ(tc.sharedArray(tc.floatTy(), 256), tc.sharedArray(tc.floatTy(), 256));
  EXPECT_NE(tc.sharedArray(tc.floatTy(), 256), tc.sharedArray(tc.floatTy(), 128));
  EXPECT_EQ(tc.sharedArray(tc.intTy(), 4)->str(), "shared int[4]");
  EXPECT_EQ(tc.ptrTo(tc.floatTy())->str(), "float*");
}

TEST(Sema, IntPlusFloatInsertsImplicitCast) {
  Checked c = body("p[0] = (int)(a + f);");
  EXPECT_EQ(c.errors, 0u) << c.diags;
  EXPECT_TRUE(contains(c.ast, "(cast:int (+:float (cast:float a) f))")) << c.ast;
}

TEST(Sema, IntAssignedToFloatIsImplicit) {
  Checked c = body("q[0] = a;");
  EXPECT_EQ(c.errors, 0u) << c.diags;
  EXPECT_TRUE(contains(c.ast, "(assign = (index:float q 0) (cast:float a))")) << c.ast;
}

TEST(Sema, FloatToIntNeedsExplicitCast) {
  EXPECT_EQ(body("p[0] = f;").errors, 1u);
  EXPECT_EQ(body("p[0] = (int)f;").errors, 0u);
}

TEST(Sema, ComparisonsAreBoolAndPromote) {
  Checked c = body("if (a < f && !(b == a)) { p[0] = 1; }");
  EXPECT_EQ(c.errors, 0u) << c.diags;
  EXPECT_TRUE(contains(c.ast, "(&&:bool (<:bool (cast:float a) f) (!:bool (==:bool b a)))"))
      << c.ast;
}

TEST(Sema, BuiltinCallTypes) {
  Checked c = body("q[0] = sqrt(a) + min(a, f) + abs(g); p[0] = max(a, b) + abs(-b);");
  EXPECT_EQ(c.errors, 0u) << c.diags;
  EXPECT_TRUE(contains(c.ast, "(call:float sqrt (cast:float a))")) << c.ast;
  EXPECT_TRUE(contains(c.ast, "(call:float min (cast:float a) f)")) << c.ast;
  EXPECT_TRUE(contains(c.ast, "(call:int max a b)")) << c.ast;
}

TEST(Sema, CompoundAssignIntoFloatPromotes) {
  Checked c = body("float x = 1.0; x += a; q[0] = x;");
  EXPECT_EQ(c.errors, 0u) << c.diags;
  EXPECT_TRUE(contains(c.ast, "(assign += x (cast:float a))")) << c.ast;
}

TEST(Sema, SymbolsResolvedWithIds) {
  Checked c = sema("kernel k(int* o, int n) { int i = n; { int j = i; o[j] = 1; } }");
  ASSERT_EQ(c.errors, 0u) << c.diags;
  const Kernel &k = *c.prog->kernels[0];
  ASSERT_EQ(k.symbols.size(), 4u);
  EXPECT_EQ(k.symbols[0]->name, "o");
  EXPECT_EQ(k.symbols[0]->kind, Symbol::Kind::Param);
  EXPECT_EQ(k.symbols[2]->name, "i");
  EXPECT_EQ(k.symbols[3]->id, 3u);
  auto *decl = dyn_cast<VarDeclStmt>(k.body->stmts[0].get());
  ASSERT_NE(decl, nullptr);
  EXPECT_EQ(decl->sym, k.symbols[2].get());
  auto *ref = dyn_cast<VarRefExpr>(decl->init.get());
  ASSERT_NE(ref, nullptr);
  EXPECT_EQ(ref->sym, k.symbols[1].get());
}

TEST(Sema, ForInitScopedToLoop) {
  EXPECT_EQ(body("for (int i = 0; i < a; i++) { p[i] = i; } p[0] = i;").errors, 1u);
  EXPECT_EQ(
      body("for (int i = 0; i < a; i++) { } for (int i = 0; i < b; i++) { p[i] = 0; }").errors, 0u);
}

TEST(Sema, BreakContinueInsideLoops) {
  EXPECT_EQ(body("while (a > 0) { if (b > 0) { break; } continue; }").errors, 0u);
  EXPECT_EQ(body("for (;;) { break; }").errors, 0u);
  EXPECT_EQ(body("if (a > 0) { break; }").errors, 1u);
}

TEST(Sema, SharedArrays) {
  Checked c = body("shared float s[16]; s[a] = f; q[0] = s[0] + s[1];");
  EXPECT_EQ(c.errors, 0u) << c.diags;
  EXPECT_TRUE(contains(c.ast, "(index:float s 0)")) << c.ast;
  EXPECT_EQ(body("shared float s[16]; float x = s;").errors, 1u);
  EXPECT_EQ(body("for (shared int s[2]; a < 1; a++) { }").errors, 1u);
}

TEST(Sema, NoCascadingErrors) {
  // An undeclared name is reported once per use, never again for enclosing expressions.
  EXPECT_EQ(body("p[0] = (y + 1) * 2 + min(y, 3);").errors, 2u);
  EXPECT_EQ(body("int x = y + 1; p[0] = x * 2;").errors, 1u);
}

TEST(Sema, WarningsDoNotFail) {
  Checked c = body("int unusedVar = 1;");
  EXPECT_EQ(c.errors, 0u);
  EXPECT_EQ(c.warnings, 1u);
}

TEST(Sema, InitializerSeesOuterName) {
  Checked c = sema("kernel k(int* o, int x) { { int x = x + 1; o[0] = x; } }");
  EXPECT_EQ(c.errors, 0u) << c.diags;
  EXPECT_EQ(c.warnings, 1u) << c.diags; // shadowing
}

TEST(Sema, BoolCannotBeStoredOrComputed) {
  EXPECT_EQ(body("p[0] = a < b;").errors, 1u);
  EXPECT_EQ(body("p[0] = (a < b) + 1;").errors, 1u);
  EXPECT_EQ(body("if ((a < b) == (b < a)) { }").errors, 1u);
  EXPECT_EQ(body("if (-(a < b)) { }").errors, 1u);
}
