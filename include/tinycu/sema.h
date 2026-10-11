#pragma once

#include "tinycu/ast.h"
#include "tinycu/ast_visitor.h"
#include "tinycu/diag.h"
#include "tinycu/types.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace tinycu {

/// Semantic analysis (rules in docs/LANGUAGE.md §5). Resolves names to Symbols, sets
/// `Expr::type` on every expression, and inserts implicit int→float casts so later phases
/// never have to guess. Errors and warnings go to the DiagEngine; nothing throws.
class Sema : public ASTVisitor {
public:
  explicit Sema(DiagEngine &diag) : diag_(diag), tc_(TypeContext::get()) {}

  /// Returns true if no errors were reported.
  bool run(Program &prog);

  void visitKernel(Kernel &k) override;

  void visitVarDecl(VarDeclStmt &s) override;
  void visitAssign(AssignStmt &s) override;
  void visitIf(IfStmt &s) override;
  void visitFor(ForStmt &s) override;
  void visitWhile(WhileStmt &s) override;
  void visitBreak(BreakStmt &s) override;
  void visitContinue(ContinueStmt &s) override;
  void visitBlock(BlockStmt &s) override;

  void visitIntLit(IntLitExpr &e, ExprPtr &slot) override;
  void visitFloatLit(FloatLitExpr &e, ExprPtr &slot) override;
  void visitBoolLit(BoolLitExpr &e, ExprPtr &slot) override;
  void visitVarRef(VarRefExpr &e, ExprPtr &slot) override;
  void visitBuiltinVar(BuiltinVarExpr &e, ExprPtr &slot) override;
  void visitIndex(IndexExpr &e, ExprPtr &slot) override;
  void visitUnary(UnaryExpr &e, ExprPtr &slot) override;
  void visitBinary(BinaryExpr &e, ExprPtr &slot) override;
  void visitCast(CastExpr &e, ExprPtr &slot) override;
  void visitCall(CallExpr &e, ExprPtr &slot) override;

private:
  struct Scope {
    std::unordered_map<std::string, Symbol *> byName;
    std::vector<Symbol *> order; ///< declaration order, for deterministic warnings
  };

  // scopes & symbols
  void pushScope();
  void popScope();
  Symbol *lookup(const std::string &name) const;
  Symbol *declare(const std::string &name, const Type *ty, Symbol::Kind kind, SourceLoc loc);

  // helpers
  const Type *check(ExprPtr &e); ///< visit and return the type
  const Type *typeOf(TypeSpec ts) const;
  void coerce(ExprPtr &e, const Type *to);
  bool convertForAssign(ExprPtr &value, const Type *target, SourceLoc loc);
  void checkCondition(ExprPtr &cond, const char *what);
  void checkSubStmt(Stmt &s); ///< statement in its own scope (if/loop bodies)

  DiagEngine &diag_;
  TypeContext &tc_;
  Kernel *kernel_ = nullptr;
  std::vector<Scope> scopes_;
  unsigned loopDepth_ = 0;
  unsigned blockDepth_ = 0; ///< 1 = kernel top-level block
  bool inForInit_ = false;
  bool indexBase_ = false; ///< visiting the base of `p[i]`: pointers are allowed
  int64_t sharedBytes_ = 0;
};

} // namespace tinycu
