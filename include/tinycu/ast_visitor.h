#pragma once

#include "tinycu/ast.h"

namespace tinycu {

/// Kind-dispatching AST visitor. Default implementations recurse into children, so a
/// subclass overrides only what it cares about. Expression visitors receive the owning
/// `ExprPtr &slot` as well, which lets a visitor replace the node (e.g. sema wrapping an
/// operand in an implicit cast).
class ASTVisitor {
public:
  virtual ~ASTVisitor() = default;

  void visitProgram(Program &p);
  void visitStmt(Stmt &s);
  void visitExpr(ExprPtr &slot);

  virtual void visitKernel(Kernel &k);

  virtual void visitVarDecl(VarDeclStmt &s);
  virtual void visitAssign(AssignStmt &s);
  virtual void visitIf(IfStmt &s);
  virtual void visitFor(ForStmt &s);
  virtual void visitWhile(WhileStmt &s);
  virtual void visitReturn(ReturnStmt &) {}
  virtual void visitBreak(BreakStmt &) {}
  virtual void visitContinue(ContinueStmt &) {}
  virtual void visitBlock(BlockStmt &s);
  virtual void visitSyncThreads(SyncThreadsStmt &) {}

  virtual void visitIntLit(IntLitExpr &, ExprPtr &) {}
  virtual void visitFloatLit(FloatLitExpr &, ExprPtr &) {}
  virtual void visitBoolLit(BoolLitExpr &, ExprPtr &) {}
  virtual void visitVarRef(VarRefExpr &, ExprPtr &) {}
  virtual void visitBuiltinVar(BuiltinVarExpr &, ExprPtr &) {}
  virtual void visitIndex(IndexExpr &e, ExprPtr &slot);
  virtual void visitUnary(UnaryExpr &e, ExprPtr &slot);
  virtual void visitBinary(BinaryExpr &e, ExprPtr &slot);
  virtual void visitCast(CastExpr &e, ExprPtr &slot);
  virtual void visitCall(CallExpr &e, ExprPtr &slot);
};

} // namespace tinycu
