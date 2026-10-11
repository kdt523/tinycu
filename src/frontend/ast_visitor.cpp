#include "tinycu/ast_visitor.h"

namespace tinycu {

void ASTVisitor::visitProgram(Program &p) {
  for (auto &k : p.kernels)
    visitKernel(*k);
}

void ASTVisitor::visitKernel(Kernel &k) { visitStmt(*k.body); }

void ASTVisitor::visitStmt(Stmt &s) {
  switch (s.kind()) {
  case StmtKind::VarDecl:
    return visitVarDecl(*cast<VarDeclStmt>(&s));
  case StmtKind::Assign:
    return visitAssign(*cast<AssignStmt>(&s));
  case StmtKind::If:
    return visitIf(*cast<IfStmt>(&s));
  case StmtKind::For:
    return visitFor(*cast<ForStmt>(&s));
  case StmtKind::While:
    return visitWhile(*cast<WhileStmt>(&s));
  case StmtKind::Return:
    return visitReturn(*cast<ReturnStmt>(&s));
  case StmtKind::Break:
    return visitBreak(*cast<BreakStmt>(&s));
  case StmtKind::Continue:
    return visitContinue(*cast<ContinueStmt>(&s));
  case StmtKind::Block:
    return visitBlock(*cast<BlockStmt>(&s));
  case StmtKind::SyncThreads:
    return visitSyncThreads(*cast<SyncThreadsStmt>(&s));
  }
}

void ASTVisitor::visitExpr(ExprPtr &slot) {
  Expr *e = slot.get();
  switch (e->kind()) {
  case ExprKind::IntLit:
    return visitIntLit(*cast<IntLitExpr>(e), slot);
  case ExprKind::FloatLit:
    return visitFloatLit(*cast<FloatLitExpr>(e), slot);
  case ExprKind::BoolLit:
    return visitBoolLit(*cast<BoolLitExpr>(e), slot);
  case ExprKind::VarRef:
    return visitVarRef(*cast<VarRefExpr>(e), slot);
  case ExprKind::BuiltinVar:
    return visitBuiltinVar(*cast<BuiltinVarExpr>(e), slot);
  case ExprKind::Index:
    return visitIndex(*cast<IndexExpr>(e), slot);
  case ExprKind::Unary:
    return visitUnary(*cast<UnaryExpr>(e), slot);
  case ExprKind::Binary:
    return visitBinary(*cast<BinaryExpr>(e), slot);
  case ExprKind::Cast:
    return visitCast(*cast<CastExpr>(e), slot);
  case ExprKind::Call:
    return visitCall(*cast<CallExpr>(e), slot);
  }
}

void ASTVisitor::visitVarDecl(VarDeclStmt &s) {
  if (s.init)
    visitExpr(s.init);
}

void ASTVisitor::visitAssign(AssignStmt &s) {
  visitExpr(s.target);
  visitExpr(s.value);
}

void ASTVisitor::visitIf(IfStmt &s) {
  visitExpr(s.cond);
  visitStmt(*s.thenStmt);
  if (s.elseStmt)
    visitStmt(*s.elseStmt);
}

void ASTVisitor::visitFor(ForStmt &s) {
  if (s.init)
    visitStmt(*s.init);
  if (s.cond)
    visitExpr(s.cond);
  if (s.step)
    visitStmt(*s.step);
  visitStmt(*s.body);
}

void ASTVisitor::visitWhile(WhileStmt &s) {
  visitExpr(s.cond);
  visitStmt(*s.body);
}

void ASTVisitor::visitBlock(BlockStmt &s) {
  for (auto &st : s.stmts)
    visitStmt(*st);
}

void ASTVisitor::visitIndex(IndexExpr &e, ExprPtr &) {
  visitExpr(e.base);
  visitExpr(e.index);
}

void ASTVisitor::visitUnary(UnaryExpr &e, ExprPtr &) { visitExpr(e.operand); }

void ASTVisitor::visitBinary(BinaryExpr &e, ExprPtr &) {
  visitExpr(e.lhs);
  visitExpr(e.rhs);
}

void ASTVisitor::visitCast(CastExpr &e, ExprPtr &) { visitExpr(e.operand); }

void ASTVisitor::visitCall(CallExpr &e, ExprPtr &) {
  for (auto &a : e.args)
    visitExpr(a);
}

} // namespace tinycu
