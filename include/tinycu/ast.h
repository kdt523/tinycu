#pragma once

#include "tinycu/diag.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace tinycu {

class Type; // semantic type, filled in by sema (types.h)

// ---------------------------------------------------------------------------
// LLVM-style RTTI helpers: every node class has `static bool classof(const Base*)`.
// ---------------------------------------------------------------------------
template <typename To, typename From> bool isa(const From *n) { return n && To::classof(n); }
template <typename To, typename From> To *dyn_cast(From *n) {
  return isa<To>(n) ? static_cast<To *>(n) : nullptr;
}
template <typename To, typename From> const To *dyn_cast(const From *n) {
  return isa<To>(n) ? static_cast<const To *>(n) : nullptr;
}
template <typename To, typename From> To *cast(From *n) { return static_cast<To *>(n); }
template <typename To, typename From> const To *cast(const From *n) {
  return static_cast<const To *>(n);
}

/// A type as written in source: `int`, `float`, `int*`, `float*`.
struct TypeSpec {
  enum class Base { Int, Float } base = Base::Int;
  bool pointer = false;
  std::string str() const;
};

// ===========================================================================
// Expressions
// ===========================================================================
enum class ExprKind {
  IntLit,
  FloatLit,
  BoolLit,
  VarRef,
  Index,
  Unary,
  Binary,
  Cast,
  BuiltinVar,
  Call
};

class Expr {
public:
  virtual ~Expr() = default;
  ExprKind kind() const { return kind_; }
  SourceLoc loc;
  const Type *type = nullptr; ///< set by sema

protected:
  Expr(ExprKind k, SourceLoc l) : loc(l), kind_(k) {}

private:
  ExprKind kind_;
};
using ExprPtr = std::unique_ptr<Expr>;

#define TINYCU_EXPR_CLASSOF(K)                                                                     \
  static bool classof(const Expr *e) { return e->kind() == ExprKind::K; }

class IntLitExpr : public Expr {
public:
  IntLitExpr(SourceLoc l, int32_t v) : Expr(ExprKind::IntLit, l), value(v) {}
  TINYCU_EXPR_CLASSOF(IntLit)
  int32_t value;
};

class FloatLitExpr : public Expr {
public:
  FloatLitExpr(SourceLoc l, float v, std::string sp)
      : Expr(ExprKind::FloatLit, l), value(v), spelling(std::move(sp)) {}
  TINYCU_EXPR_CLASSOF(FloatLit)
  float value;
  std::string spelling; ///< as written in source
};

class BoolLitExpr : public Expr {
public:
  BoolLitExpr(SourceLoc l, bool v) : Expr(ExprKind::BoolLit, l), value(v) {}
  TINYCU_EXPR_CLASSOF(BoolLit)
  bool value;
};

class VarRefExpr : public Expr {
public:
  VarRefExpr(SourceLoc l, std::string n) : Expr(ExprKind::VarRef, l), name(std::move(n)) {}
  TINYCU_EXPR_CLASSOF(VarRef)
  std::string name;
};

class IndexExpr : public Expr {
public:
  IndexExpr(SourceLoc l, ExprPtr b, ExprPtr i)
      : Expr(ExprKind::Index, l), base(std::move(b)), index(std::move(i)) {}
  TINYCU_EXPR_CLASSOF(Index)
  ExprPtr base;
  ExprPtr index;
};

enum class UnaryOp { Neg, Not };

class UnaryExpr : public Expr {
public:
  UnaryExpr(SourceLoc l, UnaryOp o, ExprPtr e)
      : Expr(ExprKind::Unary, l), op(o), operand(std::move(e)) {}
  TINYCU_EXPR_CLASSOF(Unary)
  UnaryOp op;
  ExprPtr operand;
};

enum class BinaryOp { Add, Sub, Mul, Div, Rem, Lt, Le, Gt, Ge, Eq, Ne, And, Or };
const char *binaryOpSpelling(BinaryOp op);

class BinaryExpr : public Expr {
public:
  BinaryExpr(SourceLoc l, BinaryOp o, ExprPtr a, ExprPtr b)
      : Expr(ExprKind::Binary, l), op(o), lhs(std::move(a)), rhs(std::move(b)) {}
  TINYCU_EXPR_CLASSOF(Binary)
  BinaryOp op;
  ExprPtr lhs;
  ExprPtr rhs;
};

/// Explicit `(T)e` cast, or an implicit conversion inserted by sema.
class CastExpr : public Expr {
public:
  CastExpr(SourceLoc l, TypeSpec t, ExprPtr e, bool impl = false)
      : Expr(ExprKind::Cast, l), to(t), operand(std::move(e)), implicit(impl) {}
  TINYCU_EXPR_CLASSOF(Cast)
  TypeSpec to;
  ExprPtr operand;
  bool implicit;
};

enum class BuiltinVar { ThreadIdx, BlockIdx, BlockDim, GridDim };
const char *builtinVarName(BuiltinVar v);

class BuiltinVarExpr : public Expr {
public:
  BuiltinVarExpr(SourceLoc l, BuiltinVar v) : Expr(ExprKind::BuiltinVar, l), var(v) {}
  TINYCU_EXPR_CLASSOF(BuiltinVar)
  BuiltinVar var;
};

enum class BuiltinFn { Min, Max, Abs, Sqrt };
const char *builtinFnName(BuiltinFn f);

class CallExpr : public Expr {
public:
  CallExpr(SourceLoc l, BuiltinFn f, std::vector<ExprPtr> a)
      : Expr(ExprKind::Call, l), fn(f), args(std::move(a)) {}
  TINYCU_EXPR_CLASSOF(Call)
  BuiltinFn fn;
  std::vector<ExprPtr> args;
};

#undef TINYCU_EXPR_CLASSOF

// ===========================================================================
// Statements
// ===========================================================================
enum class StmtKind {
  VarDecl,
  Assign,
  If,
  For,
  While,
  Return,
  Break,
  Continue,
  Block,
  SyncThreads
};

class Stmt {
public:
  virtual ~Stmt() = default;
  StmtKind kind() const { return kind_; }
  SourceLoc loc;

protected:
  Stmt(StmtKind k, SourceLoc l) : loc(l), kind_(k) {}

private:
  StmtKind kind_;
};
using StmtPtr = std::unique_ptr<Stmt>;

#define TINYCU_STMT_CLASSOF(K)                                                                     \
  static bool classof(const Stmt *s) { return s->kind() == StmtKind::K; }

class VarDeclStmt : public Stmt {
public:
  VarDeclStmt(SourceLoc l) : Stmt(StmtKind::VarDecl, l) {}
  TINYCU_STMT_CLASSOF(VarDecl)
  bool shared = false;
  TypeSpec typeSpec;
  std::string name;
  SourceLoc nameLoc;
  std::optional<int32_t> arraySize; ///< `[N]`
  ExprPtr init;                     ///< may be null
};

enum class AssignOp { Set, Add, Sub, Mul, Div };
const char *assignOpSpelling(AssignOp op);

/// `target op value`. `x++` / `x--` are desugared by the parser into `x += 1` / `x -= 1`.
class AssignStmt : public Stmt {
public:
  AssignStmt(SourceLoc l, AssignOp o, ExprPtr t, ExprPtr v)
      : Stmt(StmtKind::Assign, l), op(o), target(std::move(t)), value(std::move(v)) {}
  TINYCU_STMT_CLASSOF(Assign)
  AssignOp op;
  ExprPtr target; ///< VarRefExpr, IndexExpr, or BuiltinVarExpr (rejected by sema)
  ExprPtr value;
};

class IfStmt : public Stmt {
public:
  IfStmt(SourceLoc l) : Stmt(StmtKind::If, l) {}
  TINYCU_STMT_CLASSOF(If)
  ExprPtr cond;
  StmtPtr thenStmt;
  StmtPtr elseStmt; ///< may be null
};

class ForStmt : public Stmt {
public:
  ForStmt(SourceLoc l) : Stmt(StmtKind::For, l) {}
  TINYCU_STMT_CLASSOF(For)
  StmtPtr init; ///< VarDeclStmt or AssignStmt, may be null
  ExprPtr cond; ///< may be null (= true)
  StmtPtr step; ///< AssignStmt, may be null
  StmtPtr body;
};

class WhileStmt : public Stmt {
public:
  WhileStmt(SourceLoc l) : Stmt(StmtKind::While, l) {}
  TINYCU_STMT_CLASSOF(While)
  ExprPtr cond;
  StmtPtr body;
};

class ReturnStmt : public Stmt {
public:
  ReturnStmt(SourceLoc l) : Stmt(StmtKind::Return, l) {}
  TINYCU_STMT_CLASSOF(Return)
};

class BreakStmt : public Stmt {
public:
  BreakStmt(SourceLoc l) : Stmt(StmtKind::Break, l) {}
  TINYCU_STMT_CLASSOF(Break)
};

class ContinueStmt : public Stmt {
public:
  ContinueStmt(SourceLoc l) : Stmt(StmtKind::Continue, l) {}
  TINYCU_STMT_CLASSOF(Continue)
};

class BlockStmt : public Stmt {
public:
  BlockStmt(SourceLoc l) : Stmt(StmtKind::Block, l) {}
  TINYCU_STMT_CLASSOF(Block)
  std::vector<StmtPtr> stmts;
};

class SyncThreadsStmt : public Stmt {
public:
  SyncThreadsStmt(SourceLoc l) : Stmt(StmtKind::SyncThreads, l) {}
  TINYCU_STMT_CLASSOF(SyncThreads)
};

#undef TINYCU_STMT_CLASSOF

// ===========================================================================
// Top level
// ===========================================================================
struct Param {
  TypeSpec typeSpec;
  std::string name;
  SourceLoc loc;
};

struct Kernel {
  std::string name;
  SourceLoc loc;
  std::vector<Param> params;
  std::unique_ptr<BlockStmt> body;
};

struct Program {
  std::vector<std::unique_ptr<Kernel>> kernels;
};

/// Deterministic S-expression dump used by `--emit=ast` and golden tests.
std::string printAST(const Program &prog);
std::string printExpr(const Expr &e);

} // namespace tinycu
