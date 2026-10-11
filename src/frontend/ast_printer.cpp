// Deterministic S-expression printer for the AST (`--emit=ast`).
//
// Statements go one per line, indented two spaces per nesting level; expressions are
// printed inline. Example:
//   (kernel saxpy (params (float a) (float* x) (float* y) (int n))
//     (block
//       (vardecl int i (+ (* blockIdx blockDim) threadIdx))
//       (if (< i n)
//         (assign = (index y i) (+ (* a (index x i)) (index y i))))))

#include "tinycu/ast.h"

#include <string>

namespace tinycu {

std::string TypeSpec::str() const {
  std::string s = base == Base::Int ? "int" : "float";
  if (pointer)
    s += '*';
  return s;
}

const char *binaryOpSpelling(BinaryOp op) {
  switch (op) {
  case BinaryOp::Add:
    return "+";
  case BinaryOp::Sub:
    return "-";
  case BinaryOp::Mul:
    return "*";
  case BinaryOp::Div:
    return "/";
  case BinaryOp::Rem:
    return "%";
  case BinaryOp::Lt:
    return "<";
  case BinaryOp::Le:
    return "<=";
  case BinaryOp::Gt:
    return ">";
  case BinaryOp::Ge:
    return ">=";
  case BinaryOp::Eq:
    return "==";
  case BinaryOp::Ne:
    return "!=";
  case BinaryOp::And:
    return "&&";
  case BinaryOp::Or:
    return "||";
  }
  return "?";
}

const char *assignOpSpelling(AssignOp op) {
  switch (op) {
  case AssignOp::Set:
    return "=";
  case AssignOp::Add:
    return "+=";
  case AssignOp::Sub:
    return "-=";
  case AssignOp::Mul:
    return "*=";
  case AssignOp::Div:
    return "/=";
  }
  return "?";
}

const char *builtinVarName(BuiltinVar v) {
  switch (v) {
  case BuiltinVar::ThreadIdx:
    return "threadIdx";
  case BuiltinVar::BlockIdx:
    return "blockIdx";
  case BuiltinVar::BlockDim:
    return "blockDim";
  case BuiltinVar::GridDim:
    return "gridDim";
  }
  return "?";
}

const char *builtinFnName(BuiltinFn f) {
  switch (f) {
  case BuiltinFn::Min:
    return "min";
  case BuiltinFn::Max:
    return "max";
  case BuiltinFn::Abs:
    return "abs";
  case BuiltinFn::Sqrt:
    return "sqrt";
  }
  return "?";
}

namespace {

class Printer {
public:
  std::string out;

  void expr(const Expr &e) {
    switch (e.kind()) {
    case ExprKind::IntLit:
      out += std::to_string(cast<IntLitExpr>(&e)->value);
      break;
    case ExprKind::FloatLit:
      out += cast<FloatLitExpr>(&e)->spelling;
      break;
    case ExprKind::BoolLit:
      out += cast<BoolLitExpr>(&e)->value ? "true" : "false";
      break;
    case ExprKind::VarRef:
      out += cast<VarRefExpr>(&e)->name;
      break;
    case ExprKind::BuiltinVar:
      out += builtinVarName(cast<BuiltinVarExpr>(&e)->var);
      break;
    case ExprKind::Index: {
      auto *ix = cast<IndexExpr>(&e);
      out += "(index ";
      expr(*ix->base);
      out += ' ';
      expr(*ix->index);
      out += ')';
      break;
    }
    case ExprKind::Unary: {
      auto *u = cast<UnaryExpr>(&e);
      out += u->op == UnaryOp::Neg ? "(- " : "(! ";
      expr(*u->operand);
      out += ')';
      break;
    }
    case ExprKind::Binary: {
      auto *b = cast<BinaryExpr>(&e);
      out += '(';
      out += binaryOpSpelling(b->op);
      out += ' ';
      expr(*b->lhs);
      out += ' ';
      expr(*b->rhs);
      out += ')';
      break;
    }
    case ExprKind::Cast: {
      auto *c = cast<CastExpr>(&e);
      out += "(cast " + c->to.str() + ' ';
      expr(*c->operand);
      out += ')';
      break;
    }
    case ExprKind::Call: {
      auto *c = cast<CallExpr>(&e);
      out += "(call ";
      out += builtinFnName(c->fn);
      for (const ExprPtr &a : c->args) {
        out += ' ';
        expr(*a);
      }
      out += ')';
      break;
    }
    }
  }

  void newline(int indent) {
    out += '\n';
    out.append(static_cast<size_t>(indent), ' ');
  }

  void stmt(const Stmt &s, int indent) {
    switch (s.kind()) {
    case StmtKind::VarDecl: {
      auto *d = cast<VarDeclStmt>(&s);
      out += "(vardecl ";
      if (d->shared)
        out += "shared ";
      out += d->typeSpec.str() + ' ' + d->name;
      if (d->arraySize)
        out += '[' + std::to_string(*d->arraySize) + ']';
      if (d->init) {
        out += ' ';
        expr(*d->init);
      }
      out += ')';
      break;
    }
    case StmtKind::Assign: {
      auto *a = cast<AssignStmt>(&s);
      out += "(assign ";
      out += assignOpSpelling(a->op);
      out += ' ';
      expr(*a->target);
      out += ' ';
      expr(*a->value);
      out += ')';
      break;
    }
    case StmtKind::If: {
      auto *i = cast<IfStmt>(&s);
      out += "(if ";
      expr(*i->cond);
      newline(indent + 2);
      stmt(*i->thenStmt, indent + 2);
      if (i->elseStmt) {
        newline(indent + 2);
        stmt(*i->elseStmt, indent + 2);
      }
      out += ')';
      break;
    }
    case StmtKind::For: {
      auto *f = cast<ForStmt>(&s);
      out += "(for ";
      if (f->init)
        stmt(*f->init, indent);
      else
        out += "()";
      out += ' ';
      if (f->cond)
        expr(*f->cond);
      else
        out += "()";
      out += ' ';
      if (f->step)
        stmt(*f->step, indent);
      else
        out += "()";
      newline(indent + 2);
      stmt(*f->body, indent + 2);
      out += ')';
      break;
    }
    case StmtKind::While: {
      auto *w = cast<WhileStmt>(&s);
      out += "(while ";
      expr(*w->cond);
      newline(indent + 2);
      stmt(*w->body, indent + 2);
      out += ')';
      break;
    }
    case StmtKind::Return:
      out += "(return)";
      break;
    case StmtKind::Break:
      out += "(break)";
      break;
    case StmtKind::Continue:
      out += "(continue)";
      break;
    case StmtKind::SyncThreads:
      out += "(syncthreads)";
      break;
    case StmtKind::Block: {
      auto *b = cast<BlockStmt>(&s);
      out += "(block";
      for (const StmtPtr &c : b->stmts) {
        newline(indent + 2);
        stmt(*c, indent + 2);
      }
      out += ')';
      break;
    }
    }
  }

  void kernel(const Kernel &k) {
    out += "(kernel " + k.name + " (params";
    for (const Param &p : k.params)
      out += " (" + p.typeSpec.str() + ' ' + p.name + ')';
    out += ')';
    newline(2);
    stmt(*k.body, 2);
    out += ")\n";
  }
};

} // namespace

std::string printAST(const Program &prog) {
  Printer p;
  for (const auto &k : prog.kernels)
    p.kernel(*k);
  return p.out;
}

std::string printExpr(const Expr &e) {
  Printer p;
  p.expr(e);
  return p.out;
}

} // namespace tinycu
