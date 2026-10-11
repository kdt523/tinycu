#include "tinycu/sema.h"

#include <set>

namespace tinycu {

namespace {
constexpr int32_t kMaxSharedElems = 12288; ///< 48 KB of 4-byte elements
constexpr int64_t kMaxSharedBytes = 49152; ///< per kernel

std::string quoted(const Type *t) { return "'" + t->str() + "'"; }
} // namespace

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------
bool Sema::run(Program &prog) {
  unsigned errorsBefore = diag_.errorCount();
  std::set<std::string> names;
  for (auto &k : prog.kernels) {
    if (!names.insert(k->name).second)
      diag_.error(k->loc, "redefinition of kernel '" + k->name + "'");
    visitKernel(*k);
  }
  return diag_.errorCount() == errorsBefore;
}

void Sema::visitKernel(Kernel &k) {
  kernel_ = &k;
  scopes_.clear();
  loopDepth_ = 0;
  blockDepth_ = 0;
  sharedBytes_ = 0;
  pushScope(); // parameters live in the outermost scope
  for (Param &p : k.params) {
    p.sym = declare(p.name, typeOf(p.typeSpec), Symbol::Kind::Param, p.loc);
    p.sym->assigned = true;
  }
  visitStmt(*k.body);
  popScope();
  kernel_ = nullptr;
}

// ---------------------------------------------------------------------------
// Scopes
// ---------------------------------------------------------------------------
void Sema::pushScope() { scopes_.emplace_back(); }

void Sema::popScope() {
  for (Symbol *s : scopes_.back().order)
    if (s->kind != Symbol::Kind::Param && !s->used)
      diag_.warning(s->loc, "unused variable '" + s->name + "'");
  scopes_.pop_back();
}

Symbol *Sema::lookup(const std::string &name) const {
  for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
    auto f = it->byName.find(name);
    if (f != it->byName.end())
      return f->second;
  }
  return nullptr;
}

Symbol *Sema::declare(const std::string &name, const Type *ty, Symbol::Kind kind, SourceLoc loc) {
  auto sym = std::make_unique<Symbol>();
  sym->name = name;
  sym->type = ty;
  sym->kind = kind;
  sym->loc = loc;
  sym->id = static_cast<unsigned>(kernel_->symbols.size());
  Symbol *raw = sym.get();
  kernel_->symbols.push_back(std::move(sym));

  Scope &cur = scopes_.back();
  auto prev = cur.byName.find(name);
  if (prev != cur.byName.end()) {
    diag_.error(loc, "redeclaration of '" + name + "'");
    diag_.note(prev->second->loc, "previous declaration of '" + name + "' is here");
    raw->used = true; // don't also warn "unused" for the rejected duplicate
    return raw;
  }
  if (Symbol *outer = lookup(name)) {
    diag_.warning(loc, "declaration of '" + name + "' shadows an outer variable");
    diag_.note(outer->loc, "shadowed declaration is here");
  }
  cur.byName[name] = raw;
  cur.order.push_back(raw);
  return raw;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const Type *Sema::check(ExprPtr &e) {
  visitExpr(e);
  return e->type;
}

const Type *Sema::typeOf(TypeSpec ts) const {
  const Type *base = ts.base == TypeSpec::Base::Int ? tc_.intTy() : tc_.floatTy();
  return ts.pointer ? tc_.ptrTo(base) : base;
}

// Wrap `e` in an implicit conversion. Only int -> float is ever implicit.
void Sema::coerce(ExprPtr &e, const Type *to) {
  if (e->type == to)
    return;
  SourceLoc loc = e->loc;
  auto c = std::make_unique<CastExpr>(loc, TypeSpec{TypeSpec::Base::Float, false}, std::move(e),
                                      /*implicit=*/true);
  c->type = to;
  e = std::move(c);
}

bool Sema::convertForAssign(ExprPtr &value, const Type *target, SourceLoc loc) {
  const Type *vt = value->type;
  if (vt == target)
    return true;
  if (target->isFloat() && vt->isInt()) {
    coerce(value, target);
    return true;
  }
  if (target->isInt() && vt->isFloat()) {
    diag_.error(loc, "implicit conversion from float to int loses precision; use (int)");
    return false;
  }
  diag_.error(loc, "cannot assign a value of type " + quoted(vt) + " to " + quoted(target));
  return false;
}

void Sema::checkCondition(ExprPtr &cond, const char *what) {
  const Type *t = check(cond);
  if (!t->isError() && !t->isBool())
    diag_.error(cond->loc, std::string("condition of '") + what + "' must be bool, found " +
                               quoted(t) +
                               " (no implicit truthiness; compare explicitly, e.g. "
                               "'x != 0')");
}

// if/loop bodies get their own scope and are never "kernel top level".
void Sema::checkSubStmt(Stmt &s) {
  pushScope();
  ++blockDepth_;
  visitStmt(s);
  --blockDepth_;
  popScope();
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------
void Sema::visitBlock(BlockStmt &s) {
  pushScope();
  ++blockDepth_;
  for (auto &st : s.stmts)
    visitStmt(*st);
  --blockDepth_;
  popScope();
}

void Sema::visitVarDecl(VarDeclStmt &s) {
  const Type *elem = typeOf({s.typeSpec.base, false});

  if (s.shared) {
    bool ok = true;
    if (blockDepth_ != 1 || inForInit_) {
      diag_.error(s.loc, "shared arrays must be declared at kernel top level");
      ok = false;
    }
    if (s.typeSpec.pointer) {
      diag_.error(s.loc, "shared variable '" + s.name + "' cannot be a pointer");
      ok = false;
    }
    if (!s.arraySize) {
      diag_.error(s.nameLoc,
                  "shared variable '" + s.name + "' must be an array with a constant size");
      ok = false;
    } else if (*s.arraySize < 1 || *s.arraySize > kMaxSharedElems) {
      diag_.error(s.nameLoc,
                  "shared array size must be between 1 and 12288 elements (48 KB), got " +
                      std::to_string(*s.arraySize));
      ok = false;
    } else if (ok) {
      sharedBytes_ += 4 * static_cast<int64_t>(*s.arraySize);
      if (sharedBytes_ > kMaxSharedBytes) {
        diag_.error(s.nameLoc, "total shared memory of kernel '" + kernel_->name +
                                   "' exceeds 48 KB (" + std::to_string(sharedBytes_) + " bytes)");
        ok = false;
      }
    }
    if (s.init) {
      diag_.error(s.init->loc, "shared array cannot have an initializer");
      ok = false;
    }
    const Type *ty = ok ? tc_.sharedArray(elem, *s.arraySize) : tc_.errorTy();
    s.sym = declare(s.name, ty, Symbol::Kind::Shared, s.nameLoc);
    s.sym->assigned = true;
    return;
  }

  const Type *ty = typeOf(s.typeSpec);
  if (s.arraySize) {
    diag_.error(s.nameLoc, "local arrays are not supported; use a 'shared' array");
    ty = tc_.errorTy();
  } else if (s.typeSpec.pointer) {
    diag_.error(
        s.loc, "local pointer variables are not supported; pointers can only be kernel parameters");
    ty = tc_.errorTy();
  }
  // The initializer is checked before the name is declared: `int x = x;` refers to an outer x.
  if (s.init) {
    const Type *it = check(s.init);
    if (!it->isError() && !ty->isError())
      convertForAssign(s.init, ty, s.init->loc);
  }
  s.sym = declare(s.name, ty, Symbol::Kind::Local, s.nameLoc);
  s.sym->assigned = s.init != nullptr;
}

void Sema::visitAssign(AssignStmt &s) {
  Expr *t = s.target.get();
  const Type *tt = nullptr;
  Symbol *targetSym = nullptr;

  if (auto *bv = dyn_cast<BuiltinVarExpr>(t)) {
    diag_.error(bv->loc, std::string("cannot assign to read-only builtin variable '") +
                             builtinVarName(bv->var) + "'");
    bv->type = tc_.intTy();
    check(s.value);
    return;
  }
  if (auto *v = dyn_cast<VarRefExpr>(t)) {
    Symbol *sym = lookup(v->name);
    if (!sym) {
      diag_.error(v->loc, "use of undeclared identifier '" + v->name + "'");
      v->type = tc_.errorTy();
      check(s.value);
      return;
    }
    v->sym = sym;
    if (sym->type->isPtr() || sym->type->isSharedArray()) {
      diag_.error(v->loc, sym->type->isPtr() ? "cannot assign to pointer '" + v->name + "'"
                                             : "cannot assign to shared array '" + v->name +
                                                   "'; assign to an element instead");
      v->type = tc_.errorTy();
      check(s.value);
      return;
    }
    if (s.op != AssignOp::Set) { // compound assignment also reads the target
      sym->used = true;
      if (!sym->assigned && sym->kind == Symbol::Kind::Local) {
        diag_.warning(v->loc, "variable '" + v->name + "' is used uninitialized");
        sym->assigned = true;
      }
    }
    v->type = sym->type;
    tt = sym->type;
    targetSym = sym;
  } else {
    tt = check(s.target); // IndexExpr
  }

  const Type *vt = check(s.value);
  if (!tt->isError() && !vt->isError()) {
    if (s.op == AssignOp::Set) {
      convertForAssign(s.value, tt, s.value->loc);
    } else if (!vt->isArithmetic()) {
      diag_.error(s.value->loc, std::string("invalid operand to '") + assignOpSpelling(s.op) +
                                    "': " + quoted(vt));
    } else if (tt->isInt() && vt->isFloat()) {
      // x op= v means x = x op v; with float v the result is float.
      diag_.error(s.value->loc, "implicit conversion from float to int loses precision; use (int)");
    } else if (tt->isFloat() && vt->isInt()) {
      coerce(s.value, tt);
    }
  }
  if (targetSym)
    targetSym->assigned = true;
}

void Sema::visitIf(IfStmt &s) {
  checkCondition(s.cond, "if");
  checkSubStmt(*s.thenStmt);
  if (s.elseStmt)
    checkSubStmt(*s.elseStmt);
}

void Sema::visitFor(ForStmt &s) {
  pushScope(); // the init declaration is scoped to the loop
  if (s.init) {
    inForInit_ = true;
    visitStmt(*s.init);
    inForInit_ = false;
  }
  if (s.cond)
    checkCondition(s.cond, "for");
  if (s.step)
    visitStmt(*s.step);
  ++loopDepth_;
  checkSubStmt(*s.body);
  --loopDepth_;
  popScope();
}

void Sema::visitWhile(WhileStmt &s) {
  checkCondition(s.cond, "while");
  ++loopDepth_;
  checkSubStmt(*s.body);
  --loopDepth_;
}

void Sema::visitBreak(BreakStmt &s) {
  if (loopDepth_ == 0)
    diag_.error(s.loc, "'break' statement not in a loop");
}

void Sema::visitContinue(ContinueStmt &s) {
  if (loopDepth_ == 0)
    diag_.error(s.loc, "'continue' statement not in a loop");
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------
void Sema::visitIntLit(IntLitExpr &e, ExprPtr &) { e.type = tc_.intTy(); }
void Sema::visitFloatLit(FloatLitExpr &e, ExprPtr &) { e.type = tc_.floatTy(); }
void Sema::visitBoolLit(BoolLitExpr &e, ExprPtr &) { e.type = tc_.boolTy(); }
void Sema::visitBuiltinVar(BuiltinVarExpr &e, ExprPtr &) { e.type = tc_.intTy(); }

void Sema::visitVarRef(VarRefExpr &e, ExprPtr &) {
  bool asIndexBase = indexBase_;
  indexBase_ = false;
  Symbol *sym = lookup(e.name);
  if (!sym) {
    diag_.error(e.loc, "use of undeclared identifier '" + e.name + "'");
    e.type = tc_.errorTy();
    return;
  }
  e.sym = sym;
  sym->used = true;
  if (sym->type->isIndexable() && !asIndexBase) {
    diag_.error(e.loc, std::string(sym->type->isPtr() ? "pointer '" : "shared array '") + e.name +
                           "' can only be indexed (no pointer arithmetic, comparison or copies)");
    e.type = tc_.errorTy();
    return;
  }
  if (!sym->assigned && sym->kind == Symbol::Kind::Local) {
    diag_.warning(e.loc, "variable '" + e.name + "' is used uninitialized");
    sym->assigned = true; // warn once
  }
  e.type = sym->type;
}

void Sema::visitIndex(IndexExpr &e, ExprPtr &) {
  indexBase_ = isa<VarRefExpr>(e.base.get());
  const Type *bt = check(e.base);
  indexBase_ = false;
  const Type *it = check(e.index);
  e.type = tc_.errorTy();
  if (bt->isError() || it->isError())
    return;
  if (!bt->isIndexable()) {
    diag_.error(e.loc,
                "subscripted value of type " + quoted(bt) + " is not a pointer or shared array");
    return;
  }
  if (!it->isInt()) {
    diag_.error(e.index->loc, "array index must be int, found " + quoted(it));
    return;
  }
  e.type = bt->elem();
}

void Sema::visitUnary(UnaryExpr &e, ExprPtr &) {
  const Type *t = check(e.operand);
  e.type = tc_.errorTy();
  if (t->isError())
    return;
  if (e.op == UnaryOp::Neg) {
    if (!t->isArithmetic()) {
      diag_.error(e.loc, "invalid operand to unary '-': " + quoted(t));
      return;
    }
    e.type = t;
  } else {
    if (!t->isBool()) {
      diag_.error(e.loc,
                  "operand of '!' must be bool, found " + quoted(t) + " (no implicit truthiness)");
      return;
    }
    e.type = tc_.boolTy();
  }
}

void Sema::visitBinary(BinaryExpr &e, ExprPtr &) {
  const Type *lt = check(e.lhs);
  const Type *rt = check(e.rhs);
  e.type = tc_.errorTy();
  if (lt->isError() || rt->isError())
    return;
  std::string op = binaryOpSpelling(e.op);
  std::string have = " (have " + quoted(lt) + " and " + quoted(rt) + ")";

  switch (e.op) {
  case BinaryOp::Add:
  case BinaryOp::Sub:
  case BinaryOp::Mul:
  case BinaryOp::Div:
  case BinaryOp::Lt:
  case BinaryOp::Le:
  case BinaryOp::Gt:
  case BinaryOp::Ge:
  case BinaryOp::Eq:
  case BinaryOp::Ne: {
    bool cmp = e.op >= BinaryOp::Lt && e.op <= BinaryOp::Ne;
    if (!lt->isArithmetic() || !rt->isArithmetic()) {
      diag_.error(e.loc, std::string(cmp ? "invalid operands to comparison '"
                                         : "invalid operands to binary '") +
                             op + "'" + have);
      return;
    }
    const Type *common = lt;
    if (lt != rt) { // int (op) float: the int side converts to float
      common = tc_.floatTy();
      coerce(e.lhs, common);
      coerce(e.rhs, common);
    }
    e.type = cmp ? tc_.boolTy() : common;
    return;
  }
  case BinaryOp::Rem:
    if (!lt->isInt() || !rt->isInt()) {
      diag_.error(e.loc, "'%' requires int operands" + have);
      return;
    }
    e.type = tc_.intTy();
    return;
  case BinaryOp::And:
  case BinaryOp::Or:
    if (!lt->isBool() || !rt->isBool()) {
      diag_.error(e.loc,
                  "operands of '" + op + "' must be bool" + have + "; no implicit truthiness");
      return;
    }
    e.type = tc_.boolTy();
    return;
  }
}

void Sema::visitCast(CastExpr &e, ExprPtr &) {
  const Type *t = check(e.operand);
  e.type = tc_.errorTy();
  if (e.to.pointer) {
    diag_.error(e.loc, "cannot cast to pointer type '" + e.to.str() + "'");
    return;
  }
  if (t->isError())
    return;
  if (!t->isArithmetic()) {
    diag_.error(e.loc, "cannot cast " + quoted(t) + " to '" + e.to.str() + "'");
    return;
  }
  e.type = typeOf(e.to);
}

void Sema::visitCall(CallExpr &e, ExprPtr &) {
  bool argError = false;
  for (auto &a : e.args)
    argError |= check(a)->isError();
  e.type = tc_.errorTy();
  if (argError)
    return;
  std::string name = builtinFnName(e.fn);
  size_t want = (e.fn == BuiltinFn::Min || e.fn == BuiltinFn::Max) ? 2 : 1;
  if (e.args.size() != want) {
    diag_.error(e.loc, "'" + name + "' expects " + std::to_string(want) + " argument" +
                           (want == 1 ? "" : "s") + ", got " + std::to_string(e.args.size()));
    return;
  }
  for (size_t i = 0; i < e.args.size(); ++i) {
    if (!e.args[i]->type->isArithmetic()) {
      diag_.error(e.args[i]->loc, "argument " + std::to_string(i + 1) + " of '" + name +
                                      "' must be int or float, found " + quoted(e.args[i]->type));
      return;
    }
  }
  switch (e.fn) {
  case BuiltinFn::Min:
  case BuiltinFn::Max:
    if (e.args[0]->type != e.args[1]->type) {
      coerce(e.args[0], tc_.floatTy());
      coerce(e.args[1], tc_.floatTy());
    }
    e.type = e.args[0]->type;
    return;
  case BuiltinFn::Abs:
    e.type = e.args[0]->type;
    return;
  case BuiltinFn::Sqrt:
    coerce(e.args[0], tc_.floatTy());
    e.type = tc_.floatTy();
    return;
  }
}

} // namespace tinycu
