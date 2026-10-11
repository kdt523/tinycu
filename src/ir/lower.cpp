// Lowering: type-checked AST -> three-address IR in basic blocks.
//
// - Each Symbol gets one fixed named vreg (`%i`); temporaries are numbered. Assignment is `mov`.
// - `&&` / `||` are short-circuited with control flow.
// - if/while/for become the standard block patterns; `return` branches to a single `exit`.
// - Blocks are appended to the function when code starts being emitted into them, so the
//   printed order follows the source.
#include "tinycu/ast.h"
#include "tinycu/ir.h"
#include "tinycu/types.h"

#include <cassert>
#include <unordered_map>

namespace tinycu {

using namespace ir;

namespace {

IRType irType(const Type *t) {
  switch (t->kind()) {
  case Type::Kind::Bool:
    return IRType::I1;
  case Type::Kind::Int:
    return IRType::I32;
  case Type::Kind::Float:
    return IRType::F32;
  case Type::Kind::Ptr:
    return IRType::PtrGlobal;
  case Type::Kind::SharedArray:
    return IRType::PtrShared;
  default:
    assert(false && "lowering an ill-typed program");
    return IRType::I32;
  }
}

class Lowerer {
public:
  explicit Lowerer(const Kernel &k) : k_(k) {}

  std::unique_ptr<Function> run() {
    fn_ = std::make_unique<Function>();
    fn_->name = k_.name;
    symReg_.assign(k_.symbols.size(), ~0u);
    exit_ = newBlock("exit", /*numbered=*/false);
    start(newBlock("entry", false));
    for (size_t i = 0; i < k_.params.size(); ++i) {
      const Param &p = k_.params[i];
      IRType t = irType(p.sym->type);
      uint32_t r = fn_->newVReg(t, p.name);
      symReg_[p.sym->id] = r;
      IRType elem = t == IRType::PtrGlobal ? irType(p.sym->type->elem()) : t;
      fn_->params.push_back({p.name, t, elem, r});
      Inst in;
      in.op = Op::Param;
      in.type = t;
      in.dst = Value::vreg(r);
      in.aux = static_cast<int32_t>(i);
      emit(in);
    }
    stmt(*k_.body);
    if (cur_)
      br(exit_);
    start(exit_);
    Inst ret;
    ret.op = Op::Ret;
    emit(ret);
    return std::move(fn_);
  }

private:
  // ---- blocks ----
  BasicBlock *newBlock(const std::string &label, bool numbered = true) {
    auto bb = std::make_unique<BasicBlock>();
    bb->label = numbered ? label + "." + std::to_string(labelCounter_) : label;
    BasicBlock *raw = bb.get();
    pending_[raw] = std::move(bb);
    return raw;
  }
  /// Append `bb` to the function and make it the insertion point.
  void start(BasicBlock *bb) {
    auto it = pending_.find(bb);
    assert(it != pending_.end());
    fn_->blocks.push_back(std::move(it->second));
    pending_.erase(it);
    cur_ = bb;
  }
  /// Code after return/break/continue is unreachable but still lowered (into a fresh block).
  void ensureBlock() {
    if (!cur_) {
      ++labelCounter_;
      start(newBlock("dead"));
    }
  }
  void emit(const Inst &in) {
    ensureBlock();
    cur_->insts.push_back(in);
    if (in.isTerminator())
      cur_ = nullptr;
  }
  void br(BasicBlock *target) {
    Inst in;
    in.op = Op::Br;
    in.t = target;
    emit(in);
  }
  void condbr(Value p, BasicBlock *t, BasicBlock *f) {
    Inst in;
    in.op = Op::CondBr;
    in.type = IRType::I1;
    in.a = p;
    in.t = t;
    in.f = f;
    emit(in);
  }

  // ---- instruction helpers ----
  Value temp(IRType t) { return Value::vreg(fn_->newVReg(t)); }
  Value binop(Op op, IRType resTy, IRType opTy, Value a, Value b) {
    Inst in;
    in.op = op;
    in.type = opTy;
    in.dst = temp(resTy);
    in.a = a;
    in.b = b;
    emit(in);
    return in.dst;
  }
  Value unop(Op op, IRType resTy, IRType opTy, Value a) {
    return binop(op, resTy, opTy, a, Value{});
  }
  void mov(Value dst, Value src, IRType t) {
    Inst in;
    in.op = Op::Mov;
    in.type = t;
    in.dst = dst;
    in.a = src;
    emit(in);
  }

  uint32_t regFor(const Symbol *s) {
    uint32_t &r = symReg_[s->id];
    if (r == ~0u)
      r = fn_->newVReg(irType(s->type), s->name);
    return r;
  }

  // ---- addresses ----
  struct Address {
    Value addr;
    MemSpace space;
    IRType elem;
  };
  Address address(const IndexExpr &e) {
    const auto *base = cast<VarRefExpr>(e.base.get());
    Value baseVal;
    MemSpace space;
    if (base->sym->kind == Symbol::Kind::Shared) {
      baseVal = Value::shared(sharedIndex_.at(base->sym->id));
      space = MemSpace::Shared;
    } else {
      baseVal = Value::vreg(regFor(base->sym));
      space = MemSpace::Global;
    }
    Value idx = expr(*e.index);
    Inst in;
    in.op = Op::Addr;
    in.type = IRType::I64;
    in.dst = temp(IRType::I64);
    in.a = baseVal;
    in.b = idx;
    in.aux = 4;
    emit(in);
    return {in.dst, space, irType(e.type)};
  }
  Value load(const Address &ad) {
    Inst in;
    in.op = Op::Load;
    in.type = ad.elem;
    in.dst = temp(ad.elem);
    in.a = ad.addr;
    in.aux = static_cast<int32_t>(ad.space);
    emit(in);
    return in.dst;
  }
  void store(const Address &ad, Value v) {
    Inst in;
    in.op = Op::Store;
    in.type = ad.elem;
    in.a = ad.addr;
    in.b = v;
    in.aux = static_cast<int32_t>(ad.space);
    emit(in);
  }

  // ---- expressions ----
  Value expr(const Expr &e) {
    switch (e.kind()) {
    case ExprKind::IntLit:
      return Value::immI32(cast<IntLitExpr>(&e)->value);
    case ExprKind::FloatLit:
      return Value::immF32(cast<FloatLitExpr>(&e)->value);
    case ExprKind::BoolLit:
      return Value::immBool(cast<BoolLitExpr>(&e)->value);
    case ExprKind::VarRef:
      return Value::vreg(regFor(cast<VarRefExpr>(&e)->sym));
    case ExprKind::BuiltinVar: {
      static const SReg map[] = {SReg::Tid, SReg::Ctaid, SReg::Ntid, SReg::Nctaid};
      Inst in;
      in.op = Op::Sreg;
      in.type = IRType::I32;
      in.dst = temp(IRType::I32);
      in.aux = static_cast<int32_t>(map[static_cast<int>(cast<BuiltinVarExpr>(&e)->var)]);
      emit(in);
      return in.dst;
    }
    case ExprKind::Index:
      return load(address(*cast<IndexExpr>(&e)));
    case ExprKind::Unary: {
      const auto *u = cast<UnaryExpr>(&e);
      Value v = expr(*u->operand);
      IRType t = irType(e.type);
      return unop(u->op == UnaryOp::Neg ? Op::Neg : Op::Not, t, t, v);
    }
    case ExprKind::Binary:
      return binary(*cast<BinaryExpr>(&e));
    case ExprKind::Cast: {
      const auto *c = cast<CastExpr>(&e);
      Value v = expr(*c->operand);
      IRType from = irType(c->operand->type), to = irType(e.type);
      if (from == to)
        return v;
      Op op = to == IRType::F32 ? Op::CvtF32I32 : Op::CvtI32F32;
      return unop(op, to, to, v);
    }
    case ExprKind::Call: {
      const auto *c = cast<CallExpr>(&e);
      static const Op ops[] = {Op::Min, Op::Max, Op::Abs, Op::Sqrt};
      Op op = ops[static_cast<int>(c->fn)];
      IRType t = irType(e.type);
      Value a = expr(*c->args[0]);
      Value b = c->args.size() > 1 ? expr(*c->args[1]) : Value{};
      return binop(op, t, t, a, b);
    }
    }
    return Value{};
  }

  Value binary(const BinaryExpr &e) {
    if (e.op == BinaryOp::And || e.op == BinaryOp::Or)
      return shortCircuit(e);
    Value l = expr(*e.lhs);
    Value r = expr(*e.rhs);
    IRType opTy = irType(e.lhs->type);
    IRType resTy = irType(e.type);
    Op op;
    switch (e.op) {
    case BinaryOp::Add:
      op = Op::Add;
      break;
    case BinaryOp::Sub:
      op = Op::Sub;
      break;
    case BinaryOp::Mul:
      op = Op::Mul;
      break;
    case BinaryOp::Div:
      op = Op::Div;
      break;
    case BinaryOp::Rem:
      op = Op::Rem;
      break;
    case BinaryOp::Lt:
      op = Op::CmpLt;
      break;
    case BinaryOp::Le:
      op = Op::CmpLe;
      break;
    case BinaryOp::Gt:
      op = Op::CmpGt;
      break;
    case BinaryOp::Ge:
      op = Op::CmpGe;
      break;
    case BinaryOp::Eq:
      op = Op::CmpEq;
      break;
    case BinaryOp::Ne:
      op = Op::CmpNe;
      break;
    default:
      op = Op::Add;
      break;
    }
    return binop(op, resTy, opTy, l, r);
  }

  // res = lhs; if (and ? res : !res) res = rhs;   -- rhs is evaluated only when needed.
  Value shortCircuit(const BinaryExpr &e) {
    bool isAnd = e.op == BinaryOp::And;
    ++labelCounter_;
    BasicBlock *rhsBB = newBlock(isAnd ? "and.rhs" : "or.rhs");
    BasicBlock *endBB = newBlock(isAnd ? "and.end" : "or.end");
    Value res = temp(IRType::I1);
    Value l = expr(*e.lhs);
    mov(res, l, IRType::I1);
    if (isAnd)
      condbr(l, rhsBB, endBB);
    else
      condbr(l, endBB, rhsBB);
    start(rhsBB);
    Value r = expr(*e.rhs);
    mov(res, r, IRType::I1);
    br(endBB);
    start(endBB);
    return res;
  }

  // ---- statements ----
  void stmt(const Stmt &s) {
    switch (s.kind()) {
    case StmtKind::VarDecl: {
      const auto *d = cast<VarDeclStmt>(&s);
      if (d->shared) {
        uint32_t idx = static_cast<uint32_t>(fn_->shared.size());
        IRType elem = irType(d->sym->type->elem());
        fn_->shared.push_back({d->name, elem, d->sym->type->arraySize()});
        sharedIndex_[d->sym->id] = idx;
        return;
      }
      // A local without an initializer starts at zero: TinyCU defines it (sema still warns
      // about reading it), so the interpreter and the GPU agree.
      uint32_t r = regFor(d->sym);
      IRType t = irType(d->sym->type);
      Value v =
          d->init ? expr(*d->init) : (t == IRType::F32 ? Value::immF32(0.0f) : Value::immI32(0));
      mov(Value::vreg(r), v, t);
      return;
    }
    case StmtKind::Assign:
      return assign(*cast<AssignStmt>(&s));
    case StmtKind::If: {
      const auto *i = cast<IfStmt>(&s);
      unsigned n = ++labelCounter_;
      (void)n;
      BasicBlock *thenBB = newBlock("if.then");
      BasicBlock *elseBB = i->elseStmt ? newBlock("if.else") : nullptr;
      BasicBlock *endBB = newBlock("if.end");
      Value c = expr(*i->cond);
      condbr(c, thenBB, elseBB ? elseBB : endBB);
      start(thenBB);
      stmt(*i->thenStmt);
      if (cur_)
        br(endBB);
      if (elseBB) {
        start(elseBB);
        stmt(*i->elseStmt);
        if (cur_)
          br(endBB);
      }
      start(endBB);
      return;
    }
    case StmtKind::While: {
      const auto *w = cast<WhileStmt>(&s);
      ++labelCounter_;
      BasicBlock *condBB = newBlock("while.cond");
      BasicBlock *bodyBB = newBlock("while.body");
      BasicBlock *endBB = newBlock("while.end");
      br(condBB);
      start(condBB);
      condbr(expr(*w->cond), bodyBB, endBB);
      start(bodyBB);
      loops_.push_back({endBB, condBB});
      stmt(*w->body);
      loops_.pop_back();
      if (cur_)
        br(condBB);
      start(endBB);
      return;
    }
    case StmtKind::For: {
      const auto *f = cast<ForStmt>(&s);
      if (f->init)
        stmt(*f->init);
      ++labelCounter_;
      BasicBlock *condBB = newBlock("for.cond");
      BasicBlock *bodyBB = newBlock("for.body");
      BasicBlock *incBB = newBlock("for.inc");
      BasicBlock *endBB = newBlock("for.end");
      br(condBB);
      start(condBB);
      if (f->cond)
        condbr(expr(*f->cond), bodyBB, endBB);
      else
        br(bodyBB);
      start(bodyBB);
      loops_.push_back({endBB, incBB});
      stmt(*f->body);
      loops_.pop_back();
      if (cur_)
        br(incBB);
      start(incBB);
      if (f->step)
        stmt(*f->step);
      br(condBB);
      start(endBB);
      return;
    }
    case StmtKind::Return:
      br(exit_);
      return;
    case StmtKind::Break:
      br(loops_.back().breakBB);
      return;
    case StmtKind::Continue:
      br(loops_.back().continueBB);
      return;
    case StmtKind::Block:
      for (const auto &st : cast<BlockStmt>(&s)->stmts)
        stmt(*st);
      return;
    case StmtKind::SyncThreads: {
      Inst in;
      in.op = Op::Barrier;
      emit(in);
      return;
    }
    }
  }

  void assign(const AssignStmt &s) {
    static const Op compound[] = {Op::Mov, Op::Add, Op::Sub, Op::Mul, Op::Div};
    Op op = compound[static_cast<int>(s.op)];
    IRType t = irType(s.target->type);
    if (const auto *v = dyn_cast<VarRefExpr>(s.target.get())) {
      Value dst = Value::vreg(regFor(v->sym));
      Value val = expr(*s.value);
      if (s.op != AssignOp::Set)
        val = binop(op, t, t, dst, val);
      mov(dst, val, t);
      return;
    }
    // a[i] op= e: the address is computed once.
    Address ad = address(*cast<IndexExpr>(s.target.get()));
    Value old;
    if (s.op != AssignOp::Set)
      old = load(ad);
    Value val = expr(*s.value);
    if (s.op != AssignOp::Set)
      val = binop(op, t, t, old, val);
    store(ad, val);
  }

  struct Loop {
    BasicBlock *breakBB;
    BasicBlock *continueBB;
  };

  const Kernel &k_;
  std::unique_ptr<Function> fn_;
  BasicBlock *cur_ = nullptr;
  BasicBlock *exit_ = nullptr;
  std::unordered_map<BasicBlock *, std::unique_ptr<BasicBlock>> pending_;
  std::vector<uint32_t> symReg_;
  std::unordered_map<unsigned, uint32_t> sharedIndex_;
  std::vector<Loop> loops_;
  unsigned labelCounter_ = 0;
};

} // namespace

std::unique_ptr<Module> lowerProgram(const Program &prog) {
  auto m = std::make_unique<Module>();
  for (const auto &k : prog.kernels)
    m->functions.push_back(Lowerer(*k).run());
  return m;
}

} // namespace tinycu
