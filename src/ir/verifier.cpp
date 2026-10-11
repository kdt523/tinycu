// IR verifier: run after lowering and (with --verify-each) after every optimization pass.
#include "tinycu/ir.h"

#include <set>
#include <unordered_set>

namespace tinycu::ir {

namespace {

class Verifier {
public:
  explicit Verifier(const Function &fn) : fn_(fn) {}

  std::vector<std::string> run() {
    if (fn_.blocks.empty()) {
      err("function has no blocks");
      return errors_;
    }
    std::set<std::string> labels;
    for (const auto &bb : fn_.blocks) {
      blocks_.insert(bb.get());
      if (!labels.insert(bb->label).second)
        err("duplicate block label '" + bb->label + "'");
    }
    for (const IRParam &p : fn_.params)
      if (p.reg >= fn_.vregs.size())
        err("parameter '" + p.name + "' has an invalid vreg");

    std::vector<bool> defined(fn_.vregs.size(), false);
    for (const auto &bb : fn_.blocks)
      for (const Inst &in : bb->insts)
        if (in.dst.isVReg() && in.dst.reg < defined.size())
          defined[in.dst.reg] = true;

    for (const auto &bb : fn_.blocks) {
      cur_ = bb.get();
      if (bb->insts.empty()) {
        err("block is empty (needs a terminator)");
        continue;
      }
      for (size_t i = 0; i < bb->insts.size(); ++i) {
        const Inst &in = bb->insts[i];
        bool last = i + 1 == bb->insts.size();
        if (in.isTerminator() && !last)
          err("terminator '" + std::string(opName(in.op)) + "' in the middle of the block");
        if (last && !in.isTerminator())
          err("block does not end with a terminator");
        checkInst(in, defined);
      }
    }
    // The entry block must not be a branch target.
    for (const auto &bb : fn_.blocks)
      for (BasicBlock *s : successors(*bb))
        if (s == fn_.entry()) {
          cur_ = bb.get();
          err("branch to the entry block");
        }
    return errors_;
  }

private:
  void err(const std::string &msg) {
    errors_.push_back(fn_.name + (cur_ ? ":" + cur_->label : std::string()) + ": " + msg);
  }

  bool checkValue(const Value &v, const std::vector<bool> &defined, const char *what) {
    if (v.isNone()) {
      err(std::string("missing operand ") + what);
      return false;
    }
    if (v.isVReg()) {
      if (v.reg >= fn_.vregs.size()) {
        err(std::string("operand ") + what + " refers to unknown vreg " + std::to_string(v.reg));
        return false;
      }
      if (!defined[v.reg])
        err("vreg %" + fn_.vregs[v.reg].name + " is used but never defined");
    }
    if (v.kind == Value::Kind::Shared && v.reg >= fn_.shared.size()) {
      err("unknown shared array");
      return false;
    }
    return true;
  }

  void expectType(const Value &v, IRType t, const char *what) {
    IRType got = fn_.typeOf(v);
    if (got != t)
      err(std::string(what) + " has type " + typeName(got) + ", expected " + typeName(t));
  }

  void checkTarget(const BasicBlock *bb) {
    if (!bb || !blocks_.count(bb))
      err("branch to a block that is not in the function");
  }

  void checkInst(const Inst &in, const std::vector<bool> &defined) {
    auto operand = [&](const Value &v, const char *what, IRType t) {
      if (checkValue(v, defined, what))
        expectType(v, t, what);
    };
    auto result = [&](IRType t) {
      if (!in.dst.isVReg() || in.dst.reg >= fn_.vregs.size()) {
        err(std::string("'") + opName(in.op) + "' needs a vreg destination");
        return;
      }
      expectType(in.dst, t, "destination");
    };
    auto noResult = [&] {
      if (!in.dst.isNone())
        err(std::string("'") + opName(in.op) + "' must not have a destination");
    };
    bool numeric = in.type == IRType::I32 || in.type == IRType::F32;

    switch (in.op) {
    case Op::Add:
    case Op::Sub:
    case Op::Mul:
    case Op::Div:
    case Op::Min:
    case Op::Max:
      if (!numeric)
        err(std::string("'") + opName(in.op) + "' must be i32 or f32");
      result(in.type);
      operand(in.a, "a", in.type);
      operand(in.b, "b", in.type);
      return;
    case Op::Rem:
      if (in.type != IRType::I32)
        err("'rem' must be i32");
      result(IRType::I32);
      operand(in.a, "a", IRType::I32);
      operand(in.b, "b", IRType::I32);
      return;
    case Op::Neg:
    case Op::Abs:
      if (!numeric)
        err(std::string("'") + opName(in.op) + "' must be i32 or f32");
      result(in.type);
      operand(in.a, "a", in.type);
      return;
    case Op::Sqrt:
      result(IRType::F32);
      operand(in.a, "a", IRType::F32);
      return;
    case Op::And:
    case Op::Or:
      result(IRType::I1);
      operand(in.a, "a", IRType::I1);
      operand(in.b, "b", IRType::I1);
      return;
    case Op::Not:
      result(IRType::I1);
      operand(in.a, "a", IRType::I1);
      return;
    case Op::CmpEq:
    case Op::CmpNe:
    case Op::CmpLt:
    case Op::CmpLe:
    case Op::CmpGt:
    case Op::CmpGe:
      if (!numeric)
        err("compare operand type must be i32 or f32");
      result(IRType::I1);
      operand(in.a, "a", in.type);
      operand(in.b, "b", in.type);
      return;
    case Op::CvtF32I32:
      result(IRType::F32);
      operand(in.a, "a", IRType::I32);
      return;
    case Op::CvtI32F32:
      result(IRType::I32);
      operand(in.a, "a", IRType::F32);
      return;
    case Op::Mov:
      if (in.dst.isVReg() && in.dst.reg < fn_.vregs.size()) {
        if (checkValue(in.a, defined, "a"))
          expectType(in.a, fn_.typeOf(in.dst), "mov source");
      } else {
        err("'mov' needs a vreg destination");
      }
      return;
    case Op::Addr: {
      result(IRType::I64);
      if (checkValue(in.a, defined, "base")) {
        IRType bt = fn_.typeOf(in.a);
        if (bt != IRType::PtrGlobal && bt != IRType::PtrShared)
          err(std::string("addr base has type ") + typeName(bt) + ", expected a pointer");
      }
      operand(in.b, "index", IRType::I32);
      if (in.aux != 4)
        err("addr element size must be 4");
      return;
    }
    case Op::Load:
      if (in.type != IRType::I32 && in.type != IRType::F32)
        err("load type must be i32 or f32");
      result(in.type);
      operand(in.a, "address", IRType::I64);
      if (in.aux != 0 && in.aux != 1)
        err("bad memory space");
      return;
    case Op::Store:
      noResult();
      if (in.type != IRType::I32 && in.type != IRType::F32)
        err("store type must be i32 or f32");
      operand(in.a, "address", IRType::I64);
      operand(in.b, "value", in.type);
      if (in.aux != 0 && in.aux != 1)
        err("bad memory space");
      return;
    case Op::Sreg:
      result(IRType::I32);
      if (in.aux < 0 || in.aux > 3)
        err("bad special register");
      return;
    case Op::Param:
      if (in.aux < 0 || static_cast<size_t>(in.aux) >= fn_.params.size()) {
        err("param index out of range");
        return;
      }
      result(fn_.params[static_cast<size_t>(in.aux)].type);
      return;
    case Op::Barrier:
      noResult();
      return;
    case Op::Br:
      noResult();
      checkTarget(in.t);
      return;
    case Op::CondBr:
      noResult();
      operand(in.a, "condition", IRType::I1);
      checkTarget(in.t);
      checkTarget(in.f);
      return;
    case Op::Ret:
      noResult();
      return;
    }
    err("unknown opcode");
  }

  const Function &fn_;
  const BasicBlock *cur_ = nullptr;
  std::unordered_set<const BasicBlock *> blocks_;
  std::vector<std::string> errors_;
};

} // namespace

std::vector<std::string> verify(const Function &fn) { return Verifier(fn).run(); }

} // namespace tinycu::ir
