#include "tinycu/ir.h"

#include <cmath>
#include <cstring>

namespace tinycu::ir {

const char *typeName(IRType t) {
  switch (t) {
  case IRType::I1:
    return "i1";
  case IRType::I32:
    return "i32";
  case IRType::F32:
    return "f32";
  case IRType::PtrGlobal:
    return "ptr";
  case IRType::PtrShared:
    return "ptr.shared";
  case IRType::I64:
    return "i64";
  }
  return "?";
}

const char *opName(Op op) {
  switch (op) {
  case Op::Add:
    return "add";
  case Op::Sub:
    return "sub";
  case Op::Mul:
    return "mul";
  case Op::Div:
    return "div";
  case Op::Rem:
    return "rem";
  case Op::Neg:
    return "neg";
  case Op::And:
    return "and";
  case Op::Or:
    return "or";
  case Op::Not:
    return "not";
  case Op::CmpEq:
    return "cmp.eq";
  case Op::CmpNe:
    return "cmp.ne";
  case Op::CmpLt:
    return "cmp.lt";
  case Op::CmpLe:
    return "cmp.le";
  case Op::CmpGt:
    return "cmp.gt";
  case Op::CmpGe:
    return "cmp.ge";
  case Op::CvtF32I32:
    return "cvt.f32.i32";
  case Op::CvtI32F32:
    return "cvt.i32.f32";
  case Op::Min:
    return "min";
  case Op::Max:
    return "max";
  case Op::Abs:
    return "abs";
  case Op::Sqrt:
    return "sqrt";
  case Op::Mov:
    return "mov";
  case Op::Addr:
    return "addr";
  case Op::Load:
    return "load";
  case Op::Store:
    return "store";
  case Op::Sreg:
    return "sreg";
  case Op::Param:
    return "param";
  case Op::Barrier:
    return "barrier";
  case Op::Br:
    return "br";
  case Op::CondBr:
    return "condbr";
  case Op::Ret:
    return "ret";
  }
  return "?";
}

const char *sregName(SReg s) {
  switch (s) {
  case SReg::Tid:
    return "tid";
  case SReg::Ctaid:
    return "ctaid";
  case SReg::Ntid:
    return "ntid";
  case SReg::Nctaid:
    return "nctaid";
  }
  return "?";
}

bool Value::operator==(const Value &o) const {
  if (kind != o.kind)
    return false;
  switch (kind) {
  case Kind::None:
    return true;
  case Kind::VReg:
  case Kind::Shared:
    return reg == o.reg;
  case Kind::ImmI32:
    return i == o.i;
  case Kind::ImmF32: // bitwise: 0.0 != -0.0, NaN == same NaN
    return std::memcmp(&f, &o.f, sizeof f) == 0;
  case Kind::ImmBool:
    return b == o.b;
  }
  return false;
}

uint32_t Function::newVReg(IRType t, const std::string &name) {
  uint32_t id = static_cast<uint32_t>(vregs.size());
  std::string n = name.empty() ? std::to_string(id) : name;
  if (!name.empty()) {
    // Unique among named vregs: i, i.1, i.2, ... (temps are pure numbers, never clash).
    auto taken = [&](const std::string &s) {
      for (const VRegInfo &v : vregs)
        if (v.name == s)
          return true;
      return false;
    };
    for (unsigned k = 1; taken(n); ++k)
      n = name + "." + std::to_string(k);
  }
  vregs.push_back({t, n});
  return id;
}

IRType Function::typeOf(const Value &v) const {
  switch (v.kind) {
  case Value::Kind::VReg:
    return v.reg < vregs.size() ? vregs[v.reg].type : IRType::I32;
  case Value::Kind::ImmI32:
    return IRType::I32;
  case Value::Kind::ImmF32:
    return IRType::F32;
  case Value::Kind::ImmBool:
    return IRType::I1;
  case Value::Kind::Shared:
    return IRType::PtrShared;
  case Value::Kind::None:
    break;
  }
  return IRType::I32;
}

bool Function::usesBarrier() const {
  for (const auto &bb : blocks)
    for (const Inst &in : bb->insts)
      if (in.op == Op::Barrier)
        return true;
  return false;
}

Function *Module::find(const std::string &name) const {
  for (const auto &f : functions)
    if (f->name == name)
      return f.get();
  return nullptr;
}

} // namespace tinycu::ir
