// Deterministic text form of the IR (`--emit=ir`, golden tests, --dump-after).
#include "tinycu/ir.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace tinycu::ir {

namespace {

// Shortest-safe decimal that round-trips to the same f32 (%.9g), always marked as a float
// literal (`1.0f`, `2.5e-07f`). Non-finite values print their bit pattern: `f32(0x7fc00000)`.
std::string formatF32(float f) {
  char buf[48];
  if (!std::isfinite(f)) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof bits);
    std::snprintf(buf, sizeof buf, "f32(0x%08x)", bits);
    return buf;
  }
  std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>(f));
  std::string s = buf;
  if (s.find_first_of(".e") == std::string::npos)
    s += ".0";
  return s + "f";
}

} // namespace

std::string printValue(const Function &fn, const Value &v) {
  switch (v.kind) {
  case Value::Kind::None:
    return "<none>";
  case Value::Kind::VReg:
    return v.reg < fn.vregs.size() ? "%" + fn.vregs[v.reg].name
                                   : "%<bad" + std::to_string(v.reg) + ">";
  case Value::Kind::ImmI32:
    return std::to_string(v.i);
  case Value::Kind::ImmF32:
    return formatF32(v.f);
  case Value::Kind::ImmBool:
    return v.b ? "true" : "false";
  case Value::Kind::Shared:
    return v.reg < fn.shared.size() ? "@" + fn.shared[v.reg].name : "@<bad>";
  }
  return "?";
}

static std::string printInst(const Function &fn, const Inst &in) {
  auto val = [&](const Value &v) { return printValue(fn, v); };
  auto label = [](const BasicBlock *bb) { return bb ? bb->label : std::string("<null>"); };
  std::string s;
  if (in.dst.isVReg())
    s = val(in.dst) + ":" + typeName(fn.typeOf(in.dst)) + " = ";
  switch (in.op) {
  case Op::Load:
  case Op::Store:
    s += std::string(opName(in.op)) +
         (in.aux == static_cast<int32_t>(MemSpace::Shared) ? ".shared " : ".global ") + val(in.a);
    if (in.op == Op::Store)
      s += ", " + val(in.b);
    return s;
  case Op::Addr:
    return s + "addr " + val(in.a) + ", " + val(in.b) + ", " + std::to_string(in.aux);
  case Op::Sreg:
    return s + "sreg " + sregName(static_cast<SReg>(in.aux));
  case Op::Param:
    return s + "param " + std::to_string(in.aux);
  case Op::Barrier:
    return "barrier";
  case Op::Br:
    return "br " + label(in.t);
  case Op::CondBr:
    return "condbr " + val(in.a) + ", " + label(in.t) + ", " + label(in.f);
  case Op::Ret:
    return "ret";
  default:
    break;
  }
  s += opName(in.op);
  if (!in.a.isNone())
    s += " " + val(in.a);
  if (!in.b.isNone())
    s += ", " + val(in.b);
  return s;
}

std::string printFunction(const Function &fn) {
  std::string s = "kernel " + fn.name + "(";
  for (size_t i = 0; i < fn.params.size(); ++i) {
    if (i)
      s += ", ";
    const IRParam &p = fn.params[i];
    s += std::string(typeName(p.type)) + " %" + fn.vregs[p.reg].name;
  }
  s += ") {\n";
  for (const SharedArray &sh : fn.shared)
    s += "  shared " + std::string(typeName(sh.elemType)) + " @" + sh.name + "[" +
         std::to_string(sh.count) + "]\n";
  for (const auto &bb : fn.blocks) {
    s += bb->label + ":\n";
    for (const Inst &in : bb->insts)
      s += "  " + printInst(fn, in) + "\n";
  }
  return s + "}\n";
}

std::string printModule(const Module &m) {
  std::string s;
  for (size_t i = 0; i < m.functions.size(); ++i) {
    if (i)
      s += "\n";
    s += printFunction(*m.functions[i]);
  }
  return s;
}

} // namespace tinycu::ir
