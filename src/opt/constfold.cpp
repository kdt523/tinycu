// Constant folding + local constant propagation + algebraic simplification.
//
// Per basic block, a map vreg -> constant is maintained in instruction order (the IR is not
// SSA, so a redefinition kills the entry). Folding calls ir::evalPure, the exact function the
// interpreter executes, so a folded value can never differ from the run-time value.
// Never folded: division/remainder by zero, INT_MIN / -1, out-of-range float->int (all UB,
// left for the run time to report), and sqrt of a negative number.
#include "passes.h"

#include "tinycu/semantics.h"

#include <cmath>
#include <cstring>
#include <unordered_map>

namespace tinycu::opt {

using namespace ir;

namespace {

uint32_t immBits(const Value &v) {
  switch (v.kind) {
  case Value::Kind::ImmI32:
    return static_cast<uint32_t>(v.i);
  case Value::Kind::ImmBool:
    return v.b ? 1u : 0u;
  case Value::Kind::ImmF32: {
    uint32_t b;
    std::memcpy(&b, &v.f, sizeof b);
    return b;
  }
  default:
    return 0;
  }
}

Value immOfType(IRType t, uint32_t bits) {
  if (t == IRType::I1)
    return Value::immBool(bits != 0);
  if (t == IRType::F32) {
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return Value::immF32(f);
  }
  return Value::immI32(static_cast<int32_t>(bits));
}

IRType resultType(const Inst &in) {
  switch (in.op) {
  case Op::CmpEq:
  case Op::CmpNe:
  case Op::CmpLt:
  case Op::CmpLe:
  case Op::CmpGt:
  case Op::CmpGe:
  case Op::And:
  case Op::Or:
  case Op::Not:
    return IRType::I1;
  case Op::CvtF32I32:
    return IRType::F32;
  case Op::CvtI32F32:
    return IRType::I32;
  default:
    return in.type;
  }
}

bool isI(const Value &v, int32_t k) { return v.kind == Value::Kind::ImmI32 && v.i == k; }
bool isF(const Value &v, float k) { // bitwise: distinguishes +0.0 and -0.0
  return v.kind == Value::Kind::ImmF32 && std::memcmp(&v.f, &k, sizeof k) == 0;
}

void toMov(Inst &in, Value src, IRType t) {
  in.op = Op::Mov;
  in.type = t;
  in.a = src;
  in.b = Value{};
}

// Algebraic identities. Integer identities hold for all values (wrapping arithmetic).
// Float identities are restricted to ones that are exact for every input, including NaN,
// infinities and signed zeros:
//   x + (-0.0) == x, x - (+0.0) == x, x * 1.0 == x, x / 1.0 == x       (valid)
//   x + 0.0 -> x   is WRONG: -0.0 + 0.0 == +0.0
//   x * 0.0 -> 0.0 is WRONG: NaN * 0 == NaN, inf * 0 == NaN, -1 * 0 == -0.0
//   x - x   -> 0.0 is WRONG: NaN - NaN == NaN, inf - inf == NaN
// Comparisons are never rewritten: for floats !(a < b) is NOT a >= b (both differ when an
// operand is NaN), so such rewrites would be miscompiles; TinyCU does none of them.
bool simplify(Inst &in, const std::unordered_map<uint32_t, uint32_t> &notOf) {
  const Value &a = in.a, &b = in.b;
  bool f = in.type == IRType::F32;
  switch (in.op) {
  case Op::Add:
    if (!f && isI(b, 0))
      return toMov(in, a, in.type), true;
    if (!f && isI(a, 0))
      return toMov(in, b, in.type), true;
    if (f && isF(b, -0.0f))
      return toMov(in, a, in.type), true;
    if (f && isF(a, -0.0f))
      return toMov(in, b, in.type), true;
    return false;
  case Op::Sub:
    if ((!f && isI(b, 0)) || (f && isF(b, 0.0f)))
      return toMov(in, a, in.type), true;
    if (!f && a.isVReg() && a == b) // x - x == 0 for ints only
      return toMov(in, Value::immI32(0), in.type), true;
    return false;
  case Op::Mul:
    if ((!f && isI(b, 1)) || (f && isF(b, 1.0f)))
      return toMov(in, a, in.type), true;
    if ((!f && isI(a, 1)) || (f && isF(a, 1.0f)))
      return toMov(in, b, in.type), true;
    if (!f && (isI(a, 0) || isI(b, 0))) // x * 0 == 0 for ints only
      return toMov(in, Value::immI32(0), in.type), true;
    return false;
  case Op::Div:
    if ((!f && isI(b, 1)) || (f && isF(b, 1.0f)))
      return toMov(in, a, in.type), true;
    return false;
  case Op::Not: // !!p -> p
    if (a.isVReg()) {
      auto it = notOf.find(a.reg);
      if (it != notOf.end())
        return toMov(in, Value::vreg(it->second), IRType::I1), true;
    }
    return false;
  default:
    return false;
  }
}

} // namespace

bool ConstFoldPass::run(Function &fn) {
  bool changed = false;
  for (auto &bb : fn.blocks) {
    std::unordered_map<uint32_t, Value> known;    // vreg -> constant (valid in this block)
    std::unordered_map<uint32_t, uint32_t> notOf; // p -> q  where `p = not q`
    for (Inst &in : bb->insts) {
      // 1. Propagate known constants into the operands.
      for (Value *v : {&in.a, &in.b}) {
        if (v->isVReg()) {
          auto it = known.find(v->reg);
          if (it != known.end()) {
            *v = it->second;
            changed = true;
            ++changes;
          }
        }
      }
      // 2. Fold.
      if (isPureOp(in.op) && in.op != Op::Mov && in.a.isImm() && (in.b.isNone() || in.b.isImm())) {
        uint32_t out = 0;
        bool negativeSqrt = in.op == Op::Sqrt && std::signbit(in.a.f) && in.a.f != 0.0f;
        if (!negativeSqrt && evalPure(in.op, in.type, immBits(in.a),
                                      in.b.isNone() ? 0 : immBits(in.b), out) == EvalStatus::Ok) {
          IRType rt = resultType(in);
          toMov(in, immOfType(rt, out), rt);
          changed = true;
          ++changes;
        }
      } else if (in.dst.isVReg() && simplify(in, notOf)) {
        changed = true;
        ++changes;
      }
      if (in.op == Op::CondBr && in.a.kind == Value::Kind::ImmBool) {
        BasicBlock *target = in.a.b ? in.t : in.f;
        in.op = Op::Br;
        in.a = Value{};
        in.t = target;
        in.f = nullptr;
        changed = true;
        ++changes;
      }
      // 3. Update the facts for the defined vreg (kills older facts: not SSA).
      if (in.dst.isVReg()) {
        uint32_t d = in.dst.reg;
        known.erase(d);
        notOf.erase(d);
        for (auto it = notOf.begin(); it != notOf.end();)
          it = it->second == d ? notOf.erase(it) : std::next(it);
        if (in.op == Op::Mov && in.a.isImm())
          known[d] = in.a;
        else if (in.op == Op::Not && in.a.isVReg() && in.a.reg != d)
          notOf[d] = in.a.reg;
      }
    }
  }
  return changed;
}

} // namespace tinycu::opt
