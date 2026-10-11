#include "tinycu/semantics.h"

#include <climits>
#include <cmath>
#include <cstring>

namespace tinycu::ir {

namespace {

float f32(uint32_t bits) {
  float f;
  std::memcpy(&f, &bits, sizeof f);
  return f;
}
uint32_t bits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, sizeof b);
  return b;
}
int32_t i32(uint32_t b) { return static_cast<int32_t>(b); }

// PTX min.f32/max.f32: a NaN operand yields the other operand; -0 < +0.
float fmin32(float a, float b) {
  if (std::isnan(a))
    return b;
  if (std::isnan(b))
    return a;
  if (a == b)
    return std::signbit(a) ? a : b;
  return a < b ? a : b;
}
float fmax32(float a, float b) {
  if (std::isnan(a))
    return b;
  if (std::isnan(b))
    return a;
  if (a == b)
    return std::signbit(a) ? b : a;
  return a > b ? a : b;
}

} // namespace

bool isPureOp(Op op) {
  switch (op) {
  case Op::Add:
  case Op::Sub:
  case Op::Mul:
  case Op::Div:
  case Op::Rem:
  case Op::Neg:
  case Op::And:
  case Op::Or:
  case Op::Not:
  case Op::CmpEq:
  case Op::CmpNe:
  case Op::CmpLt:
  case Op::CmpLe:
  case Op::CmpGt:
  case Op::CmpGe:
  case Op::CvtF32I32:
  case Op::CvtI32F32:
  case Op::Min:
  case Op::Max:
  case Op::Abs:
  case Op::Sqrt:
  case Op::Mov:
    return true;
  default:
    return false;
  }
}

EvalStatus evalPure(Op op, IRType type, uint32_t a, uint32_t b, uint32_t &out) {
  bool f = type == IRType::F32;
  switch (op) {
  case Op::Add:
    out = f ? bits(f32(a) + f32(b)) : a + b;
    break;
  case Op::Sub:
    out = f ? bits(f32(a) - f32(b)) : a - b;
    break;
  case Op::Mul:
    out = f ? bits(f32(a) * f32(b)) : a * b;
    break;
  case Op::Div:
  case Op::Rem:
    if (f) {
      out = bits(f32(a) / f32(b)); // Rem is never f32 (verifier)
    } else {
      if (b == 0)
        return EvalStatus::DivZero;
      if (i32(a) == INT_MIN && i32(b) == -1)
        return EvalStatus::DivOverflow;
      out = static_cast<uint32_t>(op == Op::Div ? i32(a) / i32(b) : i32(a) % i32(b));
    }
    break;
  case Op::Neg:
    out = f ? bits(-f32(a)) : 0u - a;
    break;
  case Op::And:
    out = (a & b) & 1u;
    break;
  case Op::Or:
    out = (a | b) & 1u;
    break;
  case Op::Not:
    out = (a & 1u) ^ 1u;
    break;
  case Op::CmpEq:
    out = f ? f32(a) == f32(b) : a == b;
    break;
  case Op::CmpNe: // C semantics: NaN != x is true (PTX setp.neu.f32)
    out = f ? !(f32(a) == f32(b)) : a != b;
    break;
  case Op::CmpLt:
    out = f ? f32(a) < f32(b) : i32(a) < i32(b);
    break;
  case Op::CmpLe:
    out = f ? f32(a) <= f32(b) : i32(a) <= i32(b);
    break;
  case Op::CmpGt:
    out = f ? f32(a) > f32(b) : i32(a) > i32(b);
    break;
  case Op::CmpGe:
    out = f ? f32(a) >= f32(b) : i32(a) >= i32(b);
    break;
  case Op::CvtF32I32:
    out = bits(static_cast<float>(i32(a)));
    break;
  case Op::CvtI32F32: {
    float x = f32(a);
    if (std::isnan(x) || !(x >= -2147483648.0f && x < 2147483648.0f))
      return EvalStatus::BadCvt;
    out = static_cast<uint32_t>(static_cast<int32_t>(std::trunc(x)));
    break;
  }
  case Op::Min:
    out =
        f ? bits(fmin32(f32(a), f32(b))) : static_cast<uint32_t>(i32(a) < i32(b) ? i32(a) : i32(b));
    break;
  case Op::Max:
    out =
        f ? bits(fmax32(f32(a), f32(b))) : static_cast<uint32_t>(i32(a) > i32(b) ? i32(a) : i32(b));
    break;
  case Op::Abs: // abs.s32(INT_MIN) == INT_MIN
    out = f ? bits(std::fabs(f32(a))) : (i32(a) < 0 ? 0u - a : a);
    break;
  case Op::Sqrt:
    out = bits(std::sqrt(f32(a)));
    break;
  case Op::Mov:
    out = a;
    break;
  default:
    out = 0;
    break;
  }
  return EvalStatus::Ok;
}

} // namespace tinycu::ir
