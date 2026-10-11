#pragma once

// The single definition of TinyCU's arithmetic semantics for side-effect-free IR operations.
// Used by BOTH the interpreter and the constant folder, so folding can never disagree with
// execution (a classic source of miscompiles). The GPU and nvcc-twin paths remain the
// independent check on this code.

#include "tinycu/ir.h"

#include <cstdint>

namespace tinycu::ir {

enum class EvalStatus {
  Ok,
  DivZero,     ///< integer division / remainder by zero
  DivOverflow, ///< INT_MIN / -1, INT_MIN % -1
  BadCvt,      ///< float -> int of NaN or out-of-range value
};

/// Ops that evalPure handles: arithmetic, logic, compares, conversions, min/max/abs/sqrt, mov.
bool isPureOp(Op op);

/// Evaluates `op` with operation type `type` on raw 32-bit operands (i32 two's complement,
/// f32 bit pattern, i1 as 0/1). Rules: i32 wraps; `/` and `%` truncate toward zero; f32 uses
/// IEEE float without FMA; `!=` is true when an operand is NaN, other compares are false;
/// min/max follow PTX (NaN operand -> the other one, -0 < +0); abs(INT_MIN) == INT_MIN.
EvalStatus evalPure(Op op, IRType type, uint32_t a, uint32_t b, uint32_t &out);

} // namespace tinycu::ir
