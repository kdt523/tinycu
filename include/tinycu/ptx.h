#pragma once

#include "tinycu/ir.h"

#include <cstdint>
#include <string>

namespace tinycu::ptx {

struct Options {
  std::string sm = "sm_75";
  bool comments = true; ///< annotate source variable names (`// %i`)
};

/// Emits one PTX module (ISA 7.0) containing every kernel of `m`.
std::string emitModule(const ir::Module &m, const Options &opts);

/// Exact PTX spelling of an f32 constant: `0f3F800000` (hex bit pattern).
std::string f32Immediate(float f);

/// Turns an IR block label into a PTX-safe label: `if.then.1` -> `$L__if_then_1`.
std::string ptxLabel(const std::string &irLabel);

} // namespace tinycu::ptx
