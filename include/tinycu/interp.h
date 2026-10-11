#pragma once

// IR interpreter: the reference oracle. Emulates a 1-D grid of 1-D thread blocks on the CPU
// with exact TinyCU semantics (i32 wraps, f32 without FMA contraction) and turns every
// undefined behaviour (out-of-bounds access, division by zero, INT_MIN / -1, bad float->int
// conversion, barrier divergence, reading shared memory no thread has written, endless loop)
// into a reported error.

#include "tinycu/ir.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tinycu::interp {

/// One kernel argument. Scalars use `bits` (raw i32 or f32 bit pattern); pointers own a
/// buffer of 32-bit elements that the kernel reads and writes in place.
struct ArgValue {
  ir::IRType type = ir::IRType::I32; ///< I32, F32 or PtrGlobal
  ir::IRType elem = ir::IRType::I32; ///< element type when type == PtrGlobal
  uint32_t bits = 0;
  std::vector<uint32_t> data;
};

struct LaunchConfig {
  uint32_t grid = 1;
  uint32_t block = 1;
  uint64_t stepLimit = 10'000'000; ///< per thread
};

struct Result {
  bool ok = true;
  std::string error; ///< full human-readable message
  /// Machine-readable class: oob | div0 | div_overflow | bad_cvt | barrier_divergence |
  /// step_limit | uninit_shared | bad_args
  std::string errorKind;
  int64_t errorBlock = -1; ///< -1 when not tied to a thread
  int64_t errorThread = -1;
  std::string errorMsg; ///< message without the kernel/block/thread prefix
  uint64_t steps = 0;   ///< instructions executed over all threads
};

/// Runs `fn` over the grid. `args` must match fn.params in number and type.
Result run(const ir::Function &fn, const LaunchConfig &cfg, std::vector<ArgValue> &args);

// ---- JSON I/O used by `tinycu --interp` (interp_json.cpp) ----
/// Parses {"params":[{"type":"float","value":2.0},{"type":"float*","data":[...]}, ...]}.
/// Float scalars/elements may be given as "bits" instead (to carry NaN/Inf/-0).
bool parseArgsJson(const std::string &text, const ir::Function &fn, std::vector<ArgValue> &out,
                   std::string &error);
/// {"kernel":..., "grid":..., "block":..., "status":"ok"|"error", "error":..., "params":[...]}
std::string resultJson(const ir::Function &fn, const LaunchConfig &cfg, const Result &r,
                       const std::vector<ArgValue> &args);

} // namespace tinycu::interp
