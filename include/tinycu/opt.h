#pragma once

// -O1 optimizer: a tiny pass manager and four IR passes (see docs/ARCHITECTURE.md §Optimizer).
// Correctness over aggressiveness: every pass must preserve the interpreter's output exactly.

#include "tinycu/ir.h"

#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tinycu::opt {

class Pass {
public:
  virtual ~Pass() = default;
  virtual const char *name() const = 0;
  /// Transforms `fn`; returns true if anything changed (and adds to `changes`).
  virtual bool run(ir::Function &fn) = 0;
  unsigned changes = 0; ///< statistics: number of rewrites, summed over all runs
};

std::unique_ptr<Pass> createPass(const std::string &name); ///< nullptr if unknown
const std::vector<std::string> &passNames();               ///< every known pass, in -O1 order
std::vector<std::string> o1Pipeline(); ///< constfold, copyprop, simplifycfg, dce

struct PipelineConfig {
  std::vector<std::string> passes;   ///< in order; run repeatedly until nothing changes
  std::vector<std::string> disabled; ///< --disable-pass (used for bisection)
  std::string dumpAfter;             ///< print the function after every run of this pass
  bool verifyEach = false;           ///< run the IR verifier after every pass
  int maxIterations = 10;
  std::ostream *dump = nullptr; ///< destination for --dump-after (required if dumpAfter set)
};

struct PipelineResult {
  bool ok = true;
  std::string error;                                   ///< verifier failure or unknown pass name
  std::vector<std::pair<std::string, unsigned>> stats; ///< per pass: total changes
  unsigned maxIterationsUsed = 0;
};

PipelineResult runPipeline(ir::Module &m, const PipelineConfig &cfg);

/// Deliberate miscompiles for validating the fuzzer (hidden flag --inject-bug=ID, phase 7):
///   1 constfold folds int add/sub/mul with saturation instead of wrap-around
///   2 copyprop forgets to invalidate a copy when its *source* is redefined
///   3 dce deletes stores of constant values
///   4 simplifycfg swaps the targets of a condbr while threading a jump
/// 0 (default) = no bug.
void setInjectedBug(int id);
int injectedBug();

/// Splits "a,b,c" (used for --passes and --disable-pass).
std::vector<std::string> splitList(const std::string &s);

} // namespace tinycu::opt
