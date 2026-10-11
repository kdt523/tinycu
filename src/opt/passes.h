#pragma once

// Pass classes (internal to src/opt; users go through opt.h's createPass).

#include "tinycu/opt.h"

namespace tinycu::opt {

/// Local (per basic block) constant propagation + folding + algebraic simplification, and
/// `condbr <const>` -> `br`. Folding uses ir::evalPure, the interpreter's own semantics.
class ConstFoldPass : public Pass {
public:
  const char *name() const override { return "constfold"; }
  bool run(ir::Function &fn) override;
};

/// Local copy propagation: after `mov a, b`, uses of `a` become `b` until a or b is redefined.
class CopyPropPass : public Pass {
public:
  const char *name() const override { return "copyprop"; }
  bool run(ir::Function &fn) override;
};

/// Removes side-effect-free instructions whose result is never used anywhere (to fixpoint).
class DcePass : public Pass {
public:
  const char *name() const override { return "dce"; }
  bool run(ir::Function &fn) override;
};

/// condbr p, L, L -> br L; jump threading through empty blocks; unreachable block removal;
/// merging a block into its unique predecessor.
class SimplifyCfgPass : public Pass {
public:
  const char *name() const override { return "simplifycfg"; }
  bool run(ir::Function &fn) override;
};

} // namespace tinycu::opt
