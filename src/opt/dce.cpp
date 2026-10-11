// Dead code elimination: removes instructions without side effects whose destination is never
// read anywhere in the function, repeating until nothing changes (removing one use can make
// another definition dead). Loads are removable (UB-free programs only); stores, barriers and
// terminators never are.
//
// Zero-initialisation (`mov %x, 0` for `int x;`) is removed only when %x is not read anywhere,
// i.e. when it is provably irrelevant: a single use anywhere keeps it.
#include "passes.h"

#include <vector>

namespace tinycu::opt {

using namespace ir;

bool DcePass::run(Function &fn) {
  bool changed = false;
  for (;;) {
    std::vector<unsigned> uses(fn.vregs.size(), 0);
    for (const auto &bb : fn.blocks)
      for (const Inst &in : bb->insts)
        for (const Value *v : {&in.a, &in.b})
          if (v->isVReg() && v->reg < uses.size())
            ++uses[v->reg];
    bool removed = false;
    for (auto &bb : fn.blocks) {
      auto &insts = bb->insts;
      size_t out = 0;
      for (size_t i = 0; i < insts.size(); ++i) {
        const Inst &in = insts[i];
        bool selfMove = in.op == Op::Mov && in.a.isVReg() && in.dst == in.a;
        bool dead = in.dst.isVReg() && !in.hasSideEffects() && uses[in.dst.reg] == 0;
        if (selfMove || dead) {
          removed = true;
          ++changes;
          continue;
        }
        insts[out++] = insts[i];
      }
      insts.resize(out);
    }
    if (!removed)
      return changed;
    changed = true;
  }
}

} // namespace tinycu::opt
