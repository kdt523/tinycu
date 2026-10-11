// Local copy propagation. After `mov a, b` (b a vreg or a constant), later uses of `a` in the
// same block read `b` directly, until `a` or `b` is redefined (the IR is not SSA, so both kill
// the fact). The `mov` itself is left for DCE to remove once it has no uses.
#include "passes.h"

#include <unordered_map>

namespace tinycu::opt {

using namespace ir;

bool CopyPropPass::run(Function &fn) {
  bool changed = false;
  for (auto &bb : fn.blocks) {
    std::unordered_map<uint32_t, Value> copyOf; // a -> b
    for (Inst &in : bb->insts) {
      for (Value *v : {&in.a, &in.b}) {
        if (!v->isVReg())
          continue;
        auto it = copyOf.find(v->reg);
        if (it != copyOf.end()) {
          *v = it->second;
          changed = true;
          ++changes;
        }
      }
      if (!in.dst.isVReg())
        continue;
      uint32_t d = in.dst.reg;
      copyOf.erase(d);                                    // `d` has a new value
      for (auto it = copyOf.begin(); it != copyOf.end();) // facts that read the old `d`
        it = (it->second.isVReg() && it->second.reg == d) ? copyOf.erase(it) : std::next(it);
      if (in.op == Op::Mov && !(in.a.isVReg() && in.a.reg == d) && in.a.kind != Value::Kind::Shared)
        copyOf[d] = in.a;
    }
  }
  return changed;
}

} // namespace tinycu::opt
