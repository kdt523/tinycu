// CFG simplification, repeated until nothing changes:
//   1. condbr p, L, L            -> br L
//   2. jump threading: a branch to a block that only contains `br M` goes to M directly
//   3. unreachable blocks (not reachable from entry) are deleted
//   4. a block ending in `br S` absorbs S when it is S's only predecessor
// The entry block is never removed, merged into another block, or made a branch target.
#include "passes.h"

#include <unordered_map>
#include <unordered_set>

namespace tinycu::opt {

using namespace ir;

namespace {

bool isTrampoline(const BasicBlock *bb) {
  return bb->insts.size() == 1 && bb->insts[0].op == Op::Br;
}

// Follows a chain of empty `br` blocks; stops at a cycle.
BasicBlock *finalTarget(BasicBlock *bb, const BasicBlock *entry) {
  std::unordered_set<const BasicBlock *> seen;
  while (bb != entry && isTrampoline(bb) && seen.insert(bb).second) {
    BasicBlock *next = bb->insts[0].t;
    if (next == entry || seen.count(next))
      break;
    bb = next;
  }
  return bb;
}

} // namespace

bool SimplifyCfgPass::run(Function &fn) {
  bool changed = false;
  for (bool again = true; again;) {
    again = false;
    BasicBlock *entry = fn.entry();

    // 1 + 2
    for (auto &bb : fn.blocks) {
      Inst &t = bb->insts.back();
      if (t.op == Op::CondBr && t.t == t.f) {
        t.op = Op::Br;
        t.a = Value{};
        t.f = nullptr;
        again = true;
        ++changes;
      }
      for (BasicBlock **target : {&t.t, &t.f}) {
        if (!*target || *target == bb.get())
          continue;
        BasicBlock *dest = finalTarget(*target, entry);
        if (dest != *target) {
          *target = dest;
          again = true;
          ++changes;
        }
      }
    }

    // 3
    std::vector<bool> live = reachable(fn);
    for (size_t i = fn.blocks.size(); i-- > 1;) {
      if (!live[i]) {
        fn.blocks.erase(fn.blocks.begin() + static_cast<long>(i));
        again = true;
        ++changes;
      }
    }

    // 4
    auto preds = predecessors(fn);
    std::unordered_map<const BasicBlock *, size_t> predCount;
    for (size_t i = 0; i < fn.blocks.size(); ++i)
      predCount[fn.blocks[i].get()] = preds[i].size();
    for (size_t i = 0; i < fn.blocks.size(); ++i) {
      BasicBlock *bb = fn.blocks[i].get();
      Inst &t = bb->insts.back();
      if (t.op != Op::Br)
        continue;
      BasicBlock *succ = t.t;
      if (succ == bb || succ == entry || predCount[succ] != 1)
        continue;
      bb->insts.pop_back();
      bb->insts.insert(bb->insts.end(), succ->insts.begin(), succ->insts.end());
      for (size_t j = 0; j < fn.blocks.size(); ++j)
        if (fn.blocks[j].get() == succ) {
          fn.blocks.erase(fn.blocks.begin() + static_cast<long>(j));
          if (j < i)
            --i;
          break;
        }
      again = true;
      ++changes;
      break; // predecessor lists are stale now; recompute in the next round
    }
    changed |= again;
  }
  return changed;
}

} // namespace tinycu::opt
