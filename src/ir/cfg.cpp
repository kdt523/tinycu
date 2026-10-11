// CFG utilities: successors, predecessors, reverse post-order, reachability.
#include "tinycu/ir.h"

#include <algorithm>
#include <unordered_map>

namespace tinycu::ir {

std::vector<BasicBlock *> successors(const BasicBlock &bb) {
  const Inst *t = bb.terminator();
  if (!t)
    return {};
  if (t->op == Op::Br)
    return {t->t};
  if (t->op == Op::CondBr) {
    if (t->t == t->f)
      return {t->t};
    return {t->t, t->f};
  }
  return {};
}

size_t blockIndex(const Function &fn, const BasicBlock *bb) {
  for (size_t i = 0; i < fn.blocks.size(); ++i)
    if (fn.blocks[i].get() == bb)
      return i;
  return fn.blocks.size();
}

std::vector<std::vector<BasicBlock *>> predecessors(const Function &fn) {
  std::unordered_map<const BasicBlock *, size_t> index;
  for (size_t i = 0; i < fn.blocks.size(); ++i)
    index[fn.blocks[i].get()] = i;
  std::vector<std::vector<BasicBlock *>> preds(fn.blocks.size());
  for (const auto &bb : fn.blocks)
    for (BasicBlock *s : successors(*bb)) {
      auto it = index.find(s);
      if (it == index.end())
        continue;
      auto &list = preds[it->second];
      if (std::find(list.begin(), list.end(), bb.get()) == list.end())
        list.push_back(bb.get());
    }
  return preds;
}

std::vector<BasicBlock *> reversePostOrder(const Function &fn) {
  std::vector<BasicBlock *> post;
  if (fn.blocks.empty())
    return post;
  std::unordered_map<const BasicBlock *, bool> seen;
  // Iterative DFS (no recursion depth limits on big CFGs).
  struct Frame {
    BasicBlock *bb;
    std::vector<BasicBlock *> succ;
    size_t next;
  };
  std::vector<Frame> stack;
  BasicBlock *entry = fn.blocks[0].get();
  seen[entry] = true;
  stack.push_back({entry, successors(*entry), 0});
  while (!stack.empty()) {
    Frame &f = stack.back();
    if (f.next < f.succ.size()) {
      BasicBlock *s = f.succ[f.next++];
      if (!seen[s]) {
        seen[s] = true;
        stack.push_back({s, successors(*s), 0});
      }
    } else {
      post.push_back(f.bb);
      stack.pop_back();
    }
  }
  std::reverse(post.begin(), post.end());
  return post;
}

std::vector<bool> reachable(const Function &fn) {
  std::vector<bool> r(fn.blocks.size(), false);
  for (BasicBlock *bb : reversePostOrder(fn)) {
    size_t i = blockIndex(fn, bb);
    if (i < r.size())
      r[i] = true;
  }
  return r;
}

} // namespace tinycu::ir
