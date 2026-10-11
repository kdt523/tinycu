// IR: CFG utilities, verifier (negative tests on hand-built IR), lowering of all goldens.
#include "tinycu/ir.h"
#include "tinycu/parser.h"
#include "tinycu/sema.h"

#include <gtest/gtest.h>

#include <dirent.h>
#include <fstream>
#include <sstream>

using namespace tinycu;
using namespace tinycu::ir;

namespace {

// Tiny builder for hand-written IR.
struct Builder {
  std::unique_ptr<Function> fn = std::make_unique<Function>();
  BasicBlock *block(const std::string &label) {
    fn->blocks.push_back(std::make_unique<BasicBlock>());
    fn->blocks.back()->label = label;
    return fn->blocks.back().get();
  }
  static Inst br(BasicBlock *t) {
    Inst in;
    in.op = Op::Br;
    in.t = t;
    return in;
  }
  static Inst condbr(Value p, BasicBlock *t, BasicBlock *f) {
    Inst in;
    in.op = Op::CondBr;
    in.type = IRType::I1;
    in.a = p;
    in.t = t;
    in.f = f;
    return in;
  }
  static Inst ret() { return Inst{}; }
  static Inst bin(Op op, IRType t, Value d, Value a, Value b) {
    Inst in;
    in.op = op;
    in.type = t;
    in.dst = d;
    in.a = a;
    in.b = b;
    return in;
  }
};

// entry -> (a | b) -> join -> exit ; plus an unreachable block.
Builder diamond() {
  Builder b;
  BasicBlock *entry = b.block("entry"), *l = b.block("l"), *r = b.block("r"),
             *join = b.block("join"), *dead = b.block("dead"), *exit = b.block("exit");
  uint32_t p = b.fn->newVReg(IRType::I1, "p");
  Inst cmp =
      Builder::bin(Op::CmpLt, IRType::I32, Value::vreg(p), Value::immI32(1), Value::immI32(2));
  entry->insts = {cmp, Builder::condbr(Value::vreg(p), l, r)};
  l->insts = {Builder::br(join)};
  r->insts = {Builder::br(join)};
  join->insts = {Builder::br(exit)};
  dead->insts = {Builder::br(join)};
  exit->insts = {Builder::ret()};
  return b;
}

bool hasError(const std::vector<std::string> &errs, const std::string &needle) {
  for (const std::string &e : errs)
    if (e.find(needle) != std::string::npos)
      return true;
  return false;
}

std::string listErrors(const std::vector<std::string> &errs) {
  std::string s;
  for (const std::string &e : errs)
    s += e + "\n";
  return s;
}

} // namespace

TEST(CFG, SuccessorsAndPredecessors) {
  Builder b = diamond();
  Function &fn = *b.fn;
  auto succ = successors(*fn.blocks[0]);
  ASSERT_EQ(succ.size(), 2u);
  EXPECT_EQ(succ[0]->label, "l");
  EXPECT_EQ(succ[1]->label, "r");
  auto preds = predecessors(fn);
  EXPECT_TRUE(preds[0].empty());
  ASSERT_EQ(preds[3].size(), 3u); // join <- l, r, dead
  EXPECT_EQ(preds[3][0]->label, "l");
  EXPECT_TRUE(successors(*fn.blocks[5]).empty()); // ret
}

TEST(CFG, ReversePostOrderAndReachability) {
  Builder b = diamond();
  auto rpo = reversePostOrder(*b.fn);
  std::vector<std::string> labels;
  for (BasicBlock *bb : rpo)
    labels.push_back(bb->label);
  EXPECT_EQ(labels, (std::vector<std::string>{"entry", "r", "l", "join", "exit"}));
  auto reach = reachable(*b.fn);
  EXPECT_EQ(reach, (std::vector<bool>{true, true, true, true, false, true}));
}

TEST(CFG, LoopRpoVisitsEachBlockOnce) {
  Builder b;
  BasicBlock *entry = b.block("entry"), *head = b.block("head"), *body = b.block("body"),
             *exit = b.block("exit");
  entry->insts = {Builder::br(head)};
  head->insts = {Builder::condbr(Value::immBool(true), body, exit)};
  body->insts = {Builder::br(head)};
  exit->insts = {Builder::ret()};
  EXPECT_EQ(reversePostOrder(*b.fn).size(), 4u);
  EXPECT_TRUE(verify(*b.fn).empty()) << listErrors(verify(*b.fn));
}

TEST(Verifier, AcceptsWellFormed) {
  Builder b = diamond();
  EXPECT_TRUE(verify(*b.fn).empty()) << listErrors(verify(*b.fn));
}

TEST(Verifier, RejectsMissingTerminator) {
  Builder b = diamond();
  b.fn->blocks[1]->insts.clear();
  b.fn->blocks[1]->insts.push_back(Builder::bin(Op::Add, IRType::I32,
                                                Value::vreg(b.fn->newVReg(IRType::I32)),
                                                Value::immI32(1), Value::immI32(2)));
  EXPECT_TRUE(hasError(verify(*b.fn), "does not end with a terminator"));
}

TEST(Verifier, RejectsTerminatorInMiddle) {
  Builder b = diamond();
  b.fn->blocks[2]->insts.insert(b.fn->blocks[2]->insts.begin(), Builder::ret());
  EXPECT_TRUE(hasError(verify(*b.fn), "in the middle"));
}

TEST(Verifier, RejectsEmptyBlock) {
  Builder b = diamond();
  b.fn->blocks[2]->insts.clear();
  EXPECT_TRUE(hasError(verify(*b.fn), "empty"));
}

TEST(Verifier, RejectsDanglingBranchTarget) {
  Builder b = diamond();
  BasicBlock stray;
  stray.label = "stray";
  b.fn->blocks[1]->insts.back().t = &stray;
  EXPECT_TRUE(hasError(verify(*b.fn), "not in the function"));
}

TEST(Verifier, RejectsBranchToEntry) {
  Builder b = diamond();
  b.fn->blocks[1]->insts.back().t = b.fn->blocks[0].get();
  EXPECT_TRUE(hasError(verify(*b.fn), "entry"));
}

TEST(Verifier, RejectsOperandTypeMismatch) {
  Builder b = diamond();
  uint32_t f = b.fn->newVReg(IRType::F32);
  uint32_t d = b.fn->newVReg(IRType::I32);
  auto &insts = b.fn->blocks[1]->insts;
  insts.insert(insts.begin(), Builder::bin(Op::Add, IRType::I32, Value::vreg(d), Value::immI32(1),
                                           Value::immF32(2.0f)));
  insts.insert(insts.begin(),
               Builder::bin(Op::Mov, IRType::F32, Value::vreg(f), Value::immF32(1.0f), Value{}));
  EXPECT_TRUE(hasError(verify(*b.fn), "has type f32, expected i32"));
}

TEST(Verifier, RejectsNonPredicateCondition) {
  Builder b = diamond();
  b.fn->blocks[0]->insts.back().a = Value::immI32(1);
  EXPECT_TRUE(hasError(verify(*b.fn), "condition has type i32"));
}

TEST(Verifier, RejectsUseOfUndefinedVReg) {
  Builder b = diamond();
  uint32_t ghost = b.fn->newVReg(IRType::I32, "ghost");
  uint32_t d = b.fn->newVReg(IRType::I32);
  auto &insts = b.fn->blocks[1]->insts;
  insts.insert(insts.begin(), Builder::bin(Op::Add, IRType::I32, Value::vreg(d), Value::vreg(ghost),
                                           Value::immI32(1)));
  EXPECT_TRUE(hasError(verify(*b.fn), "%ghost is used but never defined"));
}

TEST(Verifier, RejectsBadAddrAndLoad) {
  Builder b = diamond();
  uint32_t addr = b.fn->newVReg(IRType::I64);
  uint32_t v = b.fn->newVReg(IRType::I32);
  Inst a =
      Builder::bin(Op::Addr, IRType::I64, Value::vreg(addr), Value::immI32(0), Value::immI32(1));
  a.aux = 4;
  Inst ld = Builder::bin(Op::Load, IRType::I32, Value::vreg(v), Value::immI32(0), Value{});
  auto &insts = b.fn->blocks[1]->insts;
  insts.insert(insts.begin(), ld);
  insts.insert(insts.begin(), a);
  auto errs = verify(*b.fn);
  EXPECT_TRUE(hasError(errs, "addr base has type i32, expected a pointer")) << listErrors(errs);
  EXPECT_TRUE(hasError(errs, "address has type i32, expected i64")) << listErrors(errs);
}

TEST(Verifier, RejectsStoreWithDestinationAndBadParam) {
  Builder b = diamond();
  Inst st = Builder::bin(Op::Store, IRType::I32, Value::vreg(b.fn->newVReg(IRType::I32)),
                         Value::immI32(0), Value::immI32(0));
  Inst pa;
  pa.op = Op::Param;
  pa.dst = Value::vreg(b.fn->newVReg(IRType::I32));
  pa.aux = 7;
  auto &insts = b.fn->blocks[1]->insts;
  insts.insert(insts.begin(), pa);
  insts.insert(insts.begin(), st);
  auto errs = verify(*b.fn);
  EXPECT_TRUE(hasError(errs, "must not have a destination")) << listErrors(errs);
  EXPECT_TRUE(hasError(errs, "param index out of range")) << listErrors(errs);
}

TEST(Verifier, RejectsDuplicateLabels) {
  Builder b = diamond();
  b.fn->blocks[2]->label = "l";
  EXPECT_TRUE(hasError(verify(*b.fn), "duplicate block label"));
}

TEST(Lowering, AllGoldenProgramsVerify) {
  DIR *dir = opendir(TINYCU_SOURCE_DIR "/tests/golden");
  ASSERT_NE(dir, nullptr);
  int count = 0;
  while (dirent *ent = readdir(dir)) {
    std::string name = ent->d_name;
    if (name.size() < 3 || name.substr(name.size() - 3) != ".tc")
      continue;
    std::ifstream in(std::string(TINYCU_SOURCE_DIR "/tests/golden/") + name);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string src = ss.str();
    DiagEngine diag(name, src);
    auto prog = parseSource(src, diag);
    ASSERT_FALSE(diag.hasErrors()) << name;
    ASSERT_TRUE(Sema(diag).run(*prog)) << name;
    auto m = lowerProgram(*prog);
    for (const auto &fn : m->functions) {
      auto errs = verify(*fn);
      EXPECT_TRUE(errs.empty()) << name << "\n" << listErrors(errs) << printFunction(*fn);
    }
    ++count;
  }
  closedir(dir);
  EXPECT_GE(count, 10);
}

TEST(Lowering, NamedVRegsAreUniqued) {
  std::string src = "kernel k(int* o) { for (int i = 0; i < 2; i++) { o[i] = i; } "
                    "for (int i = 0; i < 3; i++) { o[i] = i; } }";
  DiagEngine diag("t.tc", src);
  auto prog = parseSource(src, diag);
  ASSERT_TRUE(Sema(diag).run(*prog));
  std::string ir = printModule(*lowerProgram(*prog));
  EXPECT_NE(ir.find("%i:i32 = mov 0"), std::string::npos) << ir;
  EXPECT_NE(ir.find("%i.1:i32 = mov 0"), std::string::npos) << ir;
}

TEST(Lowering, CompoundIndexAssignComputesAddressOnce) {
  std::string src = "kernel k(float* x, int n) { x[n] += 1.0; }";
  DiagEngine diag("t.tc", src);
  auto prog = parseSource(src, diag);
  ASSERT_TRUE(Sema(diag).run(*prog));
  std::string ir = printModule(*lowerProgram(*prog));
  size_t first = ir.find("addr");
  ASSERT_NE(first, std::string::npos);
  EXPECT_EQ(ir.find("addr", first + 1), std::string::npos) << ir;
}

TEST(IRPrinter, FloatImmediatesRoundTrip) {
  Function fn;
  EXPECT_EQ(printValue(fn, Value::immF32(1.0f)), "1.0f");
  EXPECT_EQ(printValue(fn, Value::immF32(0.1f)), "0.100000001f");
  EXPECT_EQ(printValue(fn, Value::immF32(-0.0f)), "-0.0f");
  EXPECT_EQ(printValue(fn, Value::immF32(1e-40f)), "9.9999461e-41f");
  EXPECT_EQ(printValue(fn, Value::immI32(-7)), "-7");
}
