// Optimizer unit tests: hand-written IR before/after each pass (parsed with ir::parseFunction).
#include "tinycu/interp.h"
#include "tinycu/ir.h"
#include "tinycu/opt.h"
#include "tinycu/parser.h"
#include "tinycu/sema.h"

#include <gtest/gtest.h>

#include <cstring>
#include <sstream>

using namespace tinycu;
using namespace tinycu::ir;

namespace {

// Common prologue: parameters %a %b (i32), %f %g (f32), %p (ptr), defined in `entry`.
const char *kHead = "kernel k(i32 %a, i32 %b, f32 %f, f32 %g, ptr %p) {\n"
                    "entry:\n"
                    "  %a:i32 = param 0\n"
                    "  %b:i32 = param 1\n"
                    "  %f:f32 = param 2\n"
                    "  %g:f32 = param 3\n"
                    "  %p:ptr = param 4\n";

std::string fn(const std::string &body) { return kHead + body + "}\n"; }

std::unique_ptr<Function> parse(const std::string &text) {
  std::string err;
  auto f = parseFunction(text, err);
  EXPECT_NE(f, nullptr) << err << "\n" << text;
  return f;
}

// Runs `pass` on `before` and returns the printed result (verified).
std::string after(const std::string &pass, const std::string &before) {
  auto f = parse(before);
  if (!f)
    return "";
  auto p = opt::createPass(pass);
  p->run(*f);
  auto errs = verify(*f);
  EXPECT_TRUE(errs.empty()) << (errs.empty() ? "" : errs[0]) << "\n" << printFunction(*f);
  return printFunction(*f);
}

// Expected text, normalized through the parser + printer.
std::string norm(const std::string &text) {
  auto f = parse(text);
  return f ? printFunction(*f) : "";
}

#define EXPECT_PASS(pass, before, expected) EXPECT_EQ(after(pass, before), norm(expected))

} // namespace

// =========================================================================== parser round trip
TEST(IRParser, RoundTripsPrinterOutput) {
  std::string src = R"(kernel k(f32 %a, ptr %x, i32 %n) {
  shared f32 @s[4]
entry:
  %a:f32 = param 0
  %x:ptr = param 1
  %n:i32 = param 2
  %3:i32 = sreg tid
  %4:i1 = cmp.lt %3, %n
  condbr %4, then, exit
then:
  %5:i64 = addr %x, %3, 4
  %6:f32 = load.global %5
  %7:f32 = mul %6, -0.0f
  %8:i64 = addr @s, %3, 4
  store.shared %8, %7
  barrier
  %9:f32 = sqrt f32(0x7fc00000)
  %10:i32 = cvt.i32.f32 %a
  br exit
exit:
  ret
}
)";
  EXPECT_EQ(norm(src), src);
  std::string err;
  EXPECT_EQ(parseFunction("kernel k() {\nentry:\n  %x:i32 = bogus 1\n  ret\n}\n", err), nullptr);
  EXPECT_NE(err.find("unknown instruction"), std::string::npos) << err;
}

// =========================================================================== constfold
TEST(ConstFold, FoldsAndPropagatesWithinBlock) {
  EXPECT_PASS("constfold", fn("  %x:i32 = add 2, 3\n  %y:i32 = mul %x, 4\n  ret\n"),
              fn("  %x:i32 = mov 5\n  %y:i32 = mov 20\n  ret\n"));
}

TEST(ConstFold, IntOverflowWraps) {
  EXPECT_PASS(
      "constfold",
      fn("  %x:i32 = add 2147483647, 1\n  %y:i32 = mul 65536, 65536\n  %z:i32 = neg -2147483648\n  "
         "ret\n"),
      fn("  %x:i32 = mov -2147483648\n  %y:i32 = mov 0\n  %z:i32 = mov -2147483648\n  ret\n"));
}

TEST(ConstFold, DivisionByZeroIsNotFolded) {
  EXPECT_PASS("constfold",
              fn("  %z:i32 = mov 0\n  %d:i32 = div 7, %z\n  %r:i32 = rem 7, 0\n  ret\n"),
              fn("  %z:i32 = mov 0\n  %d:i32 = div 7, 0\n  %r:i32 = rem 7, 0\n  ret\n"));
}

TEST(ConstFold, IntMinDividedByMinusOneIsNotFolded) {
  EXPECT_PASS("constfold", fn("  %d:i32 = div -2147483648, -1\n  %q:i32 = div -7, 2\n  ret\n"),
              fn("  %d:i32 = div -2147483648, -1\n  %q:i32 = mov -3\n  ret\n"));
}

TEST(ConstFold, FloatTimesZeroIsNotSimplified) {
  // NaN * 0 = NaN, inf * 0 = NaN, -1 * 0 = -0: x * 0.0 -> 0.0 would be a miscompile.
  EXPECT_PASS(
      "constfold",
      fn("  %x:f32 = mul %f, 0.0f\n  %y:f32 = sub %f, %f\n  %z:f32 = add %f, 0.0f\n  ret\n"),
      fn("  %x:f32 = mul %f, 0.0f\n  %y:f32 = sub %f, %f\n  %z:f32 = add %f, 0.0f\n  ret\n"));
}

TEST(ConstFold, ExactFloatIdentities) {
  EXPECT_PASS(
      "constfold",
      fn("  %x:f32 = add %f, -0.0f\n  %y:f32 = sub %f, 0.0f\n  %z:f32 = mul 1.0f, %f\n"
         "  %w:f32 = div %f, 1.0f\n  ret\n"),
      fn("  %x:f32 = mov %f\n  %y:f32 = mov %f\n  %z:f32 = mov %f\n  %w:f32 = mov %f\n  ret\n"));
}

TEST(ConstFold, IntIdentities) {
  EXPECT_PASS("constfold",
              fn("  %x:i32 = mul %a, 0\n  %y:i32 = sub %a, %a\n  %z:i32 = add 0, %a\n"
                 "  %w:i32 = div %a, 1\n  %v:i32 = mul %a, 1\n  ret\n"),
              fn("  %x:i32 = mov 0\n  %y:i32 = mov 0\n  %z:i32 = mov %a\n  %w:i32 = mov %a\n"
                 "  %v:i32 = mov %a\n  ret\n"));
}

TEST(ConstFold, NaNComparisonsMatchCSemantics) {
  // NaN != NaN is true (setp.neu), every other comparison with NaN is false.
  EXPECT_PASS("constfold",
              fn("  %n:f32 = div 0.0f, 0.0f\n  %ne:i1 = cmp.ne %n, %n\n  %eq:i1 = cmp.eq %n, %n\n"
                 "  %lt:i1 = cmp.lt %n, 1.0f\n  %ge:i1 = cmp.ge %n, 1.0f\n  ret\n"),
              fn("  %n:f32 = mov f32(0xffc00000)\n  %ne:i1 = mov true\n  %eq:i1 = mov false\n"
                 "  %lt:i1 = mov false\n  %ge:i1 = mov false\n  ret\n"));
}

TEST(ConstFold, FloatCompareIsNeverInverted) {
  // !(f < g) must stay a `not` of `cmp.lt`: rewriting to cmp.ge is wrong when f or g is NaN.
  EXPECT_PASS("constfold", fn("  %c:i1 = cmp.lt %f, %g\n  %d:i1 = not %c\n  ret\n"),
              fn("  %c:i1 = cmp.lt %f, %g\n  %d:i1 = not %c\n  ret\n"));
}

TEST(ConstFold, DoubleNegation) {
  EXPECT_PASS("constfold",
              fn("  %c:i1 = cmp.lt %a, %b\n  %d:i1 = not %c\n  %e:i1 = not %d\n  ret\n"),
              fn("  %c:i1 = cmp.lt %a, %b\n  %d:i1 = not %c\n  %e:i1 = mov %c\n  ret\n"));
  // ...but not when the inner operand was redefined in between.
  EXPECT_PASS("constfold",
              fn("  %c:i1 = cmp.lt %a, %b\n  %d:i1 = not %c\n  %c:i1 = cmp.gt %a, %b\n"
                 "  %e:i1 = not %d\n  ret\n"),
              fn("  %c:i1 = cmp.lt %a, %b\n  %d:i1 = not %c\n  %c:i1 = cmp.gt %a, %b\n"
                 "  %e:i1 = not %d\n  ret\n"));
}

TEST(ConstFold, RedefinitionKillsTheConstant) {
  EXPECT_PASS("constfold",
              fn("  %x:i32 = mov 1\n  %x:i32 = add %x, %a\n  %y:i32 = add %x, 1\n  ret\n"),
              fn("  %x:i32 = mov 1\n  %x:i32 = add 1, %a\n  %y:i32 = add %x, 1\n  ret\n"));
}

TEST(ConstFold, UnsafeConversionsAndSqrtAreNotFolded) {
  EXPECT_PASS("constfold",
              fn("  %s:f32 = sqrt -4.0f\n  %t:f32 = sqrt 4.0f\n  %c:i32 = cvt.i32.f32 3.0e9f\n"
                 "  %d:i32 = cvt.i32.f32 -2.75f\n  %e:f32 = cvt.f32.i32 16777217\n  ret\n"),
              fn("  %s:f32 = sqrt -4.0f\n  %t:f32 = mov 2.0f\n  %c:i32 = cvt.i32.f32 3.0e9f\n"
                 "  %d:i32 = mov -2\n  %e:f32 = mov 16777216.0f\n  ret\n"));
}

TEST(ConstFold, ConstantBranchBecomesJump) {
  EXPECT_PASS("constfold",
              fn("  %c:i1 = cmp.lt 1, 2\n  condbr %c, yes, no\nyes:\n  br no\nno:\n  ret\n"),
              fn("  %c:i1 = mov true\n  br yes\nyes:\n  br no\nno:\n  ret\n"));
}

TEST(ConstFold, ConstantsDoNotCrossBlocks) {
  EXPECT_PASS("constfold", fn("  %x:i32 = mov 4\n  br next\nnext:\n  %y:i32 = add %x, 1\n  ret\n"),
              fn("  %x:i32 = mov 4\n  br next\nnext:\n  %y:i32 = add %x, 1\n  ret\n"));
}

TEST(ConstFold, AgreesWithTheInterpreter) {
  // Fold a whole program and execute both versions: identical outputs, including NaN cases.
  std::string src = R"(kernel k(int* o, float* f, float z) {
    float nan = z / z;
    if (nan != nan) { o[0] = 1; }
    if (!(nan < 1.0) && !(nan >= 1.0)) { o[1] = 1; }
    o[2] = 2147483647 + 1;
    o[3] = (int)(-7.5) / 2 + -7 % 3;
    f[0] = min(nan, 2.0) + max(-0.0, 0.0) + sqrt(2.0) + (float)16777217;
    f[1] = 0.1 * 3.0 - 0.3;
  })";
  DiagEngine diag("t.tc", src);
  auto prog = parseSource(src, diag);
  ASSERT_TRUE(Sema(diag).run(*prog));
  auto run = [&](bool optimize) {
    auto m = lowerProgram(*prog);
    if (optimize) {
      opt::PipelineConfig pc;
      pc.passes = opt::o1Pipeline();
      pc.verifyEach = true;
      EXPECT_TRUE(opt::runPipeline(*m, pc).ok);
    }
    std::vector<interp::ArgValue> args(3);
    args[0].type = IRType::PtrGlobal;
    args[0].data.assign(4, 0);
    args[1].type = IRType::PtrGlobal;
    args[1].elem = IRType::F32;
    args[1].data.assign(2, 0);
    args[2].type = IRType::F32;
    args[2].bits = 0; // z = 0.0
    interp::LaunchConfig cfg;
    EXPECT_TRUE(interp::run(*m->functions[0], cfg, args).ok);
    return std::make_pair(args[0].data, args[1].data);
  };
  auto o0 = run(false), o1 = run(true);
  EXPECT_EQ(o0.first, o1.first);
  EXPECT_EQ(o0.second, o1.second);
  EXPECT_EQ(o0.first, (std::vector<uint32_t>{1, 1, 0x80000000u, static_cast<uint32_t>(-4)}));
}

// =========================================================================== copyprop
TEST(CopyProp, ReplacesUsesOfACopy) {
  EXPECT_PASS("copyprop", fn("  %x:i32 = mov %a\n  %y:i32 = add %x, %x\n  ret\n"),
              fn("  %x:i32 = mov %a\n  %y:i32 = add %a, %a\n  ret\n"));
}

TEST(CopyProp, PropagatesConstants) {
  EXPECT_PASS("copyprop", fn("  %x:f32 = mov 1.5f\n  %y:f32 = mul %x, %f\n  ret\n"),
              fn("  %x:f32 = mov 1.5f\n  %y:f32 = mul 1.5f, %f\n  ret\n"));
}

TEST(CopyProp, StopsWhenTheCopyIsRedefined) {
  EXPECT_PASS("copyprop",
              fn("  %x:i32 = mov %a\n  %x:i32 = add %x, 1\n  %y:i32 = add %x, 2\n  ret\n"),
              fn("  %x:i32 = mov %a\n  %x:i32 = add %a, 1\n  %y:i32 = add %x, 2\n  ret\n"));
}

TEST(CopyProp, StopsWhenTheSourceIsRedefined) {
  EXPECT_PASS("copyprop",
              fn("  %t:i32 = mov %b\n  %x:i32 = mov %t\n  %t:i32 = add %a, 1\n  %y:i32 = add %x, "
                 "2\n  ret\n"),
              fn("  %t:i32 = mov %b\n  %x:i32 = mov %b\n  %t:i32 = add %a, 1\n  %y:i32 = add %b, "
                 "2\n  ret\n"));
  EXPECT_PASS("copyprop",
              fn("  %x:i32 = mov %t\n  %t:i32 = add %a, 1\n  %y:i32 = add %x, 2\n  ret\n"
                 "dummy:\n  %t:i32 = mov 0\n  ret\n"),
              fn("  %x:i32 = mov %t\n  %t:i32 = add %a, 1\n  %y:i32 = add %x, 2\n  ret\n"
                 "dummy:\n  %t:i32 = mov 0\n  ret\n"));
}

TEST(CopyProp, FollowsChains) {
  EXPECT_PASS("copyprop", fn("  %x:i32 = mov %a\n  %y:i32 = mov %x\n  %z:i32 = add %y, 1\n  ret\n"),
              fn("  %x:i32 = mov %a\n  %y:i32 = mov %a\n  %z:i32 = add %a, 1\n  ret\n"));
}

TEST(CopyProp, DoesNotCrossBlocks) {
  EXPECT_PASS("copyprop", fn("  %x:i32 = mov %a\n  br next\nnext:\n  %y:i32 = add %x, 1\n  ret\n"),
              fn("  %x:i32 = mov %a\n  br next\nnext:\n  %y:i32 = add %x, 1\n  ret\n"));
}

TEST(CopyProp, IntoBranchConditionAndStores) {
  EXPECT_PASS(
      "copyprop",
      fn("  %c:i1 = cmp.lt %a, %b\n  %d:i1 = mov %c\n  %i:i32 = mov %b\n"
         "  %ad:i64 = addr %p, %i, 4\n  store.global %ad, %i\n  condbr %d, x, x\nx:\n  ret\n"),
      fn("  %c:i1 = cmp.lt %a, %b\n  %d:i1 = mov %c\n  %i:i32 = mov %b\n"
         "  %ad:i64 = addr %p, %b, 4\n  store.global %ad, %b\n  condbr %c, x, x\nx:\n  ret\n"));
}

TEST(CopyProp, IgnoresSelfMoves) {
  EXPECT_PASS("copyprop", fn("  %x:i32 = mov %a\n  %x:i32 = mov %x\n  %y:i32 = add %x, 1\n  ret\n"),
              fn("  %x:i32 = mov %a\n  %x:i32 = mov %a\n  %y:i32 = add %a, 1\n  ret\n"));
}

// =========================================================================== dce
TEST(Dce, RemovesUnusedArithmeticAndParams) {
  EXPECT_PASS("dce",
              "kernel k(i32 %a, i32 %b) {\nentry:\n  %a:i32 = param 0\n  %b:i32 = param 1\n"
              "  %x:i32 = add %a, 1\n  ret\n}\n",
              "kernel k(i32 %a, i32 %b) {\nentry:\n  ret\n}\n");
}

TEST(Dce, RemovesDeadChainsToFixpoint) {
  EXPECT_PASS("dce",
              "kernel k(i32 %a) {\nentry:\n  %a:i32 = param 0\n  %x:i32 = add %a, 1\n"
              "  %y:i32 = mul %x, 2\n  %z:i32 = sub %y, %x\n  ret\n}\n",
              "kernel k(i32 %a) {\nentry:\n  ret\n}\n");
}

TEST(Dce, KeepsStoresAndWhatTheyUse) {
  std::string k =
      "kernel k(ptr %p, i32 %a) {\nentry:\n  %p:ptr = param 0\n  %a:i32 = param 1\n"
      "  %x:i32 = mul %a, 3\n  %ad:i64 = addr %p, %a, 4\n  store.global %ad, %x\n  ret\n}\n";
  EXPECT_PASS("dce", k, k);
}

TEST(Dce, KeepsBarriers) {
  std::string k = "kernel k() {\nentry:\n  barrier\n  ret\n}\n";
  EXPECT_PASS("dce", k, k);
}

TEST(Dce, RemovesUnusedLoads) {
  EXPECT_PASS("dce",
              "kernel k(ptr %p) {\nentry:\n  %p:ptr = param 0\n  %ad:i64 = addr %p, 0, 4\n"
              "  %v:f32 = load.global %ad\n  ret\n}\n",
              "kernel k(ptr %p) {\nentry:\n  ret\n}\n");
}

TEST(Dce, KeepsValuesUsedInOtherBlocks) {
  std::string k = "kernel k(i32 %a) {\nentry:\n  %a:i32 = param 0\n  %c:i1 = cmp.lt %a, 3\n"
                  "  br next\nnext:\n  condbr %c, exit, exit\nexit:\n  ret\n}\n";
  EXPECT_PASS("dce", k, k);
}

TEST(Dce, RemovesSelfMoves) {
  EXPECT_PASS("dce",
              "kernel k(ptr %p, i32 %a) {\nentry:\n  %p:ptr = param 0\n  %a:i32 = param 1\n"
              "  %a:i32 = mov %a\n  %ad:i64 = addr %p, %a, 4\n  store.global %ad, 1\n  ret\n}\n",
              "kernel k(ptr %p, i32 %a) {\nentry:\n  %p:ptr = param 0\n  %a:i32 = param 1\n"
              "  %ad:i64 = addr %p, %a, 4\n  store.global %ad, 1\n  ret\n}\n");
}

TEST(Dce, ZeroInitSurvivesWhenReadOnSomePath) {
  // `int x;` then `if (c) x = 5;` then use x: the `mov %x, 0` is needed on the other path.
  std::string k = "kernel k(ptr %p, i32 %a) {\nentry:\n  %p:ptr = param 0\n  %a:i32 = param 1\n"
                  "  %x:i32 = mov 0\n  %c:i1 = cmp.lt %a, 3\n  condbr %c, set, use\n"
                  "set:\n  %x:i32 = mov 5\n  br use\n"
                  "use:\n  %ad:i64 = addr %p, 0, 4\n  store.global %ad, %x\n  ret\n}\n";
  EXPECT_PASS("dce", k, k);
}

TEST(Dce, ZeroInitRemovedWhenNeverRead) {
  EXPECT_PASS("dce", "kernel k() {\nentry:\n  %x:i32 = mov 0\n  %x:i32 = mov 5\n  ret\n}\n",
              "kernel k() {\nentry:\n  ret\n}\n");
}

// =========================================================================== simplifycfg
TEST(SimplifyCfg, CondBrWithSameTargets) {
  EXPECT_PASS("simplifycfg",
              "kernel k() {\nentry:\n  %c:i1 = mov true\n  condbr %c, a, a\na:\n  barrier\n  br b\n"
              "b:\n  barrier\n  ret\n}\n",
              "kernel k() {\nentry:\n  %c:i1 = mov true\n  barrier\n  barrier\n  ret\n}\n");
}

TEST(SimplifyCfg, RemovesUnreachableBlocks) {
  EXPECT_PASS("simplifycfg",
              "kernel k(i32 %a) {\nentry:\n  %a:i32 = param 0\n  %c:i1 = cmp.lt %a, 1\n"
              "  condbr %c, x, y\ndead:\n  barrier\n  br x\nx:\n  barrier\n  ret\ny:\n  ret\n}\n",
              "kernel k(i32 %a) {\nentry:\n  %a:i32 = param 0\n  %c:i1 = cmp.lt %a, 1\n"
              "  condbr %c, x, y\nx:\n  barrier\n  ret\ny:\n  ret\n}\n");
}

TEST(SimplifyCfg, ThreadsJumpsThroughEmptyBlocks) {
  EXPECT_PASS(
      "simplifycfg",
      "kernel k(i32 %a) {\nentry:\n  %a:i32 = param 0\n  %c:i1 = cmp.lt %a, 1\n"
      "  condbr %c, t1, other\nt1:\n  br t2\nt2:\n  br target\nother:\n  barrier\n  br target\n"
      "target:\n  ret\n}\n",
      "kernel k(i32 %a) {\nentry:\n  %a:i32 = param 0\n  %c:i1 = cmp.lt %a, 1\n"
      "  condbr %c, target, other\nother:\n  barrier\n  br target\ntarget:\n  ret\n}\n");
}

TEST(SimplifyCfg, MergesBlockIntoUniquePredecessor) {
  EXPECT_PASS(
      "simplifycfg",
      "kernel k() {\nentry:\n  barrier\n  br next\nnext:\n  barrier\n  br last\nlast:\n  ret\n}\n",
      "kernel k() {\nentry:\n  barrier\n  barrier\n  ret\n}\n");
}

TEST(SimplifyCfg, DoesNotMergeJoinPoints) {
  std::string k =
      "kernel k(i32 %a) {\nentry:\n  %a:i32 = param 0\n  %c:i1 = cmp.lt %a, 1\n"
      "  condbr %c, l, r\nl:\n  barrier\n  br join\nr:\n  barrier\n  barrier\n  br join\n"
      "join:\n  barrier\n  ret\n}\n";
  EXPECT_PASS("simplifycfg", k, k);
}

TEST(SimplifyCfg, KeepsLoops) {
  std::string k = "kernel k(i32 %a) {\nentry:\n  %a:i32 = param 0\n  %i:i32 = mov 0\n  br head\n"
                  "head:\n  %c:i1 = cmp.lt %i, %a\n  condbr %c, body, exit\n"
                  "body:\n  %i:i32 = add %i, 1\n  br head\nexit:\n  ret\n}\n";
  EXPECT_PASS("simplifycfg", k, k);
}

TEST(SimplifyCfg, EmptyInfiniteLoopsTerminate) {
  // An empty self-loop and a two-block trampoline cycle must not hang the pass.
  std::string k = "kernel k(i32 %a) {\nentry:\n  %a:i32 = param 0\n  %c:i1 = cmp.lt %a, 1\n"
                  "  condbr %c, self, c1\nself:\n  br self\nc1:\n  br c2\nc2:\n  br c1\n}\n";
  std::string out = after("simplifycfg", k);
  EXPECT_NE(out.find("br self"), std::string::npos) << out;
}

TEST(SimplifyCfg, EntryIsNeverRemovedOrTargeted) {
  EXPECT_PASS("simplifycfg", "kernel k() {\nentry:\n  br next\nnext:\n  ret\n}\n",
              "kernel k() {\nentry:\n  ret\n}\n");
}

// =========================================================================== pass manager
TEST(PassManager, UnknownPassIsAnError) {
  Module m;
  opt::PipelineConfig pc;
  pc.passes = {"constfold", "nonsense"};
  EXPECT_FALSE(opt::runPipeline(m, pc).ok);
}

TEST(PassManager, IteratesToFixpointAndCountsChanges) {
  Module m;
  m.functions.push_back(parse(fn("  %x:i32 = add 1, 2\n  %y:i32 = mov %x\n  %z:i32 = mul %y, %a\n"
                                 "  %ad:i64 = addr %p, %z, 4\n  store.global %ad, %z\n  br next\n"
                                 "next:\n  ret\n")));
  opt::PipelineConfig pc;
  pc.passes = opt::o1Pipeline();
  pc.verifyEach = true;
  std::ostringstream dump;
  pc.dumpAfter = "dce";
  pc.dump = &dump;
  auto r = opt::runPipeline(m, pc);
  ASSERT_TRUE(r.ok) << r.error;
  EXPECT_GE(r.maxIterationsUsed, 2u); // the last iteration confirms nothing changes
  EXPECT_NE(dump.str().find("; *** IR after dce (iteration 1) ***"), std::string::npos);
  std::string out = printFunction(*m.functions[0]);
  EXPECT_NE(out.find("%z:i32 = mul 3, %a"), std::string::npos) << out;
  EXPECT_EQ(out.find("next:"), std::string::npos) << out;
  unsigned total = 0;
  for (auto &s : r.stats)
    total += s.second;
  EXPECT_GT(total, 0u);
}

TEST(PassManager, DisabledPassesDoNotRun) {
  Module m;
  m.functions.push_back(parse(fn("  %x:i32 = add 1, 2\n  ret\n")));
  opt::PipelineConfig pc;
  pc.passes = opt::o1Pipeline();
  pc.disabled = {"constfold", "dce"};
  ASSERT_TRUE(opt::runPipeline(m, pc).ok);
  EXPECT_NE(printFunction(*m.functions[0]).find("add 1, 2"), std::string::npos);
}
