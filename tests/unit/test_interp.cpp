// Interpreter (the oracle): semantics of every construct, and every UB check.
#include "tinycu/interp.h"
#include "tinycu/ir.h"
#include "tinycu/parser.h"
#include "tinycu/sema.h"

#include <gtest/gtest.h>

#include <climits>
#include <cmath>
#include <cstring>

using namespace tinycu;
using namespace tinycu::ir;
using interp::ArgValue;

namespace {

std::unique_ptr<Module> compile(const std::string &src) {
  static std::string keep;
  keep = src;
  DiagEngine diag("t.tc", keep);
  auto prog = parseSource(keep, diag);
  EXPECT_FALSE(diag.hasErrors());
  EXPECT_TRUE(Sema(diag).run(*prog));
  auto m = lowerProgram(*prog);
  for (const auto &fn : m->functions)
    EXPECT_TRUE(verify(*fn).empty());
  return m;
}

uint32_t fbits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, sizeof b);
  return b;
}
float bitsf(uint32_t b) {
  float f;
  std::memcpy(&f, &b, sizeof f);
  return f;
}

ArgValue I(int32_t v) {
  ArgValue a;
  a.type = IRType::I32;
  a.bits = static_cast<uint32_t>(v);
  return a;
}
ArgValue F(float v) {
  ArgValue a;
  a.type = IRType::F32;
  a.elem = IRType::F32;
  a.bits = fbits(v);
  return a;
}
ArgValue IP(std::vector<int32_t> v) {
  ArgValue a;
  a.type = IRType::PtrGlobal;
  a.elem = IRType::I32;
  for (int32_t x : v)
    a.data.push_back(static_cast<uint32_t>(x));
  return a;
}
ArgValue FP(std::vector<float> v) {
  ArgValue a;
  a.type = IRType::PtrGlobal;
  a.elem = IRType::F32;
  for (float x : v)
    a.data.push_back(fbits(x));
  return a;
}
std::vector<int32_t> ints(const ArgValue &a) {
  std::vector<int32_t> r;
  for (uint32_t x : a.data)
    r.push_back(static_cast<int32_t>(x));
  return r;
}
std::vector<float> floats(const ArgValue &a) {
  std::vector<float> r;
  for (uint32_t x : a.data)
    r.push_back(bitsf(x));
  return r;
}

interp::Result launch(const std::string &src, std::vector<ArgValue> &args, uint32_t grid,
                      uint32_t block) {
  auto m = compile(src);
  interp::LaunchConfig cfg;
  cfg.grid = grid;
  cfg.block = block;
  cfg.stepLimit = 100000;
  return interp::run(*m->functions[0], cfg, args);
}

const char *kVecAdd = R"(kernel vecadd(int* a, int* b, int* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { c[i] = a[i] + b[i]; }
})";

} // namespace

TEST(Interp, VecAddWithPartialLastBlock) {
  std::vector<ArgValue> args = {IP({1, 2, 3, 4, 5}), IP({10, 20, 30, 40, 50}), IP({0, 0, 0, 0, 0}),
                                I(5)};
  auto r = launch(kVecAdd, args, 2, 4); // 8 threads, 5 elements
  ASSERT_TRUE(r.ok) << r.error;
  EXPECT_EQ(ints(args[2]), (std::vector<int32_t>{11, 22, 33, 44, 55}));
}

TEST(Interp, Saxpy) {
  std::vector<ArgValue> args = {F(2.5f), FP({1, 2, 3}), FP({0.5f, 0.25f, -1}), I(3)};
  auto r = launch(R"(kernel saxpy(float a, float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { y[i] = a * x[i] + y[i]; } })",
                  args, 1, 32);
  ASSERT_TRUE(r.ok) << r.error;
  EXPECT_EQ(floats(args[2]), (std::vector<float>{3.0f, 5.25f, 6.5f}));
}

TEST(Interp, Relu) {
  std::vector<ArgValue> args = {FP({-1.5f, 0.0f, 2.0f, -0.0f}), FP({9, 9, 9, 9}), I(4)};
  auto r = launch(R"(kernel relu(float* x, float* y, int n) {
    int i = threadIdx.x;
    if (i < n) { float v = x[i]; if (v < 0.0) { v = 0.0; } y[i] = v; } })",
                  args, 1, 4);
  ASSERT_TRUE(r.ok) << r.error;
  auto y = floats(args[1]);
  EXPECT_EQ(y[0], 0.0f);
  EXPECT_EQ(y[2], 2.0f);
  EXPECT_TRUE(std::signbit(y[3])); // -0.0 < 0.0 is false: -0.0 passes through
}

TEST(Interp, LoopSum) {
  std::vector<ArgValue> args = {IP({0, 0, 0, 0})};
  auto r = launch(R"(kernel k(int* o) {
    int s = 0;
    for (int j = 0; j <= threadIdx.x * 10; j++) { s += j; }
    o[threadIdx.x] = s; })",
                  args, 1, 4);
  ASSERT_TRUE(r.ok) << r.error;
  EXPECT_EQ(ints(args[0]), (std::vector<int32_t>{0, 55, 210, 465}));
}

TEST(Interp, NestedIf) {
  std::vector<ArgValue> args = {IP({0, 0, 0, 0, 0, 0})};
  auto r = launch(R"(kernel k(int* o) {
    int t = threadIdx.x;
    if (t < 3) { if (t == 0) { o[t] = 10; } else { o[t] = 20; } }
    else { if (t % 2 == 0) { o[t] = 30; } else if (t == 5) { o[t] = 50; } else { o[t] = 40; } } })",
                  args, 1, 6);
  ASSERT_TRUE(r.ok) << r.error;
  EXPECT_EQ(ints(args[0]), (std::vector<int32_t>{10, 20, 20, 40, 30, 50}));
}

TEST(Interp, BreakAndContinue) {
  std::vector<ArgValue> args = {IP({0, 0})};
  auto r = launch(R"(kernel k(int* o) {
    int s = 0;
    for (int j = 0; j < 100; j++) { if (j % 2 == 1) { continue; } if (j > 10) { break; } s += j; }
    int w = 0;
    while (true) { w++; if (w >= 7) { break; } }
    o[0] = s; o[1] = w; })",
                  args, 1, 1);
  ASSERT_TRUE(r.ok) << r.error;
  EXPECT_EQ(ints(args[0]), (std::vector<int32_t>{30, 7})); // 0+2+4+6+8+10
}

TEST(Interp, ShortCircuitGuardsOutOfBounds) {
  std::vector<ArgValue> args = {IP({5, -1, 7}), IP({0, 0, 0, 0, 0, 0, 0, 0}), I(3)};
  auto r = launch(R"(kernel k(int* x, int* o, int n) {
    int i = threadIdx.x;
    if (i < n && x[i] > 0) { o[i] = 1; }
    if (i >= n || x[i] < 0) { o[i] = 2; } })",
                  args, 1, 8);
  ASSERT_TRUE(r.ok) << r.error;
  EXPECT_EQ(ints(args[1]), (std::vector<int32_t>{1, 2, 1, 2, 2, 2, 2, 2}));
}

TEST(Interp, IntegerWrapAround) {
  std::vector<ArgValue> args = {IP({0, 0, 0, 0})};
  auto r = launch(R"(kernel k(int* o) {
    int m = 2147483647;
    o[0] = m + 1;
    o[1] = (-2147483647 - 1) - 1;
    o[2] = 65536 * 65536 + 7;
    o[3] = abs(-2147483647 - 1); })",
                  args, 1, 1);
  ASSERT_TRUE(r.ok) << r.error;
  EXPECT_EQ(ints(args[0]), (std::vector<int32_t>{INT_MIN, INT_MAX, 7, INT_MIN}));
}

TEST(Interp, DivisionTruncatesTowardZero) {
  std::vector<ArgValue> args = {IP({0, 0, 0, 0})};
  auto r = launch(
      R"(kernel k(int* o) { o[0] = -7 / 2; o[1] = -7 % 2; o[2] = 7 % -2; o[3] = (int)(-2.9); })",
      args, 1, 1);
  ASSERT_TRUE(r.ok) << r.error;
  EXPECT_EQ(ints(args[0]), (std::vector<int32_t>{-3, -1, 1, -2}));
}

TEST(Interp, SharedMemoryReduction) {
  std::vector<float> in(100);
  float expect0 = 0, expect1 = 0;
  for (int i = 0; i < 100; ++i) {
    in[static_cast<size_t>(i)] = static_cast<float>(i);
    (i < 64 ? expect0 : expect1) += static_cast<float>(i);
  }
  std::vector<ArgValue> args = {FP(in), FP({0, 0}), I(100)};
  auto r = launch(R"(kernel reduce(float* in, float* out, int n) {
    shared float buf[64];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float v = 0.0;
    if (i < n) { v = in[i]; }
    buf[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
      if (tid < s) { buf[tid] += buf[tid + s]; }
      __syncthreads();
    }
    if (tid == 0) { out[blockIdx.x] = buf[0]; } })",
                  args, 2, 64);
  ASSERT_TRUE(r.ok) << r.error;
  EXPECT_EQ(floats(args[1]), (std::vector<float>{expect0, expect1}));
}

TEST(Interp, FloatSemantics) {
  std::vector<ArgValue> args = {FP({0, 0, 0, 0, 0, 0}), IP({0, 0, 0})};
  auto r = launch(R"(kernel k(float* f, int* b) {
    float z = 0.0;
    float nan = z / z;
    f[0] = sqrt(2.0);
    f[1] = min(nan, 3.0);
    f[2] = max(-0.0, 0.0);
    f[3] = (float)16777217;
    f[4] = 1.0 / z;
    f[5] = abs(-0.0);
    if (nan != nan) { b[0] = 1; }
    if (nan == nan) { b[1] = 1; }
    if (nan < 1.0 || nan >= 1.0) { b[2] = 1; } })",
                  args, 1, 1);
  ASSERT_TRUE(r.ok) << r.error;
  auto f = floats(args[0]);
  EXPECT_EQ(f[0], std::sqrt(2.0f));
  EXPECT_EQ(f[1], 3.0f);
  EXPECT_EQ(f[2], 0.0f);
  EXPECT_FALSE(std::signbit(f[2]));
  EXPECT_EQ(f[3], 16777216.0f); // round to nearest even
  EXPECT_TRUE(std::isinf(f[4]));
  EXPECT_FALSE(std::signbit(f[5]));
  EXPECT_EQ(ints(args[1]), (std::vector<int32_t>{1, 0, 0}));
}

TEST(Interp, UninitializedLocalIsZero) {
  std::vector<ArgValue> args = {IP({9}), FP({9})};
  auto r = launch("kernel k(int* o, float* f) { int x; float y; o[0] = x; f[0] = y; }", args, 1, 1);
  ASSERT_TRUE(r.ok) << r.error;
  EXPECT_EQ(ints(args[0])[0], 0);
  EXPECT_EQ(floats(args[1])[0], 0.0f);
}

TEST(Interp, GridStrideAndSpecialRegisters) {
  std::vector<ArgValue> args = {IP(std::vector<int32_t>(20, 0)), I(20)};
  auto r = launch(R"(kernel k(int* o, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
      o[i] = i * 100 + gridDim.x * 10 + blockDim.x; } })",
                  args, 2, 3);
  ASSERT_TRUE(r.ok) << r.error;
  for (int i = 0; i < 20; ++i)
    EXPECT_EQ(ints(args[0])[static_cast<size_t>(i)], i * 100 + 23);
}

// ---- undefined behaviour is reported, never executed ----

TEST(InterpErrors, OutOfBoundsLoad) {
  std::vector<ArgValue> args = {IP({1, 2, 3})};
  auto r = launch("kernel k(int* o) { o[0] = o[threadIdx.x + 1]; }", args, 1, 4);
  ASSERT_FALSE(r.ok);
  EXPECT_NE(r.error.find("out-of-bounds load from 'o' at index 3 (size 3)"), std::string::npos)
      << r.error;
  EXPECT_NE(r.error.find("thread 2"), std::string::npos) << r.error;
}

TEST(InterpErrors, OutOfBoundsNegativeStore) {
  std::vector<ArgValue> args = {IP({1})};
  auto r = launch("kernel k(int* o) { o[-1] = 5; }", args, 1, 1);
  ASSERT_FALSE(r.ok);
  EXPECT_NE(r.error.find("out-of-bounds store to 'o' at index -1"), std::string::npos) << r.error;
}

TEST(InterpErrors, SharedOutOfBounds) {
  std::vector<ArgValue> args = {IP({0})};
  auto r =
      launch("kernel k(int* o) { shared int s[4]; s[threadIdx.x] = 1; o[0] = s[0]; }", args, 1, 5);
  ASSERT_FALSE(r.ok);
  EXPECT_NE(r.error.find("'s' at index 4 (size 4)"), std::string::npos) << r.error;
}

TEST(InterpErrors, DivisionByZero) {
  std::vector<ArgValue> args = {IP({0}), I(0)};
  auto r = launch("kernel k(int* o, int z) { o[0] = 5 / z; }", args, 1, 1);
  ASSERT_FALSE(r.ok);
  EXPECT_NE(r.error.find("integer division by zero"), std::string::npos);
  r = launch("kernel k(int* o, int z) { o[0] = 5 % z; }", args, 1, 1);
  ASSERT_FALSE(r.ok);
}

TEST(InterpErrors, IntMinDividedByMinusOne) {
  std::vector<ArgValue> args = {IP({0}), I(-1)};
  auto r = launch("kernel k(int* o, int m) { o[0] = (-2147483647 - 1) / m; }", args, 1, 1);
  ASSERT_FALSE(r.ok);
  EXPECT_NE(r.error.find("INT_MIN / -1"), std::string::npos);
}

TEST(InterpErrors, FloatToIntOutOfRange) {
  std::vector<ArgValue> args = {IP({0}), F(3e9f)};
  auto r = launch("kernel k(int* o, float f) { o[0] = (int)f; }", args, 1, 1);
  ASSERT_FALSE(r.ok);
  EXPECT_NE(r.error.find("float-to-int"), std::string::npos);
  std::vector<ArgValue> ok = {IP({0}), F(-2147483648.0f)};
  EXPECT_TRUE(launch("kernel k(int* o, float f) { o[0] = (int)f; }", ok, 1, 1).ok);
}

TEST(InterpErrors, InfiniteLoopHitsStepLimit) {
  std::vector<ArgValue> args = {IP({0})};
  auto r = launch("kernel k(int* o) { while (true) { o[0] += 1; } }", args, 1, 1);
  ASSERT_FALSE(r.ok);
  EXPECT_NE(r.error.find("step limit"), std::string::npos);
}

TEST(InterpErrors, BarrierDivergenceEarlyExit) {
  std::vector<ArgValue> args = {IP({0})};
  auto r =
      launch("kernel k(int* o) { if (threadIdx.x == 0) { return; } __syncthreads(); o[0] = 1; }",
             args, 1, 4);
  ASSERT_FALSE(r.ok);
  EXPECT_NE(r.error.find("barrier divergence"), std::string::npos) << r.error;
}

TEST(InterpErrors, BarrierDivergenceDifferentBarriers) {
  std::vector<ArgValue> args = {IP({0})};
  auto r = launch(
      "kernel k(int* o) { if (threadIdx.x < 2) { __syncthreads(); } else { __syncthreads(); } }",
      args, 1, 4);
  ASSERT_FALSE(r.ok);
  EXPECT_NE(r.error.find("different __syncthreads"), std::string::npos) << r.error;
}

TEST(InterpErrors, WrongArgumentCount) {
  std::vector<ArgValue> args = {IP({0})};
  auto r = launch(kVecAdd, args, 1, 1);
  ASSERT_FALSE(r.ok);
  EXPECT_NE(r.error.find("expects 4 arguments"), std::string::npos);
}
