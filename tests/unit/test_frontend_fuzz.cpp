// Robustness: the lexer + parser must never crash, hang, or trip ASan/UBSan on garbage.
#include "tinycu/parser.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace tinycu;

namespace {

void parseNoCrash(const std::string &src) {
  DiagEngine diag("fuzz.tc", src);
  auto prog = parseSource(src, diag);
  ASSERT_NE(prog, nullptr);
  if (!diag.hasErrors())
    (void)printAST(*prog);
  std::ostringstream os;
  diag.print(os);
}

const char *kSeedProgram = R"(kernel saxpy(float a, float* x, float* y, int n) {
  shared float buf[64];
  int i = blockIdx * blockDim + threadIdx;
  for (int k = 0; k < 4; k++) { if (i < n && k != 2 || !(a > 1.0)) { y[i] += a * x[i]; } else { break; } }
  while (i > 0) { i -= (int)sqrt(abs(2.5e-1f)); continue; }
  __syncthreads(); return;
})";

} // namespace

TEST(FrontendFuzz, RandomBytes) {
  std::mt19937 rng(12345);
  for (int iter = 0; iter < 1000; ++iter) {
    std::string s(rng() % 300, '\0');
    for (char &c : s)
      c = static_cast<char>(rng() & 0xff);
    parseNoCrash(s);
  }
}

TEST(FrontendFuzz, RandomPrintableBytes) {
  std::mt19937 rng(777);
  const std::string alphabet = "kernel int float(){}[];,=+-*/%!<>&|. \n0123456789abcxyz_eEf";
  for (int iter = 0; iter < 1000; ++iter) {
    std::string s(rng() % 300, ' ');
    for (char &c : s)
      c = alphabet[rng() % alphabet.size()];
    parseNoCrash(s);
  }
}

TEST(FrontendFuzz, TokenShuffles) {
  // Lex a valid program, then shuffle / drop / duplicate its tokens.
  std::string src = kSeedProgram;
  DiagEngine d("seed.tc", src);
  Lexer lx(src, d);
  std::vector<Token> toks = lx.lexAll();
  ASSERT_FALSE(d.hasErrors());
  toks.pop_back(); // Eof
  std::mt19937 rng(4242);
  for (int iter = 0; iter < 1000; ++iter) {
    std::vector<Token> t = toks;
    switch (iter % 3) {
    case 0:
      std::shuffle(t.begin(), t.end(), rng);
      break;
    case 1:
      for (int k = 0; k < 5 && !t.empty(); ++k)
        t.erase(t.begin() + static_cast<long>(rng() % t.size()));
      break;
    default:
      for (int k = 0; k < 5; ++k)
        t.insert(t.begin() + static_cast<long>(rng() % (t.size() + 1)), toks[rng() % toks.size()]);
      break;
    }
    std::string s;
    for (const Token &tk : t) {
      s += tk.text;
      s += ' ';
    }
    parseNoCrash(s);
  }
}

TEST(FrontendFuzz, SeedProgramParses) {
  std::string src = kSeedProgram;
  DiagEngine diag("seed.tc", src);
  parseSource(src, diag);
  EXPECT_FALSE(diag.hasErrors());
}
