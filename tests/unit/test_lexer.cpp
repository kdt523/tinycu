#include "tinycu/token.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace tinycu;

namespace {

struct Lexed {
  std::vector<Token> toks;
  unsigned errors;
};

Lexed lex(std::string_view src) {
  static std::string keep; // tokens point into the source
  keep = std::string(src);
  DiagEngine diag("t.tc", keep);
  Lexer lx(keep, diag);
  Lexed r{lx.lexAll(), diag.errorCount()};
  return r;
}

std::vector<TokKind> kinds(std::string_view src) {
  std::vector<TokKind> ks;
  for (const Token &t : lex(src).toks)
    ks.push_back(t.kind);
  return ks;
}

using K = TokKind;

} // namespace

TEST(Lexer, Empty) { EXPECT_EQ(kinds(""), std::vector<K>{K::Eof}); }

TEST(Lexer, Keywords) {
  EXPECT_EQ(kinds("kernel int float shared if else for while return break continue"),
            (std::vector<K>{K::KwKernel, K::KwInt, K::KwFloat, K::KwShared, K::KwIf, K::KwElse,
                            K::KwFor, K::KwWhile, K::KwReturn, K::KwBreak, K::KwContinue, K::Eof}));
}

TEST(Lexer, BuiltinKeywords) {
  EXPECT_EQ(kinds("threadIdx blockIdx blockDim gridDim min max abs sqrt __syncthreads true false"),
            (std::vector<K>{K::KwThreadIdx, K::KwBlockIdx, K::KwBlockDim, K::KwGridDim, K::KwMin,
                            K::KwMax, K::KwAbs, K::KwSqrt, K::KwSyncThreads, K::KwTrue, K::KwFalse,
                            K::Eof}));
}

TEST(Lexer, IdentifiersAreNotKeywordPrefixes) {
  EXPECT_EQ(kinds("kernels int_ _x x1 iff"),
            (std::vector<K>{K::Ident, K::Ident, K::Ident, K::Ident, K::Ident, K::Eof}));
}

TEST(Lexer, MaximalMunchOperators) {
  EXPECT_EQ(kinds("<= >= == != && || += -= *= /= ++ --"),
            (std::vector<K>{K::LessEq, K::GreaterEq, K::EqEq, K::NotEq, K::AmpAmp, K::PipePipe,
                            K::PlusAssign, K::MinusAssign, K::StarAssign, K::SlashAssign,
                            K::PlusPlus, K::MinusMinus, K::Eof}));
}

TEST(Lexer, MunchWithoutSpaces) {
  // "a+++b" lexes as a ++ + b (maximal munch), "x<=-1" as x <= - 1.
  EXPECT_EQ(kinds("a+++b"), (std::vector<K>{K::Ident, K::PlusPlus, K::Plus, K::Ident, K::Eof}));
  EXPECT_EQ(kinds("x<=-1"), (std::vector<K>{K::Ident, K::LessEq, K::Minus, K::IntLit, K::Eof}));
}

TEST(Lexer, SingleCharOperators) {
  EXPECT_EQ(kinds("+-*/%!=<>(){}[],;"),
            (std::vector<K>{K::Plus, K::Minus, K::Star, K::Slash, K::Percent, K::NotEq, K::Less,
                            K::Greater, K::LParen, K::RParen, K::LBrace, K::RBrace, K::LBracket,
                            K::RBracket, K::Comma, K::Semi, K::Eof}));
}

TEST(Lexer, IntLiterals) {
  Lexed r = lex("0 42 2147483647 007");
  ASSERT_EQ(r.errors, 0u);
  EXPECT_EQ(r.toks[0].intVal, 0);
  EXPECT_EQ(r.toks[1].intVal, 42);
  EXPECT_EQ(r.toks[2].intVal, 2147483647);
  EXPECT_EQ(r.toks[3].intVal, 7);
}

TEST(Lexer, IntOverflowIsError) {
  Lexed r = lex("2147483648 99999999999999999999");
  EXPECT_EQ(r.errors, 2u);
  EXPECT_EQ(r.toks[0].kind, K::Error);
  EXPECT_EQ(r.toks[1].kind, K::Error);
}

TEST(Lexer, FloatLiterals) {
  Lexed r = lex("1.5 1.0f 2e-3 3E+2f 0.0");
  ASSERT_EQ(r.errors, 0u);
  for (int i = 0; i < 5; ++i)
    EXPECT_EQ(r.toks[i].kind, K::FloatLit) << i;
  EXPECT_EQ(r.toks[0].floatVal, 1.5f);
  EXPECT_EQ(r.toks[1].floatVal, 1.0f);
  EXPECT_EQ(r.toks[2].floatVal, 2e-3f);
  EXPECT_EQ(r.toks[3].floatVal, 300.0f);
  EXPECT_EQ(r.toks[1].text, "1.0f");
}

TEST(Lexer, FloatOverflowAndUnderflowAreErrors) {
  EXPECT_EQ(lex("1e39").errors, 1u);
  EXPECT_EQ(lex("1e-50").errors, 1u);
  EXPECT_EQ(lex("0e-50").errors, 0u); // zero is exactly representable
  EXPECT_EQ(lex("1e-40").errors, 0u); // subnormal, not zero
}

TEST(Lexer, DotWithoutDigitsIsNotFloat) {
  // "1." is int 1 followed by an invalid '.'; "1.x" likewise.
  Lexed r = lex("1.");
  EXPECT_EQ(r.toks[0].kind, K::IntLit);
  EXPECT_EQ(r.toks[1].kind, K::Error);
  EXPECT_EQ(r.errors, 1u);
}

TEST(Lexer, InvalidSuffix) {
  Lexed r = lex("12abc 1.5x");
  EXPECT_EQ(r.errors, 2u);
  EXPECT_EQ(r.toks[0].kind, K::Error);
  EXPECT_EQ(r.toks[0].text, "12abc");
}

TEST(Lexer, CommentsAreSkipped) {
  EXPECT_EQ(kinds("a // line comment\n/* block\n comment */ b /**/c"),
            (std::vector<K>{K::Ident, K::Ident, K::Ident, K::Eof}));
}

TEST(Lexer, UnterminatedBlockComment) {
  Lexed r = lex("a /* oops");
  EXPECT_EQ(r.errors, 1u);
  EXPECT_EQ(r.toks.back().kind, K::Eof);
}

TEST(Lexer, UnknownCharactersContinueLexing) {
  Lexed r = lex("a @ b # c & d | e .");
  EXPECT_EQ(r.errors, 5u);
  EXPECT_EQ(r.toks.size(), 11u); // 5 idents + 5 errors + eof
}

TEST(Lexer, SourceLocations) {
  Lexed r = lex("a\n  bb\n\tc");
  EXPECT_EQ(r.toks[0].loc.line, 1u);
  EXPECT_EQ(r.toks[0].loc.col, 1u);
  EXPECT_EQ(r.toks[1].loc.line, 2u);
  EXPECT_EQ(r.toks[1].loc.col, 3u);
  EXPECT_EQ(r.toks[2].loc.line, 3u);
  EXPECT_EQ(r.toks[2].loc.col, 2u);
}

TEST(Lexer, CrlfLineEndings) {
  Lexed r = lex("a\r\nb");
  EXPECT_EQ(r.errors, 0u);
  EXPECT_EQ(r.toks[1].loc.line, 2u);
  EXPECT_EQ(r.toks[1].loc.col, 1u);
}

TEST(Lexer, NulAndHighBytes) {
  std::string s("a\0b\xff", 4);
  Lexed r = lex(s);
  EXPECT_EQ(r.errors, 2u);
  EXPECT_EQ(r.toks.back().kind, K::Eof);
}

TEST(Lexer, DotOnlyAfterBuiltinVariables) {
  EXPECT_EQ(
      kinds("threadIdx.x blockIdx . x"),
      (std::vector<K>{K::KwThreadIdx, K::Dot, K::Ident, K::KwBlockIdx, K::Dot, K::Ident, K::Eof}));
  // Elsewhere '.' is not a token, and float literals are unaffected.
  Lexed r = lex("a.x 1.5");
  EXPECT_EQ(r.errors, 1u);
  EXPECT_EQ(r.toks[1].kind, K::Error);
  EXPECT_EQ(r.toks[3].kind, K::FloatLit);
}
