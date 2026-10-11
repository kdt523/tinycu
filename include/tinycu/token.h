#pragma once

#include "tinycu/diag.h"

#include <cstdint>
#include <string_view>
#include <vector>

namespace tinycu {

enum class TokKind {
  // special
  Eof,
  Error,
  // literals / names
  Ident,
  IntLit,
  FloatLit,
  // keywords
  KwKernel,
  KwInt,
  KwFloat,
  KwShared,
  KwIf,
  KwElse,
  KwFor,
  KwWhile,
  KwReturn,
  KwBreak,
  KwContinue,
  KwTrue,
  KwFalse,
  KwSyncThreads,
  KwThreadIdx,
  KwBlockIdx,
  KwBlockDim,
  KwGridDim,
  KwMin,
  KwMax,
  KwAbs,
  KwSqrt,
  // punctuation
  LParen,
  RParen,
  LBrace,
  RBrace,
  LBracket,
  RBracket,
  Comma,
  Semi,
  // operators
  Plus,
  Minus,
  Star,
  Slash,
  Percent,
  Bang,
  Assign,
  PlusAssign,
  MinusAssign,
  StarAssign,
  SlashAssign,
  PlusPlus,
  MinusMinus,
  EqEq,
  NotEq,
  Less,
  LessEq,
  Greater,
  GreaterEq,
  AmpAmp,
  PipePipe,
};

/// Human-readable spelling used in diagnostics, e.g. "';'" or "identifier".
const char *tokKindName(TokKind k);

struct Token {
  TokKind kind = TokKind::Eof;
  std::string_view text; ///< points into the source buffer
  SourceLoc loc;
  int32_t intVal = 0;   ///< valid for IntLit
  float floatVal = 0.f; ///< valid for FloatLit
};

/// Hand-written maximal-munch lexer. Errors are reported to `diag` and produce a
/// TokKind::Error token; lexing always continues to Eof.
class Lexer {
public:
  Lexer(std::string_view src, DiagEngine &diag) : src_(src), diag_(diag) {}

  Token next();
  /// Lex the whole buffer; the last token is always Eof.
  std::vector<Token> lexAll();

private:
  char peek(size_t ahead = 0) const {
    return pos_ + ahead < src_.size() ? src_[pos_ + ahead] : '\0';
  }
  void advance();
  bool skipTriviaAndComments();
  Token lexNumber(SourceLoc start, size_t begin);
  Token make(TokKind k, size_t begin, SourceLoc loc) const;

  std::string_view src_;
  DiagEngine &diag_;
  size_t pos_ = 0;
  uint32_t line_ = 1;
  uint32_t col_ = 1;
};

} // namespace tinycu
