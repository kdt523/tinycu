#include "tinycu/token.h"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <string>

namespace tinycu {

const char *tokKindName(TokKind k) {
  switch (k) {
  case TokKind::Eof:
    return "end of file";
  case TokKind::Error:
    return "invalid token";
  case TokKind::Ident:
    return "identifier";
  case TokKind::IntLit:
    return "integer literal";
  case TokKind::FloatLit:
    return "float literal";
  case TokKind::KwKernel:
    return "'kernel'";
  case TokKind::KwInt:
    return "'int'";
  case TokKind::KwFloat:
    return "'float'";
  case TokKind::KwShared:
    return "'shared'";
  case TokKind::KwIf:
    return "'if'";
  case TokKind::KwElse:
    return "'else'";
  case TokKind::KwFor:
    return "'for'";
  case TokKind::KwWhile:
    return "'while'";
  case TokKind::KwReturn:
    return "'return'";
  case TokKind::KwBreak:
    return "'break'";
  case TokKind::KwContinue:
    return "'continue'";
  case TokKind::KwTrue:
    return "'true'";
  case TokKind::KwFalse:
    return "'false'";
  case TokKind::KwSyncThreads:
    return "'__syncthreads'";
  case TokKind::KwThreadIdx:
    return "'threadIdx'";
  case TokKind::KwBlockIdx:
    return "'blockIdx'";
  case TokKind::KwBlockDim:
    return "'blockDim'";
  case TokKind::KwGridDim:
    return "'gridDim'";
  case TokKind::KwMin:
    return "'min'";
  case TokKind::KwMax:
    return "'max'";
  case TokKind::KwAbs:
    return "'abs'";
  case TokKind::KwSqrt:
    return "'sqrt'";
  case TokKind::LParen:
    return "'('";
  case TokKind::RParen:
    return "')'";
  case TokKind::LBrace:
    return "'{'";
  case TokKind::RBrace:
    return "'}'";
  case TokKind::LBracket:
    return "'['";
  case TokKind::RBracket:
    return "']'";
  case TokKind::Comma:
    return "','";
  case TokKind::Semi:
    return "';'";
  case TokKind::Dot:
    return "'.'";
  case TokKind::Plus:
    return "'+'";
  case TokKind::Minus:
    return "'-'";
  case TokKind::Star:
    return "'*'";
  case TokKind::Slash:
    return "'/'";
  case TokKind::Percent:
    return "'%'";
  case TokKind::Bang:
    return "'!'";
  case TokKind::Assign:
    return "'='";
  case TokKind::PlusAssign:
    return "'+='";
  case TokKind::MinusAssign:
    return "'-='";
  case TokKind::StarAssign:
    return "'*='";
  case TokKind::SlashAssign:
    return "'/='";
  case TokKind::PlusPlus:
    return "'++'";
  case TokKind::MinusMinus:
    return "'--'";
  case TokKind::EqEq:
    return "'=='";
  case TokKind::NotEq:
    return "'!='";
  case TokKind::Less:
    return "'<'";
  case TokKind::LessEq:
    return "'<='";
  case TokKind::Greater:
    return "'>'";
  case TokKind::GreaterEq:
    return "'>='";
  case TokKind::AmpAmp:
    return "'&&'";
  case TokKind::PipePipe:
    return "'||'";
  }
  return "token";
}

namespace {

struct Keyword {
  std::string_view text;
  TokKind kind;
};

constexpr Keyword kKeywords[] = {
    {"kernel", TokKind::KwKernel},
    {"int", TokKind::KwInt},
    {"float", TokKind::KwFloat},
    {"shared", TokKind::KwShared},
    {"if", TokKind::KwIf},
    {"else", TokKind::KwElse},
    {"for", TokKind::KwFor},
    {"while", TokKind::KwWhile},
    {"return", TokKind::KwReturn},
    {"break", TokKind::KwBreak},
    {"continue", TokKind::KwContinue},
    {"true", TokKind::KwTrue},
    {"false", TokKind::KwFalse},
    {"__syncthreads", TokKind::KwSyncThreads},
    {"threadIdx", TokKind::KwThreadIdx},
    {"blockIdx", TokKind::KwBlockIdx},
    {"blockDim", TokKind::KwBlockDim},
    {"gridDim", TokKind::KwGridDim},
    {"min", TokKind::KwMin},
    {"max", TokKind::KwMax},
    {"abs", TokKind::KwAbs},
    {"sqrt", TokKind::KwSqrt},
};

bool isIdentStart(char c) { return std::isalpha(static_cast<unsigned char>(c)) || c == '_'; }
bool isIdentChar(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }
bool isDigit(char c) { return c >= '0' && c <= '9'; }

} // namespace

void Lexer::advance() {
  if (pos_ >= src_.size())
    return;
  if (src_[pos_] == '\n') {
    ++line_;
    col_ = 1;
  } else {
    ++col_;
  }
  ++pos_;
}

Token Lexer::make(TokKind k, size_t begin, SourceLoc loc) const {
  Token t;
  t.kind = k;
  t.text = src_.substr(begin, pos_ - begin);
  t.loc = loc;
  return t;
}

// Returns false if an unterminated block comment hit end of file.
bool Lexer::skipTriviaAndComments() {
  for (;;) {
    char c = peek();
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v') {
      advance();
    } else if (c == '/' && peek(1) == '/') {
      while (pos_ < src_.size() && peek() != '\n')
        advance();
    } else if (c == '/' && peek(1) == '*') {
      SourceLoc start{line_, col_};
      advance();
      advance();
      for (;;) {
        if (pos_ >= src_.size()) {
          diag_.error(start, "unterminated block comment");
          return false;
        }
        if (peek() == '*' && peek(1) == '/') {
          advance();
          advance();
          break;
        }
        advance();
      }
    } else {
      return true;
    }
  }
}

Token Lexer::lexNumber(SourceLoc start, size_t begin) {
  bool isFloat = false;
  while (isDigit(peek()))
    advance();
  if (peek() == '.' && isDigit(peek(1))) {
    isFloat = true;
    advance();
    while (isDigit(peek()))
      advance();
  }
  if ((peek() == 'e' || peek() == 'E') &&
      (isDigit(peek(1)) || ((peek(1) == '+' || peek(1) == '-') && isDigit(peek(2))))) {
    isFloat = true;
    advance();
    if (peek() == '+' || peek() == '-')
      advance();
    while (isDigit(peek()))
      advance();
  }
  size_t numEnd = pos_;
  if (isFloat && (peek() == 'f' || peek() == 'F'))
    advance();
  if (isIdentChar(peek())) {
    while (isIdentChar(peek()))
      advance();
    diag_.error(start, "invalid suffix on numeric literal '" +
                           std::string(src_.substr(begin, pos_ - begin)) + "'");
    return make(TokKind::Error, begin, start);
  }

  std::string digits(src_.substr(begin, numEnd - begin));
  if (!isFloat) {
    uint64_t v = 0;
    for (char c : digits) {
      v = v * 10 + static_cast<uint64_t>(c - '0');
      if (v > 2147483647ull) {
        diag_.error(start,
                    "integer literal '" + digits + "' is too large for int (max 2147483647)");
        return make(TokKind::Error, begin, start);
      }
    }
    Token t = make(TokKind::IntLit, begin, start);
    t.intVal = static_cast<int32_t>(v);
    return t;
  }

  float f = std::strtof(digits.c_str(), nullptr);
  if (std::isinf(f)) {
    diag_.error(start, "float literal '" + digits + "' overflows float");
    return make(TokKind::Error, begin, start);
  }
  // A literal with any non-zero mantissa digit must not round to zero.
  bool nonZeroMantissa = false;
  for (char c : digits) {
    if (c == 'e' || c == 'E')
      break;
    if (c >= '1' && c <= '9')
      nonZeroMantissa = true;
  }
  if (f == 0.0f && nonZeroMantissa) {
    diag_.error(start, "float literal '" + digits + "' underflows to zero");
    return make(TokKind::Error, begin, start);
  }
  Token t = make(TokKind::FloatLit, begin, start);
  t.floatVal = f;
  return t;
}

Token Lexer::lexToken() {
  if (!skipTriviaAndComments()) {
    Token t;
    t.kind = TokKind::Eof;
    t.loc = {line_, col_};
    return t;
  }
  SourceLoc start{line_, col_};
  size_t begin = pos_;
  if (pos_ >= src_.size())
    return make(TokKind::Eof, begin, start);

  char c = peek();
  if (isIdentStart(c)) {
    while (isIdentChar(peek()))
      advance();
    std::string_view text = src_.substr(begin, pos_ - begin);
    for (const Keyword &kw : kKeywords)
      if (kw.text == text)
        return make(kw.kind, begin, start);
    return make(TokKind::Ident, begin, start);
  }
  if (isDigit(c))
    return lexNumber(start, begin);

  advance();
  auto two = [&](char second, TokKind twoKind, TokKind oneKind) {
    if (peek() == second) {
      advance();
      return make(twoKind, begin, start);
    }
    return make(oneKind, begin, start);
  };
  switch (c) {
  case '(':
    return make(TokKind::LParen, begin, start);
  case ')':
    return make(TokKind::RParen, begin, start);
  case '{':
    return make(TokKind::LBrace, begin, start);
  case '}':
    return make(TokKind::RBrace, begin, start);
  case '[':
    return make(TokKind::LBracket, begin, start);
  case ']':
    return make(TokKind::RBracket, begin, start);
  case ',':
    return make(TokKind::Comma, begin, start);
  case ';':
    return make(TokKind::Semi, begin, start);
  case '.':
    // `.` is only a token in the member access `threadIdx.x` (and the other builtins).
    if (lastKind_ == TokKind::KwThreadIdx || lastKind_ == TokKind::KwBlockIdx ||
        lastKind_ == TokKind::KwBlockDim || lastKind_ == TokKind::KwGridDim)
      return make(TokKind::Dot, begin, start);
    break;
  case '%':
    return make(TokKind::Percent, begin, start);
  case '+':
    if (peek() == '+') {
      advance();
      return make(TokKind::PlusPlus, begin, start);
    }
    return two('=', TokKind::PlusAssign, TokKind::Plus);
  case '-':
    if (peek() == '-') {
      advance();
      return make(TokKind::MinusMinus, begin, start);
    }
    return two('=', TokKind::MinusAssign, TokKind::Minus);
  case '*':
    return two('=', TokKind::StarAssign, TokKind::Star);
  case '/':
    return two('=', TokKind::SlashAssign, TokKind::Slash);
  case '=':
    return two('=', TokKind::EqEq, TokKind::Assign);
  case '!':
    return two('=', TokKind::NotEq, TokKind::Bang);
  case '<':
    return two('=', TokKind::LessEq, TokKind::Less);
  case '>':
    return two('=', TokKind::GreaterEq, TokKind::Greater);
  case '&':
    if (peek() == '&') {
      advance();
      return make(TokKind::AmpAmp, begin, start);
    }
    break;
  case '|':
    if (peek() == '|') {
      advance();
      return make(TokKind::PipePipe, begin, start);
    }
    break;
  default:
    break;
  }
  unsigned char uc = static_cast<unsigned char>(c);
  std::string shown = std::isprint(uc) ? std::string(1, c)
                                       : "\\x" + std::string(1, "0123456789abcdef"[uc >> 4]) +
                                             "0123456789abcdef"[uc & 15];
  diag_.error(start, "unexpected character '" + shown + "'");
  return make(TokKind::Error, begin, start);
}

Token Lexer::next() {
  Token t = lexToken();
  lastKind_ = t.kind;
  return t;
}

std::vector<Token> Lexer::lexAll() {
  std::vector<Token> toks;
  for (;;) {
    toks.push_back(next());
    if (toks.back().kind == TokKind::Eof)
      return toks;
  }
}

} // namespace tinycu
