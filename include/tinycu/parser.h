#pragma once

#include "tinycu/ast.h"
#include "tinycu/diag.h"
#include "tinycu/token.h"

#include <memory>
#include <string_view>
#include <vector>

namespace tinycu {

/// Recursive-descent parser, one method per grammar rule (docs/LANGUAGE.md).
/// On a syntax error it reports, then synchronizes to the next `;` or `}` (panic mode).
/// Always returns a Program; check `diag.hasErrors()` before using it.
class Parser {
public:
  Parser(std::vector<Token> toks, DiagEngine &diag);
  std::unique_ptr<Program> parseProgram();

private:
  // token helpers
  const Token &tok() const { return toks_[pos_]; }
  const Token &peekTok(size_t ahead) const;
  const Token &prev() const { return toks_[pos_ > 0 ? pos_ - 1 : 0]; }
  bool at(TokKind k) const { return tok().kind == k; }
  bool consume(TokKind k);
  void bump();
  bool stopped() const { return diag_.limitReached(); }

  // diagnostics
  void errorAt(SourceLoc loc, const std::string &msg);
  void errorHere(const std::string &msg);
  SourceLoc afterPrev() const;
  bool expectSemi(const char *after);
  bool expectClose(TokKind close, SourceLoc openLoc);
  void synchronize();

  // grammar rules
  std::unique_ptr<Kernel> parseKernel();
  bool parseType(TypeSpec &out);
  StmtPtr parseStmt();
  std::unique_ptr<BlockStmt> parseBlock();
  StmtPtr parseVarDecl(); ///< vardecl_noSemi
  StmtPtr parseAssign();  ///< assign (no `;`)
  StmtPtr parseIf();
  StmtPtr parseFor();
  StmtPtr parseWhile();
  ExprPtr parseExpr();
  ExprPtr parseLogicOr();
  ExprPtr parseLogicAnd();
  ExprPtr parseEquality();
  ExprPtr parseRelational();
  ExprPtr parseAdditive();
  ExprPtr parseMult();
  ExprPtr parseUnary();
  ExprPtr parsePostfix();
  ExprPtr parsePrimary();
  bool parseBuiltinMember(); ///< optional `.x` after a builtin variable

  struct DepthGuard;
  bool enterNested();

  std::vector<Token> toks_;
  DiagEngine &diag_;
  size_t pos_ = 0;
  unsigned depth_ = 0;
};

/// Convenience: lex + parse a whole buffer (error limit 20).
std::unique_ptr<Program> parseSource(std::string_view src, DiagEngine &diag);

} // namespace tinycu
