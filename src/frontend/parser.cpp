#include "tinycu/parser.h"

#include <string>

namespace tinycu {

namespace {
constexpr unsigned kMaxDepth = 200; ///< guards the recursion against stack overflow
constexpr unsigned kErrorLimit = 20;

std::string locStr(SourceLoc l) { return std::to_string(l.line) + ":" + std::to_string(l.col); }

bool isTypeStart(TokKind k) { return k == TokKind::KwInt || k == TokKind::KwFloat; }

bool isBuiltinVar(TokKind k, BuiltinVar &out) {
  switch (k) {
  case TokKind::KwThreadIdx:
    out = BuiltinVar::ThreadIdx;
    return true;
  case TokKind::KwBlockIdx:
    out = BuiltinVar::BlockIdx;
    return true;
  case TokKind::KwBlockDim:
    out = BuiltinVar::BlockDim;
    return true;
  case TokKind::KwGridDim:
    out = BuiltinVar::GridDim;
    return true;
  default:
    return false;
  }
}

bool isBuiltinFn(TokKind k, BuiltinFn &out) {
  switch (k) {
  case TokKind::KwMin:
    out = BuiltinFn::Min;
    return true;
  case TokKind::KwMax:
    out = BuiltinFn::Max;
    return true;
  case TokKind::KwAbs:
    out = BuiltinFn::Abs;
    return true;
  case TokKind::KwSqrt:
    out = BuiltinFn::Sqrt;
    return true;
  default:
    return false;
  }
}
} // namespace

struct Parser::DepthGuard {
  explicit DepthGuard(Parser &p) : p(p) {}
  ~DepthGuard() { --p.depth_; }
  Parser &p;
};

Parser::Parser(std::vector<Token> toks, DiagEngine &diag) : toks_(std::move(toks)), diag_(diag) {
  if (toks_.empty() || toks_.back().kind != TokKind::Eof)
    toks_.push_back(Token{});
}

const Token &Parser::peekTok(size_t ahead) const {
  size_t i = pos_ + ahead;
  return i < toks_.size() ? toks_[i] : toks_.back();
}

void Parser::bump() {
  if (pos_ + 1 < toks_.size())
    ++pos_;
}

bool Parser::consume(TokKind k) {
  if (!at(k))
    return false;
  bump();
  return true;
}

void Parser::errorAt(SourceLoc loc, const std::string &msg) {
  // The lexer already reported invalid tokens; don't cascade.
  if (at(TokKind::Error))
    return;
  diag_.error(loc, msg);
}

void Parser::errorHere(const std::string &msg) { errorAt(tok().loc, msg); }

SourceLoc Parser::afterPrev() const {
  if (pos_ == 0)
    return tok().loc;
  const Token &p = prev();
  return {p.loc.line, p.loc.col + static_cast<uint32_t>(p.text.size())};
}

bool Parser::expectSemi(const char *after) {
  if (consume(TokKind::Semi))
    return true;
  errorAt(afterPrev(), std::string("expected ';' after ") + after);
  return false;
}

bool Parser::expectClose(TokKind close, SourceLoc openLoc) {
  if (consume(close))
    return true;
  const char *open = close == TokKind::RParen ? "'('" : close == TokKind::RBracket ? "'['" : "'{'";
  errorHere(std::string("expected ") + tokKindName(close) + " to match " + open + " at " +
            locStr(openLoc));
  return false;
}

bool Parser::enterNested() {
  if (++depth_ > kMaxDepth) {
    errorHere("nesting too deep (limit " + std::to_string(kMaxDepth) + ")");
    return false;
  }
  return true;
}

// Panic mode: skip to just after the next ';', or stop before '}' / 'kernel' / EOF.
// A '{' met while skipping starts the body of the broken statement (e.g. `if (x {`):
// skip it as a balanced unit so its '}' does not close the enclosing block.
void Parser::synchronize() {
  while (!at(TokKind::Eof)) {
    if (consume(TokKind::Semi))
      return;
    if (at(TokKind::RBrace) || at(TokKind::KwKernel))
      return;
    if (at(TokKind::LBrace)) {
      unsigned depth = 0;
      do {
        if (at(TokKind::LBrace))
          ++depth;
        else if (at(TokKind::RBrace))
          --depth;
        bump();
      } while (depth > 0 && !at(TokKind::Eof) && !at(TokKind::KwKernel));
      return;
    }
    bump();
  }
}

// program := kernel+ EOF
std::unique_ptr<Program> Parser::parseProgram() {
  auto prog = std::make_unique<Program>();
  if (at(TokKind::Eof))
    errorHere("expected at least one 'kernel'");
  while (!at(TokKind::Eof) && !stopped()) {
    if (!at(TokKind::KwKernel)) {
      errorHere(std::string("expected 'kernel', found ") + tokKindName(tok().kind));
      do
        bump();
      while (!at(TokKind::Eof) && !at(TokKind::KwKernel));
      continue;
    }
    if (auto k = parseKernel()) {
      prog->kernels.push_back(std::move(k));
    } else {
      while (!at(TokKind::Eof) && !at(TokKind::KwKernel))
        bump();
    }
  }
  return prog;
}

// type := ("int" | "float") ["*"]
bool Parser::parseType(TypeSpec &out) {
  if (at(TokKind::KwInt))
    out.base = TypeSpec::Base::Int;
  else if (at(TokKind::KwFloat))
    out.base = TypeSpec::Base::Float;
  else
    return false;
  bump();
  out.pointer = consume(TokKind::Star);
  return true;
}

// kernel := "kernel" IDENT "(" [param ("," param)*] ")" block
std::unique_ptr<Kernel> Parser::parseKernel() {
  auto k = std::make_unique<Kernel>();
  k->loc = tok().loc;
  bump(); // 'kernel'
  if (!at(TokKind::Ident)) {
    errorHere("expected kernel name after 'kernel'");
    return nullptr;
  }
  k->name = std::string(tok().text);
  bump();
  SourceLoc lparen = tok().loc;
  if (!consume(TokKind::LParen)) {
    errorHere("expected '(' after kernel name");
    return nullptr;
  }
  if (!at(TokKind::RParen)) {
    do {
      Param p;
      if (!parseType(p.typeSpec)) {
        errorHere(std::string("expected parameter type ('int' or 'float'), found ") +
                  tokKindName(tok().kind));
        return nullptr;
      }
      if (!at(TokKind::Ident)) {
        errorHere("expected parameter name");
        return nullptr;
      }
      p.name = std::string(tok().text);
      p.loc = tok().loc;
      bump();
      k->params.push_back(std::move(p));
    } while (consume(TokKind::Comma));
  }
  if (!expectClose(TokKind::RParen, lparen))
    return nullptr;
  if (!at(TokKind::LBrace)) {
    errorHere("expected '{' to begin kernel body");
    return nullptr;
  }
  k->body = parseBlock();
  if (!k->body)
    return nullptr;
  return k;
}

// block := "{" stmt* "}"
std::unique_ptr<BlockStmt> Parser::parseBlock() {
  if (!enterNested()) {
    DepthGuard g(*this);
    return nullptr;
  }
  DepthGuard g(*this);
  auto b = std::make_unique<BlockStmt>(tok().loc);
  SourceLoc open = tok().loc;
  bump(); // '{'
  while (!at(TokKind::RBrace) && !at(TokKind::Eof) && !at(TokKind::KwKernel) && !stopped()) {
    if (StmtPtr s = parseStmt())
      b->stmts.push_back(std::move(s));
    else
      synchronize();
  }
  if (!expectClose(TokKind::RBrace, open))
    return nullptr;
  return b;
}

// stmt := vardecl | assign ";" | ifstmt | forstmt | whilestmt | "return" ";" | "break" ";"
//       | "continue" ";" | block | "__syncthreads" "(" ")" ";"
StmtPtr Parser::parseStmt() {
  if (!enterNested()) {
    DepthGuard g(*this);
    return nullptr;
  }
  DepthGuard g(*this);
  SourceLoc loc = tok().loc;
  switch (tok().kind) {
  case TokKind::KwShared:
  case TokKind::KwInt:
  case TokKind::KwFloat: {
    StmtPtr d = parseVarDecl();
    if (!d || !expectSemi("declaration"))
      return nullptr;
    return d;
  }
  case TokKind::KwIf:
    return parseIf();
  case TokKind::KwFor:
    return parseFor();
  case TokKind::KwWhile:
    return parseWhile();
  case TokKind::KwReturn:
    bump();
    if (!expectSemi("'return'"))
      return nullptr;
    return std::make_unique<ReturnStmt>(loc);
  case TokKind::KwBreak:
    bump();
    if (!expectSemi("'break'"))
      return nullptr;
    return std::make_unique<BreakStmt>(loc);
  case TokKind::KwContinue:
    bump();
    if (!expectSemi("'continue'"))
      return nullptr;
    return std::make_unique<ContinueStmt>(loc);
  case TokKind::LBrace:
    return parseBlock();
  case TokKind::KwSyncThreads: {
    bump();
    SourceLoc lp = tok().loc;
    if (!consume(TokKind::LParen)) {
      errorHere("expected '(' after '__syncthreads'");
      return nullptr;
    }
    if (!expectClose(TokKind::RParen, lp) || !expectSemi("'__syncthreads()'"))
      return nullptr;
    return std::make_unique<SyncThreadsStmt>(loc);
  }
  case TokKind::Ident:
  case TokKind::KwThreadIdx:
  case TokKind::KwBlockIdx:
  case TokKind::KwBlockDim:
  case TokKind::KwGridDim: {
    StmtPtr a = parseAssign();
    if (!a || !expectSemi("assignment"))
      return nullptr;
    return a;
  }
  default:
    errorHere(std::string("expected statement, found ") + tokKindName(tok().kind));
    return nullptr;
  }
}

// vardecl_noSemi := ["shared"] type IDENT ["[" INTLIT "]"] ["=" expr]
StmtPtr Parser::parseVarDecl() {
  auto d = std::make_unique<VarDeclStmt>(tok().loc);
  d->shared = consume(TokKind::KwShared);
  if (!parseType(d->typeSpec)) {
    errorHere("expected type after 'shared'");
    return nullptr;
  }
  if (!at(TokKind::Ident)) {
    errorHere(std::string("expected variable name, found ") + tokKindName(tok().kind));
    return nullptr;
  }
  d->name = std::string(tok().text);
  d->nameLoc = tok().loc;
  bump();
  if (at(TokKind::LBracket)) {
    SourceLoc lb = tok().loc;
    bump();
    if (!at(TokKind::IntLit)) {
      errorHere("expected integer literal array size");
      return nullptr;
    }
    d->arraySize = tok().intVal;
    bump();
    if (!expectClose(TokKind::RBracket, lb))
      return nullptr;
  }
  if (consume(TokKind::Assign)) {
    d->init = parseExpr();
    if (!d->init)
      return nullptr;
  }
  return d;
}

// assign := lvalue ("=" | "+=" | "-=" | "*=" | "/=") expr | lvalue ("++" | "--")
// lvalue := IDENT | IDENT "[" expr "]"
// Builtin variables are accepted as lvalue heads here so sema can reject them with a
// precise "read-only" diagnostic.
StmtPtr Parser::parseAssign() {
  SourceLoc loc = tok().loc;
  ExprPtr target;
  BuiltinVar bv;
  if (at(TokKind::Ident)) {
    target = std::make_unique<VarRefExpr>(loc, std::string(tok().text));
  } else if (isBuiltinVar(tok().kind, bv)) {
    target = std::make_unique<BuiltinVarExpr>(loc, bv);
  } else {
    errorHere(std::string("expected assignment target, found ") + tokKindName(tok().kind));
    return nullptr;
  }
  std::string name(tok().text);
  bump();
  if (isa<BuiltinVarExpr>(target.get()) && !parseBuiltinMember())
    return nullptr;
  if (at(TokKind::LBracket)) {
    SourceLoc lb = tok().loc;
    bump();
    ExprPtr idx = parseExpr();
    if (!idx || !expectClose(TokKind::RBracket, lb))
      return nullptr;
    target = std::make_unique<IndexExpr>(loc, std::move(target), std::move(idx));
  }
  AssignOp op;
  switch (tok().kind) {
  case TokKind::Assign:
    op = AssignOp::Set;
    break;
  case TokKind::PlusAssign:
    op = AssignOp::Add;
    break;
  case TokKind::MinusAssign:
    op = AssignOp::Sub;
    break;
  case TokKind::StarAssign:
    op = AssignOp::Mul;
    break;
  case TokKind::SlashAssign:
    op = AssignOp::Div;
    break;
  case TokKind::PlusPlus:
  case TokKind::MinusMinus: {
    // Desugar x++ / x-- into x += 1 / x -= 1.
    op = at(TokKind::PlusPlus) ? AssignOp::Add : AssignOp::Sub;
    SourceLoc opLoc = tok().loc;
    bump();
    return std::make_unique<AssignStmt>(loc, op, std::move(target),
                                        std::make_unique<IntLitExpr>(opLoc, 1));
  }
  default:
    errorHere("expected assignment operator after '" + name + "'");
    return nullptr;
  }
  bump();
  ExprPtr value = parseExpr();
  if (!value)
    return nullptr;
  return std::make_unique<AssignStmt>(loc, op, std::move(target), std::move(value));
}

// ifstmt := "if" "(" expr ")" stmt ["else" stmt]   (else binds to the nearest if)
StmtPtr Parser::parseIf() {
  auto s = std::make_unique<IfStmt>(tok().loc);
  bump();
  SourceLoc lp = tok().loc;
  if (!consume(TokKind::LParen)) {
    errorHere("expected '(' after 'if'");
    return nullptr;
  }
  s->cond = parseExpr();
  if (!s->cond || !expectClose(TokKind::RParen, lp))
    return nullptr;
  s->thenStmt = parseStmt();
  if (!s->thenStmt)
    return nullptr;
  if (consume(TokKind::KwElse)) {
    s->elseStmt = parseStmt();
    if (!s->elseStmt)
      return nullptr;
  }
  return s;
}

// forstmt := "for" "(" [vardecl_noSemi | assign] ";" [expr] ";" [assign] ")" stmt
StmtPtr Parser::parseFor() {
  auto s = std::make_unique<ForStmt>(tok().loc);
  bump();
  SourceLoc lp = tok().loc;
  if (!consume(TokKind::LParen)) {
    errorHere("expected '(' after 'for'");
    return nullptr;
  }
  if (!at(TokKind::Semi)) {
    s->init = (at(TokKind::KwShared) || isTypeStart(tok().kind)) ? parseVarDecl() : parseAssign();
    if (!s->init)
      return nullptr;
  }
  if (!consume(TokKind::Semi)) {
    errorAt(afterPrev(), "expected ';' after 'for' initializer");
    return nullptr;
  }
  if (!at(TokKind::Semi)) {
    s->cond = parseExpr();
    if (!s->cond)
      return nullptr;
  }
  if (!consume(TokKind::Semi)) {
    errorAt(afterPrev(), "expected ';' after 'for' condition");
    return nullptr;
  }
  if (!at(TokKind::RParen)) {
    s->step = parseAssign();
    if (!s->step)
      return nullptr;
  }
  if (!expectClose(TokKind::RParen, lp))
    return nullptr;
  s->body = parseStmt();
  if (!s->body)
    return nullptr;
  return s;
}

// whilestmt := "while" "(" expr ")" stmt
StmtPtr Parser::parseWhile() {
  auto s = std::make_unique<WhileStmt>(tok().loc);
  bump();
  SourceLoc lp = tok().loc;
  if (!consume(TokKind::LParen)) {
    errorHere("expected '(' after 'while'");
    return nullptr;
  }
  s->cond = parseExpr();
  if (!s->cond || !expectClose(TokKind::RParen, lp))
    return nullptr;
  s->body = parseStmt();
  if (!s->body)
    return nullptr;
  return s;
}

// expr := logic_or
ExprPtr Parser::parseExpr() { return parseLogicOr(); }

// logic_or := logic_and ("||" logic_and)*
ExprPtr Parser::parseLogicOr() {
  ExprPtr lhs = parseLogicAnd();
  while (lhs && at(TokKind::PipePipe)) {
    SourceLoc loc = tok().loc;
    bump();
    ExprPtr rhs = parseLogicAnd();
    if (!rhs)
      return nullptr;
    lhs = std::make_unique<BinaryExpr>(loc, BinaryOp::Or, std::move(lhs), std::move(rhs));
  }
  return lhs;
}

// logic_and := equality ("&&" equality)*
ExprPtr Parser::parseLogicAnd() {
  ExprPtr lhs = parseEquality();
  while (lhs && at(TokKind::AmpAmp)) {
    SourceLoc loc = tok().loc;
    bump();
    ExprPtr rhs = parseEquality();
    if (!rhs)
      return nullptr;
    lhs = std::make_unique<BinaryExpr>(loc, BinaryOp::And, std::move(lhs), std::move(rhs));
  }
  return lhs;
}

// equality := relational (("=="|"!=") relational)*
ExprPtr Parser::parseEquality() {
  ExprPtr lhs = parseRelational();
  while (lhs && (at(TokKind::EqEq) || at(TokKind::NotEq))) {
    BinaryOp op = at(TokKind::EqEq) ? BinaryOp::Eq : BinaryOp::Ne;
    SourceLoc loc = tok().loc;
    bump();
    ExprPtr rhs = parseRelational();
    if (!rhs)
      return nullptr;
    lhs = std::make_unique<BinaryExpr>(loc, op, std::move(lhs), std::move(rhs));
  }
  return lhs;
}

// relational := additive (("<"|"<="|">"|">=") additive)*
ExprPtr Parser::parseRelational() {
  ExprPtr lhs = parseAdditive();
  for (;;) {
    if (!lhs)
      return nullptr;
    BinaryOp op;
    switch (tok().kind) {
    case TokKind::Less:
      op = BinaryOp::Lt;
      break;
    case TokKind::LessEq:
      op = BinaryOp::Le;
      break;
    case TokKind::Greater:
      op = BinaryOp::Gt;
      break;
    case TokKind::GreaterEq:
      op = BinaryOp::Ge;
      break;
    default:
      return lhs;
    }
    SourceLoc loc = tok().loc;
    bump();
    ExprPtr rhs = parseAdditive();
    if (!rhs)
      return nullptr;
    lhs = std::make_unique<BinaryExpr>(loc, op, std::move(lhs), std::move(rhs));
  }
}

// additive := mult (("+"|"-") mult)*
ExprPtr Parser::parseAdditive() {
  ExprPtr lhs = parseMult();
  while (lhs && (at(TokKind::Plus) || at(TokKind::Minus))) {
    BinaryOp op = at(TokKind::Plus) ? BinaryOp::Add : BinaryOp::Sub;
    SourceLoc loc = tok().loc;
    bump();
    ExprPtr rhs = parseMult();
    if (!rhs)
      return nullptr;
    lhs = std::make_unique<BinaryExpr>(loc, op, std::move(lhs), std::move(rhs));
  }
  return lhs;
}

// mult := unary (("*"|"/"|"%") unary)*
ExprPtr Parser::parseMult() {
  ExprPtr lhs = parseUnary();
  while (lhs && (at(TokKind::Star) || at(TokKind::Slash) || at(TokKind::Percent))) {
    BinaryOp op = at(TokKind::Star)    ? BinaryOp::Mul
                  : at(TokKind::Slash) ? BinaryOp::Div
                                       : BinaryOp::Rem;
    SourceLoc loc = tok().loc;
    bump();
    ExprPtr rhs = parseUnary();
    if (!rhs)
      return nullptr;
    lhs = std::make_unique<BinaryExpr>(loc, op, std::move(lhs), std::move(rhs));
  }
  return lhs;
}

// unary := ("-"|"!") unary | "(" type ")" unary | postfix
ExprPtr Parser::parseUnary() {
  if (!enterNested()) {
    DepthGuard g(*this);
    return nullptr;
  }
  DepthGuard g(*this);
  SourceLoc loc = tok().loc;
  if (at(TokKind::Minus) || at(TokKind::Bang)) {
    UnaryOp op = at(TokKind::Minus) ? UnaryOp::Neg : UnaryOp::Not;
    bump();
    ExprPtr e = parseUnary();
    if (!e)
      return nullptr;
    return std::make_unique<UnaryExpr>(loc, op, std::move(e));
  }
  // Cast disambiguation: '(' followed by a type keyword is a cast.
  if (at(TokKind::LParen) && isTypeStart(peekTok(1).kind)) {
    bump();
    TypeSpec ts;
    parseType(ts);
    if (!expectClose(TokKind::RParen, loc))
      return nullptr;
    ExprPtr e = parseUnary();
    if (!e)
      return nullptr;
    return std::make_unique<CastExpr>(loc, ts, std::move(e));
  }
  return parsePostfix();
}

// postfix := primary ["[" expr "]"]
ExprPtr Parser::parsePostfix() {
  SourceLoc loc = tok().loc;
  ExprPtr e = parsePrimary();
  if (e && at(TokKind::LBracket)) {
    SourceLoc lb = tok().loc;
    bump();
    ExprPtr idx = parseExpr();
    if (!idx || !expectClose(TokKind::RBracket, lb))
      return nullptr;
    e = std::make_unique<IndexExpr>(loc, std::move(e), std::move(idx));
  }
  return e;
}

// primary := INTLIT | FLOATLIT | "true" | "false" | IDENT | builtin_var
//          | builtin_fn "(" [expr ("," expr)*] ")" | "(" expr ")"
ExprPtr Parser::parsePrimary() {
  const Token &t = tok();
  SourceLoc loc = t.loc;
  BuiltinVar bv;
  BuiltinFn bf;
  switch (t.kind) {
  case TokKind::IntLit: {
    auto e = std::make_unique<IntLitExpr>(loc, t.intVal);
    bump();
    return e;
  }
  case TokKind::FloatLit: {
    auto e = std::make_unique<FloatLitExpr>(loc, t.floatVal, std::string(t.text));
    bump();
    return e;
  }
  case TokKind::KwTrue:
  case TokKind::KwFalse: {
    auto e = std::make_unique<BoolLitExpr>(loc, t.kind == TokKind::KwTrue);
    bump();
    return e;
  }
  case TokKind::Ident: {
    auto e = std::make_unique<VarRefExpr>(loc, std::string(t.text));
    bump();
    return e;
  }
  case TokKind::LParen: {
    bump();
    ExprPtr e = parseExpr();
    if (!e || !expectClose(TokKind::RParen, loc))
      return nullptr;
    return e;
  }
  default:
    break;
  }
  if (isBuiltinVar(t.kind, bv)) {
    bump();
    if (!parseBuiltinMember())
      return nullptr;
    return std::make_unique<BuiltinVarExpr>(loc, bv);
  }
  if (isBuiltinFn(t.kind, bf)) {
    std::string name(t.text);
    bump();
    SourceLoc lp = tok().loc;
    if (!consume(TokKind::LParen)) {
      errorHere("expected '(' after '" + name + "'");
      return nullptr;
    }
    std::vector<ExprPtr> args;
    if (!at(TokKind::RParen)) {
      do {
        ExprPtr a = parseExpr();
        if (!a)
          return nullptr;
        args.push_back(std::move(a));
      } while (consume(TokKind::Comma));
    }
    if (!expectClose(TokKind::RParen, lp))
      return nullptr;
    return std::make_unique<CallExpr>(loc, bf, std::move(args));
  }
  errorHere(std::string("expected expression, found ") + tokKindName(t.kind));
  return nullptr;
}

// builtin_var := ("threadIdx" | "blockIdx" | "blockDim" | "gridDim") ["." "x"]
// Both spellings produce the same BuiltinVarExpr; `.y` / `.z` get a dedicated error.
bool Parser::parseBuiltinMember() {
  if (!consume(TokKind::Dot))
    return true;
  if (at(TokKind::Ident) && tok().text == "x") {
    bump();
    return true;
  }
  if (at(TokKind::Ident) && (tok().text == "y" || tok().text == "z")) {
    errorHere("only the x dimension is supported (TinyCU is 1-D)");
    return false;
  }
  errorHere(std::string("expected 'x' after '.', found ") + tokKindName(tok().kind));
  return false;
}

std::unique_ptr<Program> parseSource(std::string_view src, DiagEngine &diag) {
  diag.setErrorLimit(kErrorLimit);
  Lexer lex(src, diag);
  Parser p(lex.lexAll(), diag);
  return p.parseProgram();
}

} // namespace tinycu
