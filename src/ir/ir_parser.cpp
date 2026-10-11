// Parser for the IR text format produced by ir_printer.cpp (used by the optimizer unit tests:
// "before" text -> pass -> printed "after" text). Round-trips everything the printer emits.
#include "tinycu/ir.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>

namespace tinycu::ir {

namespace {

std::string trim(const std::string &s) {
  size_t b = s.find_first_not_of(" \t\r");
  size_t e = s.find_last_not_of(" \t\r");
  return b == std::string::npos ? "" : s.substr(b, e - b + 1);
}

std::vector<std::string> splitCommas(const std::string &s) {
  std::vector<std::string> parts;
  std::string cur;
  for (char c : s) {
    if (c == ',') {
      parts.push_back(trim(cur));
      cur.clear();
    } else {
      cur += c;
    }
  }
  if (!trim(cur).empty())
    parts.push_back(trim(cur));
  return parts;
}

bool parseType(const std::string &s, IRType &t) {
  static const std::map<std::string, IRType> types = {{"i1", IRType::I1},
                                                      {"i32", IRType::I32},
                                                      {"f32", IRType::F32},
                                                      {"ptr", IRType::PtrGlobal},
                                                      {"ptr.shared", IRType::PtrShared},
                                                      {"i64", IRType::I64}};
  auto it = types.find(s);
  if (it == types.end())
    return false;
  t = it->second;
  return true;
}

class Parser {
public:
  explicit Parser(const std::string &text) {
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
      line = trim(line);
      if (!line.empty() && line.rfind(";", 0) != 0)
        lines_.push_back(line);
    }
  }

  std::unique_ptr<Function> run(std::string &error) {
    fn_ = std::make_unique<Function>();
    if (!header() || !declare() || !body()) {
      error = error_;
      return nullptr;
    }
    return std::move(fn_);
  }

private:
  bool fail(const std::string &msg) {
    if (error_.empty())
      error_ = "line " + std::to_string(cur_ + 1) + ": " + msg;
    return false;
  }

  // kernel NAME(T %a, T %b) {
  bool header() {
    if (lines_.empty() || lines_[0].rfind("kernel ", 0) != 0)
      return fail("expected 'kernel NAME(...) {'");
    const std::string &h = lines_[0];
    size_t lp = h.find('('), rp = h.rfind(')');
    if (lp == std::string::npos || rp == std::string::npos)
      return fail("malformed kernel header");
    fn_->name = trim(h.substr(7, lp - 7));
    for (const std::string &p : splitCommas(h.substr(lp + 1, rp - lp - 1))) {
      size_t sp = p.find(' ');
      IRType t;
      if (sp == std::string::npos || !parseType(p.substr(0, sp), t) || p[sp + 1] != '%')
        return fail("bad parameter '" + p + "'");
      std::string name = p.substr(sp + 2);
      uint32_t r = fn_->newVReg(t, name);
      regs_[name] = r;
      fn_->params.push_back({name, t, t == IRType::PtrGlobal ? IRType::F32 : t, r});
    }
    return true;
  }

  // First pass: shared arrays, labels (create blocks) and every `%x:T =` definition.
  bool declare() {
    for (cur_ = 1; cur_ < lines_.size(); ++cur_) {
      const std::string &l = lines_[cur_];
      if (l == "}")
        break;
      if (l.rfind("shared ", 0) == 0) { // shared T @name[N]
        std::istringstream ss(l.substr(7));
        std::string ty, rest;
        ss >> ty >> rest;
        IRType t;
        size_t lb = rest.find('['), rb = rest.find(']');
        if (!parseType(ty, t) || rest.empty() || rest[0] != '@' || lb == std::string::npos ||
            rb == std::string::npos)
          return fail("bad shared declaration");
        std::string name = rest.substr(1, lb - 1);
        shared_[name] = static_cast<uint32_t>(fn_->shared.size());
        fn_->shared.push_back({name, t, std::atoi(rest.substr(lb + 1, rb - lb - 1).c_str())});
      } else if (l.back() == ':' && l.find(' ') == std::string::npos) {
        std::string label = l.substr(0, l.size() - 1);
        if (blocks_.count(label))
          return fail("duplicate label '" + label + "'");
        fn_->blocks.push_back(std::make_unique<BasicBlock>());
        fn_->blocks.back()->label = label;
        blocks_[label] = fn_->blocks.back().get();
      } else if (l[0] == '%') {
        size_t colon = l.find(':'), eq = l.find(" = ");
        if (colon == std::string::npos || eq == std::string::npos || colon > eq)
          return fail("expected '%name:type = ...'");
        std::string name = l.substr(1, colon - 1);
        IRType t;
        if (!parseType(l.substr(colon + 1, eq - colon - 1), t))
          return fail("unknown type in '" + l + "'");
        auto it = regs_.find(name);
        if (it == regs_.end())
          regs_[name] = fn_->newVReg(t, name);
        else if (fn_->vregs[it->second].type != t)
          return fail("vreg %" + name + " defined with two types");
      }
    }
    return true;
  }

  bool operand(const std::string &tok, Value &v) {
    if (tok.empty())
      return fail("missing operand");
    if (tok[0] == '%') {
      auto it = regs_.find(tok.substr(1));
      if (it == regs_.end())
        return fail("unknown vreg " + tok);
      v = Value::vreg(it->second);
    } else if (tok[0] == '@') {
      auto it = shared_.find(tok.substr(1));
      if (it == shared_.end())
        return fail("unknown shared array " + tok);
      v = Value::shared(it->second);
    } else if (tok == "true" || tok == "false") {
      v = Value::immBool(tok == "true");
    } else if (tok.rfind("f32(0x", 0) == 0) {
      uint32_t bits = static_cast<uint32_t>(std::strtoul(tok.c_str() + 6, nullptr, 16));
      float f;
      std::memcpy(&f, &bits, sizeof f);
      v = Value::immF32(f);
    } else if (tok.back() == 'f') {
      v = Value::immF32(std::strtof(tok.substr(0, tok.size() - 1).c_str(), nullptr));
    } else {
      char *end = nullptr;
      long x = std::strtol(tok.c_str(), &end, 10);
      if (*end != '\0')
        return fail("bad operand '" + tok + "'");
      v = Value::immI32(static_cast<int32_t>(x));
    }
    return true;
  }

  BasicBlock *label(const std::string &name) {
    auto it = blocks_.find(name);
    if (it == blocks_.end()) {
      fail("unknown label '" + name + "'");
      return nullptr;
    }
    return it->second;
  }

  bool body() {
    static const std::map<std::string, Op> ops = {{"add", Op::Add},
                                                  {"sub", Op::Sub},
                                                  {"mul", Op::Mul},
                                                  {"div", Op::Div},
                                                  {"rem", Op::Rem},
                                                  {"neg", Op::Neg},
                                                  {"and", Op::And},
                                                  {"or", Op::Or},
                                                  {"not", Op::Not},
                                                  {"cmp.eq", Op::CmpEq},
                                                  {"cmp.ne", Op::CmpNe},
                                                  {"cmp.lt", Op::CmpLt},
                                                  {"cmp.le", Op::CmpLe},
                                                  {"cmp.gt", Op::CmpGt},
                                                  {"cmp.ge", Op::CmpGe},
                                                  {"cvt.f32.i32", Op::CvtF32I32},
                                                  {"cvt.i32.f32", Op::CvtI32F32},
                                                  {"min", Op::Min},
                                                  {"max", Op::Max},
                                                  {"abs", Op::Abs},
                                                  {"sqrt", Op::Sqrt},
                                                  {"mov", Op::Mov}};
    BasicBlock *bb = nullptr;
    for (cur_ = 1; cur_ < lines_.size(); ++cur_) {
      std::string l = lines_[cur_];
      if (l == "}")
        return true;
      if (l.rfind("shared ", 0) == 0)
        continue;
      if (l.back() == ':' && l.find(' ') == std::string::npos) {
        bb = blocks_[l.substr(0, l.size() - 1)];
        continue;
      }
      if (!bb)
        return fail("instruction before the first label");
      Inst in;
      if (l[0] == '%') {
        size_t colon = l.find(':'), eq = l.find(" = ");
        in.dst = Value::vreg(regs_[l.substr(1, colon - 1)]);
        l = trim(l.substr(eq + 3));
      }
      size_t sp = l.find(' ');
      std::string op = sp == std::string::npos ? l : l.substr(0, sp);
      std::vector<std::string> args =
          sp == std::string::npos ? std::vector<std::string>{} : splitCommas(l.substr(sp + 1));
      auto need = [&](size_t n) {
        return args.size() == n || fail("'" + op + "' expects " + std::to_string(n) + " operands");
      };
      IRType dstTy = in.dst.isVReg() ? fn_->typeOf(in.dst) : IRType::I32;

      if (op == "br") {
        if (!need(1) || !(in.t = label(args[0])))
          return false;
        in.op = Op::Br;
      } else if (op == "condbr") {
        if (!need(3) || !operand(args[0], in.a) || !(in.t = label(args[1])) ||
            !(in.f = label(args[2])))
          return false;
        in.op = Op::CondBr;
        in.type = IRType::I1;
      } else if (op == "ret") {
        in.op = Op::Ret;
      } else if (op == "barrier") {
        in.op = Op::Barrier;
      } else if (op == "load.global" || op == "load.shared") {
        if (!need(1) || !operand(args[0], in.a))
          return false;
        in.op = Op::Load;
        in.type = dstTy;
        in.aux = static_cast<int32_t>(op == "load.shared" ? MemSpace::Shared : MemSpace::Global);
      } else if (op == "store.global" || op == "store.shared") {
        if (!need(2) || !operand(args[0], in.a) || !operand(args[1], in.b))
          return false;
        in.op = Op::Store;
        in.type = fn_->typeOf(in.b);
        in.aux = static_cast<int32_t>(op == "store.shared" ? MemSpace::Shared : MemSpace::Global);
      } else if (op == "addr") {
        if (!need(3) || !operand(args[0], in.a) || !operand(args[1], in.b))
          return false;
        in.op = Op::Addr;
        in.type = IRType::I64;
        in.aux = std::atoi(args[2].c_str());
      } else if (op == "sreg") {
        static const std::map<std::string, SReg> sr = {{"tid", SReg::Tid},
                                                       {"ctaid", SReg::Ctaid},
                                                       {"ntid", SReg::Ntid},
                                                       {"nctaid", SReg::Nctaid}};
        if (!need(1) || !sr.count(args[0]))
          return fail("bad special register");
        in.op = Op::Sreg;
        in.type = IRType::I32;
        in.aux = static_cast<int32_t>(sr.at(args[0]));
      } else if (op == "param") {
        if (!need(1))
          return false;
        in.op = Op::Param;
        in.type = dstTy;
        in.aux = std::atoi(args[0].c_str());
      } else if (ops.count(op)) {
        in.op = ops.at(op);
        if (args.empty() || args.size() > 2)
          return fail("'" + op + "' expects 1 or 2 operands");
        if (!operand(args[0], in.a) || (args.size() == 2 && !operand(args[1], in.b)))
          return false;
        bool cmp = in.op >= Op::CmpEq && in.op <= Op::CmpGe;
        in.type = cmp ? fn_->typeOf(in.a) : dstTy;
      } else {
        return fail("unknown instruction '" + op + "'");
      }
      bb->insts.push_back(in);
    }
    return fail("missing closing '}'");
  }

  std::vector<std::string> lines_;
  size_t cur_ = 0;
  std::unique_ptr<Function> fn_;
  std::map<std::string, uint32_t> regs_;
  std::map<std::string, uint32_t> shared_;
  std::map<std::string, BasicBlock *> blocks_;
  std::string error_;
};

} // namespace

std::unique_ptr<Function> parseFunction(const std::string &text, std::string &error) {
  return Parser(text).run(error);
}

} // namespace tinycu::ir
