// PTX backend: IR -> PTX ISA 7.0 text (instruction selection table in docs/ARCHITECTURE.md).
//
// Register mapping: every IR vreg gets one PTX virtual register of its class
// (i1 -> %p, i32 -> %r, f32 -> %f, ptr/ptr.shared/i64 -> %rd); ptxas does the real allocation.
// Float add/sub/mul/div/sqrt carry `.rn`, which also forbids ptxas from contracting into FMA,
// so results are bit-identical to the interpreter.
#include "tinycu/ptx.h"

#include "tinycu/version.h"

#include <cctype>
#include <cstdio>
#include <cstring>

namespace tinycu::ptx {

using namespace ir;

std::string f32Immediate(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof bits);
  char buf[16];
  std::snprintf(buf, sizeof buf, "0f%08X", bits);
  return buf;
}

std::string ptxLabel(const std::string &irLabel) {
  std::string s = "$L__";
  for (char c : irLabel)
    s += std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
  return s;
}

namespace {

enum class RegClass { Pred, B32, F32, B64 };

RegClass classOf(IRType t) {
  switch (t) {
  case IRType::I1:
    return RegClass::Pred;
  case IRType::I32:
    return RegClass::B32;
  case IRType::F32:
    return RegClass::F32;
  default:
    return RegClass::B64;
  }
}

const char *prefix(RegClass c) {
  switch (c) {
  case RegClass::Pred:
    return "%p";
  case RegClass::B32:
    return "%r";
  case RegClass::F32:
    return "%f";
  case RegClass::B64:
    return "%rd";
  }
  return "%r";
}

/// Is the vreg name a user variable (not a numbered temporary)?
bool isNamed(const VRegInfo &v) {
  return !v.name.empty() && !std::isdigit(static_cast<unsigned char>(v.name[0]));
}

class KernelEmitter {
public:
  KernelEmitter(const Function &fn, const Options &opts) : fn_(fn), opts_(opts) {}

  std::string run() {
    // Assign PTX registers (1-based, like nvcc) in vreg order: deterministic.
    regs_.resize(fn_.vregs.size());
    for (size_t i = 0; i < fn_.vregs.size(); ++i) {
      RegClass c = classOf(fn_.vregs[i].type);
      regs_[i] = prefix(c) + std::to_string(++count_[static_cast<int>(c)]);
    }

    std::string head = ".visible .entry " + fn_.name + "(\n";
    for (size_t i = 0; i < fn_.params.size(); ++i) {
      head += "\t.param " + paramType(fn_.params[i]) + " " + paramName(i);
      head += i + 1 < fn_.params.size() ? ",\n" : "\n";
    }
    head += ")\n{\n";

    for (size_t s = 0; s < fn_.shared.size(); ++s)
      body_ += "\t.shared .align 4 .b8 " + sharedName(s) + "[" +
               std::to_string(4 * static_cast<int64_t>(fn_.shared[s].count)) + "];\n";

    for (size_t b = 0; b < fn_.blocks.size(); ++b) {
      const BasicBlock &bb = *fn_.blocks[b];
      next_ = b + 1 < fn_.blocks.size() ? fn_.blocks[b + 1].get() : nullptr;
      if (b > 0)
        body_ += ptxLabel(bb.label) + ":\n";
      for (const Inst &in : bb.insts)
        inst(in);
    }
    body_ += "}\n";

    // Register declarations need the final counts (temps for immediates may have been added).
    std::string decls;
    static const char *names[] = {".pred", ".b32", ".f32", ".b64"};
    for (int c = 0; c < 4; ++c)
      if (count_[c])
        decls += std::string("\t.reg ") + names[c] + " \t" + prefix(static_cast<RegClass>(c)) +
                 "<" + std::to_string(count_[c] + 1) + ">;\n";
    return head + decls + "\n" + body_;
  }

private:
  // ---- naming ----
  std::string paramName(size_t i) const { return fn_.name + "_param_" + std::to_string(i); }
  static std::string paramType(const IRParam &p) {
    return p.type == IRType::PtrGlobal ? ".u64" : p.type == IRType::F32 ? ".f32" : ".s32";
  }
  std::string sharedName(size_t s) const { return "__shared_" + fn_.shared[s].name; }

  std::string fresh(RegClass c) {
    return prefix(c) + std::to_string(++count_[static_cast<int>(c)]);
  }

  std::string reg(const Value &v) const { return regs_[v.reg]; }

  /// Operand text. Immediates are written inline (PTX allows them for these instructions)
  /// except predicates and where `needReg` is set, which get a fresh register via `mov`.
  std::string operand(const Value &v, bool needReg = false) {
    switch (v.kind) {
    case Value::Kind::VReg:
      return reg(v);
    case Value::Kind::ImmI32:
      if (!needReg)
        return std::to_string(v.i);
      return materialize(RegClass::B32, "mov.u32", std::to_string(v.i));
    case Value::Kind::ImmF32:
      if (!needReg)
        return f32Immediate(v.f);
      return materialize(RegClass::F32, "mov.f32", f32Immediate(v.f));
    case Value::Kind::ImmBool:
      return materialize(RegClass::Pred, "setp.ne.u32", v.b ? "1, 0" : "0, 0");
    case Value::Kind::Shared:
      return materialize(RegClass::B64, "mov.u64", sharedName(v.reg));
    case Value::Kind::None:
      break;
    }
    return "<none>";
  }

  std::string materialize(RegClass c, const char *op, const std::string &src) {
    std::string r = fresh(c);
    line(std::string(op) + " \t" + r + ", " + src + ";");
    return r;
  }

  void line(const std::string &s, const Inst *in = nullptr) {
    body_ += "\t" + s;
    if (opts_.comments && in && in->dst.isVReg() && isNamed(fn_.vregs[in->dst.reg]))
      body_ += " \t// %" + fn_.vregs[in->dst.reg].name;
    body_ += "\n";
  }

  static const char *intOrFloat(IRType t, const char *i, const char *f) {
    return t == IRType::F32 ? f : i;
  }

  // ---- instruction selection ----
  void inst(const Inst &in) {
    std::string d = in.dst.isVReg() ? reg(in.dst) : "";
    auto bin = [&](const std::string &op) {
      std::string a = operand(in.a), b = operand(in.b);
      line(op + " \t" + d + ", " + a + ", " + b + ";", &in);
    };
    auto un = [&](const std::string &op) {
      std::string a = operand(in.a);
      line(op + " \t" + d + ", " + a + ";", &in);
    };
    switch (in.op) {
    case Op::Add:
      return bin(intOrFloat(in.type, "add.s32", "add.rn.f32"));
    case Op::Sub:
      return bin(intOrFloat(in.type, "sub.s32", "sub.rn.f32"));
    case Op::Mul:
      return bin(intOrFloat(in.type, "mul.lo.s32", "mul.rn.f32"));
    case Op::Div:
      return bin(intOrFloat(in.type, "div.s32", "div.rn.f32"));
    case Op::Rem:
      return bin("rem.s32");
    case Op::Neg:
      return un(intOrFloat(in.type, "neg.s32", "neg.f32"));
    case Op::And:
      return bin("and.pred");
    case Op::Or:
      return bin("or.pred");
    case Op::Not:
      return un("not.pred");
    case Op::CmpEq:
    case Op::CmpNe:
    case Op::CmpLt:
    case Op::CmpLe:
    case Op::CmpGt:
    case Op::CmpGe: {
      static const char *cc[] = {"eq", "ne", "lt", "le", "gt", "ge"};
      int k = static_cast<int>(in.op) - static_cast<int>(Op::CmpEq);
      std::string op;
      if (in.type == IRType::F32)
        // C semantics: `!=` is true when an operand is NaN -> the *unordered* setp.neu;
        // the others are ordered (false on NaN).
        op = std::string("setp.") + (in.op == Op::CmpNe ? "neu" : cc[k]) + ".f32";
      else
        op = std::string("setp.") + cc[k] + ".s32";
      return bin(op);
    }
    case Op::CvtF32I32:
      return un("cvt.rn.f32.s32");
    case Op::CvtI32F32:
      return un("cvt.rzi.s32.f32");
    case Op::Min:
      return bin(intOrFloat(in.type, "min.s32", "min.f32"));
    case Op::Max:
      return bin(intOrFloat(in.type, "max.s32", "max.f32"));
    case Op::Abs:
      return un(intOrFloat(in.type, "abs.s32", "abs.f32"));
    case Op::Sqrt:
      return un("sqrt.rn.f32");
    case Op::Mov: {
      IRType t = fn_.typeOf(in.dst);
      if (t == IRType::I1 && in.a.kind == Value::Kind::ImmBool)
        return line(std::string("setp.ne.u32 \t") + d + ", " + (in.a.b ? "1" : "0") + ", 0;", &in);
      const char *op = t == IRType::I1    ? "mov.pred"
                       : t == IRType::I32 ? "mov.u32"
                       : t == IRType::F32 ? "mov.f32"
                                          : "mov.u64";
      return un(op);
    }
    case Op::Addr: {
      std::string base = operand(in.a);
      std::string idx = operand(in.b, /*needReg=*/true); // mul.wide needs a register source
      std::string off = fresh(RegClass::B64);
      line("mul.wide.s32 \t" + off + ", " + idx + ", " + std::to_string(in.aux) + ";");
      line("add.s64 \t" + d + ", " + base + ", " + off + ";", &in);
      return;
    }
    case Op::Load: {
      const char *sp = in.aux == static_cast<int32_t>(MemSpace::Shared) ? "shared" : "global";
      std::string ty = intOrFloat(in.type, "s32", "f32");
      return line(std::string("ld.") + sp + "." + ty + " \t" + d + ", [" + reg(in.a) + "];", &in);
    }
    case Op::Store: {
      const char *sp = in.aux == static_cast<int32_t>(MemSpace::Shared) ? "shared" : "global";
      std::string ty = intOrFloat(in.type, "s32", "f32");
      std::string v = operand(in.b);
      return line(std::string("st.") + sp + "." + ty + " \t[" + reg(in.a) + "], " + v + ";");
    }
    case Op::Sreg: {
      static const char *sr[] = {"%tid.x", "%ctaid.x", "%ntid.x", "%nctaid.x"};
      return line(std::string("mov.u32 \t") + d + ", " + sr[in.aux] + ";", &in);
    }
    case Op::Param: {
      const IRParam &p = fn_.params[static_cast<size_t>(in.aux)];
      std::string slot = "[" + paramName(static_cast<size_t>(in.aux)) + "]";
      if (p.type == IRType::PtrGlobal) {
        // Kernel pointer params are generic addresses; convert once to the global window.
        std::string raw = fresh(RegClass::B64);
        line("ld.param.u64 \t" + raw + ", " + slot + ";");
        return line("cvta.to.global.u64 \t" + d + ", " + raw + ";", &in);
      }
      return line(std::string(p.type == IRType::F32 ? "ld.param.f32" : "ld.param.s32") + " \t" + d +
                      ", " + slot + ";",
                  &in);
    }
    case Op::Barrier:
      return line("bar.sync \t0;");
    case Op::Br:
      if (in.t != next_)
        line("bra.uni \t" + ptxLabel(in.t->label) + ";");
      return;
    case Op::CondBr: {
      if (in.a.kind == Value::Kind::ImmBool) { // constant condition: plain jump
        BasicBlock *target = in.a.b ? in.t : in.f;
        if (target != next_)
          line("bra.uni \t" + ptxLabel(target->label) + ";");
        return;
      }
      std::string p = reg(in.a);
      if (in.t == in.f) {
        if (in.t != next_)
          line("bra.uni \t" + ptxLabel(in.t->label) + ";");
      } else if (in.t == next_) {
        line("@!" + p + " bra \t" + ptxLabel(in.f->label) + ";");
      } else {
        line("@" + p + " bra \t" + ptxLabel(in.t->label) + ";");
        if (in.f != next_)
          line("bra.uni \t" + ptxLabel(in.f->label) + ";");
      }
      return;
    }
    case Op::Ret:
      return line("ret;");
    }
  }

  const Function &fn_;
  const Options &opts_;
  std::vector<std::string> regs_;
  int count_[4] = {0, 0, 0, 0};
  std::string body_;
  const BasicBlock *next_ = nullptr;
};

} // namespace

std::string emitModule(const Module &m, const Options &opts) {
  std::string s = "//\n// Generated by TinyCU " + std::string(version()) +
                  "\n//\n\n.version 7.0\n.target " + opts.sm + "\n.address_size 64\n";
  for (const auto &fn : m.functions)
    s += "\n\t// .globl\t" + fn->name + "\n" + KernelEmitter(*fn, opts).run();
  return s;
}

} // namespace tinycu::ptx
