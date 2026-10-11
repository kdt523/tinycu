#pragma once

// TinyCU three-address IR (see docs/IR.md).
//
// A Function is a list of BasicBlocks (entry = blocks[0]). Every block ends with exactly one
// terminator (br / condbr / ret). Values are virtual registers or immediates. The IR is
// deliberately *not* SSA: a source variable maps to one named vreg and assignment is `mov`.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tinycu::ir {

enum class IRType : uint8_t {
  I1,        ///< predicate
  I32,       ///< int
  F32,       ///< float
  PtrGlobal, ///< pointer parameter (global memory); printed `ptr`
  PtrShared, ///< base of a shared array; printed `ptr.shared`
  I64,       ///< computed byte address
};
const char *typeName(IRType t);

struct Value {
  enum class Kind : uint8_t { None, VReg, ImmI32, ImmF32, ImmBool, Shared };
  Kind kind = Kind::None;
  uint32_t reg = 0; ///< VReg: vreg id. Shared: index into Function::shared
  int32_t i = 0;
  float f = 0.f;
  bool b = false;

  static Value vreg(uint32_t id) {
    Value v;
    v.kind = Kind::VReg;
    v.reg = id;
    return v;
  }
  static Value immI32(int32_t x) {
    Value v;
    v.kind = Kind::ImmI32;
    v.i = x;
    return v;
  }
  static Value immF32(float x) {
    Value v;
    v.kind = Kind::ImmF32;
    v.f = x;
    return v;
  }
  static Value immBool(bool x) {
    Value v;
    v.kind = Kind::ImmBool;
    v.b = x;
    return v;
  }
  static Value shared(uint32_t idx) {
    Value v;
    v.kind = Kind::Shared;
    v.reg = idx;
    return v;
  }
  bool isNone() const { return kind == Kind::None; }
  bool isVReg() const { return kind == Kind::VReg; }
  bool isImm() const {
    return kind == Kind::ImmI32 || kind == Kind::ImmF32 || kind == Kind::ImmBool;
  }
  bool operator==(const Value &o) const;
  bool operator!=(const Value &o) const { return !(*this == o); }
};

enum class Op : uint8_t {
  // arithmetic (Inst::type = I32 or F32; Rem is I32 only)
  Add,
  Sub,
  Mul,
  Div,
  Rem,
  Neg,
  // predicate logic (I1)
  And,
  Or,
  Not,
  // comparisons: Inst::type is the *operand* type, result is I1
  CmpEq,
  CmpNe,
  CmpLt,
  CmpLe,
  CmpGt,
  CmpGe,
  // conversions
  CvtF32I32, ///< int -> float, round to nearest even
  CvtI32F32, ///< float -> int, truncate toward zero
  // builtins (Inst::type = I32 or F32; Sqrt is F32 only)
  Min,
  Max,
  Abs,
  Sqrt,
  // moves and memory
  Mov,
  Addr,  ///< dst:i64 = a (ptr base) + b (i32 index) * aux (element size)
  Load,  ///< dst = load.<space> a(i64);  aux = MemSpace
  Store, ///< store.<space> a(i64), b;     aux = MemSpace
  // special
  Sreg,    ///< dst:i32 = special register aux (SReg)
  Param,   ///< dst = kernel parameter number aux
  Barrier, ///< __syncthreads()
  // terminators
  Br,     ///< br t
  CondBr, ///< condbr a, t, f
  Ret,
};
const char *opName(Op op);

enum class MemSpace : int32_t { Global = 0, Shared = 1 };
enum class SReg : int32_t { Tid = 0, Ctaid = 1, Ntid = 2, Nctaid = 3 };
const char *sregName(SReg s);

struct BasicBlock;

struct Inst {
  Op op = Op::Ret;
  IRType type = IRType::I32; ///< operation type (operand type for compares)
  Value dst;                 ///< VReg, or None for store/barrier/terminators
  Value a, b;
  int32_t aux = 0;
  BasicBlock *t = nullptr; ///< br / condbr true target
  BasicBlock *f = nullptr; ///< condbr false target

  bool isTerminator() const { return op == Op::Br || op == Op::CondBr || op == Op::Ret; }
  /// May be removed when its result is unused (no side effects, cannot trap observably).
  bool hasSideEffects() const { return op == Op::Store || op == Op::Barrier || isTerminator(); }
};

struct BasicBlock {
  std::string label;
  std::vector<Inst> insts;
  const Inst *terminator() const {
    return !insts.empty() && insts.back().isTerminator() ? &insts.back() : nullptr;
  }
};

struct VRegInfo {
  IRType type;
  std::string name; ///< unique printable name without `%` (e.g. "i", "i.1", "7")
};

struct SharedArray {
  std::string name;
  IRType elemType; ///< I32 or F32
  int32_t count;
};

struct IRParam {
  std::string name;
  IRType type;  ///< I32, F32 or PtrGlobal
  IRType elem;  ///< element type for PtrGlobal (I32/F32), otherwise == type
  uint32_t reg; ///< vreg holding it after `param`
};

struct Function {
  std::string name;
  std::vector<IRParam> params;
  std::vector<SharedArray> shared;
  std::vector<std::unique_ptr<BasicBlock>> blocks; ///< blocks[0] is the entry
  std::vector<VRegInfo> vregs;

  /// New temporary (named by its id) or named vreg (uniqued: "i", "i.1", ...).
  uint32_t newVReg(IRType t, const std::string &name = "");
  BasicBlock *entry() const { return blocks.empty() ? nullptr : blocks[0].get(); }
  IRType typeOf(const Value &v) const;
  bool usesBarrier() const;
};

struct Module {
  std::vector<std::unique_ptr<Function>> functions;
  Function *find(const std::string &name) const;
};

// ---- printing (ir_printer.cpp) ----
std::string printValue(const Function &fn, const Value &v);
std::string printFunction(const Function &fn);
std::string printModule(const Module &m);

/// Parses the text printed by printFunction (ir_parser.cpp; used by optimizer unit tests).
/// Returns nullptr and sets `error` on malformed input.
std::unique_ptr<Function> parseFunction(const std::string &text, std::string &error);

// ---- verification (verifier.cpp) ----
/// Returns a list of problems; empty means the function is well-formed.
std::vector<std::string> verify(const Function &fn);

// ---- CFG utilities (cfg.cpp) ----
std::vector<BasicBlock *> successors(const BasicBlock &bb);
/// Predecessor lists indexed like fn.blocks (duplicates removed, in block order).
std::vector<std::vector<BasicBlock *>> predecessors(const Function &fn);
std::vector<BasicBlock *> reversePostOrder(const Function &fn);
std::vector<bool> reachable(const Function &fn); ///< indexed like fn.blocks
size_t blockIndex(const Function &fn, const BasicBlock *bb);

} // namespace tinycu::ir

namespace tinycu {
struct Program;
/// Lower a type-checked program to IR (lower.cpp).
std::unique_ptr<ir::Module> lowerProgram(const Program &prog);
} // namespace tinycu
