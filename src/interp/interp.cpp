// The TinyCU IR interpreter (reference oracle).
//
// Memory model: every pointer parameter is its own buffer, every shared array is a per-block
// buffer. An address (i64) encodes (buffer id, byte offset) so that loads/stores can be checked
// precisely; computing an out-of-range address is fine, *accessing* it is an error.
#include "tinycu/interp.h"
#include "tinycu/semantics.h"

#include <climits>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace tinycu::interp {

using namespace ir;

namespace {

constexpr int kBufShift = 40;
constexpr uint64_t kOffsetBias = 1ull << 39;
constexpr uint32_t kSharedBufBase = 1u << 20;

uint64_t baseAddress(uint32_t bufId) {
  return (static_cast<uint64_t>(bufId) << kBufShift) + kOffsetBias;
}

uint64_t fromF32(float f) {
  uint32_t b;
  std::memcpy(&b, &f, sizeof b);
  return b;
}
int32_t asI32(uint64_t bits) { return static_cast<int32_t>(static_cast<uint32_t>(bits)); }

enum class ThreadState { Running, AtBarrier, Done };

struct Thread {
  uint32_t tid = 0;
  const BasicBlock *bb = nullptr;
  size_t ip = 0;
  std::vector<uint64_t> regs;
  uint64_t steps = 0;
  ThreadState state = ThreadState::Running;
};

class Machine {
public:
  Machine(const Function &fn, const LaunchConfig &cfg, std::vector<ArgValue> &args)
      : fn_(fn), cfg_(cfg), args_(args) {}

  Result run() {
    if (args_.size() != fn_.params.size())
      return fail("kernel '" + fn_.name + "' expects " + std::to_string(fn_.params.size()) +
                  " arguments, got " + std::to_string(args_.size()));
    for (size_t i = 0; i < args_.size(); ++i)
      if (args_[i].type != fn_.params[i].type ||
          (args_[i].type == IRType::PtrGlobal && args_[i].elem != fn_.params[i].elem))
        return fail("argument " + std::to_string(i) + " ('" + fn_.params[i].name +
                    "') has the wrong type");
    if (cfg_.grid == 0 || cfg_.block == 0)
      return fail("grid and block sizes must be positive");

    bool lockstep = fn_.usesBarrier();
    for (uint32_t b = 0; b < cfg_.grid && result_.ok; ++b) {
      ctaid_ = b;
      // Shared memory is NOT zeroed on the GPU: track which elements this block has written
      // and make reading an unwritten one an error (the data value itself is irrelevant).
      shared_.assign(fn_.shared.size(), {});
      sharedWritten_.assign(fn_.shared.size(), {});
      for (size_t s = 0; s < fn_.shared.size(); ++s) {
        shared_[s].assign(static_cast<size_t>(fn_.shared[s].count), 0u);
        sharedWritten_[s].assign(static_cast<size_t>(fn_.shared[s].count), false);
      }
      lockstep ? runBlockLockstep() : runBlockSequential();
    }
    return result_;
  }

private:
  Result fail(const std::string &msg) {
    result_.ok = false;
    result_.error = msg;
    result_.errorKind = "bad_args";
    result_.errorMsg = msg;
    return result_;
  }

  bool error(const Thread &t, const char *kind, const std::string &msg) {
    if (result_.ok) {
      result_.ok = false;
      result_.error = "runtime error in kernel '" + fn_.name + "' (block " +
                      std::to_string(ctaid_) + ", thread " + std::to_string(t.tid) + "): " + msg;
      result_.errorKind = kind;
      result_.errorBlock = ctaid_;
      result_.errorThread = t.tid;
      result_.errorMsg = msg;
    }
    return false;
  }

  Thread makeThread(uint32_t tid) {
    Thread t;
    t.tid = tid;
    t.bb = fn_.entry();
    t.regs.assign(fn_.vregs.size(), 0);
    return t;
  }

  void runBlockSequential() {
    for (uint32_t tid = 0; tid < cfg_.block && result_.ok; ++tid) {
      Thread t = makeThread(tid);
      execute(t);
    }
  }

  // With barriers: every thread runs until it reaches a barrier or exits; when all threads
  // have stopped, they must all be waiting at the same barrier (then continue) or all be done.
  void runBlockLockstep() {
    std::vector<Thread> threads;
    threads.reserve(cfg_.block);
    for (uint32_t tid = 0; tid < cfg_.block; ++tid)
      threads.push_back(makeThread(tid));
    for (;;) {
      for (Thread &t : threads)
        if (t.state == ThreadState::Running && !execute(t))
          return;
      const Thread *waiting = nullptr, *done = nullptr;
      const Inst *barrier = nullptr;
      for (const Thread &t : threads) {
        if (t.state == ThreadState::Done) {
          done = &t;
        } else {
          const Inst *at = &t.bb->insts[t.ip];
          if (barrier && at != barrier) {
            error(t, "barrier_divergence",
                  "barrier divergence: threads " + std::to_string(waiting->tid) + " and " +
                      std::to_string(t.tid) + " wait at different __syncthreads()");
            return;
          }
          barrier = at;
          waiting = &t;
        }
      }
      if (!waiting)
        return; // all threads finished
      if (done) {
        error(*waiting, "barrier_divergence",
              "barrier divergence: thread " + std::to_string(waiting->tid) +
                  " waits at __syncthreads() but thread " + std::to_string(done->tid) +
                  " already exited");
        return;
      }
      for (Thread &t : threads) { // release the barrier
        ++t.ip;
        t.state = ThreadState::Running;
      }
    }
  }

  uint64_t get(const Thread &t, const Value &v) const {
    switch (v.kind) {
    case Value::Kind::VReg:
      return t.regs[v.reg];
    case Value::Kind::ImmI32:
      return static_cast<uint32_t>(v.i);
    case Value::Kind::ImmF32:
      return fromF32(v.f);
    case Value::Kind::ImmBool:
      return v.b ? 1 : 0;
    case Value::Kind::Shared:
      return baseAddress(kSharedBufBase + v.reg);
    case Value::Kind::None:
      break;
    }
    return 0;
  }

  // Resolve a checked memory access. Returns nullptr (after reporting) when invalid.
  uint32_t *access(const Thread &t, uint64_t addr, bool isStore, MemSpace space) {
    uint32_t bufId = static_cast<uint32_t>(addr >> kBufShift);
    int64_t off =
        static_cast<int64_t>(addr & ((1ull << kBufShift) - 1)) - static_cast<int64_t>(kOffsetBias);
    std::vector<uint32_t> *buf = nullptr;
    std::vector<bool> *written = nullptr;
    std::string name;
    if (space == MemSpace::Shared && bufId >= kSharedBufBase &&
        bufId - kSharedBufBase < shared_.size()) {
      buf = &shared_[bufId - kSharedBufBase];
      written = &sharedWritten_[bufId - kSharedBufBase];
      name = fn_.shared[bufId - kSharedBufBase].name;
    } else if (space == MemSpace::Global && bufId >= 1 && bufId <= args_.size() &&
               args_[bufId - 1].type == IRType::PtrGlobal) {
      buf = &args_[bufId - 1].data;
      name = fn_.params[bufId - 1].name;
    } else {
      error(t, "oob", std::string("invalid ") + (isStore ? "store" : "load") + " address");
      return nullptr;
    }
    int64_t index = off >= 0 ? off / 4 : -((-off + 3) / 4);
    if (off % 4 != 0 || off < 0 || static_cast<uint64_t>(index) >= buf->size()) {
      error(t, "oob",
            std::string("out-of-bounds ") + (isStore ? "store to '" : "load from '") + name +
                "' at index " + std::to_string(index) + " (size " + std::to_string(buf->size()) +
                ")");
      return nullptr;
    }
    size_t i = static_cast<size_t>(index);
    if (written) {
      if (isStore) {
        (*written)[i] = true;
      } else if (!(*written)[i]) {
        error(t, "uninit_shared",
              "load from shared '" + name + "' at index " + std::to_string(index) +
                  " before any thread of this block wrote it (shared memory is not "
                  "zero-initialized on the GPU)");
        return nullptr;
      }
    }
    return &(*buf)[i];
  }

  // Runs `t` until it exits (Done), reaches a barrier (AtBarrier) or fails (returns false).
  bool execute(Thread &t) {
    for (;;) {
      if (++t.steps > cfg_.stepLimit)
        return error(t, "step_limit",
                     "step limit of " + std::to_string(cfg_.stepLimit) +
                         " instructions exceeded (possible infinite loop)");
      ++result_.steps;
      const Inst &in = t.bb->insts[t.ip];
      uint64_t a = in.a.isNone() ? 0 : get(t, in.a);
      uint64_t b = in.b.isNone() ? 0 : get(t, in.b);
      uint64_t r = 0;

      if (in.op == Op::Mov) { // may carry 64-bit pointers/addresses
        r = a;
      } else if (isPureOp(in.op)) {
        // Same function as the constant folder: execution and folding cannot disagree.
        uint32_t out = 0;
        switch (evalPure(in.op, in.type, static_cast<uint32_t>(a), static_cast<uint32_t>(b), out)) {
        case EvalStatus::Ok:
          break;
        case EvalStatus::DivZero:
          return error(t, "div0", "integer division by zero");
        case EvalStatus::DivOverflow:
          return error(t, "div_overflow", "integer overflow in division (INT_MIN / -1)");
        case EvalStatus::BadCvt:
          return error(t, "bad_cvt", "float-to-int conversion of an out-of-range value");
        }
        r = out;
      }

      switch (in.op) {
      default:
        break; // pure ops: handled above
      case Op::Addr:
        r = a + static_cast<uint64_t>(static_cast<int64_t>(asI32(b)) * in.aux);
        break;
      case Op::Load: {
        uint32_t *p = access(t, a, false, static_cast<MemSpace>(in.aux));
        if (!p)
          return false;
        r = *p;
        break;
      }
      case Op::Store: {
        uint32_t *p = access(t, a, true, static_cast<MemSpace>(in.aux));
        if (!p)
          return false;
        *p = static_cast<uint32_t>(b);
        break;
      }
      case Op::Sreg:
        switch (static_cast<SReg>(in.aux)) {
        case SReg::Tid:
          r = t.tid;
          break;
        case SReg::Ctaid:
          r = ctaid_;
          break;
        case SReg::Ntid:
          r = cfg_.block;
          break;
        case SReg::Nctaid:
          r = cfg_.grid;
          break;
        }
        break;
      case Op::Param: {
        const ArgValue &arg = args_[static_cast<size_t>(in.aux)];
        r = arg.type == IRType::PtrGlobal ? baseAddress(static_cast<uint32_t>(in.aux) + 1)
                                          : arg.bits;
        break;
      }
      case Op::Barrier:
        t.state = ThreadState::AtBarrier;
        return true; // ip stays on the barrier until the block releases it
      case Op::Br:
        t.bb = in.t;
        t.ip = 0;
        continue;
      case Op::CondBr:
        t.bb = (a & 1) ? in.t : in.f;
        t.ip = 0;
        continue;
      case Op::Ret:
        t.state = ThreadState::Done;
        return true;
      }
      if (in.dst.isVReg())
        t.regs[in.dst.reg] = r;
      ++t.ip;
    }
  }

  const Function &fn_;
  LaunchConfig cfg_;
  std::vector<ArgValue> &args_;
  std::vector<std::vector<uint32_t>> shared_;
  std::vector<std::vector<bool>> sharedWritten_;
  uint32_t ctaid_ = 0;
  Result result_;
};

} // namespace

Result run(const Function &fn, const LaunchConfig &cfg, std::vector<ArgValue> &args) {
  return Machine(fn, cfg, args).run();
}

} // namespace tinycu::interp
