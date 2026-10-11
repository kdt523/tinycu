#include "passes.h"

#include <algorithm>
#include <ostream>

namespace tinycu::opt {

namespace {
int gInjectedBug = 0;
} // namespace

void setInjectedBug(int id) { gInjectedBug = id; }
int injectedBug() { return gInjectedBug; }

const std::vector<std::string> &passNames() {
  static const std::vector<std::string> names = {"constfold", "copyprop", "simplifycfg", "dce"};
  return names;
}

std::vector<std::string> o1Pipeline() { return passNames(); }

std::unique_ptr<Pass> createPass(const std::string &name) {
  if (name == "constfold")
    return std::make_unique<ConstFoldPass>();
  if (name == "copyprop")
    return std::make_unique<CopyPropPass>();
  if (name == "dce")
    return std::make_unique<DcePass>();
  if (name == "simplifycfg")
    return std::make_unique<SimplifyCfgPass>();
  return nullptr;
}

std::vector<std::string> splitList(const std::string &s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s + ",") {
    if (c == ',') {
      if (!cur.empty())
        out.push_back(cur);
      cur.clear();
    } else if (c != ' ') {
      cur += c;
    }
  }
  return out;
}

PipelineResult runPipeline(ir::Module &m, const PipelineConfig &cfg) {
  PipelineResult res;
  std::vector<std::unique_ptr<Pass>> passes;
  for (const std::string &n : cfg.passes) {
    if (std::find(cfg.disabled.begin(), cfg.disabled.end(), n) != cfg.disabled.end())
      continue;
    auto p = createPass(n);
    if (!p) {
      res.ok = false;
      res.error = "unknown pass '" + n + "'";
      return res;
    }
    passes.push_back(std::move(p));
  }
  for (const std::string &n : cfg.disabled)
    if (!createPass(n)) {
      res.ok = false;
      res.error = "unknown pass '" + n + "' in --disable-pass";
      return res;
    }

  for (auto &fn : m.functions) {
    unsigned iter = 0;
    bool changed = true;
    while (changed && iter < static_cast<unsigned>(cfg.maxIterations)) {
      changed = false;
      ++iter;
      for (auto &p : passes) {
        changed |= p->run(*fn);
        if (cfg.verifyEach) {
          std::vector<std::string> errs = ir::verify(*fn);
          if (!errs.empty()) {
            res.ok = false;
            res.error = "IR verification failed after pass '" + std::string(p->name()) + "':\n";
            for (const std::string &e : errs)
              res.error += "  " + e + "\n";
            res.error += ir::printFunction(*fn);
            return res;
          }
        }
        if (cfg.dump && cfg.dumpAfter == p->name())
          *cfg.dump << "; *** IR after " << p->name() << " (iteration " << iter << ") ***\n"
                    << ir::printFunction(*fn);
      }
    }
    res.maxIterationsUsed = std::max(res.maxIterationsUsed, iter);
  }
  for (auto &p : passes)
    res.stats.emplace_back(p->name(), p->changes);
  return res;
}

} // namespace tinycu::opt
