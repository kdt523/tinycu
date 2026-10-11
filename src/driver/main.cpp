// tinycu — command-line driver.
//
//   tinycu <input.tc> [-o out] [-O0|-O1] [--emit=ast|ir|ptx] [--dump-after=<pass>]
//          [--interp --grid N --block M --args args.json [--kernel name]]
//          [--passes=a,b] [--disable-pass=a] [--stats]
//          [--verify-each|--no-verify-each] [--sm=sm_75] [--no-ptx-comments]
//          [--version] [--help]
//
// Exit codes: 0 ok, 1 compile error, 2 usage error, 3 internal error,
//             4 kernel runtime error reported by the interpreter.

#include "tinycu/interp.h"
#include "tinycu/ir.h"
#include "tinycu/opt.h"
#include "tinycu/parser.h"
#include "tinycu/ptx.h"
#include "tinycu/sema.h"
#include "tinycu/version.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

enum ExitCode {
  kOk = 0,
  kCompileError = 1,
  kUsageError = 2,
  kInternalError = 3,
  kRuntimeError = 4
};

enum class Emit { Ast, Ir, Ptx };

struct Options {
  std::string input;
  std::string output;
  int optLevel = 0;
  Emit emit = Emit::Ptx;
  std::string dumpAfter;
  std::vector<std::string> passes; ///< --passes= (overrides the -O level)
  bool customPasses = false;
  std::vector<std::string> disabled;
  bool stats = false;
  bool interp = false;
  long grid = 1;
  long block = 1;
  std::string argsJson;
  std::string kernel;
#ifdef NDEBUG
  bool verifyEach = false;
#else
  bool verifyEach = true; // default ON in Debug builds
#endif
  std::string sm = "sm_75";
  bool ptxComments = true;
};

void printUsage(std::FILE *out) {
  std::fprintf(out,
               "usage: tinycu <input.tc> [-o out] [-O0|-O1] [--emit=ast|ir|ptx]\n"
               "              [--passes=a,b] [--disable-pass=a] [--dump-after=<pass>] [--stats]\n"
               "              [--verify-each|--no-verify-each]\n"
               "              [--interp --grid N --block M --args a.json [--kernel name]]\n"
               "              [--sm=sm_75] [--no-ptx-comments] [--version] [--help]\n"
               "exit codes: 0 ok, 1 compile error, 2 usage error, 3 internal error,\n"
               "            4 kernel runtime error (interpreter)\n");
}

int usageError(const std::string &msg) {
  std::fprintf(stderr, "tinycu: error: %s\n", msg.c_str());
  printUsage(stderr);
  return kUsageError;
}

bool startsWith(const std::string &s, const char *prefix) { return s.rfind(prefix, 0) == 0; }

bool parseLong(const std::string &s, long &out) {
  if (s.empty())
    return false;
  char *end = nullptr;
  long v = std::strtol(s.c_str(), &end, 10);
  if (*end != '\0' || v <= 0)
    return false;
  out = v;
  return true;
}

// Returns -1 on success, otherwise an exit code to return immediately.
int parseArgs(const std::vector<std::string> &args, Options &opts) {
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string &a = args[i];
    auto needValue = [&](std::string &dst) -> bool {
      if (i + 1 >= args.size())
        return false;
      dst = args[++i];
      return true;
    };
    if (a == "--help" || a == "-h") {
      printUsage(stdout);
      return kOk;
    } else if (a == "--version") {
      std::printf("tinycu %s\n", tinycu::version());
      return kOk;
    } else if (a == "-o") {
      if (!needValue(opts.output))
        return usageError("'-o' requires a file name");
    } else if (a == "-O0" || a == "-O1") {
      opts.optLevel = a[2] - '0';
    } else if (startsWith(a, "--emit=")) {
      std::string v = a.substr(7);
      if (v == "ast")
        opts.emit = Emit::Ast;
      else if (v == "ir")
        opts.emit = Emit::Ir;
      else if (v == "ptx")
        opts.emit = Emit::Ptx;
      else
        return usageError("unknown --emit kind '" + v + "'");
    } else if (startsWith(a, "--dump-after=")) {
      opts.dumpAfter = a.substr(13);
    } else if (startsWith(a, "--passes=")) {
      opts.passes = tinycu::opt::splitList(a.substr(9));
      opts.customPasses = true;
    } else if (startsWith(a, "--disable-pass=")) {
      for (const std::string &p : tinycu::opt::splitList(a.substr(15)))
        opts.disabled.push_back(p);
    } else if (a == "--stats") {
      opts.stats = true;
    } else if (startsWith(a, "--inject-bug=")) { // hidden: fuzzer validation (phase 7)
      long id = 0;
      if (!parseLong(a.substr(13), id) || id > 4)
        return usageError("--inject-bug expects 1..4");
      tinycu::opt::setInjectedBug(static_cast<int>(id));
    } else if (a == "--verify-each") {
      opts.verifyEach = true;
    } else if (a == "--no-verify-each") {
      opts.verifyEach = false;
    } else if (a == "--interp") {
      opts.interp = true;
    } else if (a == "--grid" || a == "--block") {
      std::string v;
      if (!needValue(v) || !parseLong(v, a == "--grid" ? opts.grid : opts.block))
        return usageError("'" + a + "' requires a positive integer");
    } else if (a == "--args") {
      if (!needValue(opts.argsJson))
        return usageError("'--args' requires a file name");
    } else if (a == "--kernel") {
      if (!needValue(opts.kernel))
        return usageError("'--kernel' requires a name");
    } else if (startsWith(a, "--sm=")) {
      opts.sm = a.substr(5);
      if (opts.sm.rfind("sm_", 0) != 0 || opts.sm.size() < 5)
        return usageError("'--sm' expects a target like sm_75");
    } else if (a == "--ptx-comments" || a == "--no-ptx-comments") {
      opts.ptxComments = a == "--ptx-comments";
    } else if (!a.empty() && a[0] == '-') {
      return usageError("unknown option '" + a + "'");
    } else if (opts.input.empty()) {
      opts.input = a;
    } else {
      return usageError("multiple input files given");
    }
  }
  if (opts.input.empty())
    return usageError("no input file");
  for (const std::string &p : opts.passes)
    if (!tinycu::opt::createPass(p))
      return usageError("unknown pass '" + p + "' (known: constfold, copyprop, simplifycfg, dce)");
  for (const std::string &p : opts.disabled)
    if (!tinycu::opt::createPass(p))
      return usageError("unknown pass '" + p + "' in --disable-pass");
  if (!opts.dumpAfter.empty() && !tinycu::opt::createPass(opts.dumpAfter))
    return usageError("unknown pass '" + opts.dumpAfter + "' in --dump-after");
  if (opts.interp) {
    if (opts.argsJson.empty())
      return usageError("'--interp' requires '--args <file.json>'");
    if (opts.block > 1024)
      return usageError("'--block' must be at most 1024");
    if (opts.grid > 0x7fffffffL)
      return usageError("'--grid' is too large");
  }
  return -1;
}

bool readFile(const std::string &path, std::string &out) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return false;
  std::ostringstream ss;
  ss << in.rdbuf();
  out = ss.str();
  return true;
}

int writeOutput(const Options &opts, const std::string &text) {
  if (opts.output.empty() || opts.output == "-") {
    std::cout << text;
    return kOk;
  }
  std::ofstream out(opts.output, std::ios::binary);
  if (!out) {
    std::fprintf(stderr, "tinycu: error: cannot open '%s' for writing\n", opts.output.c_str());
    return kUsageError;
  }
  out << text;
  return kOk;
}

// A verifier failure is a compiler bug: report it with the offending IR.
bool verifyModule(const tinycu::ir::Module &m, const char *after) {
  bool ok = true;
  for (const auto &fn : m.functions) {
    std::vector<std::string> errs = tinycu::ir::verify(*fn);
    if (errs.empty())
      continue;
    ok = false;
    std::fprintf(stderr, "tinycu: internal error: IR verification failed after %s:\n", after);
    for (const std::string &e : errs)
      std::fprintf(stderr, "  %s\n", e.c_str());
    std::fprintf(stderr, "%s", tinycu::ir::printFunction(*fn).c_str());
  }
  return ok;
}

int interpret(const Options &opts, const tinycu::ir::Module &m) {
  const tinycu::ir::Function *fn = nullptr;
  if (opts.kernel.empty()) {
    if (m.functions.size() != 1) {
      std::string names;
      for (const auto &f : m.functions)
        names += (names.empty() ? "" : ", ") + f->name;
      std::fprintf(stderr,
                   "tinycu: error: '%s' contains %zu kernels (%s); choose one with --kernel "
                   "<name>\n",
                   opts.input.c_str(), m.functions.size(), names.c_str());
      return kUsageError;
    }
    fn = m.functions[0].get();
  } else if (!(fn = m.find(opts.kernel))) {
    std::fprintf(stderr, "tinycu: error: no kernel named '%s'\n", opts.kernel.c_str());
    return kUsageError;
  }
  std::string text, err;
  if (!readFile(opts.argsJson, text)) {
    std::fprintf(stderr, "tinycu: error: cannot read '%s'\n", opts.argsJson.c_str());
    return kUsageError;
  }
  std::vector<tinycu::interp::ArgValue> args;
  if (!tinycu::interp::parseArgsJson(text, *fn, args, err)) {
    std::fprintf(stderr, "tinycu: error: %s\n", err.c_str());
    return kUsageError;
  }
  tinycu::interp::LaunchConfig cfg;
  cfg.grid = static_cast<uint32_t>(opts.grid);
  cfg.block = static_cast<uint32_t>(opts.block);
  tinycu::interp::Result r = tinycu::interp::run(*fn, cfg, args);
  int rc = writeOutput(opts, tinycu::interp::resultJson(*fn, cfg, r, args));
  if (!r.ok) {
    std::fprintf(stderr, "tinycu: %s\n", r.error.c_str());
    return kRuntimeError;
  }
  return rc;
}

int compile(const Options &opts) {
  std::string source;
  if (!readFile(opts.input, source)) {
    std::fprintf(stderr, "tinycu: error: cannot read '%s'\n", opts.input.c_str());
    return kUsageError;
  }
  tinycu::DiagEngine diag(opts.input, source);
  std::unique_ptr<tinycu::Program> prog = tinycu::parseSource(source, diag);
  // Sema runs only on a syntactically valid program.
  if (!diag.hasErrors())
    tinycu::Sema(diag).run(*prog);
  diag.print(std::cerr);
  if (diag.hasErrors())
    return kCompileError;

  if (opts.emit == Emit::Ast && !opts.interp)
    return writeOutput(opts, tinycu::printAST(*prog));

  std::unique_ptr<tinycu::ir::Module> module = tinycu::lowerProgram(*prog);
  if (opts.verifyEach && !verifyModule(*module, "lowering"))
    return kInternalError;

  // Optimization: -O1 runs the standard pipeline; --passes= runs a custom list instead.
  std::vector<std::string> pipeline =
      opts.customPasses
          ? opts.passes
          : (opts.optLevel >= 1 ? tinycu::opt::o1Pipeline() : std::vector<std::string>{});
  if (!pipeline.empty()) {
    tinycu::opt::PipelineConfig pc;
    pc.passes = pipeline;
    pc.disabled = opts.disabled;
    pc.dumpAfter = opts.dumpAfter;
    pc.verifyEach = opts.verifyEach;
    pc.dump = &std::cerr;
    tinycu::opt::PipelineResult pr = tinycu::opt::runPipeline(*module, pc);
    if (!pr.ok) {
      std::fprintf(stderr, "tinycu: internal error: %s\n", pr.error.c_str());
      return kInternalError;
    }
    if (opts.stats) {
      std::fprintf(stderr, "pass statistics (changes):\n");
      for (const auto &s : pr.stats)
        std::fprintf(stderr, "  %-12s %u\n", s.first.c_str(), s.second);
      std::fprintf(stderr, "  iterations   %u\n", pr.maxIterationsUsed);
    }
  }

  if (opts.interp)
    return interpret(opts, *module);
  if (opts.emit == Emit::Ir)
    return writeOutput(opts, tinycu::ir::printModule(*module));

  tinycu::ptx::Options popts;
  popts.sm = opts.sm;
  popts.comments = opts.ptxComments;
  return writeOutput(opts, tinycu::ptx::emitModule(*module, popts));
}

} // namespace

int main(int argc, char **argv) {
  std::vector<std::string> args(argv + 1, argv + argc);
  Options opts;
  int rc = parseArgs(args, opts);
  if (rc >= 0)
    return rc;
  return compile(opts);
}
