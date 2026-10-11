// tinycu — command-line driver.
//
//   tinycu <input.tc> [-o out.ptx] [-O0|-O1] [--emit=ast|ir|ptx] [--dump-after=<pass>]
//          [--interp --grid N --block M --args args.json] [--sm=sm_75] [--version] [--help]
//
// Exit codes: 0 ok, 1 compile error, 2 usage error, 3 internal error.

#include "tinycu/version.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

enum ExitCode { kOk = 0, kCompileError = 1, kUsageError = 2, kInternalError = 3 };

enum class Emit { Ast, Ir, Ptx };

struct Options {
  std::string input;
  std::string output;
  int optLevel = 0;
  Emit emit = Emit::Ptx;
  std::string dumpAfter;
  bool interp = false;
  long grid = 1;
  long block = 1;
  std::string argsJson;
  std::string sm = "sm_75";
};

void printUsage(std::FILE *out) {
  std::fprintf(out,
               "usage: tinycu <input.tc> [-o out.ptx] [-O0|-O1] [--emit=ast|ir|ptx]\n"
               "              [--dump-after=<pass>] [--interp --grid N --block M --args a.json]\n"
               "              [--sm=sm_75] [--version] [--help]\n");
}

int usageError(const std::string &msg) {
  std::fprintf(stderr, "tinycu: error: %s\n", msg.c_str());
  printUsage(stderr);
  return kUsageError;
}

bool startsWith(const std::string &s, const char *prefix) {
  return s.rfind(prefix, 0) == 0;
}

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
    } else if (a == "--interp") {
      opts.interp = true;
    } else if (a == "--grid" || a == "--block") {
      std::string v;
      if (!needValue(v) || !parseLong(v, a == "--grid" ? opts.grid : opts.block))
        return usageError("'" + a + "' requires a positive integer");
    } else if (a == "--args") {
      if (!needValue(opts.argsJson))
        return usageError("'--args' requires a file name");
    } else if (startsWith(a, "--sm=")) {
      opts.sm = a.substr(5);
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
  return -1;
}

} // namespace

int main(int argc, char **argv) {
  std::vector<std::string> args(argv + 1, argv + argc);
  Options opts;
  int rc = parseArgs(args, opts);
  if (rc >= 0)
    return rc;

  std::fprintf(stderr, "tinycu: %s: not implemented\n",
               opts.interp ? "interpreter" : "compilation");
  return kInternalError;
}
