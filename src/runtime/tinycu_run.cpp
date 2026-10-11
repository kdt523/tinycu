// tinycu-run: load a PTX module with the CUDA Driver API and launch one kernel.
//
//   tinycu-run k.ptx --kernel saxpy --grid G --block B --args args.json [--out out.json]
//              [--time] [--repeat N]
//
// args.json has the same format as `tinycu --interp` ({"params":[{"type":"float*","data":[...]},
// ...]}, float values may be given as "bits"), and the output JSON has the same shape as the
// interpreter's, so the two can be compared directly. --time reports the median kernel time
// (cuEvent) over --repeat launches (default 20); timing runs re-upload the inputs each time.
//
// Driver API flow: cuInit -> cuDeviceGet -> cuDevicePrimaryCtxRetain -> cuModuleLoadDataEx
// (with the JIT error log) -> cuModuleGetFunction -> cuMemAlloc/cuMemcpyHtoD -> cuLaunchKernel
// -> cuCtxSynchronize -> cuMemcpyDtoH.
#include <cuda.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

[[noreturn]] void die(const std::string &msg, int code = 1) {
  std::fprintf(stderr, "tinycu-run: error: %s\n", msg.c_str());
  std::exit(code);
}

void checkCu(CUresult r, const char *what, const char *file, int line) {
  if (r == CUDA_SUCCESS)
    return;
  const char *name = nullptr, *desc = nullptr;
  cuGetErrorName(r, &name);
  cuGetErrorString(r, &desc);
  std::fprintf(stderr, "tinycu-run: %s:%d: %s failed: %s (%s)\n", file, line, what,
               name ? name : "?", desc ? desc : "?");
  std::exit(3);
}
#define CHECK_CU(call) checkCu((call), #call, __FILE__, __LINE__)

std::string readFile(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    die("cannot read '" + path + "'", 2);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

uint32_t f32Bits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, sizeof b);
  return b;
}

struct Arg {
  std::string type; // int, float, int*, float*
  uint32_t scalar = 0;
  std::vector<uint32_t> host;
  CUdeviceptr dev = 0;
  bool isPtr() const { return type.back() == '*'; }
  bool isFloat() const { return type.rfind("float", 0) == 0; }
};

uint32_t parseElem(const json &j, bool isFloat, bool bits) {
  if (bits)
    return j.get<uint32_t>();
  if (isFloat)
    return f32Bits(static_cast<float>(j.get<double>()));
  return static_cast<uint32_t>(j.get<int32_t>());
}

std::vector<Arg> parseArgs(const std::string &text) {
  json doc = json::parse(text, nullptr, false);
  if (doc.is_discarded() || !doc.contains("params") || !doc["params"].is_array())
    die("args: expected an object with a \"params\" array", 2);
  std::vector<Arg> args;
  for (const json &p : doc["params"]) {
    Arg a;
    a.type = p.value("type", "");
    if (a.type != "int" && a.type != "float" && a.type != "int*" && a.type != "float*")
      die("args: unknown type '" + a.type + "'", 2);
    bool bits = p.contains("bits");
    if (a.isPtr()) {
      for (const json &x : p[bits ? "bits" : "data"])
        a.host.push_back(parseElem(x, a.isFloat(), bits));
    } else {
      a.scalar = parseElem(p[bits ? "bits" : "value"], a.isFloat(), bits);
    }
    args.push_back(std::move(a));
  }
  return args;
}

json resultJson(const std::string &kernel, unsigned grid, unsigned block,
                const std::vector<Arg> &args) {
  json doc;
  doc["kernel"] = kernel;
  doc["grid"] = grid;
  doc["block"] = block;
  doc["status"] = "ok";
  json params = json::array();
  for (const Arg &a : args) {
    json p;
    p["type"] = a.type;
    auto value = [&](uint32_t b) -> json {
      if (!a.isFloat())
        return json(static_cast<int32_t>(b));
      float f;
      std::memcpy(&f, &b, sizeof f);
      return std::isfinite(f) ? json(static_cast<double>(f)) : json(nullptr);
    };
    if (a.isPtr()) {
      json data = json::array();
      for (uint32_t v : a.host)
        data.push_back(value(v));
      p["data"] = std::move(data);
      if (a.isFloat())
        p["bits"] = a.host;
    } else {
      p["value"] = value(a.scalar);
      if (a.isFloat())
        p["bits"] = a.scalar;
    }
    params.push_back(std::move(p));
  }
  doc["params"] = std::move(params);
  return doc;
}

struct Options {
  std::string ptx, kernel, args, out;
  unsigned grid = 1, block = 1;
  bool time = false;
  int repeat = 20;
};

Options parseCli(int argc, char **argv) {
  Options o;
  auto usage = [] {
    std::fprintf(stderr, "usage: tinycu-run k.ptx --kernel NAME --grid G --block B --args a.json "
                         "[--out out.json] [--time] [--repeat N]\n");
    std::exit(2);
  };
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc)
        usage();
      return argv[++i];
    };
    if (a == "--kernel")
      o.kernel = next();
    else if (a == "--grid")
      o.grid = static_cast<unsigned>(std::stoul(next()));
    else if (a == "--block")
      o.block = static_cast<unsigned>(std::stoul(next()));
    else if (a == "--args")
      o.args = next();
    else if (a == "--out")
      o.out = next();
    else if (a == "--time")
      o.time = true;
    else if (a == "--repeat")
      o.repeat = std::max(1, std::stoi(next()));
    else if (a == "--help" || a == "-h")
      usage();
    else if (!a.empty() && a[0] != '-' && o.ptx.empty())
      o.ptx = a;
    else
      usage();
  }
  if (o.ptx.empty() || o.kernel.empty() || o.args.empty() || o.grid == 0 || o.block == 0)
    usage();
  return o;
}

} // namespace

int main(int argc, char **argv) {
  Options opt = parseCli(argc, argv);
  std::string ptx = readFile(opt.ptx);
  std::vector<Arg> args = parseArgs(readFile(opt.args));

  CHECK_CU(cuInit(0));
  CUdevice dev;
  CHECK_CU(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
  CHECK_CU(cuCtxSetCurrent(ctx));

  // Load the module and capture the JIT log, so PTX errors are shown verbatim.
  char errLog[16384] = {0}, infoLog[16384] = {0};
  CUjit_option jitOpts[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                            CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES};
  void *jitVals[] = {errLog, reinterpret_cast<void *>(sizeof errLog), infoLog,
                     reinterpret_cast<void *>(sizeof infoLog)};
  CUmodule mod;
  CUresult lr = cuModuleLoadDataEx(&mod, ptx.c_str(), 4, jitOpts, jitVals);
  if (lr != CUDA_SUCCESS) {
    std::fprintf(stderr, "tinycu-run: PTX JIT failed. JIT error log:\n%s\n", errLog);
    checkCu(lr, "cuModuleLoadDataEx", __FILE__, __LINE__);
  }
  CUfunction fn;
  CHECK_CU(cuModuleGetFunction(&fn, mod, opt.kernel.c_str()));

  for (Arg &a : args)
    if (a.isPtr())
      CHECK_CU(cuMemAlloc(&a.dev, std::max<size_t>(4, a.host.size() * 4)));
  auto upload = [&] {
    for (Arg &a : args)
      if (a.isPtr() && !a.host.empty())
        CHECK_CU(cuMemcpyHtoD(a.dev, a.host.data(), a.host.size() * 4));
  };
  std::vector<void *> params;
  for (Arg &a : args)
    params.push_back(a.isPtr() ? static_cast<void *>(&a.dev) : static_cast<void *>(&a.scalar));
  auto launch = [&] {
    CHECK_CU(
        cuLaunchKernel(fn, opt.grid, 1, 1, opt.block, 1, 1, 0, nullptr, params.data(), nullptr));
  };

  json timing;
  if (opt.time) {
    CUevent start, stop;
    CHECK_CU(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CHECK_CU(cuEventCreate(&stop, CU_EVENT_DEFAULT));
    std::vector<float> ms;
    for (int r = 0; r < opt.repeat; ++r) {
      upload();
      CHECK_CU(cuEventRecord(start, nullptr));
      launch();
      CHECK_CU(cuEventRecord(stop, nullptr));
      CHECK_CU(cuEventSynchronize(stop));
      float t = 0;
      CHECK_CU(cuEventElapsedTime(&t, start, stop));
      ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    timing["median_ms"] = ms[ms.size() / 2];
    timing["min_ms"] = ms.front();
    timing["repeat"] = opt.repeat;
    CHECK_CU(cuEventDestroy(start));
    CHECK_CU(cuEventDestroy(stop));
  }

  upload(); // the reported outputs come from one clean launch
  launch();
  CHECK_CU(cuCtxSynchronize());
  for (Arg &a : args)
    if (a.isPtr() && !a.host.empty())
      CHECK_CU(cuMemcpyDtoH(a.host.data(), a.dev, a.host.size() * 4));

  json doc = resultJson(opt.kernel, opt.grid, opt.block, args);
  if (opt.time)
    doc["time"] = timing;
  std::string text = doc.dump() + "\n";
  if (opt.out.empty()) {
    std::cout << text;
  } else {
    std::ofstream(opt.out, std::ios::binary) << text;
  }

  for (Arg &a : args)
    if (a.dev)
      cuMemFree(a.dev);
  cuModuleUnload(mod);
  cuDevicePrimaryCtxRelease(dev);
  return 0;
}
