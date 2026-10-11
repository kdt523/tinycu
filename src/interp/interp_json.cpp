// JSON argument parsing and result printing for `tinycu --interp`.
#include "tinycu/interp.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstring>

namespace tinycu::interp {

using namespace ir;
using json = nlohmann::json;

namespace {

const char *specName(IRType type, IRType elem) {
  if (type == IRType::PtrGlobal)
    return elem == IRType::F32 ? "float*" : "int*";
  return type == IRType::F32 ? "float" : "int";
}

uint32_t f32Bits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, sizeof b);
  return b;
}

bool parseElem(const json &j, IRType elem, uint32_t &out, std::string &error) {
  if (elem == IRType::F32) {
    if (!j.is_number()) {
      error = "expected a number";
      return false;
    }
    out = f32Bits(static_cast<float>(j.get<double>()));
    return true;
  }
  if (!j.is_number_integer()) {
    error = "expected an integer";
    return false;
  }
  int64_t v = j.get<int64_t>();
  if (v < INT32_MIN || v > INT32_MAX) {
    error = "integer " + std::to_string(v) + " does not fit in int32";
    return false;
  }
  out = static_cast<uint32_t>(static_cast<int32_t>(v));
  return true;
}

bool parseBits(const json &j, uint32_t &out, std::string &error) {
  if (!j.is_number_unsigned() || j.get<uint64_t>() > 0xffffffffull) {
    error = "\"bits\" must be an unsigned 32-bit integer";
    return false;
  }
  out = static_cast<uint32_t>(j.get<uint64_t>());
  return true;
}

} // namespace

bool parseArgsJson(const std::string &text, const Function &fn, std::vector<ArgValue> &out,
                   std::string &error) {
  json doc = json::parse(text, nullptr, /*allow_exceptions=*/false);
  if (doc.is_discarded() || !doc.is_object() || !doc.contains("params") ||
      !doc["params"].is_array()) {
    error = "args: expected an object with a \"params\" array";
    return false;
  }
  const json &params = doc["params"];
  if (params.size() != fn.params.size()) {
    error = "args: kernel '" + fn.name + "' has " + std::to_string(fn.params.size()) +
            " parameters but " + std::to_string(params.size()) + " were given";
    return false;
  }
  out.clear();
  for (size_t i = 0; i < params.size(); ++i) {
    const json &p = params[i];
    const IRParam &decl = fn.params[i];
    std::string where = "args: param " + std::to_string(i) + " ('" + decl.name + "'): ";
    std::string want = specName(decl.type, decl.elem);
    if (!p.is_object() || !p.contains("type") || p["type"] != want) {
      error = where + "expected \"type\": \"" + want + "\"";
      return false;
    }
    ArgValue a;
    a.type = decl.type;
    a.elem = decl.elem;
    std::string e;
    if (decl.type == IRType::PtrGlobal) {
      const char *key = p.contains("bits") ? "bits" : "data";
      if (!p.contains(key) || !p[key].is_array()) {
        error = where + "expected a \"data\" (or \"bits\") array";
        return false;
      }
      for (const json &x : p[key]) {
        uint32_t v;
        bool ok = std::string(key) == "bits" ? parseBits(x, v, e) : parseElem(x, decl.elem, v, e);
        if (!ok) {
          error = where + e;
          return false;
        }
        a.data.push_back(v);
      }
    } else {
      bool ok = p.contains("bits")    ? parseBits(p["bits"], a.bits, e)
                : p.contains("value") ? parseElem(p["value"], decl.type, a.bits, e)
                                      : (e = "expected \"value\" (or \"bits\")", false);
      if (!ok) {
        error = where + e;
        return false;
      }
    }
    out.push_back(std::move(a));
  }
  return true;
}

std::string resultJson(const Function &fn, const LaunchConfig &cfg, const Result &r,
                       const std::vector<ArgValue> &args) {
  json doc;
  doc["kernel"] = fn.name;
  doc["grid"] = cfg.grid;
  doc["block"] = cfg.block;
  doc["status"] = r.ok ? "ok" : "error";
  if (!r.ok) {
    json e;
    e["kind"] = r.errorKind;
    e["block"] = r.errorBlock >= 0 ? json(r.errorBlock) : json(nullptr);
    e["thread"] = r.errorThread >= 0 ? json(r.errorThread) : json(nullptr);
    e["msg"] = r.errorMsg;
    e["text"] = r.error;
    doc["error"] = std::move(e);
  }
  doc["steps"] = r.steps;
  json params = json::array();
  for (size_t i = 0; i < args.size(); ++i) {
    const ArgValue &a = args[i];
    json p;
    p["name"] = fn.params[i].name;
    p["type"] = specName(a.type, a.elem);
    auto value = [&](uint32_t bits, IRType t) -> json {
      if (t == IRType::F32) {
        float f;
        std::memcpy(&f, &bits, sizeof f);
        return std::isfinite(f) ? json(static_cast<double>(f)) : json(nullptr);
      }
      return json(static_cast<int32_t>(bits));
    };
    if (a.type == IRType::PtrGlobal) {
      json data = json::array();
      for (uint32_t v : a.data)
        data.push_back(value(v, a.elem));
      p["data"] = std::move(data);
      if (a.elem == IRType::F32)
        p["bits"] = a.data; // exact; NaN/Inf appear as null in "data"
    } else {
      p["value"] = value(a.bits, a.type);
      if (a.type == IRType::F32)
        p["bits"] = a.bits;
    }
    params.push_back(std::move(p));
  }
  doc["params"] = std::move(params);
  return doc.dump() + "\n";
}

} // namespace tinycu::interp
