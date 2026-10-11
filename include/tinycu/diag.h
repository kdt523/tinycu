#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace tinycu {

/// 1-based line/column position in a source file.
struct SourceLoc {
  uint32_t line = 0;
  uint32_t col = 0;
};

enum class Severity { Note, Warning, Error };

struct Diagnostic {
  Severity severity;
  SourceLoc loc;
  std::string msg;
};

/// Collects diagnostics (never throws) and renders them as
/// `file.tc:LINE:COL: error: message` followed by the source line and a caret.
class DiagEngine {
public:
  DiagEngine(std::string fileName, std::string_view source)
      : fileName_(std::move(fileName)), source_(source) {}

  void error(SourceLoc loc, std::string msg) { report(Severity::Error, loc, std::move(msg)); }
  void warning(SourceLoc loc, std::string msg) { report(Severity::Warning, loc, std::move(msg)); }
  void note(SourceLoc loc, std::string msg) { report(Severity::Note, loc, std::move(msg)); }
  void report(Severity sev, SourceLoc loc, std::string msg);

  /// After `n` errors further errors are dropped and a single note is emitted (0 = no cap).
  void setErrorLimit(unsigned n) { errorLimit_ = n; }
  bool limitReached() const { return errorLimit_ != 0 && errors_ >= errorLimit_; }

  unsigned errorCount() const { return errors_; }
  bool hasErrors() const { return errors_ != 0; }
  const std::vector<Diagnostic> &diagnostics() const { return diags_; }

  /// Print all collected diagnostics in report order.
  void print(std::ostream &os) const;

private:
  std::string_view lineText(uint32_t line) const;

  std::string fileName_;
  std::string_view source_;
  std::vector<Diagnostic> diags_;
  unsigned errors_ = 0;
  unsigned errorLimit_ = 0;
  bool limitNoted_ = false;
};

} // namespace tinycu
