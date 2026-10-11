#include "tinycu/diag.h"

#include <algorithm>
#include <ostream>

namespace tinycu {

void DiagEngine::report(Severity sev, SourceLoc loc, std::string msg) {
  if (sev == Severity::Error) {
    if (limitReached()) {
      if (!limitNoted_) {
        limitNoted_ = true;
        diags_.push_back({Severity::Note, {}, "too many errors emitted, stopping now"});
      }
      return;
    }
    ++errors_;
  }
  diags_.push_back({sev, loc, std::move(msg)});
}

std::string_view DiagEngine::lineText(uint32_t line) const {
  size_t pos = 0;
  for (uint32_t l = 1; l < line; ++l) {
    pos = source_.find('\n', pos);
    if (pos == std::string_view::npos)
      return {};
    ++pos;
  }
  size_t end = source_.find('\n', pos);
  std::string_view text = source_.substr(pos, end == std::string_view::npos ? end : end - pos);
  if (!text.empty() && text.back() == '\r')
    text.remove_suffix(1);
  return text;
}

static const char *severityName(Severity s) {
  switch (s) {
  case Severity::Note:
    return "note";
  case Severity::Warning:
    return "warning";
  case Severity::Error:
    return "error";
  }
  return "error";
}

void DiagEngine::print(std::ostream &os) const {
  // Lexer and parser report in different passes: print in source order (stable, so
  // same-location diagnostics keep report order); location-less notes go last.
  std::vector<Diagnostic> sorted = diags_;
  std::stable_sort(sorted.begin(), sorted.end(), [](const Diagnostic &a, const Diagnostic &b) {
    auto key = [](const Diagnostic &d) {
      return d.loc.line == 0 ? UINT64_MAX : (uint64_t(d.loc.line) << 32) | d.loc.col;
    };
    return key(a) < key(b);
  });
  for (const Diagnostic &d : sorted) {
    if (d.loc.line == 0 || d.loc.col == 0) {
      os << fileName_ << ": " << severityName(d.severity) << ": " << d.msg << '\n';
      continue;
    }
    os << fileName_ << ':' << d.loc.line << ':' << d.loc.col << ": " << severityName(d.severity)
       << ": " << d.msg << '\n';
    std::string_view text = lineText(d.loc.line);
    os << text << '\n';
    // Keep tabs so the caret lines up with the source line in a terminal.
    std::string caret;
    for (uint32_t i = 1; i < d.loc.col; ++i)
      caret += (i - 1 < text.size() && text[i - 1] == '\t') ? '\t' : ' ';
    os << caret << "^\n";
  }
}

} // namespace tinycu
