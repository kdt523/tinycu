#!/usr/bin/env python3
"""Single-file test runner used by ctest.

  run_file_test.py golden <tinycu> <file.tc>
      For every snapshot <file>.{ast,ir,ptx} that exists, run `tinycu --emit=<ext>`
      and require byte-identical output. For every <file>.dump-<pass>, run
      `tinycu -O1 --dump-after=<pass> --emit=ir` and compare its stderr (the pass dumps).
  run_file_test.py error <tinycu> <file.tc>
      The first lines of the file contain `// EXPECT: L:C: message` (an error) and/or
      `// EXPECT-WARN: L:C: message` (a warning). Each must appear in stderr as
      `<file>:L:C: error|warning: ...message...`. EXPECT implies exit code 1.
"""
import re
import subprocess
import sys
from pathlib import Path

EMIT_KINDS = ("ast", "ir", "ptx")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60)


def golden(tinycu, tc):
    checked = 0
    for ext in EMIT_KINDS:
        expected_path = tc.with_suffix("." + ext)
        if not expected_path.exists():
            continue
        checked += 1
        r = run([tinycu, str(tc), f"--emit={ext}"])
        if r.returncode != 0:
            print(f"FAIL {tc} --emit={ext}: exit {r.returncode}\n{r.stderr}")
            return 1
        expected = expected_path.read_text()
        if r.stdout != expected:
            import difflib
            diff = difflib.unified_diff(expected.splitlines(True), r.stdout.splitlines(True),
                                        str(expected_path), "actual")
            print(f"FAIL {tc} --emit={ext}: output differs from snapshot\n" + "".join(diff))
            return 1
    for dump in sorted(tc.parent.glob(tc.stem + ".dump-*")):
        checked += 1
        pass_name = dump.suffix[len(".dump-"):]
        r = run([tinycu, str(tc), "-O1", f"--dump-after={pass_name}", "--emit=ir"])
        if r.returncode != 0:
            print(f"FAIL {tc} --dump-after={pass_name}: exit {r.returncode}\n{r.stderr}")
            return 1
        if r.stderr != dump.read_text():
            import difflib
            diff = difflib.unified_diff(dump.read_text().splitlines(True),
                                        r.stderr.splitlines(True), str(dump), "actual")
            print(f"FAIL {tc} --dump-after={pass_name}: differs from snapshot\n" + "".join(diff))
            return 1
    if checked == 0:
        print(f"FAIL {tc}: no snapshot found (run tools/update_golden.sh)")
        return 1
    print(f"ok {tc} ({checked} snapshot(s))")
    return 0


EXPECT_RE = re.compile(r"^//\s*EXPECT(-WARN)?:\s*(\d+):(\d+):\s*(.*)$")


def error(tinycu, tc):
    expects = []
    for line in tc.read_text().splitlines():
        m = EXPECT_RE.match(line.strip())
        if m:
            sev = "warning" if m.group(1) else "error"
            expects.append((sev, int(m.group(2)), int(m.group(3)), m.group(4).strip()))
    if not expects:
        print(f"FAIL {tc}: no // EXPECT: line")
        return 1
    r = run([tinycu, str(tc), "--emit=ast"])
    want_error = any(e[0] == "error" for e in expects)
    if want_error and r.returncode != 1:
        print(f"FAIL {tc}: expected exit code 1, got {r.returncode}\n{r.stderr}")
        return 1
    if not want_error and r.returncode != 0:
        print(f"FAIL {tc}: expected success with warnings, got {r.returncode}\n{r.stderr}")
        return 1
    for sev, line, col, msg in expects:
        prefix = f"{tc}:{line}:{col}: {sev}: "
        if not any(l.startswith(prefix) and msg in l[len(prefix):] for l in r.stderr.splitlines()):
            print(f"FAIL {tc}: missing diagnostic\n  want: {prefix}{msg}\n  got:\n{r.stderr}")
            return 1
    print(f"ok {tc}")
    return 0


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("golden", "error"):
        print(__doc__)
        return 2
    mode, tinycu, tc = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    return golden(tinycu, tc) if mode == "golden" else error(tinycu, tc)


if __name__ == "__main__":
    sys.exit(main())
