"""Parallel differential fuzzing.

    python -m tcharness.fuzz --n 5000 --jobs 8 [--gpu] [--float-ratio 0.3] [--seed-start 0]
                             [--inject-bug ID] [--stop-on-first] [--keep-all] [--out fuzz_out]
                             [--summary results/fuzz_summary.json]

Each seed -> tcharness.gen program in <out>/<seed>/ -> tcharness.diff. CPU paths (A, B, ptxas)
run in a process pool; with --gpu the GPU paths (C, D, E) run afterwards in the main process
(one CUDA context). Passing programs are deleted unless --keep-all; failures keep their
directory with result.json for triage and `python -m tcharness.reduce <dir>`.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from tcharness import diff, gen
from tcharness.tinycu import REPO_ROOT


def _work(job: tuple[int, bool, str, list[str], bool]) -> tuple[int, dict, dict]:
    seed, float_mode, out, extra, keep = job
    d = Path(out) / str(seed)
    prog = gen.generate(seed, float_mode)
    gen.write(prog, seed, d)
    res, ref = diff.check_cpu(d, extra)
    refj = {k: v.tolist() for k, v in ref.items()} if res.status == "OK" else {}
    (d / "result.json").write_text(json.dumps({"seed": seed, "float": float_mode, "extra": extra,
                                               **res.to_json()}, indent=1))
    return seed, {"float": float_mode, **res.to_json()}, refj


def run(n: int, jobs: int, gpu: bool, float_ratio: float, seed_start: int, extra: list[str],
        out: Path, stop_on_first: bool = False, keep_all: bool = False, quiet: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rng_float = lambda s: (s * 2654435761 % 1000) / 1000.0 < float_ratio  # noqa: E731 deterministic
    work = [(s, rng_float(s), str(out), extra, keep_all) for s in range(seed_start, seed_start + n)]
    results: dict[int, dict] = {}
    t0 = time.time()
    first_failure = None
    with mp.get_context("fork").Pool(jobs) as pool:
        for k, (seed, res, ref) in enumerate(pool.imap_unordered(_work, work, chunksize=4), 1):
            if gpu and res["status"] == "OK":
                import numpy as np  # noqa: PLC0415
                refa = {name: np.array(v, dtype=np.float32 if name == "fout" else np.int32)
                        for name, v in ref.items()}
                g = diff.check_gpu(out / str(seed), refa, extra)
                res = {"float": res["float"], **g.to_json()}
                (out / str(seed) / "result.json").write_text(json.dumps({"seed": seed, **res}, indent=1))
            results[seed] = res
            if res["status"] == "OK" and not keep_all:
                shutil.rmtree(out / str(seed), ignore_errors=True)
            elif res["status"] != "OK" and first_failure is None:
                first_failure = (k, seed)
                if not quiet:
                    print(f"\n[{k}] seed {seed}: {res['label']}: {res['detail'][:300]}", flush=True)
                if stop_on_first:
                    pool.terminate()
                    break
            if not quiet and k % 100 == 0:
                rate = k / (time.time() - t0) * 60
                print(f"  {k}/{n} programs, {rate:.0f}/min, failures so far: "
                      f"{sum(r['status'] != 'OK' for r in results.values())}", flush=True)
    elapsed = time.time() - t0
    counts = Counter(r["label"] for r in results.values())
    failing = sorted(s for s, r in results.items() if r["status"] != "OK")
    return {"programs": len(results), "seconds": round(elapsed, 2),
            "programs_per_min": round(len(results) / elapsed * 60, 1) if elapsed else 0.0,
            "jobs": jobs, "gpu": gpu, "float_ratio": float_ratio, "seed_start": seed_start,
            "extra": extra, "counts": dict(sorted(counts.items())),
            "float_programs": sum(r["float"] for r in results.values()),
            "failing_seeds": failing[:200],
            "first_failure_index": first_failure[0] if first_failure else None,
            "first_failure_seed": first_failure[1] if first_failure else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=mp.cpu_count())
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--float-ratio", type=float, default=0.3)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--inject-bug", type=int, help="pass --inject-bug=ID to tinycu (validation)")
    ap.add_argument("--stop-on-first", action="store_true")
    ap.add_argument("--keep-all", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "fuzz_out")
    ap.add_argument("--summary", type=Path, help="write the JSON summary here")
    a = ap.parse_args(argv)
    extra = [f"--inject-bug={a.inject_bug}"] if a.inject_bug else []
    s = run(a.n, a.jobs, a.gpu, a.float_ratio, a.seed_start, extra, a.out, a.stop_on_first, a.keep_all)
    print(f"\n{s['programs']} programs in {s['seconds']} s ({s['programs_per_min']}/min, {a.jobs} jobs, "
          f"{'GPU paths A-E' if a.gpu else 'paths A-B + ptxas'})")
    width = max([len(k) for k in s["counts"]] + [6])
    for label, c in s["counts"].items():
        print(f"  {label:<{width}}  {c}")
    if s["failing_seeds"]:
        print(f"  failing seeds (first {len(s['failing_seeds'])}): {s['failing_seeds'][:20]}")
    if a.summary:
        a.summary.parent.mkdir(parents=True, exist_ok=True)
        a.summary.write_text(json.dumps(s, indent=1) + "\n")
    return 0 if all(k == "OK" for k in s["counts"]) else 1


if __name__ == "__main__":
    sys.exit(main())
