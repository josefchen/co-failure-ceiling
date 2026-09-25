"""Integrity guard for the code-domain co-failure result: a too-strict grader (blocklist refusal, tight
timeout, output-format mismatch, special-judge problems) would manufacture FAKE all-models-wrong events and
inflate beta. Before trusting any co-failure on codegen, we run each problem's KNOWN-CORRECT reference
solutions (code_contests' own accepted PYTHON3 submissions) through grade_codegen against our trimmed test
subset, and CATEGORIZE every failure. Only problems where the grader accepts a reference are retained for the
run (written back) -- so on every retained problem, an all-models-wrong event is genuine co-failure, not a
grader artifact.  Usage: python3 validate_codegen_grader.py [--diagnose]
"""
import os, sys, json, subprocess, tempfile, re
import grade

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")
RAW = os.path.join(RUNS, "codegen_problems_raw.json")   # build output, keeps ref_solutions (idempotent source)
PATH = os.path.join(RUNS, "codegen_problems.json")      # slim run file the loader reads (no refs)


def categorize(code, gold):
    """why does this reference fail/pass our grader? returns one of pass/blocked/timeout/runtime/mismatch."""
    g = json.loads(gold); tests = g["tests"]
    low = code.lower()
    if any(b in low for b in grade._CODE_BAD):
        return "blocked"
    tl = float(g.get("time_limit", 4)) or 4
    timeout = min(max(tl * 2.0, 4.0), 8.0)
    d = tempfile.mkdtemp(prefix="cgv_"); p = os.path.join(d, "s.py")
    try:
        open(p, "w").write(code)
        for t in tests:
            try:
                r = subprocess.run([sys.executable, p], input=t["input"], capture_output=True,
                                   text=True, timeout=timeout, cwd=d)
            except Exception:
                return "timeout"
            if r.returncode != 0:
                return "runtime"
            if grade._norm_out(r.stdout) != grade._norm_out(t["output"]):
                return "mismatch"
        return "pass"
    finally:
        try: os.unlink(p); os.rmdir(d)
        except Exception: pass


def main():
    diagnose = "--diagnose" in sys.argv
    probs = json.load(open(RAW))   # idempotent: always read the raw file (with refs), never the slim run file
    kept, dropped = [], []
    cats = {}
    for p in probs:
        gold = json.dumps({"tests": p["tests"], "time_limit": p.get("time_limit", 4)})
        refs = p.get("ref_solutions", [])
        if not refs:
            best = "no_ref"
        else:
            results = [categorize(s, gold) for s in refs]
            best = "pass" if "pass" in results else results[0]
        cats[best] = cats.get(best, 0) + 1
        (kept if best == "pass" else dropped).append(p)
    print(f"problems: {len(probs)} | retained (grader accepts a ref): {len(kept)} | dropped: {len(dropped)}")
    print("failure categories:", json.dumps(cats, indent=0))
    if diagnose:
        return  # do NOT slim/overwrite in diagnose mode
    slim = [{k: v for k, v in p.items() if k != "ref_solutions"} for p in kept]
    json.dump(slim, open(PATH, "w"))
    rts = [p["cf_rating"] for p in kept]
    print(f"[validate] retained {len(kept)} problems, cf_rating {min(rts) if rts else 0}-{max(rts) if rts else 0}; "
          f"wrote validated codegen_problems.json")


if __name__ == "__main__":
    main()
