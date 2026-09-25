"""Third domain, structurally independent of mathematics (math at two difficulties is one task family).
Code generation is the canonical choice: open-ended generation, but
PROGRAMMATIC (execution) grading, and a task family disjoint from competition math.

We use deepmind/code_contests (competitive programming, pure stdin->stdout, no filesystem/network), filtered
to a hard CF-rating band so frontier models genuinely co-fail (unlike easy code, where the tail is empty like
GPQA). Each problem is trimmed to public + a few small generated tests and saved locally so the run loader
needs no re-fetch.  Usage: python3 codegen_build.py [lo_rating] [hi_rating] [max_problems]
"""
import os, sys, json, urllib.request, urllib.parse

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")
DS = "deepmind/code_contests"
MAX_TESTS = 20          # STRICT: public + private + generated stress tests (was 6 toy tests)
MAX_INPUT_CHARS = 200000  # allow real stress-test inputs (the API caps generated_tests anyway)


def fetch(split, off, length=10):
    u = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
        {"dataset": DS, "config": "default", "split": split, "offset": off, "length": length})
    return [r["row"] for r in json.load(urllib.request.urlopen(u, timeout=120))["rows"]]


def trim_tests(row):
    """STRICT grading set: public (format) + PRIVATE + GENERATED stress tests, the hidden-equivalent
    suite. Prioritize private/generated (the strong ones) over the tiny public samples."""
    tests, seen = [], set()
    def add(block):
        b = row.get(block) or {}
        for i, o in zip(b.get("input", []), b.get("output", [])):
            if len(tests) >= MAX_TESTS or len(i) > MAX_INPUT_CHARS or i in seen:
                continue
            seen.add(i); tests.append({"input": i, "output": o})
    add("public_tests"); add("private_tests"); add("generated_tests")
    return tests[:MAX_TESTS]


def main():
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 3500
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 70
    out, scanned, seen = [], 0, set()
    for split in ("test", "valid"):
      off = 0
      while len(out) < cap and scanned < 4000:
        try:
            rows = fetch(split, off, 10)
        except Exception as e:
            print("fetch err", str(e)[:80]); break
        if not rows:
            break
        off += len(rows)
        for r in rows:
            scanned += 1
            if r["name"] in seen:
                continue
            cr = r.get("cf_rating") or 0
            if not (lo <= cr <= hi):
                continue
            tests = trim_tests(r)
            if len(tests) < 2:          # need at least a couple of checkable cases
                continue
            desc = r["description"]
            if len(desc) > 6000:        # keep prompt/token cost bounded
                continue
            seen.add(r["name"])
            # keep up to 3 shortest PYTHON3 reference solutions (language code 3) to VALIDATE the grader
            sol = r.get("solutions") or {}
            py3 = [s for c, s in zip(sol.get("language", []), sol.get("solution", []))
                   if c == 3 and len(s) <= 8000]
            py3 = sorted(py3, key=len)[:3]
            out.append({"name": r["name"], "cf_rating": cr, "difficulty": r.get("difficulty"),
                        "description": desc, "tests": tests, "ref_solutions": py3,
                        "time_limit": (r.get("time_limit") or {}).get("seconds", 4) if isinstance(r.get("time_limit"), dict) else 4})
            if len(out) >= cap:
                break
        print(f"scanned {scanned}, selected {len(out)}")
    path = os.path.join(RUNS, "codegen_problems_raw.json")  # raw: keeps ref_solutions; validate reads this
    json.dump(out, open(path, "w"))
    rts = [p["cf_rating"] for p in out]
    print(f"\nselected {len(out)} problems, cf_rating {min(rts) if rts else 0}-{max(rts) if rts else 0}, "
          f"median tests/problem {sorted(len(p['tests']) for p in out)[len(out)//2] if out else 0}")
    print(f"[codegen_build] wrote {path}")


if __name__ == "__main__":
    main()
