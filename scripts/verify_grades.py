"""Re-grade every released response with harness/grade.py and compare with the grade stored in the released matrices.
Needs the responses (python3 scripts/fetch_data.py --responses). Code problems are graded by executing the program
(slow; pass --with-code to include them; the five problems that accept several correct outputs use the special
checkers of harness/code_checkers.py); code verdicts depend on each test's time limit (3x the official limit, at most 12 s), so
run --with-code on an otherwise idle machine. Free-response GPQA is judge-graded and is not re-graded here.
Usage: python3 scripts/verify_grades.py [--with-code]
"""
import os, sys, json, gzip, collections
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "harness"))
import grade, code_checkers  # noqa: E402
import re  # noqa: E402
CHECK = code_checkers.CHECKERS

if __name__ == "__main__":
    S = json.load(open(os.path.join(ROOT, "runs", "items_snapshot.json")))
    code = "--with-code" in sys.argv
    n = collections.Counter(); bad = []
    for line in gzip.open(os.path.join(ROOT, "responses", "responses.jsonl.gz"), "rt"):
        r = json.loads(line)
        if "correct" not in r or r["matrix"] == "matrix_marketGPQAOPENv2.json":
            continue
        it = S[r["qid"]]
        if it["kind"] == "codegen" and not code:
            n["skipped_code"] += 1; continue
        if r["qid"] in CHECK and r["matrix"] == "matrix_marketCG_final.json":   # several correct outputs: special checker
            gd = json.loads(it["gold"]); blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", r["content"] or "", re.DOTALL)
            outs = code_checkers.run_program(blocks[-1] if blocks else (r["content"] or ""), gd["tests"], gd.get("time_limit", 4))
            g = int(outs is not None and code_checkers.check(r["qid"], gd["tests"], outs))
        else:
            g = grade.grade(it["kind"], r["content"] or "", it["gold"])
        n[r["matrix"]] += 1
        if int(g) != int(r["correct"]):
            bad.append((r["matrix"], r["qid"], r["model"], r["correct"], g))
    for k, v in sorted(n.items()):
        print(f"[verify] {k}: {v}")
    for b in bad[:20]:
        print("  MISMATCH", b)
    print(f"[verify] {sum(v for k, v in n.items() if k != 'skipped_code')} grades re-computed; {len(bad)} differ")
    sys.exit(1 if bad else 0)
