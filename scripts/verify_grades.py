"""Re-grade every released response with harness/grade.py and compare with the grade stored in the released matrices.
Needs the responses (python3 scripts/fetch_data.py --responses). Code problems are graded by executing the program
(slow; pass --with-code to include them); free-response GPQA is judge-graded and is not re-graded here.
Usage: python3 scripts/verify_grades.py [--with-code]
"""
import os, sys, json, gzip, collections
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "harness"))
import grade  # noqa: E402

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
