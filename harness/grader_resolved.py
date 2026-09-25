"""For every all-wrong event of the first release that the corrected grader resolved, list the reference answer and the
distinct answers the grader now marks correct (with how many models gave each), so a reader can confirm each resolution is
a true match and not a grader false positive. Reads runs/canonical.json and the released responses; writes
audit/grader_resolved.json. Usage: python3 grader_resolved.py (after canonical.py and scripts/fetch_data.py --responses)
"""
import os, json, gzip, collections
import grade

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); RUNS = os.path.join(ROOT, "runs")
FINAL = {"math500": "matrix_marketE3_final.json", "mathhard": "matrix_marketMH_final.json", "code": "matrix_marketCG_final.json",
         "gpqamc": "matrix_marketE2_final.json", "mmlupro": "matrix_marketE2_final.json", "mix": "matrix_stageA2v3_final.json",
         "pro15": "matrix_hardAv3_final.json"}

if __name__ == "__main__":
    C = json.load(open(os.path.join(RUNS, "canonical.json"))); S = json.load(open(os.path.join(RUNS, "items_snapshot.json")))
    want = {(FINAL[b], q) for b, tr in C["trace"].items() for q in tr["causes"].get("grader defect (corrected grader)", [])}
    found = collections.defaultdict(collections.Counter)
    for line in gzip.open(os.path.join(ROOT, "responses", "responses.jsonl.gz"), "rt"):
        r = json.loads(line)
        if (r["matrix"], r["qid"]) in want and r.get("correct"):
            it = S[r["qid"]]
            a = grade.extract_math(r["content"] or "") if it["kind"] == "math" else grade.extract(it["kind"], r["content"] or "")
            found[(r["matrix"], r["qid"])][str(a)] += 1
    out = [{"matrix": m, "qid": q, "reference": S[q]["gold"], "answers_marked_correct": dict(found[(m, q)].most_common())}
           for m, q in sorted(want)]
    json.dump(out, open(os.path.join(ROOT, "audit", "grader_resolved.json"), "w"), indent=1, ensure_ascii=False)
    print(f"[grader_resolved] {len(out)} events; {sum(1 for o in out if not o['answers_marked_correct'])} without a correct answer")
