"""Record the token budget each cell was actually run with, and re-grade every cell from its raw response.

The loaders' budgets changed during the project (e.g. the 15-model mix ran MMLU/ARC at 512 tokens; today's loader says
1,024), so a budget cannot be inferred from the item alone. For each cell we find the cached response whose request
matches (same model, prompt, temperature 0, one of the candidate budgets), prefer the budget whose re-grade reproduces
the recorded correctness, write that budget into the cell as "max_tokens", and count any cell whose re-grade
disagrees with the matrix. Code cells are not re-executed here (their grader runs the programs); their budget is
still recorded. Usage (maintainers, needs the local response cache): python3 scripts/annotate_budgets.py
"""
import os, sys, json, collections
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "harness"))
import orclient, grade  # noqa: E402

CANDS = [256, 512, 1024, 1536, 2048, 4096, 8192, 24000, 32768]
FINAL = ["matrix_marketE3_dt.json", "matrix_marketMH_dt.json", "matrix_marketCG_dt.json", "matrix_marketE2_dt.json",
         "matrix_stageA2v3_dt.json", "matrix_hardAv3.json", "matrix_stageA2v2_rr.json"]


def main():
    S = json.load(open(os.path.join(ROOT, "runs", "items_snapshot.json")))
    report = {}
    for fname in FINAL:
        path = os.path.join(ROOT, "runs", fname); R = json.load(open(path))
        st = collections.Counter(); bad = []
        for q, v in R.items():
            it = S[q]; msgs = [{"role": "user", "content": it["prompt"]}]
            for m, c in v["models"].items():
                order = ([c["max_tokens"]] if c.get("max_tokens") else []) + [it["max_tokens"]] + CANDS
                found = []
                for mt in dict.fromkeys(order):
                    p = orclient._key(m, msgs, 0.0, mt, None)
                    if os.path.exists(p):
                        found.append((mt, json.load(open(p))))
                if not found:
                    st["no_response"] += 1; continue
                if it["kind"] == "codegen":
                    c["max_tokens"] = found[0][0]; st["budget_only"] += 1; continue
                graded = [(mt, grade.grade(it["kind"], r.get("content", ""), it["gold"])) for mt, r in found]
                match = [mt for mt, g in graded if g == c["correct"]]
                if match:
                    c["max_tokens"] = match[0]; st["regrade_matches"] += 1
                else:
                    c["max_tokens"] = found[0][0]; st["regrade_DIFFERS"] += 1; bad.append((q, m, c["correct"], graded[0][1]))
        json.dump(R, open(path, "w"))
        report[fname] = {"counts": dict(st), "differs_examples": bad[:10]}
        print(fname, dict(st), bad[:3])
    json.dump(report, open(os.path.join(ROOT, "runs", "regrade_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
