"""Re-grade EVERY released artifact from its raw responses with the current grader (grade.py; 2026-09-25; no API calls).

The market-scale matrices re-grade identically from their raw responses. The earlier artifacts do not: they were
scored by older grader/extractor versions (a first-letter multiple-choice extractor that could not read MMLU-Pro's
letters E-J and picked up the word "A" from running text), although the first version of the paper said every output
had been re-graded. This script rebuilds each one from the cached responses:

  market matrices (T=0) marketE3_dt, marketMH_dt, marketE2_dt                 -> matrix_<base>_final.json
  code matrix           marketCG_dt (execution-graded; copied, multi-answer flag added) -> matrix_marketCG_final.json
  15-model/churn (T=0)  stageA2v3_dt, hardAv3, churnD                           -> matrix_<base>_final.json
  fusion samples (T=.7) fusion_eqq2, fusion_eqq2_math, fusion_eqqA            -> fusion_<tag>_final.json
  cascade               cascade_stageC2v3                                     -> cascade_stageC2v3_final.json

For every artifact it writes a change report to runs/regrade_v2_report.json. Budgets are recovered per call: a
response is looked up at the recorded budget first, then at each candidate budget (the loaders' budgets changed
during the project; e.g. the fusion runs used 768 tokens for MMLU-Pro). A call with no cached response keeps its
old value and is counted.
"""
import os, sys, json, collections
import orclient, grade
from cascade import majority as cascade_majority

HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "..", "runs")
CANDS = [256, 512, 768, 1024, 1536, 2048, 4096, 8192, 24000, 32768]
# Token budgets each run actually used. The loaders' budgets changed between the pillar experiments (June 14-17) and
# the market-scale runs, and the same prompt is often cached at both. The pillar-era budgets below were identified by
# grade agreement (stored grade vs. the response at each candidate budget: 94.4-100% at these budgets vs. 82-96% at the
# later ones; see DATA_AUDIT.md) and are confirmed by the cascade's cheap-model samples, which exist only at these.
# Market matrices carry a per-cell "max_tokens" recorded by scripts/annotate_budgets.py (every cell re-grades to its
# stored v1 grade at that budget).
PILLAR_ERA = {"gsm8k": 1024, "mmlu": 512, "arc": 512, "math500": 2048, "mmlu_pro": 768}
BUDGETS = {"stageA2v3_dt": PILLAR_ERA, "hardAv3": PILLAR_ERA, "churnD": PILLAR_ERA, "eqq2": PILLAR_ERA, "eqqA": PILLAR_ERA,
           "eqq2_math": PILLAR_ERA, "stageC2v3": PILLAR_ERA, "advC": PILLAR_ERA}
S = json.load(open(os.path.join(RUNS, "items_snapshot.json")))


def fetch(model, prompt, temp, seed, first=None, exact=False):
    """look the response up at the run's budget; with exact=True never fall back to another budget."""
    for mt in dict.fromkeys(([first] if first else []) + ([] if exact else CANDS)):
        p = orclient._key(model, [{"role": "user", "content": prompt}], temp, mt, seed)
        if os.path.exists(p):
            return mt, json.load(open(p))
    return None, None


def regrade_matrix(tag, kinds=None):
    """kinds: only re-grade cells of these item kinds (others keep the verdict already in matrix_<base>_final.json)."""
    base = tag.replace("_dt", "")
    src = f"matrix_{base}_final.json" if kinds else f"matrix_{tag}.json"
    R = json.load(open(os.path.join(RUNS, src))); st = collections.Counter(); changes = []
    orig = json.load(open(os.path.join(RUNS, f"matrix_{tag}.json")))
    era = BUDGETS.get(tag)
    for q, v in R.items():
        it = S[q]
        for m, c in v["models"].items():
            if it["kind"] == "codegen":
                st["codegen_kept"] += 1; continue          # execution-graded; see code_problem_flags()
            if kinds and it["kind"] not in kinds:
                st["kept_" + it["kind"]] += 1; continue
            budget = c.get("max_tokens") if (c.get("detruncated") or not era) else era[v["dataset"]]
            if c.get("detruncated"):
                budget = 32768
            mt, r = fetch(m, it["prompt"], 0.0, None, budget, exact=True)
            if r is None:
                st["no_response_at_budget"] += 1; continue
            if not (r.get("tok_in", 0) > 0 and r.get("tok_out", 0) > 0):
                st["corrupt_response"] += 1; c["corrupt_response"] = True
            new = grade.grade(it["kind"], r.get("content") or "", it["gold"])
            c["max_tokens"] = mt
            old = orig[q]["models"][m]["correct"]
            if new != old:
                changes.append([q, m, old, new])
            c["correct"] = new
    st["changed_vs_input"] = len(changes)
    json.dump(R, open(os.path.join(RUNS, f"matrix_{base}_final.json"), "w"))
    return {"counts": dict(st), "wrong_to_right": sum(1 for x in changes if x[3] == 1),
            "right_to_wrong": sum(1 for x in changes if x[3] == 0), "changes": changes}


def regrade_fusion(tag):
    d = json.load(open(os.path.join(RUNS, f"fusion_{tag}.json"))); st = collections.Counter(); changes = []
    for q, mm in d["samples"].items():
        it = S[q]; kind = d["meta"][q]["kind"]
        for m, arr in mm.items():
            for s in range(1, len(arr) + 1):
                mt, r = fetch(m, it["prompt"], 0.7, s, BUDGETS[tag][d["meta"][q]["dataset"]], exact=True)
                if r is None:
                    st["no_response"] += 1; continue
                new = grade.extract(kind, r.get("content") or "")
                if new != arr[s - 1]:
                    changes.append([q, m, s, arr[s - 1], new]); arr[s - 1] = new
                st["changed" if changes and changes[-1][:3] == [q, m, s] else "unchanged"] += 1
    json.dump(d, open(os.path.join(RUNS, f"fusion_{tag}_final.json"), "w"))
    return {"counts": dict(st), "changes_sample": changes[:50], "n_changes": len(changes)}


def regrade_cascade(tag):
    d = json.load(open(os.path.join(RUNS, f"cascade_{tag}.json"))); st = collections.Counter(); changes = []
    for rec in d["rec"]:
        q = rec["qid"]; it = S[q]; kind = it["kind"]
        Ls = []
        for s in range(1, d["kc"] + 1):
            mt, r = fetch(d["L"], it["prompt"], 0.7, s, BUDGETS[tag][it["dataset"]], exact=True)
            Ls.append(grade.extract(kind, r.get("content") or "") if r else None); st["L_found" if r else "L_missing"] += 1
        mtH, rH = fetch(d["H"], it["prompt"], 0.0, None, BUDGETS[tag][it["dataset"]], exact=True)
        st["H_found" if rH else "H_missing"] += 1
        modal, conf = cascade_majority(Ls)
        new = {"conf": conf, "Lcorr": grade.check(kind, modal, it["gold"]),
               "Hcorr": grade.check(kind, grade.extract(kind, rH.get("content") or ""), it["gold"]) if rH else rec["Hcorr"]}
        diff = {k: (rec[k], v) for k, v in new.items() if rec[k] != v}
        if diff:
            changes.append([q, diff]); rec.update(new)
    json.dump(d, open(os.path.join(RUNS, f"cascade_{tag}_final.json"), "w"))
    return {"counts": dict(st), "n_records_changed": len(changes), "changes_sample": changes[:30]}


MULTI = __import__("re").compile(r"(any of them|print any|any valid|any such|any possible|multiple (possible )?answers|"
                                   r"multiple solutions|any one of|several (possible )?answers|choice of paths|order of putting)", 2)


def code_problem_flags():
    """flag code problems whose statement accepts more than one correct output: exact-match grading cannot verify a
    wrong verdict on them (a different valid answer is scored wrong). Two statements (1586E, 1599A) accept any valid
    set of paths / any valid order without using a stock phrase; their patterns are listed explicitly above."""
    P = json.load(open(os.path.join(RUNS, "codegen_problems.json")))
    R = json.load(open(os.path.join(RUNS, "matrix_marketCG_dt.json")))
    for q, v in R.items():
        v["multi_answer"] = bool(MULTI.search(P[int(q.split(":")[1])]["description"]))
    json.dump(R, open(os.path.join(RUNS, "matrix_marketCG_final.json"), "w"))
    return {"problems": len(R), "multi_answer": sum(v["multi_answer"] for v in R.values())}


if __name__ == "__main__":
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
    if only:                                            # re-run selected artifacts; keep the rest of the report
        rep = json.load(open(os.path.join(RUNS, "regrade_final_report.json")))
        for name in only:
            kind, tag = name.split("_", 1)
            rep[name] = {"fusion": regrade_fusion, "cascade": regrade_cascade}[kind](tag); print(name, rep[name].get("counts"), flush=True)
        json.dump(rep, open(os.path.join(RUNS, "regrade_final_report.json"), "w"), indent=1); sys.exit(0)
    rep = {"code": code_problem_flags()}; print("code:", rep["code"], flush=True)
    import sys
    market_kinds = {"mc", "number"} if "--market-mc-only" in sys.argv else None   # math verdicts are unchanged by the MC/number fixes
    for tag in ["marketE3_dt", "marketMH_dt", "marketE2_dt", "hardAv3", "stageA2v3_dt", "churnD"]:
        rep[f"matrix_{tag}"] = regrade_matrix(tag, market_kinds if tag.startswith("market") else None); r = rep[f"matrix_{tag}"]
        print(f"matrix_{tag}: {r['counts']} | wrong->right {r['wrong_to_right']} right->wrong {r['right_to_wrong']}", flush=True)
    for tag in ["eqq2", "eqq2_math", "eqqA"]:
        rep[f"fusion_{tag}"] = regrade_fusion(tag); print(f"fusion_{tag}: {rep[f'fusion_{tag}']['counts']}", flush=True)
    rep["cascade_stageC2v3"] = regrade_cascade("stageC2v3")
    print(f"cascade_stageC2v3: {rep['cascade_stageC2v3']['counts']} records changed {rep['cascade_stageC2v3']['n_records_changed']}")
    json.dump(rep, open(os.path.join(RUNS, "regrade_final_report.json"), "w"), indent=1)
