"""Integrity fix: the market-scale all-wrong tail beta is contaminated by correlated token-cap
truncation (reasoning models hitting the cap on the same hard problems). For every currently
all-wrong HARD query, re-run the TRUNCATED models with a large token budget (32768) and re-grade.
A query that gains >=1 correct model is no longer all-wrong, so beta drops. Reports the corrected
beta and how many of the 16 events survive. Bounded cost (only truncated cells in all-wrong events).
Usage: python3 detruncate.py --tag marketE_live
"""
import os, json, argparse, concurrent.futures as cf
import numpy as np
import orclient, data, grade

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")
CAP = {"mmlu_pro": 2048, "math500": 4096, "aime25": 8192, "aime24": 8192}
BIG = 32768

def main(tag, cap_usd):
    R = json.load(open(os.path.join(RUNS, f"matrix_{tag}.json")))
    hard = ["mmlu_pro", "math500", "aime25", "aime24"]
    qs = [q for q, v in R.items() if v["dataset"] in hard]
    models = sorted(set(m for q in qs for m in R[q]["models"]))
    qs = [q for q in qs if all(m in R[q]["models"] for m in models)]
    aw = [q for q in qs if sum(R[q]["models"][m]["correct"] for m in models) == 0]
    print(f"pooled hard: {len(models)} models x {len(qs)} queries; all-wrong before = {len(aw)} (beta={len(aw)/len(qs):.4f})")
    # build prompt lookup for the hard datasets
    prompts = {}
    for ds in hard:
        for it in data.load(ds, 250):
            prompts[it["qid"]] = it
    # collect all (query, model) truncated cells in all-wrong events
    jobs = []
    for q in aw:
        ds = R[q]["dataset"]; it = prompts.get(q)
        if it is None:
            continue
        for m in models:
            if R[q]["models"][m].get("tok_out", 0) >= CAP.get(ds, 9e9) - 8:
                jobs.append((q, m, it))
    print(f"re-running {len(jobs)} truncated cells across {len(aw)} all-wrong events at max_tokens={BIG}")

    def one(q, m, it):
        r = orclient.chat(m, [{"role": "user", "content": it["prompt"]}], temperature=0.0,
                          max_tokens=BIG, budget_cap=cap_usd)
        c = grade.grade(it["kind"], r["content"], it["gold"])
        return q, m, {"correct": c, "cost": r["cost"], "tok_in": r["tok_in"],
                      "tok_out": r["tok_out"], "cached": r["cached"], "detruncated": True}
    reran = 0
    with cf.ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(one, q, m, it) for q, m, it in jobs]
        for fu in cf.as_completed(futs):
            try:
                q, m, cell = fu.result(); R[q]["models"][m] = cell; reran += 1
            except Exception as e:
                print(f"  rerun failed: {str(e)[:70]}")
    new_aw = [q for q in qs if sum(R[q]["models"][m]["correct"] for m in models) == 0]
    fixed = len(aw) - len(new_aw)  # events that gained a correct model after de-truncation
    print(f"re-ran {reran} truncated cells in all-wrong events; {fixed} events gained a correct model")
    print(f"all-wrong AFTER de-truncation = {len(new_aw)} (beta={len(new_aw)/len(qs):.4f})")
    print(f"surviving all-wrong events: {sorted(new_aw)}")
    json.dump(R, open(os.path.join(RUNS, f"matrix_{tag}_detrunc.json"), "w"))
    print(f"[detruncate] wrote matrix_{tag}_detrunc.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="marketE_live")
    ap.add_argument("--cap_usd", type=float, default=320.0)
    a = ap.parse_args(); main(a.tag, a.cap_usd)
