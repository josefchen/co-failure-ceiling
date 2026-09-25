"""Self-consistency fusion policies (verifiable-task analogue of fusion/MoA):
 - Self-MoA(k): majority vote over k samples from the SINGLE BEST model (homogeneous).
 - Hetero(k):   majority vote over 1 sample from each of the top-k DISTINCT models.
Tests H-B1 (break-even k), H-B3 (Self-MoA is the binding baseline), and the
diversification-limit prediction (optimal k smaller for higher-rho families).
All numbers are REAL (sampled via API, temp>0, multiple seeds). Nothing fabricated.
Usage: export OPENROUTER_API_KEY=...
       python3 fusion.py --datasets gsm8k,mmlu,math500 --n 100 --models <csv> --S 6 --cap 60 --tag stageB
"""
import os, sys, json, argparse, collections, concurrent.futures as cf
import numpy as np
import orclient, data, grade

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")

def majority(answers):
    answers = [a for a in answers if a is not None]
    if not answers: return None
    return collections.Counter(answers).most_common(1)[0][0]

def run(datasets, n, models, S, cap, tag, workers=12):
    items = []
    for ds in datasets: items += data.load(ds, n)
    print(f"[fusion] {len(items)} items x {len(models)} models x {S} samples = {len(items)*len(models)*S} calls; cap ${cap}")

    def one(it, model, s):
        r = orclient.chat(model, [{"role": "user", "content": it["prompt"]}],
                          temperature=0.7, max_tokens=it["max_tokens"], seed=s, budget_cap=cap)
        return it["qid"], model, s, grade.extract(it["kind"], r["content"]), r["cost"]

    # samples[qid][model] = [answers...]; cost[qid][model] = [costs...]
    samples = {it["qid"]: {m: [None]*S for m in models} for it in items}
    costs = {it["qid"]: {m: [0.0]*S for m in models} for it in items}
    meta = {it["qid"]: {"kind": it["kind"], "gold": it["gold"], "dataset": it["dataset"]} for it in items}
    jobs = [(it, m, s) for it in items for m in models for s in range(1, S+1)]
    failed = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, it, m, s) for it, m, s in jobs]
        for i, fu in enumerate(cf.as_completed(futs)):
            try:
                qid, model, s, ans, cost = fu.result()
                samples[qid][model][s-1] = ans; costs[qid][model][s-1] = cost
            except orclient.BudgetExceeded as e:
                print(f"[fusion] STOP (budget): {e}"); break
            except Exception as e:
                failed += 1
                if failed <= 5: print(f"[fusion] cell failed: {str(e)[:100]}")
            if (i+1) % 500 == 0:
                sp = orclient.spend_summary(); print(f"[fusion] {i+1}/{len(jobs)} | spent ${sp['total_usd']:.4f}")
    json.dump({"samples": samples, "costs": costs, "meta": meta, "models": models, "S": S, "datasets": datasets},
              open(os.path.join(RUNS, f"fusion_{tag}.json"), "w"))
    print(f"[fusion] saved fusion_{tag}.json | failed {failed}")
    analyze(tag)

def _acc_cost(samples, costs, meta, qids, member_fn, k):
    """member_fn(qid)->list of (model, sample_index) used at size k. Returns (acc, mean_cost)."""
    correct, tot_cost = 0, 0.0
    for q in qids:
        picks = member_fn(q, k)
        ans = [samples[q][m][s] for (m, s) in picks]
        voted = majority(ans)
        correct += grade.check(meta[q]["kind"], voted, meta[q]["gold"])
        tot_cost += sum(costs[q][m][s] for (m, s) in picks)
    return correct/len(qids), tot_cost/len(qids)

def analyze(tag):
    D = json.load(open(os.path.join(RUNS, f"fusion_{tag}.json")))
    samples, costs, meta, models, S = D["samples"], D["costs"], D["meta"], D["models"], D["S"]
    qids = list(meta.keys())
    # rank models by single-sample accuracy (sample 0)
    acc0 = {m: np.mean([grade.check(meta[q]["kind"], samples[q][m][0], meta[q]["gold"]) for q in qids]) for m in models}
    ranked = sorted(models, key=lambda m: -acc0[m])
    best = ranked[0]
    print(f"\n=== FUSION analysis tag={tag} | {len(qids)} queries | best={best} (acc0={acc0[best]:.3f}) ===")
    print(f"{'k':>3} | {'SelfMoA_acc':>11} {'SelfMoA_$/q':>11} | {'Hetero_acc':>10} {'Hetero_$/q':>10}")
    selfmoa = lambda q, k: [(best, s) for s in range(min(k, S))]
    hetero  = lambda q, k: [(ranked[j], 0) for j in range(min(k, len(ranked)))]
    rows = []
    for k in range(1, min(S, len(ranked)) + 1):
        sa, sc = _acc_cost(samples, costs, meta, qids, selfmoa, k)
        ha, hc = _acc_cost(samples, costs, meta, qids, hetero, k)
        rows.append({"k": k, "selfmoa_acc": sa, "selfmoa_cpq": sc, "hetero_acc": ha, "hetero_cpq": hc})
        print(f"{k:>3} | {sa:11.3f} {sc:11.5f} | {ha:10.3f} {hc:10.5f}")
    # per-dataset rho + optimal hetero-k (break-even on raw accuracy/cost is dataset-specific)
    print("\nper-dataset: mean inter-model rho (sample-0 correctness) and hetero argmax-acc k")
    per = {}
    for ds in D["datasets"]:
        dq = [q for q in qids if meta[q]["dataset"] == ds]
        Mb = np.array([[grade.check(meta[q]["kind"], samples[q][m][0], meta[q]["gold"]) for q in dq] for m in models], dtype=float)
        var = Mb.var(axis=1); keep = [i for i in range(len(models)) if var[i] > 1e-9]
        rho = float(np.corrcoef(Mb[keep])[np.triu_indices(len(keep), 1)].mean()) if len(keep) > 1 else float('nan')
        # hetero accuracy curve on this dataset
        hk = []
        for k in range(1, len(ranked)+1):
            ha, _ = _acc_cost(samples, costs, meta, dq, hetero, k); hk.append(ha)
        per[ds] = {"rho": rho, "hetero_argmax_k": int(np.argmax(hk)+1), "hetero_curve": hk, "n": len(dq), "best_single_acc": float(np.max(Mb.mean(axis=1)))}
        print(f"  {ds:10} n={len(dq):4} mean_rho={rho:.3f}  hetero_argmax_k={per[ds]['hetero_argmax_k']}  best_single={per[ds]['best_single_acc']:.3f}")
    json.dump({"rows": rows, "per_dataset": per, "best": best}, open(os.path.join(RUNS, f"fusion_analysis_{tag}.json"), "w"), indent=2)
    print(f"[fusion] wrote fusion_analysis_{tag}.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True); ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--models", required=True); ap.add_argument("--S", type=int, default=6)
    ap.add_argument("--cap", type=float, required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=48); ap.add_argument("--analyze_only", action="store_true")
    a = ap.parse_args()
    if a.analyze_only: analyze(a.tag)
    else: run(a.datasets.split(","), a.n, a.models.split(","), a.S, a.cap, a.tag, workers=a.workers)
