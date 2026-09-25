"""Experiment driver. Build the base correctness matrix and analyze it.
Usage:
  export OPENROUTER_API_KEY=...
  python3 experiment.py matrix --datasets gsm8k,mmlu --n 60 --tag stageA --cap 25
  python3 experiment.py analyze --tag stageA
All numbers produced are REAL (from logged API calls). Nothing is fabricated.
"""
import os, sys, json, csv, argparse, concurrent.futures as cf
import numpy as np
import orclient, data, grade

HERE = os.path.dirname(__file__)
RUNS = os.path.join(HERE, "..", "runs")

def families():
    fam = {}
    with open(os.path.join(HERE, "..", "model_registry.csv")) as f:
        for row in csv.DictReader(r for r in f if not r.startswith("#")):
            fam[row["model_id"]] = (row["provider_family"], row["tier"])
    return fam

def all_models():
    # Base pool = standard chat models. Reasoning-tier models (e.g. o4-mini) are
    # excluded here: they reject small max_tokens, ignore temperature, and bill
    # hidden reasoning tokens, so they belong in a separate arm, not the base matrix.
    return [m for m, (fam, tier) in families().items() if tier != "strong-reasoning"]

def market_models():
    # Market-wide pool (53 models, 20 families) from market_registry.csv, verified live prices.
    ms = []
    path = os.path.join(HERE, "..", "market_registry.csv")
    with open(path) as f:
        for row in csv.DictReader(r for r in f if not r.startswith("#")):
            ms.append(row["model_id"])
    return ms

def build_matrix(datasets, n, tag, cap, workers=8, offset=0, models=None):
    models = models or all_models()
    items = []
    for ds in datasets:
        items += data.load(ds, n, offset)
    print(f"[matrix] {len(items)} items x {len(models)} models = {len(items)*len(models)} cells; budget cap ${cap}")
    results = {}  # qid -> {meta, models:{model:{correct,cost,...}}}
    for it in items:
        results[it["qid"]] = {"dataset": it["dataset"], "kind": it["kind"], "gold": it["gold"], "models": {}}

    def one(it, model):
        r = orclient.chat(model, [{"role": "user", "content": it["prompt"]}],
                          temperature=0.0, max_tokens=it["max_tokens"], budget_cap=cap)
        c = grade.grade(it["kind"], r["content"], it["gold"])
        return it["qid"], model, {"correct": c, "cost": r["cost"], "tok_in": r["tok_in"],
                                  "tok_out": r["tok_out"], "cached": r["cached"]}

    jobs = [(it, m) for it in items for m in models]
    done, failed, stop = 0, 0, False
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, it, m) for it, m in jobs]
        for fu in cf.as_completed(futs):
            try:
                qid, model, cell = fu.result()
                results[qid]["models"][model] = cell
            except orclient.BudgetExceeded as e:
                if not stop:
                    print(f"[matrix] STOP (budget): {e}")
                stop = True
                continue
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"[matrix] cell failed (skipped): {str(e)[:120]}")
                continue
            done += 1
            if done % 100 == 0:
                s = orclient.spend_summary()
                print(f"[matrix] {done}/{len(jobs)} ok ({failed} failed) | spent ${s['total_usd']:.4f} | cache {s['cache_hits']}/{s['cache_hits']+s['cache_misses']}")
    print(f"[matrix] completed: {done} ok, {failed} failed of {len(jobs)}")
    out = os.path.join(RUNS, f"matrix_{tag}.json")
    json.dump(results, open(out, "w"))
    s = orclient.spend_summary()
    print(f"[matrix] wrote {out} | total spent ${s['total_usd']:.4f} | cache hit rate "
          f"{s['cache_hits']/max(1,s['cache_hits']+s['cache_misses']):.2%}")
    return out

def analyze(tag):
    results = json.load(open(os.path.join(RUNS, f"matrix_{tag}.json")))
    fam = families()
    models = all_models()
    # keep only queries fully answered by all models (for a clean aligned matrix)
    qids = [q for q, v in results.items() if all(m in v["models"] for m in models)]
    if not qids:
        print("no fully-answered queries"); return
    datasets = sorted(set(results[q]["dataset"] for q in qids))
    M = np.array([[results[q]["models"][m]["correct"] for q in qids] for m in models], dtype=float)  # models x queries
    cost = np.array([[results[q]["models"][m]["cost"] for q in qids] for m in models], dtype=float)

    acc = M.mean(axis=1)
    mean_cost = cost.mean(axis=1)  # avg $/query per model
    order = np.argsort(-acc)
    print(f"\n=== ANALYSIS tag={tag}  |  {len(qids)} aligned queries over {datasets}  |  {len(models)} models ===")
    print(f"{'model':46} {'acc':>6} {'$/q':>9} {'$/correct':>10} {'family':>10} {'tier':>8}")
    for i in order:
        m = models[i]; a = acc[i]; cpq = mean_cost[i]
        cpc = cpq / a if a > 0 else float('inf')
        print(f"{m:46} {a:6.3f} {cpq:9.5f} {cpc:10.5f} {fam[m][0]:>10} {fam[m][1]:>8}")

    # policies
    V_sb = acc.max(); sb = models[int(acc.argmax())]
    V_rand = acc.mean()
    per_query_any = (M.sum(axis=0) > 0).astype(float)  # partition-free per-query oracle
    V_oracle = per_query_any.mean()
    G = V_oracle - V_sb
    cheapest_acc_model = models[int(np.argmin([mean_cost[i] if acc[i] >= 0.5*V_sb else 1e9 for i in range(len(models))]))]
    print(f"\nsingle-best         = {V_sb:.3f}  ({sb})")
    print(f"random (mean acc)   = {V_rand:.3f}")
    print(f"oracle (per-query)  = {V_oracle:.3f}")
    print(f"ORACLE GAIN G       = {G:.3f}   (= oracle - single-best; >0 iff Q not row-dominated)")

    # rho matrix on binary correctness
    # guard: drop models with zero variance (all right or all wrong) for correlation
    var = M.var(axis=1)
    keep = [i for i in range(len(models)) if var[i] > 1e-9]
    R = np.corrcoef(M[keep])
    iu = np.triu_indices(len(keep), k=1)
    mean_rho = float(R[iu].mean())
    # block structure: within-family vs cross-family mean rho
    fam_of = [fam[models[i]][0] for i in keep]
    within, cross = [], []
    for a in range(len(keep)):
        for b in range(a+1, len(keep)):
            (within if fam_of[a] == fam_of[b] else cross).append(R[a, b])
    print(f"\ninter-model error correlation (Pearson on correctness, {len(keep)} models w/ variance):")
    print(f"  mean off-diagonal rho   = {mean_rho:.3f}")
    print(f"  within-family  mean rho = {np.mean(within):.3f}  (n={len(within)})" if within else "  within-family: n/a")
    print(f"  cross-family   mean rho = {np.mean(cross):.3f}  (n={len(cross)})" if cross else "  cross-family: n/a")
    print(f"  => block structure: within {np.mean(within):.3f} vs cross {np.mean(cross):.3f}"
          if within and cross else "")
    return {"tag": tag, "n_queries": len(qids), "datasets": datasets, "n_models": len(models),
            "V_sb": V_sb, "single_best": sb, "V_rand": V_rand, "V_oracle": V_oracle, "G": G,
            "mean_rho": mean_rho,
            "within_family_rho": float(np.mean(within)) if within else None,
            "cross_family_rho": float(np.mean(cross)) if cross else None}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    pm = sub.add_parser("matrix"); pm.add_argument("--datasets", required=True); pm.add_argument("--n", type=int, required=True)
    pm.add_argument("--tag", required=True); pm.add_argument("--cap", type=float, required=True)
    pm.add_argument("--offset", type=int, default=0); pm.add_argument("--workers", type=int, default=8)
    pm.add_argument("--models", default=None, help="comma-separated model id override (else full registry pool)")
    pm.add_argument("--pool", default=None, choices=["market"], help="use a named pool (market = 53-model market_registry)")
    pa = sub.add_parser("analyze"); pa.add_argument("--tag", required=True)
    a = ap.parse_args()
    if a.cmd == "matrix":
        sel = a.models.split(",") if a.models else (market_models() if a.pool == "market" else None)
        build_matrix(a.datasets.split(","), a.n, a.tag, a.cap, a.workers, a.offset, models=sel)
    elif a.cmd == "analyze":
        res = analyze(a.tag)
        if res:
            json.dump(res, open(os.path.join(RUNS, f"analysis_{a.tag}.json"), "w"), indent=2)
            print(f"\n[analyze] wrote analysis_{a.tag}.json")
    else:
        ap.print_help()
