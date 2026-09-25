"""Deployable learned router + cost-aware-oracle (optimal) frontier, computed offline from the
logged correctness matrix (NO new API calls). Answers Pillar A's open question: does a TRAINED
router capture the oracle gain G, and how close does it get to the optimal cost-quality frontier
(the Dekoninck-style cost-aware oracle, argmax_i [q_i(t) - lambda c_i])?

Method: TF-IDF(prompt)+domain+length features; per-model held-out logistic P(correct); router
routes to argmax_i [p_i(x) - lambda c_i] over a lambda sweep -> realized (cost, accuracy) frontier.
Baselines: single-cheapest, single-best (held-out), random, learned-router frontier, cost-aware
ORACLE frontier (optimal upper bound). Reports the fraction of oracle gain G the learned router
captures, with a query-bootstrap CI. All numbers from logged data.
Usage: python3 router_baseline.py --tag stageA2v3 --datasets gsm8k,mmlu,math500,arc --n 120
"""
import os, json, argparse
import numpy as np
import data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")

def main(tag, datasets, n, seed=0):
    R = json.load(open(os.path.join(RUNS, f"matrix_{tag}.json")))
    models = sorted({m for v in R.values() for m in v["models"]})
    # prompts by qid (reload datasets; qids are deterministic)
    prompts = {}
    for ds in datasets:
        for it in data.load(ds, n):
            prompts[it["qid"]] = (it["prompt"], it["dataset"])
    qids = [q for q in R if q in prompts and all(m in R[q]["models"] for m in models)]
    X_text = [prompts[q][0] for q in qids]
    dsets = sorted(set(prompts[q][1] for q in qids))
    dom = np.array([[1.0 if prompts[q][1] == d else 0.0 for d in dsets] for q in qids])
    length = np.array([[len(prompts[q][0]) / 1000.0] for q in qids])
    Y = np.array([[R[q]["models"][m]["correct"] for m in models] for q in qids], float)  # queries x models
    cost = np.array([np.mean([R[q]["models"][m]["cost"] for q in qids]) for m in models])  # mean $/q per model
    rng = np.random.default_rng(seed); idx = rng.permutation(len(qids)); cut = int(0.6 * len(qids))
    tr, te = idx[:cut], idx[cut:]
    vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
    Xtf = vec.fit_transform(X_text)
    import scipy.sparse as sp
    Xall = sp.hstack([Xtf, sp.csr_matrix(np.hstack([dom, length]))]).tocsr()
    Xtr, Xte = Xall[tr], Xall[te]
    # per-model held-out P(correct)
    P = np.zeros((len(te), len(models)))
    for j, m in enumerate(models):
        ytr = Y[tr, j]
        if ytr.min() == ytr.max():
            P[:, j] = ytr.mean()  # constant model
        else:
            clf = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr, ytr)
            P[:, j] = clf.predict_proba(Xte)[:, 1]
    Yte = Y[te]
    # baselines on TEST
    acc_sb_model = int(Y[tr].mean(0).argmax())          # single-best chosen on TRAIN (held-out)
    V_sb = Yte[:, acc_sb_model].mean()
    V_rand = Yte.mean()
    V_oracle = (Yte.sum(1) > 0).mean()                  # per-query oracle (any model correct)
    G = V_oracle - V_sb
    # learned router: route to argmax_i p_i  (quality-max operating point)
    route = P.argmax(1)
    V_lr = Yte[np.arange(len(te)), route].mean()
    cost_lr = cost[route].mean()
    frac = (V_lr - V_sb) / G if G > 1e-9 else float('nan')
    # bootstrap CI on fraction captured
    bs = []
    for _ in range(2000):
        b = rng.integers(0, len(te), len(te))
        sb = Yte[b, acc_sb_model].mean(); orc = (Yte[b].sum(1) > 0).mean()
        lr = Yte[b, route[b]].mean()
        if orc - sb > 1e-9:
            bs.append((lr - sb) / (orc - sb))
    fci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (float('nan'),)*2
    # cost-aware frontiers (lambda sweep): learned vs oracle
    def frontier(scoremat):
        pts = []
        for lam in np.concatenate([[0], np.logspace(-2, 3, 30)]):
            r = (scoremat - lam * cost[None, :]).argmax(1)
            pts.append((cost[r].mean(), Yte[np.arange(len(te)), r].mean()))
        return sorted(set(pts))
    fr_lr = frontier(P)
    fr_or = frontier(Yte)  # cost-aware oracle (optimal): uses true correctness
    print(f"=== Learned router ({tag}) | {len(qids)} queries ({len(te)} test) | {len(models)} models ===")
    print(f"single-best (held-out) = {V_sb:.3f}   random = {V_rand:.3f}   per-query oracle = {V_oracle:.3f}   G = {G:.3f}")
    print(f"LEARNED ROUTER (quality-max): acc = {V_lr:.3f}  (\\$/q {cost_lr:.5f})")
    print(f"  fraction of oracle gain G captured = {frac:.2f}  95% CI [{fci[0]:.2f},{fci[1]:.2f}]")
    print(f"  beats single-best? {'YES' if (fci[0] > 0) else 'not significantly'} (CI on V_lr-V_sb via G-share)")
    json.dump({"tag": tag, "V_sb": float(V_sb), "V_rand": float(V_rand), "V_oracle": float(V_oracle),
               "G": float(G), "V_learned_router": float(V_lr), "frac_G_captured": float(frac),
               "frac_ci": fci, "n_test": int(len(te)), "frontier_learned": fr_lr, "frontier_oracle": fr_or,
               "cost_per_model": dict(zip(models, cost.tolist()))},
              open(os.path.join(RUNS, f"router_{tag}.json"), "w"), indent=2)
    print(f"[router] wrote router_{tag}.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--datasets", required=True); ap.add_argument("--n", type=int, required=True)
    a = ap.parse_args(); main(a.tag, a.datasets.split(","), a.n)
