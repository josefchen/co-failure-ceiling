"""Stronger learned router, to test whether the near-zero realizable gain survives a much stronger
model than the TF-IDF+domain logistic. Three routers, offline from the logged matrix (no API calls):
  (L)  logistic on word TF-IDF + domain + length (the original baseline);
  (G)  gradient-boosted per-model P(correct) on word+char TF-IDF -> SVD(256) + domain + length;
  (M)  a direct MULTICLASS best-model predictor (gradient boosting on the same dense features),
       trained on the per-query argmax model, routing to its prediction.
Reports each router's fraction of the per-query oracle gain G captured on a held-out split, with a
query-bootstrap CI, and the BEST across the three. If even the strongest captures ~0 of G, the
"realizable routing gain is near zero" claim is robust to router strength.
Usage: python3 router_strong.py --tag stageA2v3 --datasets gsm8k,mmlu,math500,arc --n 250
"""
import os, json, argparse
import numpy as np
import data
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")

def captured(Yte, route, sb_model, rng):
    V_sb = Yte[:, sb_model].mean(); V_or = (Yte.sum(1) > 0).mean()
    G = V_or - V_sb
    V_lr = Yte[np.arange(len(Yte)), route].mean()
    frac = (V_lr - V_sb) / G if G > 1e-9 else float("nan")
    bs = []
    for _ in range(2000):
        b = rng.integers(0, len(Yte), len(Yte))
        sb = Yte[b, sb_model].mean(); orc = (Yte[b].sum(1) > 0).mean(); lr = Yte[b, route[b]].mean()
        if orc - sb > 1e-9:
            bs.append((lr - sb) / (orc - sb))
    ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (float("nan"),) * 2
    return float(V_lr), float(frac), ci

def main(tag, datasets, n, seed=0):
    R = json.load(open(os.path.join(RUNS, f"matrix_{tag}.json")))
    models = sorted({m for v in R.values() for m in v["models"]})
    prompts = {}
    for ds in datasets:
        for it in data.load(ds, n):
            prompts[it["qid"]] = (it["prompt"], it["dataset"])
    qids = [q for q in R if q in prompts and all(m in R[q]["models"] for m in models)]
    X_text = [prompts[q][0] for q in qids]
    dsets = sorted(set(prompts[q][1] for q in qids))
    dom = np.array([[1.0 if prompts[q][1] == d else 0.0 for d in dsets] for q in qids])
    length = np.array([[len(prompts[q][0]) / 1000.0, np.log1p(len(prompts[q][0]))] for q in qids])
    Y = np.array([[R[q]["models"][m]["correct"] for m in models] for q in qids], float)
    rng = np.random.default_rng(seed); idx = rng.permutation(len(qids)); cut = int(0.6 * len(qids))
    tr, te = idx[:cut], idx[cut:]
    # features
    wtf = TfidfVectorizer(max_features=6000, ngram_range=(1, 2), min_df=2).fit_transform(X_text)
    ctf = TfidfVectorizer(max_features=6000, ngram_range=(3, 5), analyzer="char_wb", min_df=2).fit_transform(X_text)
    Xsparse = sp.hstack([wtf, ctf, sp.csr_matrix(np.hstack([dom, length]))]).tocsr()
    svd = TruncatedSVD(n_components=min(256, Xsparse.shape[1] - 1), random_state=seed)
    Xdense = np.hstack([svd.fit_transform(Xsparse), dom, length])
    Yte = Y[te]; sb_model = int(Y[tr].mean(0).argmax())
    res = {}
    # (L) logistic on sparse word+char
    PL = np.full((len(te), len(models)), 0.0)
    for j in range(len(models)):
        ytr = Y[tr, j]
        PL[:, j] = ytr.mean() if ytr.min() == ytr.max() else \
            LogisticRegression(max_iter=1000).fit(Xsparse[tr], ytr).predict_proba(Xsparse[te])[:, 1]
    res["logistic"] = captured(Yte, PL.argmax(1), sb_model, rng)
    # (G) gradient-boosted per-model P(correct)
    PG = np.full((len(te), len(models)), 0.0)
    for j in range(len(models)):
        ytr = Y[tr, j]
        if ytr.min() == ytr.max():
            PG[:, j] = ytr.mean()
        else:
            PG[:, j] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08).fit(
                Xdense[tr], ytr).predict_proba(Xdense[te])[:, 1]
    res["gbm_permodel"] = captured(Yte, PG.argmax(1), sb_model, rng)
    # (M) multiclass best-model predictor
    best_tr = Y[tr].argmax(1)  # per-train-query best model (ties -> first)
    if len(set(best_tr)) > 1:
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08).fit(Xdense[tr], best_tr)
        routeM = clf.predict(Xdense[te]).astype(int)
        res["gbm_multiclass"] = captured(Yte, routeM, sb_model, rng)
    V_sb = Yte[:, sb_model].mean(); V_or = (Yte.sum(1) > 0).mean()
    print(f"=== STRONG routers ({tag}) | {len(qids)} q ({len(te)} test) | {len(models)} models | single-best={V_sb:.3f} oracle={V_or:.3f} G={V_or-V_sb:.3f} ===")
    best_frac = -9
    for name, (vlr, frac, ci) in res.items():
        print(f"  {name:16} acc={vlr:.3f}  frac of G captured={frac:+.2f}  95% CI [{ci[0]:+.2f},{ci[1]:+.2f}]  {'(beats single-best)' if ci[0]>0 else '(NOT sig. > single-best)'}")
        best_frac = max(best_frac, frac)
    print(f"  BEST router captures {best_frac:+.2f} of G")
    json.dump({"tag": tag, "V_sb": float(V_sb), "V_oracle": float(V_or), "G": float(V_or - V_sb),
               "routers": {k: {"acc": v[0], "frac_G": v[1], "ci": v[2]} for k, v in res.items()},
               "best_frac_G": float(best_frac)}, open(os.path.join(RUNS, f"router_strong_{tag}.json"), "w"), indent=2)
    print(f"[router_strong] wrote router_strong_{tag}.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--datasets", required=True); ap.add_argument("--n", type=int, default=250)
    a = ap.parse_args(); main(a.tag, a.datasets.split(","), a.n)
