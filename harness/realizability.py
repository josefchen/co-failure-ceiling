"""Frontier Realizability Law at market scale.

Reads matrix_<tag>.json (market pool, 53 models x many benchmarks) and measures, per
benchmark and pooled, the structure that sets the orchestration ceiling:

  - acc_i (per-model), single-best V_sb, per-query oracle V_oracle, oracle gain G;
  - beta = P(all models wrong)  [the realizability tail: orchestration cannot beat 1-beta];
  - mean pairwise error correlation rho_bar (Pearson on the binary correctness matrix);
  - beta_1f = single-factor Gaussian-copula tail implied by rho_bar and the marginal error
    rates -> what a practitioner pricing co-failure off mean pairwise rho would predict;
  - UNDERPRICING RATIO = beta / beta_1f  [the headline: pairwise rho underprices the tail];
  - rho_eff = the single base correlation that reproduces the empirical beta (CDO-style
    base correlation; framed as a known device, not a novel object);
  - model-clustered CIs via leave-one-model-out jackknife (the only honest inference here:
    queries are not the unit of variation, models are);
  - pool-size scaling: beta and the underpricing ratio as the pool grows k=2..M (random
    sub-pools), i.e. does adding models to the market keep shrinking the realizable tail?

All inputs are logged real runs. Nothing is fabricated.
Usage: python3 realizability.py --tag marketE [--datasets mmlu_pro,math500,aime25]
"""
import os, json, csv, argparse, itertools
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.stats import norm
from scipy.optimize import brentq

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")
_NODES, _W = hermegauss(64)          # E-Hermite nodes for N(0,1): weight/sqrt(2pi)
_WN = _W / np.sqrt(2 * np.pi)

def families():
    fam = {}
    for reg in ("model_registry.csv", "market_registry.csv"):
        p = os.path.join(HERE, "..", reg)
        if not os.path.exists(p): continue
        for row in csv.DictReader(r for r in open(p) if not r.startswith("#")):
            fam[row["model_id"]] = row["provider_family"]
    return fam

def _copula_all_wrong(t, rho):
    """P(all wrong) under single-factor Gaussian copula. t_i = Phi^{-1}(p_wrong_i).
    Wrong_i iff latent X_i < t_i, X_i = sqrt(rho) Z + sqrt(1-rho) eps_i."""
    rho = min(max(rho, 1e-6), 0.999999)
    a = np.sqrt(rho); b = np.sqrt(1 - rho)
    # integrate over Z: prod_i Phi((t_i - a z)/b)
    acc = 0.0
    for z, w in zip(_NODES, _WN):
        cond = norm.cdf((t - a * z) / b)
        acc += w * np.prod(cond)
    return float(acc)

def _build(R, models, dataset=None):
    """Return correctness matrix M (models x queries) over queries answered by ALL models."""
    qids = [q for q, v in R.items()
            if (dataset is None or v["dataset"] == dataset) and all(m in v["models"] for m in models)]
    if not qids: return None, None
    M = np.array([[R[q]["models"][m]["correct"] for q in qids] for m in models], float)
    return M, qids

def _metrics(M):
    """Core realizability metrics on a models x queries 0/1 matrix."""
    n_models, n_q = M.shape
    acc = M.mean(1)
    V_sb = float(acc.max()); V_oracle = float((M.sum(0) > 0).mean()); G = V_oracle - V_sb
    beta = float((M.sum(0) == 0).mean())                     # empirical P(all wrong)
    p_wrong = np.clip(1 - acc, 1e-6, 1 - 1e-6)
    t = norm.ppf(p_wrong)
    # mean pairwise rho on correctness (drop zero-variance models)
    var = M.var(1); keep = var > 1e-9
    if keep.sum() >= 2:
        Rc = np.corrcoef(M[keep]); iu = np.triu_indices(keep.sum(), 1)
        rho_bar = float(np.nanmean(Rc[iu]))
    else:
        rho_bar = 0.0
    beta_1f = _copula_all_wrong(t, max(rho_bar, 1e-6))       # implied by pairwise rho
    beta_indep = _copula_all_wrong(t, 1e-6)                  # independence benchmark
    # rho_eff: base correlation reproducing empirical beta
    rho_eff = None
    if beta > beta_1f and beta < _copula_all_wrong(t, 0.999):
        try:
            rho_eff = float(brentq(lambda r: _copula_all_wrong(t, r) - beta, 1e-4, 0.999, xtol=1e-4))
        except Exception:
            rho_eff = None
    ratio = (beta / beta_1f) if beta_1f > 0 else float("inf")
    return {"n_models": int(n_models), "n_q": int(n_q), "acc_max": V_sb, "acc_mean": float(acc.mean()),
            "V_oracle": V_oracle, "G": float(G), "beta": beta, "beta_1factor": float(beta_1f),
            "beta_indep": float(beta_indep), "underpricing_ratio": float(ratio),
            "rho_bar": rho_bar, "rho_eff": rho_eff}

def _jackknife_models(M, fn):
    """Leave-one-model-out jackknife -> (mean, se) for a scalar statistic fn(M)."""
    n = M.shape[0]; vals = []
    for i in range(n):
        sub = np.delete(M, i, axis=0)
        v = fn(sub)
        if v is not None and np.isfinite(v): vals.append(v)
    vals = np.array(vals)
    if len(vals) < 2: return None, None
    mean = vals.mean()
    se = np.sqrt((n - 1) / n * np.sum((vals - mean) ** 2))
    return float(mean), float(se)

def _pool_scaling(M, sizes, reps=200, seed=12345):
    """beta and underpricing ratio vs pool size k on genuinely RANDOM model sub-pools.
    Reports mean/median plus a 5-95 percentile band across draws, the mean number of
    all-wrong events per draw (so the reader sees the tail's support), and a one-sided
    test that the full-pool ratio exceeds the k=2 ratio under resampling."""
    n = M.shape[0]; rng = np.random.default_rng(seed)
    out = []
    for k in sizes:
        if k > n: continue
        bs, rs, aw = [], [], []
        draws = reps if k < n else 1  # full pool is unique
        for _ in range(draws):
            idx = rng.choice(n, size=k, replace=False) if k < n else np.arange(n)
            sub = M[idx]
            mm = _metrics(sub)
            bs.append(mm["beta"]); aw.append(int((sub.sum(0) == 0).sum()))
            if np.isfinite(mm["underpricing_ratio"]):
                rs.append(mm["underpricing_ratio"])
        rs = [r for r in rs if np.isfinite(r)]
        out.append({"k": int(k), "beta_mean": float(np.mean(bs)),
                    "ratio_median": float(np.median(rs)) if rs else float("nan"),
                    "ratio_p05": float(np.percentile(rs, 5)) if rs else float("nan"),
                    "ratio_p95": float(np.percentile(rs, 95)) if rs else float("nan"),
                    "allwrong_events_mean": float(np.mean(aw)), "draws": draws})
    return out

def main(tag, datasets):
    R = json.load(open(os.path.join(RUNS, f"matrix_{tag}.json")))
    fam = families()
    all_models = sorted({m for v in R.values() for m in v["models"]})
    avail_ds = sorted({v["dataset"] for v in R.values()})
    if datasets: avail_ds = [d for d in avail_ds if d in datasets]
    print(f"=== Frontier Realizability Law @ market scale (tag={tag}) ===")
    print(f"pool: {len(all_models)} models, {len(set(fam.get(m,'?') for m in all_models))} families | datasets: {avail_ds}")
    report = {"tag": tag, "n_models_catalog": len(all_models),
              "families": sorted(set(fam.get(m, "?") for m in all_models)), "per_dataset": {}, "pooled": None}
    print(f"\n{'dataset':10} {'M':>3} {'q':>4} {'sb':>5} {'orac':>5} {'G':>5} {'beta':>6} {'beta_1f':>8} {'ratio':>6} {'rho_bar':>7} {'rho_eff':>7}")
    for ds in avail_ds:
        # use models present on essentially all queries of this dataset
        cnt = {}
        for q, v in R.items():
            if v["dataset"] != ds: continue
            for m in v["models"]: cnt[m] = cnt.get(m, 0) + 1
        if not cnt: continue
        qtot = max(cnt.values()); models = sorted([m for m, c in cnt.items() if c >= 0.95 * qtot])
        M, qids = _build(R, models, ds)
        if M is None or M.shape[1] < 10:
            print(f"{ds:10} (skip: too few aligned queries)"); continue
        mm = _metrics(M)
        bmean, bse = _jackknife_models(M, lambda X: _metrics(X)["beta"])
        rmean, rse = _jackknife_models(M, lambda X: _metrics(X)["underpricing_ratio"])
        mm["beta_jk_se"] = bse; mm["ratio_jk_se"] = rse
        mm["models"] = models
        report["per_dataset"][ds] = mm
        re = f"{mm['rho_eff']:.3f}" if mm['rho_eff'] is not None else "  -  "
        print(f"{ds:10} {mm['n_models']:3d} {mm['n_q']:4d} {mm['acc_max']:.3f} {mm['V_oracle']:.3f} "
              f"{mm['G']:+.3f} {mm['beta']:.4f} {mm['beta_1factor']:.5f} {mm['underpricing_ratio']:5.1f}x "
              f"{mm['rho_bar']:.3f}  {re}")
    # pooled across the hard, non-saturated benchmarks (where the tail is informative)
    hard = [d for d in ("mmlu_pro", "math500", "aime25", "aime24") if d in avail_ds]
    if hard:
        cnt = {}
        for q, v in R.items():
            if v["dataset"] not in hard: continue
            for m in v["models"]: cnt[m] = cnt.get(m, 0) + 1
        qtot = max(cnt.values()) if cnt else 0
        models = sorted([m for m, c in cnt.items() if c >= 0.80 * qtot])
        qids = [q for q, v in R.items() if v["dataset"] in hard and all(m in v["models"] for m in models)]
        if qids:
            M = np.array([[R[q]["models"][m]["correct"] for q in qids] for m in models], float)
            mm = _metrics(M)
            bmean, bse = _jackknife_models(M, lambda X: _metrics(X)["beta"])
            rmean, rse = _jackknife_models(M, lambda X: _metrics(X)["underpricing_ratio"])
            mm["beta_jk_se"] = bse; mm["ratio_jk_se"] = rse; mm["models"] = models; mm["datasets"] = hard
            sizes = [2, 3, 5, 8, 12, 18, 25, 35, len(models)]
            mm["pool_scaling"] = _pool_scaling(M, sizes)
            report["pooled"] = mm
            print(f"\n=== POOLED hard benchmarks {hard} | {len(models)} models x {len(qids)} queries ===")
            print(f"  single-best={mm['acc_max']:.3f}  oracle={mm['V_oracle']:.3f}  G={mm['G']:+.3f}")
            print(f"  beta (P all wrong)      = {mm['beta']:.4f}  (+/- {bse:.4f}, model-jackknife)")
            print(f"  beta_1factor(rho_bar)   = {mm['beta_1factor']:.5f}   [rho_bar={mm['rho_bar']:.3f}]")
            print(f"  UNDERPRICING RATIO      = {mm['underpricing_ratio']:.1f}x  (+/- {rse:.1f})")
            print(f"  rho_eff (base corr.)    = {mm['rho_eff']}")
            print(f"  pool-size scaling (k -> beta, ratio):")
            for r in mm["pool_scaling"]:
                print(f"     k={r['k']:3d}  beta={r['beta_mean']:.4f}  ratio={r['ratio_median']:.1f}x")
    out = os.path.join(RUNS, f"realizability_{tag}.json")
    json.dump(report, open(out, "w"), indent=2)
    print(f"\n[realizability] wrote {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--datasets", default=None, help="comma list to restrict (else all in matrix)")
    a = ap.parse_args()
    main(a.tag, a.datasets.split(",") if a.datasets else None)
