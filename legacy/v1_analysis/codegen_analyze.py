"""Regime analysis for the third (code) domain. Loads matrix_marketCG.json, builds the aligned
models x problems correctness matrix, and reports the same realizability quantities used for MATH-500/GPQA:
single-best, per-query oracle, gain G, co-failure beta (+ Clopper-Pearson + k), mean Pearson rho, mean
TETRACHORIC rho, and the tetrachoric single-factor underpricing. This decides which regime code is in
(ceiling-bound beta>0 like open-ended math, vs realizability-bound beta~0 like multiple-choice GPQA).
No API calls.  Usage: python3 codegen_analyze.py
"""
import os, json
import numpy as np
from scipy.stats import norm, beta as betadist
import realizability as RZ
from recompute_tetrachoric import mean_tetra

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")


def cp_interval(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else betadist.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else betadist.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def main():
    R = json.load(open(os.path.join(RUNS, "matrix_marketCG.json")))
    models = sorted({m for v in R.values() for m in v["models"]})
    qids = [q for q, v in R.items() if all(m in v["models"] for m in models)]
    M = np.array([[R[q]["models"][m]["correct"] for q in qids] for m in models], float)
    n_models, n_q = M.shape
    acc = M.mean(1)
    order = np.argsort(-acc)
    print(f"=== codegen regime: {n_models} models x {n_q} aligned problems ===")
    for i in order:
        print(f"  {models[i]:42} acc={acc[i]:.3f}")
    V_sb = float(acc.max()); sb = models[int(acc.argmax())]
    V_oracle = float((M.sum(0) > 0).mean()); G = V_oracle - V_sb
    k = int((M.sum(0) == 0).sum()); beta = k / n_q
    cp = cp_interval(k, n_q)
    met = RZ._metrics(M); rho_p = met["rho_bar"]
    rho_t, t = mean_tetra(M)
    bsf_t = RZ._copula_all_wrong(t, max(rho_t, 1e-6))
    up_t = beta / bsf_t if bsf_t > 0 else None
    out = {"n_models": n_models, "n_q": n_q, "single_best": V_sb, "single_best_model": sb,
           "V_oracle": V_oracle, "G": G, "beta": beta, "k_allwrong": k,
           "beta_cp": cp, "rho_pearson": rho_p, "rho_tetrachoric": rho_t,
           "beta_sf_tetra": bsf_t, "underprice_tetra": up_t,
           "acc_min": float(acc.min()), "acc_mean": float(acc.mean())}
    json.dump(out, open(os.path.join(RUNS, "codegen_regime.json"), "w"), indent=2)
    print(f"\nsingle-best={V_sb:.3f} ({sb})  oracle={V_oracle:.3f}  G={G:.3f}")
    print(f"beta(all-wrong)={beta:.3f}  k={k}/{n_q}  CP[{cp[0]:.3f},{cp[1]:.3f}]")
    print(f"rho: Pearson={rho_p:.3f}  TETRACHORIC={rho_t:.3f}  -> beta_sf={bsf_t:.4f}  underprice={up_t:.2f}x" if up_t else "beta=0 (realizability-bound)")
    regime = "ceiling-bound (open-ended, beta>0)" if beta > 0 else "realizability-bound (beta~0)"
    print(f"REGIME: {regime}")
    print("[codegen_analyze] wrote runs/codegen_regime.json")


if __name__ == "__main__":
    main()
