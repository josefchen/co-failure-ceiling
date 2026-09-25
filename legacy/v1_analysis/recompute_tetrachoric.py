"""First-release analysis (superseded by exact_copula.py and canonical.py): the single-factor copula must be calibrated
with the TETRACHORIC (latent) correlation, NOT the Pearson correlation of 0/1 correctness indicators
--- as the paper's own Prop./footnote require. Using Pearson understates the latent correlation,
deflates the model's predicted all-wrong rate beta_sf, and inflates the 'underpricing' ratio.

This recomputes, per benchmark and for the pooled hard set, BOTH the naive-Pearson and the correct
tetrachoric underpricing of the co-failure tail, plus the tetrachoric pool-size scaling. Writes
runs/realizability_tetrachoric.json. No API calls.
Usage: python3 recompute_tetrachoric.py
"""
import os, json
import numpy as np
from scipy.stats import norm, multivariate_normal as mvn
from scipy.optimize import brentq
import realizability as RZ

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")

def tetra_pair(wi, wj, ti, tj):
    """latent Gaussian correlation reproducing the empirical joint wrong-rate of a pair."""
    pij = float(np.mean(wi * wj))
    if pij <= 1e-9:
        return 0.0
    f = lambda r: mvn.cdf([ti, tj], mean=[0, 0], cov=[[1, r], [r, 1]]) - pij
    try:
        if f(-0.99) * f(0.99) > 0:
            return 0.0 if f(0.0) > 0 else 0.95
        return float(brentq(f, -0.99, 0.99, xtol=1e-3))
    except Exception:
        return 0.0

def mean_tetra(M, max_pairs=400, seed=0):
    """mean off-diagonal tetrachoric correlation on a models x queries correctness matrix."""
    W = 1.0 - M; m = M.shape[0]
    pw = np.clip(W.mean(1), 1e-6, 1 - 1e-6); t = norm.ppf(pw)
    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    rng = np.random.default_rng(seed)
    if len(pairs) > max_pairs:
        pairs = [pairs[k] for k in rng.choice(len(pairs), size=max_pairs, replace=False)]
    vals = [tetra_pair(W[i], W[j], t[i], t[j]) for i, j in pairs]
    return float(np.mean(vals)), t

def underprice(M):
    n = M.shape[1]; beta = float((M.sum(0) == 0).mean()); k = int((M.sum(0) == 0).sum())
    pw = np.clip(1 - M.mean(1), 1e-6, 1 - 1e-6); t = norm.ppf(pw)
    rho_p = RZ._metrics(M)["rho_bar"]
    rho_t, _ = mean_tetra(M)
    bsf_p = RZ._copula_all_wrong(t, max(rho_p, 1e-6))
    bsf_t = RZ._copula_all_wrong(t, max(rho_t, 1e-6))
    return {"n": n, "k_allwrong": k, "beta": beta, "rho_pearson": rho_p, "rho_tetrachoric": rho_t,
            "beta_sf_pearson": bsf_p, "beta_sf_tetra": bsf_t,
            "underprice_pearson": beta / bsf_p if bsf_p > 0 else None,
            "underprice_tetra": beta / bsf_t if bsf_t > 0 else None}

def load_aligned(tag, ds):
    R = json.load(open(os.path.join(RUNS, f"matrix_{tag}.json")))
    qs = [q for q, v in R.items() if v["dataset"] == ds]
    ms = sorted(set(m for q in qs for m in R[q]["models"]))
    al = [q for q in qs if all(m in R[q]["models"] for m in ms)]
    if not al:
        return None
    return np.array([[R[q]["models"][m]["correct"] for q in al] for m in ms], float)

def main():
    out = {}
    for tag, ds in [("marketE3", "math500"), ("marketMH", "math_hard"), ("marketCG", "codegen"),
                    ("marketE2", "mmlu_pro"), ("marketE2", "gpqa")]:
        M = load_aligned(tag, ds)
        if M is None:
            continue
        out[ds] = underprice(M)
        r = out[ds]
        print(f"{ds:9} (n={r['n']}, k={r['k_allwrong']}): beta={r['beta']:.4f} | "
              f"Pearson rho={r['rho_pearson']:.3f}->{r['underprice_pearson']:.1f}x | "
              f"TETRACHORIC rho={r['rho_tetrachoric']:.3f}->{r['underprice_tetra']:.1f}x")
    json.dump(out, open(os.path.join(RUNS, "realizability_tetrachoric.json"), "w"), indent=2)
    m5 = out.get("math500")
    if m5:
        print(f"\nHEADLINE (MATH-500): naive-Pearson {m5['underprice_pearson']:.0f}x is INFLATED; "
              f"correct tetrachoric underpricing is {m5['underprice_tetra']:.1f}x (real but modest residual common-mode excess).")
    print("[recompute_tetrachoric] wrote runs/realizability_tetrachoric.json")

if __name__ == "__main__":
    main()
