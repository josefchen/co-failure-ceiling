"""First-release analysis (superseded by canonical.py): address the main statistical objection --
"the 2.5x co-failure underpricing is just single-factor copula MISSPECIFICATION, not a real common-mode
atom." We answer it two ways, on the existing matrices, no API:

 (1) FULL-SIGMA GAUSSIAN BASELINE. Instead of the one-parameter single-factor copula (mean tetrachoric
     rho), fit the FULL pairwise tetrachoric correlation MATRIX Sigma (every pair calibrated to its joint
     wrong-rate), project to PSD, and Monte-Carlo the all-models-wrong probability under MVN(0, Sigma).
     This is the most flexible Gaussian copula the pairwise data can license. If empirical beta STILL
     exceeds beta_fullSigma, the residual is beyond ANY Gaussian (pairwise-calibrated) dependence -- the
     signature of a common-mode atom (Prop. poolbias), not a misspecified single factor.

 (2) COMPOSITION-BOOTSTRAPPED POOL-SIZE CURVE. The pool-size divergence (calib_compare) was a single curve
     over one random sub-pool per k -- it confounds size with WHICH models. We resample the model
     COMPOSITION (random k-subsets, many per k) and report the median + 5-95 band of the tetrachoric ratio
     beta/beta_sf vs k. A band that stays >1 and rises with k separates the size effect from composition.

Writes runs/residual_decomp.json.  Usage: python3 residual_decomp.py [n_mc] [n_boot]
"""
import os, sys, json
import numpy as np
from scipy.stats import norm
from scipy.linalg import cholesky, eigh
import realizability as RZ
from recompute_tetrachoric import tetra_pair, load_aligned

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")


def tetra_matrix(W):
    """full m x m tetrachoric correlation matrix calibrated pairwise to joint wrong-rates."""
    m = W.shape[0]
    pw = np.clip(W.mean(1), 1e-6, 1 - 1e-6); t = norm.ppf(pw)
    S = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            r = tetra_pair(W[i], W[j], t[i], t[j])
            S[i, j] = S[j, i] = r
    return S, t


def psd_project(S, eps=1e-4):
    """nearest PSD correlation-ish matrix: clip eigenvalues, renormalize diagonal to 1."""
    w, V = eigh(S)
    w = np.clip(w, eps, None)
    S2 = (V * w) @ V.T
    d = np.sqrt(np.diag(S2))
    S2 = S2 / np.outer(d, d)
    return S2


def mc_all_wrong(S, t, n_mc, seed=0):
    """Monte-Carlo P(all models wrong) under MVN(0,S): wrong_i iff latent_i < t_i."""
    rng = np.random.default_rng(seed)
    L = cholesky(S, lower=True)
    m = S.shape[0]
    hits = 0; B = 20000
    done = 0
    while done < n_mc:
        b = min(B, n_mc - done)
        Z = (L @ rng.standard_normal((m, b))).T   # (b, m)
        hits += int(np.all(Z < t[None, :], axis=1).sum())
        done += b
    return hits / n_mc


def composition_curve(W, ks, n_boot, max_pairs, seed=0):
    """median + 5-95 band of tetrachoric ratio beta/beta_sf over random k-model subsets."""
    rng = np.random.default_rng(seed)
    m = W.shape[0]
    out = {}
    for k in ks:
        if k > m:
            continue
        ratios = []
        for _ in range(n_boot):
            idx = rng.choice(m, size=k, replace=False)
            Wk = W[idx]
            beta = float((Wk.prod(0) == 1).mean())
            if beta <= 0:
                ratios.append(0.0); continue
            pw = np.clip(Wk.mean(1), 1e-6, 1 - 1e-6); t = norm.ppf(pw)
            # mean tetrachoric over sampled pairs
            pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
            if len(pairs) > max_pairs:
                pairs = [pairs[q] for q in rng.choice(len(pairs), size=max_pairs, replace=False)]
            rho = float(np.mean([tetra_pair(Wk[a], Wk[b], t[a], t[b]) for a, b in pairs])) if pairs else 0.0
            bsf = RZ._copula_all_wrong(t, max(rho, 1e-6))
            ratios.append(beta / bsf if bsf > 0 else 0.0)
        nz = [r for r in ratios if r > 0]
        out[k] = {"frac_pools_with_tail": len(nz) / len(ratios),
                  "ratio_median": float(np.median(nz)) if nz else 0.0,
                  "ratio_p05": float(np.percentile(nz, 5)) if nz else 0.0,
                  "ratio_p95": float(np.percentile(nz, 95)) if nz else 0.0}
    return out


def main():
    n_mc = int(sys.argv[1]) if len(sys.argv) > 1 else 400000
    n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    out = {}
    for tag, ds in [("marketE3", "math500"), ("marketMH", "math_hard"), ("marketCG", "codegen")]:
        M = load_aligned(tag, ds)
        if M is None:
            continue
        W = 1.0 - M
        beta = float((W.prod(0) == 1).mean()); k = int((W.prod(0) == 1).sum())
        pw = np.clip(W.mean(1), 1e-6, 1 - 1e-6); t = norm.ppf(pw)
        rho_bar = float(np.mean([tetra_pair(W[i], W[j], t[i], t[j])
                                 for i in range(W.shape[0]) for j in range(i + 1, W.shape[0])]))
        beta_sf = RZ._copula_all_wrong(t, max(rho_bar, 1e-6))
        beta_indep = RZ._copula_all_wrong(t, 1e-6)
        S, t2 = tetra_matrix(W)
        S = psd_project(S)
        beta_full = mc_all_wrong(S, t, n_mc)
        out[ds] = {
            "m": int(W.shape[0]), "n": int(W.shape[1]), "k_allwrong": k, "beta_emp": beta,
            "beta_independent": beta_indep, "beta_single_factor": beta_sf, "beta_full_sigma_mc": beta_full,
            "ratio_vs_single_factor": beta / beta_sf if beta_sf > 0 else None,
            "ratio_vs_full_sigma": beta / beta_full if beta_full > 0 else None,
            "rho_bar_tetra": rho_bar, "n_mc": n_mc,
        }
        print(f"{ds:9} k={k:2d} beta_emp={beta:.4f} | indep={beta_indep:.2e}  single-factor={beta_sf:.4f}  "
              f"FULL-Sigma(MC)={beta_full:.4f} | residual vs full-Sigma = {out[ds]['ratio_vs_full_sigma']}")
        if ds == "math500":
            ks = [2, 4, 8, 16, 24, 32, 48, 67]
            out["math500_composition_curve"] = composition_curve(W, ks, n_boot, 200)
            print("  composition-bootstrapped pool-size curve (math500):")
            for kk, v in out["math500_composition_curve"].items():
                print(f"    k={kk:2d}: ratio median={v['ratio_median']:.2f} [{v['ratio_p05']:.2f},{v['ratio_p95']:.2f}]  "
                      f"(pools with a tail: {v['frac_pools_with_tail']:.0%})")
    json.dump(out, open(os.path.join(RUNS, "residual_decomp.json"), "w"), indent=2)
    print("\n[residual_decomp] wrote runs/residual_decomp.json")
    m5 = out.get("math500")
    if m5 and m5["ratio_vs_full_sigma"]:
        print(f"HEADLINE: even the FULL pairwise-tetrachoric Gaussian copula predicts beta={m5['beta_full_sigma_mc']:.4f}, "
              f"vs empirical {m5['beta_emp']:.4f} -> residual {m5['ratio_vs_full_sigma']:.2f}x BEYOND any Gaussian "
              f"pairwise dependence: a common-mode atom, not single-factor misspecification.")


if __name__ == "__main__":
    main()
