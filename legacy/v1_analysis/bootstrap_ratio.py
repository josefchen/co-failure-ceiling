"""First-release analysis (superseded by canonical.py): the co-failure UNDERPRICING RATIO
(beta / beta_sf) was reported as a point estimate. Two honesty gaps:

  (1) DENOMINATOR EFFECT. MATH-Hard's ratio (7.6x) > MATH-500's (2.5x) even though MATH-Hard's
      beta (0.044) is LOWER than MATH-500's (0.052). The larger ratio is driven by a LOWER fitted
      tetrachoric rho (0.69 vs 0.78), which shrinks the single-factor denominator beta_sf -- not by
      a fatter co-failure tail. We report the rho-matched counterfactual: MATH-Hard's ratio if it
      had MATH-500's rho.

  (2) UNPROPAGATED UNCERTAINTY. The ratio inherits sampling error in BOTH beta (a ~k-event count,
      wide Clopper-Pearson) AND the fitted tetrachoric rho. We bootstrap over QUERIES (columns),
      jointly recomputing beta (numerator) and the mean tetrachoric rho -> beta_sf (denominator),
      and report the 5-95 percentile band on the ratio. No API calls.

Writes runs/ratio_uncertainty.json.  Usage: python3 bootstrap_ratio.py [B] [max_pairs]
"""
import os, json, sys
import numpy as np
from scipy.stats import norm
import realizability as RZ
from recompute_tetrachoric import tetra_pair, load_aligned

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")


def mean_tetra_cols(W, cols, max_pairs, rng):
    """mean off-diagonal tetrachoric rho on the wrong-matrix W restricted to a column (query) subset."""
    Wc = W[:, cols]; m = Wc.shape[0]
    pw = np.clip(Wc.mean(1), 1e-6, 1 - 1e-6); t = norm.ppf(pw)
    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    if len(pairs) > max_pairs:
        pairs = [pairs[k] for k in rng.choice(len(pairs), size=max_pairs, replace=False)]
    vals = [tetra_pair(Wc[i], Wc[j], t[i], t[j]) for i, j in pairs]
    return float(np.mean(vals)), t


def ratio_from(W, cols, max_pairs, rng):
    Wc = W[:, cols]
    beta = float((Wc.prod(0) == 1).mean())          # all models wrong on a query
    rho_t, t = mean_tetra_cols(W, cols, max_pairs, rng)
    bsf = RZ._copula_all_wrong(t, max(rho_t, 1e-6))
    return beta, rho_t, (beta / bsf if bsf > 0 else None), t


def analyze(tag, ds, B, max_pairs, seed=0):
    M = load_aligned(tag, ds)
    if M is None:
        return None
    W = 1.0 - M; n = W.shape[1]
    rng = np.random.default_rng(seed)
    # point estimate on the full sample
    base_beta, base_rho, base_ratio, base_t = ratio_from(W, np.arange(n), 10**9, rng)
    # bootstrap over queries
    ratios, betas, rhos = [], [], []
    for _ in range(B):
        cols = rng.integers(0, n, size=n)
        b, r, ra, _ = ratio_from(W, cols, max_pairs, rng)
        if ra is not None and np.isfinite(ra):
            ratios.append(ra); betas.append(b); rhos.append(r)
    ratios = np.array(ratios)
    return {"n": int(n), "k_allwrong": int((W.prod(0) == 1).sum()),
            "point_beta": base_beta, "point_rho_tet": base_rho, "point_ratio": base_ratio,
            "boot_B": len(ratios),
            "ratio_p05": float(np.percentile(ratios, 5)), "ratio_p50": float(np.percentile(ratios, 50)),
            "ratio_p95": float(np.percentile(ratios, 95)),
            "beta_p05": float(np.percentile(betas, 5)), "beta_p95": float(np.percentile(betas, 95)),
            "rho_p05": float(np.percentile(rhos, 5)), "rho_p95": float(np.percentile(rhos, 95)),
            "_t": base_t}


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    max_pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    out = {}
    res = {}
    for tag, ds in [("marketE3", "math500"), ("marketMH", "math_hard"), ("marketCG", "codegen")]:
        r = analyze(tag, ds, B, max_pairs)
        if r is None:
            continue
        res[ds] = r
        print(f"{ds:9} k={r['k_allwrong']:2d}  ratio point={r['point_ratio']:.2f}  "
              f"90% CI [{r['ratio_p05']:.2f}, {r['ratio_p95']:.2f}]  "
              f"(beta {r['beta_p05']:.3f}-{r['beta_p95']:.3f}, rho_tet {r['rho_p05']:.3f}-{r['rho_p95']:.3f})")
        out[ds] = {k: v for k, v in r.items() if k != "_t"}

    # rho-matched counterfactual: MATH-Hard's ratio if it had MATH-500's tetrachoric rho
    if "math500" in res and "math_hard" in res:
        rho_ref = res["math500"]["point_rho_tet"]
        t_h = res["math_hard"]["_t"]; beta_h = res["math_hard"]["point_beta"]
        bsf_matched = RZ._copula_all_wrong(t_h, rho_ref)
        ratio_matched = beta_h / bsf_matched if bsf_matched > 0 else None
        out["rho_matched_counterfactual"] = {
            "desc": "MATH-Hard underpricing ratio if its tetrachoric rho equalled MATH-500's",
            "rho_ref_from_math500": rho_ref, "math_hard_beta": beta_h,
            "math_hard_ratio_observed": res["math_hard"]["point_ratio"],
            "math_hard_ratio_rho_matched": ratio_matched}
        print(f"\nrho-matched: MATH-Hard at MATH-500's rho={rho_ref:.3f} -> ratio={ratio_matched:.2f}x "
              f"(vs observed {res['math_hard']['point_ratio']:.2f}x). "
              f"MATH-Hard beta={beta_h:.3f} is LOWER than MATH-500's {res['math500']['point_beta']:.3f}; "
              f"the higher observed ratio is a denominator (lower-rho) effect, not a fatter tail.")

    json.dump(out, open(os.path.join(RUNS, "ratio_uncertainty.json"), "w"), indent=2)
    print("\n[bootstrap_ratio] wrote runs/ratio_uncertainty.json")


if __name__ == "__main__":
    main()
