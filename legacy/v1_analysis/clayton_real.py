"""First-release analysis (superseded by canonical.py): the full-Sigma Gaussian baseline is, fairly, "constructed
to lose" -- a Gaussian copula has ZERO lower tail dependence, so of course it cannot reach a common-mode atom.
The honest test is a copula WITH genuine lower-tail dependence, fit to the REAL matrix (not the exchangeable toy
of copula_dichotomy.py). We use an exchangeable CLAYTON copula (Archimedean, lower-tail-dependent
lambda_L = 2^{-1/theta} > 0), calibrate its single parameter theta to the matrix's MEAN pairwise both-wrong rate,
keep the real heterogeneous per-model error marginals, and Monte-Carlo the all-models-wrong rate.

If beta_clayton < beta_empirical even though Clayton HAS lower tail dependence calibrated to the same pairwise
co-failure, then no exchangeable pairwise-calibrated copula -- Gaussian OR tail-dependent -- can reach the tail:
the driver is a common-mode ATOM (Prop. poolbias), confirmed on the real 67-model data, not a Gaussian artifact.
Writes runs/clayton_real.json.  Usage: python3 clayton_real.py [n_mc]   (default 4e6 draws, chunked)
"""
import os, sys, json
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaincinv
import realizability as RZ
from recompute_tetrachoric import load_aligned

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")


def clayton_U(theta, u, E):
    """exchangeable Clayton uniforms via gamma frailty (Marshall-Olkin): V~Gamma(1/theta,1) drawn by inverse-CDF
    from a fixed uniform u, U_i=(1+E_i/V)^(-1/theta), E_i~Exp(1). Lower-tail dependent. Using fixed (u, E)
    across theta (common random numbers) makes the calibration objective smooth in theta."""
    V = gammaincinv(1.0 / theta, u)[:, None]
    return (1.0 + E / V) ** (-1.0 / theta)


def mean_pairwise_bothwrong(wrong):
    """EXACT mean over all m(m-1)/2 pairs of P(wrong_i & wrong_j), via per-draw wrong counts c: sum_pairs = C(c,2)."""
    m = wrong.shape[1]; c = wrong.sum(1).astype(float)
    return float(np.mean(c * (c - 1) / 2) / (m * (m - 1) / 2))


def main():
    # precision fix (2026-09-24): the earlier version calibrated theta on 60k fresh draws per brentq step with 200
    # sampled pairs (a noisy objective, xtol 1e-2) and used 2e5 draws for beta, so reruns scattered by ~5% on MATH-500
    # and ~10% on MATH-Hard. We now calibrate on EXACT all-pairs co-failure with common random numbers and report the
    # Monte-Carlo standard error of beta_clayton.
    n_mc = int(sys.argv[1]) if len(sys.argv) > 1 else 4000000
    n_cal, chunk = 200000, 200000
    out = {}
    for tag, ds in [("marketE3", "math500"), ("marketMH", "math_hard")]:
        M = load_aligned(tag, ds)
        if M is None:
            continue
        W = 1.0 - M; m, nq = W.shape
        pw = np.clip(W.mean(1), 1e-6, 1 - 1e-6)
        beta_emp = float((W.prod(0) == 1).mean()); k = int((W.prod(0) == 1).sum())
        q2_bar = mean_pairwise_bothwrong(W.T)
        rng = np.random.default_rng(12345)
        u = rng.random(n_cal); E = rng.exponential(1.0, size=(n_cal, m))
        f = lambda th: mean_pairwise_bothwrong(clayton_U(th, u, E) <= pw[None, :]) - q2_bar
        try:
            theta = float(brentq(f, 0.05, 30.0, xtol=1e-4, maxiter=100))
        except Exception:
            theta = None
        if theta is None:
            out[ds] = {"error": "theta calibration failed", "q2_bar": q2_bar}; continue
        lam_L = 2.0 ** (-1.0 / theta)
        hits = done = 0
        while done < n_mc:
            b = min(chunk, n_mc - done)
            U = clayton_U(theta, rng.random(b), rng.exponential(1.0, size=(b, m)))
            hits += int(np.all(U <= pw[None, :], axis=1).sum()); done += b
        beta_clayton = hits / n_mc
        se = float(np.sqrt(beta_clayton * (1 - beta_clayton) / n_mc))
        out[ds] = {
            "m": m, "n": nq, "k_allwrong": k, "beta_emp": beta_emp,
            "q2_bar_target": q2_bar, "clayton_theta": theta, "clayton_lambda_L": lam_L,
            "beta_clayton": beta_clayton, "beta_clayton_mc_se": se, "n_mc": n_mc,
            "ratio_emp_over_clayton": beta_emp / beta_clayton if beta_clayton > 0 else None,
            "ratio_mc95": [beta_emp / (beta_clayton + 1.96 * se), beta_emp / (beta_clayton - 1.96 * se)],
        }
        r = out[ds]
        print(f"{ds:9} k={k} beta_emp={beta_emp:.4f} | Clayton(theta={theta:.2f}, lambda_L={lam_L:.2f}) "
              f"calibrated to pairwise -> beta={beta_clayton:.4f} (MC se {se:.5f}) | residual {r['ratio_emp_over_clayton']:.3f}")
    json.dump(out, open(os.path.join(RUNS, "clayton_real.json"), "w"), indent=2)
    print("[clayton_real] wrote runs/clayton_real.json")
    m5 = out.get("math500")
    if m5 and m5.get("ratio_emp_over_clayton"):
        print(f"\nKEY: a Clayton copula WITH lower-tail dependence (lambda_L={m5['clayton_lambda_L']:.2f}), calibrated to the "
              f"same pairwise co-failure on the real matrix, still predicts beta={m5['beta_clayton']:.4f} vs empirical "
              f"{m5['beta_emp']:.4f} -> {m5['ratio_emp_over_clayton']:.2f}x residual. The tail is a common-mode ATOM, "
              f"not a Gaussian artifact.")


if __name__ == "__main__":
    main()
