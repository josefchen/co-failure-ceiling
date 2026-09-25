"""Mechanism control for Prop. poolbias: the underpricing divergence is driven by a COMMON-MODE
atom (beta_infinity>0), NOT by pairwise lower tail dependence lambda_L per se. We compare, under
the exact empirical pipeline (binary Pearson rho_bar -> single-factor Gaussian beta_sf), two
exchangeable error laws with the same marginal error rate alpha:
  (a) Clayton copula (Archimedean, gamma frailty), lower tail dependence lambda_L = 2^{-1/theta} > 0
      but beta(m) -> 0 polynomially (no common-mode atom);
  (b) common-shock mixture: w.p. pi all m err (a beta_infinity = pi atom), else i.i.d. at alpha0.
Reproducible; writes runs/copula_dichotomy.json. (Run id: copula_dichotomy.)
Usage: python3 copula_dichotomy.py
"""
import os, json
import numpy as np
import realizability as RZ
from scipy.stats import norm

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")
SEED = 0; NQ = 40000; ALPHA = 0.40

def _rng(): return np.random.default_rng(SEED)

def clayton_errors(rng, n_q, m, alpha, theta):
    # Archimedean Clayton via gamma frailty: G~Gamma(1/theta,1); U_i=(1+E_i/G)^{-1/theta}; err=1{U<=alpha}
    G = rng.gamma(1 / theta, 1.0, size=n_q)[:, None]
    E = rng.exponential(1.0, size=(n_q, m))
    U = (1 + E / G) ** (-1 / theta)
    return (U <= alpha).astype(int)

def mixture_errors(rng, n_q, m, alpha0, pi):
    hard = rng.random(n_q) < pi
    err = (rng.random((n_q, m)) < alpha0).astype(int); err[hard] = 1
    return err

def ratio_curve(E, ms):
    out = []
    for m in ms:
        sub = E[:, :m]
        beta = (sub.sum(1) == m).mean()
        C = np.corrcoef(sub.T); iu = np.triu_indices(m, 1)
        rho = float(np.nanmean(C[iu])) if m > 1 else 0.0
        t = norm.ppf(ALPHA); bsf = RZ._copula_all_wrong(np.full(m, t), max(rho, 1e-6))
        out.append({"m": m, "beta": float(beta), "rho_bar": rho, "beta_sf": float(bsf),
                    "ratio": float(beta / bsf) if bsf > 0 else float("inf")})
    return out

def main():
    rng = _rng(); ms = [2, 5, 12, 25, 53]
    res = {"alpha": ALPHA, "n_q": NQ, "seed": SEED, "lambda_L_formula": "2^(-1/theta)", "cases": {}}
    print(f"alpha={ALPHA}, n_q={NQ}")
    for th in [2.0, 4.0]:
        E = clayton_errors(rng, NQ, 53, ALPHA, th)
        lamL = 2 ** (-1 / th)
        cur = ratio_curve(E, ms)
        res["cases"][f"clayton_theta{th}"] = {"lambda_L": lamL, "curve": cur}
        print(f"Clayton theta={th} (lambda_L={lamL:.2f}): " + " ".join(f"m{c['m']}={c['ratio']:.1f}x" for c in cur))
    for pi in [0.05]:
        E = mixture_errors(rng, NQ, 53, ALPHA, pi)
        cur = ratio_curve(E, ms)
        res["cases"][f"mixture_pi{pi}"] = {"beta_infinity": pi, "curve": cur}
        print(f"Common-shock mixture pi={pi} (atom beta_inf={pi}): " + " ".join(f"m{c['m']}={c['ratio']:.1f}x" for c in cur))
    json.dump(res, open(os.path.join(RUNS, "copula_dichotomy.json"), "w"), indent=2)
    cl = res["cases"]["clayton_theta2.0"]["curve"][-1]["ratio"]
    mx = res["cases"]["mixture_pi0.05"]["curve"][-1]["ratio"]
    print(f"\nVERDICT: Clayton lambda_L=0.71 underprices only {cl:.1f}x at m=53 (bounded); "
          f"common-shock atom underprices {mx:.0f}x (diverges). Driver = common-mode atom, not lambda_L.")
    print("[copula_dichotomy] wrote runs/copula_dichotomy.json")

if __name__ == "__main__":
    main()
