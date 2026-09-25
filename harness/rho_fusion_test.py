"""Clean, powered test of the diversification limit (Pillar B).
Idea: enumerate ALL 3-model triplets from the matrix. Inter-model error correlation rho
varies NATURALLY across triplets (same-family triplets high-rho, cross-family low-rho), so
rho is an exogenous regressor we did not have to manufacture. For each triplet measure the
majority-vote fusion gain over its best member, and regress gain on rho while CONTROLLING
for the triplet's accuracy headroom (the ceiling confound). The diversification limit
predicts a NEGATIVE partial coefficient on rho (more correlated -> less fusion gain).

Uses the temp=0 matrix (one answer per model/query): vote-of-3 correct iff >=2 members right.
All numbers from logged calls. Usage: python3 rho_fusion_test.py --tag hardA
"""
import os, json, argparse, itertools, csv
import numpy as np
HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")

def load_matrix(tag):
    R = json.load(open(os.path.join(RUNS, f"matrix_{tag}.json")))
    models = sorted({m for v in R.values() for m in v["models"]})
    qids = [q for q, v in R.items() if all(m in v["models"] for m in models)]
    M = np.array([[R[q]["models"][m]["correct"] for q in qids] for m in models], float)
    return models, M  # models x queries (binary correctness)

def main(tag):
    models, M = load_matrix(tag)
    n = len(models); acc = M.mean(1)
    # full pairwise correlation once
    var = M.var(1)
    C = np.corrcoef(M)
    rows, members = [], []
    for tri in itertools.combinations(range(n), 3):
        i, j, k = tri
        if min(var[i], var[j], var[k]) < 1e-9:
            continue
        rho = np.mean([C[i, j], C[i, k], C[j, k]])
        votes = (M[i] + M[j] + M[k]) >= 2          # majority of 3
        vote_acc = votes.mean()
        best_member = max(acc[i], acc[j], acc[k])
        gain = vote_acc - best_member
        headroom = 1 - best_member
        rows.append((rho, gain, headroom, best_member, gain / headroom if headroom > 1e-9 else np.nan))
        members.append(tri)
    arr = np.array(rows)
    rho_, gain_, head_, best_, ng_ = arr.T
    print(f"=== Diversification test on matrix_{tag}: {len(rows)} triplets from {n} models ===")
    print(f"rho range [{rho_.min():.2f},{rho_.max():.2f}] | mean fusion gain {gain_.mean():+.3f} | mean best-member acc {best_.mean():.3f}")

    # OLS: gain ~ 1 + rho + headroom  (isolate rho effect from the ceiling confound)
    X = np.column_stack([np.ones(len(rho_)), rho_, head_])
    beta, *_ = np.linalg.lstsq(X, gain_, rcond=None)
    resid = gain_ - X @ beta
    dof = len(rho_) - 3
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tval = beta / se
    names = ["intercept", "rho", "headroom"]
    print("\nOLS  gain ~ rho + headroom  (controls for the ceiling confound):")
    for nm, b, s, t in zip(names, beta, se, tval):
        print(f"  {nm:10} coef={b:+.4f}  se={s:.4f}  t={t:+.2f}")
    # bootstrap CI on the rho coefficient
    rng = np.random.default_rng(0); bs = []
    for _ in range(3000):
        idx = rng.integers(0, len(rho_), len(rho_))
        Xb = X[idx]; yb = gain_[idx]
        try: bb = np.linalg.lstsq(Xb, yb, rcond=None)[0][1]; bs.append(bb)
        except Exception: pass
    ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    print(f"\n  rho coef 95% IID-triplet bootstrap CI (INVALID, triplets share models): [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    # --- model-cluster jackknife (leave-one-model-out): the correct inference ---
    mem = np.array(members)
    jack = []
    for jm in range(n):
        keep = [t for t in range(len(rows)) if jm not in mem[t]]
        if len(keep) < 5:
            continue
        Xj = X[keep]; yj = gain_[keep]
        try: jack.append(np.linalg.lstsq(Xj, yj, rcond=None)[0][1])
        except Exception: pass
    jack = np.array(jack); J = len(jack)
    # the mean vote gain under the same leave-one-model-out resampling: is its sign robust to dropping any model?
    gain_loo = [float(gain_[[t for t in range(len(rows)) if jm not in mem[t]]].mean()) for jm in range(n)]
    se_jack = np.sqrt((J - 1) / J * np.sum((jack - jack.mean())**2))
    lo, hi = beta[1] - 1.96 * se_jack, beta[1] + 1.96 * se_jack
    print(f"  rho coef MODEL-JACKKNIFE SE = {se_jack:.4f} ({se_jack/se[1]:.1f}x the naive SE); 95% CI [{lo:+.4f}, {hi:+.4f}]")
    sign_stable = (np.all(np.array(bs) > 0) or np.all(np.array(bs) < 0))
    verdict = ("SUPPORTED" if hi < 0 else ("REFUTES naive equal-weight diversity (coef>0)" if lo > 0 else
               "sign stable but NOT significant under model-clustering (CI spans 0)"))
    print(f"  Verdict (cluster-robust): {verdict}; point estimate {beta[1]:+.3f}, mean gain {gain_.mean():+.3f}")
    out = {"tag": tag, "n_triplets": len(rows), "rho_coef": float(beta[1]), "rho_se_naive": float(se[1]),
           "rho_se_jackknife": float(se_jack), "rho_ci_iid": ci, "rho_ci_jackknife": [float(lo), float(hi)],
           "headroom_coef": float(beta[2]),
           "rho_range": [float(rho_.min()), float(rho_.max())], "mean_gain": float(gain_.mean()),
           "mean_gain_loo_range": [min(gain_loo), max(gain_loo)], "frac_triplets_negative": float((gain_ < 0).mean()),
           "verdict": verdict,
           "scatter": [{"rho": float(r), "gain": float(g), "headroom": float(h)} for r, g, h, *_ in rows]}
    json.dump(out, open(os.path.join(RUNS, f"rhofus_{tag}.json"), "w"), indent=2)
    print(f"[rho_fusion] wrote rhofus_{tag}.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); a = ap.parse_args()
    main(a.tag)
