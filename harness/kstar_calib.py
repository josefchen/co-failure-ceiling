"""k*(rho) calibration / multi-rho test of the diversification limit, on a MATCHED-QUALITY pool.
From a fusion run's samples (e.g. eqq2: 6 accuracy-matched models), enumerate all sub-bands; rho
varies by family composition (same-family high, cross-family low). For each band measure the
majority-vote gain over the best member, controlling for member accuracy, and regress gain on rho
with MODEL-CLUSTERED (leave-one-model-out jackknife) inference. The diversification limit predicts
a NEGATIVE rho coefficient at matched quality (lower correlation -> larger diversifiable gain).
Also checks the binary majority-vote floor 1 - Phi(-Phi^{-1}(1-alpha)/sqrt(rho)) against observed.
rho estimated on EST samples (disjoint from the EVAL gain). All numbers from logged calls.
Usage: python3 kstar_calib.py --tag eqq2
"""
import os, json, argparse, itertools, collections
import numpy as np
from scipy.stats import norm
import grade

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")
EST = (0, 1, 2); EVAL = (3, 4, 5, 6, 7, 8)
_TIE = np.random.default_rng(7)

def majority(ans):
    ans = [a for a in ans if a is not None]
    if not ans: return None
    c = collections.Counter(ans).most_common(); best = c[0][1]
    tied = [a for a, n in c if n == best]
    return tied[0] if len(tied) == 1 else tied[_TIE.integers(0, len(tied))]

def main(tag):
    D = json.load(open(os.path.join(RUNS, f"fusion_{tag}.json")))
    samples, meta, models = D["samples"], D["meta"], D["models"]
    qids = list(meta.keys()); m = len(models)
    # EST continuous per-model correctness (for rho) and EVAL single-sample correctness (for gain)
    est = np.array([[np.mean([grade.check(meta[q]["kind"], samples[q][mod][s], meta[q]["gold"]) for s in EST])
                     for q in qids] for mod in models])               # m x Q
    ev0 = np.array([[grade.check(meta[q]["kind"], samples[q][mod][EVAL[0]], meta[q]["gold"]) for q in qids]
                    for mod in models], float)                         # m x Q (one held-out eval sample)
    ev_acc = ev0.mean(1)
    Cest = np.corrcoef(est) if est.shape[0] > 1 else np.array([[1.0]])
    rows, members = [], []
    for r in range(2, m + 1):
        for band in itertools.combinations(range(m), r):
            sub = list(band)
            pair_rho = [Cest[i, j] for a, i in enumerate(sub) for j in sub[a+1:]]
            rho = float(np.mean(pair_rho))
            best_member = ev_acc[sub].max()
            voted = np.array([grade.check(meta[qids[t]]["kind"],
                              majority([samples[qids[t]][models[i]][EVAL[0]] for i in sub]),
                              meta[qids[t]]["gold"]) for t in range(len(qids))], float)
            gain = voted.mean() - best_member
            head = 1 - best_member
            rows.append((rho, gain, best_member, gain/head if head > 1e-9 else np.nan)); members.append(sub)
    arr = np.array(rows); rho_, gain_, bm_, ng_ = arr.T
    # OLS gain ~ rho + best_member
    X = np.column_stack([np.ones(len(rho_)), rho_, bm_]); beta, *_ = np.linalg.lstsq(X, gain_, rcond=None)
    # model-cluster jackknife on the rho coefficient
    jack = []
    for jm in range(m):
        keep = [t for t in range(len(rows)) if jm not in members[t]]
        if len(keep) > 4:
            b = np.linalg.lstsq(X[keep], gain_[keep], rcond=None)[0]; jack.append(b[1])
    jack = np.array(jack); J = len(jack)
    se = np.sqrt((J-1)/J*np.sum((jack-jack.mean())**2)); lo, hi = beta[1]-1.96*se, beta[1]+1.96*se
    print(f"=== k*(rho) multi-band test on {tag}: {len(rows)} sub-bands of {m} matched models ===")
    print(f"rho range [{rho_.min():.2f},{rho_.max():.2f}] | mean gain {gain_.mean():+.3f} | mean best-member {bm_.mean():.3f}")
    print(f"OLS gain ~ rho + best_member:  rho coef = {beta[1]:+.4f}  (model-jackknife SE {se:.4f}, 95% CI [{lo:+.4f},{hi:+.4f}])")
    verdict = ("SUPPORTS diversification limit (rho coef < 0, CI excludes 0)" if hi < 0 else
               ("CONTRADICTS (rho coef > 0)" if lo > 0 else "sign negative but CI spans 0" if beta[1] < 0 else "inconclusive"))
    print(f"  -> {verdict}")
    # binary floor calibration: full band predicted vs observed majority-vote accuracy
    alpha = 1 - ev_acc.mean(); rho_full = float(np.mean([Cest[i, j] for i in range(m) for j in range(i+1, m)]))
    floor_err = float(norm.cdf(-norm.ppf(1-alpha)/np.sqrt(max(rho_full, 1e-6))))
    full_vote = np.mean([grade.check(meta[qids[t]]["kind"],
                         majority([samples[qids[t]][models[i]][EVAL[0]] for i in range(m)]), meta[qids[t]]["gold"])
                         for t in range(len(qids))])
    print(f"\nbinary majority-vote floor check (full {m}-model band): mean alpha={alpha:.3f}, rho={rho_full:.3f}")
    print(f"  predicted floor accuracy 1-Phi(-Phi^-1(1-alpha)/sqrt(rho)) = {1-floor_err:.3f}   observed {m}-vote = {full_vote:.3f}")
    json.dump({"tag": tag, "n_bands": len(rows), "rho_coef": float(beta[1]), "rho_coef_ci": [float(lo), float(hi)],
               "verdict": verdict, "rho_range": [float(rho_.min()), float(rho_.max())],
               "floor_predicted": float(1-floor_err), "floor_observed": float(full_vote),
               "alpha": float(alpha), "rho_full": rho_full,
               "scatter": [{"rho": float(r), "gain": float(g), "best_member": float(b)} for r, g, b, _ in rows]},
              open(os.path.join(RUNS, f"kstar_{tag}.json"), "w"), indent=2)
    print(f"[kstar] wrote kstar_{tag}.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); a = ap.parse_args(); main(a.tag)
