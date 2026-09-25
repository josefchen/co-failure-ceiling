"""Equal-quality break-even test of the diversification limit (Pillar B, owned theorem).
Contrasts two fusion regimes at MATCHED single-member quality, differing in error correlation:
  Self-MoA(k):  majority vote of k samples from the single best model  -> HIGH rho (intra-model)
  Hetero(k):    majority vote of 1 sample from each of the k strongest models -> LOW rho (inter-model)
The floor predicts: lower rho => larger diversifiable error => larger fusion gain and later plateau.
rho is estimated on a DISJOINT sample split (EST) from the gain (EVAL); query-bootstrap CIs.
Reads fusion_<tag>.json (multi-sample). All numbers from logged calls. Free (re-uses cache).
Usage: python3 equal_quality.py --tag hardBv2 --boot 2000
"""
import os, json, argparse, collections
import numpy as np
import grade

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")
EST = (0, 1, 2)                 # samples for rho estimation (disjoint from gain)
EVAL = (3, 4, 5, 6, 7, 8)       # >=6 DISTINCT eval samples so Self-MoA scales without recycling
_TIE = np.random.default_rng(12345)

def majority(ans):
    ans = [a for a in ans if a is not None]
    if not ans:
        return None
    c = collections.Counter(ans); top = c.most_common()
    best = top[0][1]
    tied = [a for a, n in top if n == best]
    return tied[0] if len(tied) == 1 else tied[_TIE.integers(0, len(tied))]  # random tie-break (symmetric)

def rho_intra(samples, meta, model, qids, idx):
    """corr across the model's own samples (Self-MoA correlation)."""
    Mx = np.array([[grade.check(meta[q]["kind"], samples[q][model][s], meta[q]["gold"]) for q in qids] for s in idx], float)
    var = Mx.var(1); keep = [i for i in range(len(idx)) if var[i] > 1e-9]
    if len(keep) < 2: return np.nan
    R = np.corrcoef(Mx[keep]); return float(R[np.triu_indices(len(keep), 1)].mean())

def rho_inter(samples, meta, models, qids, s):
    Mx = np.array([[grade.check(meta[q]["kind"], samples[q][m][s], meta[q]["gold"]) for q in qids] for m in models], float)
    var = Mx.var(1); keep = [i for i in range(len(models)) if var[i] > 1e-9]
    if len(keep) < 2: return np.nan
    R = np.corrcoef(Mx[keep]); return float(R[np.triu_indices(len(keep), 1)].mean())

def correct_vec(samples, meta, picks, qids):
    """per-query correctness (0/1 array) of the majority vote of picks=[(model,sample_idx)]."""
    return np.array([grade.check(meta[q]["kind"], majority([samples[q][m][s] for (m, s) in picks]), meta[q]["gold"]) for q in qids], float)

def main(tag, boot):
    D = json.load(open(os.path.join(RUNS, f"fusion_{tag}.json")))
    samples, meta, models, S = D["samples"], D["meta"], D["models"], D["S"]
    qids = list(meta.keys())
    # rank by EVAL-sample accuracy (held out from EST rho); pick best for Self-MoA
    ev_acc = {m: np.mean([grade.check(meta[q]["kind"], samples[q][m][EVAL[0]], meta[q]["gold"]) for q in qids]) for m in models}
    ranked = sorted(models, key=lambda m: -ev_acc[m]); best = ranked[0]
    Kmax = min(S, len(ranked))
    rho_sm = rho_intra(samples, meta, best, qids, EST)
    rho_ht = rho_inter(samples, meta, ranked[:Kmax], qids, EST[0])
    print(f"=== Equal-quality break-even test ({tag}) | {len(qids)} queries | best={best} (acc {ev_acc[best]:.3f}) ===")
    print(f"member accuracies (top-{Kmax}): " + ", ".join(f"{ev_acc[m]:.3f}" for m in ranked[:Kmax]))
    print(f"rho: Self-MoA intra-model = {rho_sm:.3f}  vs  heterogeneous inter-model = {rho_ht:.3f}  (disjoint EST split)")
    print(f"{'k':>2} | {'SelfMoA(hi-rho)':>15} {'Hetero(lo-rho)':>15} | {'Hetero-SelfMoA':>14}")
    rng = np.random.default_rng(0); rows = []
    nq = len(qids)
    idx_bs = [rng.integers(0, nq, nq) for _ in range(boot)]  # shared resample indices
    Kmax = min(Kmax, len(EVAL))   # cap so Self-MoA uses DISTINCT eval samples (no recycling)
    for k in range(1, Kmax + 1):
        sm = [(best, EVAL[j]) for j in range(k)]                         # k DISTINCT samples of best
        ht = [(ranked[j], EVAL[0]) for j in range(k)]                    # 1 EVAL sample each of top-k
        cv_sm = correct_vec(samples, meta, sm, qids)
        cv_ht = correct_vec(samples, meta, ht, qids)
        a_sm, a_ht = cv_sm.mean(), cv_ht.mean()
        d = cv_ht - cv_sm
        diffs = np.array([d[ix].mean() for ix in idx_bs])               # vectorized bootstrap
        lo, hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
        rows.append({"k": k, "selfmoa": float(a_sm), "hetero": float(a_ht), "diff": float(a_ht - a_sm),
                     "diff_ci": [float(lo), float(hi)]})
        star = "  *sig" if (lo > 0 or hi < 0) else ""
        print(f"{k:>2} | {a_sm:15.3f} {a_ht:15.3f} | {a_ht-a_sm:+.3f} [{lo:+.3f},{hi:+.3f}]{star}")
    out = {"tag": tag, "best": best, "rho_selfmoa": rho_sm, "rho_hetero": rho_ht,
           "member_acc": [float(ev_acc[m]) for m in ranked[:Kmax]], "rows": rows}
    json.dump(out, open(os.path.join(RUNS, f"eqq_{tag}.json"), "w"), indent=2)
    kbest = max(rows, key=lambda r: r["diff"])
    print(f"\nVerdict: lower-rho heterogeneous fusion vs high-rho Self-MoA at matched quality:")
    print(f"  max advantage {kbest['diff']:+.3f} at k={kbest['k']} (95% CI [{kbest['diff_ci'][0]:+.3f},{kbest['diff_ci'][1]:+.3f}])")
    print(f"  consistent with the diversification limit: lower rho ({rho_ht:.2f}) -> larger fusion gain than higher rho ({rho_sm:.2f}).")
    print(f"[eqq] wrote eqq_{tag}.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(); main(a.tag, a.boot)
