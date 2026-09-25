"""Rigorous statistics for the diversification-limit test (Pillar B), controlling the
benchmark-ceiling confound by DIFFICULTY BUCKETING and reporting BOOTSTRAP CIs.

Theory: diversifiable error fraction is (1 - rho). Fusion gain should scale with the
available headroom (1 - best_single_acc) times (1 - rho). So the ceiling-free test is:
  normalized_gain := fusion_gain / headroom    vs    (1 - rho)
within difficulty-matched buckets. Higher (1-rho) -> larger normalized gain.

Usage: python3 analyze_stats.py --tag hardB --bins 4 --boot 2000
Reads runs/fusion_<tag>.json (sampled correctness). All numbers from logged calls.
"""
import os, json, argparse, collections
import numpy as np
import grade

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")

def majority(ans):
    ans = [a for a in ans if a is not None]
    return collections.Counter(ans).most_common(1)[0][0] if ans else None

def load(tag):
    D = json.load(open(os.path.join(RUNS, f"fusion_{tag}.json")))
    return D["samples"], D["meta"], D["models"]

def rho_of(corr_mat):
    """mean off-diagonal Pearson corr of binary correctness rows (models x queries)."""
    var = corr_mat.var(axis=1); keep = [i for i in range(corr_mat.shape[0]) if var[i] > 1e-9]
    if len(keep) < 2:
        return np.nan
    R = np.corrcoef(corr_mat[keep])
    return float(R[np.triu_indices(len(keep), 1)].mean())

EST = (0, 1, 2)    # samples used to estimate difficulty + rho
EVAL = (3, 4, 5)   # disjoint samples used to measure fusion gain (breaks collider leakage)

def _corr_est(samples, meta, models, qids):
    """rho on per-query mean correctness over EST samples (disjoint from gain)."""
    Me = np.array([[np.mean([grade.check(meta[q]["kind"], samples[q][m][s], meta[q]["gold"]) for s in EST])
                    for q in qids] for m in models], dtype=float)
    return rho_of(Me), Me.mean(axis=1)

def bucket_stats(samples, meta, models, qids):
    rho, acc_est = _corr_est(samples, meta, models, qids)
    # gain measured on EVAL samples only (disjoint from EST used for rho/difficulty)
    def acc_eval(m):
        return np.mean([np.mean([grade.check(meta[q]["kind"], samples[q][m][s], meta[q]["gold"]) for s in EVAL])
                        for q in qids])
    acc_e = np.array([acc_eval(m) for m in models])
    best_single = acc_e.max()
    order = list(np.argsort(-acc_e))
    best_k = best_single
    for k in range(2, len(models) + 1):
        picks = order[:k]
        # one held-out eval sample (EVAL[0]) per model for the vote
        voted = np.mean([grade.check(meta[q]["kind"],
                          majority([samples[q][models[i]][EVAL[0]] for i in picks]), meta[q]["gold"])
                         for q in qids])
        best_k = max(best_k, voted)
    gain = best_k - best_single
    headroom = 1 - best_single
    norm_gain = gain / headroom if headroom > 1e-9 else np.nan
    return {"n": len(qids), "rho": rho, "best_single": best_single, "best_k": best_k,
            "gain": gain, "headroom": headroom, "norm_gain": norm_gain}

def main(tag, bins, boot, seed_list=(0,)):
    samples, meta, models = load(tag)
    qids = list(meta.keys())
    # difficulty = mean sample-0 correctness across models
    diff = {}
    for q in qids:
        cs = [np.mean([grade.check(meta[q]["kind"], samples[q][m][s], meta[q]["gold"]) for s in EST]) for m in models]
        diff[q] = np.mean(cs)  # difficulty from EST samples (disjoint from EVAL gain)
    # quantile difficulty buckets (exclude fully-saturated d==1 bucket from the test)
    ds = np.array([diff[q] for q in qids])
    edges = np.quantile(ds, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-9
    print(f"=== Pillar-B ceiling-free test: tag={tag} | {len(qids)} queries | {bins} difficulty buckets ===")
    print(f"{'bucket(difficulty)':>20} {'n':>4} {'rho':>6} {'1-rho':>6} {'best1':>6} {'gain':>6} {'norm_gain':>9}")
    rows = []
    rng = np.random.default_rng(0)
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        bq = [q for q in qids if lo <= diff[q] < hi]
        if len(bq) < 8:
            continue
        st = bucket_stats(samples, meta, models, bq)
        # bootstrap CI on norm_gain and rho
        ng_bs, rho_bs = [], []
        for _ in range(boot):
            samp = list(rng.choice(bq, size=len(bq), replace=True))
            s = bucket_stats(samples, meta, models, samp)
            if not np.isnan(s["norm_gain"]): ng_bs.append(s["norm_gain"])
            if not np.isnan(s["rho"]): rho_bs.append(s["rho"])
        st["norm_gain_ci"] = [float(np.percentile(ng_bs, 2.5)), float(np.percentile(ng_bs, 97.5))] if ng_bs else None
        st["rho_ci"] = [float(np.percentile(rho_bs, 2.5)), float(np.percentile(rho_bs, 97.5))] if rho_bs else None
        st["diff_lo"], st["diff_hi"] = float(lo), float(hi)
        rows.append(st)
        ng = st["norm_gain"]; ci = st["norm_gain_ci"]
        cis = f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci else "n/a"
        print(f"   [{lo:.2f},{hi:.2f})        {st['n']:>4} {st['rho']:6.3f} {1-st['rho']:6.3f} "
              f"{st['best_single']:6.3f} {st['gain']:+.3f} {ng:9.3f} {cis}")
    # association test: does norm_gain rise with (1-rho)?
    valid = [r for r in rows if not np.isnan(r["norm_gain"]) and not np.isnan(r["rho"])]
    if len(valid) >= 3:
        x = np.array([1 - r["rho"] for r in valid]); y = np.array([r["norm_gain"] for r in valid])
        r_pearson = float(np.corrcoef(x, y)[0, 1])
        # bootstrap CI on the across-bucket correlation
        corrs = []
        for _ in range(boot):
            idx = rng.choice(len(valid), len(valid), replace=True)
            if np.std(x[idx]) > 1e-9 and np.std(y[idx]) > 1e-9:
                corrs.append(np.corrcoef(x[idx], y[idx])[0, 1])
        ci = [float(np.percentile(corrs, 2.5)), float(np.percentile(corrs, 97.5))] if corrs else None
        print(f"\nAcross-bucket corr( (1-rho), normalized_gain ) = {r_pearson:.3f}  95%CI {('[%.2f,%.2f]'%(ci[0],ci[1])) if ci else 'n/a'}")
        print("Prediction (diversification limit): POSITIVE corr (more diversifiable error -> bigger fusion gain).")
        out = {"tag": tag, "buckets": rows, "corr_1mrho_vs_normgain": r_pearson, "corr_ci": ci}
    else:
        out = {"tag": tag, "buckets": rows, "corr_1mrho_vs_normgain": None, "note": "too few non-saturated buckets"}
    json.dump(out, open(os.path.join(RUNS, f"bstats_{tag}.json"), "w"), indent=2)
    print(f"[stats] wrote bstats_{tag}.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    main(a.tag, a.bins, a.boot)
