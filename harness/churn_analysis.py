"""Pillar D, made empirical: the realized option value of broad model access over a real
2-year frontier timeline. Reads matrix_<tag>.json (per-model accuracy) + churn_registry.csv
(release dates + prices). Measures, on the actual release sequence:
  - the frontier-accuracy trajectory (capability churn);
  - the realized broad-access advantage of ROUTE (always adopt current best) vs COMMIT
    (stay on the best model available at the start) -> the option value the theory predicts;
  - the build-vs-route premium delta* (per-period advantage a self-host must beat);
  - frontier $/correct churn (cost depreciation);
  - the comparative static: per-release captured gap vs capability dispersion, controlling
    for level (tests the additive/level-robust prediction).
Observational (single realized path) -- reported as such. No new inference; all from logs.
Usage: python3 churn_analysis.py --tag churnD --dataset mmlu_pro
"""
import os, json, csv, argparse
import numpy as np

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")

def load(tag, dataset):
    R = json.load(open(os.path.join(RUNS, f"matrix_{tag}.json")))
    reg = {}
    for row in csv.DictReader(r for r in open(os.path.join(HERE, "..", "churn_registry.csv")) if not r.startswith("#")):
        reg[row["model_id"]] = {"date": row["created_date"], "pin": float(row["price_in_per_mtok_usd"]),
                                "pout": float(row["price_out_per_mtok_usd"])}
    models = sorted(reg.keys())
    # per-model accuracy + mean $/query on the chosen dataset
    acc, cpq = {}, {}
    for m in models:
        qs = [q for q, v in R.items() if v["dataset"] == dataset and m in v["models"]]
        if not qs:
            continue
        acc[m] = np.mean([R[q]["models"][m]["correct"] for q in qs])
        cpq[m] = np.mean([R[q]["models"][m]["cost"] for q in qs])
    return reg, acc, cpq, dataset

def main(tag, dataset):
    reg, acc, cpq, ds = load(tag, dataset)
    rows = sorted([(reg[m]["date"], m, acc[m], cpq[m]) for m in acc], key=lambda r: r[0])
    dates = [r[0] for r in rows]
    print(f"=== Pillar D longitudinal ({tag}, {ds}) | {len(rows)} models, {dates[0]} -> {dates[-1]} ===")
    print(f"{'date':12} {'model':40} {'acc':>5} {'frontier':>8} {'gap':>6}")
    frontier = -1; capt = []; traj = []
    best_cpc_so_far = 1e9; cpc_curve = []
    for d, m, a, c in rows:
        prev = frontier
        gap = a - prev if prev >= 0 else 0.0
        capt.append({"date": d, "model": m, "acc": a, "prev_frontier": max(prev, 0), "gap": gap})
        frontier = max(frontier, a)
        traj.append((d, frontier))
        cpc = c / a if a > 0 else 1e9
        best_cpc_so_far = min(best_cpc_so_far, cpc)
        cpc_curve.append((d, best_cpc_so_far))
        print(f"{d:12} {m:40} {a:5.3f} {frontier:8.3f} {gap:+6.3f}")
    # ROUTE vs COMMIT: commit to best available in the first 90 days
    commit_acc = max(a for d, m, a, c in rows if d <= _add_days(dates[0], 90))
    route_acc = frontier
    adv = route_acc - commit_acc
    # realized option value (normalized v=1): sum of captured positive gaps
    opt_value = sum(x["gap"] for x in capt if x["gap"] > 0)
    # cost churn: frontier $/correct first vs last
    cpc_first, cpc_last = cpc_curve[0][1], cpc_curve[-1][1]
    # comparative static: captured positive gap vs available-pool dispersion, controlling for level
    disp, lev, gp = [], [], []
    seen = []
    for d, m, a, c in rows:
        seen.append(a)
        if len(seen) >= 2:
            disp.append(float(np.std(seen[:-1])))  # dispersion of pool before this release
            lev.append(float(max(seen[:-1])))       # level = prev frontier
            gp.append(max(a - max(seen[:-1]), 0.0)) # captured gap
    disp, lev, gp = map(np.array, (disp, lev, gp))
    out = {"tag": tag, "dataset": ds, "n_models": len(rows), "span": [dates[0], dates[-1]],
           "frontier_first": float(traj[0][1]), "frontier_last": float(route_acc),
           "commit_acc_90d": float(commit_acc), "broad_access_advantage": float(adv),
           "realized_option_value_sum_pos_gaps": float(opt_value),
           "frontier_cpc_first": float(cpc_first), "frontier_cpc_last": float(cpc_last),
           "cpc_drop_factor": float(cpc_first / cpc_last) if cpc_last > 0 else None,
           "trajectory": [[d, float(f)] for d, f in traj],
           "capture": capt}
    print(f"\nROUTE (always adopt best) final acc = {route_acc:.3f}")
    print(f"COMMIT (best in first 90d, '{[m for d,m,a,c in rows if d<=_add_days(dates[0],90) and a==commit_acc][0]}') = {commit_acc:.3f}")
    print(f"REALIZED BROAD-ACCESS ADVANTAGE = {adv:+.3f}  (the option value of breadth, measured)")
    print(f"realized option value (sum of captured positive gaps) = {opt_value:.3f}")
    print(f"frontier $/correct churn: {cpc_first:.5f} -> {cpc_last:.5f}  ({out['cpc_drop_factor']:.1f}x cheaper)")
    if len(gp) >= 4 and disp.std() > 1e-9:
        # partial assoc of captured gap with dispersion controlling for level
        X = np.column_stack([np.ones(len(gp)), disp, lev]); beta, *_ = np.linalg.lstsq(X, gp, rcond=None)
        r_disp = float(np.corrcoef(disp, gp)[0, 1])
        print(f"\ncomparative static (observational): corr(capture gap, pool dispersion)={r_disp:+.2f}; "
              f"OLS capture~disp+level: b_disp={beta[1]:+.3f}, b_level={beta[2]:+.3f}")
        print("  (Pillar D predicts capture loads on dispersion, ~independent of level; single path, not causal.)")
        out["b_disp"] = float(beta[1]); out["b_level"] = float(beta[2]); out["corr_disp_gap"] = r_disp
    json.dump(out, open(os.path.join(RUNS, f"churn_{tag}_{ds}.json"), "w"), indent=2)
    print(f"[churn] wrote churn_{tag}_{ds}.json")

def _add_days(datestr, days):
    import datetime
    d = datetime.date.fromisoformat(datestr) + datetime.timedelta(days=days)
    return d.isoformat()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--dataset", default="mmlu_pro")
    a = ap.parse_args(); main(a.tag, a.dataset)
