"""Held-out evaluation of the cascade (Pillar C): a HELD-OUT threshold fold, as for the routers, so the
cascade's advantage over random mixing is not an in-sample artifact. The cascade defers to H when the
cheap model L's self-confidence < tau; Prop. cascade says it beats random-mixing-at-matched-budget iff
the confidence verifier has AUC>1/2 for ranking L-correct vs L-wrong. We 5-fold cross-validate: pick tau*
on the train folds, evaluate accuracy/cost AND the advantage over random mixing at the SAME escalation
rate on the held-out fold. No API. (The optimal two-model deferral upper bound of Jitkrittum needs H's
confidence, which the matrix does not log; we flag that as the remaining cascade gap.)
Usage: python3 cascade_heldout.py [tag]
"""
import os, sys, json
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")


def cascade_acc_cost(rec, tau):
    """accept L if conf>=tau else escalate to H. returns (accuracy, mean cost, escalation rate)."""
    acc = cost = esc = 0.0
    for r in rec:
        if r["conf"] >= tau:
            acc += r["Lcorr"]; cost += r["Lcost"]
        else:
            acc += r["Hcorr"]; cost += r["Lcost"] + r["Hcost"]; esc += 1
    n = len(rec)
    return acc / n, cost / n, esc / n


def random_mix_acc(rec, e):
    """expected accuracy escalating a RANDOM fraction e to H (the matched-budget null)."""
    L = np.mean([r["Lcorr"] for r in rec]); H = np.mean([r["Hcorr"] for r in rec])
    return (1 - e) * L + e * H


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "stageC2v3"
    d = json.load(open(os.path.join(RUNS, f"cascade_{tag}.json")))
    rec = d["rec"]; n = len(rec)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 5)
    taus = np.unique([r["conf"] for r in rec])
    rows = []
    for f in range(5):
        te = set(folds[f].tolist()); tr = [rec[i] for i in range(n) if i not in te]; tst = [rec[i] for i in te]
        # tau* maximizing train accuracy (cascade objective); tie-break toward less escalation
        best = max(taus, key=lambda t: (cascade_acc_cost(tr, t)[0], -cascade_acc_cost(tr, t)[2]))
        a, c, e = cascade_acc_cost(tst, best)
        rm = random_mix_acc(tst, e)
        # held-out AUC of confidence for ranking L-correct vs L-wrong
        y = [r["Lcorr"] for r in tst]; s = [r["conf"] for r in tst]
        auc = roc_auc_score(y, s) if 0 < sum(y) < len(y) else float("nan")
        rows.append({"fold": f, "tau": float(best), "test_acc": a, "test_cost": c, "esc": e,
                     "random_mix_acc": rm, "advantage_vs_random_mix": a - rm, "test_auc_conf": auc})
        print(f"fold {f}: tau*={best:.3f} | test acc={a:.3f} cost={c:.4f} esc={e:.2f} | "
              f"random-mix@{e:.2f}={rm:.3f} -> advantage {a-rm:+.3f} | held-out AUC={auc:.3f}")
    adv = np.array([r["advantage_vs_random_mix"] for r in rows])
    aucs = np.array([r["test_auc_conf"] for r in rows])
    L = np.mean([r["Lcorr"] for r in rec]); H = np.mean([r["Hcorr"] for r in rec])
    out = {"tag": tag, "L": d["L"], "H": d["H"], "n": n,
           "always_L_acc": float(L), "always_H_acc": float(H),
           "folds": rows,
           "mean_advantage_vs_random_mix": float(adv.mean()), "sd_advantage": float(adv.std(ddof=1)),
           "mean_heldout_auc": float(np.nanmean(aucs)),
           "verdict": ("held-out cascade beats random-mixing-at-matched-budget (AUC>1/2 out of sample)"
                       if adv.mean() > 0 and np.nanmean(aucs) > 0.5 else "no out-of-sample cascade advantage")}
    json.dump(out, open(os.path.join(RUNS, f"cascade_heldout_{tag}.json"), "w"), indent=2)
    print(f"\nHELD-OUT: mean advantage over random mixing = {adv.mean():+.3f} (sd {adv.std(ddof=1):.3f}, 5 folds); "
          f"mean held-out confidence AUC = {np.nanmean(aucs):.3f}. {out['verdict']}.")
    print(f"[cascade_heldout] wrote cascade_heldout_{tag}.json")


if __name__ == "__main__":
    main()
