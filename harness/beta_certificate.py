"""The realizability certificate (Prop. cert) as a runnable tool.

Given the outcome of a multi-model pool on n queries -- either a correctness matrix (models x queries,
0/1) or just the counts (K = #queries all models got wrong, and the single-best model's accuracy) --
this computes the $0 PRE-DEPLOYMENT CERTIFICATE on how much any selection policy (router / vote /
cascade) could possibly gain over single-best:

    ceiling        = 1 - beta            (no policy can exceed this; Prop. cert (i))
    max gain        = (1 - beta) - a_sb   (= the per-query oracle gain G; no selection policy gains more)
    certified bound = (1 - beta_lo) - a_sb_lo   (CP LOWER limit on beta, LOWER limit on a_sb; an UPPER bound on the
                                                  gain of every selection policy, w.p. >= 1-delta)

Both lower limits are needed: a lower beta raises the ceiling and a lower a_sb raises the gain, so the bound stays
conservative. (Pairing the lower beta with an UPPER a_sb bound, as an earlier version did, understates the headroom
and can issue an unsound "skip" verdict.) delta is split evenly: each limit is one-sided at delta/2.

If the certified bound is below your orchestration overhead, NO policy in the class can pay for itself --
decided from one sample, $0 of routing. Pairwise rho is never needed (Prop. nonid: it cannot identify beta).

Usage:
  python3 beta_certificate.py --csv my_grades.csv [--overhead 0.02]    # your own eval: rows = questions, columns = models,
                                                                        # 1 = correct, 0 = wrong (header row optional)
  python3 beta_certificate.py --matrix runs/matrix_marketE3_final.json --dataset math500
  python3 beta_certificate.py --k 0 --n 330 --a_sb 0.988                # counts only (single best treated as pre-specified)
"""
import argparse, json, os
import numpy as np
from scipy.stats import beta as Beta


def cp_interval(k, n, delta=0.05):
    """two-sided Clopper-Pearson interval at level 1-delta (each end is a one-sided limit at delta/2)."""
    lo = 0.0 if k == 0 else Beta.ppf(delta / 2, k, n - k + 1)
    hi = 1.0 if k == n else Beta.ppf(1 - delta / 2, k + 1, n - k)
    return float(lo), float(hi)


def cp_lower(k, n, a):
    """one-sided Clopper-Pearson lower limit at level 1-a."""
    return 0.0 if k == 0 else float(Beta.ppf(a, k, n - k + 1))


def from_matrix(path, dataset):
    R = json.load(open(path))
    qs = [q for q, v in R.items() if dataset is None or v.get("dataset") == dataset]
    models = sorted({m for q in qs for m in R[q]["models"]})
    qs = [q for q in qs if all(m in R[q]["models"] and not R[q]["models"][m].get("missing") for m in models)]
    M = np.array([[R[q]["models"][m]["correct"] for q in qs] for m in models], float)
    n = M.shape[1]
    k = int((M.sum(0) == 0).sum())
    a_sb = float(M.mean(1).max())
    return k, n, a_sb, M.shape[0], [int(c) for c in M.sum(1)]


def from_csv(path):
    """0/1 grades, one row per question and one column per model; a first row that is not numeric is read as a header"""
    import csv
    rows = [r for r in csv.reader(open(path)) if r]
    try:
        [float(x) for x in rows[0]]
    except ValueError:
        rows = rows[1:]
    M = np.array([[float(x) for x in r] for r in rows]).T          # models x questions
    return int((M.sum(0) == 0).sum()), M.shape[1], float(M.mean(1).max()), M.shape[0], [int(c) for c in M.sum(1)]


def certificate(k, n, a_sb, overhead=0.0, delta=0.05, m=None, correct_counts=None):
    beta = k / n
    b_lo, b_hi = cp_interval(k, n, delta)             # b_lo: one-sided lower limit at delta/2
    if correct_counts is not None:
        # single-best is picked in-sample, so bound it simultaneously: with prob >= 1-delta/2 every model's accuracy
        # is above its Bonferroni lower limit, hence a_sb = max_i p_i >= max_i lo_i.
        asb_lo = max(cp_lower(c, n, delta / (2 * len(correct_counts))) for c in correct_counts)
        asb_note = "Bonferroni lower limit over all models (single-best chosen in-sample)"
    else:
        asb_lo = cp_lower(int(round(a_sb * n)), n, delta / 2)
        asb_note = "one-sided lower limit, treating the single-best model as pre-specified"
    ceiling = 1 - beta
    max_gain = ceiling - a_sb
    certified_bound = (1 - b_lo) - asb_lo           # upper bound on every selection policy's gain, w.p. >= 1-delta
    verdict = ("NO policy can pay for itself (certified max gain < overhead): skip orchestration"
               if certified_bound < overhead else
               "orchestration MIGHT pay: certified max gain exceeds overhead -- worth evaluating a router")
    return {
        "m_models": m, "n_queries": n, "k_all_wrong": k,
        "beta": beta, "beta_CP": [b_lo, b_hi],
        "single_best_acc": a_sb, "single_best_acc_lower": asb_lo, "single_best_lower_method": asb_note,
        "ceiling_1_minus_beta": ceiling,
        "max_gain_point": max_gain,
        "certified_max_gain_upper_bound": certified_bound,
        "overhead": overhead, "delta_total": delta,
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix"); ap.add_argument("--dataset"); ap.add_argument("--csv")
    ap.add_argument("--k", type=int); ap.add_argument("--n", type=int); ap.add_argument("--a_sb", type=float)
    ap.add_argument("--overhead", type=float, default=0.0); ap.add_argument("--delta", type=float, default=0.05)
    a = ap.parse_args()
    if a.csv:
        k, n, a_sb, m, counts = from_csv(a.csv)
    elif a.matrix:
        k, n, a_sb, m, counts = from_matrix(a.matrix, a.dataset)
    else:
        assert a.k is not None and a.n is not None and a.a_sb is not None, "need --csv, --matrix or (--k --n --a_sb)"
        k, n, a_sb, m, counts = a.k, a.n, a.a_sb, None, None
    cert = certificate(k, n, a_sb, a.overhead, a.delta, m, counts)
    print(json.dumps(cert, indent=2))
    print(f"\nCERTIFICATE: ceiling = 1-beta = {cert['ceiling_1_minus_beta']:.3f}; single-best = {a_sb:.3f}; "
          f"every selection policy gains at most {cert['certified_max_gain_upper_bound']:+.3f} over single-best "
          f"(w.p. >= {1-a.delta:.0%}).\n{cert['verdict']}")


if __name__ == "__main__":
    main()
