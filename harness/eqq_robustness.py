"""Partition-robustness of the matched-quality diversification gain (Pillar B, item 6/2b).
The +0.055 vs +0.025 discrepancy between aggregation pipelines is a sample-to-split partition
artifact. We resample the partition of the S cached samples into the ranking/eval base and the
Self-MoA distinct draws, recompute the k=3 hetero-minus-SelfMoA gain each time, and report the
distribution. FREE (reuses cached fusion samples; no inference).
Usage: python3 eqq_robustness.py --tag eqq2 --trials 60 --k 3
"""
import os, json, argparse
import numpy as np
from collections import Counter
import grade

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")

def maj(ans):
    ans = [a for a in ans if a is not None]
    return Counter(ans).most_common(1)[0][0] if ans else None

def main(tag, trials, k):
    D = json.load(open(os.path.join(RUNS, f"fusion_{tag}.json")))
    samples, meta, models, S = D["samples"], D["meta"], D["models"], D["S"]
    qids = [q for q in samples if all(m in samples[q] for m in models)]
    def acc(picks):
        return float(np.mean([grade.check(meta[q]["kind"], maj([samples[q][m][s] for (m, s) in picks]),
                                          meta[q]["gold"]) for q in qids]))
    rng = np.random.default_rng(0)
    gains = []
    for _ in range(trials):
        perm = list(rng.permutation(S))
        base = perm[k]                      # eval sample used for ranking + hetero members
        sm_idx = perm[k:2 * k]              # k distinct samples for Self-MoA
        ev = {m: np.mean([grade.check(meta[q]["kind"], samples[q][m][base], meta[q]["gold"]) for q in qids]) for m in models}
        ranked = sorted(models, key=lambda m: -ev[m]); best = ranked[0]
        ht = [(ranked[j], base) for j in range(k)]
        sm = [(best, sm_idx[j]) for j in range(k)]
        gains.append(acc(ht) - acc(sm))
    g = np.array(gains)
    out = {"tag": tag, "k": k, "trials": trials, "mean": float(g.mean()), "std": float(g.std()),
           "median": float(np.median(g)), "min": float(g.min()), "max": float(g.max()),
           "frac_positive": float(np.mean(g > 0)), "n_queries": len(qids)}
    json.dump(out, open(os.path.join(RUNS, f"eqq_robustness_{tag}.json"), "w"), indent=2)
    print(f"k={k} gain over {trials} partitions: mean={out['mean']:+.4f} std={out['std']:.4f} "
          f"range=[{out['min']:+.4f},{out['max']:+.4f}] frac>0={out['frac_positive']:.0%}")
    print(f"[eqq_robustness] wrote eqq_robustness_{tag}.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="eqq2")
    ap.add_argument("--trials", type=int, default=60); ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args(); main(a.tag, a.trials, a.k)
