"""Reconstruct a (possibly partial) correctness matrix directly from the disk cache,
without waiting for a running experiment to write its final JSON. For each (item, model)
the cache key is recomputed; if the response is cached it is loaded and graded. Writes
matrix_<tag>.json in the standard format so realizability.py / router_baseline.py / figures.py
can consume it. Anytime-safe: reads the same cache the live run fills.
Usage: python3 reconstruct.py --pool market --datasets gsm8k,mmlu,arc,math500,mmlu_pro,aime24,aime25 --n 250 --tag marketE
"""
import os, json, argparse
import orclient, data, grade, experiment

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")

def main(pool, datasets, n, tag, offset=0):
    models = experiment.market_models() if pool == "market" else experiment.all_models()
    items = []
    for ds in datasets:
        items += data.load(ds, n, offset)
    results, have, miss = {}, 0, 0
    for it in items:
        cell = {"dataset": it["dataset"], "kind": it["kind"], "gold": it["gold"], "models": {}}
        msgs = [{"role": "user", "content": it["prompt"]}]
        for m in models:
            cpath = orclient._key(m, msgs, 0.0, it["max_tokens"], None)
            if os.path.exists(cpath):
                try:
                    r = json.load(open(cpath))
                except Exception:
                    miss += 1; continue
                c = grade.grade(it["kind"], r.get("content", ""), it["gold"])
                cell["models"][m] = {"correct": c, "cost": r.get("cost", 0.0),
                                     "tok_in": r.get("tok_in", 0), "tok_out": r.get("tok_out", 0), "cached": True}
                have += 1
            else:
                miss += 1
        results[it["qid"]] = cell
    out = os.path.join(RUNS, f"matrix_{tag}.json")
    json.dump(results, open(out, "w"))
    tot = have + miss
    # per-dataset coverage
    cov = {}
    for q, v in results.items():
        d = v["dataset"]; cov.setdefault(d, [0, 0]); cov[d][0] += len(v["models"]); cov[d][1] += len(models)
    print(f"[reconstruct] {tag}: {have}/{tot} cells present ({have/max(1,tot):.1%}) over {len(models)} models")
    for d, (h, t) in sorted(cov.items()):
        print(f"   {d:10} {h}/{t} ({h/max(1,t):.0%})")
    print(f"[reconstruct] wrote {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="market"); ap.add_argument("--datasets", required=True)
    ap.add_argument("--n", type=int, required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()
    main(a.pool, a.datasets.split(","), a.n, a.tag, a.offset)
