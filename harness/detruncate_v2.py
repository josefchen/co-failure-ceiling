"""Truncation control for the headline matrices (2026-09-24). Supersedes detruncate.py, which only ever ran on
the older 53-model matrix (marketE_live) although the paper described the 67-model matrices as truncation-corrected.

For every all-wrong query in an analysed subset (queries every model answered), each cell whose output hit its
token budget (tok_out >= max_tokens - 8) is re-asked with the same prompt at max_tokens = 32768 and re-graded with
the unchanged grader. A query that gains a correct model is no longer all-wrong. Only cells inside all-wrong events
are re-run, so the control can only LOWER beta (and hence the underpricing ratio): it is conservative for every
co-failure claim. Every re-run cell is logged to runs/rerun_log.csv (step = "detruncate").

Usage: python3 detruncate_v2.py --tag marketE3_rr --datasets math500 --out marketE3_dt --cap-usd 620
       (--include-partial also covers questions that some model did not answer; used for AIME, where the only such
       all-wrong question in the study, aime25:27, sits)
"""
import os, json, csv, argparse, datetime, concurrent.futures as cf
import orclient, grade

HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "..", "runs")
BIG = 32768


def analysed_subset(R, ds):
    qs = [q for q, v in R.items() if v["dataset"] == ds]
    ms = sorted({m for q in qs for m in R[q]["models"]})
    full = [q for q in qs if all(m in R[q]["models"] and not R[q]["models"][m].get("missing") for m in ms)]
    return full, ms


def partial_subset(R, ds):
    """every question of ds, with the models that answered it (for all-wrong questions some model did not answer)"""
    qs = [q for q, v in R.items() if v["dataset"] == ds]
    ms = sorted({m for q in qs for m in R[q]["models"]})
    return qs, ms


def main(tag, datasets, out, cap_usd, workers, include_partial=False):
    subset = partial_subset if include_partial else analysed_subset
    R = json.load(open(os.path.join(RUNS, f"matrix_{tag}.json")))
    S = json.load(open(os.path.join(RUNS, "items_snapshot.json")))
    jobs, before = [], {}
    for ds in datasets:
        full, ms = subset(R, ds)
        aw = [q for q in full if any(not c.get("missing") for c in R[q]["models"].values())
              and not any(c["correct"] for c in R[q]["models"].values() if not c.get("missing"))]
        before[ds] = (len(aw), len(full), aw)
        for q in aw:
            it = S[q]
            for m in ms:
                c = R[q]["models"].get(m)
                if c is None or c.get("missing"):
                    continue
                budget = c.get("max_tokens", it["max_tokens"])   # the budget this cell actually ran with (see annotate_budgets.py)
                if c["tok_out"] >= budget - 8 and budget < BIG and not c.get("detruncated"):
                    jobs.append((q, m, it))
        print(f"[detrunc] {tag}/{ds}: all-wrong {len(aw)}/{len(full)}; truncated cells inside them: "
              f"{sum(1 for j in jobs if R[j[0]]['dataset'] == ds)}")
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    def one(q, m, it):
        r = orclient.chat(m, [{"role": "user", "content": it["prompt"]}], temperature=0.0, max_tokens=BIG,
                          budget_cap=cap_usd)
        ok = r.get("tok_in", 0) > 0 and r.get("tok_out", 0) > 0
        return q, m, {"correct": grade.grade(it["kind"], r["content"], it["gold"]) if ok else 0,
                      "cost": r["cost"], "tok_in": r["tok_in"], "tok_out": r["tok_out"], "cached": r["cached"],
                      "detruncated": stamp, "max_tokens": BIG, **({} if ok else {"missing": True})}

    log = os.path.join(RUNS, "rerun_log.csv")
    with open(log, "a", newline="") as f, cf.ThreadPoolExecutor(max_workers=workers) as ex:
        w = csv.writer(f)
        for fu in cf.as_completed([ex.submit(one, *j) for j in jobs]):
            try:
                q, m, cell = fu.result()
            except orclient.BudgetExceeded as e:
                print("[detrunc] STOP:", e); break
            except Exception as e:
                print("[detrunc] cell failed (left as-is):", str(e)[:100]); continue
            old = R[q]["models"][m]["correct"]; R[q]["models"][m] = cell
            w.writerow([stamp, "detruncate", tag, q, m, old, cell["correct"], cell.get("missing", False),
                        cell["tok_in"], cell["tok_out"], f"{cell['cost']:.6f}"])
    json.dump(R, open(os.path.join(RUNS, f"matrix_{out}.json"), "w"))
    for ds in datasets:
        full, ms = subset(R, ds)
        aw = [q for q in full if any(not c.get("missing") for c in R[q]["models"].values())
              and not any(c["correct"] for c in R[q]["models"].values() if not c.get("missing"))]
        gone = sorted(set(before[ds][2]) - set(aw))
        print(f"[detrunc] {ds}: all-wrong {before[ds][0]}/{before[ds][1]} -> {len(aw)}/{len(full)}; "
              f"events that gained a correct model: {gone}")
    print(f"[detrunc] wrote matrix_{out}.json | spend now ${orclient.spend_summary()['total_usd']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--datasets", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--cap-usd", type=float, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--include-partial", action="store_true",
                    help="also treat questions some model did not answer (all-wrong among the models that answered)")
    a = ap.parse_args(); main(a.tag, a.datasets.split(","), a.out, a.cap_usd, a.workers, a.include_partial)
