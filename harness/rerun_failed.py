"""Integrity re-run (2026-09-24): re-query cells whose API call never produced a response.

Such cells have tok_in == tok_out == 0 and cost == 0: OpenRouter returned no usage block, and the content is
empty or cut off mid-word (e.g. "Brief reasoning: To find the value of $f("). orclient cached that as a normal
response and every later run reused it, so the cell was graded "wrong" although the model never finished
answering. This script:

  1. finds every such cell in a matrix,
  2. rebuilds the ORIGINAL request (same prompt, temperature 0, same max_tokens) from the pinned item snapshot
     (runs/items_snapshot.json) and checks that a cache file exists at exactly that key -- proof that the re-run
     asks the identical question,
  3a. if that cache file now holds a complete response (usage present, e.g. a later run already refreshed it),
      re-grades it from the cache with no API call ("regrade-from-cache");
  3b. if it holds the corrupt response, moves it to harness/cache_quarantine/ (kept for audit, never deleted),
      re-queries (up to 3 attempts), and re-grades with the unchanged grader ("requery");
  4. writes matrix_<tag>_rr.json plus one line per cell to runs/rerun_log.csv.

A cell that still returns nothing after 3 attempts is written with "missing": true and is treated as missing
(not wrong) by the analysis. Usage:
  python3 rerun_failed.py --tag marketE3 --dry-run        # verify keys only, no API calls
  python3 rerun_failed.py --tag marketE3 --cap-usd 600    # cap is cumulative spend, as in orclient
"""
import os, json, csv, shutil, argparse, datetime, concurrent.futures as cf
import orclient, grade

HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "..", "runs")
QUAR = os.path.join(HERE, "cache_quarantine")
SNAPSHOT = os.path.join(RUNS, "items_snapshot.json")
LOAD_N = {"math500": 500, "aime24": 30, "aime25": 30, "math_hard": 300, "codegen": 63, "gpqa": 198,
          "mmlu_pro": 250, "gsm8k": 250, "mmlu": 250, "arc": 250}


def is_failed(cell):
    return cell.get("tok_in", 0) == 0 and cell.get("tok_out", 0) == 0 and not cell.get("missing")


def complete(r):
    """a response is complete only if the provider reported usage for it."""
    return r.get("tok_in", 0) > 0 and r.get("tok_out", 0) > 0


def item_index():
    return json.load(open(SNAPSHOT))


def main(tag, dry, cap_usd, workers):
    R = json.load(open(os.path.join(RUNS, f"matrix_{tag}.json")))
    jobs = [(q, m) for q, v in R.items() for m, c in v["models"].items() if is_failed(c)]
    items = item_index()
    plan, from_cache, unmatched = [], [], []
    for q, m in jobs:
        it = items.get(q)
        cpath = orclient._key(m, [{"role": "user", "content": it["prompt"]}], 0.0, it["max_tokens"], None) if it else None
        if not it or not os.path.exists(cpath):
            unmatched.append((q, m)); continue
        (from_cache if complete(json.load(open(cpath))) else plan).append((q, m, it, cpath))
    print(f"[rerun] {tag}: {len(jobs)} failed cells | {len(plan)} to re-query (corrupt cache entry at the exact "
          f"original key) | {len(from_cache)} to re-grade from a complete cached response | {len(unmatched)} unmatched")
    if unmatched:
        print("   unmatched (no cache file at the original key; NOT re-run, left as-is and reported):", unmatched)
    if dry:
        return
    os.makedirs(QUAR, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    def one(q, m, it, cpath):
        qpath = os.path.join(QUAR, os.path.basename(cpath))
        if os.path.exists(cpath):
            shutil.move(cpath, qpath)
        r = None
        for attempt in range(3):
            r = orclient.chat(m, [{"role": "user", "content": it["prompt"]}], temperature=0.0,
                              max_tokens=it["max_tokens"], budget_cap=cap_usd)
            if complete(r):
                break
            os.path.exists(cpath) and shutil.move(cpath, qpath + f".attempt{attempt + 1}")
        answered = complete(r)
        cell = {"correct": grade.grade(it["kind"], r["content"], it["gold"]) if answered else 0,
                "cost": r["cost"], "tok_in": r["tok_in"], "tok_out": r["tok_out"], "cached": False,
                "rerun": stamp}
        if not answered:
            cell["missing"] = True
        return q, m, cell

    log = os.path.join(RUNS, "rerun_log.csv"); new = not os.path.exists(log)
    with open(log, "a", newline="") as f, cf.ThreadPoolExecutor(max_workers=workers) as ex:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "step", "tag", "qid", "model", "old_correct", "new_correct", "missing", "tok_in",
                        "tok_out", "cost_usd"])
        for q, m, it, cpath in from_cache:
            r = json.load(open(cpath))
            cell = {"correct": grade.grade(it["kind"], r["content"], it["gold"]), "cost": r.get("cost", 0.0),
                    "tok_in": r["tok_in"], "tok_out": r["tok_out"], "cached": True, "regraded": stamp}
            old = R[q]["models"][m]["correct"]; R[q]["models"][m] = cell
            w.writerow([stamp, "regrade-from-cache", tag, q, m, old, cell["correct"], False, cell["tok_in"],
                        cell["tok_out"], f"{cell['cost']:.6f}"])
        for fu in cf.as_completed([ex.submit(one, *p) for p in plan]):
            try:
                q, m, cell = fu.result()
            except orclient.BudgetExceeded as e:
                print("[rerun] STOP:", e); break
            old = R[q]["models"][m]["correct"]; R[q]["models"][m] = cell
            w.writerow([stamp, "requery", tag, q, m, old, cell["correct"], cell.get("missing", False),
                        cell["tok_in"], cell["tok_out"], f"{cell['cost']:.6f}"])
    json.dump(R, open(os.path.join(RUNS, f"matrix_{tag}_rr.json"), "w"))
    left = sum(1 for v in R.values() for c in v["models"].values() if is_failed(c))
    miss = sum(1 for v in R.values() for c in v["models"].values() if c.get("missing"))
    print(f"[rerun] wrote matrix_{tag}_rr.json | still-failed {left} | persistent missing {miss} | "
          f"spend now ${orclient.spend_summary()['total_usd']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cap-usd", type=float, default=None); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); main(a.tag, a.dry_run, a.cap_usd, a.workers)
