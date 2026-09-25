"""Re-query the five calls that returned zero output tokens without exhausting their budget (found by a scan after
regrade_final.py; 2026-09-25). Same request (model, prompt, temperature 0, budget) as the original; the empty cached
response is moved to cache_quarantine/ (never deleted). Each call has a 300 s wall-clock limit (some providers keep the
connection open indefinitely). Updates the *_final matrices in place and logs each cell to runs/rerun_log.csv
(step "requery-empty"). Pipeline order: rerun_failed.py -> detruncate_v2.py -> regrade_final.py -> requery_empty.py ->
detruncate_v2.py (15-model mix). Needs OPENROUTER_API_KEY and the response cache (maintainers only).
"""
import json, os, shutil, csv, datetime, sys, signal
class WallClock(BaseException): pass
def _alarm(*a): raise WallClock('wall-clock limit 300s')
signal.signal(signal.SIGALRM, _alarm)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orclient, grade
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs") + os.sep
S = json.load(open(RUNS + "items_snapshot.json")); QUAR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_quarantine")
todo = [("matrix_marketE3_final.json", "aime25:1", "nvidia/nemotron-3-super-120b-a12b", 4096),
        ("matrix_marketCG_final.json", "codegen:27", "google/gemini-3.1-pro-preview", 24000),
        ("matrix_marketCG_final.json", "codegen:34", "google/gemini-3.1-pro-preview", 24000),
        ("matrix_stageA2v3_final.json", "gsm8k:43", "google/gemini-3.1-flash-lite", 1024),
        ("matrix_stageA2v3_final.json", "math500:70", "google/gemini-3.1-flash-lite", 2048)]
stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
for f, q, m, mt in todo:
    it = S[q]; p = orclient._key(m, [{"role": "user", "content": it["prompt"]}], 0.0, mt, None)
    if os.path.exists(p):
        r0 = json.load(open(p))
        if r0.get("tok_out", 0) > 0 and (r0.get("content") or "").strip():
            print(q, m, "already complete in cache; skipping re-query"); r = r0
        else:
            shutil.move(p, os.path.join(QUAR, os.path.basename(p) + ".empty")); r = None
    else:
        r = None
    try:
        for attempt in range(3):
            if r is not None and r.get("tok_out", 0) > 0: break
            signal.alarm(300)
            try:
                r = orclient.chat(m, [{"role": "user", "content": it["prompt"]}], temperature=0.0, max_tokens=mt, budget_cap=620)
            finally:
                signal.alarm(0)
    except (Exception, WallClock) as e:
        print(q, m, "FAILED:", str(e)[:100], flush=True)
        R = json.load(open(RUNS + f)); c = R[q]["models"][m]
        c.update({"missing": True, "rerun": stamp, "rerun_error": str(e)[:100]})
        json.dump(R, open(RUNS + f, "w"))
        with open(RUNS + "rerun_log.csv", "a", newline="") as fh:
            csv.writer(fh).writerow([stamp, "requery-empty-failed", f, q, m, c["correct"], c["correct"], True, 0, 0, "0.000000"])
        continue
    R = json.load(open(RUNS + f)); c = R[q]["models"][m]; old = c["correct"]
    ok = r.get("tok_out", 0) > 0
    new = grade.grade(it["kind"], r["content"], it["gold"]) if ok else 0
    c.update({"correct": new, "tok_in": r["tok_in"], "tok_out": r["tok_out"], "cost": r["cost"], "max_tokens": mt, "rerun": stamp})
    c.pop("corrupt_response", None)
    if not ok: c["missing"] = True
    json.dump(R, open(RUNS + f, "w"))
    with open(RUNS + "rerun_log.csv", "a", newline="") as fh:
        csv.writer(fh).writerow([stamp, "requery-empty", f, q, m, old, new, not ok, r["tok_in"], r["tok_out"], f"{r['cost']:.6f}"])
    print(f, q, m, "tok_out", r["tok_out"], "correct", old, "->", new, flush=True)
print("spend now $%.2f" % orclient.spend_summary()["total_usd"])
