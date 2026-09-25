"""Write runs/spend_ledger.csv: one row per API call of the project that returned a response (every cached response, incl. exploratory
and superseded runs and the entries later quarantined), with the request's cache key (sha1 of model, messages,
temperature, max_tokens, seed), the month of the call and its logged list-price cost. The paper's spend totals are sums
over this file. Maintainers only (needs the local response cache): python3 scripts/export_spend_ledger.py
"""
import os, csv, json, glob, datetime
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

if __name__ == "__main__":
    rows = []
    for d in ("cache", "cache_quarantine"):
        for p in glob.glob(os.path.join(ROOT, "harness", d, "*.json*")):
            r = json.load(open(p))
            month = datetime.datetime.fromtimestamp(os.path.getmtime(p), datetime.timezone.utc).strftime("%Y-%m")
            rows.append([os.path.basename(p).split(".")[0], month, f"{(r.get('cost') or 0):.8f}", d == "cache_quarantine"])
    rows.sort()
    with open(os.path.join(ROOT, "runs", "spend_ledger.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["request_key", "month", "cost_usd", "quarantined"]); w.writerows(rows)
    print(f"[ledger] {len(rows)} calls, ${sum(float(r[2]) for r in rows):.2f}")
