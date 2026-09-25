"""OpenRouter client: disk cache, cost meter, retry, hard budget guard.
Key is read from env only. Never logs the key. Prices from model_registry.csv.
"""
import os, json, time, hashlib, threading, csv, urllib.request, urllib.error

HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "cache")
SPEND = os.path.join(HERE, "..", "runs", "_spend.json")
REG = os.path.join(HERE, "..", "model_registry.csv")
URL = "https://openrouter.ai/api/v1/chat/completions"
os.makedirs(CACHE, exist_ok=True)

_lock = threading.Lock()

def _prices():
    p = {}
    for reg in (REG, os.path.join(HERE, "..", "churn_registry.csv"),
                os.path.join(HERE, "..", "market_registry.csv")):
        if not os.path.exists(reg):
            continue
        with open(reg) as f:
            for row in csv.DictReader(r for r in f if not r.startswith("#")):
                p[row["model_id"]] = (float(row["price_in_per_mtok_usd"]), float(row["price_out_per_mtok_usd"]))
    return p
PRICES = _prices()

class BudgetExceeded(Exception):
    pass

def _spend_get():
    if os.path.exists(SPEND):
        try:
            return json.load(open(SPEND))
        except (json.JSONDecodeError, ValueError):
            pass  # corrupted (e.g. concurrent writes) -> recover gracefully
    return {"total_usd": 0.0, "calls": 0, "cache_hits": 0, "cache_misses": 0}

def _spend_add(cost, hit):
    with _lock:
        s = _spend_get()
        s["total_usd"] += cost
        s["calls"] += 1
        s["cache_hits"] += int(hit)
        s["cache_misses"] += int(not hit)
        tmp = SPEND + ".tmp"
        with open(tmp, "w") as f:
            json.dump(s, f, indent=2)
        os.replace(tmp, SPEND)  # atomic; prevents partial/corrupt reads
        return s

def spend_summary():
    return _spend_get()

def _key(model, messages, temperature, max_tokens, seed):
    h = hashlib.sha1(json.dumps([model, messages, temperature, max_tokens, seed], sort_keys=True).encode()).hexdigest()
    return os.path.join(CACHE, h + ".json")

def chat(model, messages, temperature=0.0, max_tokens=512, seed=None, budget_cap=None, max_retries=5):
    """Returns dict: {content, tok_in, tok_out, cost, cached}. Raises BudgetExceeded."""
    cpath = _key(model, messages, temperature, max_tokens, seed)
    if os.path.exists(cpath):
        r = json.load(open(cpath))
        _spend_add(0.0, hit=True)
        r["cached"] = True
        return r
    if budget_cap is not None and _spend_get()["total_usd"] >= budget_cap:
        raise BudgetExceeded(f"budget cap ${budget_cap} reached (spent ${_spend_get()['total_usd']:.4f})")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if seed is not None:
        body["seed"] = seed
    data = json.dumps(body).encode()
    last = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(URL, data=data, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            resp = json.load(urllib.request.urlopen(req, timeout=float(os.environ.get("ORCLIENT_TIMEOUT", "120"))))
            msg = resp["choices"][0]["message"]["content"]
            u = resp.get("usage", {}) or {}
            ti, to = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
            pin, pout = PRICES.get(model, (0.0, 0.0))
            cost = ti/1e6*pin + to/1e6*pout
            r = {"content": msg or "", "tok_in": ti, "tok_out": to, "cost": cost}
            json.dump(r, open(cpath, "w"))
            _spend_add(cost, hit=False)
            r["cached"] = False
            return r
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode()[:160]}"
            if e.code in (429, 500, 502, 503, 520, 524):
                time.sleep(2 ** attempt); continue
            break
        except Exception as e:
            last = repr(e); time.sleep(2 ** attempt)
    raise RuntimeError(f"chat failed for {model}: {last}")
