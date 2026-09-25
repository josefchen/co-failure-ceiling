"""Build a market-wide model pool with VERIFIED live prices from the OpenRouter catalog.
Writes market_registry.csv (same schema as model_registry.csv). Reports missing candidates.
Curated to span families, sizes, price tiers, and release eras -- chat/instruct only
(pure 'thinking'/reasoning variants excluded to keep programmatic grading clean under a
finite max_tokens, as disclosed in the paper). No fabrication: a candidate is written only
if it is present in the live catalog with a real (non -1) price.
"""
import os, json, csv, urllib.request

HERE = os.path.dirname(__file__)
KEY = os.environ["OPENROUTER_API_KEY"]

# Curated market candidates: frontier + mid + cheap across the whole market.
# The 15 current-frontier IDs are already in model_registry.csv (cached); listed here so
# the market pool is self-contained. Pure reasoning/'thinking' variants intentionally omitted.
CANDIDATES = [
    # === ACTUAL 2026-06-19 frontier (newest per family; verified live + smoke-tested) ===
    "openai/gpt-5.5", "openai/gpt-5.4", "openai/gpt-5.2", "openai/gpt-5.1",
    "anthropic/claude-opus-4.8", "anthropic/claude-sonnet-4.6", "anthropic/claude-haiku-4.5",
    "google/gemini-3.1-pro-preview", "google/gemini-3.5-flash", "google/gemini-3.1-flash-lite",
    "x-ai/grok-4.3", "z-ai/glm-5.2", "z-ai/glm-5.1", "z-ai/glm-5",
    "qwen/qwen3.7-max", "qwen/qwen3.7-plus", "qwen/qwen3.6-max-preview",
    "qwen/qwen3.5-397b-a17b", "qwen/qwen3.5-122b-a10b", "qwen/qwen3-max",
    "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash", "deepseek/deepseek-v3.2",
    "moonshotai/kimi-k2.7-code", "moonshotai/kimi-k2-0905",
    "minimax/minimax-m3", "minimax/minimax-m2.7",
    "mistralai/mistral-medium-3-5", "mistralai/mistral-large-2512",
    "nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3-super-120b-a12b",
    "stepfun/step-3.7-flash",
    # === breadth: mid/cheap and prior-generation across the market (price/era spread) ===
    "openai/gpt-5-mini", "openai/gpt-5-nano", "openai/gpt-oss-120b", "openai/gpt-oss-20b",
    "z-ai/glm-4.7", "z-ai/glm-4.6", "qwen/qwen3-235b-a22b-2507", "qwen/qwen3-coder-plus",
    "qwen/qwen3-next-80b-a3b-instruct", "qwen/qwen-plus-2025-07-28",
    "deepseek/deepseek-chat-v3.1", "deepseek/deepseek-v3.1-terminus",
    "minimax/minimax-m2.5", "minimax/minimax-m2",
    "nvidia/nemotron-3-nano-30b-a3b", "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "meta-llama/llama-4-maverick", "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-70b-instruct", "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.2-3b-instruct", "mistralai/mistral-small-24b-instruct-2501",
    "mistralai/mistral-nemo", "google/gemma-2-27b-it", "google/gemma-3n-e4b-it",
    "microsoft/phi-4-mini-instruct", "ibm-granite/granite-4.1-8b", "ibm-granite/granite-4.0-h-micro",
    "nousresearch/hermes-4-405b", "nousresearch/hermes-4-70b", "ai21/jamba-large-1.7",
    "upstage/solar-pro-3", "inclusionai/ling-2.6-flash", "xiaomi/mimo-v2.5-pro", "writer/palmyra-x5",
]

FAMILY = {  # normalize provider prefix -> family label used in the paper
    "anthropic": "anthropic", "openai": "openai", "google": "google", "moonshotai": "moonshot",
    "qwen": "qwen", "mistralai": "mistral", "minimax": "minimax", "deepseek": "deepseek",
    "meta-llama": "meta", "z-ai": "zai", "nvidia": "nvidia", "nousresearch": "nous",
    "microsoft": "microsoft", "ibm-granite": "ibm", "ai21": "ai21", "upstage": "upstage",
    "stepfun": "stepfun", "inclusionai": "inclusionai", "xiaomi": "xiaomi", "writer": "writer",
    "x-ai": "xai",
}

def tier(pin, pout):
    blended = 0.25 * pin + 0.75 * pout  # output-weighted
    if blended >= 8: return "frontier"
    if blended >= 1.0: return "mid"
    return "cheap"

def main():
    req = urllib.request.Request("https://openrouter.ai/api/v1/models",
                                 headers={"Authorization": f"Bearer {KEY}"})
    cat = {m["id"]: m for m in json.load(urllib.request.urlopen(req, timeout=60))["data"]}
    rows, missing = [], []
    for cid in CANDIDATES:
        m = cat.get(cid)
        if not m:
            missing.append(cid); continue
        pin = float(m["pricing"]["prompt"]) * 1e6
        pout = float(m["pricing"]["completion"]) * 1e6
        if pin < 0 or pout < 0:
            missing.append(cid + " (no price)"); continue
        fam = FAMILY.get(cid.split("/")[0], cid.split("/")[0])
        ctx = int(m.get("context_length") or 0)
        rows.append({"model_id": cid, "provider_family": fam, "tier": tier(pin, pout),
                     "snapshot_date": "2026-06-19", "price_in_per_mtok_usd": f"{pin:.6g}",
                     "price_out_per_mtok_usd": f"{pout:.6g}", "context_window": ctx,
                     "notes": "verified-live-openrouter"})
    out = os.path.join(HERE, "..", "market_registry.csv")
    with open(out, "w", newline="") as f:
        f.write("# Market-wide pool, VERIFIED live prices from OpenRouter /models, snapshot 2026-06-19.\n")
        f.write("# Chat/instruct models only; pure reasoning/'thinking' variants excluded (finite max_tokens grading).\n")
        w = csv.DictWriter(f, fieldnames=["model_id", "provider_family", "tier", "snapshot_date",
                                          "price_in_per_mtok_usd", "price_out_per_mtok_usd",
                                          "context_window", "notes"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: (-(0.25*float(r["price_in_per_mtok_usd"])+0.75*float(r["price_out_per_mtok_usd"])))):
            w.writerow(r)
    fams = sorted(set(r["provider_family"] for r in rows))
    print(f"wrote {out}: {len(rows)} models, {len(fams)} families")
    print("families:", ", ".join(fams))
    by_tier = {}
    for r in rows: by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
    print("tiers:", by_tier)
    if missing:
        print(f"\nMISSING ({len(missing)}):", "; ".join(missing))

if __name__ == "__main__":
    main()
