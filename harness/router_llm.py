"""Deployment-realistic router baseline: an LLM-as-router. For each query, a
capable-but-cheap router model sees the query and a capsule of each candidate model's strengths,
and names the single model it expects to answer correctly; we route there and look up that model's
logged correctness. This is the routing a real system would deploy. We report the fraction of the
per-query oracle gain G it captures vs held-out single-best -- the test the gradient-boosted/TF-IDF
routers also fail. Runs on the 15-model mix (the pool with logged prompts).
Usage: python3 router_llm.py --tag stageA2v3_final --router openai/gpt-5-mini --cap <usd>
The router's replies are cached by prompt, so re-running on a re-graded matrix replays the same picks at no cost; every
per-query pick is written to the output ("picks") so the numbers can be recomputed without the cache.
"""
import os, json, argparse, re, csv, concurrent.futures as cf
import numpy as np
import orclient, data

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")
DATASETS = ["gsm8k", "mmlu", "math500", "arc"]
CAPSULE = {  # one-line public capsules; the router sees these, not the answer key
    "anthropic/claude-opus-4.8": "top frontier, strongest reasoning/math",
    "openai/gpt-5.1": "top frontier, strong reasoning", "google/gemini-3.1-pro-preview": "frontier, strong",
    "moonshotai/kimi-k2.7-code": "frontier, code/math", "anthropic/claude-sonnet-4.6": "strong mid",
    "openai/gpt-5-mini": "fast mid", "google/gemini-3.5-flash": "fast mid", "qwen/qwen3-235b-a22b-2507": "cheap MoE",
    "mistralai/mistral-large-2512": "mid", "minimax/minimax-m2.7": "cheap", "deepseek/deepseek-v3.2": "cheap strong math",
    "anthropic/claude-haiku-4.5": "cheap fast", "openai/gpt-5-nano": "cheapest", "google/gemini-3.1-flash-lite": "cheapest",
    "meta-llama/llama-4-maverick": "cheap open"}

def main(tag, router, cap, workers=8):
    R = json.load(open(os.path.join(RUNS, f"matrix_{tag}.json")))
    models = sorted({m for v in R.values() for m in v["models"]})
    qids = [q for q, v in R.items() if all(m in v["models"] for m in models)]
    # query text map
    prompts = {}
    for ds in DATASETS:
        for it in data.load(ds, 300):
            prompts[it["qid"]] = it["prompt"]
    qids = [q for q in qids if q in prompts]
    M = {q: {m: R[q]["models"][m]["correct"] for m in models} for q in qids}
    acc = {m: np.mean([M[q][m] for q in qids]) for m in models}
    V_sb = max(acc.values()); sb = max(acc, key=acc.get)
    V_oracle = float(np.mean([1.0 if any(M[q][m] for m in models) else 0.0 for q in qids]))
    G = V_oracle - V_sb
    menu = "\n".join(f"- {m}: {CAPSULE.get(m,'')}" for m in models)

    def route_one(q):
        prompt = (f"You are a model router. Given a query, pick the ONE model most likely to answer it correctly.\n"
                  f"Models:\n{menu}\n\nQuery:\n{prompts[q][:1200]}\n\n"
                  f"Reply with ONLY the exact model id from the list.")
        r = orclient.chat(router, [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=40, budget_cap=cap)
        out = r["content"] or ""
        pick = next((m for m in models if m in out), None)
        if pick is None:  # fallback: last token match on suffix
            pick = next((m for m in models if m.split("/")[-1] in out), sb)
        return q, pick

    picks = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for fu in cf.as_completed([ex.submit(route_one, q) for q in qids]):
            try:
                q, p = fu.result(); picks[q] = p
            except Exception:
                pass
    routed = [q for q in qids if q in picks]
    V_llm = float(np.mean([M[q][picks[q]] for q in routed]))
    frac = (V_llm - V_sb) / G if G > 0 else float("nan")
    from collections import Counter
    dist = Counter(picks.values())
    out = {"tag": tag, "router": router, "n": len(routed), "V_single_best": V_sb, "single_best": sb,
           "V_oracle": V_oracle, "G": G, "V_llm_router": V_llm, "frac_G_captured": frac,
           "routed_to_single_best_frac": dist.get(sb, 0) / max(1, len(routed)),
           "top_picks": dist.most_common(5), "picks": {q: picks[q] for q in routed}}
    json.dump(out, open(os.path.join(RUNS, f"router_llm_{tag}.json"), "w"), indent=2)
    print(f"LLM router ({router}) on {len(routed)} queries:")
    print(f"  single-best={V_sb:.3f} ({sb})  oracle={V_oracle:.3f}  G={G:.3f}")
    print(f"  LLM-router={V_llm:.3f}  -> captures {frac:+.2f} of G")
    print(f"  routed to single-best {out['routed_to_single_best_frac']:.0%} of the time; top picks {dist.most_common(4)}")
    print(f"[router_llm] wrote router_llm_{tag}.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="stageA2v3")
    ap.add_argument("--router", default="openai/gpt-5-mini"); ap.add_argument("--cap", type=float, default=300.0)
    a = ap.parse_args(); main(a.tag, a.router, a.cap)
