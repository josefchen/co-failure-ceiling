"""Export the raw model response behind every released cell to responses/responses.jsonl.gz, so any grade can be
re-checked from the original text (python3 -c "import grade; ..."). One JSON object per cell:
  {matrix, qid, model, max_tokens, content, tok_in, tok_out, cost, correct}
Temperature-0.7 samples (matched-quality fusion, cascade) carry {temperature, seed, extracted} instead of correct.
For free-response GPQA the answer is the first non-empty response on the 2,048 -> 8,192 -> 16,384 token ladder, as
used by the judges; judge votes are in runs/judge_open_v2_votes.json.
Usage (maintainers, needs the local response cache): python3 scripts/export_responses.py
"""
import os, sys, json, gzip
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "harness"))
import orclient  # noqa: E402  (only its cache-key function is used; no API calls are made)

FINAL = ["matrix_marketE3_final.json", "matrix_marketMH_final.json", "matrix_marketCG_final.json",
         "matrix_marketE2_final.json", "matrix_stageA2v3_final.json", "matrix_hardAv3_final.json",
         "matrix_churnD_final.json"]
SAMPLED = ["fusion_eqq2_final.json", "fusion_eqq2_math_final.json", "fusion_eqqA_final.json"]   # temperature 0.7, seed = sample index


def cached(model, prompt, max_tokens, temp=0.0, seed=None):
    p = orclient._key(model, [{"role": "user", "content": prompt}], temp, max_tokens, seed)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    S = json.load(open(os.path.join(ROOT, "runs", "items_snapshot.json")))
    os.makedirs(os.path.join(ROOT, "responses"), exist_ok=True)
    out = gzip.open(os.path.join(ROOT, "responses", "responses.jsonl.gz"), "wt")
    n = miss = 0
    for fname in FINAL:
        R = json.load(open(os.path.join(ROOT, "runs", fname)))
        for q, v in R.items():
            it = S.get(q)
            for m, c in v["models"].items():
                mt = c.get("max_tokens", it["max_tokens"] if it else None)
                r = cached(m, it["prompt"], mt) if it else None
                if r is None:
                    miss += 1; continue
                out.write(json.dumps({"matrix": fname, "qid": q, "model": m, "max_tokens": mt, "content": r.get("content", ""),
                                      "tok_in": r.get("tok_in"), "tok_out": r.get("tok_out"), "cost": r.get("cost"),
                                      "correct": c["correct"]}) + "\n"); n += 1
    from regrade_final import BUDGETS
    def sample(fname, q, m, s, temp, seed, mt, extracted):
        r = cached(m, S[q]["prompt"], mt, temp, seed)
        if r is None:
            return 0
        out.write(json.dumps({"matrix": fname, "qid": q, "model": m, "max_tokens": mt, "temperature": temp, "seed": seed,
                              "content": r.get("content", ""), "tok_in": r.get("tok_in"), "tok_out": r.get("tok_out"),
                              "cost": r.get("cost"), "extracted": extracted}) + "\n")
        return 1
    for fname in SAMPLED:
        d = json.load(open(os.path.join(ROOT, "runs", fname))); tag = fname[len("fusion_"):-len("_final.json")]
        for q, mm in d["samples"].items():
            mt = BUDGETS[tag][d["meta"][q]["dataset"]]
            for m, arr in mm.items():
                for s_ in range(1, len(arr) + 1):
                    ok = sample(fname, q, m, s_, 0.7, s_, mt, arr[s_ - 1]); n += ok; miss += 1 - ok
    C = json.load(open(os.path.join(ROOT, "runs", "cascade_stageC2v3_final.json")))
    for rec in C["rec"]:
        q = rec["qid"]; mt = BUDGETS["stageC2v3"][S[q]["dataset"]]
        for s_ in range(1, C["kc"] + 1):
            ok = sample("cascade_stageC2v3_final.json", q, C["L"], s_, 0.7, s_, mt, None); n += ok; miss += 1 - ok
        ok = sample("cascade_stageC2v3_final.json", q, C["H"], 0, 0.0, None, mt, None); n += ok; miss += 1 - ok
    V2 = json.load(open(os.path.join(ROOT, "runs", "matrix_marketGPQAOPENv2.json")))
    from judge_open_v2 import PROMPT
    items = {it["qid"]: it for it in json.load(open(os.path.join(ROOT, "gpqa_open_v2", "items_v2.json")))}
    for q, v in V2.items():
        it = items[q]; prompt = PROMPT.format(stem=it["stem_v2"] if it["label"] == "REWRITE" else it["stem_v1"])
        for m, c in v["models"].items():
            for mt in (2048, 8192, 16384):
                r = cached(m, prompt, mt)
                if r and (r.get("content") or "").strip():
                    out.write(json.dumps({"matrix": "matrix_marketGPQAOPENv2.json", "qid": q, "model": m, "max_tokens": mt,
                                          "content": r["content"], "tok_in": r.get("tok_in"), "tok_out": r.get("tok_out"),
                                          "cost": r.get("cost"), "correct": c["correct"]}) + "\n"); n += 1
                    break
            else:
                miss += 1
    out.close()
    print(f"[export] {n} responses written; {miss} cells without a cached response")


if __name__ == "__main__":
    main()
