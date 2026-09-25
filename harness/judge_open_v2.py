"""Free-response GPQA, v2 (2026-09-25). Implements gpqa_open_v2/PROTOCOL.md.

Differences from judge_open.py (v1), and nothing else:
  * items and stems come from gpqa_open_v2/items_v2.json: EXCLUDE items are not asked; the 5 REWRITE items are
    asked with their rewritten stem; KEEP items use the unchanged v1 stem (so their answers come from the cache);
  * judges see the model's FULL answer. v1 showed only answer[:2000], which can hide the final answer line.
The prompt template, token ladder (2048 -> 8192 -> 16384), judge panel, judge prompt wording, majority rule, tie
handling and self-judging exclusion are identical to v1. Writes runs/matrix_marketGPQAOPENv2.json,
runs/judge_open_v2_votes.json and runs/judge_open_v2_meta.json.

Usage: python3 judge_open_v2.py --cap 620
"""
import os, re, json, argparse, itertools, concurrent.futures as cf
import numpy as np
import orclient
from judge_open import JUDGES, get_answer

HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "..", "runs")
V2 = os.path.join(HERE, "..", "gpqa_open_v2", "items_v2.json")
PROMPT = ("Answer this graduate-level science question. Reason briefly, then give your final answer concisely on "
          "the last line as 'Answer: <answer>'.\n\n{stem}")          # the v1 template, verbatim (data.gpqa_open)


def judge_full(judge, question, gold, answer, cap):
    """v1 judge_one with the whole answer instead of answer[:2000]; wording otherwise identical."""
    p = (f"You are grading a graduate science answer. QUESTION:\n{question}\n\nREFERENCE (correct) ANSWER:\n{gold}\n\n"
         f"CANDIDATE ANSWER:\n{answer}\n\nIs the candidate's FINAL answer correct, i.e. equivalent to the "
         f"reference (same value/quantity/claim, allowing equivalent forms, rounding, and units)? "
         f"Answer with exactly one word: YES or NO.")
    r = orclient.chat(judge, [{"role": "user", "content": p}], temperature=0.0, max_tokens=256, budget_cap=cap)
    out = (r.get("content") or "").upper()
    yes = [m.start() for m in re.finditer(r"\bYES\b", out)]; no = [m.start() for m in re.finditer(r"\bNO\b", out)]
    if not yes and not no:
        return None
    if yes and not no: return 1
    if no and not yes: return 0
    return 1 if max(yes) > max(no) else 0


def main(cap, workers=8):
    v1 = json.load(open(os.path.join(RUNS, "matrix_marketGPQAOPEN.json")))
    models = sorted({m for v in v1.values() for m in v["models"]})
    items = [it for it in json.load(open(V2)) if it["label"] != "EXCLUDE" or it["exclude_reason"] != "option-dependent"]
    prompts = {it["qid"]: PROMPT.format(stem=it["stem_v2"] if it["label"] == "REWRITE" else it["stem_v1"]) for it in items}
    gold = {it["qid"]: it["gold"] for it in items}
    print(f"[v2] {len(items)} items asked ({sum(it['label'] == 'REWRITE' for it in items)} rewritten) x {len(models)} models")
    answers = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(get_answer, m, prompts[q], cap): (m, q) for m in models for q in prompts}
        for f in cf.as_completed(fut):
            try: answers[fut[f]] = f.result()
            except orclient.BudgetExceeded as e: print("[v2] STOP:", e); raise
            except Exception: answers[fut[f]] = ""
    votes = {}
    jobs = [(m, q, J) for (m, q), a in answers.items() if a for J in JUDGES if J != m]
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(judge_full, J, prompts[q], gold[q], answers[(m, q)], cap): (m, q, J) for m, q, J in jobs}
        for f in cf.as_completed(fut):
            try:
                v = f.result()
                if v is not None: votes[fut[f]] = v
            except orclient.BudgetExceeded as e: print("[v2] STOP:", e); raise
            except Exception: pass
    label = {it["qid"]: it["label"] for it in items}
    R = {}
    for q in prompts:
        R[q] = {"dataset": "gpqa_open_v2", "kind": "open", "gold": gold[q], "v2_label": label[q], "models": {}}
        for m in models:
            a = answers.get((m, q))
            if not a: continue
            vs = [votes[(m, q, J)] for J in JUDGES if (m, q, J) in votes]
            if not vs: continue
            R[q]["models"][m] = {"correct": int(sum(vs) > len(vs) / 2.0), "votes": vs, "answer_chars": len(a)}
    json.dump(R, open(os.path.join(RUNS, "matrix_marketGPQAOPENv2.json"), "w"))
    json.dump({f"{m}|||{q}|||{J}": v for (m, q, J), v in votes.items()},
              open(os.path.join(RUNS, "judge_open_v2_votes.json"), "w"))
    agr = {}
    for Ja, Jb in itertools.combinations(JUDGES, 2):
        both = [(m, q) for (m, q) in answers if (m, q, Ja) in votes and (m, q, Jb) in votes]
        if both:
            a = np.array([votes[(m, q, Ja)] for m, q in both]); b = np.array([votes[(m, q, Jb)] for m, q in both])
            po = float(np.mean(a == b)); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
            agr[f"{Ja.split('/')[-1]} vs {Jb.split('/')[-1]}"] = {"n": len(both), "agree": po,
                                                                  "kappa": float((po - pe) / (1 - pe)) if pe < 1 else 1.0}
    meta = {"protocol": "gpqa_open_v2/PROTOCOL.md", "models": models, "judges": JUDGES, "n_items_asked": len(prompts),
            "inter_judge": agr, "spend_usd_after": orclient.spend_summary()["total_usd"]}
    json.dump(meta, open(os.path.join(RUNS, "judge_open_v2_meta.json"), "w"), indent=2)
    full = [q for q in R if len(R[q]["models"]) == len(models)]
    print(f"[v2] complete-coverage items: {len(full)}/{len(R)} | kappa range "
          f"{min(v['kappa'] for v in agr.values()):.2f}-{max(v['kappa'] for v in agr.values()):.2f} | "
          f"spend now ${meta['spend_usd_after']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cap", type=float, required=True)
    a = ap.parse_args(); main(a.cap)
