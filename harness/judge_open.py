"""Version 1 (retired; see judge_open_v2.py): grade the OPEN-ENDED GPQA answers with an LLM-judge PANEL, so co-failure can be measured
on an open-ended domain with NO programmatic oracle. For each (question, model-answer) we ask 3 strong,
diverse judges 'is the candidate equivalent to the reference answer? YES/NO', exclude a judge from grading
its OWN model's answer, and take the majority. We report inter-judge agreement (pairwise + Fleiss-style) so
the grader's reliability is quantified -- an unreliable judge would manufacture or hide co-failure. Then beta,
tetrachoric underpricing, etc. are computed exactly as for the verifiable domains.

Answers are read from the response cache (collected by `experiment.py matrix --datasets gpqa_open`); judging
adds only short calls. Writes runs/matrix_marketGPQAOPEN.json + runs/judge_open_meta.json.
Usage: python3 judge_open.py --tag marketGPQAOPEN --n 130 --cap <usd> --models "<csv>"
"""
import os, json, argparse, re, itertools
import concurrent.futures as cf
import numpy as np
import orclient, data

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")
# Capable, direct-answering (non-reasoning-heavy) judges: they emit a verdict without burning a token budget
# on hidden reasoning (the failure mode of frontier reasoning models at small max_tokens).
JUDGES = ["anthropic/claude-sonnet-4.6", "deepseek/deepseek-v3.2", "google/gemini-3.5-flash",
          "qwen/qwen3.7-max", "mistralai/mistral-large-2512"]   # 5-judge panel (9-program: stronger consensus + kappa)

def get_answer(model, prompt, cap):
    r = orclient.chat(model, [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=2048, budget_cap=cap)
    ans = (r.get("content") or "").strip()
    if not ans:  # de-truncate: reasoning models exhaust 2048 on hidden reasoning and emit no answer (a false
        # all-wrong); re-fetch at 8192 (then 16384 for the stubborn heaviest reasoners) so the cell is a genuine
        # answer, not a truncation artifact -- pushing toward full 130-question coverage.
        r = orclient.chat(model, [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=8192, budget_cap=cap)
        ans = (r.get("content") or "").strip()
    if not ans:
        r = orclient.chat(model, [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=16384, budget_cap=cap)
        ans = (r.get("content") or "").strip()
    return ans

def judge_one(judge, question, gold, answer, cap):
    p = (f"You are grading a graduate science answer. QUESTION:\n{question}\n\nREFERENCE (correct) ANSWER:\n{gold}\n\n"
         f"CANDIDATE ANSWER:\n{answer[:2000]}\n\nIs the candidate's FINAL answer correct, i.e. equivalent to the "
         f"reference (same value/quantity/claim, allowing equivalent forms, rounding, and units)? "
         f"Answer with exactly one word: YES or NO.")
    r = orclient.chat(judge, [{"role": "user", "content": p}], temperature=0.0, max_tokens=256, budget_cap=cap)
    out = (r.get("content") or "").upper()
    yes = [m.start() for m in re.finditer(r"\bYES\b", out)]; no = [m.start() for m in re.finditer(r"\bNO\b", out)]
    if not yes and not no:
        return None
    if yes and not no: return 1
    if no and not yes: return 0
    return 1 if max(yes) > max(no) else 0   # take the last verdict mentioned

def main(tag, n, cap, models, workers=8):
    items = data.load("gpqa_open", n)
    qmap = {it["qid"]: it for it in items}
    qids = list(qmap)
    # 1) collect each model's answer (cache hit if the matrix run already ran)
    pairs = [(m, q) for m in models for q in qids]
    answers = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(get_answer, m, qmap[q]["prompt"], cap): (m, q) for m, q in pairs}
        for f in cf.as_completed(fut):
            m, q = fut[f]
            try: answers[(m, q)] = f.result()
            except Exception: answers[(m, q)] = ""
    # 2) judge every (model, question) with the panel, excluding a judge grading its own model
    judge_votes = {}   # (m,q,judge) -> 0/1
    jobs = []
    for (m, q), ans in answers.items():
        if not ans: continue
        for J in JUDGES:
            if J == m: continue  # no self-judging
            jobs.append((m, q, J, ans))
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(judge_one, J, qmap[q]["prompt"], qmap[q]["gold"], ans, cap): (m, q, J) for (m, q, J, ans) in jobs}
        for f in cf.as_completed(fut):
            m, q, J = fut[f]
            try:
                v = f.result()
                if v is not None: judge_votes[(m, q, J)] = v
            except Exception: pass
    # 3) majority-vote correctness per (model, question)
    R = {}
    for q in qids:
        R[q] = {"dataset": "gpqa_open", "kind": "open", "gold": qmap[q]["gold"], "models": {}}
        for m in models:
            if (m, q) not in answers or not answers[(m, q)]: continue
            vs = [judge_votes[(m, q, J)] for J in JUDGES if (m, q, J) in judge_votes]
            if not vs: continue
            R[q]["models"][m] = {"correct": int(sum(vs) > len(vs) / 2.0), "votes": vs}
    json.dump(R, open(os.path.join(RUNS, f"matrix_{tag}.json"), "w"))
    json.dump({f"{m}|||{q}|||{J}": v for (m, q, J), v in judge_votes.items()},
              open(os.path.join(RUNS, "judge_open_votes.json"), "w"))
    # 3b) aggregation robustness: does beta survive unanimity / lenient grading rules?
    def beta_under(rule):
        rows = {}
        for q in qids:
            cells = {}
            for m in models:
                if (m, q) not in answers or not answers[(m, q)]: continue
                vs = [judge_votes[(m, q, J)] for J in JUDGES if (m, q, J) in judge_votes]
                if not vs: continue
                if rule == "majority": ok = sum(vs) > len(vs) / 2.0
                elif rule == "unanimous": ok = all(v == 1 for v in vs)   # correct only if ALL judges agree
                else: ok = any(v == 1 for v in vs)                        # lenient: any judge says correct
                cells[m] = int(ok)
            rows[q] = cells
        full_q = [q for q in qids if len(rows[q]) == len(models)]
        if not full_q: return None
        M = np.array([[rows[q][m] for q in full_q] for m in models], float); W = 1 - M
        return {"rule": rule, "n": len(full_q), "beta": float((W.prod(0) == 1).mean()),
                "k": int((W.prod(0) == 1).sum()), "mean_acc": float(M.mean())}
    robustness = [beta_under(r) for r in ("majority", "unanimous", "lenient")]
    # 4) inter-judge agreement (pairwise) over all jointly-judged (m,q)
    agr = {}
    for Ja, Jb in itertools.combinations(JUDGES, 2):
        both = [(m, q) for (m, q) in answers if (m, q, Ja) in judge_votes and (m, q, Jb) in judge_votes]
        if both:
            a = np.array([judge_votes[(m, q, Ja)] for m, q in both]); b = np.array([judge_votes[(m, q, Jb)] for m, q in both])
            po = float(np.mean(a == b)); pa, pb = a.mean(), b.mean()
            pe = pa * pb + (1 - pa) * (1 - pb)
            kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
            agr[f"{Ja.split('/')[-1]} vs {Jb.split('/')[-1]}"] = {"n": len(both), "agree": po, "kappa": float(kappa)}
    # 5) beta on the fully-judged subset
    full = [q for q in qids if len(R[q]["models"]) == len(models)]
    meta = {"tag": tag, "n_questions": len(qids), "models": len(models), "judges": JUDGES,
            "fully_judged": len(full), "inter_judge": agr, "aggregation_robustness": robustness}
    print("aggregation robustness:", [(r["rule"], round(r["beta"], 3), "k=%d" % r["k"], "n=%d" % r["n"]) for r in robustness if r])
    if full:
        M = np.array([[R[q]["models"][m]["correct"] for q in full] for m in models], float)
        W = 1 - M; beta = float((W.prod(0) == 1).mean()); k = int((W.prod(0) == 1).sum())
        meta.update({"beta": beta, "k_allwrong": k, "mean_acc": float(M.mean()), "best_acc": float(M.mean(1).max())})
        print(f"OPEN-GPQA: {len(full)} fully-judged Qs x {len(models)} models | beta={beta:.3f} (k={k}) | "
              f"mean acc {M.mean():.3f} best {M.mean(1).max():.3f}")
    print("inter-judge kappa:", {k2: round(v["kappa"], 2) for k2, v in agr.items()})
    json.dump(meta, open(os.path.join(RUNS, "judge_open_meta.json"), "w"), indent=2)
    print(f"[judge_open] wrote matrix_{tag}.json + judge_open_meta.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="marketGPQAOPEN"); ap.add_argument("--n", type=int, default=130)
    ap.add_argument("--cap", type=float, default=600.0); ap.add_argument("--models", required=True)
    a = ap.parse_args(); main(a.tag, a.n, a.cap, [m.strip() for m in a.models.split(",") if m.strip()])
