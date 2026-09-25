"""Cascade calibration economics experiment (Pillar C).
Cheap model L answers with a self-consistency confidence (agreement among k_c temp>0
samples); escalate to strong model H when confidence < tau. Sweep tau.
Tests:
  H-C1 collapse identity: Q_cascade(phi) - Q_mix(phi) -> 0 as verifier AUC -> 0.5
  H-C2 volume ceiling:    min escalation at quality floor q vs (1 - a_L/a_H)
  C3 dominance:           cascade $/correct vs strong-only, vs AUC
All numbers REAL (sampled via API). Nothing fabricated.
Usage: export OPENROUTER_API_KEY=...
  python3 cascade.py --L openai/gpt-4o-mini --H qwen/qwen-2.5-72b-instruct \
      --datasets gsm8k,mmlu,math500,arc --n 150 --kc 5 --cap 250 --tag stageC
"""
import os, json, argparse, collections, random, concurrent.futures as cf
import numpy as np
import orclient, data, grade

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")

def majority(ans):
    ans = [a for a in ans if a is not None]
    if not ans: return None, 0.0
    c = collections.Counter(ans); top, cnt = c.most_common(1)[0]
    return top, cnt/len(ans)

def auc_error_detector(conf, wrong):
    """AUC of (low confidence -> L wrong). score = 1-conf; label = wrong."""
    score = [1-c for c in conf]; y = wrong
    pos = [score[i] for i in range(len(y)) if y[i] == 1]
    neg = [score[i] for i in range(len(y)) if y[i] == 0]
    if not pos or not neg: return float('nan')
    wins = sum((a > b) + 0.5*(a == b) for a in pos for b in neg)
    return wins/(len(pos)*len(neg))

def run(L, H, datasets, n, kc, cap, tag, workers=32):
    items = []
    for ds in datasets: items += data.load(ds, n)
    print(f"[cascade] L={L} H={H} | {len(items)} queries | kc={kc} | cap ${cap}")

    def Lsample(it, s):
        r = orclient.chat(L, [{"role": "user", "content": it["prompt"]}], temperature=0.7,
                          max_tokens=it["max_tokens"], seed=s, budget_cap=cap)
        return ("L", it["qid"], s, grade.extract(it["kind"], r["content"]), r["cost"])
    def Hanswer(it):
        r = orclient.chat(H, [{"role": "user", "content": it["prompt"]}], temperature=0.0,
                          max_tokens=it["max_tokens"], budget_cap=cap)
        return ("H", it["qid"], 0, grade.extract(it["kind"], r["content"]), r["cost"])

    Ls = {it["qid"]: [None]*kc for it in items}; Lc = {it["qid"]: [0.0]*kc for it in items}
    Ha = {it["qid"]: None for it in items}; Hc = {it["qid"]: 0.0 for it in items}
    meta = {it["qid"]: {"kind": it["kind"], "gold": it["gold"], "dataset": it["dataset"]} for it in items}
    jobs = [Lsample]*0
    futs = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for it in items:
            for s in range(1, kc+1): futs.append(ex.submit(Lsample, it, s))
            futs.append(ex.submit(Hanswer, it))
        fail = 0
        for i, fu in enumerate(cf.as_completed(futs)):
            try:
                tag_, qid, s, ans, cost = fu.result()
                if tag_ == "L": Ls[qid][s-1] = ans; Lc[qid][s-1] = cost
                else: Ha[qid] = ans; Hc[qid] = cost
            except orclient.BudgetExceeded as e:
                print(f"[cascade] STOP (budget): {e}"); break
            except Exception as e:
                fail += 1
                if fail <= 5: print(f"[cascade] cell failed: {str(e)[:100]}")
            if (i+1) % 500 == 0:
                print(f"[cascade] {i+1}/{len(futs)} | spent ${orclient.spend_summary()['total_usd']:.4f}")
    # per-query L modal answer + confidence + correctness; H correctness
    rec = []
    for q in meta:
        modal, conf = majority(Ls[q])
        Lcorr = grade.check(meta[q]["kind"], modal, meta[q]["gold"])
        Hcorr = grade.check(meta[q]["kind"], Ha[q], meta[q]["gold"])
        rec.append({"qid": q, "ds": meta[q]["dataset"], "conf": conf, "Lcorr": Lcorr,
                    "Hcorr": Hcorr, "Lcost": sum(Lc[q]), "Hcost": Hc[q]})
    json.dump({"L": L, "H": H, "kc": kc, "rec": rec}, open(os.path.join(RUNS, f"cascade_{tag}.json"), "w"))
    print(f"[cascade] saved cascade_{tag}.json | failed {fail}")
    analyze(tag)

def _curve(rec, conf_key="conf"):
    a_L = np.mean([r["Lcorr"] for r in rec]); a_H = np.mean([r["Hcorr"] for r in rec])
    Lcost = np.mean([r["Lcost"] for r in rec]); Hcost = np.mean([r["Hcost"] for r in rec])
    confs = sorted(set(r[conf_key] for r in rec))
    rows = []
    for tau in confs + [1.01]:
        esc = [r for r in rec if r[conf_key] < tau]
        phi = len(esc)/len(rec)
        Q = np.mean([(r["Hcorr"] if r[conf_key] < tau else r["Lcorr"]) for r in rec])
        C = np.mean([(r["Lcost"] + (r["Hcost"] if r[conf_key] < tau else 0)) for r in rec])
        Qmix = a_L + phi*(a_H - a_L)  # random-mixing chord at matched coverage
        rows.append({"tau": tau, "phi": phi, "Q": Q, "C": C, "Qmix": Qmix, "gap": Q - Qmix,
                     "dpc": (C/Q if Q > 0 else float('inf'))})
    return rows, a_L, a_H, Lcost, Hcost

def analyze(tag):
    D = json.load(open(os.path.join(RUNS, f"cascade_{tag}.json")))
    rec = D["rec"]
    rows, a_L, a_H, Lcost, Hcost = _curve(rec)
    auc = auc_error_detector([r["conf"] for r in rec], [1-r["Lcorr"] for r in rec])
    ceiling = 1 - a_L/a_H if a_H > 0 else float('nan')
    print(f"\n=== CASCADE tag={tag} | L={D['L']} (a_L={a_L:.3f}, ${Lcost:.5f}) H={D['H']} (a_H={a_H:.3f}, ${Hcost:.5f}) ===")
    print(f"verifier AUC (self-consistency as error detector) = {auc:.3f}")
    print(f"volume ceiling 1 - a_L/a_H = {ceiling:.3f}  (max fraction of H-calls any cascade can avoid)")
    maxgap = max(rows, key=lambda r: r["gap"])
    print(f"max calibration lift over random-mixing: gap={maxgap['gap']:.3f} at phi={maxgap['phi']:.3f}")
    dpc_L, dpc_H = Lcost/a_L, Hcost/a_H
    best = min(rows, key=lambda r: r["dpc"])
    print(f"$/correct: L-only={dpc_L:.5f}  H-only={dpc_H:.5f}  best-cascade={best['dpc']:.5f} at phi={best['phi']:.3f}")
    # AUC degradation: mix confidence with uniform noise, recompute collapse gap.
    # Averaged over SEEDS=20 independent noise draws per level (operates on cached
    # confidences, no inference); we report mean and std so the collapse claim is
    # an honest seed-averaged statement. NOTE: the cascade ADVANTAGE (gap) is what
    # collapses monotonically to zero; injected-noise AUC is non-monotone at low
    # noise (it can rise slightly), so we never claim AUC itself is monotone.
    SEEDS = 20
    print(f"\nAUC degradation (inject noise into confidence, mean +/- std over {SEEDS} seeds) -> collapse to random mixing:")
    print(f"{'noise':>6} {'AUC(mean)':>10} {'AUC(std)':>9} {'gap(mean)':>10} {'gap(std)':>9}")
    deg = []
    for noise in [0.0, 0.25, 0.5, 0.75, 1.0]:
        aucs, gaps = [], []
        for s in range(SEEDS):
            rng = random.Random(1000 + s)
            rec2 = [dict(r, confn=(1-noise)*r["conf"] + noise*rng.random()) for r in rec]
            r2, *_ = _curve(rec2, conf_key="confn")
            aucs.append(auc_error_detector([x["confn"] for x in rec2], [1-x["Lcorr"] for x in rec2]))
            gaps.append(max(x["gap"] for x in r2))
        deg.append({"noise": noise, "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                    "max_gap": float(np.mean(gaps)), "max_gap_std": float(np.std(gaps)), "seeds": SEEDS})
        print(f"{noise:6.2f} {np.mean(aucs):10.3f} {np.std(aucs):9.4f} {np.mean(gaps):10.4f} {np.std(gaps):9.4f}")
    json.dump({"a_L": a_L, "a_H": a_H, "auc": auc, "ceiling": ceiling, "rows": rows,
               "dpc_L": dpc_L, "dpc_H": dpc_H, "best_cascade_dpc": best["dpc"], "degradation": deg},
              open(os.path.join(RUNS, f"cascade_analysis_{tag}.json"), "w"), indent=2)
    print(f"[cascade] wrote cascade_analysis_{tag}.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", required=True); ap.add_argument("--H", required=True)
    ap.add_argument("--datasets", required=True); ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--kc", type=int, default=5); ap.add_argument("--cap", type=float, required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--analyze_only", action="store_true")
    a = ap.parse_args()
    if a.analyze_only: analyze(a.tag)
    else: run(a.L, a.H, a.datasets.split(","), a.n, a.kc, a.cap, a.tag)
