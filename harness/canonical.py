"""Canonical analysis, version 2 (2026-09-25): every co-failure number the paper reports, computed in one
deterministic pass from the released matrices. Writes runs/canonical.json and paper/numbers.tex (one LaTeX macro per
reported number, so the manuscript cannot drift from the pipeline).

Three blocks:
  1. market   - the corrected co-failure measurement per benchmark: all-wrong events as graded by harness/grade.py,
                what the audit makes of each (audit/adjudication_result.json, multi-answer code problems), the audited
                beta with its Clopper-Pearson interval, the $0 certificate on the largest achievable gain, and the
                single-factor prediction from pairwise (tetrachoric) correlation.
  2. trace    - every all-wrong event of the June 2026 release, traced through each correction stage
                (_rr: corrupt responses re-queried; _dt: truncation control; _final: corrected grader; audit) to the
                first stage that removes it.
  3. artifact - the June MATH-500 "tail" analysed exactly as a pairwise-correlation model saw it (tetrachoric
                single-factor, full-Sigma Gaussian copula, Clayton, pool-size curve): a real, if spurious, common-mode
                atom, and a demonstration of Prop. poolbias.
plus the free-response GPQA test (v1 retired, v2) and the 15-model pools.

Estimators (exact_copula.py): all-pairs tetrachoric correlations solved to machine precision; adaptive quadrature;
4e6-draw Monte Carlo with standard errors; B = 2000 query bootstrap with a fixed seed; exact sub-matrix means over
200 random pools per size. Usage: python3 canonical.py
"""
import os, json, csv, time
import numpy as np
from multiprocessing import Pool
from scipy.stats import beta as Beta
from scipy.optimize import brentq
from scipy.special import gammaincinv
import exact_copula as X

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); RUNS = os.path.join(ROOT, "runs")
SEED = 20260925
B_BOOT = int(os.environ.get("CANON_BOOT", 2000))
N_COMP = int(os.environ.get("CANON_COMP", 200))

# benchmark -> (dataset filter, [June release, _rr, _dt, _final] matrices; None = stage not applicable)
STAGES = {
    "math500": (["math500"], ["matrix_marketE3.json", "matrix_marketE3_rr.json", "matrix_marketE3_dt.json", "matrix_marketE3_final.json"]),
    "aime": (["aime24", "aime25"], [None, None, None, "matrix_marketE3_final.json"]),
    "mathhard": (["math_hard"], ["matrix_marketMH.json", "matrix_marketMH_rr.json", "matrix_marketMH_dt.json", "matrix_marketMH_final.json"]),
    "code": (["codegen"], ["matrix_marketCG.json", "matrix_marketCG_rr.json", "matrix_marketCG_dt.json", "matrix_marketCG_final.json"]),
    "gpqamc": (["gpqa"], ["matrix_marketE2.json", "matrix_marketE2_rr.json", "matrix_marketE2_dt.json", "matrix_marketE2_final.json"]),
    "mmlupro": (["mmlu_pro"], ["matrix_marketE2.json", "matrix_marketE2_rr.json", "matrix_marketE2_dt.json", "matrix_marketE2_final.json"]),
    "mix": (None, ["matrix_stageA2v3.json", "matrix_stageA2v3_rr.json", "matrix_stageA2v3_dt.json", "matrix_stageA2v3_final.json"]),
    "pro15": (None, ["matrix_hardAv3.json", "matrix_hardAv3.json", "matrix_hardAv3.json", "matrix_hardAv3_final.json"]),
}
STAGE_NAMES = ["corrupt response (re-queried)", "truncation (re-asked at 32,768 tokens)", "grader defect (corrected grader)"]
BIG = 32768


def post_regrade_flips():
    """cells flipped wrong -> right by re-runs applied to a *_final matrix after re-grading (runs/rerun_log.csv): the
    requery of empty responses and the truncation control of the 15-model pools. {final matrix: {qid: {model: step}}}"""
    out = {}
    for row in csv.DictReader(open(os.path.join(RUNS, "rerun_log.csv"))):
        tag = row["tag"].replace(".json", "").replace("matrix_", "")
        if tag.endswith("_final") and row["old_correct"] == "0" and row["new_correct"] == "1":
            out.setdefault(f"matrix_{tag}.json", {}).setdefault(row["qid"], {})[row["model"]] = row["step"]
    return out


FLIPS = post_regrade_flips()
ANALYSED_TAGS = {"marketE3", "marketMH", "marketCG", "marketE2", "stageA2v3", "hardAv3"}


def corrupt_summary():
    """the corrupt-response repair (rerun_failed.py, requery_empty.py), counted from runs/rerun_log.csv over the matrices
    this version analyses"""
    rows = [r for r in csv.DictReader(open(os.path.join(RUNS, "rerun_log.csv")))
            if r["step"] in ("requery", "regrade-from-cache") and r["tag"] in ANALYSED_TAGS]
    empty = [r for r in csv.DictReader(open(os.path.join(RUNS, "rerun_log.csv"))) if r["step"] == "requery-empty"]
    by_model = {}
    for r in rows:
        by_model[r["model"]] = by_model.get(r["model"], 0) + 1
    return {"cells": len(rows), "requeried": sum(r["step"] == "requery" for r in rows),
            "from_cache": sum(r["step"] == "regrade-from-cache" for r in rows),
            "requery_usd": sum(float(r["cost_usd"]) for r in rows if r["step"] == "requery"),
            "wrong_to_right": sum(r["old_correct"] == "0" and r["new_correct"] == "1" for r in rows),
            "by_model": dict(sorted(by_model.items(), key=lambda kv: -kv[1])),
            "empty_requeried": len(empty), "empty_wrong_to_right": sum(r["old_correct"] == "0" and r["new_correct"] == "1" for r in empty)}


def capped_not_reasked(R, q):
    """models whose answer to q hit its token budget and was never answered at 32,768 tokens (re-asking failed: the
    model is no longer served)."""
    return sorted(m for m, c in R[q]["models"].items()
                  if c.get("max_tokens") and c.get("tok_out", 0) >= c["max_tokens"] - 8 and c["max_tokens"] < BIG)


def load(fname, dss=None):
    """models x queries correctness over the queries every model answered; also returns the raw matrix."""
    R = json.load(open(os.path.join(RUNS, fname)))
    qs = [q for q, v in R.items() if dss is None or v["dataset"] in dss]
    ms = sorted({m for q in qs for m in R[q]["models"]})
    full = [q for q in qs if all(m in R[q]["models"] and not R[q]["models"][m].get("missing") for m in ms)]
    M = np.array([[int(R[q]["models"][m]["correct"]) for q in full] for m in ms], float)
    return M, ms, full, R


def allwrong(fname, dss):
    M, ms, full, _ = load(fname, dss)
    return {q for j, q in enumerate(full) if M[:, j].sum() == 0}, len(full)


def cp(k, n, a=0.05):
    return [0.0 if k == 0 else float(Beta.ppf(a / 2, k, n - k + 1)), 1.0 if k == n else float(Beta.ppf(1 - a / 2, k + 1, n - k))]


def cp_lower(k, n, a):
    return 0.0 if k == 0 else float(Beta.ppf(a, k, n - k + 1))


def pearson_mean(M):
    keep = M.std(1) > 0
    C = np.corrcoef(M[keep]); iu = np.triu_indices(int(keep.sum()), 1)
    return float(np.nanmean(C[iu]))


def implied_rho(t, beta):
    f = lambda r: X.single_factor_beta(t, r) - beta
    return float(brentq(f, 1e-4, 0.9999, xtol=1e-8)) if beta > 0 and f(1e-4) < 0 < f(0.9999) else None


def adjudication():
    return {r["qid"]: r["class"] for r in json.load(open(os.path.join(ROOT, "audit", "adjudication_result.json")))}


def certificate(M, k, delta=0.05):
    """$0 certificate (Prop. cert (iii); beta_certificate.py): upper bound on the gain of every selection policy over
    the single best model, with probability >= 1 - delta: beta lower limit and a Bonferroni lower limit on a_sb."""
    m, n = M.shape; c = M.sum(1)
    asb_lo = max(cp_lower(int(ci), n, delta / (2 * m)) for ci in c)
    return (1 - cp_lower(k, n, delta / 2)) - asb_lo, asb_lo


def costs():
    """spend, from released files only: the itemized June run registry, the code run's logged per-cell costs, and the
    per-call ledger of every paid API call (runs/spend_ledger.csv)"""
    reg = {r[0]: float(r[1]) for r in csv.reader(l for l in open(os.path.join(RUNS, "cost_registry.csv")) if not l.startswith("#"))
           if r and r[0] != "run_id"}
    CG = json.load(open(os.path.join(RUNS, "matrix_marketCG.json")))
    by_month = {}; calls = {}
    for r in csv.DictReader(open(os.path.join(RUNS, "spend_ledger.csv"))):
        by_month[r["month"]] = by_month.get(r["month"], 0.0) + float(r["cost_usd"]); calls[r["month"]] = calls.get(r["month"], 0) + 1
    return {"core_pillars": reg["CORE_PILLAR_SUBTOTAL"], "market_scale": reg["MARKET_SCALE_SUBTOTAL"],
            "code": sum(c.get("cost", 0) or 0 for v in CG.values() for c in v["models"].values()),
            "ledger_usd_by_month": by_month, "ledger_calls_by_month": calls}


# ---------------------------------------------------------------- 1. market
def market(name, adj):
    dss, stages = STAGES[name]
    M, ms, full, R = load(stages[3], dss); W = 1 - M; m, n = M.shape; acc = M.mean(1)
    aw = [q for j, q in enumerate(full) if W[:, j].all()]
    multi = {q for q in aw if R[q].get("multi_answer")}
    cls = {q: ("multi-answer (exact-match grading cannot verify)" if q in multi else adj.get(q, "unaudited")) for q in aw}
    genuine = [q for q in aw if cls[q] in ("genuine co-failure", "unresolved", "unaudited")]
    k = len(genuine); beta = k / n
    cert, asb_lo = certificate(M, k)
    r = {"matrix": stages[3], "datasets": dss, "m": m, "n": n, "single_best_model": ms[int(acc.argmax())],
         "single_best": float(acc.max()), "oracle": float((M.sum(0) > 0).mean()), "mean_acc": float(acc.mean()),
         "allwrong_graded": len(aw), "allwrong_class": cls, "k": k, "beta": beta, "beta_cp95": cp(k, n),
         "k_upper": len(aw) - sum(1 for q in aw if cls[q] in ("reference error", "ambiguous item")),
         "certified_max_gain": cert, "single_best_lower": asb_lo, "rho_pearson": pearson_mean(M)}
    r["G"] = r["oracle"] - r["single_best"]
    r["capped_not_reasked"] = {q: capped_not_reasked(R, q) for q in aw if capped_not_reasked(R, q)}
    # questions left out of n because some model did not answer them: all-wrong among the models that did answer
    qs = [q for q, v in R.items() if dss is None or v["dataset"] in dss]
    part = {}
    for q in qs:
        ans = [c for c in R[q]["models"].values() if not c.get("missing")]
        if q not in full and ans and not any(c["correct"] for c in ans):
            part[q] = {"answered": len(ans), "of": m}
    r["partial_allwrong"] = part
    r["n_any_answer"] = sum(1 for q in qs if any(not c.get("missing") for c in R[q]["models"].values()))
    if multi:
        uniq = [j for j, q in enumerate(full) if not R[q].get("multi_answer")]
        ku = int(W[:, uniq].all(0).sum())
        r["unique_output"] = {"n": len(uniq), "k": ku, "beta_cp95": cp(ku, len(uniq))}
    try:
        S, t, info = X.tetrachoric_matrix(W)
        r.update({"rho_tet": X.mean_offdiag(S), "tet_pairs": info, "beta_sf_tet": X.single_factor_beta(t, X.mean_offdiag(S))})
        r["ratio_tet"] = beta / r["beta_sf_tet"] if k else None
    except ValueError:                      # a model that is always right (e.g. 1.000 on AIME subsets): no finite threshold
        r.update({"rho_tet": None, "beta_sf_tet": None, "ratio_tet": None})
    return r


# ---------------------------------------------------------------- 2. trace of the June 2026 events
def trace(name, adj, final_cls):
    dss, stages = STAGES[name]
    if stages[0] is None:
        return None
    sets = [allwrong(f, dss)[0] for f in stages]; n0 = allwrong(stages[0], dss)[1]
    M0 = load(stages[0], dss)[0]
    out = {"june_k": len(sets[0]), "june_n": n0, "june_single_best": float(M0.mean(1).max()), "causes": {}}
    R = json.load(open(os.path.join(RUNS, stages[3]))); flips = FLIPS.get(stages[3], {})
    for q in sorted(sets[0]):
        cause = None
        for i in range(1, 4):
            if q not in sets[i]:
                cause = STAGE_NAMES[i - 1]; break
        if cause == STAGE_NAMES[2]:
            # resolved in the final matrix: by the grader, unless every model now right was flipped by a re-run
            # applied after re-grading (then the re-run is the cause)
            right = {m for m, c in R[q]["models"].items() if c["correct"]}
            if right and right <= set(flips.get(q, {})):
                steps = {flips[q][m] for m in right}
                cause = STAGE_NAMES[0] if steps == {"requery-empty"} else STAGE_NAMES[1]
        if cause is None:
            cause = final_cls.get(q, adj.get(q, "unaudited"))
        out["causes"].setdefault(cause, []).append(q)
    out["counts"] = {c: len(v) for c, v in out["causes"].items()}
    out["capped_not_reasked"] = {q: capped_not_reasked(R, q) for q in sorted(sets[0]) if capped_not_reasked(R, q)}
    return out


# ---------------------------------------------------------------- 3. the June MATH-500 artifact tail
def clayton(W, n_cal=200_000, n_mc=4_000_000):
    m = W.shape[0]; pw = W.mean(1)
    c = W.sum(0); q2 = float(np.mean(c * (c - 1) / 2) / (m * (m - 1) / 2))
    rng = np.random.default_rng(SEED)
    u = rng.random(n_cal); E = rng.exponential(1.0, size=(n_cal, m))
    U = lambda th, uu, EE: (1.0 + EE / gammaincinv(1.0 / th, uu)[:, None]) ** (-1.0 / th)
    def f(th):
        w = U(th, u, E) <= pw; cc = w.sum(1); return float(np.mean(cc * (cc - 1) / 2) / (m * (m - 1) / 2)) - q2
    th = float(brentq(f, 0.05, 30.0, xtol=1e-6)); hits = done = 0
    while done < n_mc:
        b = min(200_000, n_mc - done)
        hits += int(np.all(U(th, rng.random(b), rng.exponential(1.0, size=(b, m))) <= pw, axis=1).sum()); done += b
    p = hits / n_mc
    return {"theta": th, "lambda_L": 2 ** (-1 / th), "beta": p, "beta_se": float(np.sqrt(p * (1 - p) / n_mc))}


def _boot_one(args):
    W, cols = args
    Wb = W[:, cols]
    if np.any(Wb.mean(1) <= 0) or np.any(Wb.mean(1) >= 1):
        return None
    S, t, _ = X.tetrachoric_matrix(Wb); r = X.mean_offdiag(S); b = float((Wb.prod(0) == 1).mean())
    return b, r, b / X.single_factor_beta(t, r)


def _comp_one(args):
    S, t, W, idx = args
    b = float((W[idx].prod(0) == 1).mean())
    return 0.0 if b == 0 else b / X.single_factor_beta(t[idx], X.mean_offdiag(S, idx))


def artifact(fname="matrix_marketE3_dt.json", dss=("math500",)):
    """the June MATH-500 tail (grade_v1 verdicts, after the corrupt-response and truncation fixes)."""
    M, ms, full, _ = load(fname, list(dss)); W = 1 - M; m, n = M.shape
    k = int((W.prod(0) == 1).sum()); beta = k / n
    S, t, info = X.tetrachoric_matrix(W); rho = X.mean_offdiag(S); bsf = X.single_factor_beta(t, rho)
    Sp, nneg = X.nearest_psd_correlation(S); bf, se = X.full_sigma_beta(Sp, t)
    rng = np.random.default_rng(SEED)
    with Pool(max(1, (os.cpu_count() or 2) - 2)) as P:
        boot = [x for x in P.map(_boot_one, [(W, rng.integers(0, n, n)) for _ in range(B_BOOT)], chunksize=8) if x]
        comp = {}
        for kk in [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, m]:
            idxs = [np.arange(m)] if kk == m else [np.sort(rng.choice(m, kk, replace=False)) for _ in range(N_COMP)]
            rr = np.array(P.map(_comp_one, [(S, t, W, i) for i in idxs])); nz = rr[rr > 0]
            comp[str(kk)] = {"pools": len(idxs), "median": float(np.median(nz)) if len(nz) else None,
                             "p05": float(np.percentile(nz, 5)) if len(nz) else None,
                             "p95": float(np.percentile(nz, 95)) if len(nz) else None}
    ratios = np.array([b[2] for b in boot]); cl = clayton(W)
    return {"matrix": fname, "m": m, "n": n, "k": k, "beta": beta, "rho_pearson": pearson_mean(M), "rho_tet": rho,
            "beta_sf_tet": bsf, "ratio_tet": beta / bsf,
            "ratio_p05_p95": [float(np.percentile(ratios, 5)), float(np.percentile(ratios, 95))], "bootstrap_B": len(boot),
            "beta_sf_pearson": X.single_factor_beta(t, pearson_mean(M)), "rho_eff": implied_rho(t, beta),
            "full_sigma": {"neg_eigenvalues": nneg, "rho_after_psd": X.mean_offdiag(Sp), "beta": bf, "beta_se": se, "ratio": beta / bf},
            "clayton": {**cl, "ratio": beta / cl["beta"]}, "composition": comp}


# ---------------------------------------------------------------- free-response GPQA
def gpqa_open(v2_file, v1_file, items_file):
    items = {it["qid"]: it for it in json.load(open(items_file))}
    def summarize(R, qids, label):
        ms = sorted({mm for v in R.values() for mm in v["models"]})
        full = [q for q in qids if q in R and all(mm in R[q]["models"] for mm in ms)]
        M = np.array([[R[q]["models"][mm]["correct"] for q in full] for mm in ms], float); W = 1 - M
        k = int((W.prod(0) == 1).sum())
        part = [q for q in qids if q in R and q not in set(full) and R[q]["models"]]
        k_part = sum(1 for q in part if all(R[q]["models"][mm]["correct"] == 0 for mm in R[q]["models"]))
        rules = {}
        for rule in ("majority", "unanimous", "lenient"):
            Mr = np.array([[{"majority": sum(R[q]["models"][mm]["votes"]) > len(R[q]["models"][mm]["votes"]) / 2,
                             "unanimous": all(v == 1 for v in R[q]["models"][mm]["votes"]),
                             "lenient": any(v == 1 for v in R[q]["models"][mm]["votes"])}[rule]
                            for q in full] for mm in ms], float)
            kk = int(((1 - Mr).prod(0) == 1).sum()); rules[rule] = {"k": kk, "beta": kk / len(full)}
        return {"set": label, "models": len(ms), "n": len(full), "k": k, "beta": k / len(full), "beta_cp95": cp(k, len(full)),
                "with_partial_items_upper": {"n": len(full) + len(part), "k": k + k_part,
                                             "beta_cp95": cp(k + k_part, len(full) + len(part)),
                                             "min_answering": min((len(R[q]["models"]) for q in part), default=None)},
                "mean_acc": float(M.mean()), "best_acc": float(M.mean(1).max()), "rules": rules,
                "qids": full, "allwrong_qids": [q for j, q in enumerate(full) if W[:, j].all()]}
    v1 = json.load(open(os.path.join(RUNS, v1_file))); v2 = json.load(open(os.path.join(RUNS, v2_file)))
    out = {"v1_retired": summarize(v1, list(v1), "v1: all 130 items, v1 stems, 2,000-char judge window"),
           "v2_primary": summarize(v2, [q for q, it in items.items() if it["label"] != "EXCLUDE"], "v2 primary: KEEP + REWRITE"),
           "v2_secondary": summarize(v2, [q for q, it in items.items() if it["exclude_reason"] != "option-dependent"], "v2 secondary")}
    v1aw = out["v1_retired"]["allwrong_qids"]
    out["v1_allwrong_reasons"] = {"option-dependent": sum(1 for q in v1aw if items[q]["exclude_reason"] == "option-dependent"),
                                  "doubtful reference": sum(1 for q in v1aw if items[q]["exclude_reason"] == "non-unique or doubtful gold"),
                                  "kept": sum(1 for q in v1aw if items[q]["label"] != "EXCLUDE")}
    MC = json.load(open(os.path.join(RUNS, "matrix_marketE2_final.json")))
    mc_ms = sorted({mm for q, v in MC.items() if v["dataset"] == "gpqa" for mm in v["models"]})
    for key in ("v1_retired", "v2_primary", "v2_secondary"):
        s = out[key]; R = v1 if key == "v1_retired" else v2
        full = [q.replace("gpqaopen:", "gpqa:") for q in s["qids"]]
        full = [q for q in full if q in MC and all(mm in MC[q]["models"] for mm in mc_ms)]
        Mm = np.array([[MC[q]["models"][mm]["correct"] for q in full] for mm in mc_ms], float)
        kk = int(((1 - Mm).prod(0) == 1).sum())
        s["mc_same_items"] = {"models": len(mc_ms), "n": len(full), "k": kk, "beta": kk / len(full), "beta_cp95": cp(kk, len(full))}
        common = sorted(set(mc_ms) & {mm for v in R.values() for mm in v["models"]})
        both = [q for q in s["qids"] if all(mm in MC[q.replace("gpqaopen:", "gpqa:")]["models"] for mm in common)]
        Mo = np.array([[R[q]["models"][mm]["correct"] for q in both] for mm in common], float)
        Mc = np.array([[MC[q.replace("gpqaopen:", "gpqa:")]["models"][mm]["correct"] for q in both] for mm in common], float)
        s["matched_models"] = {"models": len(common), "n": len(both), "mc_mean": float(Mc.mean()), "open_mean": float(Mo.mean()),
                               "mc_best": float(Mc.mean(1).max()), "open_best": float(Mo.mean(1).max())}
    sec = out["v2_secondary"]
    sec["allwrong_flagged_doubtful_gold"] = sum(1 for q in sec["allwrong_qids"] if items[q]["exclude_reason"] == "non-unique or doubtful gold")
    ks = [v["kappa"] for v in json.load(open(os.path.join(RUNS, "judge_open_v2_meta.json")))["inter_judge"].values()]
    out["v2_kappa_range"] = [min(ks), max(ks)]
    return out


# ---------------------------------------------------------------- 15-model pools (Table A)
def small_pool(fname, fam_csv):
    M, ms, qs, _ = load(fname); m, n = M.shape
    fam = {}
    with open(fam_csv) as f:
        for row in csv.DictReader(r for r in f if not r.startswith("#")):
            fam[row["model_id"]] = row["provider_family"]
    acc = M.mean(1); orc = (M.sum(0) > 0)
    rng = np.random.default_rng(SEED); Gs = []
    for _ in range(2000):
        c = rng.integers(0, n, n); Gs.append(orc[c].mean() - M[:, c].mean(1).max())
    C = np.corrcoef(M); iu = np.triu_indices(m, 1)
    same = np.array([fam[ms[i]] == fam[ms[j]] for i, j in zip(*iu)])
    return {"matrix": fname, "m": m, "n": n, "single_best_model": ms[int(acc.argmax())], "single_best": float(acc.max()),
            "oracle": float(orc.mean()), "G": float(orc.mean() - acc.max()),
            "G_ci95": [float(np.percentile(Gs, 2.5)), float(np.percentile(Gs, 97.5))],
            "rho_pearson": float(C[iu].mean()), "rho_within_family": float(C[iu][same].mean()),
            "rho_cross_family": float(C[iu][~same].mean())}


def main():
    t0 = time.time(); adj = adjudication()
    out = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": SEED, "market": {}, "trace": {}}
    for name in STAGES:
        r = market(name, adj); out["market"][name] = r
        tr = trace(name, adj, r["allwrong_class"])
        if tr:
            out["trace"][name] = tr
        print(f"[canon] {name}: n={r['n']} m={r['m']} all-wrong graded {r['allwrong_graded']} -> genuine {r['k']} | "
              f"sb {r['single_best']:.3f} oracle {r['oracle']:.3f} cert {r['certified_max_gain']:.3f} | "
              f"June {tr['june_k'] if tr else '-'}: {tr['counts'] if tr else ''} ({time.time() - t0:.0f}s)", flush=True)
    out["artifact_math500"] = artifact()
    print(f"[canon] June MATH-500 artifact: ratio {out['artifact_math500']['ratio_tet']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    out["gpqa_open"] = gpqa_open("matrix_marketGPQAOPENv2.json", "matrix_marketGPQAOPEN.json",
                                 os.path.join(ROOT, "gpqa_open_v2", "items_v2.json"))
    out["pool15_mix"] = small_pool("matrix_stageA2v3_final.json", os.path.join(ROOT, "model_registry.csv"))
    out["pool15_hard"] = small_pool("matrix_hardAv3_final.json", os.path.join(ROOT, "model_registry.csv"))
    out["corrupt"] = corrupt_summary()
    # genuine co-failures that both adjudicators also judged not well posed (more than one defensible answer)
    flags = {r["qid"]: (r.get("A_well_posed"), r.get("B_well_posed"))
             for r in json.load(open(os.path.join(ROOT, "audit", "adjudication_result.json")))}
    gen = sorted({q for r in out["market"].values() for q, c in r["allwrong_class"].items() if c == "genuine co-failure"})
    out["genuine_questions"] = gen
    out["genuine_ill_posed_both"] = [q for q in gen if flags.get(q) == (False, False)]
    out["costs"] = costs()
    sp = os.path.join(RUNS, "audit_spend.json")
    if os.path.exists(sp):
        out["audit_spend"] = json.load(open(sp))
    json.dump(out, open(os.path.join(RUNS, "canonical.json"), "w"), indent=1)
    import make_numbers; make_numbers.write(out, os.path.join(ROOT, "paper", "numbers.tex"))
    print(f"[canon] wrote runs/canonical.json and paper/numbers.tex in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
