"""Turn runs/canonical.json into paper/numbers.tex: one \\newcommand per number the paper reports.
Generated, never hand-edited. Regenerate with `python3 canonical.py` (or `python3 make_numbers.py` to re-render)."""
import json, os, hashlib

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
CAUSES = {"corrupt response (re-queried)": "Corrupt", "truncation (re-asked at 32,768 tokens)": "Trunc",
          "grader defect (corrected grader)": "Grader", "reference error": "RefErr", "ambiguous item": "Ambig",
          "multi-answer (exact-match grading cannot verify)": "Multi", "genuine co-failure": "Genuine",
          "unresolved": "Unres", "unaudited": "Unaud"}
BENCH = [("math500", "Math"), ("aime", "Aime"), ("mathhard", "Hard"), ("code", "Code"), ("gpqamc", "Gmc"),
         ("mmlupro", "Mmlu"), ("mix", "Mix"), ("pro15", "Pro")]


def f(x, d, sign=False):
    if x is None:
        return "--"
    return ("+" if sign and x > 0 else "") + f"{x:.{d}f}"


def write(c, path):
    L = []
    def mac(name, val):
        assert name.isalpha(), name
        L.append(f"\\newcommand{{\\n{name}}}{{{val}}}")
    tot = {v: 0 for v in CAUSES.values()}; tot_june = 0; capped_june = capped_left = capped_left_q = 0
    for key, P in BENCH:
        r = c["market"][key]
        mac(P + "M", r["m"]); mac(P + "N", r["n"]); mac(P + "SB", f(r["single_best"], 3)); mac(P + "Oracle", f(r["oracle"], 3))
        mac(P + "SBName", "\\texttt{" + r["single_best_model"].split("/")[-1] + "}")
        mac(P + "G", f(r["G"], 3)); mac(P + "MeanAcc", f(r["mean_acc"], 2)); mac(P + "AwGraded", r["allwrong_graded"])
        mac(P + "K", r["k"]); mac(P + "KUpper", r["k_upper"]); mac(P + "Beta", f(r["beta"], 3))
        mac(P + "BetaLo", f(r["beta_cp95"][0], 3)); mac(P + "BetaHi", f(r["beta_cp95"][1], 3))
        mac(P + "Cert", f(r["certified_max_gain"], 3)); mac(P + "SBLo", f(r["single_best_lower"], 3))
        mac(P + "RhoP", f(r["rho_pearson"], 2)); mac(P + "RhoTet", f(r["rho_tet"], 2)); mac(P + "BsfTet", f(r["beta_sf_tet"], 3))
        mac(P + "RatioTet", f(r["ratio_tet"], 0)); mac(P + "PartK", len(r["partial_allwrong"])); mac(P + "AnyN", r["n_any_answer"])
        if "unique_output" in r:
            u = r["unique_output"]; mac(P + "UniqN", u["n"]); mac(P + "UniqK", u["k"]); mac(P + "UniqBetaHi", f(u["beta_cp95"][1], 3))
            mac(P + "MultiAw", sum(1 for v in r["allwrong_class"].values() if v.startswith("multi-answer")))
        capped_left += sum(len(v) for v in r["capped_not_reasked"].values()); capped_left_q += len(r["capped_not_reasked"])
        tr = c["trace"].get(key)
        if tr:
            capped_june += sum(len(v) for v in tr["capped_not_reasked"].values())
            mac(P + "JuneK", tr["june_k"]); mac(P + "JuneN", tr["june_n"]); tot_june += tr["june_k"]
            mac(P + "JuneBeta", f(tr["june_k"] / tr["june_n"], 3)); mac(P + "JuneSB", f(tr["june_single_best"], 3))
            for cause, short in CAUSES.items():
                n = tr["counts"].get(cause, 0); mac(P + "June" + short, n); tot[short] += n
    mac("TotJune", tot_june)
    # the pools share some questions (MATH-500 in the 67-model and 15-model pools; MMLU-Pro likewise): distinct counts
    june_q = {q for tr in c["trace"].values() for qs in tr["causes"].values() for q in qs}
    gen_q = {q for tr in c["trace"].values() for q in tr["causes"].get("genuine co-failure", [])}
    mac("TotJuneQ", len(june_q)); mac("TotGenuineQ", len(gen_q))
    mac("GenuineQ", len(c["genuine_questions"])); mac("GenuineIllPosed", len(c["genuine_ill_posed_both"]))
    mac("TotPartK", sum(len(r["partial_allwrong"]) for r in c["market"].values()))
    # budget-capped answers inside all-wrong questions that could not be re-asked (model no longer served)
    mac("CappedLeft", capped_left)
    cr = c["corrupt"]
    mac("CorruptCells", cr["cells"]); mac("CorruptRequeried", cr["requeried"]); mac("CorruptFromCache", cr["from_cache"])
    mac("CorruptUsd", f(cr["requery_usd"], 2)); mac("CorruptFlip", cr["wrong_to_right"])
    mac("CorruptLlamaSmall", cr["by_model"].get("meta-llama/llama-3.2-3b-instruct", 0))
    mac("EmptyRequeried", cr["empty_requeried"]); mac("EmptyFlip", cr["empty_wrong_to_right"])
    co = c["costs"]; lm = co["ledger_usd_by_month"]
    mac("CostCore", f(co["core_pillars"], 2)); mac("CostMarket", f(co["market_scale"], 2)); mac("CostCode", f(co["code"], 2))
    mac("SpendJune", f(lm["2026-06"], 2)); mac("SpendAudit", f(lm.get("2026-09", 0.0), 2))
    mac("CallsJune", f"{co['ledger_calls_by_month']['2026-06']:,}".replace(",", "{,}"))
    for short, n in tot.items():
        mac("Tot" + short, n)
    a = c["artifact_math500"]
    mac("ArtK", a["k"]); mac("ArtN", a["n"]); mac("ArtM", a["m"]); mac("ArtBeta", f(a["beta"], 3))
    mac("ArtRhoTet", f(a["rho_tet"], 2)); mac("ArtBsfTet", f(a["beta_sf_tet"], 3)); mac("ArtRatioTet", f(a["ratio_tet"], 1))
    mac("ArtRatioLo", f(a["ratio_p05_p95"][0], 1)); mac("ArtRatioHi", f(a["ratio_p05_p95"][1], 1))
    mac("ArtRhoP", f(a["rho_pearson"], 2)); mac("ArtBsfP", f(a["beta_sf_pearson"], 4)); mac("ArtRatioP", f(a["beta"] / a["beta_sf_pearson"], 0))
    mac("ArtRhoEff", f(a["rho_eff"], 2)); fs = a["full_sigma"]
    mac("ArtFullBeta", f(fs["beta"], 3)); mac("ArtFullRatio", f(fs["ratio"], 2)); mac("ArtNegEig", fs["neg_eigenvalues"])
    mac("ArtRhoPSD", f(fs["rho_after_psd"], 2)); cl = a["clayton"]
    mac("ArtClayLam", f(cl["lambda_L"], 2)); mac("ArtClayBeta", f(cl["beta"], 3)); mac("ArtClayRatio", f(cl["ratio"], 1))
    comp = a["composition"]; ks = sorted(comp, key=int)
    mac("ArtCompStart", f(comp[ks[0]]["median"], 1)); mac("ArtCompEnd", f(comp[ks[-1]]["median"], 1)); mac("ArtCompPools", comp[ks[1]]["pools"])
    g = c["gpqa_open"]
    for key, P in [("v1_retired", "OpenOne"), ("v2_primary", "Open"), ("v2_secondary", "OpenSec")]:
        s = g[key]
        mac(P + "N", s["n"]); mac(P + "K", s["k"]); mac(P + "Beta", f(s["beta"], 3))
        mac(P + "BetaLo", f(s["beta_cp95"][0], 3)); mac(P + "BetaHi", f(s["beta_cp95"][1], 3))
        u = s["with_partial_items_upper"]
        mac(P + "UpN", u["n"]); mac(P + "UpK", u["k"]); mac(P + "UpBetaHi", f(u["beta_cp95"][1], 3))
        mac(P + "UpMinAns", u["min_answering"] if u["min_answering"] is not None else "--")
        mac(P + "UnanK", s["rules"]["unanimous"]["k"]); mac(P + "LenK", s["rules"]["lenient"]["k"])
        mc = s["mc_same_items"]; mac(P + "McN", mc["n"]); mac(P + "McK", mc["k"]); mac(P + "McBetaHi", f(mc["beta_cp95"][1], 3))
        mm = s["matched_models"]
        mac(P + "MatchModels", mm["models"]); mac(P + "McMean", f(mm["mc_mean"], 2)); mac(P + "OpenMean", f(mm["open_mean"], 2))
        mac(P + "McBest", f(mm["mc_best"], 2)); mac(P + "OpenBest", f(mm["open_best"], 2))
    mac("OpenSecFlagged", g["v2_secondary"]["allwrong_flagged_doubtful_gold"])
    mac("OpenOneOpt", g["v1_allwrong_reasons"]["option-dependent"]); mac("OpenOneDoubt", g["v1_allwrong_reasons"]["doubtful reference"])
    mac("KappaLo", f(g["v2_kappa_range"][0], 2)); mac("KappaHi", f(g["v2_kappa_range"][1], 2))
    for key, P in [("pool15_mix", "TaMix"), ("pool15_hard", "TaPro")]:
        r = c[key]
        mac(P + "N", r["n"]); mac(P + "SB", f(r["single_best"], 3)); mac(P + "Oracle", f(r["oracle"], 3)); mac(P + "G", f(r["G"], 3))
        mac(P + "SBName", "\\texttt{" + r["single_best_model"].split("/")[-1] + "}")
        mac(P + "GLo", f(r["G_ci95"][0], 3)); mac(P + "GHi", f(r["G_ci95"][1], 3))
        mac(P + "Rho", f(r["rho_pearson"], 3)); mac(P + "RhoIn", f(r["rho_within_family"], 3)); mac(P + "RhoX", f(r["rho_cross_family"], 3))
        mac(P + "RhoGap", f(r["rho_within_family"] - r["rho_cross_family"], 3))
    au = c.get("audit_spend")
    mac("AuditCost", f(au["total_usd"], 2) if au else "--")
    src = os.path.join(ROOT, "runs", "canonical.json")
    h = hashlib.sha256(open(src, "rb").read()).hexdigest()[:16] if os.path.exists(src) else "n/a"
    open(path, "w").write("\n".join([f"% GENERATED by harness/make_numbers.py from runs/canonical.json (sha256 {h}). Do not edit by hand."] + L) + "\n")
    print(f"[numbers] wrote {len(L)} macros to {path}")


if __name__ == "__main__":
    write(json.load(open(os.path.join(ROOT, "runs", "canonical.json"))), os.path.join(ROOT, "paper", "numbers.tex"))
