"""Pillar A-C and churn numbers (2026-09-25): re-run every offline pillar analysis on the final released inputs and
write paper/numbers_pillars.tex (one LaTeX macro per reported number).

Each script below was first checked to reproduce its v1 output bit-for-bit from the v1 inputs (DATA_AUDIT.md), then run on
the v2 inputs: every matrix, fusion sample set and cascade record re-scored from the logged responses by the corrected
grader (regrade_final.py), with corrupt responses re-queried and the truncation control applied. The LLM-as-router needs
live model calls; its per-query picks are written to router_llm_stageA2v3_final.json (replayed from the response cache on
the final matrix), from which its reported numbers are read.

Usage: python3 pillars.py
"""
import os, sys, json, subprocess, hashlib

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); RUNS = os.path.join(ROOT, "runs")
PY = sys.executable
MIX = "stageA2v3_final"         # final 15-model saturated mix (corrected grader, pillar-era budgets)
SAT_B = "stageA2v3_final"       # Pillar B saturated pool: the same final mix (v1 used the un-re-graded stageA2v2)
HARD = "hardAv3_final"
STEPS = [
    ["router_baseline.py", "--tag", MIX, "--datasets", "gsm8k,mmlu,math500,arc", "--n", "120"],
    ["router_strong.py", "--tag", MIX, "--datasets", "gsm8k,mmlu,math500,arc", "--n", "120"],
    ["router_baseline.py", "--tag", HARD, "--datasets", "mmlu_pro", "--n", "200"],
    ["rho_fusion_test.py", "--tag", HARD],
    ["rho_fusion_test.py", "--tag", SAT_B],
    ["equal_quality.py", "--tag", "eqq2_final"],
    ["equal_quality.py", "--tag", "eqq2_math_final"],
    ["eqq_robustness.py", "--tag", "eqq2_final"],
    ["-c", "import fusion; fusion.analyze('eqq2_final')"],
    ["kstar_calib.py", "--tag", "eqq2_final"],
    ["kstar_calib.py", "--tag", "eqqA_final"],
    ["-c", "import cascade; cascade.analyze('stageC2v3_final')"],
    ["-c", "import cascade; cascade.analyze('advC_final')"],
    ["cascade_heldout.py", "stageC2v3_final"],
    ["churn_analysis.py", "--tag", "churnD_final", "--dataset", "mmlu_pro"],
    ["churn_analysis.py", "--tag", "churnD_final", "--dataset", "gsm8k"],
    ["copula_dichotomy.py"],
]


READ = []


def J(name):
    READ.append(name)
    return json.load(open(os.path.join(RUNS, name)))


def f(x, d, sign=False):
    return ("+" if sign and x > 0 else "") + f"{x:.{d}f}"


def main():
    for s in STEPS:
        subprocess.run([PY] + s, cwd=HERE, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    L = []
    def mac(k, v):
        assert k.isalpha(), k
        L.append(f"\\newcommand{{\\n{k}}}{{{v}}}")
    r = J(f"router_{MIX}.json")
    mac("RtSB", f(r["V_sb"], 3)); mac("RtLearned", f(r["V_learned_router"], 3)); mac("RtOracle", f(r["V_oracle"], 3))
    mac("RtFracPct", f(100 * r["frac_G_captured"], 0)); mac("RtFracLo", f(r["frac_ci"][0], 2)); mac("RtFracHi", f(r["frac_ci"][1], 2))
    rh = J(f"router_{HARD}.json")
    mac("RtHardFrac", f(rh["frac_G_captured"], 2)); mac("RtHardN", rh["n_test"]); mac("RtN", r["n_test"])
    mac("RtHardLost", round((rh["V_sb"] - rh["V_learned_router"]) * rh["n_test"])); mac("RtLost", round((r["V_sb"] - r["V_learned_router"]) * r["n_test"]))
    mac("RtFrac", f(r["frac_G_captured"], 2))
    s = J(f"router_strong_{MIX}.json")["routers"]
    mac("RtGbm", f(s["gbm_permodel"]["frac_G"], 2)); mac("RtMulti", f(s["gbm_multiclass"]["frac_G"], 2))
    llm = J(f"router_llm_{MIX}.json")
    mac("RtLlmFrac", f(llm["frac_G_captured"], 0)); mac("RtLlmRoutePct", f(100 * llm["routed_to_single_best_frac"], 0))
    b = J(f"rhofus_{HARD}.json"); bs = J(f"rhofus_{SAT_B}.json")
    mac("BSlope", f(b["rho_coef"], 2, True)); mac("BSlopeLo", f(b["rho_ci_jackknife"][0], 2, True)); mac("BSlopeHi", f(b["rho_ci_jackknife"][1], 2, True))
    mac("BGainHard", f(b["mean_gain"], 2)); mac("BGainSat", f(bs["mean_gain"], 2))
    mac("BGainLooHardMax", f(b["mean_gain_loo_range"][1], 3)); mac("BGainLooSatMax", f(bs["mean_gain_loo_range"][1], 3))
    mac("BNegHardPct", f(100 * b["frac_triplets_negative"], 0)); mac("BNegSatPct", f(100 * bs["frac_triplets_negative"], 0))
    mac("BWidthRatio", f((b["rho_ci_jackknife"][1] - b["rho_ci_jackknife"][0]) / (b["rho_ci_iid"][1] - b["rho_ci_iid"][0]), 1))
    e = J("eqq_eqq2_final.json"); em = J("eqq_eqq2_math_final.json"); er = J("eqq_robustness_eqq2_final.json")
    k3 = next(x for x in e["rows"] if x["k"] == 3); m3 = next(x for x in em["rows"] if x["k"] == 3)
    mm = max(em["rows"], key=lambda x: x["diff"])
    mac("EqRhoSelf", f(e["rho_selfmoa"], 2)); mac("EqRhoHet", f(e["rho_hetero"], 2))
    mac("EqMemLo", f(min(e["member_acc"]), 2)); mac("EqMemHi", f(max(e["member_acc"]), 3))
    fa = next(x for x in J("fusion_analysis_eqq2_final.json")["rows"] if x["k"] == 3)
    mac("EqAltAgg", f(fa["hetero_acc"] - fa["selfmoa_acc"], 3, True))
    mac("EqKThree", f(k3["diff"], 3, True)); mac("EqKThreeLo", f(k3["diff_ci"][0], 3, True)); mac("EqKThreeHi", f(k3["diff_ci"][1], 3, True))
    mac("EqMean", f(er["mean"], 3, True)); mac("EqMin", f(er["min"], 3, True)); mac("EqMax", f(er["max"], 3, True))
    mac("EqTrials", er["trials"]); mac("EqPosPct", f(100 * er["frac_positive"], 0))
    mac("EqMathRho", f(em["rho_hetero"], 2)); mac("EqMathKThree", f(m3["diff"], 3, True))
    mac("EqMathKThreeLo", f(m3["diff_ci"][0], 3, True)); mac("EqMathKThreeHi", f(m3["diff_ci"][1], 3, True))
    mac("EqMathMax", f(mm["diff"], 3, True)); mac("EqMathMaxK", mm["k"])
    em_ = max(e["rows"], key=lambda x: x["diff"]); sig = [x["k"] for x in e["rows"] if x["diff_ci"][0] > 0]
    mac("EqBestDiff", f(em_["diff"], 3, True)); mac("EqBestK", em_["k"])
    mac("EqSigK", ",".join(str(k) for k in sig) if sig else "none"); mac("EqKMax", max(x["k"] for x in e["rows"]))
    k6, k9 = J("kstar_eqq2_final.json"), J("kstar_eqqA_final.json")
    mac("KsBands", k6["n_bands"]); mac("KsBandsNine", k9["n_bands"])
    mac("KsSlopeSix", f(k6["rho_coef"], 3, True)); mac("KsSlopeSixLo", f(k6["rho_coef_ci"][0], 2, True)); mac("KsSlopeSixHi", f(k6["rho_coef_ci"][1], 2, True))
    mac("KsSlopeNine", f(k9["rho_coef"], 3, True)); mac("KsSlopeNineLo", f(k9["rho_coef_ci"][0], 2, True)); mac("KsSlopeNineHi", f(k9["rho_coef_ci"][1], 2, True))
    c = J("cascade_analysis_stageC2v3_final.json"); d = c["degradation"]; h = J("cascade_heldout_stageC2v3_final.json")
    mac("CaAL", f(c["a_L"], 3)); mac("CaAH", f(c["a_H"], 3)); mac("CaAUC", f(c["auc"], 3)); mac("CaCeil", f(c["ceiling"], 3))
    mac("CaGapStart", f(d[0]["max_gap"], 3)); mac("CaGapEnd", f(d[-1]["max_gap"], 3)); mac("CaAUCEnd", f(d[-1]["auc"], 3))
    mac("CaGapStdMax", f(max(x["max_gap_std"] for x in d), 3)); mac("CaSeeds", d[0]["seeds"])
    def dominance(a):          # interior cascade operating points (0 < escalation < 1) that undercut H-only on dollars per correct
        pts = [r for r in a["rows"] if 0 < r["phi"] < 1]
        return sum(r["dpc"] < a["dpc_H"] for r in pts), len(pts)
    mac("CaDomPts", dominance(c)[0]); mac("CaBandPts", dominance(c)[1])
    cc = J("cascade_analysis_advC_final.json")
    mac("CcAH", f(cc["a_H"], 3)); mac("CcDomPts", dominance(cc)[0]); mac("CcBandPts", dominance(cc)[1])
    ccq = max(cc["rows"], key=lambda r: r["Q"])
    mac("CcQMax", f(ccq["Q"], 3)); mac("CcDpcRatio", f(min(r["dpc"] for r in cc["rows"] if 0 < r["phi"] < 1) / cc["dpc_H"], 1))
    mac("CaHeld", f(h["mean_advantage_vs_random_mix"], 3, True)); mac("CaHeldSd", f(h["sd_advantage"], 3)); mac("CaHeldAUC", f(h["mean_heldout_auc"], 3))
    ch = J("churn_churnD_final_mmlu_pro.json"); cg = J("churn_churnD_final_gsm8k.json")
    mac("ChDropLo", f(min(ch["cpc_drop_factor"], cg["cpc_drop_factor"]), 0)); mac("ChDropHi", f(max(ch["cpc_drop_factor"], cg["cpc_drop_factor"]), 0))
    mac("ChOptPro", f(ch["broad_access_advantage"], 2, True)); mac("ChOptGsm", f(cg["broad_access_advantage"], 2, True))
    cd = J("copula_dichotomy.json")["cases"]
    c2, c4, mx = cd["clayton_theta2.0"], cd["clayton_theta4.0"], cd["mixture_pi0.05"]
    mac("DiLamA", f(c2["lambda_L"], 2)); mac("DiRatioA", f(c2["curve"][-1]["ratio"], 1)); mac("DiM", c2["curve"][-1]["m"])
    mac("DiLamB", f(c4["lambda_L"], 2)); mac("DiRatioB", f(c4["curve"][-1]["ratio"], 1))
    mac("DiRhoA", f(c2["curve"][-1]["rho_bar"], 2)); mac("DiRhoAtom", f(mx["curve"][-1]["rho_bar"], 2))
    mac("DiAtomExp", int(__import__("math").floor(__import__("math").log10(mx["curve"][-1]["ratio"]))))
    h8 = hashlib.sha256(b"".join(open(os.path.join(RUNS, p), "rb").read() for p in sorted(set(READ)))).hexdigest()[:16]
    out = os.path.join(ROOT, "paper", "numbers_pillars.tex")
    open(out, "w").write("\n".join([f"% GENERATED by harness/pillars.py (outputs digest {h8}). Do not edit by hand."] + L) + "\n")
    print(f"[pillars] wrote {len(L)} macros to {out}")


if __name__ == "__main__":
    main()
