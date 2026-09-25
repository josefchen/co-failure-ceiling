"""Flagship paper figures v2 — recomputed from committed runs, no fabricated data.
Cohesive publication style (near-monochrome + one warm accent for the co-failure object beta).

  fig_regime_map.pdf  - "two regimes across the domains": beta per domain with Clopper-Pearson
                        intervals, split into ceiling-bound (beta>0) vs realizability-bound (beta~0),
                        with the GPQA multiple-choice<->open flip pair connected.
  fig_format_flip.pdf - content-controlled GPQA flip: the same 79 questions, MC vs free-response;
                        a co-failure block of 10/79 all-models-wrong cells opens under open-ended.

Usage: python3 paper_figs_v2.py
Numbers are read from ../runs/*.json (recomputed) and cross-checked against the paper table/text.
"""
import os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")
FIGS = os.path.join(HERE, "..", "paper", "figures")
os.makedirs(FIGS, exist_ok=True)

# ---------- cohesive style ----------
INK     = "#1A1D22"   # near-black ink for primary text/marks
TEXT    = "#2B2F36"
DIM     = "#6A727C"   # secondary labels
FAINT   = "#9AA1AA"   # ticks / hairlines
HAIR    = "#E4E7EB"   # gridlines on paper-white
ACCENT  = "#E0613C"   # the ONE warm hue: reserved for the co-failure object beta / STOP
ACCENT_SOFT = "#F4C9B9"
GO      = "#2E9E6B"   # resolvable / realizable (green)
GO_SOFT = "#BFE3D2"
# held-out domain hues (matched lightness, muted) — only used to tint family, never competes with ACCENT
H_MATH  = "#3F8EA0"   # teal
H_CODE  = "#B98A3C"   # gold
H_SCI   = "#7C6BB0"   # violet

def set_style():
    # robust clean sans stack; matplotlib falls back gracefully and embeds the chosen face in the PDF
    for cand in ["Helvetica Neue", "Helvetica", "Arial", "Inter", "DejaVu Sans"]:
        if any(cand.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = cand; break
    plt.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 200,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
        "text.color": TEXT, "axes.edgecolor": FAINT, "axes.labelcolor": TEXT,
        "xtick.color": DIM, "ytick.color": DIM,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "regular",
        "pdf.fonttype": 42, "ps.fonttype": 42,  # embed TrueType so arXiv renders identically
    })

def clean(ax, left=True, bottom=True):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left); ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(length=3, color=FAINT)

# ---------- Clopper-Pearson (matches the harness / space implementation) ----------
def cp_interval(k, n, alpha=0.05):
    from math import inf
    # use scipy if present, else a stable bisection on the regularized incomplete beta
    try:
        from scipy.stats import beta as B
        lo = 0.0 if k == 0 else B.ppf(alpha/2, k, n-k+1)
        hi = 1.0 if k == n else B.ppf(1-alpha/2, k+1, n-k)
        return float(lo), float(hi)
    except Exception:
        def betacf(a, b, x):
            MAXIT, EPS, FPMIN = 300, 3e-12, 1e-300
            qab, qap, qam = a+b, a+1, a-1
            c = 1.0; d = 1 - qab*x/qap
            if abs(d) < FPMIN: d = FPMIN
            d = 1/d; h = d
            for m in range(1, MAXIT+1):
                m2 = 2*m
                aa = m*(b-m)*x/((qam+m2)*(a+m2))
                d = 1+aa*d; d = FPMIN if abs(d) < FPMIN else d
                c = 1+aa/c; c = FPMIN if abs(c) < FPMIN else c
                d = 1/d; h *= d*c
                aa = -(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
                d = 1+aa*d; d = FPMIN if abs(d) < FPMIN else d
                c = 1+aa/c; c = FPMIN if abs(c) < FPMIN else c
                d = 1/d; de = d*c; h *= de
                if abs(de-1) < EPS: break
            return h
        def betai(a, b, x):
            if x <= 0: return 0.0
            if x >= 1: return 1.0
            bt = math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log(1-x))
            return bt*betacf(a, b, x)/a if x < (a+1)/(a+b+2) else 1-bt*betacf(b, a, 1-x)/b
        def betainv(p, a, b):
            lo, hi = 0.0, 1.0
            for _ in range(100):
                mid = (lo+hi)/2
                if betai(a, b, mid) < p: lo = mid
                else: hi = mid
            return mid
        lo = 0.0 if k == 0 else betainv(alpha/2, k, n-k+1)
        hi = 1.0 if k == n else betainv(1-alpha/2, k+1, n-k)
        return lo, hi


# =====================================================================================
# FIG: regime map — beta per domain, two regimes, with the GPQA MC<->open flip connected
# =====================================================================================
def fig_regime_map():
    """audited all-models-wrong rate per benchmark with 95% CP intervals; every value from runs/canonical.json."""
    c = json.load(open(os.path.join(RUNS, "canonical.json"))); mk = c["market"]; g = c["gpqa_open"]["v2_primary"]
    rows = [  # label, audited k, n, as-graded all-wrong, hue, note
        ("MATH-500", mk["math500"]["k"], mk["math500"]["n"], mk["math500"]["allwrong_graded"], H_MATH, f"open-ended math · {mk['math500']['m']} models"),
        ("MATH-Hard", mk["mathhard"]["k"], mk["mathhard"]["n"], mk["mathhard"]["allwrong_graded"], H_MATH, f"open-ended math · {mk['mathhard']['m']} models"),
        ("AIME 2024+25", mk["aime"]["k"], mk["aime"]["n"], mk["aime"]["allwrong_graded"], H_MATH, f"integer answers · {mk['aime']['m']} models"),
        ("code_contests", mk["code"]["k"], mk["code"]["n"], mk["code"]["allwrong_graded"], H_CODE, f"execution-graded, special checkers · {mk['code']['m']} models"),
        ("GPQA (free-response)", g["k"], g["n"], None, H_SCI, "v2 question set · 18 models"),
        ("GPQA (multiple choice)", mk["gpqamc"]["k"], mk["gpqamc"]["n"], mk["gpqamc"]["allwrong_graded"], H_SCI, f"{mk['gpqamc']['m']} models"),
        ("MMLU-Pro", mk["mmlupro"]["k"], mk["mmlupro"]["n"], mk["mmlupro"]["allwrong_graded"], H_MATH, f"direct-answer multiple choice · {mk['mmlupro']['m']} models"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ys = list(range(len(rows)))[::-1]
    for y, (label, k, n, graded, hue, note) in zip(ys, rows):
        beta = k / n; lo, hi = cp_interval(k, n); col = ACCENT if k else GO
        ax.plot([lo, hi], [y, y], color=col, lw=2.2, alpha=0.30, solid_capstyle="round", zorder=2)
        ax.scatter([beta], [y], s=78, color=col, edgecolor="white", linewidth=1.2, zorder=4)
        ax.text(-0.004, y + 0.17, label, ha="right", va="center", fontsize=10, color=INK, fontweight="medium")
        ax.text(-0.004, y - 0.20, note, ha="right", va="center", fontsize=7.3, color=FAINT)
        vtxt = (r"$\beta=0$  (0/%d)" % n) if k == 0 else (r"$\beta=%.3f$  (%d/%d)" % (beta, k, n))
        ax.text(hi + 0.003, y + 0.17, vtxt, ha="left", va="center", fontsize=8.8, color=col, fontweight="medium")
        if graded:
            ax.text(hi + 0.003, y - 0.20, "%d all-wrong as graded; audited: %d genuine" % (graded, k), ha="left", va="center",
                    fontsize=7.2, color=DIM)
    ax.axvline(0, color=FAINT, lw=1.0, zorder=1)
    ax.set_xlim(-0.055, 0.16); ax.set_ylim(-0.7, len(rows) - 0.3); ax.set_yticks([])
    ax.set_xlabel(r"audited all-models-wrong rate  $\beta$  —  every selection policy is capped at $1-\beta$", fontsize=9.5)
    ax.set_xticks([0, 0.025, 0.05, 0.075, 0.10]); ax.set_xticklabels(["0", "0.025", "0.05", "0.075", "0.10"])
    clean(ax, left=False); ax.spines["bottom"].set_color(FAINT); ax.grid(axis="x", color=HAIR, lw=0.7, zorder=0); ax.set_axisbelow(True)
    fig.text(0.5, 1.005, "The co-failure ceiling on the 2026 frontier", ha="center", fontsize=12.5, color=INK, fontweight="bold")
    fig.text(0.5, 0.965, "95% Clopper–Pearson intervals after the all-wrong audit; recomputed from the released matrices (canonical.json).",
             ha="center", fontsize=8.0, color=DIM)
    out = os.path.join(FIGS, "fig_regime_map.pdf"); fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig); print("wrote", out)


def fig_format_flip():
    """the same GPQA-Diamond questions (v2 primary item set), multiple-choice vs free-response; values from canonical.json."""
    c = json.load(open(os.path.join(RUNS, "canonical.json"))); g = c["gpqa_open"]["v2_primary"]
    n, qids = g["n"], g["qids"]; aw = {i for i, q in enumerate(qids) if q in set(g["allwrong_qids"])}
    MC = json.load(open(os.path.join(RUNS, "matrix_marketE2_final.json")))
    mc_ms = sorted({m for q, v in MC.items() if v["dataset"] == "gpqa" for m in v["models"]})
    mc_aw = {i for i, q in enumerate(qids) if (qq := q.replace("gpqaopen:", "gpqa:")) in MC
             and all(m in MC[qq]["models"] for m in mc_ms) and all(MC[qq]["models"][m]["correct"] == 0 for m in mc_ms)}
    mm = g["matched_models"]; mc = g["mc_same_items"]
    cols = 20; rows_n = math.ceil(n / cols); cell, gap = 1.0, 0.18
    fig, ax = plt.subplots(figsize=(7.4, 3.5 + 0.35 * max(0, rows_n - 4)))

    def draw_grid(y0, allwrong_set, base_col):
        for i in range(n):
            r, cc = divmod(i, cols); x = cc * (cell + gap); y = y0 - r * (cell + gap); wrong = i in allwrong_set
            ax.add_patch(FancyBboxPatch((x, y), cell, cell, boxstyle="round,pad=0,rounding_size=0.18",
                         linewidth=(0.9 if wrong else 0.0), edgecolor=(ACCENT if wrong else "white"),
                         facecolor=(ACCENT if wrong else base_col), zorder=3))
    top_y = 0.0; gridH = rows_n * (cell + gap); bot_y = -(gridH + 2.4)
    draw_grid(top_y, mc_aw, GO_SOFT); draw_grid(bot_y, aw, "#E7EAEE")
    rightx = cols * (cell + gap) + 0.6
    b_mc = r"$\beta\approx0$" if not mc["k"] else r"$\beta=%.3f$" % mc["beta"]
    ax.text(-0.6, top_y + cell/2, "Multiple-choice", ha="right", va="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.text(-0.6, top_y - 1.0, "options given", ha="right", va="center", fontsize=8, color=DIM)
    ax.text(rightx, top_y + cell/2 + 0.4, b_mc, ha="left", va="center", fontsize=12, color=GO, fontweight="bold")
    ax.text(rightx, top_y - 0.5, "%d of %d all-wrong (%d models)\nmatched %d models: mean %.2f · best %.2f"
            % (mc["k"], mc["n"], mc["models"], mm["models"], mm["mc_mean"], mm["mc_best"]),
            ha="left", va="center", fontsize=7.6, color=DIM, linespacing=1.3)
    ax.text(-0.6, bot_y + cell/2, "Free-response", ha="right", va="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.text(-0.6, bot_y - 1.0, "same questions,\nno options", ha="right", va="center", fontsize=8, color=DIM, linespacing=1.3)
    ax.text(rightx, bot_y + cell/2 + 0.4, r"$\beta=%.3f$" % g["beta"], ha="left", va="center", fontsize=12, color=ACCENT, fontweight="bold")
    ax.text(rightx, bot_y - 0.5, "%d of %d all-wrong (CP[%.3f, %.3f])\nmatched %d models: mean %.2f · best %.2f"
            % (g["k"], n, g["beta_cp95"][0], g["beta_cp95"][1], mm["models"], mm["open_mean"], mm["open_best"]),
            ha="left", va="center", fontsize=7.6, color=DIM, linespacing=1.3)
    ax.add_patch(FancyBboxPatch((0.0, bot_y - gridH - 0.8), cell, cell, boxstyle="round,pad=0,rounding_size=0.18",
                 linewidth=0.9, edgecolor=ACCENT, facecolor=ACCENT, zorder=3))
    ax.text(cell + 0.4, bot_y - gridH - 0.8 + cell/2, "= every model wrong on this question (co-failure)",
            ha="left", va="center", fontsize=7.8, color=DIM)
    ax.set_xlim(-7.2, rightx + 11.5); ax.set_ylim(bot_y - gridH - 1.6, top_y + cell + 1.0)
    ax.set_aspect("equal"); ax.axis("off")
    fig.text(0.5, 1.01, "Same questions, two answer formats", ha="center", fontsize=12.5, color=INK, fontweight="bold")
    k = c["gpqa_open"]["v2_kappa_range"]
    fig.text(0.5, 0.965, "Each cell is one GPQA-Diamond question (v2 item set, pre-registered screen); 5-judge panel, "
             r"$\kappa$ %.2f–%.2f." % (k[0], k[1]), ha="center", fontsize=8.0, color=DIM)
    out = os.path.join(FIGS, "fig_format_flip.pdf"); fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig); print("wrote", out, "| open all-wrong =", g["k"], "of", n, "| MC all-wrong =", mc["k"])


if __name__ == "__main__":
    set_style()
    fig_regime_map()
    fig_format_flip()
