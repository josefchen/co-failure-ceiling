"""Flagship Fig.2: the co-failure residual survives the FULL pairwise-tetrachoric Gaussian copula
(left), and grows with pool size as a composition-robust band (right). Reads runs/canonical.json
(harness/canonical.py). Writes paper/figures/fig_realizability.pdf.

Styling shared via figstyle.py: the predictions are cool/neutral, the empirical beta is the one warm
accent. No fabricated data; every value is read from the committed runs.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle as S

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")
FIGS = os.path.join(HERE, "..", "paper", "figures")

S.set_style()
cn = json.load(open(os.path.join(RUNS, "canonical.json")))   # harness/canonical.py (exact estimators)
A = cn["artifact_math500"]    # the June MATH-500 "tail": grading artifacts, analysed as a pairwise model saw them
m5 = {"beta_single_factor": A["beta_sf_tet"], "beta_full_sigma_mc": A["full_sigma"]["beta"],
      "beta_emp": A["beta"], "ratio_vs_full_sigma": A["full_sigma"]["ratio"]}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.5))

# --- left: nested predictions for MATH-500 (predictions cool/neutral; empirical beta = warm accent) ---
labels = ["single-factor\n$\\beta(\\bar\\rho)$", "full-$\\Sigma$ Gaussian\n(all pairs, MC)", "June tail\n(artifacts)"]
vals = [m5["beta_single_factor"], m5["beta_full_sigma_mc"], m5["beta_emp"]]
colors = [S.H_MATH, S.INKBLUE, S.ACCENT]
xb = np.arange(3)
ax1.bar(xb, vals, 0.62, color=colors, zorder=3)
for i, v in enumerate(vals):
    ax1.annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=8.5, color=S.INK, xytext=(0, 2),
                 textcoords="offset points")
# bracket for the residual ratio between full-Sigma and empirical
ax1.annotate("", xy=(2, m5["beta_emp"]), xytext=(1, m5["beta_emp"]),
             arrowprops=dict(arrowstyle="<->", color=S.DIM, lw=1.0))
ax1.text(1.5, m5["beta_emp"] * 1.05, f"{m5['ratio_vs_full_sigma']:.2f}× the full\npairwise Gaussian",
         ha="center", va="bottom", fontsize=8, fontweight="bold", color=S.ACCENT_DK, linespacing=1.2)
ax1.set_xticks(xb); ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylabel("co-failure probability $\\beta$")
ax1.set_ylim(0, m5["beta_emp"] * 1.34)
S.clean(ax1, grid="y")
S.titled(ax1, "An artifact atom, invisible to pairwise ρ", "June MATH-500 tail (grading artifacts) vs. copula predictions")

# --- right: composition-bootstrapped pool-size curve ---
cc = A["composition"]
ks = sorted(int(k) for k in cc)
med = [cc[str(k)]["median"] for k in ks]
lo = [cc[str(k)]["p05"] for k in ks]
hi = [cc[str(k)]["p95"] for k in ks]
ax2.fill_between(ks, lo, hi, color=S.H_MATH, alpha=0.16, lw=0, label="5–95% over random $k$-subsets")
ax2.plot(ks, med, "o-", color=S.H_MATH, lw=1.8, ms=4.5, mec="white", mew=0.6, zorder=3, label="median ratio")
ax2.axhline(1.0, ls=":", color=S.FAINT, lw=0.9)
ax2.set_xlabel("pool size $k$ (random market sub-pools)")
ax2.set_ylabel(r"tetrachoric underpricing $\beta/\beta_{\mathrm{sf}}$")
ax2.legend(loc="upper left")
S.clean(ax2, grid="y")
S.titled(ax2, "Its mispricing grows with the pool", "random sub-pools of the 67 models; June grading")

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_realizability.pdf"))
plt.close(fig)
print("wrote fig_realizability.pdf  (full-Sigma %.4f vs empirical %.4f; curve k=%d..%d -> %.2f..%.2f)"
      % (m5["beta_full_sigma_mc"], m5["beta_emp"], ks[0], ks[-1], med[0], med[-1]))
