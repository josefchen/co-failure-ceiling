"""Generate paper figures from logged runs (no fabricated data).
  fig_cost_quality.pdf  - per-model accuracy vs $/correct + oracle/single-best/random
  fig_cascade_collapse.pdf - verifier AUC vs cascade advantage over random mixing
  fig_rho_gain.pdf      - ceiling-free diversification test: (1-rho) vs normalized fusion gain (bucketed, 95% CI)
Usage: python3 figures.py --matrix matrix_stageA2 --cascade stageC2 --bstats hardB

Styling is shared via figstyle.py (cohesive near-monochrome + one warm accent for beta); only
colours/spines/labels are styled here, every plotted number is read from ../runs/*.json.
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import figstyle as S

HERE = os.path.dirname(__file__); RUNS = os.path.join(HERE, "..", "runs")
FIGS = os.path.join(HERE, "..", "paper", "figures")
os.makedirs(FIGS, exist_ok=True)

def families():
    fam = {}
    with open(os.path.join(HERE, "..", "model_registry.csv")) as f:
        for row in csv.DictReader(r for r in f if not r.startswith("#")):
            fam[row["model_id"]] = row["provider_family"]
    return fam

_SHORT = {  # headliner labels for the 2026 frontier pool (others shown as family-colored dots)
    "openai/gpt-5.5": "GPT-5.5", "openai/gpt-5.4": "GPT-5.4", "openai/gpt-5.2": "GPT-5.2",
    "anthropic/claude-opus-4.8": "Claude Opus 4.8", "anthropic/claude-sonnet-4.6": "Claude Sonnet 4.6",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro", "x-ai/grok-4.3": "Grok-4.3",
    "z-ai/glm-5.2": "GLM-5.2", "qwen/qwen3.7-max": "Qwen3.7-Max", "moonshotai/kimi-k2.7-code": "Kimi K2.7",
    "deepseek/deepseek-v4-pro": "DeepSeek V4", "minimax/minimax-m3": "MiniMax M3",
    "mistralai/mistral-medium-3-5": "Mistral-Med 3.5", "nvidia/nemotron-3-ultra-550b-a55b": "Nemotron-3-Ultra",
    "google/gemini-3.5-flash": "Gemini 3.5 Flash", "meta-llama/llama-4-maverick": "Llama-4",
    "openai/gpt-5-nano": "GPT-5-nano"}
def _short(m): return _SHORT.get(m, m.split("/")[-1])

def fig_cost_quality(mtag):
    R = json.load(open(os.path.join(RUNS, f"matrix_{mtag}.json")))
    fam = families()
    models = sorted({m for v in R.values() for m in v["models"]})
    qids = [q for q, v in R.items() if all(m in v["models"] for m in models)]
    M = np.array([[R[q]["models"][m]["correct"] for q in qids] for m in models], float)
    cost = np.array([[R[q]["models"][m]["cost"] for q in qids] for m in models], float)
    acc = M.mean(1); cpc = cost.mean(1) / np.clip(acc, 1e-9, None)
    oracle = (M.sum(0) > 0).mean(); sb = acc.max()
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.set_xscale("log")  # set BEFORE reading get_xlim() for edge-anchored labels, so the limit is a valid positive log value
    fams = sorted(set(fam.get(m, "?") for m in models))
    pal = S.muted_qualitative(len(fams)); cmap = {f: pal[i] for i, f in enumerate(fams)}
    # headroom band: single-best -> per-query oracle (the gain G lives here)
    ax.axhspan(sb, oracle, color=S.GO_SOFT, alpha=0.45, zorder=0, lw=0)
    lab_i = 0
    for i, m in enumerate(models):
        head = m in _SHORT
        ax.scatter(cpc[i], acc[i], color=cmap[fam.get(m, "?")], s=(54 if head else 34),
                   edgecolor="white", linewidth=0.6, zorder=(4 if head else 3), alpha=0.95)
        if head:  # only annotate headliners to avoid clutter at 67 models
            dy = 9 if (lab_i % 2 == 0) else -13; lab_i += 1
            ax.annotate(_short(m), (cpc[i], acc[i]), textcoords="offset points", xytext=(0, dy),
                        ha="center", fontsize=6.2, color=S.INK, zorder=5)
    ax.axhline(oracle, ls="--", color=S.GO, lw=1.3, zorder=2)
    ax.axhline(sb, ls=":", color=S.INK, lw=1.1, zorder=2)
    x1 = ax.get_xlim()[1]
    ax.text(x1, oracle, f"per-query oracle {oracle:.3f} ", color=S.GO, fontsize=7.5, va="bottom", ha="right", fontweight="bold")
    ax.text(x1, sb, f"single-best {sb:.3f} ", color=S.INK, fontsize=7.5, va="top", ha="right")
    xg = cpc.min()*1.15
    ax.annotate("", xy=(xg, oracle), xytext=(xg, sb), arrowprops=dict(arrowstyle="<->", color=S.GO, lw=1.2))
    ax.text(xg*1.18, (oracle+sb)/2, f"oracle gain $G$ = {oracle-sb:.3f}", color=S.GO,
            fontsize=7.5, rotation=90, va="center", fontweight="bold")
    ax.set_xlabel("cost per correct answer (USD, log scale)"); ax.set_ylabel("accuracy")
    handles = [plt.Line2D([0], [0], marker='o', ls='', color=cmap[f], label=f, ms=5.5,
               markeredgecolor="white", markeredgewidth=0.5) for f in fams]
    leg = ax.legend(handles=handles, fontsize=5.6, loc="lower right", ncol=4,
                    title="provider family", title_fontsize=6.5, labelcolor=S.TEXT,
                    handletextpad=0.3, columnspacing=0.8, borderaxespad=0.4)
    leg.get_title().set_color(S.DIM)
    S.clean(ax, grid="y")
    S.titled(ax, "The 2026 frontier pool: cost–quality and the oracle gap",
             "each dot a model (hue = provider family); the band spans single-best to the per-query oracle")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_cost_quality.pdf")); plt.close(fig)
    print("wrote fig_cost_quality.pdf")

def fig_cascade_collapse(ctag):
    d = json.load(open(os.path.join(RUNS, f"cascade_analysis_{ctag}.json")))
    deg = d["degradation"]; auc = [r["auc"] for r in deg]; gap = [r["max_gap"] for r in deg]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot(auc, gap, "o-", color=S.ACCENT, lw=1.8, ms=5, mec="white", mew=0.6, zorder=3)
    ax.axvline(0.5, ls="--", color=S.FAINT, lw=1, zorder=1)
    ax.axhline(0.0, ls=":", color=S.FAINT, lw=0.9, zorder=1)
    ax.annotate("AUC = 1/2\n(collapse to random mixing)", (0.5, max(gap) * 0.6), fontsize=7.2,
                color=S.DIM, ha="center", linespacing=1.3)
    ax.set_xlabel("verifier AUC (error-detector)"); ax.set_ylabel("cascade advantage over random mixing")
    ax.invert_xaxis()
    S.clean(ax, grid="both")
    S.titled(ax, "Cascade collapse identity", "advantage over random mixing vanishes as the verifier degrades to chance (Pillar C)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_cascade_collapse.pdf")); plt.close(fig)
    print("wrote fig_cascade_collapse.pdf")

def fig_rho_gain(btag):
    # 455-triplet test: gain vs rho, colored by headroom, with OLS-controlled slope reported
    d = json.load(open(os.path.join(RUNS, f"rhofus_{btag}.json")))
    sc = d["scatter"]
    rho = np.array([s["rho"] for s in sc]); gain = np.array([s["gain"] for s in sc])
    head = np.array([s["headroom"] for s in sc])
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    scat = ax.scatter(rho, gain, c=head, cmap=S.SEQ, s=11, alpha=0.75, edgecolor="none", zorder=3)
    cb = fig.colorbar(scat); S.style_cbar(cb, "accuracy headroom (1 − best member)")
    ax.axhline(0, ls=":", color=S.FAINT, lw=0.9, zorder=1)
    co = np.polyfit(rho, gain, 1); xs = np.linspace(rho.min(), rho.max(), 20)
    ax.plot(xs, np.polyval(co, xs), "-", color=S.ACCENT, lw=2.0, zorder=4,
            label=f"OLS slope (ctrl headroom) = {d['rho_coef']:+.2f}")
    ax.set_xlabel(r"inter-model error correlation  $\rho$")
    ax.set_ylabel("majority-vote gain over best member")
    ax.legend(loc="lower right")
    S.clean(ax, grid="y")
    S.titled(ax, f"Naive heterogeneous fusion", f"{d['n_triplets']} triplets; majority-vote gain over the best member vs error correlation, controlling for headroom")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_rho_gain.pdf")); plt.close(fig)
    print("wrote fig_rho_gain.pdf")

def fig_equal_quality(tag):
    d = json.load(open(os.path.join(RUNS, f"eqq_{tag}.json")))
    r = d["rows"]; ks = [x["k"] for x in r]
    sm = [x["selfmoa"] for x in r]; ht = [x["hetero"] for x in r]
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.plot(ks, sm, "o-", color=S.ACCENT, lw=1.8, ms=5, mec="white", mew=0.6, zorder=3,
            label=fr"Self-MoA (high $\rho$ = {d['rho_selfmoa']:.2f})")
    ax.plot(ks, ht, "s-", color=S.INKBLUE, lw=1.8, ms=5, mec="white", mew=0.6, zorder=3,
            label=fr"Heterogeneous (low $\rho$ = {d['rho_hetero']:.2f})")
    for x in r:
        if x["diff_ci"][0] > 0:
            ax.annotate("*", (x["k"], x["hetero"] + 0.004), ha="center", color=S.INKBLUE, fontsize=12)
    ax.set_xlabel(r"ensemble size $k$ (matched member quality)")
    ax.set_ylabel("majority-vote accuracy")
    ax.legend(loc="lower right")
    S.clean(ax, grid="y")
    S.titled(ax, "Equal-quality break-even", r"heterogeneous fusion vs Self-MoA at matched member quality ($\ast$: 95% CI excludes 0) — Pillar B")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_equal_quality.pdf")); plt.close(fig)
    print("wrote fig_equal_quality.pdf")

def fig_router(tag):
    d = json.load(open(os.path.join(RUNS, f"router_{tag}.json")))
    fl = np.array(d["frontier_learned"]); fo = np.array(d["frontier_oracle"])
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    fo = fo[np.argsort(fo[:, 0])]; fl = fl[np.argsort(fl[:, 0])]
    ax.plot(fo[:, 0], fo[:, 1], "-", color=S.GO, lw=2.0, zorder=3, label="cost-aware oracle (optimal)")
    ax.plot(fl[:, 0], fl[:, 1], "s--", color=S.INKBLUE, ms=3.5, lw=1.5, zorder=3, label="learned router (TF-IDF + domain)")
    ax.axhline(d["V_sb"], ls=":", color=S.INK, lw=1.1, zorder=2, label=f"single-best ({d['V_sb']:.3f})")
    ax.set_xscale("log"); ax.set_xlabel("cost per query (USD, log)"); ax.set_ylabel("accuracy")
    ax.legend(loc="lower right")
    S.clean(ax, grid="y")
    S.titled(ax, "Routing frontier", fr"the learned router captures {d['frac_G_captured']*100:.0f}% of the oracle gain $G$")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_router.pdf")); plt.close(fig)
    print("wrote fig_router.pdf")

def fig_kstar(tag):
    d = json.load(open(os.path.join(RUNS, f"kstar_{tag}.json")))
    sc = d["scatter"]; rho = np.array([s["rho"] for s in sc]); gain = np.array([s["gain"] for s in sc])
    bm = np.array([s["best_member"] for s in sc])
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    s = ax.scatter(rho, gain, c=bm, cmap=S.SEQ, s=15, alpha=0.8, edgecolor="none", zorder=3)
    cb = fig.colorbar(s); S.style_cbar(cb, "best-member accuracy")
    ax.axhline(0, ls=":", color=S.FAINT, lw=0.9, zorder=1)
    co = np.polyfit(rho, gain, 1); xs = np.linspace(rho.min(), rho.max(), 20)
    ax.plot(xs, np.polyval(co, xs), "-", color=S.ACCENT, lw=2.0, zorder=4,
            label=fr"$\rho$ coef (ctrl acc) = {d['rho_coef']:+.2f}")
    ax.set_xlabel(r"inter-model error correlation $\rho$"); ax.set_ylabel("majority-vote gain over best member")
    ax.legend(loc="upper right")
    S.clean(ax, grid="y")
    S.titled(ax, r"$k^\star(\rho)$ at matched quality", f"{d['n_bands']} quality-matched bands")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_kstar.pdf")); plt.close(fig)
    print("wrote fig_kstar.pdf")

def fig_churn(tag):
    d = json.load(open(os.path.join(RUNS, f"churn_{tag}_mmlu_pro.json")))
    traj = d["trajectory"]
    import datetime
    xs = [datetime.date.fromisoformat(t[0]) for t in traj]; ys = [t[1] for t in traj]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.step(xs, ys, where="post", color=S.INK, lw=1.9, zorder=3, label="ROUTE: best available (frontier)")
    ax.axhline(d["commit_acc_90d"], ls="--", color=S.ACCENT, lw=1.5, zorder=2,
               label=f"COMMIT: best of the first 90 days ({d['commit_acc_90d']:.2f})")
    ax.annotate("", xy=(xs[-1], d["frontier_last"]), xytext=(xs[-1], d["commit_acc_90d"]),
                arrowprops=dict(arrowstyle="<->", color=S.DIM, lw=1.1))
    ax.text(xs[-1], (d["frontier_last"]+d["commit_acc_90d"])/2, f"  +{d['broad_access_advantage']:.2f}\n  option value",
            fontsize=7.2, va="center", color=S.DIM)
    ax.set_ylabel("frontier accuracy (MMLU-Pro)"); ax.set_xlabel("model release date")
    ax.legend(loc="lower right"); fig.autofmt_xdate()
    S.clean(ax, grid="y")
    S.titled(ax, "The option value of breadth", f"routing to the best available model vs committing to the first 90 days' best; cost/correct drops {d['cpc_drop_factor']:.0f}× (Pillar D)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_churn.pdf")); plt.close(fig)
    print("wrote fig_churn.pdf")

def fig_realizability(tag):
    """Market-scale realizability: empirical co-failure tail beta vs the single-factor
    (pairwise-rho) prediction, per benchmark; plus the pool-size scaling of the ratio."""
    d = json.load(open(os.path.join(RUNS, f"realizability_{tag}.json")))
    pd = d["per_dataset"]
    order = [k for k in ("gsm8k", "arc", "mmlu", "mmlu_pro", "math500", "aime24", "aime25") if k in pd]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.5))
    x = np.arange(len(order)); w = 0.38
    be = [pd[k]["beta"] for k in order]; b1 = [pd[k]["beta_1factor"] for k in order]
    ax1.bar(x - w/2, be, w, color=S.ACCENT, label=r"empirical $\beta=P(\mathrm{all\ wrong})$")
    ax1.bar(x + w/2, b1, w, color=S.INKBLUE, label=r"single-factor $\beta(\bar\rho)$ (priced off pairwise $\rho$)")
    for i, k in enumerate(order):
        r = pd[k]["underpricing_ratio"]
        if np.isfinite(r) and be[i] > 0:
            ax1.annotate(f"{r:.0f}×", (i, max(be[i], b1[i])), ha="center", va="bottom", fontsize=7.2, color=S.INK)
    ax1.set_xticks(x); ax1.set_xticklabels(order, rotation=30, ha="right", fontsize=7.5)
    ax1.set_ylabel("co-failure probability"); ax1.legend(loc="upper left")
    S.clean(ax1, grid="y"); S.titled(ax1, "Pairwise ρ underprices the tail")
    pool = d.get("pooled")
    if pool and pool.get("pool_scaling"):
        ps = pool["pool_scaling"]
        ks = [r["k"] for r in ps]; rt = [r["ratio_median"] for r in ps]
        lo = [r.get("ratio_p05", np.nan) for r in ps]; hi = [r.get("ratio_p95", np.nan) for r in ps]
        if all(np.isfinite(v) for v in lo + hi):
            ax2.fill_between(ks, lo, hi, color=S.H_MATH, alpha=0.16, lw=0, label="5–95% over random sub-pools")
        ax2.plot(ks, rt, "o-", color=S.H_MATH, lw=1.8, ms=4.5, mec="white", mew=0.6, label="median ratio")
        ax2.axhline(1.0, ls=":", color=S.FAINT, lw=0.9)
        ax2.set_xlabel(r"pool size $k$ (random market sub-pools)")
        ax2.set_ylabel(r"underpricing ratio $\beta/\beta(\bar\rho)$")
        ax2.legend(loc="upper left")
        S.clean(ax2, grid="y")
        S.titled(ax2, "Tail mispricing grows with the market", f"{pool['n_models']} models, hard benchmarks")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_realizability.pdf")); plt.close(fig)
    print("wrote fig_realizability.pdf")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--realiz", default=None)
    ap.add_argument("--churn", default="churnD_final")
    ap.add_argument("--kstar", default="eqq2_final")
    ap.add_argument("--matrix", default="stageA2v3_final")
    ap.add_argument("--cascade", default="stageC2v3_final"); ap.add_argument("--bstats", default="hardAv3_final")
    # eqq2 is the paper's matched-quality run (rho 0.42 vs 0.80); v1 of the paper drew hardBv2 here by mistake
    ap.add_argument("--eqq", default="eqq2_final"); ap.add_argument("--router", default="stageA2v3_final")
    a = ap.parse_args()
    S.set_style()
    jobs = [(fig_cost_quality, a.matrix), (fig_cascade_collapse, a.cascade),
            (fig_rho_gain, a.bstats), (fig_equal_quality, a.eqq), (fig_router, a.router),
            (fig_kstar, a.kstar), (fig_churn, a.churn)]
    if a.realiz: jobs.append((fig_realizability, a.realiz))
    for fn, arg in jobs:
        try: fn(arg)
        except Exception as e: print(f"{fn.__name__} skipped:", e)
