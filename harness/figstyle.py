"""Shared figure style for the paper — cohesive near-monochrome + ONE warm accent (the co-failure
object beta). Imported by figures.py, paper_figs_v2.py, make_fig_realizability.py so every figure in
the paper matches. Styling only; contains no data and computes no reported number.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, to_rgb

# ---- palette (shared with the hero figures) ----
INK     = "#1A1D22"   # primary ink for text / key marks
TEXT    = "#2B2F36"
DIM     = "#6A727C"   # secondary labels
FAINT   = "#9AA1AA"   # ticks / hairlines / spines
HAIR    = "#E4E7EB"   # gridlines on paper-white
ACCENT      = "#E0613C"   # the ONE warm hue: reserved for beta / the co-failure object / STOP
ACCENT_SOFT = "#F4C9B9"
ACCENT_DK   = "#B23A1C"
GO      = "#2E9E6B"   # resolvable / realizable / "good" green
GO_SOFT = "#BFE3D2"
INKBLUE = "#3B4C6B"   # muted slate-blue: a neutral second series (replaces saturated 'navy')
H_MATH  = "#3F8EA0"   # teal   — open-ended math
H_CODE  = "#B98A3C"   # gold   — code
H_SCI   = "#7C6BB0"   # violet — science

# on-brand sequential colormap (replaces viridis): light warm-grey -> teal -> deep ink-teal.
# monotone in lightness, so it still reads as an ordered scale.
SEQ = LinearSegmentedColormap.from_list("seq_brand", ["#EAF0F2", H_MATH, "#1E4651"])

def set_style():
    for cand in ["Helvetica Neue", "Helvetica", "Arial", "Inter", "DejaVu Sans"]:
        if any(cand.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 200,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "text.color": TEXT, "axes.edgecolor": FAINT, "axes.labelcolor": TEXT,
        "xtick.color": DIM, "ytick.color": DIM,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "regular",
        "legend.frameon": False, "legend.fontsize": 8, "legend.handlelength": 1.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,  # embed TrueType so arXiv renders identically
    })

def clean(ax, left=True, bottom=True, grid="y"):
    """Drop the top/right spines, recolor the rest, add a faint single-axis grid."""
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    for s in ("left", "bottom", "top", "right"):
        ax.spines[s].set_color(FAINT)
    ax.tick_params(length=3, color=FAINT)
    if grid in ("y", "both"):
        ax.grid(axis="y", color=HAIR, lw=0.7, zorder=0)
    if grid in ("x", "both"):
        ax.grid(axis="x", color=HAIR, lw=0.7, zorder=0)
    ax.set_axisbelow(True)

def titled(ax, title, subtitle=None):
    """Per-axes title block: bold ink title with an optional dim subtitle beneath it (axes coords)."""
    ax.set_title(title, color=INK, fontweight="bold", fontsize=11, loc="left", pad=(16 if subtitle else 8))
    if subtitle:
        ax.text(0.0, 1.012, subtitle, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8.0, color=DIM)

def suptitled(fig, title, subtitle=None, y=1.005, x=0.5):
    fig.text(x, y, title, ha="center", va="bottom", fontsize=12.0, color=INK, fontweight="bold")
    if subtitle:
        fig.text(x, y - 0.052, subtitle, ha="center", va="bottom", fontsize=8.0, color=DIM)

def style_cbar(cb, label=None):
    cb.outline.set_edgecolor(FAINT); cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(length=2.5, color=FAINT, labelsize=7.5, colors=DIM)
    if label is not None:
        cb.set_label(label, color=DIM, fontsize=8)

def muted_qualitative(n):
    """n harmonious, desaturated categorical colors (e.g. for 21 provider families) — muted toward a
    warm grey so no single dot screams, keeping the warm ACCENT exclusive to beta."""
    base = [to_rgb(matplotlib.cm.tab20(i / 20.0)) for i in range(20)]
    grey = to_rgb("#8A8F98")
    return [tuple(0.58 * c + 0.42 * g for c, g in zip(base[i % 20], grey)) for i in range(n)]
