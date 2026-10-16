"""Second round of analysis: what the CRF learned, and where context helped.
 
    python analyse2.py
 
Uses only files you already have (matrices/ and results.csv) -- no retraining.
Writes figures/5_*.png, 6_*.png, 7_*.png.
"""
 
import pathlib
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) 
 
from splits import LABELS
 
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#dcdcd8"
# diverging: blue <-> red with a neutral grey midpoint (never a hue at zero)
DIVERGING = matplotlib.colors.LinearSegmentedColormap.from_list(
    "div", ["#184f95", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a08a", "#d03b3b", "#8f1f1f"])
SEQ = matplotlib.colors.LinearSegmentedColormap.from_list(
    "seq", ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#184f95"])
 
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 11,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.8,
})
 
OUT = pathlib.Path("figures"); OUT.mkdir(exist_ok=True)
K = len(LABELS)
 
r = pd.read_csv("results.csv")
configs = list(r.sort_values("rf").config.unique())
rfs = [int(r[r.config == c].rf.iloc[0]) for c in configs]
folds = sorted(r.fold.unique())
 
 
def mean_matrix(kind, config):
    """Average the per-fold matrices for one config. kind is 'cm' or 'trans'."""
    ms = []
    for f in folds:
        p = pathlib.Path(f"matrices/{kind}_{config}_fold{f}.csv")
        if p.exists():
            ms.append(np.loadtxt(p, delimiter=","))
    return np.mean(ms, axis=0) if ms else None
 
 
# ===========================================================================
# 5. What the CRF learned, and how it changes with receptive field
# ===========================================================================
trans = {c: mean_matrix("trans", c) for c in configs}
have = [c for c in configs if trans[c] is not None]
 
if have:
    lim = max(np.abs(trans[c]).max() for c in have)
    fig, axes = plt.subplots(1, len(have), figsize=(3.0 * len(have), 3.4))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, have):
        im = ax.imshow(trans[c], cmap=DIVERGING, vmin=-lim, vmax=lim)
        ax.set_title(f"RF {rfs[configs.index(c)]}", fontsize=11, color=INK)
        ax.set_xticks(range(K), list(LABELS), fontsize=8)
        ax.set_yticks(range(K), list(LABELS) if ax is axes[0] else [""] * K, fontsize=8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("from")
    for ax in axes:
        ax.set_xlabel("to")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("transition score", color=INK2, fontsize=9)
    cb.outline.set_visible(False)
    fig.suptitle("The CRF learned that labels come in runs — nothing imposed it",
                 fontsize=12, color=INK, x=0.07, ha="left")
    fig.savefig(OUT / "5_transitions.png")
    plt.close(fig)
 
    # Does a narrow CNN make the CRF work harder?
    # diagonal dominance = mean(stay) - mean(switch). Higher = the CRF is
    # doing more of the smoothing because the emissions are less reliable.
    rows = []
    for c in have:
        A = trans[c]
        off = A[~np.eye(K, dtype=bool)]
        rows.append((rfs[configs.index(c)], np.diag(A).mean() - off.mean()))
    dom = pd.DataFrame(rows, columns=["rf", "dominance"]).sort_values("rf")
    dom.to_csv(OUT / "transition_strength.csv", index=False)
 
    print("diagonal dominance of the learned transitions (stay minus switch):")
    for _, x in dom.iterrows():
        print(f"  RF {int(x.rf):>3}   {x.dominance:.3f}")
 
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.plot(dom.rf, dom.dominance, color=BLUE, lw=2, marker="o", ms=7)
    ax.set_xlabel("receptive field (residues)")
    ax.set_ylabel("stay minus switch")
    ax.set_xticks(dom.rf)
    ax.set_title("How hard the CRF has to work", fontsize=12, color=INK, pad=12, loc="left")
    ax.grid(axis="y", alpha=0.6); ax.set_axisbelow(True)
    fig.savefig(OUT / "6_transition_strength.png")
    plt.close(fig)
 
 
# ===========================================================================
# 7. Where context helped: narrowest vs best, cell by cell
# ===========================================================================
narrow, best = configs[0], r.groupby("config").macro_f1.mean().idxmax()
cmN, cmB = mean_matrix("cm", narrow), mean_matrix("cm", best)
 
if cmN is not None and cmB is not None:
    nN = cmN / cmN.sum(1, keepdims=True)          # row-normalised = recall
    nB = cmB / cmB.sum(1, keepdims=True)
    delta = (nB - nN) * 100                       # percentage points
 
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for ax, m, title in ((axes[0], nN * 100, f"{narrow} (RF {rfs[0]})"),
                         (axes[1], nB * 100, f"{best} (RF {rfs[configs.index(best)]})")):
        ax.imshow(m, cmap=SEQ, vmin=0, vmax=100)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center",
                        fontsize=8, color="white" if m[i, j] > 55 else INK)
        ax.set_title(title, fontsize=11, color=INK)
 
    lim = np.abs(delta).max()
    im = axes[2].imshow(delta, cmap=DIVERGING, vmin=-lim, vmax=lim)
    for i in range(K):
        for j in range(K):
            v = delta[i, j]
            axes[2].text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=8,
                         color="white" if abs(v) > 0.65 * lim else INK)
    axes[2].set_title("change (percentage points)", fontsize=11, color=INK)
    cb = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
 
    for ax in axes:
        ax.set_xticks(range(K), list(LABELS), fontsize=9)
        ax.set_yticks(range(K), list(LABELS), fontsize=9)
        ax.set_xlabel("predicted"); ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("true")
    fig.suptitle("Where wider context changed the predictions",
                 fontsize=12, color=INK, x=0.09, ha="left")
    fig.savefig(OUT / "7_confusion_delta.png")
    plt.close(fig)
 
    print(f"\nbiggest changes from {narrow} to {best} (percentage points):")
    flat = [(delta[i, j], LABELS[i], LABELS[j]) for i in range(K) for j in range(K)]
    for v, a, b in sorted(flat, key=lambda t: -abs(t[0]))[:8]:
        kind = "correct" if a == b else "error"
        print(f"  true {a} -> predicted {b}  {v:+6.1f}  ({kind})")
 
print(f"\nwrote figures to {OUT}/")