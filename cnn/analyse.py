"""Turn results.csv + matrices/ into the numbers and figures for the talk.

    python analyse.py

Writes figures/ and prints the summary table.
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


# validated colourblind-safe palette
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
SEQ = ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
       "#256abf", "#184f95", "#104281"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.8,
})

OUT = pathlib.Path("figures")
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------- summary
r = pd.read_csv("results.csv")
order = r.sort_values("rf").config.unique()          # narrow -> wide

summary = (r.groupby(["config", "rf"])
             .agg(macro_f1=("macro_f1", "mean"), f1_se=("macro_f1", "sem"),
                  mcc=("mcc", "mean"), mcc_se=("mcc", "sem"),
                  sp_f1=("sp_macro_f1", "mean"), sp_se=("sp_macro_f1", "sem"),
                  n=("macro_f1", "size"))
             .reset_index().sort_values("rf"))

print("\nmean +/- standard error over folds\n")
print(f"{'config':<7}{'RF':>4}{'folds':>7}   {'macro-F1':>16}{'MCC':>16}{'SP-F1':>16}")
for _, x in summary.iterrows():
    print(f"{x.config:<7}{int(x.rf):>4}{int(x.n):>7}   "
          f"{x.macro_f1:>8.4f} +/-{x.f1_se:<5.4f}"
          f"{x.mcc:>8.4f} +/-{x.mcc_se:<5.4f}"
          f"{x.sp_f1:>8.4f} +/-{x.sp_se:<5.4f}")
summary.to_csv("figures/summary.csv", index=False)

# per-fold spread, so the talk can show folds individually not just the mean
print("\nper-fold macro-F1")
print(r.pivot_table(index="config", columns="fold", values="macro_f1")
       .reindex(order).round(4).to_string())


# ------------------------------------------------- figure 1: the RF curve
fig, ax = plt.subplots(figsize=(6.4, 4.2))
x = summary.rf.values

for col, se, colour, name in [("macro_f1", "f1_se", BLUE, "macro-F1"),
                              ("mcc", "mcc_se", ORANGE, "MCC"),
                              ("sp_f1", "sp_se", AQUA, "SP-F1 (S/L/T)")]:
    ax.errorbar(x, summary[col], yerr=summary[se], color=colour, lw=2,
                marker="o", ms=7, capsize=4, label=name, zorder=3)
    # direct label at the right end, so identity is never colour-alone
    ax.annotate(name, (x[-1], summary[col].iloc[-1]), xytext=(8, 0),
                textcoords="offset points", color=colour, fontsize=10,
                va="center")

ax.axhline(0.1394, color=INK2, lw=1.2, ls=":", zorder=1)
ax.annotate("majority-class baseline (0.139)", (x[0], 0.1394), xytext=(0, 7),
            textcoords="offset points", color=INK2, fontsize=9)

ax.set_xlabel("receptive field (residues)")
ax.set_ylabel("score")
ax.set_xticks(x)
ax.set_title("Wider context improves per-residue prediction",
             fontsize=12, color=INK, pad=12, loc="left")
ax.grid(axis="y", alpha=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.margins(x=0.14)
fig.savefig(OUT / "1_receptive_field.png")
plt.close(fig)


# ------------------------------- figure 2: the S->M confusion (the finding)
def load_cm(config, fold):
    p = pathlib.Path(f"matrices/cm_{config}_fold{fold}.csv")
    return np.loadtxt(p, delimiter=",") if p.exists() else None


iS, iM = LABELS.index("S"), LABELS.index("M")
rows = []
for cfg in order:
    rf = int(r[r.config == cfg].rf.iloc[0])
    sm, ms = [], []
    for fold in sorted(r[r.config == cfg].fold.unique()):
        cm = load_cm(cfg, fold)
        if cm is None:
            continue
        sm.append(cm[iS, iM] / cm[iS].sum())          # true S called M
        ms.append(cm[iM, iS] / cm[iM].sum())          # true M called S
    if sm:
        rows.append((cfg, rf, np.mean(sm), np.std(sm, ddof=1) / np.sqrt(len(sm)),
                     np.mean(ms), np.std(ms, ddof=1) / np.sqrt(len(ms))))

if rows:
    conf = pd.DataFrame(rows, columns=["config", "rf", "sm", "sm_se", "ms", "ms_se"])
    conf.to_csv("figures/sm_confusion.csv", index=False)
    print("\nS<->M confusion (mean over folds)")
    for _, c in conf.iterrows():
        print(f"  RF {int(c.rf):>3}   S called M {c.sm:6.1%} +/-{c.sm_se:.1%}   "
              f"M called S {c.ms:6.1%} +/-{c.ms_se:.1%}")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.errorbar(conf.rf, conf.sm * 100, yerr=conf.sm_se * 100, color=BLUE,
                lw=2, marker="o", ms=7, capsize=4, label="signal peptide called membrane")
    ax.errorbar(conf.rf, conf.ms * 100, yerr=conf.ms_se * 100, color=ORANGE,
                lw=2, marker="s", ms=7, capsize=4, label="membrane called signal peptide")
    ax.set_xlabel("receptive field (residues)")
    ax.set_ylabel("% of residues misclassified")
    ax.set_xticks(conf.rf)
    ax.set_title("Context resolves the signal-peptide / membrane confusion",
                 fontsize=12, color=INK, pad=12, loc="left")
    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(OUT / "2_sm_confusion.png")
    plt.close(fig)


# -------------------------------- figure 3: confusion matrix, best config
best = summary.loc[summary.macro_f1.idxmax()]
mats = [load_cm(best.config, f) for f in sorted(r[r.config == best.config].fold.unique())]
mats = [m for m in mats if m is not None]

if mats:
    cm = np.sum(mats, axis=0)
    norm = cm / cm.sum(axis=1, keepdims=True)         # row-normalised = recall

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if norm[i, j] > 0.55 else INK)

    ax.set_xticks(range(len(LABELS)), list(LABELS))
    ax.set_yticks(range(len(LABELS)), list(LABELS))
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix, {best.config} (RF={int(best.rf)}), row-normalised",
                 fontsize=12, color=INK, pad=12, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.savefig(OUT / "3_confusion_matrix.png")
    plt.close(fig)


# ------------------------------------- figure 4: per-class F1 vs RF
per = r.groupby(["config", "rf"])[[f"f1_{c}" for c in LABELS]].mean().reset_index().sort_values("rf")
fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True, sharey=True)
for axn, c in zip(axes.flat, LABELS):
    axn.plot(per.rf, per[f"f1_{c}"], color=BLUE, lw=2, marker="o", ms=6)
    axn.set_title(c, fontsize=11, color=INK)
    axn.grid(axis="y", alpha=0.6)
    axn.set_axisbelow(True)
    axn.set_ylim(0, 1)
    axn.set_xticks(per.rf)
for axn in axes[1]:
    axn.set_xlabel("receptive field")
axes[0, 0].set_ylabel("F1")
axes[1, 0].set_ylabel("F1")
fig.suptitle("Per-class F1 by receptive field", fontsize=12, color=INK, x=0.09, ha="left")
fig.savefig(OUT / "4_per_class.png")
plt.close(fig)

print(f"\nwrote {len(list(OUT.glob('*.png')))} figures + summary.csv to figures/")