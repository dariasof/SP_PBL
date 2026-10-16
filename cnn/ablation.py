"""Same CNN, with and without it.

    python ablation.py                  # rf59, all 6 folds, both variants
    python ablation.py --folds 0 --epochs 5

Two things are measured, because the CRF is supposed to buy two things:

  performance   macro-F1 / MCC / SP-F1, the usual
  structure     how often the predicted label sequence is biologically
                impossible -- returning to a signal peptide after the mature
                protein starts, mixing two SP types, crossing the membrane
                without a transmembrane segment

The CRF here is UNCONSTRAINED, so valid output is not guaranteed by
construction -- it has to be learned. That makes the structural comparison a
real experiment rather than a tautology.

Writes ablation.csv.
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, matthews_corrcoef
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) 
from model import CnnCrf
from splits import FOLDS, LABELS
from training import ALL, CONFIGS, DEVICE, SP, set_seed, tensors

# transitions that cannot happen in a real protein
ILLEGAL = set()
for a in "SLT":                       # never return to a signal peptide
    for b in "SLT":
        if a != b:
            ILLEGAL.add((a, b))       # and never mix two SP types
for a in "OMI":
    for b in "SLT":
        ILLEGAL.add((a, b))
ILLEGAL |= {("O", "I"), ("I", "O")}   # no crossing the membrane without an M


def count_violations(pred, mask):
    """Fraction of sequences whose predicted path contains an impossible step."""
    bad = 0
    for row, m in zip(pred, mask):
        path = [LABELS[int(t)] for t in row[m]]
        if any((a, b) in ILLEGAL for a, b in zip(path, path[1:])):
            bad += 1
    return bad / len(pred)


def run(dilations, train_df, val_df, use_crf, epochs=30, patience=5, seed=0):
    set_seed(seed)
    x_tr, y_tr, m_tr = tensors(train_df)
    x_va, y_va, m_va = tensors(val_df)
    loader = DataLoader(TensorDataset(x_tr, y_tr, m_tr), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    model = CnnCrf(dilations=dilations).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss(ignore_index=-100)

    best, bad = None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            opt.zero_grad()
            if use_crf:
                loss = model.loss(xb, yb, mb)
            else:
                # identical network, plain per-residue cross-entropy
                loss = ce(model.emissions(xb).reshape(-1, len(LABELS)), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(x_va), 256):
                xb, mb = x_va[i:i+256].to(DEVICE), m_va[i:i+256].to(DEVICE)
                preds.append((model.predict(xb, mb) if use_crf
                              else model.emissions(xb).argmax(-1)).cpu())
        pred = torch.cat(preds)

        t, p = y_va[m_va].numpy(), pred[m_va].numpy()
        s = {"macro_f1": f1_score(t, p, average="macro", labels=ALL, zero_division=0),
             "mcc": matthews_corrcoef(t, p),
             "sp_macro_f1": f1_score(t, p, average="macro", labels=SP, zero_division=0)}

        if best is None or s["macro_f1"] > best["macro_f1"]:
            best, bad = s, 0
            best["invalid_paths"] = count_violations(pred, m_va)
        else:
            bad += 1
            if bad >= patience:
                break
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rf59")
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(6)))
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    df = pd.read_csv("dataset.csv", comment="#")
    dil = CONFIGS[args.config]
    rows = []

    for use_crf in (True, False):
        tag = "CNN+CRF" if use_crf else "CNN only"
        for k in args.folds:
            _, tr, va = FOLDS[k]
            t0 = time.time()
            s = run(dil, df[df.partition == tr], df[df.partition == va],
                    use_crf, epochs=args.epochs)
            rows.append({"variant": tag, "fold": k, **s})
            print(f"{tag:<9} fold {k}  macro-F1 {s['macro_f1']:.4f}  "
                  f"MCC {s['mcc']:.4f}  invalid paths {s['invalid_paths']:.1%}  "
                  f"({(time.time()-t0)/60:.1f} min)")
        print()

    out = pd.DataFrame(rows)
    out.to_csv("ablation.csv", index=False)

    print(f"{'variant':<10}{'macro-F1':>20}{'MCC':>20}{'invalid paths':>18}")
    for tag in out.variant.unique():
        g = out[out.variant == tag]
        print(f"{tag:<10}{g.macro_f1.mean():>12.4f} +/-{g.macro_f1.sem():<6.4f}"
              f"{g.mcc.mean():>12.4f} +/-{g.mcc.sem():<6.4f}"
              f"{g.invalid_paths.mean():>17.1%}")

    # paired across folds: both variants saw the same six splits
    a = out[out.variant == "CNN+CRF"].set_index("fold").macro_f1
    b = out[out.variant == "CNN only"].set_index("fold").macro_f1
    if len(a) == len(b) > 1:
        d = (a - b)
        print(f"\npaired difference (CRF - no CRF): {d.mean():+.4f} +/- {d.sem():.4f}"
              f"   CRF wins on {int((d > 0).sum())}/{len(d)} folds")
