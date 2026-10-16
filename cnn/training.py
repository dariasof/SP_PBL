
"""Train the CNN+CRF on every config and every fold.
 
    python training.py                      # 5 configs x 6 folds = 30 runs
    python training.py --configs rf9 rf59   # just two configs
    python training.py --folds 0 --epochs 3 # quick check
 
Writes one row per run to results.csv as it finishes, plus the confusion
matrix and learned transition matrix for each run.
"""
 
import argparse
import csv
import pathlib
import random
import time
 
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) 
 
from model import CnnCrf
from splits import FOLDS, LABELS, encode
 
# same depth, same width, same parameter count — only the window changes
CONFIGS = {
    "rf9":  (1, 1, 1, 1),
    "rf19": (1, 2, 2, 4),
    "rf31": (1, 2, 4, 8),
    "rf59": (1, 4, 8, 16),
    "rf75": (1, 4, 16, 16),
}
 
ALL = list(range(len(LABELS)))
SP = [LABELS.index(c) for c in "SLT"]          # signal peptide states only
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
 
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
 
 
def tensors(frame):
    """DataFrame -> (x, y, mask) tensors."""
    x, y, m = encode(list(frame.itertuples(index=False, name=None)))
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)
 
 
def evaluate(model, x, y, mask, batch_size=256):
    """Per-residue scores over unpadded positions."""
    model.eval()
    preds = []
    for i in range(0, len(x), batch_size):
        xb = x[i : i + batch_size].to(DEVICE)
        mb = mask[i : i + batch_size].to(DEVICE)
        preds.append(model.predict(xb, mb).cpu())
    pred = torch.cat(preds)
 
    t, p = y[mask].numpy(), pred[mask].numpy()
    # labels=ALL matters: without it macro-F1 averages only over the classes
    # that appear, so a model that never predicts T is scored over 5 not 6.
    return {
        "macro_f1": f1_score(t, p, average="macro", labels=ALL, zero_division=0),
        "mcc": matthews_corrcoef(t, p),
        "sp_macro_f1": f1_score(t, p, average="macro", labels=SP, zero_division=0),
        "per_class": f1_score(t, p, average=None, labels=ALL, zero_division=0),
        "confusion": confusion_matrix(t, p, labels=ALL),
    }
 
 
def train_one(dilations, train_df, val_df, epochs=30, lr=2e-3,
              batch_size=512, patience=5, seed=0, verbose=True):
    """Train one config on one fold. Returns scores at the best epoch."""
    set_seed(seed)
    x_tr, y_tr, m_tr = tensors(train_df)
    x_va, y_va, m_va = tensors(val_df)
 
    loader = DataLoader(
        TensorDataset(x_tr, y_tr, m_tr), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
 
    model = CnnCrf(dilations=dilations).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
 
    best, best_epoch, bad = None, -1, 0
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            opt.zero_grad()
            loss = model.loss(xb, yb, mb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # CRF loss can spike
            opt.step()
            total += loss.item() * len(xb)
 
        s = evaluate(model, x_va, y_va, m_va)
        if verbose:
            print(f"    epoch {epoch:>2}  loss {total/len(x_tr):7.3f}   "
                  f"macro-F1 {s['macro_f1']:.4f}   MCC {s['mcc']:.4f}")
 
        # keep the best epoch, stop if it hasn't improved for `patience` epochs
        if best is None or s["macro_f1"] > best["macro_f1"]:
            best, best_epoch, bad = s, epoch, 0
            best["transitions"] = model.transition_matrix()
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    early stop (best was epoch {best_epoch})")
                break
 
    best["best_epoch"] = best_epoch
    return best
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset.csv")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(6)))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
 
    df = pd.read_csv(args.data, comment="#")
    out = pathlib.Path(args.out)
    pathlib.Path("matrices").mkdir(exist_ok=True)
    header_needed = not out.exists()
 
    print(f"device: {DEVICE}   configs: {args.configs}   folds: {args.folds}\n")
 
    with out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if header_needed:
            w.writerow(["config", "rf", "fold", "seed", "macro_f1", "mcc",
                        "sp_macro_f1", "best_epoch", "minutes"]
                       + [f"f1_{c}" for c in LABELS])
 
        for name in args.configs:
            dil = CONFIGS[name]
            for k in args.folds:
                _, train_part, val_part = FOLDS[k]
                train_df = df[df.partition == train_part]
                val_df = df[df.partition == val_part]
 
                rf = 1 + sum(2 * d for d in dil)
                print(f"{name} (RF={rf})  fold {k}  "
                      f"train=P{train_part} val=P{val_part}")
 
                t0 = time.time()
                s = train_one(dil, train_df, val_df,
                              epochs=args.epochs, seed=args.seed)
                mins = (time.time() - t0) / 60
 
                print(f"  -> macro-F1 {s['macro_f1']:.4f}  MCC {s['mcc']:.4f}  "
                      f"SP-F1 {s['sp_macro_f1']:.4f}  ({mins:.1f} min)\n")
 
                w.writerow([name, rf, k, args.seed,
                            round(s["macro_f1"], 4), round(s["mcc"], 4),
                            round(s["sp_macro_f1"], 4), s["best_epoch"], round(mins, 2)]
                           + [round(float(v), 4) for v in s["per_class"]])
                fh.flush()
 
                # saved for the confusion-matrix and transition-matrix slides
                np.savetxt(f"matrices/cm_{name}_fold{k}.csv", s["confusion"],
                           fmt="%d", delimiter=",", header=",".join(LABELS))
                np.savetxt(f"matrices/trans_{name}_fold{k}.csv", s["transitions"],
                           fmt="%.4f", delimiter=",", header=",".join(LABELS))
 
 
if __name__ == "__main__":
    main()
 