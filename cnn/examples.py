"""Concrete predicted paths: one the CRF fixes, one it doesn't.

    python examples.py                  # fold 0, rf59
    python examples.py --fold 2 -n 5

Finds validation sequences where the plain CNN produces a biologically
impossible annotation and the CNN+CRF does not, and prints them aligned with
the true labels so the difference is visible residue by residue.

Uses checkpoints/ when available; trains what's missing and saves it.
"""

import argparse
import pathlib

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ablation import ILLEGAL, count_violations, predict_all, train_crf
from model import CnnCrf
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) 
from splits import FOLDS, LABELS
from training import CONFIGS, DEVICE, set_seed, tensors


def first_violation(path):
    """(index, 'A->B') of the first impossible step, or None if the path is valid."""
    for i, step in enumerate(zip(path, path[1:])):
        if step in ILLEGAL:
            return i + 1, f"{step[0]}->{step[1]}"
    return None


def load_or_train(config, fold, train_df, val_df, use_crf, epochs):
    tag = f"{config}_fold{fold}" + ("" if use_crf else "_nocrf")
    ckpt = pathlib.Path(f"checkpoints/{tag}.pt")
    dil = CONFIGS[config]

    if ckpt.exists():
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        m = CnnCrf(dilations=blob["dilations"])
        m.load_state_dict(blob["state_dict"])
        print(f"  {tag}: loaded checkpoint")
        return m.to(DEVICE).eval()

    print(f"  {tag}: training ({epochs} epochs)...")
    if use_crf:
        _, state = train_crf(dil, train_df, val_df, epochs=epochs)
    else:
        state = train_cnn_only_state(dil, train_df, val_df, epochs)
    m = CnnCrf(dilations=dil)
    m.load_state_dict(state)
    ckpt.parent.mkdir(exist_ok=True)
    torch.save({"dilations": dil, "state_dict": state}, ckpt)
    return m.to(DEVICE).eval()


def train_cnn_only_state(dilations, train_df, val_df, epochs, patience=5, seed=0):
    """Plain cross-entropy training; returns the best epoch's weights."""
    set_seed(seed)
    x_tr, y_tr, m_tr = tensors(train_df)
    x_va, y_va, m_va = tensors(val_df)
    loader = DataLoader(TensorDataset(x_tr, y_tr, m_tr), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model = CnnCrf(dilations=dilations).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss(ignore_index=-100)

    from sklearn.metrics import f1_score
    best, best_state, bad = -1, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb, _ in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            ce(model.emissions(xb).reshape(-1, len(LABELS)), yb.reshape(-1)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        pred = predict_all(model, x_va, m_va, use_crf=False)
        f1 = f1_score(y_va[m_va].numpy(), pred[m_va].numpy(), average="macro",
                      labels=list(range(len(LABELS))), zero_division=0)
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    return best_state


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rf59")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("-n", type=int, default=3, help="examples of each kind")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    df = pd.read_csv("dataset.csv", comment="#")
    _, tr, va = FOLDS[args.fold]
    train_df, val_df = df[df.partition == tr], df[df.partition == va].reset_index(drop=True)

    crf_model = load_or_train(args.config, args.fold, train_df, val_df, True, args.epochs)
    cnn_model = load_or_train(args.config, args.fold, train_df, val_df, False, args.epochs)

    x_va, y_va, m_va = tensors(val_df)
    p_crf = predict_all(crf_model, x_va, m_va, use_crf=True)
    p_cnn = predict_all(cnn_model, x_va, m_va, use_crf=False)

    print(f"\nvalidation partition P{va}: {len(val_df)} sequences")
    print(f"  invalid paths, CNN+CRF  {count_violations(p_crf, m_va):.1%}")
    print(f"  invalid paths, CNN only {count_violations(p_cnn, m_va):.1%}\n")

    def as_string(row, m):
        return "".join(LABELS[int(t)] for t in row[m])

    fixed, both_bad = [], []
    for i in range(len(val_df)):
        m = m_va[i]
        a, b = as_string(p_crf[i], m), as_string(p_cnn[i], m)
        va_, vb = first_violation(a), first_violation(b)
        if vb and not va_:
            fixed.append((i, a, b, vb))
        elif va_ and vb:
            both_bad.append((i, a, b, va_, vb))

    def show(i, crf_path, cnn_path, note_crf, note_cnn):
        r = val_df.iloc[i]
        print(f"{r.uid}  |  {r.kingdom}  |  {r.sp_type}")
        print(f"  {'sequence':<10} {r.seq}")
        print(f"  {'true':<10} {r.labels}")
        print(f"  {'CNN+CRF':<10} {crf_path}   {note_crf}")
        print(f"  {'CNN only':<10} {cnn_path}   {note_cnn}\n")

    print(f"=== the CRF fixed it ({len(fixed)} such sequences) ===\n")
    for i, a, b, vb in fixed[:args.n]:
        pos, step = vb
        show(i, a, b, "valid", f"INVALID: {step} at position {pos}")

    if both_bad:
        print(f"=== still invalid even with the CRF ({len(both_bad)}) ===\n")
        for i, a, b, va_, vb in both_bad[:1]:
            show(i, a, b, f"INVALID: {va_[1]} at {va_[0]}",
                 f"INVALID: {vb[1]} at {vb[0]}")

    print("Paths are shown over real residues only; padding is excluded.")
