"""Does the CRF earn its place? Trains only the CNN-only variant.

    python ablation.py                  # rf59, all 6 folds
    python ablation.py --folds 0 --epochs 5

The CNN+CRF numbers are read from results.csv -- no need to retrain them.
Only the no-CRF variant is trained here, so this is roughly half the work.

Two things are measured, because the CRF is supposed to buy two things:

  performance   macro-F1 / MCC / SP-F1
  structure     how often the predicted label sequence is biologically
                impossible -- returning to a signal peptide after the mature
                protein starts, mixing two SP types, crossing the membrane
                without a transmembrane segment

The CRF here is UNCONSTRAINED, so valid output is not guaranteed by
construction; it has to be learned. That makes the structural comparison a
real experiment rather than a tautology.

Invalid-path counts for the CRF variant come from checkpoints/ if present
(inference only, a few seconds). Run training.py once with checkpoint saving
to get them; without checkpoints that column is reported as n/a.
"""

import argparse
import pathlib
import time

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, matthews_corrcoef
from torch.utils.data import DataLoader, TensorDataset

from model import CnnCrf
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) 
from splits import FOLDS, LABELS
from training import ALL, CONFIGS, DEVICE, SP, set_seed, tensors

# transitions that cannot happen in a real protein
ILLEGAL = {(a, b) for a in "SLT" for b in "SLT" if a != b}   # no mixing SP types
ILLEGAL |= {(a, b) for a in "OMI" for b in "SLT"}            # no returning to an SP
ILLEGAL |= {("O", "I"), ("I", "O")}                          # no crossing without M


def count_violations(pred, mask):
    """Fraction of sequences whose predicted path contains an impossible step."""
    bad = 0
    for row, m in zip(pred, mask):
        path = [LABELS[int(t)] for t in row[m]]
        if any(step in ILLEGAL for step in zip(path, path[1:])):
            bad += 1
    return bad / len(pred)


def scores(true, pred):
    return {"macro_f1": f1_score(true, pred, average="macro", labels=ALL, zero_division=0),
            "mcc": matthews_corrcoef(true, pred),
            "sp_macro_f1": f1_score(true, pred, average="macro", labels=SP, zero_division=0)}


def predict_all(model, x, mask, use_crf):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x), 256):
            xb, mb = x[i:i+256].to(DEVICE), mask[i:i+256].to(DEVICE)
            out.append((model.predict(xb, mb) if use_crf
                        else model.emissions(xb).argmax(-1)).cpu())
    return torch.cat(out)


def train_cnn_only(dilations, train_df, val_df, epochs=30, patience=5, seed=0):
    """Identical network, plain per-residue cross-entropy, argmax decoding."""
    set_seed(seed)
    x_tr, y_tr, m_tr = tensors(train_df)
    x_va, y_va, m_va = tensors(val_df)
    loader = DataLoader(TensorDataset(x_tr, y_tr, m_tr), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    model = CnnCrf(dilations=dilations).to(DEVICE)   # the CRF simply goes unused
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss(ignore_index=-100)

    best, bad = None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb, mb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            ce(model.emissions(xb).reshape(-1, len(LABELS)), yb.reshape(-1)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        pred = predict_all(model, x_va, m_va, use_crf=False)
        s = scores(y_va[m_va].numpy(), pred[m_va].numpy())
        if best is None or s["macro_f1"] > best["macro_f1"]:
            best, bad = s, 0
            best["invalid_paths"] = count_violations(pred, m_va)
        else:
            bad += 1
            if bad >= patience:
                break
    return best


def train_crf(dilations, train_df, val_df, epochs=30, patience=5, seed=0):
    """Same network, CRF loss and Viterbi decoding. Only needed once per fold."""
    set_seed(seed)
    x_tr, y_tr, m_tr = tensors(train_df)
    x_va, y_va, m_va = tensors(val_df)
    loader = DataLoader(TensorDataset(x_tr, y_tr, m_tr), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    model = CnnCrf(dilations=dilations).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best, best_state, bad = None, None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            opt.zero_grad()
            model.loss(xb, yb, mb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        pred = predict_all(model, x_va, m_va, use_crf=True)
        s = scores(y_va[m_va].numpy(), pred[m_va].numpy())
        if best is None or s["macro_f1"] > best["macro_f1"]:
            best, bad = s, 0
            best["invalid_paths"] = count_violations(pred, m_va)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    return best, best_state


def crf_invalid_paths(config, dilations, fold, train_df, val_df, epochs):
    """Inference from a checkpoint if we have one, else train and save one."""
    ckpt = pathlib.Path(f"checkpoints/{config}_fold{fold}.pt")
    if ckpt.exists():
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = CnnCrf(dilations=blob["dilations"])
        model.load_state_dict(blob["state_dict"])
        x_va, _, m_va = tensors(val_df)
        pred = predict_all(model.to(DEVICE), x_va, m_va, use_crf=True)
        return count_violations(pred, m_va), True

    best, state = train_crf(dilations, train_df, val_df, epochs=epochs)
    ckpt.parent.mkdir(exist_ok=True)
    torch.save({"dilations": dilations, "state_dict": state}, ckpt)
    return best["invalid_paths"], False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rf59")
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(6)))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--results", default="results.csv")
    args = ap.parse_args()

    df = pd.read_csv("dataset.csv", comment="#")
    crf = pd.read_csv(args.results).query("config == @args.config")
    if crf.empty:
        raise SystemExit(f"no rows for {args.config} in {args.results} — "
                         f"run training.py first, or pass --config")

    rows = []
    for k in args.folds:
        _, tr, va = FOLDS[k]
        val_df = df[df.partition == va]

        t0 = time.time()
        s = train_cnn_only(CONFIGS[args.config], df[df.partition == tr],
                           val_df, epochs=args.epochs)
        mins = (time.time() - t0) / 60

        ref = crf[crf.fold == k].iloc[0]           # performance from results.csv
        inv, cached = crf_invalid_paths(args.config, CONFIGS[args.config], k,
                                        df[df.partition == tr], val_df, args.epochs)

        rows.append({"fold": k,
                     "crf_macro_f1": ref.macro_f1, "cnn_macro_f1": s["macro_f1"],
                     "crf_mcc": ref.mcc, "cnn_mcc": s["mcc"],
                     "crf_invalid": inv, "cnn_invalid": s["invalid_paths"]})

        src = "cached" if cached else "trained"
        print(f"fold {k}  macro-F1  CRF {ref.macro_f1:.4f} / CNN {s['macro_f1']:.4f}   "
              f"invalid  CRF {inv:.1%} ({src}) / CNN {s['invalid_paths']:.1%}   "
              f"({(time.time()-t0)/60:.1f} min)")

    out = pd.DataFrame(rows)
    out.to_csv("ablation.csv", index=False)

    d = out.crf_macro_f1 - out.cnn_macro_f1
    print(f"\n{'':12}{'macro-F1':>22}{'MCC':>22}{'invalid paths':>18}")
    for tag, f1, mcc, inv in [("CNN+CRF", out.crf_macro_f1, out.crf_mcc, out.crf_invalid),
                              ("CNN only", out.cnn_macro_f1, out.cnn_mcc, out.cnn_invalid)]:
        print(f"{tag:<12}{f1.mean():>13.4f} +/-{f1.sem():<6.4f}"
              f"{mcc.mean():>13.4f} +/-{mcc.sem():<6.4f}{inv.mean():>17.1%}")

    if len(d) > 1:
        print(f"\npaired difference (CRF - no CRF): {d.mean():+.4f} +/- {d.sem():.4f}"
              f"   CRF wins on {int((d > 0).sum())}/{len(d)} folds")