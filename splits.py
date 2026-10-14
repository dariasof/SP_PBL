"""Dataset preparation for the SignalP 6.0 per-residue task (6 classes).

Run:  python splits.py train_set.fasta


Scheme: SignalP 6.0's nested Cross-Validation: 3 outer x 2 inner loops = 6 models.


PILIN is dropped: it carries a 7th label 'P' we don't predict.
"""

import sys
from collections import Counter
import hashlib

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {a: i + 2 for i, a in enumerate(AA)}  # 0 = padding, 1 = unknown
UNK, N_AA = 1, len(AA) + 2  # benchmark_set.fasta may contain X/U/B/Z
LABELS = "SLTOMI"
LABEL_TO_ID = {c: i for i, c in enumerate(LABELS)}
MAX_LEN, PAD, IGNORE = 70, 0, -100


def load(path, drop=("PILIN",)):
    """Parse the 3-line FASTA -> list of (uid, kingdom, sp_type, part, seq, labels)."""
    lines = [l.strip() for l in open(path) if l.strip()]
    data = []
    for i in range(0, len(lines), 3):
        uid, kingdom, sp_type, part = lines[i][1:].split("|")
        seq, labels = lines[i + 1], lines[i + 2]
        if sp_type in drop:
            continue
        assert len(seq) == len(labels) <= MAX_LEN, f"bad record {uid}"
        assert set(labels) <= set(LABELS), f"{uid}: unexpected labels {set(labels)}"
        data.append((uid, kingdom, sp_type, int(part), seq, labels))
    assert {r[3] for r in data} == {0, 1, 2}, "expected exactly 3 partitions"
    return data


# SignalP 6.0's nested Cross-Validation. 
# The outer loop holds one partition out as test; the inner loop swaps the other two between train and val. 
# Six rows, six models per hyperparameter configuration.
#          test  train  val
FOLDS = [(   0,     2,    1),
         (   0,     1,    2),
         (   1,     2,    0),
         (   1,     0,    2),
         (   2,     1,    0),
         (   2,     0,    1)]


def partition(data, p):
    """All records in partition p."""
    return [r for r in data if r[3] == p]


def encode(records):
    """-> x (N,70) aa ids, y (N,70) label ids, mask (N,70) bool.

    Pass mask to the CRF so padding contributes no emissions or transitions.
    """
    x = np.zeros((len(records), MAX_LEN), dtype=np.int64)
    y = np.full((len(records), MAX_LEN), IGNORE, dtype=np.int64)
    mask = np.zeros((len(records), MAX_LEN), dtype=bool)
    for i, (*_, seq, labels) in enumerate(records):
        n = len(seq)
        x[i, :n] = [AA_TO_ID.get(c, UNK) for c in seq]
        y[i, :n] = [LABEL_TO_ID[c] for c in labels]
        mask[i, :n] = True
    return x, y, mask

def export_csv(data, path, source_fasta):
    """Write the prepared dataset as one flat CSV.
 
    Lets anyone load it without parsing FASTA:   
            df = pd.read_csv("dataset.csv", comment="#")
            train, val = df[df.partition == 2], df[df.partition == 1]
 
    """
    md5 = hashlib.md5(open(source_fasta, "rb").read()).hexdigest()
    with open(path, "w") as f:
        f.write(f"# source={source_fasta} md5={md5}\n")
        f.write(f"# labels={LABELS} dropped=PILIN rows={len(data)}\n")
        f.write("uid,kingdom,sp_type,partition,seq,labels\n")
        for r in data:
            f.write(",".join(str(x) for x in r) + "\n")
 
 
def verify_csv(path, source_fasta):
    """Re-parse the FASTA and confirm the CSV still matches it. Raises if not."""
    fresh = load(source_fasta)
    rows = [l.strip() for l in open(path) if not l.startswith("#")][1:]
    saved = [tuple(l.split(",")) for l in rows]
    assert len(saved) == len(fresh), f"{len(saved)} rows in CSV vs {len(fresh)} in FASTA"
    for (u, k, t, p, s, lb), r in zip(saved, fresh):
        assert (u, k, t, int(p), s, lb) == r, f"CSV disagrees with FASTA at {u}"
    return len(fresh)

if __name__ == "__main__":
    data = load(sys.argv[1])
    print(f"{len(data)} sequences, {len(LABELS)} classes: {LABELS}\n")

    for k, (test, tr, va) in enumerate(FOLDS):
        train, val = partition(data, tr), partition(data, va)
        types = Counter(r[2] for r in val)
        res = Counter(c for r in val for c in r[5])
        print(
            f"fold {k}  test=P{test}  train=P{tr} ({len(train):>5})  "
            f"val=P{va} ({len(val):>5}) "
            f"val: SP={types['SP']} LIPO={types['LIPO']} TAT={types['TAT']} "
            f"TATLIPO={types['TATLIPO']}"
        )
        
        export_csv(data, "dataset.csv", sys.argv[1])
        n = verify_csv("dataset.csv", sys.argv[1])
        print(f"\nwrote dataset.csv ({n} rows), verified against the FASTA.\n")
          

    