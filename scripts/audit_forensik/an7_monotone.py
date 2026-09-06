"""Ceiling test: how far does tree-level STRUCTURE + composition get us,
with no pixels at all?  Uses only the vertical ordering of bunches on a palm."""
import json, glob
import numpy as np
from collections import defaultdict

R953 = "/workspace/SawitMVC-YOLO/json"
MAN = "/workspace/SawitMVC-YOLO/split_manifest.csv"
CID = {"B1": 0, "B2": 1, "B3": 2, "B4": 3}

split = {}
for i, line in enumerate(open(MAN, encoding="utf-8-sig")):
    if i:
        f = line.strip().split(",")
        split[f[0]] = f[-1]

trees = {}
for p in glob.glob(f"{R953}/*.json"):
    d = json.load(open(p))
    imgs = d.get("images", {})
    ann = {}
    for side, v in imgs.items():
        for a in v.get("annotations", []):
            ann[(side, a["box_index"])] = a["bbox_yolo"]
    B = []
    for b in d.get("bunches", []) or []:
        bb = [ann[(ap["side"], ap["box_index"])] for ap in b.get("appearances", [])
              if (ap["side"], ap["box_index"]) in ann]
        c = CID.get(b.get("class"), -1)
        if bb and c >= 0:
            B.append((float(np.median([x[1] for x in bb])),
                      float(np.median([x[2] * x[3] for x in bb])), c))
    if len(B) >= 2:
        trees[d["tree_id"]] = B

test = [t for t in trees if split.get(t) == "test"]
print(f"test trees: {len(test)}  bunches: {sum(len(trees[t]) for t in test)}")


def monotone_assign(B, counts, key):
    """Sort bunches by `key`, hand out classes in ordinal order following the
    tree's class-count vector.  Ripest (B1) goes to the lowest bunch."""
    order = np.argsort([-k for k in key])   # descending cy = lowest on trunk first
    out = np.empty(len(B), dtype=int)
    c, k = 0, 0
    for idx in order:
        while k >= counts[c]:
            c += 1
            k = 0
        out[idx] = c
        k += 1
    return out


acc_v = acc_s = tot = 0
pm = np.zeros((4, 4), int)
for t in test:
    B = trees[t]
    y = np.array([b[2] for b in B])
    counts = np.bincount(y, minlength=4)
    pv = monotone_assign(B, counts, [b[0] for b in B])          # by vertical position
    ps = monotone_assign(B, counts, [b[1] for b in B])          # by box size
    acc_v += (pv == y).sum(); acc_s += (ps == y).sum(); tot += len(y)
    for a, b in zip(y, pv):
        pm[a, b] += 1

print("\n--- ORACLE-COMPOSITION assignment (tree class counts known, pixels NOT used) ---")
print(f"  assign by vertical order : acc={acc_v/tot:.4f}")
print(f"  assign by box size       : acc={acc_s/tot:.4f}")
f1 = []
for c in range(4):
    tp = pm[c, c]; fp = pm[:, c].sum() - tp; fn = pm[c].sum() - tp
    f1.append(2 * tp / max(2 * tp + fp + fn, 1))
print(f"  vertical-order macro-F1  = {np.mean(f1):.4f}   per-class F1 = {[round(x,3) for x in f1]}")
print("  confusion (rows=true B1..B4):"); print(pm)

# how hard is predicting composition itself?
print("\n--- how predictable is the tree's class composition? ---")
tr = [t for t in trees if split.get(t) == "train"]
comp_tr = np.array([np.bincount([b[2] for b in trees[t]], minlength=4) /
                    len(trees[t]) for t in tr])
comp_te = np.array([np.bincount([b[2] for b in trees[t]], minlength=4) /
                    len(trees[t]) for t in test])
print(f"  global train composition : {comp_tr.mean(0).round(3)}")
print(f"  per-tree std around it   : {comp_te.std(0).round(3)}  <- trees differ a lot")
n_tr = np.array([len(trees[t]) for t in tr]); n_te = np.array([len(trees[t]) for t in test])
print(f"  bunches/tree train mean={n_tr.mean():.2f} sd={n_tr.std():.2f} | test mean={n_te.mean():.2f} sd={n_te.std():.2f}")

# what if composition comes from the global prior instead of the oracle?
acc_g = 0
gp = comp_tr.mean(0)
for t in test:
    B = trees[t]; y = np.array([b[2] for b in B]); n = len(y)
    cnt = np.floor(gp * n).astype(int)
    while cnt.sum() < n:
        cnt[np.argmax(gp * n - cnt)] += 1
    acc_g += (monotone_assign(B, cnt, [b[0] for b in B]) == y).sum()
print(f"\n  same rule but composition = GLOBAL PRIOR (no oracle): acc={acc_g/tot:.4f}")
