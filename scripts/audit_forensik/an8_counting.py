"""What limits counting: singletons, duplication-factor stability, simple baselines."""
import json, glob
import numpy as np
from collections import defaultdict, Counter

R953 = "/workspace/SawitMVC-YOLO/json"
MAN = "/workspace/SawitMVC-YOLO/split_manifest.csv"
CID = {"B1": 0, "B2": 1, "B3": 2, "B4": 3}
NAMES = ["B1", "B2", "B3", "B4"]

split = {}
for i, line in enumerate(open(MAN, encoding="utf-8-sig")):
    if i:
        f = line.strip().split(","); split[f[0]] = f[-1]

trees = {}
for p in glob.glob(f"{R953}/*.json"):
    d = json.load(open(p))
    B = [(CID.get(b.get("class"), -1), len(b.get("appearances", []))) for b in (d.get("bunches") or [])]
    B = [b for b in B if b[0] >= 0]
    trees[d["tree_id"]] = dict(B=B, nimg=len(d.get("images", {})),
                               nbox=sum(v.get("bbox_count", 0) for v in d.get("images", {}).values()),
                               sp=split.get(d["tree_id"], "train"))

# --- singletons: which classes are only ever seen once? ---
print("=== appearance count by class (all 953 trees) ===")
tab = defaultdict(Counter)
for t in trees.values():
    for c, a in t["B"]:
        tab[c][min(a, 4)] += 1
print(f"{'class':<6}{'n':>7}{'1 view':>9}{'2':>8}{'3':>8}{'4+':>8}")
for c in range(4):
    n = sum(tab[c].values())
    print(f"{NAMES[c]:<6}{n:>7}" + "".join(f"{100*tab[c][k]/n:>7.1f}%" for k in [1, 2, 3, 4]))

# --- duplication factor stability ---
te = [t for t in trees.values() if t["sp"] == "test"]
tr = [t for t in trees.values() if t["sp"] == "train"]
k_tr = np.array([t["nbox"] / max(len(t["B"]), 1) for t in tr if t["B"]])
k_te = np.array([t["nbox"] / max(len(t["B"]), 1) for t in te if t["B"]])
print(f"\n=== per-tree duplication factor k = boxes / unique bunches ===")
print(f"  train: mean={k_tr.mean():.3f} sd={k_tr.std():.3f} p10={np.percentile(k_tr,10):.2f} p90={np.percentile(k_tr,90):.2f}")
print(f"  test : mean={k_te.mean():.3f} sd={k_te.std():.3f}")

# --- baseline: count = GT boxes / k_global   (perfect detection, no linking at all) ---
kg = k_tr.mean()
y = np.array([len(t["B"]) for t in te if t["B"]])
nb = np.array([t["nbox"] for t in te if t["B"]])
for name, pred in [("boxes / mean k", nb / kg), ("boxes / 1.887", nb / 1.887)]:
    p = np.round(pred)
    print(f"  [oracle boxes] {name:<16} MAE={np.abs(p-y).mean():.3f}  exact={np.mean(p==y):.3f}  ±1={np.mean(np.abs(p-y)<=1):.3f}")

# ridge on per-class box counts (oracle boxes) -> per-class bunch counts
from sklearn.linear_model import RidgeCV
def feats(ts):
    X, Y = [], []
    for t in ts:
        if not t["B"]:
            continue
        cb = Counter(); ca = Counter()
        for c, a in t["B"]:
            cb[c] += 1; ca[c] += a
        X.append([ca[c] for c in range(4)] + [sum(ca.values()), t["nimg"]])
        Y.append([cb[c] for c in range(4)])
    return np.array(X, float), np.array(Y, float)

Xtr, Ytr = feats(tr); Xte, Yte = feats(te)
R = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Xtr, Ytr)
Pd = np.clip(np.round(R.predict(Xte)), 0, None)
print(f"\n=== per-class counting from ORACLE boxes (upper bound of the counting head) ===")
for c in range(4):
    e = np.abs(Pd[:, c] - Yte[:, c])
    print(f"  {NAMES[c]}: MAE={e.mean():.3f}  exact={np.mean(e==0):.3f}  ±1={np.mean(e<=1):.3f}")
etot = np.abs(Pd.sum(1) - Yte.sum(1))
print(f"  TOTAL: MAE={etot.mean():.3f}  exact={np.mean(etot==0):.3f}  ±1={np.mean(etot<=1):.3f}")
print(f"  macro-MAE (mean of per-class MAE) = {np.abs(Pd-Yte).mean(0).mean():.3f}")
print("  reference: project baseline (real detections) Ridge macro-MAE 1.036, class±1 77.5%")
