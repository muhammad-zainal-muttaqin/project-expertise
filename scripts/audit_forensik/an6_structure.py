import json, glob, os
import numpy as np
from collections import defaultdict

R953 = "/workspace/SawitMVC-YOLO/json"
MAN = "/workspace/SawitMVC-YOLO/split_manifest.csv"
CID = {"B1": 0, "B2": 1, "B3": 2, "B4": 3}

split = {}
for i, line in enumerate(open(MAN, encoding="utf-8-sig")):
    if i == 0:
        continue
    f = line.strip().split(",")
    split[f[0]] = f[-1]

rows = []   # (tree, split, cy, area, cls, n_app)
for p in glob.glob(f"{R953}/*.json"):
    d = json.load(open(p))
    tree = d["tree_id"]
    imgs = d.get("images", {})
    ann = {}
    for side, v in imgs.items():
        for a in v.get("annotations", []):
            ann[(side, a["box_index"])] = a["bbox_yolo"]
    for b in d.get("bunches", []) or []:
        bb = [ann[(ap["side"], ap["box_index"])] for ap in b.get("appearances", [])
              if (ap["side"], ap["box_index"]) in ann]
        if not bb:
            continue
        cy = float(np.median([x[1] for x in bb]))
        ar = float(np.median([x[2] * x[3] for x in bb]))
        c = CID.get(b.get("class"), -1)
        if c < 0:
            continue
        rows.append((tree, split.get(tree, "train"), cy, ar, c, len(bb)))

print(f"bunches: {len(rows)}  trees: {len(set(r[0] for r in rows))}")

# ---- within-tree ranks ----
bytree = defaultdict(list)
for r in rows:
    bytree[r[0]].append(r)

feats, labs, grp = [], [], []
for t, rs in bytree.items():
    n = len(rs)
    if n < 2:
        continue
    cy = np.array([r[2] for r in rs]);  ar = np.array([r[3] for r in rs])
    rcy = cy.argsort().argsort() / (n - 1)          # 0 = topmost
    rar = (-ar).argsort().argsort() / (n - 1)       # 0 = largest
    for i, r in enumerate(rs):
        feats.append([rcy[i], rar[i], cy[i], np.log(ar[i]), n, r[5],
                      cy[i] - cy.mean(), np.log(ar[i]) - np.log(ar).mean()])
        labs.append(r[4]); grp.append(r[1])

X = np.array(feats); y = np.array(labs); g = np.array(grp)
print(f"usable bunches: {len(y)}   class mix: {np.bincount(y)/len(y)}")

from scipy.stats import spearmanr
print(f"\nSpearman(class, vertical rank within tree) = {spearmanr(y, X[:,0]).statistic:+.3f}")
print(f"Spearman(class, size rank within tree)     = {spearmanr(y, X[:,1]).statistic:+.3f}")
print(f"Spearman(class, raw cy)                    = {spearmanr(y, X[:,2]).statistic:+.3f}")
print(f"Spearman(class, log area)                  = {spearmanr(y, X[:,3]).statistic:+.3f}")

tr = (g == "train"); te = (g == "test")
print(f"\ntrain bunches={tr.sum()}  test bunches={te.sum()}")

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=42)
m.fit(X[tr], y[tr])
pred = m.predict(X[te])
prior = np.bincount(y[tr]).argmax()
print(f"\n--- GEOMETRY-ONLY maturity classifier (NO pixels at all) ---")
print(f"  majority-class baseline : acc={accuracy_score(y[te], np.full(te.sum(), prior)):.4f}")
print(f"  structural features     : acc={accuracy_score(y[te], pred):.4f}  macroF1={f1_score(y[te], pred, average='macro'):.4f}")
print(f"  ordinal +-1 accuracy    : {np.mean(np.abs(pred - y[te]) <= 1):.4f}")
print("  confusion (rows=true B1..B4):")
print(confusion_matrix(y[te], pred))

# ablation: which feature carries it
for name, cols in [("vertical rank only", [0]), ("size rank only", [1]),
                   ("vert+size rank", [0, 1]), ("all", list(range(X.shape[1])))]:
    mm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=42)
    mm.fit(X[tr][:, cols], y[tr])
    pp = mm.predict(X[te][:, cols])
    print(f"  {name:<20} acc={accuracy_score(y[te], pp):.4f}  macroF1={f1_score(y[te], pp, average='macro'):.4f}")
