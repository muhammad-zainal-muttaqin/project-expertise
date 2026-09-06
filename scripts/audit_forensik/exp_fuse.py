"""Bunch-level maturity: appearance vs +structure vs +monotone tree decoding."""
import json
import numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

OUT = "/workspace/crops953"
idx = json.load(open(f"{OUT}/index.json"))
by = {s: [r for r in idx if r["split"] == s] for s in ["train", "val", "test"]}
P = {s: np.load(f"{OUT}/P_{s}.npy") for s in ["train", "val", "test"]}


def bunch_level(sp):
    """Aggregate crop probabilities to physical bunches (GT grouping)."""
    g = defaultdict(list)
    for i, r in enumerate(by[sp]):
        g[(r["tree"], r["bunch"])].append(i)
    rows = []
    for (tree, bid), ii in g.items():
        r0 = by[sp][ii[0]]
        p = P[sp][ii].mean(0)
        cy = float(np.median([by[sp][i]["cy"] for i in ii]))
        ar = float(np.median([by[sp][i]["w"] * by[sp][i]["h"] for i in ii]))
        rows.append(dict(tree=tree, y=r0["cls"], p=p, cy=cy, ar=ar, napp=len(ii)))
    # within-tree structural ranks (computed from detections, no GT needed)
    bt = defaultdict(list)
    for r in rows:
        bt[r["tree"]].append(r)
    for t, rs in bt.items():
        n = len(rs)
        cy = np.array([r["cy"] for r in rs]); ar = np.array([r["ar"] for r in rs])
        rcy = cy.argsort().argsort() / max(n - 1, 1)
        rar = (-ar).argsort().argsort() / max(n - 1, 1)
        for i, r in enumerate(rs):
            r["rcy"], r["rar"], r["n"] = rcy[i], rar[i], n
            r["dcy"] = cy[i] - cy.mean(); r["dar"] = np.log(ar[i]) - np.log(ar).mean()
    return rows, bt


def feats(rows):
    return np.array([[*r["p"], r["rcy"], r["rar"], r["cy"], np.log(r["ar"]),
                      r["n"], r["napp"], r["dcy"], r["dar"]] for r in rows])


def report(name, y, pred, rows=None):
    line = f"  {name:<34} acc={accuracy_score(y,pred):.4f}  macroF1={f1_score(y,pred,average='macro'):.4f}  ±1={np.mean(np.abs(pred-y)<=1):.4f}"
    if rows is not None:
        # per-tree per-class count error (what the counting head must deliver)
        bt = defaultdict(lambda: [np.zeros(4), np.zeros(4)])
        for r, p in zip(rows, pred):
            bt[r["tree"]][0][r["y"]] += 1
            bt[r["tree"]][1][p] += 1
        e = np.array([np.abs(a - b).sum() / 2 for a, b in bt.values()])
        line += f"  per-tree class-count MAE={e.mean():.3f}"
    print(line)


rv, btv = bunch_level("val")
rt, btt = bunch_level("test")
yv = np.array([r["y"] for r in rv]); yt = np.array([r["y"] for r in rt])
print(f"val bunches={len(rv)}  test bunches={len(rt)}  test trees={len(btt)}")

print("\n=== TEST, bunch level (GT grouping) ===")
report("1. appearance only (argmax)", yt, np.array([r["p"].argmax() for r in rt]), rt)

# structure-only reference
gs = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=42)
gs.fit(feats(rv)[:, 4:], yv)
report("2. structure only (no pixels)", yt, gs.predict(feats(rt)[:, 4:]), rt)

# fusion: appearance probs + structure, fitted on VAL
gf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, random_state=42)
gf.fit(feats(rv), yv)
Pf_t = gf.predict_proba(feats(rt))
report("3. appearance + structure (fused)", yt, Pf_t.argmax(1), rt)


def monotone_decode(P, cy, lam):
    """Ripeness must not increase as you go UP the trunk.
    Sort lowest-first, DP over 3 ordered cut points."""
    o = np.argsort(-cy)
    L = np.log(np.clip(P[o], 1e-9, 1))
    n = len(o)
    pref = np.vstack([np.zeros(4), np.cumsum(L, 0)])
    best, arg = None, None
    for i in range(n + 1):
        for j in range(i, n + 1):
            for k in range(j, n + 1):
                s = (pref[i, 0] - pref[0, 0]) + (pref[j, 1] - pref[i, 1]) + \
                    (pref[k, 2] - pref[j, 2]) + (pref[n, 3] - pref[k, 3])
                if best is None or s > best:
                    best, arg = s, (i, j, k)
    lab = np.empty(n, int)
    i, j, k = arg
    lab[:i] = 0; lab[i:j] = 1; lab[j:k] = 2; lab[k:] = 3
    out = np.empty(n, int)
    out[o] = lab
    # soft blend: keep argmax where the model is confident
    hard = P.argmax(1)
    conf = P.max(1)
    return np.where(conf > lam, hard, out)


for lam in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]:
    pv = np.concatenate([monotone_decode(gf.predict_proba(feats(rs)),
                                         np.array([r["cy"] for r in rs]), lam)
                         for rs in btv.values()])
    yv2 = np.concatenate([np.array([r["y"] for r in rs]) for rs in btv.values()])
    a, f = accuracy_score(yv2, pv), f1_score(yv2, pv, average="macro")
    print(f"    [VAL tuning] lam={lam:.2f} acc={a:.4f} macroF1={f:.4f}")

LAM = 0.7
pt = np.concatenate([monotone_decode(gf.predict_proba(feats(rs)),
                                     np.array([r["cy"] for r in rs]), LAM)
                     for rs in btt.values()])
yt2 = np.concatenate([np.array([r["y"] for r in rs]) for rs in btt.values()])
rt2 = [r for rs in btt.values() for r in rs]
report(f"4. fused + monotone decode(l={LAM})", yt2, pt, rt2)

print("\n  confusion, fused+monotone (rows=true B1..B4):")
print(confusion_matrix(yt2, pt))
print("\n  reference: project best end-to-end 953 matched class acc = 0.7442, macro-F1 = 0.6034")
