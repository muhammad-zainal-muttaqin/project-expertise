"""Clean protocol:
   - appearance model  A(y|crop)      : ConvNeXt-Tiny trained on TRAIN crops
   - structure model   S(y|geometry)  : GBM trained on TRAIN bunches, geometry only
   - fusion            log A + w*(log S - log prior)   ; w tuned on VAL
   - monotone tree decoding                            ; lambda tuned on VAL
   TEST is touched once, at the end.
"""
import json
import numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

OUT = "/workspace/crops953"
idx = json.load(open(f"{OUT}/index.json"))
by = {s: [r for r in idx if r["split"] == s] for s in ["train", "val", "test"]}
P = {s: np.load(f"{OUT}/P_{s}.npy") for s in ["train", "val", "test"]}


def bunches(sp):
    g = defaultdict(list)
    for i, r in enumerate(by[sp]):
        g[(r["tree"], r["bunch"])].append(i)
    rows = []
    for (tree, bid), ii in g.items():
        rows.append(dict(tree=tree, y=by[sp][ii[0]]["cls"], p=P[sp][ii].mean(0),
                         cy=float(np.median([by[sp][i]["cy"] for i in ii])),
                         ar=float(np.median([by[sp][i]["w"] * by[sp][i]["h"] for i in ii])),
                         napp=len(ii)))
    bt = defaultdict(list)
    for r in rows:
        bt[r["tree"]].append(r)
    for t, rs in bt.items():
        n = len(rs)
        cy = np.array([r["cy"] for r in rs]); ar = np.array([r["ar"] for r in rs])
        rcy = cy.argsort().argsort() / max(n - 1, 1)
        rar = (-ar).argsort().argsort() / max(n - 1, 1)
        for i, r in enumerate(rs):
            r.update(rcy=rcy[i], rar=rar[i], n=n,
                     dcy=cy[i] - cy.mean(), dar=np.log(ar[i]) - np.log(ar).mean())
    return rows, bt


def G(rows):
    return np.array([[r["rcy"], r["rar"], r["cy"], np.log(r["ar"]), r["n"], r["napp"],
                      r["dcy"], r["dar"]] for r in rows])


rtr, _ = bunches("train"); rv, btv = bunches("val"); rt, btt = bunches("test")
ytr = np.array([r["y"] for r in rtr]); yv = np.array([r["y"] for r in rv]); yt = np.array([r["y"] for r in rt])

S = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, l2_regularization=1.0,
                                   random_state=42).fit(G(rtr), ytr)
prior = np.bincount(ytr, minlength=4) / len(ytr)
LP = np.log(prior)


def fuse(rows, w):
    A = np.log(np.clip(np.array([r["p"] for r in rows]), 1e-9, 1))
    Sp = np.log(np.clip(S.predict_proba(G(rows)), 1e-9, 1))
    z = A + w * (Sp - LP)
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def mono(Pm, cy, conf_keep):
    o = np.argsort(-cy)
    L = np.log(np.clip(Pm[o], 1e-9, 1)); n = len(o)
    pref = np.vstack([np.zeros(4), np.cumsum(L, 0)])
    best = None
    for i in range(n + 1):
        for j in range(i, n + 1):
            for k in range(j, n + 1):
                s = pref[i, 0] + (pref[j, 1] - pref[i, 1]) + (pref[k, 2] - pref[j, 2]) + (pref[n, 3] - pref[k, 3])
                if best is None or s > best:
                    best, arg = s, (i, j, k)
    lab = np.empty(n, int); i, j, k = arg
    lab[:i], lab[i:j], lab[j:k], lab[k:] = 0, 1, 2, 3
    out = np.empty(n, int); out[o] = lab
    return np.where(Pm.max(1) > conf_keep, Pm.argmax(1), out)


print("--- tune w on VAL (appearance+structure weight) ---")
bw, bs = 0.0, -1
for w in [0, .2, .4, .6, .8, 1.0, 1.3, 1.6, 2.0]:
    p = fuse(rv, w).argmax(1)
    a, f = accuracy_score(yv, p), f1_score(yv, p, average="macro")
    print(f"    w={w:<4} val acc={a:.4f} macroF1={f:.4f}")
    if a + f > bs:
        bs, bw = a + f, w
print(f"  -> chosen w={bw}")

print("--- tune monotone keep-threshold on VAL ---")
bl, bs = 1.01, -1
for lam in [1.01, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0]:
    pv, yy = [], []
    for t, rs in btv.items():
        pv.append(mono(fuse(rs, bw), np.array([r["cy"] for r in rs]), lam))
        yy.append([r["y"] for r in rs])
    pv = np.concatenate(pv); yy = np.concatenate(yy)
    a, f = accuracy_score(yy, pv), f1_score(yy, pv, average="macro")
    print(f"    keep>{lam:<5} val acc={a:.4f} macroF1={f:.4f}")
    if a + f > bs:
        bs, bl = a + f, lam
print(f"  -> chosen keep>{bl}")


def rep(name, rows, pred):
    y = np.array([r["y"] for r in rows])
    bt = defaultdict(lambda: [np.zeros(4), np.zeros(4)])
    for r, p in zip(rows, pred):
        bt[r["tree"]][0][r["y"]] += 1; bt[r["tree"]][1][p] += 1
    cmae = np.mean([np.abs(a - b).sum() / 2 for a, b in bt.values()])
    # ripe (B1) count error per tree - the agronomically useful number
    rr = np.array([[a[0], b[0]] for a, b in bt.values()])
    print(f"  {name:<38} acc={accuracy_score(y,pred):.4f} macroF1={f1_score(y,pred,average='macro'):.4f} "
          f"±1={np.mean(np.abs(pred-y)<=1):.4f} classcountMAE={cmae:.3f} "
          f"ripeMAE={np.abs(rr[:,0]-rr[:,1]).mean():.3f} ripe±1={np.mean(np.abs(rr[:,0]-rr[:,1])<=1):.3f}")


print("\n=== TEST (touched once) ===")
rep("A. appearance only", rt, np.array([r["p"].argmax() for r in rt]))
rep("B. structure only (no pixels)", rt, S.predict(G(rt)))
rep(f"C. fused (w={bw})", rt, fuse(rt, bw).argmax(1))
pt, rr = [], []
for t, rs in btt.items():
    pt.append(mono(fuse(rs, bw), np.array([r["cy"] for r in rs]), bl)); rr += rs
pt = np.concatenate(pt)
rep(f"D. fused + monotone (keep>{bl})", rr, pt)
print("\n  confusion D (rows=true B1..B4):"); print(confusion_matrix([r["y"] for r in rr], pt))

# 2-class agronomic view: harvest-ready (B1) vs not
for name, pred, rows in [("A. appearance", np.array([r["p"].argmax() for r in rt]), rt),
                         (f"D. fused+monotone", pt, rr)]:
    y = np.array([r["y"] for r in rows])
    yb, pb = (y == 0).astype(int), (pred == 0).astype(int)
    print(f"  [harvest-ready B1 vs rest] {name:<22} acc={accuracy_score(yb,pb):.4f} F1={f1_score(yb,pb):.4f}")
    yb2, pb2 = (y <= 1).astype(int), (pred <= 1).astype(int)
    print(f"  [B1+B2 vs B3+B4]           {name:<22} acc={accuracy_score(yb2,pb2):.4f} F1={f1_score(yb2,pb2):.4f}")
