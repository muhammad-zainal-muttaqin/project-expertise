"""Perbaikan akhir: penaut dilatih pada ambang inferensi + pencacahan B1-saja.

Dua koreksi terhadap jalankan sebelumnya:
  1. Penaut tepi dilatih pada pasangan dengan conf >= ambang inferensi, bukan
     >= 0,10, sehingga distribusi latih dan inferensi cocok.
  2. Target pencacahan ditambah B1-saja (siap panen menurut kartu dataset:
     B1 = optimal harvest stage; B2 masih transitioning).
"""
import json, pickle, sys, itertools
import numpy as np
sys.path.insert(0, "/tmp/claude-1001/-workspace/ebcbd941-6775-4113-b727-404085458263/scratchpad")
from panen_pipeline import (evaluate, GT, RES, EMPAT_SISI, build_pairs, cluster_tree)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score, average_precision_score

D = pickle.load(open(f"{RES}/dets.pkl", "rb"))
P = json.load(open(f"{RES}/panen_results.json"))["profil"]
CONF = P["det_conf"]
TR = [t for t in D["train"] if t in EMPAT_SISI]
VA = [t for t in D["val"] if t in EMPAT_SISI]
TE = [t for t in D["test"] if t in EMPAT_SISI]


def filt(sp, trees):
    return {t: [d for d in D[sp].get(t, []) if d["conf"] >= CONF] for t in trees}


print(f"=== melatih ulang penaut pada conf>={CONF} ===", flush=True)
Ftr, Fva, Fte = filt("train", TR), filt("val", VA), filt("test", TE)
Xtr, ytr, _ = build_pairs(Ftr, TR)
Xva, yva, _ = build_pairs(Fva, VA)
E2 = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                    l2_regularization=1.0, random_state=42).fit(Xtr, ytr)
pv = E2.predict_proba(Xva)[:, 1]
print(f"  pasangan latih {len(ytr)} (positif {int(ytr.sum())})  "
      f"VAL AUC={roc_auc_score(yva, pv):.4f} AP={average_precision_score(yva, pv):.4f}",
      flush=True)
pickle.dump(E2, open(f"{RES}/edge_model_v2.pkl", "wb"))

print("=== menala ulang topologi pada VAL ===", flush=True)
best = None
for lt, ms, st in itertools.product([0.25, 0.35, 0.45, 0.55], [2, 3], [0.35, 0.45, 0.55]):
    r, _ = evaluate(D["val"], VA, E2, lt, ms, st, P["t_coarse"], P["t_b1b2"],
                    P["t_b3b4"], CONF)
    j = r["physical_f1"] + r["class2_acc"]
    if best is None or j > best[0]:
        best = (j, lt, ms, st, r)
_, LT, MS, ST, rvbest = best
print(f"  terpilih link={LT} max_size={MS} single={ST} -> F1 VAL {rvbest['physical_f1']:.4f}",
      flush=True)

rv, _ = evaluate(D["val"], VA, E2, LT, MS, ST, P["t_coarse"], P["t_b1b2"], P["t_b3b4"], CONF)
rt, _ = evaluate(D["test"], TE, E2, LT, MS, ST, P["t_coarse"], P["t_b1b2"], P["t_b3b4"], CONF)
print(f"\nF1 fisik  VAL {rv['physical_f1']:.4f}  TEST {rt['physical_f1']:.4f} "
      f"(P {rt['precision']:.4f} / R {rt['recall']:.4f})", flush=True)
print(f"kelas 2   TEST akurasi {rt['class2_acc']:.4f} F1 {rt['class2_f1']:.4f}", flush=True)
print(f"kelas 4   TEST akurasi {rt['class4_acc']:.4f} makro {rt['class4_macro_f1']:.4f} "
      f"ordinal±1 {rt['class4_within1']:.4f}", flush=True)

# ------------------------------------------------- lapisan pencacahan Ridge
THR = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


def feats(sp, trees):
    X, Y = [], []
    for t in trees:
        raw = D[sp].get(t, [])
        ds = [d for d in raw if d["conf"] >= CONF]
        if ds:
            Xp, _, keys = build_pairs({t: ds}, [t], labelled=False)
            pr = E2.predict_proba(Xp)[:, 1] if len(Xp) else []
            ep = {(min(i, j), max(i, j)): float(p) for (_, i, j), p in zip(keys, pr)}
            groups = cluster_tree(ds, ep, LT, MS)
        else:
            groups = []
        clus = []
        for g in groups:
            mem = [ds[i] for i in g]
            w = np.array([m["conf"] for m in mem])
            if len(mem) == 1 and w[0] < ST:
                continue
            clus.append(dict(n=len(mem), conf=float(w.mean()),
                             score=float(np.average([m["score"] for m in mem], weights=w))))
        f = []
        for th in THR:
            k = [d for d in raw if d["conf"] >= th]
            f += [len(k),
                  sum(1 for d in k if d["score"] < P["t_b1b2"]),          # B1
                  sum(1 for d in k if d["score"] < P["t_coarse"]),        # B1+B2
                  sum(1 for d in k if d["score"] >= P["t_coarse"])]
        for lo, hi in [(1, 1), (2, 2), (3, 9)]:
            f.append(sum(1 for c in clus if lo <= c["n"] <= hi))
        f += [len(clus),
              sum(1 for c in clus if c["score"] < P["t_b1b2"]),
              sum(1 for c in clus if c["score"] < P["t_coarse"]),
              float(np.mean([c["conf"] for c in clus])) if clus else 0.,
              float(np.mean([c["n"] for c in clus])) if clus else 0.]
        g = GT.get(t, [])
        X.append(f)
        Y.append([len(g), sum(1 for b in g if b["c"] == 0),
                  sum(1 for b in g if b["c"] <= 1), sum(1 for b in g if b["c"] > 1)])
    return np.array(X, float), np.array(Y, float)


Xtr2, Ytr2 = feats("train", TR)
Xva2, Yva2 = feats("val", VA)
Xte2, Yte2 = feats("test", TE)
mu, sd = Xtr2.mean(0), Xtr2.std(0) + 1e-9
R = RidgeCV(alphas=np.logspace(-2, 4, 30)).fit((Xtr2 - mu) / sd, Ytr2)
LAB = ["total", "B1 siap panen", "B1+B2 matang", "B3+B4 belum"]
out = {}
for tag, X, Y in [("VALIDATION", Xva2, Yva2), ("TEST (sekali)", Xte2, Yte2)]:
    p = np.clip(np.round(R.predict((X - mu) / sd)), 0, None)
    print(f"\n=== pencacahan Ridge — {tag} ===")
    out[tag] = {}
    for k, lab in enumerate(LAB):
        e = np.abs(p[:, k] - Y[:, k])
        out[tag][lab] = dict(mae=float(e.mean()), exact=float((e == 0).mean()),
                             within1=float((e <= 1).mean()), mean_true=float(Y[:, k].mean()))
        print(f"  {lab:<14}: MAE {e.mean():.3f}  tepat {(e==0).mean():.3f}  "
              f"±1 {(e<=1).mean():.3f}   (rerata acuan {Y[:,k].mean():.2f}/pohon)")

json.dump(dict(profil=dict(det_conf=CONF, link_thr=LT, max_size=MS, single_thr=ST,
                           t_coarse=P["t_coarse"], t_b1b2=P["t_b1b2"], t_b3b4=P["t_b3b4"]),
               linker_val_auc=float(roc_auc_score(yva, pv)),
               linker_val_ap=float(average_precision_score(yva, pv)),
               val=rv, test=rt, counting=out),
          open(f"{RES}/panen_final.json", "w"), indent=1)
print("\nFINAL DONE")
