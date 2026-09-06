"""Lapisan pencacahan Ridge di atas klaster (menggantikan cacah klaster mentah).

Cacah klaster mentah terbukti bias ke bawah karena recall fisik < 1. Lapisan ini
memetakan statistik klaster + statistik deteksi multi-ambang (gaya F_all) ke
jumlah tandan per pohon, dilatih pada TRAIN, dipilih pada VAL, TEST sekali.
"""
import json, pickle, sys
import numpy as np
from collections import defaultdict
sys.path.insert(0, "/tmp/claude-1001/-workspace/ebcbd941-6775-4113-b727-404085458263/scratchpad")
from panen_pipeline import (evaluate, GT, RES, EMPAT_SISI, build_pairs, cluster_tree)
from sklearn.linear_model import RidgeCV

D = pickle.load(open(f"{RES}/dets.pkl", "rb"))
E = pickle.load(open(f"{RES}/edge_model.pkl", "rb"))
P = json.load(open(f"{RES}/panen_results.json"))["profil"]
print("profil topologi:", P, flush=True)
THR = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


def per_tree(sp, trees):
    """Klaster sekali per pohon, lalu rakit fitur pencacahan."""
    X, Y, meta = [], [], []
    for t in trees:
        raw = D[sp].get(t, [])
        ds = [d for d in raw if d["conf"] >= P["det_conf"]]
        if ds:
            Xp, _, keys = build_pairs({t: ds}, [t], labelled=False)
            pr = E.predict_proba(Xp)[:, 1] if len(Xp) else []
            ep = {(min(i, j), max(i, j)): float(p) for (_, i, j), p in zip(keys, pr)}
            groups = cluster_tree(ds, ep, P["link_thr"], P["max_size"])
        else:
            groups = []
        clus = []
        for g in groups:
            mem = [ds[i] for i in g]
            w = np.array([m["conf"] for m in mem])
            if len(mem) == 1 and w[0] < P["single_thr"]:
                continue
            clus.append(dict(n=len(mem), conf=float(w.mean()),
                             score=float(np.average([m["score"] for m in mem], weights=w))))
        f = []
        # statistik deteksi multi-ambang, per sisi dan total
        for th in THR:
            k = [d for d in raw if d["conf"] >= th]
            f.append(len(k))
            f.append(len(k) / 4.0)
            f.append(sum(1 for d in k if d["score"] < P["t_coarse"]))
        # statistik klaster
        for lo, hi in [(0, 99), (1, 1), (2, 2), (3, 99)]:
            f.append(sum(1 for c in clus if lo <= c["n"] <= hi))
        f += [sum(1 for c in clus if c["score"] < P["t_coarse"]),
              sum(1 for c in clus if c["score"] >= P["t_coarse"]),
              float(np.mean([c["conf"] for c in clus])) if clus else 0.,
              float(np.sum([c["conf"] for c in clus])) if clus else 0.,
              float(np.mean([c["n"] for c in clus])) if clus else 0.,
              len({d["side"] for d in ds})]
        gt = GT.get(t, [])
        X.append(f)
        Y.append([len(gt), sum(1 for b in gt if b["c"] <= 1), sum(1 for b in gt if b["c"] > 1)])
        meta.append(dict(tree=t, raw_clusters=len(clus),
                         raw_matang=sum(1 for c in clus if c["score"] < P["t_coarse"])))
    return np.array(X, float), np.array(Y, float), meta


TR = [t for t in D["train"] if t in EMPAT_SISI]
VA = [t for t in D["val"] if t in EMPAT_SISI]
TE = [t for t in D["test"] if t in EMPAT_SISI]
Xtr, Ytr, _ = per_tree("train", TR)
Xva, Yva, Mva = per_tree("val", VA)
Xte, Yte, Mte = per_tree("test", TE)
print(f"fitur {Xtr.shape[1]} · train {len(TR)} · val {len(VA)} · test {len(TE)}", flush=True)

mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
R = RidgeCV(alphas=np.logspace(-2, 4, 30)).fit((Xtr - mu) / sd, Ytr)


def report(tag, X, Y, meta):
    p = np.clip(np.round(R.predict((X - mu) / sd)), 0, None)
    raw = np.array([[m["raw_clusters"], m["raw_matang"],
                     m["raw_clusters"] - m["raw_matang"]] for m in meta], float)
    print(f"\n=== {tag} ===")
    out = {}
    for k, lab in enumerate(["total ", "MATANG", "BELUM "]):
        for name, pred in [("klaster mentah", raw[:, k]), ("Ridge         ", p[:, k])]:
            e = np.abs(pred - Y[:, k])
            print(f"  {lab} {name}: MAE {e.mean():.3f}  tepat {(e==0).mean():.3f}  "
                  f"±1 {(e<=1).mean():.3f}")
            out[f"{lab.strip()}_{name.strip()}"] = dict(
                mae=float(e.mean()), exact=float((e == 0).mean()),
                within1=float((e <= 1).mean()))
    return out


rv = report("VALIDATION", Xva, Yva, Mva)
rt = report("TEST (sekali)", Xte, Yte, Mte)
json.dump(dict(profil=P, val=rv, test=rt), open(f"{RES}/panen_count.json", "w"), indent=1)
print("\n--- pembanding: V2-E-045 total MAE 1,393 · ±1 0,6148 | GSP MAE 1,363 · ±1 0,6370")
print("COUNT DONE")
