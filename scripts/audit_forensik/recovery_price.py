"""Berapa harga presisi untuk memulihkan 227 tandan yang punya kandidat?

`recovery_budget_val.json` membuktikan kandidatnya ADA. Yang belum diukur:
apakah kandidat itu dapat dipertahankan tanpa meruntuhkan presisi. Skrip ini
menurunkan ambang dan mencatat kurva operasi identitas fisik pada VAL, lalu
membandingkannya dengan batas atas penyaring sempurna (oracle).

Jika presisi bertahan saat recall naik  -> penyaring terpelajar punya ruang nyata.
Jika presisi runtuh seketika            -> kandidat itu tidak terpisahkan dari
                                           derau oleh ambang apa pun, sehingga
                                           model baru harus menemukan sinyal baru.
"""
import json, pickle, sys
import numpy as np
from collections import defaultdict
sys.path.insert(0, "/tmp/claude-1001/-workspace/ebcbd941-6775-4113-b727-404085458263/scratchpad")
from panen_pipeline import GT, RES, EMPAT_SISI, build_pairs, cluster_tree, iou1, yolo_to_xyxy

D = pickle.load(open(f"{RES}/dets.pkl", "rb"))
E = pickle.load(open(f"{RES}/edge_model_v2.pkl", "rb"))
P = json.load(open(f"{RES}/panen_final.json"))["profil"]
VA = [t for t in D["val"] if t in EMPAT_SISI]
print(f"VAL {len(VA)} pohon · {sum(len(GT[t]) for t in VA)} tandan acuan", flush=True)


def run(det_conf, single_thr, link_thr, max_size, oracle_filter=False):
    TP = FP = FN = 0
    for t in VA:
        ds = [d for d in D["val"].get(t, []) if d["conf"] >= det_conf]
        gt = GT.get(t, [])
        if ds:
            X, _, keys = build_pairs({t: ds}, [t], labelled=False)
            pr = E.predict_proba(X)[:, 1] if len(X) else []
            ep = {(min(i, j), max(i, j)): float(p) for (_, i, j), p in zip(keys, pr)}
            groups = cluster_tree(ds, ep, link_thr, max_size)
        else:
            groups = []
        clus = []
        for g in groups:
            mem = [ds[i] for i in g]
            w = np.array([m["conf"] for m in mem])
            if len(mem) == 1 and w[0] < single_thr:
                continue
            clus.append(dict(members=mem, conf=float(w.mean())))
        used, match = set(), 0
        for c in sorted(clus, key=lambda k: -k["conf"]):
            best, bi = 0.5, -1
            for gi, b in enumerate(gt):
                if gi in used:
                    continue
                v = 0.
                for m in c["members"]:
                    for s, bb in b["app"]:
                        if s == m["side"]:
                            v = max(v, iou1(m["box"], yolo_to_xyxy(bb)))
                if v > best:
                    best, bi = v, gi
            if bi >= 0:
                used.add(bi); match += 1
        TP += match
        FP += (0 if oracle_filter else len(clus) - match)   # oracle: buang semua FP
        FN += len(gt) - match
    P_ = TP / max(TP + FP, 1); R_ = TP / max(TP + FN, 1)
    return dict(tp=TP, fp=FP, fn=FN, precision=P_, recall=R_,
                f1=2 * P_ * R_ / max(P_ + R_, 1e-9))


print("\n=== kurva operasi identitas fisik (VAL, penaut & link_thr tetap) ===")
print(f"{'det_conf':>9}{'single':>8}{'recall':>9}{'presisi':>9}{'F1':>8}{'TP':>7}{'FP':>7}")
base = None
rows = []
for dc, st in [(0.30, 0.45), (0.25, 0.40), (0.20, 0.35), (0.15, 0.30),
               (0.12, 0.25), (0.10, 0.20), (0.10, 0.10), (0.10, 0.00)]:
    r = run(dc, st, P["link_thr"], P["max_size"])
    rows.append(dict(det_conf=dc, single_thr=st, **r))
    if base is None:
        base = r
    print(f"{dc:>9.2f}{st:>8.2f}{r['recall']:>9.4f}{r['precision']:>9.4f}"
          f"{r['f1']:>8.4f}{r['tp']:>7d}{r['fp']:>7d}")

print("\n=== batas atas: kandidat longgar + penyaring SEMPURNA (oracle) ===")
orc = run(0.10, 0.00, P["link_thr"], P["max_size"], oracle_filter=True)
print(f"  recall {orc['recall']:.4f} · TP {orc['tp']} · FN {orc['fn']}  "
      f"(presisi 1,0 menurut definisi)")

loose = rows[-1]
print(f"\n=== harga pemulihan ===")
print(f"  profil terpilih : recall {base['recall']:.4f} presisi {base['precision']:.4f} "
      f"(TP {base['tp']}, FP {base['fp']})")
print(f"  ambang longgar  : recall {loose['recall']:.4f} presisi {loose['precision']:.4f} "
      f"(TP {loose['tp']}, FP {loose['fp']})")
d_tp = loose['tp'] - base['tp']; d_fp = loose['fp'] - base['fp']
print(f"  Δ               : +{d_tp} tandan dipulihkan dengan +{d_fp} positif palsu "
      f"-> {d_fp/max(d_tp,1):.1f} FP per tandan")
print(f"  penyaring terpelajar harus membuang {d_fp} dari {loose['tp']+loose['fp']} klaster "
      f"sambil mempertahankan {loose['tp']}")
print(f"  batas atas recall dengan kandidat longgar: {orc['recall']:.4f} "
      f"({orc['fn']} tandan tetap tak terjangkau)")

json.dump(dict(curve=rows, oracle_loose=orc,
               recovered=d_tp, fp_cost=d_fp, fp_per_recovered=d_fp / max(d_tp, 1)),
          open("/workspace/results_panen/recovery_price_val.json", "w"), indent=1)
