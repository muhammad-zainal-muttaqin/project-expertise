"""PT-E-016c — CI tingkat pohon untuk selisih GNN vs baseline.

PT-E-016b memberi delta test F1 +1,06 pp dan ARI +3,45 pp, tapi ranking val dan
test berbalik (val: baseline menang; test: GNN menang). Ranking yang berbalik
antar-split adalah tanda selisihnya sebanding dengan derau, jadi delta titik
tanpa CI tidak boleh dibaca sebagai putusan.

Resampling di tingkat POHON, bukan pasangan: pasangan di dalam satu pohon jauh
dari independen (mereka berbagi kotak yang sama), jadi bootstrap per-pasangan
akan melaporkan CI yang terlalu sempit. Ini konvensi yang sama dengan
`EP.bootstrap_pohon`.

Ambang untuk masing-masing metode DIKUNCI dari val (baseline 0,25; GNN 0,90),
sama seperti PT-E-016b -- bukan dipilih ulang per resample, yang akan bocor.

Pemakaian:
    .venv/bin/python pipeline-pertandan/scripts/ci_gnn.py
"""

from __future__ import annotations

import itertools
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).parent))
import penaut_pertandan as PP           # noqa: E402
import gnn_penaut as GN                 # noqa: E402

SUB = PP.SUB
SEED = 0
N_BOOT = 2000


def stat_per_pohon(g, S, ambang):
    """tp/fp/fn + ARI + cakupan untuk SATU pohon, supaya bisa di-resample."""
    lab = GN.rakit(g, S, ambang)
    kotak = g["kotak"]
    gt = [b["bid"] if b["bid"] is not None else -1000 - k for k, b in enumerate(kotak)]
    tp = fp = fn = 0
    for i, j in itertools.combinations(range(len(kotak)), 2):
        if kotak[i]["s"] == kotak[j]["s"]:
            continue
        P, G = lab[i] == lab[j], gt[i] == gt[j]
        tp += P and G; fp += P and not G; fn += G and not P
    per_gt = defaultdict(list)
    for k, bid in enumerate(gt):
        per_gt[bid].append(k)
    n_multi = n_ketemu = 0
    for bid, anggota in per_gt.items():
        if bid < -999 or len(anggota) < 2:
            continue
        n_multi += 1
        if max(Counter(lab[k] for k in anggota).values()) >= 2:
            n_ketemu += 1
    return dict(tp=tp, fp=fp, fn=fn, ari=adjusted_rand_score(gt, lab),
                n_multi=n_multi, n_ketemu=n_ketemu)


def agregat(stats, idx):
    tp = sum(stats[i]["tp"] for i in idx)
    fp = sum(stats[i]["fp"] for i in idx)
    fn = sum(stats[i]["fn"] for i in idx)
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    nm = sum(stats[i]["n_multi"] for i in idx)
    nk = sum(stats[i]["n_ketemu"] for i in idx)
    return (2 * p * r / (p + r + 1e-9),
            float(np.mean([stats[i]["ari"] for i in idx])),
            nk / max(nm, 1))


def main() -> int:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    dev = "cpu"
    t0 = time.time()

    man = PP.muat_manifest()
    pohon = {s: [t for t, v in man.items() if v == s] for s in ("train", "val", "test")}
    semua = pohon["train"] + pohon["val"] + pohon["test"]
    desk = PP.bangun_deskriptor(semua, SUB / "results" / "deskriptor_crop.npz")
    z = np.load(SUB / "results" / "reid_embedding.npz", allow_pickle=True)
    emb = {k: z[k] for k in z.files}
    for fo in range(2):
        fz = SUB / "results" / f"reid_embedding_f{fo}.npz"
        if fz.exists():
            zz = np.load(fz, allow_pickle=True)
            ditahan = {t for i, t in enumerate(sorted(pohon["train"])) if i % 2 == fo}
            for k in zz.files:
                if k.split("|")[0] in ditahan:
                    emb[k] = zz[k]
    PP.HARAP = PP.hitung_harapan_geser(pohon["train"])
    prob = PP.bangun_prob_prediksi(pohon)
    graf = {s: [g for g in (GN.bangun_graf(t, desk, emb, prob) for t in pohon[s])
                if g is not None] for s in ("train", "test")}

    Xtr = np.concatenate([g["E"] for g in graf["train"]])
    ytr = np.concatenate([g["y"] for g in graf["train"]])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         random_state=SEED).fit(Xtr, ytr)
    ck = torch.load(SUB / "runs" / "gnn_penaut" / "best.pt", map_location=dev)
    m = GN.PenautGNN(ck["d_v"], ck["d_e"], ck["lebar"], ck["lapis"]).to(dev)
    m.load_state_dict(ck["model"])

    A_BASE, A_GNN = 0.25, 0.90          # dikunci dari val (PT-E-016b)
    st_b, st_g = [], []
    for g in graf["test"]:
        st_b.append(stat_per_pohon(g, clf.predict_proba(g["E"])[:, 1], A_BASE))
        st_g.append(stat_per_pohon(g, GN.skor_semua(m, [g], dev)[0], A_GNN))

    n = len(st_b)
    penuh = {"baseline": agregat(st_b, range(n)), "gnn": agregat(st_g, range(n))}
    rng = np.random.default_rng(SEED)
    d_f1, d_ari, d_cak = [], [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        fb, ab, cb = agregat(st_b, idx)
        fg, ag, cg = agregat(st_g, idx)
        d_f1.append(fg - fb); d_ari.append(ag - ab); d_cak.append(cg - cb)

    def ringkas(d, titik):
        d = np.array(d) * 100
        return {"delta_pp": round(titik * 100, 2),
                "ci95": [round(float(np.percentile(d, 2.5)), 2),
                         round(float(np.percentile(d, 97.5)), 2)],
                "P(delta>0)": round(float((d > 0).mean()), 3)}

    hasil = {"pt_e": "016c", "n_pohon_test": n, "n_boot": N_BOOT,
             "ambang": {"baseline": A_BASE, "gnn": A_GNN},
             "titik": {k: {"f1": round(v[0], 4), "ari": round(v[1], 4),
                           "cakupan": round(v[2], 4)} for k, v in penuh.items()},
             "delta": {
                 "f1": ringkas(d_f1, penuh["gnn"][0] - penuh["baseline"][0]),
                 "ari": ringkas(d_ari, penuh["gnn"][1] - penuh["baseline"][1]),
                 "cakupan": ringkas(d_cak, penuh["gnn"][2] - penuh["baseline"][2])}}
    hasil["detik"] = round(time.time() - t0, 1)
    f = SUB / "results" / "pt_e_016c_ci.json"
    f.write_text(json.dumps(hasil, indent=1, ensure_ascii=False))
    print(json.dumps(hasil, indent=1, ensure_ascii=False))
    print(f"-> {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
