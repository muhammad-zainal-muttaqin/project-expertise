"""PT-E-016b — sapuan ambang GNN dengan grid diperluas.

Run pertama PT-E-016 memakai grid ambang [0,10 .. 0,50] yang diwarisi dari
baseline. Untuk baseline grid itu cukup: F1 val-nya memuncak di 0,25, optimum
interior yang sah. Untuk GNN tidak: F1 val-nya naik MONOTON sampai 0,50 lalu
grid-nya habis, jadi konfigurasi terbaiknya tidak pernah diuji dan selisih
-3,36 pp yang tercatat mengukur GNN yang sengaja dilumpuhkan.

Penyebabnya bisa dijelaskan: skor GNN dilatih dengan `pos_weight` 14,2 (dari
prevalensi positif 6,6%), yang menggeser seluruh distribusi skor ke atas. Ambang
yang setara untuk baseline dan GNN karena itu tidak sama angkanya.

Tidak melatih ulang apa pun -- bobot dimuat dari `runs/gnn_penaut/best.pt`,
graf dibangun ulang identik (seed sama, fitur sama).

Pemakaian:
    .venv/bin/python pipeline-pertandan/scripts/sapu_ambang_gnn.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import penaut_pertandan as PP           # noqa: E402
import gnn_penaut as GN                 # noqa: E402

SUB = PP.SUB
SEED = 0
# grid diperluas: rapat di daerah baru, tetap memuat rentang lama supaya
# angka run pertama bisa dicocokkan ulang
GRID = [round(x, 3) for x in
        [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
         0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975]]


def main() -> int:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    dev = "cpu"          # GPU sedang dipakai rantai; skoring GNN murah di CPU
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

    print("membangun graf...")
    graf = {s: [g for g in (GN.bangun_graf(t, desk, emb, prob) for t in pohon[s])
                if g is not None] for s in pohon}

    # baseline: dilatih ulang identik (deterministik, seed sama)
    Xtr = np.concatenate([g["E"] for g in graf["train"]])
    ytr = np.concatenate([g["y"] for g in graf["train"]])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         random_state=SEED).fit(Xtr, ytr)
    skor_b = {s: [clf.predict_proba(g["E"])[:, 1] for g in graf[s]] for s in graf}

    ck = torch.load(SUB / "runs" / "gnn_penaut" / "best.pt", map_location=dev)
    m = GN.PenautGNN(ck["d_v"], ck["d_e"], ck["lebar"], ck["lapis"]).to(dev)
    m.load_state_dict(ck["model"])
    skor_g = {s: GN.skor_semua(m, graf[s], dev) for s in graf}

    hasil = {"pt_e": "016b", "grid": GRID, "catatan":
             "sapuan ulang; tidak ada training baru", "sapuan_val": {}}
    for nama, S in (("baseline", skor_b), ("gnn", skor_g)):
        auc = float(roc_auc_score(np.concatenate([g["y"] for g in graf["val"]]),
                                  np.concatenate(S["val"])))
        sap = {}
        print(f"\n=== {nama} (AUC val {auc:.4f}) ===")
        for a in GRID:
            sap[f"{a:.3f}"] = GN.nilai(graf["val"], S["val"], a)
            r = sap[f"{a:.3f}"]
            print(f"  ambang {a:.3f}: F1 {r['f1']:.4f} ARI {r['ari']:.4f} "
                  f"presisi {r['presisi']:.4f} recall {r['recall']:.4f} "
                  f"cakupan {r['cakupan_tandan']:.4f} bias {r['bias_jumlah']:+.2f}")
        best = max(GRID, key=lambda a: sap[f"{a:.3f}"]["f1"])
        di_tepi = best in (GRID[0], GRID[-1])
        hasil["sapuan_val"][nama] = {
            "auc_val": round(auc, 4), "sapuan": sap,
            "ambang_terbaik_val": best, "di_tepi_grid": di_tepi,
            "val": sap[f"{best:.3f}"],
            "test_sekali": GN.nilai(graf["test"], S["test"], best)}
        print(f"  -> ambang val {best} (di tepi grid: {di_tepi})")
        print(f"  TEST: {hasil['sapuan_val'][nama]['test_sekali']}")

    b = hasil["sapuan_val"]["baseline"]; g = hasil["sapuan_val"]["gnn"]
    hasil["putusan"] = {
        "ambang_baseline": b["ambang_terbaik_val"],
        "ambang_gnn": g["ambang_terbaik_val"],
        "salah_satu_di_tepi_grid": b["di_tepi_grid"] or g["di_tepi_grid"],
        "delta_f1_test_pp": round((g["test_sekali"]["f1"] - b["test_sekali"]["f1"]) * 100, 2),
        "delta_ari_test_pp": round((g["test_sekali"]["ari"] - b["test_sekali"]["ari"]) * 100, 2),
        "delta_cakupan_test_pp": round((g["test_sekali"]["cakupan_tandan"] -
                                        b["test_sekali"]["cakupan_tandan"]) * 100, 2),
        "delta_f1_run_pertama_pp": -3.36,
    }
    hasil["detik"] = round(time.time() - t0, 1)
    f = SUB / "results" / "pt_e_016b_sapu_ambang.json"
    f.write_text(json.dumps(hasil, indent=1, ensure_ascii=False))
    print("\n" + json.dumps(hasil["putusan"], indent=1, ensure_ascii=False))
    print(f"-> {f}  ({hasil['detik']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
