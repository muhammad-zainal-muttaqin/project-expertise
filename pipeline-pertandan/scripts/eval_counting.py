"""PT-E-004 — Counting per pohon: pipeline per-tandan vs pipeline yang sudah ada.

Pertanyaannya: kalau tandan sudah dikelompokkan menjadi pool, apakah MENGHITUNG
POOL mengalahkan cara lama yang menaksir jumlah dari hitungan deteksi mentah?

Empat penghitung, semuanya dari DETEKSI YANG SAMA supaya perbandingannya adil:

  C1 naif           jumlah deteksi per kelas, apa adanya
  C2 k global       jumlah deteksi / 1,8905 (rasio duplikasi korpus)
  C3 k per kelas    jumlah deteksi / k_c, dengan k_c dipas di split train
  C4 hitung pool    jumlah POOL per kelas  <- pipeline ini
  C5 Ridge + F_all  fitur 67-dim -> RidgeCV; jalur counting yang sudah mapan
                    di repo (pola `../scripts/run_counting_v2repro.py`)

## Kaveat perbandingan yang WAJIB ikut dibaca

Angka `E-007/report_test.json` untuk "Koreksi global k=1,8905" (macro MAE
0,356) TIDAK sebanding dengan angka di sini: nilai sebagus itu hanya mungkin
kalau pembilangnya hitungan KOTAK GT, bukan deteksi. Di sini semua penghitung
memakai deteksi detektor yang sama, jadi yang dibandingkan adalah pilihan
penghitungnya — bukan mutu detektornya. Untuk konteks, varian berbasis kotak GT
ikut dihitung dan dilaporkan sebagai plafon.

## Metrik (mengikuti pelaporan repo induk)

  macro MAE      rata-rata |taksiran - benar| atas 4 kelas dan seluruh pohon
  class +-1 acc  fraksi sel (pohon, kelas) yang meleset <= 1
  tree +-1 acc   fraksi pohon yang KEEMPAT kelasnya meleset <= 1

Pemakaian:
    .venv/bin/python pipeline-pertandan/scripts/eval_counting.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import penaut_pertandan as PP           # noqa: E402
import eval_pertandan as EP             # noqa: E402
import eval_endtoend as EE              # noqa: E402
import reid_pertandan as RD              # noqa: E402

SUB = PP.SUB
KELAS = PP.KELAS
K_GLOBAL = 1.8905


def metrik(pred: np.ndarray, benar: np.ndarray) -> dict:
    d = np.abs(pred - benar)
    return {"macro_mae": round(float(d.mean()), 4),
            "class_pm1_acc": round(float((d <= 1).mean()), 4),
            "tree_pm1_acc": round(float((d <= 1).all(1).mean()), 4),
            "bias_total": round(float((pred - benar).sum(1).mean()), 4),
            "n_pohon": len(benar)}


def fitur_fall(per_sisi: list[Counter], skor: dict) -> list[float]:
    """F_all ringkas: statistik per kelas atas sisi + keyakinan + geometri.

    Mengikuti pola `../scripts/run_counting_v2repro.py::extract_all_features`.
    """
    f = []
    for k, c in enumerate(KELAS):
        ps = np.array([s[k] for s in per_sisi], float)
        cf = np.array(skor[c]["conf"]) if skor[c]["conf"] else np.zeros(0)
        ar = np.array(skor[c]["area"]) if skor[c]["area"] else np.zeros(0)
        cy = np.array(skor[c]["cy"]) if skor[c]["cy"] else np.zeros(0)
        n = len(cf)
        f += [ps.sum(), ps.max(), ps.mean(), ps.std(), ps.min(),
              ps.std() / (ps.mean() + 1e-6), float((ps > 0).sum()),
              1.0 / (1.0 + ps.std()),
              float(cf.sum()), float(cf.mean()) if n else 0.0,
              float(cf.max()) if n else 0.0,
              float((cf >= 0.5).sum()), float((cf >= 0.6).sum()),
              float(cy.mean()) if n else 0.5, float(ar.mean()) if n else 0.0,
              float(ar.std()) if n else 0.0]
    f.append(float(len(per_sisi)))
    f.append(float(sum(s.total() for s in per_sisi)))
    f.append(float(sum(s.total() for s in per_sisi)) / max(len(per_sisi), 1))
    return f


def kumpulkan(pohon, z, conf, clf, ambang, skema, tau, cfg=None):
    """Untuk tiap pohon: hitungan GT, hitungan deteksi, hitungan pool, fitur F_all."""
    baris = []
    for P in pohon:
        cfg = cfg or {}
        det = EE.deteksi_pohon(P, z, conf, cfg.get("reid"))
        benar = np.zeros(4)
        for kls in P["tandan"].values():
            benar[kls] += 1
        naif = np.zeros(4)
        per_sisi = defaultdict(Counter)
        skor = {c: {"conf": [], "area": [], "cy": []} for c in KELAS}
        for d in det:
            k = int(np.argmax(d["p"]))
            naif[k] += 1
            per_sisi[d["s"]][k] += 1
            skor[KELAS[k]]["conf"].append(d["conf"])
            skor[KELAS[k]]["area"].append(d["w"] * d["h"])
            skor[KELAS[k]]["cy"].append(d["cy"])
        pool = np.zeros(4)
        if det:
            lab = EE.klaster_deteksi(clf, P["n_sisi"], det, ambang,
                                     cfg.get("pakai_reid", False),
                                     cfg.get("pakai_kelas", True),
                                     cfg.get("pakai_prob", False))
            grup = defaultdict(list)
            for i, d in enumerate(det):
                grup[lab[i]].append(d)
            for anggota in grup.values():
                pool[EP.prediksi(anggota, "R4", skema, tau)] += 1
        sisi_urut = [per_sisi[s] for s in sorted(set(d["s"] for d in det))] or [Counter()]
        baris.append({"tree": P["tree"], "benar": benar, "naif": naif, "pool": pool,
                      "fitur": fitur_fall(sisi_urut, skor)})
    return baris


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keluaran", default=str(SUB / "results" / "pt_e_004_counting.json"))
    args = ap.parse_args()

    o = json.loads((SUB / "results" / "pt_e_001_oracle.json").read_text())
    conf, skema, tau = o["conf_dikunci"], o["skema_bobot_dikunci"], tuple(o["tau_R4_dikunci"])
    pen = json.loads((SUB / "results" / "pt_e_002_penaut.json").read_text())
    nama_var = pen.get("varian_dipakai_endtoend", pen["varian_terbaik_di_val"])
    V = pen["varian"][nama_var]
    ambang = V["ambang_dikunci_dari_val"]
    cfg = {"pakai_kelas": V.get("pakai_kelas_sama", True),
           "pakai_prob": V.get("pakai_prob_prediksi", False),
           "pakai_reid": V.get("pakai_reid", False), "reid": None}
    print(f"penaut dipakai: {nama_var} (ambang {ambang}, cfg {cfg})")

    man = PP.muat_manifest()
    ids = {s: [t for t, v in man.items() if v == s] for s in ["train", "val", "test"]}
    desk = PP.bangun_deskriptor(ids["train"] + ids["val"] + ids["test"],
                                SUB / "results" / "deskriptor_crop.npz")
    # WAJIB: konstanta pergeseran arah-putar hidup sebagai global di
    # penaut_pertandan dan hanya terisi saat skrip itu dijalankan langsung.
    # Tanpa baris ini, fitur arah TIDAK aktif di sini dan hasilnya diam-diam
    # kembali ke fitur lama — terlihat sebagai angka yang identik persis.
    PP.HARAP = PP.hitung_harapan_geser(ids["train"])
    print(f"konstanta arah-putar dipas di train: {len(PP.HARAP)} entri")
    from sklearn.ensemble import HistGradientBoostingClassifier
    emb = None
    if cfg["pakai_reid"]:
        zz = np.load(SUB / "results" / "reid_embedding.npz", allow_pickle=True)
        emb = {k: zz[k] for k in zz.files}
        import torch
        m = RD.Reid().cuda().eval()
        m.load_state_dict(torch.load(SUB / "runs" / "reid_resnet18" / "best.pt"))

        def _reid(crops):
            out = []
            with torch.no_grad():
                for i in range(0, len(crops), 256):
                    out.append(m(RD.ke_tensor(crops[i:i + 256], False, "cuda"))
                               .float().cpu().numpy())
            return np.concatenate(out)
        cfg["reid"] = _reid
    prob = PP.bangun_prob_prediksi(
        {k: ids[k] for k in ["train", "val", "test"]}) if cfg["pakai_prob"] else None
    Xtr, ytr = PP.pasangan(ids["train"], desk, True, emb, cfg["pakai_kelas"], prob)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         random_state=PP.SEED).fit(Xtr, ytr)

    splits = ["train", "val", "test"]
    z = {s: np.load(SUB / "results" / f"pred_skorpenuh_{s}.npz", allow_pickle=True)
         for s in splits}
    data = {}
    for s in splits:
        print(f"mengumpulkan {s}...")
        data[s] = kumpulkan([EP.muat_pohon(t) for t in ids[s]], z[s], conf,
                            clf, ambang, skema, tau, cfg)

    B = {s: np.stack([r["benar"] for r in data[s]]) for s in data}
    N = {s: np.stack([r["naif"] for r in data[s]]) for s in data}
    PO = {s: np.stack([r["pool"] for r in data[s]]) for s in data}
    F = {s: np.array([r["fitur"] for r in data[s]], float) for s in data}

    kc = N["train"].sum(0) / np.maximum(B["train"].sum(0), 1)   # k per kelas dari train
    ridge = Pipeline([("sc", StandardScaler()),
                      ("rg", RidgeCV(alphas=np.logspace(-3, 3, 25)))]).fit(F["train"], B["train"])

    hasil = {"conf": conf, "ambang_penaut": ambang, "tau": list(tau),
             "k_global": K_GLOBAL, "k_per_kelas_dari_train": [round(float(x), 4) for x in kc],
             "catatan_pas": "C3 dan C5 dipas di TRAIN (716 pohon), dievaluasi di TEST — protokol repo induk",
             "split": {}}
    for s in splits:
        hasil["split"][s] = {
            "C1_naif": metrik(N[s], B[s]),
            "C2_k_global": metrik(N[s] / K_GLOBAL, B[s]),
            "C3_k_per_kelas": metrik(N[s] / kc, B[s]),
            "C4_hitung_pool": metrik(PO[s], B[s]),
            "C5_ridge_fall": metrik(np.clip(ridge.predict(F[s]), 0, None), B[s]),
        }
        print(f"\n--- {s} ---")
        for k, v in hasil["split"][s].items():
            print(f"  {k:16s} macroMAE {v['macro_mae']:.4f}  class+-1 {v['class_pm1_acc']:.4f}"
                  f"  tree+-1 {v['tree_pm1_acc']:.4f}  bias {v['bias_total']:+.2f}")

    t = hasil["split"]["test"]
    hasil["gerbang_G3"] = {
        "syarat": "macro MAE C4 (hitung pool) < pembanding terbaik lainnya di test",
        "C4": t["C4_hitung_pool"]["macro_mae"],
        "pembanding_terbaik": min((v["macro_mae"], k) for k, v in t.items()
                                  if k != "C4_hitung_pool"),
        "putusan": ("LOLOS" if t["C4_hitung_pool"]["macro_mae"] <
                    min(v["macro_mae"] for k, v in t.items() if k != "C4_hitung_pool")
                    else "GUGUR"),
    }
    Path(args.keluaran).write_text(json.dumps(hasil, indent=1, ensure_ascii=False))
    print("\n" + json.dumps(hasil["gerbang_G3"], indent=1, ensure_ascii=False))
    print(f"-> {args.keluaran}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
