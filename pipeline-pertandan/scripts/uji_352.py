"""PT-E-010 — Konfigurasi terbaik diuji ulang di SawitMVC-Depth (352 pohon, RGB).

Satu percobaan, konfigurasi terbaik apa adanya. Tujuannya bukan menyetel ulang
melainkan menjawab: **apakah temuan di korpus 953 bertahan di sesi akuisisi yang
berbeda?**

Kenapa uji ini kuat. SawitMVC-Depth direkam terpisah, ~80 hari setelah korpus
953 (`../results/pergeseran_temporal.json`), dengan kamera berbeda dan citra
landscape 1280x800 alih-alih portrait 960x1280. Kalau temuan intinya bertahan di
sana, ia sifat protokol pengambilan — bukan kebetulan satu sesi.

Bukti awal yang sudah dihitung sebelum skrip ini (probe langsung, 352 pohon):

    offset +1  ->  dx +0,163, 98,4% ke kanan   (953: +0,241, 98,6%)
    offset +3  ->  dx -0,175, 99,0% ke kiri    (953: -0,260, 99,7%)
    pasangan salah -> ~0,000, ~50/50           (sama)

Arah dan konsistensinya identik; besarannya lebih kecil karena citranya lebih
lebar, jadi pergeseran sudut yang sama menutupi fraksi lebar yang lebih kecil.

## Konfigurasi (dikunci dari korpus 953, TIDAK disetel ulang di sini)

    detektor   runs/yolo26l_e60_i1280_rgb352/weights/best.pt  (detektor 352 sendiri)
    penaut     varian E — geometri + arah putar + kelas prediksi lunak + re-ID
    re-ID      bobot dilatih di 953; DIPINDAH apa adanya (uji transfer)
    aturan     R4 ekspektasi ordinal
    conf       0,10   (PT-E-009: sudah optimal, menaikkannya memburuk monoton)

Konstanta arah putar dan ambang ordinal `tau` dipas ulang di split TRAIN 352 —
itu bukan penyetelan konfigurasi, melainkan kalibrasi yang memang harus mengikuti
datanya.

Pemakaian:
    .venv/bin/python pipeline-pertandan/scripts/uji_352.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).parent))
import penaut_pertandan as PP           # noqa: E402
import eval_pertandan as EP             # noqa: E402
import eval_endtoend as EE              # noqa: E402
import reid_pertandan as RD             # noqa: E402

D352 = Path("/workspace/SawitMVC-Depth")
SPLITDIR = D352 / "splits" / "canonical_70_15_15"
SUB = PP.SUB


def pasang_profil_352():
    """Arahkan seluruh modul ke dataset 352. Skemanya identik dengan 953
    (diverifikasi di `../docs/SCHEMA-PERTREE.md`), jadi cukup mengganti akar,
    daftar split, dan tata letak citra."""
    PP.DS = D352
    EP.DS = D352
    EE.DS = D352
    RD.DS = D352
    PP.TAG = "_352"

    def manifest_352():
        man = {}
        for s in ("train", "val", "test"):
            for t in (SPLITDIR / f"{s}_trees.txt").read_text().splitlines():
                if t.strip():
                    man[t.strip()] = s
        return man

    def cari_citra_352(stem: str):
        q = D352 / "images" / f"{stem}.jpg"
        return q if q.exists() else None

    PP.muat_manifest = manifest_352
    PP.cari_citra = cari_citra_352


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--ambang", nargs="+", type=float,
                    default=[0.25, 0.45, 0.65])
    ap.add_argument("--keluaran", default=str(SUB / "results" / "pt_e_010_uji_352.json"))
    args = ap.parse_args()

    pasang_profil_352()
    man = PP.muat_manifest()
    ids = {s: [t for t, v in man.items() if v == s] for s in ("train", "val", "test")}
    print(f"352 pohon: train {len(ids['train'])} / val {len(ids['val'])} / "
          f"test {len(ids['test'])}")

    PP.HARAP = PP.hitung_harapan_geser(ids["train"])
    print("konstanta arah-putar dipas di TRAIN 352:")
    for k in sorted(PP.HARAP):
        print(f"    n_sisi={k[0]} offset={k[1]}: {PP.HARAP[k]:+.3f}")

    desk = PP.bangun_deskriptor(ids["train"] + ids["val"] + ids["test"],
                                SUB / "results" / "deskriptor_crop_352.npz")
    print(f"deskriptor: {len(desk)} potongan")

    # re-ID: bobot 953 dipindah apa adanya
    import torch
    model = RD.Reid().cuda().eval()
    model.load_state_dict(torch.load(SUB / "runs" / "reid_resnet18" / "best.pt"))

    def reid_fn(crops):
        keluar = []
        with torch.no_grad():
            for i in range(0, len(crops), 256):
                keluar.append(model(RD.ke_tensor(crops[i:i+256], False, "cuda"))
                              .float().cpu().numpy())
        return np.concatenate(keluar)

    img, kunci, _, _ = RD.bangun_potongan(
        ids["train"] + ids["val"] + ids["test"],
        SUB / "results" / "potongan_reid_352.npz")
    E = reid_fn(img)
    emb = {k: e for k, e in zip(kunci, E)}
    print(f"embedding re-ID (transfer dari 953): {len(emb)} potongan")

    prob = PP.bangun_prob_prediksi({k: ids[k] for k in ("train", "val", "test")})
    print("melatih penaut di TRAIN 352...")
    Xtr, ytr = PP.pasangan(ids["train"], desk, True, emb, False, prob)
    print(f"  {len(ytr)} pasangan, {int(ytr.sum())} positif ({100*ytr.mean():.2f}%)")
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         random_state=PP.SEED).fit(Xtr, ytr)

    # mutu penautan di kotak GT (sebanding dengan PT-E-002/008)
    penaut = {}
    terbaik = (None, -1.0)
    for a in args.ambang:
        m = PP.nilai_klaster(clf, ids["val"], desk, True, a, emb, False, prob)
        penaut[f"{a:.2f}"] = m
        print(f"  penaut val ambang {a:.2f}: F1 {m['f1']} ARI {m['ari']} "
              f"MAE_n {m['mae_jumlah']}")
        if m["f1"] > terbaik[1]:
            terbaik = (a, m["f1"])
    ambang = terbaik[0]
    penaut_test = PP.nilai_klaster(clf, ids["test"], desk, True, ambang, emb, False, prob)
    print(f"  penaut TEST (ambang {ambang}): {penaut_test}")

    # pipeline utuh
    cfg = {"pakai_kelas": False, "pakai_prob": True, "pakai_reid": True}
    pohon = {s: [EP.muat_pohon(t) for t in ids[s]] for s in ("val", "test")}
    z = {s: np.load(SUB / "results" / f"pred_skorpenuh_352_{s}.npz", allow_pickle=True)
         for s in ("val", "test")}

    skema = "conf_luas"
    pv = EP.bangun_pool(pohon["val"], z["val"], args.conf)
    tau = EP.cari_tau(pv, skema)
    print(f"tau dipas di val 352: {tau}")

    hasil = {"dataset": str(D352), "n_pohon": {k: len(v) for k, v in ids.items()},
             "detektor": "runs/yolo26l_e60_i1280_rgb352/weights/best.pt",
             "reid": "bobot 953 dipindah apa adanya (transfer)",
             "conf": args.conf, "skema": skema, "tau": list(tau),
             "harapan_geser": {f"{k[0]}|{k[1]}": round(v, 5)
                               for k, v in sorted(PP.HARAP.items())},
             "penaut_kotak_GT": {"sapuan_val": penaut, "ambang_dikunci": ambang,
                                 "test": penaut_test},
             "split": {}}

    for s in ("val", "test"):
        pools = EP.bangun_pool(pohon[s], z[s], args.conf)
        blok = {"ORACLE": {a: EP.nilai(pools, a, skema, tau)
                           for a in ("R0", "R0cal", "R2", "R4")},
                "cakupan": EP.cakupan(pohon[s], pools),
                "PEMBANDING_per_kemunculan": EP.akurasi_per_kemunculan(pools),
                "n_pool": len(pools),
                "n_multi": sum(1 for q in pools if len(q["pool"]) >= 2)}
        multi = [q for q in pools if len(q["pool"]) >= 2]
        blok["ORACLE_multi"] = {a: EP.nilai(multi, a, skema, tau)
                                for a in ("R0cal", "R4")}
        blok["G0_penggabungan_multi"] = EP.bootstrap_pohon(multi, "R4", "R0cal",
                                                           skema, tau)
        blok["rekalibrasi"] = EP.bootstrap_pohon(pools, "R0cal", "R0", skema, tau)
        blok["endtoend"] = EE.jalankan(s, pohon[s], z[s], clf, ambang, args.conf,
                                       skema, tau, reid_fn, True, False, True)
        hasil["split"][s] = blok
        g = blok["G0_penggabungan_multi"]
        e = blok["endtoend"]
        print(f"\n--- {s} ---")
        print(f"  cakupan: recall/tandan {blok['cakupan']['recall_per_tandan']} "
              f"recall/kemunculan {blok['cakupan']['recall_per_kemunculan']}")
        print(f"  ORACLE  R0 {blok['ORACLE']['R0']['akurasi']} "
              f"R2 {blok['ORACLE']['R2']['akurasi']} R4 {blok['ORACLE']['R4']['akurasi']}")
        print(f"  G0 penggabungan (pool >=2): {g['delta_pp']:+.2f} pp "
              f"CI95 [{g['ci95_pp'][0]:+.2f}, {g['ci95_pp'][1]:+.2f}]")
        print(f"  rekalibrasi: {blok['rekalibrasi']['delta_pp']:+.2f} pp")
        print(f"  END-TO-END R4 {e['PENAUT_NYATA']['R4']['akurasi']} "
              f"(oracle {e['TAUTAN_ORACLE']['R4']['akurasi']}), "
              f"F1 penaut {e['penautan_di_atas_deteksi']['f1']}")
        print(f"  PEMBANDING per-kemunculan (pipeline lama): "
              f"{blok['PEMBANDING_per_kemunculan']['akurasi']}")

    t = hasil["split"]["test"]
    hasil["ringkas_test"] = {
        "recall_per_kemunculan": t["cakupan"]["recall_per_kemunculan"],
        "recall_per_tandan": t["cakupan"]["recall_per_tandan"],
        "kelas_per_kemunculan_lama": t["PEMBANDING_per_kemunculan"]["akurasi"],
        "kelas_per_tandan_oracle_R4": t["ORACLE"]["R4"]["akurasi"],
        "kelas_per_tandan_endtoend_R4": t["endtoend"]["PENAUT_NYATA"]["R4"]["akurasi"],
        "G0_penggabungan_pp": t["G0_penggabungan_multi"]["delta_pp"],
        "G0_ci95": t["G0_penggabungan_multi"]["ci95_pp"],
        "penaut_F1_kotakGT": penaut_test["f1"],
        "penaut_ARI_kotakGT": penaut_test["ari"],
    }
    Path(args.keluaran).write_text(json.dumps(hasil, indent=1, ensure_ascii=False,
                                              default=float))
    print("\n" + json.dumps(hasil["ringkas_test"], indent=1, ensure_ascii=False))
    print(f"-> {args.keluaran}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
