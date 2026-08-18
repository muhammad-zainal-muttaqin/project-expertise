"""Baseline dan plafon oracle per-tandan khusus DAMIMAS.

Ambang operasional dipilih di validation dengan sasaran end-to-end yang mudah
ditafsirkan: fraksi seluruh tandan fisik yang terdeteksi *dan* kelas akhirnya
benar. Test hanya dinilai setelah ambang, pembobotan view, dan tau ordinal tetap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).parent))
import eval_pertandan as EP  # noqa: E402


SUB = EP.SUB
KELAS = EP.KELAS


def rinci_kelas(pools, skema, tau):
    y = np.array([p["gt"] for p in pools], int)
    yh = np.array([EP.prediksi(p["pool"], "R4", skema, tau) for p in pools], int)
    pr, rc, f1, sup = precision_recall_fscore_support(
        y, yh, labels=np.arange(4), zero_division=0)
    return {KELAS[k]: {"precision": float(pr[k]), "recall": float(rc[k]),
                       "f1": float(f1[k]), "support_terdeteksi": int(sup[k])}
            for k in range(4)}


def recall_fisik_per_kelas(pohon, pools):
    gt = np.zeros(4, int); kena = np.zeros(4, int)
    for p in pohon:
        for c in p["tandan"].values():
            gt[c] += 1
    for p in pools:
        kena[p["gt"]] += 1
    return {KELAS[k]: {"gt": int(gt[k]), "terdeteksi": int(kena[k]),
                       "recall": float(kena[k] / max(gt[k], 1))}
            for k in range(4)}


def blok(pohon, pools, skema, tau):
    dasar = EP.blok_split(pohon, pools, skema, tau)
    dasar["R4_precision_recall_f1_per_kelas"] = rinci_kelas(pools, skema, tau)
    dasar["recall_fisik_per_kelas"] = recall_fisik_per_kelas(pohon, pools)
    dasar["akurasi_x_recall_fisik"] = round(
        dasar["semua_pool"]["R4"]["akurasi"] *
        dasar["cakupan"]["recall_per_tandan"], 4)
    return dasar


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, nargs="+",
                    default=[.01, .03, .05, .075, .10, .15, .20, .25, .30])
    ap.add_argument("--pred-val", type=Path,
                    default=SUB / "results" / "pred_skorpenuh_val.npz")
    ap.add_argument("--pred-test", type=Path,
                    default=SUB / "results" / "pred_skorpenuh_test.npz")
    ap.add_argument("--nama-detektor", default="YOLO26l 953 lama, difilter DAMIMAS")
    ap.add_argument("--keluaran", type=Path,
                    default=SUB / "results" / "damimas_baseline_pertandan.json")
    args = ap.parse_args()

    man = EP.muat_manifest()
    ids = {s: [t for t, sp in man.items()
               if sp == s and t.startswith("DAMIMAS_")] for s in ("val", "test")}
    pohon = {s: [EP.muat_pohon(t) for t in ids[s]] for s in ids}
    z = {"val": np.load(args.pred_val, allow_pickle=True),
         "test": np.load(args.pred_test, allow_pickle=True)}

    sapuan, terbaik = {}, None
    for conf in args.conf:
        pv = EP.bangun_pool(pohon["val"], z["val"], conf)
        skema = max(("seragam", "conf", "luas", "conf_luas", "conf_luas_tepi"),
                    key=lambda x: EP.nilai(pv, "R3", x, (.5, 1.5, 2.5))["akurasi"])
        tau = EP.cari_tau(pv, skema)
        m = EP.nilai(pv, "R4", skema, tau)
        cak = EP.cakupan(pohon["val"], pv)
        util = m["akurasi"] * cak["recall_per_tandan"]
        sapuan[f"{conf:.3f}"] = {"skema": skema, "tau": tau,
                                  "akurasi_R4": m["akurasi"],
                                  "macro_f1": m["macro_f1"],
                                  "recall_fisik": cak["recall_per_tandan"],
                                  "akurasi_x_recall": round(util, 6),
                                  "n_tandan": len(pv)}
        kunci = (util, m["akurasi"], m["macro_f1"], -conf)
        if terbaik is None or kunci > terbaik[0]:
            terbaik = (kunci, conf, skema, tau)
        print(f"conf={conf:.3f} recall={cak['recall_per_tandan']:.4f} "
              f"R4={m['akurasi']:.4f} util={util:.4f}", flush=True)

    _, conf, skema, tau = terbaik
    pools = {s: EP.bangun_pool(pohon[s], z[s], conf) for s in ids}
    hasil = {
        "dataset": "SawitMVC-YOLO-Damimas",
        "tautan": "oracle; plafon klasifikasi/agregasi, bukan hasil deploy",
        "detektor": args.nama_detektor,
        "prediksi": {"val": str(args.pred_val), "test": str(args.pred_test)},
        "pemilihan_val": "maksimum accuracy_R4 * recall_fisik",
        "sapuan_val": sapuan,
        "terkunci": {"conf": conf, "skema": skema, "tau": tau},
        "split": {s: blok(pohon[s], pools[s], skema, tau) for s in ids},
    }
    args.keluaran.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
    print(json.dumps({s: {"cakupan": hasil["split"][s]["cakupan"],
                          "R4": hasil["split"][s]["semua_pool"]["R4"],
                          "util": hasil["split"][s]["akurasi_x_recall_fisik"]}
                      for s in ids}, indent=2, ensure_ascii=False))
    print(f"-> {args.keluaran}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
