"""PT-E-006 — Menguji penghitung Baseline-SawitMVC (M01-M05) di deteksi yang sama.

Konteks. Repo `ULM-SawitMVC/Baseline-SawitMVC` memuat lima algoritma dedup
heuristik (`algorithms/M0*.py`) dengan angka tercatat **Acc+-1 87,62% dan macro
MAE 0,375 pada 953 pohon**. Angka itu jauh lebih baik daripada apa pun yang
diukur di PT-E-004 (terbaik: Ridge+F_all, macro MAE 1,0542 di test).

Selisih sebesar itu hampir pasti bukan soal algoritmanya, melainkan soal
**masukannya**. Spesifikasi input di `algorithms/README.md` menyebut
"From YOLO labels: column 1 = x_norm" — dan berkas `labels/*.txt` di
SawitMVC-YOLO adalah KOTAK GT, bukan deteksi. README utama repo itu juga
memisahkan keduanya secara eksplisit: dengan deteksi GT penghitung terbaik
mencapai 98,05% Class+-1, sedangkan dengan deteksi YOLO26m nyata hanya 77,48%.

Skrip ini menutup dugaan itu dengan mengukur langsung: algoritma yang sama,
dua jenis masukan.

  masukan A  kotak GT           -> mestinya mereproduksi ~0,375
  masukan B  deteksi YOLO26l    -> angka yang sebanding dengan PT-E-004

Yang TIDAK dilakukan: menyalahkan algoritmanya. Kalau A tereproduksi dan B jauh
lebih buruk, kesimpulannya adalah keduanya menjawab pertanyaan berbeda, bukan
bahwa salah satu keliru.

Pemakaian:
    .venv/bin/python pipeline-pertandan/scripts/eval_counting_baseline.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import penaut_pertandan as PP           # noqa: E402

BASE = Path("/workspace/Baseline-SawitMVC")
SUB = PP.SUB
DS = PP.DS
KELAS = PP.KELAS


def muat_algoritma():
    sys.path.insert(0, str(BASE))
    from algorithms import RANKING       # noqa: E402
    return {k: v["predict"] for k, v in RANKING.items()}


def deteksi_gt(tree: str) -> list[dict]:
    """Kotak GT sebagai 'deteksi' — masukan A."""
    d = json.loads((DS / "json" / f"{tree}.json").read_text(encoding="utf-8-sig"))
    out = []
    for im in d["images"].values():
        for a in im["annotations"]:
            cx, cy = a["bbox_yolo"][0], a["bbox_yolo"][1]
            out.append({"class": a["class_name"], "x_norm": float(cx),
                        "y_norm": float(cy), "side_index": int(im["side_index"])})
    return out


def deteksi_detektor(tree: str, z, conf: float) -> list[dict]:
    """Deteksi YOLO26l — masukan B. Disatukan per anchor dulu."""
    d = json.loads((DS / "json" / f"{tree}.json").read_text(encoding="utf-8-sig"))
    out = []
    for im in d["images"].values():
        stem = im["filename"].rsplit(".", 1)[0]
        w, h = im["width"], im["height"]
        D = z[stem] if stem in z.files else np.zeros((0, 11))
        if len(D):
            _, u = np.unique(D[:, 10], return_index=True)
            D = D[np.sort(u)]
            D = D[D[:, 6:10].max(1) >= conf]
        for r in D:
            out.append({"class": KELAS[int(np.argmax(r[6:10]))],
                        "x_norm": float((r[0] + r[2]) / 2 / w),
                        "y_norm": float((r[1] + r[3]) / 2 / h),
                        "side_index": int(im["side_index"])})
    return out


def deteksi_y26mv2(tree: str) -> list[dict]:
    """Deteksi ter-cache dari detektor repo Baseline-SawitMVC (YOLO26m y26mv2).

    Masukan C: detektor MEREKA, algoritma MEREKA. Menutup kemungkinan bahwa
    penurunan di masukan B cuma efek detektor yang berbeda.
    """
    f = BASE / "predictions" / "y26mv2_per_tree" / f"{tree}.json"
    if not f.exists():
        return []
    d = json.loads(f.read_text(encoding="utf-8-sig"))
    out = []
    for im in d.get("images", {}).values():
        si = int(im.get("side_index", 0))
        for a in im.get("annotations", []):
            if a.get("class_name") not in KELAS:
                continue
            b = a.get("bbox_yolo", [0, 0, 0, 0])
            out.append({"class": a["class_name"], "x_norm": float(b[0]),
                        "y_norm": float(b[1]), "side_index": si})
    return out


def benar_gt(tree: str) -> np.ndarray:
    d = json.loads((DS / "json" / f"{tree}.json").read_text(encoding="utf-8-sig"))
    v = np.zeros(4)
    for b in d["bunches"]:
        v[KELAS.index(b["class"])] += 1
    return v


def metrik(pred: np.ndarray, benar: np.ndarray) -> dict:
    d = np.abs(pred - benar)
    return {"macro_mae": round(float(d.mean()), 4),
            "class_pm1_acc": round(float((d <= 1).mean()), 4),
            "tree_pm1_acc": round(float((d <= 1).all(1).mean()), 4),
            "total_mae": round(float(np.abs(pred.sum(1) - benar.sum(1)).mean()), 4),
            "bias_total": round(float((pred - benar).sum(1).mean()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--keluaran", default=str(SUB / "results" / "pt_e_006_baseline_counting.json"))
    args = ap.parse_args()

    algo = muat_algoritma()
    print("algoritma dimuat:", list(algo))
    man = PP.muat_manifest()
    ids = {s: [t for t, v in man.items() if v == s] for s in ["train", "val", "test"]}
    semua = sorted(man)

    z = {s: np.load(SUB / "results" / f"pred_skorpenuh_{s}.npz", allow_pickle=True)
         for s in ["train", "val", "test"]}
    milik = {t: s for s in ids for t in ids[s]}

    hasil = {"conf_deteksi": args.conf, "detektor": "YOLO26l @1280 (sel5)",
             "acuan_repo": {"M01": {"acc1": 0.8762, "macro_mae": 0.3746,
                                    "total_count_mae": 1.3305, "korpus": "953 pohon"}},
             "hasil": {}}

    for himpunan, daftar in [("953_semua", semua), ("141_test", ids["test"])]:
        B = np.stack([benar_gt(t) for t in daftar])
        for masukan in ["A_kotak_GT", "B_deteksi_yolo26l", "C_deteksi_y26mv2_repo"]:
            if masukan.startswith("A"):
                dets = {t: deteksi_gt(t) for t in daftar}
            elif masukan.startswith("B"):
                dets = {t: deteksi_detektor(t, z[milik[t]], args.conf) for t in daftar}
            else:
                dets = {t: deteksi_y26mv2(t) for t in daftar}
            for nama, fn in algo.items():
                P = np.stack([[fn(dets[t])[c] for c in KELAS] for t in daftar], dtype=float)
                hasil["hasil"].setdefault(himpunan, {}).setdefault(masukan, {})[nama] = \
                    metrik(P, B)
            # pembanding: naif tanpa dedup
            P = np.stack([[sum(1 for x in dets[t] if x["class"] == c) for c in KELAS]
                          for t in daftar], dtype=float)
            hasil["hasil"][himpunan][masukan]["naif_tanpa_dedup"] = metrik(P, B)

        print(f"\n===== {himpunan} ({len(daftar)} pohon) =====")
        for masukan in ["A_kotak_GT", "B_deteksi_yolo26l", "C_deteksi_y26mv2_repo"]:
            print(f"  --- masukan {masukan} ---")
            for nama, m in hasil["hasil"][himpunan][masukan].items():
                print(f"    {nama:22s} macroMAE {m['macro_mae']:.4f}  "
                      f"class±1 {m['class_pm1_acc']:.4f}  tree±1 {m['tree_pm1_acc']:.4f}  "
                      f"totalMAE {m['total_mae']:.4f}")

    m01_gt = hasil["hasil"]["953_semua"]["A_kotak_GT"]["M01_selector_b2b3"]
    m01_det = hasil["hasil"]["141_test"]["B_deteksi_yolo26l"]["M01_selector_b2b3"]
    hasil["kesimpulan"] = {
        "M01_di_kotak_GT_953": m01_gt,
        "M01_di_deteksi_test141": m01_det,
        "reproduksi_acuan": abs(m01_gt["macro_mae"] - 0.3746) <= 0.02,
        "arti": ("kalau reproduksi_acuan benar dan angka di deteksi jauh lebih buruk, "
                 "maka angka 0,375 memang diukur di kotak GT — bukan cacat algoritma, "
                 "melainkan pertanyaan yang berbeda"),
    }
    Path(args.keluaran).write_text(json.dumps(hasil, indent=1, ensure_ascii=False))
    print("\n" + json.dumps(hasil["kesimpulan"], indent=1, ensure_ascii=False))
    print(f"-> {args.keluaran}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
