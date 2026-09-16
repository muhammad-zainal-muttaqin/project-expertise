"""Metrik deteksi lengkap per korpus latih (V2-E-050c).

Menghitung Presisi, Recall, F1, AP50, dan AP50-95 per kelas dan makro untuk
sembilan kombinasi korpus latih dan detektor, langsung dari *dump* prediksi
`.npz` serta anotasi acuan. Presisi, Recall, dan F1 dilaporkan pada ambang
skor keyakinan yang memaksimalkan F1 makro, mengikuti konvensi keluaran
validasi Ultralytics.

Pemakaian:
    python scripts/metrik_deteksi_perkorpus.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CLASSES = ["B1", "B2", "B3", "B4"]
IOU_GRID = np.round(np.arange(0.50, 0.951, 0.05), 2)
CONF_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
CONF_TETAP = 0.25
DETEKTOR = [("yolo26l", "YOLO26l"), ("rtdetr_l", "RT-DETR-L"), ("rfdetr_l", "RF-DETR-L")]


# --------------------------------------------------------------------------
# Anotasi acuan
# --------------------------------------------------------------------------
def gt_953(baseline_root: Path, pohon_uji: set[str], awalan: str = "") -> dict[str, np.ndarray]:
    kotak: dict[str, np.ndarray] = {}
    for berkas in sorted((baseline_root / "ground_truth" / "annotations").glob("*.json")):
        pohon = berkas.stem
        if pohon not in pohon_uji:
            continue
        data = json.loads(berkas.read_text(encoding="utf-8-sig"))
        for nama_sisi, sisi in data.get("images", {}).items():
            idx = nama_sisi.split("_")[-1]
            kunci = f"{awalan}{pohon}_{idx}"
            baris = []
            for ann in sisi.get("annotations", []):
                if ann.get("class_name") not in CLASSES:
                    continue
                x1, y1, x2, y2 = ann["bbox_pixel"]
                baris.append([x1, y1, x2, y2, CLASSES.index(ann["class_name"])])
            kotak[kunci] = np.array(baris, dtype=float).reshape(-1, 5)
    return kotak


def gt_763(depth_root: Path, split: tuple[str, ...], awalan: str = "") -> dict[str, np.ndarray]:
    kotak: dict[str, np.ndarray] = {}
    for folder in split:
        for berkas in sorted((depth_root / folder / "linked").glob("*.json")):
            data = json.loads(berkas.read_text(encoding="utf-8-sig"))
            pohon = data["tree_id"]
            for nama_sisi, sisi in data.get("images", {}).items():
                idx = nama_sisi.split("_")[-1]
                kunci = f"{awalan}{pohon}_{idx}"
                baris = []
                for ann in sisi.get("annotations", []):
                    if ann.get("class_name") not in CLASSES:
                        continue
                    x1, y1, x2, y2 = ann["bbox_pixel"]
                    baris.append([x1, y1, x2, y2, CLASSES.index(ann["class_name"])])
                kotak[kunci] = np.array(baris, dtype=float).reshape(-1, 5)
    return kotak


def muat_deteksi(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    keluaran = {}
    for kunci in data.files:
        kotak = data[kunci]
        if kotak.size == 0:
            keluaran[kunci] = np.zeros((0, 6), dtype=float)
            continue
        kelas = kotak[:, 5].astype(int)
        keluaran[kunci] = kotak[(kelas >= 0) & (kelas <= 3)].astype(float)
    return keluaran


# --------------------------------------------------------------------------
# Pencocokan dan metrik
# --------------------------------------------------------------------------
def iou_matriks(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    irisan = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    luas_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    luas_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return irisan / np.maximum(luas_a[:, None] + luas_b[None, :] - irisan, 1e-9)


def ap_dari_kurva(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> float:
    """AP interpolasi 101 titik mengikuti konvensi COCO."""
    if n_gt == 0:
        return float("nan")
    urut = np.argsort(-conf)
    tp = tp[urut]
    tp_kum = np.cumsum(tp)
    fp_kum = np.cumsum(1 - tp)
    recall = tp_kum / n_gt
    presisi = tp_kum / np.maximum(tp_kum + fp_kum, 1e-9)
    presisi = np.maximum.accumulate(presisi[::-1])[::-1]
    titik = np.linspace(0, 1, 101)
    return float(np.interp(titik, recall, presisi, left=presisi[0] if len(presisi) else 0, right=0).mean())


def evaluasi(gt: dict[str, np.ndarray], det: dict[str, np.ndarray]) -> dict:
    citra = sorted(set(gt) & set(det))
    hasil_kelas = {}
    tp_conf_perkelas = {}

    for c in range(4):
        n_gt = int(sum((gt[i][:, 4] == c).sum() for i in citra))
        conf_all, tp_all = [], {float(t): [] for t in IOU_GRID}
        for i in citra:
            g = gt[i][gt[i][:, 4] == c]
            d = det[i][det[i][:, 5] == c]
            if len(d) == 0:
                continue
            d = d[np.argsort(-d[:, 4])]
            conf_all.append(d[:, 4])
            iou = iou_matriks(d[:, :4], g[:, :4])
            for t in IOU_GRID:
                dipakai = np.zeros(len(g), dtype=bool)
                tp = np.zeros(len(d))
                for j in range(len(d)):
                    if len(g) == 0:
                        break
                    kandidat = iou[j].copy()
                    kandidat[dipakai] = 0
                    terbaik = int(np.argmax(kandidat)) if len(kandidat) else -1
                    if terbaik >= 0 and kandidat[terbaik] >= t:
                        dipakai[terbaik] = True
                        tp[j] = 1
                tp_all[float(t)].append(tp)
        conf_gab = np.concatenate(conf_all) if conf_all else np.zeros(0)
        ap = {}
        for t in IOU_GRID:
            tp_gab = np.concatenate(tp_all[float(t)]) if tp_all[float(t)] else np.zeros(0)
            ap[float(t)] = ap_dari_kurva(tp_gab, conf_gab, n_gt) if len(tp_gab) else 0.0
        tp50 = np.concatenate(tp_all[0.5]) if tp_all[0.5] else np.zeros(0)
        tp_conf_perkelas[c] = (tp50, conf_gab, n_gt)
        hasil_kelas[CLASSES[c]] = {
            "n_gt": n_gt,
            "ap50": ap[0.5],
            "ap50_95": float(np.nanmean([ap[float(t)] for t in IOU_GRID])),
        }

    # Presisi, Recall, F1 pada ambang yang memaksimalkan F1 makro.
    terbaik = (-1.0, 0.25, None)
    for conf_t in CONF_GRID:
        f1s, prs = [], []
        for c in range(4):
            tp50, conf, n_gt = tp_conf_perkelas[c]
            pilih = conf >= conf_t
            tp = float(tp50[pilih].sum())
            fp = float(pilih.sum() - tp)
            p = tp / max(tp + fp, 1e-9)
            r = tp / max(n_gt, 1e-9)
            f1 = 2 * p * r / max(p + r, 1e-9)
            f1s.append(f1)
            prs.append((p, r, f1))
        f1_makro = float(np.mean(f1s))
        if f1_makro > terbaik[0]:
            terbaik = (f1_makro, float(conf_t), prs)

    _, conf_opt, prs = terbaik
    for c in range(4):
        p, r, f1 = prs[c]
        hasil_kelas[CLASSES[c]].update({"presisi": p, "recall": r, "f1": f1})

    # Titik kerja pembanding pada ambang seragam 0,25.
    tetap = {}
    for c in range(4):
        tp50, conf, n_gt = tp_conf_perkelas[c]
        pilih = conf >= CONF_TETAP
        tp = float(tp50[pilih].sum())
        fp = float(pilih.sum() - tp)
        p = tp / max(tp + fp, 1e-9)
        r = tp / max(n_gt, 1e-9)
        tetap[CLASSES[c]] = {
            "presisi": p,
            "recall": r,
            "f1": 2 * p * r / max(p + r, 1e-9),
        }
    tetap["makro"] = {
        kunci: float(np.mean([tetap[c][kunci] for c in CLASSES]))
        for kunci in ("presisi", "recall", "f1")
    }

    return {
        "n_citra": len(citra),
        "conf_optimal": conf_opt,
        "conf_tetap": CONF_TETAP,
        "pada_conf_tetap": tetap,
        "per_kelas": hasil_kelas,
        "makro": {
            "presisi": float(np.mean([hasil_kelas[c]["presisi"] for c in CLASSES])),
            "recall": float(np.mean([hasil_kelas[c]["recall"] for c in CLASSES])),
            "f1": float(np.mean([hasil_kelas[c]["f1"] for c in CLASSES])),
            "map50": float(np.mean([hasil_kelas[c]["ap50"] for c in CLASSES])),
            "map50_95": float(np.mean([hasil_kelas[c]["ap50_95"] for c in CLASSES])),
            "n_gt": int(sum(hasil_kelas[c]["n_gt"] for c in CLASSES)),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-root", default="D:/Work/Assisten-Dosen/Baseline-SawitMVC")
    ap.add_argument("--depth-root", default="D:/Work/Assisten-Dosen/SawitMVC-Depth/SawitMVC-Depth-YOLO")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--out", default="results/counting_koefisien_2026-09-16/metrik_deteksi_perkorpus.json")
    arg = ap.parse_args()

    akar = Path(arg.project_root).resolve()
    baseline = Path(arg.baseline_root)
    depth = Path(arg.depth_root)

    import csv

    pohon_uji_953 = set()
    with (baseline / "ground_truth" / "split_manifest.csv").open(encoding="utf-8-sig", newline="") as fh:
        for baris in csv.DictReader(fh):
            if baris["new_split"].strip() == "test":
                pohon_uji_953.add(baris["tree_id"].strip())

    gt_per_korpus = {
        "953": gt_953(baseline, pohon_uji_953),
        "763": gt_763(depth, ("test",)),
        "1716": {
            **gt_953(baseline, pohon_uji_953, awalan="SAWIT_"),
            **gt_763(depth, ("train", "valid", "test"), awalan="DEPTH_"),
        },
    }
    dump_per_korpus = {
        "953": "results/pred_{slug}_v2repro_953_test.npz",
        "763": "results/new763/predictions/{slug}_rgb_s42_i1280__test.npz",
        "1716": "results/combined1716/predictions/combined1716_{slug}_rgb_s42_i1280__test.npz",
    }

    # Kombinasi silang: detektor satu korpus dievaluasi pada partisi uji korpus lain.
    silang = {
        ("763", "953"): ("results/cross_eval/predictions/new763_{slug}__on_953__test.npz", "953"),
        ("1716", "953"): ("results/cross_eval/predictions/combined1716_{slug}__on_953__test.npz", "953"),
        ("1716", "763"): ("results/combined1716/predictions/combined1716_{slug}_rgb_s42_i1280__test.npz", "763dep"),
        ("953", "763"): ("results/cross_eval/predictions/v2repro953_{slug}__on_763__test.npz", "763"),
    }
    gt_per_korpus["763dep"] = gt_763(depth, ("test",), awalan="DEPTH_")

    baris_keluaran = []
    for korpus, pola in dump_per_korpus.items():
        for slug, label in DETEKTOR:
            path = akar / pola.format(slug=slug)
            if not path.exists():
                print(f"[lewat] {path}")
                continue
            hasil = evaluasi(gt_per_korpus[korpus], muat_deteksi(path))
            hasil.update({"korpus_latih": korpus, "korpus_uji": korpus, "detektor": label,
                          "sumber_dump": str(path.relative_to(akar)).replace("\\", "/")})
            baris_keluaran.append(hasil)
            m = hasil["makro"]
            print(
                f"{korpus:>4} {label:>10} n={hasil['n_citra']:>4} conf*={hasil['conf_optimal']:.2f} "
                f"P={m['presisi']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
                f"mAP50={m['map50']:.4f} mAP50-95={m['map50_95']:.4f} "
                f"| conf 0,25: P={hasil['pada_conf_tetap']['makro']['presisi']:.4f} "
                f"R={hasil['pada_conf_tetap']['makro']['recall']:.4f} "
                f"F1={hasil['pada_conf_tetap']['makro']['f1']:.4f}"
            )

    for (latih, uji), (pola, kunci_gt) in silang.items():
        for slug, label in DETEKTOR:
            path = akar / pola.format(slug=slug)
            if not path.exists():
                print(f"[lewat silang] {path.name}")
                continue
            hasil = evaluasi(gt_per_korpus[kunci_gt], muat_deteksi(path))
            hasil.update({"korpus_latih": latih, "korpus_uji": uji, "detektor": label,
                          "sumber_dump": str(path.relative_to(akar)).replace("\\", "/")})
            baris_keluaran.append(hasil)
            m = hasil["makro"]
            print(
                f"{latih:>4}->{uji:<5} {label:>10} n={hasil['n_citra']:>4} "
                f"P={m['presisi']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
                f"mAP50={m['map50']:.4f} mAP50-95={m['map50_95']:.4f}"
            )

    keluaran = Path(arg.out)
    keluaran.parent.mkdir(parents=True, exist_ok=True)
    keluaran.write_text(
        json.dumps(
            {
                "_meta": {
                    "eksperimen": "V2-E-050c",
                    "tanggal": "2026-09-16",
                    "evaluator": "implementasi lokal, pencocokan serakah IoU, AP interpolasi 101 titik",
                    "catatan": "Presisi, Recall, dan F1 dilaporkan pada ambang skor keyakinan yang memaksimalkan F1 makro.",
                },
                "baris": baris_keluaran,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nDitulis: {keluaran} ({len(baris_keluaran)} baris)")


if __name__ == "__main__":
    main()
