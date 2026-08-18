"""Evaluasi dump deteksi pada subset DAMIMAS tanpa menjalankan inferensi ulang.

Metrik ranking dihitung dengan pycocotools. Ambang operasi per kelas dipilih
sekali di val dengan F1 maksimum, lalu diterapkan apa adanya ke test untuk
melaporkan precision/recall/F1 pada titik operasi yang sah.

Format setiap array prediksi: ``x1,y1,x2,y2,score,class,...``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


NAMA = ("B1", "B2", "B3", "B4")


def bangun_gt(root: Path, split: str):
    images, anns, ann_id, per_stem = [], [], 1, {}
    paths = sorted((root / "images" / split).iterdir())
    for image_id, path in enumerate(paths, 1):
        w, h = Image.open(path).size
        images.append({"id": image_id, "file_name": path.name,
                       "width": w, "height": h})
        baris = []
        label = root / "labels" / split / f"{path.stem}.txt"
        for line in label.read_text().splitlines():
            if not line.strip():
                continue
            c, cx, cy, bw, bh = map(float, line.split())
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
            baris.append([int(c), x1, y1, x2, y2])
            anns.append({"id": ann_id, "image_id": image_id,
                         "category_id": int(c) + 1,
                         "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
            ann_id += 1
        per_stem[path.stem] = np.asarray(baris, float) if baris else np.zeros((0, 5))
    coco = COCO()
    coco.dataset = {"images": images, "annotations": anns,
                    "categories": [{"id": i + 1, "name": n}
                                   for i, n in enumerate(NAMA)]}
    coco.createIndex()
    return coco, paths, per_stem


def muat_prediksi(path: Path, stems: set[str]):
    z = np.load(path, allow_pickle=True)
    return {s: np.asarray(z[s], float)[:, :6] if s in z.files else np.zeros((0, 6))
            for s in stems}


def iou_satu_ke_banyak(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.zeros(0)
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a = np.maximum(0, box[2] - box[0]) * np.maximum(0, box[3] - box[1])
    b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-12)


def kurva_kelas(gt: dict[str, np.ndarray], pred: dict[str, np.ndarray], kelas: int):
    kandidat = []
    jumlah_gt = 0
    gt_kelas = {}
    for stem, g in gt.items():
        gg = g[g[:, 0] == kelas, 1:5]
        gt_kelas[stem] = gg
        jumlah_gt += len(gg)
        p = pred[stem]
        for row in p[p[:, 5].astype(int) == kelas]:
            kandidat.append((float(row[4]), stem, row[:4]))
    kandidat.sort(key=lambda x: x[0], reverse=True)
    dipakai = {s: np.zeros(len(v), bool) for s, v in gt_kelas.items()}
    tp, fp, skor = [], [], []
    for conf, stem, box in kandidat:
        ious = iou_satu_ke_banyak(box, gt_kelas[stem])
        tersedia = np.where(~dipakai[stem])[0]
        if len(tersedia):
            j = tersedia[int(np.argmax(ious[tersedia]))]
            cocok = ious[j] >= 0.5
        else:
            cocok = False
        if cocok:
            dipakai[stem][j] = True
        tp.append(int(cocok)); fp.append(int(not cocok)); skor.append(conf)
    tp = np.cumsum(tp); fp = np.cumsum(fp)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(jumlah_gt, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return np.asarray(skor), precision, recall, f1, jumlah_gt


def pilih_ambang(gt, pred):
    hasil = {}
    for k, nama in enumerate(NAMA):
        skor, p, r, f1, n = kurva_kelas(gt, pred, k)
        if not len(skor):
            hasil[nama] = {"ambang": 1.0, "precision": 0.0,
                           "recall": 0.0, "f1": 0.0, "n_gt": n}
            continue
        i = int(np.argmax(f1))
        hasil[nama] = {"ambang": float(skor[i]), "precision": float(p[i]),
                       "recall": float(r[i]), "f1": float(f1[i]), "n_gt": n}
    return hasil


def nilai_ambang(gt, pred, ambang):
    hasil = {}
    for k, nama in enumerate(NAMA):
        skor, p, r, f1, n = kurva_kelas(gt, pred, k)
        keep = np.where(skor >= ambang[nama])[0]
        if len(keep):
            i = int(keep[-1]); pp, rr, ff = p[i], r[i], f1[i]
        else:
            pp = rr = ff = 0.0
        hasil[nama] = {"ambang": float(ambang[nama]), "precision": float(pp),
                       "recall": float(rr), "f1": float(ff), "n_gt": n}
    hasil["macro"] = {m: float(np.mean([hasil[n][m] for n in NAMA]))
                      for m in ("precision", "recall", "f1")}
    return hasil


def nilai_coco(coco: COCO, paths: list[Path], pred: dict[str, np.ndarray]):
    det = []
    for image_id, path in enumerate(paths, 1):
        for x1, y1, x2, y2, conf, kelas in pred[path.stem]:
            det.append({"image_id": image_id, "category_id": int(kelas) + 1,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(conf)})
    ev = COCOeval(coco, coco.loadRes(det), "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    precision = ev.eval["precision"]
    ap50 = {}
    for k, nama in enumerate(NAMA):
        x = precision[0, :, k, 0, 2]
        ap50[nama] = float(x[x > -1].mean()) if (x > -1).any() else 0.0
    return {"mAP50_95": float(ev.stats[0]), "mAP50": float(ev.stats[1]),
            "AP50_per_kelas": ap50}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path,
                    default=Path("/workspace/SawitMVC-YOLO-Damimas"))
    ap.add_argument("--pred-val", type=Path, required=True)
    ap.add_argument("--pred-test", type=Path, required=True)
    ap.add_argument("--keluaran", type=Path, required=True)
    args = ap.parse_args()

    data = {}
    muatan = {}
    for split, sumber in (("val", args.pred_val), ("test", args.pred_test)):
        coco, paths, gt = bangun_gt(args.dataset, split)
        pred = muat_prediksi(sumber, set(gt))
        muatan[split] = (gt, pred)
        data[split] = {"n_citra": len(paths), "n_box_gt": sum(map(len, gt.values())),
                       **nilai_coco(coco, paths, pred)}
    dipilih = pilih_ambang(*muatan["val"])
    ambang = {n: dipilih[n]["ambang"] for n in NAMA}
    data["ambang_dipilih_di_val"] = dipilih
    data["titik_operasi"] = {
        split: nilai_ambang(*muatan[split], ambang) for split in ("val", "test")}
    data["titik_operasi_per_sisi"] = {}
    for split in ("val", "test"):
        gt, pred = muatan[split]
        sisi = sorted({s.rsplit("_", 1)[-1] for s in gt}, key=int)
        data["titik_operasi_per_sisi"][split] = {}
        for v in sisi:
            stems = [s for s in gt if s.rsplit("_", 1)[-1] == v]
            g = {s: gt[s] for s in stems}; p = {s: pred[s] for s in stems}
            data["titik_operasi_per_sisi"][split][f"sisi_{v}"] = {
                "n_citra": len(stems), **nilai_ambang(g, p, ambang)}
    data["sumber"] = {"val": str(args.pred_val), "test": str(args.pred_test)}

    args.keluaran.parent.mkdir(parents=True, exist_ok=True)
    args.keluaran.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"-> {args.keluaran}")


if __name__ == "__main__":
    main()
