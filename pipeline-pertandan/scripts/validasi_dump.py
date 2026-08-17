"""Validasi dump skor-penuh: apakah ia mereproduksi mAP50 yang sudah tercatat?

Gerbang kewarasan sebelum dump ini dipakai untuk apa pun. Kalau kotaknya
salah tempat atau memuat duplikat, seluruh angka per-tandan di hilirnya ikut
salah — dan itu tidak akan terlihat dari metrik per-tandan sendiri.

Sudah terbukti berguna: versi pertama skrip inferensi memakai tensor mentah
`(1,300,6)` apa adanya dan menghasilkan mAP50 test 0,1342, bukan 0,5436. Yang
salah bukan kotaknya (koordinatnya cocok sampai ~0,4 px dengan dump lama)
melainkan adanya baris duplikat yang jadi positif palsu.

Angka acuan (dari `../results/eval_sel5_953_rgb_test.json`):
    test mAP50 = 0,5436   mAP50-95 = 0,2565

Pemakaian:
    .venv/bin/python pipeline-pertandan/scripts/validasi_dump.py --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DS = Path("/workspace/SawitMVC-YOLO")
SUB = Path(__file__).resolve().parents[1]
KELAS = ["B1", "B2", "B3", "B4"]
ACUAN = {"test": {"mAP50": 0.5436, "mAP50_95": 0.2565}}


def bangun_gt(split: str):
    idir, ldir = DS / "images" / split, DS / "labels" / split
    paths = sorted(p for p in idir.iterdir() if p.suffix.lower() == ".jpg")
    images, anns, aid = [], [], 1
    for i, p in enumerate(paths, 1):
        w, h = Image.open(p).size
        images.append({"id": i, "file_name": p.name, "width": w, "height": h})
        lf = ldir / f"{p.stem}.txt"
        if lf.is_file():
            for ln in lf.read_text().splitlines():
                if not ln.strip():
                    continue
                c, cx, cy, bw, bh = map(float, ln.split())
                anns.append({"id": aid, "image_id": i, "category_id": int(c) + 1,
                             "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
                             "area": bw * w * bh * h, "iscrowd": 0})
                aid += 1
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns,
                  "categories": [{"id": i + 1, "name": n} for i, n in enumerate(KELAS)]}
    gt.createIndex()
    return gt, paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf-min", type=float, default=0.001)
    args = ap.parse_args()

    gt, paths = bangun_gt(args.split)
    z = np.load(SUB / "results" / f"pred_skorpenuh_{args.split}.npz", allow_pickle=True)
    dt, n_det = [], 0
    for i, p in enumerate(paths, 1):
        D = z[p.stem] if p.stem in z.files else np.zeros((0, 8))
        for r in D:
            s = float(r[4])                      # conf resmi, BUKAN max vektor
            if s < args.conf_min:
                continue
            n_det += 1
            dt.append({"image_id": i, "category_id": int(r[5]) + 1,   # cls resmi
                       "bbox": [float(r[0]), float(r[1]),
                                float(r[2] - r[0]), float(r[3] - r[1])], "score": s})
    E = COCOeval(gt, gt.loadRes(dt), "bbox")
    E.evaluate(); E.accumulate(); E.summarize()

    hasil = {"split": args.split, "n_citra": len(paths), "n_deteksi": n_det,
             "mAP50": round(float(E.stats[1]), 4), "mAP50_95": round(float(E.stats[0]), 4)}
    if args.split in ACUAN:
        a = ACUAN[args.split]
        hasil["acuan"] = a
        hasil["selisih_mAP50"] = round(hasil["mAP50"] - a["mAP50"], 4)
        hasil["lolos"] = abs(hasil["selisih_mAP50"]) <= 0.005
    print("\n" + json.dumps(hasil, indent=1))
    f = SUB / "results" / f"validasi_dump_{args.split}.json"
    f.write_text(json.dumps(hasil, indent=1))
    print(f"-> {f}")
    return 0 if hasil.get("lolos", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
