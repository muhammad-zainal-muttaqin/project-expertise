"""Dump prediksi top-1 YOLO/RT-DETR pada dataset DAMIMAS.

Format NPZ kompatibel dengan ``eval_dump_damimas.py``: setiap stem memuat
``[x1,y1,x2,y2,score,class]``. Untuk YOLO26 yang membutuhkan empat skor kelas
per anchor bagi R4, gunakan ``pipeline-pertandan/scripts/infer_skor_penuh.py``.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from ultralytics import RTDETR, YOLO


ROOT = Path(__file__).resolve().parents[1]
DS = Path("/workspace/SawitMVC-YOLO-Damimas")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=("yolo", "rtdetr"), required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", nargs="+", default=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--augment", action="store_true")
    args = ap.parse_args()
    model = (YOLO if args.arch == "yolo" else RTDETR)(args.weights)

    for split in args.split:
        paths = sorted((DS / "images" / split).glob("*.jpg"))
        out = {}; t0 = time.time()
        for awal in range(0, len(paths), args.batch):
            blok = paths[awal:awal + args.batch]
            rr = model.predict([str(p) for p in blok], imgsz=args.imgsz,
                               conf=.001, iou=.7, max_det=300,
                               augment=args.augment, verbose=False)
            for p, r in zip(blok, rr):
                b = r.boxes
                out[p.stem] = (np.c_[b.xyxy.cpu().numpy(),
                                      b.conf.cpu().numpy(),
                                      b.cls.cpu().numpy()].astype(np.float32)
                               if b is not None and len(b)
                               else np.zeros((0, 6), np.float32))
            n = min(awal + args.batch, len(paths))
            if n % 100 < args.batch or n == len(paths):
                print(f"{split}: {n}/{len(paths)} ({time.time()-t0:.0f}s)", flush=True)
        suffix = "_tta" if args.augment else ""
        tujuan = ROOT / "results" / f"pred_{args.tag}{suffix}_{split}.npz"
        np.savez_compressed(tujuan, **out)
        print(f"-> {tujuan}")


if __name__ == "__main__":
    main()
