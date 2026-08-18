"""Dump prediksi top-1 YOLO/RT-DETR pada dataset DAMIMAS.

Format NPZ kompatibel dengan ``eval_dump_damimas.py``: enam kolom pertama
memuat ``[x1,y1,x2,y2,score,class]``. Empat skor kelas top-1 dan ID anchor
ditambahkan pada kolom 6:11 agar dump juga sah sebagai bank counting. Untuk
YOLO26 yang membutuhkan distribusi empat kelas asli bagi R4, gunakan
``pipeline-pertandan/scripts/infer_skor_penuh.py``.
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
                if b is not None and len(b):
                    box = b.xyxy.cpu().numpy().astype(np.float32)
                    conf = b.conf.cpu().numpy().astype(np.float32)
                    kelas = b.cls.cpu().numpy().astype(np.int64)
                    if ((kelas < 0) | (kelas >= 4)).any():
                        raise RuntimeError(
                            f"class_id di luar 0..3 pada {p}: {np.unique(kelas)}")
                    pk = np.zeros((len(box), 4), np.float32)
                    pk[np.arange(len(box)), kelas] = conf
                    out[p.stem] = np.c_[
                        box, conf, kelas.astype(np.float32), pk,
                        np.arange(len(box), dtype=np.float32),
                    ].astype(np.float32)
                else:
                    out[p.stem] = np.zeros((0, 11), np.float32)
            n = min(awal + args.batch, len(paths))
            if n % 100 < args.batch or n == len(paths):
                print(f"{split}: {n}/{len(paths)} ({time.time()-t0:.0f}s)", flush=True)
        suffix = "_tta" if args.augment else ""
        tujuan = ROOT / "results" / f"pred_{args.tag}{suffix}_{split}.npz"
        np.savez_compressed(tujuan, **out)
        print(f"-> {tujuan}")


if __name__ == "__main__":
    main()
