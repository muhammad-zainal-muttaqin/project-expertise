"""Inferensi tiled/SAHI-style untuk objek tandan kecil di adegan padat.

Tile saling tumpang tindih, prediksi yang terpotong di batas internal dibuang,
lalu NMS global per kelas dilakukan dalam koordinat citra asli. Dump ini tetap
menjadi anggota terpisah; keputusan apakah tile membantu dibuat di VAL oleh
``fusi_detektor_damimas.py``.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import RTDETR, YOLO


ROOT = Path(__file__).resolve().parents[1]
DS = Path("/workspace/SawitMVC-YOLO-Damimas")


def posisi(panjang: int, tile: int, overlap: float) -> list[int]:
    if panjang <= tile:
        return [0]
    langkah = max(1, int(round(tile * (1 - overlap))))
    out = list(range(0, panjang - tile + 1, langkah))
    if out[-1] != panjang - tile:
        out.append(panjang - tile)
    return out


def iou(box, boxes):
    if not len(boxes):
        return np.zeros(0)
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


def nms(D, ambang):
    keep = []
    for k in range(4):
        idx = np.flatnonzero(D[:, 5].astype(int) == k)
        urut = idx[np.argsort(-D[idx, 4])]
        while len(urut):
            i = int(urut[0]); keep.append(i); urut = urut[1:]
            if len(urut):
                urut = urut[iou(D[i, :4], D[urut, :4]) < ambang]
    return D[np.asarray(keep, int)] if keep else np.zeros((0, 6), np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=("yolo", "rtdetr"), required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", nargs="+", default=["val", "test"])
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=.25)
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="resolusi masukan model untuk setiap tile")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--nms-iou", type=float, default=.65)
    ap.add_argument("--batas", type=int, default=3,
                    help="buang box yang menyentuh batas tile internal")
    args = ap.parse_args()
    model = (YOLO if args.arch == "yolo" else RTDETR)(args.weights)

    for split in args.split:
        paths = sorted((DS / "images" / split).glob("*.jpg"))
        out = {}; t0 = time.time()
        for ni, path in enumerate(paths, 1):
            im = cv2.imread(str(path))
            if im is None:
                raise RuntimeError(f"Citra gagal dibaca: {path}")
            H, W = im.shape[:2]
            tugas = []
            for y0 in posisi(H, args.tile, args.overlap):
                for x0 in posisi(W, args.tile, args.overlap):
                    y1, x1 = min(y0 + args.tile, H), min(x0 + args.tile, W)
                    tugas.append((im[y0:y1, x0:x1], x0, y0, x1, y1))
            rows = []
            for awal in range(0, len(tugas), args.batch):
                blok = tugas[awal:awal + args.batch]
                rr = model.predict([q[0] for q in blok], imgsz=args.imgsz,
                                   conf=.001, iou=.7, max_det=300, verbose=False)
                for r, (_crop, x0, y0, xakhir, yakhir) in zip(rr, blok):
                    b = r.boxes
                    if b is None or not len(b):
                        continue
                    D = np.c_[b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                              b.cls.cpu().numpy()].astype(np.float32)
                    tw, th = xakhir - x0, yakhir - y0
                    m = np.ones(len(D), bool)
                    if x0 > 0:
                        m &= D[:, 0] > args.batas
                    if xakhir < W:
                        m &= D[:, 2] < tw - args.batas
                    if y0 > 0:
                        m &= D[:, 1] > args.batas
                    if yakhir < H:
                        m &= D[:, 3] < th - args.batas
                    D = D[m]
                    D[:, [0, 2]] += x0; D[:, [1, 3]] += y0
                    rows.append(D)
            D = np.concatenate(rows) if rows else np.zeros((0, 6), np.float32)
            out[path.stem] = nms(D, args.nms_iou)
            if ni % 50 == 0 or ni == len(paths):
                print(f"{split}: {ni}/{len(paths)} ({time.time()-t0:.0f}s)", flush=True)
        tujuan = ROOT / "results" / f"pred_{args.tag}_{split}.npz"
        np.savez_compressed(tujuan, **out)
        print(f"-> {tujuan}")


if __name__ == "__main__":
    main()
