"""Latih YOLO26l 4-kanal (RGB+D) dengan modality dropout — Fase 5 lever representasi.

Adaptasi dari `Research-Pipeline/pipeline/fourch.py::patch_loader` (copy+adapt,
BUKAN cross-import — Volume 1 read-only). Beda dengan fourch.py: dataset kita
sudah berupa TIFF 4-kanal yang di-bake di disk (bukan compose dari JPG+PNG
depth terpisah saat load), jadi patch di sini hanya perlu MENOL-kan kanal
depth (indeks 3) secara acak saat TRAIN — bukan menyusun ulang dari sumber
lain.

Kenapa: cakupan depth valid cuma 71% (lihat depth_meta.json), dan tepat di
area paling butuh depth (B4, tertutup pelepah) itulah yang paling sering
invalid (line-of-sight sensor depth ikut terhalang, sama seperti RGB).
Modality dropout memaksa network tidak "bergantung mati" pada kanal depth —
selaras scope Fase 5 lever representasi ("augmentasi khusus kanal depth" di
docs/RENCANA.md), BUKAN eksperimen baru di luar dua lever yang diizinkan.

Mekanisme monkeypatch: `ultralytics.utils.patches.imread` diberi perlakuan
khusus untuk file .tiff (baca cv2.IMREAD_UNCHANGED, mengabaikan parameter
`flags`) — diverifikasi langsung dari source terpasang (ultralytics==8.4.103).
Supaya imread yang stateless ini tetap tahu train vs eval, dipakai trik sentinel
`flags` yang sama seperti fourch.py: BaseDataset.__init__ di-patch untuk
menyetel `self.cv2_flag` ke sentinel tergantung `self.augment`, lalu imread
yang dipatch membaca sentinel itu untuk memutuskan apakah dropout berlaku.

Usage:
    .venv/bin/python train_yolo_4ch_dropout.py \
        --data /workspace/SawitMVC-Depth-4ch/data_rgbd_352.yaml \
        --dropout 0.25 --epochs 15 --patience 3 \
        --name yolo26l_screening_dropout352
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

_TRAIN_FLAG = -997
_EVAL_FLAG = -999


def patch_modality_dropout(dropout: float) -> None:
    import cv2
    import ultralytics.data.base as base
    from ultralytics.data.base import BaseDataset

    if getattr(base, "_dropout_patched", False):
        return

    orig_init = BaseDataset.__init__

    def patched_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        if getattr(self, "channels", 3) == 4:
            self.cv2_flag = _TRAIN_FLAG if getattr(self, "augment", False) else _EVAL_FLAG

    BaseDataset.__init__ = patched_init

    orig_imread = base.imread

    def imread_dropout(filename, flags=cv2.IMREAD_COLOR):
        if flags not in (_TRAIN_FLAG, _EVAL_FLAG):
            return orig_imread(filename, flags)
        im = orig_imread(filename, cv2.IMREAD_UNCHANGED)  # flags asli diabaikan untuk .tiff
        if im is not None and flags == _TRAIN_FLAG and im.ndim == 3 and im.shape[2] == 4:
            if random.random() < dropout:
                im = im.copy()
                im[:, :, 3] = 0
        return im

    base.imread = imread_dropout
    base._dropout_patched = True
    print(f"patch: modality dropout aktif, p={dropout} (kanal depth di-nol-kan acak, TRAIN saja)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/SawitMVC-Depth-4ch/data_rgbd_352.yaml")
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="yolo26l_dropout352")
    ap.add_argument("--project", default="/workspace/project-expertise/runs")
    ap.add_argument("--weights", default="yolo26l.pt")
    args = ap.parse_args()

    patch_modality_dropout(args.dropout)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    mulai = time.time()
    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        cos_lr=True,
        project=args.project,
        name=args.name,
    )
    durasi = time.time() - mulai

    meta = {
        "modal": "rgbd_dropout",
        "dropout": args.dropout,
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "durasi_detik": round(durasi, 1),
    }
    out_dir = Path(args.project) / args.name
    (out_dir / "hasil.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
