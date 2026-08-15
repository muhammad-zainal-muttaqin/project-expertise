"""Screening generik YOLO26l 4-kanal (early fusion arsitektur V2-E-005, HANYA
encoding kanal depth yang beda) -- Fase 5 lever representasi.

Dipakai untuk kandidat encoding (edge/clipped/valid_mask): arsitektur early
fusion TIDAK diubah (itu tetap arsitektur yang sama dengan V2-E-005), yang
diuji cuma apakah representasi kanal depth yang berbeda mengubah hasil.
Baseline 'inverse' TIDAK di-rerun di sini -- angkanya sudah ada di V2-E-005.

Usage:
    .venv/bin/python train_yolo_4ch_screening.py \
        --data /workspace/SawitMVC-Depth-4ch-edge/data_rgbd_352_edge.yaml \
        --epochs 15 --patience 3 --name yolo26l_screening_edge352
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--resume", metavar="LAST_PT",
                    help="lanjutkan run yang terputus dari weights/last.pt. "
                         "Resep (epochs/batch/imgsz/seed) dibaca ultralytics dari "
                         "args.yaml run itu, jadi TIDAK bisa berubah diam-diam.")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default="/workspace/project-expertise/runs")
    ap.add_argument("--weights", default="yolo26l.pt")
    args = ap.parse_args()

    from ultralytics import YOLO

    mulai = time.time()
    if args.resume:
        # Sengaja TIDAK mengoper ulang epochs/batch/imgsz/seed: ultralytics
        # membacanya dari args.yaml run yang bersangkutan. Mengopernya lagi
        # justru membuka celah resep berubah di tengah jalan tanpa jejak.
        if not Path(args.resume).is_file():
            raise SystemExit(f"FATAL: {args.resume} tidak ada — tidak bisa resume")
        model = YOLO(args.resume)
        model.train(resume=True)
        out_dir = Path(args.resume).resolve().parent.parent
        args.name = out_dir.name
    else:
        if not args.data or not args.name:
            raise SystemExit("FATAL: --data dan --name wajib kalau bukan --resume")
        model = YOLO(args.weights)
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
        out_dir = Path(args.project) / args.name
    durasi = time.time() - mulai

    meta = {
        "modal": "rgbd_encoding_screening",
        "data": args.data,
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "durasi_detik": round(durasi, 1),
        "resume_dari": args.resume or None,
    }
    (out_dir / "hasil.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
