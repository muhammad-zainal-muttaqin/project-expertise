"""Latih RT-DETR-L pada split kanonik DAMIMAS-only.

Arsitektur ini merupakan anggota kedua bank kandidat deteksi. Keluaran akhirnya
akan digabung dengan YOLO melalui pemilihan/weighted-box-fusion di validation,
bukan dipaksa menggantikan YOLO pada semua kelas.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import RTDETR


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/workspace/SawitMVC-YOLO-Damimas/data.yaml")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="rtdetr-l.pt")
    ap.add_argument("--name", default="rtdetr_l_damimas_s42")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=18)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = RTDETR(args.weights)
    model.train(
        data=str(DATA), project=str(ROOT / "runs"), name=args.name,
        exist_ok=False, epochs=args.epochs, patience=args.patience,
        imgsz=args.imgsz, batch=args.batch, workers=8, seed=args.seed,
        deterministic=True, cos_lr=True,
        hsv_h=.005, hsv_s=.15, hsv_v=.25,
        translate=.10, scale=.40, fliplr=.5,
        val=True, plots=False, save=True, save_period=5,
    )


if __name__ == "__main__":
    main()
