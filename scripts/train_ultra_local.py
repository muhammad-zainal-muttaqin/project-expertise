"""Replikasi persis train_yolo26l.py / train_rtdetr.py dari research-pipeline,
HANYA beda satu hal: --data menunjuk ke yaml lokal (disk overlay, bukan
network mount /workspace) untuk menghindari I/O stall dataloader.

Tidak mengedit file apa pun di research-pipeline -- skrip berdiri sendiri di
project-expertise. Hyperparameter disalin persis dari skrip asli supaya tetap
setara dengan protokol E-021.

Usage:
    .venv/bin/python train_ultra_local.py --arch rtdetr \
        --weights /workspace/research-pipeline/reproduce/experiments/rtdetr-l.pt \
        --data /home/claudeuser/data-cache/data_rgb_local.yaml \
        --imgsz 1280 --epochs 60 --batch 4 --name rtdetr_l_e60_i1280_v2repro
"""
from __future__ import annotations

import argparse
from pathlib import Path

EVIDENCE_ROOT = Path("/workspace/research-pipeline/evidence/experiments")

ap = argparse.ArgumentParser()
ap.add_argument("--arch", choices=["yolo", "rtdetr"], required=True)
ap.add_argument("--weights", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--name", required=True)
a = ap.parse_args()

if a.arch == "yolo":
    from ultralytics import YOLO as Model
else:
    from ultralytics import RTDETR as Model

m = Model(a.weights)
m.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        seed=42, name=a.name, project=str(EVIDENCE_ROOT / "runs"),
        exist_ok=True, cos_lr=True,
        hsv_h=0.005, hsv_s=0.15, hsv_v=0.25,
        plots=False, patience=60, val=True)
r = m.val(data=a.data, split="val", imgsz=a.imgsz, batch=2, plots=False, verbose=False)
print("val mAP50:", r.box.map50, "val mAP50-95:", r.box.map)
