"""Retrain RT-DETR-L V2-E-001 (953 pohon SawitMVC-YOLO), 2026-09-07.

Bobot v2repro asli hilang (tidak ada di repo maupun bucket cadangan HF).
Konfigurasi mereplikasi persis entri V2-E-001 di experiments/EKSPERIMEN.md:
epochs=60, imgsz=1280, batch=4, seed=42, cos_lr=True, patience=60.
Tujuan: mengisi kolom 'detection' (AP50 class-agnostic) yang masih '·'
pada metrics/recap.md, tabel 953, baris RT-DETR-L.
"""
from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")
model.train(
    data="configs/sawitmvc_yolo_953_retrain.yaml",
    epochs=60,
    imgsz=1280,
    batch=4,
    seed=42,
    cos_lr=True,
    patience=60,
    workers=20,
    project="runs",
    name="rtdetr_l_e60_i1280_v2repro_retrain",
)
