"""Retrain RF-DETR-L V2-E-001 (953 pohon SawitMVC-YOLO), 2026-09-07.

Bobot v2repro asli hilang (tidak ada di repo maupun bucket cadangan HF).
Konfigurasi mereplikasi persis entri V2-E-001 di experiments/EKSPERIMEN.md:
RFDETRLarge(resolution=1280, gradient_checkpointing=True).train(epochs=60,
batch_size=4, grad_accum_steps=4, seed=42). num_workers dimaksimalkan ke
jumlah core CPU mesin ini (20) untuk mempercepat dataloader.
"""
from rfdetr import RFDETRLarge

model = RFDETRLarge(resolution=1280, gradient_checkpointing=True)
model.train(
    dataset_dir="/workspace/SawitMVC-YOLO-coco",
    output_dir="runs/rfdetr_l_e60_i1280_v2repro_retrain",
    epochs=60,
    batch_size=4,
    grad_accum_steps=4,
    seed=42,
    num_workers=20,
    checkpoint_interval=1,
    run_test=False,
    tensorboard=False,
    device="cuda",
)
