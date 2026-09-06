"""Detektor tandan class-agnostic berkapasitas lebih tinggi untuk pipeline panen."""
import os
os.environ["YOLO_VERBOSE"] = "false"
from ultralytics import YOLO

RUNS = "/workspace/runs_panen"
m = YOLO("yolo26m.pt")
m.train(data="/workspace/ds/may1/data.yaml", imgsz=1280, epochs=40, batch=12,
        workers=32, cache="ram", device=0, seed=42, deterministic=True,
        project=RUNS, name="agnostik_m1280", exist_ok=True,
        patience=10, plots=False, verbose=False)

r = YOLO(f"{RUNS}/agnostik_m1280/weights/best.pt").val(
    data="/workspace/ds/may1/data.yaml", split="test", imgsz=1280, batch=12,
    workers=32, device=0, verbose=False, plots=False,
    project=RUNS, name="val_test", exist_ok=True)
print(f"AGNOSTIK test AP50={r.box.map50:.4f} P={r.box.mp:.4f} R={r.box.mr:.4f}", flush=True)
print("DET DONE")
