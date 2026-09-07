import json
from pathlib import Path
import numpy as np
from ultralytics import RTDETR
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import sys
sys.path.insert(0, "/workspace/project-expertise/scripts")
from eval_new763_pycoco import build_gt

DATASET = Path("/workspace/SawitMVC-YOLO")
WEIGHTS = "/workspace/project-expertise/runs/detect/runs/rtdetr_l_e60_i1280_v2repro_retrain/weights/best.pt"
OUT_NPZ = Path("/workspace/project-expertise/results/pred_rtdetr_l_v2repro_953_test.npz")

model = RTDETR(WEIGHTS)
gt, paths = build_gt(DATASET, "test")

dump = {}
for i in range(0, len(paths), 8):
    chunk = paths[i:i+8]
    results = model.predict([str(p) for p in chunk], imgsz=1280, conf=0.001, iou=0.7,
                             max_det=300, device=0, verbose=False)
    for p, r in zip(chunk, results):
        boxes = r.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            rows = np.concatenate([xyxy, conf[:, None], cls[:, None]], axis=1)
        else:
            rows = np.zeros((0, 6), dtype=np.float32)
        dump[p.stem] = rows

np.savez(OUT_NPZ, **dump)
print(f"saved {OUT_NPZ}, {len(dump)} images")

# class-aware mAP50
stem_to_id = {p.stem: i for i, p in enumerate(paths, 1)}
dt_list = []
for stem, rows in dump.items():
    image_id = stem_to_id.get(stem)
    if image_id is None:
        continue
    for x1, y1, x2, y2, score, cls in rows:
        dt_list.append({"image_id": image_id, "category_id": int(cls) + 1,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score)})
dt = gt.loadRes(dt_list) if dt_list else gt.loadRes([])
ev = COCOeval(gt, dt, "bbox")
ev.evaluate(); ev.accumulate(); ev.summarize()
mAP50 = float(ev.stats[1])
mAP50_95 = float(ev.stats[0])
print(f"class-aware mAP50={mAP50:.4f} mAP50-95={mAP50_95:.4f}")

# class-agnostic AP50 (fold to 1 class)
gt2, _ = build_gt(DATASET, "test")
for ann in gt2.dataset["annotations"]:
    ann["category_id"] = 1
gt2.dataset["categories"] = [{"id": 1, "name": "tandan"}]
gt2.createIndex()
dt_list2 = []
for stem, rows in dump.items():
    image_id = stem_to_id.get(stem)
    if image_id is None:
        continue
    for x1, y1, x2, y2, score, cls in rows:
        dt_list2.append({"image_id": image_id, "category_id": 1,
                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                         "score": float(score)})
dt2 = gt2.loadRes(dt_list2) if dt_list2 else gt2.loadRes([])
ev2 = COCOeval(gt2, dt2, "bbox")
ev2.evaluate(); ev2.accumulate(); ev2.summarize()
AP50_agnostic = float(ev2.stats[1])
AP50_95_agnostic = float(ev2.stats[0])
print(f"AP50_agnostic={AP50_agnostic:.4f} AP50_95_agnostic={AP50_95_agnostic:.4f}")

out = {
    "node": "V2-E-001",
    "model": "RT-DETR-L",
    "dataset": "953",
    "split": "test",
    "n_images": len(paths),
    "weights": WEIGHTS,
    "retrain_reason": "bobot v2repro asli hilang (tidak ada di repo maupun bucket cadangan HF); retrain 60 epoch mengikuti protokol persis experiments/EKSPERIMEN.md V2-E-001 (imgsz=1280, batch=4, seed=42, cos_lr=True, patience=60)",
    "mAP50_classaware": mAP50,
    "mAP50_95_classaware": mAP50_95,
    "AP50_agnostic": AP50_agnostic,
    "AP50_95_agnostic": AP50_95_agnostic,
    "reference_class_aware_mAP50_original_2026-08-09": 0.5781,
    "predictions": str(OUT_NPZ),
}
Path("/workspace/project-expertise/results/rtdetr_l_v2repro_953_retrain_2026-09-07.json").write_text(json.dumps(out, indent=2))
print("done")
