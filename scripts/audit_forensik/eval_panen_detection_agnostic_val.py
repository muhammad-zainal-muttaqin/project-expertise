import json
from pathlib import Path
import sys
import numpy as np
from ultralytics import YOLO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, "/workspace/project-expertise/scripts")
from eval_new763_pycoco import build_gt

DATASET = Path("/workspace/SawitMVC-YOLO")
WEIGHTS = "/workspace/runs_panen/agnostik_m1280/weights/best.pt"

model = YOLO(WEIGHTS)
gt, paths = build_gt(DATASET, "val")
for ann in gt.dataset["annotations"]:
    ann["category_id"] = 1
gt.dataset["categories"] = [{"id": 1, "name": "tandan"}]
gt.createIndex()
stem_to_id = {p.stem: i for i, p in enumerate(paths, 1)}

dt_list = []
dump = {}
for i in range(0, len(paths), 16):
    chunk = paths[i:i+16]
    results = model.predict([str(p) for p in chunk], imgsz=1280, conf=0.001, iou=0.7,
                             max_det=300, device=0, verbose=False)
    for p, r in zip(chunk, results):
        image_id = stem_to_id[p.stem]
        boxes = r.boxes
        rows = []
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), score in zip(xyxy, conf):
                dt_list.append({"image_id": image_id, "category_id": 1,
                                "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                                "score": float(score)})
                rows.append([x1, y1, x2, y2, score, 0.0])
        dump[p.stem] = np.asarray(rows, dtype=np.float32).reshape(-1, 6)

np.savez("/workspace/project-expertise/results/pred_panen_agnostik_953_val.npz", **dump)

dt = gt.loadRes(dt_list) if dt_list else gt.loadRes([])
ev = COCOeval(gt, dt, "bbox")
ev.evaluate(); ev.accumulate(); ev.summarize()
AP50 = float(ev.stats[1]); AP50_95 = float(ev.stats[0])
print(f"FINAL: AP50_agnostic={AP50:.4f} AP50_95_agnostic={AP50_95:.4f} n_images={len(paths)}")

out = {
    "node": "AF-E-012",
    "model": "agnostik_m1280 (YOLO26m class-agnostic, Pipeline Panen)",
    "dataset": "953", "split": "val",
    "weights": "audit_forensik_2026-09-06/runs_panen/agnostik_m1280/weights/best.pt (bucket)",
    "protocol": "imgsz=1280, conf=0.001, iou=0.7, max_det=300 -- sebanding dengan standar conf=0.001 di seluruh kolom detection lain (beda dari dets.pkl yang terpotong conf>=0.10)",
    "AP50_agnostic": AP50, "AP50_95_agnostic": AP50_95, "n_images": len(paths),
    "predictions": "results/pred_panen_agnostik_953_val.npz",
}
Path("/workspace/project-expertise/results/agnostic_ap50_panen_val_2026-09-07.json").write_text(json.dumps(out, indent=2))
print("saved")
