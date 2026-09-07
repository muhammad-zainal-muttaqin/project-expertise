import json
from pathlib import Path
import numpy as np
from rfdetr import RFDETRLarge
from pycocotools.cocoeval import COCOeval

import sys
sys.path.insert(0, "/workspace/project-expertise/scripts")
from eval_new763_pycoco import build_gt

DATASET = Path("/workspace/SawitMVC-YOLO")
WEIGHTS = "/workspace/project-expertise/runs/rfdetr_l_e60_i1280_v2repro_retrain/checkpoint_best_ema.pth"
OUT_NPZ = Path("/workspace/project-expertise/results/pred_rfdetr_l_v2repro_953_test.npz")

model = RFDETRLarge(pretrain_weights=WEIGHTS, resolution=1280)
gt, paths = build_gt(DATASET, "test")

dump = {}
dt_list = []
stem_to_id = {p.stem: i for i, p in enumerate(paths, 1)}
for start in range(0, len(paths), 8):
    chunk = paths[start:start + 8]
    results = model.predict([str(p) for p in chunk], threshold=0.001)
    if not isinstance(results, list):
        results = [results]
    for path, det in zip(chunk, results):
        image_id = stem_to_id[path.stem]
        rows = []
        for xyxy, score, klass in zip(det.xyxy, det.confidence, det.class_id):
            x1, y1, x2, y2 = map(float, xyxy)
            score, klass = float(score), int(klass)
            rows.append([x1, y1, x2, y2, score, float(klass)])
            dt_list.append({"image_id": image_id, "category_id": klass + 1,
                            "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
        dump[path.stem] = np.asarray(rows, dtype=np.float32).reshape(-1, 6)

np.savez(OUT_NPZ, **dump)
print(f"saved {OUT_NPZ}, {len(dump)} images")

dt = gt.loadRes(dt_list) if dt_list else gt.loadRes([])
ev = COCOeval(gt, dt, "bbox")
ev.evaluate(); ev.accumulate(); ev.summarize()
mAP50 = float(ev.stats[1]); mAP50_95 = float(ev.stats[0])
print(f"class-aware mAP50={mAP50:.4f} mAP50-95={mAP50_95:.4f}")

gt2, _ = build_gt(DATASET, "test")
for ann in gt2.dataset["annotations"]:
    ann["category_id"] = 1
gt2.dataset["categories"] = [{"id": 1, "name": "tandan"}]
gt2.createIndex()
dt_list2 = [{**d, "category_id": 1} for d in dt_list]
dt2 = gt2.loadRes(dt_list2) if dt_list2 else gt2.loadRes([])
ev2 = COCOeval(gt2, dt2, "bbox")
ev2.evaluate(); ev2.accumulate(); ev2.summarize()
AP50_agnostic = float(ev2.stats[1]); AP50_95_agnostic = float(ev2.stats[0])
print(f"AP50_agnostic={AP50_agnostic:.4f} AP50_95_agnostic={AP50_95_agnostic:.4f}")

out = {
    "node": "V2-E-001", "model": "RF-DETR-L", "dataset": "953", "split": "test",
    "n_images": len(paths), "weights": WEIGHTS,
    "checkpoint_epoch": "best_ema (epoch 5, still training to 60 in background)",
    "mAP50_classaware": mAP50, "mAP50_95_classaware": mAP50_95,
    "AP50_agnostic": AP50_agnostic, "AP50_95_agnostic": AP50_95_agnostic,
    "reference_class_aware_mAP50_original_2026-08-09": 0.6012,
    "predictions": str(OUT_NPZ),
}
Path("/workspace/project-expertise/results/rfdetr_l_v2repro_953_retrain_2026-09-07.json").write_text(json.dumps(out, indent=2))
print("done")
