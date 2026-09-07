import json
import pickle
from pathlib import Path
import sys
import numpy as np
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, "/workspace/project-expertise/scripts")
from eval_new763_pycoco import build_gt

DATASET = Path("/workspace/SawitMVC-YOLO")
D = pickle.load(open("/workspace/results_panen/dets.pkl", "rb"))

results = {}
for split_key, yolo_split in [("val", "val"), ("test", "test")]:
    dets_by_tree = D[split_key]
    gt, paths = build_gt(DATASET, yolo_split)
    for ann in gt.dataset["annotations"]:
        ann["category_id"] = 1
    gt.dataset["categories"] = [{"id": 1, "name": "tandan"}]
    gt.createIndex()
    stem_to_id = {p.stem: i for i, p in enumerate(paths, 1)}

    dt_list = []
    for tree, dlist in dets_by_tree.items():
        for d in dlist:
            stem = Path(d["img"]).stem
            image_id = stem_to_id.get(stem)
            if image_id is None:
                continue
            x1, y1, x2, y2 = d["px"]
            dt_list.append({"image_id": image_id, "category_id": 1,
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(d["conf"])})
    dt = gt.loadRes(dt_list) if dt_list else gt.loadRes([])
    ev = COCOeval(gt, dt, "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    AP50 = float(ev.stats[1]); AP50_95 = float(ev.stats[0])
    print(f"{split_key}: AP50_agnostic={AP50:.4f} AP50_95_agnostic={AP50_95:.4f} n_images={len(paths)} n_dets={len(dt_list)}")
    results[split_key] = {"AP50_agnostic": AP50, "AP50_95_agnostic": AP50_95,
                          "n_images": len(paths), "n_dets": len(dt_list)}

out = {
    "node": "AF-E-012",
    "model": "agnostik_m1280 (YOLO26m class-agnostic, Pipeline Panen)",
    "dataset": "953",
    "weights": "audit_forensik_2026-09-06/runs_panen/agnostik_m1280/weights/best.pt (bucket)",
    "note": "AP50 class-agnostic dihitung dari dets.pkl (dump deteksi mentah conf>=0.10, hasil regenerasi stage-1 panen_pipeline.py sesi 2026-09-07) terhadap GT standar (labels/{split} YOLO format, ULM-DS-Lab/SawitMVC-YOLO). Kolom 'detection' baris AF-E-012 val pada recap.md masih '.'; nilai test/uji (0,8104) sudah cocok dengan AF-E-011 (detektor yang sama).",
    "results": results,
}
Path("/workspace/project-expertise/results/agnostic_ap50_panen_2026-09-07.json").write_text(json.dumps(out, indent=2))
print("saved")
