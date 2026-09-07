import json
import os
from pathlib import Path
import sys
sys.path.insert(0, "/workspace/project-expertise/scripts")
from eval_new763_pycoco import build_gt

SRC = Path("/workspace/SawitMVC-YOLO")
DST = Path("/workspace/SawitMVC-YOLO-coco")

SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}

for yolo_split, coco_split in SPLIT_MAP.items():
    out_dir = DST / coco_split
    out_dir.mkdir(parents=True, exist_ok=True)
    gt, paths = build_gt(SRC, yolo_split)
    for p in paths:
        link = out_dir / p.name
        if not link.exists():
            os.symlink(p.resolve(), link)
    (out_dir / "_annotations.coco.json").write_text(json.dumps(gt.dataset))
    print(f"{coco_split}: {len(gt.dataset['images'])} images, {len(gt.dataset['annotations'])} annotations")

print("done")
