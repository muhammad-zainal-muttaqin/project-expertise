"""Adaptor: RT-DETR-L (ultralytics) -> JSON per-pohon (docs/SCHEMA-PERTREE.md).

Usage:
    .venv/bin/python rtdetr_to_pertree.py \
        --weights /path/to/rtdetr_l_e60_i1280_v2repro/weights/best.pt \
        --image-dir /workspace/SawitMVC/data/images \
        --split-file /workspace/SawitMVC/test.txt --split-name test \
        --out-dir /workspace/project-expertise/runs/pertree/rtdetr_l_v2repro \
        --detector-name rtdetr_l_v2repro

Tanpa --split-file, memproses seluruh citra di --image-dir sebagai satu
"split" (dipakai untuk smoke test).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import RTDETR

from common_pertree import (
    CLASS_NAMES,
    group_images_by_tree,
    load_split_stems,
    parse_tree_and_side,
    write_pertree_json,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image-dir", required=True, type=Path)
    ap.add_argument("--split-file", type=Path, default=None)
    ap.add_argument("--split-name", default="all")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--detector-name", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--limit-trees", type=int, default=None, help="Smoke test: proses N pohon pertama saja")
    args = ap.parse_args()

    stems = load_split_stems(args.split_file) if args.split_file else None
    trees = group_images_by_tree(args.image_dir, stems)
    tree_names = sorted(trees)
    if args.limit_trees:
        tree_names = tree_names[: args.limit_trees]

    model = RTDETR(args.weights)

    n_written = 0
    for tree_name in tree_names:
        images_payload: dict[str, dict] = {}
        for img_path in sorted(trees[tree_name]):
            _, side = parse_tree_and_side(img_path)
            result = model.predict(source=str(img_path), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            annotations = []
            for box in result.boxes:
                cls_id = int(box.cls.item())
                class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls{cls_id}"
                cx, cy, w, h = box.xywhn[0].tolist()
                annotations.append(
                    {
                        "class_name": class_name,
                        "bbox_yolo": [cx, cy, w, h],
                        "conf": float(box.conf.item()),
                    }
                )
            images_payload[f"side_{side}"] = {"side_index": side - 1, "annotations": annotations}

        write_pertree_json(args.out_dir, tree_name, args.split_name, args.detector_name, images_payload)
        n_written += 1

    print(f"Selesai: {n_written} pohon ditulis ke {args.out_dir}")


if __name__ == "__main__":
    main()
