"""Adaptor: RF-DETR-L (rfdetr) -> JSON per-pohon (docs/SCHEMA-PERTREE.md).

RF-DETR mengembalikan supervision.Detections (xyxy piksel, class_id, confidence)
-- beda dari ultralytics (xywhn ternormalisasi) -- jadi perlu normalisasi
manual pakai ukuran citra.

Usage:
    .venv/bin/python rfdetr_to_pertree.py \
        --weights /path/to/rfdetr_l_e60_i1280_v2repro/checkpoint_best_ema.pth \
        --image-dir /workspace/SawitMVC/data/images \
        --split-file /workspace/SawitMVC/test.txt --split-name test \
        --out-dir /workspace/project-expertise/runs/pertree/rfdetr_l_v2repro \
        --detector-name rfdetr_l_v2repro

Tanpa --weights, memakai bobot dasar RF-DETR-L pretrained COCO (untuk smoke
test pipa data saja -- kelas dan angka tidak bermakna).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from rfdetr import RFDETRLarge

from common_pertree import (
    CLASS_NAMES,
    group_images_by_tree,
    load_split_stems,
    parse_tree_and_side,
    write_pertree_json,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="checkpoint .pth; kosong = bobot dasar COCO pretrained")
    ap.add_argument("--image-dir", required=True, type=Path)
    ap.add_argument("--split-file", type=Path, default=None)
    ap.add_argument("--split-name", default="all")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--detector-name", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--resolution", type=int, default=1280)
    ap.add_argument("--limit-trees", type=int, default=None, help="Smoke test: proses N pohon pertama saja")
    args = ap.parse_args()

    stems = load_split_stems(args.split_file) if args.split_file else None
    trees = group_images_by_tree(args.image_dir, stems)
    tree_names = sorted(trees)
    if args.limit_trees:
        tree_names = tree_names[: args.limit_trees]

    kwargs = {"resolution": args.resolution}
    if args.weights:
        kwargs["pretrain_weights"] = args.weights
    model = RFDETRLarge(**kwargs)

    n_written = 0
    for tree_name in tree_names:
        images_payload: dict[str, dict] = {}
        for img_path in sorted(trees[tree_name]):
            _, side = parse_tree_and_side(img_path)
            pil_img = Image.open(img_path).convert("RGB")
            w, h = pil_img.size
            detections = model.predict(pil_img, threshold=args.conf)

            annotations = []
            for (x1, y1, x2, y2), conf, cls_id in zip(
                detections.xyxy, detections.confidence, detections.class_id
            ):
                cls_id = int(cls_id)
                class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls{cls_id}"
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                annotations.append(
                    {
                        "class_name": class_name,
                        "bbox_yolo": [float(cx), float(cy), float(bw), float(bh)],
                        "conf": float(conf),
                    }
                )
            images_payload[f"side_{side}"] = {"side_index": side - 1, "annotations": annotations}

        write_pertree_json(args.out_dir, tree_name, args.split_name, args.detector_name, images_payload)
        n_written += 1

    print(f"Selesai: {n_written} pohon ditulis ke {args.out_dir}")


if __name__ == "__main__":
    main()
