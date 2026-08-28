#!/usr/bin/env python3
"""Validation-only late fusion of fixed new763 RGB/RGB+D4 predictions.

This is deliberately an inference-only follow-up to the 4-channel ablation.
It consumes prediction dumps that have already been frozen, evaluates only
the 468-image ``valid`` split, and has no test argument or test path.  The
recipes are fixed before evaluation:

* ``*_nms_iou060``: class-aware greedy NMS on one frozen source;
* ``union_nms_iou060``: class-aware NMS on RGB + RGB+D4 detections;
* ``union_wbf_avg_iou060``: class-aware greedy weighted-box fusion on the
  union, with score equal to the mean member score.

IoU=0.60 is the project's existing WBF operating point.  There is no
validation parameter sweep here; this script is a cheap screen for whether
the two modalities contain complementary evidence before any new training.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from eval_new763_pycoco import build_gt, evaluate


N_CLASSES = 4
FUSION_IOU = 0.60


def load_dump(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        out: dict[str, np.ndarray] = {}
        for stem in archive.files:
            rows = np.asarray(archive[stem], dtype=np.float32).reshape(-1, 6)
            if not np.isfinite(rows).all():
                raise ValueError(f"non-finite rows in {path}:{stem}")
            if len(rows) and (rows[:, 4] < 0).any():
                raise ValueError(f"negative scores in {path}:{stem}")
            # RF-DETR's frozen prediction API emits a low-score background
            # sentinel as class id 4 in addition to the four task classes.
            # The canonical COCO evaluator ignores category 5; remove it
            # here so fusion has exactly the same semantics as the source
            # evaluation rather than letting an invalid sentinel enter a
            # fusion cluster.
            if len(rows) and (rows[:, 5] < 0).any():
                raise ValueError(f"negative class ids in {path}:{stem}")
            out[stem] = rows[rows[:, 5] < N_CLASSES].copy()
    return out


def iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = max(float(box[2] - box[0]), 0.0) * max(float(box[3] - box[1]), 0.0)
    area_b = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def nms_class_aware(rows: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS, independently for the four predicted classes."""
    if len(rows) == 0:
        return np.empty((0, 6), dtype=np.float32)
    kept: list[np.ndarray] = []
    for cls in range(N_CLASSES):
        part = rows[rows[:, 5].astype(np.int64) == cls]
        order = np.argsort(-part[:, 4], kind="mergesort")
        while len(order):
            index = int(order[0])
            kept.append(part[index])
            order = order[1:]
            if len(order):
                overlap = iou_one(part[index, :4], part[order, :4])
                order = order[overlap < iou_threshold]
    if not kept:
        return np.empty((0, 6), dtype=np.float32)
    result = np.stack(kept).astype(np.float32)
    return result[np.argsort(-result[:, 4], kind="mergesort")]


def wbf_class_aware(rows: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy class-aware WBF with a fixed mean confidence rule.

    All rows in the frozen dumps are retained as input, including the
    detector's low-score tail.  A cluster's coordinates are score-weighted;
    its output score is the arithmetic mean of its member scores.  This is a
    standard, deterministic recipe and intentionally has no learned or
    validation-fitted weight.
    """
    if len(rows) == 0:
        return np.empty((0, 6), dtype=np.float32)
    output: list[np.ndarray] = []
    for cls in range(N_CLASSES):
        part = rows[rows[:, 5].astype(np.int64) == cls]
        if len(part) == 0:
            continue
        part = part[np.argsort(-part[:, 4], kind="mergesort")]
        groups: list[list[np.ndarray]] = []
        centers: list[np.ndarray] = []
        for row in part:
            if centers:
                overlap = np.asarray(
                    [float(iou_one(row[:4], center[None, :])[0]) for center in centers]
                )
                index = int(np.argmax(overlap))
            else:
                overlap, index = np.zeros(0), -1
            if len(overlap) and overlap[index] >= iou_threshold:
                groups[index].append(row)
                group = np.stack(groups[index])
                weights = np.maximum(group[:, 4], 1e-9)
                centers[index] = (group[:, :4] * weights[:, None]).sum(0) / weights.sum()
            else:
                groups.append([row])
                centers.append(row[:4].copy())
        for group, center in zip(groups, centers):
            member = np.stack(group)
            output.append(
                np.asarray(
                    [*center, float(member[:, 4].mean()), float(cls)], dtype=np.float32
                )
            )
    if not output:
        return np.empty((0, 6), dtype=np.float32)
    result = np.stack(output)
    return result[np.argsort(-result[:, 4], kind="mergesort")]


def as_dt(paths: list[Path], predictions: dict[str, np.ndarray]) -> list[dict]:
    stem_to_id = {path.stem: index for index, path in enumerate(paths, 1)}
    unknown = set(predictions).difference(stem_to_id)
    if unknown:
        raise ValueError(f"unknown VAL stems, e.g. {sorted(unknown)[:3]}")
    detections = []
    for stem, rows in predictions.items():
        image_id = stem_to_id[stem]
        for x1, y1, x2, y2, score, cls in rows:
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": int(cls) + 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                }
            )
    return detections


def evaluate_dump(gt, paths: list[Path], predictions: dict[str, np.ndarray]) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        return evaluate(gt, as_dt(paths, predictions))


def fused_dump(
    rgb: dict[str, np.ndarray], rgbd: dict[str, np.ndarray], recipe: str
) -> dict[str, np.ndarray]:
    if recipe == "rgb":
        return rgb
    if recipe == "rgbd4":
        return rgbd
    out: dict[str, np.ndarray] = {}
    for stem in sorted(set(rgb) | set(rgbd)):
        a = rgb.get(stem, np.empty((0, 6), dtype=np.float32))
        b = rgbd.get(stem, np.empty((0, 6), dtype=np.float32))
        joined = np.concatenate([a, b], axis=0)
        if recipe == "rgb_nms_iou060":
            out[stem] = nms_class_aware(a, FUSION_IOU)
        elif recipe == "rgbd4_nms_iou060":
            out[stem] = nms_class_aware(b, FUSION_IOU)
        elif recipe == "union_nms_iou060":
            out[stem] = nms_class_aware(joined, FUSION_IOU)
        elif recipe == "union_wbf_avg_iou060":
            out[stem] = wbf_class_aware(joined, FUSION_IOU)
        else:
            raise ValueError(recipe)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("yolo26l", "rtdetr_l", "rfdetr_l"), required=True)
    parser.add_argument("--rgb-predictions", type=Path, required=True)
    parser.add_argument("--rgbd-predictions", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/new763_rgbd4"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    args = parser.parse_args()

    gt, paths = build_gt(args.dataset, "val")
    if len(paths) != 468:
        raise ValueError(f"new763 RGBD4 VAL must contain 468 images, got {len(paths)}")
    rgb = load_dump(args.rgb_predictions)
    rgbd = load_dump(args.rgbd_predictions)
    expected = {path.stem for path in paths}
    if set(rgb) != expected or set(rgbd) != expected:
        raise ValueError("both prediction dumps must cover exactly the same 468 VAL stems")

    recipes = ("rgb", "rgbd4", "rgb_nms_iou060", "rgbd4_nms_iou060",
               "union_nms_iou060", "union_wbf_avg_iou060")
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "analysis": "fixed late fusion of RGB and RGB+D4; new763 VAL only",
        "protocol": {
            "n_images": len(paths),
            "evaluator": "pycocotools.COCOeval",
            "fusion_iou": FUSION_IOU,
            "wbf_score": "arithmetic mean of member scores",
            "parameter_selection": "none; recipes fixed before evaluation",
            "test_access": "forbidden; this script has no test path or test option",
        },
        "inputs": {
            "dataset": str(args.dataset.resolve()),
            "rgb_predictions": str(args.rgb_predictions.resolve()),
            "rgbd_predictions": str(args.rgbd_predictions.resolve()),
        },
        "recipes": {},
    }
    for recipe in recipes:
        predictions = fused_dump(rgb, rgbd, recipe)
        metrics = evaluate_dump(gt, paths, predictions)
        pred_path = args.pred_dir / f"{args.arch}_{recipe}__val.npz"
        np.savez_compressed(pred_path, **predictions)
        n_detections = int(sum(len(rows) for rows in predictions.values()))
        result["recipes"][recipe] = {
            "metrics": metrics,
            "n_detections": n_detections,
            "predictions": str(pred_path.resolve()),
        }
        print(json.dumps({"recipe": recipe, **metrics, "n_detections": n_detections}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
