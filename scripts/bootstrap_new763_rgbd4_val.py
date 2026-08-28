#!/usr/bin/env python3
"""Paired image-level bootstrap for the new763 RGB versus RGB+D4 VAL.

This script has one legal split (``val``) by design.  It compares two already
trained, fixed submissions on the same 468 validation images using the same
COCO ground truth and ``pycocotools.COCOeval`` path as the project evaluator.
Each bootstrap replicate resamples image indices with replacement and feeds
the identical resample to RGB and RGB+D4, which preserves the paired design.

The script performs no model selection and never reads a test directory.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval_new763_pycoco as enp  # noqa: E402

NAMES = enp.NAMES
N_CLASSES = len(NAMES)


def load_gt_struct(dataset: Path):
    """Extract compact, pickle-friendly VAL GT structures."""
    with redirect_stdout(io.StringIO()):
        gt, paths = enp.build_gt(dataset, "val")
    stems = [p.stem for p in paths]
    image_wh = [None] * len(paths)
    for image in gt.dataset["images"]:
        image_wh[int(image["id"]) - 1] = (int(image["width"]), int(image["height"]))
    gt_anns = [[] for _ in paths]
    for ann in gt.dataset["annotations"]:
        idx = int(ann["image_id"]) - 1
        gt_anns[idx].append(
            (int(ann["category_id"]), [float(x) for x in ann["bbox"]], float(ann["area"]))
        )
    if any(x is None for x in image_wh):
        raise RuntimeError("VAL GT contains an image without width/height")
    return stems, image_wh, gt_anns


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    """Load a prediction dump and normalize every row to x1,y1,x2,y2,score,cls."""
    with np.load(path, allow_pickle=True) as data:
        out: dict[str, np.ndarray] = {}
        for key in data.files:
            rows = np.asarray(data[key], dtype=np.float32)
            if rows.size == 0:
                rows = np.empty((0, 6), dtype=np.float32)
            rows = rows.reshape(-1, 6)
            if not np.isfinite(rows).all():
                raise ValueError(f"non-finite prediction rows in {path}:{key}")
            out[key] = rows
    return out


def preds_to_dt_by_idx(stems: list[str], predictions: dict[str, np.ndarray]) -> list[list[tuple]]:
    stem_to_idx = {stem: idx for idx, stem in enumerate(stems)}
    dt = [[] for _ in stems]
    unknown = set(predictions).difference(stem_to_idx)
    if unknown:
        raise ValueError(f"prediction dump has unknown VAL stems, e.g. {sorted(unknown)[:3]}")
    for stem, rows in predictions.items():
        idx = stem_to_idx[stem]
        for x1, y1, x2, y2, score, klass in rows:
            dt[idx].append(
                (
                    int(klass) + 1,
                    [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    float(score),
                )
            )
    return dt


def build_gt_coco(idx_array: np.ndarray, image_wh: list[tuple[int, int]], gt_anns: list[list[tuple]]) -> COCO:
    images, annotations = [], []
    ann_id = 1
    for new_idx, original_idx in enumerate(idx_array):
        width, height = image_wh[int(original_idx)]
        image_id = new_idx + 1
        images.append({"id": image_id, "width": width, "height": height, "file_name": f"{new_idx}.jpg"})
        for category_id, bbox, area in gt_anns[int(original_idx)]:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    coco = COCO()
    coco.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": idx + 1, "name": name} for idx, name in enumerate(NAMES)],
    }
    with redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def eval_resample(gt: COCO, idx_array: np.ndarray, dt_by_idx: list[list[tuple]]) -> float:
    detections = []
    for new_idx, original_idx in enumerate(idx_array):
        image_id = new_idx + 1
        for category_id, bbox, score in dt_by_idx[int(original_idx)]:
            detections.append(
                {"image_id": image_id, "category_id": category_id, "bbox": bbox, "score": score}
            )
    with redirect_stdout(io.StringIO()):
        dt = gt.loadRes(detections) if detections else gt.loadRes([])
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return float(evaluator.stats[1])


_WORKER: dict[str, Any] = {}


def worker_init(image_wh, gt_anns, rgb_dt, rgbd_dt):
    _WORKER.update(image_wh=image_wh, gt_anns=gt_anns, rgb_dt=rgb_dt, rgbd_dt=rgbd_dt)


def process_resample(task):
    resample_id, indices = task
    gt = build_gt_coco(indices, _WORKER["image_wh"], _WORKER["gt_anns"])
    rgb = eval_resample(gt, indices, _WORKER["rgb_dt"])
    rgbd = eval_resample(gt, indices, _WORKER["rgbd_dt"])
    return int(resample_id), rgb, rgbd


def ci95(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/SawitMVC-Depth-YOLO"))
    parser.add_argument("--rgb-predictions", type=Path, required=True)
    parser.add_argument("--rgbd-predictions", type=Path, required=True)
    parser.add_argument("--rgb-result", type=Path, required=True)
    parser.add_argument("--rgbd-result", type=Path, required=True)
    parser.add_argument("--n-resamples", type=int, default=500)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.n_resamples < 100 or args.workers < 1:
        raise ValueError("use at least 100 resamples and one worker")
    for path in (args.rgb_predictions, args.rgbd_predictions, args.rgb_result, args.rgbd_result):
        if not path.is_file():
            raise FileNotFoundError(path)

    stems, image_wh, gt_anns = load_gt_struct(args.dataset)
    if len(stems) != 468:
        raise ValueError(f"new763 VAL must contain 468 images, got {len(stems)}")
    rgb_predictions = load_predictions(args.rgb_predictions)
    rgbd_predictions = load_predictions(args.rgbd_predictions)
    if set(rgb_predictions) != set(stems) or set(rgbd_predictions) != set(stems):
        raise ValueError("both prediction dumps must cover exactly the same 468 VAL stems")
    rgb_dt = preds_to_dt_by_idx(stems, rgb_predictions)
    rgbd_dt = preds_to_dt_by_idx(stems, rgbd_predictions)

    rgb_result = json.loads(args.rgb_result.read_text())
    rgbd_result = json.loads(args.rgbd_result.read_text())
    rgb_point = float(rgb_result["splits"]["val"]["mAP50"])
    rgbd_point = float(rgbd_result["metrics"]["mAP50"])

    identity = np.arange(len(stems), dtype=np.int64)
    identity_gt = build_gt_coco(identity, image_wh, gt_anns)
    identity_rgb = eval_resample(identity_gt, identity, rgb_dt)
    identity_rgbd = eval_resample(identity_gt, identity, rgbd_dt)
    if abs(identity_rgb - rgb_point) > 1e-6 or abs(identity_rgbd - rgbd_point) > 1e-6:
        raise RuntimeError(
            "identity sanity check failed: "
            f"RGB {identity_rgb:.9f}/{rgb_point:.9f}, "
            f"RGBD {identity_rgbd:.9f}/{rgbd_point:.9f}"
        )
    print(
        f"identity sanity OK: RGB={identity_rgb:.6f}, RGBD={identity_rgbd:.6f}, "
        f"delta={identity_rgbd - identity_rgb:+.6f}",
        flush=True,
    )

    rng = np.random.RandomState(args.seed)
    resamples = rng.randint(0, len(stems), size=(args.n_resamples, len(stems)))
    rgb_values = np.empty(args.n_resamples, dtype=np.float64)
    rgbd_values = np.empty(args.n_resamples, dtype=np.float64)
    started = time.time()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=worker_init,
        initargs=(image_wh, gt_anns, rgb_dt, rgbd_dt),
    ) as pool:
        tasks = ((idx, resamples[idx]) for idx in range(args.n_resamples))
        for done, (resample_id, rgb_value, rgbd_value) in enumerate(
            pool.map(process_resample, tasks, chunksize=2), start=1
        ):
            rgb_values[resample_id] = rgb_value
            rgbd_values[resample_id] = rgbd_value
            if done % 50 == 0 or done == args.n_resamples:
                print(f"{done}/{args.n_resamples} resamples; {time.time() - started:.1f}s", flush=True)

    delta = rgbd_values - rgb_values
    result = {
        "schema_version": 1,
        "analysis": "paired image-level bootstrap, new763 VAL only",
        "protocol": {
            "n_images": len(stems),
            "n_resamples": args.n_resamples,
            "seed": args.seed,
            "same_resample_indices_for_both_models": True,
            "evaluator": "pycocotools.COCOeval",
            "selection": "none; weights and prediction dumps fixed before bootstrap",
            "test_access": "forbidden; this script has no test path or test option",
        },
        "point": {
            "rgb_mAP50": rgb_point,
            "rgbd_mAP50": rgbd_point,
            "delta_rgbd_minus_rgb": rgbd_point - rgb_point,
        },
        "bootstrap": {
            "rgb_mAP50_ci95": ci95(rgb_values),
            "rgbd_mAP50_ci95": ci95(rgbd_values),
            "delta_ci95": ci95(delta),
            "delta_mean": float(delta.mean()),
            "fraction_delta_gt_zero": float((delta > 0).mean()),
            "significant_excludes_zero": bool(np.percentile(delta, 2.5) > 0 or np.percentile(delta, 97.5) < 0),
        },
        "identity_sanity": {"rgb": identity_rgb, "rgbd": identity_rgbd},
        "inputs": {
            "dataset": str(args.dataset.resolve()),
            "rgb_predictions": str(args.rgb_predictions.resolve()),
            "rgbd_predictions": str(args.rgbd_predictions.resolve()),
            "rgb_result": str(args.rgb_result.resolve()),
            "rgbd_result": str(args.rgbd_result.resolve()),
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
