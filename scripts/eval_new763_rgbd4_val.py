#!/usr/bin/env python3
"""Validation-only COCO evaluation for a new763 RGB+D detector.

The CLI has no test option by design.  It evaluates the 468-image ``valid``
split and compares the result with the corresponding locked RGB baseline
number already recorded in ``results/new763_summary.json``.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_new763_pycoco import build_gt, evaluate  # noqa: E402
from train_new763_rgbd4 import (  # noqa: E402
    _depth_train_stats,
    patch_rfdetr_backbone,
    patch_rfdetr_loader,
    patch_rfdetr_normalize,
    patch_ultralytics_rgbd,
    validate_dataset,
)


def _rgbd(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"expected BGRD TIFF {path}, got {None if image is None else image.shape}")
    return np.ascontiguousarray(image[..., [2, 1, 0, 3]])


def predict_ultra_rgbd(model, paths: list[Path], imgsz: int, batch: int, agnostic_nms: bool = False):
    predictions, dump = [], {}
    for start in range(0, len(paths), batch):
        chunk = paths[start : start + batch]
        images = [_rgbd(path) for path in chunk]
        results = model.predict(
            images,
            imgsz=imgsz,
            conf=0.001,
            iou=0.7,
            max_det=300,
            agnostic_nms=agnostic_nms,
            verbose=False,
            save=False,
        )
        for offset, (path, result) in enumerate(zip(chunk, results)):
            rows = []
            boxes = result.boxes
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.detach().cpu().numpy()
                conf = boxes.conf.detach().cpu().numpy()
                cls = boxes.cls.detach().cpu().numpy()
                for (x1, y1, x2, y2), score, klass in zip(xyxy, conf, cls):
                    rows.append([float(x1), float(y1), float(x2), float(y2), float(score), float(klass)])
                    predictions.append(
                        {
                            "image_id": start + offset + 1,
                            "category_id": int(klass) + 1,
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(score),
                        }
                    )
            dump[path.stem] = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
    return predictions, dump


def _state_from_checkpoint(path: Path) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    model = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if isinstance(model, nn.Module):
        return model.state_dict()
    if isinstance(model, dict):
        return model
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        return checkpoint["state_dict"]
    raise ValueError(f"cannot extract RF-DETR state dict from {path}")


def load_rfdetr_4ch(weights: Path, dataset: Path, imgsz: int, seed: int):
    """Build a 4ch RF-DETR without pretrain auto-load, then load 4ch state."""
    import torch
    from rfdetr import RFDETRLarge

    mean_d, std_d, stats = _depth_train_stats(dataset, seed)
    # These patches are harmless for inference but make the provenance and
    # model construction identical to the training process.
    patch_rfdetr_loader()
    patch_rfdetr_normalize(mean_d, std_d)
    patch_rfdetr_backbone()
    model = RFDETRLarge(
        gradient_checkpointing=False,
        resolution=imgsz,
        num_channels=4,
        num_classes=4,
        pretrain_weights=None,
    )
    network = model.model.model if hasattr(model.model, "model") else model.model
    state = _state_from_checkpoint(weights)
    incompatible = network.load_state_dict(state, strict=False)
    missing = [str(k) for k in incompatible.missing_keys]
    unexpected = [str(k) for k in incompatible.unexpected_keys]
    # A 4ch trained checkpoint must at least load the 4ch patch projection.
    patch_key = "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight"
    if patch_key not in state or tuple(state[patch_key].shape)[1] != 4:
        raise ValueError(f"RF-DETR checkpoint is not a 4-channel checkpoint: {patch_key}={getattr(state.get(patch_key), 'shape', None)}")
    if any(not (k.startswith("criterion.") or k.startswith("postprocess.")) for k in missing):
        raise RuntimeError(f"unexpected missing RF-DETR weights: {missing[:10]}")
    model.means = [0.485, 0.456, 0.406, mean_d]
    model.stds = [0.229, 0.224, 0.225, std_d]
    return model, {"depth_stats": stats, "missing_keys": missing, "unexpected_keys": unexpected}


def predict_rfdetr_rgbd(model, paths: list[Path], batch: int):
    predictions, dump = [], {}
    for start in range(0, len(paths), batch):
        chunk = paths[start : start + batch]
        detections = model.predict([_rgbd(path) for path in chunk], threshold=0.001, include_source_image=False)
        if not isinstance(detections, list):
            detections = [detections]
        for offset, (path, det) in enumerate(zip(chunk, detections)):
            rows = []
            for xyxy, score, klass in zip(det.xyxy, det.confidence, det.class_id):
                x1, y1, x2, y2 = map(float, xyxy)
                score, klass = float(score), int(klass)
                rows.append([x1, y1, x2, y2, score, float(klass)])
                predictions.append(
                    {
                        "image_id": start + offset + 1,
                        "category_id": klass + 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score,
                    }
                )
            dump[path.stem] = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
    return predictions, dump


def main() -> int:
    if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
        cv2.utils.logging.setLogLevel(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=("yolo26l", "rtdetr_l", "rfdetr_l"), required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=Path("/workspace/new763_rgbd4"))
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--pred-dir", type=Path, required=True)
    args = ap.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    data_yaml = validate_dataset(args.dataset)
    # The evaluator has one legal split, hard-coded here intentionally.
    gt, paths = build_gt(args.dataset, "val")
    if len(paths) != 468:
        raise ValueError(f"new763 validation is expected to contain 468 images, got {len(paths)}")

    extra: dict[str, Any] = {}
    if args.arch == "yolo26l":
        from ultralytics import YOLO

        patch_ultralytics_rgbd()
        model = YOLO(str(args.weights))
        dt_list, dump = predict_ultra_rgbd(model, paths, args.imgsz, args.batch)
    elif args.arch == "rtdetr_l":
        from ultralytics import RTDETR

        patch_ultralytics_rgbd()
        model = RTDETR(str(args.weights))
        dt_list, dump = predict_ultra_rgbd(model, paths, args.imgsz, args.batch)
    else:
        model, extra = load_rfdetr_4ch(args.weights, args.dataset, args.imgsz, args.seed)
        dt_list, dump = predict_rfdetr_rgbd(model, paths, args.batch)

    metrics = evaluate(gt, dt_list)
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.pred_dir / f"{args.run_name}__val.npz"
    np.savez_compressed(pred_path, **dump)

    baseline_path = Path(__file__).resolve().parents[1] / "results" / "new763_summary.json"
    baseline = json.loads(baseline_path.read_text())
    baseline_val = next(r["val_mAP50"] for r in baseline["runs"] if r["arch"] == args.arch)
    result = {
        "schema_version": 1,
        "run_name": args.run_name,
        "arch": args.arch,
        "weights": str(args.weights.resolve()),
        "dataset": str(args.dataset.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "split": "val only (468 images); test forbidden",
        "imgsz": args.imgsz,
        "evaluator": "pycocotools.COCOeval",
        "metrics": metrics,
        "rgb_baseline_val_mAP50": baseline_val,
        "delta_vs_rgb_baseline_val_mAP50": round(metrics["mAP50"] - baseline_val, 6),
        "predictions": str(pred_path.resolve()),
        "extra": extra,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)
    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
