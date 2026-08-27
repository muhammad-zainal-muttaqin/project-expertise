"""Validation-only sweep for class-agnostic WBF localization.

The remote detector bank already contains the raw predictions, so this script
does not spend GPU time re-running inference.  It searches only post-processing
parameters (WBF IoU, input score floor, and optionally model weights) on VAL,
where the choice is legal.  The selected recipe can then be re-run once on
TEST with the exact values recorded in the output JSON.

This measures localization independently of the four-class head: every fused
box is assigned one category and evaluated against class-agnostic ground truth.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_remote_pipeline_postprocess as base  # noqa: E402


ARTIFACT_ROOT = Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27")


def config_for(dataset: str, split: str) -> dict:
    if dataset == "depth":
        name, short = "SawitMVC-Depth-YOLO", "depth"
    else:
        name, short = "SawitMVC-YOLO", "953"
    source = base.CONFIGS[name]
    pred_root = ARTIFACT_ROOT / "predictions_combined1716"
    cfg = {**source, "predictions": {}}
    for model in ("yolo26l", "rtdetr_l", "rfdetr_l"):
        cfg["predictions"][model] = (
            pred_root /
            f"remote_combined1716_{model}_{short}_{split}__{split}.npz")
    return cfg


def parse_weight_sets(values: list[str] | None) -> list[np.ndarray | None]:
    if not values:
        return [None]
    out = []
    for value in values:
        weights = np.asarray([float(x) for x in value.split(",")], float)
        if len(weights) != 3 or np.any(weights <= 0):
            raise ValueError("model weights harus berupa tiga angka positif, mis. 1,1,1")
        out.append(weights)
    return out


def evaluate_dataset(dataset: str, split: str, ious: list[float],
                     score_mins: list[float], weight_sets: list[np.ndarray | None],
                     workers: int, save_best_root: Path | None) -> dict:
    cfg = config_for(dataset, split)
    records = base.load_records(cfg, split)
    bank = base.load_prediction_bank(cfg)
    rows = []
    best = None
    for weights in weight_sets:
        label = "equal" if weights is None else ",".join(map(str, weights.tolist()))
        for iou_threshold in ious:
            for score_min in score_mins:
                _ca, agnostic, _vote = base.fuse_corpus(
                    records, bank, iou_threshold, score_min, workers,
                    weights)
                # COCOeval.summarize() is useful interactively but emits a
                # dozen lines per candidate; keep the sweep log machine-sized.
                with contextlib.redirect_stdout(io.StringIO()):
                    metrics = base.coco_metrics(
                        cfg["data_root"], agnostic, True, split)
                item = {
                    "dataset": dataset, "split": split,
                    "iou_threshold": float(iou_threshold),
                    "input_score_min": float(score_min),
                    "model_weights": (None if weights is None else
                                       weights.tolist()),
                    "model_weights_label": label,
                    "metrics": metrics,
                }
                rows.append(item)
                score = (metrics["mAP50"], metrics["mAP50_95"])
                if best is None or score > best["_score"]:
                    best = {"_score": score, "item": item, "predictions": agnostic}
                print(json.dumps({
                    "dataset": dataset, "split": split,
                    "iou": iou_threshold, "score_min": score_min,
                    "weights": label,
                    **metrics,
                }, ensure_ascii=False), flush=True)
    if best is None:
        raise RuntimeError("sweep kosong")
    best_item = best["item"]
    if save_best_root is not None:
        out = save_best_root / split
        out.mkdir(parents=True, exist_ok=True)
        safe = "SawitMVC_Depth_YOLO" if dataset == "depth" else "SawitMVC_YOLO"
        np.savez_compressed(out / f"{safe}__wbf_agnostic_best.npz",
                            **best["predictions"])
        best_item["saved_predictions"] = str(
            out / f"{safe}__wbf_agnostic_best.npz")
    return {"dataset": dataset, "split": split, "n_records": len(records),
            "best": best_item, "results": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--datasets", nargs="+", choices=("depth", "953"),
                    default=("depth", "953"))
    ap.add_argument("--ious", nargs="+", type=float,
                    default=[.40, .45, .50, .55, .60, .65, .70, .75, .80])
    ap.add_argument("--score-mins", nargs="+", type=float,
                    default=[.001, .005, .01, .025, .05, .075, .10])
    ap.add_argument("--model-weight-sets", nargs="*",
                    help="optional comma-separated triples, e.g. 1,1,1 .75,1,1.5")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--save-best-root", type=Path, default=None)
    args = ap.parse_args()
    if any(not 0 <= x <= 1 for x in args.ious):
        ap.error("IoU harus berada di [0,1]")
    if any(x < 0 for x in args.score_mins):
        ap.error("score minimum tidak boleh negatif")
    weights = parse_weight_sets(args.model_weight_sets)
    result = {
        "protocol": "validation-only WBF localization sweep",
        "split": args.split, "datasets": {},
        "ious": args.ious, "score_mins": args.score_mins,
        "model_weight_sets": [None if x is None else x.tolist() for x in weights],
        "workers": args.workers,
    }
    for dataset in args.datasets:
        result["datasets"][dataset] = evaluate_dataset(
            dataset, args.split, args.ious, args.score_mins, weights,
            args.workers, args.save_best_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"-> {args.output}")
    for dataset, data in result["datasets"].items():
        print(json.dumps({"dataset": dataset, "best": data["best"]},
                         ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
