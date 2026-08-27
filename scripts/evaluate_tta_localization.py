"""Evaluate RF-DETR photometric TTA as an extra localization source.

The original RF-DETR predictions and each transformed-image prediction are
first fused into one RF-DETR source.  That prevents the support penalty from
treating TTA copies as independent detectors.  The resulting source is then
fused with the original YOLO26l and RT-DETR sources using the normal WBF code.
All choices are intended to be made on validation.
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
from sweep_wbf_localization import config_for  # noqa: E402


ARTIFACT_ROOT = Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key], np.float32) for key in archive.files}


def combine_rf(original: dict[str, np.ndarray], variants: list[dict[str, np.ndarray]],
               inner_iou: float, inner_score: float) -> dict[str, np.ndarray]:
    out = {}
    stems = sorted(set(original) | {stem for vote in variants for stem in vote})
    for stem in stems:
        parts = [original.get(stem, np.zeros((0, 6), np.float32))]
        parts.extend(vote.get(stem, np.zeros((0, 6), np.float32))
                     for vote in variants)
        rows = [x for x in parts if len(x)]
        if not rows:
            out[stem] = np.zeros((0, 6), np.float32)
            continue
        # All rows deliberately use source id 0: these are views of one
        # detector, not independent ensemble members.
        joined = np.concatenate([np.c_[x[:, :6], np.zeros((len(x), 1))]
                                 for x in rows], axis=0)
        groups = base.fuse_groups(joined, inner_iou, inner_score, 1)
        out[stem] = np.asarray(
            [[*g["box"], g["score"], int(np.argmax(g["p"]))] for g in groups],
            np.float32).reshape(-1, 6)
    return out


def evaluate(dataset: str, split: str, variants: list[str], inner_iou: float,
             inner_score: float, outer_iou: float, outer_score: float,
             workers: int, tta_dir: Path) -> dict:
    cfg = config_for(dataset, split)
    records = base.load_records(cfg, split)
    raw = base.load_prediction_bank(cfg)
    tta = [load_npz(tta_dir / f"{dataset}_{split}_{variant}.npz")
           for variant in variants]
    raw["rfdetr_l"] = combine_rf(raw["rfdetr_l"], tta, inner_iou, inner_score)
    with contextlib.redirect_stdout(io.StringIO()):
        _ca, agnostic, _vote = base.fuse_corpus(
            records, raw, outer_iou, outer_score, workers)
        metrics = base.coco_metrics(cfg["data_root"], agnostic, True, split)
    return {
        "dataset": dataset, "split": split, "tta_variants": variants,
        "inner_iou": inner_iou, "inner_score": inner_score,
        "outer_iou": outer_iou, "outer_score": outer_score,
        "metrics": metrics,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("depth", "953"), required=True)
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--variants", nargs="+", choices=("hflip", "vflip", "rot180",
                                                        "clahe", "unsharp", "gamma095", "gamma105", "hue2"),
                    required=True)
    ap.add_argument("--inner-ious", nargs="+", type=float, default=[.5, .6, .7])
    ap.add_argument("--inner-score", type=float, default=.025)
    ap.add_argument("--outer-ious", nargs="+", type=float, default=[.55, .6])
    ap.add_argument("--outer-scores", nargs="+", type=float, default=[.025, .05])
    ap.add_argument("--tta-dir", type=Path,
                    default=Path("/workspace/model_artifacts/project-expertise/tta_rfdetr"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    results = []
    for inner in args.inner_ious:
        for outer in args.outer_ious:
            for outer_score in args.outer_scores:
                item = evaluate(args.dataset, args.split, args.variants, inner,
                                args.inner_score, outer, outer_score, args.workers,
                                args.tta_dir)
                results.append(item)
                print(json.dumps(item, ensure_ascii=False), flush=True)
    best = max(results, key=lambda x: (x["metrics"]["mAP50"],
                                       x["metrics"]["mAP50_95"]))
    output = {"protocol": "RF-DETR photometric TTA localization",
              "results": results, "best": best}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
