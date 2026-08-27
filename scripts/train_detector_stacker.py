"""Fit a train-only class stacker from the three detector sources.

The normal WBF soft vote reduces every source to one class vector.  This
experiment keeps source-specific evidence (best-overlap class, score, IoU,
and multiplicity) and learns a small proposal classifier.  It writes the same
soft-vote NPZ format, so the existing linker/head evaluator can measure the
effect without changing boxes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_remote_pipeline_postprocess as base  # noqa: E402
from sweep_wbf_localization import config_for  # noqa: E402


K = len(base.NAMES)
MODEL_NAMES = ("yolo26l", "rtdetr_l", "rfdetr_l")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key], np.float32)
                for key in archive.files}


def ious(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, np.float32)
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = max(float(box[2] - box[0]), 0.) * max(float(box[3] - box[1]), 0.)
    bb = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None)
    return (inter / (aa + bb - inter + 1e-9)).astype(np.float32)


def gt_by_stem(dataset: str, split: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    name = "SawitMVC-Depth-YOLO" if dataset == "depth" else "SawitMVC-YOLO"
    cfg = base.CONFIGS[name]
    out = {}
    for rec in base.load_records(cfg, split).values():
        for view in rec["views"].values():
            boxes, labels = [], []
            for ann in view.get("annotations", []):
                if ann.get("bbox_pixel") is None:
                    continue
                boxes.append(ann["bbox_pixel"])
                labels.append(int(ann.get("class_id", -1)))
            out[view["stem"]] = (np.asarray(boxes, np.float32).reshape(-1, 4),
                                  np.asarray(labels, np.int64))
    return out


def proposal_features(stem: str, row: np.ndarray,
                      raw: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    values = []
    box = row[:4]
    for name in MODEL_NAMES:
        source = np.asarray(raw[name].get(stem, np.zeros((0, 6))), np.float32)
        ov = ious(box, source[:, :4])
        if len(ov) == 0:
            values.extend([0.] * (K + 4)); continue
        order = np.argsort(-ov)
        best = int(order[0]); matched = ov >= .30
        # Source class is a one-hot prediction; score and overlap indicate
        # whether that class was actually supported by a good source box.
        one = np.zeros(K, np.float32)
        cls = int(source[best, 5])
        if 0 <= cls < K:
            one[cls] = 1.
        values.extend(one.tolist())
        values.extend([float(source[best, 4]), float(ov[best]),
                       float(matched.sum()),
                       float(source[matched, 4].max()) if matched.any() else 0.])
    h, w = max(float(row[3]), 1.), max(float(row[2]), 1.)
    values.extend([float(row[4]), float(row[0] / max(w, 1.)),
                   float(row[1] / max(h, 1.)),
                   float((row[2] - row[0]) / max(w, 1.)),
                   float((row[3] - row[1]) / max(h, 1.))])
    p = np.maximum(row[5:5 + K], 0.)
    p /= max(float(p.sum()), 1e-9)
    values.extend(p.tolist())
    return np.asarray(values, np.float32)


def label_rows(stem: str, rows: np.ndarray,
               gt: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    boxes, labels = gt.get(stem, (np.zeros((0, 4), np.float32),
                                  np.zeros(0, np.int64)))
    y = np.full(len(rows), -1, np.int64)
    for i, row in enumerate(rows):
        ov = ious(row[:4], boxes)
        if len(ov) and float(ov.max()) >= .5:
            cls = int(labels[int(ov.argmax())])
            if 0 <= cls < K:
                y[i] = cls
    return y


def estimator(name: str, seed: int):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=.1, max_iter=1000, solver="lbfgs", random_state=seed))
    if name == "logreg_low":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=.02, max_iter=1000, solver="lbfgs", random_state=seed))
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=500, max_features="sqrt",
                                    min_samples_leaf=5, n_jobs=8,
                                    random_state=seed)
    raise ValueError(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("depth", "953"), required=True)
    ap.add_argument("--artifact-root", type=Path, default=Path(
        "/workspace/model_artifacts/project-expertise/eval_2026-08-27"))
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=("logreg", "logreg_low", "extra_trees"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    split_data = {}
    # Validation is sufficient to screen the stacker.  The historical test
    # directory contains only class-aware/agnostic dumps (not soft-vote rows),
    # so do not manufacture a test input before a candidate is locked.
    for split in ("train", "val"):
        cfg = config_for(args.dataset, split)
        raw = base.load_prediction_bank(cfg)
        fused_dir = args.artifact_root / ("fused_combined1716" if split == "test"
                                           else f"fused_combined1716_{split}")
        safe = "SawitMVC_Depth_YOLO" if args.dataset == "depth" else "SawitMVC_YOLO"
        fused = load_npz(fused_dir / f"{safe}__wbf_softvote.npz")
        gt = gt_by_stem(args.dataset, split)
        x, y, row_keys = [], [], []
        for stem, rows in fused.items():
            for i, row in enumerate(rows):
                x.append(proposal_features(stem, row, raw))
                row_keys.append((stem, i))
            y.extend(label_rows(stem, rows, gt).tolist())
        split_data[split] = (fused, np.asarray(x, np.float32),
                             np.asarray(y, np.int64),
                             row_keys)
        print(json.dumps({"split": split, "rows": len(x),
                          "labelled": int(np.sum(np.asarray(y) >= 0)),
                          "features": int(x[0].shape[0]) if x else 0}, ensure_ascii=False),
              flush=True)
    train_fused, x_train, y_train, _ = split_data["train"]
    keep = y_train >= 0
    results = []
    for name in args.models:
        clf = estimator(name, args.seed)
        clf.fit(x_train[keep], y_train[keep])
        train_acc = float((clf.predict(x_train[keep]) == y_train[keep]).mean())
        out_dir = args.output_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for split, (fused, x, y, keys) in split_data.items():
            p = clf.predict_proba(x)
            output = {k: np.asarray(v, np.float32).copy() for k, v in fused.items()}
            for (stem, index), prob in zip(keys, p):
                output[stem][index, 5:5 + K] = prob
            np.savez_compressed(out_dir / f"fused_{split}__wbf_softvote.npz",
                                **output)
        val_fused, x_val, y_val, _ = split_data["val"]
        val_keep = y_val >= 0
        val_acc = float((clf.predict(x_val[val_keep]) == y_val[val_keep]).mean())
        item = {"model": name, "train_proposal_accuracy": train_acc,
                "val_proposal_accuracy": val_acc,
                "output_root": str(out_dir)}
        results.append(item); print(json.dumps(item, ensure_ascii=False), flush=True)
    output = {"protocol": "train-only per-detector class stacker",
              "dataset": args.dataset, "fit_split": "train",
              "selection_split": "val", "models": results}
    (args.output_root / "metadata.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
