"""Fit a train-only class-probability calibrator for WBF proposals.

This is a deliberately conservative residual layer: ``softmax(W log p + b)``
starts at the detector identity mapping and is strongly regularized toward it.
It can correct systematic B1--B4 confusion without changing boxes, scores,
linking, or counting.  Parameters are fitted only from train proposals with
IoU >= 0.5 to a labelled bunch.  Validation/test are transformed only after
the mapping has been fitted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_remote_pipeline_postprocess as base  # noqa: E402


K = len(base.NAMES)
IDENTITY = np.eye(K, dtype=np.float64)


def vote_path(root: Path, safe: str, split: str) -> Path:
    folder = root / ("fused_combined1716" if split == "test"
                     else f"fused_combined1716_{split}")
    path = folder / f"{safe}__wbf_softvote.npz"
    if path.exists():
        return path
    if split == "test":
        return (Path(__file__).resolve().parents[1] /
                "results" / "remote_eval_2026-08-27" /
                "fused_combined1716" / f"{safe}__wbf_softvote.npz")
    return path


def load_vote(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key], np.float32)
                for key in archive.files}


def image_gt(view: dict) -> tuple[np.ndarray, np.ndarray]:
    boxes, labels = [], []
    for ann in view.get("annotations", []):
        box = ann.get("bbox_pixel")
        if box is None:
            continue
        boxes.append([float(x) for x in box])
        labels.append(int(ann.get("class_id", -1)))
    return np.asarray(boxes, np.float64).reshape(-1, 4), np.asarray(labels, np.int64)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(x2 - x1, 0.) * max(y2 - y1, 0.)
    aa = max(float(a[2] - a[0]), 0.) * max(float(a[3] - a[1]), 0.)
    bb = max(float(b[2] - b[0]), 0.) * max(float(b[3] - b[1]), 0.)
    return inter / (aa + bb - inter + 1e-9)


def matched_probabilities(cfg: dict, split: str,
                          vote: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rows, labels = [], []
    records = base.load_records(cfg, split)
    for rec in records.values():
        for view in rec["views"].values():
            proposals = np.asarray(vote.get(view["stem"],
                                            np.zeros((0, 5 + K))), np.float32)
            gt, gt_cls = image_gt(view)
            for row in proposals:
                best, best_cls = 0., -1
                for box, cls in zip(gt, gt_cls):
                    ov = iou(row[:4], box)
                    if ov > best:
                        best, best_cls = ov, int(cls)
                if best >= .5 and 0 <= best_cls < K:
                    p = np.maximum(row[5:5 + K], 0.)
                    p /= max(float(p.sum()), 1e-9)
                    rows.append(np.log(p + 1e-4))
                    labels.append(best_cls)
    return np.asarray(rows, np.float64).reshape(-1, K), np.asarray(labels, np.int64)


def fit_mapping(logp: np.ndarray, y: np.ndarray, regularization: float,
                balanced: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    counts = np.bincount(y, minlength=K).astype(np.float64)
    class_weight = 1. / np.sqrt(np.maximum(counts, 1.)) if balanced else np.ones(K)
    class_weight /= class_weight.mean()
    start = np.r_[IDENTITY.ravel(), np.zeros(K)]

    def unpack(theta):
        return theta[:K * K].reshape(K, K), theta[K * K:]

    def objective(theta):
        W, b = unpack(theta)
        z = logp @ W.T + b
        log_z = z - np.logaddexp.reduce(z, axis=1, keepdims=True)
        nll = -np.mean(log_z[np.arange(len(y)), y] * class_weight[y])
        penalty = regularization * (((W - IDENTITY) ** 2).mean() +
                                    (b ** 2).mean())
        return float(nll + penalty)

    def gradient(theta):
        W, b = unpack(theta)
        p = softmax(logp @ W.T + b, axis=1)
        q = p.copy()
        q[np.arange(len(y)), y] -= 1.
        q *= class_weight[y, None]
        q /= max(len(y), 1)
        grad_w = q.T @ logp + 2. * regularization * (W - IDENTITY) / K**2
        grad_b = q.sum(0) + 2. * regularization * b / K
        return np.r_[grad_w.ravel(), grad_b]

    result = minimize(objective, start, jac=gradient, method="L-BFGS-B",
                      options={"maxiter": 1000, "ftol": 1e-12,
                               "gtol": 1e-9})
    W, b = unpack(result.x)
    return W, b, {"success": bool(result.success), "message": result.message,
                   "iterations": int(result.nit), "regularization": regularization,
                   "balanced": balanced, "class_counts": counts.astype(int).tolist()}


def classify_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p.argmax(1)
    f1 = []
    for c in range(K):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {"n": int(len(y)), "accuracy": float((pred == y).mean()),
            "macro_f1": float(np.mean(f1)),
            "f1_per_class": dict(zip(base.NAMES, f1))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("depth", "953"), required=True)
    ap.add_argument("--fused-root", type=Path,
                    default=Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27"))
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--regularization", type=float, default=10.)
    ap.add_argument("--balanced", action="store_true")
    args = ap.parse_args()
    dataset_name = "SawitMVC-Depth-YOLO" if args.dataset == "depth" else "SawitMVC-YOLO"
    safe = "SawitMVC_Depth_YOLO" if args.dataset == "depth" else "SawitMVC_YOLO"
    cfg = base.CONFIGS[dataset_name]
    votes, split_stats = {}, {}
    for split in ("train", "val", "test"):
        path = vote_path(args.fused_root, safe, split)
        if not path.exists():
            raise FileNotFoundError(path)
        votes[split] = load_vote(path)
        logp, y = matched_probabilities(cfg, split, votes[split])
        split_stats[split] = {"matched": int(len(y)),
                             "baseline": classify_metrics(y, np.exp(logp))}
        if split == "train":
            train_logp, train_y = logp, y
        else:
            split_stats[split]["input_path"] = str(path)
    W, b, fit_meta = fit_mapping(train_logp, train_y, args.regularization,
                                  args.balanced)
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = {
        "dataset": dataset_name, "fit_split": "train",
        "regularization": args.regularization, "balanced": args.balanced,
        "fit": fit_meta, "W": W.tolist(), "b": b.tolist(),
        "matched_metrics": split_stats,
    }
    for split in ("train", "val", "test"):
        source = votes[split]
        out_arrays = {}
        for key, rows in source.items():
            out_rows = np.asarray(rows, np.float32).copy()
            if len(out_rows):
                p = np.maximum(out_rows[:, 5:5 + K], 0.)
                p /= np.maximum(p.sum(1, keepdims=True), 1e-9)
                out_rows[:, 5:5 + K] = softmax(
                    np.log(p + 1e-4) @ W.T + b, axis=1).astype(np.float32)
            out_arrays[key] = out_rows
        np.savez_compressed(args.output_root / f"fused_{split}__wbf_softvote.npz",
                            **out_arrays)
    (args.output_root / "metadata.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
