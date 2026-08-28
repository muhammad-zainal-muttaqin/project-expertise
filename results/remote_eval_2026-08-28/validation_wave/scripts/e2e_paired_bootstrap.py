"""Paired bootstrap for the already-locked end-to-end test summaries.

This is an analysis-only script.  It consumes the saved per-tree summaries
from the original count-reconciled baseline and the locked GSP result; it
does not load images, detector predictions, or fit/select any model.  The
same resampled tree indices are used for both systems, so the comparison is
paired at the tree level.

The legacy baseline stores ``matched`` as the physical true-positive count
per tree (its aggregate equals ``physical_detection.tp``).  The locked GSP
artifact stores the equivalent value as ``tp``.  The class-aware baseline
does not contain per-tree class-correct counts/confusion matrices, so this
script deliberately reports only metrics that can be reconstructed exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path("/workspace")
DEFAULTS = {
    "953": {
        "baseline": ROOT / "model_artifacts/project-expertise/eval_2026-08-27/count_reconciled_calibration_953_test_locked.json",
        "candidate": ROOT / "gsp_linker/artifacts/953/results_test_locked.json",
    },
    "depth": {
        "baseline": ROOT / "model_artifacts/project-expertise/eval_2026-08-27/count_reconciled_calibration_depth_test_locked.json",
        "candidate": ROOT / "gsp_linker/artifacts/depth/results_test_locked.json",
    },
}


def _load(path: Path, candidate: bool) -> tuple[dict, list[dict]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if candidate:
        metrics = data["metrics"]
        rows = data["per_tree"]
    else:
        metrics = data["results"][0]["metrics"]
        rows = metrics["per_tree"]
    return metrics, rows


def _validate(dataset: str, baseline: dict, candidate: dict,
              base_rows: list[dict], cand_rows: list[dict]) -> None:
    if len(base_rows) != len(cand_rows):
        raise RuntimeError(f"{dataset}: tree count mismatch")
    base_ids = [str(r["tree_id"]) for r in base_rows]
    cand_ids = [str(r["tree_id"]) for r in cand_rows]
    if base_ids != cand_ids:
        raise RuntimeError(f"{dataset}: baseline/candidate tree order or IDs differ")
    for b, c in zip(base_rows, cand_rows):
        if int(b["gt_count"]) != int(c["gt_bunches"]):
            raise RuntimeError(f"{dataset}: GT mismatch at {b['tree_id']}")
    btp = sum(int(r["matched"]) for r in base_rows)
    ctp = sum(int(r["tp"]) for r in cand_rows)
    if btp != int(baseline["physical_detection"]["tp"]):
        raise RuntimeError(f"{dataset}: baseline matched does not reproduce aggregate TP")
    if ctp != int(candidate["physical_detection"]["tp"]):
        raise RuntimeError(f"{dataset}: candidate TP does not reproduce aggregate TP")
    for r in base_rows:
        if int(r["count_delta"]) != int(r["pred_count"]) - int(r["gt_count"]):
            raise RuntimeError(f"{dataset}: baseline count delta mismatch at {r['tree_id']}")
    for r in cand_rows:
        if int(r["count_delta"]) != int(r["pred_clusters"]) - int(r["gt_bunches"]):
            raise RuntimeError(f"{dataset}: candidate count delta mismatch at {r['tree_id']}")


def _aggregate(rows: list[dict], idx: np.ndarray, candidate: bool) -> dict[str, float]:
    if candidate:
        tp = np.asarray([r["tp"] for r in rows], dtype=float)
        pred = np.asarray([r["pred_clusters"] for r in rows], dtype=float)
        gt = np.asarray([r["gt_bunches"] for r in rows], dtype=float)
    else:
        tp = np.asarray([r["matched"] for r in rows], dtype=float)
        pred = np.asarray([r["pred_count"] for r in rows], dtype=float)
        gt = np.asarray([r["gt_count"] for r in rows], dtype=float)
    delta = np.asarray([r["count_delta"] for r in rows], dtype=float)
    sampled_tp = float(tp[idx].sum())
    sampled_pred = float(pred[idx].sum())
    sampled_gt = float(gt[idx].sum())
    precision = sampled_tp / max(sampled_pred, 1.0)
    recall = sampled_tp / max(sampled_gt, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "physical_f1": float(f1),
        "mae": float(np.abs(delta[idx]).mean()),
        "exact_accuracy": float((delta[idx] == 0).mean()),
        "plus_minus_1_accuracy": float((np.abs(delta[idx]) <= 1).mean()),
    }


def _point(metrics: dict) -> dict[str, float]:
    return {
        "physical_f1": float(metrics["physical_detection"]["f1"]),
        "mae": float(metrics["counting"]["mae"]),
        "exact_accuracy": float(metrics["counting"]["exact_accuracy"]),
        "plus_minus_1_accuracy": float(metrics["counting"]["plus_minus_1_accuracy"]),
    }


def run_dataset(dataset: str, n_resamples: int, seed: int) -> dict:
    paths = DEFAULTS[dataset]
    base_metrics, base_rows = _load(paths["baseline"], candidate=False)
    cand_metrics, cand_rows = _load(paths["candidate"], candidate=True)
    _validate(dataset, base_metrics, cand_metrics, base_rows, cand_rows)

    n = len(base_rows)
    # Vectorized paired resampling.  Each row is a tree; both systems use the
    # identical index matrix, preserving within-tree correlation.
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n), dtype=np.int64)
    b = {k: np.empty(n_resamples, dtype=np.float64)
         for k in ("physical_f1", "mae", "exact_accuracy", "plus_minus_1_accuracy")}
    c = {k: np.empty(n_resamples, dtype=np.float64) for k in b}

    # Aggregate the ratio metrics directly for speed and exact equivalence to
    # the evaluator; averaging per-tree F1 would answer a different question.
    btp = np.asarray([r["matched"] for r in base_rows], dtype=np.float64)
    bpred = np.asarray([r["pred_count"] for r in base_rows], dtype=np.float64)
    bgt = np.asarray([r["gt_count"] for r in base_rows], dtype=np.float64)
    ctp = np.asarray([r["tp"] for r in cand_rows], dtype=np.float64)
    cpred = np.asarray([r["pred_clusters"] for r in cand_rows], dtype=np.float64)
    cgt = np.asarray([r["gt_bunches"] for r in cand_rows], dtype=np.float64)
    bdelta = np.asarray([r["count_delta"] for r in base_rows], dtype=np.float64)
    cdelta = np.asarray([r["count_delta"] for r in cand_rows], dtype=np.float64)

    def f1(tp_sum, pred_sum, gt_sum):
        p = tp_sum / np.maximum(pred_sum, 1.0)
        r = tp_sum / np.maximum(gt_sum, 1.0)
        return 2.0 * p * r / np.maximum(p + r, 1e-12)

    b["physical_f1"] = f1(btp[indices].sum(axis=1), bpred[indices].sum(axis=1),
                            bgt[indices].sum(axis=1))
    c["physical_f1"] = f1(ctp[indices].sum(axis=1), cpred[indices].sum(axis=1),
                            cgt[indices].sum(axis=1))
    b["mae"] = np.abs(bdelta[indices]).mean(axis=1)
    c["mae"] = np.abs(cdelta[indices]).mean(axis=1)
    b["exact_accuracy"] = (bdelta[indices] == 0).mean(axis=1)
    c["exact_accuracy"] = (cdelta[indices] == 0).mean(axis=1)
    b["plus_minus_1_accuracy"] = (np.abs(bdelta[indices]) <= 1).mean(axis=1)
    c["plus_minus_1_accuracy"] = (np.abs(cdelta[indices]) <= 1).mean(axis=1)

    base_point = _point(base_metrics)
    cand_point = _point(cand_metrics)
    rows = {}
    for metric in b:
        # Positive means improvement: higher is better for F1/accuracy, while
        # lower is better for MAE.
        sign = -1.0 if metric == "mae" else 1.0
        diff = sign * (c[metric] - b[metric])
        point = sign * (cand_point[metric] - base_point[metric])
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rows[metric] = {
            "baseline": base_point[metric],
            "candidate": cand_point[metric],
            "improvement_positive": point,
            "paired_bootstrap_ci95": [float(lo), float(hi)],
            "p_improvement_positive": float(np.mean(diff > 0.0)),
            "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
        }
    return {
        "dataset": dataset,
        "n_trees": n,
        "n_resamples": n_resamples,
        "seed": seed,
        "protocol": "paired tree bootstrap over existing locked per-tree summaries; no image/model inference; no selection",
        "baseline_artifact": str(paths["baseline"]),
        "candidate_artifact": str(paths["candidate"]),
        "tree_alignment": "exact ID and order match; GT counts match per tree",
        "metrics": rows,
        "class_aware_scope": "not computed: legacy baseline lacks per-tree class-correct/confusion decomposition",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=tuple(DEFAULTS), default=list(DEFAULTS))
    ap.add_argument("--resamples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "ci_boot/artifacts/e2e_paired_test.json")
    args = ap.parse_args()
    result = {
        "generated_at": "2026-08-28",
        "analysis": "paired end-to-end test bootstrap",
        "datasets": {ds: run_dataset(ds, args.resamples, args.seed + i)
                     for i, ds in enumerate(args.datasets)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
