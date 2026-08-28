#!/usr/bin/env python3
"""Validation-only class-logit calibration for both frozen pipelines.

The detector, linker, selected groups, and count targets are fixed.  This
script fits no model and reads no TEST split; it evaluates a declared bias
grid on the probabilities of TRAIN-fitted class experts.  The purpose is to
find whether the strong B2 calibration used on 953 also has a useful,
generalizable analogue for the Depth class profile, especially its weak B4
recall.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np

import harness
import large_member_head as large
import member_head as mh
import multiscale_member_head as ms


OUT = Path("/workspace/cluster_head/artifacts")
K = harness.K


def short(metrics: dict) -> dict:
    return {
        "physical_f1": metrics["physical_detection"]["f1"],
        "mae": metrics["counting"]["mae"],
        "pm1": metrics["counting"]["plus_minus_1_accuracy"],
        "matched_class_accuracy": metrics["classification"]["matched_class_accuracy"],
        "matched": metrics["classification"]["matched"],
        "macro_f1": metrics["classification"]["macro_f1_end_to_end"],
        "per_class_f1": metrics["classification"]["per_class_f1_end_to_end"],
    }


def pool(member_probs: np.ndarray, data: dict, pooling: str) -> np.ndarray:
    flat = [group for _record, groups in data["groups"] for group in groups]
    output = []
    for gi, rows in enumerate(data["group_rows"]):
        q = member_probs[rows]
        if pooling == "mean":
            w = np.asarray([float(m["score"]) for m in flat[gi]["members"]],
                           dtype=np.float32)
            w /= max(float(w.sum()), 1e-8)
            output.append((q * w[:, None]).sum(axis=0))
        elif pooling == "max":
            output.append(q.max(axis=0))
        elif pooling == "top":
            j = int(np.argmax([float(m["score"]) for m in flat[gi]["members"]]))
            output.append(q[j])
        else:
            raise ValueError(pooling)
    output = np.maximum(np.asarray(output, dtype=np.float32), 1e-8)
    return output / np.maximum(output.sum(axis=1, keepdims=True), 1e-8)


def feature_view(base_data: dict, fmap: dict, dim: int, feature_fn) -> dict:
    rows, features, keys = [], [], []
    for _record, groups in base_data["groups"]:
        for group in groups:
            current = []
            for member in group["members"]:
                current.append(len(features))
                features.append(feature_fn(member, fmap, dim))
            rows.append(current)
            keys.append(mh.harness_group_key(group))
    if keys != base_data["keys"]:
        raise RuntimeError("expert feature view changed frozen group order")
    return {"X": np.asarray(features, dtype=np.float32),
            "group_rows": rows, "groups": base_data["groups"], "keys": keys}


def static_evaluator(data: dict, fixed: dict):
    gt_by_group = []
    unmatched_gt = []
    for record, groups in data["groups"]:
        matches = dict(harness.count.tree_matches(record, groups))
        matched_gt = set(matches.values())
        for index in range(len(groups)):
            gt_by_group.append(
                int(record["bunches"][matches[index]]["cls"])
                if index in matches else -1
            )
        unmatched_gt.extend(
            int(record["bunches"][index]["cls"])
            for index in range(len(record["bunches"]))
            if index not in matched_gt and 0 <= int(record["bunches"][index]["cls"]) < K
        )
    gt = np.asarray(gt_by_group, dtype=np.int64)
    matched = gt >= 0
    unmatched_gt = np.asarray(unmatched_gt, dtype=np.int64)
    if len(gt) != len(data["keys"]):
        raise RuntimeError("group/evaluator length mismatch")

    def evaluate(prediction: np.ndarray) -> dict:
        prediction = np.asarray(prediction, dtype=np.int64)
        if prediction.shape != gt.shape:
            raise RuntimeError("prediction/evaluator length mismatch")
        cm = np.zeros((K + 1, K + 1), dtype=np.int64)
        np.add.at(cm, (prediction[matched], gt[matched]), 1)
        np.add.at(cm, (prediction[~matched],
                       np.full(int((~matched).sum()), K)), 1)
        if len(unmatched_gt):
            np.add.at(cm, (np.full(len(unmatched_gt), K), unmatched_gt), 1)
        f1s = []
        for cls in range(K):
            tp = cm[cls, cls]
            fp = cm[cls].sum() - tp
            fn = cm[:, cls].sum() - tp
            f1s.append(float(2 * tp / max(2 * tp + fp + fn, 1)))
        return {
            **fixed,
            "matched_class_accuracy": float(
                (prediction[matched] == gt[matched]).sum()
                / max(int(matched.sum()), 1)
            ),
            "matched": int(matched.sum()),
            "macro_f1": float(np.mean(f1s)),
            "per_class_f1": dict(zip(harness.NAMES, f1s)),
            "confusion": cm.tolist(),
        }

    return evaluate


def build_views(dataset: str, data: dict) -> dict[str, np.ndarray]:
    views = {}
    base_model = joblib.load(OUT / f"{dataset}_member_logistic.joblib")
    views["base_logistic_max"] = pool(
        np.asarray(base_model.predict_proba(data["X"]), dtype=np.float32), data, "max"
    )
    extra_model = joblib.load(OUT / f"{dataset}_member_extra.joblib")
    views["base_extra_max"] = pool(
        np.asarray(extra_model.predict_proba(data["X"]), dtype=np.float32), data, "max"
    )
    # Independent pretrained views are cheap to reuse once their feature
    # matrices exist.  Keep them as optional opinions so the same calibration
    # harness can test both datasets without changing the frozen topology.
    large_data = feature_view(
        data, *large.load_fmap(dataset, "val"), large.member_feature
    )
    for head in ("large_hist", "large_logistic", "large_extra"):
        model_path = OUT / f"{dataset}_{head}.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
            views[f"{head}_mean"] = pool(
                np.asarray(model.predict_proba(large_data["X"]), dtype=np.float32),
                large_data, "mean"
            )
    multi_base, _ = mh._load_fmap(dataset, "val")
    multi_maps = {tag: ms.load_map(dataset, "val", tag) for tag in ms.SCALES}
    multi_data = feature_view(
        data, multi_base, 1536,
        lambda member, fmap, _dim: ms.make_member_feature(member, fmap, multi_maps)
    )
    for head in ("ms_extra", "ms_logistic"):
        model_path = OUT / f"{dataset}_{head}.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
            views[f"multiscale_{head.removeprefix('ms_')}_mean"] = pool(
                np.asarray(model.predict_proba(multi_data["X"]), dtype=np.float32),
                multi_data, "mean"
            )
    return views


def top_rows(rows: list[dict], n: int = 12) -> list[dict]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row["metrics"]["matched_class_accuracy"],
            row["metrics"]["macro_f1"],
        ),
        reverse=True,
    )
    return ordered[:n]


def consider_top(bucket: list[dict], row: dict, n: int = 12) -> None:
    """Keep a small audit trail without materializing every scale row."""
    bucket.append(row)
    bucket.sort(
        key=lambda item: (
            item["metrics"]["matched_class_accuracy"],
            item["metrics"]["macro_f1"],
        ),
        reverse=True,
    )
    del bucket[n:]


def consider_macro(bucket: list[dict], row: dict, n: int = 12) -> None:
    bucket.append(row)
    bucket.sort(
        key=lambda item: (
            item["metrics"]["macro_f1"],
            item["metrics"]["matched_class_accuracy"],
        ),
        reverse=True,
    )
    del bucket[n:]


def run(dataset: str, seed: int) -> dict:
    del seed  # kept in the report/CLI for consistent experiment identifiers
    started = time.time()
    data = mh.collect(dataset, "val")
    base_metrics = short(harness.evaluate_clusters(
        data["payload"], data["targets"], harness.PROFILES[dataset]
    ))
    fixed = {key: base_metrics[key] for key in ("physical_f1", "mae", "pm1")}
    evaluate = static_evaluator(data, fixed)
    detector = np.asarray(
        [np.asarray(group["p"], dtype=np.float32)
         for _record, groups in data["groups"] for group in groups]
    )
    detector = np.maximum(detector, 1e-8)
    detector /= np.maximum(detector.sum(axis=1, keepdims=True), 1e-8)
    views = build_views(dataset, data)

    if dataset == "953":
        specs = {
            "robust_953_anchor": {
                "weights": {"large_hist_mean": .15, "base_logistic_max": .05,
                            "multiscale_extra_mean": 0.},
                "bias": [0., .15, 0., 0.],
            },
            "member_stack": {
                "weights": {"base_extra_max": .45, "base_logistic_max": .10},
                "bias": [0., 0., 0., 0.],
            },
        }
    else:
        specs = {
            "member_stack_macro": {
                "weights": {"base_extra_max": .10, "base_logistic_max": .15},
                "bias": [0., 0., 0., 0.],
            },
            "member_stack_matched": {
                "weights": {"base_extra_max": .30, "base_logistic_max": .30},
                "bias": [0., 0., 0., 0.],
            },
            "multiscale_macro": {
                "weights": {"base_logistic_max": .20,
                            "multiscale_logistic_mean": .15},
                "bias": [0., 0., 0., 0.],
            },
            "large_multiscale_skip": {
                "weights": {"base_extra_max": .10, "base_logistic_max": .15,
                            "large_logistic_mean": .05,
                            "multiscale_logistic_mean": .05},
                "bias": [0., 0., 0., 0.],
            },
        }

    values = (-.30, -.20, -.15, -.10, -.05, 0., .05, .10, .15, .20, .30)
    report_specs = {}
    for spec_name, spec in specs.items():
        raw_logits = np.log(detector)
        for view_name, weight in spec["weights"].items():
            raw_logits = raw_logits + float(weight) * np.log(
                np.maximum(views[view_name], 1e-8)
            )
        logits = raw_logits + np.asarray(spec["bias"], dtype=np.float32)
        anchor = evaluate(np.argmax(logits, axis=1))
        rows = []
        for b2, b3, b4 in product(values, repeat=3):
            bias = np.asarray([0., b2, b3, b4], dtype=np.float32)
            metrics = evaluate(np.argmax(raw_logits + bias, axis=1))
            rows.append({"bias": bias.tolist(), "metrics": metrics})
        scale_values = (0.85, 1.0, 1.15)
        scale_top_match: list[dict] = []
        scale_top_macro: list[dict] = []
        # Temperature is applied to the fused log-opinion logits before the
        # class bias.  This tests calibration, not a new detector/linker.
        scale_bias_values = (-.15, -.10, -.05, 0., .05, .10, .15)
        scale_grid_size = len(scale_values) ** K * len(scale_bias_values) ** 3
        for scales in product(scale_values, repeat=K):
            scaled = raw_logits * np.asarray(scales, dtype=np.float32)
            for b2, b3, b4 in product(scale_bias_values, repeat=3):
                bias = np.asarray([0., b2, b3, b4], dtype=np.float32)
                scale_row = {
                    "scales": list(scales),
                    "bias": bias.tolist(),
                    "metrics": evaluate(np.argmax(scaled + bias, axis=1)),
                }
                consider_top(scale_top_match, scale_row)
                consider_macro(scale_top_macro, scale_row)
        report_specs[spec_name] = {
            "weights": spec["weights"],
            "anchor_bias": spec["bias"],
            "anchor": anchor,
            "grid_values": values,
            "grid_size": len(rows),
            "best_by_matched": top_rows(rows, 1)[0],
            "best_by_macro": max(
                rows,
                key=lambda row: (row["metrics"]["macro_f1"],
                                 row["metrics"]["matched_class_accuracy"]),
            ),
            "top_by_matched": top_rows(rows),
            "top_by_macro": sorted(
                rows,
                key=lambda row: (row["metrics"]["macro_f1"],
                                 row["metrics"]["matched_class_accuracy"]),
                reverse=True,
            )[:12],
            "scale_grid": {
                "scale_values": scale_values,
                "bias_values": scale_bias_values,
                "grid_size": scale_grid_size,
                "best_by_matched": scale_top_match[0],
                "best_by_macro": scale_top_macro[0],
                "top_by_matched": scale_top_match,
                "top_by_macro": scale_top_macro,
            },
        }

    output = {
        "dataset": dataset,
        "protocol": "fixed TRAIN-fitted class heads; bias grid selected VAL; no TEST",
        "seed": 20260828,
        "baseline_val": base_metrics,
        "groups": int(len(data["keys"])),
        "views": sorted(views),
        "specs": report_specs,
        "elapsed_sec": time.time() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{dataset}_class_bias_general_results_val.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(json.dumps({
        "dataset": dataset,
        "baseline": base_metrics,
        "specs": {
            name: {
                "best_matched": value["best_by_matched"],
                "best_macro": value["best_by_macro"],
            }
            for name, value in report_specs.items()
        },
        "report": str(path),
    }, ensure_ascii=False), flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("953", "depth"), required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    run(args.dataset, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
