#!/usr/bin/env python3
"""Fuse independent timm opinions into the frozen best 953 VAL stack.

The physical linker and count layer are immutable.  The existing DINOv2
stack and B2 calibration are the fixed starting point; this experiment adds
small declared weights for timm opinions and searches only TRAIN-derived
models on VAL.  TEST is not accepted or read.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np

import harness
import large_member_head as large
import member_head as mh
import multiscale_member_head as ms
import timm_member_head as tm


OUT = Path("/workspace/cluster_head/artifacts")
DATASET = "953"


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


def normalized_detector(data: dict) -> np.ndarray:
    detector = np.asarray([np.asarray(group["p"], dtype=np.float32)
                           for _record, groups in data["groups"] for group in groups])
    detector = np.maximum(detector, 1e-8)
    return detector / np.maximum(detector.sum(axis=1, keepdims=True), 1e-8)


def feature_data_from(base_data: dict, fmap: dict, feature_dim: int,
                      feature_fn) -> dict:
    """Build a feature view over one already-materialized VAL topology."""
    rows, features, keys = [], [], []
    for _record, groups in base_data["groups"]:
        for group in groups:
            group_rows = []
            for member in group["members"]:
                group_rows.append(len(features))
                features.append(feature_fn(member, fmap, feature_dim))
            rows.append(group_rows)
            keys.append(mh.harness_group_key(group))
    if keys != base_data["keys"]:
        raise RuntimeError("feature view changed frozen group order")
    return {"X": np.asarray(features, dtype=np.float32),
            "group_rows": rows, "keys": keys, "groups": base_data["groups"]}


def multiscale_feature_data(base_data: dict) -> dict:
    base_map, _ = mh._load_fmap(DATASET, "val")
    maps = {tag: ms.load_map(DATASET, "val", tag) for tag in ms.SCALES}
    rows, features, keys = [], [], []
    for _record, groups in base_data["groups"]:
        for group in groups:
            group_rows = []
            for member in group["members"]:
                group_rows.append(len(features))
                features.append(ms.make_member_feature(member, base_map, maps))
            rows.append(group_rows)
            keys.append(mh.harness_group_key(group))
    if keys != base_data["keys"]:
        raise RuntimeError("multiscale feature view changed frozen group order")
    return {"X": np.asarray(features, dtype=np.float32),
            "group_rows": rows, "keys": keys, "groups": base_data["groups"]}


def static_class_evaluator(data: dict, fixed: dict):
    """Precompute tree matches so the fusion grid never reruns geometry IoU."""
    gt_by_group = []
    unmatched_gt = []
    for record, groups in data["groups"]:
        matches = dict(harness.count.tree_matches(record, groups))
        matched_gt_indices = set(matches.values())
        for group_index in range(len(groups)):
            if group_index in matches:
                gt_by_group.append(int(record["bunches"][matches[group_index]]["cls"]))
            else:
                gt_by_group.append(-1)
        unmatched_gt.extend(
            int(record["bunches"][index]["cls"])
            for index in range(len(record["bunches"]))
            if index not in matched_gt_indices
            and 0 <= int(record["bunches"][index]["cls"]) < harness.K
        )
    gt = np.asarray(gt_by_group, dtype=np.int64)
    matched = gt >= 0
    unmatched_gt = np.asarray(unmatched_gt, dtype=np.int64)
    if len(gt) != len(data["keys"]):
        raise RuntimeError("static evaluator group count mismatch")

    def evaluate_prediction(prediction: np.ndarray) -> dict:
        cm = np.zeros((harness.K + 1, harness.K + 1), dtype=np.int64)
        np.add.at(cm, (prediction[matched], gt[matched]), 1)
        np.add.at(cm, (prediction[~matched],
                       np.full(int((~matched).sum()), harness.K)), 1)
        if len(unmatched_gt):
            np.add.at(cm, (np.full(len(unmatched_gt), harness.K), unmatched_gt), 1)
        f1s = []
        for cls in range(harness.K):
            tp = cm[cls, cls]
            fp = cm[cls, :].sum() - tp
            fn = cm[:, cls].sum() - tp
            f1s.append(float(2 * tp / max(2 * tp + fp + fn, 1)))
        return {
            "physical_f1": fixed["physical_f1"], "mae": fixed["mae"],
            "pm1": fixed["pm1"],
            "matched_class_accuracy": float(
                (prediction[matched] == gt[matched]).sum()
                / max(int(matched.sum()), 1)),
            "matched": int(matched.sum()), "macro_f1": float(np.mean(f1s)),
            "per_class_f1": dict(zip(harness.NAMES, f1s)),
            "confusion": cm.tolist(),
        }
    return evaluate_prediction


def opinion(model, data: dict, pooling: str) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(data["X"]), dtype=np.float32)
    return tm.pool(probabilities, data, pooling)


def invariant(metrics: dict, baseline: dict) -> bool:
    return bool(
        abs(metrics["physical_f1"] - baseline["physical_f1"]) < 1e-10
        and abs(metrics["mae"] - baseline["mae"]) < 1e-10
        and abs(metrics["pm1"] - baseline["pm1"]) < 1e-10)


def rank_key(row: dict):
    return (row["metrics"]["matched_class_accuracy"], row["metrics"]["macro_f1"])


def run(seed: int) -> dict:
    started = time.time()
    data = mh.collect(DATASET, "val")
    baseline = short(harness.evaluate_clusters(
        data["payload"], data["targets"], harness.PROFILES[DATASET]))
    evaluate_prediction = static_class_evaluator(data, baseline)
    large_fmap, large_dim = large.load_fmap(DATASET, "val")
    large_data = feature_data_from(data, large_fmap, large_dim, large.member_feature)
    multi_data = multiscale_feature_data(data)
    detector = normalized_detector(data)

    # This is the already-selected VAL candidate, held fixed as the anchor
    # for the independent-backbone ablation.
    base_model = joblib.load(OUT / "953_member_logistic.joblib")
    large_model = joblib.load(OUT / "953_large_hist.joblib")
    multi_model = joblib.load(OUT / "953_ms_extra.joblib")
    q_base = opinion(base_model, data, "max")
    q_large = opinion(large_model, large_data, "mean")
    q_multi = opinion(multi_model, multi_data, "mean")
    anchor_weights = {"large": .15, "base_logistic": .05, "multiscale": 0.}
    anchor_bias = np.asarray([0., .15, 0., 0.], dtype=np.float32)
    anchor_logits = (np.log(detector)
                     + anchor_weights["large"] * np.log(q_large)
                     + anchor_weights["base_logistic"] * np.log(q_base)
                     + anchor_weights["multiscale"] * np.log(q_multi)
                     + anchor_bias)
    anchor = evaluate_prediction(np.argmax(anchor_logits, axis=1).astype(int))

    # Keep all pooling/model opinions available for complementarity analysis,
    # then limit the search to a small deterministic coordinate grid.
    opinions = {}
    for backbone in tm.MODELS:
        timm_fmap, timm_dim = tm.load_fmap(DATASET, "val", backbone)
        tm_data = feature_data_from(data, timm_fmap, timm_dim, tm.member_feature)
        for head in ("logistic", "extra", "hist"):
            model = joblib.load(OUT / f"953_timm_{backbone}_{head}.joblib")
            for pooling in ("mean", "max", "top"):
                opinions[f"{backbone}:{head}:{pooling}"] = opinion(model, tm_data, pooling)

    rows = []
    def add(stage: str, weights: dict[str, float], bias_b2: float = .15):
        logits = anchor_logits - anchor_bias + np.asarray([0., bias_b2, 0., 0.], dtype=np.float32)
        for name, weight in weights.items():
            logits = logits + weight * np.log(np.maximum(opinions[name], 1e-8))
        metrics = evaluate_prediction(np.argmax(logits, axis=1).astype(int))
        rows.append({"stage": stage, "weights": weights.copy(),
                     "bias_b2": bias_b2, "metrics": metrics,
                     "physical_count_invariant": invariant(metrics, baseline)})

    # One-opinion scan identifies useful error diversity before pair/triple
    # search; zero is always included so an independent head cannot hurt by
    # construction of the selected anchor.
    scan_values = (0., .01, .02, .03, .05, .08, .12, .16, .20)
    for name in sorted(opinions):
        for weight in scan_values:
            add("single", {name: weight})

    single_rows = [row for row in rows if row["stage"] == "single"]
    useful = sorted(
        [row for row in single_rows
         if any(float(weight) > 0. for weight in row["weights"].values())],
        key=rank_key, reverse=True)
    # At most one member of each backbone enters the interaction search to
    # keep the experiment low-variance and avoid a combinatorial VAL hunt.
    selected_names = []
    for row in useful:
        name = next(iter(row["weights"]))
        backbone = name.split(":", 1)[0]
        if backbone not in {x.split(":", 1)[0] for x in selected_names}:
            selected_names.append(name)
        if len(selected_names) == 3:
            break

    pair_values = (0., .02, .05, .08, .12)
    for pair in combinations(selected_names, 2):
        for wa in pair_values:
            for wb in pair_values:
                add("pair", {pair[0]: wa, pair[1]: wb})

    triple_values = (0., .02, .04, .06, .08, .12)
    if len(selected_names) == 3:
        for wa in triple_values:
            for wb in triple_values:
                for wc in triple_values:
                    add("triple", {selected_names[0]: wa,
                                    selected_names[1]: wb,
                                    selected_names[2]: wc})

    # A short B2 sensitivity scan on the best fusion prevents the new
    # opinion from being judged against an accidentally over-specific bias.
    candidates = [row for row in rows if row["physical_count_invariant"]]
    best_before_bias = max(candidates, key=rank_key)
    best_weights = best_before_bias["weights"]
    for b2 in (0., .05, .10, .15, .20, .25):
        add("best_bias", best_weights, b2)

    selected = max([row for row in rows if row["physical_count_invariant"]],
                   key=rank_key)
    best_macro = max([row for row in rows if row["physical_count_invariant"]],
                     key=lambda row: (row["metrics"]["macro_f1"],
                                      row["metrics"]["matched_class_accuracy"]))
    report = {
        "dataset": DATASET,
        "protocol": "fixed TRAIN-fitted heads; anchor and timm fusion selected VAL; no TEST",
        "seed": seed,
        "anchor": {"weights": anchor_weights, "bias": anchor_bias.tolist(),
                   "metrics": anchor},
        "baseline_val": baseline,
        "opinion_count": len(opinions),
        "selected_interaction_opinions": selected_names,
        "search_sizes": {"single": len(single_rows),
                          "pair": sum(row["stage"] == "pair" for row in rows),
                          "triple": sum(row["stage"] == "triple" for row in rows),
                          "best_bias": sum(row["stage"] == "best_bias" for row in rows)},
        "selected_validation": selected,
        "best_by_macro": best_macro,
        "all_rows": rows,
        "elapsed_sec": time.time() - started,
    }
    path = OUT / "953_timm_stack_fusion_results_val.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(json.dumps({"anchor": report["anchor"],
                      "selected_validation": selected,
                      "best_by_macro": best_macro,
                      "report": str(path)}, ensure_ascii=False), flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    run(args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
