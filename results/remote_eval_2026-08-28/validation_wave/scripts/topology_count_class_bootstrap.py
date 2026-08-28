#!/usr/bin/env python3
"""Paired tree bootstrap for the best cross-layer Depth VAL composition.

Candidate: original GSP topology + TRAIN-fitted v2 geo count targets + the
predeclared class ``scale_macro`` calibration.  The baseline is the original
GSP topology/count/class profile.  No TEST split is accepted or read.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, "/workspace/cluster_head")
sys.path.insert(0, "/workspace/pipeline_v2")

import class_bias_general as calibration  # noqa: E402
import harness  # noqa: E402
import pipeline_v2 as v2  # noqa: E402
import topology_count_class_combo as combo  # noqa: E402


OUT = Path("/workspace/cluster_head/artifacts")
K = harness.K
RESAMPLES = 5000
SEED = 20260828


def tree_stats(rec: dict, groups: list[dict]) -> dict:
    cm = np.zeros((K + 1, K + 1), dtype=np.int64)
    matches = harness.count.tree_matches(rec, groups)
    matched_pred = {i for i, _ in matches}
    matched_gt = {j for _, j in matches}
    for i, j in matches:
        pred, gt = int(groups[i]["cls"]), int(rec["bunches"][j]["cls"])
        if 0 <= pred < K and 0 <= gt < K:
            cm[pred, gt] += 1
    for i, group in enumerate(groups):
        pred = int(group["cls"])
        if i not in matched_pred and 0 <= pred < K:
            cm[pred, K] += 1
    for j, bunch in enumerate(rec["bunches"]):
        gt = int(bunch["cls"])
        if j not in matched_gt and 0 <= gt < K:
            cm[K, gt] += 1
    delta = len(groups) - len(rec["bunches"])
    return {"cm": cm, "pred": len(groups), "gt": len(rec["bunches"]),
            "tp": len(matches), "abs_count": abs(delta),
            "exact": int(delta == 0), "pm1": int(abs(delta) <= 1)}


def aggregate(stats: list[dict], indices: np.ndarray | None = None) -> dict:
    chosen = stats if indices is None else [stats[int(i)] for i in indices]
    cm = np.sum([row["cm"] for row in chosen], axis=0)
    pred = sum(row["pred"] for row in chosen)
    gt = sum(row["gt"] for row in chosen)
    tp = sum(row["tp"] for row in chosen)
    precision = tp / max(pred, 1)
    recall = tp / max(gt, 1)
    f1 = 2. * precision * recall / max(precision + recall, 1e-12)
    f1s = []
    for c in range(K):
        true_pos = int(cm[c, c])
        fp = int(cm[c].sum() - true_pos)
        fn = int(cm[:, c].sum() - true_pos)
        f1s.append(2. * true_pos / max(2 * true_pos + fp + fn, 1))
    matched = int(cm[:K, :K].sum())
    return {
        "physical_f1": float(f1),
        "mae": float(np.mean([row["abs_count"] for row in chosen])),
        "exact": float(np.mean([row["exact"] for row in chosen])),
        "pm1": float(np.mean([row["pm1"] for row in chosen])),
        "matched_class_accuracy": float(np.trace(cm[:K, :K]) / max(matched, 1)),
        "matched": matched,
        "macro_f1": float(np.mean(f1s)),
        "per_class_f1": dict(zip(harness.NAMES, f1s)),
        "pred_clusters": int(pred),
        "gt_bunches": int(gt),
    }


def main() -> int:
    dataset = "depth"
    base_profile = dict(harness.PROFILES[dataset])

    records, base_payload, base_targets, _ = harness.build_payload(dataset, "val")
    cfg = v2.edge.cfg_for(dataset)
    v2_records = v2.count.four_side(v2.base.load_records(cfg, "val"))
    prior = v2.base.build_rotation_prior(v2.base.load_records(cfg, "train"))
    vote = v2.edge.load_vote(combo.DEPTH_V2 / "vote_v2_geo_val.npz")
    edge_model = joblib.load(combo.DEPTH_V2 / "edge_v2_geo.joblib")
    per_tree = v2.build_dets_and_candidates(v2_records, vote, prior, edge_model)
    v2_payload, solver_tags = v2.payload_for_tau(v2_records, per_tree, .25, 3)
    x_val, _y_val, ids = v2.build_count_features(v2_records, vote, per_tree)
    ridge = joblib.load(combo.DEPTH_V2 / "count_ridge_geo.joblib")
    v2_targets = {tree_id: int(n) for tree_id, n in
                  zip(ids, v2.count.predict_count(ridge, x_val))}
    if list(records) != list(v2_records) or list(base_targets) != list(v2_targets):
        raise RuntimeError("VAL tree ordering mismatch")

    baseline_groups = combo.build_selected(base_payload, base_targets, base_profile)
    candidate_groups = combo.build_selected(base_payload, v2_targets, base_profile)
    combo.calibrated_classes(dataset, baseline_groups, "detector")
    combo.calibrated_classes(dataset, candidate_groups, "scale_macro")
    baseline_stats = [tree_stats(rec, groups)
                      for rec, groups in baseline_groups]
    candidate_stats = [tree_stats(rec, groups)
                       for rec, groups in candidate_groups]
    baseline_point = aggregate(baseline_stats)
    candidate_point = aggregate(candidate_stats)
    if len(baseline_stats) != len(candidate_stats):
        raise RuntimeError("paired tree counts differ")

    rng = np.random.RandomState(SEED)
    delta = {key: np.empty(RESAMPLES, dtype=np.float64)
             for key in ("physical_f1", "mae", "exact", "pm1",
                         "matched_class_accuracy", "macro_f1")}
    for i in range(RESAMPLES):
        sample = rng.randint(0, len(baseline_stats), size=len(baseline_stats))
        b = aggregate(baseline_stats, sample)
        c = aggregate(candidate_stats, sample)
        for key in delta:
            delta[key][i] = c[key] - b[key]
    bootstrap = {}
    for key, values in delta.items():
        bootstrap[key] = {
            "mean_delta": float(values.mean()),
            "ci95": [float(x) for x in np.percentile(values, [2.5, 97.5])],
            "p_positive": float(np.mean(values > 0.)),
        }

    output = {
        "dataset": dataset,
        "protocol": "paired tree bootstrap; fit TRAIN, select VAL; no TEST",
        "seed": SEED, "resamples": RESAMPLES,
        "candidate": {
            "topology": "original_gsp",
            "count_targets": "v2_geo_ridge_fit_train",
            "class_calibration": "depth_member_stack_macro_scale_macro",
            "v2_solver_tags": solver_tags,
        },
        "tree_count": len(baseline_stats),
        "baseline_point": baseline_point,
        "candidate_point": candidate_point,
        "delta_point": {key: candidate_point[key] - baseline_point[key]
                        for key in ("physical_f1", "mae", "exact", "pm1",
                                    "matched_class_accuracy", "macro_f1")},
        "bootstrap": bootstrap,
    }
    path = OUT / "depth_topology_count_class_bootstrap_val.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"baseline_point": baseline_point,
                      "candidate_point": candidate_point,
                      "delta_point": output["delta_point"],
                      "bootstrap": bootstrap, "report": str(path)},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
