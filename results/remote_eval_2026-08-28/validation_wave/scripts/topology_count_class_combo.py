#!/usr/bin/env python3
"""VAL-only cross-layer composition for the Depth pipeline.

The preceding experiments optimized the linker/topology and class head in
separate layers.  This script evaluates their Cartesian product on VAL:

* original GSP topology or the v2 geo GSP topology;
* original count targets or the v2 geo count targets;
* detector class assignment or one of two already selected class calibrations.

This is a composition audit, not a new test-set search.  All learned pieces
are fit on TRAIN and every reported choice is evaluated on VAL only.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, "/workspace/cluster_head")
sys.path.insert(0, "/workspace/pipeline_v2")

import class_bias_general as calibration  # noqa: E402
import harness  # noqa: E402
import head_aware_selection as evaluator  # noqa: E402
import member_head as mh  # noqa: E402
import pipeline_v2 as v2  # noqa: E402


OUT = Path("/workspace/cluster_head/artifacts")
DEPTH_V2 = Path("/workspace/pipeline_v2/artifacts/depth")


def build_selected(payload: list[tuple], targets: dict[str, int], profile: dict):
    return [(rec, [copy.deepcopy(g) for g in groups])
            for rec, groups in harness.make_groups(payload, targets, profile)]


def view_data(dataset: str, grouped: list[tuple]) -> dict:
    fmap, dim = mh._load_fmap(dataset, "val")
    all_x, rows, keys = [], [], []
    for _rec, groups in grouped:
        for group in groups:
            current = []
            for member in group["members"]:
                current.append(len(all_x))
                all_x.append(mh.member_feature(member, fmap, dim))
            rows.append(current)
            keys.append(mh.harness_group_key(group))
    return {"groups": grouped, "X": np.asarray(all_x, dtype=np.float32),
            "group_rows": rows, "keys": keys}


def calibrated_classes(dataset: str, grouped: list[tuple], mode: str) -> None:
    data = view_data(dataset, grouped)
    detector = np.asarray([np.asarray(g["p"], dtype=np.float32)
                           for _rec, groups in grouped for g in groups])
    detector = np.maximum(detector, 1e-8)
    detector /= np.maximum(detector.sum(axis=1, keepdims=True), 1e-8)
    if mode == "detector":
        logits = np.log(detector)
    else:
        report = json.loads((calibration.OUT /
                             f"{dataset}_class_bias_general_results_val.json").read_text())
        spec_name = "member_stack_macro"
        spec = report["specs"][spec_name]
        chosen_name = "best_by_matched" if mode == "scale_matched" else "best_by_macro"
        chosen = spec["scale_grid"][chosen_name]
        logits = np.log(detector)
        views = calibration.build_views(dataset, data)
        for view_name, weight in spec["weights"].items():
            logits += float(weight) * np.log(np.maximum(views[view_name], 1e-8))
        logits = logits * np.asarray(chosen["scales"], dtype=np.float32)
        logits += np.asarray(chosen["bias"], dtype=np.float32)
    offset = 0
    for _rec, groups in grouped:
        for group in groups:
            group["cls"] = int(np.argmax(logits[offset]))
            offset += 1


def short(m: dict) -> dict:
    return evaluator.short(m)


def run(seed: int) -> dict:
    del seed
    started = time.time()
    dataset = "depth"
    base_profile = dict(harness.PROFILES[dataset])
    v2_profile = {"link_threshold": .5, "singleton_min": .15,
                  "max_size": 3, "rank_mode": "score"}

    records, base_payload, base_targets, _ = harness.build_payload(dataset, "val")

    cfg = v2.edge.cfg_for(dataset)
    v2_records = v2.count.four_side(v2.base.load_records(cfg, "val"))
    prior = v2.base.build_rotation_prior(v2.base.load_records(cfg, "train"))
    vote = v2.edge.load_vote(DEPTH_V2 / "vote_v2_geo_val.npz")
    edge_model = joblib.load(DEPTH_V2 / "edge_v2_geo.joblib")
    per_tree = v2.build_dets_and_candidates(v2_records, vote, prior, edge_model)
    v2_payload, solver_tags = v2.payload_for_tau(v2_records, per_tree, .25, 3)
    x_val, _y_val, ids = v2.build_count_features(v2_records, vote, per_tree)
    ridge = joblib.load(DEPTH_V2 / "count_ridge_geo.joblib")
    v2_targets = {tree_id: int(n) for tree_id, n in
                  zip(ids, v2.count.predict_count(ridge, x_val))}
    if list(records) != list(v2_records) or list(base_targets) != list(v2_targets):
        raise RuntimeError("VAL tree ordering mismatch between frozen and v2 branches")

    combinations = {
        "original_topology__original_count": (base_payload, base_targets, base_profile),
        "original_topology__v2_count": (base_payload, v2_targets, base_profile),
        "v2_topology__original_count": (v2_payload, base_targets, v2_profile),
        "v2_topology__v2_count": (v2_payload, v2_targets, v2_profile),
    }
    rows = []
    for combo_name, (payload, targets, profile) in combinations.items():
        for class_mode in ("detector", "scale_matched", "scale_macro"):
            grouped = build_selected(payload, targets, profile)
            calibrated_classes(dataset, grouped, class_mode)
            metrics = short(evaluator.evaluate_grouped(grouped))
            rows.append({"combination": combo_name, "class_mode": class_mode,
                         "metrics": metrics})

    baseline = next(r["metrics"] for r in rows if
                    r["combination"] == "original_topology__original_count" and
                    r["class_mode"] == "detector")
    eligible = [r for r in rows if (
        r["metrics"]["physical_f1"] >= baseline["physical_f1"]
        and r["metrics"]["mae"] <= baseline["mae"]
        and r["metrics"]["matched_class_accuracy"] >= baseline["matched_class_accuracy"])]
    best_macro = max(rows, key=lambda r: (r["metrics"]["macro_f1"],
                                          r["metrics"]["matched_class_accuracy"],
                                          r["metrics"]["physical_f1"]))
    best_allround = max(eligible, key=lambda r: (r["metrics"]["macro_f1"],
                                                 r["metrics"]["matched_class_accuracy"],
                                                 r["metrics"]["physical_f1"]),
                        default=None)
    report = {
        "dataset": dataset,
        "protocol": "fixed TRAIN-fitted topology/count/class branches; Cartesian composition VAL only; no TEST",
        "baseline_val": baseline,
        "v2_solver_tags": solver_tags,
        "count_target_summary": {
            "original_mean": float(np.mean(list(base_targets.values()))),
            "v2_mean": float(np.mean(list(v2_targets.values()))),
        },
        "rows": rows,
        "best_by_macro": best_macro,
        "best_allrounder_guardrail": best_allround,
        "elapsed_sec": time.time() - started,
    }
    path = OUT / "depth_topology_count_class_combo_results_val.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"baseline": baseline, "best_by_macro": best_macro,
                      "best_allrounder_guardrail": best_allround,
                      "report": str(path)}, ensure_ascii=False), flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()
    run(args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
