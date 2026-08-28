#!/usr/bin/env python3
"""Train a composition-aware class head for the 953 VAL candidate.

The 953 pipeline has a promising class calibration on the original
Hungarian/count profile, while the V2 count target has not yet been crossed
with a head trained on that exact composition.  This script changes only the
class head: original Hungarian groups are selected with the TRAIN-fitted V2
geo Ridge count target, member labels are inherited from matched TRAIN
groups, and several declared skip/blend settings are evaluated on VAL.

Only TRAIN and VAL are accepted.  No TEST path exists in this file.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, "/workspace/cluster_head")
sys.path.insert(0, "/workspace/pipeline_v2")

import harness  # noqa: E402
import head_aware_selection as evaluator  # noqa: E402
import member_head as mh  # noqa: E402
import pipeline_v2 as v2  # noqa: E402


OUT = Path("/workspace/cluster_head/artifacts")
V2_ROOT = Path("/workspace/pipeline_v2/artifacts/953")
DATASET = "953"
SEED = 20260828


def short(metrics: dict) -> dict:
    return evaluator.short(metrics)


def v2_count_targets(split: str) -> dict[str, int]:
    """Use the saved TRAIN-fitted V2 geo Ridge target for a split."""
    cfg = v2.edge.cfg_for(DATASET)
    records = v2.count.four_side(v2.base.load_records(cfg, split))
    prior = v2.base.build_rotation_prior(v2.base.load_records(cfg, "train"))
    vote = v2.edge.load_vote(V2_ROOT / f"vote_v2_geo_{split}.npz")
    edge_model = joblib.load(V2_ROOT / "edge_v2_geo.joblib")
    per_tree = v2.build_dets_and_candidates(records, vote, prior, edge_model)
    features, _y, ids = v2.build_count_features(records, vote, per_tree)
    ridge = joblib.load(V2_ROOT / "count_ridge_geo.joblib")
    prediction = v2.count.predict_count(ridge, features)
    return {tree_id: int(n) for tree_id, n in zip(ids, prediction)}


def composition_data(split: str, targets: dict[str, int]) -> dict:
    records, payload, _old_targets, _class_prior = harness.build_payload(
        DATASET, split)
    groups = [(rec, [copy.deepcopy(g) for g in gs])
              for rec, gs in harness.make_groups(
                  payload, targets, harness.PROFILES[DATASET])]
    fmap, dim = mh._load_fmap(DATASET, split)
    features, labels, group_rows, keys = [], [], [], []
    missing = 0
    for rec, tree_groups in groups:
        matches = dict(harness.count.tree_matches(rec, tree_groups))
        for gi, group in enumerate(tree_groups):
            gt = int(rec["bunches"][matches[gi]]["cls"]) if gi in matches else -1
            current = []
            for member in group["members"]:
                key = (str(member["stem"]), int(member["row_index"]))
                missing += int(key not in fmap)
                current.append(len(features))
                features.append(mh.member_feature(member, fmap, dim))
                labels.append(gt)
            group_rows.append(current)
            keys.append(mh.harness_group_key(group))
    return {
        "records": records, "payload": payload, "groups": groups,
        "X": np.asarray(features, dtype=np.float32),
        "y": np.asarray(labels, dtype=np.int64), "group_rows": group_rows,
        "keys": keys, "missing": missing,
    }


def pool(probabilities: np.ndarray, data: dict, mode: str) -> np.ndarray:
    flat = [g for _rec, groups in data["groups"] for g in groups]
    output = []
    for gi, rows in enumerate(data["group_rows"]):
        q = probabilities[rows]
        if mode == "max":
            p = q.max(axis=0)
        elif mode == "mean":
            weights = np.asarray([float(m["score"])
                                 for m in flat[gi]["members"]], dtype=np.float32)
            weights /= max(float(weights.sum()), 1e-8)
            p = (q * weights[:, None]).sum(axis=0)
        elif mode == "top":
            j = int(np.argmax([float(m["score"])
                               for m in flat[gi]["members"]]))
            p = q[j]
        else:
            raise ValueError(mode)
        output.append(p)
    p = np.maximum(np.asarray(output, dtype=np.float32), 1e-8)
    return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-8)


def apply_prediction(data: dict, prediction: np.ndarray) -> list[tuple]:
    candidate = [(rec, [copy.deepcopy(g) for g in groups])
                 for rec, groups in data["groups"]]
    if len(prediction) != sum(len(groups) for _rec, groups in candidate):
        raise RuntimeError("group prediction length mismatch")
    offset = 0
    for _rec, groups in candidate:
        for group in groups:
            group["cls"] = int(prediction[offset])
            offset += 1
    return candidate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    started = time.time()
    train_targets = v2_count_targets("train")
    val_targets = v2_count_targets("val")
    train = composition_data("train", train_targets)
    val = composition_data("val", val_targets)
    mask = train["y"] >= 0
    if mask.sum() < 20 or len(np.unique(train["y"][mask])) < harness.K:
        raise RuntimeError("insufficient matched composition labels")

    baseline = short(evaluator.evaluate_grouped(
        apply_prediction(val, np.asarray([
            int(np.argmax(g["p"]))
            for _rec, groups in val["groups"] for g in groups]))))
    rows = []
    biases = ((0., 0., 0.), (.10, 0., 0.), (.15, -.05, 0.),
              (.15, -.10, 0.), (.15, -.10, -.10),
              (.10, -.10, -.10), (.05, -.10, -.05))
    for name, model in mh.models(args.seed):
        fit_start = time.time()
        model.fit(train["X"][mask], train["y"][mask])
        q = np.asarray(model.predict_proba(val["X"]), dtype=np.float32)
        for pooling in ("mean", "max", "top"):
            opinion = pool(q, val, pooling)
            detector = np.asarray([
                np.asarray(g["p"], dtype=np.float32)
                for _rec, groups in val["groups"] for g in groups])
            detector = np.maximum(detector, 1e-8)
            detector /= np.maximum(detector.sum(axis=1, keepdims=True), 1e-8)
            for alpha in (.05, .10, .15, .20, .30, .45, .60, .85, 1.0):
                for b2, b3, b4 in biases:
                    logits = (np.log(detector)
                              + alpha * np.log(np.maximum(opinion, 1e-8)))
                    logits += np.asarray([0., b2, b3, b4], dtype=np.float32)
                    pred = np.argmax(logits, axis=1)
                    metrics = short(evaluator.evaluate_grouped(
                        apply_prediction(val, pred)))
                    rows.append({"model": name, "pooling": pooling,
                                 "alpha": alpha, "bias": [0., b2, b3, b4],
                                 "metrics": metrics,
                                 "fit_elapsed_sec": time.time() - fit_start})
        OUT.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, OUT / f"953_composition_{name}.joblib", compress=3)
        print(json.dumps({"model": name, "rows": len(rows),
                          "elapsed_sec": time.time() - fit_start},
                         ensure_ascii=False), flush=True)

    best_matched = max(rows, key=lambda r: (
        r["metrics"]["matched_class_accuracy"], r["metrics"]["macro_f1"]))
    best_macro = max(rows, key=lambda r: (
        r["metrics"]["macro_f1"], r["metrics"]["matched_class_accuracy"]))
    eligible = [r for r in rows if (
        r["metrics"]["physical_f1"] >= baseline["physical_f1"]
        and r["metrics"]["mae"] <= baseline["mae"]
        and r["metrics"]["pm1"] >= baseline["pm1"])]
    report = {
        "dataset": DATASET,
        "protocol": "fit composition-aware member heads on TRAIN; select VAL; no TEST",
        "composition": "original_hungarian_topology + v2_geo_ridge_count",
        "train": {"groups": len(train["group_rows"]),
                  "matched_members": int(mask.sum()),
                  "missing": int(train["missing"])},
        "val": {"groups": len(val["group_rows"]),
                "missing": int(val["missing"])},
        "baseline_composition_val": baseline,
        "best_by_matched": best_matched,
        "best_by_macro": best_macro,
        "best_allrounder_guardrail": max(
            eligible, key=lambda r: (r["metrics"]["macro_f1"],
                                     r["metrics"]["matched_class_accuracy"]),
            default=None),
        "rows": rows,
        "elapsed_sec": time.time() - started,
    }
    path = OUT / "953_composition_aware_head_results_val.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"baseline": baseline,
                      "best_by_matched": best_matched,
                      "best_by_macro": best_macro,
                      "best_allrounder_guardrail": report["best_allrounder_guardrail"],
                      "report": str(path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
