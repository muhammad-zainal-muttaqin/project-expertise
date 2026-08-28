#!/usr/bin/env python3
"""TRAIN/VAL policy between the original Depth GSP and V2-geo GSP.

The two candidate linkers are fixed before the policy is fit.  A policy sees
only candidate graph/detection summaries and predicted count targets; it
never sees ground-truth boxes/classes as features and accepts no TEST split.
The experiment is designed to attack the observed Depth physical-F1 versus
counting-MAE compromise without changing the locked 953 branch.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/workspace/cluster_head")
sys.path.insert(0, "/workspace/gsp_linker")
sys.path.insert(0, "/workspace/pipeline_v2")

import harness  # noqa: E402
import link_global_setpartition as gsp  # noqa: E402
import pipeline_v2 as v2  # noqa: E402


OUT = Path("/workspace/cluster_head/artifacts")
DEPTH_V2 = Path("/workspace/pipeline_v2/artifacts/depth")
K = harness.K

BASE_PROFILE = dict(harness.PROFILES["depth"])
V2_PROFILE = {
    "link_threshold": .5, "singleton_min": .15,
    "max_size": 3, "rank_mode": "score",
}
V2_TAU = .25


def groups_for(item: tuple, targets: dict, profile: dict) -> list[dict]:
    rec, dets, edges = item
    return harness.count.selected_clusters(
        dets, edges, profile["link_threshold"], profile["singleton_min"],
        profile["max_size"], int(targets[rec["tree_id"]]), profile["rank_mode"])


def summarize(groups: list[dict], edges: list[tuple], dets: list[dict],
             target: int) -> list[float]:
    values = [
        np.asarray([float(group["score"]) for group in groups], dtype=float),
        np.asarray([len(group["members"]) for group in groups], dtype=float),
        np.asarray([float(edge[0]) for edge in edges], dtype=float),
        np.asarray([float(det["score"]) for det in dets], dtype=float),
    ]
    out = [float(target), float(len(dets)), float(len(edges)),
           float(len(groups)), float(len(groups) - target)]
    for array in values:
        if len(array):
            out.extend([float(array.mean()), float(array.std()),
                        float(array.min()), float(np.median(array)),
                        float(array.max()), float((array >= .5).sum())])
        else:
            out.extend([0.] * 6)
    out.extend(float(sum(int(det["side"]) == side for det in dets))
                for side in range(4))
    out.extend(float(sum(len(group["members"]) == size for group in groups))
                for size in (1, 2, 3, 4))
    if groups:
        probs = np.asarray([group["p"] for group in groups], dtype=float)
        out.extend(float(probs[:, cls].mean()) for cls in range(K))
        out.extend([float(probs.max(axis=1).mean()),
                    float(probs.max(axis=1).std())])
    else:
        out.extend([0.] * (K + 2))
    return out


def policy_features(base_item: tuple, v2_item: tuple,
                    base_targets: dict, v2_targets: dict) -> np.ndarray:
    br, bd, be = base_item
    vr, vd, ve = v2_item
    bg = groups_for(base_item, base_targets, BASE_PROFILE)
    vg = groups_for(v2_item, v2_targets, V2_PROFILE)
    a = np.asarray(summarize(bg, be, bd, int(base_targets[br["tree_id"]])),
                   dtype=np.float32)
    b = np.asarray(summarize(vg, ve, vd, int(v2_targets[vr["tree_id"]])),
                   dtype=np.float32)
    return np.concatenate([a, b, b - a]).astype(np.float32)


def one_quality(item: tuple, targets: dict, profile: dict) -> dict:
    rec, _dets, _edges = item
    groups = groups_for(item, targets, profile)
    matches = harness.count.tree_matches(rec, groups)
    correct = sum(int(groups[i]["cls"] == rec["bunches"][j]["cls"])
                  for i, j in matches)
    return {"tree_id": str(rec["tree_id"]), "tp": len(matches),
            "pred": len(groups), "gt": len(rec["bunches"]),
            "abs_count": abs(len(groups) - len(rec["bunches"])),
            "class_correct": correct}


def evaluate_mixed(items: list[tuple], targets: list[dict],
                   profiles: list[dict]) -> dict:
    if not (len(items) == len(targets) == len(profiles)):
        raise RuntimeError("mixed candidate lengths differ")
    cm = np.zeros((K + 1, K + 1), dtype=np.int64)
    total_pred = total_gt = total_tp = 0
    abs_count = exact = pm1 = vector_exact = class_correct = matched = 0
    per_tree = []
    for item, target_map, profile in zip(items, targets, profiles):
        rec, _dets, _edges = item
        groups = groups_for(item, target_map, profile)
        matches = harness.count.tree_matches(rec, groups)
        bunches = rec["bunches"]
        total_pred += len(groups); total_gt += len(bunches); total_tp += len(matches)
        delta = len(groups) - len(bunches)
        abs_count += abs(delta); exact += int(delta == 0); pm1 += int(abs(delta) <= 1)
        pred_count = np.bincount([group["cls"] for group in groups], minlength=K)
        gt_count = np.bincount([bunch["cls"] for bunch in bunches if bunch["cls"] >= 0],
                               minlength=K)
        vector_exact += int(np.array_equal(pred_count, gt_count))
        matched += len(matches)
        matched_pred = {i for i, _ in matches}
        matched_gt = {j for _, j in matches}
        for i, j in matches:
            pred_cls, gt_cls = int(groups[i]["cls"]), int(bunches[j]["cls"])
            if 0 <= pred_cls < K and 0 <= gt_cls < K:
                cm[pred_cls, gt_cls] += 1
                class_correct += int(pred_cls == gt_cls)
        for i, group in enumerate(groups):
            if i not in matched_pred and 0 <= int(group["cls"]) < K:
                cm[int(group["cls"]), K] += 1
        for j, bunch in enumerate(bunches):
            if j not in matched_gt and 0 <= int(bunch["cls"]) < K:
                cm[K, int(bunch["cls"])] += 1
        per_tree.append({"tree_id": str(rec["tree_id"]),
                         "gt_count": len(bunches), "pred_count": len(groups),
                         "count_delta": delta, "matched": len(matches),
                         "predicted_target": int(target_map[rec["tree_id"]])})
    precision = total_tp / max(total_pred, 1)
    recall = total_tp / max(total_gt, 1)
    f1 = 2. * precision * recall / max(precision + recall, 1e-12)
    f1s = []
    for cls in range(K):
        tp = cm[cls, cls]
        fp = cm[cls, :].sum() - tp
        fn = cm[:, cls].sum() - tp
        f1s.append(2. * tp / max(2 * tp + fp + fn, 1))
    return {
        "physical_detection": {"precision": precision, "recall": recall,
                                "f1": f1, "tp": total_tp,
                                "pred_clusters": total_pred, "gt_bunches": total_gt},
        "counting": {"mae": abs_count / max(len(items), 1),
                      "exact_accuracy": exact / max(len(items), 1),
                      "plus_minus_1_accuracy": pm1 / max(len(items), 1),
                      "vector_exact_accuracy": vector_exact / max(len(items), 1)},
        "classification": {
            "matched_class_accuracy": class_correct / max(matched, 1),
            "matched": matched, "macro_f1_end_to_end": float(np.mean(f1s)),
            "per_class_f1_end_to_end": dict(zip(harness.NAMES, f1s)),
            "confusion_prediction_rows": cm.tolist()},
        "per_tree": per_tree,
    }


def short(metrics: dict) -> dict:
    return {"physical_f1": metrics["physical_detection"]["f1"],
            "mae": metrics["counting"]["mae"],
            "pm1": metrics["counting"]["plus_minus_1_accuracy"],
            "matched_class_accuracy": metrics["classification"]["matched_class_accuracy"],
            "matched": metrics["classification"]["matched"],
            "macro_f1": metrics["classification"]["macro_f1_end_to_end"],
            "per_class_f1": metrics["classification"]["per_class_f1_end_to_end"]}


def model_bank(seed: int, jobs: int):
    yield "logistic", Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=.10, max_iter=700,
                                    class_weight="balanced", random_state=seed))])
    yield "extra", ExtraTreesClassifier(
        n_estimators=360, min_samples_leaf=6, max_features=.70,
        class_weight="balanced", n_jobs=jobs, random_state=seed)
    yield "hist", HistGradientBoostingClassifier(
        max_iter=240, learning_rate=.05, max_leaf_nodes=7,
        min_samples_leaf=12, l2_regularization=3., random_state=seed)


def load_v2_candidate(split: str, records: dict, prior: dict,
                      edge_model, vote: dict, target_map: dict) -> list[tuple]:
    per_tree = v2.build_dets_and_candidates(records, vote, prior, edge_model)
    payload, _tags = v2.payload_for_tau(records, per_tree, V2_TAU, 3)
    return payload


def run(seed: int, jobs: int) -> dict:
    started = time.time()
    cfg = v2.edge.cfg_for("depth")
    train_records = v2.count.four_side(v2.base.load_records(cfg, "train"))
    val_records = v2.count.four_side(v2.base.load_records(cfg, "val"))
    prior = v2.base.build_rotation_prior(v2.base.load_records(cfg, "train"))

    # Original GSP candidate and targets are produced by the validation-only
    # harness; this reproduces the locked depth anchor.
    _base_train_records, base_train_payload, base_train_targets, _ = harness.build_payload(
        "depth", "train")
    _base_val_records, base_val_payload, base_val_targets, _ = harness.build_payload(
        "depth", "val")
    if list(train_records) != list(_base_train_records) or list(val_records) != list(_base_val_records):
        raise RuntimeError("tree order mismatch between V2 and locked harness")

    vote_train = v2.edge.load_vote(DEPTH_V2 / "vote_v2_geo_train.npz")
    vote_val = v2.edge.load_vote(DEPTH_V2 / "vote_v2_geo_val.npz")
    edge_model = joblib.load(DEPTH_V2 / "edge_v2_geo.joblib")
    v2_train_payload = load_v2_candidate("train", train_records, prior,
                                         edge_model, vote_train, {})
    v2_val_payload = load_v2_candidate("val", val_records, prior,
                                       edge_model, vote_val, {})
    x_train, y_train, _ids_train = v2.build_count_features(
        train_records, vote_train,
        v2.build_dets_and_candidates(train_records, vote_train, prior, edge_model))
    x_val, _y_val, ids_val = v2.build_count_features(
        val_records, vote_val,
        v2.build_dets_and_candidates(val_records, vote_val, prior, edge_model))
    ridge = joblib.load(DEPTH_V2 / "count_ridge_geo.joblib")
    v2_train_targets = {tree_id: int(n) for tree_id, n in
                        zip(train_records, v2.count.predict_count(ridge, x_train))}
    v2_val_targets = {tree_id: int(n) for tree_id, n in
                      zip(ids_val, v2.count.predict_count(ridge, x_val))}

    train_items = list(zip(base_train_payload, v2_train_payload))
    val_items = list(zip(base_val_payload, v2_val_payload))
    train_base_quality = [one_quality(a, base_train_targets, BASE_PROFILE)
                          for a, _b in train_items]
    train_v2_quality = [one_quality(b, v2_train_targets, V2_PROFILE)
                        for _a, b in train_items]
    q_base = {name: np.asarray([row[name] for row in train_base_quality],
                               dtype=np.float32)
              for name in ("tp", "abs_count", "class_correct")}
    q_v2 = {name: np.asarray([row[name] for row in train_v2_quality],
                             dtype=np.float32)
            for name in ("tp", "abs_count", "class_correct")}
    utility_specs = {
        "balanced": {"tp": 1.0, "count": .75, "class": .25},
        "physical": {"tp": 2.0, "count": .25, "class": .10},
        "count": {"tp": .50, "count": 1.50, "class": .10},
    }
    X_train = np.asarray([policy_features(a, b, base_train_targets, v2_train_targets)
                          for a, b in train_items], dtype=np.float32)
    X_val = np.asarray([policy_features(a, b, base_val_targets, v2_val_targets)
                        for a, b in val_items], dtype=np.float32)
    baseline = short(evaluate_mixed(
        base_val_payload, [base_val_targets] * len(base_val_payload),
        [BASE_PROFILE] * len(base_val_payload)))
    v2_only = short(evaluate_mixed(
        v2_val_payload, [v2_val_targets] * len(v2_val_payload),
        [V2_PROFILE] * len(v2_val_payload)))
    rows = [{"policy": "always_original_gsp", "metrics": baseline},
            {"policy": "always_v2_geo", "metrics": v2_only}]
    policy_details = []
    for utility_name, weights in utility_specs.items():
        utility = (weights["tp"] * (q_v2["tp"] - q_base["tp"])
                   - weights["count"] * (q_v2["abs_count"] - q_base["abs_count"])
                   + weights["class"] * (q_v2["class_correct"] - q_base["class_correct"]))
        labels = (utility > 0.).astype(np.int64)
        if len(np.unique(labels)) < 2:
            policy_details.append({"utility": utility_name,
                                   "skipped": "single-class labels"})
            continue
        for name, model in model_bank(seed, jobs):
            model.fit(X_train, labels)
            choose = model.predict(X_val).astype(bool)
            mixed = [v2_val_payload[i] if choose[i] else base_val_payload[i]
                     for i in range(len(choose))]
            targets = [v2_val_targets if choose[i] else base_val_targets
                       for i in range(len(choose))]
            profiles = [V2_PROFILE if choose[i] else BASE_PROFILE
                        for i in range(len(choose))]
            metrics = short(evaluate_mixed(mixed, targets, profiles))
            path = OUT / f"depth_adaptive_v2_{utility_name}_{name}.joblib"
            joblib.dump(model, path, compress=3)
            row = {"policy": f"{utility_name}:{name}",
                   "utility": utility_name, "model": name,
                   "gsp_selected_val": int(choose.sum()), "metrics": metrics}
            rows.append(row)
            policy_details.append({"utility": utility_name, "model": name,
                                   "train_positive": int(labels.sum()),
                                   "train_positive_fraction": float(labels.mean()),
                                   "utility_mean": float(utility.mean()),
                                   "utility_std": float(utility.std()),
                                   "gsp_selected_val": int(choose.sum())})

    best = max(rows, key=lambda row: (
        row["metrics"]["matched_class_accuracy"],
        row["metrics"]["physical_f1"], -row["metrics"]["mae"],
        row["metrics"]["macro_f1"]))
    best_physical = max(rows, key=lambda row: (
        row["metrics"]["physical_f1"],
        row["metrics"]["matched_class_accuracy"], -row["metrics"]["mae"]))
    best_count = min(rows, key=lambda row: (
        row["metrics"]["mae"], -row["metrics"]["pm1"],
        -row["metrics"]["physical_f1"]))
    report = {
        "dataset": "depth",
        "protocol": "fixed original-GSP/V2-geo candidates; policy fit TRAIN; evaluate VAL; no TEST",
        "seed": seed, "candidate": {"v2_tau_prob": V2_TAU, **V2_PROFILE},
        "train_trees": len(train_items), "val_trees": len(val_items),
        "feature_dim": int(X_train.shape[1]),
        "baseline_val": baseline, "v2_only_val": v2_only,
        "utility_specs": utility_specs, "policy_details": policy_details,
        "rows": rows, "selected_validation": best,
        "best_physical": best_physical, "best_count": best_count,
        "elapsed_sec": time.time() - started,
    }
    path = OUT / "depth_adaptive_v2_results_val.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(json.dumps({"baseline": baseline, "v2_only": v2_only,
                      "selected_validation": best,
                      "best_physical": best_physical,
                      "best_count": best_count, "report": str(path)},
                     ensure_ascii=False), flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    run(args.seed, max(1, args.jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
