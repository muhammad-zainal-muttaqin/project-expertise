#!/usr/bin/env python3
"""Train/evaluate timm visual member heads on frozen TRAIN/VAL topology.

Each model consumes one independent ImageNet backbone embedding extracted by
``timm_extract_features.py``.  Labels are inherited only for matched TRAIN
members; the linker, selected clusters, target counts, and all VAL metrics
remain under ``harness``.  TEST is intentionally not an accepted split.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import harness


FEATURE_ROOT = Path("/workspace/dino_head/features_timm")
OUT = Path("/workspace/cluster_head/artifacts")
MODELS = ("convnext_small", "swin_tiny", "efficientnetv2_rw_s")
K = harness.K


def group_key(group: dict) -> tuple:
    return tuple(sorted((int(m["side"]), str(m["stem"]), int(m["row_index"]))
                        for m in group["members"]))


def load_fmap(dataset: str, split: str, model_key: str):
    index_path = f"/workspace/dino_head/crops/{dataset}/{split}_index.npz"
    feature_path = FEATURE_ROOT / model_key / dataset / f"{split}_feat.npy"
    with np.load(index_path, allow_pickle=True) as z:
        stems = z["stem"].astype(str)
        rows = np.asarray(z["row_index"], dtype=np.int64)
    features = np.load(feature_path, mmap_mode="r")
    if len(stems) != len(features):
        raise RuntimeError(f"feature/index mismatch: {feature_path}")
    fmap = {(str(stem), int(row)): np.asarray(features[i], dtype=np.float32)
            for i, (stem, row) in enumerate(zip(stems, rows))}
    return fmap, int(features.shape[1])


def member_feature(member: dict, fmap: dict, dim: int) -> np.ndarray:
    feature = fmap.get((str(member["stem"]), int(member["row_index"])))
    if feature is None:
        feature = np.zeros(dim, dtype=np.float32)
    probs = np.asarray(member["p"], dtype=np.float32)
    side = np.zeros(4, dtype=np.float32)
    if 0 <= int(member["side"]) < 4:
        side[int(member["side"])] = 1.0
    scalars = np.asarray([
        float(member["score"]), float(member["cx"]), float(member["cy"]),
        float(member["w"]), float(member["h"]),
        float(member.get("rank_cx", 0.)), float(member.get("rank_cy", 0.)),
        float(member.get("z_side_x", 0.)), float(member.get("z_side_y", 0.)),
        float(member.get("z_side_area", 0.)), float(member.get("side_count", 1.)),
    ], dtype=np.float32)
    return np.concatenate([feature, probs, side, scalars])


def collect(dataset: str, split: str, model_key: str) -> dict:
    records, payload, targets, _prior = harness.build_payload(dataset, split)
    groups = harness.make_groups(payload, targets, harness.PROFILES[dataset])
    fmap, feature_dim = load_fmap(dataset, split, model_key)
    all_x, all_y, group_rows, keys = [], [], [], []
    matched_members = 0
    missing = 0
    for record, tree_groups in groups:
        matches = dict(harness.count.tree_matches(record, tree_groups))
        for group_index, group in enumerate(tree_groups):
            gt = int(record["bunches"][matches[group_index]]["cls"]
                     ) if group_index in matches else -1
            rows = []
            for member in group["members"]:
                if (str(member["stem"]), int(member["row_index"])) not in fmap:
                    missing += 1
                rows.append(len(all_x))
                all_x.append(member_feature(member, fmap, feature_dim))
                all_y.append(gt)
            group_rows.append(rows)
            keys.append(group_key(group))
            if gt >= 0:
                matched_members += len(rows)
    return {
        "records": records, "payload": payload, "targets": targets,
        "groups": groups, "X": np.asarray(all_x, dtype=np.float32),
        "y": np.asarray(all_y, dtype=np.int64), "group_rows": group_rows,
        "keys": keys, "feature_dim": feature_dim,
        "matched_members": matched_members, "missing": missing,
    }


def pool(member_probs: np.ndarray, data: dict, mode: str) -> np.ndarray:
    flat = [group for _record, groups in data["groups"] for group in groups]
    output = []
    for index, rows in enumerate(data["group_rows"]):
        probs = member_probs[rows]
        members = flat[index]["members"]
        if mode == "mean":
            weights = np.asarray([float(m["score"]) for m in members], dtype=np.float32)
            weights /= max(float(weights.sum()), 1e-8)
            output.append((probs * weights[:, None]).sum(axis=0))
        elif mode == "max":
            output.append(probs.max(axis=0))
        elif mode == "top":
            top = int(np.argmax([float(m["score"]) for m in members]))
            output.append(probs[top])
        else:
            raise ValueError(mode)
    output = np.maximum(np.asarray(output, dtype=np.float32), 1e-8)
    return output / np.maximum(output.sum(axis=1, keepdims=True), 1e-8)


def models(seed: int, feature_dim: int, jobs: int):
    # A PCA bottleneck keeps all three backbones on the same statistical
    # footing and limits the downstream head's variance on the small crops.
    pca_log = min(160, feature_dim + 15)
    pca_tree = min(128, feature_dim + 15)
    pca_hist = min(96, feature_dim + 15)
    yield "logistic", Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=pca_log, whiten=True, random_state=20260828)),
        ("clf", LogisticRegression(C=.15, max_iter=450, solver="lbfgs",
                                    class_weight="balanced", random_state=seed)),
    ])
    yield "extra", Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=pca_tree, whiten=True, random_state=20260828)),
        ("clf", ExtraTreesClassifier(n_estimators=480, min_samples_leaf=3,
                                     max_features="sqrt", class_weight="balanced",
                                     n_jobs=jobs, random_state=seed)),
    ])
    yield "hist", Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=pca_hist, whiten=True, random_state=20260828)),
        ("clf", HistGradientBoostingClassifier(max_iter=240, learning_rate=.05,
                                                max_leaf_nodes=15,
                                                min_samples_leaf=16,
                                                l2_regularization=2.,
                                                random_state=seed)),
    ])


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


def run(model_key: str, dataset: str, seed: int, jobs: int) -> dict:
    started = time.time()
    train = collect(dataset, "train", model_key)
    val = collect(dataset, "val", model_key)
    mask = train["y"] >= 0
    if mask.sum() < 20 or len(np.unique(train["y"][mask])) < K:
        raise RuntimeError(f"{dataset}/{model_key}: insufficient matched labels")
    baseline = short(harness.evaluate_clusters(
        val["payload"], val["targets"], harness.PROFILES[dataset]))
    detector = np.asarray([np.asarray(group["p"], dtype=np.float32)
                           for _record, groups in val["groups"] for group in groups])
    detector = np.maximum(detector, 1e-8)
    detector /= np.maximum(detector.sum(axis=1, keepdims=True), 1e-8)
    rows = []
    for name, model in models(seed, train["feature_dim"], jobs):
        fit_started = time.time()
        model.fit(train["X"][mask], train["y"][mask])
        member_probs = np.asarray(model.predict_proba(val["X"]), dtype=np.float32)
        for pooling in ("mean", "max", "top"):
            opinion = pool(member_probs, val, pooling)
            for alpha in (.05, .10, .15, .20, .30, .45, .60, .80, 1.0):
                logits = (np.log(detector)
                          + alpha * np.log(np.maximum(opinion, 1e-8)))
                prediction = np.argmax(logits, axis=1).astype(int)
                prediction_map = {key: int(cls)
                                  for key, cls in zip(val["keys"], prediction)}
                metrics = short(harness.evaluate_clusters(
                    val["payload"], val["targets"], harness.PROFILES[dataset],
                    lambda group, prediction_map=prediction_map:
                    prediction_map[group_key(group)]))
                metrics["physical_count_invariant"] = bool(
                    abs(metrics["physical_f1"] - baseline["physical_f1"]) < 1e-10
                    and abs(metrics["mae"] - baseline["mae"]) < 1e-10
                    and abs(metrics["pm1"] - baseline["pm1"]) < 1e-10)
                rows.append({"model": name, "pooling": pooling, "alpha": alpha,
                             "metrics": metrics,
                             "fit_elapsed_sec": time.time() - fit_started})
        joblib.dump(model, OUT / f"{dataset}_timm_{model_key}_{name}.joblib",
                    compress=3)
        top = sorted(rows, key=lambda row: (
            row["metrics"]["matched_class_accuracy"],
            row["metrics"]["macro_f1"]), reverse=True)[:3]
        print(json.dumps({"dataset": dataset, "backbone": model_key,
                          "model": name, "top": top}, ensure_ascii=False),
              flush=True)
    eligible = [row for row in rows if row["metrics"]["physical_count_invariant"]]
    best_match = max(eligible, key=lambda row: (
        row["metrics"]["matched_class_accuracy"], row["metrics"]["macro_f1"]))
    best_macro = max(eligible, key=lambda row: (
        row["metrics"]["macro_f1"], row["metrics"]["matched_class_accuracy"]))
    report = {
        "dataset": dataset, "backbone": model_key,
        "protocol": "fit timm member heads on matched TRAIN; select VAL; no TEST",
        "feature_dim": int(train["feature_dim"]),
        "train": {"members": int(len(train["X"])),
                  "matched_members": int(train["matched_members"]),
                  "missing": int(train["missing"])},
        "val": {"members": int(len(val["X"])),
                "matched_members": int(val["matched_members"]),
                "missing": int(val["missing"])},
        "baseline_val": baseline, "best_by_matched": best_match,
        "best_by_macro": best_macro, "results": rows,
        "elapsed_sec": time.time() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{dataset}_timm_{model_key}_results_val.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(json.dumps({"dataset": dataset, "backbone": model_key,
                      "best_by_matched": best_match,
                      "best_by_macro": best_macro, "report": str(path)},
                     ensure_ascii=False), flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("953", "depth"), required=True)
    ap.add_argument("--backbone", choices=MODELS, required=True)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    run(args.backbone, args.dataset, args.seed, max(1, args.jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
