"""Rich train-only count models for the frozen multi-view pipeline.

The original count head sees proposal counts and soft class mass.  This
experiment adds only inference-available statistics: edge-score distributions,
graph connectivity, and candidate-cluster distributions at fixed thresholds.
It does not change detector boxes or the linker and never accepts a TEST split.

The aim is to cover the known precision/recall compromise at the count layer
without using a ground-truth count as an input.  Models are compared by
grouped cross-validation on TRAIN, then fitted on all TRAIN trees and evaluated
on VAL.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, "/workspace/cluster_head")
sys.path.insert(0, "/workspace/gsp_linker")
import harness  # noqa: E402
import sweep_remote_pipeline as sweep  # noqa: E402


OUT = Path("/workspace/pipeline_v3_count/artifacts")
K = harness.K
EDGE_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90)
CLUSTER_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70)
SINGLETON_THRESHOLDS = (0.05, 0.10, 0.15, 0.20)
CLUSTER_MAX_SIZES = (3, 4)


def _stats(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return [0., 0., 0., 0., 0., 0., 0.]
    return [float(values.size), float(values.sum()), float(values.mean()),
            float(values.std()), float(values.min()), float(np.median(values)),
            float(values.max())]


def _cluster_stats(groups: list[dict]) -> list[float]:
    sizes = np.asarray([len(g["members"]) for g in groups], dtype=float)
    scores = np.asarray([float(g["score"]) for g in groups], dtype=float)
    out = [*_stats(scores)]
    out.extend(float(np.sum(sizes == s)) for s in (1, 2, 3, 4))
    out.extend([float(sizes.sum()) if len(sizes) else 0.,
                float(sizes.mean()) if len(sizes) else 0.,
                float(sizes.std()) if len(sizes) else 0.,
                float(sizes.max()) if len(sizes) else 0.])
    if groups:
        p = np.asarray([g["p"] for g in groups], dtype=float)
        w = np.asarray([max(float(g["score"]), 1e-6) for g in groups])
        w /= max(float(w.sum()), 1e-8)
        out.extend((p * w[:, None]).sum(axis=0).tolist())
        out.extend(p.max(axis=1).tolist()[:0])
        out.extend(_stats(p.max(axis=1)))
    else:
        out.extend([0.] * (K + 7))
    return out


def tree_features(rec: dict, dets: list[dict], edges: list[tuple[float, int, int]]) -> np.ndarray:
    """Inference-only graph and proposal statistics for one tree."""
    out: list[float] = []
    by_side: dict[int, list[dict]] = {s: [] for s in range(4)}
    for d in dets:
        by_side.setdefault(int(d["side"]), []).append(d)

    # Proposal counts, confidence, geometry, and soft class mass per view.
    for side in range(4):
        rows = by_side.get(side, [])
        scores = np.asarray([float(d["score"]) for d in rows], dtype=float)
        out.extend(_stats(scores))
        if rows:
            p = np.asarray([np.asarray(d["p"], dtype=float) for d in rows])
            p = np.maximum(p, 0.)
            p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-8)
            w = scores / max(float(scores.sum()), 1e-8)
            out.extend((p * w[:, None]).sum(axis=0).tolist())
            boxes = np.asarray([d["box"] for d in rows], dtype=float)
            area = np.maximum(boxes[:, 2] - boxes[:, 0], 0.) * \
                np.maximum(boxes[:, 3] - boxes[:, 1], 0.)
            out.extend(_stats(area / max(float(rec["views"][side]["width"] *
                                              rec["views"][side]["height"]), 1.)))
            out.extend([float(p.max(axis=1).mean()), float(p.max(axis=1).std())])
        else:
            out.extend([0.] * (K + 7 + 2))

    all_scores = np.asarray([float(d["score"]) for d in dets], dtype=float)
    out.extend(_stats(all_scores))
    out.extend([float(np.sum(all_scores >= t)) for t in EDGE_THRESHOLDS])

    # Edge score and edge-degree summaries are still available at inference.
    edge_scores = np.asarray([float(e[0]) for e in edges], dtype=float)
    out.extend(_stats(edge_scores))
    out.extend([float(np.sum(edge_scores >= t)) for t in EDGE_THRESHOLDS])
    degree = np.zeros(len(dets), dtype=float)
    for score, i, j in edges:
        if score >= 0.05:
            degree[int(i)] += 1.
            degree[int(j)] += 1.
    out.extend(_stats(degree))

    # Candidate-cluster count and score statistics at fixed operating points.
    for threshold in CLUSTER_THRESHOLDS:
        for singleton in SINGLETON_THRESHOLDS:
            for max_size in CLUSTER_MAX_SIZES:
                groups = sweep.clusters(dets, edges, threshold, singleton, max_size)
                out.extend(_cluster_stats(groups))

    # A compact view of the profile's untruncated cluster count is useful for
    # correcting the learned count target while remaining GT-free.
    for threshold in (0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        groups = sweep.clusters(dets, edges, threshold, 0.15, 4)
        out.append(float(len(groups)))
        out.append(float(sum(len(g["members"]) for g in groups)))
    return np.asarray(out, dtype=np.float32)


def collect(dataset: str, split: str) -> dict:
    if split not in ("train", "val"):
        raise ValueError("this experiment accepts only train or val")
    records, payload, targets, _prior = harness.build_payload(dataset, split)
    x, y, ids = [], [], []
    for rec, dets, edges in payload:
        x.append(tree_features(rec, dets, edges))
        y.append(float(len(rec["bunches"])))
        ids.append(str(rec["tree_id"]))
    X = np.asarray(x, dtype=np.float32)
    return {"dataset": dataset, "split": split, "records": records,
            "payload": payload, "targets": targets, "X": X,
            "y": np.asarray(y, dtype=np.float32), "ids": np.asarray(ids)}


def model_bank(seed: int, jobs: int):
    yield "ridge", Pipeline([
        ("scale", StandardScaler()), ("model", Ridge(alpha=10.0))])
    yield "extra", ExtraTreesRegressor(
        n_estimators=360, min_samples_leaf=5, max_features=.70,
        random_state=seed, n_jobs=jobs, criterion="squared_error")
    yield "forest", RandomForestRegressor(
        n_estimators=300, min_samples_leaf=5, max_features=.70,
        random_state=seed, n_jobs=jobs, criterion="squared_error")
    yield "hist", HistGradientBoostingRegressor(
        max_iter=220, learning_rate=.04, max_leaf_nodes=7,
        l2_regularization=3.0, random_state=seed)


def predict_int(model, X: np.ndarray) -> np.ndarray:
    return np.maximum(0, np.rint(np.asarray(model.predict(X), dtype=float))).astype(int)


def cv_mae(factory, X: np.ndarray, y: np.ndarray, ids: np.ndarray,
           seed: int, jobs: int) -> dict:
    trees = np.asarray(sorted(set(ids)), dtype=str)
    kf = KFold(n_splits=min(5, len(trees)), shuffle=True, random_state=seed)
    fold_scores = []
    for fit_tree_idx, hold_tree_idx in kf.split(trees):
        fit_trees = set(trees[fit_tree_idx].tolist())
        hold_trees = set(trees[hold_tree_idx].tolist())
        fit = np.asarray([i for i, t in enumerate(ids) if t in fit_trees])
        hold = np.asarray([i for i, t in enumerate(ids) if t in hold_trees])
        model = factory()
        model.fit(X[fit], y[fit])
        pred = predict_int(model, X[hold])
        fold_scores.append(float(np.abs(pred - y[hold]).mean()))
    return {"fold_mae": fold_scores, "cv_mae": float(np.mean(fold_scores))}


def short(m: dict) -> dict:
    return {
        "physical_f1": m["physical_detection"]["f1"],
        "mae": m["counting"]["mae"],
        "pm1": m["counting"]["plus_minus_1_accuracy"],
        "matched_class_accuracy": m["classification"]["matched_class_accuracy"],
        "matched": m["classification"]["matched"],
        "macro_f1": m["classification"]["macro_f1_end_to_end"],
        "per_class_f1": m["classification"]["per_class_f1_end_to_end"],
    }


def evaluate(data: dict, counts: np.ndarray, rank_mode: str) -> dict:
    profile = dict(harness.PROFILES[data["dataset"]])
    profile["rank_mode"] = rank_mode
    return harness.evaluate_clusters(data["payload"],
                                     {t: int(c) for t, c in zip(data["ids"], counts)},
                                     profile)


def run(dataset: str, seed: int, jobs: int) -> dict:
    started = time.time()
    train, val = collect(dataset, "train"), collect(dataset, "val")
    if train["X"].shape[1] != val["X"].shape[1]:
        raise RuntimeError("train/val rich count feature dimensions differ")
    baseline_metrics = harness.evaluate_clusters(
        val["payload"], val["targets"], harness.PROFILES[dataset])
    baseline = short(baseline_metrics)
    rank_modes = ("score", "support", "max_member", "class_conf")
    rows = []
    fitted = {}
    for name, prototype in model_bank(seed, jobs):
        def factory(name=name):
            for candidate_name, candidate in model_bank(seed, jobs):
                if candidate_name == name:
                    return candidate
            raise KeyError(name)
        cv = cv_mae(factory, train["X"], train["y"], train["ids"], seed, jobs)
        model = factory()
        model.fit(train["X"], train["y"])
        fitted[name] = model
        val_counts = predict_int(model, val["X"])
        train_counts = predict_int(model, train["X"])
        joblib.dump(model, OUT / f"{dataset}_rich_count_{name}.joblib", compress=3)
        for rank in rank_modes:
            metrics = short(evaluate(val, val_counts, rank))
            rows.append({"model": name, "rank_mode": rank,
                         "cv": cv, "direct_count_mae_val": float(np.abs(val_counts - val["y"]).mean()),
                         "metrics": metrics})
        print(json.dumps({"dataset": dataset, "model": name, "cv_mae": cv["cv_mae"],
                          "val_count_mean": float(val_counts.mean())},
                         ensure_ascii=False), flush=True)

    # A small, declared ensemble of the two models with the lowest TRAIN CV
    # error is tested because count bias and variance are different failure
    # modes.  It is not selected from TEST.
    cv_by_model = {}
    for row in rows:
        cv_by_model[row["model"]] = row["cv"]["cv_mae"]
    top_models = sorted(cv_by_model, key=lambda n: (cv_by_model[n], n))[:2]
    if len(top_models) == 2:
        ensemble_counts = np.rint(np.mean(
            [predict_int(fitted[n], val["X"]) for n in top_models], axis=0)).astype(int)
        for rank in rank_modes:
            rows.append({"model": "+".join(top_models), "rank_mode": rank,
                         "cv": {"ensemble_of": top_models},
                         "direct_count_mae_val": float(np.abs(ensemble_counts - val["y"]).mean()),
                         "metrics": short(evaluate(val, ensemble_counts, rank))})

    eligible = [r for r in rows if (
        r["metrics"]["matched_class_accuracy"] > baseline["matched_class_accuracy"]
        and r["metrics"]["mae"] <= baseline["mae"])]
    best_matched = max(rows, key=lambda r: (r["metrics"]["matched_class_accuracy"],
                                             r["metrics"]["macro_f1"],
                                             -r["metrics"]["mae"]))
    best_allrounder = max(eligible, key=lambda r: (
        r["metrics"]["matched_class_accuracy"], r["metrics"]["physical_f1"],
        r["metrics"]["macro_f1"], -r["metrics"]["mae"])) if eligible else None
    report = {
        "dataset": dataset,
        "protocol": "rich inference-only count features; fit TRAIN; select VAL; no TEST",
        "seed": seed, "train_trees": int(len(train["ids"])),
        "val_trees": int(len(val["ids"])), "feature_dim": int(train["X"].shape[1]),
        "feature_design": {"edge_thresholds": EDGE_THRESHOLDS,
                           "cluster_thresholds": CLUSTER_THRESHOLDS,
                           "singleton_thresholds": SINGLETON_THRESHOLDS,
                           "cluster_max_sizes": CLUSTER_MAX_SIZES},
        "baseline_val": baseline, "models_cv": cv_by_model,
        "rows": rows, "best_by_matched": best_matched,
        "best_allrounder_guardrail": best_allrounder,
        "elapsed_sec": time.time() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{dataset}_rich_count_results_val.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(json.dumps({"dataset": dataset, "best_by_matched": best_matched,
                      "best_allrounder_guardrail": best_allrounder,
                      "report": str(path), "seconds": report["elapsed_sec"]},
                     ensure_ascii=False), flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("953", "depth"), required=True)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    run(args.dataset, args.seed, args.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
