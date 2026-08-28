#!/usr/bin/env python3
"""Train-only multi-view count meta-ensemble for both corpora.

The earlier count heads were fitted per proposal/reranker branch.  This
experiment asks a narrower but useful question: can one train-only regressor
combine the raw, inference-available statistics from the original vote and
both V2 vote modes, then feed that count target into the frozen original
Hungarian/GSP selection?  It is a residual count layer with the original
profile as the skip path.

All feature blocks are generated from TRAIN/VAL predictions and saved
TRAIN-fitted edge/count artifacts.  The meta regressors are fit on TRAIN and
evaluated once on VAL.  The command intentionally has no TEST option.
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
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/workspace/project-expertise/scripts")
sys.path.insert(0, "/workspace/cluster_head")
sys.path.insert(0, "/workspace/pipeline_v2")

import evaluate_remote_count_reconciled as count  # noqa: E402
import harness  # noqa: E402
import pipeline_v2 as v2  # noqa: E402


OUT = Path("/workspace/cluster_head/artifacts")
V2_ROOT = Path("/workspace/pipeline_v2/artifacts")
SEED = 20260828
RANK_MODES = ("score", "support", "max_member", "class_conf",
              "class_conf_power_0.25", "class_conf_power_0.50")


def short(metrics: dict) -> dict:
    return {
        "physical_f1": float(metrics["physical_detection"]["f1"]),
        "mae": float(metrics["counting"]["mae"]),
        "pm1": float(metrics["counting"]["plus_minus_1_accuracy"]),
        "matched_class_accuracy": float(
            metrics["classification"]["matched_class_accuracy"]),
        "matched": int(metrics["classification"]["matched"]),
        "macro_f1": float(metrics["classification"]["macro_f1_end_to_end"]),
        "per_class_f1": metrics["classification"]["per_class_f1_end_to_end"],
    }


def raw_count_features(dataset: str, split: str) -> np.ndarray:
    cfg = v2.edge.cfg_for(dataset)
    records = v2.count.four_side(v2.base.load_records(cfg, split))
    root = v2.ORIG_FUSED_ROOT
    vote = v2.edge.load_vote(v2.edge.vote_path(root, dataset, split))
    return np.stack([
        count.feature_vector(rec, vote, v2.PROPOSAL_MIN_V2)
        for rec in records.values()
    ]).astype(np.float32)


def v2_feature_block(dataset: str, mode: str, split: str):
    cfg = v2.edge.cfg_for(dataset)
    records = v2.count.four_side(v2.base.load_records(cfg, split))
    prior = v2.base.build_rotation_prior(v2.base.load_records(cfg, "train"))
    root = V2_ROOT / dataset
    vote = v2.edge.load_vote(root / f"vote_v2_{mode}_{split}.npz")
    edge_model = joblib.load(root / f"edge_v2_{mode}.joblib")
    if hasattr(edge_model, "n_jobs"):
        edge_model.n_jobs = 1
    per_tree = v2.build_dets_and_candidates(records, vote, prior, edge_model)
    x, _y, ids = v2.build_count_features(records, vote, per_tree)
    return np.asarray(x, dtype=np.float32), list(ids)


def build_split(dataset: str, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cfg = v2.edge.cfg_for(dataset)
    records = v2.count.four_side(v2.base.load_records(cfg, split))
    ids = list(records)
    blocks = [raw_count_features(dataset, split)]
    for mode in ("ptp", "geo"):
        x, mode_ids = v2_feature_block(dataset, mode, split)
        if mode_ids != ids:
            raise RuntimeError(f"{dataset}/{split}: tree order mismatch for {mode}")
        blocks.append(x)
    X = np.concatenate(blocks, axis=1).astype(np.float32)
    y = np.asarray([count.target_count(rec) for rec in records.values()],
                   dtype=np.float32)
    return X, y, ids


def model_bank(seed: int, jobs: int):
    yield "ridge_cv", Pipeline([
        ("scale", StandardScaler()),
        ("model", RidgeCV(alphas=(0.1, 1., 10., 100., 1000.), cv=5,
                           scoring="neg_mean_absolute_error")),
    ])
    yield "hist", HistGradientBoostingRegressor(
        max_iter=260, learning_rate=.045, max_leaf_nodes=9,
        min_samples_leaf=18, l2_regularization=4., random_state=seed)
    yield "extra", ExtraTreesRegressor(
        n_estimators=520, min_samples_leaf=5, max_features=.70,
        criterion="absolute_error", n_jobs=jobs, random_state=seed)


def integer_prediction(model, X: np.ndarray) -> np.ndarray:
    return np.maximum(0, np.rint(np.asarray(model.predict(X), dtype=float))).astype(int)


def cv_mae(name: str, X: np.ndarray, y: np.ndarray, seed: int, jobs: int) -> dict:
    factories = {n: m for n, m in model_bank(seed, jobs)}
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for fit_idx, hold_idx in kf.split(X):
        # Recreate the estimator so no fold shares fitted state.
        model = next(m for n, m in model_bank(seed, jobs) if n == name)
        model.fit(X[fit_idx], y[fit_idx])
        pred = integer_prediction(model, X[hold_idx])
        scores.append(float(np.abs(pred - y[hold_idx]).mean()))
    return {"fold_mae": scores, "cv_mae": float(np.mean(scores))}


def evaluate_counts(data: dict, counts: np.ndarray, rank_mode: str) -> dict:
    profile = dict(harness.PROFILES[data["dataset"]])
    profile["rank_mode"] = rank_mode
    target = {tree_id: int(n) for tree_id, n in zip(data["ids"], counts)}
    return harness.evaluate_clusters(data["payload"], target, profile)


def baseline_data(dataset: str, split: str) -> dict:
    records, payload, targets, _prior = harness.build_payload(dataset, split)
    return {"dataset": dataset, "records": records, "payload": payload,
            "targets": targets, "ids": list(records)}


def dominates(m: dict, b: dict) -> bool:
    return (m["physical_f1"] >= b["physical_f1"]
            and m["mae"] <= b["mae"]
            and m["pm1"] >= b["pm1"]
            and m["matched_class_accuracy"] >= b["matched_class_accuracy"]
            and m["macro_f1"] >= b["macro_f1"])


def run(dataset: str, seed: int, jobs: int) -> dict:
    started = time.time()
    Xtr, ytr, train_ids = build_split(dataset, "train")
    Xva, yva, val_ids = build_split(dataset, "val")
    if Xtr.shape[1] != Xva.shape[1] or train_ids[:0] != val_ids[:0]:
        raise RuntimeError(f"{dataset}: train/VAL feature contract mismatch")
    train_data = baseline_data(dataset, "train")
    val_data = baseline_data(dataset, "val")
    if train_ids != list(train_data["ids"]) or val_ids != list(val_data["ids"]):
        raise RuntimeError(f"{dataset}: feature/tree order mismatch")

    baseline = short(harness.evaluate_clusters(
        val_data["payload"], val_data["targets"], harness.PROFILES[dataset]))
    models = {}
    cv = {}
    predictions = {}
    rows = []
    for name, model in model_bank(seed, jobs):
        cv[name] = cv_mae(name, Xtr, ytr, seed, jobs)
        model.fit(Xtr, ytr)
        models[name] = model
        pred = integer_prediction(model, Xva)
        predictions[name] = pred
        joblib.dump(model, OUT / f"{dataset}_count_meta_{name}.joblib", compress=3)
        print(json.dumps({"dataset": dataset, "model": name,
                          "cv_mae": cv[name]["cv_mae"],
                          "val_direct_mae": float(np.abs(pred - yva).mean())},
                         ensure_ascii=False), flush=True)

    # Both mean and median are fixed, tiny ensembles of all three trained
    # regressors.  They are not selected using VAL prediction error; the
    # downstream VAL table is the declared selection surface.
    predictions["ensemble_mean"] = np.rint(np.mean(
        [predictions[n] for n in ("ridge_cv", "hist", "extra")], axis=0)).astype(int)
    predictions["ensemble_median"] = np.rint(np.median(
        [predictions[n] for n in ("ridge_cv", "hist", "extra")], axis=0)).astype(int)

    for model_name, pred in predictions.items():
        for rank_mode in RANK_MODES:
            metrics = short(evaluate_counts(val_data, pred, rank_mode))
            rows.append({"model": model_name, "rank_mode": rank_mode,
                         "metrics": metrics,
                         "val_direct_target_mae": float(np.abs(pred - yva).mean())})

    best_by_match = max(rows, key=lambda r: (
        r["metrics"]["matched_class_accuracy"], r["metrics"]["macro_f1"],
        r["metrics"]["physical_f1"], -r["metrics"]["mae"]))
    best_by_mae = min(rows, key=lambda r: (
        r["metrics"]["mae"], -r["metrics"]["physical_f1"],
        -r["metrics"]["matched_class_accuracy"]))
    allround = [r for r in rows if dominates(r["metrics"], baseline)]

    # For 953, apply the already declared robust class calibrations to the
    # top count candidates only.  This is a composition audit, not a new
    # class grid; it keeps the class layer's skip connection explicit.
    calibrated_rows = []
    if dataset == "953":
        import cross_layer_953 as composition
        shortlist = []
        seen = set()
        for r in sorted(rows, key=lambda x: (
                x["metrics"]["matched_class_accuracy"],
                x["metrics"]["macro_f1"]), reverse=True)[:8] + \
                sorted(rows, key=lambda x: (
                    x["metrics"]["mae"], -x["metrics"]["physical_f1"]))[:8]:
            key = (r["model"], r["rank_mode"])
            if key not in seen:
                seen.add(key); shortlist.append(r)
        for r in shortlist:
            pred = predictions[r["model"]]
            target = {tree_id: int(n) for tree_id, n in zip(val_ids, pred)}
            profile = dict(harness.PROFILES[dataset])
            profile["rank_mode"] = r["rank_mode"]
            # Reuse the same original payload but run the class choices on a
            # fresh group copy for each mode.
            for mode in ("scale_matched", "scale_macro"):
                metrics, _groups = composition.evaluate_candidate(
                    val_data["payload"], target, profile, mode)
                calibrated_rows.append({"model": r["model"],
                                        "rank_mode": r["rank_mode"],
                                        "class_mode": mode, "metrics": metrics})

    all_rows = rows + calibrated_rows
    best_calibrated_match = max(calibrated_rows, key=lambda r: (
        r["metrics"]["matched_class_accuracy"], r["metrics"]["macro_f1"],
        r["metrics"]["physical_f1"], -r["metrics"]["mae"]), default=None)
    best_calibrated_macro = max(calibrated_rows, key=lambda r: (
        r["metrics"]["macro_f1"], r["metrics"]["matched_class_accuracy"],
        r["metrics"]["physical_f1"], -r["metrics"]["mae"]), default=None)
    report = {
        "dataset": dataset,
        "protocol": "raw original+V2 count features fit TRAIN; count/profile composition selected VAL; no TEST",
        "seed": seed, "feature_dim": int(Xtr.shape[1]),
        "feature_blocks": ["original_vote_stats", "v2_ptp_stats", "v2_geo_stats"],
        "train_trees": len(train_ids), "val_trees": len(val_ids),
        "models_cv": cv, "baseline_val": baseline,
        "rows": all_rows,
        "best_by_matched": best_by_match,
        "best_by_target_mae": best_by_mae,
        "best_allrounder_guardrail": max(
            allround, key=lambda r: (r["metrics"]["matched_class_accuracy"],
                                     r["metrics"]["macro_f1"]), default=None),
        "calibrated_shortlist": {
            "n": len(calibrated_rows),
            "best_by_matched": best_calibrated_match,
            "best_by_macro": best_calibrated_macro,
        },
        "elapsed_sec": time.time() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{dataset}_count_meta_ensemble_results_val.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"dataset": dataset, "baseline": baseline,
                      "best_by_matched": best_by_match,
                      "best_by_target_mae": best_by_mae,
                      "best_allrounder": report["best_allrounder_guardrail"],
                      "calibrated_best": best_calibrated_match,
                      "report": str(path)}, ensure_ascii=False), flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=("953", "depth"),
                    default=("953", "depth"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    for dataset in args.datasets:
        run(dataset, args.seed, max(1, args.jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
