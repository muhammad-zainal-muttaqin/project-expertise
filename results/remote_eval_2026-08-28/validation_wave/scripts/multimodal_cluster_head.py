"""Parallel RGB/auxiliary DINO cluster heads, TRAIN/VAL only.

The frozen physical clusters come from ``cluster_head.harness``.  This module
adds a parallel auxiliary branch (mono-depth for 953, calibrated sensor depth
for Depth) and retains the detector probability/scalar path as an explicit
skip connection.  No test file is accepted or loaded.
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

import sys
sys.path.insert(0, "/workspace/cluster_head")
import cluster_head_experiment as base  # noqa: E402


OUT = Path("/workspace/aux_modal/artifacts")
EMB_BLOCKS = 3 * 1536  # mean, score-weighted mean, per-dimension max


def aux_fmap(dataset: str, split: str):
    idx_path = Path(f"/workspace/dino_head/crops/{dataset}/{split}_index.npz")
    feat_path = Path(f"/workspace/aux_modal/features/{dataset}/{split}_aux_dinofeat.npy")
    with np.load(idx_path, allow_pickle=True) as z:
        stems = z["stem"].astype(str)
        rows = np.asarray(z["row_index"], dtype=np.int64)
    feat = np.load(feat_path, mmap_mode="r")
    if len(stems) != len(feat):
        raise RuntimeError(f"aux feature/index mismatch {dataset}/{split}")
    return ({(str(stem), int(row)): np.asarray(feat[i], dtype=np.float32)
             for i, (stem, row) in enumerate(zip(stems, rows))},
            int(feat.shape[1]))


def modal_matrix(data: dict, dataset: str, split: str, kind: str):
    amap, adim = aux_fmap(dataset, split)
    rows = []
    missing = 0
    for _rec, groups in data["groups"]:
        for group in groups:
            vec, miss = base.group_features(group, amap, adim)
            rows.append(vec)
            missing += miss
    aux = np.asarray(rows, dtype=np.float32)
    rgb = np.asarray(data["X"], dtype=np.float32)
    if len(aux) != len(rgb):
        raise RuntimeError("group feature row mismatch")
    # Keep the low-dimensional detector/geometry features once.  The two
    # DINO branches remain separate so the classifier can learn a residual
    # correction instead of allowing one modality to erase the other.
    if kind == "aux_only":
        X = np.concatenate([aux[:, :EMB_BLOCKS], rgb[:, EMB_BLOCKS:]], axis=1)
    elif kind == "base_aux":
        X = np.concatenate([rgb[:, :EMB_BLOCKS], aux[:, :EMB_BLOCKS],
                            rgb[:, EMB_BLOCKS:]], axis=1)
    else:
        raise ValueError(kind)
    return X, {"missing_members": int(missing), "feature_dim": int(X.shape[1])}


def models(seed: int):
    # One compact linear and two non-linear heads.  PCA is fit inside each
    # pipeline on TRAIN only, so no VAL statistics leak into the head.
    yield "logistic_pca160", Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=160, whiten=True, random_state=seed)),
        ("clf", LogisticRegression(C=.20, max_iter=350, solver="lbfgs",
                                    class_weight="balanced", random_state=seed)),
    ])
    yield "extra_pca128", Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=128, whiten=True, random_state=seed)),
        ("clf", ExtraTreesClassifier(n_estimators=320, min_samples_leaf=3,
                                     max_features="sqrt", class_weight="balanced",
                                     n_jobs=8, random_state=seed)),
    ])
    yield "hist_pca96", Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=96, whiten=True, random_state=seed)),
        ("clf", HistGradientBoostingClassifier(max_iter=220, learning_rate=.05,
                                                max_leaf_nodes=15,
                                                l2_regularization=2., random_state=seed)),
    ])


def metrics_short(m: dict) -> dict:
    return {
        "physical_f1": m["physical_detection"]["f1"],
        "mae": m["counting"]["mae"],
        "pm1": m["counting"]["plus_minus_1_accuracy"],
        "matched_class_accuracy": m["classification"]["matched_class_accuracy"],
        "matched": m["classification"]["matched"],
        "macro_f1": m["classification"]["macro_f1_end_to_end"],
        "per_class_f1": m["classification"]["per_class_f1_end_to_end"],
    }


def run(dataset: str, seed: int):
    started = time.time()
    # This calls only the frozen TRAIN/VAL harness and its anchor-compatible
    # linker.  It never reads any test archive.
    train = base.collect_groups(dataset, "train")
    val = base.collect_groups(dataset, "val")
    train_mask = train["y"] >= 0
    y = train["y"][train_mask]
    if len(y) < 20 or len(np.unique(y)) < base.K:
        raise RuntimeError("insufficient matched training labels")
    baseline = base.harness.evaluate_clusters(
        val["payload"], val["targets"], base.harness.PROFILES[dataset])
    all_results = []
    for kind in ("aux_only", "base_aux"):
        Xtr, tr_diag = modal_matrix(train, dataset, "train", kind)
        Xva, va_diag = modal_matrix(val, dataset, "val", kind)
        for name, model in models(seed):
            fit_start = time.time()
            model.fit(Xtr[train_mask], y)
            q = np.asarray(model.predict_proba(Xva), dtype=np.float32)
            detector = np.asarray([np.asarray(g["p"], dtype=np.float32)
                                   for _rec, gs in val["groups"] for g in gs])
            detector = np.maximum(detector, 1e-8)
            detector /= detector.sum(axis=1, keepdims=True)
            for mode, alpha in [("head", 1.), ("blend", .15), ("blend", .25),
                                ("blend", .40), ("blend", .60), ("blend", .85),
                                ("blend", 1.), ("residual", .25),
                                ("residual", .50), ("residual", .75)]:
                if mode == "head":
                    probs = q
                elif mode == "blend":
                    z = np.log(detector) + alpha * np.log(np.maximum(q, 1e-8))
                    probs = base._softmax(z)
                else:
                    residual = np.log(np.maximum(q, 1e-8)) - np.log(1. / base.K)
                    probs = base._softmax(np.log(detector) + alpha * residual)
                pred = np.argmax(probs, axis=1).astype(int)
                pmap = base._prediction_map(val["keys"], pred)
                m = base.evaluate_with_map(val, pmap)
                short = metrics_short(m)
                short["physical_count_invariant"] = bool(
                    abs(short["physical_f1"] - baseline["physical_detection"]["f1"]) < 1e-10
                    and abs(short["mae"] - baseline["counting"]["mae"]) < 1e-10
                    and abs(short["pm1"] - baseline["counting"]["plus_minus_1_accuracy"]) < 1e-10)
                all_results.append({"kind": kind, "model": name, "mode": mode,
                                    "alpha": alpha, "metrics": short,
                                    "fit_elapsed_sec": time.time() - fit_start,
                                    "train_diag": tr_diag, "val_diag": va_diag})
            OUT.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, OUT / f"{dataset}_{kind}_{name}.joblib", compress=3)
            top = sorted(all_results, key=lambda r: (
                r["metrics"]["matched_class_accuracy"], r["metrics"]["macro_f1"]),
                reverse=True)[:2]
            print(json.dumps({"dataset": dataset, "kind": kind, "model": name,
                              "top": top}, ensure_ascii=False), flush=True)
    eligible = [r for r in all_results if r["metrics"]["physical_count_invariant"]]
    best = max(eligible, key=lambda r: (r["metrics"]["matched_class_accuracy"],
                                        r["metrics"]["macro_f1"])) if eligible else None
    report = {"dataset": dataset,
              "protocol": "fit TRAIN matched clusters; select VAL; no TEST",
              "baseline_val": metrics_short(baseline), "results": all_results,
              "selected_validation": best, "elapsed_sec": time.time() - started}
    path = OUT / f"{dataset}_multimodal_results.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"dataset": dataset, "selected_validation": best,
                      "report": str(path)}, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("953", "depth"), required=True)
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()
    run(args.dataset, args.seed)


if __name__ == "__main__":
    main()
