"""Validation-only cluster class-head experiments.

The physical linker and count layer are frozen to the profiles in
``harness.py``.  This script learns a class decision on matched TRAIN
clusters and evaluates it on VAL only.  It never opens a test archive.

The fusion is deliberately residual/skip based: the detector probability
vector is always present in the features, and candidate predictions can be
combined with the detector's log-probabilities.  A candidate is useful only
if it improves the validation class metrics without changing the physical or
counting metrics.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import harness


warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used.*",
    category=UserWarning,
)


ROOT = Path("/workspace/cluster_head")
ARTIFACTS = ROOT / "artifacts"
K = harness.K


def _fmap(dataset: str, split: str) -> tuple[dict[tuple[str, int], np.ndarray], int]:
    """Load DINO features keyed by the exact proposal identity."""
    idx_path = Path(f"/workspace/dino_head/crops/{dataset}/{split}_index.npz")
    feat_path = Path(f"/workspace/dino_head/features/{dataset}/{split}_dinofeat.npy")
    with np.load(idx_path, allow_pickle=True) as z:
        stems = z["stem"].astype(str)
        rows = np.asarray(z["row_index"], dtype=np.int64)
    features = np.load(feat_path, mmap_mode="r")
    if len(stems) != len(features):
        raise RuntimeError(f"feature/index length mismatch {dataset}/{split}: "
                           f"{len(features)} vs {len(stems)}")
    out = {(str(stem), int(row)): np.asarray(features[i], dtype=np.float32)
           for i, (stem, row) in enumerate(zip(stems, rows))}
    return out, int(features.shape[1])


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float32)
    return float(-(p * np.log(np.maximum(p, 1e-8))).sum())


def group_key(group: dict) -> tuple:
    """Stable identity for a group reconstructed by harness.make_groups."""
    return tuple(sorted((int(m["side"]), str(m["stem"]), int(m["row_index"]))
                        for m in group["members"]))


def group_features(group: dict, fmap: dict[tuple[str, int], np.ndarray],
                    feature_dim: int) -> tuple[np.ndarray, int]:
    """Create a detector+geometry+DINO feature with explicit skip inputs."""
    members = group["members"]
    feats = []
    missing = 0
    for member in members:
        f = fmap.get((str(member["stem"]), int(member["row_index"])))
        if f is None:
            missing += 1
        else:
            feats.append(f)
    if feats:
        fmat = np.asarray(feats, dtype=np.float32)
        weights = np.asarray([max(float(m["score"]), 1e-5)
                              for m in members
                              if (str(m["stem"]), int(m["row_index"])) in fmap],
                            dtype=np.float32)
        weights /= max(float(weights.sum()), 1e-8)
        mean = fmat.mean(axis=0)
        weighted = (fmat * weights[:, None]).sum(axis=0)
        vmax = fmat.max(axis=0)
    else:
        mean = weighted = vmax = np.zeros(feature_dim, dtype=np.float32)

    p = np.asarray(group["p"], dtype=np.float32)
    p = np.maximum(p, 1e-8)
    p /= max(float(p.sum()), 1e-8)
    logp = np.log(p)
    member_scores = np.asarray([float(m["score"]) for m in members], dtype=np.float32)
    member_p = np.asarray([np.asarray(m["p"], dtype=np.float32) for m in members])
    boxes = np.asarray([np.asarray(m["box"], dtype=np.float32) for m in members])
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    sizes = np.maximum(boxes[:, 2:] - boxes[:, :2], 1e-6)
    side_onehot = np.zeros(4, dtype=np.float32)
    for side in {int(m["side"]) for m in members}:
        if 0 <= side < 4:
            side_onehot[side] = 1.0

    # These low-dimensional values are the explicit residual/skip path.
    scalars = np.concatenate([
        p, logp, np.asarray(group.get("head_p", p), dtype=np.float32),
        np.asarray([float(group["score"]), float(len(members)),
                    float(member_scores.max()), float(member_scores.mean()),
                    float(member_scores.std()), float(member_scores.min()),
                    float(centers[:, 0].mean()), float(centers[:, 1].mean()),
                    float(centers[:, 0].std()), float(centers[:, 1].std()),
                    float(sizes[:, 0].mean()), float(sizes[:, 1].mean()),
                    float(sizes[:, 0].std()), float(sizes[:, 1].std()),
                    _entropy(p), float(p.max() - np.partition(p, -2)[-2]),
                    float(member_p.max(axis=0).max()),
                    float(member_p.mean(axis=0).max())], dtype=np.float32),
        side_onehot,
    ])
    vector = np.concatenate([mean, weighted, vmax, scalars]).astype(np.float32)
    return vector, missing


def collect_groups(dataset: str, split: str):
    """Build frozen groups and labels; labels come only from TRAIN matches."""
    profile = harness.PROFILES[dataset]
    records, payload, targets, _prior = harness.build_payload(dataset, split)
    grouped = harness.make_groups(payload, targets, profile)
    fmap, feature_dim = _fmap(dataset, split)
    rows = []
    labels = []
    keys = []
    missing_members = 0
    n_members = 0
    matched_groups = 0
    for rec, groups in grouped:
        matches = dict(harness.count.tree_matches(rec, groups))
        for i, group in enumerate(groups):
            vector, missing = group_features(group, fmap, feature_dim)
            missing_members += missing
            n_members += len(group["members"])
            rows.append(vector)
            keys.append(group_key(group))
            if i in matches:
                labels.append(int(rec["bunches"][matches[i]]["cls"]))
                matched_groups += 1
            else:
                labels.append(-1)
    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return {
        "dataset": dataset, "split": split, "records": records,
        "payload": payload, "targets": targets, "groups": grouped,
        "X": X, "y": y, "keys": keys, "fmap": fmap,
        "feature_dim": feature_dim, "matched_groups": matched_groups,
        "n_groups": len(rows), "missing_members": missing_members,
        "n_members": n_members,
    }


def _make_models(seed: int):
    """Small diverse model bank; each candidate is fit on TRAIN only."""
    # The PCA branch is a compact representation suitable for a paper's
    # reproducible ablation; the full linear branch tests whether more detail
    # helps.  Tree models see the compact representation to avoid overfitting.
    yield "linear_full", Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.10, max_iter=350, solver="lbfgs",
                                    class_weight="balanced", random_state=seed)),
    ])
    yield "linear_pca128", Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=128, whiten=True, random_state=seed)),
        ("clf", LogisticRegression(C=0.25, max_iter=300, solver="lbfgs",
                                    class_weight="balanced", random_state=seed)),
    ])
    yield "extra_compact", Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=96, whiten=True, random_state=seed)),
        ("clf", ExtraTreesClassifier(n_estimators=240, min_samples_leaf=3,
                                     max_features="sqrt", class_weight="balanced",
                                     n_jobs=8, random_state=seed)),
    ])
    yield "hist_compact", Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=64, whiten=True, random_state=seed)),
        ("clf", HistGradientBoostingClassifier(max_iter=180, learning_rate=.06,
                                                max_leaf_nodes=15, l2_regularization=2.,
                                                random_state=seed)),
    ])


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-8)


def candidate_predictions(model, X: np.ndarray, groups: list[tuple[dict, list[dict]]],
                           alpha: float, mode: str) -> tuple[np.ndarray, dict]:
    q = np.asarray(model.predict_proba(X), dtype=np.float32)
    if q.shape[1] != K:
        # A split with an absent class is not expected, but keeping this
        # explicit makes the failure auditable instead of silently remapping.
        raise RuntimeError(f"classifier probability shape {q.shape}; expected (*,{K})")
    d = []
    for _rec, gs in groups:
        for group in gs:
            p = np.asarray(group["p"], dtype=np.float32)
            p = np.maximum(p, 1e-8)
            p /= max(float(p.sum()), 1e-8)
            d.append(p)
    detector = np.asarray(d, dtype=np.float32)
    if mode == "head":
        probs = q
    elif mode == "blend":
        # Log-opinion pool: alpha=0 is the exact detector skip path.
        probs = _softmax(np.log(detector) + alpha * np.log(np.maximum(q, 1e-8)))
    elif mode == "residual":
        # Centered log-probability residual; the detector remains the anchor.
        residual = np.log(np.maximum(q, 1e-8)) - np.log(1.0 / K)
        probs = _softmax(np.log(detector) + alpha * residual)
    else:
        raise ValueError(mode)
    return np.argmax(probs, axis=1).astype(int), {
        "mean_head_conf": float(q.max(axis=1).mean()),
        "mean_fused_conf": float(probs.max(axis=1).mean()),
    }


def _prediction_map(keys: list[tuple], pred: np.ndarray) -> dict[tuple, int]:
    if len(keys) != len(pred):
        raise RuntimeError("prediction/key length mismatch")
    return {key: int(cls) for key, cls in zip(keys, pred)}


def evaluate_with_map(data: dict, prediction_map: dict[tuple, int]) -> dict:
    def class_fn(group):
        key = group_key(group)
        if key not in prediction_map:
            raise KeyError(f"missing prediction for group {key}")
        return prediction_map[key]
    return harness.evaluate_clusters(data["payload"], data["targets"],
                                     harness.PROFILES[data["dataset"]], class_fn)


def compact_metrics(metrics: dict) -> dict:
    return {
        "physical_f1": metrics["physical_detection"]["f1"],
        "mae": metrics["counting"]["mae"],
        "pm1": metrics["counting"]["plus_minus_1_accuracy"],
        "matched_class_accuracy": metrics["classification"]["matched_class_accuracy"],
        "matched": metrics["classification"]["matched"],
        "macro_f1": metrics["classification"]["macro_f1_end_to_end"],
        "per_class_f1": metrics["classification"]["per_class_f1_end_to_end"],
    }


def run_dataset(dataset: str, seed: int) -> dict:
    t0 = time.time()
    train = collect_groups(dataset, "train")
    val = collect_groups(dataset, "val")
    train_mask = train["y"] >= 0
    if train_mask.sum() < 20 or len(np.unique(train["y"][train_mask])) < K:
        raise RuntimeError(f"insufficient matched TRAIN labels: {train['y'][train_mask]}")
    baseline = harness.evaluate_clusters(train["payload"], train["targets"],
                                         harness.PROFILES[dataset])
    val_baseline = harness.evaluate_clusters(val["payload"], val["targets"],
                                             harness.PROFILES[dataset])
    results = []
    Xtr, ytr = train["X"][train_mask], train["y"][train_mask]
    for name, model in _make_models(seed):
        started = time.time()
        model.fit(Xtr, ytr)
        for mode, alpha in [("head", 1.0), ("blend", .25), ("blend", .50),
                            ("blend", .75), ("blend", 1.0),
                            ("residual", .25), ("residual", .50),
                            ("residual", .75), ("residual", 1.0)]:
            pred, diag = candidate_predictions(model, val["X"], val["groups"], alpha, mode)
            pmap = _prediction_map(val["keys"], pred)
            metrics = evaluate_with_map(val, pmap)
            results.append({"model": name, "mode": mode, "alpha": alpha,
                            "metrics": compact_metrics(metrics), "diag": diag,
                            "fit_elapsed_sec": time.time() - started})
        joblib.dump(model, ARTIFACTS / f"{dataset}_{name}.joblib", compress=3)
        print(json.dumps({"dataset": dataset, "model": name,
                          "top": sorted(results, key=lambda r: (
                              r["metrics"]["matched_class_accuracy"],
                              r["metrics"]["macro_f1"]), reverse=True)[:2]},
                         ensure_ascii=False), flush=True)

    # Select by matched accuracy, then macro-F1, while requiring the frozen
    # physical/count metrics to remain exact to numerical tolerance.
    for row in results:
        m = row["metrics"]
        m["physical_count_invariant"] = bool(
            abs(m["physical_f1"] - val_baseline["physical_detection"]["f1"]) < 1e-10
            and abs(m["mae"] - val_baseline["counting"]["mae"]) < 1e-10
            and abs(m["pm1"] - val_baseline["counting"]["plus_minus_1_accuracy"]) < 1e-10)
    eligible = [r for r in results if r["metrics"]["physical_count_invariant"]]
    best = max(eligible, key=lambda r: (r["metrics"]["matched_class_accuracy"],
                                        r["metrics"]["macro_f1"],
                                        -r["metrics"]["mae"])) if eligible else None
    report = {
        "dataset": dataset, "protocol": "fit TRAIN matched clusters; select VAL; no TEST",
        "train": {k: train[k] for k in ("n_groups", "matched_groups", "missing_members",
                                         "n_members", "feature_dim")},
        "val": {k: val[k] for k in ("n_groups", "matched_groups", "missing_members",
                                     "n_members", "feature_dim")},
        "baseline_train": compact_metrics(baseline),
        "baseline_val": compact_metrics(val_baseline),
        "results": results,
        "selected_validation": best,
        "elapsed_sec": time.time() - t0,
    }
    out = ARTIFACTS / f"{dataset}_cluster_head_results.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": dataset, "selected_validation": best,
                      "report": str(out)}, ensure_ascii=False), flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=("953", "depth"),
                    default=("953", "depth"))
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    all_reports = {dataset: run_dataset(dataset, args.seed) for dataset in args.datasets}
    # Parallel invocations are intentional for the two independent datasets;
    # avoid a shared summary-file race when a caller launches one process per
    # dataset to keep all CPU resources busy.
    if len(args.datasets) > 1:
        out = ARTIFACTS / "cluster_head_results_all.json"
        out.write_text(json.dumps(all_reports, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        print(f"-> {out}")
    else:
        print("single-dataset run: per-dataset report already written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
