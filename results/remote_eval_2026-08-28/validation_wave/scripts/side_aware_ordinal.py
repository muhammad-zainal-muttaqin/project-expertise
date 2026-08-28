"""Side-aware group classifier and ordinal head, TRAIN/VAL only.

The established member heads classify each crop before pooling.  This
experiment tests the complementary representation: project the frozen
DINOv2-Large embedding once, aggregate it separately for each camera side,
and classify the resulting physical cluster.  A cumulative ordinal bank is
included because B1--B4 are ordered labels.  The detector probability remains
available as a skip path and the frozen topology is untouched.
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
import large_member_head as large


OUT = Path("/workspace/cluster_head/artifacts")
K = harness.K
SIDE_COUNT = 4
PROJECTION_DIM = 64


def key(group: dict) -> tuple:
    return tuple(sorted((int(m["side"]), str(m["stem"]), int(m["row_index"]))
                        for m in group["members"]))


def group_labels(data: dict) -> np.ndarray:
    labels = []
    for rec, groups in data["groups"]:
        matches = dict(harness.count.tree_matches(rec, groups))
        for gi, _group in enumerate(groups):
            labels.append(int(rec["bunches"][matches[gi]]["cls"]) if gi in matches else -1)
    return np.asarray(labels, dtype=np.int64)


def fit_projection(train: dict) -> PCA:
    x = np.asarray(train["X"][:, :train["dim"]], dtype=np.float32)
    n = min(PROJECTION_DIM, x.shape[0], x.shape[1])
    pca = PCA(n_components=n, whiten=True, random_state=20260828)
    pca.fit(x)
    return pca


def make_features(data: dict, pca: PCA) -> tuple[np.ndarray, list[tuple]]:
    z = pca.transform(np.asarray(data["X"][:, :data["dim"]], dtype=np.float32))
    rows, keys = [], []
    for rec, groups in data["groups"]:
        for group in groups:
            row_ids = []
            side_ids = []
            for m in group["members"]:
                ident = (str(m["stem"]), int(m["row_index"]))
                # The collector's member rows are stored in group_rows; use a
                # stable key lookup rather than assuming feature-map order.
                side_ids.append(int(m["side"]))
            # Reconstruct the corresponding row indices from the collector's
            # ordered group list.
            gi_global = len(rows)
            row_ids = data["group_rows"][gi_global]
            side_means = np.zeros((SIDE_COUNT, z.shape[1]), dtype=np.float32)
            side_max = np.full((SIDE_COUNT, z.shape[1]), -1e4, dtype=np.float32)
            side_n = np.zeros(SIDE_COUNT, dtype=np.float32)
            scores = []
            for ri, member in zip(row_ids, group["members"]):
                side = int(member["side"])
                if 0 <= side < SIDE_COUNT:
                    side_means[side] += z[ri]
                    side_max[side] = np.maximum(side_max[side], z[ri])
                    side_n[side] += 1.
                scores.append(float(member["score"]))
            side_means /= np.maximum(side_n[:, None], 1.)
            side_max = np.where(side_n[:, None] > 0., side_max, 0.)
            weights = np.asarray(scores, dtype=np.float32)
            weights /= max(float(weights.sum()), 1e-8)
            global_mean = (z[row_ids] * weights[:, None]).sum(axis=0)
            global_max = z[row_ids].max(axis=0)
            p = np.asarray(group["p"], dtype=np.float32)
            p = np.maximum(p, 1e-8); p /= max(float(p.sum()), 1e-8)
            entropy = float(-(p * np.log(p)).sum())
            sp = np.sort(p)
            scalars = np.asarray([
                *p.tolist(), *np.log(p).tolist(), entropy, float(sp[-1] - sp[-2]),
                float(group["score"]), float(len(group["members"])),
                float(np.max(scores)), float(np.mean(scores)), float(np.std(scores)),
                *((side_n > 0.).astype(np.float32)).tolist(),
            ], dtype=np.float32)
            rows.append(np.concatenate([side_means.ravel(), side_max.ravel(),
                                        global_mean, global_max, scalars]))
            keys.append(key(group))
    return np.asarray(rows, dtype=np.float32), keys


def model_bank(seed: int, jobs: int):
    yield "group_logistic", Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=.10, max_iter=800,
                                    class_weight="balanced", random_state=seed))])
    yield "group_extra", ExtraTreesClassifier(
        n_estimators=320, min_samples_leaf=4, max_features="sqrt",
        class_weight="balanced", n_jobs=jobs, random_state=seed)
    yield "group_hist", HistGradientBoostingClassifier(
        max_iter=180, learning_rate=.05, max_leaf_nodes=9,
        l2_regularization=3., random_state=seed)


def ordinal_model(seed: int):
    return [Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=.10, max_iter=800,
                                    class_weight="balanced", random_state=seed + k))
    ]) for k in range(K - 1)]


def ordinal_predict(models, X: np.ndarray) -> np.ndarray:
    cumulative = []
    for threshold, model in enumerate(models):
        # P(Y > threshold), with both binary classes guaranteed by TRAIN.
        cumulative.append(model.predict_proba(X)[:, 1])
    cumulative = np.asarray(cumulative).T
    q = np.zeros((len(X), K), dtype=np.float32)
    q[:, 0] = 1. - cumulative[:, 0]
    q[:, 1] = cumulative[:, 0] - cumulative[:, 1]
    q[:, 2] = cumulative[:, 1] - cumulative[:, 2]
    q[:, 3] = cumulative[:, 2]
    q = np.maximum(q, 1e-7)
    return q / np.maximum(q.sum(axis=1, keepdims=True), 1e-8)


def evaluate(data: dict, q: np.ndarray, detector: np.ndarray, keys: list[tuple],
             mode: str, alpha: float) -> dict:
    if mode == "head":
        z = np.log(np.maximum(q, 1e-8))
    elif mode == "blend":
        z = np.log(np.maximum(detector, 1e-8)) + alpha * np.log(np.maximum(q, 1e-8))
    else:
        raise ValueError(mode)
    pred = np.argmax(z, axis=1).astype(int)
    pmap = {k: int(c) for k, c in zip(keys, pred)}
    m = harness.evaluate_clusters(
        data["payload"], data["targets"], harness.PROFILES[data["dataset"]],
        lambda group, pmap=pmap: pmap[key(group)])
    return {"physical_f1": m["physical_detection"]["f1"],
            "mae": m["counting"]["mae"],
            "pm1": m["counting"]["plus_minus_1_accuracy"],
            "matched_class_accuracy": m["classification"]["matched_class_accuracy"],
            "matched": m["classification"]["matched"],
            "macro_f1": m["classification"]["macro_f1_end_to_end"],
            "per_class_f1": m["classification"]["per_class_f1_end_to_end"]}


def run(dataset: str, seed: int, jobs: int) -> dict:
    started = time.time()
    train = large.collect(dataset, "train")
    val = large.collect(dataset, "val")
    train["dataset"], val["dataset"] = dataset, dataset
    y_train, y_val = group_labels(train), group_labels(val)
    if len(y_train) != len(train["group_rows"]) or len(y_val) != len(val["group_rows"]):
        raise RuntimeError("group label/row mismatch")
    pca = fit_projection(train)
    X_train, train_keys = make_features(train, pca)
    X_val, val_keys = make_features(val, pca)
    if train_keys != train["keys"] or val_keys != val["keys"]:
        raise RuntimeError("group feature key order mismatch")
    mask = y_train >= 0
    detector = np.asarray([np.asarray(g["p"], dtype=np.float32)
                           for _r, gs in val["groups"] for g in gs], dtype=np.float32)
    detector = np.maximum(detector, 1e-8)
    detector /= np.maximum(detector.sum(1, keepdims=True), 1e-8)
    baseline = evaluate(val, detector, detector, val_keys, "head", 1.)
    rows = []
    for name, model in model_bank(seed, jobs):
        model.fit(X_train[mask], y_train[mask])
        q = np.asarray(model.predict_proba(X_val), dtype=np.float32)
        joblib.dump(model, OUT / f"{dataset}_{name}_sideaware.joblib", compress=3)
        for mode, alpha in (("head", 1.), ("blend", .15), ("blend", .25),
                            ("blend", .50), ("blend", .75), ("blend", 1.)):
            rows.append({"model": name, "mode": mode, "alpha": alpha,
                         "metrics": evaluate(val, q, detector, val_keys, mode, alpha)})

    ord_models = ordinal_model(seed)
    for threshold, model in enumerate(ord_models):
        model.fit(X_train[mask], (y_train[mask] > threshold).astype(np.int64))
    q_ord = ordinal_predict(ord_models, X_val)
    joblib.dump({"pca": pca, "models": ord_models},
                OUT / f"{dataset}_sideaware_ordinal.joblib", compress=3)
    for mode, alpha in (("head", 1.), ("blend", .15), ("blend", .25),
                        ("blend", .50), ("blend", .75), ("blend", 1.)):
        rows.append({"model": "ordinal_logistic", "mode": mode, "alpha": alpha,
                     "metrics": evaluate(val, q_ord, detector, val_keys, mode, alpha)})

    eligible = [r for r in rows if (
        abs(r["metrics"]["physical_f1"] - baseline["physical_f1"]) < 1e-10
        and abs(r["metrics"]["mae"] - baseline["mae"]) < 1e-10
        and abs(r["metrics"]["pm1"] - baseline["pm1"]) < 1e-10)]
    best = max(eligible, key=lambda r: (r["metrics"]["matched_class_accuracy"],
                                        r["metrics"]["macro_f1"]))
    report = {"dataset": dataset,
              "protocol": "side-aware group features and ordinal head; fit TRAIN; select VAL; no TEST",
              "seed": seed, "projection_dim": int(pca.n_components_),
              "feature_dim": int(X_train.shape[1]),
              "train_groups": int(len(y_train)), "train_matched_groups": int(mask.sum()),
              "val_groups": int(len(y_val)), "baseline_val": baseline,
              "results": rows, "selected_validation": best,
              "elapsed_sec": time.time() - started}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{dataset}_side_aware_ordinal_results_val.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(json.dumps({"dataset": dataset, "selected_validation": best,
                      "report": str(path), "seconds": report["elapsed_sec"]},
                     ensure_ascii=False), flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("953", "depth"), required=True)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    run(args.dataset, args.seed, args.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
