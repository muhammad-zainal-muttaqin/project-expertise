"""DINOv2-Large nearest-neighbor/prototype opinions, TRAIN/VAL only.

This is a deliberately different expert from the tree-based crop heads.  It
uses cosine similarity in the frozen DINOv2-Large embedding space, pools
member opinions across each frozen cluster, and tests only a small residual
weight on top of the best existing stack.  The detector probability remains
the anchor and no TEST split is accepted.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

import harness
import large_member_head as large
import member_head as mh
import multiscale_member_head as ms


OUT = Path("/workspace/cluster_head/artifacts")
K = harness.K


def group_key(group: dict) -> tuple:
    return tuple(sorted((int(m["side"]), str(m["stem"]), int(m["row_index"]))
                        for m in group["members"]))


def pool(q_member: np.ndarray, data: dict, pooling: str) -> np.ndarray:
    out = []
    for group, rows in zip(data["groups_flat"], data["group_rows"]):
        q = q_member[rows]
        if pooling == "mean":
            w = np.asarray([float(m["score"]) for m in group["members"]], dtype=np.float32)
            w /= max(float(w.sum()), 1e-8)
            out.append((q * w[:, None]).sum(axis=0))
        elif pooling == "max":
            out.append(q.max(axis=0))
        elif pooling == "top":
            j = int(np.argmax([float(m["score"]) for m in group["members"]]))
            out.append(q[j])
        else:
            raise ValueError(pooling)
    q = np.maximum(np.asarray(out, dtype=np.float32), 1e-8)
    return q / np.maximum(q.sum(axis=1, keepdims=True), 1e-8)


def adapt(data: dict) -> dict:
    return {"groups_flat": [g for _rec, gs in data["groups"] for g in gs],
            "group_rows": data["group_rows"]}


def detector(data: dict) -> np.ndarray:
    q = np.asarray([np.asarray(g["p"], dtype=np.float32)
                    for g in data["groups_flat"]], dtype=np.float32)
    q = np.maximum(q, 1e-8)
    return q / np.maximum(q.sum(axis=1, keepdims=True), 1e-8)


def evaluate(data: dict, pred: np.ndarray) -> dict:
    pmap = {key: int(cls) for key, cls in zip(data["keys"], pred)}
    m = harness.evaluate_clusters(
        data["payload"], data["targets"], harness.PROFILES[data["dataset"]],
        lambda group, pmap=pmap: pmap[group_key(group)])
    return {"physical_f1": m["physical_detection"]["f1"],
            "mae": m["counting"]["mae"],
            "pm1": m["counting"]["plus_minus_1_accuracy"],
            "matched_class_accuracy": m["classification"]["matched_class_accuracy"],
            "matched": m["classification"]["matched"],
            "macro_f1": m["classification"]["macro_f1_end_to_end"],
            "per_class_f1": m["classification"]["per_class_f1_end_to_end"]}


def fit_similarity(train: dict, val: dict, dim: int, max_k: int):
    mask = train["y"] >= 0
    xtr = np.asarray(train["X"][mask, :dim], dtype=np.float32)
    ytr = np.asarray(train["y"][mask], dtype=np.int64)
    xva = np.asarray(val["X"][:, :dim], dtype=np.float32)
    xtr /= np.maximum(np.linalg.norm(xtr, axis=1, keepdims=True), 1e-8)
    xva /= np.maximum(np.linalg.norm(xva, axis=1, keepdims=True), 1e-8)
    nn = NearestNeighbors(n_neighbors=max_k, metric="cosine", algorithm="brute",
                          n_jobs=8)
    nn.fit(xtr)
    distances, indices = nn.kneighbors(xva, return_distance=True)
    out = {}
    for weighting in ("uniform", "inverse"):
        for k in (3, 7, 15):
            if k > max_k:
                continue
            d = distances[:, :k]
            labels = ytr[indices[:, :k]]
            if weighting == "uniform":
                w = np.ones_like(d, dtype=np.float32)
            else:
                w = 1. / np.maximum(d, 1e-3)
            q = np.zeros((len(xva), K), dtype=np.float32)
            for c in range(K):
                q[:, c] = (w * (labels == c)).sum(axis=1)
            q = np.maximum(q, 1e-8)
            out[f"knn_{weighting}_{k}"] = q / np.maximum(q.sum(1, keepdims=True), 1e-8)
    centroids = []
    for c in range(K):
        z = xtr[ytr == c].mean(axis=0)
        z /= max(float(np.linalg.norm(z)), 1e-8)
        centroids.append(z)
    sim = xva @ np.asarray(centroids).T
    for temperature in (.05, .10, .20):
        z = sim / temperature
        z -= z.max(axis=1, keepdims=True)
        q = np.exp(z)
        out[f"prototype_t{temperature:.2f}"] = q / np.maximum(q.sum(1, keepdims=True), 1e-8)
    return out, nn


def member_feature_map(data: dict, kind: str):
    if kind == "large":
        return np.asarray(data["X"][:, :data["dim"]], dtype=np.float32)
    raise ValueError(kind)


def run(dataset: str, seed: int) -> dict:
    started = time.time()
    train_large = large.collect(dataset, "train")
    val_large = large.collect(dataset, "val")
    # A second copy of the fixed topology supplies the established experts.
    val_base = mh.collect(dataset, "val")
    val_ms = ms.collect(dataset, "val")
    if val_large["keys"] != val_base["keys"] or val_large["keys"] != val_ms["keys"]:
        raise RuntimeError("fixed topology differs across feature collectors")
    train_data = {**train_large, "y": train_large["y"]}
    val_data = {**val_large, "dataset": dataset,
                "groups_flat": [g for _r, gs in val_large["groups"] for g in gs]}
    train_data["dataset"] = dataset
    train_data["groups_flat"] = [g for _r, gs in train_large["groups"] for g in gs]
    # Build expert opinions from the already fitted TRAIN-only models.
    extra = joblib.load(OUT / f"{dataset}_member_extra.joblib")
    logistic = joblib.load(OUT / f"{dataset}_member_logistic.joblib")
    large_hist = joblib.load(OUT / f"{dataset}_large_hist.joblib")
    ms_extra = joblib.load(OUT / f"{dataset}_ms_extra.joblib")
    q_extra = pool(extra.predict_proba(val_base["X"]),
                   {"groups_flat": [g for _r, gs in val_base["groups"] for g in gs],
                    "group_rows": val_base["group_rows"]}, "max")
    q_log = pool(logistic.predict_proba(val_base["X"]),
                 {"groups_flat": [g for _r, gs in val_base["groups"] for g in gs],
                  "group_rows": val_base["group_rows"]}, "max")
    q_large = pool(large_hist.predict_proba(val_large["X"]),
                   {"groups_flat": val_data["groups_flat"],
                    "group_rows": val_large["group_rows"]}, "mean")
    q_ms = pool(ms_extra.predict_proba(val_ms["X"]),
                {"groups_flat": [g for _r, gs in val_ms["groups"] for g in gs],
                 "group_rows": val_ms["group_rows"]}, "mean")
    d = detector(val_data)
    base_z = (np.log(d) + .10 * np.log(q_large) + .20 * np.log(q_extra)
              + .05 * np.log(q_log) + .10 * np.log(q_ms))
    base_pred = np.argmax(base_z, axis=1).astype(int)
    baseline = evaluate(val_data, base_pred)
    q_sim, nn = fit_similarity(train_large, val_large, train_large["dim"], 15)
    rows = []
    for name, q_member in q_sim.items():
        for pooling in ("mean", "max", "top"):
            q = pool(q_member, {"groups_flat": val_data["groups_flat"],
                                "group_rows": val_large["group_rows"]}, pooling)
            for beta in (.05, .10, .15, .20, .30, .45, .60, 1.0):
                pred = np.argmax(base_z + beta * np.log(q), axis=1).astype(int)
                rows.append({"expert": name, "pooling": pooling, "beta": beta,
                             "metrics": evaluate(val_data, pred)})
    selected = max(rows, key=lambda r: (r["metrics"]["matched_class_accuracy"],
                                        r["metrics"]["macro_f1"]))
    joblib.dump(nn, OUT / f"{dataset}_dino_large_knn.joblib", compress=3)
    report = {"dataset": dataset,
              "protocol": "DINOv2-Large cosine KNN/prototype fit TRAIN; residual stack selected VAL; no TEST",
              "seed": seed, "train_members": int(len(train_large["X"])),
              "train_matched_members": int((train_large["y"] >= 0).sum()),
              "val_members": int(len(val_large["X"])), "embedding_dim": int(train_large["dim"]),
              "baseline_val": baseline, "rows": rows, "selected_validation": selected,
              "model_path": str(OUT / f"{dataset}_dino_large_knn.joblib"),
              "elapsed_sec": time.time() - started}
    path = OUT / f"{dataset}_knn_prototype_results_val.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(json.dumps({"dataset": dataset, "selected_validation": selected,
                      "report": str(path), "seconds": report["elapsed_sec"]},
                     ensure_ascii=False), flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("953", "depth"), required=True)
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()
    run(args.dataset, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
