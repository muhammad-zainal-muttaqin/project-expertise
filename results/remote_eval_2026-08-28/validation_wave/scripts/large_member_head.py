"""DINOv2-Large member head, TRAIN/VAL only.

The physical linker and count layer are inherited from ``harness`` and stay
frozen.  This is a single stronger-backbone ablation over the same proposal
identities as the established DINOv2-Base branch.
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


OUT = Path("/workspace/cluster_head/artifacts")
K = harness.K


def key(m):
    return tuple(sorted((int(m["side"]), str(m["stem"]), int(m["row_index"]))
                        for m in m["members"]))


def load_fmap(dataset: str, split: str):
    with np.load(f"/workspace/dino_head/crops/{dataset}/{split}_index.npz",
                 allow_pickle=True) as z:
        stems = z["stem"].astype(str)
        rows = np.asarray(z["row_index"], dtype=np.int64)
    feat = np.load(f"/workspace/dino_head/features_large/{dataset}/{split}_dinolargefeat.npy",
                   mmap_mode="r")
    if len(stems) != len(feat):
        raise RuntimeError(f"DINO-Large index mismatch {dataset}/{split}")
    return {(str(s), int(r)): np.asarray(feat[i], dtype=np.float32)
            for i, (s, r) in enumerate(zip(stems, rows))}, int(feat.shape[1])


def member_feature(m: dict, fmap: dict, dim: int) -> np.ndarray:
    f = fmap.get((str(m["stem"]), int(m["row_index"])))
    if f is None:
        f = np.zeros(dim, dtype=np.float32)
    p = np.asarray(m["p"], dtype=np.float32)
    side = np.zeros(4, dtype=np.float32)
    if 0 <= int(m["side"]) < 4:
        side[int(m["side"])] = 1.
    scalars = np.asarray([
        float(m["score"]), float(m["cx"]), float(m["cy"]), float(m["w"]),
        float(m["h"]), float(m.get("rank_cx", 0.)), float(m.get("rank_cy", 0.)),
        float(m.get("z_side_x", 0.)), float(m.get("z_side_y", 0.)),
        float(m.get("z_side_area", 0.)), float(m.get("side_count", 1.)),
    ], dtype=np.float32)
    return np.concatenate([f, p, side, scalars])


def collect(dataset: str, split: str):
    records, payload, targets, _prior = harness.build_payload(dataset, split)
    groups = harness.make_groups(payload, targets, harness.PROFILES[dataset])
    fmap, dim = load_fmap(dataset, split)
    x, y, group_rows, keys = [], [], [], []
    matched_members = 0
    for rec, gs in groups:
        matches = dict(harness.count.tree_matches(rec, gs))
        for gi, group in enumerate(gs):
            gt = int(rec["bunches"][matches[gi]]["cls"]) if gi in matches else -1
            rows = []
            for m in group["members"]:
                rows.append(len(x)); x.append(member_feature(m, fmap, dim)); y.append(gt)
            group_rows.append(rows)
            keys.append(key(group))
            if gt >= 0:
                matched_members += len(rows)
    return {"records": records, "payload": payload, "targets": targets,
            "groups": groups, "X": np.asarray(x, dtype=np.float32),
            "y": np.asarray(y, dtype=np.int64), "group_rows": group_rows,
            "keys": keys, "matched_members": matched_members, "dim": dim}


def pool(q, data, pooling):
    flat = [g for _rec, gs in data["groups"] for g in gs]
    out = []
    for gi, rows in enumerate(data["group_rows"]):
        z = q[rows]
        if pooling == "mean":
            w = np.asarray([float(m["score"]) for m in flat[gi]["members"]], dtype=np.float32)
            w /= max(float(w.sum()), 1e-8)
            out.append((z * w[:, None]).sum(axis=0))
        elif pooling == "max":
            out.append(z.max(axis=0))
        elif pooling == "top":
            j = int(np.argmax([float(m["score"]) for m in flat[gi]["members"]]))
            out.append(z[j])
        else:
            raise ValueError(pooling)
    return np.asarray(out, dtype=np.float32)


def models(seed):
    yield "large_logistic", Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=160, whiten=True, random_state=seed)),
        ("clf", LogisticRegression(C=.15, max_iter=400, solver="lbfgs",
                                    class_weight="balanced", random_state=seed)),
    ])
    yield "large_extra", Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=128, whiten=True, random_state=seed)),
        ("clf", ExtraTreesClassifier(n_estimators=420, min_samples_leaf=3,
                                     max_features="sqrt", class_weight="balanced",
                                     n_jobs=8, random_state=seed)),
    ])
    yield "large_hist", Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=96, whiten=True, random_state=seed)),
        ("clf", HistGradientBoostingClassifier(max_iter=220, learning_rate=.05,
                                                max_leaf_nodes=15,
                                                l2_regularization=2., random_state=seed)),
    ])


def short(m):
    return {"physical_f1": m["physical_detection"]["f1"],
            "mae": m["counting"]["mae"],
            "pm1": m["counting"]["plus_minus_1_accuracy"],
            "matched_class_accuracy": m["classification"]["matched_class_accuracy"],
            "matched": m["classification"]["matched"],
            "macro_f1": m["classification"]["macro_f1_end_to_end"],
            "per_class_f1": m["classification"]["per_class_f1_end_to_end"]}


def run(dataset: str, seed: int):
    started = time.time()
    train, val = collect(dataset, "train"), collect(dataset, "val")
    mask = train["y"] >= 0
    baseline_m = harness.evaluate_clusters(val["payload"], val["targets"],
                                            harness.PROFILES[dataset])
    baseline = short(baseline_m)
    detector = np.asarray([np.asarray(g["p"], dtype=np.float32)
                           for _rec, gs in val["groups"] for g in gs])
    detector = np.maximum(detector, 1e-8)
    detector /= detector.sum(axis=1, keepdims=True)
    rows = []
    for name, model in models(seed):
        t0 = time.time()
        model.fit(train["X"][mask], train["y"][mask])
        q_member = np.asarray(model.predict_proba(val["X"]), dtype=np.float32)
        for pooling in ("mean", "max", "top"):
            q = pool(q_member, val, pooling)
            for alpha in (.05, .10, .15, .20, .30, .45, .60, .80, 1.0):
                probs = np.exp(np.log(detector) + alpha * np.log(np.maximum(q, 1e-8)))
                pred = np.argmax(probs, axis=1).astype(int)
                pmap = {k: int(c) for k, c in zip(val["keys"], pred)}
                m = harness.evaluate_clusters(
                    val["payload"], val["targets"], harness.PROFILES[dataset],
                    lambda g, pmap=pmap: pmap[key(g)])
                s = short(m)
                s["physical_count_invariant"] = bool(
                    abs(s["physical_f1"] - baseline["physical_f1"]) < 1e-10
                    and abs(s["mae"] - baseline["mae"]) < 1e-10
                    and abs(s["pm1"] - baseline["pm1"]) < 1e-10)
                rows.append({"model": name, "pooling": pooling, "alpha": alpha,
                             "metrics": s})
        joblib.dump(model, OUT / f"{dataset}_{name}.joblib", compress=3)
        print(json.dumps({"dataset": dataset, "model": name,
                          "elapsed_sec": time.time() - t0,
                          "top": sorted(rows, key=lambda r:(r["metrics"]["matched_class_accuracy"],
                                                              r["metrics"]["macro_f1"]),
                                         reverse=True)[:3]}, ensure_ascii=False), flush=True)
    eligible = [r for r in rows if r["metrics"]["physical_count_invariant"]]
    best = max(eligible, key=lambda r:(r["metrics"]["matched_class_accuracy"],
                                       r["metrics"]["macro_f1"]))
    best_macro = max(eligible, key=lambda r:(r["metrics"]["macro_f1"],
                                              r["metrics"]["matched_class_accuracy"]))
    report = {"dataset": dataset,
              "protocol": "fit DINOv2-Large member head on matched TRAIN; select VAL; no TEST",
              "feature_dim": train["dim"],
              "train": {"members": int(len(train["X"])),
                        "matched_members": int(train["matched_members"])},
              "val": {"members": int(len(val["X"])),
                      "matched_members": int(val["matched_members"])},
              "baseline_val": baseline, "best_by_matched": best,
              "best_by_macro": best_macro, "results": rows,
              "elapsed_sec": time.time() - started}
    out = OUT / f"{dataset}_large_member_head_results.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({"dataset": dataset, "best_by_matched": best,
                      "best_by_macro": best_macro, "report": str(out)},
                     ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("953", "depth"), required=True)
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()
    run(args.dataset, args.seed)


if __name__ == "__main__":
    main()
