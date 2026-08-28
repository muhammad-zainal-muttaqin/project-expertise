"""Multi-scale member class heads, TRAIN/VAL only."""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import member_head as mh

warnings.filterwarnings(
    "ignore", message=r"`sklearn\.utils\.parallel\.delayed` should be used.*",
    category=UserWarning,
)

OUT = Path("/workspace/cluster_head/artifacts")
SCALES = ("ctx100", "ctx200")


def load_map(dataset, split, tag):
    with np.load(f"/workspace/dino_head/crops/{dataset}/{split}_index.npz",
                 allow_pickle=True) as z:
        stems = z["stem"].astype(str)
        rows = np.asarray(z["row_index"], dtype=np.int64)
    feat = np.load(f"/workspace/multiscale/features/{dataset}/{split}_{tag}_dinofeat.npy",
                   mmap_mode="r")
    if len(feat) != len(stems):
        raise RuntimeError(f"index/feature mismatch for {dataset}/{split}/{tag}")
    return {(str(s), int(r)): np.asarray(feat[i], dtype=np.float32)
            for i, (s, r) in enumerate(zip(stems, rows))}


def make_member_feature(m, base_map, maps):
    f0 = base_map.get((str(m["stem"]), int(m["row_index"])))
    if f0 is None:
        f0 = np.zeros(1536, dtype=np.float32)
    fs = [maps[tag].get((str(m["stem"]), int(m["row_index"])))
          for tag in SCALES]
    fs = [f if f is not None else np.zeros(1536, dtype=np.float32) for f in fs]
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
    return np.concatenate([np.asarray(f0), *fs, p, side, scalars])


def collect(dataset, split):
    records, payload, targets, _prior = mh.harness.build_payload(dataset, split)
    grouped = mh.harness.make_groups(payload, targets, mh.harness.PROFILES[dataset])
    base_map, _ = mh._load_fmap(dataset, split)
    maps = {tag: load_map(dataset, split, tag) for tag in SCALES}
    X, y, group_rows, keys, matched_members = [], [], [], [], 0
    for rec, groups in grouped:
        matches = dict(mh.harness.count.tree_matches(rec, groups))
        for gi, group in enumerate(groups):
            rows = []
            gt = int(rec["bunches"][matches[gi]]["cls"]) if gi in matches else -1
            for m in group["members"]:
                rows.append(len(X)); X.append(make_member_feature(m, base_map, maps)); y.append(gt)
            group_rows.append(rows)
            keys.append(tuple(sorted((int(m["side"]), str(m["stem"]), int(m["row_index"]))
                                     for m in group["members"])))
            if gt >= 0:
                matched_members += len(rows)
    return {"records": records, "payload": payload, "targets": targets,
            "groups": grouped, "X": np.asarray(X, dtype=np.float32),
            "y": np.asarray(y, dtype=np.int64), "group_rows": group_rows,
            "keys": keys, "matched_members": matched_members}


def model_bank(seed):
    yield "ms_logistic", Pipeline([
        ("scale", StandardScaler()), ("pca", PCA(n_components=128, whiten=True, random_state=seed)),
        ("clf", LogisticRegression(C=.15, max_iter=350, solver="lbfgs",
                                    class_weight="balanced", random_state=seed))])
    yield "ms_extra", Pipeline([
        ("scale", StandardScaler()), ("pca", PCA(n_components=96, whiten=True, random_state=seed)),
        ("clf", ExtraTreesClassifier(n_estimators=320, min_samples_leaf=3,
                                     max_features="sqrt", class_weight="balanced",
                                     n_jobs=8, random_state=seed))])


def short(m):
    return {"physical_f1": m["physical_detection"]["f1"],
            "mae": m["counting"]["mae"],
            "pm1": m["counting"]["plus_minus_1_accuracy"],
            "matched_class_accuracy": m["classification"]["matched_class_accuracy"],
            "matched": m["classification"]["matched"],
            "macro_f1": m["classification"]["macro_f1_end_to_end"],
            "per_class_f1": m["classification"]["per_class_f1_end_to_end"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("953", "depth"), required=True)
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()
    started = time.time()
    train, val = collect(args.dataset, "train"), collect(args.dataset, "val")
    mask = train["y"] >= 0
    if mask.sum() < 20 or len(np.unique(train["y"][mask])) < mh.harness.K:
        raise RuntimeError("insufficient matched member labels")
    baseline = mh.harness.evaluate_clusters(val["payload"], val["targets"],
                                            mh.harness.PROFILES[args.dataset])
    detector = np.asarray([np.asarray(g["p"], dtype=np.float32)
                           for _rec, gs in val["groups"] for g in gs])
    detector = np.maximum(detector, 1e-8); detector /= detector.sum(axis=1, keepdims=True)
    results = []
    for name, model in model_bank(args.seed):
        fit_start = time.time()
        model.fit(train["X"][mask], train["y"][mask])
        qmem = np.asarray(model.predict_proba(val["X"]), dtype=np.float32)
        flat = [g for _rec, gs in val["groups"] for g in gs]
        for pooling in ("mean", "max", "top"):
            q = []
            for gi, rows in enumerate(val["group_rows"]):
                z = qmem[rows]
                if pooling == "mean":
                    w = np.asarray([float(m["score"]) for m in flat[gi]["members"]], dtype=np.float32)
                    w /= max(float(w.sum()), 1e-8); q.append((z * w[:, None]).sum(0))
                elif pooling == "max":
                    q.append(z.max(0))
                else:
                    j = int(np.argmax([float(m["score"]) for m in flat[gi]["members"]]))
                    q.append(z[j])
            q = np.asarray(q, dtype=np.float32)
            for alpha in (.10, .20, .30, .45, .60, .80, 1.0):
                z = np.log(detector) + alpha * np.log(np.maximum(q, 1e-8))
                pred = np.argmax(z, axis=1).astype(int)
                pmap = {key: int(cls) for key, cls in zip(val["keys"], pred)}
                m = mh.harness.evaluate_clusters(
                    val["payload"], val["targets"], mh.harness.PROFILES[args.dataset],
                    lambda g, pmap=pmap: pmap[mh.harness_group_key(g)])
                s = short(m)
                s["physical_count_invariant"] = bool(
                    abs(s["physical_f1"] - baseline["physical_detection"]["f1"]) < 1e-10
                    and abs(s["mae"] - baseline["counting"]["mae"]) < 1e-10
                    and abs(s["pm1"] - baseline["counting"]["plus_minus_1_accuracy"]) < 1e-10)
                results.append({"model": name, "pooling": pooling, "alpha": alpha,
                                "metrics": s, "fit_elapsed_sec": time.time() - fit_start})
        joblib.dump(model, OUT / f"{args.dataset}_{name}.joblib", compress=3)
        top = sorted(results, key=lambda r: (r["metrics"]["matched_class_accuracy"],
                                             r["metrics"]["macro_f1"]), reverse=True)[:3]
        print(json.dumps({"dataset": args.dataset, "model": name, "top": top}, ensure_ascii=False), flush=True)
    eligible = [r for r in results if r["metrics"]["physical_count_invariant"]]
    best = max(eligible, key=lambda r: (r["metrics"]["matched_class_accuracy"], r["metrics"]["macro_f1"]))
    report = {"dataset": args.dataset,
              "protocol": "fit multi-scale member TRAIN; pool VAL; no TEST",
              "train": {"members": len(train["X"]), "matched_members": train["matched_members"]},
              "val": {"members": len(val["X"]), "matched_members": val["matched_members"]},
              "baseline_val": short(baseline), "results": results,
              "selected_validation": best, "elapsed_sec": time.time() - started}
    path = OUT / f"{args.dataset}_multiscale_member_results.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"dataset": args.dataset, "selected_validation": best,
                      "report": str(path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
