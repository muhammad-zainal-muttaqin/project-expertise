"""Multi-view member-level class head, validation-only.

Each proposal/member crop is assigned the GT class of its matched TRAIN
cluster.  A classifier is fit on those member examples, then its probabilities
are pooled across the members of each frozen VAL cluster.  The physical
linker, target count, and cluster selection are untouched; TEST is forbidden.
"""
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

import harness

warnings.filterwarnings(
    "ignore", message=r"`sklearn\.utils\.parallel\.delayed` should be used.*",
    category=UserWarning,
)

OUT = Path("/workspace/cluster_head/artifacts")


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
    return np.concatenate([np.asarray(f, dtype=np.float32), p, side, scalars])


def collect(dataset: str, split: str):
    records, payload, targets, _prior = harness.build_payload(dataset, split)
    grouped = harness.make_groups(payload, targets, harness.PROFILES[dataset])
    fmap, dim = _load_fmap(dataset, split)
    all_x, all_labels, group_rows, keys = [], [], [], []
    matched_members = 0
    missing = 0
    for rec, groups in grouped:
        matches = dict(harness.count.tree_matches(rec, groups))
        for gi, group in enumerate(groups):
            indices = []
            gt_cls = int(rec["bunches"][matches[gi]]["cls"]) if gi in matches else -1
            for m in group["members"]:
                key = (str(m["stem"]), int(m["row_index"]))
                if key not in fmap:
                    missing += 1
                indices.append(len(all_x))
                all_x.append(member_feature(m, fmap, dim))
                all_labels.append(gt_cls)
            group_rows.append(indices)
            keys.append(tuple(sorted((int(m["side"]), str(m["stem"]), int(m["row_index"]))
                                     for m in group["members"])))
            if gt_cls >= 0:
                matched_members += len(indices)
    return {"records": records, "payload": payload, "targets": targets,
            "groups": grouped, "X": np.asarray(all_x, dtype=np.float32),
            "y": np.asarray(all_labels, dtype=np.int64), "group_rows": group_rows,
            "keys": keys, "dim": dim, "missing": missing,
            "matched_members": matched_members}


def _load_fmap(dataset: str, split: str):
    with np.load(f"/workspace/dino_head/crops/{dataset}/{split}_index.npz",
                 allow_pickle=True) as z:
        stems = z["stem"].astype(str)
        rows = np.asarray(z["row_index"], dtype=np.int64)
    feat = np.load(f"/workspace/dino_head/features/{dataset}/{split}_dinofeat.npy",
                   mmap_mode="r")
    if len(stems) != len(feat):
        raise RuntimeError("DINO index/feature mismatch")
    return ({(str(s), int(r)): np.asarray(feat[i], dtype=np.float32)
             for i, (s, r) in enumerate(zip(stems, rows))}, int(feat.shape[1]))


def models(seed: int):
    yield "member_logistic", Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=128, whiten=True, random_state=seed)),
        ("clf", LogisticRegression(C=.15, max_iter=300, solver="lbfgs",
                                    class_weight="balanced", random_state=seed)),
    ])
    yield "member_extra", Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=96, whiten=True, random_state=seed)),
        ("clf", ExtraTreesClassifier(n_estimators=320, min_samples_leaf=3,
                                     max_features="sqrt", class_weight="balanced",
                                     n_jobs=8, random_state=seed)),
    ])


def short(m):
    return {"physical_f1": m["physical_detection"]["f1"],
            "mae": m["counting"]["mae"],
            "pm1": m["counting"]["plus_minus_1_accuracy"],
            "matched_class_accuracy": m["classification"]["matched_class_accuracy"],
            "matched": m["classification"]["matched"],
            "macro_f1": m["classification"]["macro_f1_end_to_end"],
            "per_class_f1": m["classification"]["per_class_f1_end_to_end"]}


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("953", "depth"), required=True)
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()
    started = time.time()
    train, val = collect(args.dataset, "train"), collect(args.dataset, "val")
    mask = train["y"] >= 0
    if mask.sum() < 20 or len(np.unique(train["y"][mask])) < harness.K:
        raise RuntimeError("insufficient matched member labels")
    baseline = harness.evaluate_clusters(val["payload"], val["targets"],
                                         harness.PROFILES[args.dataset])
    detector = np.asarray([np.asarray(g["p"], dtype=np.float32)
                           for _rec, gs in val["groups"] for g in gs])
    detector = np.maximum(detector, 1e-8)
    detector /= detector.sum(axis=1, keepdims=True)
    results = []
    for name, model in models(args.seed):
        fit_start = time.time()
        model.fit(train["X"][mask], train["y"][mask])
        q_member = np.asarray(model.predict_proba(val["X"]), dtype=np.float32)
        # The above group index is global; build the flattened group list once
        # and use score-weighted and max-member pooling variants.
        flat_groups = [g for _rec, gs in val["groups"] for g in gs]
        rows_out = []
        for gi, rows in enumerate(val["group_rows"]):
            q = q_member[rows]
            scores = np.asarray([float(m["score"]) for m in flat_groups[gi]["members"]], dtype=np.float32)
            scores /= max(float(scores.sum()), 1e-8)
            mean_q = (q * scores[:, None]).sum(axis=0)
            max_q = q.max(axis=0)
            top_q = q[int(np.argmax([float(m["score"]) for m in flat_groups[gi]["members"]]))]
            rows_out.append((mean_q, max_q, top_q))
        for pooling in ("mean", "max", "top"):
            q = np.asarray([x[{"mean": 0, "max": 1, "top": 2}[pooling]] for x in rows_out])
            for alpha in (.10, .20, .30, .45, .60, .80, 1.0):
                probs = softmax(np.log(detector) + alpha * np.log(np.maximum(q, 1e-8)))
                pred = np.argmax(probs, axis=1).astype(int)
                pmap = {key: int(cls) for key, cls in zip(val["keys"], pred)}
                met = harness.evaluate_clusters(val["payload"], val["targets"],
                                                 harness.PROFILES[args.dataset],
                                                 lambda g, pmap=pmap: pmap[harness_group_key(g)])
                results.append({"model": name, "pooling": pooling, "alpha": alpha,
                                "metrics": short(met),
                                "fit_elapsed_sec": time.time() - fit_start})
        OUT.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, OUT / f"{args.dataset}_{name}.joblib", compress=3)
        top = sorted(results, key=lambda r: (r["metrics"]["matched_class_accuracy"],
                                             r["metrics"]["macro_f1"]), reverse=True)[:3]
        print(json.dumps({"dataset": args.dataset, "model": name, "top": top},
                         ensure_ascii=False), flush=True)
    eligible = [r for r in results if (
        abs(r["metrics"]["physical_f1"] - baseline["physical_detection"]["f1"]) < 1e-10
        and abs(r["metrics"]["mae"] - baseline["counting"]["mae"]) < 1e-10
        and abs(r["metrics"]["pm1"] - baseline["counting"]["plus_minus_1_accuracy"]) < 1e-10)]
    best = max(eligible, key=lambda r: (r["metrics"]["matched_class_accuracy"],
                                        r["metrics"]["macro_f1"]))
    report = {"dataset": args.dataset,
              "protocol": "fit matched TRAIN members; pool frozen VAL clusters; no TEST",
              "train": {"members": len(train["X"]), "matched_members": train["matched_members"],
                        "missing": train["missing"]},
              "val": {"members": len(val["X"]), "matched_members": val["matched_members"],
                      "missing": val["missing"]},
              "baseline_val": short(baseline), "results": results,
              "selected_validation": best, "elapsed_sec": time.time() - started}
    path = OUT / f"{args.dataset}_member_head_results.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"dataset": args.dataset, "selected_validation": best,
                      "report": str(path)}, ensure_ascii=False), flush=True)


def harness_group_key(group):
    return tuple(sorted((int(m["side"]), str(m["stem"]), int(m["row_index"]))
                        for m in group["members"]))


if __name__ == "__main__":
    main()
