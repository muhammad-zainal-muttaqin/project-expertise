"""Out-of-fold multi-expert class head, TRAIN/VAL only.

This experiment addresses a specific weakness of the earlier member stack:
its expert probabilities were fitted in-sample before the opinion weights were
selected.  Here each TRAIN tree receives predictions from models that were not
fitted on that tree.  A small meta head is then fitted to those out-of-fold
opinions and evaluated on the untouched VAL trees.

The physical linker, target count, and cluster selection remain frozen.  The
detector probability is retained as an explicit skip path in every meta
feature.  No TEST path is accepted by this module.
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


ROOT = Path("/workspace/cluster_head")
OUT = ROOT / "artifacts"
K = harness.K


def _key(group: dict) -> tuple:
    return tuple(sorted((int(m["side"]), str(m["stem"]), int(m["row_index"]))
                        for m in group["members"]))


def _load_map(dataset: str, split: str, kind: str):
    with np.load(f"/workspace/dino_head/crops/{dataset}/{split}_index.npz",
                 allow_pickle=True) as z:
        stems = z["stem"].astype(str)
        rows = np.asarray(z["row_index"], dtype=np.int64)
    if kind == "base":
        path = f"/workspace/dino_head/features/{dataset}/{split}_dinofeat.npy"
    elif kind == "large":
        path = f"/workspace/dino_head/features_large/{dataset}/{split}_dinolargefeat.npy"
    elif kind in ("ctx100", "ctx200"):
        path = f"/workspace/multiscale/features/{dataset}/{split}_{kind}_dinofeat.npy"
    else:
        raise ValueError(kind)
    feat = np.load(path, mmap_mode="r")
    if len(stems) != len(feat):
        raise RuntimeError(f"feature/index mismatch for {dataset}/{split}/{kind}")
    return {(str(s), int(r)): np.asarray(feat[i], dtype=np.float32)
            for i, (s, r) in enumerate(zip(stems, rows))}, int(feat.shape[1])


def _member_extra(m: dict) -> np.ndarray:
    p = np.asarray(m["p"], dtype=np.float32)
    side = np.zeros(4, dtype=np.float32)
    if 0 <= int(m["side"]) < 4:
        side[int(m["side"])] = 1.0
    scalars = np.asarray([
        float(m["score"]), float(m["cx"]), float(m["cy"]), float(m["w"]),
        float(m["h"]), float(m.get("rank_cx", 0.)), float(m.get("rank_cy", 0.)),
        float(m.get("z_side_x", 0.)), float(m.get("z_side_y", 0.)),
        float(m.get("z_side_area", 0.)), float(m.get("side_count", 1.)),
    ], dtype=np.float32)
    return np.concatenate([p, side, scalars])


def _feature(m: dict, maps: dict[str, dict], dims: dict[str, int], expert: str) -> np.ndarray:
    ident = (str(m["stem"]), int(m["row_index"]))
    if expert.startswith("base"):
        f = maps["base"].get(ident, np.zeros(dims["base"], dtype=np.float32))
        return np.concatenate([f, _member_extra(m)])
    if expert.startswith("large"):
        f = maps["large"].get(ident, np.zeros(dims["large"], dtype=np.float32))
        return np.concatenate([f, _member_extra(m)])
    if expert.startswith("ms"):
        fs = [maps[tag].get(ident, np.zeros(dims[tag], dtype=np.float32))
              for tag in ("base", "ctx100", "ctx200")]
        return np.concatenate([*fs, _member_extra(m)])
    raise ValueError(expert)


def _group_context(group: dict) -> np.ndarray:
    p = np.asarray(group["p"], dtype=np.float32)
    p = np.maximum(p, 1e-8)
    p /= max(float(p.sum()), 1e-8)
    members = group["members"]
    scores = np.asarray([float(m["score"]) for m in members], dtype=np.float32)
    sides = np.zeros(4, dtype=np.float32)
    for m in members:
        if 0 <= int(m["side"]) < 4:
            sides[int(m["side"])] = 1.0
    entropy = float(-(p * np.log(p)).sum())
    sorted_p = np.sort(p)
    margin = float(sorted_p[-1] - sorted_p[-2])
    return np.asarray([
        *np.log(p).tolist(), *p.tolist(), entropy, margin,
        float(group["score"]), float(len(members)), float(scores.max()),
        float(scores.mean()), float(scores.std()), float(scores.min()),
        *sides.tolist(),
    ], dtype=np.float32)


def _collect(dataset: str, split: str, experts: tuple[str, ...]) -> dict:
    if split not in ("train", "val"):
        raise ValueError("this experiment accepts only train or val")
    records, payload, targets, _prior = harness.build_payload(dataset, split)
    groups = harness.make_groups(payload, targets, harness.PROFILES[dataset])
    needed = {"base"}
    if any(x.startswith("large") for x in experts):
        needed.add("large")
    if any(x.startswith("ms") for x in experts):
        needed.update(("ctx100", "ctx200"))
    loaded = {kind: _load_map(dataset, split, kind) for kind in sorted(needed)}
    maps = {kind: value[0] for kind, value in loaded.items()}
    dims = {kind: int(value[1]) for kind, value in loaded.items()}

    rows = {name: [] for name in experts}
    labels = []
    group_rows = {name: [] for name in experts}
    keys, tree_ids, contexts, flat_groups = [], [], [], []
    matched_groups = 0
    for rec, tree_groups in groups:
        matches = dict(harness.count.tree_matches(rec, tree_groups))
        for gi, group in enumerate(tree_groups):
            gt = int(rec["bunches"][matches[gi]]["cls"]) if gi in matches else -1
            labels.append(gt)
            key = _key(group)
            keys.append(key)
            tree_ids.append(str(rec["tree_id"]))
            contexts.append(_group_context(group))
            flat_groups.append(group)
            if gt >= 0:
                matched_groups += 1
            for name in experts:
                idx = []
                for m in group["members"]:
                    idx.append(len(rows[name]))
                    rows[name].append(_feature(m, maps, dims, name))
                group_rows[name].append(idx)
    return {
        "dataset": dataset, "split": split, "records": records,
        "payload": payload, "targets": targets, "groups": groups,
        "flat_groups": flat_groups, "keys": keys, "tree_ids": np.asarray(tree_ids),
        "labels": np.asarray(labels, dtype=np.int64),
        "contexts": np.asarray(contexts, dtype=np.float32),
        "X": {name: np.asarray(value, dtype=np.float32) for name, value in rows.items()},
        "group_rows": group_rows, "matched_groups": matched_groups,
    }


def _model(name: str, seed: int, n_jobs: int):
    if name == "base_extra":
        return Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=128, whiten=True, random_state=seed)),
            ("clf", ExtraTreesClassifier(
                n_estimators=280, min_samples_leaf=3, max_features="sqrt",
                class_weight="balanced", n_jobs=n_jobs, random_state=seed)),
        ])
    if name == "large_hist":
        return Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=96, whiten=True, random_state=seed)),
            ("clf", HistGradientBoostingClassifier(
                max_iter=220, learning_rate=.05, max_leaf_nodes=15,
                l2_regularization=2., random_state=seed)),
        ])
    if name == "ms_extra":
        return Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=96, whiten=True, random_state=seed)),
            ("clf", ExtraTreesClassifier(
                n_estimators=260, min_samples_leaf=3, max_features="sqrt",
                class_weight="balanced", n_jobs=n_jobs, random_state=seed)),
        ])
    raise ValueError(name)


def _pool(q_member: np.ndarray, data: dict, name: str) -> np.ndarray:
    out = []
    for group, rows in zip(data["flat_groups"], data["group_rows"][name]):
        q = q_member[rows]
        if not len(q):
            out.append(np.full(K, 1.0 / K, dtype=np.float32))
            continue
        if name == "base_extra" or name == "ms_extra":
            weights = np.asarray([float(m["score"]) for m in group["members"]],
                                 dtype=np.float32)
            weights /= max(float(weights.sum()), 1e-8)
            out.append((q * weights[:, None]).sum(axis=0))
        else:
            weights = np.asarray([float(m["score"]) for m in group["members"]],
                                 dtype=np.float32)
            weights /= max(float(weights.sum()), 1e-8)
            out.append((q * weights[:, None]).sum(axis=0))
    q = np.asarray(out, dtype=np.float32)
    q = np.maximum(q, 1e-8)
    return q / np.maximum(q.sum(axis=1, keepdims=True), 1e-8)


def _detector(data: dict) -> np.ndarray:
    q = np.asarray([np.asarray(g["p"], dtype=np.float32)
                    for g in data["flat_groups"]], dtype=np.float32)
    q = np.maximum(q, 1e-8)
    return q / np.maximum(q.sum(axis=1, keepdims=True), 1e-8)


def _meta_features(data: dict, opinions: dict[str, np.ndarray]) -> np.ndarray:
    d = _detector(data)
    blocks = [np.log(d)]
    for name in opinions:
        q = np.maximum(opinions[name], 1e-8)
        q /= np.maximum(q.sum(axis=1, keepdims=True), 1e-8)
        blocks.extend((np.log(q), q - d))
    return np.concatenate([*blocks, data["contexts"]], axis=1).astype(np.float32)


def _fused_predictions(detector: np.ndarray, q: np.ndarray, mode: str,
                       alpha: float) -> np.ndarray:
    if mode == "head":
        z = np.log(np.maximum(q, 1e-8))
    elif mode == "blend":
        z = np.log(np.maximum(detector, 1e-8)) + alpha * np.log(np.maximum(q, 1e-8))
    else:
        raise ValueError(mode)
    return np.argmax(z, axis=1).astype(int)


def _short(m: dict) -> dict:
    return {
        "physical_f1": m["physical_detection"]["f1"],
        "mae": m["counting"]["mae"],
        "pm1": m["counting"]["plus_minus_1_accuracy"],
        "matched_class_accuracy": m["classification"]["matched_class_accuracy"],
        "matched": m["classification"]["matched"],
        "macro_f1": m["classification"]["macro_f1_end_to_end"],
        "per_class_f1": m["classification"]["per_class_f1_end_to_end"],
    }


def _evaluate(data: dict, pred: np.ndarray) -> dict:
    pmap = {key: int(cls) for key, cls in zip(data["keys"], pred)}
    return harness.evaluate_clusters(
        data["payload"], data["targets"], harness.PROFILES[data["dataset"]],
        lambda group, pmap=pmap: pmap[_key(group)])


def run(dataset: str, seed: int, folds: int, n_jobs: int) -> dict:
    started = time.time()
    experts = ("base_extra", "large_hist", "ms_extra")
    train = _collect(dataset, "train", experts)
    val = _collect(dataset, "val", experts)
    y_group = train["labels"]
    train_trees = np.asarray(sorted(set(train["tree_ids"])), dtype=str)
    if len(train_trees) < folds:
        raise RuntimeError("fewer TRAIN trees than requested folds")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(train_trees)
    fold_for_tree = {tree: i % folds for i, tree in enumerate(perm)}
    fold_id = np.asarray([fold_for_tree[t] for t in train["tree_ids"]], dtype=int)
    oof = {name: np.zeros((len(y_group), K), dtype=np.float32) for name in experts}
    fold_reports = []
    for fi in range(folds):
        hold_groups = np.flatnonzero(fold_id == fi)
        fit_groups = np.flatnonzero(fold_id != fi)
        fit_groups = fit_groups[y_group[fit_groups] >= 0]
        hold_member_rows = {
            name: np.asarray([j for gi in hold_groups for j in train["group_rows"][name][gi]],
                             dtype=np.int64)
            for name in experts
        }
        fit_member_rows = {
            name: np.asarray([j for gi in fit_groups for j in train["group_rows"][name][gi]],
                             dtype=np.int64)
            for name in experts
        }
        fold_info = {"fold": fi, "fit_groups": int(len(fit_groups)),
                     "hold_groups": int(len(hold_groups))}
        for name in experts:
            model = _model(name, seed + fi, n_jobs)
            model.fit(train["X"][name][fit_member_rows[name]],
                      np.repeat(y_group[fit_groups],
                                [len(train["group_rows"][name][gi]) for gi in fit_groups]))
            q = model.predict_proba(train["X"][name][hold_member_rows[name]])
            temp = {"flat_groups": [train["flat_groups"][gi] for gi in hold_groups],
                    "group_rows": {name: []}}
            offset = 0
            for gi in hold_groups:
                n = len(train["group_rows"][name][gi])
                temp["group_rows"][name].append(list(range(offset, offset + n)))
                offset += n
            oof[name][hold_groups] = _pool(np.asarray(q, dtype=np.float32), temp, name)
            fold_info[f"{name}_fit_members"] = int(len(fit_member_rows[name]))
        fold_reports.append(fold_info)

    matched_mask = y_group >= 0
    oof_meta = _meta_features(train, oof)
    val_models = {}
    val_opinions = {}
    for name in experts:
        fit_rows = np.asarray([j for gi in np.flatnonzero(matched_mask)
                               for j in train["group_rows"][name][gi]], dtype=np.int64)
        fit_y = np.repeat(y_group[matched_mask],
                          [len(train["group_rows"][name][gi])
                           for gi in np.flatnonzero(matched_mask)])
        model = _model(name, seed, n_jobs)
        model.fit(train["X"][name][fit_rows], fit_y)
        val_opinions[name] = _pool(
            np.asarray(model.predict_proba(val["X"][name]), dtype=np.float32), val, name)
        val_models[name] = model
        joblib.dump(model, OUT / f"{dataset}_oof_{name}.joblib", compress=3)

    # The meta bank is intentionally small and fixed before VAL is read.
    meta_specs = [
        ("logistic_c030", LogisticRegression(C=.30, max_iter=800,
                                               class_weight="balanced",
                                               solver="lbfgs", random_state=seed)),
        ("logistic_c010", LogisticRegression(C=.10, max_iter=800,
                                               class_weight="balanced",
                                               solver="lbfgs", random_state=seed)),
        ("hist_meta", HistGradientBoostingClassifier(
            max_iter=180, learning_rate=.05, max_leaf_nodes=7,
            l2_regularization=3., random_state=seed)),
    ]
    baseline = _short(harness.evaluate_clusters(
        val["payload"], val["targets"], harness.PROFILES[dataset]))
    val_meta = _meta_features(val, val_opinions)
    detector_val = _detector(val)
    rows = []
    for name, meta in meta_specs:
        meta.fit(oof_meta[matched_mask], y_group[matched_mask])
        qmeta_train = np.asarray(meta.predict_proba(oof_meta), dtype=np.float32)
        qmeta_val = np.asarray(meta.predict_proba(val_meta), dtype=np.float32)
        joblib.dump(meta, OUT / f"{dataset}_oof_meta_{name}.joblib", compress=3)
        for mode, alpha in (("head", 1.0), ("blend", .25), ("blend", .50),
                            ("blend", .75), ("blend", 1.0)):
            pred_val = _fused_predictions(detector_val, qmeta_val, mode, alpha)
            metrics = _short(_evaluate(val, pred_val))
            oof_metrics = _short(_evaluate(train, _fused_predictions(
                _detector(train), qmeta_train, mode, alpha)))
            rows.append({"meta_model": name, "mode": mode, "alpha": alpha,
                         "metrics": metrics, "oof_metrics": oof_metrics})

    eligible = [r for r in rows if (
        abs(r["metrics"]["physical_f1"] - baseline["physical_f1"]) < 1e-10
        and abs(r["metrics"]["mae"] - baseline["mae"]) < 1e-10
        and abs(r["metrics"]["pm1"] - baseline["pm1"]) < 1e-10)]
    best = max(eligible, key=lambda r: (r["metrics"]["matched_class_accuracy"],
                                        r["metrics"]["macro_f1"]))
    report = {
        "dataset": dataset,
        "protocol": "group/tree OOF expert fitting on TRAIN; meta selection on VAL; no TEST",
        "seed": seed, "folds": folds, "experts": list(experts),
        "train": {"groups": int(len(train["labels"])),
                  "matched_groups": int(train["matched_groups"]),
                  "trees": int(len(train_trees))},
        "val": {"groups": int(len(val["labels"])),
                "matched_groups": int(val["matched_groups"])},
        "folds_detail": fold_reports, "baseline_val": baseline,
        "results": rows, "selected_validation": best,
        "elapsed_sec": time.time() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{dataset}_oof_expert_stack_results.json"
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
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    run(args.dataset, args.seed, args.folds, args.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
