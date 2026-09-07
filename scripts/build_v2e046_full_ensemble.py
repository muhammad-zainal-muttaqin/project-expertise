"""V2-E-046 penuh: ansambel 4 pakar (DINOv2-Large + Base + Multiscale + detektor).

Melanjutkan build_v2e046_dinov2_stacker.py (yang hanya mereplikasi
large_member_head.py, satu cabang dari empat) dengan menambah tiga cabang
lagi persis seperti large_stacker.py:
  - q_extra, q_logistic: DINOv2-Base member_head.py (pooling "max")
  - q_large: DINOv2-Large large_member_head.py (pooling "mean") -- model
    large_hist yang menang pada eksplorasi awal
  - q_ms: Multiscale (base+ctx100+ctx200 digabung) multiscale_member_head.py
    (pooling "mean", model ms_extra)
Digabung lewat blend log-probabilitas berbobot 4-dimensi, persis grid
large_stacker.py (large_w x base_w x log_w x ms_w = 1200 kombinasi),
mengambil yang terbaik dengan physical_f1 tak berubah dari baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, "/workspace/project-expertise/scripts")
import eval_remote_pipeline_postprocess as base_mod  # noqa: E402
import sweep_remote_pipeline as sweep  # noqa: E402
import evaluate_remote_count_reconciled as rec_mod  # noqa: E402
from build_v2e046_dinov2_stacker import (  # noqa: E402
    PROFILE, VOTE_DIR, prepare_payload_with_stem, evaluate_clusters,
    key_of, tree_matches, member_feature as member_feature_large, load_fmap as load_fmap_large,
)

K = 4
SEED = 20260828


def load_fmap_generic(path_npy: str):
    with np.load("/workspace/dino_head/crops/953/train_index.npz", allow_pickle=True) as z:
        pass  # placeholder, real per-split loading below


def load_fmap(dataset, split, feat_path):
    with np.load(f"/workspace/dino_head/crops/{dataset}/{split}_index.npz", allow_pickle=True) as z:
        stems = z["stem"].astype(str)
        rows = np.asarray(z["row_index"], dtype=np.int64)
    feat = np.load(feat_path, mmap_mode="r")
    return ({(str(s), int(r)): np.asarray(feat[i], dtype=np.float32)
             for i, (s, r) in enumerate(zip(stems, rows))}, int(feat.shape[1]))


def member_feature_base(m, fmap, dim):
    f = fmap.get((str(m["stem"]), int(m["row_index"])))
    if f is None:
        f = np.zeros(dim, dtype=np.float32)
    p = np.asarray(m["p"], dtype=np.float32)
    side = np.zeros(4, dtype=np.float32)
    if 0 <= int(m["side"]) < 4:
        side[int(m["side"])] = 1.
    scalars = np.asarray([
        float(m["score"]), float(m["cx"]), float(m["cy"]), float(m["w"]), float(m["h"]),
        float(m.get("rank_cx", 0.)), float(m.get("rank_cy", 0.)),
        float(m.get("z_side_x", 0.)), float(m.get("z_side_y", 0.)),
        float(m.get("z_side_area", 0.)), float(m.get("side_count", 1.)),
    ], dtype=np.float32)
    return np.concatenate([np.asarray(f, dtype=np.float32), p, side, scalars])


def member_feature_ms(m, base_map, ctx100_map, ctx200_map):
    f0 = base_map.get((str(m["stem"]), int(m["row_index"])))
    if f0 is None:
        f0 = np.zeros(1536, dtype=np.float32)
    f1 = ctx100_map.get((str(m["stem"]), int(m["row_index"])))
    f1 = f1 if f1 is not None else np.zeros(1536, dtype=np.float32)
    f2 = ctx200_map.get((str(m["stem"]), int(m["row_index"])))
    f2 = f2 if f2 is not None else np.zeros(1536, dtype=np.float32)
    p = np.asarray(m["p"], dtype=np.float32)
    side = np.zeros(4, dtype=np.float32)
    if 0 <= int(m["side"]) < 4:
        side[int(m["side"])] = 1.
    scalars = np.asarray([
        float(m["score"]), float(m["cx"]), float(m["cy"]), float(m["w"]), float(m["h"]),
        float(m.get("rank_cx", 0.)), float(m.get("rank_cy", 0.)),
        float(m.get("z_side_x", 0.)), float(m.get("z_side_y", 0.)),
        float(m.get("z_side_area", 0.)), float(m.get("side_count", 1.)),
    ], dtype=np.float32)
    return np.concatenate([np.asarray(f0), np.asarray(f1), np.asarray(f2), p, side, scalars])


def pool(q, groups_flat, group_rows, pooling):
    out = []
    for gi, rows in enumerate(group_rows):
        z = q[rows]
        members = groups_flat[gi]["members"]
        if pooling == "mean":
            w = np.asarray([float(m["score"]) for m in members], dtype=np.float32)
            w /= max(float(w.sum()), 1e-8)
            out.append((z * w[:, None]).sum(axis=0))
        elif pooling == "max":
            out.append(z.max(axis=0))
        else:
            j = int(np.argmax([float(m["score"]) for m in members]))
            out.append(z[j])
    return np.asarray(out, dtype=np.float32)


def main():
    dataset_name = "SawitMVC-YOLO"
    cfg = base_mod.CONFIGS[dataset_name]
    train_records = rec_mod.four_side(base_mod.load_records(cfg, "train"))
    val_records = rec_mod.four_side(base_mod.load_records(cfg, "val"))
    prior = base_mod.build_rotation_prior(base_mod.load_records(cfg, "train"))
    train_vote = rec_mod.load_vote(VOTE_DIR["train"])
    val_vote = rec_mod.load_vote(VOTE_DIR["val"])

    class_prior = np.bincount(
        [b["cls"] for r in train_records.values() for b in r["bunches"] if 0 <= b["cls"] < K],
        minlength=K).astype(float)
    class_prior /= max(float(class_prior.sum()), 1.)
    profile = {**PROFILE, "class_prior": class_prior}

    y_train_count = np.asarray([rec_mod.target_count(r) for r in train_records.values()], float)
    X_count_train = np.stack([rec_mod.feature_vector(r, train_vote, PROFILE["proposal_min"])
                              for r in train_records.values()])
    alpha, cv = rec_mod.choose_alpha(X_count_train, y_train_count)
    count_model = rec_mod.fit_ridge(X_count_train, y_train_count, alpha)

    train_payload, train_targets = prepare_payload_with_stem(
        train_records, train_vote, prior, PROFILE["proposal_min"], PROFILE["pair_mode"], count_model)
    val_payload, val_targets = prepare_payload_with_stem(
        val_records, val_vote, prior, PROFILE["proposal_min"], PROFILE["pair_mode"], count_model)

    baseline = evaluate_clusters(val_payload, val_targets, profile)
    print(f"baseline: f1={baseline['physical_f1']:.4f} acc={baseline['matched_class_accuracy']:.4f}", flush=True)

    def collect_groups(payload, targets):
        groups_flat, group_rows, y, keys = [], [], [], []
        for r, dets, edges in payload:
            groups = rec_mod.selected_clusters(
                dets, edges, PROFILE["link_threshold"], PROFILE["singleton_min"],
                PROFILE["max_size"], targets[r["tree_id"]], PROFILE["rank_mode"])
            matches = dict(tree_matches(r, groups))
            for gi, g in enumerate(groups):
                gt = int(r["bunches"][matches[gi]]["cls"]) if gi in matches else -1
                y.append((gt, r, gi))
                groups_flat.append(g)
                keys.append(key_of(g))
        return groups_flat, keys, y

    train_groups, train_keys, train_y_meta = collect_groups(train_payload, train_targets)
    val_groups, val_keys, val_y_meta = collect_groups(val_payload, val_targets)

    def member_rows(groups_flat):
        group_rows = []
        idx = 0
        starts = []
        for g in groups_flat:
            n = len(g["members"])
            starts.append((idx, idx + n))
            idx += n
        return starts

    # ---- fitur 1: DINOv2-Large (dari skrip sebelumnya) ----
    fmap_l_tr, dim_l = load_fmap_large("953", "train")
    fmap_l_va, _ = load_fmap_large("953", "val")
    # ---- fitur 2/3: DINOv2-Base ----
    fmap_b_tr, dim_b = load_fmap("953", "train", "/workspace/dino_head/features/953/train_dinofeat.npy")
    fmap_b_va, _ = load_fmap("953", "val", "/workspace/dino_head/features/953/val_dinofeat.npy")
    # ---- fitur 4: Multiscale ----
    fmap_ctx100_tr, _ = load_fmap("953", "train", "/workspace/multiscale/features/953/train_ctx100_dinofeat.npy")
    fmap_ctx200_tr, _ = load_fmap("953", "train", "/workspace/multiscale/features/953/train_ctx200_dinofeat.npy")
    fmap_ctx100_va, _ = load_fmap("953", "val", "/workspace/multiscale/features/953/val_ctx100_dinofeat.npy")
    fmap_ctx200_va, _ = load_fmap("953", "val", "/workspace/multiscale/features/953/val_ctx200_dinofeat.npy")

    def build_member_arrays(groups_flat, y_meta, feature_fn):
        X, y, rows = [], [], []
        for g, (gt, _r, _gi) in zip(groups_flat, y_meta):
            idxs = []
            for m in g["members"]:
                idxs.append(len(X)); X.append(feature_fn(m)); y.append(gt)
            rows.append(idxs)
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), rows

    Xl_tr, yl_tr, _ = build_member_arrays(train_groups, train_y_meta, lambda m: member_feature_large(m, fmap_l_tr, dim_l))
    Xl_va, _, rows_va = build_member_arrays(val_groups, val_y_meta, lambda m: member_feature_large(m, fmap_l_va, dim_l))
    Xb_tr, yb_tr, _ = build_member_arrays(train_groups, train_y_meta, lambda m: member_feature_base(m, fmap_b_tr, dim_b))
    Xb_va, _, _ = build_member_arrays(val_groups, val_y_meta, lambda m: member_feature_base(m, fmap_b_va, dim_b))
    Xms_tr, yms_tr, _ = build_member_arrays(train_groups, train_y_meta, lambda m: member_feature_ms(m, fmap_b_tr, fmap_ctx100_tr, fmap_ctx200_tr))
    Xms_va, _, _ = build_member_arrays(val_groups, val_y_meta, lambda m: member_feature_ms(m, fmap_b_va, fmap_ctx100_va, fmap_ctx200_va))

    mask_l = yl_tr >= 0
    mask_b = yb_tr >= 0
    mask_ms = yms_tr >= 0
    print(f"members: large={len(Xl_tr)} base={len(Xb_tr)} ms={len(Xms_tr)}  matched={int(mask_l.sum())}", flush=True)

    # ---- latih 4 model ----
    print("fit large_hist ...", flush=True)
    large_hist = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=96, whiten=True, random_state=SEED)),
                           ("clf", HistGradientBoostingClassifier(max_iter=220, learning_rate=.05, max_leaf_nodes=15,
                                                                   l2_regularization=2., random_state=SEED))])
    large_hist.fit(Xl_tr[mask_l], yl_tr[mask_l])
    q_large_member = np.asarray(large_hist.predict_proba(Xl_va), dtype=np.float32)

    print("fit member_extra ...", flush=True)
    member_extra = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=96, whiten=True, random_state=SEED)),
                             ("clf", ExtraTreesClassifier(n_estimators=320, min_samples_leaf=3, max_features="sqrt",
                                                          class_weight="balanced", n_jobs=8, random_state=SEED))])
    member_extra.fit(Xb_tr[mask_b], yb_tr[mask_b])
    q_extra_member = np.asarray(member_extra.predict_proba(Xb_va), dtype=np.float32)

    print("fit member_logistic ...", flush=True)
    member_logistic = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=128, whiten=True, random_state=SEED)),
                                ("clf", LogisticRegression(C=.15, max_iter=300, solver="lbfgs",
                                                            class_weight="balanced", random_state=SEED))])
    member_logistic.fit(Xb_tr[mask_b], yb_tr[mask_b])
    q_logistic_member = np.asarray(member_logistic.predict_proba(Xb_va), dtype=np.float32)

    print("fit ms_extra ...", flush=True)
    ms_extra = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=96, whiten=True, random_state=SEED)),
                         ("clf", ExtraTreesClassifier(n_estimators=320, min_samples_leaf=3, max_features="sqrt",
                                                      class_weight="balanced", n_jobs=8, random_state=SEED))])
    ms_extra.fit(Xms_tr[mask_ms], yms_tr[mask_ms])
    q_ms_member = np.asarray(ms_extra.predict_proba(Xms_va), dtype=np.float32)

    q_large = pool(q_large_member, val_groups, rows_va, "mean")
    q_extra = pool(q_extra_member, val_groups, rows_va, "max")
    q_logistic = pool(q_logistic_member, val_groups, rows_va, "max")
    q_ms = pool(q_ms_member, val_groups, rows_va, "mean")

    detector = np.asarray([np.asarray(g["p"], dtype=np.float32) for g in val_groups])
    detector = np.maximum(detector, 1e-8)
    detector /= detector.sum(axis=1, keepdims=True)

    large_w = (0., .05, .10, .15, .20, .30, .45, .60, .80, 1.0)
    base_w = (0., .05, .10, .20, .30, .45)
    log_w = (0., .05, .10, .15, .20)
    ms_w = (0., .05, .10, .15)

    best = None
    all_results = []
    for wl in large_w:
        for we in base_w:
            for wx in log_w:
                for wm in ms_w:
                    z = (np.log(detector) + wl * np.log(np.maximum(q_large, 1e-8))
                         + we * np.log(np.maximum(q_extra, 1e-8))
                         + wx * np.log(np.maximum(q_logistic, 1e-8))
                         + wm * np.log(np.maximum(q_ms, 1e-8)))
                    pred = np.argmax(z, axis=1).astype(int)
                    pmap = {k: int(c) for k, c in zip(val_keys, pred)}
                    m = evaluate_clusters(val_payload, val_targets, profile,
                                          class_fn=lambda g, pmap=pmap: pmap[key_of(g)])
                    count_invariant = abs(m["physical_f1"] - baseline["physical_f1"]) < 1e-9
                    row = {"large_w": wl, "base_w": we, "log_w": wx, "ms_w": wm,
                           "matched_class_accuracy": m["matched_class_accuracy"],
                           "physical_f1": m["physical_f1"], "count_invariant": count_invariant,
                           "confusion_matrix": m["confusion_matrix"]}
                    all_results.append(row)
                    if count_invariant and (best is None or
                            row["matched_class_accuracy"] > best["matched_class_accuracy"]):
                        best = row

    print(f"grid size={len(all_results)}", flush=True)
    print(json.dumps({"baseline": baseline, "best": {k: v for k, v in best.items() if k != 'confusion_matrix'}}, indent=2))
    Path("/workspace/project-expertise/results/v2e046_full_ensemble_2026-09-07.json").write_text(
        json.dumps({"baseline": baseline, "best": best, "grid_size": len(all_results),
                    "profile": PROFILE}, indent=2))
    print("saved -> results/v2e046_full_ensemble_2026-09-07.json")


if __name__ == "__main__":
    main()
