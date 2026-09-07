"""V2-E-046: WBF[...] + Hungarian Anchor A + stacking DINOv2-Large, 953 val.

harness.py (skrip orkestrasi asli untuk seluruh keluarga eksperimen "validation
wave") tidak bisa dijalankan sesi ini -- ia mengimpor link_global_setpartition
(solver MILP untuk 763-depth) di level modul, dan modul itu tidak ada di
manapun (bukan cuma untuk 763-depth; import-nya unconditional, jadi bahkan
jalur 953 pun gagal memuat). Skrip ini mereplikasi jalur 953 murni dari
fungsi-fungsi yang MASIH tersedia dan sudah tervalidasi sesi ini
(scripts/evaluate_remote_count_reconciled.py, dipakai untuk V2-E-045), plus
menyalin dua bagian yang hilang secara aditif:

  1. sweep_remote_pipeline.make_detections() tidak melacak stem/row_index
     (dibutuhkan untuk mencocokkan fitur DINOv2 per-crop) -- disalin lokal
     di sini dengan tambahan stem/row_index (mengikuti enumerasi baris
     mentah sebelum filter proposal_min, sesuai konvensi
     train_detection_edge_linker.make_detections()) dan medan rank/z-score
     (rank_cx, rank_cy, z_side_x, z_side_y, z_side_area, side_count) --
     formula disalin persis dari train_detection_edge_linker.py.
  2. Logika evaluasi cluster->confusion matrix dengan class assignment
     KUSTOM (bukan class_prior_exponent bawaan) -- direplikasi dari
     evaluate_payload() di evaluate_remote_count_reconciled.py.

Model stacker (3 kandidat: logistic/extra-trees/histgb + PCA), skema fitur
member (DINOv2 2048-d + probabilitas kelas + one-hot sisi + 11 skalar), dan
pooling+blending (mean/max/top x 9 nilai alpha) disalin PERSIS dari
large_member_head.py -- termasuk seed=20260828 bawaannya.

Profil linking 953 (proposal_min=0.125, link_threshold=0.3, singleton_min=0.15,
max_size=3, pair_mode=adjacent, rank_mode=max_member, class_prior_exponent=-0.25)
adalah profil tervalidasi V2-E-045 sesi ini, bukan tebakan.
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
import eval_remote_pipeline_postprocess as base  # noqa: E402
import sweep_remote_pipeline as sweep  # noqa: E402
import evaluate_remote_count_reconciled as rec_mod  # noqa: E402

K = 4
NAMES = ["B1", "B2", "B3", "B4"]
SEED = 20260828

PROFILE = dict(proposal_min=0.125, link_threshold=0.30, singleton_min=0.15,
               max_size=3, pair_mode="adjacent", rank_mode="max_member",
               class_prior_exponent=-0.25)

VOTE_DIR = {
    "train": Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27/fused_combined1716_train/SawitMVC_YOLO__wbf_softvote.npz"),
    "val": Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27/fused_combined1716_val/SawitMVC_YOLO__wbf_softvote.npz"),
}


# ------------------------------------------------------------- deteksi+stem
def make_detections_with_stem(rec, vote, proposal_min):
    from collections import defaultdict
    dets = []
    for side, view in rec["views"].items():
        rows = vote.get(view["stem"], np.zeros((0, 5 + K)))
        for row_index, row in enumerate(rows):
            if row[4] < proposal_min:
                continue
            x1, y1, x2, y2, score = row[:5]
            p = np.zeros(K, float)
            p[:] = np.maximum(row[5:5 + K], 0.)
            p /= max(float(p.sum()), 1e-9)
            width, height = max(view["width"], 1), max(view["height"], 1)
            dets.append({
                "side": int(side), "box": np.asarray([x1, y1, x2, y2], float),
                "score": float(score), "p": p, "head_p": p.copy(),
                "stem": view["stem"], "row_index": int(row_index),
                "cx": float((x1 + x2) / 2 / width), "cy": float((y1 + y2) / 2 / height),
                "w": float(max(x2 - x1, 1.) / width), "h": float(max(y2 - y1, 1.) / height),
            })
    by_side = defaultdict(list)
    for i, det in enumerate(dets):
        by_side[det["side"]].append(i)
    for indices in by_side.values():
        values = np.asarray([[dets[i]["cx"], dets[i]["cy"],
                              np.log(max(dets[i]["w"] * dets[i]["h"], 1e-8))]
                             for i in indices], float)
        for col, key_ in enumerate(("cx", "cy")):
            order = np.argsort(values[:, col], kind="stable")
            ranks = np.empty(len(order), float)
            ranks[order] = np.arange(len(order)) / max(len(order) - 1, 1)
            for j, idx in enumerate(indices):
                dets[idx][f"rank_{key_}"] = float(ranks[j])
        med = np.median(values, axis=0)
        scale = 1.4826 * np.median(np.abs(values - med), axis=0)
        scale = np.maximum(scale, [0.03, 0.03, 0.20])
        for j, idx in enumerate(indices):
            dets[idx]["z_side_x"] = float((values[j, 0] - med[0]) / scale[0])
            dets[idx]["z_side_y"] = float((values[j, 1] - med[1]) / scale[1])
            dets[idx]["z_side_area"] = float((values[j, 2] - med[2]) / scale[2])
            dets[idx]["side_count"] = float(len(indices))
    return dets


def prepare_payload_with_stem(records, vote, prior, proposal_min, pair_mode, count_model):
    payload, count_features = [], {}
    for tree_id, r in records.items():
        dets = make_detections_with_stem(r, vote, proposal_min)
        edges = sweep.build_edges(dets, r["n_sides"], prior, pair_mode)
        count_features[tree_id] = rec_mod.feature_vector(r, vote, proposal_min)
        payload.append((r, dets, edges))
    X = np.stack([count_features[tid] for tid in records])
    counts = rec_mod.predict_count(count_model, X)
    return payload, {tid: int(n) for tid, n in zip(records, counts)}


# -------------------------------------------------------------- fitur member
def key_of(group):
    return tuple(sorted((int(m["side"]), str(m["stem"]), int(m["row_index"]))
                        for m in group["members"]))


def load_fmap(dataset, split):
    with np.load(f"/workspace/dino_head/crops/{dataset}/{split}_index.npz", allow_pickle=True) as z:
        stems = z["stem"].astype(str)
        rows = np.asarray(z["row_index"], dtype=np.int64)
    feat = np.load(f"/workspace/dino_head/features_large/{dataset}/{split}_dinolargefeat.npy", mmap_mode="r")
    return ({(str(s), int(r)): np.asarray(feat[i], dtype=np.float32)
             for i, (s, r) in enumerate(zip(stems, rows))}, int(feat.shape[1]))


def member_feature(m, fmap, dim):
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
    return np.concatenate([f, p, side, scalars])


# --------------------------------------------------------- evaluasi cluster
def tree_matches(rec, groups):
    return rec_mod.tree_matches(rec, groups)


def evaluate_clusters(payload, target_counts, profile, class_fn=None):
    cm = np.zeros((K + 1, K + 1), int)
    total_pred = total_gt = total_tp = 0
    for r, dets, edges in payload:
        target = target_counts[r["tree_id"]]
        groups = rec_mod.selected_clusters(
            dets, edges, profile["link_threshold"], profile["singleton_min"],
            profile["max_size"], target, profile["rank_mode"])
        if class_fn is None:
            exponent = profile["class_prior_exponent"]
            prior = np.maximum(profile.get("class_prior", np.ones(K) / K), 1e-9)
            for g in groups:
                g["cls"] = int(np.argmax(g["p"] * np.power(prior, exponent)))
        else:
            for g in groups:
                g["cls"] = class_fn(g)
        matches = dict(tree_matches(r, groups))
        total_pred += len(groups); total_gt += len(r["bunches"]); total_tp += len(matches)
        matched_pred, matched_gt = set(matches.keys()), set(matches.values())
        for gi, gt_j in matches.items():
            pred_cls, gt_cls = groups[gi]["cls"], r["bunches"][gt_j]["cls"]
            if 0 <= pred_cls < K and 0 <= gt_cls < K:
                cm[pred_cls, gt_cls] += 1
        for gi, g in enumerate(groups):
            if gi not in matched_pred and 0 <= g["cls"] < K:
                cm[g["cls"], K] += 1
        for j, b in enumerate(r["bunches"]):
            if j not in matched_gt and 0 <= b["cls"] < K:
                cm[K, b["cls"]] += 1
    precision = total_tp / max(total_pred, 1)
    recall = total_tp / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    matched = int(cm[:K, :K].sum())
    class_correct = int(np.trace(cm[:K, :K]))
    return {"physical_f1": f1, "precision": precision, "recall": recall,
            "matched": matched, "matched_class_accuracy": class_correct / max(matched, 1),
            "confusion_matrix": cm.tolist()}


def pool(q_member, groups_flat, group_rows, pooling):
    out = []
    for gi, rows in enumerate(group_rows):
        z = q_member[rows]
        if pooling == "mean":
            w = np.asarray([float(m["score"]) for m in groups_flat[gi]["members"]], dtype=np.float32)
            w /= max(float(w.sum()), 1e-8)
            out.append((z * w[:, None]).sum(axis=0))
        elif pooling == "max":
            out.append(z.max(axis=0))
        else:  # top
            j = int(np.argmax([float(m["score"]) for m in groups_flat[gi]["members"]]))
            out.append(z[j])
    return np.asarray(out, dtype=np.float32)


def models(seed):
    yield "large_logistic", Pipeline([
        ("scale", StandardScaler()), ("pca", PCA(n_components=160, whiten=True, random_state=seed)),
        ("clf", LogisticRegression(C=.15, max_iter=400, solver="lbfgs",
                                    class_weight="balanced", random_state=seed)),
    ])
    yield "large_extra", Pipeline([
        ("scale", StandardScaler()), ("pca", PCA(n_components=128, whiten=True, random_state=seed)),
        ("clf", ExtraTreesClassifier(n_estimators=420, min_samples_leaf=3, max_features="sqrt",
                                     class_weight="balanced", n_jobs=8, random_state=seed)),
    ])
    yield "large_hist", Pipeline([
        ("scale", StandardScaler()), ("pca", PCA(n_components=96, whiten=True, random_state=seed)),
        ("clf", HistGradientBoostingClassifier(max_iter=220, learning_rate=.05, max_leaf_nodes=15,
                                                l2_regularization=2., random_state=seed)),
    ])


def main():
    dataset_name = "SawitMVC-YOLO"
    cfg = base.CONFIGS[dataset_name]
    train_records = rec_mod.four_side(base.load_records(cfg, "train"))
    val_records = rec_mod.four_side(base.load_records(cfg, "val"))
    prior = base.build_rotation_prior(base.load_records(cfg, "train"))
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
    print(f"count model alpha={alpha} cv={cv}", flush=True)

    train_payload, train_targets = prepare_payload_with_stem(
        train_records, train_vote, prior, PROFILE["proposal_min"], PROFILE["pair_mode"], count_model)
    val_payload, val_targets = prepare_payload_with_stem(
        val_records, val_vote, prior, PROFILE["proposal_min"], PROFILE["pair_mode"], count_model)

    baseline = evaluate_clusters(val_payload, val_targets, profile)
    print(f"baseline (class_prior_exponent, val): f1={baseline['physical_f1']:.4f} "
          f"matched_class_acc={baseline['matched_class_accuracy']:.4f}", flush=True)

    def collect(payload, targets, fmap, dim):
        x, y, group_rows, groups_flat = [], [], [], []
        for r, dets, edges in payload:
            groups = rec_mod.selected_clusters(
                dets, edges, PROFILE["link_threshold"], PROFILE["singleton_min"],
                PROFILE["max_size"], targets[r["tree_id"]], PROFILE["rank_mode"])
            matches = dict(tree_matches(r, groups))
            for gi, g in enumerate(groups):
                gt = int(r["bunches"][matches[gi]]["cls"]) if gi in matches else -1
                rows = []
                for m in g["members"]:
                    rows.append(len(x)); x.append(member_feature(m, fmap, dim)); y.append(gt)
                group_rows.append(rows)
                groups_flat.append(g)
        return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64), group_rows, groups_flat

    fmap_train, dim = load_fmap("953", "train")
    fmap_val, _ = load_fmap("953", "val")
    Xtr, ytr, _, _ = collect(train_payload, train_targets, fmap_train, dim)
    Xva, yva, group_rows_va, groups_flat_va = collect(val_payload, val_targets, fmap_val, dim)
    mask = ytr >= 0
    print(f"train members={len(Xtr)} matched={int(mask.sum())}  val members={len(Xva)}", flush=True)

    detector = np.asarray([np.asarray(g["p"], dtype=np.float32) for g in groups_flat_va])
    detector = np.maximum(detector, 1e-8)
    detector /= detector.sum(axis=1, keepdims=True)

    all_results = []
    best_overall = None
    for name, model in models(SEED):
        model.fit(Xtr[mask], ytr[mask])
        q_member = np.asarray(model.predict_proba(Xva), dtype=np.float32)
        for pooling in ("mean", "max", "top"):
            q = pool(q_member, groups_flat_va, group_rows_va, pooling)
            for a in (.05, .10, .15, .20, .30, .45, .60, .80, 1.0):
                probs = np.exp(np.log(detector) + a * np.log(np.maximum(q, 1e-8)))
                pred = np.argmax(probs, axis=1).astype(int)
                pmap = {key_of(g): int(c) for g, c in zip(groups_flat_va, pred)}
                m = evaluate_clusters(val_payload, val_targets, profile,
                                      class_fn=lambda g, pmap=pmap: pmap[key_of(g)])
                count_invariant = (abs(m["physical_f1"] - baseline["physical_f1"]) < 1e-9)
                row = {"model": name, "pooling": pooling, "alpha": a,
                       "matched_class_accuracy": m["matched_class_accuracy"],
                       "physical_f1": m["physical_f1"], "count_invariant": count_invariant,
                       "confusion_matrix": m["confusion_matrix"]}
                all_results.append(row)
                if count_invariant and (best_overall is None or
                        row["matched_class_accuracy"] > best_overall["matched_class_accuracy"]):
                    best_overall = row
        print(f"{name} done, best so far: {best_overall['matched_class_accuracy']:.4f} "
              f"({best_overall['pooling']}, alpha={best_overall['alpha']})", flush=True)

    print(json.dumps({"baseline": baseline, "best": {k: v for k, v in best_overall.items()
                                                      if k != 'confusion_matrix'}}, indent=2))
    Path("/workspace/project-expertise/results/v2e046_dinov2_stacker_2026-09-07.json").write_text(
        json.dumps({"baseline": baseline, "best": best_overall,
                    "n_candidates": len(all_results), "profile": PROFILE,
                    "n_train_members": len(Xtr), "n_train_matched": int(mask.sum()),
                    "n_val_members": len(Xva), "n_val_groups": len(groups_flat_va)},
                   indent=2))
    print("saved -> results/v2e046_dinov2_stacker_2026-09-07.json")


if __name__ == "__main__":
    main()
