"""Detector-metric (COCO mAP) boosting pipeline for the oil-palm four-view project.

Layer 1: deep-tail WBF refusion (cached per dataset/split/floor).
Layer 2: learned TP re-ranker (HistGradientBoostingClassifier), fit on TRAIN only.
Layer 3: scoring + submission (agnostic and class-aware), selected on VAL only.

Test data is never opened in this script (existence checks by filename only).
All knobs are selected on VAL. Fixed seed 42 everywhere a seed is accepted.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/workspace/project-expertise/scripts")
import eval_remote_pipeline_postprocess as base  # noqa: E402
import train_detection_edge_linker as edge  # noqa: E402

# --------------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------------
ROOT = Path("/workspace/map_boost")
CACHE = ROOT / "cache"
ARTIFACTS = ROOT / "artifacts"
CACHE.mkdir(parents=True, exist_ok=True)
ARTIFACTS.mkdir(parents=True, exist_ok=True)

K = len(base.NAMES)  # 4 classes B1..B4

RAW_PRED_DIR = Path(
    "/workspace/model_artifacts/project-expertise/eval_2026-08-27/predictions_combined1716"
)
MODELS = ["yolo26l", "rtdetr_l", "rfdetr_l"]
FLOORS = [0.05, 0.02, 0.01]
IOU_THRESHOLD = 0.60
WORKERS = 24

EDGE_MODEL_PATHS = {
    "953": "/workspace/model_artifacts/project-expertise/detection_edge_linker_953_v2/extra.joblib",
    "depth": "/workspace/model_artifacts/project-expertise/detection_edge_linker_depth_v1/extra.joblib",
}
DEPTH_ROOT = Path("/workspace/SawitMVC-Depth-YOLO")

ANCHOR_DIR = Path(
    "/workspace/model_artifacts/project-expertise/eval_2026-08-27/fused_combined1716_val"
)
ANCHOR_EXPECTED = {
    ("953", "agnostic"): 0.8312,
    ("depth", "agnostic"): 0.8648,
    ("953", "classaware"): 0.5689,
    ("depth", "classaware"): 0.6595,
}
ANCHOR_TOL = 0.004

AGNOSTIC_COMBOS = [(1.0, 0.0), (0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (1.0, 0.5)]
GAMMAS = [0.5, 0.75, 1.0]

FEATURE_NAMES_BASE = [
    "score", "log_score", "support", "p1", "p2", "p3", "p4", "maxp", "entropy", "margin",
    "cx", "cy", "w", "h", "area", "log_area", "aspect",
    "n_boxes_image", "norm_rank", "score_over_max",
    "z_side_x", "z_side_y", "z_side_area", "side_count",
    "cv_max_prev", "cv_max_next", "cv_mean_two_max", "cv_count_gt03", "cv_count_gt05",
    "cv_sum_top2_logit", "has_multiview",
]
FEATURE_NAMES_DEPTH_EXTRA = [
    "depth_box_median", "depth_box_mean", "depth_box_std", "depth_frac_valid",
    "depth_img_median", "depth_box_minus_img_median", "has_depth",
]


def feature_names(dataset: str) -> list[str]:
    if dataset == "depth":
        return FEATURE_NAMES_BASE + FEATURE_NAMES_DEPTH_EXTRA
    return list(FEATURE_NAMES_BASE)


def json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def cfg_for(dataset: str) -> dict:
    return base.CONFIGS["SawitMVC-Depth-YOLO" if dataset == "depth" else "SawitMVC-YOLO"]


def raw_pred_path(dataset: str, model: str, split: str) -> Path:
    return RAW_PRED_DIR / f"remote_combined1716_{model}_{dataset}_{split}__{split}.npz"


# --------------------------------------------------------------------------
# Step 2: anchor gate
# --------------------------------------------------------------------------
def load_npz_rows(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {k: np.asarray(z[k], dtype=np.float64) for k in z.files}


def anchor_gate() -> dict:
    print("=== ANCHOR GATE ===", flush=True)
    files = {
        ("953", "agnostic"): ANCHOR_DIR / "SawitMVC_YOLO__wbf_agnostic.npz",
        ("depth", "agnostic"): ANCHOR_DIR / "SawitMVC_Depth_YOLO__wbf_agnostic.npz",
        ("953", "classaware"): ANCHOR_DIR / "SawitMVC_YOLO__wbf_classaware.npz",
        ("depth", "classaware"): ANCHOR_DIR / "SawitMVC_Depth_YOLO__wbf_classaware.npz",
    }
    results = {}
    ok = True
    for (dataset, kind), path in files.items():
        cfg = cfg_for(dataset)
        predictions = load_npz_rows(path)
        agnostic = kind == "agnostic"
        metrics = base.coco_metrics(cfg["data_root"], predictions, agnostic=agnostic, split="val")
        actual = metrics["mAP50"]
        expected = ANCHOR_EXPECTED[(dataset, kind)]
        passed = abs(actual - expected) <= ANCHOR_TOL
        ok = ok and passed
        results[f"{dataset}_{kind}"] = {
            "actual": actual, "expected": expected, "tol": ANCHOR_TOL, "passed": passed,
        }
        print(f"  {dataset:6s} {kind:11s} actual={actual:.4f} expected={expected:.4f} "
              f"{'PASS' if passed else 'FAIL'}", flush=True)
    results["all_passed"] = ok
    if not ok:
        print("ANCHOR GATE FAILED -- stopping per spec.", flush=True)
    return results


# --------------------------------------------------------------------------
# Step 3: verify raw npz inputs
# --------------------------------------------------------------------------
def verify_inputs() -> dict:
    print("=== VERIFY INPUTS ===", flush=True)
    report = {}
    for dataset in ["953", "depth"]:
        for split in ["train", "val"]:
            for model in MODELS:
                p = raw_pred_path(dataset, model, split)
                if not p.exists():
                    raise FileNotFoundError(f"missing required input: {p}")
                z = np.load(p)
                n = len(z.files)
                report[f"{dataset}_{split}_{model}"] = {"path": str(p), "n_images": n}
                print(f"  {dataset:6s} {split:5s} {model:9s} n_images={n} ({p.name})", flush=True)
        for model in MODELS:
            # test: existence check by filename only -- never opened.
            p = raw_pred_path(dataset, model, "test")
            exists = p.exists()
            report[f"{dataset}_test_{model}"] = {"path": str(p), "exists": exists}
            print(f"  {dataset:6s} test  {model:9s} exists={exists} (name-only check, not opened)",
                  flush=True)
            if not exists:
                raise FileNotFoundError(f"missing test file (name check): {p}")
    return report


# --------------------------------------------------------------------------
# Layer 1: WBF refusion with caching
# --------------------------------------------------------------------------
def load_bank(dataset: str, split: str) -> dict[str, dict[str, np.ndarray]]:
    bank = {}
    for model in MODELS:
        p = raw_pred_path(dataset, model, split)
        z = np.load(p)
        bank[model] = {stem: np.asarray(z[stem], dtype=np.float64) for stem in z.files}
    return bank


def vote_cache_path(dataset: str, split: str, floor: float) -> Path:
    return CACHE / f"vote_{dataset}_{split}_floor{floor:.2f}.joblib"


def build_or_load_vote(dataset: str, split: str, floor: float) -> dict:
    path = vote_cache_path(dataset, split, floor)
    if path.exists():
        data = joblib.load(path)
        print(f"[fuse] cache hit {dataset} {split} floor={floor}: "
              f"{data['seconds']:.1f}s (cached), {data['n_trees']} trees, "
              f"{data['n_images']} images", flush=True)
        return data
    cfg = cfg_for(dataset)
    records = base.load_records(cfg, split)
    n_images = sum(len(rec["views"]) for rec in records.values())
    bank = load_bank(dataset, split)
    t0 = time.time()
    _, _, vote = base.fuse_corpus(records, bank, iou_threshold=IOU_THRESHOLD,
                                   score_min=floor, workers=WORKERS)
    dt = time.time() - t0
    slim = {
        stem: [
            {"box": np.asarray(g["box"], dtype=np.float32),
             "score": float(g["score"]),
             "p": np.asarray(g["p"], dtype=np.float32),
             "support": int(g["support"])}
            for g in groups
        ]
        for stem, groups in vote.items()
    }
    n_boxes = sum(len(v) for v in slim.values())
    data = {"vote": slim, "seconds": dt, "n_trees": len(records), "n_images": n_images,
            "n_boxes": n_boxes}
    joblib.dump(data, path, compress=3)
    print(f"[fuse] {dataset} {split} floor={floor}: {dt:.1f}s, {len(records)} trees, "
          f"{n_images} images, {n_boxes} fused boxes", flush=True)
    return data


def synth_vote_arrays(vote_slim: dict) -> dict[str, np.ndarray]:
    out = {}
    for stem, groups in vote_slim.items():
        if not groups:
            out[stem] = np.zeros((0, 5 + K), dtype=np.float32)
            continue
        rows = np.zeros((len(groups), 5 + K), dtype=np.float32)
        for i, g in enumerate(groups):
            rows[i, :4] = g["box"]
            rows[i, 4] = g["score"]
            rows[i, 5:5 + K] = g["p"]
        out[stem] = rows
    return out


# --------------------------------------------------------------------------
# Depth features (depth dataset only)
#
# NOTE on data mismatch vs spec: the spec describes pre-built uint8 PNG depth
# maps (0=invalid, 1..255=inverse depth) under {split}/depth/. What actually
# exists in this environment is the raw sensor dump: {stem}.raw (uint16LE mm,
# native depth-camera grid 848x480) + {stem}.json (calibration/metadata),
#0=invalid. The json's own alignmentNote states the buffer is NOT yet
# depth-to-color reprojected onto the 1280x800 RGB plane; a naive resize
# misses by a median of 29px on that plane (full reprojection would need the
# embedded camera intrinsics/extrinsics). Given this is a supplementary
# feature set for a learned re-ranker (not a physical measurement), we use a
# nearest-neighbor resize from the native depth grid to the RGB image size as
# a deterministic, documented substitute, applied identically at train and
# val time. Values are used directly as depth in mm (not inverted); a
# tree-based ranker is insensitive to monotone rescaling. This is a factual
# deviation from the spec's data description, not a preference choice.
# --------------------------------------------------------------------------
def depth_paths(split: str, stem: str) -> tuple[Path, Path]:
    folder = "valid" if split == "val" else split
    base_dir = DEPTH_ROOT / folder / "depth"
    return base_dir / f"{stem}.raw", base_dir / f"{stem}.json"


def load_depth_raw(split: str, stem: str, target_w: int, target_h: int,
                    cache: dict) -> np.ndarray | None:
    key = (split, stem)
    if key in cache:
        return cache[key]
    raw_p, json_p = depth_paths(split, stem)
    if not raw_p.exists() or not json_p.exists():
        cache[key] = None
        return None
    try:
        meta = json.loads(json_p.read_text(encoding="utf-8-sig"))
        w, h = int(meta["width"]), int(meta["height"])
        arr = np.fromfile(raw_p, dtype="<u2")
        if arr.size != w * h:
            cache[key] = None
            return None
        arr = arr.reshape(h, w).astype(np.float32)
        if (w, h) != (target_w, target_h):
            arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    except Exception:
        cache[key] = None
        return None
    if not np.any(arr > 0):
        cache[key] = None
        return None
    cache[key] = arr
    return arr


def depth_features_for_box(depth_arr: np.ndarray | None, img_median: float,
                            box: np.ndarray) -> np.ndarray:
    if depth_arr is None:
        return np.zeros(7, dtype=np.float32)
    h, w = depth_arr.shape
    x1, y1, x2, y2 = box
    xi1, yi1 = max(int(math.floor(x1)), 0), max(int(math.floor(y1)), 0)
    xi2, yi2 = min(int(math.ceil(x2)), w), min(int(math.ceil(y2)), h)
    if xi2 <= xi1 or yi2 <= yi1:
        valid = np.zeros(0, dtype=np.float32)
        patch_size = 0
    else:
        patch = depth_arr[yi1:yi2, xi1:xi2]
        valid = patch[patch > 0]
        patch_size = patch.size
    if valid.size:
        bmed, bmean, bstd = float(np.median(valid)), float(valid.mean()), float(valid.std())
        frac = float(valid.size) / float(max(patch_size, 1))
    else:
        bmed = bmean = bstd = 0.0
        frac = 0.0
    return np.array([bmed, bmean, bstd, frac, img_median, bmed - img_median, 1.0],
                     dtype=np.float32)


# --------------------------------------------------------------------------
# Cross-view link evidence (n_sides == 4 trees only)
# --------------------------------------------------------------------------
def logit_clip(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def crossview_features_for_tree(rec: dict, dets: list[dict], edge_model, prior: dict
                                 ) -> dict[int, np.ndarray]:
    n = rec["n_sides"]
    out = {i: np.zeros(6, dtype=np.float32) for i in range(len(dets))}
    if n != 4:
        return out
    by_side = defaultdict(list)
    for i, d in enumerate(dets):
        by_side[d["side"]].append(i)
    anchors_by_side = {s: [i for i in idxs if dets[i]["score"] >= 0.125]
                       for s, idxs in by_side.items()}
    pair_meta = []  # (query_local_idx, direction) direction 0=prev, 1=next
    feats = []
    for s, idxs in by_side.items():
        prev_s, next_s = (s - 1) % 4, (s + 1) % 4
        for qi in idxs:
            for aj in anchors_by_side.get(prev_s, []):
                feats.append(edge.pair_features(dets[qi], dets[aj], n, prior))
                pair_meta.append((qi, 0))
            for aj in anchors_by_side.get(next_s, []):
                feats.append(edge.pair_features(dets[qi], dets[aj], n, prior))
                pair_meta.append((qi, 1))
    if not feats:
        return out
    X = np.stack(feats).astype(np.float32)
    probs = edge_model.predict_proba(X)[:, 1]
    per_query = defaultdict(lambda: ([], []))
    for (qi, direction), pr in zip(pair_meta, probs):
        per_query[qi][direction].append(float(pr))
    for qi, (prev_probs, next_probs) in per_query.items():
        max_prev = max(prev_probs) if prev_probs else 0.0
        max_next = max(next_probs) if next_probs else 0.0
        mean_two = (max_prev + max_next) / 2.0
        pooled = prev_probs + next_probs
        count_03 = float(sum(1 for x in pooled if x > 0.3))
        count_05 = float(sum(1 for x in pooled if x > 0.5))
        if pooled:
            top2 = sorted(pooled, reverse=True)[:2]
            sum_top2_logit = float(np.sum(logit_clip(np.asarray(top2))))
        else:
            sum_top2_logit = 0.0
        out[qi] = np.array([max_prev, max_next, mean_two, count_03, count_05, sum_top2_logit],
                           dtype=np.float32)
    return out


# --------------------------------------------------------------------------
# Layer 2: feature + label construction
# --------------------------------------------------------------------------
def greedy_labels(boxes: np.ndarray, scores: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    n = len(boxes)
    labels = np.zeros(n, dtype=np.int8)
    if n == 0 or len(gt_boxes) == 0:
        return labels
    order = np.argsort(-scores, kind="stable")
    matched_gt = np.zeros(len(gt_boxes), dtype=bool)
    for idx in order:
        ious = base.iou_one(boxes[idx], gt_boxes)
        ious = np.where(matched_gt, -1.0, ious)
        j = int(np.argmax(ious))
        if ious[j] >= 0.5:
            matched_gt[j] = True
            labels[idx] = 1
    return labels


def features_cache_path(dataset: str, split: str, floor: float) -> Path:
    return CACHE / f"features_{dataset}_{split}_floor{floor:.2f}.joblib"


def build_features(dataset: str, split: str, floor: float, edge_model, prior: dict,
                    depth_cache: dict) -> dict:
    cache_path = features_cache_path(dataset, split, floor)
    if cache_path.exists():
        data = joblib.load(cache_path)
        print(f"[features] cache hit {dataset} {split} floor={floor}: "
              f"{data['seconds']:.1f}s (cached), {len(data['y'])} boxes", flush=True)
        return data

    cfg = cfg_for(dataset)
    records = base.load_records(cfg, split)
    vote_data = build_or_load_vote(dataset, split, floor)
    vote_slim = vote_data["vote"]
    synth = synth_vote_arrays(vote_slim)
    is_depth = dataset == "depth"

    stem_gt: dict[str, np.ndarray] = {}
    for rec in records.values():
        for side, view in rec["views"].items():
            boxes = [app["box"] for bunch in rec["bunches"] for app in bunch["appearances"]
                     if int(app["side"]) == int(side)]
            stem_gt[view["stem"]] = (np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
                                     if boxes else np.zeros((0, 4)))

    feats_list, labels_list = [], []
    stems_out, row_index_out, box_out, score_out, p_out = [], [], [], [], []

    t0 = time.time()
    n_trees = 0
    for tree_id, rec in records.items():
        n_trees += 1
        dets = edge.make_detections(rec, synth, proposal_min=0.0)
        if not dets:
            continue
        cross = crossview_features_for_tree(rec, dets, edge_model, prior)
        has_multiview = 1.0 if rec["n_sides"] == 4 else 0.0

        by_stem = defaultdict(list)
        for i, d in enumerate(dets):
            by_stem[d["stem"]].append(i)

        for stem, idxs in by_stem.items():
            boxes = np.stack([dets[i]["box"] for i in idxs])
            scores = np.array([dets[i]["score"] for i in idxs])
            n_boxes = len(idxs)
            max_score = float(scores.max())
            order = np.argsort(-scores, kind="stable")
            rank = np.empty(n_boxes)
            rank[order] = np.arange(n_boxes)
            norm_rank = rank / max(n_boxes - 1, 1)
            gt_boxes = stem_gt.get(stem, np.zeros((0, 4)))
            labels = greedy_labels(boxes, scores, gt_boxes)

            depth_arr = img_median = None
            if is_depth:
                side0 = dets[idxs[0]]["side"]
                width = rec["views"][side0]["width"]
                height = rec["views"][side0]["height"]
                depth_arr = load_depth_raw(split, stem, width, height, depth_cache)
                if depth_arr is not None:
                    valid = depth_arr[depth_arr > 0]
                    img_median = float(np.median(valid)) if valid.size else 0.0
                else:
                    img_median = 0.0

            for pos, i in enumerate(idxs):
                d = dets[i]
                p = np.asarray(d["p"], dtype=np.float64)
                score = float(d["score"])
                support = int(vote_slim[stem][d["row_index"]]["support"])
                p_sorted = np.sort(p)[::-1]
                margin = float(p_sorted[0] - p_sorted[1])
                entropy = float(-(p * np.log(np.clip(p, 1e-8, None))).sum())
                detector_feats = [score, math.log(score + 1e-6), support, *p.tolist(),
                                  float(p.max()), entropy, margin]
                area = d["w"] * d["h"]
                geometry_feats = [d["cx"], d["cy"], d["w"], d["h"], area,
                                  math.log(max(area, 1e-9)), d["w"] / max(d["h"], 1e-9)]
                context_feats = [float(n_boxes), float(norm_rank[pos]),
                                 score / max(max_score, 1e-9)]
                side_feats = [d["z_side_x"], d["z_side_y"], d["z_side_area"], d["side_count"]]
                cf = cross.get(i, np.zeros(6, dtype=np.float32))
                cross_feats = [*cf.tolist(), has_multiview]
                feat = detector_feats + geometry_feats + context_feats + side_feats + cross_feats
                if is_depth:
                    feat += depth_features_for_box(depth_arr, img_median, d["box"]).tolist()
                feats_list.append(feat)
                labels_list.append(int(labels[pos]))
                stems_out.append(stem)
                row_index_out.append(int(d["row_index"]))
                box_out.append(d["box"])
                score_out.append(score)
                p_out.append(p.astype(np.float32))

    dt = time.time() - t0
    X = np.asarray(feats_list, dtype=np.float32)
    y = np.asarray(labels_list, dtype=np.int8)
    data = {
        "X": X, "y": y,
        "stems": np.asarray(stems_out, dtype=object),
        "row_index": np.asarray(row_index_out, dtype=np.int32),
        "box": np.asarray(box_out, dtype=np.float32).reshape(-1, 4),
        "score": np.asarray(score_out, dtype=np.float32),
        "p": np.asarray(p_out, dtype=np.float32).reshape(-1, K),
        "feature_names": feature_names(dataset),
        "seconds": dt, "n_trees": n_trees, "n_boxes": len(y),
    }
    joblib.dump(data, cache_path, compress=3)
    print(f"[features] {dataset} {split} floor={floor}: {dt:.1f}s, {n_trees} trees, "
          f"{len(y)} boxes, positive_rate={y.mean():.4f}", flush=True)
    return data


# --------------------------------------------------------------------------
# Layer 2: ranker training
# --------------------------------------------------------------------------
def ranker_path(dataset: str, floor: float) -> Path:
    d = ARTIFACTS / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d / f"ranker_floor{floor:.2f}.joblib"


def train_or_load_ranker(dataset: str, floor: float, train_data: dict, val_data: dict) -> dict:
    path = ranker_path(dataset, floor)
    if path.exists():
        model = joblib.load(path)
        train_prob = model.predict_proba(train_data["X"])[:, 1]
        val_prob = model.predict_proba(val_data["X"])[:, 1]
        train_auc = float(roc_auc_score(train_data["y"], train_prob))
        val_auc = float(roc_auc_score(val_data["y"], val_prob))
        print(f"[ranker] cache hit {dataset} floor={floor}: train_auc={train_auc:.4f} "
              f"val_auc={val_auc:.4f}", flush=True)
        return {"model": model, "train_auc": train_auc, "val_auc": val_auc, "cached": True}

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=10.0, random_state=42,
    )
    t0 = time.time()
    model.fit(train_data["X"], train_data["y"])
    dt = time.time() - t0
    train_prob = model.predict_proba(train_data["X"])[:, 1]
    val_prob = model.predict_proba(val_data["X"])[:, 1]
    train_auc = float(roc_auc_score(train_data["y"], train_prob))
    val_auc = float(roc_auc_score(val_data["y"], val_prob))
    joblib.dump(model, path, compress=3)
    print(f"[ranker] {dataset} floor={floor}: fit {dt:.1f}s, train_auc={train_auc:.4f} "
          f"val_auc={val_auc:.4f}", flush=True)
    return {"model": model, "train_auc": train_auc, "val_auc": val_auc, "seconds": dt,
            "cached": False}


# --------------------------------------------------------------------------
# Layer 3: scoring + submission
# --------------------------------------------------------------------------
def score_agnostic(vote_slim: dict, p_tp_map: dict, a: float, b: float,
                    cap: int = 100) -> dict[str, np.ndarray]:
    preds = {}
    for stem, groups in vote_slim.items():
        rows = []
        for row_index, g in enumerate(groups):
            score = g["score"]
            p_tp = p_tp_map.get((stem, row_index))
            if p_tp is None:
                continue
            s = (score ** a) * (p_tp ** b)
            x1, y1, x2, y2 = g["box"]
            rows.append((x1, y1, x2, y2, s, 0.0))
        if not rows:
            continue
        arr = np.asarray(rows, dtype=np.float64)
        order = np.argsort(-arr[:, 4])[:cap]
        preds[stem] = arr[order]
    return preds


def score_classaware(vote_slim: dict, p_tp_map: dict, a: float, b: float, gamma: float,
                      cap: int = 100, min_pc: float = 0.01) -> dict[str, np.ndarray]:
    preds = {}
    for stem, groups in vote_slim.items():
        rows = []
        for row_index, g in enumerate(groups):
            score = g["score"]
            p_tp = p_tp_map.get((stem, row_index))
            if p_tp is None:
                continue
            s = (score ** a) * (p_tp ** b)
            x1, y1, x2, y2 = g["box"]
            p = g["p"]
            for c in range(K):
                pc = float(p[c])
                if pc >= min_pc:
                    rows.append((x1, y1, x2, y2, s * (pc ** gamma), float(c)))
        if not rows:
            continue
        arr = np.asarray(rows, dtype=np.float64)
        order = np.argsort(-arr[:, 4])[:cap]
        preds[stem] = arr[order]
    return preds


def p_tp_map_from(feat_data: dict, model) -> dict:
    probs = model.predict_proba(feat_data["X"])[:, 1]
    out = {}
    for stem, row_index, p in zip(feat_data["stems"], feat_data["row_index"], probs):
        out[(stem, int(row_index))] = float(p)
    return out


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def run_dataset(dataset: str, wall_times: dict) -> dict:
    print(f"\n########## DATASET {dataset} ##########", flush=True)
    cfg = cfg_for(dataset)
    edge_model = joblib.load(EDGE_MODEL_PATHS[dataset])
    prior = base.build_rotation_prior(base.load_records(cfg, "train"))
    depth_cache: dict = {}

    result = {"floors": {}, "agnostic_grid": [], "classaware_grid": []}

    for floor in FLOORS:
        print(f"\n--- {dataset} floor={floor} ---", flush=True)
        t0 = time.time()
        train_feat = build_features(dataset, "train", floor, edge_model, prior, depth_cache)
        val_feat = build_features(dataset, "val", floor, edge_model, prior, depth_cache)
        ranker_info = train_or_load_ranker(dataset, floor, train_feat, val_feat)
        model = ranker_info["model"]
        val_vote = build_or_load_vote(dataset, "val", floor)["vote"]
        p_tp_map = p_tp_map_from(val_feat, model)
        dt = time.time() - t0
        wall_times[f"{dataset}_floor{floor}"] = dt

        result["floors"][str(floor)] = {
            "train_auc": ranker_info["train_auc"], "val_auc": ranker_info["val_auc"],
            "train_boxes": int(train_feat["n_boxes"]), "val_boxes": int(val_feat["n_boxes"]),
            "seconds": dt,
        }

        for (a, b) in AGNOSTIC_COMBOS:
            preds = score_agnostic(val_vote, p_tp_map, a, b)
            metrics = base.coco_metrics(cfg["data_root"], preds, agnostic=True, split="val")
            row = {"floor": floor, "a": a, "b": b, "AP50": metrics["mAP50"]}
            result["agnostic_grid"].append(row)
            print(f"  [agnostic] floor={floor} a={a} b={b} AP50={metrics['mAP50']:.4f}",
                  flush=True)

        for (a, b) in AGNOSTIC_COMBOS:
            for gamma in GAMMAS:
                preds = score_classaware(val_vote, p_tp_map, a, b, gamma)
                metrics = base.coco_metrics(cfg["data_root"], preds, agnostic=False, split="val")
                row = {"floor": floor, "a": a, "b": b, "gamma": gamma,
                       "mAP50": metrics["mAP50"], "per_class_AP50": metrics["per_class_AP50"]}
                result["classaware_grid"].append(row)
                print(f"  [classaware] floor={floor} a={a} b={b} gamma={gamma} "
                      f"mAP50={metrics['mAP50']:.4f}", flush=True)

    best_agnostic = max(result["agnostic_grid"], key=lambda r: r["AP50"])
    best_classaware = max(result["classaware_grid"], key=lambda r: r["mAP50"])
    result["selected_agnostic_profile"] = best_agnostic
    result["selected_classaware_profile"] = best_classaware
    print(f"\n>>> {dataset} BEST agnostic: {best_agnostic}", flush=True)
    print(f">>> {dataset} BEST classaware: {best_classaware}", flush=True)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("anchor", "verify", "all"), default="all")
    args = ap.parse_args()

    wall_times = {}
    t_start = time.time()

    anchors = anchor_gate()
    if not anchors["all_passed"]:
        print(json.dumps(anchors, indent=2, default=json_default))
        return 1
    if args.stage == "anchor":
        return 0

    verify = verify_inputs()
    if args.stage == "verify":
        return 0

    out = {
        "anchors": anchors, "verify": verify,
        "baselines": {
            "agnostic": {"953": 0.8312, "depth": 0.8648},
            "classaware": {"953": 0.5689, "depth": 0.6595},
        },
        "datasets": {},
    }
    for dataset in ["953", "depth"]:
        out["datasets"][dataset] = run_dataset(dataset, wall_times)
        out["datasets"][dataset]["feature_list"] = feature_names(dataset)
        out["datasets"][dataset]["file_paths"] = {
            "edge_model": EDGE_MODEL_PATHS[dataset],
            "raw_predictions": {m: str(raw_pred_path(dataset, m, s))
                                for m in MODELS for s in ("train", "val")},
            "vote_cache": {str(f): floor for floor in FLOORS
                          for f in [vote_cache_path(dataset, sp, floor)
                                    for sp in ("train", "val")]},
            "ranker": {str(floor): str(ranker_path(dataset, floor)) for floor in FLOORS},
        }
        result_path = ARTIFACTS / dataset / "results_val.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps({"dataset": dataset, **out["datasets"][dataset],
                       "anchors": anchors, "baselines": out["baselines"]},
                      indent=2, default=json_default) + "\n"
        )
        print(f"-> wrote {result_path}", flush=True)

    wall_times["total_seconds"] = time.time() - t_start
    out["wall_times"] = wall_times
    combined_path = ARTIFACTS / "results_val.json"
    combined_path.write_text(json.dumps(out, indent=2, default=json_default) + "\n")
    print(f"-> wrote {combined_path}", flush=True)
    print(json.dumps(wall_times, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
