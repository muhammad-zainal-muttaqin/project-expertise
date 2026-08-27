"""Evaluate the downloaded detector bank with the four-view post-processing.

This is an evaluation harness for the remote ``new763`` detector bank.  It
does not train or download anything.  The detector dumps are produced by
``eval_new763_pycoco.py``; this script applies class-agnostic WBF, keeps a
soft class vote for the downstream head, and links detections across ordered
views with a train-only signed-rotation prior.

The multi-view number is deliberately labelled as a *baseline*: the
separate crop-classifier and learned proposal-linker heads are not silently
substituted here.  This makes the result reproducible on both corpora and
avoids applying a DAMIMAS-only head to MARIHAT/TOPAZ.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).parent))
from eval_new763_pycoco import NAMES, build_gt  # noqa: E402


K = len(NAMES)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


CONFIGS = {
    "SawitMVC-Depth-YOLO": {
        "kind": "depth",
        "data_root": Path("/workspace/SawitMVC-Depth-YOLO"),
        "meta_root": Path("/workspace/SawitMVC-Depth-YOLO"),
        "predictions": {
            "yolo26l": Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27/predictions/remote_new763_yolo26l_depth_test__test.npz"),
            "rtdetr_l": Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27/predictions/remote_new763_rtdetr_l_depth_test__test.npz"),
            "rfdetr_l": Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27/predictions/remote_new763_rfdetr_l_depth_test__test.npz"),
        },
    },
    "SawitMVC-YOLO": {
        "kind": "yolo953_adapter",
        "data_root": Path("/workspace/SawitMVC-YOLO"),
        "meta_root": Path("/workspace/SawitMVC-YOLO"),
        "predictions": {
            "yolo26l": Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27/predictions/remote_new763_yolo26l_953_test__test.npz"),
            "rtdetr_l": Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27/predictions/remote_new763_rtdetr_l_953_test__test.npz"),
            "rfdetr_l": Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27/predictions/remote_new763_rfdetr_l_953_test__test.npz"),
        },
    },
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def metadata_paths(cfg: dict, split: str) -> list[Path]:
    if cfg["kind"] == "depth":
        folder = "valid" if split == "val" and (cfg["meta_root"] / "valid").is_dir() else split
        return sorted((cfg["meta_root"] / folder / "linked").glob("*.json"))
    manifest = {}
    manifest_path = cfg["meta_root"] / "split_manifest.csv"
    if manifest_path.exists():
        import csv
        with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                manifest[row["tree_id"]] = row["new_split"]
    paths = []
    for path in sorted((cfg["meta_root"] / "json").glob("*.json")):
        data = read_json(path)
        # The regenerated YOLO folders follow split_manifest.csv.  The JSON
        # field ``split`` belongs to the source export and is not authoritative
        # after the tree-level re-split.
        selected_split = manifest.get(data.get("tree_id"), data.get("split"))
        if selected_split == split:
            paths.append(path)
    return paths


def class_index(name: str | None) -> int:
    value = str(name or "").upper()
    return NAMES.index(value) if value in NAMES else -1


def tree_record(data: dict) -> dict:
    views = {}
    for image in data.get("images", {}).values():
        filename = image.get("filename")
        if not filename:
            continue
        side = int(image.get("side_index", len(views)))
        width = int(image.get("width", 0))
        height = int(image.get("height", 0))
        views[side] = {
            "stem": Path(filename).stem,
            "filename": filename,
            "width": width,
            "height": height,
            "annotations": image.get("annotations", []),
        }

    bunches = []
    for bunch in data.get("bunches", []):
        appearances = []
        for app in bunch.get("appearances", []):
            box = app.get("bbox_pixel")
            if box is None:
                continue
            appearances.append({
                "side": int(app.get("side_index", 0)),
                "box_index": int(app.get("box_index", 0)),
                "box": [float(x) for x in box],
            })
        if appearances:
            bunches.append({
                "id": str(bunch.get("bunch_id", len(bunches))),
                "cls": class_index(bunch.get("class") or bunch.get("class_name")),
                "appearances": appearances,
            })

    # Defensive fallback for metadata files without an explicit bunch list.
    if not bunches:
        for side, view in views.items():
            for ann in view["annotations"]:
                box = ann.get("bbox_pixel")
                if box is None:
                    continue
                bunches.append({
                    "id": f"{side}:{ann.get('box_index', len(bunches))}",
                    "cls": int(ann.get("class_id", -1)),
                    "appearances": [{
                        "side": side,
                        "box_index": int(ann.get("box_index", 0)),
                        "box": [float(x) for x in box],
                    }],
                })

    return {
        "tree_id": data.get("tree_id", ""),
        "n_sides": len(views),
        "views": views,
        "bunches": bunches,
    }


def load_records(cfg: dict, split: str) -> dict[str, dict]:
    out = {}
    for path in metadata_paths(cfg, split):
        rec = tree_record(read_json(path))
        if rec["tree_id"]:
            out[rec["tree_id"]] = rec
    return out


def center_features(rec: dict, app: dict) -> tuple[float, float, float, float]:
    view = rec["views"][app["side"]]
    x1, y1, x2, y2 = app["box"]
    w, h = max(view["width"], 1), max(view["height"], 1)
    bw, bh = max(x2 - x1, 1.), max(y2 - y1, 1.)
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, bw / w, bh / h)


def build_rotation_prior(records: dict[str, dict]) -> dict[tuple[int, int], tuple[float, ...]]:
    values = defaultdict(list)
    for rec in records.values():
        n = rec["n_sides"]
        for bunch in rec["bunches"]:
            for left, right in combinations(bunch["appearances"], 2):
                if left["side"] == right["side"]:
                    continue
                a, b = sorted((left, right), key=lambda x: x["side"])
                d = (b["side"] - a["side"]) % n
                xa, ya, wa, ha = center_features(rec, a)
                xb, yb, wb, hb = center_features(rec, b)
                values[(n, d)].append((xb - xa, yb - ya,
                                       np.log((wb * hb) / (wa * ha))))
    prior = {}
    for key, rows in values.items():
        a = np.asarray(rows, float)
        med = np.median(a, axis=0)
        mad = 1.4826 * np.median(np.abs(a - med), axis=0)
        prior[key] = (
            float(med[0]), float(med[1]), max(float(mad[0]), .025),
            max(float(mad[1]), .025), max(float(mad[2]), .15), len(rows),
        )
    return prior


def pair_score(a: dict, b: dict, n: int,
               prior: dict[tuple[int, int], tuple[float, ...]]) -> float:
    if a["side"] > b["side"]:
        a, b = b, a
    d = (b["side"] - a["side"]) % n
    xa, ya, wa, ha = a["cx"], a["cy"], a["w"], a["h"]
    xb, yb, wb, hb = b["cx"], b["cy"], b["w"], b["h"]
    mux, muy, sx, sy, sa, _count = prior.get(
        (n, d), (0., 0., .20, .15, .70, 0))
    zdx = ((xb - xa) - mux) / max(sx, .025)
    zdy = ((yb - ya) - muy) / max(sy, .025)
    zarea = (np.log(max(wb * hb, 1e-8) / max(wa * ha, 1e-8))) / max(sa, .15)
    zshape = (np.log(max(wb / hb, 1e-8) / max(wa / ha, 1e-8))) / .85
    class_sim = float(np.sqrt(np.maximum(a["p"], 0) * np.maximum(b["p"], 0)).sum())
    cost = .5 * (zdx * zdx + zdy * zdy) + .12 * zarea * zarea
    cost += .08 * zshape * zshape + .10 * (1. - class_sim)
    return float(np.exp(-min(cost, 40.)))


def calibrate_link_threshold(records: dict[str, dict],
                             prior: dict[tuple[int, int], tuple[float, ...]]) -> dict:
    scores, labels = [], []
    for rec in records.values():
        dets = []
        for bunch in rec["bunches"]:
            p = np.zeros(K, float)
            if 0 <= bunch["cls"] < K:
                p[bunch["cls"]] = 1.
            for app in bunch["appearances"]:
                x, y, w, h = center_features(rec, app)
                dets.append({"side": app["side"], "cx": x, "cy": y,
                             "w": w, "h": h, "p": p,
                             "bid": bunch["id"]})
        for a, b in combinations(dets, 2):
            if a["side"] == b["side"]:
                continue
            scores.append(pair_score(a, b, rec["n_sides"], prior))
            labels.append(int(a["bid"] == b["bid"]))
    if not scores:
        return {"threshold": .25, "pairs": 0, "f1": 0.}
    s, y = np.asarray(scores), np.asarray(labels, int)
    best = (0., .25, 0, 0, 0)
    for threshold in np.arange(.05, .96, .01):
        pred = s >= threshold
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-12)
        if f1 > best[0]:
            best = (float(f1), float(threshold), tp, fp, fn)
    return {"threshold": best[1], "pairs": int(len(y)), "f1": best[0],
            "tp": best[2], "fp": best[3], "fn": best[4]}


def iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, float)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = max((box[2] - box[0]) * (box[3] - box[1]), 0.)
    area_b = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None)
    return inter / (area_a + area_b - inter + 1e-9)


def fuse_groups(rows: np.ndarray, iou_threshold: float,
                score_min: float, n_models: int,
                model_weights: np.ndarray | None = None,
                class_model_weights: np.ndarray | None = None) -> list[dict]:
    """Fast greedy WBF; rows are x1,y1,x2,y2,score,class,model."""
    if len(rows) == 0:
        return []
    rows = rows[rows[:, 4] >= score_min]
    if len(rows) == 0:
        return []
    if model_weights is None:
        model_weights = np.ones(n_models, dtype=float)
        weighted_mode = False
    else:
        model_weights = np.asarray(model_weights, dtype=float)
        if len(model_weights) != n_models or np.any(model_weights <= 0):
            raise ValueError("model_weights must contain one positive weight per model")
        weighted_mode = True
    if class_model_weights is None:
        class_model_weights = np.ones((n_models, K), dtype=float)
    else:
        class_model_weights = np.asarray(class_model_weights, dtype=float)
        if (class_model_weights.shape != (n_models, K)
                or np.any(class_model_weights < 0)
                or np.any(class_model_weights.sum(axis=0) <= 0)):
            raise ValueError(
                "class_model_weights must be nonnegative with shape "
                f"({n_models}, {K}) and positive support per class")
    rows = rows[np.argsort(-(rows[:, 4] * model_weights[rows[:, 6].astype(int)]))]
    representatives: list[np.ndarray] = []
    groups: list[list[np.ndarray]] = []
    for row in rows:
        if representatives:
            overlaps = np.asarray([iou_one(row[:4], np.asarray([r]))[0]
                                   for r in representatives])
            index = int(np.argmax(overlaps))
            if overlaps[index] >= iou_threshold:
                groups[index].append(row)
                g = np.asarray(groups[index])
                weights = g[:, 4:5] * model_weights[g[:, 6].astype(int), None]
                representatives[index] = (g[:, :4] * weights).sum(0) / max(
                    float(weights.sum()), 1e-9)
                continue
        groups.append([row])
        representatives.append(row[:4].copy())

    out = []
    for group in groups:
        g = np.asarray(group)
        weights = g[:, 4] * model_weights[g[:, 6].astype(int)]
        box = (g[:, :4] * weights[:, None]).sum(0) / max(float(weights.sum()), 1e-9)
        support = len(set(g[:, 6].astype(int).tolist()))
        if weighted_mode:
            # Missing sources still reduce the score through support/n_models;
            # among present sources, the configured model weights only affect
            # the confidence average and the box coordinate.
            present = np.unique(g[:, 6].astype(int))
            present_weight = model_weights[present].sum()
            score_base = float(weights.sum() / max(present_weight, 1e-9))
        else:
            score_base = float(weights.mean())
        score = float(score_base * min(support, n_models) / n_models)
        probs = np.zeros(K, float)
        for row, row_weight in zip(g, weights):
            c = int(row[5])
            if 0 <= c < K:
                model_id = int(row[6])
                probs[c] += float(row_weight * class_model_weights[model_id, c])
        if probs.sum() <= 0:
            probs[:] = 1. / K
        else:
            probs /= probs.sum()
        out.append({"box": box.astype(float), "score": score, "p": probs,
                    "support": support, "members": g})
    return out


def load_prediction_bank(cfg: dict) -> dict[str, dict[str, np.ndarray]]:
    bank = {}
    for model_name, path in cfg["predictions"].items():
        z = np.load(path)
        bank[model_name] = {stem: np.asarray(z[stem], float)
                            for stem in z.files}
    return bank


def _fuse_one(task: tuple[str, np.ndarray, float, float, int,
                            np.ndarray | None, np.ndarray | None]):
    (stem, rows, iou_threshold, score_min, n_models, model_weights,
     class_model_weights) = task
    all_groups = fuse_groups(rows, iou_threshold, score_min, n_models,
                             model_weights, class_model_weights)
    agnostic = np.asarray(
        [[*g["box"], g["score"], 0.] for g in all_groups], float
    ).reshape(-1, 6)
    ca_groups = []
    for c in range(K):
        ca_groups.extend(fuse_groups(rows[rows[:, 5] == c], iou_threshold,
                                     score_min, n_models, model_weights,
                                     class_model_weights))
    classaware = np.asarray(
        [[*g["box"], g["score"], int(np.argmax(g["p"]))]
         for g in ca_groups], float
    ).reshape(-1, 6)
    return stem, classaware, agnostic, all_groups


def fuse_corpus(records: dict[str, dict], bank: dict[str, dict[str, np.ndarray]],
                iou_threshold: float, score_min: float,
                workers: int = 1,
                model_weights: np.ndarray | None = None,
                class_model_weights: np.ndarray | None = None) -> tuple[dict, dict, dict]:
    n_models = len(bank)
    stems = [view["stem"] for rec in records.values()
             for view in rec["views"].values()]
    classaware, agnostic, vote = {}, {}, {}
    tasks = []
    for stem in stems:
        parts = []
        for model_id, model_name in enumerate(bank):
            rows = bank[model_name].get(stem, np.zeros((0, 6)))
            if len(rows):
                parts.append(np.c_[rows[:, :6],
                                   np.full(len(rows), model_id, float)])
        rows = np.concatenate(parts, axis=0) if parts else np.zeros((0, 7))
        tasks.append((stem, rows, iou_threshold, score_min, n_models,
                      model_weights, class_model_weights))
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            fused = pool.map(_fuse_one, tasks, chunksize=1)
            for stem, ca_rows, agn_rows, groups in fused:
                classaware[stem], agnostic[stem], vote[stem] = (
                    ca_rows, agn_rows, groups)
    else:
        for task in tasks:
            stem, ca_rows, agn_rows, groups = _fuse_one(task)
            classaware[stem], agnostic[stem], vote[stem] = (
                ca_rows, agn_rows, groups)
    return classaware, agnostic, vote


def coco_metrics(data_root: Path, predictions: dict[str, np.ndarray],
                 agnostic: bool = False, split: str = "test") -> dict:
    gt, paths = build_gt(data_root, split)
    if agnostic:
        for ann in gt.dataset["annotations"]:
            ann["category_id"] = 1
        gt.dataset["categories"] = [{"id": 1, "name": "tandan"}]
        gt.createIndex()
    stem_to_id = {path.stem: i for i, path in enumerate(paths, 1)}
    dt = []
    for stem, rows in predictions.items():
        image_id = stem_to_id.get(stem)
        if image_id is None:
            continue
        for x1, y1, x2, y2, score, cls in rows:
            dt.append({"image_id": image_id,
                       "category_id": 1 if agnostic else int(cls) + 1,
                       "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                       "score": float(score)})
    results = gt.loadRes(dt) if dt else gt.loadRes([])
    ev = COCOeval(gt, results, "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    precision = ev.eval["precision"]
    per_class = {}
    if agnostic:
        per_class["tandan"] = float(precision[0, :, 0, 0, 2][
            precision[0, :, 0, 0, 2] > -1].mean())
    else:
        for index, name in enumerate(NAMES):
            values = precision[0, :, index, 0, 2]
            values = values[values > -1]
            per_class[name] = float(values.mean()) if len(values) else 0.
    return {"n_images": len(paths), "mAP50": float(ev.stats[1]),
            "mAP50_95": float(ev.stats[0]), "per_class_AP50": per_class}


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def link_clusters(dets: list[dict], n: int,
                  prior: dict[tuple[int, int], tuple[float, ...]],
                  threshold: float, pair_mode: str = "all",
                  max_cluster_size: int | None = None) -> list[list[dict]]:
    if not dets:
        return []
    per_side = defaultdict(list)
    for i, det in enumerate(dets):
        per_side[det["side"]].append(i)
    candidates = []
    for sa, sb in combinations(sorted(per_side), 2):
        if (pair_mode == "adjacent" and
                (sb - sa) % n not in (1, n - 1)):
            continue
        aa, bb = per_side[sa], per_side[sb]
        scores = np.asarray([[pair_score(dets[i], dets[j], n, prior)
                              for j in bb] for i in aa], float)
        if scores.size == 0:
            continue
        for ri, ci in zip(*linear_sum_assignment(-scores)):
            if scores[ri, ci] >= threshold:
                candidates.append((float(scores[ri, ci]), aa[ri], bb[ci]))
    candidates.sort(reverse=True)
    uf = UnionFind(len(dets))
    side_sets = {i: {dets[i]["side"]} for i in range(len(dets))}
    sizes = {i: 1 for i in range(len(dets))}
    max_size = (max_cluster_size if max_cluster_size is not None
                else (3 if n == 4 else 6))
    for score, i, j in candidates:
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj or side_sets[ri] & side_sets[rj]:
            continue
        if sizes[ri] + sizes[rj] > max_size:
            continue
        uf.union(ri, rj)
        root = uf.find(ri)
        sizes[root] = sizes[ri] + sizes[rj]
        side_sets[root] = side_sets[ri] | side_sets[rj]
    groups = defaultdict(list)
    for i, det in enumerate(dets):
        groups[uf.find(i)].append(det)
    return list(groups.values())


def cluster_summary(group: list[dict]) -> dict:
    weights = np.asarray([max(x["score"], 1e-6) for x in group])
    p = np.average(np.stack([x["p"] for x in group]), axis=0, weights=weights)
    p /= max(float(p.sum()), 1e-9)
    return {"members": group, "p": p, "cls": int(np.argmax(p)),
            "score": float(weights.mean())}


def count_iou(pred: dict, bunch: dict) -> float:
    best = 0.
    for member in pred["members"]:
        for app in bunch["appearances"]:
            if member["side"] != app["side"]:
                continue
            best = max(best, float(iou_one(member["box"],
                                           np.asarray([app["box"]]))[0]))
    return best


def multiview_metrics(records: dict[str, dict], vote: dict[str, list[dict]],
                      prior: dict[tuple[int, int], tuple[float, ...]],
                      threshold: float, proposal_min: float,
                      singleton_min: float = 0.0,
                      pair_mode: str = "all",
                      max_cluster_size: int | None = None) -> dict:
    # Four-side capture is the requested product contract. Eight-side test
    # trees are reported as excluded rather than mixed into this benchmark.
    usable = [r for r in records.values() if r["n_sides"] == 4]
    skipped = len(records) - len(usable)
    total_gt = total_pred = total_tp = 0
    count_abs = count_exact = count_pm1 = count_vec_exact = 0
    class_correct = matched = 0
    cm = np.zeros((K + 1, K + 1), int)  # row=prediction, col=ground truth
    per_tree = []
    for rec in usable:
        dets = []
        for side, view in rec["views"].items():
            for item in vote.get(view["stem"], []):
                if item["score"] < proposal_min:
                    continue
                x1, y1, x2, y2 = item["box"]
                width = max(view["width"], 1)
                height = max(view["height"], 1)
                dets.append({"side": side, "box": np.asarray(item["box"], float),
                             "score": item["score"], "p": item["p"],
                             "cx": ((x1 + x2) / 2) / width,
                             "cy": ((y1 + y2) / 2) / height,
                             "w": max(x2 - x1, 1.) / width,
                             "h": max(y2 - y1, 1.) / height})
        clusters = []
        for group in link_clusters(dets, rec["n_sides"], prior, threshold,
                                    pair_mode, max_cluster_size):
            summary = cluster_summary(group)
            if len(group) == 1 and summary["score"] < singleton_min:
                continue
            clusters.append(summary)
        bunches = rec["bunches"]
        matrix = np.zeros((len(clusters), len(bunches)), float)
        for i, pred in enumerate(clusters):
            for j, bunch in enumerate(bunches):
                matrix[i, j] = count_iou(pred, bunch)
        matches = []
        if matrix.size:
            for i, j in zip(*linear_sum_assignment(-matrix)):
                if matrix[i, j] >= .5:
                    matches.append((int(i), int(j)))
        matched_pred = {i for i, _ in matches}
        matched_gt = {j for _, j in matches}
        total_gt += len(bunches); total_pred += len(clusters)
        total_tp += len(matches); matched += len(matches)
        pred_count = np.bincount([x["cls"] for x in clusters], minlength=K)
        gt_count = np.bincount([x["cls"] for x in bunches if x["cls"] >= 0], minlength=K)
        delta = len(clusters) - len(bunches)
        count_abs += abs(delta); count_exact += int(delta == 0)
        count_pm1 += int(abs(delta) <= 1)
        count_vec_exact += int(np.array_equal(pred_count, gt_count))
        for i, j in matches:
            pc, gc = clusters[i]["cls"], bunches[j]["cls"]
            if 0 <= pc < K and 0 <= gc < K:
                cm[pc, gc] += 1
                class_correct += int(pc == gc)
        for i, pred in enumerate(clusters):
            if i not in matched_pred and 0 <= pred["cls"] < K:
                cm[pred["cls"], K] += 1
        for j, bunch in enumerate(bunches):
            if j not in matched_gt and 0 <= bunch["cls"] < K:
                cm[K, bunch["cls"]] += 1
        per_tree.append({"tree_id": rec["tree_id"], "n_sides": rec["n_sides"],
                         "gt_count": len(bunches), "pred_count": len(clusters),
                         "matched": len(matches), "count_delta": delta})

    p = total_tp / max(total_pred, 1)
    r = total_tp / max(total_gt, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    macro_f1 = []
    for c in range(K):
        tp = cm[c, c]
        fp = int(cm[c, :].sum() - tp)
        fn = int(cm[:, c].sum() - tp)
        macro_f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {
        "protocol": "4-side only; WBF proposal + weighted class probabilities + train-prior heuristic linker; counting is raw linked-cluster count, not Ridge F_all",
        "counting_method": "raw linked-cluster count (Ridge F_all not applied to this remote detector dump)",
        "n_trees": len(usable), "n_8_side_excluded": skipped,
        "proposal_conf_min": proposal_min, "link_threshold": threshold,
        "singleton_conf_min": singleton_min, "link_pair_mode": pair_mode,
        "max_cluster_size": (max_cluster_size if max_cluster_size is not None
                              else 3),
        "physical_detection": {"precision": p, "recall": r, "f1": f1,
                                "tp": total_tp, "pred_clusters": total_pred,
                                "gt_bunches": total_gt},
        "counting": {"mae": count_abs / max(len(usable), 1),
                     "exact_accuracy": count_exact / max(len(usable), 1),
                     "plus_minus_1_accuracy": count_pm1 / max(len(usable), 1),
                     "vector_exact_accuracy": count_vec_exact / max(len(usable), 1)},
        "classification": {
            "matched_class_accuracy": class_correct / max(matched, 1),
            "matched": matched, "macro_f1_end_to_end": float(np.mean(macro_f1)),
            "per_class_f1_end_to_end": dict(zip(NAMES, macro_f1)),
        },
        "per_tree": per_tree,
    }


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", choices=("new763", "combined1716"),
                    default="new763")
    ap.add_argument("--iou-threshold", type=float, default=.60)
    ap.add_argument("--score-min", type=float, default=.05,
                    help="minimum detector score entering WBF")
    ap.add_argument("--proposal-min", type=float, default=.05,
                    help="minimum fused score entering four-view linker")
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() or 1, 1),
                    help="CPU workers for independent per-image WBF jobs")
    ap.add_argument("--link-threshold", type=float, default=None,
                    help="override the train-calibrated link threshold")
    ap.add_argument("--singleton-min", type=float, default=0.0,
                    help="drop singleton clusters below this fused score")
    ap.add_argument("--pair-mode", choices=("all", "adjacent"), default="all",
                    help="side pairs considered by the linker")
    ap.add_argument("--max-cluster-size", type=int, default=None,
                    help="maximum members in one linked cluster")
    ap.add_argument("--model-weights", nargs="+", type=float, default=None,
                    help="optional positive weights in model order: YOLO, RT-DETR, RF-DETR")
    ap.add_argument("--class-model-weights", nargs="+", type=float, default=None,
                    help="optional class-vote-only weights, row-major model x B1..B4; "
                         "does not change boxes, scores, or linking")
    ap.add_argument("--rfdetr-override", type=Path, default=None,
                    help="replace the RF-DETR prediction dump for SawitMVC-YOLO")
    ap.add_argument("--prediction-override", action="append", default=[],
                    metavar="DATASET:MODEL=PATH",
                    help="replace one prediction dump; DATASET is 953 or depth")
    ap.add_argument("--split", choices=("train", "val", "test"),
                    default="test", help="dataset split to evaluate")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--fused-dir", type=Path, default=None)
    args = ap.parse_args()
    if args.model_weights is not None and len(args.model_weights) != 3:
        ap.error("--model-weights requires exactly three values: YOLO RT-DETR RF-DETR")
    if args.class_model_weights is not None and len(args.class_model_weights) != 12:
        ap.error("--class-model-weights requires exactly 12 values "
                 "(YOLO B1..B4, RT-DETR B1..B4, RF-DETR B1..B4)")
    class_model_weights = (np.asarray(args.class_model_weights, dtype=float).reshape(3, K)
                           if args.class_model_weights is not None else None)
    if class_model_weights is not None and (
            np.any(class_model_weights < 0)
            or np.any(class_model_weights.sum(axis=0) <= 0)):
        ap.error("--class-model-weights must be nonnegative with positive support per class")

    artifact_root = Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27")
    if args.output is None:
        suffix = "testsets" if args.split == "test" else f"{args.split}sets"
        args.output = artifact_root / f"remote_pipeline_{args.bank}_{suffix}.json"
    if args.fused_dir is None:
        suffix = "" if args.split == "test" else f"_{args.split}"
        args.fused_dir = artifact_root / f"fused_{args.bank}{suffix}"
    configs = CONFIGS
    if args.bank == "combined1716":
        pred_root = artifact_root / "predictions_combined1716"
        configs = {}
        for dataset_name, cfg in CONFIGS.items():
            short = "depth" if dataset_name == "SawitMVC-Depth-YOLO" else "953"
            configs[dataset_name] = {
                **cfg,
                "predictions": {
                    "yolo26l": pred_root / f"remote_combined1716_yolo26l_{short}_{args.split}__{args.split}.npz",
                    "rtdetr_l": pred_root / f"remote_combined1716_rtdetr_l_{short}_{args.split}__{args.split}.npz",
                    "rfdetr_l": pred_root / f"remote_combined1716_rfdetr_l_{short}_{args.split}__{args.split}.npz",
                },
            }
    elif args.split != "test":
        configs = {
            dataset_name: {
                **cfg,
                "predictions": {
                    model_name: path.with_name(
                        path.name.replace("_test__test.npz",
                                         f"_{args.split}__{args.split}.npz"))
                    for model_name, path in cfg["predictions"].items()
                },
            }
            for dataset_name, cfg in CONFIGS.items()
        }

    if args.rfdetr_override is not None:
        if args.bank != "combined1716" or args.split != "val":
            ap.error("--rfdetr-override currently requires --bank combined1716 --split val")
        configs["SawitMVC-YOLO"]["predictions"]["rfdetr_l"] = args.rfdetr_override

    for spec in args.prediction_override:
        if "=" not in spec or ":" not in spec.split("=", 1)[0]:
            ap.error("--prediction-override format is DATASET:MODEL=PATH")
        target, raw_path = spec.split("=", 1)
        dataset_key, model_name = target.split(":", 1)
        dataset_name = {
            "953": "SawitMVC-YOLO", "yolo953": "SawitMVC-YOLO",
            "depth": "SawitMVC-Depth-YOLO",
        }.get(dataset_key.lower())
        if dataset_name is None or model_name not in configs[dataset_name]["predictions"]:
            ap.error(f"unknown prediction override target: {target}")
        configs[dataset_name]["predictions"][model_name] = Path(raw_path)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": f"{args.bank}: YOLO26l + RT-DETR-L + RF-DETR-L",
        "wbf": {"iou_threshold": args.iou_threshold,
                "input_score_min": args.score_min,
                "model_weights": args.model_weights,
                "class_model_weights": (class_model_weights.tolist()
                                         if class_model_weights is not None else None)},
        "datasets": {},
    }
    for dataset_name, cfg in configs.items():
        print(f"[{dataset_name}] loading metadata and prediction bank", flush=True)
        split_records = load_records(cfg, args.split)
        train_records = load_records(cfg, "train")
        prior = build_rotation_prior(train_records)
        calibration = calibrate_link_threshold(train_records, prior)
        link_threshold = (args.link_threshold if args.link_threshold is not None
                          else calibration["threshold"])
        bank = load_prediction_bank(cfg)
        model_weights = (np.asarray(args.model_weights, dtype=float)
                         if args.model_weights is not None else None)
        ca, agn, vote = fuse_corpus(split_records, bank, args.iou_threshold,
                                    args.score_min, args.workers,
                                    model_weights, class_model_weights)
        ca_arrays = {k: v for k, v in ca.items()}
        agn_arrays = {k: v for k, v in agn.items()}
        vote_arrays = {
            stem: np.asarray([[*item["box"], item["score"],
                               int(np.argmax(item["p"]))] for item in items], float
                             ).reshape(-1, 6)
            for stem, items in vote.items()
        }
        softvote_arrays = {
            stem: np.asarray([[*item["box"], item["score"], *item["p"]]
                              for item in items], float).reshape(-1, 5 + K)
            for stem, items in vote.items()
        }
        safe = dataset_name.replace("/", "_").replace("-", "_")
        save_npz(args.fused_dir / f"{safe}__wbf_classaware.npz", ca_arrays)
        save_npz(args.fused_dir / f"{safe}__wbf_agnostic.npz", agn_arrays)
        save_npz(args.fused_dir / f"{safe}__wbf_classvote.npz", vote_arrays)
        save_npz(args.fused_dir / f"{safe}__wbf_softvote.npz", softvote_arrays)
        wbf_metrics = {
            "classaware": coco_metrics(cfg["data_root"], ca_arrays, False,
                                        args.split),
            "agnostic": coco_metrics(cfg["data_root"], agn_arrays, True,
                                      args.split),
        }
        mv = multiview_metrics(
            split_records, vote, prior, link_threshold, args.proposal_min,
            args.singleton_min, args.pair_mode, args.max_cluster_size)
        result["datasets"][dataset_name] = {
            f"n_{args.split}_trees_metadata": len(split_records),
            "rotation_prior_train": {f"{n}|{d}": list(v)
                                      for (n, d), v in sorted(prior.items())},
            "link_calibration_train": calibration,
            "wbf_metrics": wbf_metrics,
            "multiview_metrics": mv,
            "fused_files": {
                "classaware": str(args.fused_dir / f"{safe}__wbf_classaware.npz"),
                "agnostic": str(args.fused_dir / f"{safe}__wbf_agnostic.npz"),
                "classvote": str(args.fused_dir / f"{safe}__wbf_classvote.npz"),
                "softvote": str(args.fused_dir / f"{safe}__wbf_softvote.npz"),
            },
        }
        print(json.dumps({"dataset": dataset_name, "wbf": wbf_metrics,
                          "multiview": {k: v for k, v in mv.items()
                                         if k != "per_tree"}}, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
