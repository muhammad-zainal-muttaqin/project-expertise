"""Train a detector-space cross-view linker on train-only proposal pairs.

The original remote pipeline links proposals with a hand-written geometric
score.  This experiment keeps the same four-view protocol, but learns the
probability that two *detector proposals* belong to one physical bunch.  GT
is used only to label train pairs; validation is used for profile selection.
No test data is read by this script.

This is deliberately a small, auditable layer:

    WBF proposals -> pair feature model -> per-side assignment -> constrained
    union-find -> count reconciliation/classification metrics

The learned score is not allowed to merge two proposals from the same side or
to make a cluster larger than the number of views.  The default evaluator is
the same evaluator used by the baseline remote pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_remote_pipeline_postprocess as base  # noqa: E402
import evaluate_remote_class_head as head_eval  # noqa: E402
import evaluate_remote_count_reconciled as count  # noqa: E402
import sweep_remote_pipeline as sweep  # noqa: E402


K = len(base.NAMES)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27")


def safe_name(dataset: str) -> str:
    return "SawitMVC_Depth_YOLO" if dataset == "depth" else "SawitMVC_YOLO"


def cfg_for(dataset: str) -> dict:
    return base.CONFIGS["SawitMVC-Depth-YOLO" if dataset == "depth"
                       else "SawitMVC-YOLO"]


def vote_path(root: Path, dataset: str, split: str) -> Path:
    folder = root / ("fused_combined1716" if split == "test"
                     else f"fused_combined1716_{split}")
    return folder / f"{safe_name(dataset)}__wbf_softvote.npz"


def load_vote(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key], np.float32)
                for key in archive.files}


def normalise(p: np.ndarray) -> np.ndarray:
    p = np.maximum(np.asarray(p, np.float32), 0.)
    return p / max(float(p.sum()), 1e-9)


def make_detections(rec: dict, vote: dict[str, np.ndarray],
                    proposal_min: float) -> list[dict]:
    """Build proposal records and side-local ranks used by the edge model."""
    dets = []
    for side, view in rec["views"].items():
        rows = np.asarray(vote.get(view["stem"], np.zeros((0, 5 + K))),
                          np.float32)
        for row_index, row in enumerate(rows):
            if len(row) < 5 or float(row[4]) < proposal_min:
                continue
            x1, y1, x2, y2 = [float(x) for x in row[:4]]
            p = normalise(row[5:5 + K])
            width, height = max(int(view["width"]), 1), max(int(view["height"]), 1)
            dets.append({
                "side": int(side), "box": np.asarray([x1, y1, x2, y2], float),
                "score": float(row[4]), "p": p, "head_p": p.copy(),
                "stem": view["stem"], "row_index": int(row_index),
                "cx": (x1 + x2) / (2. * width),
                "cy": (y1 + y2) / (2. * height),
                "w": max(x2 - x1, 1.) / width,
                "h": max(y2 - y1, 1.) / height,
            })

    # Proposal ranks and robust within-side coordinates make the model less
    # sensitive to a camera's absolute framing and to the number of proposals.
    by_side = defaultdict(list)
    for i, det in enumerate(dets):
        by_side[det["side"]].append(i)
    for indices in by_side.values():
        values = np.asarray([[dets[i]["cx"], dets[i]["cy"],
                              np.log(max(dets[i]["w"] * dets[i]["h"], 1e-8))]
                             for i in indices], float)
        for col, key in enumerate(("cx", "cy")):
            order = np.argsort(values[:, col], kind="stable")
            ranks = np.empty(len(order), float)
            ranks[order] = np.arange(len(order)) / max(len(order) - 1, 1)
            for j, idx in enumerate(indices):
                dets[idx][f"rank_{key}"] = float(ranks[j])
        med = np.median(values, axis=0)
        scale = 1.4826 * np.median(np.abs(values - med), axis=0)
        scale = np.maximum(scale, [0.03, 0.03, 0.20])
        for j, idx in enumerate(indices):
            dets[idx]["z_side_x"] = float((values[j, 0] - med[0]) / scale[0])
            dets[idx]["z_side_y"] = float((values[j, 1] - med[1]) / scale[1])
            dets[idx]["z_side_area"] = float((values[j, 2] - med[2]) / scale[2])
            dets[idx]["side_count"] = float(len(indices))
    return dets


def pair_features(a: dict, b: dict, n: int,
                  prior: dict[tuple[int, int], tuple[float, ...]]) -> np.ndarray:
    """Feature vector for one ordered cross-view proposal pair."""
    if a["side"] > b["side"]:
        a, b = b, a
    d = (b["side"] - a["side"]) % n
    mux, muy, sx, sy, sarea, _ = prior.get(
        (n, d), (0., 0., .20, .15, .70, 0.))
    dx, dy = b["cx"] - a["cx"], b["cy"] - a["cy"]
    log_area = np.log(max(b["w"] * b["h"], 1e-8) /
                       max(a["w"] * a["h"], 1e-8))
    log_shape = np.log(max(b["w"] / max(b["h"], 1e-8), 1e-8) /
                        max(a["w"] / max(a["h"], 1e-8), 1e-8))
    zdx = (dx - mux) / max(sx, .025)
    zdy = (dy - muy) / max(sy, .025)
    zarea = log_area / max(sarea, .15)
    zshape = log_shape / .85
    pa, pb = a["p"], b["p"]
    sqrt_sim = float(np.sqrt(np.maximum(pa, 0.) * np.maximum(pb, 0.)).sum())
    dot = float(np.dot(pa, pb))
    l1 = float(np.abs(pa - pb).sum())
    ent_a = float(-(pa * np.log(np.maximum(pa, 1e-8))).sum())
    ent_b = float(-(pb * np.log(np.maximum(pb, 1e-8))).sum())
    geom_cost = .5 * (zdx * zdx + zdy * zdy) + .12 * zarea * zarea
    geom_cost += .08 * zshape * zshape + .10 * (1. - sqrt_sim)
    hand_score = float(np.exp(-min(geom_cost, 40.)))
    side_onehot = [float(d == k) for k in range(1, n)]
    return np.asarray([
        # absolute normalized geometry of both proposals
        a["cx"], a["cy"], a["w"], a["h"],
        b["cx"], b["cy"], b["w"], b["h"],
        # relative geometry and robust signed-rotation residuals
        dx, dy, abs(dx), abs(dy), log_area, abs(log_area),
        log_shape, abs(log_shape), zdx, zdy, abs(zdx), abs(zdy),
        zarea, abs(zarea), zshape, abs(zshape), zdx * zdx + zdy * zdy,
        hand_score,
        # proposal confidence and soft class agreement
        a["score"], b["score"], min(a["score"], b["score"]),
        max(a["score"], b["score"]), a["score"] * b["score"],
        sqrt_sim, dot, l1, ent_a, ent_b, abs(ent_a - ent_b),
        float(np.argmax(pa) == np.argmax(pb)), float(pa.max()), float(pb.max()),
        # position of each proposal among same-side candidates
        a.get("rank_cx", .5), a.get("rank_cy", .5), a.get("rank_cx", .5),
        b.get("rank_cx", .5), b.get("rank_cy", .5), b.get("rank_cx", .5),
        a.get("z_side_x", 0.), a.get("z_side_y", 0.), a.get("z_side_area", 0.),
        b.get("z_side_x", 0.), b.get("z_side_y", 0.), b.get("z_side_area", 0.),
        a.get("side_count", 0.), b.get("side_count", 0.), *side_onehot,
        *pa.tolist(), *pb.tolist(),
    ], np.float32)


def gt_labels(rec: dict, dets: list[dict]) -> list[str | None]:
    """One-to-one IoU assignment of proposals to GT appearances."""
    out: list[str | None] = [None] * len(dets)
    for side, view in rec["views"].items():
        indices = [i for i, d in enumerate(dets) if d["side"] == int(side)]
        apps = [(b["id"], np.asarray(a["box"], float))
                for b in rec["bunches"] for a in b["appearances"]
                if int(a["side"]) == int(side)]
        if not indices or not apps:
            continue
        matrix = np.zeros((len(indices), len(apps)), float)
        for r, idx in enumerate(indices):
            for c, (_, box) in enumerate(apps):
                matrix[r, c] = sweep.iou(dets[idx]["box"], box)
        rr, cc = linear_sum_assignment(-matrix)
        for r, c in zip(rr, cc):
            if matrix[r, c] >= .5:
                out[indices[int(r)]] = apps[int(c)][0]
    return out


def build_pair_data(records: dict[str, dict], vote: dict[str, np.ndarray],
                    prior: dict, proposal_min: float,
                    pair_mode: str) -> tuple[np.ndarray, np.ndarray, dict]:
    X, y = [], []
    stats = {"trees": 0, "detections": 0, "pairs": 0, "positive_pairs": 0}
    for rec in records.values():
        dets = make_detections(rec, vote, proposal_min)
        labels = gt_labels(rec, dets)
        by_side = defaultdict(list)
        for i, det in enumerate(dets):
            by_side[det["side"]].append(i)
        sides = sorted(by_side)
        for sa, sb in combinations(sides, 2):
            delta = (sb - sa) % rec["n_sides"]
            if pair_mode == "adjacent" and delta not in (1, rec["n_sides"] - 1):
                continue
            for i in by_side[sa]:
                for j in by_side[sb]:
                    X.append(pair_features(dets[i], dets[j], rec["n_sides"], prior))
                    positive = labels[i] is not None and labels[i] == labels[j]
                    y.append(int(positive))
        stats["trees"] += 1
        stats["detections"] += len(dets)
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.int8)
    stats["pairs"] = int(len(y))
    stats["positive_pairs"] = int(y.sum())
    return X, y, stats


def build_edges(dets: list[dict], n: int, prior: dict, model,
                pair_mode: str) -> list[tuple[float, int, int]]:
    by_side = defaultdict(list)
    for idx, det in enumerate(dets):
        by_side[det["side"]].append(idx)
    edges = []
    sides = sorted(by_side)
    for sa, sb in combinations(sides, 2):
        delta = (sb - sa) % n
        if pair_mode == "adjacent" and delta not in (1, n - 1):
            continue
        aa, bb = by_side[sa], by_side[sb]
        features = np.asarray([
            pair_features(dets[i], dets[j], n, prior)
            for i in aa for j in bb
        ], np.float32)
        if len(features) == 0:
            continue
        scores = model.predict_proba(features)[:, 1].reshape(len(aa), len(bb))
        # Preserve the baseline's one-to-one candidate selection per side
        # pair; only the learned edge quality changes.
        ri, ci = linear_sum_assignment(-scores)
        edges.extend((float(scores[r, c]), aa[r], bb[c])
                      for r, c in zip(ri, ci))
    edges.sort(reverse=True)
    return edges


def payload_for(records: dict[str, dict], vote: dict[str, np.ndarray],
                prior: dict, proposal_min: float, pair_mode: str,
                model=None):
    payload = []
    for rec in records.values():
        dets = make_detections(rec, vote, proposal_min)
        if model is None:
            edges = sweep.build_edges(dets, rec["n_sides"], prior, pair_mode)
        else:
            edges = build_edges(dets, rec["n_sides"], prior, model, pair_mode)
        payload.append((rec, dets, edges))
    return payload


def target_counts(cfg: dict, train_records: dict[str, dict],
                  records: dict[str, dict], train_vote: dict,
                  split_vote: dict, proposal_min: float) -> tuple[dict, dict]:
    y = np.asarray([count.target_count(r) for r in train_records.values()], float)
    x_train = np.stack([count.feature_vector(r, train_vote, proposal_min)
                        for r in train_records.values()])
    alpha, cv = count.choose_alpha(x_train, y)
    model = count.fit_ridge(x_train, y, alpha)
    x_split = np.stack([count.feature_vector(r, split_vote, proposal_min)
                        for r in records.values()])
    pred = count.predict_count(model, x_split)
    return ({tree_id: int(n) for tree_id, n in zip(records, pred)},
            {"alpha": alpha, "cv": cv})


def metric(payload, targets: dict[str, int], link: float, singleton: float,
           max_size: int, rank_mode: str, class_prior: np.ndarray,
           count_blend: float = 0.) -> dict:
    # Reuse the locked metric implementation.  Head probability equals the
    # detector probability, and head_weight=0 makes this a pure linker test.
    return head_eval.evaluate_payload(
        payload, targets, link, singleton, max_size, rank_mode,
        0., class_prior, 0., None, "mean", count_blend)


def metric_key(item: dict) -> tuple:
    m = item["metrics"]
    p = m["physical_detection"]
    c = m["classification"]
    n = m["counting"]
    return (float(c["matched_class_accuracy"]), float(p["f1"]),
            -float(n["mae"]), float(c["macro_f1_end_to_end"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("depth", "953"), required=True)
    ap.add_argument("--fused-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--proposal-min", type=float, default=.125)
    ap.add_argument("--pair-mode", choices=("all", "adjacent"), default="adjacent")
    ap.add_argument("--max-size", type=int, default=3)
    ap.add_argument("--workers", type=int, default=-1,
                    help="ExtraTrees workers; -1 uses all CPU cores")
    ap.add_argument("--models", nargs="+",
                    choices=("hist", "hist_deep", "extra"),
                    default=("hist", "hist_deep", "extra"),
                    help="models to fit; use one or two for a fast focused run")
    args = ap.parse_args()

    cfg = cfg_for(args.dataset)
    train_records = count.four_side(base.load_records(cfg, "train"))
    val_records = count.four_side(base.load_records(cfg, "val"))
    train_vote = load_vote(vote_path(args.fused_root, args.dataset, "train"))
    val_vote = load_vote(vote_path(args.fused_root, args.dataset, "val"))
    prior = base.build_rotation_prior(base.load_records(cfg, "train"))
    print(json.dumps({"dataset": args.dataset, "train_trees": len(train_records),
                      "val_trees": len(val_records),
                      "proposal_min": args.proposal_min,
                      "pair_mode": args.pair_mode}, ensure_ascii=False), flush=True)

    x_train, y_train, pair_stats = build_pair_data(
        train_records, train_vote, prior, args.proposal_min, args.pair_mode)
    x_val, y_val, val_pair_stats = build_pair_data(
        val_records, val_vote, prior, args.proposal_min, args.pair_mode)
    print(json.dumps({"train_pairs": pair_stats, "val_pairs": val_pair_stats,
                      "features": int(x_train.shape[1]),
                      "positive_rate": float(y_train.mean())}, ensure_ascii=False), flush=True)
    if y_train.min() == y_train.max():
        raise RuntimeError("pair labels contain only one class")

    pos_weight = (len(y_train) - int(y_train.sum())) / max(int(y_train.sum()), 1)
    sample_weight = np.where(y_train == 1, min(pos_weight, 30.), 1.).astype(np.float32)
    candidates = {
        "hist": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=.05, max_leaf_nodes=31,
            l2_regularization=10., random_state=42),
        "hist_deep": HistGradientBoostingClassifier(
            max_iter=260, learning_rate=.045, max_leaf_nodes=63,
            l2_regularization=15., random_state=43),
        "extra": ExtraTreesClassifier(
            # 180 trees is enough for stable ranking on these pair features;
            # larger forests made 953 validation disproportionately slow.
            n_estimators=180, min_samples_leaf=3, max_features=.8,
            class_weight="balanced", n_jobs=args.workers, random_state=42),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    joblib.dump({"X": x_train, "y": y_train, "stats": pair_stats},
                args.output_root / "pair_training_data.joblib", compress=3)

    train_class_prior = np.bincount(
        [b["cls"] for r in train_records.values() for b in r["bunches"]
         if 0 <= b["cls"] < K], minlength=K).astype(float)
    train_class_prior /= max(float(train_class_prior.sum()), 1.)
    targets, count_info = target_counts(
        cfg, train_records, val_records, train_vote, val_vote, args.proposal_min)
    base_payload = payload_for(val_records, val_vote, prior, args.proposal_min,
                               args.pair_mode)
    links = np.arange(.10, .91, .05)
    singletons = (.10, .15, .20, .25)
    ranks = ("score", "support", "max_member")
    results = []

    def sweep_model(name: str, model, weights=None):
        if weights is None:
            model.fit(x_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(x_train, y_train)
        train_prob = model.predict_proba(x_train)[:, 1]
        val_prob = model.predict_proba(x_val)[:, 1]
        diagnostics = {
            "train_auc": float(roc_auc_score(y_train, train_prob)),
            "val_auc": float(roc_auc_score(y_val, val_prob)),
            "train_ap": float(average_precision_score(y_train, train_prob)),
            "val_ap": float(average_precision_score(y_val, val_prob)),
        }
        print(json.dumps({"model": name, **diagnostics}, ensure_ascii=False), flush=True)
        model_path = args.output_root / f"{name}.joblib"
        joblib.dump(model, model_path, compress=3)
        learned_payload = payload_for(val_records, val_vote, prior,
                                      args.proposal_min, args.pair_mode, model)
        for link in links:
            for singleton in singletons:
                for rank in ranks:
                    m = metric(learned_payload, targets, float(link), float(singleton),
                               args.max_size, rank, train_class_prior)
                    item = {"model": name, "link_threshold": float(link),
                            "singleton_min": float(singleton),
                            "rank_mode": rank, "metrics": m}
                    results.append(item)
        return diagnostics

    diagnostics = {}
    diagnostics["baseline"] = {
        "note": "hand-written pair score",
    }
    base_best = None
    for link in links:
        for singleton in singletons:
            for rank in ranks:
                m = metric(base_payload, targets, float(link), float(singleton),
                           args.max_size, rank, train_class_prior)
                item = {"model": "baseline", "link_threshold": float(link),
                        "singleton_min": float(singleton), "rank_mode": rank,
                        "metrics": m}
                results.append(item)
                if base_best is None or metric_key(item) > metric_key(base_best):
                    base_best = item
    diagnostics["baseline_best"] = base_best
    for name, model in candidates.items():
        if name not in args.models:
            continue
        diagnostics[name] = sweep_model(name, model)

    best_class = max(results, key=metric_key)
    best_physical = max(results, key=lambda item: (
        item["metrics"]["physical_detection"]["f1"],
        item["metrics"]["classification"]["matched_class_accuracy"],
        -item["metrics"]["counting"]["mae"]))
    output = {
        "dataset": args.dataset, "fit_split": "train",
        "selection_split": "val", "test_used": False,
        "proposal_min": args.proposal_min, "pair_mode": args.pair_mode,
        "max_size": args.max_size, "feature_count": int(x_train.shape[1]),
        "train_pair_stats": pair_stats, "val_pair_stats": val_pair_stats,
        "count_model": count_info, "class_prior": train_class_prior.tolist(),
        "diagnostics": diagnostics, "best_by_class": best_class,
        "best_by_physical_f1": best_physical, "results": results,
    }
    out_path = args.output_root / "results.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"best_by_class": best_class, "best_by_physical_f1": best_physical,
                      "output": str(out_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
