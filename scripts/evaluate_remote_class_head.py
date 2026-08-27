"""Evaluate a post-cluster B1--B4 head without changing physical linking.

The detector probabilities remain the sole input to WBF, four-view linking,
and count reconciliation.  A second probability vector (for example from a
colour/context head) is averaged only inside the already-linked physical
cluster and used for the final class decision.  This isolates classification
gain from association/counting gain and prevents a weak crop head from
changing the geometry of the pipeline.

All count features, rotation priors, and class-head training data come from
TRAIN.  The script is intended for validation selection; TEST must be run
only after the blend/profile is locked.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_remote_pipeline_postprocess as base  # noqa: E402
import evaluate_remote_count_reconciled as count  # noqa: E402
import sweep_remote_pipeline as sweep  # noqa: E402


K = len(base.NAMES)


def load_vote(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {stem: np.asarray(archive[stem], np.float32)
                for stem in archive.files}


def resolve_head_path(folder: Path, safe: str, split: str) -> Path:
    candidates = (
        folder / f"{safe}__wbf_softvote.npz",
        folder / f"fused_{split}__wbf_softvote.npz",
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def normalise(p: np.ndarray) -> np.ndarray:
    p = np.maximum(np.asarray(p, np.float32), 0.)
    return p / max(float(p.sum()), 1e-9)


def make_detections(rec: dict, base_vote: dict[str, np.ndarray],
                    head_vote: dict[str, np.ndarray],
                    proposal_min: float) -> list[dict]:
    """Create linker detections with a separate, post-cluster class vector."""
    dets = []
    for side, view in rec["views"].items():
        rows = np.asarray(base_vote.get(view["stem"], np.zeros((0, 5 + K))),
                          np.float32)
        hrows = np.asarray(head_vote.get(view["stem"], rows), np.float32)
        if len(rows) != len(hrows):
            raise ValueError(f"proposal count mismatch for {view['stem']}")
        for i, row in enumerate(rows):
            if row[4] < proposal_min:
                continue
            p = normalise(row[5:5 + K])
            hp = normalise(hrows[i, 5:5 + K])
            x1, y1, x2, y2 = [float(x) for x in row[:4]]
            width, height = max(view["width"], 1), max(view["height"], 1)
            dets.append({
                "side": int(side), "box": np.asarray([x1, y1, x2, y2]),
                "score": float(row[4]), "p": p, "head_p": hp,
                # Keep the originating proposal address available to later
                # group-level heads.  The fused vote row is immutable, so
                # this is metadata only and cannot change linking/counting.
                "stem": view["stem"], "row_index": int(i),
                "cx": (x1 + x2) / (2 * width),
                "cy": (y1 + y2) / (2 * height),
                "w": max(x2 - x1, 1.) / width,
                "h": max(y2 - y1, 1.) / height,
            })
    return dets


def build_payload(records: dict[str, dict], base_vote: dict[str, np.ndarray],
                  head_vote: dict[str, np.ndarray], prior: dict,
                  proposal_min: float, pair_mode: str):
    payload = []
    for rec in records.values():
        dets = make_detections(rec, base_vote, head_vote, proposal_min)
        edges = sweep.build_edges(dets, rec["n_sides"], prior, pair_mode)
        payload.append((rec, dets, edges))
    return payload


def aggregate_probability(group: dict, key: str, mode: str) -> np.ndarray:
    members = group["members"]
    values = np.stack([normalise(x[key]) for x in members])
    weights = np.asarray([max(x["score"], 1e-6) for x in members], np.float32)
    if mode == "max_member":
        p = values[int(np.argmax(weights))]
    elif mode == "max_class":
        p = values.max(0)
    elif mode == "geometric":
        p = np.exp(np.average(np.log(np.maximum(values, 1e-6)), axis=0,
                              weights=weights))
    elif mode == "median":
        p = np.median(values, axis=0)
    else:
        p = np.average(values, axis=0, weights=weights)
    return normalise(p)


def head_probability(group: dict, head_weight: float,
                     max_base_margin: float | None = None,
                     aggregation: str = "mean") -> np.ndarray:
    members = group["members"]
    hp = aggregate_probability(group, "head_p", aggregation)
    bp = (normalise(group["p"]) if aggregation == "mean"
          else aggregate_probability(group, "p", aggregation))
    if max_base_margin is not None:
        ordered = np.sort(bp)
        if float(ordered[-1] - ordered[-2]) > max_base_margin:
            return bp
    return normalise((1. - head_weight) * bp + head_weight * hp)


def evaluate_payload(payload, target_counts: dict[str, int], link_threshold: float,
                     singleton_min: float, max_size: int, rank_mode: str,
                     head_weight: float, class_prior: np.ndarray,
                     class_prior_exponent: float,
                     max_base_margin: float | None = None,
                     aggregation: str = "mean", count_blend: float = 0.) -> dict:
    cm = np.zeros((K + 1, K + 1), int)
    total_pred = total_gt = total_tp = 0
    abs_count = exact = pm1 = vector_exact = 0
    class_correct = matched = 0
    for rec, dets, edges in payload:
        raw = sweep.clusters(dets, edges, link_threshold, singleton_min,
                             max_size)
        raw_count = len(sweep.clusters(dets, edges, link_threshold,
                                       singleton_min, max_size))
        target = int(round((1. - count_blend) * target_counts[rec["tree_id"]]
                           + count_blend * raw_count))
        groups = count.selected_clusters(dets, edges, link_threshold,
                                         singleton_min, max_size, target,
                                         rank_mode)
        for group in groups:
            p = head_probability(group, head_weight, max_base_margin,
                                 aggregation)
            if class_prior_exponent:
                p = normalise(p * np.power(np.maximum(class_prior, 1e-9),
                                           class_prior_exponent))
            group["cls"] = int(np.argmax(p))
        matches = count.tree_matches(rec, groups)
        bunches = rec["bunches"]
        total_pred += len(groups); total_gt += len(bunches)
        total_tp += len(matches); matched += len(matches)
        delta = len(groups) - len(bunches)
        abs_count += abs(delta); exact += int(delta == 0)
        pm1 += int(abs(delta) <= 1)
        pred_count = np.bincount([x["cls"] for x in groups], minlength=K)
        gt_count = np.bincount([x["cls"] for x in bunches if x["cls"] >= 0],
                               minlength=K)
        vector_exact += int(np.array_equal(pred_count, gt_count))
        matched_pred = {i for i, _ in matches}
        matched_gt = {j for _, j in matches}
        for i, j in matches:
            pc, gc = groups[i]["cls"], bunches[j]["cls"]
            if 0 <= pc < K and 0 <= gc < K:
                cm[pc, gc] += 1
                class_correct += int(pc == gc)
        for i, group in enumerate(groups):
            if i not in matched_pred and 0 <= group["cls"] < K:
                cm[group["cls"], K] += 1
        for j, bunch in enumerate(bunches):
            if j not in matched_gt and 0 <= bunch["cls"] < K:
                cm[K, bunch["cls"]] += 1
    p = total_tp / max(total_pred, 1)
    r = total_tp / max(total_gt, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    f1s = []
    for c in range(K):
        tp = cm[c, c]
        fp = int(cm[c, :].sum() - tp)
        fn = int(cm[:, c].sum() - tp)
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {
        "physical_detection": {
            "precision": p, "recall": r, "f1": f1,
            "tp": total_tp, "pred_clusters": total_pred,
            "gt_bunches": total_gt,
        },
        "counting": {
            "mae": abs_count / max(len(payload), 1),
            "exact_accuracy": exact / max(len(payload), 1),
            "plus_minus_1_accuracy": pm1 / max(len(payload), 1),
            "vector_exact_accuracy": vector_exact / max(len(payload), 1),
        },
        "classification": {
            "matched_class_accuracy": class_correct / max(matched, 1),
            "matched": matched,
            "macro_f1_end_to_end": float(np.mean(f1s)),
            "per_class_f1_end_to_end": dict(zip(base.NAMES, f1s)),
            "confusion_prediction_rows": cm.tolist(),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("depth", "953"), required=True)
    ap.add_argument("--base-fused-dir", type=Path, required=True,
                    help="directory containing baseline __wbf_softvote.npz")
    ap.add_argument("--head-fused-dir", type=Path, required=True,
                    help="directory containing post-head __wbf_softvote.npz")
    ap.add_argument("--fit-base-fused-dir", type=Path, required=True)
    ap.add_argument("--fit-head-fused-dir", type=Path, required=True)
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--proposal-min", type=float, required=True)
    ap.add_argument("--link-threshold", type=float, required=True)
    ap.add_argument("--singleton-min", type=float, required=True)
    ap.add_argument("--max-size", type=int, default=3)
    ap.add_argument("--pair-mode", choices=("all", "adjacent"), default="adjacent")
    ap.add_argument("--rank-mode", choices=("score", "support", "max_member", "class_conf",
                                             "head_conf", "joint_conf", "support_head",
                                             "class_conf_power_0.25", "class_conf_power_0.5",
                                             "class_conf_power_0.75", "head_conf_power_0.25",
                                             "head_conf_power_0.5", "head_conf_power_0.75",
                                             "score_conf_0.5_0.25", "score_conf_0.75_0.25",
                                             "score_conf_1_0.1", "score_conf_1_0.2",
                                             "score_conf_1_0.35", "score_conf_1.25_0.25",
                                             "score_conf_1.5_0.25", "score_conf_2_0.25"),
                    default="support")
    ap.add_argument("--head-weights", nargs="+", type=float,
                    default=[0., .25, .5, .75, 1.])
    ap.add_argument("--max-base-margins", nargs="+", type=float,
                    default=[-1.],
                    help="apply head only when detector p top1-top2 <= margin; -1 disables gate")
    ap.add_argument("--aggregations", nargs="+",
                    choices=("mean", "max_member", "max_class", "geometric", "median"),
                    default=["mean"],
                    help="how member-view class probabilities are combined after linking")
    ap.add_argument("--class-prior-exponents", nargs="+", type=float,
                    default=[-.25, 0.])
    ap.add_argument("--count-blends", nargs="+", type=float, default=[0.],
                    help="mix predicted count with raw linked-cluster count")
    ap.add_argument("--workers", type=int, default=1,
                    help="reserved for future payload parallelism")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not all(0. <= x <= 1. for x in args.head_weights):
        ap.error("head weight harus berada di [0,1]")
    dataset_name = "SawitMVC-Depth-YOLO" if args.dataset == "depth" else "SawitMVC-YOLO"
    cfg = base.CONFIGS[dataset_name]
    safe = "SawitMVC_Depth_YOLO" if args.dataset == "depth" else "SawitMVC_YOLO"
    split_base = resolve_head_path(args.base_fused_dir, safe, args.split)
    split_head = resolve_head_path(args.head_fused_dir, safe, args.split)
    train_base = resolve_head_path(args.fit_base_fused_dir, safe, "train")
    train_head = resolve_head_path(args.fit_head_fused_dir, safe, "train")
    for path in (split_base, split_head, train_base, train_head):
        if not path.exists():
            raise FileNotFoundError(path)
    train_records = count.four_side(base.load_records(cfg, "train"))
    split_records = count.four_side(base.load_records(cfg, args.split))
    base_train_vote = load_vote(train_base)
    head_train_vote = load_vote(train_head)
    base_split_vote = load_vote(split_base)
    head_split_vote = load_vote(split_head)
    prior = base.build_rotation_prior(base.load_records(cfg, "train"))
    train_class_prior = np.bincount(
        [b["cls"] for rec in train_records.values() for b in rec["bunches"]
         if 0 <= b["cls"] < K], minlength=K).astype(float)
    train_class_prior /= max(float(train_class_prior.sum()), 1.)
    y_train = np.asarray([count.target_count(r) for r in train_records.values()], float)
    X_train = np.stack([count.feature_vector(r, base_train_vote, args.proposal_min)
                        for r in train_records.values()])
    alpha, cv = count.choose_alpha(X_train, y_train)
    model = count.fit_ridge(X_train, y_train, alpha)
    split_payload = build_payload(split_records, base_split_vote, head_split_vote,
                                  prior, args.proposal_min, args.pair_mode)
    target_counts = {}
    X_split = np.stack([count.feature_vector(r, base_split_vote, args.proposal_min)
                        for r in split_records.values()])
    for tree_id, n in zip(split_records, count.predict_count(model, X_split)):
        target_counts[tree_id] = int(n)
    results = []
    for aggregation in args.aggregations:
        for margin in args.max_base_margins:
            gate = None if margin < 0 else margin
            for exponent in args.class_prior_exponents:
                for weight in args.head_weights:
                    for count_blend in args.count_blends:
                        result = evaluate_payload(
                            split_payload, target_counts, args.link_threshold,
                            args.singleton_min, args.max_size, args.rank_mode, weight,
                            train_class_prior, exponent, gate, aggregation,
                            count_blend)
                        results.append({"head_weight": weight,
                                        "class_prior_exponent": exponent,
                                        "max_base_margin": margin,
                                        "aggregation": aggregation,
                                        "count_blend": count_blend,
                                        "metrics": result})
                        m = result
                        print(json.dumps({"head_weight": weight,
                                          "class_prior_exponent": exponent,
                                          "max_base_margin": margin,
                                          "aggregation": aggregation,
                                          "count_blend": count_blend,
                                          "physical_f1": m["physical_detection"]["f1"],
                                          "count_mae": m["counting"]["mae"],
                                          "matched_class_accuracy": m["classification"]["matched_class_accuracy"],
                                          "macro_f1": m["classification"]["macro_f1_end_to_end"]},
                                         ensure_ascii=False), flush=True)
    output = {
        "dataset": dataset_name, "split": args.split,
        "protocol": "baseline linker/count + post-cluster class head",
        "fit_split": "train", "selection_split": "validation",
        "proposal_min": args.proposal_min,
        "link_threshold": args.link_threshold,
        "singleton_min": args.singleton_min,
        "max_size": args.max_size, "pair_mode": args.pair_mode,
        "rank_mode": args.rank_mode, "count_model_alpha": alpha,
        "max_base_margins": args.max_base_margins,
        "aggregations": args.aggregations,
        "count_cv": cv, "class_prior_train": train_class_prior.tolist(),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
