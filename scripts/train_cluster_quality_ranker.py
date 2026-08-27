"""Fit a train-only linked-cluster quality ranker.

The normal count reconciler selects the requested number of clusters by a
detector score.  This harness learns an objectness/quality probability for a
linked cluster from TRAIN-only geometry, support and detector votes, then
tests quality-weighted ranking on validation.  The class decision remains a
separate post-cluster RGB head, so a gain can be attributed to selecting the
right physical candidates rather than to label leakage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_remote_pipeline_postprocess as base  # noqa: E402
import evaluate_remote_class_head as post  # noqa: E402
import evaluate_remote_count_reconciled as count  # noqa: E402
import sweep_remote_pipeline as sweep  # noqa: E402

K = len(base.NAMES)


def load_vote(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key], np.float32)
                for key in archive.files}


def vote_path(folder: Path, safe: str) -> Path:
    return folder / f"{safe}__wbf_softvote.npz"


def quality_features(group: dict, rec: dict) -> np.ndarray:
    members = group["members"]
    p = np.stack([post.normalise(x["p"]) for x in members])
    scores = np.asarray([x["score"] for x in members], float)
    geom = np.asarray([[x["cx"], x["cy"], x["w"], x["h"]]
                       for x in members], float)
    sides = {int(x["side"]) for x in members}
    # These are all available before GT matching at inference.
    return np.asarray([
        float(group["score"]), float(scores.max()), float(scores.min()),
        float(scores.mean()), float(scores.std()), float(len(members)),
        *[float(i in sides) for i in range(rec["n_sides"])],
        *p.mean(0).tolist(), *p.max(0).tolist(), *p.min(0).tolist(),
        *p.std(0).tolist(),
        float(p.mean(0).max()),
        float(np.sort(p.mean(0))[-1] - np.sort(p.mean(0))[-2]),
        float((-p.mean(0) * np.log(np.maximum(p.mean(0), 1e-8))).sum()),
        *geom.mean(0).tolist(), *geom.std(0).tolist(),
    ], np.float32)


def max_group_iou(rec: dict, group: dict) -> float:
    best = 0.
    for member in group["members"]:
        for bunch in rec["bunches"]:
            for appearance in bunch["appearances"]:
                if int(member["side"]) != int(appearance["side"]):
                    continue
                best = max(best, float(base.iou_one(
                    member["box"], np.asarray([appearance["box"]]))[0]))
    return best


def make_groups(cfg: dict, split: str, vote: dict[str, np.ndarray],
                prior: dict, proposal_min: float, link: float,
                singleton: float, max_size: int):
    records = count.four_side(base.load_records(cfg, split))
    out = []
    for rec in records.values():
        # Use the metadata-preserving post head constructor so each member
        # remains addressable when the optional visual head is injected.
        dets = post.make_detections(rec, vote, vote, proposal_min)
        edges = sweep.build_edges(dets, rec["n_sides"], prior, "adjacent")
        groups = sweep.clusters(dets, edges, link, singleton, max_size)
        out.append((rec, dets, edges, groups))
    return records, out


def fit_quality(x: np.ndarray, y: np.ndarray, name: str):
    if name == "logreg":
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=.3, max_iter=2000, class_weight="balanced"))
    elif name == "hist_gbdt":
        model = HistGradientBoostingClassifier(
            max_iter=180, learning_rate=.04, max_leaf_nodes=15,
            min_samples_leaf=20, l2_regularization=2., random_state=42)
    elif name == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=400, max_features="sqrt", min_samples_leaf=8,
            class_weight="balanced", n_jobs=16, random_state=42)
    else:
        raise ValueError(name)
    model.fit(x, y)
    return model


def select(groups, target: int, model, mode: str):
    if model is None:
        values = np.asarray([g["score"] for g in groups], float)
    else:
        q = model.predict_proba(np.stack([g["quality_x"] for g in groups]))[:, 1]
        if mode == "quality":
            values = q
        else:
            _, power = mode.split("_", 1)
            values = np.asarray([g["score"] for g in groups], float) * np.maximum(q, 1e-6) ** float(power)
    order = np.argsort(-values, kind="stable")
    return [groups[int(i)] for i in order[:max(int(target), 0)]]


def evaluate(groups_payload, targets, model, rank_mode, head_weight):
    cm = np.zeros((K + 1, K + 1), int)
    total_pred = total_gt = total_tp = correct = matched = 0
    abs_count = exact = pm1 = 0
    for rec, _dets, _edges, groups in groups_payload:
        chosen = select(groups, targets[rec["tree_id"]], model, rank_mode)
        for g in chosen:
            p = post.head_probability(g, head_weight, None, "mean")
            g["cls"] = int(np.argmax(p))
        matches = count.tree_matches(rec, chosen)
        total_pred += len(chosen); total_gt += len(rec["bunches"])
        total_tp += len(matches); matched += len(matches)
        delta = len(chosen) - len(rec["bunches"])
        abs_count += abs(delta); exact += int(delta == 0); pm1 += int(abs(delta) <= 1)
        for i, j in matches:
            pc, gc = chosen[i]["cls"], rec["bunches"][j]["cls"]
            if 0 <= pc < K and 0 <= gc < K:
                cm[pc, gc] += 1; correct += int(pc == gc)
        mp = {i for i, _ in matches}; mg = {j for _, j in matches}
        for i, g in enumerate(chosen):
            if i not in mp and 0 <= g["cls"] < K: cm[g["cls"], K] += 1
        for j, b in enumerate(rec["bunches"]):
            if j not in mg and 0 <= b["cls"] < K: cm[K, b["cls"]] += 1
    precision = total_tp / max(total_pred, 1); recall = total_tp / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    f1s = []
    for c in range(K):
        tp = cm[c, c]; fp = int(cm[c].sum() - tp); fn = int(cm[:, c].sum() - tp)
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {"physical_f1": f1, "precision": precision, "recall": recall,
            "pred_clusters": total_pred, "gt_bunches": total_gt, "tp": total_tp,
            "count_mae": abs_count / max(len(groups_payload), 1),
            "count_exact": exact / max(len(groups_payload), 1),
            "count_pm1": pm1 / max(len(groups_payload), 1),
            "matched_class_accuracy": correct / max(matched, 1),
            "matched": matched, "macro_f1": float(np.mean(f1s)),
            "per_class_f1": dict(zip(base.NAMES, f1s)), "confusion": cm.tolist()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("953", "depth"), default="953")
    ap.add_argument("--base-train-dir", type=Path, required=True)
    ap.add_argument("--base-val-dir", type=Path, required=True)
    ap.add_argument("--head-train-dir", type=Path, required=True)
    ap.add_argument("--head-val-dir", type=Path, required=True)
    ap.add_argument("--proposal-min", type=float, default=.125)
    ap.add_argument("--link-threshold", type=float, default=.30)
    ap.add_argument("--singleton-min", type=float, default=.15)
    ap.add_argument("--max-size", type=int, default=3)
    ap.add_argument("--models", nargs="+", default=["logreg", "hist_gbdt", "extra_trees"])
    ap.add_argument("--quality-modes", nargs="+",
                    default=["score", "quality", "score_0.25", "score_0.5",
                             "score_0.75", "score_1.0"])
    ap.add_argument("--head-weights", nargs="+", type=float, default=[0., .2, .3, .4])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    name = "SawitMVC-Depth-YOLO" if args.dataset == "depth" else "SawitMVC-YOLO"
    safe = "SawitMVC_Depth_YOLO" if args.dataset == "depth" else "SawitMVC_YOLO"
    cfg = base.CONFIGS[name]
    btr = load_vote(vote_path(args.base_train_dir, safe)); bva = load_vote(vote_path(args.base_val_dir, safe))
    htr = load_vote(args.head_train_dir / "fused_train__wbf_softvote.npz")
    hva = load_vote(args.head_val_dir / "fused_val__wbf_softvote.npz")
    prior = base.build_rotation_prior(base.load_records(cfg, "train"))
    train_records, train_payload = make_groups(cfg, "train", btr, prior, args.proposal_min,
                                                args.link_threshold, args.singleton_min, args.max_size)
    val_records, val_payload = make_groups(cfg, "val", bva, prior, args.proposal_min,
                                            args.link_threshold, args.singleton_min, args.max_size)
    # Inject class-head probabilities into the same groups used by the quality model.
    for payload, hv in ((train_payload, htr), (val_payload, hva)):
        for rec, dets, edges, groups in payload:
            for g in groups:
                for m in g["members"]:
                    if m["stem"] in hv:
                        row = hv[m["stem"]][m["row_index"]]
                        m["head_p"] = post.normalise(row[5:5 + K])
                g["quality_x"] = quality_features(g, rec)
    x_train, y_train = [], []
    for rec, _dets, _edges, groups in train_payload:
        for g in groups:
            x_train.append(g["quality_x"]); y_train.append(int(max_group_iou(rec, g) >= .5))
    x_train = np.asarray(x_train, np.float32); y_train = np.asarray(y_train, np.int64)
    x_val = []
    for _rec, _dets, _edges, groups in val_payload:
        x_val.extend(g["quality_x"] for g in groups)
    print(json.dumps({"train_groups": len(y_train), "positive": int(y_train.sum()),
                      "val_groups": len(x_val), "features": x_train.shape[1]}, ensure_ascii=False), flush=True)
    # Train the count model on TRAIN only.
    y_count = np.asarray([count.target_count(r) for r in train_records.values()], float)
    xc = np.stack([count.feature_vector(r, btr, args.proposal_min) for r in train_records.values()])
    alpha, count_cv = count.choose_alpha(xc, y_count); count_model = count.fit_ridge(xc, y_count, alpha)
    xv = np.stack([count.feature_vector(r, bva, args.proposal_min) for r in val_records.values()])
    targets = {k: int(v) for k, v in zip(val_records, count.predict_count(count_model, xv))}
    results = []
    for model_name in args.models:
        model = None if model_name == "none" else fit_quality(x_train, y_train, model_name)
        for mode in args.quality_modes:
            if mode == "score" and model is not None:
                continue
            for weight in args.head_weights:
                metrics = evaluate(val_payload, targets, model, mode, weight)
                item = {"model": model_name, "rank_mode": mode,
                        "head_weight": weight, "metrics": metrics}
                results.append(item)
                print(json.dumps({"model": model_name, "rank_mode": mode,
                                  "head_weight": weight, **{k: metrics[k] for k in
                                  ("physical_f1", "count_mae", "matched_class_accuracy", "macro_f1")}}),
                      flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"protocol": "train-only cluster quality ranker",
                                       "dataset": name, "fit_split": "train",
                                       "selection_split": "val", "proposal_min": args.proposal_min,
                                       "link_threshold": args.link_threshold,
                                       "singleton_min": args.singleton_min, "max_size": args.max_size,
                                       "train_groups": len(y_train), "positive_groups": int(y_train.sum()),
                                       "count_model_alpha": alpha, "count_cv": count_cv,
                                       "results": results}, indent=2) + "\n", encoding="utf-8")
    best = max(results, key=lambda x: x["metrics"]["matched_class_accuracy"])
    print(json.dumps({"output": str(args.output), "best": best}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
