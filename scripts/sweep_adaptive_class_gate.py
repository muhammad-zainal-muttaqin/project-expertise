"""Learn when a visual crop head should override the detector class vote.

The physical proposal/link/count path is fixed.  A binary gate is fitted only
on matched TRAIN clusters where the detector and crop head disagree and one
of them is correct.  Validation is then used to select the gate threshold;
the script does not inspect TEST labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_remote_pipeline_postprocess as base  # noqa: E402
import evaluate_remote_class_head as post  # noqa: E402
import evaluate_remote_count_reconciled as count  # noqa: E402


K = len(base.NAMES)


def load_vote(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key], np.float32)
                for key in archive.files}


def vote_path(folder: Path, dataset: str, split: str) -> Path:
    safe = "SawitMVC_Depth_YOLO" if dataset == "depth" else "SawitMVC_YOLO"
    return folder / f"{safe}__wbf_softvote.npz"


def probability(group: dict, key: str, exponent: float,
                class_prior: np.ndarray) -> np.ndarray:
    p = post.aggregate_probability(group, key, "mean")
    if exponent:
        p = post.normalise(p * np.power(np.maximum(class_prior, 1e-9), exponent))
    return p


def feature_vector(group: dict, bp: np.ndarray, hp: np.ndarray) -> np.ndarray:
    members = group["members"]
    scores = np.asarray([x["score"] for x in members], np.float32)
    support = np.asarray(sorted({int(x["side"]) for x in members}), np.float32)
    bp_margin = np.sort(bp)[-1] - np.sort(bp)[-2]
    hp_margin = np.sort(hp)[-1] - np.sort(hp)[-2]
    bp_entropy = -float(np.sum(bp * np.log(np.maximum(bp, 1e-8))))
    hp_entropy = -float(np.sum(hp * np.log(np.maximum(hp, 1e-8))))
    onehot = np.asarray([float(i in support) for i in range(4)], np.float32)
    return np.asarray([
        *bp.tolist(), *hp.tolist(),
        *np.log(np.maximum(bp, 1e-8)).tolist(),
        *np.log(np.maximum(hp, 1e-8)).tolist(),
        *(hp - bp).tolist(), *np.abs(hp - bp).tolist(),
        float(bp.max()), float(hp.max()), float(bp_margin), float(hp_margin),
        bp_entropy, hp_entropy,
        float(np.argmax(bp)), float(np.argmax(hp)),
        float(np.argmax(bp) == np.argmax(hp)),
        float(scores.mean()), float(scores.max()), float(scores.min()),
        float(scores.std()), float(len(members)), *onehot,
    ], np.float32)


def make_payload(cfg: dict, split: str, base_vote: dict, head_vote: dict,
                 prior: dict, proposal_min: float, pair_mode: str):
    records = count.four_side(base.load_records(cfg, split))
    payload = post.build_payload(records, base_vote, head_vote, prior,
                                 proposal_min, pair_mode)
    return records, payload


def collect_gate_data(payload, target_counts, class_prior, prior_exponent,
                      link_threshold, singleton_min, max_size, rank_mode):
    features, labels, stats = [], [], {"matched": 0, "disagreements": 0,
                                       "useful": 0}
    for rec, dets, edges in payload:
        groups = count.selected_clusters(
            dets, edges, link_threshold, singleton_min, max_size,
            target_counts[rec["tree_id"]], rank_mode)
        matches = count.tree_matches(rec, groups)
        for i, j in matches:
            gt = int(rec["bunches"][j]["cls"])
            bp = probability(groups[i], "p", prior_exponent, class_prior)
            hp = probability(groups[i], "head_p", prior_exponent, class_prior)
            bc, hc = int(bp.argmax()), int(hp.argmax())
            stats["matched"] += 1
            if bc == hc:
                continue
            stats["disagreements"] += 1
            base_ok, head_ok = bc == gt, hc == gt
            if base_ok == head_ok:
                continue
            stats["useful"] += 1
            features.append(feature_vector(groups[i], bp, hp))
            labels.append(int(head_ok))
    return np.asarray(features, np.float32), np.asarray(labels, np.int64), stats


def evaluate(payload, target_counts, class_prior, prior_exponent, gate=None,
             threshold=.5, blend=.1, link_threshold=.30,
             singleton_min=.15, max_size=3, rank_mode="max_member"):
    cm = np.zeros((K + 1, K + 1), np.int64)
    total_gt = total_pred = total_tp = correct = matched = 0
    abs_count = exact = pm1 = 0
    for rec, dets, edges in payload:
        groups = count.selected_clusters(
            dets, edges, link_threshold, singleton_min, max_size,
            target_counts[rec["tree_id"]], rank_mode)
        for item in groups:
            bp = probability(item, "p", prior_exponent, class_prior)
            hp = probability(item, "head_p", prior_exponent, class_prior)
            bc, hc = int(bp.argmax()), int(hp.argmax())
            if gate is None:
                item["cls"] = int(np.argmax((1. - blend) * bp + blend * hp))
            elif bc == hc:
                item["cls"] = bc
            else:
                x = feature_vector(item, bp, hp)[None]
                q = float(gate.predict_proba(x)[0, 1])
                item["cls"] = hc if q >= threshold else bc
        matches = count.tree_matches(rec, groups)
        bunches = rec["bunches"]
        total_gt += len(bunches); total_pred += len(groups)
        total_tp += len(matches); matched += len(matches)
        delta = len(groups) - len(bunches)
        abs_count += abs(delta); exact += int(delta == 0); pm1 += int(abs(delta) <= 1)
        for i, j in matches:
            pc, gc = groups[i]["cls"], bunches[j]["cls"]
            if 0 <= pc < K and 0 <= gc < K:
                cm[pc, gc] += 1
                correct += int(pc == gc)
        matched_pred = {i for i, _ in matches}
        matched_gt = {j for _, j in matches}
        for i, item in enumerate(groups):
            if i not in matched_pred and 0 <= item["cls"] < K:
                cm[item["cls"], K] += 1
        for j, item in enumerate(bunches):
            if j not in matched_gt and 0 <= item["cls"] < K:
                cm[K, item["cls"]] += 1
    precision = total_tp / max(total_pred, 1)
    recall = total_tp / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    f1s = []
    for c in range(K):
        tp = cm[c, c]
        fp = int(cm[c].sum() - tp); fn = int(cm[:, c].sum() - tp)
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {
        "physical_detection": {"precision": precision, "recall": recall,
                                "f1": f1, "tp": total_tp,
                                "pred_clusters": total_pred,
                                "gt_bunches": total_gt},
        "counting": {"mae": abs_count / max(len(payload), 1),
                     "exact_accuracy": exact / max(len(payload), 1),
                     "plus_minus_1_accuracy": pm1 / max(len(payload), 1)},
        "classification": {
            "matched_class_accuracy": correct / max(matched, 1),
            "matched": matched, "macro_f1_end_to_end": float(np.mean(f1s)),
            "per_class_f1_end_to_end": dict(zip(base.NAMES, f1s)),
            "confusion_prediction_rows": cm.tolist(),
        },
    }


def build_model(name: str, seed: int):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=.2, max_iter=2000, class_weight="balanced",
            random_state=seed))
    if name == "logreg_low_c":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=.03, max_iter=2000, class_weight="balanced",
            random_state=seed))
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=500, max_features="sqrt",
                                    min_samples_leaf=8, class_weight="balanced",
                                    random_state=seed, n_jobs=8)
    if name == "hist_gbdt":
        return HistGradientBoostingClassifier(
            max_iter=180, learning_rate=.04, max_leaf_nodes=7,
            min_samples_leaf=12, l2_regularization=2., random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500, max_features="sqrt", min_samples_leaf=8,
            class_weight="balanced", random_state=seed, n_jobs=8)
    raise ValueError(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("depth", "953"), required=True)
    ap.add_argument("--base-train-dir", type=Path, required=True)
    ap.add_argument("--base-val-dir", type=Path, required=True)
    ap.add_argument("--head-dir", type=Path, required=True)
    ap.add_argument("--proposal-min", type=float, default=.125)
    ap.add_argument("--link-threshold", type=float, default=.30)
    ap.add_argument("--singleton-min", type=float, default=.15)
    ap.add_argument("--max-size", type=int, default=3)
    ap.add_argument("--pair-mode", choices=("all", "adjacent"), default="adjacent")
    ap.add_argument("--rank-mode", choices=("score", "support", "max_member", "class_conf"),
                    default="max_member")
    ap.add_argument("--prior-exponent", type=float, default=0.)
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[.35, .45, .50, .55, .65, .75])
    ap.add_argument("--models", nargs="+",
                    default=["logreg", "logreg_low_c", "extra_trees"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    cfg = base.CONFIGS["SawitMVC-Depth-YOLO" if args.dataset == "depth"
                       else "SawitMVC-YOLO"]
    train_records = count.four_side(base.load_records(cfg, "train"))
    val_records = count.four_side(base.load_records(cfg, "val"))
    btrain = load_vote(vote_path(args.base_train_dir, args.dataset, "train"))
    bval = load_vote(vote_path(args.base_val_dir, args.dataset, "val"))
    htrain = load_vote(args.head_dir / f"fused_train__wbf_softvote.npz")
    hval = load_vote(args.head_dir / f"fused_val__wbf_softvote.npz")
    prior_rotation = base.build_rotation_prior(base.load_records(cfg, "train"))
    prior_class = np.bincount(
        [b["cls"] for r in train_records.values() for b in r["bunches"]
         if 0 <= b["cls"] < K], minlength=K).astype(float)
    prior_class /= max(float(prior_class.sum()), 1.)
    x_count = np.stack([count.feature_vector(r, btrain, args.proposal_min)
                        for r in train_records.values()])
    y_count = np.asarray([count.target_count(r) for r in train_records.values()], float)
    alpha, cv = count.choose_alpha(x_count, y_count)
    count_model = count.fit_ridge(x_count, y_count, alpha)
    train_targets = {k: int(v) for k, v in zip(
        train_records, count.predict_count(count_model, x_count))}
    x_val_count = np.stack([count.feature_vector(r, bval, args.proposal_min)
                            for r in val_records.values()])
    val_targets = {k: int(v) for k, v in zip(
        val_records, count.predict_count(count_model, x_val_count))}
    train_records, train_payload = make_payload(
        cfg, "train", btrain, htrain, prior_rotation, args.proposal_min,
        args.pair_mode)
    val_records, val_payload = make_payload(
        cfg, "val", bval, hval, prior_rotation, args.proposal_min,
        args.pair_mode)
    x_train, y_train, train_stats = collect_gate_data(
        train_payload, train_targets, prior_class, args.prior_exponent,
        args.link_threshold, args.singleton_min, args.max_size, args.rank_mode)
    x_val, y_val, val_stats = collect_gate_data(
        val_payload, val_targets, prior_class, args.prior_exponent,
        args.link_threshold, args.singleton_min, args.max_size, args.rank_mode)
    if len(x_train) < 30 or len(np.unique(y_train)) < 2:
        raise RuntimeError(f"gate training data insufficient: {train_stats}")
    results = []
    for model_name in args.models:
        model = build_model(model_name, args.seed)
        model.fit(x_train, y_train)
        for threshold in args.thresholds:
            metrics = evaluate(
                val_payload, val_targets, prior_class, args.prior_exponent,
                model, threshold, link_threshold=args.link_threshold,
                singleton_min=args.singleton_min, max_size=args.max_size,
                rank_mode=args.rank_mode)
            item = {"model": model_name, "threshold": threshold,
                    "metrics": metrics}
            results.append(item)
            c = metrics["classification"]
            print(json.dumps({"model": model_name, "threshold": threshold,
                              "matched_class_accuracy": c["matched_class_accuracy"],
                              "macro_f1": c["macro_f1_end_to_end"],
                              "physical_f1": metrics["physical_detection"]["f1"]}),
                  flush=True)
    output = {
        "protocol": "train-only binary gate between detector and visual crop head",
        "dataset": cfg["kind"], "fit_split": "train", "selection_split": "val",
        "proposal_min": args.proposal_min, "link_threshold": args.link_threshold,
        "singleton_min": args.singleton_min, "max_size": args.max_size,
        "pair_mode": args.pair_mode, "rank_mode": args.rank_mode,
        "prior_exponent": args.prior_exponent, "class_prior": prior_class.tolist(),
        "count_model_alpha": alpha, "count_cv": cv,
        "train_gate_stats": train_stats, "val_gate_stats": val_stats,
        "train_gate_rows": int(len(x_train)), "val_gate_rows": int(len(x_val)),
        "models": args.models, "thresholds": args.thresholds, "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
