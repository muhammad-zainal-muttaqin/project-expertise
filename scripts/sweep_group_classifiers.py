"""Train-only group classifiers for the final B1--B4 decision.

This is a validation-only experiment harness.  The detector, WBF proposals,
linker, and train-only count model stay fixed; only the class decision for
count-reconciled clusters is replaced by a scikit-learn classifier trained on
matched TRAIN groups.  It is useful for testing nonlinear interactions in
view consistency and detector probabilities without touching TEST labels.
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
import evaluate_remote_count_reconciled as count  # noqa: E402
import train_group_class_meta as group  # noqa: E402


K = len(base.NAMES)


def vote_path(folder: Path) -> Path:
    return folder / "SawitMVC_YOLO__wbf_softvote.npz"


def load_vote(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key], np.float32)
                for key in archive.files}


def estimator(name: str, seed: int, jobs: int):
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.3, max_iter=1000, solver="lbfgs",
                               class_weight=None, random_state=seed),
        )
    if name == "logreg_balanced":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.3, max_iter=1000, solver="lbfgs",
                               class_weight="balanced", random_state=seed),
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500, max_features="sqrt", min_samples_leaf=5,
            class_weight=None, random_state=seed, n_jobs=jobs,
        )
    if name == "extra_trees_balanced":
        return ExtraTreesClassifier(
            n_estimators=500, max_features="sqrt", min_samples_leaf=5,
            class_weight="balanced", random_state=seed, n_jobs=jobs,
        )
    if name == "hist_gbdt":
        return HistGradientBoostingClassifier(
            max_iter=250, learning_rate=.04, max_leaf_nodes=15,
            min_samples_leaf=20, l2_regularization=2., random_state=seed,
        )
    raise ValueError(f"unknown model: {name}")


def evaluate(payload, target_counts: dict[str, int], model, mean: np.ndarray,
             scale: np.ndarray, link_threshold: float, singleton_min: float,
             max_size: int, rank_mode: str, class_prior: np.ndarray,
             prior_exponent: float) -> dict:
    cm = np.zeros((K + 1, K + 1), int)
    total_pred = total_gt = total_tp = 0
    abs_count = exact = pm1 = vector_exact = 0
    correct = matched = 0
    for rec, dets, edges, _raw_groups in payload:
        target = target_counts[rec["tree_id"]]
        groups = count.selected_clusters(
            dets, edges, link_threshold, singleton_min, max_size, target,
            rank_mode)
        if groups:
            x = np.stack([group.group_features(g, rec) for g in groups])
            probs = model.predict_proba((x - mean) / scale)
        else:
            probs = np.zeros((0, K), float)
        for item, p in zip(groups, probs):
            if prior_exponent:
                p = p * np.power(np.maximum(class_prior, 1e-9),
                                 prior_exponent)
                p /= max(float(p.sum()), 1e-9)
            item["cls"] = int(np.argmax(p))
        matches = count.tree_matches(rec, groups)
        bunches = rec["bunches"]
        total_pred += len(groups); total_gt += len(bunches)
        total_tp += len(matches); matched += len(matches)
        delta = len(groups) - len(bunches)
        abs_count += abs(delta); exact += int(delta == 0)
        pm1 += int(abs(delta) <= 1)
        pred_count = np.bincount([x["cls"] for x in groups], minlength=K)
        gt_count = np.bincount([x["cls"] for x in bunches
                                if 0 <= x["cls"] < K], minlength=K)
        vector_exact += int(np.array_equal(pred_count, gt_count))
        matched_pred = {i for i, _ in matches}
        matched_gt = {j for _, j in matches}
        for i, j in matches:
            pc, gc = groups[i]["cls"], bunches[j]["cls"]
            if 0 <= pc < K and 0 <= gc < K:
                cm[pc, gc] += 1
                correct += int(pc == gc)
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
        fp = int(cm[c].sum() - tp)
        fn = int(cm[:, c].sum() - tp)
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {
        "physical_detection": {
            "precision": precision, "recall": recall, "f1": f1,
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
            "matched_class_accuracy": correct / max(matched, 1),
            "matched": matched,
            "macro_f1_end_to_end": float(np.mean(f1s)),
            "per_class_f1_end_to_end": dict(zip(base.NAMES, f1s)),
            "confusion_prediction_rows": cm.tolist(),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-train-dir", type=Path, required=True)
    ap.add_argument("--base-val-dir", type=Path, required=True)
    ap.add_argument("--proposal-min", type=float, default=.125)
    ap.add_argument("--link-threshold", type=float, default=.30)
    ap.add_argument("--singleton-min", type=float, default=.15)
    ap.add_argument("--max-size", type=int, default=3)
    ap.add_argument("--pair-mode", choices=("all", "adjacent"), default="adjacent")
    ap.add_argument("--rank-mode", choices=("score", "support", "max_member", "class_conf"),
                    default="class_conf")
    ap.add_argument("--models", nargs="+", default=[
        "logreg", "logreg_balanced", "extra_trees",
        "extra_trees_balanced", "hist_gbdt",
    ])
    ap.add_argument("--prior-exponents", nargs="+", type=float,
                    default=[-.25, 0.])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    cfg = base.CONFIGS["SawitMVC-YOLO"]
    train_vote = load_vote(vote_path(args.base_train_dir))
    val_vote = load_vote(vote_path(args.base_val_dir))
    prior = base.build_rotation_prior(base.load_records(cfg, "train"))
    train_payload = group.make_payload(
        cfg, "953", "train", train_vote, train_vote, prior,
        args.proposal_min, args.pair_mode, args.link_threshold,
        args.singleton_min, args.max_size)
    val_payload = group.make_payload(
        cfg, "953", "val", val_vote, val_vote, prior,
        args.proposal_min, args.pair_mode, args.link_threshold,
        args.singleton_min, args.max_size)
    x_train, y_train = group.matched_training_data(train_payload)
    if len(x_train) < 100:
        raise RuntimeError("too few matched training groups")
    mean, scale = x_train.mean(0), x_train.std(0)
    scale[scale < 1e-6] = 1.
    x_train_z = (x_train - mean) / scale

    train_records = count.four_side(base.load_records(cfg, "train"))
    val_records = count.four_side(base.load_records(cfg, "val"))
    y_count = np.asarray([count.target_count(r) for r in train_records.values()], float)
    x_count = np.stack([count.feature_vector(r, train_vote, args.proposal_min)
                        for r in train_records.values()])
    alpha, count_cv = count.choose_alpha(x_count, y_count)
    count_model = count.fit_ridge(x_count, y_count, alpha)
    x_val_count = np.stack([count.feature_vector(r, val_vote, args.proposal_min)
                            for r in val_records.values()])
    target_counts = {tree_id: int(n) for tree_id, n in zip(
        val_records, count.predict_count(count_model, x_val_count))}
    class_prior = np.bincount(
        [b["cls"] for r in train_records.values() for b in r["bunches"]
         if 0 <= b["cls"] < K], minlength=K).astype(float)
    class_prior /= max(float(class_prior.sum()), 1.)

    results = []
    for model_name in args.models:
        model = estimator(model_name, args.seed, args.jobs)
        model.fit(x_train_z, y_train)
        for exponent in args.prior_exponents:
            metrics = evaluate(
                val_payload, target_counts, model, mean, scale,
                args.link_threshold, args.singleton_min, args.max_size,
                args.rank_mode, class_prior, exponent)
            item = {"model": model_name, "prior_exponent": exponent,
                    "metrics": metrics}
            results.append(item)
            print(json.dumps({
                "model": model_name, "prior_exponent": exponent,
                "physical_f1": metrics["physical_detection"]["f1"],
                "count_mae": metrics["counting"]["mae"],
                "matched_class_accuracy": metrics["classification"]["matched_class_accuracy"],
                "macro_f1": metrics["classification"]["macro_f1_end_to_end"],
            }), flush=True)

    output = {
        "protocol": "train-only nonlinear group classifier; validation selection",
        "dataset": "SawitMVC-YOLO", "fit_split": "train",
        "selection_split": "val", "proposal_min": args.proposal_min,
        "link_threshold": args.link_threshold,
        "singleton_min": args.singleton_min, "max_size": args.max_size,
        "pair_mode": args.pair_mode, "rank_mode": args.rank_mode,
        "train_groups_matched": int(len(y_train)),
        "train_class_counts": np.bincount(y_train, minlength=K).tolist(),
        "count_model_alpha": alpha, "count_cv": count_cv,
        "class_prior_train": class_prior.tolist(), "models": args.models,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
