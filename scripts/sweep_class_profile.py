"""Fast validation sweep for class-aware 4-view selection profiles.

The detector probabilities and crop-head probabilities are fixed.  Only the
cluster selection/ranking profile is varied, so this isolates whether the
remaining gap is caused by which physical clusters are retained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import eval_remote_pipeline_postprocess as base
import evaluate_remote_class_head as head
import evaluate_remote_count_reconciled as count
import sweep_remote_pipeline as sweep


def load_vote(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key], np.float32)
                for key in archive.files}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("depth", "953"), default="953")
    ap.add_argument("--base-fused-dir", type=Path, required=True)
    ap.add_argument("--head-fused-dir", type=Path, required=True)
    ap.add_argument("--fit-base-fused-dir", type=Path, required=True)
    ap.add_argument("--fit-head-fused-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--proposal-min", nargs="+", type=float,
                    default=[.10, .125, .15])
    ap.add_argument("--link-threshold", nargs="+", type=float,
                    default=[.25, .30, .35])
    ap.add_argument("--singleton-min", nargs="+", type=float,
                    default=[.10, .15, .20])
    ap.add_argument("--max-size", nargs="+", type=int, default=[3, 4])
    ap.add_argument("--rank-mode", nargs="+", default=["max_member", "support", "class_conf"],
                    choices=["score", "support", "max_member", "class_conf",
                             "head_conf", "joint_conf", "support_head",
                             "class_conf_power_0.25", "class_conf_power_0.5",
                             "class_conf_power_0.75", "head_conf_power_0.25",
                             "head_conf_power_0.5", "head_conf_power_0.75",
                             "score_conf_0.5_0.25", "score_conf_0.75_0.25",
                             "score_conf_1_0.1", "score_conf_1_0.2",
                             "score_conf_1_0.35", "score_conf_1.25_0.25",
                             "score_conf_1.5_0.25", "score_conf_2_0.25"])
    ap.add_argument("--head-weight", nargs="+", type=float,
                    default=[.15, .25, .30, .40])
    ap.add_argument("--class-prior-exponent", nargs="+", type=float,
                    default=[-.25, 0.0])
    ap.add_argument("--margin", nargs="+", type=float, default=[.05])
    ap.add_argument("--aggregation", default="mean",
                    choices=["mean", "max_member", "max_class", "geometric", "median"])
    ap.add_argument("--count-blend", nargs="+", type=float, default=[0.])
    args = ap.parse_args()
    dataset_name = "SawitMVC-Depth-YOLO" if args.dataset == "depth" else "SawitMVC-YOLO"
    cfg = base.CONFIGS[dataset_name]
    safe = "SawitMVC_Depth_YOLO" if args.dataset == "depth" else "SawitMVC_YOLO"
    split_base = load_vote(args.base_fused_dir / f"{safe}__wbf_softvote.npz")
    split_head = load_vote(head.resolve_head_path(args.head_fused_dir, safe, "val"))
    train_base = load_vote(args.fit_base_fused_dir / f"{safe}__wbf_softvote.npz")
    train_head = load_vote(head.resolve_head_path(args.fit_head_fused_dir, safe, "train"))
    train_records = count.four_side(base.load_records(cfg, "train"))
    split_records = count.four_side(base.load_records(cfg, "val"))
    prior = base.build_rotation_prior(base.load_records(cfg, "train"))
    y_train = np.asarray([count.target_count(r) for r in train_records.values()], float)
    X_train = np.stack([count.feature_vector(r, train_base, .125)
                        for r in train_records.values()])
    alpha, cv = count.choose_alpha(X_train, y_train)
    model = count.fit_ridge(X_train, y_train, alpha)
    results = []
    class_prior = np.bincount(
        [b["cls"] for rec in train_records.values() for b in rec["bunches"]
         if 0 <= b["cls"] < head.K], minlength=head.K).astype(float)
    class_prior /= max(float(class_prior.sum()), 1.)
    # Build each proposal/link graph once per proposal-min and pair mode.
    for proposal_min in args.proposal_min:
        X_split = np.stack([count.feature_vector(r, split_base, proposal_min)
                            for r in split_records.values()])
        target_counts = {
            tree_id: int(n) for tree_id, n in
            zip(split_records, count.predict_count(model, X_split))}
        payload = head.build_payload(split_records, split_base, split_head,
                                     prior, proposal_min, "adjacent")
        for link_threshold in args.link_threshold:
            for singleton_min in args.singleton_min:
                for max_size in args.max_size:
                    for rank_mode in args.rank_mode:
                        for exponent in args.class_prior_exponent:
                            for weight in args.head_weight:
                                for margin in args.margin:
                                    for count_blend in args.count_blend:
                                        metrics = head.evaluate_payload(
                                            payload, target_counts, link_threshold,
                                            singleton_min, max_size, rank_mode, weight,
                                            class_prior, exponent, margin,
                                            args.aggregation, count_blend)
                                        results.append({
                                            "proposal_min": proposal_min,
                                            "link_threshold": link_threshold,
                                            "singleton_min": singleton_min,
                                            "max_size": max_size,
                                            "rank_mode": rank_mode,
                                            "head_weight": weight,
                                            "class_prior_exponent": exponent,
                                            "max_base_margin": margin,
                                            "aggregation": args.aggregation,
                                            "count_blend": count_blend,
                                            "metrics": metrics,
                                        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {"dataset": dataset_name, "split": "val",
              "fit_split": "train", "count_model_alpha": alpha,
              "count_cv": cv, "n_results": len(results),
              "results": results}
    args.output.write_text(json.dumps(output, indent=2) + "\n",
                           encoding="utf-8")
    best = max(results, key=lambda x: x["metrics"]["classification"]["matched_class_accuracy"])
    print(json.dumps({"output": str(args.output), "n_results": len(results),
                      "best": {**{k: best[k] for k in (
                          "proposal_min", "link_threshold", "singleton_min",
                          "max_size", "rank_mode", "head_weight",
                          "class_prior_exponent")},
                               "metrics": best["metrics"]}}, ensure_ascii=False),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
