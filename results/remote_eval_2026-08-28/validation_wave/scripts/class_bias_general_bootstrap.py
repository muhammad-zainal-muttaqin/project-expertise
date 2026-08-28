#!/usr/bin/env python3
"""Paired tree bootstrap for the validation-only class-bias candidates.

This analysis reconstructs only TRAIN-fitted expert opinions and VAL frozen
groups.  It aggregates per-tree confusion matrices, so matched accuracy and
end-to-end macro-F1 are resampled as paired deltas.  It never accepts or reads
a TEST split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import class_bias_general as experiment
import harness


OUT = Path("/workspace/cluster_head/artifacts")
K = harness.K


def tree_confusions(data: dict, baseline_pred: np.ndarray,
                    candidate_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    baseline = []
    candidate = []
    offset = 0
    for record, groups in data["groups"]:
        matches = dict(harness.count.tree_matches(record, groups))
        matched_gt = set(matches.values())
        base_cm = np.zeros((K + 1, K + 1), dtype=np.int64)
        cand_cm = np.zeros((K + 1, K + 1), dtype=np.int64)
        for group_index in range(len(groups)):
            pred_b = int(baseline_pred[offset + group_index])
            pred_c = int(candidate_pred[offset + group_index])
            if group_index in matches:
                gt = int(record["bunches"][matches[group_index]]["cls"])
                if 0 <= gt < K:
                    base_cm[pred_b, gt] += 1
                    cand_cm[pred_c, gt] += 1
            else:
                base_cm[pred_b, K] += 1
                cand_cm[pred_c, K] += 1
        for gt_index, bunch in enumerate(record["bunches"]):
            if gt_index not in matched_gt:
                gt = int(bunch["cls"])
                if 0 <= gt < K:
                    base_cm[K, gt] += 1
                    cand_cm[K, gt] += 1
        baseline.append(base_cm)
        candidate.append(cand_cm)
        offset += len(groups)
    if offset != len(baseline_pred) or offset != len(candidate_pred):
        raise RuntimeError("tree confusion/prediction length mismatch")
    return np.asarray(baseline), np.asarray(candidate)


def metrics_from_cm(cm: np.ndarray) -> tuple[float, float, int]:
    matched = int(cm[:K, :K].sum())
    correct = int(np.trace(cm[:K, :K]))
    f1 = []
    for cls in range(K):
        tp = int(cm[cls, cls])
        fp = int(cm[cls].sum() - tp)
        fn = int(cm[:, cls].sum() - tp)
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return correct / max(matched, 1), float(np.mean(f1)), matched


def delta_ci(base_tree: np.ndarray, cand_tree: np.ndarray,
             seed: int, resamples: int) -> dict:
    rng = np.random.RandomState(seed)
    n = len(base_tree)
    deltas_matched = np.empty(resamples, dtype=np.float64)
    deltas_macro = np.empty(resamples, dtype=np.float64)
    positive_matched = 0
    positive_macro = 0
    for i in range(resamples):
        indices = rng.randint(0, n, size=n)
        base_cm = base_tree[indices].sum(axis=0)
        cand_cm = cand_tree[indices].sum(axis=0)
        base_matched, base_macro, _ = metrics_from_cm(base_cm)
        cand_matched, cand_macro, _ = metrics_from_cm(cand_cm)
        deltas_matched[i] = cand_matched - base_matched
        deltas_macro[i] = cand_macro - base_macro
        positive_matched += int(deltas_matched[i] > 0.)
        positive_macro += int(deltas_macro[i] > 0.)
    return {
        "resamples": int(resamples),
        "unit": "tree",
        "seed": int(seed),
        "matched_delta": float(deltas_matched.mean()),
        "matched_ci95": [float(x) for x in np.percentile(deltas_matched, [2.5, 97.5])],
        "matched_p_positive": float(positive_matched / resamples),
        "macro_delta": float(deltas_macro.mean()),
        "macro_ci95": [float(x) for x in np.percentile(deltas_macro, [2.5, 97.5])],
        "macro_p_positive": float(positive_macro / resamples),
    }


def run(dataset: str, spec_name: str, selection: str,
        seed: int, resamples: int) -> dict:
    report_path = OUT / f"{dataset}_class_bias_general_results_val.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    spec = report["specs"][spec_name]
    data = experiment.mh.collect(dataset, "val")
    views = experiment.build_views(dataset, data)
    detector = np.asarray(
        [np.asarray(group["p"], dtype=np.float32)
         for _record, groups in data["groups"] for group in groups]
    )
    detector = np.maximum(detector, 1e-8)
    detector /= np.maximum(detector.sum(axis=1, keepdims=True), 1e-8)
    logits = np.log(detector)
    for view_name, weight in spec["weights"].items():
        logits += float(weight) * np.log(np.maximum(views[view_name], 1e-8))
    if selection not in ("bias_matched", "bias_macro", "scale_matched", "scale_macro"):
        raise ValueError(selection)
    candidate_key = {
        "bias_matched": "best_by_matched",
        "bias_macro": "best_by_macro",
        "scale_matched": "scale_grid",
        "scale_macro": "scale_grid",
    }[selection]
    chosen = (spec[candidate_key]
              if selection.startswith("bias")
              else spec["scale_grid"]["best_by_matched"
                                      if selection == "scale_matched"
                                      else "best_by_macro"])
    if selection.startswith("scale"):
        logits = logits * np.asarray(chosen["scales"], dtype=np.float32)
    candidate = np.argmax(
        logits + np.asarray(chosen["bias"], dtype=np.float32), axis=1
    )
    baseline = np.argmax(detector, axis=1)
    base_tree, cand_tree = tree_confusions(data, baseline, candidate)
    base_point = metrics_from_cm(base_tree.sum(axis=0))
    cand_point = metrics_from_cm(cand_tree.sum(axis=0))
    output = {
        "dataset": dataset,
        "protocol": "paired tree bootstrap for TRAIN-fitted class head; VAL only; no TEST",
        "spec": spec_name,
        "candidate_selection": selection,
        "candidate_bias": chosen["bias"],
        "candidate_scales": chosen.get("scales"),
        "baseline_point": {
            "matched_class_accuracy": base_point[0],
            "macro_f1": base_point[1], "matched": base_point[2],
        },
        "candidate_point": {
            "matched_class_accuracy": cand_point[0],
            "macro_f1": cand_point[1], "matched": cand_point[2],
        },
        "bootstrap": delta_ci(base_tree, cand_tree, seed, resamples),
        "tree_count": int(len(base_tree)),
    }
    path = OUT / f"{dataset}_class_bias_general_bootstrap_{spec_name}_{selection}.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(json.dumps({"dataset": dataset, "spec": spec_name,
                      "baseline": output["baseline_point"],
                      "candidate": output["candidate_point"],
                      "bootstrap": output["bootstrap"],
                      "report": str(path)}, ensure_ascii=False), flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("953", "depth"), required=True)
    parser.add_argument("--spec", default=None)
    parser.add_argument("--selection", default=None,
                        choices=("bias_matched", "bias_macro",
                                 "scale_matched", "scale_macro"))
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--resamples", type=int, default=5000)
    args = parser.parse_args()
    if args.spec is None:
        args.spec = "robust_953_anchor" if args.dataset == "953" else "member_stack_macro"
    if args.selection is None:
        args.selection = "bias_matched" if args.dataset == "953" else "scale_matched"
    run(args.dataset, args.spec, args.selection, args.seed, args.resamples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
