#!/usr/bin/env python3
"""Validation-only head-aware cluster selection.

The frozen linker emits more candidate clusters than the predicted tree count
in some cases.  The reference pipeline ranks those candidates by linker
score.  This experiment gives the already TRAIN-fitted class head a residual
skip path: candidate confidence may break the truncation tie, while the
linker score remains the primary signal.  The detector, edge linker, count
targets, and TEST split are otherwise untouched.

The experiment is deliberately a small, declared frontier.  It tests the
two strongest existing member heads, fractional confidence exponents, and
the already locked validation class calibration.  It is not allowed to fit
or read a TEST artifact.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import joblib
import numpy as np

import class_bias_general as calibration
import harness
import member_head as mh
import sweep_remote_pipeline as sweep


OUT = Path("/workspace/cluster_head/artifacts")
K = harness.K


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    values = np.exp(logits)
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-8)


def short(metrics: dict) -> dict:
    return {
        "physical_f1": float(metrics["physical_detection"]["f1"]),
        "mae": float(metrics["counting"]["mae"]),
        "pm1": float(metrics["counting"]["plus_minus_1_accuracy"]),
        "matched_class_accuracy": float(metrics["classification"]["matched_class_accuracy"]),
        "matched": int(metrics["classification"]["matched"]),
        "macro_f1": float(metrics["classification"]["macro_f1_end_to_end"]),
        "per_class_f1": metrics["classification"]["per_class_f1_end_to_end"],
    }


def member_features_for_dets(dataset: str, dets: list[dict], split: str = "val"):
    fmap, dim = mh._load_fmap(dataset, split)
    x, keys, missing = [], [], 0
    for det in dets:
        key = (str(det["stem"]), int(det["row_index"]))
        if key not in fmap:
            missing += 1
        x.append(mh.member_feature(det, fmap, dim))
        keys.append(key)
    if not x:
        return np.zeros((0, dim + K + 4 + 11), dtype=np.float32), keys, missing
    return np.asarray(x, dtype=np.float32), keys, missing


def raw_group_data(dataset: str, payload: list[tuple], targets: dict[str, int],
                   profile: dict) -> tuple[dict, list[list[dict]]]:
    """Build all raw candidate groups and a member-feature view for them."""
    grouped = []
    flat_dets = []
    for rec, dets, edges in payload:
        groups = sweep.clusters(
            dets, edges, profile["link_threshold"], profile["singleton_min"],
            profile["max_size"]
        )
        # The clustering helper returns fresh group dictionaries, but keep a
        # private copy so adding head probabilities cannot alter an anchor.
        groups = [copy.deepcopy(g) for g in groups]
        grouped.append((rec, groups))
        flat_dets.extend(dets)

    # Detections repeat across trees only through their own stem/row key; the
    # map is keyed below, so a single batched prediction is safe.
    x, keys, missing = member_features_for_dets(dataset, flat_dets, "val")
    return {
        "groups": grouped,
        "detection_features": x,
        "detection_keys": keys,
        "missing_features": missing,
    }, grouped


def attach_member_head(groups: list[dict], det_q: dict[tuple[str, int], np.ndarray],
                       model_name: str) -> None:
    for group in groups:
        q = np.asarray([det_q[(str(m["stem"]), int(m["row_index"]))]
                        for m in group["members"]], dtype=np.float32)
        if model_name.endswith("_max"):
            head_p = q.max(axis=0)
        elif model_name.endswith("_mean"):
            weights = np.asarray([float(m["score"]) for m in group["members"]],
                                 dtype=np.float32)
            weights /= max(float(weights.sum()), 1e-8)
            head_p = (q * weights[:, None]).sum(axis=0)
        else:
            raise ValueError(model_name)
        head_p = np.maximum(head_p, 1e-8)
        group["head_p"] = head_p / max(float(head_p.sum()), 1e-8)


def select_groups(groups: list[dict], target: int, rank_source: str,
                  power: float) -> list[dict]:
    out = [copy.deepcopy(g) for g in groups]
    if rank_source == "linker":
        key = lambda g: (float(g["score"]), len(g["members"]))
    elif rank_source == "head":
        key = lambda g: (
            float(g["score"]) * max(float(np.asarray(g["head_p"]).max()), 1e-8) ** power,
            float(g["score"]), len(g["members"]))
    else:
        raise ValueError(rank_source)
    out.sort(key=key, reverse=True)
    return out[:max(int(target), 0)]


def evaluate_grouped(grouped: list[tuple[dict, list[dict]]]) -> dict:
    cm = np.zeros((K + 1, K + 1), dtype=np.int64)
    total_pred = total_gt = total_tp = 0
    abs_count = exact = pm1 = vector_exact = class_correct = matched = 0
    for rec, groups in grouped:
        bunches = rec["bunches"]
        total_pred += len(groups)
        total_gt += len(bunches)
        matches = harness.count.tree_matches(rec, groups)
        total_tp += len(matches)
        delta = len(groups) - len(bunches)
        abs_count += abs(delta)
        exact += int(delta == 0)
        pm1 += int(abs(delta) <= 1)
        pred_count = np.bincount([int(g["cls"]) for g in groups], minlength=K)
        gt_count = np.bincount([int(b["cls"]) for b in bunches if int(b["cls"]) >= 0],
                               minlength=K)
        vector_exact += int(np.array_equal(pred_count, gt_count))
        matched += len(matches)
        matched_pred = {i for i, _ in matches}
        matched_gt = {j for _, j in matches}
        for i, j in matches:
            pc, gc = int(groups[i]["cls"]), int(bunches[j]["cls"])
            if 0 <= pc < K and 0 <= gc < K:
                cm[pc, gc] += 1
                class_correct += int(pc == gc)
        for i, g in enumerate(groups):
            if i not in matched_pred and 0 <= int(g["cls"]) < K:
                cm[int(g["cls"]), K] += 1
        for j, b in enumerate(bunches):
            if j not in matched_gt and 0 <= int(b["cls"]) < K:
                cm[K, int(b["cls"])] += 1
    precision = total_tp / max(total_pred, 1)
    recall = total_tp / max(total_gt, 1)
    f1 = 2. * precision * recall / max(precision + recall, 1e-12)
    f1s = []
    for c in range(K):
        tp = int(cm[c, c])
        fp = int(cm[c].sum() - tp)
        fn = int(cm[:, c].sum() - tp)
        f1s.append(2. * tp / max(2 * tp + fp + fn, 1))
    n = len(grouped)
    return {
        "physical_detection": {"precision": precision, "recall": recall,
                                "f1": f1, "tp": total_tp,
                                "pred_clusters": total_pred, "gt_bunches": total_gt},
        "counting": {"mae": abs_count / max(n, 1),
                      "exact_accuracy": exact / max(n, 1),
                      "plus_minus_1_accuracy": pm1 / max(n, 1),
                      "vector_exact_accuracy": vector_exact / max(n, 1)},
        "classification": {
            "matched_class_accuracy": class_correct / max(matched, 1),
            "matched": matched, "macro_f1_end_to_end": float(np.mean(f1s)),
            "per_class_f1_end_to_end": dict(zip(harness.NAMES, f1s)),
            "confusion_prediction_rows": cm.tolist(),
        },
    }


def attach_classes(grouped: list[tuple[dict, list[dict]]],
                   logits_by_group: dict[int, np.ndarray]) -> None:
    index = 0
    for _rec, groups in grouped:
        for group in groups:
            group["cls"] = int(np.argmax(logits_by_group[index]))
            index += 1


def run(dataset: str, seed: int) -> dict:
    del seed
    started = time.time()
    profile = harness.PROFILES[dataset]
    records, payload, targets, _prior = harness.build_payload(dataset, "val")
    data, raw_grouped = raw_group_data(dataset, payload, targets, profile)
    all_dets = []
    for _rec, dets, _edges in payload:
        all_dets.extend(dets)
    if len(all_dets) != len(data["detection_features"]):
        raise RuntimeError("detection feature count mismatch")

    model_paths = {
        "member_logistic_max": OUT / f"{dataset}_member_logistic.joblib",
        "member_extra_max": OUT / f"{dataset}_member_extra.joblib",
    }
    raw_q = {}
    for name, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        model = joblib.load(path)
        q = np.asarray(model.predict_proba(data["detection_features"]), dtype=np.float32)
        raw_q[name] = q
    q_by_key = {
        name: {key: q[i] for i, key in enumerate(data["detection_keys"])}
        for name, q in raw_q.items()
    }

    # The class calibration is fixed from the strongest prior VAL result; no
    # new class grid is selected inside this truncation experiment.
    calibration_report = json.loads(
        (calibration.OUT / f"{dataset}_class_bias_general_results_val.json").read_text()
    )
    spec_name = "robust_953_anchor" if dataset == "953" else "member_stack_macro"
    spec = calibration_report["specs"][spec_name]
    chosen = spec["scale_grid"]["best_by_matched"]

    # Build per-group fused class logits over the raw candidate set using the
    # same train-fitted views as the class calibration experiment.
    # ``build_views`` needs the member topology, so construct the minimal
    # feature-view object around the raw candidate groups.
    grouped = data["groups"] = raw_grouped
    all_x = []
    group_rows = []
    keys = []
    fmap, dim = mh._load_fmap(dataset, "val")
    for _rec, groups in grouped:
        for group in groups:
            rows = []
            for member in group["members"]:
                rows.append(len(all_x))
                all_x.append(mh.member_feature(member, fmap, dim))
            group_rows.append(rows)
            keys.append(mh.harness_group_key(group))
    data["X"] = np.asarray(all_x, dtype=np.float32)
    data["group_rows"] = group_rows
    data["keys"] = keys
    views = calibration.build_views(dataset, data)
    detector = np.asarray([np.asarray(g["p"], dtype=np.float32)
                           for _rec, groups in grouped for g in groups])
    detector = np.maximum(detector, 1e-8)
    detector /= np.maximum(detector.sum(axis=1, keepdims=True), 1e-8)
    raw_logits = np.log(detector)
    for view_name, weight in spec["weights"].items():
        raw_logits += float(weight) * np.log(np.maximum(views[view_name], 1e-8))
    fused_logits = raw_logits * np.asarray(chosen["scales"], dtype=np.float32)
    fused_logits += np.asarray(chosen["bias"], dtype=np.float32)

    # Add head confidence to each raw group.  The same member opinions are
    # used both for a confidence ranking and for the fixed class assignment.
    flat_index = 0
    for _rec, groups in grouped:
        for group in groups:
            for name, q in raw_q.items():
                member_rows = []
                for member in group["members"]:
                    key = (str(member["stem"]), int(member["row_index"]))
                    # Resolve through the batched detection order.  A map is
                    # safer than an index because side views can share rows.
                    member_rows.append(q_by_key[name][key])
                if name.endswith("_max"):
                    head = np.max(member_rows, axis=0)
                else:
                    weights = np.asarray([float(m["score"]) for m in group["members"]])
                    weights /= max(float(weights.sum()), 1e-8)
                    head = np.sum(np.asarray(member_rows) * weights[:, None], axis=0)
                group[name] = np.asarray(head, dtype=np.float32)
            group["calibrated_p"] = softmax(
                fused_logits[flat_index:flat_index + 1]
            )[0]
            flat_index += 1

    baseline = short(harness.evaluate_clusters(payload, targets, profile))
    rows = []
    # Baseline class assignment is detector argmax; the calibrated class
    # assignment is the fixed known candidate.  Neither changes count target.
    configs = [("detector", "detector_p", "linker", 0.0),
               ("calibrated", "calibrated_p", "linker", 0.0)]
    powers = (0.10, 0.25, 0.50, 0.75, 1.0)
    for class_name, class_key, _unused, _p in configs:
        for rank_name, rank_key in (("member_logistic_max", "member_logistic_max"),
                                    ("member_extra_max", "member_extra_max")):
            for _rec, groups in grouped:
                for group in groups:
                    group["head_p"] = group[rank_key]
            for rank_source, power_list in (("linker", (0.0,)), ("head", powers)):
                for power in power_list:
                    candidate = []
                    for rec, groups in grouped:
                        target = int(targets[rec["tree_id"]])
                        selected = select_groups(groups, target, rank_source, power)
                        for group in selected:
                            p = group["p"] if class_key == "detector_p" else group["calibrated_p"]
                            group["cls"] = int(np.argmax(p))
                        candidate.append((rec, selected))
                    met = short(evaluate_grouped(candidate))
                    rows.append({"class_head": class_name, "rank_head": rank_name,
                                 "rank_source": rank_source, "power": power,
                                 "metrics": met,
                                 "count_targets_invariant": True})

    # Keep only metrics produced by actual head ranking for the primary
    # selection; the linker rows are explicit controls/anchors.
    best_head_matched = max(
        [r for r in rows if r["rank_source"] == "head"],
        key=lambda r: (r["metrics"]["matched_class_accuracy"],
                       r["metrics"]["macro_f1"], r["metrics"]["physical_f1"])
    )
    best_head_macro = max(
        [r for r in rows if r["rank_source"] == "head"],
        key=lambda r: (r["metrics"]["macro_f1"],
                       r["metrics"]["matched_class_accuracy"],
                       r["metrics"]["physical_f1"])
    )
    best_allround = max(
        [r for r in rows if r["metrics"]["physical_f1"] >= baseline["physical_f1"]
         and r["metrics"]["mae"] <= baseline["mae"]
         and r["metrics"]["matched_class_accuracy"] >= baseline["matched_class_accuracy"]],
        key=lambda r: (r["metrics"]["macro_f1"], r["metrics"]["matched_class_accuracy"]),
        default=None,
    )
    report = {
        "dataset": dataset,
        "protocol": "fixed TRAIN-fitted heads; head-aware truncation selected VAL; no TEST",
        "baseline_val": baseline,
        "calibration_spec": {"name": spec_name, "weights": spec["weights"],
                              "scales": chosen["scales"], "bias": chosen["bias"]},
        "raw_candidates": int(sum(len(gs) for _r, gs in grouped)),
        "missing_member_features": int(data["missing_features"]),
        "rows": rows,
        "best_head_by_matched": best_head_matched,
        "best_head_by_macro": best_head_macro,
        "best_allrounder_guardrail": best_allround,
        "elapsed_sec": time.time() - started,
    }
    path = OUT / f"{dataset}_head_aware_selection_results_val.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"dataset": dataset, "baseline": baseline,
                      "best_head_by_matched": best_head_matched,
                      "best_head_by_macro": best_head_macro,
                      "best_allrounder_guardrail": best_allround,
                      "report": str(path)}, ensure_ascii=False), flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("953", "depth"), required=True)
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()
    run(args.dataset, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
