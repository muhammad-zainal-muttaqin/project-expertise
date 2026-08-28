"""Reference harness for the cluster-level class head experiment.

This module builds the EXACT locked-linker payloads used by the E2E
pipeline for 953 (learned Hungarian edges) and depth (GSP MILP), and
provides an evaluator that reproduces ``evaluate_remote_class_head
.evaluate_payload`` bit-for-bit while letting an external ``class_fn``
decide each cluster's class instead of the built-in detector/head blend.

VALIDATION ONLY.  This module never loads a test split.  Everything it
reads comes from read-only trees (/workspace/project-expertise,
/workspace/model_artifacts, /workspace/gsp_linker); everything it writes
goes under /workspace/cluster_head.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, "/workspace/project-expertise/scripts")
import eval_remote_pipeline_postprocess as base  # noqa: E402
import evaluate_remote_class_head as head_eval  # noqa: E402
import evaluate_remote_count_reconciled as count  # noqa: E402
import sweep_remote_pipeline as sweep  # noqa: E402
import train_detection_edge_linker as edge  # noqa: E402

sys.path.insert(0, "/workspace/gsp_linker")
import link_global_setpartition as gsp  # noqa: E402


K = len(base.NAMES)
NAMES = base.NAMES

# Locked linker profiles that define the physical clusters (Step 0 of the
# task).  These are NOT tunable here: they reproduce the already-selected
# E2E profiles for 953 (Hungarian edges from a learned pair model) and depth
# (global set-partition / MILP over the same learned pair model).  The
# "expect" block is the anchor this harness must reproduce on VAL before any
# cluster-head work is trusted.
PROFILES = {
    "953": {
        "linker": "hungarian_learned",
        "model_name": "extra",
        "link_threshold": 0.15,
        "singleton_min": 0.15,
        "max_size": 4,
        "rank_mode": "score",
        "expect": {"f1": 0.8232, "mae": 1.2527, "pm1": 0.6703, "matched": 0.7542,
                   "matched_count": 773, "macro_f1": 0.6014},
    },
    "depth": {
        "linker": "gsp",
        "model_name": "extra",
        "tau_prob": 0.10,
        "link_threshold": 0.5,      # GSP hands over fully-decided (score=1.0) edges
        "singleton_min": 0.20,
        "max_size": 3,
        "rank_mode": "support",
        "expect": {"f1": 0.8526, "mae": 0.9316, "pm1": 0.7863, "matched": 0.8457,
                   "matched_count": 460, "macro_f1": 0.6807},
    },
}

ANCHOR_TOLERANCE = 0.003

_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _load_edge_model(dataset: str, name: str = "extra"):
    key = (dataset, name)
    if key not in _MODEL_CACHE:
        model = joblib.load(gsp.MODEL_PATHS[key])
        # ``build_edges`` predicts one small side-pair matrix at a time.
        # Keeping the training-time ExtraTrees n_jobs=-1 here creates a
        # process-pool/warning storm for tiny batches and is slower than a
        # single in-process prediction.  This does not alter predictions.
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def _train_class_prior(train_records: dict[str, dict]) -> np.ndarray:
    prior = np.bincount(
        [b["cls"] for rec in train_records.values() for b in rec["bunches"]
         if 0 <= b["cls"] < K], minlength=K).astype(float)
    return prior / max(float(prior.sum()), 1.)


def build_payload(dataset: str, split: str):
    """Build (records, payload, targets, class_prior) for one dataset/split.

    ``payload`` is a list of ``(rec, dets, edges)`` tuples exactly as
    consumed by ``count.selected_clusters`` / ``count.tree_matches``, built
    with the dataset's locked linker profile:

    * 953  -> learned pair-probability model + per-side-pair Hungarian
              assignment (``train_detection_edge_linker.build_edges``).
    * depth -> the same learned pair model feeding a global set-partition
               MILP (``link_global_setpartition``), whose chosen clusters
               are exposed as fully-decided (score=1.0) edges.

    No test split is ever accepted here.
    """
    if dataset not in PROFILES:
        raise ValueError(f"unknown dataset: {dataset!r}")
    if split not in ("train", "val"):
        raise ValueError("this task is VALIDATION ONLY; split must be train or val, "
                          f"got {split!r}")
    profile = PROFILES[dataset]
    proposal_min = gsp.PROPOSAL_MIN
    pair_mode = gsp.PAIR_MODE

    cfg = edge.cfg_for(dataset)
    train_records_full = base.load_records(cfg, "train")
    train_records = count.four_side(train_records_full)
    split_records = count.four_side(base.load_records(cfg, split))
    # Rotation prior is built from the FULL (not four-side-filtered) train
    # records, matching every other evaluation script in this codebase.
    prior = base.build_rotation_prior(train_records_full)

    train_vote = edge.load_vote(edge.vote_path(gsp.FUSED_ROOT, dataset, "train"))
    split_vote = edge.load_vote(edge.vote_path(gsp.FUSED_ROOT, dataset, split))

    targets, count_info = edge.target_counts(
        cfg, train_records, split_records, train_vote, split_vote, proposal_min)
    class_prior = _train_class_prior(train_records)
    model = _load_edge_model(dataset, profile["model_name"])

    dets_per_tree = {tid: edge.make_detections(rec, split_vote, proposal_min)
                      for tid, rec in split_records.items()}

    payload = []
    if dataset == "953":
        for tid, rec in split_records.items():
            dets = dets_per_tree[tid]
            edges = edge.build_edges(dets, rec["n_sides"], prior, model, pair_mode)
            payload.append((rec, dets, edges))
    else:  # depth: GSP MILP
        tau = math.log(profile["tau_prob"] / (1. - profile["tau_prob"]))
        for tid, rec in split_records.items():
            dets = dets_per_tree[tid]
            probs = gsp.tree_pair_probs(dets, rec["n_sides"], prior, model, pair_mode)
            candidates, _floor = gsp.enumerate_candidates(
                dets, probs, gsp.P_FLOOR, gsp.ENUM_MAX_SIZE)
            item, _tag = gsp.gsp_payload_for_tree(rec, dets, candidates, tau,
                                                   profile["max_size"])
            payload.append(item)

    return split_records, payload, targets, class_prior


def make_groups(payload, targets: dict[str, int], profile: dict):
    """Cluster selection only (class-decision-independent).

    Returns a list of ``(rec, groups)``; ``groups`` is the exact
    ``count.selected_clusters`` output for the locked profile.  This is
    factored out so that a class-decision sweep (Step 4) never has to
    recompute clustering: physical detection / counting are invariant to
    the class head by construction.
    """
    out = []
    for rec, dets, edges in payload:
        target = int(targets[rec["tree_id"]])
        groups = count.selected_clusters(
            dets, edges, profile["link_threshold"], profile["singleton_min"],
            profile["max_size"], target, profile["rank_mode"])
        out.append((rec, groups))
    return out


def default_class_fn(group: dict) -> int:
    """Argmax of the cluster's weighted-mean detector probability."""
    return int(np.argmax(group["p"]))


def evaluate_clusters(payload, targets: dict[str, int], profile: dict,
                       class_fn=default_class_fn) -> dict:
    """Reproduces ``evaluate_remote_class_head.evaluate_payload`` exactly,
    for the fixed (head_weight=0, class_prior_exponent=0, no margin gate,
    aggregation="mean", count_blend=0) configuration that defines our
    locked profiles, but assigns ``group["cls"]`` via ``class_fn`` instead
    of the built-in detector/head blend.
    """
    cm = np.zeros((K + 1, K + 1), int)
    total_pred = total_gt = total_tp = 0
    abs_count = exact = pm1 = vector_exact = 0
    class_correct = matched = 0
    for rec, groups in make_groups(payload, targets, profile):
        for group in groups:
            group["cls"] = int(class_fn(group))
        matches = count.tree_matches(rec, groups)
        bunches = rec["bunches"]
        total_pred += len(groups)
        total_gt += len(bunches)
        total_tp += len(matches)
        delta = len(groups) - len(bunches)
        abs_count += abs(delta)
        exact += int(delta == 0)
        pm1 += int(abs(delta) <= 1)
        pred_count = np.bincount([x["cls"] for x in groups], minlength=K)
        gt_count = np.bincount([x["cls"] for x in bunches if x["cls"] >= 0], minlength=K)
        vector_exact += int(np.array_equal(pred_count, gt_count))
        matched += len(matches)
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
            "precision": p, "recall": r, "f1": f1, "tp": total_tp,
            "pred_clusters": total_pred, "gt_bunches": total_gt,
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
            "per_class_f1_end_to_end": dict(zip(NAMES, f1s)),
            "confusion_prediction_rows": cm.tolist(),
        },
    }


def _head_eval_cross_check(payload, targets, class_prior, profile) -> dict:
    return head_eval.evaluate_payload(
        payload, targets, profile["link_threshold"], profile["singleton_min"],
        profile["max_size"], profile["rank_mode"], 0.0, class_prior, 0.0,
        None, "mean", 0.0)


def run_anchor_check(dataset: str, split: str = "val",
                      tol: float = ANCHOR_TOLERANCE) -> dict:
    profile = PROFILES[dataset]
    records, payload, targets, class_prior = build_payload(dataset, split)
    metrics = evaluate_clusters(payload, targets, profile)
    cross = _head_eval_cross_check(payload, targets, class_prior, profile)

    actual = {
        "f1": metrics["physical_detection"]["f1"],
        "mae": metrics["counting"]["mae"],
        "pm1": metrics["counting"]["plus_minus_1_accuracy"],
        "matched": metrics["classification"]["matched_class_accuracy"],
    }
    checks = {}
    for key in ("f1", "mae", "pm1", "matched"):
        exp = profile["expect"][key]
        diff = abs(actual[key] - exp)
        checks[key] = {"expected": exp, "actual": actual[key], "diff": diff,
                       "ok": diff <= tol}
    checks["matched_count"] = {
        "expected": profile["expect"]["matched_count"],
        "actual": metrics["classification"]["matched"],
        "ok": metrics["classification"]["matched"] == profile["expect"]["matched_count"],
    }
    macro_actual = metrics["classification"]["macro_f1_end_to_end"]
    macro_expected = profile["expect"]["macro_f1"]
    checks["macro_f1"] = {
        "expected": macro_expected, "actual": macro_actual,
        "diff": abs(macro_actual - macro_expected),
        "ok": abs(macro_actual - macro_expected) <= tol,
    }

    agree = {
        "f1": bool(np.isclose(metrics["physical_detection"]["f1"],
                               cross["physical_detection"]["f1"])),
        "mae": bool(np.isclose(metrics["counting"]["mae"], cross["counting"]["mae"])),
        "matched": bool(np.isclose(metrics["classification"]["matched_class_accuracy"],
                                    cross["classification"]["matched_class_accuracy"])),
        "macro_f1": bool(np.isclose(metrics["classification"]["macro_f1_end_to_end"],
                                     cross["classification"]["macro_f1_end_to_end"])),
    }
    passed = all(v["ok"] for v in checks.values()) and all(agree.values())
    return {
        "dataset": dataset, "split": split, "passed": bool(passed),
        "checks": checks, "cross_check_agrees": agree,
        "metrics": metrics, "n_trees": len(records),
    }


def main() -> int:
    all_ok = True
    report = {}
    for dataset in ("953", "depth"):
        result = run_anchor_check(dataset, "val")
        report[dataset] = result
        all_ok = all_ok and result["passed"]
        print(json.dumps({
            "dataset": dataset, "passed": result["passed"],
            "checks": result["checks"], "cross_check_agrees": result["cross_check_agrees"],
            "n_trees": result["n_trees"],
        }, indent=2, ensure_ascii=False))
    out = Path("/workspace/cluster_head/artifacts/anchor_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                               default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
                    + "\n", encoding="utf-8")
    print(f"-> {out}")
    if not all_ok:
        print("ANCHOR CHECK FAILED", file=sys.stderr)
        return 1
    print("ANCHOR CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
