"""TRAIN/VAL adaptive selector between two fixed linker topologies.

The 953 validation frontier exposes a clear trade-off: Hungarian linking has
better count stability, while GSP can recover physical matches.  This module
trains a tree-level policy on TRAIN to choose between those two *fixed*
candidate payloads using only inference-time graph statistics.  It never uses
ground-truth information as a feature and refuses a TEST split.

The GSP operating point is declared from the earlier validation sweep; this
script does not perform a new test-driven search.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/workspace/cluster_head")
sys.path.insert(0, "/workspace/gsp_linker")
import harness  # noqa: E402
import link_global_setpartition as gsp  # noqa: E402


OUT = Path("/workspace/cluster_head/artifacts")
K = harness.K

# Fixed, previously observed VAL frontier for the 953 candidate.  The
# selector is the new component; these values are not fitted by this script.
GSP_CONFIG = {"tau_prob": .05, "singleton_min": .25,
              "max_size": 3, "rank_mode": "score"}


def _tree_groups(item: tuple, targets: dict, profile: dict):
    rec, dets, edges = item
    groups = harness.count.selected_clusters(
        dets, edges, profile["link_threshold"], profile["singleton_min"],
        profile["max_size"], int(targets[rec["tree_id"]]), profile["rank_mode"])
    return rec, dets, edges, groups


def _summarize_groups(groups: list[dict], edges: list[tuple[float, int, int]],
                      dets: list[dict], target: int) -> list[float]:
    scores = np.asarray([float(g["score"]) for g in groups], dtype=float)
    sizes = np.asarray([len(g["members"]) for g in groups], dtype=float)
    edge_scores = np.asarray([float(e[0]) for e in edges], dtype=float)
    det_scores = np.asarray([float(d["score"]) for d in dets], dtype=float)
    out = [float(target), float(len(dets)), float(len(edges)), float(len(groups)),
           float(len(groups) - target)]
    for values in (scores, sizes, edge_scores, det_scores):
        if len(values):
            out.extend([float(values.mean()), float(values.std()),
                        float(values.min()), float(np.median(values)),
                        float(values.max()), float((values >= .5).sum())])
        else:
            out.extend([0.] * 6)
    out.extend(float(sum(int(d["side"]) == s for d in dets)) for s in range(4))
    out.extend(float(sum(len(g["members"]) == s for g in groups))
                for s in (1, 2, 3, 4))
    if groups:
        p = np.asarray([g["p"] for g in groups], dtype=float)
        out.extend([float(p[:, c].mean()) for c in range(K)])
        out.extend([float(p.max(axis=1).mean()), float(p.max(axis=1).std())])
    else:
        out.extend([0.] * (K + 2))
    return out


def _build_gsp_payload(dataset: str, split: str):
    if split not in ("train", "val"):
        raise ValueError("this experiment accepts only train or val")
    ctx = gsp.load_context(dataset, gsp.FUSED_ROOT, split, gsp.PROPOSAL_MIN)
    model = joblib.load(gsp.MODEL_PATHS[(dataset, "extra")])
    tau = math.log(GSP_CONFIG["tau_prob"] / (1. - GSP_CONFIG["tau_prob"]))
    payload = []
    tags = defaultdict(int)
    for tree_id, rec in ctx["split_records"].items():
        dets = ctx["dets_per_tree"][tree_id]
        probs = gsp.tree_pair_probs(dets, rec["n_sides"], ctx["prior"],
                                    model, gsp.PAIR_MODE)
        candidates, _floor = gsp.enumerate_candidates(
            dets, probs, gsp.P_FLOOR, gsp.ENUM_MAX_SIZE)
        chosen, tag = gsp.solve_partition(
            len(dets), candidates, tau, GSP_CONFIG["max_size"])
        tags[tag] += 1
        payload.append((rec, dets, gsp.decided_edges(chosen)))
    return ctx["split_records"], payload, ctx["targets"], dict(tags)


def _per_tree_quality(item: tuple, targets: dict, profile: dict) -> dict:
    rec, dets, edges, groups = _tree_groups(item, targets, profile)
    matches = harness.count.tree_matches(rec, groups)
    correct = 0
    for i, j in matches:
        correct += int(groups[i]["cls"] == rec["bunches"][j]["cls"])
    return {"tree_id": str(rec["tree_id"]), "tp": int(len(matches)),
            "pred": int(len(groups)), "gt": int(len(rec["bunches"])),
            "abs_count": int(abs(len(groups) - len(rec["bunches"]))),
            "class_correct": int(correct)}


def _policy_features(base_item: tuple, gsp_item: tuple, targets: dict,
                     base_profile: dict, gsp_profile: dict) -> np.ndarray:
    br, bd, be, bg = _tree_groups(base_item, targets, base_profile)
    gr, gd, ge, gg = _tree_groups(gsp_item, targets, gsp_profile)
    a = _summarize_groups(bg, be, bd, int(targets[br["tree_id"]]))
    b = _summarize_groups(gg, ge, gd, int(targets[gr["tree_id"]]))
    # Difference features expose the topology compromise while retaining both
    # raw opinions as a skip path.
    return np.asarray([*a, *b, *np.asarray(b) - np.asarray(a)], dtype=np.float32)


def _evaluate_with_profiles(payload: list[tuple], targets: dict,
                            profiles: list[dict]) -> dict:
    """Evaluate one payload with a possibly different profile per tree."""
    if len(payload) != len(profiles):
        raise RuntimeError("payload/profile length mismatch")
    cm = np.zeros((K + 1, K + 1), dtype=int)
    total_pred = total_gt = total_tp = 0
    abs_count = exact = pm1 = vector_exact = 0
    class_correct = matched = 0
    for item, profile in zip(payload, profiles):
        rec, dets, edges, groups = _tree_groups(item, targets, profile)
        matches = harness.count.tree_matches(rec, groups)
        total_pred += len(groups)
        total_gt += len(rec["bunches"])
        total_tp += len(matches)
        delta = len(groups) - len(rec["bunches"])
        abs_count += abs(delta)
        exact += int(delta == 0)
        pm1 += int(abs(delta) <= 1)
        pred_count = np.bincount([g["cls"] for g in groups], minlength=K)
        gt_count = np.bincount([b["cls"] for b in rec["bunches"] if b["cls"] >= 0],
                               minlength=K)
        vector_exact += int(np.array_equal(pred_count, gt_count))
        matched += len(matches)
        matched_pred = {i for i, _ in matches}
        matched_gt = {j for _, j in matches}
        for i, j in matches:
            pc, gc = int(groups[i]["cls"]), int(rec["bunches"][j]["cls"])
            if 0 <= pc < K and 0 <= gc < K:
                cm[pc, gc] += 1
                class_correct += int(pc == gc)
        for i, group in enumerate(groups):
            if i not in matched_pred and 0 <= int(group["cls"]) < K:
                cm[int(group["cls"]), K] += 1
        for j, bunch in enumerate(rec["bunches"]):
            if j not in matched_gt and 0 <= int(bunch["cls"]) < K:
                cm[K, int(bunch["cls"])] += 1
    precision = total_tp / max(total_pred, 1)
    recall = total_tp / max(total_gt, 1)
    f1 = 2. * precision * recall / max(precision + recall, 1e-12)
    f1s = []
    for c in range(K):
        tp = int(cm[c, c])
        fp = int(cm[c, :].sum() - tp)
        fn = int(cm[:, c].sum() - tp)
        f1s.append(2. * tp / max(2 * tp + fp + fn, 1))
    return {"physical_detection": {"precision": precision, "recall": recall,
                                    "f1": f1, "tp": total_tp,
                                    "pred_clusters": total_pred,
                                    "gt_bunches": total_gt},
            "counting": {"mae": abs_count / max(len(payload), 1),
                         "exact_accuracy": exact / max(len(payload), 1),
                         "plus_minus_1_accuracy": pm1 / max(len(payload), 1),
                         "vector_exact_accuracy": vector_exact / max(len(payload), 1)},
            "classification": {
                "matched_class_accuracy": class_correct / max(matched, 1),
                "matched": matched, "macro_f1_end_to_end": float(np.mean(f1s)),
                "per_class_f1_end_to_end": dict(zip(harness.NAMES, f1s)),
                "confusion_prediction_rows": cm.tolist()}}


def _short(m: dict) -> dict:
    return {"physical_f1": m["physical_detection"]["f1"],
            "mae": m["counting"]["mae"],
            "pm1": m["counting"]["plus_minus_1_accuracy"],
            "matched_class_accuracy": m["classification"]["matched_class_accuracy"],
            "matched": m["classification"]["matched"],
            "macro_f1": m["classification"]["macro_f1_end_to_end"],
            "per_class_f1": m["classification"]["per_class_f1_end_to_end"]}


def _mixed(base_payload: list[tuple], gsp_payload: list[tuple], choose_gsp: np.ndarray):
    if len(base_payload) != len(gsp_payload) or len(base_payload) != len(choose_gsp):
        raise RuntimeError("payload/policy length mismatch")
    return [gsp_payload[i] if bool(choose_gsp[i]) else base_payload[i]
            for i in range(len(base_payload))]


def _model_bank(seed: int, jobs: int):
    yield "logistic", Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=.10, max_iter=600,
                                    class_weight="balanced", random_state=seed))])
    yield "extra", ExtraTreesClassifier(
        n_estimators=260, min_samples_leaf=8, max_features=.70,
        class_weight="balanced", n_jobs=jobs, random_state=seed)
    yield "hist", HistGradientBoostingClassifier(
        max_iter=180, learning_rate=.05, max_leaf_nodes=7,
        l2_regularization=3., random_state=seed)


def run(seed: int, jobs: int) -> dict:
    started = time.time()
    dataset = "953"
    profile = dict(harness.PROFILES[dataset])
    gsp_profile = dict(profile)
    gsp_profile.update({"link_threshold": .5,
                        "singleton_min": GSP_CONFIG["singleton_min"],
                        "max_size": GSP_CONFIG["max_size"],
                        "rank_mode": GSP_CONFIG["rank_mode"]})
    base_train = harness.build_payload(dataset, "train")
    base_val = harness.build_payload(dataset, "val")
    _train_records, gsp_train, targets_train, train_tags = _build_gsp_payload(dataset, "train")
    _val_records, gsp_val, targets_val, val_tags = _build_gsp_payload(dataset, "val")
    train_records, base_train_payload, base_targets, _prior = base_train
    val_records, base_val_payload, val_targets, _prior2 = base_val
    if list(train_records) != list(_train_records) or list(val_records) != list(_val_records):
        raise RuntimeError("Hungarian/GSP tree order mismatch")
    train_items = list(zip(base_train_payload, gsp_train))
    val_items = list(zip(base_val_payload, gsp_val))
    train_quality_base = [_per_tree_quality(a, base_targets, profile) for a, _ in train_items]
    train_quality_gsp = [_per_tree_quality(b, targets_train, gsp_profile) for _, b in train_items]
    # Positive utility favors recovered physical matches, but penalizes count
    # deviation so a policy cannot simply select the denser graph.
    utilities = np.asarray([
        (g["tp"] - b["tp"]) - .75 * (g["abs_count"] - b["abs_count"])
        + .25 * (g["class_correct"] - b["class_correct"])
        for b, g in zip(train_quality_base, train_quality_gsp)], dtype=np.float32)
    labels = (utilities > 0.).astype(np.int64)
    if len(np.unique(labels)) < 2:
        raise RuntimeError("adaptive labels have only one class")
    X_train = np.asarray([_policy_features(a, b, base_targets, profile, gsp_profile)
                          for a, b in train_items], dtype=np.float32)
    X_val = np.asarray([_policy_features(a, b, val_targets, profile, gsp_profile)
                        for a, b in val_items], dtype=np.float32)
    base_profiles_val = [profile] * len(base_val_payload)
    gsp_profiles_val = [gsp_profile] * len(gsp_val)
    baseline = _short(_evaluate_with_profiles(
        base_val_payload, val_targets, base_profiles_val))
    always_base = _short(_evaluate_with_profiles(
        base_val_payload, val_targets, base_profiles_val))
    always_gsp = _short(_evaluate_with_profiles(
        gsp_val, targets_val, gsp_profiles_val))
    rows = [
        {"policy": "always_hungarian", "metrics": always_base},
        {"policy": "always_gsp", "metrics": always_gsp},
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, model in _model_bank(seed, jobs):
        model.fit(X_train, labels)
        choose = model.predict(X_val).astype(bool)
        mixed_payload = _mixed(base_val_payload, gsp_val, choose)
        mixed_profiles = [gsp_profile if bool(flag) else profile for flag in choose]
        metrics = _short(_evaluate_with_profiles(mixed_payload, val_targets,
                                                 mixed_profiles))
        joblib.dump(model, OUT / f"953_adaptive_linker_{name}.joblib", compress=3)
        rows.append({"policy": name, "gsp_selected_val": int(choose.sum()),
                     "metrics": metrics})

    # Oracle is diagnostic only and is explicitly excluded from deployment.
    oracle = _mixed(base_train_payload, gsp_train,
                    np.asarray([u > 0. for u in utilities], dtype=bool))
    oracle_profiles = [gsp_profile if u > 0. else profile for u in utilities]
    oracle_train = _short(_evaluate_with_profiles(oracle, targets_train, oracle_profiles))
    best = max(rows, key=lambda r: (r["metrics"]["matched_class_accuracy"],
                                    r["metrics"]["physical_f1"],
                                    -r["metrics"]["mae"],
                                    r["metrics"]["macro_f1"]))
    report = {
        "dataset": dataset,
        "protocol": "fixed Hungarian/GSP candidates; policy fit TRAIN; evaluate VAL; no TEST",
        "seed": seed, "feature_dim": int(X_train.shape[1]),
        "train_trees": int(len(X_train)), "val_trees": int(len(X_val)),
        "gsp_config": GSP_CONFIG, "train_gsp_solver_tags": train_tags,
        "val_gsp_solver_tags": val_tags,
        "train_policy_positive": int(labels.sum()),
        "train_utility": {"mean": float(utilities.mean()),
                           "positive_fraction": float(labels.mean()),
                           "oracle_diagnostic": oracle_train},
        "baseline_val": baseline, "rows": rows, "selected_validation": best,
        "elapsed_sec": time.time() - started,
    }
    path = OUT / "953_adaptive_linker_results_val.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(json.dumps({"selected_validation": best, "oracle_train": oracle_train,
                      "report": str(path), "seconds": report["elapsed_sec"]},
                     ensure_ascii=False), flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    run(args.seed, args.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
