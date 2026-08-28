#!/usr/bin/env python3
"""Train a class head on the exact cross-layer composition (Depth VAL only).

The previous composition used a class head trained on the original count
targets.  This follow-up aligns the TRAIN labels with the selected
original-GSP + V2-count composition, then evaluates fresh member heads on the
same composition in VAL.  It is a targeted residual/skip experiment; the
detector, GSP topology, and inference count path are not changed, and TEST is
never read.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, "/workspace/cluster_head")
sys.path.insert(0, "/workspace/pipeline_v2")

import harness  # noqa: E402
import head_aware_selection as evaluator  # noqa: E402
import member_head as mh  # noqa: E402
import pipeline_v2 as v2  # noqa: E402
import topology_count_class_combo as combo  # noqa: E402


OUT = Path("/workspace/cluster_head/artifacts")
DEPTH_V2 = combo.DEPTH_V2


def v2_count_targets(split: str, seed: int = 20260828) -> dict[str, int]:
    del seed
    dataset = "depth"
    cfg = v2.edge.cfg_for(dataset)
    records = v2.count.four_side(v2.base.load_records(cfg, split))
    prior = v2.base.build_rotation_prior(v2.base.load_records(cfg, "train"))
    vote = v2.edge.load_vote(DEPTH_V2 / f"vote_v2_geo_{split}.npz")
    edge_model = joblib.load(DEPTH_V2 / "edge_v2_geo.joblib")
    per_tree = v2.build_dets_and_candidates(records, vote, prior, edge_model)
    x, _y, ids = v2.build_count_features(records, vote, per_tree)
    ridge = joblib.load(DEPTH_V2 / "count_ridge_geo.joblib")
    pred = v2.count.predict_count(ridge, x)
    return {tree_id: int(n) for tree_id, n in zip(ids, pred)}


def composition_data(split: str, targets: dict[str, int]):
    records, payload, _old_targets, _ = harness.build_payload("depth", split)
    profile = harness.PROFILES["depth"]
    grouped = [(rec, [copy.deepcopy(g) for g in groups])
               for rec, groups in harness.make_groups(payload, targets, profile)]
    fmap, dim = mh._load_fmap("depth", split)
    x, y, rows, keys = [], [], [], []
    missing = 0
    for rec, groups in grouped:
        matches = dict(harness.count.tree_matches(rec, groups))
        for gi, group in enumerate(groups):
            gt_cls = int(rec["bunches"][matches[gi]]["cls"]) if gi in matches else -1
            current = []
            for member in group["members"]:
                key = (str(member["stem"]), int(member["row_index"]))
                if key not in fmap:
                    missing += 1
                current.append(len(x))
                x.append(mh.member_feature(member, fmap, dim))
                y.append(gt_cls)
            rows.append(current)
            keys.append(mh.harness_group_key(group))
    return {"records": records, "payload": payload, "groups": grouped,
            "X": np.asarray(x, dtype=np.float32), "y": np.asarray(y, dtype=np.int64),
            "group_rows": rows, "keys": keys, "missing": missing}


def pool(q: np.ndarray, data: dict, mode: str) -> np.ndarray:
    flat = [g for _rec, groups in data["groups"] for g in groups]
    out = []
    for i, rows in enumerate(data["group_rows"]):
        member_q = q[rows]
        if mode == "max":
            p = member_q.max(axis=0)
        elif mode == "mean":
            weights = np.asarray([float(m["score"]) for m in flat[i]["members"]])
            weights /= max(float(weights.sum()), 1e-8)
            p = (member_q * weights[:, None]).sum(axis=0)
        elif mode == "top":
            j = int(np.argmax([float(m["score"]) for m in flat[i]["members"]]))
            p = member_q[j]
        else:
            raise ValueError(mode)
        out.append(p)
    p = np.maximum(np.asarray(out, dtype=np.float32), 1e-8)
    return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-8)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-8)


def short(m: dict) -> dict:
    return evaluator.short(m)


def run(seed: int) -> dict:
    started = time.time()
    train_targets = v2_count_targets("train", seed)
    val_targets = v2_count_targets("val", seed)
    train = composition_data("train", train_targets)
    val = composition_data("val", val_targets)
    mask = train["y"] >= 0
    if mask.sum() < 20 or len(np.unique(train["y"][mask])) < harness.K:
        raise RuntimeError("insufficient matched composition TRAIN labels")

    baseline_groups = combo.build_selected(
        val["payload"], val_targets, harness.PROFILES["depth"])
    combo.calibrated_classes("depth", baseline_groups, "scale_macro")
    composition_baseline = short(evaluator.evaluate_grouped(baseline_groups))
    detector = np.asarray([np.asarray(g["p"], dtype=np.float32)
                           for _rec, groups in val["groups"] for g in groups])
    detector = np.maximum(detector, 1e-8)
    detector /= np.maximum(detector.sum(axis=1, keepdims=True), 1e-8)

    rows = []
    for name, model in mh.models(seed):
        fit_start = time.time()
        model.fit(train["X"][mask], train["y"][mask])
        q = np.asarray(model.predict_proba(val["X"]), dtype=np.float32)
        for pooling in ("mean", "max", "top"):
            member_p = pool(q, val, pooling)
            for alpha in (0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.85, 1.0):
                for b2, b3, b4 in ((0., 0., 0.), (.10, 0., 0.),
                                   (.15, -.05, 0.), (.15, -.10, 0.),
                                   (.15, -.10, -.10)):
                    logits = np.log(detector) + alpha * np.log(member_p)
                    logits += np.asarray([0., b2, b3, b4], dtype=np.float32)
                    pred = np.argmax(logits, axis=1)
                    candidate = [(rec, [copy.deepcopy(g) for g in groups])
                                 for rec, groups in val["groups"]]
                    offset = 0
                    for _rec, groups in candidate:
                        for group in groups:
                            group["cls"] = int(pred[offset])
                            offset += 1
                    metrics = short(evaluator.evaluate_grouped(candidate))
                    rows.append({"model": name, "pooling": pooling,
                                 "alpha": alpha, "bias": [0., b2, b3, b4],
                                 "metrics": metrics,
                                 "fit_elapsed_sec": time.time() - fit_start})
        joblib.dump(model, OUT / f"depth_composition_{name}.joblib", compress=3)
        print(json.dumps({"model": name, "rows": len(rows),
                          "elapsed_sec": time.time() - fit_start}, ensure_ascii=False),
              flush=True)

    best_matched = max(rows, key=lambda r: (r["metrics"]["matched_class_accuracy"],
                                            r["metrics"]["macro_f1"]))
    best_macro = max(rows, key=lambda r: (r["metrics"]["macro_f1"],
                                          r["metrics"]["matched_class_accuracy"]))
    eligible = [r for r in rows if (
        r["metrics"]["physical_f1"] >= composition_baseline["physical_f1"]
        and r["metrics"]["mae"] <= composition_baseline["mae"]
        and r["metrics"]["pm1"] >= composition_baseline["pm1"])]
    report = {
        "dataset": "depth",
        "protocol": "fit composition-aware member heads on TRAIN; select VAL; no TEST",
        "composition": "original_gsp_topology + v2_geo_ridge_count",
        "train": {"groups": len(train["group_rows"]),
                  "matched_members": int(mask.sum()), "missing": train["missing"]},
        "val": {"groups": len(val["group_rows"]),
                "missing": val["missing"]},
        "baseline_composition_val": composition_baseline,
        "best_by_matched": best_matched,
        "best_by_macro": best_macro,
        "best_allrounder_guardrail": max(
            eligible, key=lambda r: (r["metrics"]["macro_f1"],
                                     r["metrics"]["matched_class_accuracy"]),
            default=None),
        "rows": rows,
        "elapsed_sec": time.time() - started,
    }
    path = OUT / "depth_composition_aware_head_results_val.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"baseline": composition_baseline,
                      "best_by_matched": best_matched,
                      "best_by_macro": best_macro,
                      "best_allrounder_guardrail": report["best_allrounder_guardrail"],
                      "report": str(path)}, ensure_ascii=False), flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()
    run(args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
