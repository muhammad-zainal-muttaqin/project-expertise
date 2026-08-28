#!/usr/bin/env python3
"""VAL-only cross-layer composition search for the 953 corpus.

This closes a gap left by the earlier Depth-only composition audit.  The
experiment crosses three already-trained, independently declared topology
families (original Hungarian, original-vote GSP, and the V2 proposal/GSP
branches) with the already-fit count targets.  Class decisions are then
applied as an explicit skip/repair layer using the predeclared 953
calibration profiles.

No component is fitted on VAL.  The only fitting happened in the existing
TRAIN artifacts (edge linkers, count regressors, and class heads).  This
module accepts only ``train`` indirectly through those artifacts and the
``val`` split; it has no TEST code path.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, "/workspace/cluster_head")
sys.path.insert(0, "/workspace/pipeline_v2")
sys.path.insert(0, "/workspace/gsp_linker")

import class_bias_general as calibration  # noqa: E402
import harness  # noqa: E402
import head_aware_selection as evaluator  # noqa: E402
import member_head as mh  # noqa: E402
import pipeline_v2 as v2  # noqa: E402
import link_global_setpartition as gsp  # noqa: E402


OUT = Path("/workspace/cluster_head/artifacts")
V2_ROOT = Path("/workspace/pipeline_v2/artifacts/953")
DATASET = "953"
SEED = 20260828

# The axes below are predeclared before inspecting this run's output.  They
# are the union of the established GSP/V2 frontiers, not a test-derived grid.
TAUS = (.05, .10, .15, .20, .25, .35)
SINGLETONS = (.15, .20, .25)
RANKS = ("score", "support", "max_member")
MAX_SIZE = 3
CLASS_MODES = ("detector", "scale_matched", "scale_macro")


def short(metrics: dict) -> dict:
    return {
        "physical_f1": float(metrics["physical_detection"]["f1"]),
        "mae": float(metrics["counting"]["mae"]),
        "pm1": float(metrics["counting"]["plus_minus_1_accuracy"]),
        "matched_class_accuracy": float(
            metrics["classification"]["matched_class_accuracy"]),
        "matched": int(metrics["classification"]["matched"]),
        "macro_f1": float(metrics["classification"]["macro_f1_end_to_end"]),
        "per_class_f1": metrics["classification"]["per_class_f1_end_to_end"],
    }


def clone_groups(payload: list[tuple], targets: dict[str, int], profile: dict):
    """Materialize frozen groups; class labels are attached later."""
    return [(rec, [copy.deepcopy(g) for g in groups])
            for rec, groups in harness.make_groups(payload, targets, profile)]


def view_data(grouped: list[tuple]) -> dict:
    """Build the member-feature view in the exact group order."""
    fmap, dim = mh._load_fmap(DATASET, "val")
    features, rows, keys = [], [], []
    for _rec, groups in grouped:
        for group in groups:
            current = []
            for member in group["members"]:
                current.append(len(features))
                features.append(mh.member_feature(member, fmap, dim))
            rows.append(current)
            keys.append(mh.harness_group_key(group))
    return {"groups": grouped, "X": np.asarray(features, dtype=np.float32),
            "group_rows": rows, "keys": keys}


def attach_classes(grouped: list[tuple], mode: str) -> None:
    """Attach detector or fixed, TRAIN-fitted calibrated classes."""
    if mode == "detector":
        for _rec, groups in grouped:
            for group in groups:
                group["cls"] = int(np.argmax(np.asarray(group["p"])))
        return

    report = json.loads(
        (calibration.OUT / "953_class_bias_general_results_val.json").read_text())
    spec = report["specs"]["robust_953_anchor"]
    chosen_name = "best_by_matched" if mode == "scale_matched" else "best_by_macro"
    chosen = spec["scale_grid"][chosen_name]
    data = view_data(grouped)
    detector = np.asarray(
        [np.asarray(g["p"], dtype=np.float32)
         for _rec, groups in grouped for g in groups], dtype=np.float32)
    detector = np.maximum(detector, 1e-8)
    detector /= np.maximum(detector.sum(axis=1, keepdims=True), 1e-8)
    logits = np.log(detector)
    views = calibration.build_views(DATASET, data)
    for view_name, weight in spec["weights"].items():
        logits += float(weight) * np.log(np.maximum(views[view_name], 1e-8))
    logits = logits * np.asarray(chosen["scales"], dtype=np.float32)
    logits += np.asarray(chosen["bias"], dtype=np.float32)
    offset = 0
    for _rec, groups in grouped:
        for group in groups:
            group["cls"] = int(np.argmax(logits[offset]))
            offset += 1


def evaluate_candidate(payload: list[tuple], targets: dict[str, int],
                       profile: dict, class_mode: str | None = None) -> dict:
    grouped = clone_groups(payload, targets, profile)
    if class_mode is not None:
        attach_classes(grouped, class_mode)
        return evaluator.short(evaluator.evaluate_grouped(grouped)), grouped
    # Detector-only evaluation through the same grouped evaluator keeps the
    # physical/count implementation identical to class-head rows.
    attach_classes(grouped, "detector")
    return evaluator.short(evaluator.evaluate_grouped(grouped)), grouped


def load_count_targets(base_records: dict, v2_records: dict,
                       prior: dict) -> tuple[dict[str, dict[str, int]], dict, dict]:
    """Load all saved V2 count predictions, without refitting anything."""
    if list(base_records) != list(v2_records):
        raise RuntimeError("VAL tree order differs between base and V2 records")
    targets = {"original": harness.build_payload(DATASET, "val")[2]}
    v2_payloads = {}
    diagnostics = {}
    for mode in ("ptp", "geo"):
        vote = v2.edge.load_vote(V2_ROOT / f"vote_v2_{mode}_val.npz")
        edge_model = joblib.load(V2_ROOT / f"edge_v2_{mode}.joblib")
        per_tree = v2.build_dets_and_candidates(
            v2_records, vote, prior, edge_model)
        x_val, _y, ids = v2.build_count_features(v2_records, vote, per_tree)
        ridge = joblib.load(V2_ROOT / f"count_ridge_{mode}.joblib")
        hgb = joblib.load(V2_ROOT / f"count_hgb_{mode}.joblib")
        models = {"ridge": ridge, "hgb": hgb}
        for kind in ("ridge", "hgb"):
            pred = v2.predict_counts(kind, models, x_val)
            targets[f"v2_{mode}_{kind}"] = {
                tree_id: int(n) for tree_id, n in zip(ids, pred)}
        diagnostics[mode] = {
            "vote": str(V2_ROOT / f"vote_v2_{mode}_val.npz"),
            "edge_model": str(V2_ROOT / f"edge_v2_{mode}.joblib"),
            "count_features": int(x_val.shape[1]),
            "target_means": {
                kind: float(np.mean(list(targets[f"v2_{mode}_{kind}"].values())))
                for kind in ("ridge", "hgb")},
        }
        # Store the precomputed per-tree candidates for the topology builder.
        v2_payloads[mode] = {"vote": vote, "per_tree": per_tree}
    return targets, v2_payloads, diagnostics


def build_original_gsp_payloads(records: dict, prior: dict) -> dict[tuple, tuple]:
    """Build the original-vote GSP frontier once per declared configuration."""
    ctx = gsp.load_context(DATASET, gsp.FUSED_ROOT, "val", gsp.PROPOSAL_MIN)
    model = joblib.load(gsp.MODEL_PATHS[(DATASET, "extra")])
    out = {}
    for tau_prob in TAUS:
        tau = math.log(tau_prob / (1. - tau_prob))
        payload = []
        for tree_id, rec in ctx["split_records"].items():
            dets = ctx["dets_per_tree"][tree_id]
            probs = gsp.tree_pair_probs(
                dets, rec["n_sides"], ctx["prior"], model, gsp.PAIR_MODE)
            candidates, _floor = gsp.enumerate_candidates(
                dets, probs, gsp.P_FLOOR, gsp.ENUM_MAX_SIZE)
            chosen, _tag = gsp.solve_partition(
                len(dets), candidates, tau, MAX_SIZE)
            payload.append((rec, dets, gsp.decided_edges(chosen)))
        for singleton in SINGLETONS:
            for rank in RANKS:
                # Groups are selected downstream, so singleton/rank belong in
                # the profile rather than in the GSP solver itself.
                profile = {"link_threshold": .5, "singleton_min": singleton,
                           "max_size": MAX_SIZE, "rank_mode": rank}
                out[("original_gsp", tau_prob, singleton, rank)] = (payload, profile)
    if list(ctx["split_records"]) != list(records):
        raise RuntimeError("original GSP VAL tree order mismatch")
    return out


def build_v2_payloads(v2_records: dict, v2_payloads: dict) -> dict[tuple, tuple]:
    out = {}
    for mode, info in v2_payloads.items():
        for tau_prob in TAUS:
            payload, tags = v2.payload_for_tau(
                v2_records, info["per_tree"], tau_prob, MAX_SIZE)
            for singleton in SINGLETONS:
                for rank in RANKS:
                    profile = {"link_threshold": .5, "singleton_min": singleton,
                               "max_size": MAX_SIZE, "rank_mode": rank}
                    out[(f"v2_{mode}", tau_prob, singleton, rank)] = (
                        payload, profile, tags)
    return out


def dominates_baseline(m: dict, baseline: dict) -> bool:
    return (m["physical_f1"] >= baseline["physical_f1"]
            and m["mae"] <= baseline["mae"]
            and m["pm1"] >= baseline["pm1"]
            and m["matched_class_accuracy"] >= baseline["matched_class_accuracy"]
            and m["macro_f1"] >= baseline["macro_f1"])


def metric_key(m: dict) -> tuple:
    return (m["matched_class_accuracy"], m["macro_f1"],
            m["physical_f1"], -m["mae"], m["pm1"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    del args.seed  # fixed axes and deterministic saved models
    started = time.time()

    records, original_payload, _original_targets, _class_prior = harness.build_payload(
        DATASET, "val")
    cfg = v2.edge.cfg_for(DATASET)
    prior = v2.base.build_rotation_prior(v2.base.load_records(cfg, "train"))
    base_profile = dict(harness.PROFILES[DATASET])
    targets, v2_payloads, target_diag = load_count_targets(records, records, prior)

    # Original Hungarian with each count target is the smallest missing
    # composition.  The class calibration is applied in the second pass.
    candidates: list[dict] = []
    for source, target in targets.items():
        m, _groups = evaluate_candidate(
            original_payload, target, base_profile, "detector")
        candidates.append({"family": "original_hungarian", "config": {},
                           "target_source": source, "class_mode": "detector",
                           "metrics": m})

    # Original-vote GSP and V2 GSP topologies cross all saved count targets.
    original_gsp = build_original_gsp_payloads(records, prior)
    for key, (payload, profile) in original_gsp.items():
        family, tau, singleton, rank = key
        for source, target in targets.items():
            m, _groups = evaluate_candidate(payload, target, profile, "detector")
            candidates.append({
                "family": family,
                "config": {"tau_prob": tau, "singleton_min": singleton,
                           "rank_mode": rank},
                "target_source": source, "class_mode": "detector", "metrics": m,
            })
    v2_topologies = build_v2_payloads(records, v2_payloads)
    for key, value in v2_topologies.items():
        family, tau, singleton, rank = key
        payload, profile, tags = value
        for source, target in targets.items():
            m, _groups = evaluate_candidate(payload, target, profile, "detector")
            candidates.append({
                "family": family,
                "config": {"tau_prob": tau, "singleton_min": singleton,
                           "rank_mode": rank, "solver_tags": tags},
                "target_source": source, "class_mode": "detector", "metrics": m,
            })

    baseline = next(row["metrics"] for row in candidates
                    if row["family"] == "original_hungarian"
                    and row["target_source"] == "original")

    # Frontier reduction is deterministic and declared: class heads are only
    # spent on the best physical, count, class, macro, and no-regression rows.
    selected = []
    seen = set()
    ranked_sets = [
        sorted(candidates, key=lambda r: metric_key(r["metrics"]), reverse=True)[:30],
        sorted(candidates, key=lambda r: (r["metrics"]["physical_f1"],
                                          -r["metrics"]["mae"],
                                          r["metrics"]["matched_class_accuracy"]),
               reverse=True)[:30],
        sorted(candidates, key=lambda r: (-r["metrics"]["mae"],
                                          r["metrics"]["physical_f1"],
                                          r["metrics"]["matched_class_accuracy"]),
               reverse=True)[:30],
        [r for r in candidates if dominates_baseline(r["metrics"], baseline)],
    ]
    for group in ranked_sets:
        for row in group:
            key = (row["family"], json.dumps(row["config"], sort_keys=True),
                   row["target_source"])
            if key not in seen:
                seen.add(key)
                selected.append(row)

    # Rebuild only the selected topology/target rows and apply the fixed
    # calibration profiles.  Detector rows remain in the report as controls.
    lookup = {}
    lookup[("original_hungarian", "original")] = (original_payload, base_profile)
    for source in targets:
        lookup[("original_hungarian", source)] = (original_payload, base_profile)
    for key, value in original_gsp.items():
        family, tau, singleton, rank = key
        payload, profile = value
        config_key = json.dumps({"tau_prob": tau, "singleton_min": singleton,
                                 "rank_mode": rank}, sort_keys=True)
        for source in targets:
            lookup[(family + "|" + config_key, source)] = (payload, profile)
    for key, value in v2_topologies.items():
        family, tau, singleton, rank = key
        payload, profile, tags = value
        config_key = json.dumps({"tau_prob": tau, "singleton_min": singleton,
                                 "rank_mode": rank, "solver_tags": tags},
                                sort_keys=True)
        for source in targets:
            lookup[(family + "|" + config_key, source)] = (payload, profile)

    expanded = []
    for row in selected:
        if row["family"] == "original_hungarian":
            payload, profile = original_payload, base_profile
        else:
            cfg = row["config"]
            # solver_tags are metadata, not a lookup axis.
            cfg_no_tags = {k: v for k, v in cfg.items() if k != "solver_tags"}
            if row["family"] == "original_gsp":
                payload, profile = original_gsp[(
                    row["family"], cfg_no_tags["tau_prob"],
                    cfg_no_tags["singleton_min"], cfg_no_tags["rank_mode"])]
            else:
                payload, profile, _tags = v2_topologies[(
                    row["family"], cfg_no_tags["tau_prob"],
                    cfg_no_tags["singleton_min"], cfg_no_tags["rank_mode"])]
        target = targets[row["target_source"]]
        for class_mode in CLASS_MODES[1:]:
            m, _groups = evaluate_candidate(payload, target, profile, class_mode)
            expanded.append({**row, "class_mode": class_mode, "metrics": m})
    all_rows = candidates + expanded

    # Compact candidates for the report while retaining every detector grid
    # row and all class variants on the selected frontier.
    def best(key_fn):
        return max(all_rows, key=lambda r: key_fn(r["metrics"]))

    best_match = best(lambda m: (m["matched_class_accuracy"], m["macro_f1"],
                                 m["physical_f1"], -m["mae"]))
    best_macro = best(lambda m: (m["macro_f1"], m["matched_class_accuracy"],
                                m["physical_f1"], -m["mae"]))
    allround = [r for r in all_rows if dominates_baseline(r["metrics"], baseline)]
    best_allround = max(allround, key=lambda r: metric_key(r["metrics"]),
                        default=None)

    report = {
        "dataset": DATASET,
        "protocol": "saved TRAIN-fitted topology/count/head branches; VAL-only frontier composition; no TEST",
        "seed": SEED,
        "axes": {"taus": TAUS, "singletons": SINGLETONS, "ranks": RANKS,
                 "max_size": MAX_SIZE, "class_modes": CLASS_MODES},
        "target_sources": list(targets),
        "target_diagnostics": target_diag,
        "baseline_val": baseline,
        "frontier_selection": {
            "detector_rows": len(candidates), "frontier_rows": len(selected),
            "expanded_class_rows": len(expanded),
            "selection": "top-30 class, top-30 physical/count, top-30 MAE, plus baseline-dominating detector rows",
        },
        "best_by_matched": best_match,
        "best_by_macro": best_macro,
        "best_allrounder_guardrail": best_allround,
        "rows": all_rows,
        "elapsed_sec": time.time() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "953_cross_layer_composition_results_val.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"baseline": baseline,
                      "detector_rows": len(candidates),
                      "frontier_rows": len(selected),
                      "best_by_matched": best_match,
                      "best_by_macro": best_macro,
                      "best_allrounder_guardrail": best_allround,
                      "report": str(path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
