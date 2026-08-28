#!/usr/bin/env python3
"""Validation-only edge-model ensemble for the 953 GSP topology.

The original GSP run used one ExtraTrees edge model for 953.  This audit
combines that model with two independently saved, TRAIN-fitted histogram
edge models before candidate enumeration.  The ensemble is evaluated over a
small declared GSP frontier and the existing count targets.  It changes the
physical linker layer, so every metric is recomputed; no class or test output
is used to choose the model weights.

Only the original TRAIN/VAL proposal dumps are loaded.  There is deliberately
no TEST argument or TEST fallback.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, "/workspace/project-expertise/scripts")
sys.path.insert(0, "/workspace/cluster_head")
sys.path.insert(0, "/workspace/pipeline_v2")
sys.path.insert(0, "/workspace/gsp_linker")

import harness  # noqa: E402
import head_aware_selection as evaluator  # noqa: E402
import pipeline_v2 as v2  # noqa: E402
import train_detection_edge_linker as edge  # noqa: E402
import link_global_setpartition as gsp  # noqa: E402


OUT = Path("/workspace/cluster_head/artifacts")
DATASET = "953"
SEED = 20260828
TAUS = (.10, .15, .20, .25, .35, .50)
SINGLETONS = (.15, .20, .25)
RANKS = ("score", "support", "max_member")
MAX_SIZE = 3

MODEL_PATHS = {
    "extra_v2": gsp.MODEL_PATHS[(DATASET, "extra")],
    "hist_v1": Path(
        "/workspace/model_artifacts/project-expertise/"
        "detection_edge_linker_953_v1/hist.joblib"),
    "hist_deep_v1": Path(
        "/workspace/model_artifacts/project-expertise/"
        "detection_edge_linker_953_v1/hist_deep.joblib"),
    "hist_all_v2": Path(
        "/workspace/model_artifacts/project-expertise/"
        "detection_edge_linker_953_all_hist/hist.joblib"),
}

# The first three are independent mixtures; the fourth uses the duplicate
# histogram retrain as a stability control.  Equal/logit averaging is fixed
# before VAL selection and recorded in the report.
MIXES = {
    "extra_only": {"extra_v2": 1.0},
    "extra_histdeep_logit": {"extra_v2": .70, "hist_deep_v1": .30},
    "extra_hist_logit": {"extra_v2": .70, "hist_all_v2": .30},
    "threeway_logit": {"extra_v2": .60, "hist_v1": .20, "hist_deep_v1": .20},
}


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


def pair_probability_tables(ctx: dict, models: dict) -> dict[str, dict]:
    """Compute each model's pair probabilities once, all on VAL."""
    out = {name: {} for name in models}
    for tree_id, rec in ctx["split_records"].items():
        dets = ctx["dets_per_tree"][tree_id]
        for name, model in models.items():
            out[name][tree_id] = gsp.tree_pair_probs(
                dets, rec["n_sides"], ctx["prior"], model, gsp.PAIR_MODE)
    return out


def mix_probs(tables: dict[str, dict], tree_id: str, weights: dict) -> dict:
    names = list(weights)
    keys = set(tables[names[0]][tree_id])
    if any(set(tables[name][tree_id]) != keys for name in names[1:]):
        raise RuntimeError("edge probability key mismatch")
    out = {}
    for key in keys:
        # Logit averaging avoids a high-entropy histogram model washing out a
        # confident ExtraTrees edge, while remaining a deterministic opinion
        # pool with no validation labels in the computation.
        z = 0.0
        for name in names:
            p = float(np.clip(tables[name][tree_id][key], 1e-4, 1. - 1e-4))
            z += float(weights[name]) * math.log(p / (1. - p))
        out[key] = float(1. / (1. + math.exp(-np.clip(z, -40., 40.))))
    return out


def build_payloads(ctx: dict, tables: dict, weights: dict) -> dict[float, list[tuple]]:
    out = {}
    for tau_prob in TAUS:
        tau = math.log(tau_prob / (1. - tau_prob))
        payload = []
        for tree_id, rec in ctx["split_records"].items():
            dets = ctx["dets_per_tree"][tree_id]
            probs = mix_probs(tables, tree_id, weights)
            candidates, _floor = gsp.enumerate_candidates(
                dets, probs, gsp.P_FLOOR, gsp.ENUM_MAX_SIZE)
            chosen, _tag = gsp.solve_partition(
                len(dets), candidates, tau, MAX_SIZE)
            payload.append((rec, dets, gsp.decided_edges(chosen)))
        out[tau_prob] = payload
    return out


def evaluate(payload: list[tuple], targets: dict[str, int], singleton: float,
             rank: str) -> dict:
    profile = {"link_threshold": .5, "singleton_min": singleton,
               "max_size": MAX_SIZE, "rank_mode": rank}
    grouped = [(rec, [copy.deepcopy(g) for g in groups])
               for rec, groups in harness.make_groups(payload, targets, profile)]
    for _rec, groups in grouped:
        for group in groups:
            group["cls"] = int(np.argmax(group["p"]))
    return evaluator.short(evaluator.evaluate_grouped(grouped))


def v2_targets(ctx: dict) -> dict[str, dict[str, int]]:
    """Load saved V2 count targets as additional, fixed count opinions."""
    records = ctx["split_records"]
    out = {"original": dict(ctx["targets"])}
    cfg = v2.edge.cfg_for(DATASET)
    v2_records = v2.count.four_side(v2.base.load_records(cfg, "val"))
    if list(v2_records) != list(records):
        raise RuntimeError("V2/original VAL tree order mismatch")
    prior = v2.base.build_rotation_prior(v2.base.load_records(cfg, "train"))
    for mode in ("ptp", "geo"):
        root = Path("/workspace/pipeline_v2/artifacts") / DATASET
        vote = v2.edge.load_vote(root / f"vote_v2_{mode}_val.npz")
        model = joblib.load(root / f"edge_v2_{mode}.joblib")
        per_tree = v2.build_dets_and_candidates(v2_records, vote, prior, model)
        x, _y, ids = v2.build_count_features(v2_records, vote, per_tree)
        count_models = {
            "ridge": joblib.load(root / f"count_ridge_{mode}.joblib"),
            "hgb": joblib.load(root / f"count_hgb_{mode}.joblib"),
        }
        for kind in ("ridge", "hgb"):
            pred = v2.predict_counts(kind, count_models, x)
            out[f"v2_{mode}_{kind}"] = {
                tree_id: int(n) for tree_id, n in zip(ids, pred)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    del args.seed
    started = time.time()
    ctx = gsp.load_context(DATASET, gsp.FUSED_ROOT, "val", gsp.PROPOSAL_MIN)
    models = {}
    for name, path in MODEL_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        model = joblib.load(path)
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
        models[name] = model
    tables = pair_probability_tables(ctx, models)
    targets = v2_targets(ctx)
    rows = []
    for mix_name, weights in MIXES.items():
        payloads = build_payloads(ctx, tables, weights)
        for tau_prob, payload in payloads.items():
            for target_name, target in targets.items():
                for singleton in SINGLETONS:
                    for rank in RANKS:
                        metrics = evaluate(payload, target, singleton, rank)
                        rows.append({
                            "mix": mix_name, "weights": weights,
                            "tau_prob": tau_prob, "target_source": target_name,
                            "singleton_min": singleton, "rank_mode": rank,
                            "metrics": metrics,
                        })
        print(json.dumps({"mix": mix_name, "rows": len(rows)}, ensure_ascii=False),
              flush=True)

    baseline = next(r["metrics"] for r in rows
                    if r["mix"] == "extra_only" and r["tau_prob"] == .20
                    and r["target_source"] == "original"
                    and r["singleton_min"] == .25 and r["rank_mode"] == "max_member")
    best_match = max(rows, key=lambda r: (
        r["metrics"]["matched_class_accuracy"], r["metrics"]["macro_f1"],
        r["metrics"]["physical_f1"], -r["metrics"]["mae"]))
    best_physical = max(rows, key=lambda r: (
        r["metrics"]["physical_f1"], r["metrics"]["matched_class_accuracy"],
        -r["metrics"]["mae"]))
    best_macro = max(rows, key=lambda r: (
        r["metrics"]["macro_f1"], r["metrics"]["matched_class_accuracy"],
        r["metrics"]["physical_f1"]))
    allround = [r for r in rows if (
        r["metrics"]["physical_f1"] >= baseline["physical_f1"]
        and r["metrics"]["mae"] <= baseline["mae"]
        and r["metrics"]["pm1"] >= baseline["pm1"]
        and r["metrics"]["matched_class_accuracy"] >= baseline["matched_class_accuracy"]
        and r["metrics"]["macro_f1"] >= baseline["macro_f1"])]
    report = {
        "dataset": DATASET,
        "protocol": "TRAIN-fitted edge opinions + saved count opinions; GSP grid selected VAL; no TEST",
        "seed": SEED, "axes": {"taus": TAUS, "singletons": SINGLETONS,
                                  "ranks": RANKS, "max_size": MAX_SIZE},
        "model_paths": {k: str(v) for k, v in MODEL_PATHS.items()},
        "mixes": MIXES, "target_sources": list(targets),
        "baseline_val": baseline, "best_by_matched": best_match,
        "best_by_physical": best_physical, "best_by_macro": best_macro,
        "best_allrounder_guardrail": max(
            allround, key=lambda r: (r["metrics"]["matched_class_accuracy"],
                                     r["metrics"]["physical_f1"],
                                     -r["metrics"]["mae"]), default=None),
        "n_rows": len(rows), "rows": rows,
        "elapsed_sec": time.time() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "953_edge_ensemble_gsp_results_val.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"baseline": baseline, "best_by_matched": best_match,
                      "best_by_physical": best_physical,
                      "best_by_macro": best_macro,
                      "best_allrounder": report["best_allrounder_guardrail"],
                      "n_rows": len(rows), "report": str(path)}, ensure_ascii=False),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
