"""Small validation-only stacker for detector/base-DINO/aux-DINO opinions.

The three experts are already fit on matched TRAIN clusters.  This script
only combines their probability outputs on VAL and never reads TEST.  The
grid is intentionally narrow and centered on the residual skip weights used
by the preceding experiments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

import sys
sys.path.insert(0, "/workspace/cluster_head")
import cluster_head_experiment as exp  # noqa: E402
sys.path.insert(0, "/workspace/aux_modal")
import multimodal_cluster_head as mm  # noqa: E402


OUT = Path("/workspace/aux_modal/artifacts")
WEIGHTS = (0.0, 0.10, 0.15, 0.25, 0.40, 0.60, 0.85, 1.0)


def short(metrics):
    return {
        "physical_f1": metrics["physical_detection"]["f1"],
        "mae": metrics["counting"]["mae"],
        "pm1": metrics["counting"]["plus_minus_1_accuracy"],
        "matched_class_accuracy": metrics["classification"]["matched_class_accuracy"],
        "matched": metrics["classification"]["matched"],
        "macro_f1": metrics["classification"]["macro_f1_end_to_end"],
        "per_class_f1": metrics["classification"]["per_class_f1_end_to_end"],
    }


def run(dataset: str):
    data = exp.collect_groups(dataset, "val")
    aux_x, aux_diag = mm.modal_matrix(data, dataset, "val", "aux_only")
    base_model = joblib.load(exp.ARTIFACTS / f"{dataset}_extra_compact.joblib")
    aux_model = joblib.load(OUT / f"{dataset}_aux_only_extra_pca128.joblib")
    q_base = np.asarray(base_model.predict_proba(data["X"]), dtype=np.float32)
    q_aux = np.asarray(aux_model.predict_proba(aux_x), dtype=np.float32)
    detector = np.asarray([np.asarray(g["p"], dtype=np.float32)
                           for _rec, gs in data["groups"] for g in gs])
    detector = np.maximum(detector, 1e-8)
    detector /= detector.sum(axis=1, keepdims=True)
    baseline = exp.harness.evaluate_clusters(
        data["payload"], data["targets"], exp.harness.PROFILES[dataset])
    rows = []
    for wb in WEIGHTS:
        for wa in WEIGHTS:
            probs = exp._softmax(np.log(detector) + wb * np.log(np.maximum(q_base, 1e-8))
                                 + wa * np.log(np.maximum(q_aux, 1e-8)))
            pred = np.argmax(probs, axis=1).astype(int)
            pmap = exp._prediction_map(data["keys"], pred)
            metrics = exp.evaluate_with_map(data, pmap)
            s = short(metrics)
            s["physical_count_invariant"] = bool(
                abs(s["physical_f1"] - baseline["physical_detection"]["f1"]) < 1e-10
                and abs(s["mae"] - baseline["counting"]["mae"]) < 1e-10
                and abs(s["pm1"] - baseline["counting"]["plus_minus_1_accuracy"]) < 1e-10)
            rows.append({"base_weight": wb, "aux_weight": wa, "metrics": s})
    eligible = [r for r in rows if r["metrics"]["physical_count_invariant"]]
    best = max(eligible, key=lambda r: (r["metrics"]["matched_class_accuracy"],
                                        r["metrics"]["macro_f1"]))
    report = {"dataset": dataset,
              "protocol": "experts fit TRAIN; opinion weights selected VAL; no TEST",
              "weights": WEIGHTS, "baseline_val": short(baseline),
              "aux_diag": aux_diag, "results": rows,
              "selected_validation": best}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{dataset}_stacked_heads_results.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"dataset": dataset, "selected_validation": best,
                      "report": str(path)}, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("953", "depth"), required=True)
    args = ap.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()
