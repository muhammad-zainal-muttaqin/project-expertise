"""Targeted validation-only class-logit bias sweep for known 953 errors.

The base ExtraTrees head is fixed from TRAIN.  Only a tiny B2/B3/B4 bias grid
is considered because the locked confusion analysis identified B2->B3 as the
dominant error.  Physical linker and count outputs remain frozen; TEST is not
loaded.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

import sys
sys.path.insert(0, "/workspace/cluster_head")
import cluster_head_experiment as exp  # noqa: E402


def short(m):
    return {"physical_f1": m["physical_detection"]["f1"],
            "mae": m["counting"]["mae"],
            "pm1": m["counting"]["plus_minus_1_accuracy"],
            "matched_class_accuracy": m["classification"]["matched_class_accuracy"],
            "matched": m["classification"]["matched"],
            "macro_f1": m["classification"]["macro_f1_end_to_end"],
            "per_class_f1": m["classification"]["per_class_f1_end_to_end"]}


def main():
    data = exp.collect_groups("953", "val")
    model = joblib.load(exp.ARTIFACTS / "953_extra_compact.joblib")
    q = np.asarray(model.predict_proba(data["X"]), dtype=np.float32)
    detector = np.asarray([np.asarray(g["p"], dtype=np.float32)
                           for _rec, gs in data["groups"] for g in gs])
    detector = np.maximum(detector, 1e-8)
    detector /= detector.sum(axis=1, keepdims=True)
    baseline = exp.harness.evaluate_clusters(
        data["payload"], data["targets"], exp.harness.PROFILES["953"])
    values = (-.20, -.10, -.05, 0., .05, .10, .20)
    rows = []
    for b2 in values:
        for b3 in values:
            for b4 in values:
                bias = np.asarray([0., b2, b3, b4], dtype=np.float32)
                probs = exp._softmax(np.log(detector) + .25 * np.log(np.maximum(q, 1e-8)) + bias)
                pred = np.argmax(probs, axis=1).astype(int)
                m = exp.evaluate_with_map(data, exp._prediction_map(data["keys"], pred))
                rows.append({"bias": bias.tolist(), "metrics": short(m)})
    best_acc = max(rows, key=lambda r: (r["metrics"]["matched_class_accuracy"],
                                        r["metrics"]["macro_f1"]))
    best_macro = max(rows, key=lambda r: (r["metrics"]["macro_f1"],
                                          r["metrics"]["matched_class_accuracy"]))
    report = {"dataset": "953", "protocol": "TRAIN-fitted head, targeted B2/B3/B4 bias selected VAL; no TEST",
              "values": values, "baseline_val": short(baseline),
              "best_by_matched": best_acc, "best_by_macro": best_macro,
              "results": rows}
    out = Path("/workspace/cluster_head/artifacts/953_class_bias_results.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"best_by_matched": best_acc, "best_by_macro": best_macro,
                      "report": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
