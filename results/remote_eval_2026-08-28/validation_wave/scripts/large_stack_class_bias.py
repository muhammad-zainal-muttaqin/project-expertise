"""Targeted class-logit calibration over the best 953 opinion stack.

The stack topology and all expert models are fixed from TRAIN.  This module
only tests a small, declared B2/B3/B4 logit-bias grid on VAL to address the
known B2-to-B3 confusion.  It never reads a TEST split.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

import harness
import member_head as mh
import multiscale_member_head as ms
import large_member_head as large


OUT = Path("/workspace/cluster_head/artifacts")
K = harness.K


def key(group: dict) -> tuple:
    return tuple(sorted((int(m["side"]), str(m["stem"]), int(m["row_index"]))
                        for m in group["members"]))


def pool(q, data, pooling):
    flat = [g for _rec, gs in data["groups"] for g in gs]
    out = []
    for gi, rows in enumerate(data["group_rows"]):
        z = q[rows]
        if pooling == "mean":
            w = np.asarray([float(m["score"]) for m in flat[gi]["members"]], dtype=np.float32)
            w /= max(float(w.sum()), 1e-8)
            out.append((z * w[:, None]).sum(axis=0))
        elif pooling == "max":
            out.append(z.max(axis=0))
        else:
            j = int(np.argmax([float(m["score"]) for m in flat[gi]["members"]]))
            out.append(z[j])
    q = np.maximum(np.asarray(out, dtype=np.float32), 1e-8)
    return q / np.maximum(q.sum(axis=1, keepdims=True), 1e-8)


def short(m):
    return {"physical_f1": m["physical_detection"]["f1"],
            "mae": m["counting"]["mae"],
            "pm1": m["counting"]["plus_minus_1_accuracy"],
            "matched_class_accuracy": m["classification"]["matched_class_accuracy"],
            "matched": m["classification"]["matched"],
            "macro_f1": m["classification"]["macro_f1_end_to_end"],
            "per_class_f1": m["classification"]["per_class_f1_end_to_end"]}


def evaluate(data, z, detector):
    pred = np.argmax(z, axis=1).astype(int)
    pmap = {k: int(c) for k, c in zip(data["keys"], pred)}
    m = harness.evaluate_clusters(data["payload"], data["targets"],
                                  harness.PROFILES["953"],
                                  lambda group, pmap=pmap: pmap[key(group)])
    return short(m)


def run(seed: int) -> dict:
    started = time.time()
    dataset = "953"
    val = mh.collect(dataset, "val")
    larg = large.collect(dataset, "val")
    msc = ms.collect(dataset, "val")
    if val["keys"] != larg["keys"] or val["keys"] != msc["keys"]:
        raise RuntimeError("fixed topology differs across experts")
    extra = joblib.load(OUT / "953_member_extra.joblib")
    logistic = joblib.load(OUT / "953_member_logistic.joblib")
    large_hist = joblib.load(OUT / "953_large_hist.joblib")
    ms_extra = joblib.load(OUT / "953_ms_extra.joblib")
    q_extra = pool(extra.predict_proba(val["X"]), val, "max")
    q_log = pool(logistic.predict_proba(val["X"]), val, "max")
    q_large = pool(large_hist.predict_proba(larg["X"]), larg, "mean")
    q_ms = pool(ms_extra.predict_proba(msc["X"]), msc, "mean")
    detector = np.asarray([np.asarray(g["p"], dtype=np.float32)
                           for _r, gs in val["groups"] for g in gs], dtype=np.float32)
    detector = np.maximum(detector, 1e-8)
    detector /= np.maximum(detector.sum(1, keepdims=True), 1e-8)
    stack_specs = {
        "best_matched": (0.10, 0.20, 0.05, 0.10),
        "best_macro": (0.15, 0.00, 0.05, 0.00),
    }
    values = (-.15, -.10, -.05, 0., .05, .10, .15)
    rows = []
    for stack_name, (wl, we, wx, wm) in stack_specs.items():
        z0 = (np.log(detector) + wl * np.log(q_large) + we * np.log(q_extra)
              + wx * np.log(q_log) + wm * np.log(q_ms))
        rows.append({"stack": stack_name, "bias": [0., 0., 0., 0.],
                     "metrics": evaluate(val, z0, detector)})
        # Keep the grid focused on the documented confusion pair and the
        # underrepresented B4 class; B1 is the fixed reference logit.
        for b2 in values:
            for b3 in values:
                for b4 in values:
                    bias = np.asarray([0., b2, b3, b4], dtype=np.float32)
                    rows.append({"stack": stack_name, "bias": bias.tolist(),
                                 "metrics": evaluate(val, z0 + bias, detector)})
    # Both stack variants change only class decisions, so their physical and
    # count metrics are invariant by construction.  Compare all rows and keep
    # the class-metric winner; the second stack has different opinion weights.
    best = max(rows, key=lambda r: (r["metrics"]["matched_class_accuracy"],
                                    r["metrics"]["macro_f1"]))
    report = {"dataset": dataset,
              "protocol": "fixed TRAIN experts; targeted B2/B3/B4 bias selected VAL; no TEST",
              "seed": seed, "values": values, "grid_size": len(rows),
              "rows": rows, "selected_validation": best,
              "elapsed_sec": time.time() - started}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "953_large_stack_class_bias_results_val.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(json.dumps({"selected_validation": best, "report": str(path),
                      "seconds": report["elapsed_sec"]}, ensure_ascii=False), flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()
    run(args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
