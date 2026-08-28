"""Validation-only stack of the DINOv2-Large and Base member opinions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

import harness
import large_member_head as large
import member_head as mh
import multiscale_member_head as ms


OUT = Path("/workspace/cluster_head/artifacts")


def pool(q, data, pooling):
    flat = [g for _rec, gs in data["groups"] for g in gs]
    result = []
    for gi, rows in enumerate(data["group_rows"]):
        z = q[rows]
        members = flat[gi]["members"]
        if pooling == "mean":
            w = np.asarray([float(m["score"]) for m in members], dtype=np.float32)
            w /= max(float(w.sum()), 1e-8)
            result.append((z * w[:, None]).sum(axis=0))
        elif pooling == "max":
            result.append(z.max(axis=0))
        elif pooling == "top":
            j = int(np.argmax([float(m["score"]) for m in members]))
            result.append(z[j])
        else:
            raise ValueError(pooling)
    return np.asarray(result, dtype=np.float32)


def short(m):
    return {"physical_f1": m["physical_detection"]["f1"],
            "mae": m["counting"]["mae"],
            "pm1": m["counting"]["plus_minus_1_accuracy"],
            "matched_class_accuracy": m["classification"]["matched_class_accuracy"],
            "matched": m["classification"]["matched"],
            "macro_f1": m["classification"]["macro_f1_end_to_end"],
            "per_class_f1": m["classification"]["per_class_f1_end_to_end"]}


def run(dataset: str):
    base = mh.collect(dataset, "val")
    larg = large.collect(dataset, "val")
    msc = ms.collect(dataset, "val")
    if base["keys"] != larg["keys"] or base["keys"] != msc["keys"]:
        raise RuntimeError("frozen topology differs between feature collectors")
    extra = joblib.load(OUT / f"{dataset}_member_extra.joblib")
    logistic = joblib.load(OUT / f"{dataset}_member_logistic.joblib")
    large_hist = joblib.load(OUT / f"{dataset}_large_hist.joblib")
    ms_extra = joblib.load(OUT / f"{dataset}_ms_extra.joblib")
    q_extra = pool(np.asarray(extra.predict_proba(base["X"]), dtype=np.float32), base, "max")
    q_logistic = pool(np.asarray(logistic.predict_proba(base["X"]), dtype=np.float32), base, "max")
    q_large = pool(np.asarray(large_hist.predict_proba(larg["X"]), dtype=np.float32), larg, "mean")
    q_ms = pool(np.asarray(ms_extra.predict_proba(msc["X"]), dtype=np.float32), msc, "mean")
    detector = np.asarray([np.asarray(g["p"], dtype=np.float32)
                           for _rec, gs in base["groups"] for g in gs])
    detector = np.maximum(detector, 1e-8)
    detector /= detector.sum(axis=1, keepdims=True)
    baseline_m = harness.evaluate_clusters(base["payload"], base["targets"],
                                            harness.PROFILES[dataset])
    baseline = short(baseline_m)
    large_w = (0., .05, .10, .15, .20, .30, .45, .60, .80, 1.0)
    base_w = (0., .05, .10, .20, .30, .45)
    log_w = (0., .05, .10, .15, .20)
    ms_w = (0., .05, .10, .15)
    rows = []
    for wl in large_w:
        for we in base_w:
            for wx in log_w:
                for wm in ms_w:
                    z = (np.log(detector) + wl * np.log(np.maximum(q_large, 1e-8))
                         + we * np.log(np.maximum(q_extra, 1e-8))
                         + wx * np.log(np.maximum(q_logistic, 1e-8))
                         + wm * np.log(np.maximum(q_ms, 1e-8)))
                    pred = np.argmax(z, axis=1).astype(int)
                    pmap = {k: int(c) for k, c in zip(base["keys"], pred)}
                    m = harness.evaluate_clusters(
                        base["payload"], base["targets"], harness.PROFILES[dataset],
                        lambda g, pmap=pmap: pmap[mh.harness_group_key(g)])
                    s = short(m)
                    s["physical_count_invariant"] = bool(
                        abs(s["physical_f1"] - baseline["physical_f1"]) < 1e-10
                        and abs(s["mae"] - baseline["mae"]) < 1e-10
                        and abs(s["pm1"] - baseline["pm1"]) < 1e-10)
                    rows.append({"large_weight": wl, "base_extra_weight": we,
                                 "base_logistic_weight": wx, "multiscale_weight": wm,
                                 "metrics": s})
    eligible = [r for r in rows if r["metrics"]["physical_count_invariant"]]
    best_match = max(eligible, key=lambda r:(r["metrics"]["matched_class_accuracy"],
                                              r["metrics"]["macro_f1"]))
    best_macro = max(eligible, key=lambda r:(r["metrics"]["macro_f1"],
                                               r["metrics"]["matched_class_accuracy"]))
    report = {"dataset": dataset,
              "protocol": "fit experts TRAIN; stack weights/pooling fixed then selected VAL; no TEST",
              "experts": ["detector_skip", "DINOv2-Large_hist_mean",
                          "DINOv2-Base_member_extra_max", "DINOv2-Base_member_logistic_max",
                          "DINOv2-Base_multiscale_extra_mean"],
              "baseline_val": baseline, "grid_size": len(rows),
              "best_by_matched": best_match, "best_by_macro": best_macro,
              "results": rows}
    out = OUT / f"{dataset}_large_stacker_results.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({"dataset": dataset, "best_by_matched": best_match,
                      "best_by_macro": best_macro, "report": str(out)},
                     ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("953", "depth"), required=True)
    args = ap.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()
