"""Validation-only stacker for the two member-level experts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

import member_head as mh


OUT = Path("/workspace/cluster_head/artifacts")
WEIGHTS = (0., .05, .10, .15, .20, .30, .45, .60, .80)


def short(m):
    return {"physical_f1": m["physical_detection"]["f1"],
            "mae": m["counting"]["mae"],
            "pm1": m["counting"]["plus_minus_1_accuracy"],
            "matched_class_accuracy": m["classification"]["matched_class_accuracy"],
            "matched": m["classification"]["matched"],
            "macro_f1": m["classification"]["macro_f1_end_to_end"],
            "per_class_f1": m["classification"]["per_class_f1_end_to_end"]}


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-8)


def pool(q_member, data, pooling):
    flat_groups = [g for _rec, gs in data["groups"] for g in gs]
    out = []
    for gi, rows in enumerate(data["group_rows"]):
        q = q_member[rows]
        if pooling == "mean":
            scores = np.asarray([float(m["score"]) for m in flat_groups[gi]["members"]], dtype=np.float32)
            scores /= max(float(scores.sum()), 1e-8)
            out.append((q * scores[:, None]).sum(axis=0))
        elif pooling == "max":
            out.append(q.max(axis=0))
        elif pooling == "top":
            j = int(np.argmax([float(m["score"]) for m in flat_groups[gi]["members"]]))
            out.append(q[j])
        else:
            raise ValueError(pooling)
    return np.asarray(out, dtype=np.float32)


def run(dataset):
    train, val = mh.collect(dataset, "train"), mh.collect(dataset, "val")
    extra = joblib.load(OUT / f"{dataset}_member_extra.joblib")
    logistic = joblib.load(OUT / f"{dataset}_member_logistic.joblib")
    qe = np.asarray(extra.predict_proba(val["X"]), dtype=np.float32)
    ql = np.asarray(logistic.predict_proba(val["X"]), dtype=np.float32)
    detector = np.asarray([np.asarray(g["p"], dtype=np.float32)
                           for _rec, gs in val["groups"] for g in gs])
    detector = np.maximum(detector, 1e-8)
    detector /= detector.sum(axis=1, keepdims=True)
    baseline = mh.harness.evaluate_clusters(val["payload"], val["targets"],
                                            mh.harness.PROFILES[dataset])
    rows = []
    for pooling in ("mean", "max", "top"):
        pe, pl = pool(qe, val, pooling), pool(ql, val, pooling)
        for we in WEIGHTS:
            for wl in WEIGHTS:
                probs = softmax(np.log(detector) + we * np.log(np.maximum(pe, 1e-8))
                                + wl * np.log(np.maximum(pl, 1e-8)))
                pred = np.argmax(probs, axis=1).astype(int)
                pmap = {key: int(cls) for key, cls in zip(val["keys"], pred)}
                metrics = mh.harness.evaluate_clusters(
                    val["payload"], val["targets"], mh.harness.PROFILES[dataset],
                    lambda g, pmap=pmap: pmap[mh.harness_group_key(g)])
                s = short(metrics)
                s["physical_count_invariant"] = bool(
                    abs(s["physical_f1"] - baseline["physical_detection"]["f1"]) < 1e-10
                    and abs(s["mae"] - baseline["counting"]["mae"]) < 1e-10
                    and abs(s["pm1"] - baseline["counting"]["plus_minus_1_accuracy"]) < 1e-10)
                rows.append({"pooling": pooling, "extra_weight": we,
                             "logistic_weight": wl, "metrics": s})
    eligible = [r for r in rows if r["metrics"]["physical_count_invariant"]]
    best = max(eligible, key=lambda r: (r["metrics"]["matched_class_accuracy"],
                                        r["metrics"]["macro_f1"]))
    best_macro = max(eligible, key=lambda r: (r["metrics"]["macro_f1"],
                                              r["metrics"]["matched_class_accuracy"]))
    report = {"dataset": dataset,
              "protocol": "member experts fit TRAIN; pooling/weights selected VAL; no TEST",
              "baseline_val": short(baseline), "weights": WEIGHTS,
              "best_by_matched": best, "best_by_macro": best_macro,
              "results": rows}
    path = OUT / f"{dataset}_member_stacker_results.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"dataset": dataset, "best_by_matched": best,
                      "best_by_macro": best_macro, "report": str(path)},
                     ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("953", "depth"), required=True)
    args = ap.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()
