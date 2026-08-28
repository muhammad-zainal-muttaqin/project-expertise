"""Targeted VAL stack of multi-scale ExtraTrees and base member-logistic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

import member_head as mh
import multiscale_member_head as ms


OUT = Path("/workspace/cluster_head/artifacts")
WEIGHTS = (0., .05, .10, .15, .20, .30, .45, .60, .80)


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-8)


def short(m):
    return {"physical_f1": m["physical_detection"]["f1"],
            "mae": m["counting"]["mae"],
            "pm1": m["counting"]["plus_minus_1_accuracy"],
            "matched_class_accuracy": m["classification"]["matched_class_accuracy"],
            "matched": m["classification"]["matched"],
            "macro_f1": m["classification"]["macro_f1_end_to_end"],
            "per_class_f1": m["classification"]["per_class_f1_end_to_end"]}


def pool(q, data, pooling):
    flat = [g for _rec, gs in data["groups"] for g in gs]
    out = []
    for gi, rows in enumerate(data["group_rows"]):
        z = q[rows]
        if pooling == "mean":
            w = np.asarray([float(m["score"]) for m in flat[gi]["members"]], dtype=np.float32)
            w /= max(float(w.sum()), 1e-8)
            out.append((z * w[:, None]).sum(0))
        elif pooling == "max":
            out.append(z.max(0))
        else:
            j = int(np.argmax([float(m["score"]) for m in flat[gi]["members"]]))
            out.append(z[j])
    return np.asarray(out, dtype=np.float32)


def run(dataset):
    # Both collectors use the same frozen linker profile; use the multiscale
    # collector for the validation groups and the member collector for the
    # base member features.  Group keys/order are checked explicitly.
    val_ms = ms.collect(dataset, "val")
    val_base = mh.collect(dataset, "val")
    if val_ms["keys"] != val_base["keys"]:
        raise RuntimeError("frozen group topology differs between collectors")
    ms_model = joblib.load(OUT / f"{dataset}_ms_extra.joblib")
    mem_model = joblib.load(OUT / f"{dataset}_member_logistic.joblib")
    qms_mem = np.asarray(ms_model.predict_proba(val_ms["X"]), dtype=np.float32)
    qmem = np.asarray(mem_model.predict_proba(val_base["X"]), dtype=np.float32)
    detector = np.asarray([np.asarray(g["p"], dtype=np.float32)
                           for _rec, gs in val_ms["groups"] for g in gs])
    detector = np.maximum(detector, 1e-8); detector /= detector.sum(axis=1, keepdims=True)
    baseline = mh.harness.evaluate_clusters(val_ms["payload"], val_ms["targets"],
                                            mh.harness.PROFILES[dataset])
    rows = []
    for pooling in ("mean", "max", "top"):
        pms, pmem = pool(qms_mem, val_ms, pooling), pool(qmem, val_base, pooling)
        for wm in WEIGHTS:
            for wl in WEIGHTS:
                z = np.log(detector) + wm * np.log(np.maximum(pms, 1e-8)) \
                    + wl * np.log(np.maximum(pmem, 1e-8))
                pred = np.argmax(z, axis=1).astype(int)
                pmap = {key: int(cls) for key, cls in zip(val_ms["keys"], pred)}
                m = mh.harness.evaluate_clusters(
                    val_ms["payload"], val_ms["targets"], mh.harness.PROFILES[dataset],
                    lambda g, pmap=pmap: pmap[mh.harness_group_key(g)])
                s = short(m)
                s["physical_count_invariant"] = bool(
                    abs(s["physical_f1"] - baseline["physical_detection"]["f1"]) < 1e-10
                    and abs(s["mae"] - baseline["counting"]["mae"]) < 1e-10
                    and abs(s["pm1"] - baseline["counting"]["plus_minus_1_accuracy"]) < 1e-10)
                rows.append({"pooling": pooling, "multiscale_weight": wm,
                             "member_logistic_weight": wl, "metrics": s})
    eligible = [r for r in rows if r["metrics"]["physical_count_invariant"]]
    best = max(eligible, key=lambda r: (r["metrics"]["matched_class_accuracy"],
                                        r["metrics"]["macro_f1"]))
    best_macro = max(eligible, key=lambda r: (r["metrics"]["macro_f1"],
                                              r["metrics"]["matched_class_accuracy"]))
    report = {"dataset": dataset,
              "protocol": "experts fit TRAIN; weights/pooling selected VAL; no TEST",
              "baseline_val": short(baseline), "weights": WEIGHTS,
              "best_by_matched": best, "best_by_macro": best_macro, "results": rows}
    path = OUT / f"{dataset}_multiscale_stacker_results.json"
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
