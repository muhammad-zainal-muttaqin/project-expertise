#!/usr/bin/env python3
"""Fine local refinement of the validation-only class temperature profiles.

The coarse scale/bias search has already identified a promising point.  This
script searches only a small neighborhood around that point on the same
TRAIN-fitted opinions and frozen VAL groups.  No detector, linker, count
target, or TEST artifact is changed or read.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np

import class_bias_general as base


OUT = Path("/workspace/cluster_head/artifacts")


def keep_top(bucket: list[dict], row: dict, key_name: str, n: int = 16) -> None:
    bucket.append(row)
    bucket.sort(
        key=lambda item: (
            item["metrics"][key_name],
            item["metrics"][
                "macro_f1" if key_name == "matched_class_accuracy"
                else "matched_class_accuracy"
            ],
        ),
        reverse=True,
    )
    del bucket[n:]


def run(dataset: str, spec_name: str) -> dict:
    started = time.time()
    coarse = json.loads(
        (OUT / f"{dataset}_class_bias_general_results_val.json").read_text(
            encoding="utf-8"
        )
    )
    spec = coarse["specs"][spec_name]
    data = base.mh.collect(dataset, "val")
    views = base.build_views(dataset, data)
    detector = np.asarray(
        [np.asarray(group["p"], dtype=np.float32)
         for _record, groups in data["groups"] for group in groups]
    )
    detector = np.maximum(detector, 1e-8)
    detector /= np.maximum(detector.sum(axis=1, keepdims=True), 1e-8)
    logits = np.log(detector)
    for view_name, weight in spec["weights"].items():
        logits += float(weight) * np.log(np.maximum(views[view_name], 1e-8))
    evaluate = base.static_evaluator(data, {
        "physical_f1": coarse["baseline_val"]["physical_f1"],
        "mae": coarse["baseline_val"]["mae"],
        "pm1": coarse["baseline_val"]["pm1"],
    })

    if dataset == "953":
        scales = ((.95, 1., 1.05), (1., 1.05, 1.10, 1.15),
                  (.90, .95, 1., 1.05, 1.10),
                  (.75, .80, .85, .90, .95))
        bias_values = ((.10, .15, .20), (-.10, -.05, 0., .05),
                       (-.20, -.15, -.10, -.05, 0.))
    else:
        scales = ((.95, 1., 1.05), (1.10, 1.15, 1.20, 1.25),
                  (.75, .80, .85, .90, .95),
                  (.75, .80, .85, .90, .95))
        bias_values = ((.10, .15, .20), (-.10, -.05, 0., .05),
                       (-.10, -.05, 0., .05))

    top_match: list[dict] = []
    top_macro: list[dict] = []
    grid_size = 1
    for values in scales:
        grid_size *= len(values)
    for values in bias_values:
        grid_size *= len(values)
    for scale_tuple in product(*scales):
        scaled = logits * np.asarray(scale_tuple, dtype=np.float32)
        for b2, b3, b4 in product(*bias_values):
            bias = np.asarray([0., b2, b3, b4], dtype=np.float32)
            row = {
                "scales": list(scale_tuple), "bias": bias.tolist(),
                "metrics": evaluate(np.argmax(scaled + bias, axis=1)),
            }
            keep_top(top_match, row, "matched_class_accuracy")
            keep_top(top_macro, row, "macro_f1")

    output = {
        "dataset": dataset,
        "protocol": "fine local class temperature/bias refinement; VAL only; no TEST",
        "spec": spec_name,
        "coarse_best_by_matched": spec["scale_grid"]["best_by_matched"],
        "coarse_best_by_macro": spec["scale_grid"]["best_by_macro"],
        "grid": {"scales": [list(x) for x in scales],
                 "bias_values": [list(x) for x in bias_values],
                 "size": grid_size},
        "best_by_matched": top_match[0],
        "best_by_macro": top_macro[0],
        "top_by_matched": top_match,
        "top_by_macro": top_macro,
        "elapsed_sec": time.time() - started,
    }
    path = OUT / f"{dataset}_class_bias_refine_results_val.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(json.dumps({"dataset": dataset,
                      "coarse": output["coarse_best_by_matched"],
                      "best_by_matched": output["best_by_matched"],
                      "best_by_macro": output["best_by_macro"],
                      "grid_size": grid_size, "report": str(path)},
                     ensure_ascii=False), flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("953", "depth"), required=True)
    parser.add_argument("--spec", default=None)
    args = parser.parse_args()
    if args.spec is None:
        args.spec = "robust_953_anchor" if args.dataset == "953" else "member_stack_macro"
    run(args.dataset, args.spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
