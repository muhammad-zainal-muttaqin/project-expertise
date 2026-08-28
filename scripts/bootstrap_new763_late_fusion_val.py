#!/usr/bin/env python3
"""Paired image-level bootstrap for a frozen late-fusion VAL submission.

The script is intentionally generic over two already-produced prediction
dumps, but the only legal split is the 468-image new763 ``valid`` split.  It
never accepts a split argument or constructs a test path.  Use it after a
fixed fusion recipe has been screened; it does not select weights or recipes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bootstrap_new763_rgbd4_val import (  # noqa: E402
    build_gt_coco,
    ci95,
    eval_resample,
    load_gt_struct,
    load_predictions,
    preds_to_dt_by_idx,
)


_WORKER: dict = {}


def worker_init(image_wh, gt_anns, baseline_dt, candidate_dt):
    _WORKER.update(
        image_wh=image_wh,
        gt_anns=gt_anns,
        baseline_dt=baseline_dt,
        candidate_dt=candidate_dt,
    )


def process_resample(task):
    resample_id, indices = task
    gt = build_gt_coco(indices, _WORKER["image_wh"], _WORKER["gt_anns"])
    baseline = eval_resample(gt, indices, _WORKER["baseline_dt"])
    candidate = eval_resample(gt, indices, _WORKER["candidate_dt"])
    return int(resample_id), baseline, candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--baseline-metric", type=float, required=True)
    parser.add_argument("--candidate-metric", type=float, required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/new763_rgbd4"))
    parser.add_argument("--n-resamples", type=int, default=500)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.n_resamples < 100 or args.workers < 1:
        raise ValueError("use at least 100 resamples and one worker")
    for path in (args.baseline_predictions, args.candidate_predictions):
        if not path.is_file():
            raise FileNotFoundError(path)

    stems, image_wh, gt_anns = load_gt_struct(args.dataset)
    if len(stems) != 468:
        raise ValueError(f"new763 VAL must contain 468 images, got {len(stems)}")
    baseline = load_predictions(args.baseline_predictions)
    candidate = load_predictions(args.candidate_predictions)
    if set(baseline) != set(stems) or set(candidate) != set(stems):
        raise ValueError("both prediction dumps must cover exactly the same 468 VAL stems")
    baseline_dt = preds_to_dt_by_idx(stems, baseline)
    candidate_dt = preds_to_dt_by_idx(stems, candidate)

    identity = np.arange(len(stems), dtype=np.int64)
    identity_gt = build_gt_coco(identity, image_wh, gt_anns)
    identity_baseline = eval_resample(identity_gt, identity, baseline_dt)
    identity_candidate = eval_resample(identity_gt, identity, candidate_dt)
    if abs(identity_baseline - args.baseline_metric) > 1e-6:
        raise RuntimeError(
            f"baseline identity mismatch: {identity_baseline:.9f}/{args.baseline_metric:.9f}"
        )
    if abs(identity_candidate - args.candidate_metric) > 1e-6:
        raise RuntimeError(
            f"candidate identity mismatch: {identity_candidate:.9f}/{args.candidate_metric:.9f}"
        )
    print(
        f"identity sanity OK: baseline={identity_baseline:.6f}, "
        f"candidate={identity_candidate:.6f}, "
        f"delta={identity_candidate - identity_baseline:+.6f}",
        flush=True,
    )

    rng = np.random.RandomState(args.seed)
    resamples = rng.randint(0, len(stems), size=(args.n_resamples, len(stems)))
    baseline_values = np.empty(args.n_resamples, dtype=np.float64)
    candidate_values = np.empty(args.n_resamples, dtype=np.float64)
    started = time.time()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=worker_init,
        initargs=(image_wh, gt_anns, baseline_dt, candidate_dt),
    ) as pool:
        tasks = ((idx, resamples[idx]) for idx in range(args.n_resamples))
        for done, (resample_id, baseline_value, candidate_value) in enumerate(
            pool.map(process_resample, tasks, chunksize=2), start=1
        ):
            baseline_values[resample_id] = baseline_value
            candidate_values[resample_id] = candidate_value
            if done % 50 == 0 or done == args.n_resamples:
                print(f"{done}/{args.n_resamples} resamples; {time.time() - started:.1f}s", flush=True)

    delta = candidate_values - baseline_values
    result = {
        "schema_version": 1,
        "analysis": "paired image-level bootstrap, new763 VAL only",
        "candidate": args.candidate_label,
        "protocol": {
            "n_images": len(stems),
            "n_resamples": args.n_resamples,
            "seed": args.seed,
            "same_resample_indices_for_both_submissions": True,
            "evaluator": "pycocotools.COCOeval",
            "selection": "none inside bootstrap; candidate recipe was fixed before this analysis",
            "test_access": "forbidden; this script has no test path or test option",
        },
        "point": {
            "baseline_mAP50": args.baseline_metric,
            "candidate_mAP50": args.candidate_metric,
            "delta_candidate_minus_baseline": args.candidate_metric - args.baseline_metric,
        },
        "bootstrap": {
            "baseline_mAP50_ci95": ci95(baseline_values),
            "candidate_mAP50_ci95": ci95(candidate_values),
            "delta_ci95": ci95(delta),
            "delta_mean": float(delta.mean()),
            "fraction_delta_gt_zero": float((delta > 0).mean()),
            "significant_excludes_zero": bool(
                np.percentile(delta, 2.5) > 0 or np.percentile(delta, 97.5) < 0
            ),
        },
        "identity_sanity": {
            "baseline": identity_baseline,
            "candidate": identity_candidate,
        },
        "inputs": {
            "dataset": str(args.dataset.resolve()),
            "baseline_predictions": str(args.baseline_predictions.resolve()),
            "candidate_predictions": str(args.candidate_predictions.resolve()),
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
