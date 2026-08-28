"""LOCKED TEST STAGE driver for the map_boost detector-metric (COCO mAP) pipeline.

This script performs the single authorized test opening for the oil-palm
four-view detection project. It imports and reuses `rank_and_emit.py`
functions by reference (no copy-paste, no reimplementation) so the test
feature pipeline is bit-identical to the val pipeline that selected the four
locked profiles below. It does NOT refit or retrain anything: it only loads
the already-fitted rankers under artifacts/{953,depth}/ranker_floor*.joblib
and reuses the val-selected scoring/emission rules on the test split.

Locked profiles (fixed by the orchestrating task; no other configuration may
be evaluated here):
  953   agnostic:   floor=0.02, a=0,   b=1               (score = p_tp)
  953   classaware: floor=0.01, a=0,   b=1,   gamma=1.0  (p_tp * p_c)
  depth agnostic:   floor=0.05, a=1,   b=0.5             (wbf_score * p_tp^0.5)
  depth classaware: floor=0.02, a=1,   b=0,   gamma=1.0  (wbf_score * p_c)

Guard: refuses to run if either artifacts/{dataset}/results_test_locked.json
already exists -- this is a single-test-opening pipeline, never re-select
after seeing test numbers.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, "/workspace/map_boost")
sys.path.insert(0, "/workspace/project-expertise/scripts")

import rank_and_emit as rae  # noqa: E402

base = rae.base  # eval_remote_pipeline_postprocess
edge = rae.edge  # train_detection_edge_linker

ARTIFACTS = rae.ARTIFACTS

LOCKED_PROFILES = {
    "953": {
        "agnostic": {"floor": 0.02, "a": 0.0, "b": 1.0},
        "classaware": {"floor": 0.01, "a": 0.0, "b": 1.0, "gamma": 1.0},
    },
    "depth": {
        "agnostic": {"floor": 0.05, "a": 1.0, "b": 0.5},
        "classaware": {"floor": 0.02, "a": 1.0, "b": 0.0, "gamma": 1.0},
    },
}
TEST_BASELINES = {
    "agnostic": {"953": 0.8350, "depth": 0.8764},
    "classaware": {"953": 0.5861, "depth": 0.6691},
}


def guard() -> None:
    existing = []
    for dataset in ("953", "depth"):
        p = ARTIFACTS / dataset / "results_test_locked.json"
        if p.exists():
            existing.append(str(p))
    if existing:
        print("GUARD FAILED: test-locked results already exist -- refusing to "
              "reopen the single test-opening for this track.", file=sys.stderr)
        for p in existing:
            print(f"  exists: {p}", file=sys.stderr)
        raise SystemExit(1)


def floors_needed(dataset: str) -> list[float]:
    profs = LOCKED_PROFILES[dataset]
    return sorted({profs["agnostic"]["floor"], profs["classaware"]["floor"]}, reverse=True)


def bootstrap_ci_note() -> dict:
    return {
        "attempted": True,
        "ran": False,
        "reason": (
            "bootstrap_map.py hardcodes dataset /workspace/SawitMVC-Depth (the "
            "352 canonical split, its own YOLO-txt GT loader, fixed 1280x800 "
            "image size) with no way to point it at the SawitMVC-YOLO / "
            "SawitMVC-Depth-YOLO COCO configs used in this pipeline -- not "
            "compatible without modification. bootstrap_map_from_npz.py "
            "hardcodes a CORPORA dict (new763, combined1716) with fixed "
            "single-model prediction paths and a fixed output path, its "
            "main() takes no CLI arguments, and it re-implements AP50 itself "
            "(a hand-rolled greedy-IoU/101-point routine, not "
            "base.coco_metrics/pycocotools) rather than exposing a generic "
            "entry point that accepts an arbitrary submission dict. Wiring "
            "either script to this task's fused/re-ranked test submissions "
            "would require writing new glue code around their internals, "
            "which the task explicitly instructs against ('do not build new "
            "bootstrap machinery'). Skipped per instructions; no CI computed "
            "for the four locked test metrics."
        ),
    }


def run_dataset(dataset: str) -> dict:
    print(f"\n########## TEST-LOCKED {dataset} ##########", flush=True)
    cfg = rae.cfg_for(dataset)
    edge_model = joblib.load(rae.EDGE_MODEL_PATHS[dataset])
    prior = base.build_rotation_prior(base.load_records(cfg, "train"))
    depth_cache: dict = {}

    records_test = base.load_records(cfg, "test")
    n_trees = len(records_test)
    n_images = sum(len(rec["views"]) for rec in records_test.values())
    record_view_stems = {view["stem"] for rec in records_test.values()
                          for view in rec["views"].values()}

    raw_stems: set[str] = set()
    for model in rae.MODELS:
        p = rae.raw_pred_path(dataset, model, "test")
        z = np.load(p)
        raw_stems |= set(z.files)

    # Diagnostics only -- fuse_corpus already handles missing stems on either
    # side gracefully (a stem with no raw detections from any model simply
    # fuses to zero boxes; a raw stem outside the metadata is never scored).
    missing_predictions = sorted(record_view_stems - raw_stems)
    extra_predictions = sorted(raw_stems - record_view_stems)

    result: dict = {
        "dataset": dataset, "n_trees": n_trees, "n_images": n_images,
        "n_raw_stems": len(raw_stems),
        "missing_records_count": len(missing_predictions),
        "extra_predictions_count": len(extra_predictions),
        "floors": {},
        "profiles": {},
    }

    profs = LOCKED_PROFILES[dataset]
    per_floor_ctx = {}
    for floor in floors_needed(dataset):
        t0 = time.time()
        vote_data = rae.build_or_load_vote(dataset, "test", floor)
        feat_data = rae.build_features(dataset, "test", floor, edge_model, prior, depth_cache)
        model_path = rae.ranker_path(dataset, floor)
        if not model_path.exists():
            raise FileNotFoundError(f"locked ranker missing (must not refit): {model_path}")
        model = joblib.load(model_path)
        p_tp_map = rae.p_tp_map_from(feat_data, model)
        dt = time.time() - t0
        per_floor_ctx[floor] = {"vote": vote_data["vote"], "p_tp_map": p_tp_map,
                                "n_boxes_fused": vote_data["n_boxes"]}
        result["floors"][str(floor)] = {
            "seconds": dt,
            "n_boxes_fused": vote_data["n_boxes"],
            "n_boxes_scored": int(feat_data["n_boxes"]),
            "ranker_path": str(model_path),
        }
        print(f"  floor={floor}: {dt:.1f}s, fused_boxes={vote_data['n_boxes']}, "
              f"scored_boxes={feat_data['n_boxes']}", flush=True)

    if dataset == "depth":
        depth_missing = sum(1 for (split, _stem), v in depth_cache.items()
                            if split == "test" and v is None)
        depth_total = sum(1 for (split, _stem) in depth_cache if split == "test")
        result["missing_depth_count"] = depth_missing
        result["depth_total_stems"] = depth_total

    # --- agnostic (locked profile) ---
    prof_a = profs["agnostic"]
    ctx = per_floor_ctx[prof_a["floor"]]
    preds_a = rae.score_agnostic(ctx["vote"], ctx["p_tp_map"], prof_a["a"], prof_a["b"])
    metrics_a = base.coco_metrics(cfg["data_root"], preds_a, agnostic=True, split="test")
    result["profiles"]["agnostic"] = {
        "profile": prof_a,
        "AP50": metrics_a["mAP50"], "mAP50_95": metrics_a["mAP50_95"],
        "n_images_eval": metrics_a["n_images"],
        "baseline": TEST_BASELINES["agnostic"][dataset],
        "delta": metrics_a["mAP50"] - TEST_BASELINES["agnostic"][dataset],
        "n_boxes_fused_at_floor": ctx["n_boxes_fused"],
    }

    # --- class-aware (locked profile) ---
    prof_c = profs["classaware"]
    ctx = per_floor_ctx[prof_c["floor"]]
    preds_c = rae.score_classaware(ctx["vote"], ctx["p_tp_map"], prof_c["a"], prof_c["b"],
                                   prof_c["gamma"])
    metrics_c = base.coco_metrics(cfg["data_root"], preds_c, agnostic=False, split="test")
    result["profiles"]["classaware"] = {
        "profile": prof_c,
        "mAP50": metrics_c["mAP50"], "mAP50_95": metrics_c["mAP50_95"],
        "per_class_AP50": metrics_c["per_class_AP50"],
        "n_images_eval": metrics_c["n_images"],
        "baseline": TEST_BASELINES["classaware"][dataset],
        "delta": metrics_c["mAP50"] - TEST_BASELINES["classaware"][dataset],
        "n_boxes_fused_at_floor": ctx["n_boxes_fused"],
    }

    print(f"  [agnostic]   AP50={metrics_a['mAP50']:.4f} "
          f"(baseline {TEST_BASELINES['agnostic'][dataset]:.4f}, "
          f"delta={result['profiles']['agnostic']['delta']:+.4f})", flush=True)
    print(f"  [classaware] mAP50={metrics_c['mAP50']:.4f} "
          f"(baseline {TEST_BASELINES['classaware'][dataset]:.4f}, "
          f"delta={result['profiles']['classaware']['delta']:+.4f})", flush=True)
    return result


def main() -> int:
    guard()
    t_start = time.time()
    generated_at = datetime.now(timezone.utc).isoformat()
    ci_note = bootstrap_ci_note()

    per_dataset = {}
    for dataset in ("953", "depth"):
        per_dataset[dataset] = run_dataset(dataset)

    wall_seconds_total = time.time() - t_start

    for dataset in ("953", "depth"):
        d = per_dataset[dataset]
        payload = {
            "dataset": dataset,
            "status": "test-locked",
            "generated_at": generated_at,
            "profile_agnostic": LOCKED_PROFILES[dataset]["agnostic"],
            "profile_classaware": LOCKED_PROFILES[dataset]["classaware"],
            "test_metrics": {
                "agnostic": d["profiles"]["agnostic"],
                "classaware": d["profiles"]["classaware"],
            },
            "test_baselines": {
                "agnostic": TEST_BASELINES["agnostic"][dataset],
                "classaware": TEST_BASELINES["classaware"][dataset],
            },
            "fused_box_counts_by_floor": d["floors"],
            "n_images": d["n_images"],
            "n_trees": d["n_trees"],
            "n_raw_stems": d["n_raw_stems"],
            "missing_records_count": d["missing_records_count"],
            "extra_predictions_count": d["extra_predictions_count"],
            "bootstrap_ci": ci_note,
        }
        if dataset == "depth":
            payload["missing_depth_count"] = d["missing_depth_count"]
            payload["depth_total_stems"] = d["depth_total_stems"]
        result_path = ARTIFACTS / dataset / "results_test_locked.json"
        result_path.write_text(
            json.dumps(payload, indent=2, default=rae.json_default) + "\n")
        print(f"-> wrote {result_path}", flush=True)

    combined_path = ARTIFACTS / "results_test_locked.json"
    combined = {
        "generated_at": generated_at,
        "status": "test-locked",
        "locked_profiles": LOCKED_PROFILES,
        "test_baselines": TEST_BASELINES,
        "bootstrap_ci": ci_note,
        "wall_seconds_total": wall_seconds_total,
        "datasets": per_dataset,
    }
    combined_path.write_text(json.dumps(combined, indent=2, default=rae.json_default) + "\n")
    print(f"-> wrote {combined_path}", flush=True)
    print(f"TOTAL WALL TIME: {wall_seconds_total:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
