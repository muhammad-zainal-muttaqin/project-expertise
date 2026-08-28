"""Rebuild the four locked TEST submissions bit-identically from cache +
rankers (reusing rank_and_emit.py functions by reference, exactly as
run_test_locked.py does), load the four baseline TEST prediction dicts from
the recorded npz, and verify the four new-submission point AP50/mAP50 values
against the locked results_test_locked.json numbers to 4 decimals.

Read-only against /workspace/map_boost and /workspace/model_artifacts.
Writes only under /workspace/ci_boot/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, "/workspace/map_boost")
sys.path.insert(0, "/workspace/project-expertise/scripts")

import rank_and_emit as rae  # noqa: E402
import run_test_locked as rtl  # noqa: E402  (safe: only defines names at import time)

base = rae.base  # eval_remote_pipeline_postprocess

CI_ROOT = Path("/workspace/ci_boot")
CACHE_DIR = CI_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_DIR = Path(
    "/workspace/model_artifacts/project-expertise/eval_2026-08-27/fused_combined1716_test_rebuilt"
)
BASELINE_FILES = {
    ("953", "agnostic"): BASELINE_DIR / "SawitMVC_YOLO__wbf_agnostic.npz",
    ("953", "classaware"): BASELINE_DIR / "SawitMVC_YOLO__wbf_classaware.npz",
    ("depth", "agnostic"): BASELINE_DIR / "SawitMVC_Depth_YOLO__wbf_agnostic.npz",
    ("depth", "classaware"): BASELINE_DIR / "SawitMVC_Depth_YOLO__wbf_classaware.npz",
}

# Known point values (task spec + results_test_locked.json), for a 4-decimal
# match gate.
EXPECTED = {
    ("953", "agnostic"): 0.8419,
    ("953", "classaware"): 0.5970,
    ("depth", "agnostic"): 0.8783,
    ("depth", "classaware"): 0.6552,
}
EXPECTED_BASELINE = {
    ("953", "agnostic"): 0.8350,
    ("953", "classaware"): 0.5861,
    ("depth", "agnostic"): 0.8764,
    ("depth", "classaware"): 0.6691,
}


def load_npz_rows(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {k: np.asarray(z[k], dtype=np.float64) for k in z.files}


def rebuild_dataset(dataset: str) -> dict:
    print(f"\n=== rebuild {dataset} ===", flush=True)
    cfg = rae.cfg_for(dataset)
    profs = rtl.LOCKED_PROFILES[dataset]
    per_floor_ctx = {}
    for floor in rtl.floors_needed(dataset):
        t0 = time.time()
        vote_data = rae.build_or_load_vote(dataset, "test", floor)
        feat_data = rae.build_features(dataset, "test", floor, None, None, {})
        model_path = rae.ranker_path(dataset, floor)
        if not model_path.exists():
            raise FileNotFoundError(f"locked ranker missing: {model_path}")
        model = joblib.load(model_path)
        p_tp_map = rae.p_tp_map_from(feat_data, model)
        per_floor_ctx[floor] = {"vote": vote_data["vote"], "p_tp_map": p_tp_map}
        print(f"  floor={floor}: {time.time() - t0:.1f}s", flush=True)

    prof_a = profs["agnostic"]
    ctx = per_floor_ctx[prof_a["floor"]]
    preds_a = rae.score_agnostic(ctx["vote"], ctx["p_tp_map"], prof_a["a"], prof_a["b"])
    metrics_a = base.coco_metrics(cfg["data_root"], preds_a, agnostic=True, split="test")

    prof_c = profs["classaware"]
    ctx = per_floor_ctx[prof_c["floor"]]
    preds_c = rae.score_classaware(ctx["vote"], ctx["p_tp_map"], prof_c["a"], prof_c["b"],
                                    prof_c["gamma"])
    metrics_c = base.coco_metrics(cfg["data_root"], preds_c, agnostic=False, split="test")

    print(f"  [agnostic]   AP50={metrics_a['mAP50']:.4f} (expected {EXPECTED[(dataset,'agnostic')]:.4f})",
          flush=True)
    print(f"  [classaware] mAP50={metrics_c['mAP50']:.4f} (expected {EXPECTED[(dataset,'classaware')]:.4f})",
          flush=True)

    return {
        "preds_agnostic": preds_a, "preds_classaware": preds_c,
        "AP50_agnostic": metrics_a["mAP50"], "mAP50_classaware": metrics_c["mAP50"],
        "n_images_agnostic": metrics_a["n_images"], "n_images_classaware": metrics_c["n_images"],
    }


def main() -> int:
    t_start = time.time()
    submissions = {}
    ok = True

    for dataset in ("953", "depth"):
        result = rebuild_dataset(dataset)
        submissions[dataset] = result
        for kind, actual_key in (("agnostic", "AP50_agnostic"), ("classaware", "mAP50_classaware")):
            actual = round(result[actual_key], 4)
            expected = EXPECTED[(dataset, kind)]
            passed = actual == expected
            ok = ok and passed
            print(f"CHECK new {dataset:6s} {kind:11s} actual={actual:.4f} "
                  f"expected={expected:.4f} {'PASS' if passed else 'FAIL'}", flush=True)

    baseline_preds = {}
    baseline_point_values = {}
    for (dataset, kind), path in BASELINE_FILES.items():
        cfg = rae.cfg_for(dataset)
        preds = load_npz_rows(path)
        baseline_preds[(dataset, kind)] = preds
        metrics = base.coco_metrics(cfg["data_root"], preds, agnostic=(kind == "agnostic"),
                                     split="test")
        baseline_point_values[(dataset, kind)] = metrics["mAP50"]
        actual = round(metrics["mAP50"], 4)
        expected = EXPECTED_BASELINE[(dataset, kind)]
        passed = actual == expected
        ok = ok and passed
        print(f"CHECK baseline {dataset:6s} {kind:11s} actual={actual:.4f} "
              f"expected={expected:.4f} {'PASS' if passed else 'FAIL'}", flush=True)

    if not ok:
        print("MISMATCH DETECTED -- stopping per spec.", flush=True)
        return 1

    payload = {
        "new": {
            (dataset, "agnostic"): submissions[dataset]["preds_agnostic"]
            for dataset in ("953", "depth")
        }
        | {
            (dataset, "classaware"): submissions[dataset]["preds_classaware"]
            for dataset in ("953", "depth")
        },
        "baseline": baseline_preds,
        "baseline_point_values": baseline_point_values,
        "point_values": {
            (dataset, "agnostic"): submissions[dataset]["AP50_agnostic"]
            for dataset in ("953", "depth")
        }
        | {
            (dataset, "classaware"): submissions[dataset]["mAP50_classaware"]
            for dataset in ("953", "depth")
        },
    }
    out_path = CACHE_DIR / "submissions.joblib"
    joblib.dump(payload, out_path, compress=3)
    print(f"\n-> wrote {out_path}", flush=True)
    print(f"ALL CHECKS PASSED. total {time.time() - t_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
