"""Paired image-level bootstrap CI for the four locked map_boost TEST
submissions vs. their recorded baselines.

No new model decisions here: the four submissions and the four baselines are
already fixed (rebuilt bit-identically in build_submissions.py / recorded
npz). This script only quantifies sampling uncertainty of AP50 (agnostic)
and mAP50 (class-aware) via a paired bootstrap over test images, per
dataset (953: 588 stems; depth: 440 stems).

Method: for each dataset, draw N_RESAMPLES resamples of image indices with
replacement using numpy.random.RandomState(SEED). The SAME resample index
arrays are reused for all four series of that dataset (agnostic-new,
agnostic-baseline, classaware-new, classaware-baseline) -- this is what
makes the new-vs-baseline delta a *paired* bootstrap (same resampled image
multiset scores both submissions). A repeated stem is materialized as a
distinct COCO image id in both the GT and the DT list for that resample, so
pycocotools evaluates it as if it were sampled that many times. GT/DT are
built directly (no wrapper machinery around bootstrap_map*.py, per the
orchestrating task).

Read-only against /workspace/map_boost, /workspace/model_artifacts,
/workspace/project-expertise. Writes only under /workspace/ci_boot/.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, "/workspace/map_boost")
sys.path.insert(0, "/workspace/project-expertise/scripts")

import rank_and_emit as rae  # noqa: E402
import eval_new763_pycoco as enp  # noqa: E402

NAMES = enp.NAMES
K = len(NAMES)

CI_ROOT = Path("/workspace/ci_boot")
CACHE_DIR = CI_ROOT / "cache"
ARTIFACTS_DIR = CI_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DATASETS = ["953", "depth"]


# --------------------------------------------------------------------------
# GT / DT preparation (main process only)
# --------------------------------------------------------------------------
def load_gt_struct(data_root: Path):
    with redirect_stdout(io.StringIO()):
        gt, paths = enp.build_gt(data_root, "test")
    n = len(paths)
    stems = [p.stem for p in paths]
    image_wh = [None] * n
    for img in gt.dataset["images"]:
        image_wh[img["id"] - 1] = (int(img["width"]), int(img["height"]))
    gt_anns = [[] for _ in range(n)]
    for ann in gt.dataset["annotations"]:
        i = ann["image_id"] - 1
        gt_anns[i].append((int(ann["category_id"]), [float(x) for x in ann["bbox"]],
                            float(ann["area"])))
    return stems, image_wh, gt_anns


def preds_to_dt_by_idx(stems: list[str], predictions: dict, mode: str) -> list[list[tuple]]:
    n = len(stems)
    dt = [[] for _ in range(n)]
    stem_to_idx = {s: i for i, s in enumerate(stems)}
    for stem, rows in predictions.items():
        i = stem_to_idx.get(stem)
        if i is None:
            continue
        for row in rows:
            x1, y1, x2, y2, score, cls = row
            cat = 1 if mode == "agnostic" else int(cls) + 1
            dt[i].append((cat, [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                          float(score)))
    return dt


# --------------------------------------------------------------------------
# Per-resample COCO construction + evaluation (used in both main process
# for the identity-resample sanity check, and in worker processes)
# --------------------------------------------------------------------------
def build_gt_coco(idx_array: np.ndarray, image_wh: list, gt_anns: list, agnostic: bool) -> COCO:
    images, annotations = [], []
    ann_id = 1
    for k, orig in enumerate(idx_array):
        img_id = k + 1
        w, h = image_wh[orig]
        images.append({"id": img_id, "width": w, "height": h, "file_name": f"{k}.jpg"})
        for cat, bbox, area in gt_anns[orig]:
            annotations.append({
                "id": ann_id, "image_id": img_id,
                "category_id": 1 if agnostic else cat,
                "bbox": bbox, "area": area, "iscrowd": 0,
            })
            ann_id += 1
    categories = ([{"id": 1, "name": "tandan"}] if agnostic
                  else [{"id": c + 1, "name": NAMES[c]} for c in range(K)])
    gt = COCO()
    gt.dataset = {"images": images, "annotations": annotations, "categories": categories}
    with redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt


def eval_dt(gt: COCO, idx_array: np.ndarray, dt_by_idx: list) -> float:
    dt_list = []
    for k, orig in enumerate(idx_array):
        img_id = k + 1
        for cat, bbox, score in dt_by_idx[orig]:
            dt_list.append({"image_id": img_id, "category_id": cat, "bbox": bbox,
                             "score": score})
    with redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dt_list) if dt_list else gt.loadRes([])
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[1])


# --------------------------------------------------------------------------
# Worker-process globals (set once per worker via ProcessPoolExecutor
# initializer -- NOT re-pickled per task)
# --------------------------------------------------------------------------
_W: dict = {}


def _worker_init(image_wh, gt_anns, dt_map):
    _W["image_wh"] = image_wh
    _W["gt_anns"] = gt_anns
    _W["dt_map"] = dt_map


def _process_resample(args):
    resample_id, idx_array = args
    image_wh, gt_anns, dt_map = _W["image_wh"], _W["gt_anns"], _W["dt_map"]
    gt_agn = build_gt_coco(idx_array, image_wh, gt_anns, agnostic=True)
    agn_new = eval_dt(gt_agn, idx_array, dt_map["agnostic_new"])
    agn_base = eval_dt(gt_agn, idx_array, dt_map["agnostic_baseline"])
    gt_ca = build_gt_coco(idx_array, image_wh, gt_anns, agnostic=False)
    ca_new = eval_dt(gt_ca, idx_array, dt_map["classaware_new"])
    ca_base = eval_dt(gt_ca, idx_array, dt_map["classaware_baseline"])
    return resample_id, agn_new, agn_base, ca_new, ca_base


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------
def ci95(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(point_new, point_base, arr_new, arr_base, delta) -> dict:
    lo, hi = ci95(delta)
    return {
        "point_new": float(point_new),
        "point_baseline": float(point_base),
        "point_delta": float(point_new - point_base),
        "ci95_new": ci95(arr_new),
        "ci95_baseline": ci95(arr_base),
        "ci95_delta": [lo, hi],
        "delta_mean_bootstrap": float(delta.mean()),
        "frac_delta_gt_0": float((delta > 0).mean()),
        "significant_excludes_zero": bool(lo > 0 or hi < 0),
    }


# --------------------------------------------------------------------------
# Per-dataset driver
# --------------------------------------------------------------------------
def run_dataset(dataset: str, sub: dict, n_resamples: int, max_workers: int) -> dict:
    cfg = rae.cfg_for(dataset)
    data_root = cfg["data_root"]
    stems, image_wh, gt_anns = load_gt_struct(data_root)
    n = len(stems)
    print(f"[{dataset}] n_images={n}", flush=True)

    dt_map = {
        "agnostic_new": preds_to_dt_by_idx(stems, sub["new"][(dataset, "agnostic")], "agnostic"),
        "agnostic_baseline": preds_to_dt_by_idx(
            stems, sub["baseline"][(dataset, "agnostic")], "agnostic"),
        "classaware_new": preds_to_dt_by_idx(
            stems, sub["new"][(dataset, "classaware")], "classaware"),
        "classaware_baseline": preds_to_dt_by_idx(
            stems, sub["baseline"][(dataset, "classaware")], "classaware"),
    }

    # Sanity check: identity resample (each image exactly once, in order)
    # must reproduce the recorded point values via this script's independent
    # pycocotools construction path.
    identity = np.arange(n)
    gt_agn = build_gt_coco(identity, image_wh, gt_anns, agnostic=True)
    check_new_agn = eval_dt(gt_agn, identity, dt_map["agnostic_new"])
    check_base_agn = eval_dt(gt_agn, identity, dt_map["agnostic_baseline"])
    gt_ca = build_gt_coco(identity, image_wh, gt_anns, agnostic=False)
    check_new_ca = eval_dt(gt_ca, identity, dt_map["classaware_new"])
    check_base_ca = eval_dt(gt_ca, identity, dt_map["classaware_baseline"])

    expected_new_agn = sub["point_values"][(dataset, "agnostic")]
    expected_new_ca = sub["point_values"][(dataset, "classaware")]
    expected_base_agn = sub["baseline_point_values"][(dataset, "agnostic")]
    expected_base_ca = sub["baseline_point_values"][(dataset, "classaware")]

    for label, actual, expected in [
        ("agnostic_new", check_new_agn, expected_new_agn),
        ("agnostic_baseline", check_base_agn, expected_base_agn),
        ("classaware_new", check_new_ca, expected_new_ca),
        ("classaware_baseline", check_base_ca, expected_base_ca),
    ]:
        if abs(actual - expected) > 1e-6:
            raise RuntimeError(
                f"{dataset} {label}: identity-resample eval {actual!r} != "
                f"recorded point value {expected!r}")
    print(f"[{dataset}] identity-resample sanity check OK "
          f"(agn_new={check_new_agn:.6f}, agn_base={check_base_agn:.6f}, "
          f"ca_new={check_new_ca:.6f}, ca_base={check_base_ca:.6f})", flush=True)

    rs = np.random.RandomState(SEED)
    resamples = rs.randint(0, n, size=(n_resamples, n))
    tasks = [(r, resamples[r]) for r in range(n_resamples)]

    agn_new_arr = np.empty(n_resamples)
    agn_base_arr = np.empty(n_resamples)
    ca_new_arr = np.empty(n_resamples)
    ca_base_arr = np.empty(n_resamples)

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init,
                              initargs=(image_wh, gt_anns, dt_map)) as pool:
        for resample_id, agn_new, agn_base, ca_new, ca_base in pool.map(
                _process_resample, tasks, chunksize=4):
            agn_new_arr[resample_id] = agn_new
            agn_base_arr[resample_id] = agn_base
            ca_new_arr[resample_id] = ca_new
            ca_base_arr[resample_id] = ca_base
            done += 1
            if done % 50 == 0 or done == n_resamples:
                print(f"[{dataset}] {done}/{n_resamples} resamples done, "
                      f"{time.time() - t0:.1f}s elapsed", flush=True)
    dt_total = time.time() - t0
    print(f"[{dataset}] bootstrap finished in {dt_total:.1f}s", flush=True)

    delta_agn = agn_new_arr - agn_base_arr
    delta_ca = ca_new_arr - ca_base_arr

    return {
        "n_images": n,
        "n_resamples": n_resamples,
        "seed": SEED,
        "agnostic": summarize(expected_new_agn, expected_base_agn,
                               agn_new_arr, agn_base_arr, delta_agn),
        "classaware": summarize(expected_new_ca, expected_base_ca,
                                 ca_new_arr, ca_base_arr, delta_ca),
        "wall_seconds": dt_total,
        "raw": {
            "agnostic_new": agn_new_arr.tolist(),
            "agnostic_baseline": agn_base_arr.tolist(),
            "classaware_new": ca_new_arr.tolist(),
            "classaware_baseline": ca_base_arr.tolist(),
        },
    }


def json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_summary_md(results: dict, path: Path, wall_seconds_total: float) -> None:
    lines = []
    lines.append("# CI Bootstrap Summary -- map_boost TEST (paired, image-level)")
    lines.append("")
    lines.append(f"seed=42, n_resamples=500, method=paired image-level bootstrap "
                 f"(pycocotools COCOeval, per-dataset RandomState(42))")
    lines.append(f"total wall time: {wall_seconds_total:.1f}s")
    lines.append("")
    lines.append("| Dataset | Metric | Point (new) | CI95 new | Point (baseline) | "
                 "CI95 baseline | Delta (new-baseline) | CI95 delta | P(delta>0) | Significant |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for dataset in DATASETS:
        for metric_key, metric_label in (("agnostic", "AP50 agnostic"),
                                          ("classaware", "mAP50 class-aware")):
            m = results[dataset][metric_key]
            lines.append(
                f"| {dataset} | {metric_label} | {m['point_new']:.4f} | "
                f"[{m['ci95_new'][0]:.4f}, {m['ci95_new'][1]:.4f}] | "
                f"{m['point_baseline']:.4f} | "
                f"[{m['ci95_baseline'][0]:.4f}, {m['ci95_baseline'][1]:.4f}] | "
                f"{m['point_delta']:+.4f} | "
                f"[{m['ci95_delta'][0]:+.4f}, {m['ci95_delta'][1]:+.4f}] | "
                f"{m['frac_delta_gt_0']:.3f} | "
                f"{'YES' if m['significant_excludes_zero'] else 'no'} |"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-resamples", type=int, default=500)
    ap.add_argument("--max-workers", type=int, default=10)
    ap.add_argument("--out-json", type=Path, default=ARTIFACTS_DIR / "ci_test.json")
    ap.add_argument("--out-md", type=Path, default=ARTIFACTS_DIR / "CI_SUMMARY.md")
    args = ap.parse_args()

    sub = joblib.load(CACHE_DIR / "submissions.joblib")

    t_start = time.time()
    results = {}
    for dataset in DATASETS:
        print(f"\n########## BOOTSTRAP {dataset} ##########", flush=True)
        results[dataset] = run_dataset(dataset, sub, args.n_resamples, args.max_workers)
    wall_seconds_total = time.time() - t_start

    payload = {
        "generated_at_seed": SEED,
        "n_resamples": args.n_resamples,
        "max_workers": args.max_workers,
        "method": ("paired image-level bootstrap; per-dataset RandomState(42) draws "
                   "n_images resample-index arrays with replacement, shared across "
                   "agnostic/classaware and new/baseline series of that dataset; "
                   "repeated stems materialized as distinct COCO image ids with "
                   "duplicated GT/DT; pycocotools COCOeval per resample."),
        "wall_seconds_total": wall_seconds_total,
        "datasets": results,
    }
    args.out_json.write_text(json.dumps(payload, indent=2, default=json_default) + "\n")
    print(f"\n-> wrote {args.out_json}", flush=True)

    write_summary_md(results, args.out_md, wall_seconds_total)
    print(f"-> wrote {args.out_md}", flush=True)

    print(f"\nTOTAL WALL TIME: {wall_seconds_total:.1f}s", flush=True)
    for dataset in DATASETS:
        for metric_key in ("agnostic", "classaware"):
            m = results[dataset][metric_key]
            print(f"{dataset:6s} {metric_key:11s} point_new={m['point_new']:.4f} "
                  f"point_base={m['point_baseline']:.4f} delta={m['point_delta']:+.4f} "
                  f"CI95_delta=[{m['ci95_delta'][0]:+.4f}, {m['ci95_delta'][1]:+.4f}] "
                  f"P(delta>0)={m['frac_delta_gt_0']:.3f} "
                  f"significant={m['significant_excludes_zero']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
