"""Create train/VAL-only auxiliary depth crops for the multimodal branch.

The proposal index is the existing locked WBF index.  Only its boxes are
used; no labels and no test split are loaded.  Each auxiliary image is
resized to the corresponding RGB resolution before the exact same square
context crop is taken.  Output is uint8 three-channel grayscale so DINOv2
can consume it through the same image processor as RGB.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from PIL import Image


OUT_ROOT = Path("/workspace/aux_modal/crops")
INDEX_ROOT = Path("/workspace/dino_head/crops")
DATA = {
    "953": {
        "rgb": {"train": Path("/workspace/SawitMVC-YOLO/images/train"),
                "val": Path("/workspace/SawitMVC-YOLO/images/val")},
        "aux": Path("/workspace/depth_assets/mono_953"),
        "aux_layout": "flat",
    },
    "depth": {
        "rgb": {"train": Path("/workspace/SawitMVC-Depth-YOLO/train/images"),
                "val": Path("/workspace/SawitMVC-Depth-YOLO/valid/images")},
        "aux": Path("/workspace/depth_assets/aligned_depthds"),
        "aux_layout": "split",
    },
}
OUT_SIZE = 224
CONTEXT_FACTOR = 1.5


def context_box(x1, y1, x2, y2, w, h):
    cx, cy = (x1 + x2) / 2., (y1 + y2) / 2.
    side = max(x2 - x1, y2 - y1, 1.) * CONTEXT_FACTOR
    half = side / 2.
    a, b, c, d = max(0., cx - half), max(0., cy - half), min(float(w), cx + half), min(float(h), cy + half)
    return a, b, max(c, a + 1.), max(d, b + 1.)


def _normalise(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    valid = arr > 0
    if not valid.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    vals = arr[valid]
    lo, hi = np.percentile(vals, [2., 98.])
    if hi <= lo:
        lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (np.clip(arr, lo, hi) - lo) / (hi - lo) * 255.
    out[~valid] = 0.
    return out.astype(np.uint8)


def _job(job):
    try:
        aux = Image.open(job["aux_path"])
        rgb = Image.open(job["rgb_path"])
        w, h = rgb.size
        if aux.size != (w, h):
            aux = aux.resize((w, h), Image.BILINEAR)
        aux_np = np.asarray(aux)
        out = np.empty((len(job["rows"]), OUT_SIZE, OUT_SIZE, 3), dtype=np.uint8)
        for i, row in enumerate(job["rows"]):
            x1, y1, x2, y2 = [float(x) for x in row[:4]]
            box = context_box(x1, y1, x2, y2, w, h)
            crop = Image.fromarray(_normalise(np.asarray(
                aux.crop(box).resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR))))
            gray = np.asarray(crop, dtype=np.uint8)
            out[i] = np.repeat(gray[..., None], 3, axis=2)
        return {"offset": job["offset"], "out": out}
    except Exception as exc:
        return {"offset": job["offset"], "error": f"{job['aux_path']}: {exc}"}


def run(dataset: str, split: str, workers: int) -> dict:
    spec = DATA[dataset]
    idx_path = INDEX_ROOT / dataset / f"{split}_index.npz"
    with np.load(idx_path, allow_pickle=True) as z:
        stems = z["stem"].astype(str)
        rows = np.column_stack([z[k] for k in ("x1", "y1", "x2", "y2", "score",
                                                "p1", "p2", "p3", "p4")]).astype(float)
    n = len(stems)
    groups = {}
    for i, stem in enumerate(stems):
        groups.setdefault(str(stem), []).append(i)
    jobs = []
    for stem, indices in groups.items():
        aux_dir = spec["aux"] if spec["aux_layout"] == "flat" else spec["aux"] / ("valid" if split == "val" else "train")
        aux_path = aux_dir / f"{stem}.png"
        rgb_path = spec["rgb"][split] / f"{stem}.jpg"
        if not aux_path.exists() or not rgb_path.exists():
            raise FileNotFoundError(f"missing paired input: {aux_path} / {rgb_path}")
        jobs.append({"offset": indices[0], "rows": rows[indices].tolist(),
                     "aux_path": str(aux_path), "rgb_path": str(rgb_path)})
    out_dir = OUT_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}_aux224.npy"
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8,
                                   shape=(n, OUT_SIZE, OUT_SIZE, 3))
    errors = []
    t0 = time.time()
    with mp.Pool(max(1, min(workers, 8, mp.cpu_count()))) as pool:
        for result in pool.imap_unordered(_job, jobs, chunksize=2):
            if "error" in result:
                errors.append(result["error"])
            else:
                mm[result["offset"]:result["offset"] + len(result["out"])] = result["out"]
    mm.flush()
    report = {"dataset": dataset, "split": split, "n_crops": n,
              "n_stems": len(jobs), "n_errors": len(errors), "errors": errors[:10],
              "source": "mono_depth_953" if dataset == "953" else "calibrated_sensor_depth_reprojected",
              "out_path": str(out_path), "elapsed_sec": time.time() - t0}
    (out_dir / f"{split}_aux_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=tuple(DATA), default=("953", "depth"))
    ap.add_argument("--splits", nargs="+", choices=("train", "val"), default=("train", "val"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    reports = [run(ds, sp, args.workers) for ds in args.datasets for sp in args.splits]
    out = OUT_ROOT / "extract_aux_crops_report.json"
    out.write_text(json.dumps(reports, indent=2) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
