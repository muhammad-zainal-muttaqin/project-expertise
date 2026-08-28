#!/usr/bin/env python3
"""Build the new763 RGB+D four-channel dataset for a fair RGB control.

This builder intentionally materializes only ``train`` and ``valid``.  The
test split is not inspected, copied, or reprojection-processed by this
experiment so that the new RGB+D comparison remains validation-only until a
separate, explicitly approved test opening.

The source RGB images are the canonical new763 JPEGs.  Each output TIFF is
lossless and stores ``[B, G, R, D]`` bytes because OpenCV is the common disk
reader.  The training adapters convert only the first three channels to
``[R, G, B, D]``; depth stays channel four throughout.

Depth construction:
  raw Y16 depth (millimetres, 0 invalid)
      -> per-sidecar calibrated depth-camera to color-camera reprojection
      -> z-buffer at the native 1280x800 color grid
      -> fixed inverse-depth uint8 encoding:
           0                    = invalid/hole
           1..255               = valid, 20m..0.3m (nearer is larger)

No train/validation/test-derived normalization or threshold is used in the
encoding.  Model-side depth mean/std, if needed, is estimated from TRAIN only
and recorded by the training script.

Usage:
    python build_new763_rgbd4.py --output /workspace/new763_rgbd4 --workers 12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np


SOURCE_ROOT = Path("/workspace/SawitMVC-Depth-YOLO")
DEFAULT_OUTPUT = Path("/workspace/new763_rgbd4")
SPLITS = ("train", "valid")
RGB_W, RGB_H = 1280, 800
DEPTH_W, DEPTH_H = 848, 480
Z_NEAR_MM, Z_FAR_MM = 300.0, 20_000.0
MIN_FREE_GB = 8.0
TIFF_COMPRESSION = 8  # Deflate; lossless and materially smaller than raw TIFF.


def _load_reprojection_helpers():
    """Load the already audited calibration parser/reprojector."""
    helper_dir = Path("/workspace/depth_assets").resolve()
    if str(helper_dir) not in sys.path:
        sys.path.insert(0, str(helper_dir))
    from reproject_depth import load_depth_raw, parse_camera_param, reproject

    return load_depth_raw, parse_camera_param, reproject


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _encode_inverse_depth(depth_mm: np.ndarray) -> np.ndarray:
    """Encode millimetre depth to uint8 while reserving zero for invalid."""
    out = np.zeros(depth_mm.shape, dtype=np.uint8)
    valid = depth_mm > 0
    if not np.any(valid):
        return out
    z = depth_mm[valid].astype(np.float32)
    inv = (1.0 / z - 1.0 / Z_FAR_MM) / (1.0 / Z_NEAR_MM - 1.0 / Z_FAR_MM)
    out[valid] = (1.0 + np.rint(254.0 * np.clip(inv, 0.0, 1.0))).astype(np.uint8)
    return out


def _list_items(source_root: Path, split: str) -> list[tuple[str, Path, Path, Path, Path]]:
    """Return (stem, rgb, label, raw, sidecar) items and enforce pairing."""
    image_dir = source_root / split / "images"
    label_dir = source_root / split / "labels"
    depth_dir = source_root / split / "depth"
    if not image_dir.is_dir() or not label_dir.is_dir() or not depth_dir.is_dir():
        raise FileNotFoundError(f"incomplete source split: {source_root / split}")

    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"no JPEG images in {image_dir}")
    items = []
    for rgb in images:
        stem = rgb.stem
        label = label_dir / f"{stem}.txt"
        raw = depth_dir / f"{stem}.raw"
        sidecar = depth_dir / f"{stem}.json"
        missing = [str(p) for p in (label, raw, sidecar) if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"{split}/{stem}: missing {missing}")
        items.append((stem, rgb, label, raw, sidecar))

    # Reject silent extras in the paired modalities.  This catches accidental
    # partial exports before any output is written.
    rgb_stems = {p.stem for p in images}
    label_stems = {p.stem for p in label_dir.glob("*.txt")}
    raw_stems = {p.stem for p in depth_dir.glob("*.raw")}
    json_stems = {p.stem for p in depth_dir.glob("*.json")}
    for name, stems in (("labels", label_stems), ("raw", raw_stems), ("sidecars", json_stems)):
        extras = sorted(stems - rgb_stems)
        if extras:
            raise RuntimeError(f"{split}: {name} has {len(extras)} unpaired stems, e.g. {extras[:3]}")
    return items


def _worker_impl(task: tuple[str, str, str, str, str, str, str]) -> dict:
    split, stem, rgb_s, label_s, raw_s, sidecar_s, out_s = task
    rgb_path, label_path = Path(rgb_s), Path(label_s)
    raw_path, sidecar_path, out_path = Path(raw_s), Path(sidecar_s), Path(out_s)
    if out_path.is_file():
        return {"status": "skip", "split": split, "stem": stem, "bytes": out_path.stat().st_size}

    load_depth_raw, parse_camera_param, reproject = _load_reprojection_helpers()
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None or rgb.shape != (RGB_H, RGB_W, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"{split}/{stem}: RGB shape/dtype {None if rgb is None else (rgb.shape, rgb.dtype)}")

    with sidecar_path.open() as handle:
        meta = json.load(handle)
    dw, dh = int(meta["width"]), int(meta["height"])
    if (dw, dh) != (DEPTH_W, DEPTH_H):
        raise ValueError(f"{split}/{stem}: unexpected depth shape {(dw, dh)}")
    expected_bytes = dw * dh * 2
    if raw_path.stat().st_size != expected_bytes:
        raise ValueError(f"{split}/{stem}: raw size {raw_path.stat().st_size} != {expected_bytes}")
    dump = meta["calibration"]["cameraParamDump"]
    cp = parse_camera_param(dump)
    depth = load_depth_raw(str(raw_path), dw, dh)
    aligned_mm = reproject(depth, cp, RGB_W, RGB_H)
    depth8 = _encode_inverse_depth(aligned_mm)

    # OpenCV writes/reads the four bytes in this order.  The model adapters
    # perform the BGRD -> RGBD conversion at the framework boundary.
    rgbd_bgr = np.dstack((rgb, depth8))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + f".tmp-{os.getpid()}.tiff")
    ok = cv2.imwrite(str(tmp_path), rgbd_bgr, [cv2.IMWRITE_TIFF_COMPRESSION, TIFF_COMPRESSION])
    if not ok:
        tmp_path.unlink(missing_ok=True)
        raise IOError(f"failed to write {tmp_path}")
    os.replace(tmp_path, out_path)

    # Labels are copied by the parent process.  Returning compact statistics
    # lets the manifest describe the actual generated depth, not a guess.
    valid = aligned_mm > 0
    return {
        "status": "ok",
        "split": split,
        "stem": stem,
        "bytes": out_path.stat().st_size,
        "valid_pixels": int(valid.sum()),
        "total_pixels": int(valid.size),
        "valid_min_mm": int(aligned_mm[valid].min()) if np.any(valid) else None,
        "valid_max_mm": int(aligned_mm[valid].max()) if np.any(valid) else None,
        "source_rgb_size": rgb_path.stat().st_size,
        "source_label_size": label_path.stat().st_size,
        "calibration_sha256": hashlib.sha256(dump.encode()).hexdigest(),
    }


def _worker(task: tuple[str, str, str, str, str, str, str]) -> dict:
    """Pool-safe wrapper: report a bad frame without hiding its identity."""
    split, stem = task[:2]
    try:
        return _worker_impl(task)
    except Exception as exc:  # noqa: BLE001
        return {"status": "err", "split": split, "stem": stem, "error": repr(exc)}


def _write_yaml(output: Path) -> Path:
    path = output / "data.yaml"
    path.write_text(
        "# new763 RGB+D, validation-only construction (test intentionally absent)\n"
        f"path: {output.resolve()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "channels: 4\n"
        "nc: 4\n"
        "names:\n"
        "  0: B1\n"
        "  1: B2\n"
        "  2: B3\n"
        "  3: B4\n"
    )
    return path


def _copy_labels(items_by_split: dict[str, list[tuple[str, Path, Path, Path, Path]]], output: Path) -> None:
    for split, items in items_by_split.items():
        label_dir = output / split / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        for stem, _rgb, label, _raw, _sidecar in items:
            target = label_dir / f"{stem}.txt"
            if not target.exists():
                shutil.copy2(label, target)


def _validate_outputs(items_by_split: dict[str, list[tuple[str, Path, Path, Path, Path]]], output: Path) -> dict:
    """Re-open every output with the production TIFF reader and verify pairs."""
    from ultralytics.utils.patches import imread

    checks = {"n_images": 0, "n_labels": 0, "bad": []}
    for split, items in items_by_split.items():
        for stem, rgb_path, label_path, _raw, _sidecar in items:
            out_path = output / split / "images" / f"{stem}.tiff"
            label_out = output / split / "labels" / f"{stem}.txt"
            arr = imread(str(out_path))
            if arr is None or arr.shape != (RGB_H, RGB_W, 4) or arr.dtype != np.uint8:
                checks["bad"].append(f"{split}/{stem}: decoded={None if arr is None else (arr.shape, str(arr.dtype))}")
                continue
            source = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if not np.array_equal(arr[..., :3], source):
                checks["bad"].append(f"{split}/{stem}: RGB bytes changed in TIFF")
            if not label_out.is_file() or label_out.read_bytes() != label_path.read_bytes():
                checks["bad"].append(f"{split}/{stem}: label mismatch")
            checks["n_images"] += 1
            checks["n_labels"] += int(label_out.is_file())
    if checks["bad"]:
        raise RuntimeError("output validation failed: " + "; ".join(checks["bad"][:5]))
    return checks


def build(source: Path, output: Path, workers: int, force: bool = False) -> Path:
    if source.resolve() == output.resolve():
        raise ValueError("output must not overwrite the source dataset")
    free_gb = shutil.disk_usage(output.parent).free / (1024**3)
    if free_gb < MIN_FREE_GB:
        raise RuntimeError(f"only {free_gb:.2f} GiB free; refusing to start below {MIN_FREE_GB} GiB")
    if output.exists() and force:
        # Scope is the explicitly named generated dataset, never a broad
        # workspace path.  Normal runs are resumable and do not use force.
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    items_by_split = {split: _list_items(source, split) for split in SPLITS}
    _copy_labels(items_by_split, output)
    tasks = []
    for split, items in items_by_split.items():
        image_dir = output / split / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for stem, rgb, label, raw, sidecar in items:
            tasks.append((split, stem, str(rgb), str(label), str(raw), str(sidecar), str(image_dir / f"{stem}.tiff")))

    before = shutil.disk_usage(output.parent).free / (1024**3)
    print(f"[build] source={source} output={output} tasks={len(tasks)} free_before={before:.2f}GiB", flush=True)
    started = time.time()
    rows = []
    n_skip = 0
    resumed_by_split = {split: 0 for split in SPLITS}
    errors = []
    # Fork is efficient here because each worker re-opens only its own frame;
    # no large array is sent through IPC.
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    with ctx.Pool(max(1, workers)) as pool:
        for idx, result in enumerate(pool.imap_unordered(_worker, tasks, chunksize=2), 1):
            if result.get("status") == "skip":
                n_skip += 1
                resumed_by_split[result["split"]] += 1
            else:
                if result.get("status") == "ok":
                    rows.append(result)
            if result.get("status") == "err":
                errors.append(result)
            if idx % 50 == 0 or idx == len(tasks):
                now_free = shutil.disk_usage(output.parent).free / (1024**3)
                print(f"[build] {idx}/{len(tasks)} done; skipped={n_skip}; free={now_free:.2f}GiB", flush=True)
                if now_free < MIN_FREE_GB:
                    pool.terminate()
                    raise RuntimeError(f"free space fell below {MIN_FREE_GB} GiB")
    if errors:
        raise RuntimeError(f"{len(errors)} worker errors: {errors[:3]}")

    yaml_path = _write_yaml(output)
    checks = _validate_outputs(items_by_split, output)
    all_rows = rows
    by_split = {}
    for split in SPLITS:
        split_rows = [r for r in all_rows if r["split"] == split]
        by_split[split] = {
            "n_images": len(items_by_split[split]),
            "n_generated": sum(r["status"] == "ok" for r in split_rows),
            "n_resumed": resumed_by_split[split],
            "bytes": int(sum(r.get("bytes", 0) for r in split_rows)),
            "valid_fraction_mean": float(np.mean([r["valid_pixels"] / r["total_pixels"] for r in split_rows])) if split_rows else None,
            "valid_fraction_median": float(np.median([r["valid_pixels"] / r["total_pixels"] for r in split_rows])) if split_rows else None,
        }
    calibration_counts: dict[str, int] = {}
    for row in all_rows:
        key = row.get("calibration_sha256")
        if key:
            calibration_counts[key] = calibration_counts.get(key, 0) + 1

    manifest = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_root": str(source.resolve()),
        "output_root": str(output.resolve()),
        "dataset": "SawitMVC-Depth-YOLO v2.0.0 / new763",
        "split_unit": "tree; inherited source train/valid split",
        "splits_materialized": list(SPLITS),
        "test_policy": "test directory was intentionally not read or materialized",
        "image_shape": [RGB_H, RGB_W],
        "channels": 4,
        "disk_channel_order": "BGRD",
        "model_channel_order": "RGBD",
        "depth_source": {
            "raw_shape": [DEPTH_H, DEPTH_W],
            "unit": "millimetres",
            "invalid_raw": 0,
            "alignment": "per-image cameraParamDump; calibrated depth->color reprojection with Brown-Conrady distortion and z-buffer",
            "reprojection_helper": str((Path("/workspace/depth_assets") / "reproject_depth.py").resolve()),
            "target_grid": [RGB_H, RGB_W],
        },
        "depth_encoding": {
            "dtype": "uint8",
            "invalid_value": 0,
            "valid_values": "1..255",
            "formula": "d8=1+round(254*clip((1/z-1/z_far)/(1/z_near-1/z_far),0,1))",
            "z_near_mm": Z_NEAR_MM,
            "z_far_mm": Z_FAR_MM,
            "statistics_source": "fixed physical bounds; no split-derived statistic",
        },
        "compression": {"format": "TIFF", "codec": "Deflate", "opencv_value": TIFF_COMPRESSION, "lossless": True},
        "source_pairing": {
            "rule": "same stem across canonical RGB JPEG, YOLO label, raw depth, and JSON sidecar",
            "source_rgb_is_canonical": True,
            "source_rgb_files_untouched": True,
            "output_rgb_pixels": "decoded canonical JPEG pixels copied losslessly into the TIFF payload",
        },
        "splits": by_split,
        "calibration_variant_sha256_counts": calibration_counts,
        "output_validation": checks,
        "data_yaml": str(yaml_path.resolve()),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    manifest_path = output / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[build] validated {checks['n_images']} TIFFs; manifest={manifest_path}", flush=True)
    return manifest_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE_ROOT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--workers", type=int, default=min(12, max(1, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--force", action="store_true", help="remove only the named generated output before rebuilding")
    args = ap.parse_args()
    build(args.source, args.output, args.workers, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
