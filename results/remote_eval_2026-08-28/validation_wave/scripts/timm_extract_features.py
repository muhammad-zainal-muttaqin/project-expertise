#!/usr/bin/env python3
"""Extract independent timm backbone features for the TRAIN/VAL head.

This is an intentionally separate visual opinion from the DINOv2 branches.
It reads only the existing RGB proposal crops, accepts only ``train`` and
``val`` splits, and writes float16 global-pooled ImageNet features.  No
linking, count, or test artifact is touched.

The pretrained weights are expected to be present in the local Hugging Face
cache.  Keeping the extractor offline makes the provenance of this wave
explicit and avoids a hidden network dependency.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


CROP_ROOT = Path("/workspace/dino_head/crops")
FEATURE_ROOT = Path("/workspace/dino_head/features_timm")

MODELS = {
    "convnext_small": "convnext_small.fb_in22k_ft_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in1k",
    "efficientnetv2_rw_s": "efficientnetv2_rw_s.ra2_in1k",
}


class CropDataset(Dataset):
    def __init__(self, path: Path, transform):
        self.array = np.load(path, mmap_mode="r")
        self.transform = transform

    def __len__(self) -> int:
        return int(self.array.shape[0])

    def __getitem__(self, index: int):
        image = Image.fromarray(np.asarray(self.array[index], dtype=np.uint8), mode="RGB")
        return self.transform(image)


def load_backbone(model_key: str, device: str):
    import timm
    from timm.data import create_transform, resolve_model_data_config

    model_name = MODELS[model_key]
    model = timm.create_model(model_name, pretrained=True, num_classes=0,
                              global_pool="avg")
    model.to(device).eval()
    transform = create_transform(**resolve_model_data_config(model),
                                 is_training=False)
    return model_name, model, transform


@torch.inference_mode()
def extract(model, loader, device: str, output_path: Path,
            batch_size: int, model_key: str, dataset: str, split: str) -> dict:
    loader_iter = iter(loader)
    first = next(loader_iter, None)
    if first is None:
        raise RuntimeError(f"empty crop loader for {dataset}/{split}")
    first = first.to(device, non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16,
                        enabled=device.startswith("cuda")):
        probe = model(first)
    probe = probe.reshape(probe.shape[0], -1)
    dim = int(probe.shape[1])
    n = len(loader.dataset)
    partial = output_path.with_name(output_path.name + ".partial.npy")
    mm = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16,
                                   shape=(n, dim))
    mm[:len(probe)] = probe.float().cpu().numpy().astype(np.float16)
    done = len(probe)
    started = time.time()
    for batch in loader_iter:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=device.startswith("cuda")):
            features = model(batch).reshape(batch.shape[0], -1)
        end = done + len(features)
        mm[done:end] = features.float().cpu().numpy().astype(np.float16)
        done = end
        if done == n or done % (batch_size * 20) == 0:
            print(json.dumps({"model": model_key, "dataset": dataset,
                              "split": split, "done": done, "total": n},
                             ensure_ascii=False), flush=True)
    if done != n:
        raise RuntimeError(f"feature count mismatch: wrote {done}, expected {n}")
    mm.flush()
    del mm
    os.replace(partial, output_path)
    elapsed = time.time() - started
    return {
        "model_key": model_key, "dataset": dataset, "split": split,
        "model": MODELS[model_key], "n_crops": n, "feature_dim": dim,
        "dtype": "float16", "output": str(output_path),
        "batch_size": batch_size, "workers": loader.num_workers,
        "elapsed_sec": elapsed, "crops_per_sec": n / max(elapsed, 1e-9),
    }


def run(model_key: str, datasets: list[str], splits: list[str],
        batch_size: int, workers: int, device: str, overwrite: bool) -> list[dict]:
    model_name, model, transform = load_backbone(model_key, device)
    print(json.dumps({"model_key": model_key, "model": model_name,
                      "device": device, "batch_size": batch_size},
                     ensure_ascii=False), flush=True)
    reports = []
    for dataset in datasets:
        for split in splits:
            crop_path = CROP_ROOT / dataset / f"{split}_rgb224.npy"
            if not crop_path.exists():
                raise FileNotFoundError(crop_path)
            out_dir = FEATURE_ROOT / model_key / dataset
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{split}_feat.npy"
            expected_n = int(np.load(crop_path, mmap_mode="r").shape[0])
            if out_path.exists() and not overwrite:
                old = np.load(out_path, mmap_mode="r")
                if old.shape[0] == expected_n:
                    reports.append({"model_key": model_key, "dataset": dataset,
                                    "split": split, "model": model_name,
                                    "n_crops": expected_n,
                                    "feature_dim": int(old.shape[1]),
                                    "output": str(out_path),
                                    "skipped_existing": True})
                    print(json.dumps(reports[-1], ensure_ascii=False), flush=True)
                    continue
            ds = CropDataset(crop_path, transform)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                num_workers=workers, pin_memory=device.startswith("cuda"),
                                persistent_workers=workers > 0,
                                prefetch_factor=2 if workers > 0 else None)
            reports.append(extract(model, loader, device, out_path, batch_size,
                                   model_key, dataset, split))
            reports[-1]["skipped_existing"] = False
            print(json.dumps(reports[-1], ensure_ascii=False), flush=True)
    return reports


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=tuple(MODELS),
                    default=list(MODELS))
    ap.add_argument("--datasets", nargs="+", choices=("953", "depth"),
                    default=("953", "depth"))
    ap.add_argument("--splits", nargs="+", choices=("train", "val"),
                    default=("train", "val"))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--report", type=Path,
                    default=FEATURE_ROOT / "extract_timm_features_report.json")
    args = ap.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(20260828)
    torch.set_float32_matmul_precision("high")
    reports = []
    for model_key in args.models:
        reports.extend(run(model_key, args.datasets, args.splits,
                           args.batch_size, args.workers, args.device,
                           args.overwrite))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    print(f"WROTE {len(reports)} feature reports -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
