#!/usr/bin/env python3
"""Extract a stronger DINOv2-Large embedding for TRAIN/VAL proposal crops.

This is a validation-only feature branch.  It reads the existing RGB crop
arrays, never opens a test crop, and writes a separate feature namespace so
the established DINOv2-Base artifacts remain untouched.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


CROPS_ROOT = Path("/workspace/dino_head/crops")
FEATURES_ROOT = Path("/workspace/dino_head/features_large")
MODEL_NAME = "facebook/dinov2-large"


def load_model(device: str):
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float16)
    model.to(device).eval()
    return proc, model


@torch.no_grad()
def embed_batch(proc, model, device: str, batch_np: np.ndarray) -> np.ndarray:
    inputs = proc(images=[batch_np[i] for i in range(len(batch_np))],
                  return_tensors="pt")
    inputs = {k: v.to(device, dtype=torch.float16 if v.is_floating_point() else v.dtype)
              for k, v in inputs.items()}
    hidden = model(**inputs).last_hidden_state
    return (torch.cat([hidden[:, 0, :], hidden[:, 1:, :].mean(dim=1)], dim=1)
            .to(torch.float16).cpu().numpy())


def run(dataset: str, split: str, proc, model, device: str,
        batch_size: int, overwrite: bool) -> dict:
    rgb_path = CROPS_ROOT / dataset / f"{split}_rgb224.npy"
    if not rgb_path.exists():
        raise FileNotFoundError(rgb_path)
    rgb = np.load(rgb_path, mmap_mode="r")
    n = int(rgb.shape[0])
    dim = int(2 * model.config.hidden_size)
    out_dir = FEATURES_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}_dinolargefeat.npy"
    if out_path.exists() and not overwrite:
        existing = np.load(out_path, mmap_mode="r")
        if tuple(existing.shape) == (n, dim):
            return {"dataset": dataset, "split": split, "model": MODEL_NAME,
                    "n_crops": n, "feature_dim": dim, "out_path": str(out_path),
                    "skipped_existing": True}
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16,
                                   shape=(n, dim))
    started = time.time()
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        mm[start:end] = embed_batch(proc, model, device,
                                    np.ascontiguousarray(rgb[start:end]))
        if end == n or end % (batch_size * 20) == 0:
            print(json.dumps({"dataset": dataset, "split": split,
                              "done": end, "total": n}, ensure_ascii=False),
                  flush=True)
    mm.flush()
    elapsed = time.time() - started
    return {"dataset": dataset, "split": split, "model": MODEL_NAME,
            "n_crops": n, "feature_dim": dim, "out_path": str(out_path),
            "elapsed_sec": elapsed, "images_per_sec": n / max(elapsed, 1e-9),
            "batch_size": batch_size, "skipped_existing": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=("953", "depth"),
                    default=("953", "depth"))
    ap.add_argument("--splits", nargs="+", choices=("train", "val"),
                    default=("train", "val"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--report", type=Path,
                    default=FEATURES_ROOT / "extract_large_features_report.json")
    args = ap.parse_args()
    torch.manual_seed(42)
    proc, model = load_model(args.device)
    print(json.dumps({"model": MODEL_NAME, "device": args.device,
                      "hidden_size": int(model.config.hidden_size),
                      "batch_size": args.batch_size}), flush=True)
    reports = []
    for dataset in args.datasets:
        for split in args.splits:
            reports.append(run(dataset, split, proc, model, args.device,
                               args.batch_size, args.overwrite))
            print(json.dumps(reports[-1], ensure_ascii=False), flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"-> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
