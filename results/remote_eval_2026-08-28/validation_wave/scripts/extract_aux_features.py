"""GPU DINOv2 embeddings for auxiliary train/VAL-only crops."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/dino_head")
from extract_features import embed_batch, load_model  # noqa: E402


def run(dataset, split, proc, model, device, name, batch_size):
    src = Path(f"/workspace/aux_modal/crops/{dataset}/{split}_aux224.npy")
    out_dir = Path(f"/workspace/aux_modal/features/{dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.load(src, mmap_mode="r")
    dim = int(model.config.hidden_size) * 2
    out = np.lib.format.open_memmap(out_dir / f"{split}_aux_dinofeat.npy", mode="w+",
                                    dtype=np.float16, shape=(len(rgb), dim))
    t0 = time.time()
    for start in range(0, len(rgb), batch_size):
        end = min(start + batch_size, len(rgb))
        out[start:end] = embed_batch(proc, model, device,
                                     np.ascontiguousarray(rgb[start:end]))
    out.flush()
    rep = {"dataset": dataset, "split": split, "model": name, "n_crops": len(rgb),
           "feature_dim": dim, "out_path": str(out.filename),
           "elapsed_sec": time.time() - t0, "images_per_sec": len(rgb) / max(time.time() - t0, 1e-9)}
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=("953", "depth"), default=("953", "depth"))
    ap.add_argument("--splits", nargs="+", choices=("train", "val"), default=("train", "val"))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(42)
    name, proc, model = load_model(args.device)
    print(f"[aux_features] using model={name} device={args.device}", flush=True)
    reports = [run(ds, sp, proc, model, args.device, name, args.batch_size)
               for ds in args.datasets for sp in args.splits]
    out = Path("/workspace/aux_modal/features/extract_aux_features_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
