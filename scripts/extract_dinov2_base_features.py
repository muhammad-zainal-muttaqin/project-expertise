"""Ekstraksi fitur DINOv2-Base pada crop yang sudah ada (untuk V2-E-046 member_head.py).

Meniru extract_large_features.py tetapi model "facebook/dinov2-base" dan
keluaran ke dino_head/features/ (bukan features_large/), sesuai path yang
dibaca member_head.py._load_fmap().
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-base"
CROPS_ROOT = Path("/workspace/dino_head/crops")
FEATURES_ROOT = Path("/workspace/dino_head/features")


def load_model(device):
    proc = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float16)
    model.to(device).eval()
    return proc, model


@torch.no_grad()
def embed_batch(proc, model, device, batch_np):
    inputs = proc(images=[batch_np[i] for i in range(len(batch_np))], return_tensors="pt")
    inputs = {k: v.to(device, dtype=torch.float16 if v.is_floating_point() else v.dtype)
              for k, v in inputs.items()}
    hidden = model(**inputs).last_hidden_state
    return (torch.cat([hidden[:, 0, :], hidden[:, 1:, :].mean(dim=1)], dim=1)
            .to(torch.float16).cpu().numpy())


def run(dataset, split, proc, model, device, batch_size=32):
    rgb_path = CROPS_ROOT / dataset / f"{split}_rgb224.npy"
    rgb = np.load(rgb_path, mmap_mode="r")
    n = int(rgb.shape[0])
    dim = int(2 * model.config.hidden_size)
    out_dir = FEATURES_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}_dinofeat.npy"
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n, dim))
    started = time.time()
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        mm[start:end] = embed_batch(proc, model, device, np.ascontiguousarray(rgb[start:end]))
        if end == n or end % (batch_size * 40) == 0:
            print(json.dumps({"dataset": dataset, "split": split, "done": end, "total": n}), flush=True)
    mm.flush()
    print(f"{dataset}/{split}: {n} crops, dim={dim}, {time.time()-started:.1f}s -> {out_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc, model = load_model(device)
    for dataset, split in [("953", "train"), ("953", "val")]:
        run(dataset, split, proc, model, device)
    print("done")
