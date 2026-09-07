"""Ekstraksi fitur DINOv2-Base pada crop multi-skala (ctx100, ctx200).

Meniru extract_dinov2_base_features.py tetapi membaca dari
/workspace/multiscale/crops/{dataset}/{split}_{tag}_rgb224.npy dan menulis
ke /workspace/multiscale/features/{dataset}/{split}_{tag}_dinofeat.npy,
sesuai path yang dibaca multiscale_member_head.load_map().
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-base"
CROPS_ROOT = Path("/workspace/multiscale/crops")
FEATURES_ROOT = Path("/workspace/multiscale/features")
TAGS = ("ctx100", "ctx200")


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


def run(dataset, split, tag, proc, model, device, batch_size=32):
    rgb_path = CROPS_ROOT / dataset / f"{split}_{tag}_rgb224.npy"
    rgb = np.load(rgb_path, mmap_mode="r")
    n = int(rgb.shape[0])
    dim = int(2 * model.config.hidden_size)
    out_dir = FEATURES_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}_{tag}_dinofeat.npy"
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n, dim))
    started = time.time()
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        mm[start:end] = embed_batch(proc, model, device, np.ascontiguousarray(rgb[start:end]))
        if end == n or end % (batch_size * 60) == 0:
            print(json.dumps({"dataset": dataset, "split": split, "tag": tag, "done": end, "total": n}), flush=True)
    mm.flush()
    print(f"{dataset}/{split}/{tag}: {n} crops, {time.time()-started:.1f}s -> {out_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc, model = load_model(device)
    for dataset, split in [("953", "train"), ("953", "val")]:
        for tag in TAGS:
            run(dataset, split, tag, proc, model, device)
    print("done")
