"""Bangun crop multi-skala (ctx100, ctx200) untuk cabang multiscale_member_head.py.

Sama seperti build_dino_crop_index_v2e046.py tetapi context_factor berbeda
(1.0 dan 2.0, bukan 1.5), memakai indeks (stem, row_index) yang SAMA persis
(index.npz sudah dibangun) supaya urutan crop identik dan bisa langsung
dipetakan lewat (stem, row_index) di multiscale_member_head.load_map().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

OUT_SIZE = 224
K = 4

VOTE_PATHS = {
    ("953", "train"): Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27/fused_combined1716_train/SawitMVC_YOLO__wbf_softvote.npz"),
    ("953", "val"): Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27/fused_combined1716_val/SawitMVC_YOLO__wbf_softvote.npz"),
}
IMAGE_DIRS = {
    ("953", "train"): Path("/workspace/SawitMVC-YOLO/images/train"),
    ("953", "val"): Path("/workspace/SawitMVC-YOLO/images/val"),
}
OUT_ROOT = Path("/workspace/multiscale/features")  # placeholder overwritten below
CROPS_STAGE_ROOT = Path("/workspace/multiscale/crops")


def context_box(x1, y1, x2, y2, w, h, context_factor):
    cx, cy = (x1 + x2) / 2., (y1 + y2) / 2.
    side = max(x2 - x1, y2 - y1, 1.) * context_factor
    half = side / 2.
    a, b = max(0., cx - half), max(0., cy - half)
    c, d = min(float(w), cx + half), min(float(h), cy + half)
    return a, b, max(c, a + 1.), max(d, b + 1.)


def build(dataset, split, tag, context_factor):
    vote_path = VOTE_PATHS[(dataset, split)]
    image_dir = IMAGE_DIRS[(dataset, split)]
    vote = np.load(vote_path)

    crops_out = []
    for stem in vote.files:
        rows = np.asarray(vote[stem], dtype=np.float32)
        if rows.size == 0:
            continue
        img_path = image_dir / f"{stem}.jpg"
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            for row_index, row in enumerate(rows):
                x1, y1, x2, y2 = [float(v) for v in row[:4]]
                box = context_box(x1, y1, x2, y2, w, h, context_factor)
                crop = im.crop(box).resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR)
                crops_out.append(np.asarray(crop, dtype=np.uint8))

    out_dir = CROPS_STAGE_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_arr = np.stack(crops_out, axis=0) if crops_out else np.zeros((0, OUT_SIZE, OUT_SIZE, 3), np.uint8)
    np.save(out_dir / f"{split}_{tag}_rgb224.npy", rgb_arr)
    print(f"{dataset}/{split}/{tag}: {len(crops_out)} crops (context_factor={context_factor})")


if __name__ == "__main__":
    for dataset, split in VOTE_PATHS:
        build(dataset, split, "ctx100", 1.0)
        build(dataset, split, "ctx200", 2.0)
    print("done")
