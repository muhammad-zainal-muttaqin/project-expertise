"""Bangun ulang indeks crop + array rgb224 untuk cabang fitur DINOv2 (V2-E-046).

Skrip pembangun aslinya tidak pernah tersimpan di repo maupun bucket cadangan
(hanya skrip pembacanya -- extract_large_features.py, member_head.py --
yang tersimpan). Direkonstruksi sesi 2026-09-07 dengan menurunkan skema dari
kode konsumennya:
  - row_index = posisi baris pada array softvote MENTAH (belum difilter
    proposal_min), sesuai train_detection_edge_linker.make_detections().
  - stem = nama file citra (tanpa ekstensi).
  - Box dalam softvote sudah dalam skala piksel citra asli.
  - Crop konteks persegi 1.5x sisi terpanjang box, di-resize ke 224x224,
    mengikuti context_box() di extract_aux_crops.py (yang eksplisit meniru
    crop RGB utama).

Sumber box: proposal WBF softvote hasil fusi sesi ini (V2-E-045), bukan
inferensi ulang -- fusi sudah tervalidasi (F1/class4_acc cocok <0,004 dari
nilai terpublikasi).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

CONTEXT_FACTOR = 1.5
OUT_SIZE = 224
K = 4  # B1..B4

VOTE_PATHS = {
    ("953", "train"): Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27/fused_combined1716_train/SawitMVC_YOLO__wbf_softvote.npz"),
    ("953", "val"): Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27/fused_combined1716_val/SawitMVC_YOLO__wbf_softvote.npz"),
}
IMAGE_DIRS = {
    ("953", "train"): Path("/workspace/SawitMVC-YOLO/images/train"),
    ("953", "val"): Path("/workspace/SawitMVC-YOLO/images/val"),
}
OUT_ROOT = Path("/workspace/dino_head/crops")


def context_box(x1, y1, x2, y2, w, h):
    cx, cy = (x1 + x2) / 2., (y1 + y2) / 2.
    side = max(x2 - x1, y2 - y1, 1.) * CONTEXT_FACTOR
    half = side / 2.
    a, b = max(0., cx - half), max(0., cy - half)
    c, d = min(float(w), cx + half), min(float(h), cy + half)
    return a, b, max(c, a + 1.), max(d, b + 1.)


def build(dataset: str, split: str):
    vote_path = VOTE_PATHS[(dataset, split)]
    image_dir = IMAGE_DIRS[(dataset, split)]
    vote = np.load(vote_path)

    stems_out, rows_out, crops_out = [], [], []
    n_rows_total = 0
    n_images_missing = 0
    for stem in vote.files:
        rows = np.asarray(vote[stem], dtype=np.float32)
        if rows.size == 0:
            continue
        img_path = image_dir / f"{stem}.jpg"
        if not img_path.exists():
            for ext in (".jpeg", ".png"):
                alt = image_dir / f"{stem}{ext}"
                if alt.exists():
                    img_path = alt
                    break
        if not img_path.exists():
            n_images_missing += 1
            continue
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            for row_index, row in enumerate(rows):
                n_rows_total += 1
                x1, y1, x2, y2 = [float(v) for v in row[:4]]
                box = context_box(x1, y1, x2, y2, w, h)
                crop = im.crop(box).resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR)
                crops_out.append(np.asarray(crop, dtype=np.uint8))
                stems_out.append(stem)
                rows_out.append(row_index)

    out_dir = OUT_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_arr = np.stack(crops_out, axis=0) if crops_out else np.zeros((0, OUT_SIZE, OUT_SIZE, 3), np.uint8)
    np.save(out_dir / f"{split}_rgb224.npy", rgb_arr)
    np.savez(out_dir / f"{split}_index.npz",
             stem=np.asarray(stems_out, dtype=object),
             row_index=np.asarray(rows_out, dtype=np.int64))
    print(f"{dataset}/{split}: {len(crops_out)} crops dari {n_rows_total} baris "
          f"({len(vote.files)} citra, {n_images_missing} citra tidak ditemukan)")


if __name__ == "__main__":
    for dataset, split in VOTE_PATHS:
        build(dataset, split)
    print("done")
