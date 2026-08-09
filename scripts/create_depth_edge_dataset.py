"""Create modified 4ch datasets with alternative depth encodings for Fase 5 screening.

Generates BGRD TIFF files where the D channel uses a different encoding:
  - 'edge': Sobel gradient magnitude of depth (aligns with F-002 finding)
  - 'inverse': Inverse depth 255/(d+1) — emphasizes near-range
  - 'clipped': Clip [0,80] → stretch to [0,255] — near-field focus

Usage:
    python create_depth_edge_dataset.py --encoding edge \
        --src /workspace/SawitMVC-Depth-4ch/images \
        --dst /workspace/SawitMVC-Depth-4ch-edge/images
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np


def encode_edge(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    sx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    mag = np.clip(mag / mag.max() * 255, 0, 255).astype(np.uint8) if mag.max() > 0 else mag.astype(np.uint8)
    return mag


def encode_inverse(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    inv = 255.0 / (d + 1.0)
    return np.clip(inv, 0, 255).astype(np.uint8)


def encode_clipped(depth: np.ndarray, clip_max: int = 80) -> np.ndarray:
    d = depth.astype(np.float32)
    d = np.clip(d, 0, clip_max)
    d = d / clip_max * 255.0
    return d.astype(np.uint8)


ENCODERS = {
    "edge": encode_edge,
    "inverse": encode_inverse,
    "clipped": encode_clipped,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", choices=list(ENCODERS.keys()), required=True)
    ap.add_argument("--src", default="/workspace/SawitMVC-Depth-4ch/images")
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    encoder = ENCODERS[args.encoding]

    tiffs = sorted(src.glob("*.tiff")) + sorted(src.glob("*.tif"))
    print(f"Processing {len(tiffs)} TIFF files with encoding '{args.encoding}'")

    for i, fp in enumerate(tiffs):
        img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3 or img.shape[2] < 4:
            print(f"SKIP {fp.name}: invalid shape")
            continue
        depth = img[:, :, 3]
        new_d = encoder(depth)
        img[:, :, 3] = new_d
        out = dst / fp.name
        cv2.imwrite(str(out), img)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(tiffs)}")

    print(f"Done: {len(tiffs)} files → {dst}")


if __name__ == "__main__":
    main()
