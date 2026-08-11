"""Create modified 4ch datasets with alternative depth encodings for Fase 5 screening.

Generates BGRD TIFF files where the D channel uses a different encoding:
  - 'edge': Sobel gradient magnitude of depth (aligns with F-002 finding)
  - 'inverse': Inverse depth 255/(d+1) — emphasizes near-range
  - 'clipped': Clip [0,80] → stretch to [0,255] — near-field focus
  - 'valid_mask': separates sensor holes from far-but-valid depth (Fase 5,
    2026-08-10). Kontrak input mengikuti reproject_depth.py/fourch.py:
    0 = tidak ada data (lubang sensor), 1..255 = inverse depth. Masalahnya:
    0 (invalid) bertetangga langsung dengan 1 (valid tapi paling jauh) di
    skala kontinu yang sama — conv net tidak punya sinyal eksplisit untuk
    membedakan "sensor gagal" dari "sekadar sedikit lebih jauh". Encoding ini
    memampatkan rentang valid ke [40,220], menyisakan celah numerik terhadap
    sentinel invalid di 0, supaya keduanya lebih mudah dipelajari sebagai
    kelas berbeda alih-alih titik berdekatan pada satu kontinum.

Usage:
    python create_depth_edge_dataset.py --encoding edge \
        --src /workspace/SawitMVC-Depth-4ch/images \
        --dst /workspace/SawitMVC-Depth-4ch-edge/images
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
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


def encode_valid_mask(depth: np.ndarray) -> np.ndarray:
    """Pisahkan lubang sensor (0) dari valid-terjauh (semula 1) secara numerik.

    Input mengikuti kontrak reproject_depth.py: 0=invalid, 1..255=inverse
    depth. Rentang valid dimampatkan ke [40,220], menyisakan celah terhadap
    sentinel invalid di 0 — supaya "tidak ada data" dan "jauh" tidak lagi
    hanya beda satu increment pada skala yang sama.
    """
    d = depth.astype(np.float32)
    valid = d > 0
    out = np.zeros_like(d)
    out[valid] = 40.0 + (d[valid] - 1.0) / 254.0 * 180.0
    return out.astype(np.uint8)


ENCODERS = {
    "edge": encode_edge,
    "inverse": encode_inverse,
    "clipped": encode_clipped,
    "valid_mask": encode_valid_mask,
}


def _kerja(args: tuple[str, str, str]) -> str:
    src_path, dst_path, encoding = args
    encoder = ENCODERS[encoding]
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] < 4:
        return f"SKIP (invalid shape): {Path(src_path).name}"
    img[:, :, 3] = encoder(img[:, :, 3])
    cv2.imwrite(dst_path, img)
    return f"OK: {Path(src_path).name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", choices=list(ENCODERS.keys()), required=True)
    ap.add_argument("--src", default="/workspace/SawitMVC-Depth-4ch/images")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    tiffs = sorted(src.glob("*.tiff")) + sorted(src.glob("*.tif"))
    print(f"Processing {len(tiffs)} TIFF files with encoding '{args.encoding}' ({args.workers} workers)")

    tasks = [(str(fp), str(dst / fp.name), args.encoding) for fp in tiffs]
    n_ok = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_kerja, tasks, chunksize=16)):
            n_ok += r.startswith("OK")
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(tiffs)}")

    print(f"Done: {n_ok}/{len(tiffs)} files → {dst}")


if __name__ == "__main__":
    main()
