"""Gabungkan RGB + depth PNG menjadi citra 4-kanal (BGRD) untuk training.

Kontrak depth PNG: uint8, 1 kanal, 1280x800, 0=tidak ada data, 1..255=inverse depth.
Output: TIFF 4 kanal [B,G,R,D], ukuran sama dengan RGB.

Usage:
    python build_4ch_dataset.py \
        --rgb-dir /workspace/SawitMVC-Depth/images \
        --depth-dir /workspace/depth_png_352 \
        --out-dir /workspace/SawitMVC-Depth-4ch/images \
        --workers 8
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def merge_one(args: tuple[Path, Path, Path]) -> str:
    rgb_path, depth_path, out_path = args
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return f"SKIP (cannot read RGB): {rgb_path.name}"

    if depth_path.exists():
        d = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
        if d is None:
            d = np.zeros(bgr.shape[:2], np.uint8)
        elif d.shape[:2] != bgr.shape[:2]:
            d = cv2.resize(d, (bgr.shape[1], bgr.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    else:
        d = np.zeros(bgr.shape[:2], np.uint8)

    bgrd = np.dstack([bgr, d])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgrd)
    return f"OK: {rgb_path.name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-dir", type=Path, required=True)
    ap.add_argument("--depth-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rgb_files = sorted(args.rgb_dir.glob("*.jpg"))
    print(f"Found {len(rgb_files)} RGB images")

    # Gagal KERAS. Ini langkah PERTAMA rantai regenerasi (docs/REGENERASI.md);
    # sebelumnya ia menulis direktori kosong lalu exit 0, sehingga salah ketik
    # --rgb-dir baru ketahuan berjam-jam kemudian saat training.
    if not rgb_files:
        sebab = "tidak ada" if not args.rgb_dir.is_dir() else "tidak memuat satu pun .jpg"
        print(f"GAGAL: {args.rgb_dir} {sebab} — "
              f"harusnya /workspace/SawitMVC-Depth/images")
        return 2
    if not args.depth_dir.is_dir():
        print(f"GAGAL: {args.depth_dir} tidak ada — "
              f"harusnya /workspace/depth_png_352 (lihat docs/REGENERASI.md)")
        return 2

    tasks = []
    n_missing_depth = 0
    for rgb in rgb_files:
        depth = args.depth_dir / f"{rgb.stem}.png"
        out = args.out_dir / f"{rgb.stem}.tiff"
        if not depth.exists():
            n_missing_depth += 1
        tasks.append((rgb, depth, out))

    if n_missing_depth:
        print(f"WARNING: {n_missing_depth} images have no depth PNG (will use zeros)")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(merge_one, tasks, chunksize=16))

    ok = sum(1 for r in results if r.startswith("OK"))
    print(f"\nDone: {ok}/{len(tasks)} images merged to 4-channel TIFF")
    print(f"Output: {args.out_dir}")
    if ok != len(tasks):
        print(f"GAGAL: {len(tasks) - ok} citra gagal digabung")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
