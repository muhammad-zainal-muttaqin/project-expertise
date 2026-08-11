"""Materialize direktori per-split (images/ + labels/) dari dataset 4ch flat
+ file split .txt, untuk kompatibilitas dengan eval_pycoco_rgbd352.py /
run_counting_rgbd352.py (pola lama, mengharapkan `{split}/images`,
`{split}/labels` -- bukan direktori flat + train/val/test.txt seperti yang
dipakai native ultralytics training).

Symlink saja (tidak menyalin byte) -- murah, tidak menduplikasi data di
storage network-mounted.

Usage:
    python materialize_split_dirs.py \
        --src-images /workspace/SawitMVC-Depth-4ch/images \
        --src-labels /workspace/SawitMVC-Depth/labels \
        --splits-dir /workspace/SawitMVC-Depth/splits/canonical_70_15_15_tiff \
        --out-root /workspace/SawitMVC-Depth-4ch-YOLO
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-images", type=Path, required=True)
    ap.add_argument("--src-labels", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    args = ap.parse_args()

    for split in ("train", "val", "test"):
        split_file = args.splits_dir / f"{split}.txt"
        names = [Path(line.strip()).name for line in split_file.read_text().splitlines() if line.strip()]
        img_dir = args.out_root / split / "images"
        lbl_dir = args.out_root / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for name in names:
            stem = Path(name).stem
            src_img = args.src_images / name
            src_lbl = args.src_labels / f"{stem}.txt"
            if not src_img.exists():
                print(f"WARNING: {src_img} tidak ada, dilewati")
                continue
            dst_img = img_dir / src_img.name
            if not dst_img.exists():
                dst_img.symlink_to(src_img)
            if src_lbl.exists():
                dst_lbl = lbl_dir / src_lbl.name
                if not dst_lbl.exists():
                    dst_lbl.symlink_to(src_lbl)
            n += 1
        print(f"{split}: {n}/{len(names)} citra ter-link -> {img_dir}")


if __name__ == "__main__":
    main()
