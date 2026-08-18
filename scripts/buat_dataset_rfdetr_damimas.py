"""Buat tampilan direktori DAMIMAS yang dikenali loader YOLO RF-DETR.

Dataset utama memakai tata letak Ultralytics ``images/train``. RF-DETR
mendeteksi format YOLO dari ``data.yaml`` dengan folder ``train/images`` dan
``valid/images``. Skrip ini hanya membuat symlink direktori; gambar/label tidak
disalin dan split tetap identik.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def taut_dir(sumber: Path, tujuan: Path) -> None:
    tujuan.parent.mkdir(parents=True, exist_ok=True)
    if tujuan.is_symlink():
        if tujuan.resolve() != sumber.resolve():
            raise RuntimeError(f"Symlink salah: {tujuan} -> {tujuan.resolve()}")
        return
    if tujuan.exists():
        raise FileExistsError(f"Tujuan ada tetapi bukan symlink: {tujuan}")
    tujuan.symlink_to(sumber.resolve(), target_is_directory=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumber", type=Path,
                    default=Path("/workspace/SawitMVC-YOLO-Damimas"))
    ap.add_argument("--keluaran", type=Path,
                    default=Path("/workspace/SawitMVC-YOLO-Damimas-RFDETR"))
    args = ap.parse_args()
    for rf, asli in (("train", "train"), ("valid", "val"), ("test", "test")):
        taut_dir(args.sumber / "images" / asli, args.keluaran / rf / "images")
        taut_dir(args.sumber / "labels" / asli, args.keluaran / rf / "labels")
    yaml = """path: .
train: train/images
val: valid/images
test: test/images

nc: 4
names:
  0: B1
  1: B2
  2: B3
  3: B4
"""
    (args.keluaran / "data.yaml").write_text(yaml)
    print(f"RF-DETR view -> {args.keluaran}")


if __name__ == "__main__":
    main()
