"""Buat file split .txt berisi path citra ABSOLUT untuk satu varian dataset 4ch.

Alasan: `ultralytics.data.base.BaseDataset.get_img_files()` hanya menulis
ulang baris yang diawali literal "./" (diganti direktori txt file itu
sendiri); baris relatif biasa (mis. "images/X.tiff") dibiarkan apa adanya
dan berakhir diresolve terhadap CWD proses saat file dibuka -- sumber bug
yang mirip dengan yang didokumentasikan di docs/CATATAN-TEKNIS-FASE1.md
untuk `data_rgb.yaml` (`path:` relatif-ke-CWD), hanya titik kegagalannya
beda (di sini baris di DALAM txt file, bukan `path:` yaml). Path absolut
membuatnya CWD-independent sepenuhnya -- tidak perlu lagi mengatur CWD saat
menjalankan skrip training.

Usage:
    python make_absolute_split.py \
        --stems-from /workspace/SawitMVC-Depth/splits/canonical_70_15_15_tiff \
        --images-dir /workspace/SawitMVC-Depth-4ch/images \
        --out-dir /workspace/SawitMVC-Depth-4ch/splits_abs
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems-from", type=Path, required=True,
                     help="dir berisi train.txt/val.txt/test.txt (baris relatif, mis. images/X.tiff)")
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        src = args.stems_from / f"{split}.txt"
        names = [Path(line.strip()).name for line in src.read_text().splitlines() if line.strip()]
        out_lines = [str(args.images_dir / name) for name in names]
        missing = [p for p in out_lines if not Path(p).exists()]
        if missing:
            print(f"WARNING {split}: {len(missing)}/{len(out_lines)} file tidak ada, contoh: {missing[0]}")
        (args.out_dir / f"{split}.txt").write_text("\n".join(out_lines) + "\n")
        print(f"{split}: {len(out_lines)} baris -> {args.out_dir / f'{split}.txt'}")


if __name__ == "__main__":
    main()
