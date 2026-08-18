"""Materialisasi crop GT resolusi tinggi untuk classifier DAMIMAS.

Cache re-ID historis hanya 128x128 dan dibuat untuk dua varietas. Meng-upsample
cache itu tidak mengembalikan tekstur buah yang hilang. Skrip ini membaca citra
asli satu kali, memotong hanya anotasi DAMIMAS, lalu menyimpan tensor uint8 NPY
yang dapat di-memory-map oleh eksperimen classifier berikutnya.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


SUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import penaut_pertandan as PP  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ukuran", type=int, default=224)
    ap.add_argument("--pad", type=float, default=.15)
    ap.add_argument("--prefix-varietas", default="DAMIMAS_")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    out = args.output or SUB / "results" / f"crop_damimas_{args.ukuran}.npy"
    meta = out.with_name(out.stem + "_meta.npz")
    if out.exists() or meta.exists():
        raise FileExistsError(f"Cache sudah ada; tidak ditimpa: {out} / {meta}")

    man = PP.muat_manifest()
    ids = {s: sorted(t for t, split in man.items()
                     if split == s and t.startswith(args.prefix_varietas))
           for s in ("train", "val", "test")}
    records = []
    for split, trees in ids.items():
        for tree in trees:
            _nv, boxes = PP.muat_pohon(tree)
            for b in boxes:
                records.append((split, tree, b))
    if not records:
        raise RuntimeError("Tidak ada kotak yang cocok dengan scope varietas")

    out.parent.mkdir(parents=True, exist_ok=True)
    sementara = out.with_name(out.stem + ".partial.npy")
    mm = np.lib.format.open_memmap(
        sementara, mode="w+", dtype=np.uint8,
        shape=(len(records), args.ukuran, args.ukuran, 3))
    kunci, split_of, tree_of, bunch_of, y_of = [], [], [], [], []
    per_stem: dict[str, list[tuple[int, str, str, dict]]] = {}
    for i, (split, tree, b) in enumerate(records):
        per_stem.setdefault(b["stem"], []).append((i, split, tree, b))

    gagal = 0
    for n, (stem, rows) in enumerate(sorted(per_stem.items()), 1):
        f = PP.cari_citra(stem)
        im = cv2.imread(str(f)) if f else None
        for i, split, tree, b in rows:
            if im is None:
                crop = np.zeros((args.ukuran, args.ukuran, 3), np.uint8)
                gagal += 1
            else:
                h, w = im.shape[:2]
                x1, y1, x2, y2 = b["px"]
                dx, dy = (x2 - x1) * args.pad, (y2 - y1) * args.pad
                a1, b1 = max(0, int(x1 - dx)), max(0, int(y1 - dy))
                a2, b2 = min(w, int(x2 + dx)), min(h, int(y2 + dy))
                crop = (cv2.resize(im[b1:b2, a1:a2], (args.ukuran, args.ukuran),
                                   interpolation=cv2.INTER_AREA)
                        if a2 - a1 > 3 and b2 - b1 > 3
                        else np.zeros((args.ukuran, args.ukuran, 3), np.uint8))
            mm[i] = crop
            kunci.append((i, f"{tree}|{b['s']}|{b['i']}"))
            split_of.append((i, split)); tree_of.append((i, tree))
            bunch_of.append((i, -1 if b["bid"] is None else int(b["bid"])))
            y_of.append((i, int(b["c"])))
        if n % 250 == 0:
            mm.flush()
            print(f"  citra {n}/{len(per_stem)}", flush=True)
    mm.flush(); del mm

    # Karena iterasi dikelompokkan per stem, metadata diurutkan kembali ke indeks
    # tensor agar satu kunci selalu menunjuk crop yang tepat.
    def urut(rows):
        return np.asarray([v for _i, v in sorted(rows)], dtype=object)

    np.savez_compressed(meta, kunci=urut(kunci), split=urut(split_of),
                        tree=urut(tree_of), bunch=np.asarray(urut(bunch_of), int),
                        y=np.asarray(urut(y_of), int), ukuran=args.ukuran,
                        pad=args.pad, prefix_varietas=args.prefix_varietas)
    os.replace(sementara, out)
    manifest = out.with_name(out.stem + "_manifest.json")
    manifest.write_text(json.dumps({
        "dataset": "SawitMVC-YOLO-Damimas", "ukuran": args.ukuran,
        "pad": args.pad, "jumlah_crop": len(records), "gagal_baca": gagal,
        "pohon": {s: len(v) for s, v in ids.items()},
        "berkas": {"image": str(out), "metadata": str(meta)},
    }, indent=2, ensure_ascii=False))
    print(f"-> {out} ({len(records)} crop; gagal={gagal})")


if __name__ == "__main__":
    main()
