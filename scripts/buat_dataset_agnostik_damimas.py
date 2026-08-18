"""Bangun view satu-kelas dari SawitMVC-YOLO-Damimas tanpa mengubah split.

Gambar tetap berupa symlink. Hanya label yang dimaterialisasi ulang dengan
seluruh class id menjadi 0 (``tandan``). Generator idempoten dan memverifikasi
jumlah citra/label serta keterpisahan ID pohon antar-split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SPLITS = ("train", "val", "test")


def tree_id(stem: str) -> str:
    return stem.rsplit("_", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumber", type=Path,
                    default=Path("/workspace/SawitMVC-YOLO-Damimas"))
    ap.add_argument("--keluaran", type=Path,
                    default=Path("/workspace/SawitMVC-YOLO-Damimas-Agnostic"))
    args = ap.parse_args()
    args.keluaran.mkdir(parents=True, exist_ok=True)
    statistik, sets = {}, {}
    h = hashlib.sha256()
    for split in SPLITS:
        src_img = (args.sumber / "images" / split).resolve()
        dst_img = args.keluaran / "images" / split
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        if dst_img.is_symlink():
            if dst_img.resolve() != src_img:
                raise RuntimeError(f"Symlink gambar salah: {dst_img}")
        elif dst_img.exists():
            raise FileExistsError(f"Harus symlink, tetapi sudah ada: {dst_img}")
        else:
            dst_img.symlink_to(src_img, target_is_directory=True)

        dst_lab = args.keluaran / "labels" / split
        dst_lab.mkdir(parents=True, exist_ok=True)
        paths = sorted(p for p in src_img.iterdir() if p.suffix.lower() == ".jpg")
        sets[split] = {tree_id(p.stem) for p in paths}
        nbox = 0
        for p in paths:
            src = args.sumber / "labels" / split / f"{p.stem}.txt"
            baris = []
            for line in src.read_text().splitlines():
                q = line.split()
                if not q:
                    continue
                if len(q) != 5:
                    raise ValueError(f"Label bukan bbox YOLO: {src}")
                baris.append("0 " + " ".join(q[1:]))
            teks = "\n".join(baris) + ("\n" if baris else "")
            dst = dst_lab / src.name
            if not dst.exists() or dst.read_text() != teks:
                dst.write_text(teks)
            h.update(f"{split}/{src.name}\0{teks}".encode())
            nbox += len(baris)
        lebih = {p.stem for p in dst_lab.glob("*.txt")} - {p.stem for p in paths}
        if lebih:
            raise RuntimeError(f"Label yatim di {split}: {sorted(lebih)[:3]}")
        statistik[split] = {"pohon": len(sets[split]), "citra": len(paths),
                            "kotak": nbox}

    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            if sets[a] & sets[b]:
                raise RuntimeError(f"Kebocoran pohon {a}-{b}")
    yaml = (f"path: {args.keluaran.resolve()}\n"
            "train: images/train\nval: images/val\ntest: images/test\n\n"
            "nc: 1\nnames:\n  0: tandan\n")
    (args.keluaran / "data.yaml").write_text(yaml)
    ringkas = {"sumber": str(args.sumber.resolve()),
               "dataset": "SawitMVC-YOLO-Damimas-Agnostic",
               "split": statistik, "sha256_label_terurut": h.hexdigest(),
               "catatan": "split pohon identik; citra symlink; hanya class id dilipat menjadi 0"}
    (args.keluaran / "dataset_summary.json").write_text(
        json.dumps(ringkas, indent=2, ensure_ascii=False))
    print(json.dumps(ringkas, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
