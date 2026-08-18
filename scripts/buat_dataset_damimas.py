"""Bangun turunan YOLO yang hanya memuat varietas DAMIMAS.

Split tidak dibuat ulang. Pohon DAMIMAS mempertahankan ``new_split`` kanonik
dari ``split_manifest.csv`` agar tidak ada kebocoran antar-split dan hasilnya
tetap dapat dibandingkan dengan eksperimen SawitMVC 953 sebelumnya.

Data citra dan label ditautkan dengan symlink; tidak ada byte dataset yang
digandakan. Skrip idempotent dan tidak menghapus isi direktori keluaran.

Pemakaian:
    python scripts/buat_dataset_damimas.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


SPLITS = ("train", "val", "test")


def baca_manifest(path: Path, varietas: str) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [r for r in reader if r["variety"].upper() == varietas.upper()]
    if not rows:
        raise RuntimeError(f"Tidak ada varietas {varietas!r} di {path}")
    return fieldnames, rows


def tree_id_dari_stem(stem: str) -> str:
    """Nama citra berakhir ``_<nomor-sisi>``; sisanya adalah ID pohon."""
    tree_id, sep, sisi = stem.rpartition("_")
    if not sep or not sisi.isdigit():
        raise ValueError(f"Nama citra tidak mengikuti pola <tree>_<sisi>: {stem}")
    return tree_id


def tautkan(src: Path, dst: Path) -> None:
    if dst.is_symlink():
        if dst.resolve() != src.resolve():
            raise RuntimeError(f"Symlink tujuan mengarah ke sumber lain: {dst}")
        return
    if dst.exists():
        raise FileExistsError(f"Tujuan sudah ada dan bukan symlink: {dst}")
    dst.symlink_to(src.resolve())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumber", type=Path, default=Path("/workspace/SawitMVC-YOLO"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("/workspace/SawitMVC/split_manifest.csv"))
    ap.add_argument("--keluaran", type=Path,
                    default=Path("/workspace/SawitMVC-YOLO-Damimas"))
    ap.add_argument("--varietas", default="DAMIMAS")
    args = ap.parse_args()

    fieldnames, rows = baca_manifest(args.manifest, args.varietas)
    split_tree = {s: {r["tree_id"] for r in rows if r["new_split"] == s}
                  for s in SPLITS}
    if any(split_tree[a] & split_tree[b]
           for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]):
        raise RuntimeError("Ada ID pohon yang muncul di lebih dari satu split")

    statistik: dict[str, dict[str, object]] = {}
    for split in SPLITS:
        dst_img = args.keluaran / "images" / split
        dst_lbl = args.keluaran / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        citra = []
        for src in sorted((args.sumber / "images" / split).iterdir()):
            if src.is_dir() or tree_id_dari_stem(src.stem) not in split_tree[split]:
                continue
            citra.append(src)
            tautkan(src, dst_img / src.name)
            label = args.sumber / "labels" / split / f"{src.stem}.txt"
            if not label.exists():
                raise FileNotFoundError(f"Label tidak ditemukan: {label}")
            tautkan(label, dst_lbl / label.name)

        tree_terlihat = {tree_id_dari_stem(p.stem) for p in citra}
        hilang = split_tree[split] - tree_terlihat
        if hilang:
            raise RuntimeError(f"{split}: {len(hilang)} pohon tidak punya citra: {sorted(hilang)[:3]}")

        kelas: Counter[int] = Counter()
        kosong = 0
        for src in citra:
            label = args.sumber / "labels" / split / f"{src.stem}.txt"
            baris = [x for x in label.read_text().splitlines() if x.strip()]
            kosong += not baris
            for line in baris:
                kelas[int(float(line.split()[0]))] += 1
        statistik[split] = {
            "pohon": len(tree_terlihat), "citra": len(citra),
            "box": sum(kelas.values()), "citra_tanpa_box": kosong,
            "kelas": dict(sorted(kelas.items())),
        }

    args.keluaran.mkdir(parents=True, exist_ok=True)
    with (args.keluaran / "split_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    yaml = (
        f"path: {args.keluaran.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "nc: 4\n"
        "names:\n"
        "  0: B1\n"
        "  1: B2\n"
        "  2: B3\n"
        "  3: B4\n"
    )
    (args.keluaran / "data.yaml").write_text(yaml)

    ringkasan = {
        "nama": f"SawitMVC-YOLO-{args.varietas.title()}",
        "varietas": args.varietas.upper(),
        "sumber": str(args.sumber.resolve()),
        "manifest_sumber": str(args.manifest.resolve()),
        "sha256_manifest_sumber": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "split": statistik,
        "catatan": ("split pohon dipertahankan dari new_split kanonik; citra dan "
                     "label adalah symlink read-only secara semantik ke dataset sumber"),
    }
    (args.keluaran / "dataset_summary.json").write_text(
        json.dumps(ringkasan, indent=2, ensure_ascii=False))

    if any(not p.name.upper().startswith(args.varietas.upper() + "_")
           for s in SPLITS for p in (args.keluaran / "images" / s).iterdir()):
        raise RuntimeError("Verifikasi gagal: keluaran memuat varietas lain")

    print(f"Dataset {args.varietas} -> {args.keluaran}")
    for split in SPLITS:
        print(f"  {split}: {statistik[split]}")


if __name__ == "__main__":
    main()
