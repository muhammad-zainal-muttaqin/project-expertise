"""Bangun dataset deteksi CLASS-AGNOSTIC 4-kanal untuk 352 pohon — `V2-E-024`.

Lubang yang ditutup: `agnostic352_4ch` adalah dataset di balik satu-satunya
temuan positif proyek (depth menaikkan lokalisasi, AP50 0,7636 vs 0,7358),
tapi dulu dibangun ad-hoc. Di seluruh repo ia cuma muncul sebagai *masukan*
(`docs/REPRODUKSI-FASE6.md`), tidak pernah sebagai keluaran skrip mana pun —
jadi hasil terpenting proyek ini bergantung pada direktori yang tidak bisa
dibuat ulang. Skrip ini menjadikannya reproducible.

Bedanya dengan `agnostic352` (RGB) hanya dua:
  - citra menunjuk ke TIFF 4-kanal, bukan jpg;
  - `data.yaml` memuat `channels: 4` supaya ultralytics meng-inflate bobot
    stem 3->4 kanal.
Split, label, dan pelipatan kelas ke 1 identik — memakai `tulis_label()` yang
sama persis dengan `make_agnostic_dataset.py`, bukan salinannya.

Usage:
    .venv/bin/python scripts/buat_agnostic352_4ch.py
    .venv/bin/python scripts/buat_agnostic352_4ch.py --out /tmp/uji --periksa
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_agnostic_dataset import tulis_label  # noqa: E402  (satu sumber kebenaran)

D352 = Path("/workspace/SawitMVC-Depth")
SPLIT352 = D352 / "splits" / "canonical_70_15_15"


def bangun(src_images: Path, akar: Path) -> dict:
    (akar / "images").mkdir(parents=True, exist_ok=True)
    (akar / "labels").mkdir(parents=True, exist_ok=True)
    (akar / "splits").mkdir(parents=True, exist_ok=True)

    ringkas, hilang = {}, []
    for sp in ("train", "val", "test"):
        jalur, n_box = [], 0
        for b in (SPLIT352 / f"{sp}.txt").read_text().splitlines():
            b = b.strip()
            if not b:
                continue
            stem = Path(b).stem
            img, lbl = src_images / f"{stem}.tiff", D352 / "labels" / f"{stem}.txt"
            if not (img.exists() and lbl.exists()):
                hilang.append(stem)
                continue
            tautan = akar / "images" / img.name
            if not tautan.exists():
                tautan.symlink_to(img)
            n_box += tulis_label(lbl, akar / "labels" / f"{stem}.txt")
            # Jalur ABSOLUT — alasan sama dengan make_agnostic_dataset.py.
            jalur.append(str(tautan))
        (akar / "splits" / f"{sp}.txt").write_text("\n".join(jalur) + "\n")
        ringkas[sp] = {"citra": len(jalur), "box": n_box}

    (akar / "data.yaml").write_text("\n".join([
        f"path: {akar}",
        *(f"{sp}: {akar}/splits/{sp}.txt" for sp in ("train", "val", "test")),
        "channels: 4",          # WAJIB: tanpa ini ultralytics memuat 3 kanal
        "nc: 1", "names:", "  0: tandan", "",
    ]))
    if hilang:
        ringkas["hilang"] = hilang
    return ringkas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-images", type=Path,
                    default=Path("/workspace/SawitMVC-Depth-4ch-edge/images"),
                    help="direktori TIFF 4-kanal (default: encoding `edge`, pemenang Fase 5)")
    ap.add_argument("--out", type=Path, default=Path("/workspace/agnostic352_4ch"))
    ap.add_argument("--periksa", type=Path, default=None,
                    help="bandingkan hasil dengan direktori acuan, lalu keluar taknol kalau beda")
    args = ap.parse_args()

    if not args.src_images.is_dir():
        print(f"GAGAL: {args.src_images} tidak ada — bangun dulu lewat "
              f"create_depth_edge_dataset.py (lihat docs/REGENERASI.md)")
        return 2

    ringkas = bangun(args.src_images, args.out)
    print(json.dumps(ringkas, indent=2))

    if args.periksa:
        beda = []
        for sp in ("train", "val", "test"):
            a = (args.out / "splits" / f"{sp}.txt").read_text().splitlines()
            b = (args.periksa / "splits" / f"{sp}.txt").read_text().splitlines()
            if len(a) != len(b):
                beda.append(f"{sp}: {len(a)} vs {len(b)} citra")
            for stem in (Path(x).stem for x in a):
                p, q = args.out / "labels" / f"{stem}.txt", args.periksa / "labels" / f"{stem}.txt"
                if not q.exists() or p.read_text() != q.read_text():
                    beda.append(f"label beda: {stem}")
                    break
        if beda:
            print("BEDA dari acuan:\n  " + "\n  ".join(beda))
            return 1
        print(f"COCOK dengan {args.periksa} — citra dan label identik")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
