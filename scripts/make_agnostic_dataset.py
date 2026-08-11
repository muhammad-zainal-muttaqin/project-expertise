"""Bangun dataset deteksi CLASS-AGNOSTIC (1 kelas "tandan") — Fase 6.

Kenapa: plafon mAP50 = AP50 lokalisasi. Diukur pada bobot RGB-352 yang ada,
AP50 class-agnostic = 0,6677 sementara mAP50 class-aware cuma 0,3707 — jadi
44,5% kemampuan detektor hangus karena salah kelas, bukan karena gagal
menemukan tandan. Dengan memisahkan lokalisasi dari klasifikasi:

- detektor melihat 2.299 positif untuk satu tugas, bukan terpecah jadi
  215 (B3) dan 98 (B4) yang bikin kelas langka tidak terlatih;
- kematangan ditangani classifier crop terpisah (train_crop_classifier.py),
  di mana kelas langka bisa di-oversample bebas — hal yang tidak bisa
  dilakukan detektor.

Menghasilkan dua dataset:
  agnostic953  - pretraining, 846 pohon bebas bocor terhadap val/test 352
  agnostic352  - finetune + evaluasi, split kanonik 70/15/15 yang sama
                 dengan seluruh Fase 1-5

Usage:
    .venv/bin/python scripts/make_agnostic_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

D953 = Path("/workspace/SawitMVC-YOLO")
D352 = Path("/workspace/SawitMVC-Depth")
SPLIT352 = D352 / "splits" / "canonical_70_15_15"
SPLIT6 = Path("/workspace/project-expertise/splits_fase6")
OUT = Path("/workspace")


def tulis_label(src: Path, dst: Path) -> int:
    """Salin label YOLO dengan semua id kelas dilipat jadi 0. Kelas -1 dibuang."""
    baris = []
    for ln in src.read_text().splitlines():
        p = ln.split()
        if len(p) < 5 or int(p[0]) < 0:
            continue
        baris.append(" ".join(["0"] + p[1:5]))
    dst.write_text("\n".join(baris) + ("\n" if baris else ""))
    return len(baris)


def siapkan(nama: str, pasangan: dict[str, list[tuple[Path, Path]]]) -> dict:
    akar = OUT / nama
    (akar / "images").mkdir(parents=True, exist_ok=True)
    (akar / "labels").mkdir(parents=True, exist_ok=True)
    (akar / "splits").mkdir(parents=True, exist_ok=True)

    ringkas = {}
    for sp, item in pasangan.items():
        jalur, n_box = [], 0
        for img, lbl in item:
            tautan = akar / "images" / img.name
            if not tautan.exists():
                tautan.symlink_to(img)
            n_box += tulis_label(lbl, akar / "labels" / f"{img.stem}.txt")
            jalur.append(str(tautan))
        # Jalur ABSOLUT: get_img_files() ultralytics hanya menulis ulang baris
        # berawalan "./", sisanya diresolusi terhadap CWD proses -> sumber bug.
        (akar / "splits" / f"{sp}.txt").write_text("\n".join(jalur) + "\n")
        ringkas[sp] = {"citra": len(jalur), "box": n_box}

    yaml = akar / "data.yaml"
    isi = [f"path: {akar}"]
    for sp in pasangan:
        kunci = {"train": "train", "val": "val", "test": "test"}[sp]
        isi.append(f"{kunci}: {akar}/splits/{sp}.txt")
    isi += ["nc: 1", "names:", "  0: tandan", ""]
    yaml.write_text("\n".join(isi))
    return ringkas


def main() -> int:
    # --- agnostic953: 846 pohon bebas bocor, dipecah per-POHON 90/10 -----------
    citra = [Path(b.strip()) for b in (SPLIT6 / "pretrain953_images.txt").read_text().splitlines() if b.strip()]
    pohon = sorted({f.stem.rsplit("_", 1)[0] for f in citra})
    rng = np.random.RandomState(42)
    rng.shuffle(pohon)
    val_pohon = set(pohon[:len(pohon) // 10])

    p953 = {"train": [], "val": []}
    for f in citra:
        lbl = f.parent.parent.parent / "labels" / f.parent.name / f"{f.stem}.txt"
        if not lbl.exists():
            continue
        sp = "val" if f.stem.rsplit("_", 1)[0] in val_pohon else "train"
        p953[sp].append((f, lbl))
    r953 = siapkan("agnostic953", p953)

    # --- agnostic352: split kanonik yang sama dengan Fase 1-5 -----------------
    p352 = {}
    for sp in ("train", "val", "test"):
        item = []
        for b in (SPLIT352 / f"{sp}.txt").read_text().splitlines():
            b = b.strip()
            if not b:
                continue
            stem = Path(b).stem
            img, lbl = D352 / "images" / f"{stem}.jpg", D352 / "labels" / f"{stem}.txt"
            if img.exists() and lbl.exists():
                item.append((img, lbl))
        p352[sp] = item
    r352 = siapkan("agnostic352", p352)

    # Verifikasi kebocoran: tidak boleh ada pohon val/test-352 di train-953.
    pohon953 = {Path(l).stem.rsplit("_", 1)[0]
                for l in (OUT / "agnostic953" / "splits" / "train.txt").read_text().splitlines() if l.strip()}
    terlarang = {b.strip() for f in ("val_trees.txt", "test_trees.txt")
                 for b in (SPLIT352 / f).read_text().splitlines() if b.strip()}
    bocor = pohon953 & terlarang
    assert not bocor, f"BOCOR: {len(bocor)} pohon"

    ringkas = {"agnostic953": r953, "agnostic352": r352, "irisan_bocor": 0}
    print(json.dumps(ringkas, indent=2))
    (SPLIT6 / "agnostic_ringkas.json").write_text(json.dumps(ringkas, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
