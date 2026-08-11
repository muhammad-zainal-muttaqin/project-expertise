"""Bangun daftar pohon SawitMVC-953 yang BEBAS BOCOR terhadap split 352 — Fase 6.

Dataset SawitMVC-Depth (352 pohon) adalah subset ID dari SawitMVC (953 pohon),
tapi split-nya dibuat independen: 44 dari 55 pohon test-352 justru ada di
train-953. Jadi memakai dataset 953 sebagai sumber pretraining TIDAK sah kecuali
seluruh pohon val/test-352 dibuang lebih dulu.

Motivasi Fase 6: kelas B3/B4 yang bikin metrik 352 ambruk (215 dan 98 instance)
justru melimpah di 953 (7.333 dan 2.513). Setelah 107 pohon val+test-352
dibuang, sisa pohon dipakai untuk pretraining detektor dan classifier crop.

Usage:
    .venv/bin/python scripts/make_pretrain_split.py
"""

from __future__ import annotations

import json
from pathlib import Path

D953 = Path("/workspace/SawitMVC-YOLO")
D352 = Path("/workspace/SawitMVC-Depth")
SPLIT352 = D352 / "splits" / "canonical_70_15_15"
OUT = Path("/workspace/project-expertise/splits_fase6")


def baca_pohon(p: Path) -> set[str]:
    return {b.strip() for b in p.read_text().splitlines() if b.strip()}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    terlarang = baca_pohon(SPLIT352 / "val_trees.txt") | baca_pohon(SPLIT352 / "test_trees.txt")
    print(f"pohon val+test 352 (dilarang dipakai pretraining): {len(terlarang)}")

    # Semua citra 953, lintas split aslinya — split 953 tidak relevan lagi di sini.
    semua = sorted(D953.glob("images/*/*.jpg"))
    print(f"citra 953 total: {len(semua)}")

    dipakai, dibuang = [], []
    pohon_dipakai = set()
    for f in semua:
        pohon = f.stem.rsplit("_", 1)[0]
        if pohon in terlarang:
            dibuang.append(f)
        else:
            dipakai.append(f)
            pohon_dipakai.add(pohon)

    # Verifikasi keras: nol irisan.
    bocor = pohon_dipakai & terlarang
    assert not bocor, f"BOCOR: {len(bocor)} pohon masih ada — {sorted(bocor)[:5]}"

    (OUT / "pretrain953_images.txt").write_text("\n".join(str(f) for f in dipakai) + "\n")
    (OUT / "pretrain953_trees.txt").write_text("\n".join(sorted(pohon_dipakai)) + "\n")

    meta = {
        "sumber": str(D953),
        "pohon_dilarang": len(terlarang),
        "pohon_dipakai": len(pohon_dipakai),
        "citra_dipakai": len(dipakai),
        "citra_dibuang": len(dibuang),
        "irisan_dengan_val_test_352": 0,
    }
    (OUT / "pretrain953_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
