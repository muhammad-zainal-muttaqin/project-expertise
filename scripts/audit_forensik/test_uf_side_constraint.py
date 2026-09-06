"""Uji regresi kendala sisi pada `sweep_remote_pipeline.UF` (rujukan AF-E-010).

Sebelum perbaikan, `self.sides` diisi indeks larik proposal sehingga uji irisan
sisi tidak pernah aktif dan satu klaster dapat memuat dua deteksi dari tampak
yang sama. Uji ini memaksa seluruh pasangan disatukan, lalu memastikan tidak ada
klaster yang melanggar kendala sisi maupun batas ukuran.

    python scripts/audit_forensik/test_uf_side_constraint.py
"""
import importlib.util
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
spec = importlib.util.spec_from_file_location(
    "swp", os.path.join(ROOT, "scripts", "sweep_remote_pipeline.py"))
swp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(swp)

KASUS = [
    # (sisi tiap proposal, max_size, jumlah klaster yang diharapkan)
    ([s for s in range(4) for _ in range(3)], 4, 3),   # 4 sisi x 3 proposal
    ([0, 0, 0, 1, 1, 2, 3], 3, 3),
    ([0, 1, 2, 3], 4, 1),
    ([0, 0], 2, 2),                                    # sisi sama -> tak boleh menyatu
    ([0, 1, 0, 1], 2, 2),
]


def jalankan(sides, max_size):
    uf = swp.UF(sides, max_size)
    for i in range(len(sides)):
        for j in range(i + 1, len(sides)):
            uf.union(i, j)
    grup = defaultdict(list)
    for i, s in enumerate(sides):
        grup[uf.find(i)].append(s)
    return list(grup.values())


def main() -> int:
    gagal = 0
    for sides, max_size, n_harap in KASUS:
        grup = jalankan(sides, max_size)
        langgar_sisi = [g for g in grup if len(g) != len(set(g))]
        langgar_ukuran = [g for g in grup if len(g) > max_size]
        ok = not langgar_sisi and not langgar_ukuran and len(grup) == n_harap
        gagal += not ok
        print(f"{'LULUS' if ok else 'GAGAL'}  sisi={sides} max_size={max_size} "
              f"-> klaster={len(grup)} (harap {n_harap}) "
              f"langgar_sisi={len(langgar_sisi)} langgar_ukuran={len(langgar_ukuran)}")
    print("\nSemua kasus lulus." if not gagal else f"\n{gagal} kasus gagal.")
    return 1 if gagal else 0


if __name__ == "__main__":
    raise SystemExit(main())
