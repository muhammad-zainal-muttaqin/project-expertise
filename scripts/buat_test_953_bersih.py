"""Bangun split test class-agnostic 953 yang BERSIH untuk `agn953_full`.

Lubang yang ditutup: `make_agnostic_dataset.py` hanya membuat split train+val
untuk `agnostic953` (baris `p953 = {"train": [], "val": []}`), sehingga
`agn953_full` tidak pernah punya angka test sama sekali — yang dilaporkan
selama ini (0,8101) adalah val, dan angka "test-953 = 0,7374" yang sempat
dikutip berasal dari model LAIN (detektor class-aware v2repro yang
prediksinya dilipat jadi 1 kelas), bukan dari detektor agnostik ini.

Karena `pretrain953_images.txt` mengambil semua 846 pohon yang bebas bocor
terhadap val/test 352 tanpa menghormati split kanonik 953, sebagian besar
pohon test kanonik 953 ikut terpakai saat training. Yang benar-benar tak
tersentuh hanya sebagian kecil. Skrip ini mengambil tepat pohon-pohon itu.

Keluaran ada dua supaya efek kontaminasi terlihat, bukan disembunyikan:
  test_bersih.txt  - pohon test kanonik 953 yang TIDAK dipakai pretraining
  test_penuh.txt   - seluruh test kanonik 953 (sebagian besar terkontaminasi)

Usage:
    .venv/bin/python scripts/buat_test_953_bersih.py
"""

from __future__ import annotations

import json
from pathlib import Path

D953 = Path("/workspace/SawitMVC-YOLO")
AGN = Path("/workspace/agnostic953")
PRE = Path("/workspace/project-expertise/splits_fase6/pretrain953_images.txt")


def pohon(stem: str) -> str:
    return stem.rsplit("_", 1)[0]


def main() -> int:
    dipakai = {pohon(Path(l.strip()).stem)
               for l in PRE.read_text().splitlines() if l.strip()}
    kanon_test = sorted({p.stem for p in (D953 / "labels" / "test").glob("*.txt")})
    bersih = [s for s in kanon_test if pohon(s) not in dipakai]

    print(f"pohon dipakai pretraining        : {len(dipakai)}")
    print(f"citra test kanonik 953           : {len(kanon_test)} "
          f"({len({pohon(s) for s in kanon_test})} pohon)")
    print(f"citra test BERSIH (tak tersentuh): {len(bersih)} "
          f"({len({pohon(s) for s in bersih})} pohon)")

    if not bersih:
        print("FATAL: tidak ada pohon test yang bersih — angka test tidak bisa dibuat")
        return 1

    # Citra 19 pohon bersih itu justru pohon yang DIBUANG dari pretraining
    # (versi Juli-nya ada di val/test 352), jadi belum pernah di-symlink ke
    # agnostic953/. Materialisasi di direktori terpisah supaya agnostic953 asli
    # tidak berubah.
    (AGN / "splits").mkdir(parents=True, exist_ok=True)
    for nama, daftar in (("test_bersih", bersih), ("test_penuh", kanon_test)):
        akar = Path(f"/workspace/agnostic953_{nama}")
        (akar / "images").mkdir(parents=True, exist_ok=True)
        (akar / "labels").mkdir(parents=True, exist_ok=True)
        jalur = []
        for s in daftar:
            src_i = D953 / "images" / "test" / f"{s}.jpg"
            src_l = D953 / "labels" / "test" / f"{s}.txt"
            if not src_i.is_file() or not src_l.is_file():
                continue
            dst_i = akar / "images" / f"{s}.jpg"
            if not dst_i.exists():
                dst_i.symlink_to(src_i)
            # lipat 4 kelas -> 1 kelas "tandan"
            baris = []
            for ln in src_l.read_text().splitlines():
                q = ln.split()
                if len(q) >= 5 and int(q[0]) >= 0:
                    baris.append("0 " + " ".join(q[1:5]))
            (akar / "labels" / f"{s}.txt").write_text("\n".join(baris) + "\n")
            jalur.append(str(dst_i))
        (akar / "splits").mkdir(exist_ok=True)
        (akar / "splits" / "test.txt").write_text("\n".join(jalur) + "\n")
        (akar / "data.yaml").write_text(
            f"path: {akar}\ntrain: {akar}/splits/test.txt\n"
            f"val: {akar}/splits/test.txt\nnc: 1\nnames:\n  0: tandan\n")
        n_kotak = sum(len((akar / "labels" / f"{Path(j).stem}.txt").read_text().split("\n")) - 1
                      for j in jalur)
        print(f"  {nama}: {len(jalur)} citra, ~{n_kotak} kotak -> {akar}")

    # data.yaml terpisah supaya data.yaml asli tidak diubah (append-only)
    for nama in ("test_bersih", "test_penuh"):
        (AGN / f"data_{nama}.yaml").write_text(
            f"path: {AGN}\n"
            f"train: {AGN}/splits/train.txt\n"
            f"val: {AGN}/splits/{nama}.txt\n"
            "nc: 1\nnames:\n  0: tandan\n")
    meta = {
        "pohon_pretraining": len(dipakai),
        "test_kanonik_953_citra": len(kanon_test),
        "test_bersih_citra": len(bersih),
        "test_bersih_pohon": len({pohon(s) for s in bersih}),
        "peringatan": "test_bersih kecil; CI-nya akan lebar dan wajib dilaporkan "
                      "bersama angkanya",
    }
    Path("/workspace/project-expertise/results/test953_bersih.json").write_text(
        json.dumps(meta, indent=2))
    print("\n-> agnostic953/data_test_bersih.yaml dan data_test_penuh.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
