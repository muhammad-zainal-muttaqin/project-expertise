"""Temukan dan buang TIFF korup di dataset N-kanal turunan, lalu laporkan.

Kenapa skrip ini ada. Pada 2026-08-15 eval sel 6 gagal dengan "gagal membaca
...tiff". Berkasnya ada dan berukuran 8,5 MB, tapi tidak bisa didekode oleh
cv2.imread, pembaca ultralytics, maupun cv2.imdecodemulti. Pemindaian
menunjukkan kerusakannya luas: 22 dari 588 citra test dan 10 dari 404 citra val
di d953_rgbmono, plus 6 dari 980 citra train di d352_rgbmono.

Yang membuatnya berbahaya bukan kerusakannya, tapi cara ultralytics
menanganinya: citra korup DILEWATI dengan peringatan, training tetap jalan
sampai selesai. Akibatnya metrik val sel 6 selama 31 epoch dihitung atas 394
citra, sementara baseline sel 5 dihitung atas 404 — dan tidak ada satu pun
tanda di angka akhirnya. Perbandingan yang tampak sah ternyata dilakukan di
atas himpunan data yang berbeda.

Dugaan penyebab: penulisan yang terputus saat dataset dibangun. Dua tanda
tangan galat yang muncul, `TIFFReadRGBAStrip` gagal (data terpotong) dan
`TIFFGetField PHOTOMETRIC` gagal (header rusak), keduanya khas berkas yang
tidak sempat ditulis utuh.

Berkas korup DIHAPUS supaya `buat_dataset_nch.py` mau membangkitkannya ulang —
builder itu melewati berkas yang sudah ada dan tidak kosong, dan berkas korup
memenuhi kedua syarat itu, jadi tanpa penghapusan ia akan dilewati selamanya.
Yang dihapus hanya citra turunan yang bisa dibangun ulang dalam hitungan menit
(lihat docs/REGENERASI.md), BUKAN bobot. Ekstensi setiap sasaran diperiksa
sebelum dihapus, sesuai ATURAN #1 di /workspace/CLAUDE.md.

Usage:
    .venv/bin/python scripts/perbaiki_tiff_korup.py --periksa-saja
    .venv/bin/python scripts/perbaiki_tiff_korup.py --hapus
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

DATASET = [
    Path("/workspace/d953_rgbmono"),
    Path("/workspace/d352_rgbmono"),
    Path("/workspace/d352_rgbedgemono"),
]
EKS = (".tiff", ".tif")


def bisa_dibaca(jalur: str) -> tuple[str, bool, int]:
    """(jalur, terbaca, jumlah_kanal). Meniru pembaca ultralytics persis."""
    try:
        mentah = np.fromfile(jalur, np.uint8)
        ok, bingkai = cv2.imdecodemulti(mentah, cv2.IMREAD_UNCHANGED)
    except Exception:
        return jalur, False, 0
    if not ok or not bingkai:
        return jalur, False, 0
    if len(bingkai) == 1:
        b = bingkai[0]
        return jalur, True, (b.shape[2] if b.ndim == 3 else 1)
    return jalur, True, len(bingkai)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hapus", action="store_true",
                    help="hapus berkas korup (default: hanya melaporkan)")
    ap.add_argument("--periksa-saja", action="store_true")
    ap.add_argument("--proses", type=int, default=8)
    ap.add_argument("--out", default="results/tiff_korup.json")
    args = ap.parse_args()

    laporan: dict[str, dict] = {}
    total_rusak = 0

    for ds in DATASET:
        if not ds.is_dir():
            print(f"LEWAT {ds} (tidak ada)")
            continue
        laporan[ds.name] = {}
        for sp in ("train", "val", "test"):
            d = ds / "images" / sp
            if not d.is_dir():
                continue
            berkas = sorted(str(p) for p in d.iterdir() if p.suffix.lower() in EKS)
            if not berkas:
                continue
            rusak, kanal = [], {}
            with ProcessPoolExecutor(max_workers=args.proses) as ex:
                for jalur, ok, nk in ex.map(bisa_dibaca, berkas, chunksize=8):
                    if ok:
                        kanal[nk] = kanal.get(nk, 0) + 1
                    else:
                        rusak.append(jalur)
            total_rusak += len(rusak)
            laporan[ds.name][sp] = {
                "total": len(berkas), "rusak": len(rusak),
                "kanal": kanal, "berkas_rusak": [Path(x).name for x in rusak],
            }
            print(f"{ds.name}/{sp}: {len(berkas) - len(rusak)} ok, {len(rusak)} rusak, "
                  f"kanal {kanal}")
            sys.stdout.flush()

            if rusak and args.hapus and not args.periksa_saja:
                for jalur in rusak:
                    p = Path(jalur)
                    # ATURAN #1: hanya citra turunan. Berhenti total kalau ada
                    # yang bukan .tiff/.tif — jangan pernah menebak di sini.
                    if p.suffix.lower() not in EKS:
                        sys.exit(f"FATAL: {p} bukan TIFF — dibatalkan, tidak ada yang dihapus")
                    p.unlink()
                print(f"   -> {len(rusak)} berkas dihapus, siap dibangun ulang")

        # Cache label menyimpan daftar korup hasil pemindaian lama; kalau tidak
        # dibuang, ultralytics akan memakai daftar itu dan tetap melewati citra
        # yang sudah diperbaiki.
        if args.hapus and not args.periksa_saja:
            for c in (ds / "labels").glob("*.cache"):
                c.unlink()
                print(f"   cache dibuang: {c.name}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(laporan, indent=2))
    print(f"\ntotal korup: {total_rusak} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
