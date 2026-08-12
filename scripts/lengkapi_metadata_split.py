"""Tambahkan metadata split eksplisit ke berkas hasil yang belum menyatakannya.

Lubang yang ditutup (audit LAPORAN-AKHIR §8.1): beberapa `results/*.json`
memuat angka yang benar tetapi tidak menuliskan sendiri split-nya, sehingga
pembaca harus menelusuri `EKSPERIMEN.md` untuk tahu angka itu val atau test.

Skrip ini HANYA menambahkan kunci `_meta`. Setiap nilai yang sudah ada tidak
disentuh, dan itu diverifikasi: muatan asli di-hash sebelum dan sesudah
(dengan `_meta` dilepas kembali), lalu dibandingkan. Kalau hash berubah,
skrip berhenti dan berkas dikembalikan seperti semula.

Usage:
    .venv/bin/python scripts/lengkapi_metadata_split.py [--periksa]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"

D352 = {"dataset": "SawitMVC-Depth (352 pohon)",
        "split_kanonik": "canonical_70_15_15 (245/52/55 pohon)"}
D953 = {"dataset": "SawitMVC-YOLO (953 pohon)",
        "split_kanonik": "split_manifest.csv (716/96/141 pohon)"}

META = {
    "counting_rgb352.json": {
        **D352, "split_evaluasi": "test", "n_pohon_evaluasi": 55,
        "fit": "Ridge dipasang pada train+val, dilaporkan pada test",
        "metrik": "Class ±1 Acc, MAE, bias per kelas",
        "pipeline": "Ridge + F_all (67 dim), Baseline-SawitMVC",
        "eksperimen": "V2-E-004"},
    "counting_rgbd352.json": {
        **D352, "split_evaluasi": "test", "n_pohon_evaluasi": 55,
        "fit": "Ridge dipasang pada train+val, dilaporkan pada test",
        "metrik": "Class ±1 Acc, MAE, bias per kelas",
        "pipeline": "Ridge + F_all (67 dim), Baseline-SawitMVC",
        "eksperimen": "V2-E-006, V2-E-010 (baris `edge`)"},
    "counting_v2repro.json": {
        **D953, "split_evaluasi": "test", "n_pohon_evaluasi": 141,
        "fit": "Ridge dipasang pada train+val, dilaporkan pada test",
        "metrik": "Class ±1 Acc, MAE, bias per kelas",
        "pipeline": "Ridge + F_all (67 dim), Baseline-SawitMVC",
        "eksperimen": "V2-E-002"},
    "counting_twostage.json": {
        **D352, "split_evaluasi": "test", "n_pohon_evaluasi": 55,
        "fit": "Ridge dipasang pada train+val, dilaporkan pada test",
        "metrik": "Class ±1 Acc, macro MAE, akurasi per kelas",
        "pipeline": "Ridge + F_all (67 dim), fungsi SAMA dengan Fase 1-5",
        "eksperimen": "V2-E-021"},
    "bootstrap_ci_352.json": {
        **D352, "split_evaluasi": "test", "n_pohon_evaluasi": 55,
        "metrik": "CI 95% bootstrap untuk metrik COUNTING (bukan mAP50)",
        "metode": "resampling tingkat pohon; selisih RGB vs RGB+D berpasangan",
        "eksperimen": "V2-E-011",
        "catatan": "CI untuk mAP50 deteksi ada di bootstrap_map*.json (V2-E-023)"},
    "matrix_compiled.json": {
        "catatan": "split sudah tersirat di nama kunci (`test_mAP50`, "
                   "`test_mAP50_95`); metadata ini hanya menegaskannya",
        "split_evaluasi": "test", "kolom_dataset": "lihat kunci `dataset` tiap baris",
        "peringatan": "baris 953 dan 352 TIDAK sebanding — sesi akuisisi "
                      "terpisah ~80 hari (V2-E-022); selisih antar-baris di "
                      "dalam 352 sebagian besar di bawah derau (V2-E-023)",
        "eksperimen": "V2-E-007"},
}


def sidik(obj) -> str:
    """Hash muatan tanpa `_meta`, supaya perubahan angka apa pun ketahuan."""
    bersih = {k: v for k, v in obj.items() if k != "_meta"}
    return hashlib.sha256(
        json.dumps(bersih, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--periksa", action="store_true",
                    help="hanya laporkan, jangan tulis")
    args = ap.parse_args()

    ubah = gagal = lewat = 0
    for nama, meta in META.items():
        p = R / nama
        if not p.is_file():
            print(f"  LEWAT  {nama} — tidak ada"); lewat += 1; continue
        asli = json.loads(p.read_text())
        sebelum = sidik(asli)
        if "_meta" in asli:
            print(f"  LEWAT  {nama} — sudah punya _meta"); lewat += 1; continue
        if args.periksa:
            print(f"  PERLU  {nama}"); ubah += 1; continue

        baru = {"_meta": meta, **asli}
        p.write_text(json.dumps(baru, indent=2, ensure_ascii=False))
        sesudah = sidik(json.loads(p.read_text()))
        if sebelum != sesudah:
            p.write_text(json.dumps(asli, indent=2, ensure_ascii=False))
            print(f"  GAGAL  {nama} — muatan berubah, dikembalikan"); gagal += 1
        else:
            print(f"  OK     {nama} — _meta ditambahkan, {len(asli)} kunci asli utuh")
            ubah += 1

    print(f"\n{ubah} berkas, {gagal} gagal, {lewat} dilewati")
    return 1 if gagal else 0


if __name__ == "__main__":
    raise SystemExit(main())
