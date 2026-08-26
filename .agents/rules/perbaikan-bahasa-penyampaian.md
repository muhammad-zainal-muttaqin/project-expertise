---
trigger: always_on
description: Standar Bahasa & Penulisan Ilmiah Baku (EYD V / PUEBI), Anti-Calque, Notasi Matematika, dan Format Lembar Bukti
---

# Standar Bahasa & Penulisan Ilmiah Baku (EYD Edisi V / PUEBI)

Seluruh teks narasi, judul, kesimpulan, temuan teknis node, dan dokumentasi markdown wajib mematuhi kaidah penulisan ilmiah formal (EYD Edisi V / PUEBI):

## 1. Prinsip Anti-Calque (Pencegahan Terjemahan Harfiah / Mesin)
- Gunakan "penurunan performa yang signifikan" atau "degradasi performa" (bukan "kerugian signifikan" atau "loss").
- Gunakan "selang kepercayaan 95% mencakup nilai nol (tidak signifikan secara statistik)" (bukan "CI95 memuat nol").
- Gunakan "tidak menunjukkan keunggulan performa" atau "mengalami penurunan" (bukan "tidak pernah menang" atau "kalah").
- Gunakan "disimpulkan sebagai peningkatan" atau "terbukti meningkatkan" (bukan "menyebut kenaikan").
- Gunakan "kemunculan objek (*appearance*)" (bukan "appearance" mentah).
- Gunakan "nilai acuan kebenaran (*ground truth*)" (bukan "ground truth" mentah).
- Gunakan "model batas atas teoretis (*oracle*)" (bukan "oracle" mentah).
- Gunakan "garis dasar pembanding (*baseline*)" (bukan "baseline" mentah).
- Gunakan "pencacahan (*counting*)" (bukan "counting" mentah).
- Gunakan "citra terpotong (*crop*)" (bukan "crop" mentah).
- Gunakan "variasi acak (*noise*) / derau" (bukan "noise" mentah).

## 2. Notasi Matematika, Statistika, dan Angka
- Tanda Desimal & Ribuan: koma (`,`) untuk desimal (misal 0,6012), titik (`.`) untuk ribuan (misal 3.992 citra, 2.612 objek).
- Tanda Minus Matematis: Simbol minus asli `−` (`\u2212` atau `$\minus$`), bukan tanda hubung biasa `-` (contoh: $\minus 0,0476$).
- Selang Kepercayaan: format `[min; max]` menggunakan kurung siku dan titik koma (contoh: $[−0,0270; +0,0739]$).
- Simbol Variabel: Cetak miring simbol matematis/variabel seperti *$p$-value*, *$n$ sampel*, *IoU*, *$\Delta$ mAP*, *$M_{shuf}$*, *F1*, *ARI*, *mAP50*, *AP50*.
- Rentang Satuan: Gunakan en dash (`–`) untuk rentang: B1–B4, 10–11 Agu 2026.

## 3. Struktur Narasi Empat Bagian (Lembar Bukti)
1. Rancangan Eksperimen
2. Temuan Empiris Terukur
3. Keputusan Metodologis
4. Batasan Validitas & Audit
