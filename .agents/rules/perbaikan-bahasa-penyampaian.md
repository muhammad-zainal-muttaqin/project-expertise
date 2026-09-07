---
trigger: always_on
description: Standar Bahasa & Penulisan Ilmiah Baku (EYD V / PUEBI), Anti-Calque, Notasi Matematika, Larangan Negatif, dan Format Lembar Bukti
---

# Standar Bahasa & Penulisan Ilmiah Baku (EYD Edisi V / PUEBI)

Seluruh teks narasi, judul, kesimpulan, temuan teknis node, dan dokumentasi markdown wajib mematuhi kaidah penulisan ilmiah formal (EYD Edisi V / PUEBI):

## 1. Prinsip Anti-Calque (Pencegahan Terjemahan Harfiah Mesin)
- **Fungsi Rugi / Degradasi Performa** (bukan "loss", "rugi model", atau "kerugian signifikan").
- **Selang Kepercayaan 95% mencakup nilai nol (tidak signifikan secara statistik)** (bukan "CI95 memuat nol" atau "overlap nol").
- **Tidak menunjukkan keunggulan performa / mengalami penurunan** (bukan "kalah telak" atau "tidak pernah menang").
- **Disimpulkan sebagai peningkatan / terbukti meningkatkan** (bukan "menyebut kenaikan" atau "klaim naik").
- **Kemunculan objek (*appearance*) / fitur visual** (bukan "penampilan objek" atau "appearance").
- **Nilai acuan kebenaran (*ground truth*) / label acuan riil** (bukan "ground truth" mentah).
- **Model batas atas teoretis (*oracle*)** (bukan "oracle" mentah).
- **Garis dasar pembanding (*baseline*) / model acuan** (bukan "baseline" mentah).
- **Pencacahan (*counting*)** (bukan "counting" mentah).
- **Citra terpotong (*crop*) / pemotongan wilayah objek** (bukan "crop citra" mentah).
- **Variasi acak (*noise*) / derau sensor** (bukan "noise" mentah atau "berderau").
- **Gugur secara empiris (*falsified*) / tertolak** (bukan "dipalsukan").

## 2. Larangan Khusus (Negative Constraints / Hal yang Dilarang)
- **Larangan Antropomorfisme Model**: Dilarang menulis "model bingung", "detektor tahu", atau "model berpikir". Gunakan "terjadi konfusi representasi", "fitur tidak terbedakan", atau "koordinat spasial terdeteksi presisi".
- **Larangan Bahasa Informal**: Dilarang menggunakan kata "banget", "cuma", "lumayan", "nggak", "kayak", "jeblok/hancur", "mentok", atau "curang". Gunakan padanan formal seperti "secara signifikan", "hanya", "memadai", "mengalami penurunan performa substansial", "mencapai batas teoretis", dan "mengalami kebocoran informasi (*data leakage*)".
- **Larangan Klaim Kausalitas Palsu**: Dilarang mengklaim superioritas jika selang kepercayaan masih mencakup nilai nol.

## 3. Notasi Matematika, Statistika, dan Angka
- **Tanda Desimal & Ribuan**: Gunakan koma (`,`) untuk desimal (misal 0,6012, 74,39%) dan titik (`.`) untuk ribuan (misal 3.992 citra, 2.612 objek, 18.540 kotak).
- **Tanda Minus Matematis**: Simbol minus asli `−` (*Unicode* `U+2212`), bukan tanda hubung biasa `-` (contoh: $−0,0476$).
- **Selang Kepercayaan**: format `[min; max]` menggunakan kurung siku dan titik koma (contoh: $[−0,0671; −0,0274]$).
- **Simbol Variabel**: Cetak miring simbol matematis/variabel seperti *$p$-value*, *$n$ sampel*, *IoU*, *$\Delta$ mAP*, *$M_{shuf}$*, *F1*, *ARI*, *MAE*, *mAP50*, *mAP50–95*, *AP50*.
- **Rentang Satuan**: Gunakan en dash (`–`) untuk rentang: B1–B4, 10–11 Agu 2026.

## 4. Struktur Narasi Empat Bagian (Lembar Bukti)
1. **Rancangan Eksperimen**
2. **Temuan Empiris Terukur**
3. **Keputusan Metodologis**
4. **Batasan Validitas & Audit**
