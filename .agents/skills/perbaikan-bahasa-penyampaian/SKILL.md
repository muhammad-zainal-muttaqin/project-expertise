---
name: perbaikan-bahasa-penyampaian
description: >-
  Standardizes scientific writing, Indonesian formal grammar (EYD Edisi V / PUEBI), anti-calque translations, mathematical/statistical notations, and clear explanatory narrative structure for academic and technical documentation (.md files, reports, summaries). Activate when the user requests language improvement, rewriting in standard Indonesian, fixing calque translations, organizing chronological evidence sheets, or formatting mathematical/statistical symbols.
---

# Standar Penulisan Ilmiah Baku & Perbaikan Bahasa Penyampaian (EYD V / PUEBI)

Skill ini memuat panduan komprehensif untuk menyelaraskan, memperbaiki, dan menulis ulang dokumen teknis/ilmiah (`.md`, laporan akhir, ringkasan eksekutif, dan log penelitian) agar mematuhi standar penulisan ilmiah formal Bahasa Indonesia (EYD Edisi V / PUEBI), prinsip anti-*calque*, notasi matematika-statistika baku, dan struktur narasi yang lugas.

---

## 1. Prinsip Anti-Calque (Pencegahan Terjemahan Harfiah Mesin)

Wajib mengganti terjemahan harfiah/mesin (*calque*) yang kaku dengan padanan ilmiah formal yang lazim dan berterima:

| Terjemahan Harfiah / Calque (Hindari) | Padanan Ilmiah Baku EYD V (Wajib Digunakan) |
|---|---|
| *loss / kerugian signifikan* | **penurunan performa yang signifikan / degradasi performa** |
| *CI95 memuat nol* | **selang kepercayaan 95% mencakup nilai nol (tidak signifikan secara statistik)** |
| *tidak pernah menang / kalah* | **tidak menunjukkan keunggulan performa / mengalami penurunan** |
| *menyebut kenaikan* | **disimpulkan sebagai peningkatan / terbukti meningkatkan** |
| *appearance* | **kemunculan objek (*appearance*)** |
| *train on test* | **pelatihan pada data uji / kebocoran partisi (*train-on-test*)** |
| *holdout* | **partisi data terisolasi / himpunan uji terpisah** |
| *best observed* | **nilai terbaik yang teramati** |
| *ground truth* | **nilai acuan kebenaran (*ground truth*) / data acuan riil** |
| *oracle* | **model batas atas teoretis (*oracle*)** |
| *baseline* | **garis dasar pembanding (*baseline*) / model acuan** |
| *bounding box* | **kotak pembatas (*bounding box*)** |
| *crop* | **citra terpotong (*crop*) / pemotongan objek** |
| *counting* | **pencacahan (*counting*)** |
| *screening* | **penyaringan awal (*screening*)** |
| *early stopping* | **penghentian dini (*early stopping*)** |
| *fine-tuning* | **penyesuaian terarah (*fine-tuning*) / adaptasi model** |
| *spatial pooling* | **agregasi spasial (*spatial pooling*)** |
| *temporal shift* | **pergeseran temporal (*temporal shift*)** |
| *noise* | **variasi acak (*noise*) / derau** |

---

## 2. Notasi Matematika, Statistika, dan Angka

1. **Tanda Desimal & Pemisah Ribuan**:
   - Gunakan tanda **koma (`,`)** untuk bilangan desimal (contoh: $0,6012$; $0,4500$; $74,39\%$).
   - Gunakan tanda **titik (`.`)** untuk pemisah ribuan (contoh: $3.992\text{ citra}$; $18.540\text{ kotak}$; $1.716\text{ pohon}$).
2. **Tanda Minus Tipografis Asli**:
   - Gunakan simbol minus matematis asli `$\minus$` atau `−` (`\u2212`), **bukan** tanda hubung biasa `-` (contoh: $\minus 0,0476$; $\minus 5,1\text{ cm}$).
3. **Format Selang Kepercayaan (*Confidence Interval*)**:
   - Gunakan kurung siku dan titik koma: $[\text{min}; \text{max}]$ (contoh: CI95 $[\minus 0,0671; \minus 0,0274]$ atau $[0,7144; 0,8123]$).
4. **Simbol Variabel & Metrik Matematis**:
   - Cetak miring seluruh simbol variabel dan metrik: *$p$-value*, *$n$ sampel*, *IoU*, *$\Delta$ mAP*, *$M_{shuf}$*, *F1*, *ARI*, *MAE*, *mAP50*, *mAP50–95*, *AP50*.
5. **Rentang Satuan dan Tanggal**:
   - Gunakan tanda pisah en dash (`–`): B1–B4, Mei–Juli 2026, 10–11 Agu 2026.

---

## 3. Struktur Narasi Empat Bagian (Lembar Bukti Ilmiah)

Setiap pembahasan simpul eksperimen atau bab analisis wajib menyusun narasi ke dalam 4 bagian:

1. **Rancangan Eksperimen**: Desain komparasi, variasi arsitektur/modalitas, parameter optimasi, dan batasan kontrol.
2. **Temuan Empiris Terukur**: Penyajian kuantitatif metrik performa beserta signifikansi statistik ($p$-value, lebar CI bootstrap).
3. **Keputusan Metodologis**: Rasional keputusan ilmiah (apakah hipotesis terkonfirmasi, gugur, atau memerlukan perubahan paradigma).
4. **Batasan Validitas & Audit**: Catatan integritas silsilah data (*data lineage*), irisan pohon, dan peringatan generalisasi.

---

## 4. Prosedur Eksekusi Perbaikan Dokumen

Saat diminta melakukan standardisasi atau perbaikan dokumen:

1. **Analisis Struktur & Sasaran Dokumen**:
   - Periksa apakah berkas merupakan laporan akhir, diagnostik teknis, lembar spesifikasi, atau panduan prosedur.
2. **Audit Terminologi & Calque**:
   - Pindai kata-kata serapan/terjemahan mesin dan ubah ke istilah ilmiah baku.
3. **Konversi Notasi Angka & Rumus**:
   - Ubah seluruh desimal titik (`0.6012` $\to$ `0,6012`), minus tanda hubung (`-0.05` $\to$ `−0,05` / `$\minus 0,05$`), dan format selang kepercayaan (`[-0.01, 0.05]` $\to$ `[−0,01; +0,05]`).
4. **Validasi Tautan & Keterlacakan (Traceability)**:
   - Pastikan seluruh nama skrip, berkas log (`.log`), berkas hasil (`.json`), dan citra pendukung memiliki tautan markdown aktif yang valid (menggunakan `file:///` atau tautan relatif).
5. **Pemeriksaan Integritas Log**:
   - Jangan pernah mengubah data angka riil historis pada log eksperimen *append-only*.
