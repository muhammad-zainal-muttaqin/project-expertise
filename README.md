# Project Expertise — Deteksi & Pencacahan Tandan Kelapa Sawit RGB+D

Volume 2 dari rangkaian riset deteksi tandan buah segar (TBS) kelapa sawit.
Volume 1 ([Research-Pipeline](https://github.com/muhammad-zainal-muttaqin/Research-Pipeline)) memuat tinjauan pustaka komprehensif atas 182 makalah ilmiah dan eksperimen diagnostik awal E-001 s.d. F-007. Repositori ini menjalankan eksperimen empiris terarah dengan metodologi yang lebih terukur.

## Tujuan Penelitian

Membandingkan tiga arsitektur detektor modern — **YOLO26l, RT-DETR-L, dan RF-DETR-L** — pada dataset **RGB** dan **RGB+Depth (4-kanal)**, lalu mengukur pengaruhnya terhadap **lokalisasi**, **klasifikasi tingkat kematangan (B1–B4)**, dan **pencacahan (*counting*) per pohon**.

Sejak **Fase 6**, ruang lingkup diperluas secara sistematis: tidak lagi terbatas pada komparasi arsitektur detektor satu-tahap konvensional, melainkan mengadopsi **pipeline dua-tahap modular** (lokalisasi *class-agnostic* terpisah dari klasifikasi kematangan ordinal) guna mengatasi hambatan struktural yang teridentifikasi pada analisis diagnostik (lihat [docs/DIAGNOSIS-DEPTH.md](docs/DIAGNOSIS-DEPTH.md)).

## Status Terkini

Seluruh fase eksperimen utama (`V2-E-001` s.d. `V2-E-042` serta `PT-E-000` s.d. `PT-E-036`) telah tuntas dijalankan dan diverifikasi penuh.

> [!IMPORTANT]
> **Rujukan Utama Laporan & Alur Kerja**
> - **[docs/WORKFLOW_KRONOLOGIS.md](docs/WORKFLOW_KRONOLOGIS.md)**: Rekonstruksi kronologis menyeluruh per tanggal, metrik kuantitatif, bukti visual, dan tautan log eksekusi untuk setiap simpul eksperimen.
> - **[docs/LAPORAN-AKHIR.md](docs/LAPORAN-AKHIR.md)**: Sintesis komprehensif temuan riset, analisis ancaman validitas, dan rekomendasi penerapan (*deployment*).

### Ringkasan Temuan Penutup

1. **Pergeseran Temporal (*Temporal Domain Shift*, V2-E-022)**:
   Dataset SawitMVC-YOLO (953 pohon, direkam Mei 2026) dan SawitMVC-Depth (352 pohon, direkam Juli 2026) terpisah oleh jeda waktu **$\sim 80\text{ hari}$** ($\approx 5\text{--}11$ rotasi panen). Proporsi kelas B3 menyusut dari $55,3\%$ menjadi $14,0\%$ pada pohon yang sama, sehingga perbandingan 4-kelas lintas-dataset **dinyatakan tidak valid secara ilmiah**.
2. **Keterbatasan Daya Statistik Split 352 (V2-E-023)**:
   Split uji 352 (410 kotak acuan) menghasilkan lebar selang kepercayaan $mAP50$ sebesar **$\pm 0,058$** ($0,1167$). Selisih performa antar-varian model berada di dalam rentang ketidakpastian tersebut.
3. **Efektivitas Sinyal Kedalaman (V2-E-024)**:
   Modalitas kedalaman terbukti **efektif meningkatkan lokalisasi objek** ($AP50 = \mathbf{0,7636}$ pada model 4-kanal vs $\mathbf{0,7358}$ pada kontrol RGB, $\Delta = +0,0278$, $P(\Delta > 0) = 92,1\%$), namun bersifat **redundan terhadap fitur visual RGB untuk klasifikasi kematangan**.
4. **Rekor Lokalisasi Agnostik Tertinggi (V2-E-039)**:
   Ensembel WBF 3-detektor pada korpus Combined-1716 mencetak rekor lokalisasi tertinggi sebesar **$AP50 = \mathbf{0,8106}$ ($81,06\%$)** pada 1.052 citra uji kanonik.

### Matriks Hasil Utama (Split Uji: $mAP50$ pycocotools / $\text{Class }\pm 1\text{ Acc}$ Ridge+$F_{all}$)

| Dataset Acuan | YOLO26l | RT-DETR-L | RF-DETR-L |
|---|---|---|---|
| RGB 953 Pohon | 0,5435 / 72,16% | 0,5781 / 76,24% | **0,6012** / 76,24% |
| RGB 352 Pohon | 0,3606 / 89,55% | 0,4343 / **90,91%** | **0,4544** / 88,18% |
| RGB+D 352 Pohon (*early fusion* invers) | 0,3919 / 87,73% | 0,3877 / 88,64% | 0,4186 / 88,18% |
| RGB+D 352 Pohon (Sobel `edge`, Fase 5) | **0,4316** / 87,27% | — | — |
| **Pipeline Dua-Tahap (Fase 6)** | **0,4500** / 85,91% | — | — |
| **new763 Pohon (Fase Ekspansi)** | 0,5163 | 0,5580 | **0,6129** |
| **Combined-1716 (Fase Ekspansi)** | 0,5389 | 0,5746 | **0,5960** |

### Verifikasi Remote Terbaru (V2-E-042)

Verifikasi lokal pada 27 Agustus 2026 memakai enam bobot terpilih dari bucket
Hugging Face, bukan seluruh bucket. Ringkasan lengkap, batas klaim, dan
provenans tersedia pada [laporan verifikasi remote](results/remote_eval_2026-08-27/README.md)
dan [manifest artefak](results/remote_eval_2026-08-27/MANIFEST.md).

| Bank / Test | RF-DETR-L tunggal mAP50 | WBF class-aware mAP50 | WBF agnostik AP50 | Pipeline counting MAE |
|---|---:|---:|---:|---:|
| `combined1716` / SawitMVC-Depth | **0,6711** | 0,6691 | **0,8764** | 4,52 |
| `combined1716` / SawitMVC-YOLO 953 | **0,5890** | 0,5861 | **0,8350** | 14,99 |
| `new763` / SawitMVC-Depth | **0,6125** | 0,6062 | **0,8451** | 3,28 |
| `new763` / SawitMVC-YOLO 953 | 0,1776 | 0,2018 | 0,4974 | 6,56 |

`0,8350`/`83,50%` adalah AP50 lokalisasi class-agnostic, bukan akurasi
klasifikasi B1–B4 atau akurasi pencacahan. Pipeline empat sisi masih berupa
prototipe: pada test SawitMVC-YOLO, akurasi counting tepat dan ±1 pada bank
`combined1716` masih `0%`.

Rincian per fase tersedia di [experiments/STATUS.md](experiments/STATUS.md).

---

## Navigasi Dokumen

| Dokumen | Deskripsi Isi |
|---|---|
| [docs/WORKFLOW_KRONOLOGIS.md](docs/WORKFLOW_KRONOLOGIS.md) | **Alur Kerja Kronologis & Lembar Bukti** — Rekonstruksi runut waktu seluruh simpul eksperimen, metrik, visualisasi, dan tautan log tersemat sesuai kaidah EYD V / PUEBI. |
| [docs/LAPORAN-AKHIR.md](docs/LAPORAN-AKHIR.md) | **Laporan Akhir** — Sintesis menyeluruh hasil riset, analisis ancaman validitas, dan rekomendasi penerapan. |
| [docs/DIAGNOSIS-DEPTH.md](docs/DIAGNOSIS-DEPTH.md) | **Diagnostik Sinyal Depth (Fase 6)** — Penemuan sifat fisik sinyal kedalaman (relief ordinal vs skala metrik), rasio *SNR*, dan bukti redundansi kematangan. |
| [docs/REPRODUKSI-FASE6.md](docs/REPRODUKSI-FASE6.md) | **Panduan Reproduksi** — Prosedur eksekusi langkah-demi-langkah beserta katalog 9 jebakan operasional (*silent failures*). |
| [docs/NEW763_BASELINE.md](docs/NEW763_BASELINE.md) | **Baseline Korpus 763 Pohon** — Spesifikasi rilis SawitMVC-Depth v2.0.0 dan evaluasi multi-kampanye. |
| [docs/EDA-COMBINED1716.md](docs/EDA-COMBINED1716.md) | **Analisis Eksploratif Data** — Karakteristik distribusi kelas dan sebaran spasial korpus gabungan 1.716 pohon. |
| [docs/REGENERASI.md](docs/REGENERASI.md) | Prosedur pembentukan ulang data turunan multi-kanal, citra terpotong (*crop*), dan partisi symlink. |
| [docs/REKAP.md](docs/REKAP.md) | Rekapitulasi komparasi, percobaan gagal, dan sintesis pembelajaran dari Volume 1 & Volume 2. |
| [docs/DATASET.md](docs/DATASET.md) | Spesifikasi teknis dataset SawitMVC-YOLO dan SawitMVC-Depth. |
| [docs/RENCANA.md](docs/RENCANA.md) | Rencana kerja dan metodologi per fase. |
| [experiments/EKSPERIMEN.md](experiments/EKSPERIMEN.md) | Log *append-only* per hipotesis (`V2-E-001` s.d. `V2-E-042`). |
| [pipeline-pertandan/](pipeline-pertandan/) | Subproyek mandiri asosiasi multi-tampak dan klasifikasi tingkat tandan fisik. |
| [results/](results/) | Direktori artefak metrik kuantitatif JSON, CSV, dump prediksi NPZ, dan laporan verifikasi remote. |

---

## Modul & Skrip Penelitian

### 1. Deteksi Satu-Tahap (Fase 1–5 & Ekspansi Korpus)
- `build_4ch_dataset.py`: Pembentukan dataset TIFF 4-kanal BGRD.
- `create_depth_edge_dataset.py`: Pembangkitan varian representasi depth (`edge`, `clipped`, `valid_mask`).
- `train_yolo_4ch_screening.py`: Pelatihan detektor YOLO26l generik (3- atau 4-kanal).
- `train_yolo_4ch_dropout.py`: Augmentasi *modality dropout* pada kanal kedalaman.
- `train_yolo_midfusion.py`: Implementasi cabang *mid-fusion* ber-gerbang skalar.
- `train_rfdetr_4ch.py`: Adaptasi dan pelatihan arsitektur RF-DETR-L 4-kanal.
- `eval_pycoco_*.py`, `run_counting_*.py`, `bootstrap_ci.py`: Modul evaluasi deteksi, pencacahan, dan estimasi selang kepercayaan bootstrap.

### 2. Pipeline Dua-Tahap (Fase 6)
- `probe_depth_signal.py`: Rangkaian 5 diagnostik *read-only* pengujian sifat sinyal kedalaman.
- `make_pretrain_split.py`: Pemisahan partisi pohon 953 yang bebas kebocoran terhadap himpunan validasi/uji 352.
- `make_agnostic_dataset.py`: Pembangkitan dataset deteksi lokalisasi murni 1-kelas ("tandan").
- `build_crop_dataset.py`: Ekstraksi citra terpotong (*crop*) tandan dengan mask kotak pembatas dan kanal relief kedalaman.
- `train_crop_classifier.py`: Pelatihan model pengklasifikasi kematangan ConvNeXt dengan *loss* ordinal/hybrid.
- `probe_fitur_depth.py`: Pengujian kontribusi statistik kedalaman teragregasi (*pooled depth*).
- `eval_detector_agnostic.py`: Evaluasi $AP50$ lokalisasi murni dan perakitan ensembel WBF.
- `eval_remote_pipeline_postprocess.py`: Fusi tiga detektor, kalibrasi prior rotasi, penaut empat sisi, dan evaluasi pencacahan.
- `sweep_inferensi.py`: Penelusuran kombinasi resolusi citra dan ambang NMS IoU pada split validasi.
- `eval_twostage.py`: Rekomposisi inferensi dua-tahap menuju metrik $mAP50$ deteksi kematangan.
- `run_counting_twostage.py`: Pipeline pencacahan *Ridge +* $F_{all}$ di atas estimasi dua-tahap.

### 3. Audit Validitas & Analisis Silsilah Data
- `probe_pergeseran_temporal.py`: Analisis perbandingan label citra ber-ID identik antar-tanggal akuisisi.
- `bootstrap_map.py` & `bootstrap_map_from_npz.py`: Estimasi selang kepercayaan $mAP50$ dan $AP50$ lokalisasi berpasangan tingkat citra.
- `eval_agnostic_from_npz.py`: Evaluasi lokalisasi murni *class-agnostic* langsung dari dump prediksi tanpa inferensi ulang.
- `eval_confusion_from_npz.py`: Analisis matriks konfusi bersyarat dan retensi performa lokalisasi.
- `buat_test_953_bersih.py`: Pembangunan partisi uji 953 pohon bersih (19 pohon bebas kontaminasi prapelatihan).

---

## Data Turunan & Lingkungan Eksekusi

Seluruh data turunan berukuran besar dapat diregenerasi secara deterministik mengikuti panduan di [docs/REGENERASI.md](docs/REGENERASI.md):
- `depth_png_352/`: Citra kedalaman tereproyeksi piksel-ke-piksel ke bidang koordinat kamera RGB (uint8 kanonik).
- `SawitMVC-Depth-4ch-edge/`: Dataset TIFF 4-kanal ber-encoding gradien Sobel (`edge`).
- `agnostic953/` & `agnostic352/`: Dataset anotasi 1-kelas untuk lokalisasi murni.
- `agnostic352_4ch/`: Varian 4-kanal untuk pengujian lokalisasi berbasis modalitas depth.
- `splits_fase6/`: Berkas pembagian partisi bebas kebocoran yang terlacak langsung di repositori Git.
- `requirements-freeze.txt`: Daftar 181 paket Python ter-pin untuk menjamin reprodusibilitas komputasi.

---

## Repositori Terkait

- [Research-Pipeline](https://github.com/muhammad-zainal-muttaqin/Research-Pipeline): Volume 1 — Tinjauan pustaka dan pengujian diagnostik awal.
- [Baseline-SawitMVC](https://github.com/ULM-SawitMVC/Baseline-SawitMVC): Pipa pencacahan acuan YOLO26m + *Ridge Regression*.
