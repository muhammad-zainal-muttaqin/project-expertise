# Project Expertise — Deteksi & Counting Tandan Sawit RGB+D

Volume 2 dari riset deteksi tandan buah segar (TBS) kelapa sawit.
Volume 1 ([Research-Pipeline](https://github.com/muhammad-zainal-muttaqin/Research-Pipeline))
berisi tinjauan pustaka 182 makalah dan eksperimen diagnostik E-001 s.d. F-007.
Repo ini memulai eksperimen baru dengan tujuan yang lebih tajam.

## Tujuan

Membandingkan tiga arsitektur detektor — **YOLO26l, RT-DETR-L, RF-DETR-L** —
pada dataset **RGB** dan **RGB+Depth (4-kanal)**, lalu mengukur dampaknya
terhadap **deteksi**, **klasifikasi kematangan (B1–B4)**, dan **counting
per pohon**.

### Target

| # | Target | Kriteria |
|---|---|---|
| 1 | **Tidak ada regresi** | RGB+D &ge; RGB pada setiap pasangan arsitektur |
| 2 | **Peningkatan terukur** | RGB+D > RGB secara signifikan, idealnya mAP50 &ge; 0,70 |

### Matriks perbandingan

| Dataset | YOLO26l | RT-DETR-L | RF-DETR-L |
|---|---|---|---|
| **RGB 953 pohon** (SawitMVC) | Deteksi: ada | Deteksi: ada | Deteksi: ada |
| **RGB 352 pohon** (SawitMVC-Depth, tanpa depth) | Belum | Belum | Belum |
| **RGB+D 352 pohon** (SawitMVC-Depth, 4-kanal) | Belum | Belum | Belum |

Setiap sel dievaluasi dua metrik:
- **Deteksi:** mAP50, mAP50-95, precision, recall (per kelas B1–B4)
- **Counting:** Class &plusmn;1 Acc, Tree &plusmn;1 Acc, Macro MAE (per pohon)

Pipeline counting diambil dari
[Baseline-SawitMVC](https://github.com/ULM-SawitMVC/Baseline-SawitMVC)
(Ridge + F_all 67-dim) — sudah established, tinggal ganti detektor.

## Navigasi

| Dokumen | Isi |
|---|---|
| [docs/REKAP.md](docs/REKAP.md) | Seluruh angka, percobaan gagal/berhasil, dan pelajaran dari Volume 1 |
| [docs/DATASET.md](docs/DATASET.md) | Spesifikasi kedua dataset |
| [docs/RENCANA.md](docs/RENCANA.md) | Rencana kerja 4 fase |
| [experiments/](experiments/) | Eksperimen baru (masih kosong) |
| [results/](results/) | Hasil eksperimen (masih kosong) |

## Repo terkait

| Repo | Peran |
|---|---|
| [Research-Pipeline](https://github.com/muhammad-zainal-muttaqin/Research-Pipeline) | Volume 1: tinjauan pustaka + eksperimen diagnostik |
| [Baseline-SawitMVC](https://github.com/ULM-SawitMVC/Baseline-SawitMVC) | Pipeline counting YOLO26m + Ridge, angka baseline |
