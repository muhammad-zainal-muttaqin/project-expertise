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

Sejak **Fase 6** tujuannya diperluas: bukan lagi hanya membandingkan tiga
arsitektur satu-tahap, tapi **memaksimalkan metrik dengan cara apa pun** —
termasuk pipeline dua-tahap dan model non-YOLO — karena diagnostik menunjukkan
pembatasan ke satu detektor tunggal-lah yang menahan angkanya (lihat
[docs/DIAGNOSIS-DEPTH.md](docs/DIAGNOSIS-DEPTH.md)).

## Status

Fase 0–6 **selesai** (`V2-E-001` s.d. `V2-E-024`). Pengumpulan metrik
dihentikan 2026-08-12.

**Baca [docs/LAPORAN-AKHIR.md](docs/LAPORAN-AKHIR.md) lebih dulu** — di situ
seluruh hasil, batasnya, dan alasan berhentinya dirangkum. Dua temuan penutup:

- **Kedua dataset terpisah ~80 hari akuisisi** (SawitMVC-YOLO Mei 2026,
  SawitMVC-Depth Juli 2026). Pada 1.408 citra ber-ID sama, B3 berbanding
  3.604 lawan 321 — perbandingan RGB-vs-RGB+D lintas-dataset mengukur populasi
  buah yang berbeda, bukan efek depth (`V2-E-022`).
- **Split test 352 tidak punya daya statistik** untuk pertanyaan ini: CI 95%
  mAP50 selebar ±0,058 pada 410 kotak GT, sementara selisih yang diperebutkan
  0,0044 (`V2-E-023`).

Satu temuan positif menutup proyek: **depth menaikkan lokalisasi, bukan
kematangan.** Uji berpasangan terakhir (resep identik, hanya kanal masukan yang
beda) memberi AP50 lokalisasi **0,7636 dengan depth vs 0,7358 tanpa** —
menembus plafon 0,733 yang sebelumnya dikira batas dataset, ternyata batas
modalitas RGB. Selisih +0,0278 dengan P(Δ>0)=0,921; CI masih memuat nol karena
split ini tidak mampu memisahkan efek di bawah ~0,10 (`V2-E-024`).

Ringkasan per fase: [experiments/STATUS.md](experiments/STATUS.md).

### Matriks hasil (test split; mAP50 pycocotools / Class ±1 Acc Ridge+F_all)

| Dataset | YOLO26l | RT-DETR-L | RF-DETR-L |
|---|---|---|---|
| RGB 953 pohon | 0,5435 / 72,16% | 0,5781 / 76,24% | 0,6012 / 76,24% |
| RGB 352 pohon | 0,3606 / 89,55% | 0,4343 / 90,91% | 0,4544 / 88,18% |
| RGB+D 352 pohon (`inverse`) | 0,3919 / 87,73% | 0,3877 / 88,64% | 0,4186 / 88,18% |
| RGB+D 352 pohon (`edge`, Fase 5) | 0,4316 / 87,27% | — | — |
| **Dua-tahap (Fase 6)** | **0,4500 / 85,91%** | — | — |

**Dua peringatan wajib sebelum mengutip matriks ini:**

1. **Baris 953 dan 352 tidak sebanding.** Bukan hanya karena ketimpangan kelas,
   tapi karena kedua dataset direkam terpisah ~80 hari pada fase kematangan
   kebun yang berbeda (`V2-E-022`). Keduanya mengukur populasi buah yang
   berbeda.
2. **Selisih antar-sel sebagian besar di bawah derau.** CI 95% mAP50 pada
   split test 352 selebar ±0,058 (`V2-E-023`). Jangan memperlakukan matriks ini
   sebagai peringkat.

Detailnya di [docs/LAPORAN-AKHIR.md](docs/LAPORAN-AKHIR.md) dan
[docs/DIAGNOSIS-DEPTH.md](docs/DIAGNOSIS-DEPTH.md).

## Navigasi

| Dokumen | Isi |
|---|---|
| [docs/LAPORAN-AKHIR.md](docs/LAPORAN-AKHIR.md) | **Mulai dari sini** — seluruh hasil, ancaman validitas, dan rekomendasi lanjutan |
| [docs/DIAGNOSIS-DEPTH.md](docs/DIAGNOSIS-DEPTH.md) | **Fase 6** — jalan penemuan kenapa RGB+D tidak menaikkan mAP, lengkap dengan probe yang bisa dijalankan ulang; §9 memuat koreksi sebab-akibat |
| [docs/REPRODUKSI-FASE6.md](docs/REPRODUKSI-FASE6.md) | **Fase 6** — urutan perintah persis untuk membangun ulang seluruh hasil, plus 9 jebakan yang wajib dihindari |
| [docs/REKAP.md](docs/REKAP.md) | Seluruh angka, percobaan gagal/berhasil, dan pelajaran dari Volume 1 |
| [docs/DATASET.md](docs/DATASET.md) | Spesifikasi kedua dataset |
| [docs/RENCANA.md](docs/RENCANA.md) | Rencana kerja per fase |
| [experiments/EKSPERIMEN.md](experiments/EKSPERIMEN.md) | Log append-only per hipotesis (`V2-E-0xx`) |
| [experiments/STATUS.md](experiments/STATUS.md) | Status fase + matriks hasil terkini |
| [results/](results/) | JSON hasil tiap eksperimen (sumber setiap angka yang dikutip) |

## Skrip

### Fase 1–5 (satu-tahap: detektor 3-kanal / 4-kanal)

| Skrip | Fungsi |
|---|---|
| `build_4ch_dataset.py` | Susun dataset BGRD TIFF 4-kanal |
| `create_depth_edge_dataset.py` | Varian encoding kanal depth (`edge`, `clipped`, `valid_mask`, …) |
| `train_yolo_4ch_screening.py` | Training YOLO26l generik (3- atau 4-kanal) |
| `train_yolo_4ch_dropout.py` | Modality dropout pada kanal depth |
| `train_yolo_midfusion.py` | Mid-fusion + gate taknol (lever arsitektur Fase 5) |
| `train_rfdetr_4ch.py` | RF-DETR-L 4-kanal |
| `eval_pycoco_*.py`, `run_counting_*.py`, `bootstrap_ci.py` | Evaluasi deteksi, counting, dan CI |
| `make_absolute_split.py`, `materialize_split_dirs.py` | Utilitas split (hindari bug resolusi path ultralytics) |

### Fase 6 (dua-tahap: lokalisasi terpisah dari kematangan)

| Skrip | Fungsi |
|---|---|
| `probe_depth_signal.py` | 5 diagnostik read-only yang mendasari Fase 6 — jalankan untuk memverifikasi tiap angka di `DIAGNOSIS-DEPTH.md` |
| `make_pretrain_split.py` | Daftar pohon 953 yang **bebas bocor** terhadap val/test 352 (846 pohon) |
| `make_agnostic_dataset.py` | Dataset deteksi 1-kelas ("tandan") untuk memisahkan lokalisasi dari klasifikasi |
| `build_crop_dataset.py` | Crop tandan + kanal **relief depth** + mask box target (`--sisi` mengatur resolusi) |
| `train_crop_classifier.py` | Classifier kematangan; `--tahap pretrain/finetune/gabung`, head ordinal/hybrid, gate taknol, loss auxiliary RGB-only |
| `probe_fitur_depth.py` | Uji apakah statistik depth **terpool** menambah info di atas RGB (dasar V2-E-016) |
| `eval_detector_agnostic.py` | AP50 lokalisasi murni + WBF antar-detektor |
| `pilih_detektor.py` | Pilih kombinasi detektor terbaik — **selalu di split val** |
| `sweep_inferensi.py` | Sweep imgsz × NMS IoU tanpa training |
| `eval_twostage.py` | Rekomposisi dua-tahap → mAP50 yang sebanding dengan Fase 1–5 |
| `run_counting_twostage.py` | Counting Ridge+F_all memakai fungsi yang **sama** dengan Fase 1–5 |

### Penutupan (validitas dan daya statistik)

| Skrip | Fungsi |
|---|---|
| `probe_pergeseran_temporal.py` | Bandingkan label kedua dataset pada citra ber-ID sama + baca tanggal akuisisi — dasar `V2-E-022` |
| `bootstrap_map.py` | CI 95% mAP50/AP50 dengan resampling tingkat citra; selisih antar-model **berpasangan** — dasar `V2-E-023` |
| `dump_classaware.py` | Dump prediksi detektor (RGB jpg atau 4-kanal TIFF) ke format yang bisa difusikan/di-bootstrap; `--agnostik` untuk AP50 lokalisasi |
| `fuse_final.py` | Fusi per-kelas lintas-jalur, bobot dipilih di val lalu dikunci |
| `buat_test_953_bersih.py` | Bangun split test 953 yang benar-benar tak tersentuh pretraining (19 pohon) |

## Data turunan (di luar repo, di `/workspace/`)

Semua regenerable dari skrip di atas — sengaja tidak ikut git karena besar:

| Folder | Dibuat oleh | Isi |
|---|---|---|
| `crops_fase6/` | `build_crop_dataset.py` | 16.542 crop 953 + 2.299 crop 352 (RGB, relief depth, mask) |
| `agnostic953/`, `agnostic352/` | `make_agnostic_dataset.py` | Dataset YOLO 1-kelas (symlink citra + label ditulis ulang) |
| `depth_png_352/` | `reproject_depth.py` (Volume 1) | Depth tereproyeksi ke frame color, uint8 kanonik |
| `SawitMVC-Depth-4ch*/` | `build_4ch_dataset.py`, `create_depth_edge_dataset.py` | Dataset TIFF 4-kanal per varian encoding |

Di dalam repo: `splits_fase6/` (daftar split bebas-bocor, kecil, ikut git)
dan `runs_fase6/` (output training Fase 6).

## Repo terkait

| Repo | Peran |
|---|---|
| [Research-Pipeline](https://github.com/muhammad-zainal-muttaqin/Research-Pipeline) | Volume 1: tinjauan pustaka + eksperimen diagnostik |
| [Baseline-SawitMVC](https://github.com/ULM-SawitMVC/Baseline-SawitMVC) | Pipeline counting YOLO26m + Ridge, angka baseline |
