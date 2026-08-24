# EDA — combined1716 (SawitMVC + SawitMVC-Depth-RGB)

Dihasilkan 2026-08-24 dari `combined_manifest.json` dan label YOLO dataset gabungan
`SawitMVC-Combined-1716-RGB` (dibangun lokal, tidak disimpan di repo ini karena
ukurannya ~3,2 GB). Analisis mandiri, mencakup komposisi sumber, distribusi kelas,
dan geometri box — tanpa metadata lokasi/GPS (tidak relevan, dataset ini tidak
punya metadata itu).

**Cara membangun ulang dataset ini kalau dibutuhkan lagi:**
```
python scripts/build_combined_rgb_dataset.py \
  --sawit <path_ke_SawitMVC-YOLO> \
  --depth-rgb <path_ke_SawitMVC-Depth-YOLO> \
  --output <path_tujuan>
```
Sumbernya: `SawitMVC-YOLO` (953 pohon) dan `SawitMVC-Depth-YOLO` v2.0.0/new763
(763 pohon). Skrip builder ada di `scripts/build_combined_rgb_dataset.py`,
sudah ada di repo ini.

## 1. Komposisi dataset

- Total gambar: **7044** (24412 box total)
- Sumber 1 (`sawitmvc`): **953** pohon, 3992 gambar
- Sumber 2 (`depth_rgb`): **763** pohon, 3052 gambar
- Pohon yang difoto ulang di kedua sumber (tree-ID sama, sesi akuisisi berbeda, ~5 bulan berselang): **352**
- Union pohon fisik unik: **1364** (953 + 763 − 352 = 1.364, bukan 1.716 — 1.716 adalah jumlah *record* gambar-pohon, bukan pohon unik)

| Split | Gambar | Pohon unik | dari sawitmvc | dari depth_rgb |
|---|---|---|---|---|
| train | 5184 | 1005 | 716 | 546 |
| valid | 808 | 160 | 96 | 101 |
| test | 1052 | 199 | 141 | 116 |

**Catatan split.** Kebijakan builder: 352 pohon yang difoto ulang mengikuti split asal SawitMVC (bukan split Depth), supaya kedua sesi akuisisi pohon yang sama tidak pernah pecah ke split berbeda — group-safe per definisi (lihat `split_policy` di `combined_manifest.json`).

## 2. Distribusi kelas (B1–B4)

| Kelas | Box | Fraksi |
|---|---|---|
| B1 | 3166 | 13.0% |
| B2 | 5558 | 22.8% |
| B3 | 11952 | 49.0% |
| B4 | 3736 | 15.3% |

Rasio kelas mayoritas/minoritas: **3.8x** (B3 vs B1). B4 jauh paling langka di ketiga split — konsisten dengan pola lama proyek ini (kematangan ekstrem lebih jarang tercapture daripada kelas tengah).

![Distribusi kelas](eda_figures_combined1716/01_distribusi_kelas.png)

![Distribusi kelas per split](eda_figures_combined1716/02_distribusi_kelas_per_split.png)

## 3. Kelas per sumber — potensi domain shift

| Kelas | sawitmvc | depth_rgb |
|---|---|---|
| B1 | 2032 (11.0%) | 1134 (19.3%) |
| B2 | 3500 (18.9%) | 2058 (35.0%) |
| B3 | 9701 (52.3%) | 2251 (38.3%) |
| B4 | 3307 (17.8%) | 429 (7.3%) |

![Distribusi kelas per sumber](eda_figures_combined1716/03_distribusi_kelas_per_sumber.png)

Perbedaan ini kemungkinan besar bukan artefak sampling, melainkan **kematangan sungguhan yang berubah** — 352 pohon di antara kedua sumber adalah pohon fisik yang sama, difoto ulang ~5 bulan kemudian, jadi tandan yang B3 di sesi pertama wajar sudah bergeser kelas di sesi kedua. Tetap perlu diwaspadai: model bisa saja belajar shortcut "sumber/gaya gambar" alih-alih kematangan murni — cek visual di atas sebelum training, jangan diasumsikan aman.

## 4. Kepadatan objek per gambar

- Rata-rata box/gambar: **3.47**
- Median: **3**
- Maks: **10**
- Gambar tanpa box sama sekali (background): **364** (5.2%)
  - dari `sawitmvc`: 66
  - dari `depth_rgb`: 298

![Box per gambar](eda_figures_combined1716/04_box_per_gambar.png)

## 5. Geometri box

- Area ternormalisasi (w×h): median **0.0121**, rentang [0.00002, 0.1338]
- Aspect ratio (w/h): median **1.16**, rentang [0.17, 10.00]

![Ukuran box](eda_figures_combined1716/05_ukuran_box.png)

## 6. Resolusi citra native (sebelum resize training)

| Sumber | Resolusi | Jumlah gambar |
|---|---|---|
| sawitmvc | 960x1280 | 3992 |
| depth_rgb | 1280x800 | 3052 |

Masing-masing sumber punya **satu resolusi native seragam**, tapi kedua sumber beresolusi **berbeda** — training di `imgsz` tetap (1280) akan me-resize keduanya, jadi rasio upscale/downscale berbeda per sumber. Ini bukan masalah baru (sudah begini sejak dataset asalnya), tapi layak diketahui saat menafsirkan hasil per sumber.

## 7. Subset LONSUM (99 pohon)

Dari 953 pohon `sawitmvc`, 99 di antaranya berasal dari kebun **LONSUM** (sisanya
854 DAMIMAS). LONSUM tetap ikut penuh di `combined1716` (tidak difilter oleh
skrip builder), tapi karakter kelasnya jauh berbeda dari populasi keseluruhan:

| Kelas | Box LONSUM | Fraksi LONSUM | Fraksi keseluruhan (§2) |
|---|---|---|---|
| B1 | 17 | 1,6% | 13,0% |
| B2 | 159 | 14,5% | 22,8% |
| B3 | 761 | **69,5%** | 49,0% |
| B4 | 158 | 14,4% | 15,3% |

396 gambar LONSUM total (train 300 / valid 40 / test 56), 1.095 box, rata-rata
2,77 box/gambar, 27 gambar tanpa box. **B3 mendominasi ekstrem dan B1 nyaris
hilang** — sinyal kuat bahwa kebun LONSUM punya karakter kematangan berbeda dari
DAMIMAS saat pengambilan gambar, bukan sekadar imbalans acak. Evaluasi pada
`results/local_eval_combined1716_no_lonsum/` di repo ini sengaja **meng-exclude
LONSUM** dari test set atas keputusan pengguna (kualitas datanya dianggap kurang
representatif).

## 8. Ringkasan cepat

- 7044 gambar, 24412 box, 4 kelas (B1-B4).
- Union 1364 pohon fisik unik dari 2 sumber (953 + 763; 352 di antaranya difoto ulang di kedua sesi, ~5 bulan berselang).
- Kelas timpang: B4 termarjinal di semua split — pertimbangkan class weighting atau minimal jangan kaget kalau recall B4 rendah.
- 5.2% gambar tanpa box (background) — normal untuk deteksi, tapi pastikan proporsinya sama di train/val/test.
- Dua sumber (`sawitmvc`, `depth_rgb`) punya resolusi native berbeda dan proporsi kelas berbeda — lihat §3 dan §6 sebelum menyimpulkan model "generalisasi" lintas sumber.
