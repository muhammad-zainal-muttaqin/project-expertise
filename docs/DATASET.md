# Spesifikasi Dataset

## 1. SawitMVC (953 pohon, RGB)

| Properti | Nilai |
|---|---|
| Sumber | [ULM-DS-Lab/SawitMVC-YOLO](https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-YOLO) |
| Lisensi | CC BY-NC 4.0 |
| Pohon | 953 (908 difoto 4 sisi, 45 difoto 8 sisi) |
| Citra | 3.992 |
| Resolusi | 960 x 1280 (potret) |
| Bbox | 18.540 |
| Tandan unik | 9.823 |
| Kelas | B1, B2, B3, B4 |
| Sisi per pohon | 4 atau 8 |
| Split (per pohon) | 716 train / 96 val / 141 test |
| k (duplikasi lintas-sisi) | 1,89 |
| Kebun | DAMIMAS dan LONSUM, Kab. Tanah Laut, Kalsel |
| Perangkat | 10 model smartphone, eksposur otomatis |

### Distribusi kelas (seluruh dataset)

| Kelas | Tandan unik | Proporsi |
|---|---:|---:|
| B3 | terbanyak | ~38% |
| B4 | | ~20% |
| B2 | | ~18% |
| B1 | | ~12% |

### Arah kelas

**B1 = MATANG** (jingga-merah, besar, posisi paling bawah di pohon).
Menurun sampai **B4 = MENTAH** (gelap kehijauan, kecil, posisi paling atas).

### Raw master (Sawit, 3.992 citra 3024x4032)

Ada master mentah beresolusi tinggi di `/workspace/Sawit/data`. Rasio aspek
identik (0,75), sehingga koordinat YOLO ternormalisasi dari MVC berlaku
persis. Nama berkas raw **tidak unik** secara global (936 nama kembar antar
folder) — perlu pencocokan berbasis isi untuk pemetaan.

## 2. SawitMVC-Depth (352 pohon, RGB + Depth)

| Properti | Nilai |
|---|---|
| Sumber | [ULM-DS-Lab/SawitMVC-Depth](https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-Depth) |
| Lisensi | CC BY-NC 4.0 |
| Repo | **Private** (butuh token HuggingFace) |
| Pohon | 352 |
| Citra RGB | 1.408 |
| Resolusi RGB | 1.280 x 800 (lanskap) |
| Bbox | 2.299 |
| Kelas | B1, B2, B3, B4 |
| Depth | Orbbec sensor, Y16 848x480, uint16 milimeter |
| Split | **Belum ada — perlu dibuat** |

### Distribusi kelas (terbalik dari SawitMVC)

| Kelas | Bbox | Proporsi |
|---|---:|---:|
| B2 | 1.000 | 43,5% |
| B1 | 831 | 36,1% |
| B3 | 322 | 14,0% |
| B4 | 148 | 6,4% |

Kepadatan: 1,63 bbox/citra (vs 4,64 di SawitMVC). **Angka mAP di dataset ini
TIDAK sebanding dengan angka SawitMVC.**

### Tiga sifat penting depth (sudah terverifikasi)

1. **Sidecar `alignedTo: color` menyesatkan.** Buffer depth masih di grid
   kamera depth. Reproyeksi penuh diperlukan (`reproject_depth.py` dari
   Research-Pipeline `experiments/code/build/`), bukan `cv2.resize` naif.
2. **Dua unit kamera** dengan kalibrasi berbeda (fx_depth 416,55 vs 414,38).
   Kalibrasi wajib dibaca per berkas dari sidecar.
3. **Rentang Z_NEAR/Z_FAR** di sidecar (0,3–8,0 m) tidak cocok — gunakan
   **0,8–15,0 m**.

## 3. Perbandingan Kedua Dataset

| | SawitMVC | SawitMVC-Depth |
|---|---|---|
| Jumlah pohon | 953 | 352 |
| Citra | 3.992 | 1.408 |
| Resolusi | 960x1280 (potret) | 1280x800 (lanskap) |
| Orientasi | Potret | Lanskap |
| Bbox/citra | 4,64 | 1,63 |
| Kelas dominan | B3 | B2 |
| Kelas langka | B1 | B4 (hanya 148) |
| Depth | Tidak | Ya |
| Split resmi | Ada | Belum |

**Konsekuensi:** Perbandingan RGB vs RGB+D hanya sah pada dataset yang sama
(352 pohon). Angka dari 953 pohon adalah referensi terpisah.
