# Spesifikasi Teknis Dataset Penelitian

Dokumen ini memuat spesifikasi teknis, karakteristik sensor, distribusi kategori kematangan, dan protokol partisi untuk dataset **SawitMVC-YOLO** dan **SawitMVC-Depth**.

---

## 1. Dataset SawitMVC (953 Pohon, Modalitas RGB)

| Parameter Properti | Nilai Spesifikasi |
|---|---|
| Sumber Repositori | [ULM-DS-Lab/SawitMVC-YOLO](https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-YOLO) |
| Lisensi Data | CC BY-NC 4.0 |
| Populasi Pohon | 953 pohon (908 pohon 4 sisi pandang, 45 pohon 8 sisi pandang) |
| Total Citra | 3.992 citra |
| Resolusi Asli Citra | $960 \times 1.280\text{ piksel}$ (orientasi potret) |
| Total Kotak Pembatas (*Bounding Box*) | 18.540 kotak |
| Entitas Tandan Buah Fisik Unik | 9.823 tandan terdeduplikasi |
| Kategori Kematangan | B1 (lewat matang/panen) s.d. B4 (mentah) |
| Pembagian Partisi (*Tree-Stratified*) | 716 latih / 96 validasi / 141 uji |
| Faktor Duplikasi Lintas-Sisi ($k$) | Rerata 1,89 kemunculan per tandan |
| Lokasi Perekaman Lapangan | Perkebunan DAMIMAS dan LONSUM, Kab. Tanah Laut, Kalimantan Selatan |
| Perangkat Kamera | 10 model ponsel pintar, sistem eksposur otomatis |
| **Rentang Waktu Perekaman** | **30 April – 16 Mei 2026** |

### Taksonomi Tingkat Kematangan Buah
* **B1 (Lewat Matang / Siap Panen)**: Warna jingga-kemerahan cerah, ukuran tandan besar, posisi terbawah di lingkar batang.
* **B2 (Matang Optimal)**: Warna oranye kemerahan dengan semburat ungu kehitaman.
* **B3 (Matang Awal / Mengkal)**: Warna ungu kemerahan kehitaman.
* **B4 (Mentah / Muda)**: Warna hitam kehijauan pekat, ukuran tandan relatif kecil, posisi teratas dan tertanam di sela pelepah.

---

## 2. Dataset SawitMVC-Depth (352 Pohon, Modalitas RGB+Depth)

| Parameter Properti | Nilai Spesifikasi |
|---|---|
| Sumber Repositori | [ULM-DS-Lab/SawitMVC-Depth](https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-Depth) |
| Lisensi Data | CC BY-NC 4.0 (Akses Terbatas / *Private Repository*) |
| Populasi Pohon | 352 pohon (sub-populasi DAMIMAS) |
| Total Citra RGB | 1.408 citra ($1.280 \times 800\text{ piksel}$, orientasi lanskap) |
| Total Kotak Pembatas | 2.299 kotak |
| Sensor Kedalaman | Sensor inframerah Orbbec Y16 ($848 \times 480\text{ piksel}$, unit uint16 milimeter) |
| Pembagian Partisi Kanonik | 245 latih / 52 validasi / 55 uji ($70\% / 15\% / 15\%$, *seed* 10) |
| **Rentang Waktu Perekaman** | **28 – 29 Juli 2026 (Jeda $\sim 80\text{ hari}$ dari SawitMVC)** |

### Karakteristik Teknis Sensor Kedalaman
1. **Kalibrasi dan Reproyeksi Spasial**:
   Buffer kedalaman mentah berada pada bidang koordinat sensor inframerah tersendiri. Reproyeksi piksel-ke-piksel penuh ke koordinat kamera warna RGB (`scripts/reproject_depth.py`) wajib dilakukan untuk mengoreksi pergeseran fisik median 29 piksel.
2. **Variasi Parameter Intrinsik Unit Kamera**:
   Dua unit kamera Orbbec yang digunakan memiliki panjang fokus berbeda ($f_{x} = 416,55$ vs $414,38$). Parameter kalibrasi intrinsik dibaca secara dinamis per berkas *sidecar* JSON.
3. **Rentang Normalisasi Kedalaman Efektif**:
   Rentang kuantisasi dioptimalkan pada interval $[0,8; 15,0]\text{ meter}$.

---

## 3. Komparasi Karakteristik Antar-Dataset

| Parameter Komparasi | SawitMVC (953 Pohon) | SawitMVC-Depth (352 Pohon) |
|---|---|---|
| Volume Citra | 3.992 citra | 1.408 citra |
| Orientasi Citra | Potret ($960 \times 1.280\text{ px}$) | Lanskap ($1.280 \times 800\text{ px}$) |
| Kepadatan Objek per Citra | 4,64 kotak/citra | 1,63 kotak/citra |
| Kategori Dominan | B3 ($55,3\%$) | B2 ($43,5\%$) dan B1 ($36,1\%$) |
| Kategori Paling Langka | B1 ($8,7\%$) | B4 ($6,4\%$, hanya 148 kotak) |
| Modalitas Kedalaman Fisik | Tidak Tersedia | Tersedia (Sensor Orbbec) |
| Periode Akuisisi Lapangan | Mei 2026 | Juli 2026 |

> [!WARNING]
> **Batasan Validitas Komparasi**:
> Akibat perbedaan periode perekaman 80 hari dan pergeseran fenologi kebun, perbandingan efektivitas modalitas RGB vs RGB+Depth **hanya sah dilakukan di dalam partisi dataset 352 pohon**, bukan membandingkan metrik absolut 953 pohon terhadap 352 pohon.
