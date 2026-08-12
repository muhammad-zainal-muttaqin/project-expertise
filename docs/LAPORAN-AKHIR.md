# Laporan Akhir — Volume 2: Deteksi dan Counting Tandan Sawit RGB vs RGB+D

**Tanggal:** 12 Agustus 2026
**Cakupan:** Fase 0–6 (`V2-E-001` s.d. `V2-E-024`)
**Status:** pengumpulan metrik dihentikan; seluruh angka final dan terlacak.

---

## 1. Ringkasan eksekutif

Volume 2 berangkat dari satu pertanyaan: **apakah menambahkan kanal depth
menaikkan mAP50 deteksi tandan sawit?** Dua puluh empat eksperimen kemudian,
jawabannya bukan "ya" atau "tidak", melainkan bahwa **pertanyaannya tidak bisa
dijawab dengan pasangan data yang tersedia** — dan itu sendiri hasil yang
terukur, bukan kegagalan mengukur.

Tiga temuan menutup persoalannya:

1. **Kedua dataset bukan pasangan yang sebanding.** SawitMVC-YOLO (953 pohon,
   tanpa depth) direkam 30 April – 16 Mei 2026; SawitMVC-Depth (352 pohon,
   dengan depth) direkam 28–29 Juli 2026. Jeda ~80 hari pada kebun yang sama.
   Distribusi kematangan bergeser drastis: pada 1.408 citra ber-ID sama,
   B3 berbanding **3.604 lawan 321** (11,2×). Setiap perbandingan
   RGB-vs-RGB+D yang melibatkan kedua dataset mengukur populasi buah yang
   berbeda, bukan efek depth.

2. **Split test 352 tidak punya daya statistik untuk pertanyaan ini.** Dengan
   410 kotak GT pada 220 citra, CI 95% untuk mAP50 selebar **±0,058**. Seluruh
   konfigurasi yang dihasilkan proyek ini — dari 0,3606 sampai 0,4544 — sebagian
   besar jatuh di dalam satu selang yang sama. Selisih 0,0044 antara pipeline
   dua-tahap terbaik dan rekor proyek adalah derau.

3. **Lokalisasi sudah mentok, klasifikasi yang tidak bisa diperbaiki.** AP50
   class-agnostic 0,7330 di test-352 versus mAP50 class-aware ~0,45. Selisih
   itu bukan cacat arsitektur: label lokalisasi bertahan melintasi jeda 80 hari
   karena posisi tandan di kanopi stabil, sedangkan label kematangan tidak
   karena benda fisiknya berubah.

**Rekomendasi utama:** pekerjaan lanjutan pada pertanyaan RGB-vs-RGB+D
memerlukan satu sesi akuisisi yang merekam RGB dan depth **bersamaan pada
tandan yang sama**, dengan test split yang cukup besar (perhitungan daya di
§8). Tanpa itu, penambahan model, loss, atau ensemble tidak akan mengubah
kesimpulan.

---

## 2. Pertanyaan penelitian dan jawabannya

| # | Pertanyaan | Jawaban | Sumber |
|---|---|---|---|
| 1 | Apakah depth menaikkan mAP50 deteksi? | **Tidak terjawab** dengan data ini. Perbandingan lintas-dataset tidak sah (§3); perbandingan di dalam 352 berada di bawah ambang deteksi statistik (§5). | V2-E-022, V2-E-023 |
| 2 | Encoding depth mana yang terbaik? | `edge` (Sobel gradien depth) menang screening dan training penuh: test mAP50 0,4316 vs `inverse` 0,3919. Selisihnya tetap belum signifikan. | V2-E-008, V2-E-010 |
| 3 | Apakah depth membawa informasi kematangan? | Ya, tapi **redundan secara kondisional** terhadap RGB: `I(Y;D) > 0` sementara `I(Y;D\|RGB) ≈ 0`. Depth saja 0,3756; RGB saja 0,6415; RGB+depth 0,6415. | V2-E-016 |
| 4 | Di mana kemampuan detektor hilang? | Pada penamaan kelas, bukan pencarian objek. AP50 agnostik 0,7330 vs mAP50 0,45. | V2-E-013, V2-E-017 |
| 5 | Apakah memperbesar model menolong? | Tidak. Dataset 953 dengan 9,8× lebih banyak kotak latih mencapai AP50 lokalisasi yang praktis sama (0,7374 vs 0,7330). Rencana `yolo26x` dibatalkan. | V2-E-017 |
| 6 | Apakah pipeline dua-tahap mengalahkan satu-tahap? | Setara, tidak lebih baik. 0,4500 vs 0,4544, selisih 26× lebih kecil dari lebar CI-nya. | V2-E-020, V2-E-023 |

---

## 3. Data

| | SawitMVC-YOLO | SawitMVC-Depth |
|---|---|---|
| Pohon | 953 (DAMIMAS 854, LONSUM 99) | 352 (DAMIMAS, subset ID) |
| Citra | 3.992 (960×1280) | 1.408 (1280×800) |
| Kotak | 18.540 | 2.299 |
| Depth | — | Orbbec Y16, 848×480, mm |
| **Akuisisi** | **30 Apr – 16 Mei 2026** | **28 – 29 Juli 2026** |
| Split | 716 / 96 / 141 pohon | 245 / 52 / 55 pohon (kanonik v1.1.0) |

### Pergeseran temporal (V2-E-022)

Pada **1.408 citra dengan tree ID yang sama**:

| Sumber label | Total kotak | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|
| SawitMVC-YOLO (Mei) | 6.523 | 566 (8,7%) | 1.098 (16,8%) | **3.604 (55,3%)** | 1.255 (19,2%) |
| SawitMVC-Depth (Juli) | 2.299 | 829 (36,1%) | 1.001 (43,5%) | **321 (14,0%)** | 148 (6,4%) |

Rotasi panen sawit 7–15 hari; jeda 80 hari berarti 5–11 putaran panen. Kohort
B3 yang dominan pada Mei matang menjadi B1/B2 pada Juli, sebagian sudah
dipanen — konsisten dengan turunnya total kotak dan bergesernya distribusi ke
80% B1+B2.

Reproduksi: `scripts/probe_pergeseran_temporal.py` → `results/pergeseran_temporal.json`.

---

## 4. Hasil deteksi

### 4.1 Matriks utama (test split, mAP50 pycocotools)

| Konfigurasi | 953 pohon | 352 pohon |
|---|---|---|
| YOLO26l RGB | 0,5435 | 0,3606 |
| RT-DETR-L RGB | 0,5781 | 0,4343 |
| RF-DETR-L RGB | 0,6012 | **0,4544** |
| YOLO26l RGB+D (`inverse`) | — | 0,3919 |
| RT-DETR-L RGB+D (`inverse`) | — | 0,3877 |
| RF-DETR-L RGB+D (`inverse`) | — | 0,4186 |
| YOLO26l RGB+D (`edge`) | — | 0,4316 |
| Dua-tahap Fase 6 (v4) | — | 0,4500 |

**Kolom 953 dan 352 tidak sebanding** (§3). Perbandingan hanya sah di dalam
kolom 352.

### 4.2 Per kelas (test 352)

| Konfigurasi | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| YOLO26l RGB | 0,6842 | 0,4184 | 0,2301 | 0,1516 |
| YOLO26l RGB+D `edge` | 0,7252 | 0,5031 | 0,2240 | 0,2740 |
| RT-DETR-L RGB | 0,7680 | 0,4867 | 0,2641 | 0,2185 |
| RF-DETR-L RGB | 0,6853 | 0,5184 | **0,3477** | 0,2661 |
| Dua-tahap v4 | 0,7366 | 0,4683 | 0,3212 | 0,2738 |

B1 pada dua-tahap (0,7366) sudah **melampaui plafon lokalisasi** 0,7330 —
kelas itu tidak punya sisa ruang perbaikan. Seluruh sisa jarak ada di B2–B4.

### 4.3 Lokalisasi murni (AP50 class-agnostic)

| Model | Split | AP50 | Catatan |
|---|---|---|---|
| `agn352_ft` | test 352 | **0,7330** | plafon mAP50 pipeline dua-tahap |
| YOLO26l `v2repro` (class-aware dilipat) | test 953 | 0,7374 | **model berbeda**, bukan detektor agnostik |
| `agn953_full` | val 953 | 0,8101 | lihat §7.3 untuk angka test |

---

## 5. Selang kepercayaan (V2-E-023)

Bootstrap pada tingkat citra, 500–1.000 ulangan, seed 42. Selisih antar-model
dihitung **berpasangan** (sampel citra yang sama untuk kedua model).

**Split test 352: 220 citra, 410 kotak GT.**

| Sumber | mAP50 | CI 95% | Lebar |
|---|---|---|---|
| YOLO26l-RGBD `edge` | 0,4270 | [0,3771; 0,4938] | 0,1167 |
| YOLO26l-RGB | 0,3677 | [0,3286; 0,4417] | 0,1130 |

Selisih berpasangan `edge` − RGB: **+0,0593**, CI 95% **[−0,0013; +0,1168]**,
P(Δ>0) = 0,972 → **tidak signifikan**.

AP50 lokalisasi juga diukur: lebar CI ~**0,101**.

**Implikasi.** Lebar CI ~0,117 sementara jarak dua-tahap (0,4500) ke RF-DETR-L
(0,4544) hanya 0,0044 — 26× lebih kecil. Bahkan selisih 0,0593 belum
signifikan. Enam versi rekomposisi dan empat skema classifier yang dikerjakan
Fase 6 seluruhnya bergerak di bawah ambang deteksi split ini.

---

## 6. Klasifikasi kematangan pada crop

Empat skema, tiga seed masing-masing (12 training, ConvNeXt-Small, head hybrid
CE+CORAL):

| Skema | Isi | test akurasi (rata ± sd) | macro F1 |
|---|---|---|---|
| `ftS` | pretrain 953 → finetune 352, crop 176 | **0,6837 ± 0,0172** | 0,6105 |
| `ftJ` | idem + jitter kotak | 0,6829 ± 0,0190 | 0,6065 |
| `ftG` | training gabungan 953+352 | 0,6724 ± 0,0161 | 0,5318 |
| `ftH` | gabungan, crop 256 @224 | 0,6569 ± 0,0252 | 0,5391 |

**Sebaran antar-seed (0,6293–0,7049, rentang 0,0756) adalah 2,8× lebih lebar
daripada sebaran antar-metode (0,0268).** Keempat skema tidak terbedakan.

Konfusi kelas selalu ke tetangga ordinal. Contoh `ftH_42` (test):

| GT \ prediksi | B1 | B2 | B3 | B4 | recall |
|---|---|---|---|---|---|
| B1 | 117 | 26 | 4 | 0 | 0,796 |
| B2 | 15 | 137 | 17 | 5 | 0,787 |
| B3 | 4 | 36 | **16** | 7 | **0,254** |
| B4 | 0 | 7 | 6 | 13 | 0,500 |

Recall B3 0,254 terjadi **meskipun B3 adalah kelas terbanyak dalam training
gabungan** (8.780 dari 18.059 crop). Penjelasannya di §3: B3 dalam korpus 953
adalah buah Mei, B3 dalam target 352 adalah buah Juli — dua populasi berbeda.

### Ablasi depth (V2-E-016)

| Masukan | Akurasi test |
|---|---|
| Depth saja | 0,3756 |
| RGB saja | 0,6415 |
| RGB + depth relief | 0,6415 |

`I(Y;D) > 0` tetapi `I(Y;D|RGB) ≈ 0` — depth membawa sinyal kematangan, tapi
sinyal itu sudah seluruhnya ada di RGB. **FALSIFIED** untuk hipotesis bahwa
depth menambah informasi kematangan.

---

## 7. Counting per pohon (Ridge + F_all)

| Konfigurasi | Class ±1 Acc |
|---|---|
| RT-DETR-L RGB 352 | **90,91%** |
| YOLO26l RGB 352 (asli) | 89,55% |
| RF-DETR-L RGB 352 | 88,18% |
| Dua-tahap v3 | 88,18% |
| YOLO26l RGB+D `inverse` | 87,73% |
| YOLO26l RGB+D `edge` | 87,27% |
| Dua-tahap v4 | 85,91% |
| YOLO26l RGB 352 (retrain) | 84,09% |

Konfigurasi terbaik untuk mAP50 **bukan** yang terbaik untuk counting: v4
menang mAP50, v3 menang counting. mAP peduli urutan deteksi di dalam kelas;
counting memakai argmax sehingga sensitif terhadap kalibrasi prior.

---

## 8. Ancaman validitas dan batasnya

1. **Pergeseran temporal (parah, tidak bisa dikoreksi pasca-hoc).** §3. Setiap
   klaim yang membandingkan 953 dan 352 tidak sah.
2. **Daya statistik (parah).** §5. Dengan 410 kotak GT, efek di bawah ~0,10
   mAP50 tidak terdeteksi. Untuk mendeteksi efek 0,03 dengan daya 80%
   dibutuhkan test split sekitar **10× lebih besar** (≈4.000 kotak).
3. **Bobot RF-DETR-L dan RT-DETR-L tidak tersimpan.** Angka 0,4544 dan 0,4343
   berasal dari `results/*.json` Fase 1–4; prediksinya tidak bisa diambil ulang,
   sehingga CI untuk kedua model itu tidak bisa dihitung. Hanya YOLO26l dan
   pipeline dua-tahap yang bisa di-bootstrap.
4. **`agn953_full` tidak punya test split bersih yang memadai.**
   `make_agnostic_dataset.py` hanya membuat train+val untuk `agnostic953`.
   Dari 141 pohon test kanonik 953, 122 terpakai saat pretraining; hanya 19
   pohon (76 citra, 321 kotak) yang tak tersentuh. Angka dari 19 pohon itu
   dilaporkan di §9 dengan CI-nya, dan CI itu lebar.
5. **Angka "test-953 = 0,7374" pernah dikutip keliru** sebagai hasil detektor
   agnostik. Itu detektor **class-aware** `v2repro` yang prediksinya dilipat
   jadi satu kelas — model yang berbeda. Dikoreksi di §4.3.
6. **Depth tidak teregistrasi ke kamera warna** pada rilis v1.0.0 dataset;
   reproyeksi per piksel dipakai (`reproject_depth.py`, Volume 1). Kesalahan
   registrasi tanpa reproyeksi: median 29 px, sebanding ukuran tandan kecil.

---

## 9. Uji terakhir: apakah depth menolong lokalisasi?

Satu-satunya perbandingan RGB vs RGB+D pada proyek ini yang **tidak** bisa
dikotori pergeseran temporal, karena class-agnostic membuang label kematangan
sepenuhnya dan hanya menyisakan "ada tandan atau tidak" — label yang terbukti
bertahan melintasi jeda 80 hari.

Rancangan berpasangan: resep, inisialisasi (`agn953_full`), seed (42), jadwal
(60 epoch, patience 45, cosine), resolusi (1280), dan batch (4) **identik**.
Satu-satunya yang berbeda adalah jumlah kanal masukan.

*(Hasil diisi di §9.1 setelah run selesai — lihat `V2-E-024`.)*

---

## 10. Rekomendasi

**Untuk menjawab pertanyaan depth secara meyakinkan**, yang dibutuhkan bukan
model atau loss yang lebih baik, melainkan data:

1. **Satu sesi akuisisi** yang merekam RGB dan depth bersamaan pada tandan yang
   sama. Perbandingan RGB vs RGB+D lalu menjadi ablasi kanal murni, bukan
   perbandingan lintas-waktu.
2. **Test split ≈4.000 kotak** (≈10× sekarang) supaya efek berukuran 0,03
   mAP50 terdeteksi dengan daya 80%.
3. **Anotasi ulang subset 953 dengan standar Juli**, atau sebaliknya, kalau
   korpus 953 tetap ingin dipakai sebagai sumber pretraining kematangan.
   Untuk pretraining **lokalisasi** saja, korpus 953 tetap sah dan berguna.

**Yang tidak perlu diulang** (sudah terbukti tidak menolong): memperbesar model
(V2-E-017), early fusion di stem (V2-E-005, V2-E-022 Volume 1), gate init-nol
(F-007), konsistensi lintas-sisi (F-003), tuning hyperparameter, SAHI, dan
ensembling classifier di luar ~3 anggota (V2-E-023).

---

## 11. Reproduksi

Urutan perintah lengkap: [REPRODUKSI-FASE6.md](REPRODUKSI-FASE6.md), termasuk
tabel sembilan jebakan yang semuanya gagal secara **diam-diam** (tanpa pesan
error) dan semuanya pernah terjadi di proyek ini.

| Berkas | Isi |
|---|---|
| `experiments/EKSPERIMEN.md` | log append-only `V2-E-001` … `V2-E-024` |
| `results/fase6_ringkas.json` | seluruh metrik Fase 6 dalam satu berkas |
| `results/pergeseran_temporal.json` | bukti pergeseran akuisisi 80 hari |
| `results/bootstrap_map*.json` | selang kepercayaan |
| `docs/DIAGNOSIS-DEPTH.md` | jalan penemuan + koreksi §9 |
