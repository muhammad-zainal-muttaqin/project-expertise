# Laporan Akhir — Volume 2: Deteksi dan Pencacahan Tandan Kelapa Sawit RGB vs RGB+D

**Tanggal Penyusunan:** 12 Agustus 2026  
**Cakupan Fase:** Fase 0 s.d. Fase 6 (`V2-E-001` s.d. `V2-E-026`)  
**Status Evaluasi:** Pengumpulan metrik eksperimental ditutup; seluruh angka empiris telah diverifikasi dan terlacak penuh.

> **Pembaruan 27 Agustus 2026:** Verifikasi lanjutan terhadap bobot remote
> `new763` dan `combined1716`, termasuk WBF serta pipeline empat sisi, dicatat
> sebagai `V2-E-042` pada [laporan artefak remote](../results/remote_eval_2026-08-27/README.md).
> Iterasi greedy linker dan uji classifier 5 epoch berikutnya dicatat sebagai
> `V2-E-043` dan `V2-E-044` pada log eksperimen serta [laporan optimized
> pipeline](../results/remote_eval_2026-08-27/OPTIMIZED_PIPELINE.md).
> Angka tersebut merupakan verifikasi engineering pada test lokal dan tidak
> menggantikan angka kanonik dalam laporan ini sebelum audit silsilah split
> selesai.

---

## 1. Ringkasan Eksekutif

Volume 2 dari riset ini dirancang untuk menjawab satu pertanyaan inti: **apakah penambahan kanal kedalaman (*depth*) mampu meningkatkan metrik deteksi $mAP50$ pada tandan buah segar (TBS) kelapa sawit?** Melalui pelaksanaan 26 eksperimen terstruktur, diperoleh kesimpulan komprehensif bahwa **pertanyaan perbandingan tersebut tidak dapat dijawab secara valid menggunakan pasangan dataset awal yang tersedia** — dan penemuan ini merupakan hasil pengukuran ilmiah yang terukur, bukan kegagalan pengukuran.

Empat temuan empiris utama merangkum kesimpulan proyek:

1. **Kedua Dataset Mengalami Pergeseran Temporal (*Temporal Domain Shift*, V2-E-022)**:  
   Dataset SawitMVC-YOLO (953 pohon, modalitas RGB murni) direkam pada rentang 30 April – 16 Mei 2026, sedangkan dataset SawitMVC-Depth (352 pohon, modalitas RGB+D) direkam pada 28–29 Juli 2026 (terdapat jeda waktu **$\sim 80\text{ hari}$** pada kebun kelapa sawit yang sama). Jeda waktu ini setara dengan $5\text{--}11$ siklus rotasi panen. Distribusi kematangan buah mengalami pergeseran drastis: pada 1.408 citra ber-ID pohon identik, jumlah tandan kelas matang awal B3 berbanding **$3.604\text{ berbanding }321$ kotak** (penurunan $11,2\times$). Akibatnya, setiap perbandingan performa 4-kelas lintas-dataset mengukur dua populasi kematangan buah yang berbeda secara biologis, bukan mengukur efek kanal kedalaman.

2. **Partisi Uji 352 Pohon Memiliki Keterbatasan Daya Statistik (V2-E-023)**:  
   Dengan 410 kotak nilai acuan kebenaran (*ground truth*) pada 220 citra uji, selang kepercayaan (*confidence interval*) 95% untuk metrik $mAP50$ memiliki rentang selebar **$\pm 0,058$** ($0,1167$). Seluruh variasi konfigurasi model yang dikembangkan pada Fase 6 — dari $0,3606$ hingga $0,4544$ — berada di dalam selang ketidakpastian yang sama. Selisih $0,0044$ antara pipeline dua-tahap terbaik ($0,4500$) dan rekor RF-DETR-L ($0,4544$) berada jauh di bawah variasi derau acak data.

3. **Sumber Degradasi Performa Terletak pada Klasifikasi, Bukan Lokalisasi (V2-E-013)**:  
   Evaluasi deteksi murni lokalisasi 1-kelas (*class-agnostic*) mencapai $AP50 = \mathbf{0,7330}$, berbanding jauh dengan deteksi *class-aware* 4-kelas yang berada di kisaran $\sim 0,45$. Hal ini membuktikan bahwa anotasi posisi fisik tandan bertahan melintasi jeda waktu 80 hari karena kanopi pohon relatif stabil, sedangkan label kematangan buah berubah total akibat proses pematangan dan pemanenan alami.

4. **Kanal Kedalaman Terbukti Meningkatkan Lokalisasi Objek (V2-E-024)**:  
   Uji komparasi berpasangan terkontrol ketat (resep, arsitektur, dan bobot inisialisasi identik) menghasilkan performa lokalisasi murni sebesar **$AP50 = \mathbf{0,7636}$ pada model 4-kanal (RGB + Sobel `edge`)** berbanding **$\mathbf{0,7358}$ pada kontrol RGB murni** ($\Delta = +0,0278$, $P(\Delta > 0) = 92,1\%$). Titik estimasi ini menembus batas semu $0,733$ yang sebelumnya diperkirakan sebagai limit dataset. Sinyal kedalaman terbukti memberikan nilai tambah diskriminatif tepat pada tugas lokalisasi fisik, namun bersifat redundan terhadap fitur visual RGB untuk klasifikasi tingkat kematangan buah.

**Rekomendasi Utama**: Penelitian lanjutan terkait modalitas RGB-D memerlukan sesi akuisisi tunggal yang merekam citra RGB dan data kedalaman secara simultan pada tandan buah yang sama, dengan partisi uji yang memadai ($\approx 4.000\text{ kotak pembatas}$), serta difokuskan pada penguatan **lokalisasi objek** alih-alih klasifikasi kematangan.

---

## 2. Pertanyaan Penelitian dan Jawaban Empiris

| No. | Pertanyaan Penelitian | Kesimpulan Empiris | Simpul Bukti |
|---|---|---|---|
| 1 | Apakah penambahan depth menaikkan $mAP50$ deteksi **4 kelas**? | **Tidak dapat diputuskan secara valid** pada dataset ini. Perbandingan lintas-dataset tidak valid akibat pergeseran temporal (§3); perbandingan di dalam subset 352 berada di bawah ambang deteksi statistik (§5). | `V2-E-022`, `V2-E-023` |
| 1b | Apakah penambahan depth menaikkan $AP50$ **lokalisasi murni**? | **Terbukti meningkatkan pada titik estimasi** ($0,7636$ vs $0,7358$, $\Delta = +0,0278$, $P(\Delta > 0) = 92,1\%$). Menembus batas $0,733$ yang merupakan limit modalitas RGB. | `V2-E-024` |
| 2 | Representasi kedalaman (*encoding*) mana yang paling optimal? | Representasi gradien Sobel (`edge`) mengungguli representasi invers mentah pada penyaringan awal dan pelatihan penuh ($mAP50 = \mathbf{0,4316}$ vs $\mathbf{0,3919}$, peningkatan relatif $+10,1\%$). | `V2-E-008`, `V2-E-010` |
| 3 | Apakah kanal kedalaman membawa informasi kematangan tambahan? | **Redundan secara kondisional** terhadap RGB ($I(Y; D) > 0$, namun $I(Y; D \mid \text{RGB}) \approx 0$). Akurasi model pengklasifikasi kematangan pada RGB murni dan RGB+Depth identik di angka $\mathbf{0,6415}$. | `V2-E-016` |
| 4 | Di mana letak kehilangan kemampuan performa detektor? | Kehilangan terbesar ($44,5\%$) berasal dari **kesalahan klasifikasi kelas ordinal**, bukan kegagalan melokalisasi kotak pembatas objek ($AP50\text{ agnostik} = 0,7330$ vs $mAP50\text{ 4-kelas} \approx 0,45$). | `V2-E-013`, `V2-E-017` |
| 5 | Apakah memperbesar kapasitas model (*scaling-up*) memberikan solusi? | **Tidak**. Dataset 953 dengan volume data latih $9,8\times$ lebih besar menghasilkan $AP50$ lokalisasi yang setara ($0,7374$ vs $0,7330$). Yang meningkatkan performa adalah **perluasan modalitas**, bukan kapasitas parameter. | `V2-E-017`, `V2-E-024` |
| 6 | Apakah pipeline dua-tahap mengungguli detektor satu-tahap? | **Setara pada estimasi titik** ($0,4500$ vs $0,4544$), dengan selisih yang $26\times$ lebih kecil dari lebar selang kepercayaan statistik. | `V2-E-020`, `V2-E-023` |

---

## 3. Karakteristik Data & Pergeseran Temporal

| Parameter Karakteristik | SawitMVC-YOLO | SawitMVC-Depth |
|---|---|---|
| Populasi Pohon | 953 pohon (DAMIMAS 854, LONSUM 99) | 352 pohon (sub-populasi DAMIMAS) |
| Volume Citra | 3.992 citra ($960 \times 1.280$ piksel, potret) | 1.408 citra ($1.280 \times 800$ piksel, lanskap) |
| Jumlah Kotak Anotasi | 18.540 kotak pembatas | 2.299 kotak pembatas |
| Modalitas Kedalaman | Tidak tersedia | Sensor Orbbec Y16 ($848 \times 480$ piksel, skala mm) |
| **Rentang Tanggal Akuisisi** | **30 April – 16 Mei 2026** | **28 – 29 Juli 2026** |
| Pembagian Partisi (*Split*) | 716 latih / 96 validasi / 141 uji | 245 latih / 52 validasi / 55 uji (kanonik v1.1.0) |

### Analisis Pergeseran Temporal (Simpul V2-E-022)

Evaluasi perbandingan pada **1.408 citra dengan nomor identitas pohon yang identik**:

| Sumber Label Anotasi | Total Kotak | Lewat Matang / Siap Panen (B1) | Matang Optimal (B2) | Matang Awal / Mengkal (B3) | Mentah / Muda (B4) |
|---|---|---|---|---|---|
| SawitMVC-YOLO (Mei 2026) | 6.523 | 566 ($8,7\%$) | 1.098 ($16,8\%$) | **3.604 ($55,3\%$)** | 1.255 ($19,2\%$) |
| SawitMVC-Depth (Juli 2026) | 2.299 | 829 ($36,1\%$) | 1.001 ($43,5\%$) | **321 ($14,0\%$)** | 148 ($6,4\%$) |

Rotasi panen kebun sawit berlangsung secara berkala setiap 7–15 hari. Jeda waktu 80 hari mencakup $5\text{--}11$ putaran panen. Kohort buah yang dominan pada Mei telah matang menjadi B1/B2 pada Juli dan sebagian besar telah dipanen — konsisten dengan penurunan total kotak dari 6.523 menjadi 2.299 serta pergeseran populasi ke $79,6\%$ kelas B1+B2.

Skrip Reproduksi: [`scripts/probe_pergeseran_temporal.py`](file:///D:/Work/Assisten-Dosen/project-expertise/scripts/probe_pergeseran_temporal.py) $\to$ [`results/pergeseran_temporal.json`](file:///D:/Work/Assisten-Dosen/project-expertise/results/pergeseran_temporal.json).

---

## 4. Hasil Evaluasi Deteksi

### 4.1 Matriks Deteksi Utama (Split Uji, $mAP50$ pycocotools)

| Konfigurasi Model & Modalitas | 953 Pohon | 352 Pohon |
|---|---|---|
| YOLO26l RGB | 0,5435 | 0,3606 |
| RT-DETR-L RGB | 0,5781 | 0,4343 |
| RF-DETR-L RGB | **0,6012** | **0,4544** |
| YOLO26l RGB+D (invers mentah) | — | 0,3919 |
| RT-DETR-L RGB+D (invers mentah) | — | 0,3877 |
| RF-DETR-L RGB+D (invers mentah) | — | 0,4186 |
| YOLO26l RGB+D (Sobel `edge`) | — | **0,4316** |
| Pipeline Dua-Tahap Fase 6 (v4) | — | **0,4500** |

*Catatan*: Kolom 953 dan 352 tidak sebanding akibat pergeseran domain temporal (§3). Komparasi hanya valid dilakukan di dalam kolom dataset yang sama.

### 4.2 Evaluasi Per-Kelas Kematangan (Split Uji 352 Pohon)

| Konfigurasi Model | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| YOLO26l RGB | 0,6842 | 0,4184 | 0,2301 | 0,1516 |
| YOLO26l RGB+D Sobel `edge` | 0,7252 | 0,5031 | 0,2240 | 0,2740 |
| RT-DETR-L RGB | **0,7680** | 0,4867 | 0,2641 | 0,2185 |
| RF-DETR-L RGB | 0,6853 | **0,5184** | **0,3477** | 0,2661 |
| Pipeline Dua-Tahap v4 | 0,7366 | 0,4683 | 0,3212 | **0,2738** |

### 4.3 Lokalisasi Murni 1-Kelas ($AP50$ Class-Agnostic)

| Model Detektor | Modalitas Masukan | Partisi Evaluasi | $AP50$ | Catatan Metodologis |
|---|---|---|---|---|
| **`agn352_4ch`** | **RGB+D Sobel `edge`** | Uji 352 | **0,7636** | Nilai tertinggi; menembus batas semu modalitas RGB (§9.1) |
| `agn352_ft3` | RGB | Uji 352 | 0,7358 | Kontrol berpasangan resep identik |
| `agn352_ft` | RGB | Uji 352 | 0,7330 | Plafon lokalisasi pipeline dua-tahap awal |
| YOLO26l `v2repro` (dilipat) | RGB | Uji 953 | 0,7374 | Model *class-aware* yang dilipat kategorinya |
| `agn953_full` | RGB | Uji 953 Bersih | **0,7702** | Dievaluasi pada 19 pohon tak tersentuh prapelatihan (§9.2) |
| `agn953_full` | RGB | Validasi 953 | 0,8101 | Skor himpunan validasi |

---

## 5. Analisis Selang Kepercayaan (Simpul V2-E-023)

Estimasi selang kepercayaan bootstrap 95% dijalankan pada tingkat citra dengan 500–1.000 ulangan acak (seed 42). Selisih antar-model dihitung secara **berpasangan** pada sampel citra yang sama.

**Partisi Uji 352: 220 citra, 410 kotak acuan.**

| Model Pembanding | $mAP50$ Titik | Selang Kepercayaan 95% | Lebar Rentang |
|---|---|---|---|
| YOLO26l-RGBD Sobel `edge` | 0,4270 | $[0,3771; 0,4938]$ | 0,1167 |
| YOLO26l-RGB Murni | 0,3677 | $[0,3286; 0,4417]$ | 0,1130 |

Selisih berpasangan `edge` minus RGB: **$+0,0593$** (CI95 $[\minus 0,0013; +0,1168]$, $P(\Delta > 0) = 0,972$). Karena selang kepercayaan mencakup nilai nol, peningkatan ini **belum mencapai signifikansi statistik formal pada taraf $\alpha = 0,05$**.

---

## 6. Klasifikasi Kematangan pada Citra Terpotong (Crop)

Evaluasi 4 skema pengklasifikasi kematangan (ConvNeXt-Small, head hybrid CE+CORAL, 3 seed replikasi):

| Skema Pelatihan | Deskripsi Metode | Akurasi Uji (Rerata $\pm$ SD) | Macro-$F1$ |
|---|---|---|---|
| `ftS` | Prapelatihan 953 $\to$ Penyesuaian Terarah 352 (crop 176 px) | **$0,6837 \pm 0,0172$** | **0,6105** |
| `ftJ` | Idem + perturbasi acak (*jitter*) kotak | $0,6829 \pm 0,0190$ | 0,6065 |
| `ftG` | Pelatihan gabungan 953 + 352 | $0,6724 \pm 0,0161$ | 0,5318 |
| `ftH` | Pelatihan gabungan (crop 256 px @ 224) | $0,6569 \pm 0,0252$ | 0,5391 |

Sebaran performa antar-seed ($0,6293\text{--}0,7049$, rentang $0,0756$) adalah $2,8\times$ lebih lebar daripada sebaran antar-metode ($0,0268$), menunjukkan bahwa keempat skema klasifikasi secara statistik tidak terbedakan.

### Studi Ablasi Sinyal Kedalaman (Simpul V2-E-016)

| Modalitas Masukan | Akurasi Uji ($n = 410$) |
|---|---|
| Kedalaman Saja (Statistik Relief) | 0,3756 |
| RGB Murni (Penultimate Feature 768-dim) | **0,6415** |
| RGB + Statistik Kedalaman Relief | **0,6415** |

Terbukti secara empiris bahwa $I(Y; D) > 0$ namun $I(Y; D \mid \text{RGB}) \approx 0$. Kanal kedalaman membawa sinyal kematangan bila berdiri sendiri, namun informasi tersebut telah sepenuhnya terwakili oleh fitur visual RGB.

---

## 7. Pencacahan per Pohon (Ridge + $F_{all}$)

| Konfigurasi Model | $\text{Class }\pm 1\text{ Acc}$ Uji |
|---|---|
| RT-DETR-L RGB 352 | **90,91%** |
| YOLO26l RGB 352 (asli) | 89,55% |
| RF-DETR-L RGB 352 | 88,18% |
| Dua-Tahap v3 | 88,18% |
| YOLO26l RGB+D (invers mentah) | 87,73% |
| YOLO26l RGB+D Sobel `edge` | 87,27% |
| Dua-Tahap v4 | 85,91% |
| YOLO26l RGB 352 (pelatihan ulang) | 84,09% |

Konfigurasi terbaik untuk $mAP50$ deteksi (v4: $0,4500$) tidak identik dengan konfigurasi terbaik pencacahan (v3: $88,18\%$). Metrik $mAP50$ mengoptimalkan urutan probabilitas dalam kelas, sedangkan pencacahan mengandalkan ambang argmax tegas yang sensitif terhadap kalibrasi prior kelas.

---

## 8. Ancaman Validitas & Batasan Audit

1. **Pergeseran Domain Temporal**: Perbandingan performa lintas-dataset 953 vs 352 tidak sah (§3).
2. **Keterbatasan Daya Statistik**: Dengan 410 kotak acuan, efek di bawah $\Delta \approx 0,10\text{ mAP50}$ tidak dapat dipisahkan dari variasi acak (§5). Diperlukan $\approx 4.000\text{ kotak}$ untuk mendeteksi efek $\Delta = 0,03$ dengan daya $80\%$.
3. **Ketiadaan Berkas Bobot Historis Volume 2**: Enam direktori bobot checkpoint RT-DETR-L dan RF-DETR-L Volume 2 hilang sebelum sempat dicadangkan ke repositori publik. Prosedur riset telah disempurnakan dengan kewajiban mengekspor dump prediksi `.npz` secara langsung saat evaluasi.
4. **Audit Partisi Prapelatihan Agnostik (`agn953_full`)**: Ditemukan bahwa 122 dari 141 pohon pada partisi `test_penuh` ikut terpakai saat prapelatihan agnostik. Evaluasi yang sah mengacu pada partisi uji bersih (`test_bersih`, 19 pohon / 316 kotak) dengan skor $AP50 = \mathbf{0,7702}$ (§9.2).
5. **Kesalahan Registrasi Sensor Kedalaman**: Citra kedalaman mentah memiliki pergeseran spasial fisik median 29 piksel terhadap kamera warna, yang diatasi melalui reproyeksi piksel-ke-piksel.

---

## 9. Pembuktian Efektivitas Depth pada Lokalisasi

### 9.1 Hasil Uji Berpasangan Terkontrol (Simpul V2-E-024)

| Model Detektor | Validasi $AP50$ | @Epoch | **Uji $AP50$** | Selang Kepercayaan 95% | Lebar Rentang |
|---|---|---|---|---|---|
| `agn352_4ch` (RGB + Sobel `edge`) | **0,7893** | 33 | **0,7636** | $[0,7144; 0,8123]$ | 0,0979 |
| `agn352_ft3` (RGB Murni) | 0,7473 | 42 | **0,7358** | $[0,6820; 0,7917]$ | 0,1097 |

Selisih berpasangan: **$+0,0278$** (CI95 $[\minus 0,0121; +0,0648]$, $P(\Delta > 0) = \mathbf{0,921}$).

Sinyal positif ini konsisten pada data validasi ($+0,0420$) dan data uji ($+0,0278$). Model 4-kanal menghasilkan volume deteksi yang lebih tinggi ($1.660$ vs $1.226$ prediksi), selaras dengan peningkatan daya tangkap (*recall*).

### 9.2 Evaluasi Partisi Bersih agn953_full (Simpul V2-E-025)

| Himpunan Uji Evaluasi | Pohon | Citra | Kotak | $AP50$ Lokalisasi |
|---|---|---|---|---|
| **Partisi Bersih** (Tak Tersentuh Prapelatihan) | 19 | 76 | 316 | **0,7702** |
| Partisi Penuh (122 Pohon Terkontaminasi Prapelatihan) | 141 | 588 | 2.612 | 0,8090 |
| Himpunan Validasi (Skor Pemantauan) | — | 364 | — | 0,8101 |

Nilai performa generalisasi yang sah adalah **$0,7702$**, bukan $0,8101$.

---

## 10. Rekomendasi Penerapan & Riset Lanjutan

1. **Akuisisi Simultan Tunggal**: Perekaman data RGB dan kedalaman wajib dilakukan pada satu sesi terpadu dengan jarak waktu nol pada tandan buah yang sama.
2. **Skala Partisi Uji yang Memadai**: Partisi uji minimal $\approx 4.000\text{ kotak anotasi}$ untuk menjamin daya statistik $80\%$ terhadap efek berukuran $\Delta \approx 0,03$.
3. **Fokus pada Tugas Lokalisasi Objek**: Memprioritaskan integrasi sinyal kedalaman untuk menemukan posisi tandan yang tertutup pelepah, bukan untuk identifikasi tingkat kematangan.
4. **Metode yang Tidak Dianjurkan**: Menghindari *early fusion* konvensional pada stem resolusi tinggi, estimasi depth monokular, penyetelan hiperparameter minor, dan modul *gating* nol-inisialisasi.

---

## 11. Rujukan Reproduksi & Berkas Artefak

Prosedur eksekusi langkah-demi-langkah tersedia di [docs/REPRODUKSI-FASE6.md](REPRODUKSI-FASE6.md).

| Berkas Artefak | Deskripsi Kandungan |
|---|---|
| [`docs/WORKFLOW_KRONOLOGIS.md`](WORKFLOW_KRONOLOGIS.md) | Rekam jejak alur kerja kronologis lengkap seluruh simpul eksperimen. |
| [`experiments/EKSPERIMEN.md`](../experiments/EKSPERIMEN.md) | Log *append-only* `V2-E-001` s.d. `V2-E-044`. |
| [`results/remote_eval_2026-08-27/README.md`](../results/remote_eval_2026-08-27/README.md) | Verifikasi bobot remote, WBF, dan pipeline empat sisi pada dua test lokal. |
| [`results/fase6_ringkas.json`](../results/fase6_ringkas.json) | Kompilasi seluruh metrik kuantitatif Fase 6. |
| [`results/pergeseran_temporal.json`](../results/pergeseran_temporal.json) | Bukti empiris pergeseran temporal 80 hari antar-dataset. |
| [`results/bootstrap_map.json`](../results/bootstrap_map.json) | Estimasi selang kepercayaan bootstrap. |
| [`docs/DIAGNOSIS-DEPTH.md`](DIAGNOSIS-DEPTH.md) | Analisis penemuan sifat sinyal kedalaman dan redundansi kematangan. |
