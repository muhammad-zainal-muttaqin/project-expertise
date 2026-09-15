# Laporan Kesiapan Monev — Pengembangan Perangkat Mobile dengan Teknologi Depth Sensor untuk Penghitungan dan Klasifikasi Tandan Kelapa Sawit Berbasis Deep Learning

**Tanggal penyusunan:** 15 September 2026  
**Monev dijadwalkan:** Rabu, 23 September 2026, pukul 10.00–11.00 WITA  
**Penyaji:** Bu Fatma (Dosen Ilmu Komputer)  
**Disusun oleh:** Muhammad Zainal Muttaqin (Asisten Dosen)  
**Sumber data:** Repositori `project-expertise`, komit terakhir per 15 September 2026

---

## Daftar Isi

1. [Ringkasan Eksekutif](#1-ringkasan-eksekutif)
2. [Apa Tujuan Penelitian Ini?](#2-apa-tujuan-penelitian-ini)
3. [Seberapa Jauh Penelitian Sudah Berjalan?](#3-seberapa-jauh-penelitian-sudah-berjalan)
4. [Berapa Angka Kinerja Model Terbaik?](#4-berapa-angka-kinerja-model-terbaik)
5. [Di Mana Letak Kelemahan Utama?](#5-di-mana-letak-kelemahan-utama)
6. [Apakah Sensor Kedalaman Memberikan Manfaat?](#6-apakah-sensor-kedalaman-memberikan-manfaat)
7. [Upaya Apa Saja yang Sudah Dilakukan?](#7-upaya-apa-saja-yang-sudah-dilakukan)
8. [Ceklist Kesiapan Monev](#8-ceklist-kesiapan-monev)
9. [Visualisasi Data Pendukung](#9-visualisasi-data-pendukung)
10. [Lampiran Teknis](#10-lampiran-teknis)

---

## 1. Ringkasan Eksekutif

Penelitian ini bertujuan mengembangkan perangkat mobile yang menggunakan sensor kedalaman (*depth sensor*) untuk menghitung dan mengklasifikasikan tandan buah segar (TBS) kelapa sawit secara otomatis dengan pendekatan *deep learning*. Inti sistem adalah model kecerdasan buatan yang menerima citra RGB dan data kedalaman dari empat sisi pohon, kemudian mendeteksi lokasi tandan, menggabungkan deteksi antarsisi agar tidak terjadi penghitungan ganda, menentukan tingkat kematangan (B1–B4), dan menghasilkan jumlah tandan per kelas.

Setelah menjalankan lebih dari 80 konfigurasi eksperimen yang terdokumentasi, proyek telah menghasilkan capaian berikut:

- **Deteksi lokasi tandan** yang cukup kuat: F1 fisik 0,84–0,85.
- **Klasifikasi kematangan** yang belum memenuhi target: akurasi 74,4% pada korpus RGB (target internal 75%) dan 81,6% pada korpus *depth*.
- **Pencacahan total** yang mendekati acuan: galat rata-rata 0,77–1,36 tandan per pohon.
- **Pencacahan per kelas** yang masih menyimpang besar: kelas B2 kurang terhitung hingga 41%, sementara B3 berlebih 17% pada korpus RGB.
- **Manfaat sensor kedalaman** yang belum konsisten: terbukti meningkatkan lokalisasi, tetapi redundan untuk klasifikasi kematangan.
- **Aplikasi mobile dan integrasi perangkat** yang belum terverifikasi dari repositori ini.

Dokumen ini menyajikan seluruh data secara lengkap dan transparan.

---

## 2. Apa Tujuan Penelitian Ini?

Judul penelitian menetapkan empat komponen utama yang harus terpenuhi:

| No. | Komponen | Pertanyaan Kunci |
|-----|----------|-----------------|
| 1 | **Perangkat mobile** | Apakah sudah tersedia perangkat yang dapat digunakan di lapangan? |
| 2 | **Teknologi depth sensor** | Apakah sensor kedalaman memberikan nilai tambah yang terukur? |
| 3 | **Penghitungan tandan** | Apakah sistem mampu menghitung jumlah tandan per pohon dengan akurat? |
| 4 | **Klasifikasi kematangan** | Apakah sistem mampu menentukan tingkat kematangan B1–B4 dengan tepat? |

Keempat komponen di atas saling bergantung: perangkat mobile adalah wadah pengiriman, sensor kedalaman adalah modalitas tambahan, dan penghitungan serta klasifikasi adalah kemampuan inti yang bergantung pada kualitas model *deep learning*. Laporan ini berfokus pada komponen 2, 3, dan 4 karena ketiganya merupakan substansi ilmiah yang memerlukan pembuktian empiris. Komponen 1 (aplikasi mobile) bersifat teknik integrasi yang secara arsitektur sudah dirancang dan dapat diselesaikan setelah model inti terbukti.

---

## 3. Seberapa Jauh Penelitian Sudah Berjalan?

### 3.1 Komponen yang Sudah Selesai

| Komponen | Bukti | Rujukan |
|----------|-------|---------|
| **Dataset berlabel** — tiga korpus RGB dan RGB+D dengan anotasi kematangan B1–B4 | 953 pohon (18.540 kotak), 763 pohon *depth* (data diperluas), 1.716 pohon gabungan (1.364 pohon fisik unik) | `docs/DATASET.md` |
| **Tiga arsitektur detektor** — YOLO26l, RT-DETR-L, RF-DETR-L | Dilatih pada resolusi 1.280 px, dievaluasi pada partisi uji terpisah | `experiments/EKSPERIMEN.md`, `V2-E-001` |
| **Ensembel WBF** (*Weighted Box Fusion*) | AP50 lokalisasi agnostik-kelas: 0,8350 (953), 0,8764 (763) | `metrics/recap.md` |
| **Pipeline empat sisi** — penaut lintas-sisi, deduplikasi, pencacahan | Beberapa metode penaut (Hungarian, GSP MILP, prior rotasi, *learned edge*) | `PROPOSAL-Pipeline.md` |
| **Reproyeksi data kedalaman** | Menggunakan kalibrasi sensor Orbbec, intrinsik/ekstrinsik dua kamera, koreksi distorsi Brown–Conrady | `docs/NEW763_RGBD4_RESULTS.md` |
| **Evaluasi terkontrol RGB vs RGB+D** | Protokol *bootstrap* berpasangan, *seed* tetap, partisi pohon bukan citra | Berbagai artefak JSON di `results/` |
| **Artefak evaluasi terkunci uji** | JSON metrik per pohon dengan selang kepercayaan 95% | `results/remote_eval_2026-08-28/` |
| **Log eksperimen kronologis** | Lebih dari 80 simpul eksperimen (`V2-E-001`–`V2-E-048`, `PT-E-001`–`PT-E-036`, `AF-E-001`–`AF-E-016`) | `experiments/EKSPERIMEN.md`, `HANDOFF.md` |

### 3.2 Komponen yang Belum Selesai atau Belum Terverifikasi

| Komponen | Kondisi | Dampak |
|----------|---------|--------|
| **Aplikasi mobile** | Arsitektur sudah dirancang; implementasi belum ditemukan di repositori | Belum dapat didemonstrasikan sebagai perangkat terintegrasi |
| **Pemeriksaan kualitas foto** | Modul penolakan foto buram, sisi kurang, atau urutan keliru belum selesai | Kualitas masukan belum terjamin secara otomatis |
| **Pengukuran latensi pada perangkat** | Belum ada bukti pengukuran waktu inferensi pada perangkat sasaran | Kelayakan waktu-nyata belum diketahui |
| **Antarmuka pengguna** | Rancangan keluaran sudah didokumentasikan; implementasi belum terverifikasi | Pengguna belum dapat berinteraksi dengan hasil |
| **Panduan pengambilan empat sisi** | Prosedur sudah terdokumentasi; penerapan dalam aplikasi belum terverifikasi | Konsistensi pengambilan data di lapangan belum terjamin |

---

## 4. Berapa Angka Kinerja Model Terbaik?

### 4.1 Model Deteksi Terbaik (Tugas per Citra)

Angka berikut merupakan hasil terbaik pada masing-masing tugas. Setiap baris dapat berasal dari konfigurasi yang berbeda.

| Tugas | Metrik | RGB 953 | Depth 763 | Catatan |
|-------|--------|---------|-----------|---------|
| Lokalisasi tanpa kelas | AP50 | 0,8419 | 0,8783 | WBF + *re-ranker* |
| Deteksi + klasifikasi sekaligus | mAP50 | 0,6012 | 0,6711 | RF-DETR-L (763: bank `combined1716`) |

**Cara membaca:** AP50 mengukur kemampuan menemukan lokasi tandan pada citra tunggal tanpa menilai kelasnya. mAP50 mengukur kemampuan mendeteksi sekaligus menentukan kelas B1–B4 dengan benar. Selisih antara AP50 dan mAP50 menunjukkan bahwa model sudah cukup baik dalam menemukan lokasi tandan, tetapi sering salah menentukan kelasnya.

### 4.2 Pipeline Lengkap Terbaik (Tugas per Pohon, Hasil Uji Terkunci)

Pipeline lengkap adalah rangkaian penuh dari deteksi pada empat sisi, penggabungan lintas-sisi, klasifikasi, hingga pencacahan. Angka berikut berasal dari evaluasi pada partisi uji yang tidak digunakan selama pengembangan profil.

| Metrik | Penjelasan | RGB 953 (135 pohon) | Depth 763 (110 pohon) |
|--------|------------|---------------------|----------------------|
| **F1 Fisik** | Keseimbangan antara presisi dan daya tangkap identifikasi tandan fisik | **0,8387** | **0,8534** |
| Presisi fisik | Proporsi prediksi tandan yang memang ada | 0,8444 | 0,8926 |
| Daya tangkap fisik | Proporsi tandan acuan yang berhasil ditemukan | 0,8331 | 0,8175 |
| **Akurasi klasifikasi** | Ketepatan kelas pada tandan yang berhasil dicocokkan | **74,42%** | **81,62%** |
| Jumlah tandan dicocokkan | Denominator akurasi klasifikasi | 1.118 tandan | 457 tandan |
| **Makro-F1 ujung ke ujung** | F1 rata-rata empat kelas, termasuk objek terlewat dan prediksi berlebih | **0,6034** | **0,6519** |
| **Galat rata-rata pencacahan** (MAE) | Rata-rata selisih mutlak jumlah total per pohon | **1,363** | **0,773** |
| Toleransi cacah ±1 | Proporsi pohon dengan galat total maksimal 1 tandan | 63,70% | 85,45% |
| Ketepatan jumlah total | Proporsi pohon dengan jumlah total persis tepat | 27,41% | 44,55% |
| **Ketepatan vektor empat kelas** | Proporsi pohon dengan jumlah B1, B2, B3, *dan* B4 semuanya tepat sekaligus | **5,19%** | **27,27%** |

**Konfigurasi pipeline:**
- RGB 953: Tiga detektor → WBF agnostik → Penaut Hungarian *Anchor A* → Pencacahan Ridge
- Depth 763: Tiga detektor → WBF agnostik → Penaut GSP MILP → Pencacahan Ridge

### 4.3 Selang Kepercayaan 95% Metrik Utama

Dihitung melalui simulasi *bootstrap* berpasangan sebanyak 2.000 ulangan (*random seed* 42):

| Metrik | RGB 953 | Depth 763 |
|--------|---------|-----------|
| F1 fisik | 0,8387 [0,8174; 0,8587] | 0,8534 [0,8301; 0,8761] |
| Akurasi klasifikasi | 0,7442 [0,7112; 0,7735] | 0,8162 [0,7765; 0,8556] |
| Makro-F1 ujung ke ujung | 0,6034 [0,5655; 0,6382] | 0,6519 [0,6046; 0,6918] |
| MAE cacah total | 1,363 [1,163; 1,585] | 0,773 [0,609; 0,945] |
| Toleransi cacah ±1 | 0,6370 [0,5556; 0,7185] | 0,8545 [0,7818; 0,9182] |

Selang kepercayaan ini menunjukkan rentang ketidakpastian setiap estimasi titik. Angka akurasi klasifikasi 74,42% pada korpus 953 memiliki batas bawah 71,12% dan batas atas 77,35%, yang berarti target 75% belum secara meyakinkan tercapai maupun secara meyakinkan gagal.

### 4.4 Bias Pencacahan per Kelas (Sasaran Utama Pencacahan)

Tabel berikut menunjukkan perbedaan antara jumlah yang diprediksi dan jumlah acuan untuk setiap kelas kematangan, dijumlahkan atas seluruh pohon uji.

**RGB 953 — Partisi Uji, 135 Pohon (Penaut Hungarian *Anchor A*):**

| Kelas | Prediksi | Acuan | Bias | Bias Relatif |
|-------|----------|-------|------|--------------|
| B1 (lewat matang) | 104 | 113 | −9 | −7,96% |
| B2 (matang optimal) | 145 | 246 | −101 | **−41,06%** |
| B3 (matang awal) | 824 | 706 | +118 | **+16,71%** |
| B4 (mentah) | 251 | 277 | −26 | −9,39% |
| **Total** | **1.324** | **1.342** | **−18** | **−1,34%** |

Makro-rerata nilai mutlak bias relatif: **18,78%**

**Depth 763 — Partisi Uji, 110 Pohon (Penaut GSP MILP):**

| Kelas | Prediksi | Acuan | Bias | Bias Relatif |
|-------|----------|-------|------|--------------|
| B1 (lewat matang) | 67 | 94 | −27 | −28,72% |
| B2 (matang optimal) | 227 | 199 | +28 | +14,07% |
| B3 (matang awal) | 177 | 215 | −38 | −17,67% |
| B4 (mentah) | 41 | 50 | −9 | −18,00% |
| **Total** | **512** | **558** | **−46** | **−8,24%** |

Makro-rerata nilai mutlak bias relatif: **19,62%**

**Temuan penting:** Jumlah total yang mendekati acuan (bias −1,34% pada RGB) menyembunyikan kesalahan komposisi. Pada RGB, tandan B2 kurang terhitung 41% sementara B3 berlebih 17% — keduanya nyaris saling mengimbangi pada jumlah total, tetapi distribusi kematangan yang dilaporkan tidak mencerminkan kondisi sebenarnya. Pada Depth, bias tersebar lebih merata antarempat kelas, tetapi total masih kurang terhitung 8,24%.

### 4.5 Kinerja Makro-F1 per Kelas (Ujung ke Ujung)

| Kelas | RGB 953 | Depth 763 |
|-------|---------|-----------|
| B1 | 0,7465 | 0,7578 |
| B2 | 0,4706 | 0,7230 |
| B3 | 0,6850 | 0,7092 |
| B4 | 0,5114 | 0,4176 |

Kelas B2 pada RGB (0,47) dan B4 pada kedua korpus (0,51 dan 0,42) merupakan titik terlemah. B2 adalah tandan matang optimal yang menjadi sasaran utama panen — kesalahan pada kelas ini berdampak langsung terhadap keputusan operasional.

---

## 5. Di Mana Letak Kelemahan Utama?

### 5.1 Kelemahan Kinerja

| Prioritas | Kelemahan | Angka Terukur | Dampak |
|-----------|----------|---------------|--------|
| 1 | **Pencacahan per kelas tidak akurat** | Vektor tepat hanya 5,19–27,27% pohon; bias B2 sampai −41% | Distribusi kematangan yang dilaporkan tidak dapat diandalkan untuk keputusan panen |
| 2 | **Klasifikasi kematangan belum konsisten** | Akurasi 74,42% (953) belum mencapai target internal 75%; 25,58% tandan salah kelas | Satu dari empat tandan yang terdeteksi diklasifikasikan keliru |
| 3 | **Makro-F1 ujung ke ujung masih rendah** | 0,60–0,65 | Gabungan kesalahan deteksi, penautan, dan klasifikasi |
| 4 | **Manfaat depth belum konsisten** | Lokalisasi naik; klasifikasi dan pencacahan tidak selalu membaik | Belum dapat diklaim bahwa depth sensor memberikan keunggulan menyeluruh |

### 5.2 Kelemahan Metodologis

| Aspek | Penjelasan |
|-------|------------|
| **Pergeseran temporal antardataset** | Korpus 953 (Mei 2026) dan 352/763 (Juli 2026) memiliki distribusi kematangan yang sangat berbeda akibat jeda 80 hari (~5–11 siklus panen). Perbandingan 4-kelas lintas-dataset tidak valid secara metodologis. |
| **Daya statistik partisi uji terbatas** | Partisi uji 352 pohon hanya memiliki 410 kotak acuan; selang kepercayaan selebar ±0,058 membuat banyak selisih kecil tidak dapat dibedakan dari derau. |
| **Kebocoran partisi data** | 44 dari 55 pohon uji dataset 352 termuat di partisi latih dataset 953; 87% citra uji evaluasi agnostik beririsan dengan data prapelatihan. |
| **Reproduksi sebagian hasil historis** | Beberapa bobot model asli hilang dan dilatih ulang (selisih reproduksi ~0,005 mAP50); satu konfigurasi (*stacking* DINOv2-Large) tidak berhasil direproduksi (selisih 5,2 poin persentase). |

### 5.3 Sumber Kehilangan Kinerja Terbesar

Analisis dekomposisi galat (simpul V2-E-013) menunjukkan bahwa **44,5% kapasitas model tereduksi akibat kesalahan klasifikasi kelas ordinal**, bukan kegagalan melokalisasi tandan. Lokalisasi agnostik-kelas mencapai AP50 0,73–0,88, jauh di atas deteksi 4-kelas yang hanya 0,45–0,67. Artinya, model sudah cukup mampu menemukan di mana tandan berada, tetapi sering keliru menentukan tingkat kematangannya.

Temuan ini mengarahkan upaya perbaikan ke **peningkatan kemampuan klasifikasi kematangan**, bukan penambahan kapasitas detektor.

---

## 6. Apakah Sensor Kedalaman Memberikan Manfaat?

### 6.1 Bukti Positif

| Temuan | Angka | Rujukan |
|--------|-------|---------|
| Lokalisasi agnostik-kelas meningkat dengan depth Sobel `edge` | AP50: 0,7636 (RGB+D) vs 0,7358 (RGB), delta +0,0278 | `V2-E-024` |
| Pipeline depth menghasilkan F1 fisik dan pencacahan yang lebih baik | F1: 0,8534 vs 0,8387; MAE: 0,773 vs 1,363 | Profil uji terkunci |
| *Late fusion* RGB dan RGB+D4 memberikan sinyal yang saling melengkapi | YOLO union-WBF: +0,038 mAP50 pada validasi (CI positif) | `NEW763_RGBD4_RESULTS.md` |

### 6.2 Bukti Negatif atau Belum Meyakinkan

| Temuan | Angka | Rujukan |
|--------|-------|---------|
| *Early fusion* 4-kanal tidak konsisten: YOLO datar, RF turun, RT naik kecil | Seluruh CI 95% melintasi nol | `NEW763_RGBD4_RESULTS.md` |
| Kedalaman redundan untuk klasifikasi kematangan | Informasi kondisional $I(Y; D \mid \text{RGB}) \approx 0$ | `V2-E-016` |
| Kedalaman monokular merugikan pada 953 | Delta −0,0476, CI [−0,0671; −0,0274] — **kalah signifikan** | `V2-E-029` |
| Menambah kanal kelima (sensor + monokular) mengencerkan sinyal | Delta −0,0504, CI [−0,1038; −0,0015] — **kalah signifikan** | `V2-E-030` |
| Selisih lokalisasi depth vs RGB belum signifikan secara statistik pada 95% | CI delta [−0,0121; +0,0648], $P(\Delta>0)$ = 92,1% | `bootstrap_lokalisasi.json` |

### 6.3 Kesimpulan Kontribusi Depth

Sensor kedalaman memberikan kontribusi yang **terukur tetapi terbatas**:

- **Membantu lokalisasi**: sinyal fisik depth berhasil menembus batas yang sebelumnya diperkirakan sebagai limit dataset. Namun, bukti ini belum signifikan pada tingkat kepercayaan 95%.
- **Tidak membantu klasifikasi**: fitur kedalaman bersifat redundan terhadap fitur visual RGB untuk menentukan kematangan. Warna dan tekstur tandan sudah membawa informasi kematangan yang memadai.
- **Potensi *late fusion***: kombinasi prediksi model RGB dan RGB+D yang memiliki pola kesalahan berbeda memberikan sinyal yang menjanjikan pada validasi, tetapi belum dikonfirmasi pada data uji baru.

---

## 7. Upaya Apa Saja yang Sudah Dilakukan?

Tabel ini merangkum seluruh kategori pendekatan yang telah diuji beserta hasilnya, untuk menunjukkan bahwa penelitian telah menjelajahi ruang solusi secara menyeluruh.

### 7.1 Arsitektur dan Pelatihan Detektor

| Pendekatan | Eksperimen | Hasil |
|------------|-----------|-------|
| Tiga keluarga detektor (YOLO, RT-DETR, RF-DETR) | V2-E-001 | RF-DETR-L unggul mAP50; RT-DETR-L paling rapuh terhadap pergeseran domain |
| Resolusi 960 px vs 1.280 px | AF-E-006 | 1.280 px lebih baik untuk mAP50 |
| Pelatihan pada bank `combined1716` (1.716 pohon) | V2-E-035–041 | Meningkatkan AP50; menambah data tidak otomatis menaikkan semua model |
| *Weighted WBF* dengan bobot berbeda per detektor | V2-E-043 | Menaikkan sebagian mAP tetapi menurunkan F1 fisik hilir |
| *Deep-tail proposal* dan pemeringkat ulang $p_{tp}$ | MAP_BOOST | AP50 naik 0,8350 → 0,8419 (953); CI mendukung kenaikan |

### 7.2 Representasi dan Fusi Kedalaman

| Pendekatan | Eksperimen | Hasil |
|------------|-----------|-------|
| Invers mentah (*inverse raw*) | V2-E-005 | Tidak konsisten: YOLO naik, RT dan RF turun |
| Gradien Sobel (`edge`) | V2-E-008, V2-E-010 | Terbaik untuk lokalisasi; tidak berpengaruh pada pencacahan |
| *Dropout*, *clipped*, *valid mask* | V2-E-009 | Kalah dari `edge` pada penyaringan 15 *epoch* |
| *Mid-fusion* dengan cabang terpisah dan *gate* | V2-E-009 | Gagal: puncak hanya 0,2087 |
| *Early fusion* 4-kanal pada 763 | V2-E-034 | Seluruh CI melintasi nol; tidak signifikan |
| *Late fusion* union-NMS dan WBF | Follow-up | Menjanjikan pada validasi; WBF tidak universal |
| Kedalaman monokular (*monocular depth*) | V2-E-027–032 | Dua kali kalah signifikan; ditolak |
| Lima kanal (sensor + monokular) | V2-E-030 | Kalah signifikan; mengencerkan sinyal |

### 7.3 Penautan Lintas-Sisi dan Deduplikasi

| Pendekatan | Eksperimen | Hasil |
|------------|-----------|-------|
| *Union-Find* dan Hungarian awal | PT-E-002 | F1 linker hanya 0,4282; tidak cukup |
| Prior arah rotasi searah jarum jam | PT-E-008 | F1 linker naik 0,40 → 0,65; **keberhasilan penting** |
| Re-ID embedding | PT-E-003 | Bocor (AUC train 1,0; test 0,72); diperbaiki tetapi belum menutup target |
| GNN untuk penautan | PT-E-017 | F1 naik ke 0,38; tetap belum cukup |
| *Learned edge linker* dengan 65 fitur | HANDOFF | AUC 0,95 pada validasi; belum dikonfirmasi uji |
| Greedy strict + sweep parameter | V2-E-043 | F1 0,83–0,86; parameter dari test (bukan generalisasi) |
| GSP MILP (*global set-partition*) | V2-E-045 | Kenaikan F1 fisik yang nyata; constraint 1 deteksi per sisi |
| Rekonstruksi 3D kaku | PT-E-009–013 | AUC 0,45–0,51 (setara acak); ditolak |

### 7.4 Klasifikasi Kematangan

| Pendekatan | Eksperimen | Hasil |
|------------|-----------|-------|
| Klasifikasi langsung dari detektor (satu tahap) | V2-E-001 | mAP50 sebagai ukuran gabungan; akurasi kelas terbatas |
| ConvNeXt-Tiny / crop classifier | PT-E-012, 014 | Mengalahkan detektor pada benchmark crop, tetapi tidak bertahan ujung ke ujung |
| CORAL ordinal loss | PT-E-014 | Runtuh ke 33,05% pada subtes; CORN lebih baik (69,83%) |
| DINOv2-Large *stacking* empat pakar | V2-E-046 | Validasi 76,8%; reproduksi hanya 71,6% (selisih tidak terjelaskan) |
| Komposisi lintas-lapis | V2-E-047 | Validasi 85,0% (depth); solver MILP hilang, tidak dapat direproduksi |
| Head-aware ranking | Wave 2 | Menaikkan akurasi kelas tetapi menurunkan F1 fisik |
| Kalibrasi kelas (*scale_macro*) | Wave 2 | Kandidat terbaik VAL; CI delta melintasi nol |
| ConvNeXt-Small, Swin-Tiny, EfficientNetV2-S | Wave 2 | Kenaikan nominal; satu pohon lebih baik dari anchor |
| Augmentasi fotometrik (hue, CLAHE, gray-world, dll.) | Ditolak | Tidak memberi perbaikan menyeluruh |
| Pembobotan kelas, *focal loss*, *mixed loss* | Ditolak | Tidak menyelesaikan konflik tujuan deteksi vs klasifikasi |

### 7.5 Pencacahan dan Rekonsiliasi

| Pendekatan | Eksperimen | Hasil |
|------------|-----------|-------|
| Jumlah pool mentah | PT-E-004 | Macro-MAE 3,34; **gagal telak** |
| Ridge Regression + $F_{\text{all}}$ | V2-E-045 | MAE 1,36 (953), 0,77 (763); lebih baik |
| Rekonsiliasi jumlah per kelas | PT-E-005–007 | Memperbaiki bias tetapi tidak menyembuhkan identitas tandan |
| Count meta-ensemble | Wave 2 | Matched naik, MAE/±1 turun; ditolak sebagai *all-rounder* |
| Regresor kaya fitur non-linear | Wave 2 | CV train membaik; VAL memburuk |

### 7.6 Rangkuman: Ruang Solusi yang Sudah Dijelajahi

Penelitian ini telah menguji **3 keluarga detektor, 6+ representasi kedalaman, 8+ metode penautan, 10+ pengklasifikasi dan teknik fusi, dan 5+ pendekatan pencacahan**, dengan total lebih dari 2.800 baris evaluasi pada wave terakhir saja. Pendekatan yang gagal tidak dihapus, melainkan didokumentasikan sebagai studi ablasi dan kontrol negatif. Ketidakmampuan menemukan konfigurasi yang menaikkan semua metrik secara bersamaan bukan karena kurangnya eksplorasi, melainkan karena **konflik inheren antara tujuan lokalisasi, identitas fisik, klasifikasi kelas, dan pencacahan**.

---

## 8. Ceklist Kesiapan Monev

### 8.1 Komponen yang Tersedia untuk Disajikan

| Status | Komponen | Keterangan |
|--------|----------|------------|
| Tersedia | Dataset RGB dan Depth | Tiga korpus, pembagian partisi terdokumentasi |
| Tersedia | Tiga arsitektur detektor | YOLO26l, RT-DETR-L, RF-DETR-L — terlatih dan terevaluasi |
| Tersedia | Ensembel WBF | AP50 lokalisasi 0,84–0,88 |
| Tersedia | Pipeline empat sisi | Penaut, deduplikasi, pencacahan — beberapa varian |
| Tersedia | Reproyeksi kedalaman | Kalibrasi Orbbec, koreksi distorsi |
| Tersedia | Perbandingan RGB vs RGB+D | Terkontrol, *bootstrap* berpasangan |
| Tersedia | Evaluasi terkunci uji | JSON terlacak, selang kepercayaan tersedia |

### 8.2 Komponen yang Belum Memenuhi Target atau Belum Terverifikasi

| Status | Komponen | Keterangan |
|--------|----------|------------|
| Belum memenuhi | Akurasi klasifikasi ≥75% | 74,4% pada 953; 81,6% pada 763 |
| Belum memenuhi | Pencacahan per kelas yang akurat | Bias B2: −41%; vektor tepat hanya 5–27% |
| Belum konsisten | Manfaat depth sensor | Lokalisasi naik; klasifikasi redundan |
| Belum terverifikasi | Aplikasi mobile | Belum ditemukan implementasi di repositori |
| Belum terverifikasi | Demonstrasi ujung ke ujung | Belum ada bukti sistem terintegrasi berjalan |

---

## 9. Visualisasi Data Pendukung

Dokumen ini disertai lima grafik pendukung yang tersimpan di `monev-assets/`:

1. **`01_perbandingan_pipeline.png`** — Perbandingan seluruh metrik pipeline terbaik antara RGB 953 dan Depth 763.
2. **`02_bias_pencacahan_kelas.png`** — Bias pencacahan per kelas B1–B4 pada kedua korpus.
3. **`03_rgb_vs_depth.png`** — Kinerja per tugas RGB vs Depth.
4. **`04_ceklist_kesiapan.png`** — Status komponen yang sudah dan belum selesai.
5. **`05_selang_kepercayaan.png`** — Selang kepercayaan 95% metrik utama.

---

## 10. Lampiran Teknis

### 10.1 Karakteristik Dataset

| Parameter | SawitMVC-YOLO (953) | SawitMVC-Depth (763) | Combined1716 |
|-----------|--------------------|-----------------------|--------------|
| Jumlah pohon | 953 | 763 (v2.0.0) | 1.716 (1.364 unik) |
| Jumlah citra | 3.992 | ~3.052 | ~6.864 |
| Jumlah kotak anotasi | 18.540 | ~5.000+ | ~23.000+ |
| Resolusi | 960×1.280 px (potret) | 1.280×800 px (lanskap) | Campuran |
| Modalitas kedalaman | Tidak tersedia | Sensor Orbbec Y16 (848×480 px, mm) | Sebagian |
| Tanggal akuisisi | 30 Apr – 16 Mei 2026 | 28–29 Jul 2026 | Gabungan |
| Partisi uji | 141 pohon | 110 pohon (evaluasi pipeline) | 1.052 citra uji |

### 10.2 Karakteristik Visual Kelas Kematangan

| Kelas | Tingkat Kematangan | Karakteristik Visual | Ukuran Median (px) |
|-------|--------------------|---------------------|-------------------|
| B1 | Lewat matang (siap panen) | Jingga kemerahan cerah, posisi terbawah kanopi | 133 |
| B2 | Matang optimal (siap panen) | Oranye kemerahan bersemburat ungu kehitaman | 120 |
| B3 | Matang awal / mengkal | Ungu kemerahan kehitaman | 107 |
| B4 | Mentah / muda | Hitam kehijauan pekat, di sela pelepah | 93 |

Kelas B2 dan B3 memiliki perbedaan visual yang halus (keduanya berwarna gelap keunguan), yang menjelaskan sebagian besar kesalahan klasifikasi terjadi antara kedua kelas ini. Ukuran median B4 yang paling kecil (93 px) juga menyulitkan deteksi karena resolusi fitur yang terbatas.

### 10.3 Arsitektur Pipeline

```
4 foto terarah (searah jarum jam)
  │
  ├─ Pemeriksaan kualitas (belum selesai)
  │
  ├─ Deteksi: YOLO26l + RT-DETR-L + RF-DETR-L
  │     resolusi masukan: 1.280 px
  │     keluaran: kotak pembatas + probabilitas B1–B4
  │
  ├─ Fusi proposal: WBF agnostik-kelas (IoU 0,60)
  │     keluaran: proposal tandan fisik tanpa label kelas
  │
  ├─ Penautan lintas-sisi
  │     metode: Hungarian Anchor A (953) / GSP MILP (763)
  │     fitur: posisi, ukuran, prior rotasi, kemiripan kelas
  │     constraint: maks 1 deteksi per sisi per tandan
  │
  ├─ Klasifikasi per tandan
  │     keluaran: probabilitas B1–B4 per tandan fisik
  │
  ├─ Pencacahan: Ridge Regression + rekonsiliasi
  │     keluaran: jumlah B1, B2, B3, B4 per pohon
  │
  └─ Laporan (belum selesai)
```

### 10.4 Rujukan Berkas Utama

| Topik | Berkas |
|-------|--------|
| Rekap metrik dan papan peringkat | `metrics/recap.md` |
| Hasil pipeline uji terkunci RGB 953 | `results/remote_eval_2026-08-28/gsp_artifacts/953/results_test_locked.json` |
| Hasil pipeline uji terkunci Depth 763 | `results/remote_eval_2026-08-28/gsp_artifacts/depth/results_test_locked.json` |
| Log eksperimen kronologis | `experiments/EKSPERIMEN.md` |
| Status evaluasi | `experiments/STATUS.md` |
| Handoff dan riwayat progres | `HANDOFF.md` |
| Laporan akhir Volume 2 | `docs/LAPORAN-AKHIR.md` |
| Hasil RGB+D 4-kanal | `docs/NEW763_RGBD4_RESULTS.md` |
| Selang kepercayaan lokalisasi | `results/bootstrap_lokalisasi.json` |
| Proposal arsitektur pipeline | `PROPOSAL-Pipeline.md` |
| Audit forensik | `docs/AUDIT-FORENSIK-2026-09-06.md` |

---

*Laporan ini disusun berdasarkan pembacaan menyeluruh terhadap repositori `project-expertise` per 15 September 2026. Seluruh angka merujuk langsung ke artefak JSON dan dokumen Markdown yang tercantum. Tidak ada angka yang dibulatkan atau diubah dari sumber aslinya.*
