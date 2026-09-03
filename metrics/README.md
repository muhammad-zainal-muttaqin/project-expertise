# Atlas Metrik & Rangkuman Hasil Penelitian Terpadu

Selamat datang di **Atlas Metrik dan Rangkuman Penelitian Komprehensif** repositori `project-expertise` (Volume 2). Dokumen ini berfungsi sebagai gerbang utama (*master index*) dan atlas komparatif terstruktur yang merangkum seluruh hasil eksperimen, evaluasi empiris, ablasi arsitektur, dan pengembangan pipeline dari awal penelitian hingga iterasi mutakhir.

Atlas ini dirancang agar seluruh data kuantitatif dapat ditelusuri (*traceable*), diperbandingkan secara konsisten antar-eksperimen, serta dilengkapi konteks validitas ilmiah, batasan statistik, dan rujukan berkas sumber langsung.

---

## 1. Ringkasan Eksekutif & Statistik Global

| Parameter Inventaris | Nilai Terukur / Cakupan | Keterangan Metodologis |
|---|---|---|
| **Total Simpul Eksperimen Terlacak** | **85 Simpul** | 45 simpul Volume 2 (`V2-E-001`–`V2-E-045`), 36 simpul pipeline pertandan (`PT-E-000`–`PT-E-036`), dan 4 gelombang verifikasi/ablasi mutakhir (`remote_eval_2026-08-27`, `validation_wave_2026-08-28`, `new763_rgbd4`, dan `pipeline_damimas`). |
| **Rentang Tanggal Penelitian** | **09 Agustus 2026 – 30 Agustus 2026** | Fondasi tinjauan pustaka & eksplorasi awal bersumber dari Volume 1 (Mei–Juli 2026). |
| **Arsitektur Detektor Utama** | **YOLO26l, RT-DETR-L, RF-DETR-L** | Detektor berbasis satu-tahap konvensional, Transformer hibrida waktu-nyata, dan arsitektur DETR modern. |
| **Modalitas Citra Diuji** | **RGB, RGB+Depth (Sensor Y16), RGB+Mono (Depth Monokular), RGB+Depth+Mono (5-kanal)** | Evaluasi perbandingan *early fusion*, *mid-fusion*, dan fusi lanjut (*late fusion*). |
| **Korpus & Partisi Data** | **SawitMVC-YOLO (953 pohon), SawitMVC-Depth (352 & 763 pohon), Combined-1716 (1.716 pohon)** | Pembagian partisi terkontrol (*train, val, test*) untuk mencegah kebocoran partisi data (*data leakage*). |
| **Status Validitas Ilmiah** | **52 VALID, 18 FALSIFIED / INVALID, 9 SUPERSEDED, 4 INCOMPLETE / SCREENING, 2 RETRACTED / AUDITED** | Klasifikasi ketat berdasarkan keterpenuhan kontrol variabel dan pengujian hipotesis statistik. |

### Taksonomi Status Validitas Bukti
Setiap baris hasil eksperimen dalam atlas ini diberi label validitas resmi:
1. `VALID`: Eksperimen memenuhi kontrol variabel metodologis, partisi data terkunci (*validation/test-locked*), dan kesimpulan didukung data empiris.
2. `FALSIFIED`: Hipotesis eksperimen diuji secara sah namun tertolak oleh bukti empiris (misal kanal kedalaman tidak meningkatkan klasifikasi kematangan).
3. `INVALID`: Rancangan pengujian mengandung cacat metodologis mendasar yang membatalkan perbandingan ilmiah (misal perbandingan 4-kelas lintas dataset akibat pergeseran temporal $\sim 80	ext{ hari}$ pada `V2-E-022`).
4. `SUPERSEDED`: Hasil empiris valid pada fasenya, namun telah digantikan oleh model, parameter, atau pipeline generasi berikutnya yang lebih optimal.
5. `INCOMPLETE`: Eksperimen penyaringan cepat (*screening* pendek $\le 15	ext{ epoch}$) atau dihentikan sebelum konvergen penuh.
6. `RETRACTED` / `CORRECTED`: Klaim atau angka awal yang ditarik/dikoreksi setelah audit audit kebocoran data atau galat perkakas evaluasi (misal audit pretrain `agn953_full` pada `V2-E-025` dan audit TIFF korup pada `V2-E-028`).

---

## 2. Struktur Atlas & Navigasi Berkas Spesialisasi

Untuk memudahkan penelaahan mendalam per domain tugas, atlas ini dipecah ke dalam 7 berkas spesialisasi berikut:

| Berkas Spesialisasi | Cakupan Domain & Metrik Utama |
|---|---|
| **[01_deteksi_dan_lokalisasi.md](01_deteksi_dan_lokalisasi.md)** | Komparasi detektor 1-tahap (RGB vs RGB+D vs Mono), studi *encoding* kedalaman (Invers, Sobel `edge`), fusi WBF, TTA, *class-aware* ($mAP50$, $mAP50	ext{--}95$, per kelas B1–B4) vs *class-agnostic* ($AP50_{agn}$). |
| **[02_klasifikasi_kematangan.md](02_klasifikasi_kematangan.md)** | Model pengklasifikasi kematangan pada citra terpotong (*crop*), loss ordinal (CORAL, CORN), evaluasi multi-tampak (C1–C3), *mixture-of-experts*, aturan keputusan per tandan (R0–R4), akurasi, akurasi toleransi $\pm 1$, dan MAE ordinal. |
| **[03_pencacahan_per_pohon.md](03_pencacahan_per_pohon.md)** | Pencacahan jumlah tandan per pohon via Ridge regression ($F_{all}$), penghitung pool fisik, model multi-bank, CatBoost regularized, meta-ensemble cacah, Macro MAE, Total MAE, Bias bertanda, dan akurasi $\pm 1$. |
| **[04_pengaitan_multi_tampak.md](04_pengaitan_multi_tampak.md)** | Sistem penaut (*linker*) asosiasi spasial antar-sudut pandang kanonik (Hungarian, Union-Find, GNN, Global Linker DAMIMAS, GSP Linker), F1 pasangan, *Adjusted Rand Index* (ARI), cakupan tandan multi-sisi, dan fraksi pool palsu. |
| **[05_pipeline_end_to_end.md](05_pipeline_end_to_end.md)** | Evaluasi sistem pipeline terpadu (Deteksi $	o$ Asosiasi Multi-Tampak $	o$ Klasifikasi $	o$ Rekonsiliasi Cacah), F1 deteksi fisik, MAE pencacahan per pohon, *matched class accuracy*, dan macro-F1 *end-to-end*. |
| **[06_komputasi_dan_diagnostik.md](06_komputasi_dan_diagnostik.md)** | Kompleksitas komputasi (jumlah parameter, GFLOPs, durasi pelatihan/inferensi), diagnostik fisik sinyal kedalaman (*relief SNR*, kuantisasi, uji Kruskal–Wallis, korelasi Spearman), audit derau TIFF, serta selang kepercayaan *bootstrap* 95%. |
| **[07_buku_besar_eksperimen.md](07_buku_besar_eksperimen.md)** | Buku besar (*master ledger*) kronologis seluruh simpul eksperimen `V2-E-001` s.d. `V2-E-045`, `PT-E-000` s.d. `PT-E-036`, serta rangkaian verifikasi remote dengan status pembuktian lengkap. |

---

## 3. Standar Normalisasi & Aturan Pembacaan Metrik

Guna mencegah kerancuan dan salah tafsir lintas domain, seluruh metrik dalam repositori dinormalisasi mengikuti kaidah baku berikut:

1. **Pembedaan Penyebut dan Unit Evaluasi:**
   - **Tingkat Kemunculan (*Per-Appearance / Box*):** Menilai kualitas deteksi kotak pembatas pada citra tunggal ($mAP50$, $mAP50	ext{--}95$, $AP_{agn}$).
   - **Tingkat Tandan Fisik (*Per-Physical Bunch*):** Menilai ketepatan klaster gabungan multi-tampak yang merepresentasikan 1 tandan buah riil di pohon (F1 fisik, *matched class accuracy*).
   - **Tingkat Pohon (*Per-Tree*):** Menilai akurasi agregasi cacah buah per pohon (Macro MAE, Total MAE, Akurasi Pohon $\pm 1$).
   - **Tingkat Pasangan (*Per-Pair*):** Menilai ketepatan relasi biner antara dua kotak dari sudut pandang berbeda (Presisi/Recall/F1 Pasangan, ARI).

2. **Pencegahan Penyamaan Metrik Berbeda Makna:**
   - `$AP50_{agnostic}$` adalah plafon lokalisasi spasial tanpa label kelas; **tidak boleh disamakan** dengan akurasi klasifikasi B1–B4 atau $mAP50$ 4-kelas.
   - `class_pm1_acc` pada modul pencacahan mengukur persentase sel pohon–kelas dengan galat cacah $\le 1$; **bukan akurasi klasifikasi citra biasa**.
   - Metrik bertanda minus tipografis asli `−` (seperti degradasi performa $\Delta = −0,0476$ atau bias cacah $−0,15$) menunjukkan penurunan/galat *under-predict*, bukan tanda hubung teks.

3. **Notasi Matematika & Simbol:**
   - Desimal menggunakan koma (misal: $0,6012$), pemisah ribuan menggunakan titik (misal: $3.992	ext{ citra}$).
   - Selang kepercayaan 95% ditulis dalam format kurung siku bertitik koma: $[min; max]$ (misal: $[−0,0671; −0,0274]$).
   - Seluruh selisih performa $\Delta$ yang selang kepercayaannya mencakup nilai nol dinyatakan **tidak signifikan secara statistik**.
   - Nilai metrik yang tidak tercatat/tersedia diisi secara konsisten dengan simbol `−`.

---

## 4. Tabel Master Ringkas Komparasi Utama

Berikut adalah intisari perbandingan kuantitatif pada simpul-simpul eksperimen kunci di setiap kategori.

### 4.1 Deteksi dan Lokalisasi Kotak Pembatas
*Evaluasi menggunakan `pycocotools.COCOeval` pada partisi data uji kanonik.*

| ID Simpul | Konfigurasi Model & Modalitas | Dataset & Partisi | $mAP50$ | $mAP50	ext{--}95$ | $AP_{B1}$ | $AP_{B2}$ | $AP_{B3}$ | $AP_{B4}$ | $AP50_{agn}$ | Status Bukti | Rujukan Artefak |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `V2-E-024` | YOLO26l (RGB+Sobel `edge`, Agnostik) | SawitMVC-Depth (352, Test) | − | − | − | − | − | − | **0,7636** | `VALID` | `results/V2-E-024/` |
| `V2-E-042` | RF-DETR-L Remote Combined | SawitMVC-Depth Test (110 pohon) | **0,6711** | − | − | − | − | − | **0,8764** | `VALID` | `results/remote_eval_2026-08-27/` |
| `V2-E-013` | RF-DETR-L (RGB, Agnostik) | SawitMVC-Depth (352, Test) | − | − | − | − | − | − | **0,6677** | `VALID` | `scripts/eval_twostage.py` (nilai lama `0,7330` keliru — itu angka V2-E-017, bukan V2-E-013) |
| `V2-E-034` | RF-DETR-L (RGB, Seed 42) | SawitMVC-Depth v2 (763, Test) | **0,6129** | **0,3172** | **0,6988** | **0,6541** | **0,6422** | **0,4567** | 0,7951 | `VALID` | `results/riwayat_epoch_new763/` |
| `V2-E-001` | RF-DETR-L (RGB) | SawitMVC-YOLO (953, Test) | **0,6012** | **0,2747** | **0,8150** | **0,5184** | **0,6553** | **0,4160** | − | `VALID` | `results/perkelas_pycoco_v2repro.json` |
| `V2-E-035` | RF-DETR-L (RGB, Seed 42) | Combined-1716 (1716, Test) | **0,5960** | **0,2831** | **0,7612** | **0,5580** | **0,6520** | **0,4128** | 0,7934 | `VALID` | `results/riwayat_epoch_combined1716/` |
| `V2-E-039` | Ensembel WBF 3-Detektor | Combined-1716 (1716, Test) | 0,5861 | 0,2770 | − | − | − | − | **0,8106** | `VALID` | `results/V2-E-039/` |
| `V2-E-001` | RT-DETR-L (RGB) | SawitMVC-YOLO (953, Test) | 0,5781 | 0,2629 | 0,7874 | 0,4614 | 0,6371 | 0,4266 | − | `VALID` | `results/perkelas_pycoco_v2repro.json` |
| `V2-E-001` | YOLO26l (RGB) | SawitMVC-YOLO (953, Test) | 0,5435 | 0,2564 | 0,7705 | 0,4479 | 0,6050 | 0,3506 | − | `VALID` | `results/perkelas_pycoco_v2repro.json` |
| `V2-E-027` | YOLO26l (RGB+Mono 4-ch) | SawitMVC-YOLO (953, Test) | 0,4960 | 0,2241 | 0,7212 | 0,3960 | 0,5612 | 0,3055 | − | `FALSIFIED` | `results/V2-E-027/` |
| `V2-E-003` | RF-DETR-L (RGB) | SawitMVC-Depth (352, Test) | **0,4544** | **0,1599** | **0,6853** | **0,5184** | **0,3477** | **0,2661** | − | `VALID` | `results/perkelas_pycoco_rgb352.json` |
| `V2-E-003` | RT-DETR-L (RGB) | SawitMVC-Depth (352, Test) | 0,4343 | 0,1503 | 0,7680 | 0,4867 | 0,2641 | 0,2185 | − | `VALID` | `results/perkelas_pycoco_rgb352.json` |
| `V2-E-010` | YOLO26l (RGB+Sobel `edge` 4-ch) | SawitMVC-Depth (352, Test) | **0,4316** | 0,1441 | 0,7252 | 0,5031 | 0,2240 | 0,2740 | − | `VALID` | `results/perkelas_pycoco_rgbd352.json` (kunci `YOLO26l-RGBD-edge`) |
| `V2-E-005` | YOLO26l (RGB+D Invers 4-ch) | SawitMVC-Depth (352, Test) | 0,3919 | 0,1408 | 0,6857 | 0,4579 | 0,2637 | 0,1601 | − | `SUPERSEDED` | `results/perkelas_pycoco_rgbd352.json` |
| `V2-E-003` | YOLO26l (RGB) | SawitMVC-Depth (352, Test) | 0,3606 | 0,1246 | 0,6804 | 0,4320 | 0,2001 | 0,1299 | − | `VALID` | `results/perkelas_pycoco_rgb352.json` |

---

### 4.2 Klasifikasi Tingkat Kematangan Citra Terpotong (*Crop*)
*Evaluasi performa klasifikasi kematangan 4 kelas (B1–B4) pada citra terpotong.*

| ID Simpul | Metode / Model Pengklasifikasi | Modalitas Input | Dataset & Partisi | Akurasi | Akurasi $\pm 1$ | MAE Ordinal | Macro-F1 | Status Bukti | Rujukan Artefak |
|---|---|---|---|---:|---:|---:|---:|---|---|
| `PT-E-029` | Weighted Average Ensemble | Multi-Head Crop | DAMIMAS (Test, n=1.316) | **0,7439** | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | **0,7048** | `VALID` | `pipeline-pertandan/results/pt_e_029_ensemble_kelas_damimas.json` (`ensemble.macro_f1_test`) — Akurasi cocok sumber; Macro-F1 lama `0,7290` dikoreksi ke `0,7048`; ±1/MAE belum ditemukan pada level JSON yang dicek. |
| `PT-E-001` | R4 (agregasi ordinal, seluruh pool) | RGB Crop | DAMIMAS 953 (Test, n=1.269) | **0,7360** | **0,9984** | **0,2656** | **0,7084** | `VALID` | `pipeline-pertandan/results/pt_e_001_oracle.json` (`split.test.semua_pool.R4`) — seluruh 4 nilai lama (`0,8410/0,9910/0,1650/0,8320`) diganti nilai JSON asli. |
| `PT-E-030` | ResNet18 + Loss Ordinal CORN | RGB Crop | DAMIMAS (Test, n=1.316) | **0,6983** | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | `VALID` | `pipeline-pertandan/results/damimas_classifier_corn_s42.json` (`test_akurasi`) — nilai lama `0,7100` mendekati tapi tidak presisi (val 0,7095); kolom ±1/MAE/Macro-F1 belum ditemukan pada JSON level atas, perlu telusur lebih dalam. |
| `PT-E-012` | Pengklasifikasi Multi-Tampak C3 | RGB Multi-View | DAMIMAS (Test, n=1.404) | **0,6781** | ⚠ TIDAK-BISA-DIVERIFIKASI | **0,3369** | **0,6451** | `FALSIFIED` | `pipeline-pertandan/results/pt_e_012_c3.json` (`split.test.C3`) — C3 tidak melaporkan metrik ±1; Putusan asli DIPALSUKAN (C3 kalah dari C1/C2), bukan status campuran. |
| `V2-E-015` | Classifier crop (rata-rata 3 seed) | RGB | SawitMVC-Depth (352, Test) | **0,6309** | ⚠ TBD | ⚠ TBD | ⚠ TBD | `VALID` | `runs_fase6/sd{101,202,303}_rgb/hasil.json` — nilai lama `0,6415/0,9670/0,3915/0,6012` tidak cocok sumber; kolom ±1/MAE/Macro-F1 tidak dilaporkan entri asli untuk baris ini (⚠ TIDAK-BISA-DIVERIFIKASI). |
| `V2-E-044` | Pengklasifikasi C2 5-Epoch (val internal) | RGB Crop (Jitter 10%) | SawitMVC-YOLO 953 (Val internal) | **0,6217** | **0,9932** | **0,385 (MAE kelas)** | **0,6296** | `VALID` | `results/remote_eval_2026-08-27/classifier_c2/remote953_c2_rgb_5ep_jitter10.json` — seluruh 4 nilai lama (`0,5872/0,9380/0,4420/0,5510`) diganti angka validasi internal epoch terbaik yang benar. |
| `PT-E-015` | ResNet18 + Loss Ordinal CORAL (C2) | RGB Crop | DAMIMAS (Test, n=1.404) | **0,5686** | **0,9679** | **0,5271** | **0,4865** | `FALSIFIED` | `pipeline-pertandan/results/pt_e_014_c_resnet18_coral.json` (`split.test.C2.R0`) — CE pembanding test 0,6522; CORAL lebih buruk di TEST (−8,36pp) meski narasi log sempat melaporkan CORAL unggul tipis di VAL seed 0 (+2,35pp) — arah val dan test berlawanan, seluruh 4 nilai lama tidak cocok sumber. |
| `V2-E-016` | ResNet18 (*Crop Head*) | RGB+Depth (4-ch) | SawitMVC-Depth (352, Test) | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TBD | ⚠ TBD | ⚠ TBD | `FALSIFIED` | `runs_fase6/sd*/hasil.json` — entri asli hanya mencatat **Verdict: FALSIFIED** tanpa tabel angka mandiri di teks log; perlu baca JSON langsung. |

---

### 4.3 Pencacahan (*Counting*) Tandan Buah per Pohon
*Evaluasi jumlah tandan terestimasi dibandingkan nilai acuan riil per pohon.*

| ID Simpul | Detektor / Model Masukan | Metode Estimasi Cacah | Dataset & Pohon Uji | Macro MAE | Total MAE | Total Bias | Class $\pm 1$ Acc | Tree $\pm 1$ Acc | Status Bukti | Rujukan Artefak |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| `V2-E-004` | RT-DETR-L (RGB) | Ridge + $F_{all}$ (67-dim) | 352 (55 pohon) | **0,532** | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | **90,91%** | **67,27%** | `VALID` | `results/counting_rgb352.json` (lihat catatan silsilah data di atas) |
| `V2-E-004` | YOLO26l (RGB) | Ridge + $F_{all}$ (67-dim) | 352 (55 pohon) | 0,577 | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | 89,55% | 69,09% | `VALID` | `results/counting_rgb352.json` — Macro MAE dikoreksi dari narasi log (`experiments/EKSPERIMEN.md` V2-E-004); **catatan**: JSON hasil yang hidup saat ini melaporkan angka berbeda (macro\_mae≈0,618, macro\_acc≈84,09%), indikasi berkas telah diregenerasi setelah entri log ditulis — isu silsilah data pra-eksisting, di luar cakupan audit ini. |
| `V2-E-006` | RF-DETR-L (RGB+D 4-ch) | Ridge + $F_{all}$ (67-dim) | 352 (55 pohon) | 0,586 (RGBD MAE) | ⚠ TBD | ⚠ TBD | 88,18% | 67,27% | `FALSIFIED` | `results/counting_rgbd352.json` — nilai lama `0,409` tidak cocok narasi log (RGBD MAE asli 0,673/0,632/0,586 untuk YOLO26l/RT-DETR-L/RF-DETR-L); Tree$\pm1$ dikoreksi dari 65,45% ke 67,27% sesuai `results/counting_rgbd352.json`. |
| `V2-E-002` | RF-DETR-L (RGB) | Ridge + $F_{all}$ (67-dim) | 953 (141 pohon) | **0,993** | ⚠ TBD | ⚠ TBD | **76,24%** | **36,17%** | `VALID` | `results/counting_v2repro.json` |
| `V2-E-002` | RT-DETR-L (RGB) | Ridge + $F_{all}$ (67-dim) | 953 (141 pohon) | 0,997 | ⚠ TBD | ⚠ TBD | 76,24% | 34,04% | `VALID` | `results/counting_v2repro.json` |
| `PT-E-026` | Multi-Bank Anchor (varian *full*) | Multi-Bank Regresor | DAMIMAS (Test, 127 pohon) | **1,0374** | **1,8504** | **−1,2205** | **75,79%** | **30,71%** | `FALSIFIED` | `pipeline-pertandan/results/damimas_counting_multibank_full.json` (`test.*`) — seluruh 5 nilai lama salah; Putusan asli DIPALSUKAN untuk klaim gain macro (varian *compact* diterima HANYA untuk kepala total: total\_mae 1,7795). |
| `PT-E-004` | Deteksi Nyata (C5 Ridge+$F_{all}$) | Multi-View Cluster Count | DAMIMAS (Test, 141 pohon) | **1,0542** | ⚠ TBD | ⚠ TBD | **60,64%** | **14,89%** | `FALSIFIED` | `pipeline-pertandan/results/pt_e_004_counting.json` — nilai lama (`0,812/81,20%/46,80%`) tidak cocok sumber; Putusan asli DIPALSUKAN (G3 GUGUR, C4 hitung-pool 3,3422 vs C5 1,0542), bukan CONFIRMED/VALID. |
| `V2-E-002` | YOLO26l (RGB) | Ridge + $F_{all}$ (67-dim) | 953 (141 pohon) | 1,090 | ⚠ TBD | ⚠ TBD | 72,16% | 30,50% | `VALID` | `results/counting_v2repro.json` — Macro MAE dan Class$\pm1$ terverifikasi; Total MAE/Total Bias per model tidak tersedia sebagai field tunggal di JSON (⚠ TIDAK-BISA-DIVERIFIKASI). |
| `V2-E-045` | Layer count-aware (validation-locked) | Layered Reconciled Count | SawitMVC-Depth (110 pohon) | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TBD | ⚠ TBD | **80,91%** | ⚠ TBD | `VALID` | `experiments/EKSPERIMEN.md` (STATUS.md §7) — nilai `±1=83,64%/61,82%` lama sebenarnya milik V2-E-043 (*greedy/test-tuned*); V2-E-045 versi *validation-locked* asli: F1 0,8069, MAE 0,891, $\pm1$ 80,91%, match 80,31%, macro-F1 0,6047. |
| `V2-E-045` | Layer count-aware (validation-locked) | Layered Reconciled Count | SawitMVC-YOLO 953 (135 pohon) | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TBD | ⚠ TBD | **61,48%** | ⚠ TBD | `VALID` | Idem — 953 versi *validation-locked* asli: F1 0,8043, MAE 1,393, $\pm1$ 61,48%, match 71,11%, macro-F1 0,5384 (nilai lama `54,07%` tertukar dengan V2-E-043). |

---

### 4.4 Pengaitan Multi-Tampak (*Association & Clustering*)
*Evaluasi penaut hubungan spasial antar-sudut pandang kanonik (4 sisi per pohon).*

| ID Simpul | Algoritma / Modul Penaut | Ruang Masukan | Presisi Pasangan | Recall Pasangan | F1 Pasangan | ARI (*Rand Index*) | Cakupan Multi-Sisi | Status Bukti | Rujukan Artefak |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| `PT-E-001` | Oracle Association Linker (tautan GT) | Kotak Acuan (*Ground Truth*) | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 100,0% | `VALID` | `pipeline-pertandan/results/pt_e_001_oracle.json` (`.tautan = "ORACLE (GT confirmedLinks)"`) — nilai trivial benar secara definisi, terverifikasi. |
| `V2-E-043` | Greedy Strict Linker (F1 fisik cluster, bukan pairing murni) | WBF Proposal Gabungan, *Depth* | **0,8799** | **0,8390** | **0,8590** | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | `VALID` | `experiments/EKSPERIMEN.md` (V2-E-043) — angka ini adalah F1 **fisik** level cluster (lihat §4.5), bukan F1 pairing/ARI murni; ARI tidak dilaporkan entri V2-E-043; nilai lama `0,8412/0,7820/0,8105/0,7680/74,20%` tidak cocok sumber manapun yang ditemukan. |
| `Wave-V2 Depth` | GSP Linker (*Graph Shortest Path*, profil terkunci *depth*) | Multi-View, *test-locked* | **0,8926** | **0,8175** | **0,8534** | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | `VALID` | `results/remote_eval_2026-08-28/GSP_LINKER.md` (Hasil TEST-LOCKED *depth*) — nilai lama `0,8650/0,8110/0,8371/0,7940/78,10%` mendekati tapi tidak presisi untuk salah satu dari dua dataset asli; digabung dalam satu baris "Wave-V2" tanpa nama dataset sebelumnya adalah sumber kerancuan itu sendiri. |
| `Wave-V2 953` | Hungarian+UF Anchor (profil terkunci 953) | Multi-View, *test-locked* | **0,8444** | **0,8331** | **0,8387** | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | `VALID` | `results/remote_eval_2026-08-28/GSP_LINKER.md` (Hasil TEST-LOCKED 953) — **koreksi penting**: profil terkunci untuk 953 adalah **Hungarian+union-find**, BUKAN GSP; ARI/Cakupan tidak dilaporkan dokumen ini (memakai *matched\_class\_accuracy*=0,7442 sebagai gantinya, lihat §4.5). |
| `PT-E-008` | Rotation-Aware Signed Prior (varian E) | 4 Sisi Berurutan | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | **0,6486** | **0,5904** | ⚠ TIDAK-BISA-DIVERIFIKASI | `VALID` | `results/harapan_geser.json`, `results/pt_e_002_penaut.json` (test) — F1/ARI dikoreksi dari `0,7390/0,6890`; val F1=0,6718/ARI=0,6139 (gerbang G1 LOLOS). |
| `PT-E-016` | Relational GNN Linker | Kotak Acuan (*Ground Truth*) | **0,6718** | ⚠ TIDAK-BISA-DIVERIFIKASI | **0,6349** | **0,6047** | ⚠ TIDAK-BISA-DIVERIFIKASI | `INCONCLUSIVE` | `pipeline-pertandan/results/pt_e_016_gnn.json` (test) — seluruh nilai lama (`0,8450/0,7920/0,8176/0,7740/76,50%`) tidak cocok; Putusan asli TIDAK KONKLUSIF (F1 tak terpisahkan dari nol; val memilih baseline, test memilih GNN), bukan VALID. |
| `PT-E-020` | Global Linker Khusus DAMIMAS | Matriks Afinitas Global | **0,4359** | **0,4940** | **0,4631** | **0,4228** | **56,28%** | `VALID` | `pipeline-pertandan/results/damimas_linker_global.json` (`.test.*`) — seluruh 5 nilai lama (`0,7920/0,7460/0,7683/0,7210/71,40%`) kira-kira dua kali lipat dari angka riil. |
| `PT-E-002` | Varian A: geometri + `kelas_sama` (GT) | Kotak Acuan (*Ground Truth*) | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | **0,4282** | **0,3912** | ⚠ TIDAK-BISA-DIVERIFIKASI | `FALSIFIED` | `pipeline-pertandan/results/pt_e_002_penaut.json` (test) — F1/ARI dikoreksi dari `0,7019/0,6420` (SALAH TOTAL); Putusan asli DIPALSUKAN (Gerbang G1 GUGUR di semua varian), bukan SUPERSEDED; Presisi/Recall pasangan per-varian tidak ditemukan di narasi log. |
| `PT-E-017` | Relational GNN Linker (varian C, di deteksi) | Ruang Prediksi Deteksi | **0,3915** | **0,3669** | **0,3788** | **0,3221** | **38,39%** | `VALID` | `pipeline-pertandan/results/pt_e_017_gnn_deteksi.json` (test 953) — seluruh 5 nilai lama diganti; nilai riil sekitar separuh dari klaim lama (domain shift B−A=+15,88pp F1; penalaran GNN C−B=+7,08pp F1). |

---

### 4.5 Sistem Pipeline *End-to-End*
*Evaluasi integrasi menyeluruh: Deteksi $	o$ Asosiasi Multi-Tampak $	o$ Klasifikasi $	o$ Rekonsiliasi Cacah.*

| ID Simpul | Konfigurasi Sistem Pipeline | Kumpulan Uji Terkunci | Presisi Fisik | Recall Fisik | F1 Fisik | Counting MAE | Tree $\pm 1$ Acc | Matched Class Acc | Macro-F1 E2E | Status Bukti | Catatan Audit |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `PT-E-019` | Pipeline Utuh (penaut lama × ensemble, terbaik) | DAMIMAS Test | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | R4=**0,7311** (bukan macro-F1) | `FALSIFIED` | `pipeline-pertandan/results/pt_e_019_gabungan.json` — sumber asli HANYA mengukur R4 (akurasi kelas 1-ke-1) pada 4 konfigurasi, TIDAK memuat metrik fisik (presisi/recall/F1/MAE/Tree$\pm1$/matched-class/macro-F1) sama sekali; seluruh kolom lama pada baris ini tidak berdasar. Putusan asli DIPALSUKAN pada klaim berlipat, bukan SUPERSEDED. |
| `Wave-V2` | Locked GSP Pipeline (profil terkunci: GSP) | SawitMVC-Depth Test (110, *test-locked*) | **0,8926** | **0,8175** | **0,8534** | **0,7727** | **85,45%** | **0,8162** | **0,6519** | `VALID` | `results/remote_eval_2026-08-28/GSP_LINKER.md` (Hasil TEST-LOCKED *depth*) — Matched Class Acc dikoreksi dari `0,6840` ke `0,8162`; Macro-F1 dari `0,6340` ke `0,6519`; F1 fisik dari `0,8728` ke `0,8534`. Tree$\pm1$ (`85,45%`) sudah cocok. |
| `V2-E-043` | Optimized Greedy Pipeline | SawitMVC-Depth Test (110) | **0,8799** | **0,8390** | **0,8590** | **0,818** | **83,64%** | ⚠ TIDAK-BISA-DIVERIFIKASI | **0,6419** | `VALID` | `experiments/EKSPERIMEN.md` (V2-E-043) — Presisi/Recall dikoreksi (`0,8450/0,8735`→`0,8799/0,8390`); Macro-F1 E2E dikoreksi ke `0,6419` (lama `0,6120`); "Matched Class Acc" tidak dilaporkan entri ini dengan nama itu. |
| `Wave-V2` | Locked Hungarian+UF Pipeline (profil terkunci 953: **Hungarian**, bukan GSP) | SawitMVC-YOLO 953 Test (135, *test-locked*) | **0,8444** | **0,8331** | **0,8387** | **1,3630** | **63,70%** | **0,7442** | **0,6034** | `VALID` | `results/remote_eval_2026-08-28/GSP_LINKER.md` (Hasil TEST-LOCKED 953) — Matched Class Acc dikoreksi dari `0,6010` ke `0,7442` (selisih besar); MAE dari `1,510` ke `1,3630`; Tree$\pm1$ dari `57,78%` ke `63,70%`. |
| `PT-E-025` | Global DAMIMAS 1-to-1 Pipeline | DAMIMAS Test | **0,8530** | **0,8116** | **0,8318** (dihitung dari P/R) | **1,638** | ⚠ TIDAK-BISA-DIVERIFIKASI | **0,7322** | **0,5867** | `VALID` | `pipeline-pertandan/results/damimas_endtoend_global.json` (`.test.*`) — MAE lama `0,734` salah besar (riil `1,638`); Presisi/Recall/Matched/Macro-F1 dikoreksi dari `0,7890/0,7540/.../0,7410/0,6980`. |
| `V2-E-043` | Optimized Greedy Pipeline | SawitMVC-YOLO 953 Test (135) | **0,8247** | **0,8346** | **0,8296** | **1,644** | **54,07%** | ⚠ TIDAK-BISA-DIVERIFIKASI | **0,5469** | `VALID` | `experiments/EKSPERIMEN.md` (V2-E-043) — Presisi/Recall dikoreksi (`0,8120/0,8480`→`0,8247/0,8346`); Macro-F1 E2E dikoreksi ke `0,5469` (lama `0,5230`). |
| `V2-E-042` | Baseline Remote (WBF + Hungarian) | SawitMVC-Depth Test (110) | **0,4705** | **0,8837** | **0,6140** | **4,518** | **18,18%** | ⚠ TIDAK-BISA-DIVERIFIKASI | **0,4726** | `SUPERSEDED` | `experiments/EKSPERIMEN.md` (V2-E-043, baris "Baseline") — seluruh nilai lama (`0,7980/0,8210/0,8093/.../12,73%/.../0,5840`) salah kecuali MAE (`4,520`≈`4,518`, cocok); baris 953 (MAE 14,993) hilang sepenuhnya dari tabel ini. |
| `V2-E-020` | Two-Stage (YOLO26l Edge + ResNet18) | SawitMVC-Depth 352 Test | − | − | − | − | − | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ lihat catatan | `VALID` | Entri asli hanya melaporkan $mAP50=0,4500$ (deteksi 4-kelas), bukan pipeline asosiasi/klasifikasi dengan *matched-class-accuracy*; kolom "Macro-F1 E2E" lama (`0,4500`) sebenarnya adalah $mAP50$, bukan macro-F1, dan "Matched Class Acc" (`0,6415`) tidak ditemukan di entri V2-E-020 manapun. |

---

## 5. Ringkasan Sintesis Ilmiah Utama

1. **Efektivitas Sinyal Kedalaman Terkonsentrasi pada Lokalisasi (`V2-E-024` & `V2-E-016`):**
   Kanal kedalaman terbukti memberikan nilai tambah diskriminatif yang signifikan pada lokalisasi batas fisik tandan ($AP50_{agn} = \mathbf{0,7636}$ vs kontrol RGB $\mathbf{0,7358}$, $\Delta = +0,0278$, $P(\Delta > 0) = 92,1\%$), namun bersifat redundan secara kondisional terhadap fitur visual RGB untuk klasifikasi tingkat kematangan buah ($I(Y; D \mid 	ext{RGB}) pprox 0$).
2. **Ketiadaan Keunggulan Depth Monokular (`V2-E-027` s.d. `V2-E-032`):**
   Penambahan estimasi kedalaman monokular (*monocular depth estimation*) sebagai kanal ke-4 tidak menunjukkan keunggulan performa, bahkan menyebabkan degradasi performa yang signifikan secara statistik pada korpus 953 pohon ($\Delta mAP50 = \mathbf{−0,0476}$, $CI95 = [−0,0671; −0,0274]$).
3. **Pergeseran Temporal Membatalkan Komparasi 4-Kelas Lintas-Dataset (`V2-E-022`):**
   Jeda waktu $\sim 80	ext{ hari}$ antara SawitMVC-YOLO (Mei 2026) dan SawitMVC-Depth (Juli 2026) mencakup $5	ext{--}11$ rotasi panen, menyebabkan proporsi kelas B3 menyusut dari $55,3\%$ menjadi $14,0\%$. Oleh karena itu, evaluasi lintas dataset wajib dikontrol atau dievaluasi pada lokalisasi agnostik.
4. **Keunggulan Arsitektur DETR pada Objek Kanopi Kompleks (`V2-E-001`, `V2-E-034`, `V2-E-038`):**
   Urutan keunggulan deteksi **RF-DETR-L > RT-DETR-L > YOLO26l** terbukti konsisten dan signifikan secara statistik melintasi korpus 953, 763, dan 1.716 pohon melalui pengujian selang kepercayaan *bootstrap* berpasangan.

---

## 6. Peta Tautan Artefak dan Skrip Terkait

- **Dokumentasi Metodologi & Sintesis:**
  - [`docs/LAPORAN-AKHIR.md`](../docs/LAPORAN-AKHIR.md) — Laporan akhir resmi Volume 2.
  - [`docs/WORKFLOW_KRONOLOGIS.md`](../docs/WORKFLOW_KRONOLOGIS.md) — Rekonstruksi kronologis per simpul eksperimen.
  - [`docs/DIAGNOSIS-DEPTH.md`](../docs/DIAGNOSIS-DEPTH.md) — Analisis diagnostik fisik sinyal kedalaman.
  - [`docs/ANALISIS_PIPELINE_MENDALAM.md`](../docs/ANALISIS_PIPELINE_MENDALAM.md) — Analisis mendalam seluruh jalur pipeline.
- **Skrip Evaluasi & Reproduksi:**
  - [`scripts/eval_new763_pycoco.py`](../scripts/eval_new763_pycoco.py) & [`scripts/eval_pycoco_352.py`](../scripts/eval_pycoco_352.py) — Evaluasi deteksi pycocotools standar.
  - [`scripts/bootstrap_map.py`](../scripts/bootstrap_map.py) — Pengujian selang kepercayaan dan uji signifikansi statistik.
  - [`pipeline-pertandan/scripts/eval_endtoend_global_damimas.py`](../pipeline-pertandan/scripts/eval_endtoend_global_damimas.py) — Evaluasi sistem pipeline multi-tampak.
