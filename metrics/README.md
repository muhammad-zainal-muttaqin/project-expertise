# Atlas Metrik & Rangkuman Hasil Penelitian Terpadu

Selamat datang di **Atlas Metrik dan Rangkuman Penelitian Komprehensif** repositori `project-expertise` (Volume 2). Dokumen ini berfungsi sebagai gerbang utama (*master index*) dan atlas komparatif terstruktur yang merangkum seluruh hasil eksperimen, evaluasi empiris, ablasi arsitektur, dan pengembangan pipeline dari awal penelitian hingga iterasi mutakhir.

Atlas ini dirancang agar seluruh data kuantitatif dapat ditelusuri (*traceable*), diperbandingkan secara konsisten antar-eksperimen, serta dilengkapi konteks validitas ilmiah, batasan statistik, dan rujukan berkas sumber langsung.

---

## 1. Ringkasan Eksekutif & Statistik Global

| Parameter Inventaris | Nilai Terukur / Cakupan | Keterangan Metodologis |
|---|---|---|
| **Total Simpul Eksperimen Terlacak** | **101 Simpul** | 45 simpul Volume 2 (`V2-E-001`–`V2-E-045`), 36 simpul pipeline pertandan (`PT-E-000`–`PT-E-036`), 16 simpul audit forensik (`AF-E-001`–`AF-E-016`, 6 September 2026), dan 4 gelombang verifikasi/ablasi mutakhir (`remote_eval_2026-08-27`, `validation_wave_2026-08-28`, `new763_rgbd4`, dan `pipeline_damimas`). |
| **Rentang Tanggal Penelitian** | **09 Agustus 2026 – 6 September 2026** | Fondasi tinjauan pustaka & eksplorasi awal bersumber dari Volume 1 (Mei–Juli 2026). |
| **Arsitektur Detektor Utama** | **YOLO26l, RT-DETR-L, RF-DETR-L** | Detektor berbasis satu-tahap konvensional, Transformer hibrida waktu-nyata, dan arsitektur DETR modern. |
| **Modalitas Citra Diuji** | **RGB, RGB+Depth (Sensor Y16), RGB+Mono (Depth Monokular), RGB+Depth+Mono (5-kanal)** | Evaluasi perbandingan *early fusion*, *mid-fusion*, dan fusi lanjut (*late fusion*). |
| **Korpus & Partisi Data** | **SawitMVC-YOLO (953 pohon), SawitMVC-Depth (352 & 763 pohon), Combined-1716 (1.716 pohon)** | Pembagian partisi terkontrol (*train, val, test*) untuk mencegah kebocoran partisi data (*data leakage*). |
| **Status Validitas Ilmiah** | **Ditentukan per baris dan per protokol** | Angka metrik dan status pada berkas spesialis telah diaudit terhadap artefak sumber; `CORRECTED`, `N/A`, dan alasan ketidaktersediaan ditulis eksplisit, bukan dihitung dari klaim lama. |

### Taksonomi Status Validitas Bukti
Setiap baris hasil eksperimen dalam atlas ini diberi label validitas resmi:
1. `VALID`: Eksperimen memenuhi kontrol variabel metodologis, partisi data terkunci (*validation/test-locked*), dan kesimpulan didukung data empiris.
2. `FALSIFIED`: Hipotesis eksperimen diuji secara sah namun tertolak oleh bukti empiris (misal kanal kedalaman tidak meningkatkan klasifikasi kematangan).
3. `INVALID`: Rancangan pengujian mengandung cacat metodologis mendasar yang membatalkan perbandingan ilmiah (misal perbandingan 4-kelas lintas dataset akibat pergeseran temporal $\sim 80\text{ hari}$ pada `V2-E-022`).
4. `SUPERSEDED`: Hasil empiris valid pada fasenya, namun telah digantikan oleh model, parameter, atau pipeline generasi berikutnya yang lebih optimal.
5. `INCOMPLETE`: Eksperimen penyaringan cepat (*screening* pendek $\le 15\text{ epoch}$) atau dihentikan sebelum konvergen penuh.
6. `RETRACTED` / `CORRECTED`: Klaim atau angka awal yang ditarik/dikoreksi setelah audit kebocoran data atau galat perkakas evaluasi (misal audit pretrain `agn953_full` pada `V2-E-025` dan audit TIFF korup pada `V2-E-028`).

### Protokol audit metrik (2026-09-03)
Seluruh tabel di folder ini dibandingkan dengan artefak lokal yang tersedia dan, bila diperlukan, artefak Hugging Face yang diunduh selektif. Audit tidak menjalankan training ulang. Prediksi/dump yang sudah ada dipakai untuk menghitung ulang metrik; bobot dan data hanya diambil ketika diperlukan untuk menutup celah evaluasi. `N/A — tidak dilaporkan` berarti sumber tidak menyediakan field atau protokolnya, sedangkan `N/A — bukan metrik uji` berarti metrik tersebut tidak bermakna untuk baris terkait.

---

## 2. Struktur Atlas & Navigasi Berkas Spesialisasi

Untuk memudahkan penelaahan mendalam per domain tugas, atlas ini dipecah ke dalam 7 berkas spesialisasi berikut:

| Berkas Spesialisasi | Cakupan Domain & Metrik Utama |
|---|---|
| **[01_deteksi_dan_lokalisasi.md](01_deteksi_dan_lokalisasi.md)** | Komparasi detektor 1-tahap (RGB vs RGB+D vs Mono), studi *encoding* kedalaman (Invers, Sobel `edge`), fusi WBF, TTA, *class-aware* ($mAP50$, $mAP50\text{--}95$, per kelas B1–B4) vs *class-agnostic* ($AP50_{agn}$). |
| **[02_klasifikasi_kematangan.md](02_klasifikasi_kematangan.md)** | Model pengklasifikasi kematangan pada citra terpotong (*crop*), loss ordinal (CORAL, CORN), evaluasi multi-tampak (C1–C3), *mixture-of-experts*, aturan keputusan per tandan (R0–R4), akurasi, akurasi toleransi $\pm 1$, dan MAE ordinal. |
| **[03_pencacahan_per_pohon.md](03_pencacahan_per_pohon.md)** | Pencacahan jumlah tandan per pohon via Ridge regression ($F_{all}$), penghitung pool fisik, model multi-bank, CatBoost regularized, meta-ensemble cacah, Macro MAE, Total MAE, Bias bertanda, dan akurasi $\pm 1$. |
| **[04_pengaitan_multi_tampak.md](04_pengaitan_multi_tampak.md)** | Sistem penaut (*linker*) asosiasi spasial antar-sudut pandang kanonik (Hungarian, Union-Find, GNN, Global Linker DAMIMAS, GSP Linker), F1 pasangan, *Adjusted Rand Index* (ARI), cakupan tandan multi-sisi, dan fraksi pool palsu. |
| **[05_pipeline_end_to_end.md](05_pipeline_end_to_end.md)** | Evaluasi sistem pipeline terpadu (Deteksi $\to$ Asosiasi Multi-Tampak $\to$ Klasifikasi $\to$ Rekonsiliasi Cacah), F1 deteksi fisik, MAE pencacahan per pohon, *matched class accuracy*, dan macro-F1 *end-to-end*. |
| **[06_komputasi_dan_diagnostik.md](06_komputasi_dan_diagnostik.md)** | Kompleksitas komputasi (jumlah parameter, GFLOPs, durasi pelatihan/inferensi), diagnostik fisik sinyal kedalaman (*relief SNR*, kuantisasi, uji Kruskal–Wallis, korelasi Spearman), audit derau TIFF, serta selang kepercayaan *bootstrap* 95%. |
| **[07_buku_besar_eksperimen.md](07_buku_besar_eksperimen.md)** | Buku besar (*master ledger*) kronologis seluruh simpul eksperimen `V2-E-001` s.d. `V2-E-045`, `PT-E-000` s.d. `PT-E-036`, serta rangkaian verifikasi remote dengan status pembuktian lengkap. |

---

## 3. Standar Normalisasi & Aturan Pembacaan Metrik

Guna mencegah kerancuan dan salah tafsir lintas domain, seluruh metrik dalam repositori dinormalisasi mengikuti kaidah baku berikut:

1. **Pembedaan Penyebut dan Unit Evaluasi:**
   - **Tingkat Kemunculan (*Per-Appearance / Box*):** Menilai kualitas deteksi kotak pembatas pada citra tunggal ($mAP50$, $mAP50\text{--}95$, $AP_{agn}$).
   - **Tingkat Tandan Fisik (*Per-Physical Bunch*):** Menilai ketepatan klaster gabungan multi-tampak yang merepresentasikan 1 tandan buah riil di pohon (F1 fisik, *matched class accuracy*).
   - **Tingkat Pohon (*Per-Tree*):** Menilai akurasi agregasi cacah buah per pohon (Macro MAE, Total MAE, Akurasi Pohon $\pm 1$).
   - **Tingkat Pasangan (*Per-Pair*):** Menilai ketepatan relasi biner antara dua kotak dari sudut pandang berbeda (Presisi/Recall/F1 Pasangan, ARI).

2. **Pencegahan Penyamaan Metrik Berbeda Makna:**
   - `$AP50_{agnostic}$` adalah plafon lokalisasi spasial tanpa label kelas; **tidak boleh disamakan** dengan akurasi klasifikasi B1–B4 atau $mAP50$ 4-kelas.
   - `class_pm1_acc` pada modul pencacahan mengukur persentase sel pohon–kelas dengan galat cacah $\le 1$; **bukan akurasi klasifikasi citra biasa**.
   - Metrik bertanda minus tipografis asli `−` (seperti degradasi performa $\Delta = −0,0476$ atau bias cacah $−0,15$) menunjukkan penurunan/galat *under-predict*, bukan tanda hubung teks.

3. **Notasi Matematika & Simbol:**
   - Desimal menggunakan koma (misal: $0,6012$), pemisah ribuan menggunakan titik (misal: $3.992\text{ citra}$).
   - Selang kepercayaan 95% ditulis dalam format kurung siku bertitik koma: $[min; max]$ (misal: $[−0,0671; −0,0274]$).
   - Seluruh selisih performa $\Delta$ yang selang kepercayaannya mencakup nilai nol dinyatakan **tidak signifikan secara statistik**.
   - Nilai metrik yang tidak tercatat/tersedia diisi `N/A` beserta alasannya; simbol `−` hanya dipakai bila kolom memang tidak berlaku secara definisi.

---

## 4. Tabel Master Ringkas Komparasi Utama

Berikut adalah intisari perbandingan kuantitatif pada simpul-simpul eksperimen kunci di setiap kategori.

### 4.1 Deteksi dan Lokalisasi Kotak Pembatas
*Evaluasi menggunakan `pycocotools.COCOeval` pada partisi data uji kanonik.*

| ID Simpul | Konfigurasi Model & Modalitas | Dataset & Partisi | $mAP50$ | $mAP50\text{--}95$ | $AP_{B1}$ | $AP_{B2}$ | $AP_{B3}$ | $AP_{B4}$ | $AP50_{agn}$ | Status Bukti | Rujukan Artefak |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `V2-E-024` | YOLO26l (RGB+Sobel `edge`, Agnostik) | SawitMVC-Depth (352, Test) | `N/A — bukan evaluasi 4-kelas` | `N/A — bukan evaluasi 4-kelas` | `N/A — bukan evaluasi 4-kelas` | `N/A — bukan evaluasi 4-kelas` | `N/A — bukan evaluasi 4-kelas` | `N/A — bukan evaluasi 4-kelas` | **0,7636** | `VALID` | `results/class_agnostic_metrics_audit_2026-09-03.json` |
| `V2-E-013` | YOLO26l (RGB, Agnostik) | SawitMVC-Depth (352, Test) | `N/A — bukan evaluasi 4-kelas` | `N/A — bukan evaluasi 4-kelas` | `N/A — bukan evaluasi 4-kelas` | `N/A — bukan evaluasi 4-kelas` | `N/A — bukan evaluasi 4-kelas` | `N/A — bukan evaluasi 4-kelas` | **0,6676** | `CORRECTED` | `results/class_agnostic_metrics_audit_2026-09-03.json` (checkpoint YOLO26l; label RF-DETR dan path lama dikoreksi) |
| `V2-E-042` | RF-DETR-L Remote Combined | SawitMVC-Depth Test (110 pohon) | **0,6711** | **0,2748** | **0,8044** | **0,7187** | **0,7373** | **0,4239** | **0,8764** | `VALID` | `results/remote_eval_2026-08-27/README.md` |
| `V2-E-034` | RF-DETR-L (RGB, Seed 42) | SawitMVC-Depth v2 (763, Test) | **0,6129** | **0,2335** | **0,7758** | **0,6353** | **0,6997** | **0,3406** | 0,7951 | `CORRECTED` | `results/new763/rfdetr_l_rgb_s42_i1280.json` + `results/class_agnostic_metrics_audit_2026-09-03.json` |
| `V2-E-001` | RF-DETR-L (RGB) | SawitMVC-YOLO (953, Test) | **0,6012** | **0,2747** | **0,8150** | **0,5184** | **0,6553** | **0,4160** | `N/A — tidak dihitung pada evaluasi 4-kelas` | `VALID` | `results/perkelas_pycoco_v2repro.json` |
| `V2-E-035` | RF-DETR-L (RGB, Seed 42) | Combined-1716 (1716, Test) | **0,5960** | **0,2522** | **0,7654** | **0,5394** | **0,6652** | **0,4141** | 0,7850 | `CORRECTED` | `results/combined1716/combined1716_rfdetr_l_rgb_s42_i1280.json` + `results/class_agnostic_metrics_audit_2026-09-03.json` |
| `V2-E-001` | RT-DETR-L (RGB) | SawitMVC-YOLO (953, Test) | 0,5781 | 0,2629 | 0,7874 | 0,4614 | 0,6371 | 0,4266 | `N/A — tidak dihitung pada evaluasi 4-kelas` | `VALID` | `results/perkelas_pycoco_v2repro.json` |
| `V2-E-039` | Ensembel WBF 3-Detektor | Combined-1716 (1716, Test) | 0,5538 | `N/A — tidak dilaporkan` | 0,7286 | 0,4732 | 0,6372 | 0,3760 | **0,8104** | `CORRECTED` | `results/extra_metrics_sesi2026-08.json` + `results/class_agnostic_metrics_audit_2026-09-03.json` (0,8106 adalah evaluator custom; 0,8104 COCOeval) |
| `V2-E-001` | YOLO26l (RGB) | SawitMVC-YOLO (953, Test) | 0,5435 | 0,2564 | 0,7705 | 0,4479 | 0,6050 | 0,3506 | `N/A — tidak dihitung pada evaluasi 4-kelas` | `VALID` | `results/perkelas_pycoco_v2repro.json` |
| `V2-E-027` | YOLO26l (RGB+Mono 4-ch) | SawitMVC-YOLO (953, Test) | 0,4960 | 0,2322 | 0,6902 | 0,4097 | 0,5635 | 0,3206 | `N/A — tidak dihitung pada audit ini` | `CORRECTED` | `results/eval_sel6_953_rgbmono_test.json` |
| `V2-E-003` | RF-DETR-L (RGB) | SawitMVC-Depth (352, Test) | **0,4544** | **0,1599** | **0,6853** | **0,5184** | **0,3477** | **0,2661** | `N/A — tidak dihitung pada evaluasi 4-kelas` | `VALID` | `results/perkelas_pycoco_rgb352.json` |
| `V2-E-003` | RT-DETR-L (RGB) | SawitMVC-Depth (352, Test) | 0,4343 | 0,1503 | 0,7680 | 0,4867 | 0,2641 | 0,2185 | `N/A — tidak dihitung pada evaluasi 4-kelas` | `VALID` | `results/perkelas_pycoco_rgb352.json` |
| `V2-E-010` | YOLO26l (RGB+Sobel `edge` 4-ch) | SawitMVC-Depth (352, Test) | **0,4316** | 0,1441 | 0,7252 | 0,5031 | 0,2240 | 0,2740 | `N/A — tidak dihitung pada evaluasi 4-kelas` | `VALID` | `results/perkelas_pycoco_rgbd352.json` (kunci `YOLO26l-RGBD-edge`) |
| `V2-E-005` | YOLO26l (RGB+D Invers 4-ch) | SawitMVC-Depth (352, Test) | 0,3919 | 0,1408 | 0,6857 | 0,4579 | 0,2637 | 0,1601 | `N/A — tidak dihitung pada evaluasi 4-kelas` | `SUPERSEDED` | `results/perkelas_pycoco_rgbd352.json` |
| `V2-E-003` | YOLO26l (RGB) | SawitMVC-Depth (352, Test) | 0,3606 | 0,1246 | 0,6804 | 0,4320 | 0,2001 | 0,1299 | `N/A — tidak dihitung pada evaluasi 4-kelas` | `VALID` | `results/perkelas_pycoco_rgb352.json` |

---

### 4.2 Klasifikasi Tingkat Kematangan Citra Terpotong (*Crop*)
*Evaluasi performa klasifikasi kematangan 4 kelas (B1–B4) pada citra terpotong.*

| ID Simpul | Metode / Model Pengklasifikasi | Modalitas Input | Dataset & Partisi | Akurasi | Akurasi $\pm 1$ | MAE Ordinal | Macro-F1 | Status Bukti | Rujukan Artefak |
|---|---|---|---|---:|---:|---:|---:|---|---|
| `PT-E-029` | Weighted Average Ensemble | Multi-Head Crop | DAMIMAS (Test, n=1.316) | **0,7409** | **0,9970** | **0,2622** | **0,7048** | `CORRECTED` | `pipeline-pertandan/results/pt_e_029_ensemble_kelas_damimas_pred.npz` + JSON — dump TEST memberi Akurasi/±1/MAE/Macro-F1; angka lama `0,7439` tidak cocok dengan dump n=1.316. |
| `PT-E-001` | R4 (agregasi ordinal, seluruh pool) | RGB Crop | DAMIMAS 953 (Test, n=1.269) | **0,7360** | **0,9984** | **0,2656** | **0,7084** | `VALID` | `pipeline-pertandan/results/pt_e_001_oracle.json` (`split.test.semua_pool.R4`) — seluruh 4 nilai lama (`0,8410/0,9910/0,1650/0,8320`) diganti nilai JSON asli. |
| `PT-E-030` | ResNet18 + Loss Ordinal CORN | RGB Crop | DAMIMAS (Test, n=1.316) | **0,6983** | **0,9954** | **0,3062** | **0,6554** | `CORRECTED` | `pipeline-pertandan/results/damimas_classifier_corn_s42_pred.npz` + JSON — metrik ordinal dihitung dari dump TEST; akurasi dikoreksi `0,7100`→`0,6983`. |
| `PT-E-012` | Pengklasifikasi Multi-Tampak C3 | RGB Multi-View | DAMIMAS (Test, n=1.404) | **0,6781** | `N/A — tidak dilaporkan` | **0,3369** | **0,6451** | `FALSIFIED` | `pipeline-pertandan/results/pt_e_012_c3.json` (`split.test.C3`) — artefak tidak melaporkan metrik ±1. |
| `V2-E-015` | Classifier crop (rata-rata 3 seed) | RGB | SawitMVC-Depth (352, Test) | **0,6309** | **0,9569** | **0,4163** | **0,5525** | `CORRECTED` | `results/fase6_classifier.json` (`ablasi_depth_multiseed.test`, rata-rata 3 seed); metrik ordinal dihitung dari keluaran test tiap seed. |
| `V2-E-044` | Pengklasifikasi C2 5-Epoch (val internal) | RGB Crop (Jitter 10%) | SawitMVC-YOLO 953 (Val internal) | **0,6217** | **0,9932** | **0,385 (MAE kelas)** | **0,6296** | `VALID` | `results/remote_eval_2026-08-27/classifier_c2/remote953_c2_rgb_5ep_jitter10.json` — seluruh 4 nilai lama (`0,5872/0,9380/0,4420/0,5510`) diganti angka validasi internal epoch terbaik yang benar. |
| `V2-E-016` | ResNet18 (*Crop Head*) | RGB+Depth (4-ch) | SawitMVC-Depth (352, Test) | **0,6106** | **0,9561** | **0,4374** | **0,5432** | `FALSIFIED` | `results/fase6_classifier.json` (`ablasi_depth_multiseed.test`, rata-rata 3 seed); depth tidak mengungguli RGB (`0,6106` vs `0,6309`). |
| `PT-E-015` | ResNet18 + Loss Ordinal CORAL (C2) | RGB Crop | DAMIMAS (Test, n=1.404) | **0,5686** | **0,9679** | **0,5271** | **0,4865** | `FALSIFIED` | `pipeline-pertandan/results/pt_e_014_c_resnet18_coral.json` (`split.test.C2.R0`) — CE pembanding test 0,6522; CORAL lebih buruk di TEST (−8,36pp) meski narasi log sempat melaporkan CORAL unggul tipis di VAL seed 0 (+2,35pp) — arah val dan test berlawanan, seluruh 4 nilai lama tidak cocok sumber. |

---

### 4.3 Pencacahan (*Counting*) Tandan Buah per Pohon
*Evaluasi jumlah tandan terestimasi dibandingkan nilai acuan riil per pohon.*

| ID Simpul | Detektor / Model Masukan | Metode Estimasi Cacah | Dataset & Pohon Uji | Macro MAE | Total MAE | Total Bias | Class $\pm 1$ Acc | Tree $\pm 1$ Acc | Status Bukti | Rujukan Artefak |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| `V2-E-004` | RT-DETR-L (RGB) | Ridge + $F_{all}$ (67-dim) | 352 (55 pohon) | **0,5318** | `N/A — tidak tersedia` | **−0,1636 (turunan)** | **90,91%** | **67,27%** | `CORRECTED` | `results/counting_rgb352.json`; Total Bias dijumlahkan dari empat bias kelas, sedangkan Total MAE/Exact tidak disimpan. |
| `V2-E-006` | RF-DETR-L (RGB+D 4-ch) | Ridge + $F_{all}$ (67-dim) | 352 (55 pohon) | **0,5864** | `N/A — tidak tersedia` | **−0,3455 (turunan)** | **88,18%** | **67,27%** | `FALSIFIED` | `results/counting_rgbd352.json`; angka berasal dari kunci `RF-DETR-L-RGBD`; Total MAE/Exact tidak disimpan. |
| `V2-E-004` | YOLO26l (RGB) | Ridge + $F_{all}$ (67-dim) | 352 (55 pohon) | **0,6182** | `N/A — tidak tersedia` | **−0,5455 (turunan)** | **84,09%** | **54,55%** | `CORRECTED` | `results/counting_rgb352.json`; JSON hidup adalah sumber kanonik, sehingga angka lama `0,577/89,55%/69,09%` dikoreksi. |
| `V2-E-002` | RF-DETR-L (RGB) | Ridge + $F_{all}$ (67-dim) | 953 (141 pohon) | **0,9929** | `N/A — tidak tersedia` | **−0,1135 (turunan)** | **76,24%** | **36,17%** | `CORRECTED` | `results/counting_v2repro.json`; Total Bias dijumlahkan dari bias B1–B4, Total MAE/Exact tidak tersedia. |
| `V2-E-002` | RT-DETR-L (RGB) | Ridge + $F_{all}$ (67-dim) | 953 (141 pohon) | **0,9965** | `N/A — tidak tersedia` | **−0,1135 (turunan)** | **76,24%** | **34,04%** | `CORRECTED` | `results/counting_v2repro.json`; Total Bias dijumlahkan dari bias B1–B4, Total MAE/Exact tidak tersedia. |
| `PT-E-026` | Multi-Bank Anchor (varian *full*) | Multi-Bank Regresor | DAMIMAS (Test, 127 pohon) | **1,0374** | **1,8504** | **−1,2205** | **75,79%** | **30,71%** | `FALSIFIED` | `pipeline-pertandan/results/damimas_counting_multibank_full.json` (`test.*`) — seluruh 5 nilai lama salah; Putusan asli DIPALSUKAN untuk klaim gain macro (varian *compact* diterima HANYA untuk kepala total: total\_mae 1,7795). |
| `PT-E-004` | Deteksi Nyata (C5 Ridge+$F_{all}$) | Multi-View Cluster Count | DAMIMAS (Test, 141 pohon) | **1,0542** | `N/A — tidak tersedia` | **−0,0427** | **60,64%** | **14,89%** | `FALSIFIED` | `pipeline-pertandan/results/pt_e_004_counting.json`; Total Bias tersedia, Total MAE/Exact tidak dilaporkan. |
| `V2-E-002` | YOLO26l (RGB) | Ridge + $F_{all}$ (67-dim) | 953 (141 pohon) | **1,0904** | `N/A — tidak tersedia` | **−0,3050 (turunan)** | **72,16%** | **30,50%** | `CORRECTED` | `results/counting_v2repro.json`; Total Bias dijumlahkan dari bias B1–B4, Total MAE/Exact tidak tersedia sebagai field tunggal. |
| `V2-E-045` | Layer count-aware (validation-locked) | Layered Reconciled Count | SawitMVC-Depth (110 pohon) | `N/A — sumber hanya MAE total` | **0,891** | `N/A — tidak dilaporkan` | `N/A — tidak dilaporkan` | **80,91%** | `VALID` | `experiments/EKSPERIMEN.md` (STATUS.md §7); ±1 adalah Tree±1, sedangkan Macro MAE/Class±1/Bias/Exact tidak disimpan. |
| `V2-E-045` | Layer count-aware (validation-locked) | Layered Reconciled Count | SawitMVC-YOLO 953 (135 pohon) | `N/A — sumber hanya MAE total` | **1,393** | `N/A — tidak dilaporkan` | `N/A — tidak dilaporkan` | **61,48%** | `VALID` | `experiments/EKSPERIMEN.md` (STATUS.md §7); ±1 adalah Tree±1, sedangkan Macro MAE/Class±1/Bias/Exact tidak disimpan. |

---

### 4.4 Pengaitan Multi-Tampak (*Association & Clustering*)
*Evaluasi penaut hubungan spasial antar-sudut pandang kanonik (4 sisi per pohon).*

| ID Simpul | Algoritma / Modul Penaut | Ruang Masukan | Presisi Pasangan | Recall Pasangan | F1 Pasangan | ARI (*Rand Index*) | Cakupan Multi-Sisi | Status Bukti | Rujukan Artefak |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| `PT-E-001` | Oracle Association Linker (tautan GT) | Kotak Acuan (*Ground Truth*) | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 100,0% | `VALID` | `pipeline-pertandan/results/pt_e_001_oracle.json` (`.tautan = "ORACLE (GT confirmedLinks)"`) — nilai trivial benar secara definisi, terverifikasi. |
| `V2-E-043` | Greedy Strict Linker (F1 fisik cluster, bukan pairing murni) | WBF Proposal Gabungan, *Depth* | **0,8799 (fisik)** | **0,8390 (fisik)** | **0,8590 (fisik)** | `N/A — bukan pair metric` | `N/A — tidak dilaporkan` | `VALID` | `experiments/EKSPERIMEN.md` (V2-E-043) — P/R/F1 adalah physical cluster, bukan F1 pairing/ARI. |
| `Wave-V2 Depth` | GSP Linker (profil terkunci *depth*) | Multi-View, *test-locked* | **0,8926 (fisik)** | **0,8175 (fisik)** | **0,8534 (fisik)** | `N/A — bukan pair metric` | `N/A — tidak dilaporkan` | `CORRECTED` | `results/remote_eval_2026-08-28/gsp_artifacts/depth/results_test_locked.json` — P/R/F1 berasal dari metrik physical detection. |
| `Wave-V2 953` | Hungarian+UF Anchor (profil terkunci 953) | Multi-View, *test-locked* | **0,8444 (fisik)** | **0,8331 (fisik)** | **0,8387 (fisik)** | `N/A — bukan pair metric` | `N/A — tidak dilaporkan` | `CORRECTED` | `results/remote_eval_2026-08-28/gsp_artifacts/953/results_test_locked.json` — profil 953 adalah Hungarian+union-find, bukan GSP. |
| `PT-E-008` | Rotation-Aware Signed Prior (varian E) | 4 Sisi Berurutan | **0,6679** | **0,6303** | **0,6486** | **0,5904** | `N/A — tidak dilaporkan` | `VALID` | `pipeline-pertandan/results/pt_e_002_penaut.json` (`E_reid_plus_kelas_prediksi.test_sekali`) — P/R/F1/ARI berasal dari varian E yang sama. |
| `PT-E-016` | Relational GNN Linker | Kotak Acuan (*Ground Truth*) | **0,5204** | **0,6831** | **0,5907** | **0,5586** | **69,08% (coverage tandan)** | `INCONCLUSIVE` | `pipeline-pertandan/results/pt_e_016_gnn.json` (`gnn.test`) — GNN menurunkan F1 titik operasi dibanding baseline, tetapi menaikkan coverage dan AUC. |
| `PT-E-020` | Global Linker Khusus DAMIMAS | Matriks Afinitas Global | **0,4359** | **0,4940** | **0,4631** | **0,4228** | **56,28%** | `VALID` | `pipeline-pertandan/results/damimas_linker_global.json` (`.test.*`) — seluruh 5 nilai lama (`0,7920/0,7460/0,7683/0,7210/71,40%`) kira-kira dua kali lipat dari angka riil. |
| `PT-E-002` | Varian A: geometri + `kelas_sama` (GT) | Kotak Acuan (*Ground Truth*) | **0,3742** | **0,5003** | **0,4282** | **0,3912** | `N/A — tidak dilaporkan` | `FALSIFIED` | `pipeline-pertandan/results/pt_e_002_penaut.json` (`A_geometri_saja.test_sekali`) — P/R diambil dari varian A yang sama. |
| `PT-E-017` | Relational GNN Linker (varian C, di deteksi) | Ruang Prediksi Deteksi | **0,3915** | **0,3669** | **0,3788** | **0,3221** | **38,39%** | `VALID` | `pipeline-pertandan/results/pt_e_017_gnn_deteksi.json` (test 953) — seluruh 5 nilai lama diganti; nilai riil sekitar separuh dari klaim lama (domain shift B−A=+15,88pp F1; penalaran GNN C−B=+7,08pp F1). |

---

### 4.5 Sistem Pipeline *End-to-End*
*Evaluasi integrasi menyeluruh: Deteksi $\to$ Asosiasi Multi-Tampak $\to$ Klasifikasi $\to$ Rekonsiliasi Cacah.*

| ID Simpul | Konfigurasi Sistem Pipeline | Kumpulan Uji Terkunci | Presisi Fisik | Recall Fisik | F1 Fisik | Counting MAE | Tree $\pm 1$ Acc | Matched Class Acc | Macro-F1 E2E | Status Bukti | Catatan Audit |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `PT-E-019` | Pipeline Utuh (penaut lama × ensemble, terbaik) | DAMIMAS Test | `N/A — tidak diukur` | `N/A — tidak diukur` | `N/A — tidak diukur` | `N/A — tidak diukur` | `N/A — tidak diukur` | `N/A — tidak diukur` | `R4=0,7311 (bukan macro-F1)` | `FALSIFIED` | `pipeline-pertandan/results/pt_e_019_gabungan.json` — sumber asli hanya mengukur R4; seluruh metrik fisik/counting/matched-class/macro-F1 tidak ada. |
| `V2-E-020` | Two-Stage (YOLO26l Edge + ResNet18) | SawitMVC-Depth 352 Test | `N/A — bukan metrik E2E` | `N/A — bukan metrik E2E` | `N/A — bukan metrik E2E` | `N/A — bukan metrik E2E` | `N/A — bukan metrik E2E` | `N/A — bukan metrik E2E` | `N/A — bukan metrik E2E` | `INVALID` | Entri asli hanya melaporkan $mAP50=0,4500$ deteksi 4-kelas; nilai itu bukan macro-F1 E2E dan tidak ada metrik pipeline yang cocok. |
| `Wave-V2` | Locked GSP Pipeline (profil terkunci: GSP) | SawitMVC-Depth Test (110, *test-locked*) | **0,8926** | **0,8175** | **0,8534** | **0,7727** | **85,45%** | **0,8162** | **0,6519** | `VALID` | `results/remote_eval_2026-08-28/GSP_LINKER.md` (Hasil TEST-LOCKED *depth*) — Matched Class Acc dikoreksi dari `0,6840` ke `0,8162`; Macro-F1 dari `0,6340` ke `0,6519`; F1 fisik dari `0,8728` ke `0,8534`. Tree$\pm1$ (`85,45%`) sudah cocok. |
| `V2-E-043` | Optimized Greedy Pipeline | SawitMVC-Depth Test (110) | **0,8799** | **0,8390** | **0,8590** | **0,818** | **83,64%** | `N/A — tidak dilaporkan` | **0,6419** | `VALID` | `experiments/EKSPERIMEN.md` (V2-E-043) — Matched Class Acc tidak dilaporkan dengan nama itu. |
| `Wave-V2` | Locked Hungarian+UF Pipeline (profil terkunci 953: **Hungarian**, bukan GSP) | SawitMVC-YOLO 953 Test (135, *test-locked*) | **0,8444** | **0,8331** | **0,8387** | **1,3630** | **63,70%** | **0,7442** | **0,6034** | `VALID` | `results/remote_eval_2026-08-28/GSP_LINKER.md` (Hasil TEST-LOCKED 953) — Matched Class Acc dikoreksi dari `0,6010` ke `0,7442` (selisih besar); MAE dari `1,510` ke `1,3630`; Tree$\pm1$ dari `57,78%` ke `63,70%`. |
| `PT-E-025` | Global DAMIMAS 1-to-1 Pipeline | DAMIMAS Test | **0,8530** | **0,8116** | **0,8318** (dihitung dari P/R) | **1,638** | `N/A — tidak dilaporkan` | **0,7322** | **0,5867** | `VALID` | `pipeline-pertandan/results/damimas_endtoend_global.json` (`.test.*`) — Tree±1 tidak ada di artefak; MAE lama `0,734` dikoreksi ke `1,638`. |
| `V2-E-043` | Optimized Greedy Pipeline | SawitMVC-YOLO 953 Test (135) | **0,8247** | **0,8346** | **0,8296** | **1,644** | **54,07%** | `N/A — tidak dilaporkan` | **0,5469** | `VALID` | `experiments/EKSPERIMEN.md` (V2-E-043) — Matched Class Acc tidak dilaporkan. |
| `V2-E-042` | Baseline Remote (WBF + Hungarian) | SawitMVC-Depth Test (110) | **0,4705** | **0,8837** | **0,6140** | **4,518** | **18,18%** | `N/A — tidak dilaporkan` | **0,4726** | `SUPERSEDED` | `experiments/EKSPERIMEN.md` (V2-E-043, baris Baseline); Matched Class Acc tidak dilaporkan. |
| `V2-E-042` | Baseline Remote (WBF + Hungarian) | SawitMVC-YOLO 953 Test (135) | **0,3725** | **0,9344** | **0,5327** | **14,993** | **0%** | `N/A — tidak dilaporkan` | **0,3762** | `SUPERSEDED` | `results/remote_eval_2026-08-28/gsp_artifacts/953/results_test_locked.json`; baris 953 ditambahkan dari artefak locked yang sesuai. |

---

## 5. Ringkasan Sintesis Ilmiah Utama

1. **Efektivitas Sinyal Kedalaman Terkonsentrasi pada Lokalisasi (`V2-E-024` & `V2-E-016`):**
   Kanal kedalaman terbukti memberikan nilai tambah diskriminatif yang signifikan pada lokalisasi batas fisik tandan ($AP50_{agn} = \mathbf{0,7636}$ vs kontrol RGB $\mathbf{0,7358}$, $\Delta = +0,0278$, $P(\Delta > 0) = 92,1\%$), namun bersifat redundan secara kondisional terhadap fitur visual RGB untuk klasifikasi tingkat kematangan buah ($I(Y; D \mid \text{RGB}) \approx 0$).
2. **Ketiadaan Keunggulan Depth Monokular (`V2-E-027` s.d. `V2-E-032`):**
   Penambahan estimasi kedalaman monokular (*monocular depth estimation*) sebagai kanal ke-4 tidak menunjukkan keunggulan performa, bahkan menyebabkan degradasi performa yang signifikan secara statistik pada korpus 953 pohon ($\Delta mAP50 = \mathbf{−0,0476}$, $CI95 = [−0,0671; −0,0274]$).
3. **Pergeseran Temporal Membatalkan Komparasi 4-Kelas Lintas-Dataset (`V2-E-022`):**
   Jeda waktu $\sim 80\text{ hari}$ antara SawitMVC-YOLO (Mei 2026) dan SawitMVC-Depth (Juli 2026) mencakup $5\text{--}11$ rotasi panen, menyebabkan proporsi kelas B3 menyusut dari $55,3\%$ menjadi $14,0\%$. Oleh karena itu, evaluasi lintas dataset wajib dikontrol atau dievaluasi pada lokalisasi agnostik.
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
