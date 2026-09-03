# Atlas Metrik: Komputasi, Kompleksitas Model, dan Diagnostik Sinyal

Dokumen ini memuat inventaris profil komputasi (jumlah parameter, GFLOPs, durasi pelatihan/inferensi), diagnostik fisik sinyal modalitas kedalaman (*depth signal SNR*, kuantisasi piksel, korelasi Spearman, uji Kruskal–Wallis), audit integritas data, serta pengujian ketidakpastian statistik berbasis *bootstrap confidence interval* 95%.

---

## 1. Kompleksitas Komputasi Arsitektur Detektor

*Pengukuran parameter dan beban komputasi teoritis pada resolusi $1.280 \times 1.280$ piksel.*

| Arsitektur Model | Jumlah Parameter | GFLOPs ($1.280\text{ px}$) | Tipe Arsitektur / *Backbone* | Latensi Inferensi / Citra (GPU T4/V100) | Waktu Latih (60 ep, 953 pohon) | Rujukan Artefak |
|---|---:|---:|---|---:|---:|---|
| **RF-DETR-L** (RGB, 3-ch) | **35,7 Juta** | ⚠ TIDAK-BISA-DIVERIFIKASI | ResNet50-DINO / Deformable DETR L | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | `results/perkelas_pycoco_v2repro.json` (`params_juta`) — parameter terverifikasi; GFLOPs/latensi/durasi tidak berdasar. |
| **RT-DETR-L** (RGB, 3-ch) | **33,0 Juta** | ⚠ TIDAK-BISA-DIVERIFIKASI | HGNetv2 + Hybrid Transformer Encoder/Decoder | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | `results/perkelas_pycoco_v2repro.json` (`params_juta`) — parameter terverifikasi; GFLOPs/latensi/durasi tidak berdasar. |
| **YOLO26l** (RGB, 3-ch) | **26,3 Juta** | ⚠ TIDAK-BISA-DIVERIFIKASI | CSP-DarkNet berbasis CNN konvensional | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | `results/perkelas_pycoco_v2repro.json` (`params_juta`) — HANYA jumlah parameter yang terverifikasi ke sumber; kata "GFLOPs"/"latency"/"latensi" TIDAK MUNCUL SATU KALI PUN di seluruh repositori (`scripts/`, `results/`, dokumentasi) selain di berkas `metrics/` ini sendiri — tidak ada bukti profiling GFLOPs/latensi pernah dijalankan. Path lama `models/yolo26l_e60_i1280_v2repro/` bukan sumber untuk klaim komputasi. |
| **YOLO26l** (RGB+D, 4-ch) | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | CSP-DarkNet *Conv1 Stem* modifikasi 4 kanal | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | Path lama `results/V2-E-010/` fiktif; seluruh angka baris ini tidak berdasar (lihat catatan GFLOPs di atas). |
| **ResNet18** (Crop Head) | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | ResNet18 Standar ($224 \times 224\text{ px}$) | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | Path lama `results/V2-E-015/` fiktif; seluruh angka baris ini tidak berdasar. |
| **GNN Penaut** (Relational) | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | 2-Layer Relational Graph Conv | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | Seluruh angka baris ini tidak berdasar. |

> **Catatan audit menyeluruh Tabel 1**: kolom GFLOPs, Latensi Inferensi, dan Waktu Latih pada SELURUH baris tabel ini tidak dapat diverifikasi ke sumber mana pun. Pencarian `grep -rli "gflop\|latency\|latensi"` di seluruh repositori (skrip, hasil JSON, dokumentasi) hanya menemukan istilah tersebut di dalam `metrics/06_komputasi_dan_diagnostik.md` dan `metrics/README.md` sendiri — indikasi kuat bahwa profiling komputasi ini **tidak pernah benar-benar dijalankan** dan seluruh angka di 3 kolom tersebut kemungkinan besar dikarang untuk melengkapi struktur tabel. Hanya kolom "Jumlah Parameter" yang tervalidasi (`params_juta` di JSON hasil pycocotools).

---

## 2. Diagnostik Fisik Sinyal Kedalaman (*Depth Diagnostics*)

*Analisis properti fisik sensor Orbbec Y16 (skala mm) dan komparasi terhadap kedalaman monokular (`V2-E-014` & `probe_depth_signal.py`).*

| Parameter Diagnostik | Nilai Terukur | Interpretasi Fisik & Metodologis | Simpul Terkait |
|---|---|---|---|
| **Fraksi Piksel Valid dalam Kotak Objek** | **$95,1\%$** ($4,9\%$ *invalid*) | *Drop-out* akibat refleksi permukaan pelepah dan bayangan optik sensor ToF/IR. Nilai lama `88,4%` salah; catatan penting entri asli: angka "29% invalid" yang lama beredar di proyek ini ternyata dihitung di LATAR, bukan objek. | `V2-E-014` |
| **Relief Lokal Median per Kelas (Tandan vs Kanopi Sekitar)** | **B1 $+2,8\text{ cm}$ · B2 $0,0\text{ cm}$ · B3 $-1,5\text{ cm}$ · B4 $-5,1\text{ cm}$** | Monoton sempurna menurut kelas kematangan — relief menyusut seiring matang. Nilai lama (`Δz≈142mm` tunggal) tidak cocok sumber; relief bervariasi per kelas, bukan angka tunggal. | `V2-E-014` |
| **Langkah Kuantisasi Resolusi (di Z=2,5m, median dataset)** | **$\approx 2,91\text{ cm / level}$** (bukan mm) | Sinyal relief kelas (0,8–5,1 cm) hanya setara 0,27–1,8 *level* kuantisasi — SNR per piksel ≈ 0,3, sangat lemah. Nilai lama `3,92 mm/LSB` salah satuan/besaran (real ≈29,1 mm, bukan 3,92mm) dan kesimpulannya justru TERBALIK — kuantisasi ini dinyatakan entri asli sebagai penyebab sinyal SULIT dipakai, bukan "memadai". | `V2-E-014` |
| **Uji Kruskal–Wallis Relief Lintas Kelas (B1–B4)** | **$H = 99,8$, $p = 1,7\times10^{-21}$** ($p \ll 0,001$) | **Koreksi kritis — kesimpulan lama TERBALIK.** Relief SANGAT signifikan berbeda antarkelas (monoton sempurna), BUKAN independen. Nilai lama `H=3,12,p=0,374` salah total dan membalik arah temuan ilmiah entri V2-E-014 (yang sebenarnya berjudul "Hasil B — CONFIRMED... Monoton sempurna"). | `V2-E-014` |
| **Korelasi Spearman (Sensor vs Mono)** | ⚠ TIDAK-BISA-DIVERIFIKASI | Nilai `ρ=0,4820`/`ρ=0,1210` belum ditemukan di entri narasi V2-E-016 yang dibaca (entri tersebut hanya mencatat **Verdict: FALSIFIED** tanpa tabel angka Spearman); perlu baca `results/probe_fitur_depth.json` atau skrip `probe_mono_vs_sensor.py` langsung. | `probe_mono_vs_sensor.py` |
| **Korelasi Spearman dalam Kotak Objek** | ⚠ TIDAK-BISA-DIVERIFIKASI | Sama seperti di atas — belum diverifikasi ke sumber JSON. | `probe_mono_vs_sensor.py` |

---

## 3. Audit Integritas Data & Artefak

| ID Temuan Audit | Deskripsi Masalah | Dampak Metodologis | Tindakan Koreksi | Simpul Rujukan |
|---|---|---|---|---|
| **TIFF Korup** | 39 berkas citra TIFF depth korup di dataset turunan 953 pohon | Ultralytics melewati citra korup secara diam-diam tanpa melempar *exception*. | Pembersihan dataset dan pembentukan ulang partisi symlink. | `V2-E-028` |
| **Pretrain Contamination** | Bobot pretrain `agn953_full` mengandung irisan pohon partisi validasi | Estimasi awal $AP50 = 0,8101$ terdistorsi optimisme semu. | Dihitung ulang pada evaluasi ketat menjadi $AP50 = \mathbf{0,7702}$. | `V2-E-025` |
| **Temporal Shift** | Jeda akuisisi $\sim 80\text{ hari}$ antara korpus 953 dan 352 | Distribusi kelas B3 anjlok dari $55,3\%$ ke $14,0\%$. | Komparasi 4-kelas lintas dataset dinyatakan tidak valid secara ilmiah. | `V2-E-022` |

---

## 4. Selang Kepercayaan Statistik (*Bootstrap Confidence Intervals*)

*Estimasi ketidakpastian 95% ($1.000\text{ replikasi bootstrap}$) pada metrik evaluasi utama.*

| Parameter Evaluasi | Estimasi Titik | Lebar Selang Kepercayaan ($CI95$) | Selang Kepercayaan $CI95$ | Status Signifikansi Statistik | Simpul Rujukan | Catatan Audit |
|---|---:|---:|---|---|---|---|
| $mAP50$ RF-DETR-L (SawitMVC-YOLO 953 Test) | 0,6012 | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | `V2-E-001` | CI ini belum ditemukan di entri V2-E-001/V2-E-038 yang dibaca (V2-E-038 melaporkan CI untuk korpus 763/1716, bukan 953 v2repro asli); perlu baca `results/bootstrap_map.py` output langsung. |
| $mAP50$ **YOLO26l-RGBD `edge`** (SawitMVC-Depth 352 Test) | **0,4270** *(bukan RF-DETR-L 0,4544)* | 0,1167 | $[0,3771; 0,4938]$ | **Rentang lebar karena ukuran uji kecil (410 kotak)** — 26× lebih lebar dari jarak antar-model teratas | `V2-E-023` | **Koreksi**: baris lama mencampur titik-estimasi RF-DETR-L (dari V2-E-003) dengan lebar-CI milik model LAIN (YOLO26l-`edge`, dari V2-E-023) — dua eksperimen berbeda digabung jadi satu baris keliru. Nilai di atas adalah CI riil V2-E-023 yang benar-benar koheren (satu model, satu CI). |
| $\Delta AP50_{agn}$ Depth Edge vs RGB Kontrol | **$+0,0278$** | 0,0769 | $[-0,0121; +0,0648]$ | **Belum signifikan** ($P(\Delta > 0) = 92,1\%$, mendekati tapi tidak mencapai ambang 95%) | `V2-E-024` | **Koreksi kritis — arah kesimpulan lama TERBALIK.** CI lama `[+0,0010; +0,0546]` (seluruhnya positif) tidak cocok sumber dan secara keliru menyiratkan signifikan; CI riil `[−0,0121; +0,0648]` MEMUAT NOL — Verdict asli entri ini eksplisit "POSITIF TAPI BELUM KONKLUSIF", bukan "signifikan". |
| $\Delta mAP50$ Mono 4-ch vs RGB (953 Test) | **$−0,0476$** | 0,0397 | $[−0,0671; −0,0274]$ | **Signifikan mengalami penurunan performa** | `V2-E-029` | Cocok sumber persis. |
| $\Delta mAP50$ Mono 4-ch vs RGB (352 Test) | $+0,0266$ | 0,0780 | $[−0,0124; +0,0656]$ | Tidak signifikan (mencakup nilai nol) | `V2-E-030` | Cocok sumber persis. |
| $\Delta mAP50$ 5-ch vs Sensor Depth (352 Test) | **$−0,0504$** | 0,0410 | $[−0,0709; −0,0299]$ | **Signifikan mengalami penurunan performa** | `V2-E-031` | Cocok sumber persis. |
