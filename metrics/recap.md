# Leaderboard Hasil Eksperimen

Dokumen ini mengonsolidasi seluruh hasil evaluasi dari [atlas metrik](README.md) menjadi papan peringkat terpadu untuk setiap korpus data uji pada sistem.

---

## Daftar task dan metrik

| Kolom | Arti | Satuan | Metrik |
|---|---|---|---|
| detection | temukan kotak tandan, tanpa kelas (= lokalisasi) | per citra | AP50 |
| detection + classification (gabungan) | kotak + label sekaligus, dari detektor one-stage | per citra | mAP50 class-aware |
| deduplication | tandan sama lintas 4 sisi, tanpa hitung ganda | per pohon | F1 fisik |
| classification | ketepatan kelas pada tandan yang sudah tertaut benar | per pohon | akurasi matched-class |
| counting | jumlah tandan per kelas B1–B4 di tingkat kohort (sasaran utama BBC) | per kohort (validation set atau test set) | bias per kelas dan macro-avg \|bias\|; MAE cacah total per pohon |

---

## Satu table leaderboard per dataset

- **combined1716:** leaderboard utama RGB, daya statistik terkuat.
- **763-depth:** RGB+D
- **953:** RGB

---

## 1. 953 (RGB)

### Leaderboard 953

| Metode/sistem | detection | det+class | dedup | classification | avg \|bias\| | MAE | Status | ID Simpul |
|---|---:|---:|---:|---:|---:|---:|:---:|---|
| YOLO26l | 0,7388 | 0,5435 | — | — | — | — | uji | `V2-E-001` |
| YOLO26s 960 px | 0,8057 | 0,5433 | — | — | — | — | uji | `AF-E-006` |
| YOLO26m 1.280 px | 0,8104 | — | — | — | — | — | uji | `AF-E-011` |
| RT-DETR-L | · | 0,5781 | — | — | — | — | uji | `V2-E-001` |
| RF-DETR-L | · | **0,6012** | — | — | — | — | uji | `V2-E-001` |
| WBF [YOLO+RT+RF] | 0,8350 | 0,5861 | — | — | — | — | uji | `V2-E-042` |
| WBF + *re-ranker* | **0,8419** | 0,5970 | — | — | — | — | uji | `MAP_BOOST` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + rotation-prior linker + Ridge counter | 0,8350 | 0,5861 | 0,8043 | 71,1% | · | 1,393 | uji | `V2-E-045` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + greedy strict | 0,8350 | 0,5861 | 0,8296 | — | 20,96% | 1,644 | uji | `V2-E-043` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + Hungarian *Anchor A* + Ridge counter | 0,8350 | 0,5861 | **0,8387** | **74,4%** | 18,78% | 1,363 | uji | `Wave-V2` |
| YOLO26m + penaut terlatih + Ridge (Pipeline Panen) | 0,8104 | — | 0,7619 | 71,6% | · | 1,402 | uji | `AF-E-012` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + rotation-prior linker + Ridge counter | 0,8373 | 0,5613 | 0,8087 | 70,0% | · | 1,253 | val | `V2-E-045` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + Hungarian *Anchor A* + Ridge counter | 0,8373 | 0,5613 | 0,8232 | 75,4% | **15,21%** | 1,253 | val | `Wave-V2` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + Hungarian *Anchor A* + *stacking* DINOv2-Large | 0,8373 | 0,5613 | 0,8232 | **76,8%** | · | 1,253 | val | `V2-E-046` |
| YOLO26m + penaut terlatih + Ridge (Pipeline Panen) | · | — | 0,7586 | 71,7% | · | 1,374 | val | `AF-E-012` |
| Plafon lokalisasi sempurna (*oracle*) + ConvNeXt-Tiny / Ridge | — | 0,6569 | — | — | — | **1,058** | oracle | `AF-E-005` |

*Legenda sel:* **—** = tidak berlaku bagi konfigurasi tersebut (detektor tanpa penaut tidak memiliki metrik tingkat pohon; korpus `combined1716` tidak memiliki nilai acuan multi-sisi tingkat pohon). **·** = dapat dihitung dari dump prediksi yang tersimpan, tetapi belum pernah dievaluasi. Alasan spesifik tiap sel `·` yang masih tersisa dirinci per node di [§6 Batasan Validitas dan Kaveat Audit](#6-batasan-validitas-dan-kaveat-audit).

### Counting kohort per kelas (sasaran BBC)

Tabel ini memuat satu-satunya metrik penilaian akhir tugas *counting*, yaitu bias per kelas. Nilainya dihitung terpisah untuk tiap split (validasi dan uji) pada tiap dataset, dengan menjumlahkan seluruh pohon dalam split tersebut.

#### Partisi Uji (Hungarian *Anchor A*, 135 Pohon)

| Kelas | Total prediksi | Total acuan | Bias absolut | Bias relatif |
|---|---:|---:|---:|---:|
| B1 | 104 | 113 | −9 | −7,96% |
| B2 | 145 | 246 | −101 | −41,06% |
| B3 | 824 | 706 | +118 | +16,71% |
| B4 | 251 | 277 | −26 | −9,39% |
| **Total** | **1.324** | **1.342** | **−18** | **−1,34%** |

*Makro-rerata nilai mutlak bias relatif (macro-avg \|bias\|):* **18,78%**

#### Partisi Validasi (Hungarian *Anchor A*, 91 Pohon)

| Kelas | Total prediksi | Total acuan | Bias absolut | Bias relatif |
|---|---:|---:|---:|---:|
| B1 | 84 | 86 | −2 | −2,33% |
| B2 | 112 | 186 | −74 | −39,78% |
| B3 | 560 | 476 | +84 | +17,65% |
| B4 | 186 | 188 | −2 | −1,06% |
| **Total** | **942** | **936** | **+6** | **+0,64%** |

*Makro-rerata nilai mutlak bias relatif (macro-avg \|bias\|):* **15,21%**

---

## 2. 763-depth (RGB+D)

### Leaderboard 763-depth

| Metode/sistem | detection | det+class | dedup | classification | avg \|bias\| | MAE | Status | ID Simpul |
|---|---:|---:|---:|---:|---:|---:|:---:|---|
| YOLO26l native | 0,7161 | 0,5163 | — | — | — | — | uji | `V2-E-034` |
| RT-DETR-L native | 0,7712 | 0,5580 | — | — | — | — | uji | `V2-E-034` |
| YOLO26l, bank `combined1716` | 0,7812 | 0,5765 | — | — | — | — | uji | `V2-E-042` |
| RF-DETR-L native | 0,7951 | 0,6129 | — | — | — | — | uji | `V2-E-034` |
| RT-DETR-L, bank `combined1716` | 0,8243 | 0,6309 | — | — | — | — | uji | `V2-E-042` |
| WBF + *re-ranker* | **0,8783** | 0,6552 | — | — | — | — | uji | `MAP_BOOST` |
| WBF [YOLO26l+RT-DETR-L+RF-DETR-L], bank `combined1716` | 0,8764 | 0,6691 | — | — | — | — | uji | `V2-E-042` |
| RF-DETR-L, bank `combined1716` | 0,8329 | **0,6711** | — | — | — | — | uji | `V2-E-042` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + greedy strict | 0,8764 | 0,6691 | **0,8590** | — | **18,65%** | 0,818 | uji | `V2-E-043` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + rotation-prior linker + Ridge counter | 0,8764 | 0,6691 | 0,8069 | 80,3% | · | 0,891 | uji | `V2-E-045` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + GSP MILP + Ridge counter (Wave-V2) | 0,8764 | 0,6691 | 0,8534 | **81,6%** | 19,62% | 0,773 | uji | `Wave-V2` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + rotation-prior linker + Ridge counter | 0,8648 | 0,6595 | 0,8257 | 83,6% | · | **0,726** | val | `V2-E-045` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + GSP MILP + Ridge counter (Wave-V2) | 0,8648 | 0,6595 | 0,8526 | 84,6% | 25,88% | 0,932 | val | `Wave-V2` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + GSP MILP + komposisi lintas-lapis | 0,8648 | 0,6595 | 0,8542 | **85,0%** | · | 0,915 | val | `V2-E-047` |

*Legenda sel:* **—** = tidak berlaku bagi konfigurasi tersebut (detektor tanpa penaut tidak memiliki metrik tingkat pohon; korpus `combined1716` tidak memiliki nilai acuan multi-sisi tingkat pohon). **·** = dapat dihitung dari dump prediksi yang tersimpan, tetapi belum pernah dievaluasi. Alasan spesifik tiap sel `·` yang masih tersisa dirinci per node di [§6 Batasan Validitas dan Kaveat Audit](#6-batasan-validitas-dan-kaveat-audit).

### Counting kohort per kelas (sasaran BBC)

Tabel ini memuat satu-satunya metrik penilaian akhir tugas *counting*, yaitu bias per kelas. Nilainya dihitung terpisah untuk tiap split (validasi dan uji) pada tiap dataset, dengan menjumlahkan seluruh pohon dalam split tersebut.

#### Partisi Uji (GSP MILP, 110 Pohon)

| Kelas | Total prediksi | Total acuan | Bias absolut | Bias relatif |
|---|---:|---:|---:|---:|
| B1 | 67 | 94 | −27 | −28,72% |
| B2 | 227 | 199 | +28 | +14,07% |
| B3 | 177 | 215 | −38 | −17,67% |
| B4 | 41 | 50 | −9 | −18,00% |
| **Total** | **512** | **558** | **−46** | **−8,24%** |

*Makro-rerata nilai mutlak bias relatif (macro-avg \|bias\|):* **19,62%**

#### Partisi Validasi (GSP MILP, 117 Pohon)

| Kelas | Total prediksi | Total acuan | Bias absolut | Bias relatif |
|---|---:|---:|---:|---:|
| B1 | 66 | 96 | −30 | −31,25% |
| B2 | 234 | 205 | +29 | +14,15% |
| B3 | 176 | 216 | −40 | −18,52% |
| B4 | 32 | 53 | −21 | −39,62% |
| **Total** | **508** | **570** | **−62** | **−10,88%** |

*Makro-rerata nilai mutlak bias relatif (macro-avg \|bias\|):* **25,88%**

---

## 3. combined1716 (RGB)

Korpus gabungan berkapasitas terbesar (1.716 pohon, 1.052 citra uji) difungsikan sebagai bank data pelatihan modul detektor. Korpus ini tidak memiliki label nilai acuan kebenaran (*ground truth*) multi-sisi tingkat pohon, sehingga evaluasinya terfokus pada tugas deteksi citra (`detection` dan `det+class`).

### Leaderboard combined1716

| Metode/sistem | detection | det+class | dedup | classification | avg \|bias\| | MAE | Status | ID Simpul |
|---|---:|---:|---:|---:|---:|---:|:---:|---|
| YOLO26l native | 0,7250 | 0,5389 | — | — | — | — | uji | `V2-E-035` |
| WBF native | **0,8104** | 0,5538 | — | — | — | — | uji | `V2-E-039` |
| RT-DETR-L native | 0,7577 | 0,5745 | — | — | — | — | uji | `V2-E-035` |
| RF-DETR-L native | 0,7850 | **0,5960** | — | — | — | — | uji | `V2-E-035` |

*Legenda sel:* **—** = tidak berlaku bagi konfigurasi tersebut (detektor tanpa penaut tidak memiliki metrik tingkat pohon; korpus `combined1716` tidak memiliki nilai acuan multi-sisi tingkat pohon). **·** = dapat dihitung dari dump prediksi yang tersimpan, tetapi belum pernah dievaluasi. Alasan spesifik tiap sel `·` yang masih tersisa dirinci per node di [§6 Batasan Validitas dan Kaveat Audit](#6-batasan-validitas-dan-kaveat-audit).

---

## 4. Karakteristik Visual & Skala Kematangan Tandan Sawit ([`docs/DATASET.md`](../docs/DATASET.md) §1)

| Kelas Kematangan | Tingkat Kematangan & Status Panen | Karakteristik Visual Dominan | Ukuran Kotak Median (Korpus 953) |
|---|---|---|---:|
| B1 | Lewat matang (siap panen) | Jingga kemerahan cerah, posisi lingkaran terbawah kanopi | 133 piksel |
| B2 | Matang optimal (siap panen) | Oranye kemerahan bersemburat ungu kehitaman | 120 piksel |
| B3 | Matang awal (mengkal / belum siap) | Ungu kemerahan kehitaman | 107 piksel |
| B4 | Mentah (muda / belum siap) | Hitam kehijauan pekat, tertanam rapat di sela pelepah | 93 piksel |

---

## 5. Estimasi Selang Kepercayaan 95% Profil Terkunci Uji

Dihitung melalui simulasi *bootstrap* berpasangan sebanyak 2.000 ulangan (*random seed* 42) pada seluruh metrik alur kerja:

| Metrik Evaluasi | Korpus 953: Hungarian *Anchor A* (135 pohon) | Korpus 763-depth: GSP MILP (110 pohon) |
|---|---|---|
| *F1 fisik* | 0,8387 [0,8174; 0,8587] | 0,8534 [0,8301; 0,8761] |
| Akurasi *matched-class* | 0,7442 [0,7112; 0,7735] | 0,8162 [0,7765; 0,8556] |
| Makro-*F1* ujung ke ujung | 0,6034 [0,5655; 0,6382] | 0,6519 [0,6046; 0,6918] |
| MAE cacah total | 1,363 [1,163; 1,585] | 0,773 [0,609; 0,945] |
| Akurasi toleransi cacah $\pm 1$ | 0,6370 [0,5556; 0,7185] | 0,8545 [0,7818; 0,9182] |

---

## 6. Batasan Validitas dan Kaveat Audit

| Aspek Batasan / Kaveat | Implikasi terhadap Pembacaan dan Interpretasi Data | Sumber Rujukan |
|---|---|---|
| Ambiguitas Makro-*F1* (`AF-E-012`) | Terdapat dua nilai: $0,6692$ (klaster terpasangkan) dan $0,5201$ (ujung ke ujung). Hanya nilai $0,5201$ yang setara dengan baseline $0,6034$. | [`EVIDENCE.md`](../docs/research_2026-09-06/EVIDENCE.md) |
| Pipeline Panen (`AF-E-012`, `AF-E-013`) | Menggunakan 132 pohon, detektor tunggal YOLO26m. Unggul pada estimasi B1 toleransi $\pm 1$ ($0,970$) dan akurasi ordinal ($0,9946$). | `AF-E-012`, `AF-E-013` |
| Pelanggaran kendala fisik (`AF-E-010`) | Telah dikoreksi oleh `AF-E-014`: pada profil terkunci dengan `max_size` $\le 3$, tingkat pelanggaran adalah $0,00\%$. | `AF-E-014`, `AF-E-016` |
| Status eksperimen `AF-E` | Eksperimen audit (`AF-E`) berfungsi sebagai diagnostik pelengkap, bukan pengganti angka acuan profil terkunci. | [Atlas 07](07_audit_forensik.md) |
| Status gelombang validasi (`V2-E-046`, `V2-E-047`, `V2-E-048`) | Ketiganya dipilih pada partisi validasi tanpa menyentuh partisi uji, sehingga tidak menggantikan angka uji terkunci. `V2-E-048` (pelatihan ulang *head* sadar-komposisi) tidak memberi kenaikan: makro-*F1* turun $0,6890 \to 0,6850$ dan disimpan sebagai kontrol negatif. | `experiments/STATUS.md` §8–10 |
| Kebocoran data (*leakage*) `combined1716` | Irisan identitas pohon (`tree_id`) antar-partisi belum diaudit tuntas. | [`ANALISIS_PIPELINE.md`](../docs/ANALISIS_PIPELINE_MENDALAM.md) |
| Sel `detection` tak terisi, RT-DETR-L dan RF-DETR-L (`V2-E-001`, tabel 953) | Bobot `runs/rtdetr_l_e60_i1280_v2repro/` dan `runs/rfdetr_l_e60_i1280_v2repro/` tidak ada di repositori maupun bucket cadangan Hugging Face; tidak ada dump prediksi tersimpan. Tidak dapat diisi tanpa pelatihan ulang 60 *epoch* penuh — di luar cakupan audit ini. | Audit sesi 2026-09-07 |
| Sel `detection` tak terisi, Pipeline Panen (`AF-E-012`, val, tabel 953) | Bobot detektor class-agnostic bespoke (`runs_panen/agnostik_m1280/`) tidak ada di repositori maupun bucket. Kolom `det+class` pada baris ini dikoreksi dari `·` menjadi `—` karena detektornya class-agnostic murni (setara baris uji), bukan sekadar belum dievaluasi. | Audit sesi 2026-09-07 |
| Sel `avg \|bias\|` tak terisi (7 sel: `V2-E-045` ×4, `V2-E-046`, `V2-E-047`, `AF-E-012` ×2) | Formula (makro-rerata \|bias relatif\| B1–B4) diverifikasi persis terhadap keempat nilai `Wave-V2` yang sudah terisi (dari medan `gsp_best_by_class.metrics.classification.confusion_prediction_rows`). Diagnosis presisi per node (sesi 2026-09-07, CPU-only karena GPU dipakai retrain `V2-E-001`): **`V2-E-045`** (4 sel) memerlukan inferensi GPU pada partisi *train* (model *count* dilatih di sana). **`V2-E-046`** (953 val): `harness.py` menghitung `confusion_prediction_rows` secara internal tetapi `large_stacker.py` membuangnya sebelum disimpan ke disk; regenerasi butuh fitur DINOv2-Large yang sudah diekstrak (`/workspace/dino_head/features_large/953/val_dinolargefeat.npy`) dan model *stacker* (`/workspace/cluster_head/artifacts/953_*.joblib`) — keduanya tidak ada di kontainer sesi ini. **`V2-E-047`** (763-depth val): pola sama; `composition_aware_head.py` bergantung pada `/workspace/cluster_head` dan `/workspace/pipeline_v2` (peta fitur, `edge_v2_geo.joblib`, `count_ridge_geo.joblib`) — tidak ada. **`AF-E-012`** (kedua split): `panen_eval.py`/`panen_final.py` bergantung pada `pickle.load(dets.pkl)` di `/workspace/results_panen` (tidak ada); regenerasi butuh dua bobot yang sudah diketahui hilang (`runs_panen/agnostik_m1280/weights/best.pt`, `crops953/corn_best.pt`). Kohort 132 pohon terverifikasi dari `panen_results.json.per_tree_test`, tetapi hanya berisi medan biner matang/belum, bukan rincian B1–B4. **Pola umum:** `V2-E-043` berhasil diisi karena dump proposal box-nya tersimpan di `results/` yang dilacak Git; keempat sel lain bergantung pada artefak antara (cache fitur, model *joblib*, dump deteksi) yang hidup di luar repo dan tidak ikut ter-*checkout* di kontainer sesi ini — bukan kekurangan metodologi, melainkan keterbatasan lingkungan eksekusi. Baris oracle `AF-E-005` dikoreksi dari `·` menjadi `—`: studi ini murni plafon klasifikasi per-*crop*, tanpa tahapan penautan/pencacahan tingkat pohon, sejalan dengan kolom `dedup`/`classification` yang sudah `—`. `V2-E-043` (kedua korpus, uji) terisi sesi ini: medan `confusion_matrix` ditambahkan ke `multiview_metrics()` (`scripts/eval_remote_pipeline_postprocess.py`), dijalankan pada proposal WBF uji yang sudah tersimpan, hasil F1 dan total acuan tervalidasi presisi penuh terhadap nilai `dedup` yang sudah dipublikasikan (0,8296 dan 0,8590). | Audit sesi 2026-09-07 |

---

## 7. Sumber Data dan Keterlacakan Artefak

| Rujukan Bagian | Berkas Artefak Sumber Data |
|---|---|
| detection, det+class | [`combined1716`](../results/combined1716), [`new763`](../results/new763); [`detector_matrix.json`](../results/audit_forensik_2026-09-06/detector_matrix.json), [`class_agnostic_metrics_audit_2026-09-03.json`](../results/class_agnostic_metrics_audit_2026-09-03.json), [`agnostic_ap50_sesi2026-08.json`](../results/agnostic_ap50_sesi2026-08.json) |
| dedup, classification | [`pipeline_combined1716_generalization_locked.json`](../results/remote_eval_2026-08-27/metrics/pipeline_combined1716_generalization_locked.json) |
| Gelombang validasi (`V2-E-046`, `V2-E-047`) | [`953_large_stacker_bias_val_bootstrap.json`](../results/remote_eval_2026-08-28/validation_wave/reports/953_large_stacker_bias_val_bootstrap.json), [`depth_composition_aware_head_results_val.json`](../results/remote_eval_2026-08-28/validation_wave/reports/depth_composition_aware_head_results_val.json) |
| Counting kohort per kelas | Diturunkan dari medan `metrics.classification.confusion_prediction_rows` pada artefak `Wave-V2` |
| Analisis audit dan plafon | [`experiments/AUDIT-FORENSIK-2026-09-06.md`](../experiments/AUDIT-FORENSIK-2026-09-06.md) |
| `detection` bank `combined1716` uji, 763-depth (baris YOLO26l/RT-DETR-L/RF-DETR-L) | Dihitung ulang dari dump prediksi tersimpan (`results/remote_eval_2026-08-27/predictions/`) terhadap GT `ULM-DS-Lab/SawitMVC-Depth-YOLO` (test); lihat [`agnostic_ap50_combined1716_depth_test_2026-09-07.json`](../results/agnostic_ap50_combined1716_depth_test_2026-09-07.json) |
| `detection` YOLO26l uji, 953 (`V2-E-001`) | Inferensi baru bobot lokal `models/yolo26l_e60_i1280_v2repro/best.pt` terhadap GT `ULM-DS-Lab/SawitMVC-YOLO` (test); lihat [`agnostic_ap50_v2repro_953_2026-09-07.json`](../results/agnostic_ap50_v2repro_953_2026-09-07.json) |
| `detection`/`det+class` gelombang validasi (`V2-E-045`, `Wave-V2`, `V2-E-046`, `V2-E-047`) | Inferensi baru tiga bobot bank `combined1716` pada partisi validasi kedua korpus, difusi WBF (protokol identik `results/remote_eval_2026-08-27/README.md` §1); lihat [`wbf_val_combined1716_2026-09-07.json`](../results/wbf_val_combined1716_2026-09-07.json) |
| `avg \|bias\|` uji, `V2-E-043` (kedua korpus) | Dihitung dari medan `confusion_matrix` baru pada `multiview_metrics()` (`scripts/eval_remote_pipeline_postprocess.py`), dijalankan pada proposal WBF uji tersimpan dengan hiperparameter dari `pipeline_combined1716_greedy_test_tuned.json`; lihat [`class_bias_v2e043_953_uji_2026-09-07.json`](../results/class_bias_v2e043_953_uji_2026-09-07.json), [`class_bias_v2e043_depth_uji_2026-09-07.json`](../results/class_bias_v2e043_depth_uji_2026-09-07.json) |
