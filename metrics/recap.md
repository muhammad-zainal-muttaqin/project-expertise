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

Urutan baris: terbaik di atas. Karena satu tabel memuat tahapan yang berbeda, pengurutan dilakukan di dalam kelompok yang sebanding, yaitu detektor menurut `det+class`, lalu pipeline lengkap menurut `classification`, uji sebelum validasi, dan plafon *oracle* di baris terakhir.

---

## 1. 953 (RGB)

### Leaderboard 953

| Metode/sistem | imgsz | detection | det+class | dedup | classification | avg \|bias\| | MAE | Status | ID Simpul |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| RF-DETR-L | 1.280 px | 0,7752 | **0,6012** | — | — | — | — | uji | `V2-E-001` |
| WBF + *re-ranker* | 1.280 px | **0,8419** | 0,5970 | — | — | — | — | uji | `MAP_BOOST` |
| WBF [YOLO+RT+RF] | 1.280 px | 0,8350 | 0,5861 | — | — | — | — | uji | `V2-E-042` |
| RT-DETR-L | 1.280 px | 0,7437 | 0,5781 | — | — | — | — | uji | `V2-E-001` |
| YOLO26l | 1.280 px | 0,7388 | 0,5435 | — | — | — | — | uji | `V2-E-001` |
| YOLO26s | 960 px | 0,8057 | 0,5433 | — | — | — | — | uji | `AF-E-006` |
| YOLO26m | 1.280 px | 0,8104 | — | — | — | — | — | uji | `AF-E-011` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + Hungarian *Anchor A* + Ridge counter | 1.280 px | 0,8350 | 0,5861 | **0,8387** | **74,4%** | 18,78% | 1,363 | uji | `Wave-V2` |
| YOLO26m + penaut terlatih + Ridge (Pipeline Panen) | 1.280 px | 0,8104 | — | 0,7619 | 71,6% | 23,22% | 1,402 | uji | `AF-E-012` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + rotation-prior linker + Ridge counter | 1.280 px | 0,8350 | 0,5861 | 0,8043 | 71,1% | 16,82% | 1,393 | uji | `V2-E-045` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + greedy strict | 1.280 px | 0,8350 | 0,5861 | 0,8296 | — | 20,96% | 1,644 | uji | `V2-E-043` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + Hungarian *Anchor A* + *stacking* DINOv2-Large | 1.280 px | 0,8313 | 0,5691 | 0,8232 | **76,8%** | · | 1,253 | val | `V2-E-046` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + Hungarian *Anchor A* + Ridge counter | 1.280 px | 0,8313 | 0,5691 | 0,8232 | 75,4% | **15,21%** | 1,253 | val | `Wave-V2` |
| YOLO26m + penaut terlatih + Ridge (Pipeline Panen) | 1.280 px | 0,8125 | — | 0,7586 | 71,7% | 25,65% | 1,374 | val | `AF-E-012` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + rotation-prior linker + Ridge counter | 1.280 px | 0,8313 | 0,5691 | 0,8087 | 70,0% | 17,49% | 1,253 | val | `V2-E-045` |
| Plafon lokalisasi sempurna (*oracle*) + ConvNeXt-Tiny / Ridge | — | — | 0,6569 | — | — | — | **1,058** | oracle | `AF-E-005` |

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

| Metode/sistem | imgsz | detection | det+class | dedup | classification | avg \|bias\| | MAE | Status | ID Simpul |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| RF-DETR-L, bank `combined1716` | 1.280 px | 0,8329 | **0,6711** | — | — | — | — | uji | `V2-E-042` |
| WBF [YOLO26l+RT-DETR-L+RF-DETR-L], bank `combined1716` | 1.280 px | 0,8764 | 0,6691 | — | — | — | — | uji | `V2-E-042` |
| WBF + *re-ranker* | 1.280 px | **0,8783** | 0,6552 | — | — | — | — | uji | `MAP_BOOST` |
| RT-DETR-L, bank `combined1716` | 1.280 px | 0,8243 | 0,6309 | — | — | — | — | uji | `V2-E-042` |
| RF-DETR-L native | 1.280 px | 0,7951 | 0,6129 | — | — | — | — | uji | `V2-E-034` |
| YOLO26l, bank `combined1716` | 1.280 px | 0,7812 | 0,5765 | — | — | — | — | uji | `V2-E-042` |
| RT-DETR-L native | 1.280 px | 0,7712 | 0,5580 | — | — | — | — | uji | `V2-E-034` |
| YOLO26l native | 1.280 px | 0,7161 | 0,5163 | — | — | — | — | uji | `V2-E-034` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + GSP MILP + Ridge counter (Wave-V2) | 1.280 px | 0,8764 | 0,6691 | 0,8534 | **81,6%** | 19,62% | 0,773 | uji | `Wave-V2` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + rotation-prior linker + Ridge counter | 1.280 px | 0,8764 | 0,6691 | 0,8069 | 80,3% | **13,93%** | 0,891 | uji | `V2-E-045` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + greedy strict | 1.280 px | 0,8764 | 0,6691 | **0,8590** | — | 18,65% | 0,818 | uji | `V2-E-043` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + GSP MILP + komposisi lintas-lapis | 1.280 px | 0,8648 | 0,6595 | 0,8542 | **85,0%** | · | 0,915 | val | `V2-E-047` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + GSP MILP + Ridge counter (Wave-V2) | 1.280 px | 0,8648 | 0,6595 | 0,8526 | 84,6% | 25,88% | 0,932 | val | `Wave-V2` |
| WBF[YOLO26l+RT-DETR-L+RF-DETR-L] + rotation-prior linker + Ridge counter | 1.280 px | 0,8648 | 0,6595 | 0,8257 | 83,6% | 14,10% | **0,726** | val | `V2-E-045` |

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

| Metode/sistem | imgsz | detection | det+class | dedup | classification | avg \|bias\| | MAE | Status | ID Simpul |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| RF-DETR-L native | 1.280 px | 0,7850 | **0,5960** | — | — | — | — | uji | `V2-E-035` |
| RT-DETR-L native | 1.280 px | 0,7577 | 0,5745 | — | — | — | — | uji | `V2-E-035` |
| WBF native | 1.280 px | **0,8104** | 0,5538 | — | — | — | — | uji | `V2-E-039` |
| YOLO26l native | 1.280 px | 0,7250 | 0,5389 | — | — | — | — | uji | `V2-E-035` |

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
| Sel `detection` RF-DETR-L (`V2-E-001`, tabel 953) terisi via retrain | Bobot `runs/rfdetr_l_e60_i1280_v2repro/` asli hilang, sama seperti RT-DETR-L. Dilatih ulang mengikuti protokol persis `experiments/EKSPERIMEN.md` `V2-E-001` (`RFDETRLarge(resolution=1280, gradient_checkpointing=True).train(epochs=60, batch_size=4, grad_accum_steps=4, seed=42)`) — konfigurasi dan data (716/96/141 pohon train/val/test, dataset tidak berubah sejak commit terakhir 2026-05-19) diverifikasi identik dengan run asli sebelum dipercaya. *Val* mAP50 selama pelatihan sempat terlihat jauh dari klaim "0,6" karena keliru dibandingkan dengan mAP50 *test* asli, bukan *val* asli (0,5741) — setelah dikoreksi, keduanya sebanding. Checkpoint `best_ema` (puncak sekitar epoch 5, EMA mAP50-95 0,2641) dievaluasi pada test: reproduksi `det+class` mAP50=0,5965 vs. nilai asli 0,6012 (selisih 0,0047, sebanding RT-DETR-L). Bobot diunggah ke bucket cadangan. | [`rfdetr_l_v2repro_953_retrain_2026-09-07.json`](../results/rfdetr_l_v2repro_953_retrain_2026-09-07.json) |
| Sel `detection` RT-DETR-L (`V2-E-001`, tabel 953) terisi via retrain | Bobot `runs/rtdetr_l_e60_i1280_v2repro/` asli juga hilang, sama seperti RF-DETR-L, tetapi dilatih ulang 60 *epoch* penuh sesi ini mengikuti protokol persis `experiments/EKSPERIMEN.md` `V2-E-001` (imgsz=1280, batch=4, seed=42, cos\_lr=True, patience=60). Reproduksi `det+class` mAP50=0,5718 vs. nilai asli 0,5781 (selisih 0,0063, dalam batas wajar variasi pelatihan ulang). Bobot baru diunggah ke bucket cadangan (`project-expertise/runs/rtdetr_l_e60_i1280_v2repro_retrain/weights/best.pt`) agar tidak hilang lagi. | [`rtdetr_l_v2repro_953_retrain_2026-09-07.json`](../results/rtdetr_l_v2repro_953_retrain_2026-09-07.json) |
| Sel `detection` Pipeline Panen (`AF-E-012`, val, tabel 953) terisi setelah dua percobaan | **Koreksi:** klaim awal bahwa bobot detektor (`runs_panen/agnostik_m1280/`) dan classifier (`crops953/corn_best.pt`) tidak ada di bucket cadangan adalah **keliru** — keduanya ditemukan di bawah prefix `audit_forensik_2026-09-06/`. Percobaan pertama mengisi sel ini dari dump `dets.pkl` (ambang `conf=0,10` bawaan `detect()`) memberi AP50=0,7487 pada test, selisih 0,062 dari nilai `detection` AF-E-011/AF-E-012 yang sudah terpublikasi (0,8104, detektor identik) — terlalu besar untuk derau biasa, dibatalkan. Diinferensi ulang dengan `conf=0,001` (sebanding kolom `detection` lain): AP50=**0,8125** pada val, selisih hanya 0,002 dari nilai test 0,8104 — mengonfirmasi angka benar. Kolom `det+class` pada baris ini tetap `—` karena detektornya class-agnostic murni (setara baris uji), bukan sekadar belum dievaluasi. | [`agnostic_ap50_panen_val_2026-09-07.json`](../results/agnostic_ap50_panen_val_2026-09-07.json) |
| **Koreksi:** `detection`/`det+class` baris val 953 (`V2-E-045`, `Wave-V2`, `V2-E-046`) salah pada pengisian sesi 2026-09-07 sebelumnya | Dump prediksi bank `combined1716` untuk partisi val 953 yang dipakai untuk mengisi sel ini sebelumnya (`wbf_val_combined1716_2026-09-07.json`) ternyata hanya memuat 263/404 citra val — seluruh 40 citra `LONSUM_*` dan 101 dari 364 citra `DAMIMAS_*` hilang (kemungkinan bug pemotongan pada inferensi sesi tersebut). Nilai lama **0,8373 / 0,5613** salah, dihitung dari ~65% data. Diinferensi ulang penuh (404/404 citra terverifikasi) dan dikoreksi menjadi **0,8313 / 0,5691**. Partisi val 763-depth pada dump yang sama diperiksa dan terkonfirmasi lengkap (468/468) — tidak perlu koreksi. Ditemukan saat mengisi `avg \|bias\|` `V2-E-045`, karena F1 hasil rekonstruksi (0,69) jauh dari nilai `dedup` yang sudah dipublikasikan (0,8087), yang memicu audit lebih lanjut. | [`class_bias_v2e045_2026-09-07.json`](../results/class_bias_v2e045_2026-09-07.json) |
| Sel `avg \|bias\|` tak terisi (2 sel: `V2-E-046`, `V2-E-047`) | Formula (makro-rerata \|bias relatif\| B1–B4) diverifikasi persis terhadap keempat nilai `Wave-V2` yang sudah terisi. `V2-E-045` (4 sel), `V2-E-043` (2 sel), dan `AF-E-012` (2 sel) berhasil diisi sesi ini karena artefak sumbernya (dump proposal box, atau bobot detektor+*classifier*) ternyata tersimpan di `results/` Git atau bucket cadangan. `V2-E-046`/`V2-E-047` diinvestigasi mendalam (sesi 2026-09-07) sampai akar penyebabnya, bukan sekadar dicoba sekali: **`V2-E-046`** (953 val, *stacking* DINOv2-Large) butuh *pipeline* fitur bertingkat yang tidak pernah tersimpan: indeks *crop* (`dino_head/crops/{dataset}/{split}_index.npz` + `{split}_rgb224.npy`) dan fitur DINOv2-Large hasil ekstraksi (`dino_head/features_large/.../dinolargefeat.npy`) — skrip ekstraksi fiturnya (`extract_large_features.py`) ada, tetapi skrip **pembangun indeks *crop*-nya sendiri tidak ada di manapun** (diperiksa seluruh repo dan bucket) — hanya skrip yang MEMBACA indeks itu yang tersimpan. Secara prinsip dapat direkonstruksi (skema *crop* dapat diturunkan dari kode konsumennya, proposal box sudah tersedia dari kerja `V2-E-045` sesi ini) — diperkirakan ~2–4 jam kerja (ekstraksi DINOv2-Large + rekayasa ulang pembangun indeks), tetapi **belum dikerjakan** atas keputusan eksplisit sesi ini. **`V2-E-047`** (763-depth val, komposisi lintas-lapis) jauh lebih terhambat: rantai impornya (`composition_aware_head.py` → `pipeline_v2.py` → `harness.py`) butuh `link_global_setpartition.py` — solver **MILP *global set-partition*** untuk penautan proposal 763-depth — dan modul ini **tidak ada sama sekali** di repositori maupun bucket cadangan (dicari menyeluruh). Ini bukan berkas data yang bisa diregenerasi ulang, melainkan kode algoritma optimisasi itu sendiri; merekonstruksinya berarti menulis ulang solver MILP dari nol tanpa rujukan formulasi asli — kategori pekerjaan yang jauh lebih besar dan berisiko dibanding seluruh sel lain yang berhasil diisi sesi ini, sehingga **tidak dikerjakan**. Baris oracle `AF-E-005` dikoreksi dari `·` menjadi `—`: studi ini murni plafon klasifikasi per-*crop*, tanpa tahapan penautan/pencacahan tingkat pohon, sejalan dengan kolom `dedup`/`classification` yang sudah `—`. `V2-E-043` (kedua korpus, uji): medan `confusion_matrix` ditambahkan ke `multiview_metrics()` (`scripts/eval_remote_pipeline_postprocess.py`), F1 dan total acuan tervalidasi presisi penuh terhadap nilai `dedup` terpublikasi (0,8296 dan 0,8590). `AF-E-012` (kedua split): bobot ditemukan di bucket (`audit_forensik_2026-09-06/`), pipeline diregenerasi penuh, F1/`class4_acc` cocok dalam <0,005. `V2-E-045` (4 sel): medan `confusion_matrix` ditambahkan ke `evaluate_payload()` (`scripts/evaluate_remote_count_reconciled.py`), F1/`class4_acc` hasil rekonstruksi cocok dalam <0,004 dari nilai terpublikasi di keempat sel (satu cocok presisi penuh). | Audit sesi 2026-09-07 |

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
| `detection` RT-DETR-L uji, 953 (`V2-E-001`) | Retrain 60 *epoch* penuh (bobot asli hilang), inferensi pada test split; lihat [`rtdetr_l_v2repro_953_retrain_2026-09-07.json`](../results/rtdetr_l_v2repro_953_retrain_2026-09-07.json), dump [`pred_rtdetr_l_v2repro_953_test.npz`](../results/pred_rtdetr_l_v2repro_953_test.npz) |
| `detection` RF-DETR-L uji, 953 (`V2-E-001`) | Retrain (bobot asli hilang), checkpoint `best_ema` dievaluasi pada test split; lihat [`rfdetr_l_v2repro_953_retrain_2026-09-07.json`](../results/rfdetr_l_v2repro_953_retrain_2026-09-07.json), dump [`pred_rfdetr_l_v2repro_953_test.npz`](../results/pred_rfdetr_l_v2repro_953_test.npz) |
| `avg \|bias\|` uji+val, `AF-E-012` (953) | Pipeline Panen diregenerasi penuh (bobot ditemukan di bucket cadangan `audit_forensik_2026-09-06/`) via `scripts/audit_forensik/panen_pipeline.py` → `panen_eval.py` → `panen_final.py`, medan `pred_class4_counts`/`gt_class4_counts` baru ditambahkan ke `evaluate()`; lihat [`class_bias_afe012_953_2026-09-07.json`](../results/class_bias_afe012_953_2026-09-07.json) |
| `avg \|bias\|` `V2-E-045` (kedua korpus × uji/val); koreksi `detection`/`det+class` val 953 | Inferensi GPU baru tiga detektor bank `combined1716` pada partisi *train* kedua korpus, model *count* Ridge di-fit via `scripts/evaluate_remote_count_reconciled.py` (medan `confusion_matrix` baru ditambahkan); proses ini menemukan dan memperbaiki bug cakupan citra pada dump val 953 sesi sebelumnya; lihat [`class_bias_v2e045_2026-09-07.json`](../results/class_bias_v2e045_2026-09-07.json) |
