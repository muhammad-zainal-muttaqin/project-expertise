# Map Boost — Lembar Bukti Eksperimen

- **Tanggal pencatatan:** 2026-08-28 (eksekusi val: 2026-08-27/28; eksekusi test-locked: 2026-08-28)
- **Skrip:** `scripts/rank_and_emit.py` (fusi + fitur + pelatihan *re-ranker* + grid VAL, disalin dari `/workspace/map_boost/rank_and_emit.py`) dan `scripts/run_test_locked.py` (*driver* tahap test-locked yang mengimpor ulang fungsi `rank_and_emit.py`, tidak menyalin logikanya; berkas asli berada di luar repo karena `/workspace/map_boost` bersifat *read-only* bagi sesi ini)
- **Artefak numerik:** `results/remote_eval_2026-08-28/map_boost_artifacts/{953,depth}/results_val.json`, `results/remote_eval_2026-08-28/map_boost_artifacts/{953,depth}/results_test_locked.json`
- **Status:** catatan ini berada di *staging mirror* (`/workspace/repo_staging/project-expertise/`) dan belum mendapat ID eksperimen resmi (`V2-E-###`/`PT-E-###`); penomoran dan integrasi ke `experiments/` serta `docs/LAPORAN-AKHIR.md` menjadi keputusan pemelihara repo yang sebenarnya.

---

## Rancangan Eksperimen

**Tiga lapisan pemrosesan pascadeteksi.** Pipeline ini menaikkan mAP COCO dari keluaran mentah tiga detektor (`yolo26l`, `rtdetr_l`, `rfdetr_l`) pada dua korpus (953 dan *depth*) lewat tiga lapisan yang dijalankan berurutan, seluruhnya dipilih di VAL dan dikunci sebelum test dibuka:

1. **Fusi *deep-tail* WBF (*weighted box fusion*).** `base.fuse_corpus` menggabungkan deteksi ketiga model per citra memakai `iou_threshold=0,60`, dievaluasi pada tiga ambang skor minimum (*floor*) $\in \{0,05;\ 0,02;\ 0,01\}$ — ambang yang lebih longgar menyertakan ekor distribusi skor rendah (*deep-tail*) yang biasanya dibuang, dengan asumsi lapisan 2 dapat memilah proposal benar (*true positive*, TP) dari populasi yang lebih besar itu. Hasil fusi disimpan per `{dataset, split, floor}` di `/workspace/map_boost/cache/vote_*.joblib` sehingga jalur VAL dan *test* berbagi kode fusi yang identik-bit.
2. **Pengklasifikasi ulang TP terlatih (*TP re-ranker*).** `HistGradientBoostingClassifier` (`max_iter=300`, `learning_rate=0,05`, `max_leaf_nodes=31`, `l2_regularization=10,0`, `random_state=42`) dilatih **hanya pada TRAIN**, per dataset dan per *floor*, dengan label dari pencocokan serakah IoU ≥ 0,5 terhadap nilai acuan kebenaran (*ground truth*). Fitur berjumlah 30 dimensi untuk dataset 953 dan 37 dimensi untuk dataset *depth* (30 fitur dasar + 7 fitur *depth*), mencakup lima kelompok: (a) kepercayaan detektor dan distribusi kelas lunak (*skor, entropi, margin, $p_1$..$p_4$*); (b) geometri kotak (*posisi, ukuran, rasio aspek*); (c) konteks citra (*jumlah kotak per citra, peringkat skor ternormalisasi*); (d) populasi sisi (*z-score* posisi/luas relatif terhadap median populasi kotak pada sisi kamera yang sama); (e) bukti tautan lintas sisi dari model *edge-linker* (`ExtraTreesClassifier` terlatih terpisah — `detection_edge_linker_953_v2/extra.joblib` dan `detection_edge_linker_depth_v1/extra.joblib` — dipakai murni sebagai penghasil fitur `max_prev`/`max_next`/`mean_two_max`/`count>0,3`/`count>0,5`/`sum_top2_logit`, bukan sebagai pengambil keputusan langsung); khusus dataset *depth* ditambah statistik *patch* kedalaman per kotak (median, rata-rata, simpangan baku, fraksi valid, median citra, selisih kotak−citra).
3. **Skor dan emisi (dipilih di VAL).** Skor agnostik $= \text{skor}_{\text{WBF}}^{a} \times p_{tp}^{b}$; skor sadar-kelas $= \text{skor}_{\text{WBF}}^{a} \times p_{tp}^{b} \times p_c^{\gamma}$, satu baris per kelas dengan $p_c \geq 0,01$, dibatasi maksimum 100 baris per citra (`score_agnostic`/`score_classaware`). Grid VAL: kombinasi $(a,b) \in \{(1;0);\ (0;1);\ (0{,}5;1);\ (1;1);\ (1;0{,}5)\}$ pada ketiga *floor* untuk agnostik, ditambah $\gamma \in \{0{,}5;\ 0{,}75;\ 1{,}0\}$ untuk sadar-kelas.

**Gerbang anchor (*anchor gate*), wajib lulus sebelum grid VAL berjalan.** Empat dump WBF-agnostik yang sudah ada (`fused_combined1716_val`) dievaluasi ulang lewat `base.coco_metrics` dan dibandingkan dengan angka acuan (toleransi ±0,004); kegagalan menghentikan seluruh sesi tanpa menjalankan grid.

**Disiplin *single-look* pada data test.** `run_test_locked.py` menolak (`SystemExit`) berjalan apabila `artifacts/{953,depth}/results_test_locked.json` sudah ada, mengevaluasi **hanya** empat profil yang sudah dikunci di VAL (bukan grid ulang), dan memuat *ranker* yang sudah terlatih (`ranker_floor*.joblib`) tanpa penyesuaian (*fit*) apa pun — labelnya `raise FileNotFoundError` apabila berkas *ranker* terkunci tidak ada, bukan melatih ulang secara diam-diam. Fusi dan fitur test memakai fungsi yang sama persis dari `rank_and_emit.py` (`build_or_load_vote`, `build_features`, `p_tp_map_from`, `score_agnostic`, `score_classaware`) lewat impor, bukan salinan kode.

---

## Temuan Empiris Terukur

### Gerbang anchor — VAL, seluruh 4 profil eksak (toleransi ±0,004)

| Dataset | Jenis | Acuan | Aktual | Status |
|---|---|---|---|---|
| 953 | agnostik | 0,8312 | 0,8312 | LULUS |
| *depth* | agnostik | 0,8648 | 0,8648 | LULUS |
| 953 | sadar-kelas | 0,5689 | 0,5689 | LULUS |
| *depth* | sadar-kelas | 0,6595 | 0,6595 | LULUS |

### Ringkasan grid VAL

**AUC *re-ranker* per *floor*** (`train_auc`/`val_auc`, dari `results_val.json`):

| Dataset | *Floor* | `train_auc` | `val_auc` | Kotak train | Kotak val |
|---|---|---|---|---|---|
| 953 | 0,05 | 0,9860 | 0,9544 | 74.063 | 10.228 |
| 953 | 0,02 | 0,9911 | 0,9761 | 178.931 | 24.520 |
| 953 | 0,01 | 0,9952 | 0,9867 | 447.106 | 59.768 |
| *depth* | 0,05 | 0,9937 | 0,9706 | 23.684 | 5.142 |
| *depth* | 0,02 | 0,9963 | 0,9869 | 82.941 | 17.691 |
| *depth* | 0,01 | 0,9977 | 0,9900 | 222.315 | 48.017 |

*Catatan audit terkait AUC ini ada di bagian Batasan Validitas & Audit (poin 2) — AUC tinggi mengukur diskriminasi peringkat, bukan kalibrasi.*

**Profil terbaik-VAL per dataset** (hasil pemilihan grid, sebelum dikunci):

| Dataset | Jenis | *Floor* | $a$ | $b$ | $\gamma$ | Metrik VAL |
|---|---|---|---|---|---|---|
| 953 | agnostik | 0,02 | 0 | 1 | — | AP50 = 0,8403 |
| 953 | sadar-kelas | 0,01 | 0 | 1 | 1,0 | mAP50 = 0,5866 |
| *depth* | agnostik | 0,05 | 1 | 0,5 | — | AP50 = 0,8663 |
| *depth* | sadar-kelas | 0,02 | 1 | 0 | 1,0 | mAP50 = 0,6623 |

AP50 per kelas VAL (profil terpilih): 953 — B1=0,7776; B2=0,4766; B3=0,6271; B4=0,4651. *Depth* — B1=0,8267; B2=0,6967; B3=0,7362; B4=0,3896.

### Hasil TEST-LOCKED vs. baseline test yang dirujuk orkestrator

| Dataset | Jenis | Profil terkunci | AP50/mAP50 test | Baseline test | Δ |
|---|---|---|---|---|---|
| 953 | agnostik | *floor*=0,02; $a$=0; $b$=1 | 0,8419 | 0,8350 | +0,0069 |
| 953 | sadar-kelas | *floor*=0,01; $a$=0; $b$=1; $\gamma$=1,0 | 0,5970 | 0,5861 | +0,0109 |
| *depth* | agnostik | *floor*=0,05; $a$=1; $b$=0,5 | 0,8783 | 0,8764 | +0,0019 |
| *depth* | sadar-kelas | *floor*=0,02; $a$=1; $b$=0; $\gamma$=1,0 | 0,6552 | 0,6691 | **−0,0139** |

**AP50 per kelas — TEST, 953** (profil sadar-kelas terkunci): B1=0,8042; B2=0,4942; B3=0,6566; B4=0,4328.
**AP50 per kelas — TEST, *depth*** (profil sadar-kelas terkunci): B1=0,8288; B2=0,6863; B3=0,7362; B4=0,3693.

**Cakupan citra dan populasi kotak terfusi (test):**

| Dataset | $n_{pohon}$ | $n_{citra}$ | *Floor* agnostik → kotak terfusi | *Floor* sadar-kelas → kotak terfusi |
|---|---|---|---|---|
| 953 | 141 | 588 | 0,02 → 36.427 | 0,01 → 86.768 |
| *depth* | 110 | 440 | 0,05 → 4.965 | 0,02 → 17.229 |

*Missing-records* (citra bermetadata tanpa deteksi mentah dari model manapun) = 0 pada kedua dataset; *extra-predictions* (deteksi mentah tanpa metadata terkait) = 0 pada kedua dataset; *missing-depth* (stem test tanpa berkas `.raw`/`.json` kedalaman valid) = 0/440 pada dataset *depth*. Waktu total eksekusi test-locked (fusi + fitur + skor + evaluasi COCO, kedua dataset, empat profil): 511,2 detik.

**Selang kepercayaan *bootstrap*:** tersedia pada
[`validation_wave/ci_artifacts/CI_SUMMARY.md`](validation_wave/ci_artifacts/CI_SUMMARY.md).
CI tersebut memakai 500 resampling berpasangan pada tingkat citra, dengan
sanity check yang mereproduksi keempat angka test-locked sebelum resampling.

---

## Keputusan Metodologis

1. **Empat profil dikunci dari VAL, tanpa pengecualian.** Untuk tiap dataset × jenis emisi, profil dengan AP50/mAP50 VAL tertinggi pada grid (lima kombinasi $(a,b)$ × tiga *floor* untuk agnostik; ditambah tiga $\gamma$ untuk sadar-kelas) dikunci sebagai satu-satunya kandidat yang boleh dievaluasi pada test — sesuai hasil `selected_agnostic_profile`/`selected_classaware_profile` pada `results_val.json`. Tidak ada pemilihan ulang berdasarkan hasil test.
2. **Data test dibuka tepat satu kali** pada sesi ini (2026-08-28), mencakup keempat profil sekaligus dalam satu eksekusi `run_test_locked.py`. Disiplin ini ditegakkan secara teknis: skrip menolak (`SystemExit`) berjalan apabila `results_test_locked.json` untuk `{dataset}` sudah ada di `artifacts/`, dan memuat *ranker* terkunci lewat `joblib.load` tanpa jalur pelatihan ulang (kegagalan memuat memicu `FileNotFoundError`, bukan *fallback* ke pelatihan diam-diam). Guard ini diverifikasi kosong (berkas belum ada) sebelum eksekusi dimulai.
3. **Hasil dilaporkan apa adanya, termasuk regresi.** Tiga dari empat metrik menunjukkan peningkatan performa kecil dan konsisten arah dengan VAL (953 agnostik +0,0069; 953 sadar-kelas +0,0109; *depth* agnostik +0,0019). **Profil sadar-kelas *depth* justru mengalami penurunan performa pada test (Δ=−0,0139, dari baseline 0,6691 menjadi 0,6552)**, berlawanan arah dengan keunggulan yang terlihat di VAL (0,6623 vs. baseline VAL 0,6595, yaitu +0,0028). Bootstrap test image-level mengonfirmasi bahwa CI delta 953 agnostik `[+0,0027; +0,0135]` dan 953 sadar-kelas `[+0,0030; +0,0194]` tidak mencakup nol; CI *depth* agnostik `[-0,0016; +0,0045]` dan sadar-kelas `[-0,0284; +0,0016]` melintasi nol. Keputusan tetap **mempertahankan profil sadar-kelas *depth* apa adanya** dan mencatat regresinya sebagai temuan generalisasi negatif, tanpa memilih ulang berdasarkan angka test.
4. **CI test-locked dihitung setelah test dibuka, tanpa seleksi ulang.** Artefak
`ci_artifacts/ci_test.json` memakai 500 resampling citra berpasangan,
`RandomState(42)`, dan identity-resample sanity check yang mereproduksi
keempat point estimate. Ini mengukur ketidakpastian sampling atas hasil yang
sudah terkunci; bukan izin untuk mengulang tuning test.

---

## Batasan Validitas & Audit

1. **Data *depth* mentah belum direproyeksi ke bidang RGB.** Berkas yang sebenarnya tersedia di lingkungan ini adalah dump sensor mentah (`{stem}.raw`, *uint16LE* dalam mm, kisi kamera kedalaman asli 848×480) beserta `{stem}.json`, **bukan** peta kedalaman *uint8* yang telah diproyeksikan ke bidang RGB 1280×800 seperti dideskripsikan pada spesifikasi awal. Metadata JSON kedalaman sendiri menyatakan (`alignmentNote`) bahwa *buffer* tersebut belum direproyeksi kedalaman-ke-warna; pengubahan ukuran naif (*nearest-neighbor resize*) ke bidang RGB meleset median 29 piksel dari posisi sejati pada bidang tersebut. Karena fitur kedalaman di sini adalah fitur pendukung untuk *re-ranker* yang dipelajari (bukan pengukuran fisik langsung), pengubahan ukuran *nearest-neighbor* dipakai sebagai substitusi deterministik yang diterapkan **identik** pada TRAIN, VAL, dan TEST (lihat komentar `rank_and_emit.py` baris 246–259) — sehingga tidak menimbulkan kebocoran informasi antar-*split*, tetapi nilai kedalaman absolut pada level kotak tetap harus dibaca sebagai perkiraan kasar, bukan koordinat piksel yang presisi.
2. **Skor *re-ranker* ($p_{tp}$) tidak dikalibrasi.** `HistGradientBoostingClassifier.predict_proba` menghasilkan skor yang terbukti mendiskriminasi TP/*false positive* dengan baik (AUC VAL 0,9544–0,9900 di seluruh *floor* dan dataset — lihat tabel AUC), tetapi skor ini tidak diverifikasi terkalibrasi (mis. lewat *reliability diagram* atau *Brier score*) sebagai estimasi probabilitas TP yang sebenarnya. Skor tersebut sah dipakai sebagaimana adanya di sini — sebagai **peringkat relatif** dalam rumus skor gabungan $\text{skor}_{\text{WBF}}^{a} \times p_{tp}^{b}$ (dan turunannya untuk sadar-kelas) — tetapi **tidak sah** ditafsirkan sebagai probabilitas kalibrasi absolut (mis. "kotak ini 92% kemungkinan benar").
3. **Interpretasi CI.** CI image-level di atas tidak menggantikan paired
bootstrap tree-level untuk metrik end-to-end. Untuk E2E, artefak terpisah
`validation_wave/ci_artifacts/e2e_paired_test.json` memakai 5.000 resampling
pohon dari ringkasan per-pohon yang sudah terkunci dan tidak melakukan
inferensi atau seleksi. Keduanya harus dibaca sebagai ketidakpastian sampling,
bukan sebagai bukti bahwa target 75%/80%/95% tercapai.
4. **Data test kedua dataset kini sudah terbuka** (dibuka tepat satu kali, "pembukaan pertama", pada 2026-08-28 melalui sesi ini, mencakup keempat profil terkunci sekaligus). **Setiap eksekusi ulang `run_test_locked.py` berikutnya terhadap dataset yang sama wajib diberi label eksplisit sebagai "pembukaan kedua" (atau seterusnya)** di catatan eksperimen, karena pembukaan berulang tanpa pelabelan melanggar protokol *single-look* pada data test. Guard teknis pada skrip (`SystemExit` bila `results_test_locked.json` sudah ada di `artifacts/{dataset}/`) mencegah penimpaan tidak sengaja pada lokasi keluaran yang sama, tetapi menjalankan dengan `ARTIFACTS`/lokasi keluaran yang berbeda akan melewati guard ini sepenuhnya — disiplin pelabelan pembukaan berulang tetap menjadi tanggung jawab manual, bukan tanggung jawab skrip.
5. **Cakupan grid VAL tidak identik antara agnostik dan sadar-kelas.** Grid agnostik hanya menjelajahi $(a,b)$ pada tiga *floor* (15 baris per dataset); grid sadar-kelas menambah dimensi $\gamma$ (45 baris per dataset). Kombinasi *floor* × $(a,b)$ × $\gamma$ yang lebih kasar dari itu (mis. $\gamma$ di luar $\{0{,}5;\ 0{,}75;\ 1{,}0\}$, atau ambang $p_c$ selain 0,01) tidak dievaluasi dan berada di luar cakupan sesi ini.
