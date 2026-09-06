# Matriks bukti dan kesenjangan audit

Status: audit berlangsung; daftar ini tidak menyatakan seluruh repositori
telah diperiksa atau dampak performa perbaikan telah diukur.

| ID | Temuan | Bukti lokal / primer | Status dan pertanyaan berikutnya |
|---|---|---|---|
| A01 | UF historis menyimpan indeks proposal sebagai sisi | `scripts/sweep_remote_pipeline.py:119`, commit `e6bddc9` | SUDAH DIPERBAIKI pengguna. Probe baru memverifikasi kendala. Angka pelanggaran AF-E-010 memakai graf geometri pengganti tanpa Hungarian; bukan prevalensi pada profil terkunci. |
| A02 | ExtraTrees menerima `class_weight=balanced` sekaligus bobot sampel positif berdasarkan rasio kelas | `scripts/train_detection_edge_linker.py`, blok `candidates` dan `sweep_model`; dokumentasi resmi scikit-learn | Kedua bobot dikalikan. Rasio efektif dapat menjadi kuadrat rasio kelas, dengan batas bobot sampel 30. Pengaruh terhadap skor dan generalisasi perlu ablasi; jangan mengklaim seluruh regresi disebabkan ini. |
| A03 | Probabilitas WBF berasal dari kelas pemenang detektor | `scripts/eval_new763_pycoco.py:predict_*`; `eval_remote_pipeline_postprocess.py:fuse_groups` | Terkonfirmasi. Dump enam kolom tidak menyimpan seluruh probabilitas model. WBF menghitung distribusi suara, bukan memulihkan probabilitas yang sudah dibuang. |
| A04 | Pencocokan fisik memakai IoU maksimum dari satu anggota klaster saja | `eval_remote_pipeline_postprocess.py:count_iou`; `evaluate_remote_count_reconciled.py:tree_matches` | Definisi metrik sah sebagai keberadaan objek, tetapi tidak menguji kemurnian identitas semua anggota. Uji klaster tercampur yang masih memperoleh F1 fisik tinggi. |
| A05 | Hungarian diikuti ambang tidak memaksimalkan jumlah pasangan yang lolos ambang | `sweep_remote_pipeline.py:evaluate_tree`; `train_detection_edge_linker.py:gt_labels` | Terkonfirmasi dengan kotak nyata secara geometris: evaluator memilih 1 pasangan, padahal 2 dapat memenuhi IoU ≥0,5. Ini keputusan definisi pencocokan; perubahan harus diberi versi. |
| A06 | Regresor jumlah menjadi batas keras yang hanya membuang klaster | `evaluate_remote_count_reconciled.py:selected_clusters` | Terkonfirmasi; tidak dapat memulihkan tandan tanpa proposal. Uji galat sebelum/sesudah pembatasan dan ketergantungan pada kepadatan/domain. |
| A07 | Fitur pencacahan menyambung statistik sisi menurut indeks absolut | `evaluate_remote_count_reconciled.py:feature_vector` | Terkonfirmasi; tidak menjamin invariansi terhadap perubahan sisi awal pengambilan foto. Perlu uji pergeseran siklik, dengan urutan rotasi tetap. |
| A08 | Dua fitur `rank_cx` diulang untuk masing-masing anggota pasangan | `train_detection_edge_linker.py:pair_features` | Terkonfirmasi; dugaan salah ketik. Jangan mengubah skema fitur bobot lama tanpa versi dan pelatihan ulang. |
| A09 | Penaut hanya memakai geometri, statistik skor, dan suara kelas; tidak memakai deskriptor visual identitas objek | `train_detection_edge_linker.py:pair_features` | Terkonfirmasi untuk jalur remote ini; eksperimen re-ID terdahulu ada di subproyek dan masih harus diperiksa sebelum mengusulkan ulang. |
| A10 | CSV split memiliki prioritas dalam builder gabungan, sehingga test sumber dapat dipindah ke train gabungan | `build_combined_rgb_dataset.py:group_split`; manifest belum tersedia penuh | Bukan kebocoran internal gabungan secara otomatis. Evaluasi bobot gabungan pada test sumber perlu audit irisan khusus. |
| A11 | Evaluator menganggap label hilang sebagai citra tanpa objek, melewati label malformed, dan melewati kunci dump tidak dikenal | `eval_new763_pycoco.py:build_gt`; `eval_agnostic_from_npz.py` | Terkonfirmasi dari kode; integritas harus diverifikasi sebelum metrik. |
| A12 | Evaluator COCO memanggil `loadRes([])` pada prediksi kosong | `eval_new763_pycoco.py:evaluate` dan evaluator lain | Reproduksi perilaku pustaka pada lingkungan audit. |
| A13 | Head dan proposal disejajarkan hanya dengan panjang baris | `evaluate_remote_class_head.py:make_detections` | Terkonfirmasi; urutan berbeda dengan panjang sama dapat memasangkan probabilitas ke kotak yang salah. Uji permutasi baris. |
| A14 | Skrip kalibrasi membaca label test secara otomatis sebelum fitting walau fitting sendiri memakai train | `fit_fused_probability_calibrator.py:main` | Paparan test yang tidak diperlukan, bukan bukti test digunakan dalam optimisasi. Pisahkan jalur fit/apply/evaluate. |
| A15 | Jumlah tepat sering merupakan pembatalan galat FP/FN | `results/audit_2026-09-06/count_error_cancellation.json` | Artefak historis 953: 24/37 pohon exact; Depth: 17/49. Ini memakai definisi pencocokan fisik lama; bukan evaluasi identitas ketat. |
| A16 | Klaim ketiadaan CI akurasi kelas di analisis mendalam bertentangan dengan artefak asli | `docs/ANALISIS_PIPELINE_MENDALAM.md`; `gsp_artifacts/*/results_test_locked.json:bootstrap_ci_95` | CI absolut tersedia untuk kedua domain; CI delta berpasangan dengan baseline adalah pertanyaan berbeda. Koreksi narasi baru dan sumber prioritas. |
| A17 | Skrip GSP dan map boost yang disebut laporan tidak ditemukan di inventaris scripts | `GSP_LINKER.md`, `MAP_BOOST.md`, inventaris lokal | Kesenjangan reproduksi, perlu cek semua referensi/import dan kemungkinan sumber eksternal setelah clone. |
| A18 | Kegagalan resep tertentu ditulis sebagai larangan umum mengganti detektor atau melatih model lebih besar | `pipeline-pertandan/CLAUDE.md`; `HANDOFF.md` | Kesimpulan terlalu luas: PT-E-011 tidak membuktikan detektor optimal pada semua domain. Gunakan dekomposisi galat dan kontrol anggaran. |

## Sumber primer yang sudah ditemukan

- Bolya dkk., ECCV 2020, [TIDE](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/849_ECCV_2020_paper.php): dekomposisi enam jenis kesalahan deteksi dari dump prediksi. Mendukung rancangan diagnosis, bukan menjamin kenaikan mAP.
- Koh dkk., ICML 2021, [WILDS](https://proceedings.mlr.press/v139/koh21a.html): pergeseran domain dan subpopulasi memerlukan evaluasi tersendiri.
- Gulrajani dan Lopez-Paz, [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434): kriteria pemilihan model bagian penting protokol generalisasi; perlu membaca naskah untuk membatasi klaim.
- scikit-learn, [ExtraTreesClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html): `class_weight` dikalikan `sample_weight`.
- Guo dkk., ICML 2017, [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html): confidence perlu diuji kalibrasinya; kalibrasi sendiri tidak memulihkan objek atau menjamin generalisasi.
- Matos dkk., CVPR Workshop 2024, [Tracking and Counting Apples](https://openaccess.thecvf.com/content/CVPR2024W/Vision4Ag/html/Matos_Tracking_and_Counting_Apples_in_Orchards_Under_Intermittent_Occlusions_and_CVPRW_2024_paper.html): kandidat jalur geometri 3D; memerlukan pose, intrinsics, depth, dan overlap. Empat foto tanpa pose belum tentu memenuhi syarat.

Seluruh sumber web diakses 6 September 2026. Pencarian gelombang awal:
dekomposisi galat deteksi, generalisasi domain, pencacahan buah multi-tampak;
gelombang tindak lanjut: pembobotan ganda dan kalibrasi probabilitas.

## Temuan lanjutan dan pembaruan artefak

| ID | Temuan | Bukti | Status / batas |
|---|---|---|---|
| A19 | WBF dengan bobot eksplisit sama menghasilkan skor berbeda dari bobot implisit | `implementation_probes.json:confidence.wbf` | Terbukti: 0,90 vs 1,35 bila satu detektor menyumbang dua kotak; belum mengukur frekuensi pada dump historis. |
| A20 | Sampel re-ID tidak didominasi jumlah negatif dari pohon sama sebagaimana komentar | `reid_pertandan.py` | Pada 16 pohon ×20 crop, hanya 18 dari 318 negatif berasal dari pohon sama. Tidak menyimpulkan dominasi gradien tanpa pengukuran. |
| A21 | Padding masuk ke BatchNorm sebelum mask C3 ResNet | `implementation_probes.json:c3_padding` | Terbukti pada bobot acak tanpa pelatihan; selisih logit 0,8989 saat train, sekitar 0,00000036 saat eval. BN blok yang dibekukan tetap berubah. Tidak berlaku umum pada Set Transformer fitur-cache. |
| A22 | Makna B1/B4 bertentangan antar-dokumen | `docs/DATASET.md`, `docs/LAPORAN-AKHIR.md` | Inkonsistensi narasi; belum ada bukti label numerik tertukar pada seluruh training. |
| A23 | Mode head_conf tidak memakai head_p anggota klaster | `evaluate_remote_count_reconciled.py:181`, probe | Terbukti karena cluster tidak mempunyai head_p agregat. Head untuk keputusan kelas akhir tetap dapat digunakan; bug berada pada ranking. |
| A24 | Crop Fase 6 BGR dengan normalisasi RGB ImageNet | `build_crop_dataset.py`, `train_crop_classifier.py`, probe | Terbukti; train/infer sama-sama BGR. Tidak berlaku pada Panen berbasis PIL RGB. |
| A25 | AF-E-005 menganggap hasil satu classifier sebagai plafon dataset | `exp_ceiling.py`, `exp_ceiling.log` | 0,6569 bersyarat pada P_test classifier; bukan batas Bayes/kualitas anotasi. Simulasi akurasi 0,90 juga bersyarat pada resep galat/skor buatan. |
| A26 | AF-E-004 menganggap divisor/regresor sebagai plafon pencacahan universal | `an8_counting.py` | Kotak GT dipadatkan menjadi jumlah; informasi identitas tidak dimanfaatkan. Tidak membuktikan semua penaut tidak bisa melampaui exact 0,29. |
| A27 | B1 ±1 dipengaruhi skala target | `latest_artifacts_review.json:constant_baselines` | Konstanta dipilih TRAIN: selalu 1 memberi 131/141 =92,91%; model AF-E-008 135/141=95,74% dan MAE jauh lebih rendah. |
| A28 | Pohon tanpa tandan dikeluarkan dari Panen | `panen_pipeline.py:57`, `panen_eval.py:9`, artefak review | 132 vs135; tiga pohon kosong ditemukan, satu juga absen dari cache deteksi. Profil tetap dievaluasi ulang pada135. |
| A29 | Panen matched-only macro-F1 dibandingkan dengan end-to-end lama | `panen_pipeline.py:297`, AF-E-012, artefak review | Final matched 0,669206; E2E132 0,520113; E2E135 0,519788. Klaim kenaikan +0,0658 tidak sah. Pencocokan antar-pipeline masih berbeda. |
| A30 | Perbaikan ambang linker mengubah populasi validasi | `panen_final.py`, `panen_pipeline.py` | AP0,3609→0,5562 bukan ablasi murni karena pasangan/rasio positif yang dinilai ikut berubah. Perlu kedua model pada pasangan yang sama. |
| A31 | Struktur AF-E-009 dibentuk setelah seleksi memakai GT | `e4b_fuse.py:collect` | Rank/count/context memakai hanya proposal yang lolos IoU GT, bukan semua proposal inferensi. Hasil adalah diagnostik pada proposal cocok, bukan fusi siap inferensi penuh. |
| A32 | 39 pohon test Depth menjadi train menurut builder gabungan | `latest_artifacts_review.json:split_provenance` | Terkonfirmasi pada sumber saat ini; 5 lainnya menjadi validasi. Untuk menyatakan checkpoint lama bocor perlu manifest pelatihannya. |
| A33 | Distribusi CORN direduksi menjadi rerata ordinal | `panen_pipeline.py:131`, `corn_cached_diagnostic` | Kehilangan informasi secara matematis. Pada cache crop saat ini aturan kasar berbeda hanya 2/2.612 test, tanpa perbaikan akurasi bersih; jangan menjanjikan perbaikan besar. |
| A34 | Ridge Panen menghasilkan angka terpisah tanpa rekonsiliasi identitas | `panen_final.py:feats` dan blok prediksi | Angka B1/total dapat diperbaiki tanpa memperbaiki objek fisik; pembulatan per-target juga tidak menjamin total = B1+B2 + B3+B4. Bobot Ridge dan prediksi per-pohon final belum tersimpan pada artefak yang dibaca. |

Probe sintetis: `scripts/audit_implementation_contracts.py`.
Audit artefak dan populasi: `scripts/audit_latest_artifacts.py`.
Hasil disimpan di `results/audit_2026-09-06/`. Pengujian ini tidak mengubah
parameter model, hasil historis, maupun anotasi.

Sumber primer tambahan:

- Shi, Cao, Raschka, 2021, [CORN](https://arxiv.org/abs/2111.08851) dan [kode penulis](https://github.com/Raschka-research-group/corn-ordinal-neuralnet): probabilitas kumulatif diperoleh melalui hasil kali probabilitas terkondisi.
- PyTorch, [BatchNorm2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.batchnorm.BatchNorm2d.html), dokumentasi daring diakses 6 September 2026: statistik batch saat train berbeda dari running statistics saat eval.
- Dwork dkk., NeurIPS 2015, [Generalization in Adaptive Data Analysis and Holdout Reuse](https://proceedings.neurips.cc/paper/2015/hash/bad5f33780c42f2588878a9d07405083-Abstract.html): penggunaan ulang himpunan uji secara adaptif memerlukan protokol khusus untuk mempertahankan validitas.

Pencarian dihentikan setelah setiap keputusan review utama memiliki bukti kode,
artefak, atau batasan eksplisit. Pencarian arsitektur tambahan belum menjawab
kesalahan perbandingan metrik yang sudah terbukti, sehingga tidak diprioritaskan.
