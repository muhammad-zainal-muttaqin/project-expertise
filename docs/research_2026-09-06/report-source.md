# Review implementasi dan validitas eksperimen project-expertise

Untuk pengembang dan peneliti proyek · 6 September 2026

## Kesimpulan utama

Proyek mempunyai hambatan nyata pada deteksi, pengaitan identitas lintas-tampak, dan klasifikasi. Namun, sebagian keputusan riset juga didasarkan pada perbandingan metrik yang tidak setara atau pada hasil satu resep yang ditafsirkan sebagai batas kemampuan dataset. Bukti yang diperiksa belum membuktikan bahwa target mAP50 empat kelas 0,85 mustahil; bukti itu juga belum menunjukkan cara yang sudah tervalidasi untuk mencapainya.

Temuan paling mendesak adalah klaim peningkatan makro-F1 Pipeline Panen. Nilai 0,6692 hanya menghitung tandan yang berhasil dicocokkan. Pembanding 0,6034 memasukkan prediksi berlebih dan tandan yang terlewat. Dengan model, ambang, dan pencocokan Panen final tetap, makro-F1 ujung ke ujung sebenarnya 0,5201 pada 132 pohon, atau 0,5198 pada seluruh 135 pohon empat sisi. Karena itu, klaim peningkatan +0,0658 tidak dapat dipakai untuk memilih pengklasifikasi atau menggabungkan pipeline. [Evaluator Panen](../../scripts/audit_forensik/panen_pipeline.py), [reproduksi dan matriks konfusi](../../results/audit_2026-09-06/latest_artifacts_review.json).

Review ini mempertahankan sasaran awal empat kelas dan pencacahan total. Pencacahan B1 dengan toleransi ±1 merupakan sasaran tambahan yang berbeda. Riwayat eksperimen digunakan untuk menghindari pengulangan: re-ID, pengklasifikasi multiview, CORAL/CORN, Set Transformer, fusi depth, regresi jumlah, dan berbagai ansambel sudah pernah dicoba. Rekomendasi berikut berfokus pada perbaikan kontrak implementasi serta pembandingan yang dapat dipercaya.

## Apa yang sudah diverifikasi

Audit mencakup kode utama evaluator, WBF, penaut remote, rekonsiliasi jumlah, kepala kelas, pembentukan crop, pelatihan RGBD4, sebagian jalur multiview historis, dossier pada project_expertise_experiment_map, dan tambahan AF-E-001–013. Skrip Panen awal dan final dibaca seluruhnya. Evaluasi ulang menggunakan cache deteksi dan bobot penaut yang sudah tersedia; tidak ada pelatihan ulang, inferensi GPU, atau perubahan anotasi.

Perbaikan UF pada commit e6bddc9 sudah benar dan lolos pemeriksaan kendala sisi. Ini merupakan perbaikan pengguna selama jeda, bukan kontribusi baru review ini. Hasil historis dan perubahan pengguna dipertahankan. Lingkungan diagnostik memakai torch 2.8.0, torchvision 0.23.0, numpy 2.1.2, serta lingkungan audit terpisah; pengujian ini bukan reproduksi seluruh lingkungan pelatihan historis. [Probe sintetis](../../scripts/audit_implementation_contracts.py), [audit cache](../../scripts/audit_latest_artifacts.py).

## 1. Hasil terbaru perlu dibandingkan pada definisi yang sama

### Makro-F1 kondisional dan ujung ke ujung

Panen menambahkan baris klasifikasi hanya untuk pasangan klaster–tandan yang cocok. F1 dari baris tersebut mengabaikan seluruh objek terlewat dan klaster tanpa pasangan. Dalam review, pencatatan matriks konfusi ditambahkan ke fungsi evaluator melalui instrumentasi AST, tanpa mengubah keputusan deteksi, asosiasi, atau klasifikasi. Nilai F1 fisik, akurasi kelas, dan makro-F1 kondisional hasil asli berhasil direproduksi.

| Besaran | Panen final, 132 pohon | Panen final, 135 pohon |
|---|---:|---:|
| F1 fisik | 0,7619 | 0,7612 |
| Akurasi kelas pada pasangan cocok | 0,7161 | 0,7161 |
| Makro-F1 pada pasangan cocok | 0,6692 | 0,6692 |
| Makro-F1 ujung ke ujung | 0,5201 | 0,5198 |
| MAE jumlah klaster mentah | 2,3712 | 2,3333 |
| Jumlah klaster mentah, toleransi ±1 | 0,4015 | 0,4074 |

Ridge final dilaporkan menghasilkan MAE total 1,4015 dan toleransi ±1 sebesar 0,5682 pada 132 pohon. Angka Ridge tersebut dibaca dari artefak, belum direproduksi karena model Ridge serta prediksi per-pohon final tidak disimpan bersama hasil yang diperiksa. Metrik Ridge tidak boleh disamakan dengan kualitas himpunan tandan terdeteksi. [Hasil Panen final](../../results/audit_forensik_2026-09-06/panen/panen_final.json), [hasil diagnostik](../../results/audit_2026-09-06/latest_artifacts_review.json).

Pembanding lama pada 953 memakai Hungarian+UF dan mencapai makro-F1 ujung ke ujung 0,6034. GSP Global Set-Partition dipilih pada domain Depth; nama direktori gsp_artifacts tidak berarti setiap domain memakai solver GSP. Metode pencocokan Panen juga serakah menurut confidence, sedangkan evaluator lama memakai Hungarian. Penyamaan makro-F1 memperbaiki satu kesalahan besar, tetapi pembandingan akhir tetap memerlukan evaluator yang sama. [Artefak terkunci 953](../../results/remote_eval_2026-08-28/gsp_artifacts/953/results_test_locked.json).

### Tiga pohon kosong hilang dari evaluasi

EMPAT_SISI dibentuk dengan syarat bahwa pohon mempunyai tandan berlabel, lalu daftar test diambil dari kunci cache deteksi. Akibatnya, DAMIMAS_A21B_0230, DAMIMAS_A21B_0421, dan DAMIMAS_A21B_0793 dikeluarkan. Ketiganya mempunyai empat citra dan nol tandan acuan; pohon terakhir juga tidak mempunyai entri cache deteksi. Daftar evaluasi seharusnya berasal dari manifest dan jumlah citra, dengan prediksi kosong tetap disertakan. Pada data ini penghilangan tersebut tidak selalu menaikkan metrik; masalah utamanya ialah perubahan populasi tanpa penjelasan. [Pemilihan populasi](../../scripts/audit_forensik/panen_eval.py), [identitas pohon dan hasil ulang](../../results/audit_2026-09-06/latest_artifacts_review.json).

## 2. Klaim plafon dan keberhasilan 95% terlalu luas

### Angka 0,6569 bukan batas kemampuan dataset

AF-E-005 menempatkan probabilitas satu ConvNeXt pada kotak acuan. Angka 0,6569 adalah performa pengklasifikasi tersebut dengan lokalisasi yang dibuat sempurna dalam evaluator identitas crop. Tidak ada pengukuran ketidakpastian label antaranotator atau batas galat Bayes di dalam eksperimen itu. Skrip yang sama menghasilkan 1,0000 bila kelas juga dibuat sempurna. Karena itu, rasio 0,5970/0,6569 tidak membuktikan bahwa detektor hanya menyisakan enam poin perbaikan. [Kode eksperimen](../../scripts/audit_forensik/exp_ceiling.py), [log asli](../../logs_ringkas/audit_forensik_2026-09-06/exp_ceiling.log).

Simulasi yang menyatakan akurasi sekitar 0,90 diperlukan untuk mAP50 sebesar 0,85 memakai pola kesalahan tetangga dan distribusi confidence buatan. Kesimpulan tersebut berlaku untuk simulasi itu; AP tidak ditentukan oleh akurasi saja. Demikian pula exact-count 0,29 berasal dari jumlah kotak GT yang dibagi faktor duplikasi, bukan dari penaut identitas sempurna. Kegagalan estimator yang sudah membuang informasi identitas tidak membatasi semua metode asosiasi. [Simulasi sensitivitas](../../scripts/audit_forensik/exp_sensitivity.py), [baseline pencacahan](../../scripts/audit_forensik/an8_counting.py).

### B1 jarang, sehingga toleransi ±1 relatif longgar

Pada 141 pohon test AF-E-008, jumlah B1 adalah nol pada 71 pohon, satu pada 33, dan dua pada 27. Konstanta satu dipilih dari TRAIN untuk memaksimalkan toleransi ±1, lalu dievaluasi pada TEST. Hasilnya 131/141 atau 92,91%, tanpa gambar. Model AF-E-008 mencapai 135/141 atau 95,74%. Model tetap memberi manfaat: MAE-nya 0,3688 dibandingkan 0,8582 untuk konstanta. Jadi hasilnya tidak diabaikan, tetapi toleransi ±1 perlu disertai MAE, exact-count, dan pemisahan pohon kosong/padat. [Baseline konstan](../../results/audit_2026-09-06/latest_artifacts_review.json), [AF-E-008](../../results/audit_forensik_2026-09-06/e345.json).

Panen final melaporkan B1 ±1 sebesar 96,97% pada 132 pohon, dengan exact-count 62,88%; pada validasi B1 ±1 sebesar 87,91%. Ini belum menunjukkan konsistensi di atas 95% lintas-domain. B1+B2 dan total merupakan target berbeda; hasil baik pada B1 tidak memenuhi sasaran pencacahan total secara otomatis. [Panen final](../../results/audit_forensik_2026-09-06/panen/panen_final.json).

## 3. Cacat implementasi yang masih dapat direproduksi

| Komponen | Temuan terkonfirmasi | Implikasi yang dapat dipertanggungjawabkan |
|---|---|---|
| WBF remote | Tiga kotak dengan skor 0,9 dari dua model menghasilkan 0,90 tanpa bobot eksplisit, tetapi 1,35 dengan bobot [1; 1], bila satu model menyumbang dua kotak | Normalisasi confidence bergantung cabang kode; perbandingan bobot dan kalibrasi dapat berubah tanpa perubahan bobot relatif |
| Ranking klaster | head_conf dan joint_conf memakai fallback probabilitas dasar karena head_p tidak diagregasi pada klaster | Beberapa mode yang tampak berbeda sebenarnya tidak memakai kepala kelas untuk ranking |
| Penyelarasan kepala | Jumlah baris diperiksa, identitas/koordinat proposal tidak | Baris yang tertukar diterima dan probabilitas dapat ditempelkan pada objek lain |
| ExtraTrees penaut | class_weight balanced dikalikan sample_weight positif | Rasio kelas 9:1 menjadi bobot positif/negatif 81:1 dalam contoh; dampak pada data nyata masih perlu ablasi |
| C3 ResNet historis | Padding masuk backbone sebelum mask; BatchNorm blok beku tetap berubah saat train | Menambahkan tampak palsu mengubah fitur tampak valid; membatasi validitas kesimpulan negatif resep tersebut |
| Crop Fase 6 | BGR dinormalisasi dengan statistik RGB pralatih | Ketidakcocokan dengan pralatih; train dan inferensi sendiri konsisten BGR. Panen memakai PIL RGB dan tidak terkena temuan ini |
| Evaluator kosong | loadRes([]) menghasilkan IndexError | Prediksi kosong belum ditangani sebagai hasil evaluasi yang sah |

Semua kasus di atas mempunyai reproduksi CPU tanpa dataset atau pelatihan. Hasil lengkap beserta hash sumber tersimpan pada [implementation_probes.json](../../results/audit_2026-09-06/implementation_probes.json). Contoh ini membuktikan perilaku kode, bukan besarnya kenaikan mAP setelah perbaikan. Pembobotan ganda sesuai perilaku yang dinyatakan [dokumentasi ExtraTrees scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html); perubahan statistik BatchNorm sesuai [dokumentasi PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.batchnorm.BatchNorm2d.html).

Dua keterbatasan evaluator perlu dipisahkan dari bug biasa. Pertama, Hungarian dengan tujuan jumlah IoU maksimum lalu ambang 0,5 dapat menghasilkan satu pasangan, walaupun dua pasangan dapat memenuhi ambang. Tujuan pencocokan harus ditetapkan eksplisit. Kedua, IoU klaster memakai anggota terbaik saja. Dua klaster yang masing-masing mencampur identitas A dan B dapat memperoleh F1 fisik sempurna. Metrik ini mengukur cakupan keberadaan objek, belum kemurnian seluruh anggotanya. [Kasus sintetis geometris dan identitas](../../results/audit_2026-09-06/implementation_probes.json).

Pada hasil terkunci lama, 24 dari 37 pohon dengan jumlah tepat di 953 masih mempunyai FP dan FN yang saling membatalkan; di Depth terdapat 17 dari 49. Bahkan pencocokan tersebut memakai definisi fisik yang longgar. Angka jumlah benar tidak cukup untuk menyatakan objek yang dihitung sudah benar. [Audit pembatalan galat](../../results/audit_2026-09-06/count_error_cancellation.json).

## 4. Tantangan generalisasi terbesar: supervisi dan antarmodul

### Identitas pohon dan kampanye harus ditelusuri per checkpoint

Builder gabungan memakai split 953 sebagai prioritas untuk 352 ID pohon yang sama. Pada sumber saat ini, 39 pohon test Depth berpindah ke train gabungan dan lima ke validasi gabungan. Kebijakan ini mencegah satu pohon terpecah di dalam dataset gabungan; masalah muncul jika checkpoint gabungan kemudian diuji pada test sumber Depth seolah-olah seluruhnya belum pernah dilihat. Eksposur checkpoint historis belum diputuskan tanpa manifest pelatihannya. [Builder gabungan](../../scripts/build_combined_rgb_dataset.py), [daftar 39 identitas](../../results/audit_2026-09-06/latest_artifacts_review.json).

Perubahan jumlah label dan ukuran kotak lintas-kampanye pada AF-E-001/007 merupakan bukti pergeseran distribusi. Namun, posisi pusat prediksi di dalam kotak GT tidak dengan sendirinya membuktikan perbedaan pedoman anotasi sebagai satu-satunya penyebab. Audit e1c juga menandai setiap prediksi dengan IoU terbaik ≥0,5 sebagai TP tanpa penugasan satu-ke-satu, sehingga duplikat dapat hilang dari kategori FP. Ketidaklengkapan anotasi, perubahan biologis, framing, dan galat model masih perlu dipisahkan melalui anotasi ulang sampel berstrata. [Kode kategorisasi FP](../../scripts/audit_forensik/e1c_fpkind.py).

### Fitur downstream belum selalu menyerupai kondisi pemakaian

Pada AF-E-009, fitur struktur dibuat setelah proposal disaring memakai kecocokan terhadap GT. Jumlah objek dan peringkat posisi kemudian dihitung dari proposal yang lolos saja. Seleksi demikian tidak tersedia saat pemakaian tanpa anotasi. Hasil +0,0058 makro-F1 tetap merupakan diagnostik pada deteksi cocok, tetapi belum membuktikan manfaat jalur inferensi penuh. [Fusi struktur](../../scripts/audit_forensik/e4b_fuse.py).

Pelatihan ulang penaut Panen final pada conf ≥0,30 memperbaiki kecocokan ambang. Akan tetapi AP validasi 0,3609→0,5562 membandingkan himpunan pasangan yang ikut berubah, bukan hanya dua model. Kedua model perlu dinilai pada pasangan yang sama sebelum selisih itu diatribusikan pada pelatihan. Detektor dan kepala fitur TRAIN juga menghasilkan prediksi pada data yang dipakai melatihnya; mengurangi ambang belum menghilangkan seluruh perbedaan kualitas prediksi TRAIN dan pemakaian. [Panen awal](../../scripts/audit_forensik/panen_pipeline.py), [Panen final](../../scripts/audit_forensik/panen_final.py).

CORN sendiri membentuk probabilitas ordinal secara benar melalui hasil kali probabilitas terkondisi. Reduksi seluruh distribusi menjadi satu skor rerata membuang informasi; skor sama dapat berasal dari distribusi berbeda. Namun, pada cache crop saat ini, penggantian aturan kasar skor 1,5 dengan probabilitas kumulatif 0,5 hanya mengubah dua dari 2.612 keputusan test dan tidak memberi kenaikan akurasi bersih. Temuan ini bukan alasan untuk mengulang pelatihan CORN atau menjanjikan lompatan besar. [CORN, Shi dkk., 2021](https://arxiv.org/abs/2111.08851), [diagnostik cache](../../results/audit_2026-09-06/latest_artifacts_review.json).

## 5. Urutan pemulihan yang tidak mengulang riset lama

1. Tetapkan satu kontrak evaluasi: manifest pohon, B1–B4, metrik gambar dan tandan terpisah, exact-count dan toleransi ±1 terpisah, serta F1 kondisional dan ujung ke ujung terpisah. Tambahkan metrik kemurnian asosiasi. Semua profil lama dinilai dengan versi evaluator yang dicatat; hasil historis tidak ditimpa.
2. Perbaiki kontrak data dan probabilitas yang terbukti salah: pemeriksaan identitas proposal, confidence WBF, head_p untuk ranking, prediksi kosong, serta penelusuran split terhadap checkpoint. Perbaikan ini dapat diperiksa sebelum melatih model baru.
3. Dekomposisikan galat pada satu bank prediksi tetap: objek tanpa proposal, lokalisasi, duplikat, kelas salah, penggabungan identitas, pemecahan identitas, dan klaster yang dibuang rekonsiliasi jumlah. TIDE menyediakan kerangka galat deteksi; asosiasi fisik memerlukan tambahan khusus proyek. [TIDE, Bolya dkk., ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/849_ECCV_2020_paper.php).
4. Ulangi hanya eksperimen yang hipotesisnya berubah akibat bug terkonfirmasi, satu komponen setiap kali pada TRAIN/VAL. Tidak mengulang seluruh penelusuran secara otomatis. Angka pelanggaran UF 45,3% dari AF-E-010 memakai graf geometri pengganti tanpa Hungarian; itu belum membuktikan profil terkunci mengalami prevalensi sama. [Implementasi audit UF](../../scripts/audit_forensik/run_e345.py).
5. Gunakan evaluasi antarkampanye dan per kelompok pohon untuk generalisasi. Kehadiran beragam sumber di train dan test bukan bukti generalisasi ke domain baru. Pemilihan model juga bagian protokol generalisasi. Riwayat test yang telah digunakan berkali-kali harus diakui sebagai benchmark pengembangan. [WILDS, Koh dkk., ICML 2021](https://proceedings.mlr.press/v139/koh21a.html), [DomainBed, Gulrajani dan Lopez-Paz](https://arxiv.org/abs/2007.01434), [Dwork dkk., NeurIPS 2015](https://proceedings.neurips.cc/paper/2015/hash/bad5f33780c42f2588878a9d07405083-Abstract.html).

Rancangan yang layak diperiksa adalah satu himpunan kandidat tandan fisik dengan probabilitas keberadaan, keanggotaan tampak, dan distribusi kelas yang konsisten; jumlah diturunkan dari himpunan yang sama. Ini merupakan perbaikan konsistensi pipeline yang sudah ada, bukan klaim arsitektur baru. Regresi jumlah boleh menjadi pembanding, tetapi peningkatan angkanya harus disertai pemeriksaan apakah identitas tandan ikut membaik. Jalur 3D belum menjadi prioritas karena sejarah pose/depth proyek sudah menunjukkan keterbatasan pengambilan empat tampak; pendekatan buah berbasis 3D memerlukan kondisi pose dan overlap yang memadai. [Matos dkk., CVPR Workshop 2024](https://openaccess.thecvf.com/content/CVPR2024W/Vision4Ag/html/Matos_Tracking_and_Counting_Apples_in_Orchards_Under_Intermittent_Occlusions_and_CVPRW_2024_paper.html).

## Batas review dan hasil kerja

Review ini selesai untuk cakupan kode dan artefak yang disebut, bukan audit setiap berkas repositori. Dampak perbaikan terhadap mAP belum diukur. Source GSP/reranker historis yang disebut beberapa laporan belum seluruhnya ditemukan; evaluasi baru tidak mengarang implementasinya. Klaim 0,85 dan konsistensi >95% tetap belum tercapai untuk sasaran awal pengguna.

Keluaran review terdiri atas laporan ini, dua skrip diagnostik, hasil JSON dengan hash sumber, dan matriks bukti terpisah. Kode produksi tidak diubah oleh review ini. Laporan disiapkan sebagai DOCX; pemeriksaan struktur, tautan, dan kesesuaian angka dilakukan, tetapi render visual tidak tersedia di lingkungan sesi.
