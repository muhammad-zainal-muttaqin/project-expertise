# Hasil Eksperimen new763 RGB+D Empat Kanal

## Ringkasan

Eksperimen ini menguji satu pertanyaan terkontrol: apakah menambahkan depth
sebagai kanal keempat meningkatkan deteksi pada `new763`? Pembanding RGB dan
RGB+D4 memakai 468 citra VALID yang sama, label yang sama, ukuran 1.280, seed
42, dan evaluator `pycocotools.COCOeval`. Dataset `combined1716` tidak diberi
depth pada eksperimen ini karena cakupan depth-nya tidak lengkap.

| Arsitektur | RGB mAP50 | RGB+D4 mAP50 | Δ mAP50 | RGB mAP50:95 | RGB+D4 mAP50:95 | Δ mAP50:95 | CI95 paired Δ mAP50 | Kesimpulan |
|---|---:|---:|---:|---:|---:|---:|---|---|
| YOLO26l | 0,529357 | 0,529523 | +0,000166 | 0,197855 | 0,195487 | −0,002368 | [−0,024195; +0,028892] | Tidak ada gain |
| RF-DETR-L v2 | 0,608233 | 0,597070 | −0,011163 | 0,227471 | 0,226946 | −0,000525 | [−0,037049; +0,018074] | Point turun; belum signifikan |
| RT-DETR-L | 0,577766 | 0,584088 | +0,006322 | 0,211041 | 0,212043 | +0,001002 | [−0,026770; +0,039670] | Point naik; belum signifikan |

Bootstrap memakai 500 resample level-citra dengan indeks yang sama untuk
ketiga model. Untuk YOLO26l, fraksi resample dengan Δ positif adalah 0,558;
untuk RF-DETR-L v2 0,232; untuk RT-DETR-L 0,650. Semua CI masih melintasi
nol, sehingga data ini belum mendukung klaim signifikansi bahwa depth menaikkan
mAP pada `new763`. Checkpoint regular RF
yang tidak dipilih memberi mAP50 0,604826 dan mAP50:95 0,226367; karena
seleksi protokol memakai mAP50:95, angka utama tetap checkpoint `best_total`.

## Follow-up: fixed late fusion pada VAL

Karena early fusion 4-channel belum konsisten, prediksi RGB dan RGB+D4 yang
sudah dibekukan diuji sebagai dua sumber yang saling melengkapi. Ini bukan
penyetelan ulang model: hanya ada satu resep fixed, yaitu union class-aware
NMS pada IoU 0,60; WBF mean-score IoU 0,60 disertakan sebagai kontrol karena
WBF adalah modul historis proyek. NMS single-source juga dihitung sebagai
kontrol post-processing. Semua resep dievaluasi pada 468 citra VALID yang
sama, tanpa grid parameter dan tanpa membaca TEST.

| Arsitektur | RGB mAP50 / mAP50:95 | RGB+D4 mAP50 / mAP50:95 | Union-NMS mAP50 / mAP50:95 | Union-WBF mAP50 / mAP50:95 |
|---|---:|---:|---:|---:|
| YOLO26l | 0,529357 / 0,197855 | 0,529523 / 0,195487 | 0,561982 / 0,205965 | **0,567718 / 0,207955** |
| RF-DETR-L v2 | 0,608233 / 0,227471 | 0,597070 / 0,226946 | 0,606856 / **0,231472** | 0,528041 / 0,191830 |
| RT-DETR-L | 0,577766 / 0,211041 | 0,584088 / 0,212043 | **0,606368 / 0,220330** | 0,407071 / 0,145957 |

Single-source NMS sendiri hanya mengubah mAP50 menjadi 0,550878 (YOLO),
0,610903 (RF), dan 0,579054 (RT). Jadi kenaikan union bukan sekadar efek
menekan duplikasi dalam satu model. Untuk RT, union-NMS memberi +0,028602
mAP50 terhadap RGB; untuk YOLO, union-WBF memberi +0,038361.

Paired bootstrap level-citra dilakukan setelah resep fixed dipilih untuk
screening: YOLO union-WBF memakai 500 resample dan menghasilkan Δ rata-rata
0,037912, CI95 [0,016060; 0,059120], seluruh resample positif. RT union-NMS
memakai 200 resample sebagai screen cepat dan menghasilkan Δ rata-rata
0,028492, CI95 [0,009231; 0,047236], seluruh resample positif. Kedua hasil
ini signifikan pada VAL, tetapi karena resep ditemukan melalui screen VAL,
keduanya harus disebut exploratory/validation-selected dan belum menjadi
klaim generalisasi sebelum evaluasi held-out baru. RF union-NMS tidak dipaksa
masuk bootstrap karena mAP50 point-nya sedikit di bawah RGB (0,606856 vs
0,608233), walaupun mAP50:95 naik +0,004001.

Interpretasi sementara: sinyal paling menjanjikan bukan “depth selalu
menang” pada early stem, melainkan kombinasi detector RGB dan RGB+D4 yang
memiliki error berbeda, dengan NMS mengurangi duplikasi query RT dan WBF
memberi manfaat khusus pada YOLO. WBF class-aware naif tidak universal dan
jelas merusak RF/RT; tidak boleh dipakai sebagai modul umum tanpa validasi
terpisah.

## Perubahan per kelas

| Model | B1 Δ AP50 | B2 Δ AP50 | B3 Δ AP50 | B4 Δ AP50 |
|---|---:|---:|---:|---:|
| YOLO26l RGB+D4 − RGB | −0,018841 | −0,002292 | +0,026175 | −0,004377 |
| RF-DETR-L v2 RGB+D4 − RGB | +0,011954 | −0,010345 | −0,022481 | −0,023780 |
| RT-DETR-L RGB+D4 − RGB | −0,011399 | +0,009733 | +0,024001 | +0,002951 |

Pada YOLO26l, depth membantu B3 tetapi hampir seluruh gain itu diimbangi
turunnya B1, B2, dan B4. Pada RF-DETR-L v2, B1 naik, tetapi B3 dan B4 turun;
B4 tetap menjadi kelas yang paling lemah. Run RF-DETR v1 tidak masuk tabel
karena ekspansi stem 3→4 terjadi setelah optimizer dibuat sehingga kanal depth
tidak pernah dilatih. Pada RT-DETR-L, point gain terutama datang dari B3;
paired CI masih melintasi nol sehingga hasil ini belum signifikan.
Pola ini menunjukkan bahwa kanal keempat terbaca dan dapat berkontribusi,
terutama pada RT-DETR, tetapi belum memberi bukti generalisasi yang konsisten
di seluruh arsitektur.

## Desain yang membuat perbandingan adil

- Unit split diwariskan dari `SawitMVC-Depth-YOLO v2.0.0`: 536 pohon / 2.144
  citra TRAIN dan 117 pohon / 468 citra VALID. Semua sudut pandang satu pohon
  tetap berada pada split yang sama.
- Builder hanya membaca dan mematerialkan TRAIN/VALID. Direktori TEST tidak
  dibaca, tidak dibuat dalam dataset RGBD4, dan evaluator tidak mempunyai opsi
  TEST.
- RGB dan depth dipasangkan lewat stem yang sama. Label YOLO disalin lalu
  diverifikasi identik. RGB JPEG didekode sebagai payload RGB yang sama ke TIFF
  lossless; tidak ada re-encode RGB.
- Raw depth Y16 848×480 mm diproyeksikan ke grid warna 1.280×800 memakai
  calibration sidecar per citra, intrinsics kedua kamera, ekstrinsik,
  Brown–Conrady distortion, dan z-buffer. Depth dikodekan dengan batas fisik
  tetap 0,3–20 m; nilai 0 berarti invalid.
- Payload TIFF berada dalam urutan OpenCV `[B,G,R,D]`. Adapter mengubah hanya
  tiga kanal pertama menjadi `[R,G,B,D]`; depth tidak dibalik dan padding
  depth bernilai 0. Augmentasi HSV hanya menyentuh RGB.
- YOLO26l mengawali stem empat kanal dengan tiga bobot RGB generic yang sama
  dan kanal depth nol. Setelah interruption, resume mempertahankan kanal
  depth yang sudah dipelajari; hasil final bukan warm-start dari checkpoint
  RGB new763.
- RF-DETR-L v2 memakai `rf-detr-large-2026.pth` generic yang sama dengan
  baseline RGB (MD5 `5cb72153541cbcb9aa6efa26222acc75`). Patch projection
  DINO memiliki bentuk `(384, 4, 16, 16)`, kanal depth awal nol, dan head COCO
  di-reinitialize menjadi empat kelas pada kedua recipe.
- RT-DETR-L memakai `/workspace/rtdetr-l.pt` generic yang sama dengan baseline
  RGB, membangun HGStem dengan `ch=4` sebelum optimizer, menyalin bobot RGB ke
  tiga kanal pertama, dan menginisialisasi kanal depth ke nol.
- Checkpoint dipilih oleh validation mAP50:95 framework masing-masing sebelum
  prediksi dibekukan. Bootstrap dijalankan setelah prediksi dan checkpoint
  terkunci; tidak ada pemilihan ulang dari test.

## Artefak dan reproduksi

| Jenis | Lokasi |
|---|---|
| Ringkasan JSON/CSV | [`results/new763_rgbd4/`](../results/new763_rgbd4/) |
| Hasil YOLO JSON | [`results/new763_yolo26l_rgbd4_val.json`](../results/new763_yolo26l_rgbd4_val.json) |
| Bootstrap YOLO JSON | [`results/new763_yolo26l_rgbd4_val_bootstrap.json`](../results/new763_yolo26l_rgbd4_val_bootstrap.json) |
| Hasil RF JSON | [`results/new763_rfdetr_l_rgbd4_val.json`](../results/new763_rfdetr_l_rgbd4_val.json) |
| Bootstrap RF JSON | [`results/new763_rfdetr_l_rgbd4_val_bootstrap.json`](../results/new763_rfdetr_l_rgbd4_val_bootstrap.json) |
| Hasil RT JSON | [`results/new763_rtdetr_l_rgbd4_val.json`](../results/new763_rtdetr_l_rgbd4_val.json) |
| Bootstrap RT JSON | [`results/new763_rtdetr_l_rgbd4_val_bootstrap.json`](../results/new763_rtdetr_l_rgbd4_val_bootstrap.json) |
| Fixed late-fusion JSON | [`results/new763_rgbd4/`](../results/new763_rgbd4/) — berkas `*_late_fusion_val.json` dan bootstrap terkait |
| Audit RT checkpoint | [`results/new763_rgbd4/rtdetr_l_rgbd4_checkpoint_audit.json`](../results/new763_rgbd4/rtdetr_l_rgbd4_checkpoint_audit.json) |
| Grafik agregat | [`results/figures/new763_rgbd4_val_comparison.png`](../results/figures/new763_rgbd4_val_comparison.png) |
| Grafik per kelas | [`results/figures/new763_rgbd4_per_class_delta.png`](../results/figures/new763_rgbd4_per_class_delta.png) |
| Script builder/training/eval/CI | [`scripts/`](../scripts/) dengan nama `*new763_rgbd4*` |

Checkpoint besar tidak dimasukkan ke GitHub. Best model sudah diunggah
langsung ke bucket Hugging Face:

- `hf://buckets/ULM-DS-Lab/project-expertise-backup/runs/new763_rgbd4/rfdetr_l_rgbd4_s42_i1280_fair_v2_checkpoint_best_total.pth`
- `hf://buckets/ULM-DS-Lab/project-expertise-backup/runs/new763_rgbd4/rtdetr_l_rgbd4_s42_i1280_fair_best.pt`
- `hf://buckets/ULM-DS-Lab/project-expertise-backup/runs/new763_rgbd4/yolo26l_rgbd4_s42_i1280_best.pt`

Hash dan ukuran ketiga model tercantum dalam
[`results/new763_rgbd4/new763_rgbd4_summary.json`](../results/new763_rgbd4/new763_rgbd4_summary.json).
Dataset TIFF RGBD4 sekitar 6,9 GiB juga sengaja tidak dimasukkan ke GitHub;
manifest, konfigurasi, metrics training, prediction dump VALID, dan seluruh
JSON hasil sudah dicatat di repository.

## Keputusan

Dengan protokol ini, depth belum layak menggantikan baseline RGB pada
`new763`: YOLO26l pada dasarnya imbang, RF-DETR-L v2 memiliki point estimate
lebih rendah, dan RT-DETR-L memberi gain point kecil yang belum signifikan. Ini
bukan bukti bahwa depth tidak berguna secara umum; cakupan
valid sensor hanya sekitar 0,286–0,288 dari grid warna dan eksperimen ini
baru menguji early 4-channel fusion dengan depth stem nol. Eksperimen fusion
lain harus tetap diperlakukan sebagai ablation baru dan tidak boleh membuka
TEST secara diam-diam.

Catatan kualitas RT-DETR: kolom validation loss framework menjadi `NaN` pada
sejumlah epoch akhir, tetapi metrik COCO dan seluruh tensor checkpoint tetap
finite. Anomali ini dicatat di `rtdetr_l_rgbd4_checkpoint_audit.json` dan harus
diungkapkan bila hasil dipakai dalam naskah.
