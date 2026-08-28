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

Bootstrap memakai 500 resample level-citra dengan indeks yang sama untuk
kedua model. Untuk YOLO26l, fraksi resample dengan Δ positif adalah 0,558;
untuk RF-DETR-L v2 0,232. Kedua CI masih melintasi nol, sehingga data ini tidak
mendukung klaim bahwa depth menaikkan mAP pada `new763`. Checkpoint regular RF
yang tidak dipilih memberi mAP50 0,604826 dan mAP50:95 0,226367; karena
seleksi protokol memakai mAP50:95, angka utama tetap checkpoint `best_total`.

## Perubahan per kelas

| Model | B1 Δ AP50 | B2 Δ AP50 | B3 Δ AP50 | B4 Δ AP50 |
|---|---:|---:|---:|---:|
| YOLO26l RGB+D4 − RGB | −0,018841 | −0,002292 | +0,026175 | −0,004377 |
| RF-DETR-L v2 RGB+D4 − RGB | +0,011954 | −0,010345 | −0,022481 | −0,023780 |

Pada YOLO26l, depth membantu B3 tetapi hampir seluruh gain itu diimbangi
turunnya B1, B2, dan B4. Pada RF-DETR-L v2, B1 naik, tetapi B3 dan B4 turun;
B4 tetap menjadi kelas yang paling lemah. Run RF-DETR v1 tidak masuk tabel
karena ekspansi stem 3→4 terjadi setelah optimizer dibuat sehingga kanal depth
tidak pernah dilatih.
Pola ini konsisten dengan kesimpulan agregat: kanal keempat terbaca dan
berkontribusi, tetapi belum memberi sinyal generalisasi yang lebih baik.

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
| Grafik agregat | [`results/figures/new763_rgbd4_val_comparison.png`](../results/figures/new763_rgbd4_val_comparison.png) |
| Grafik per kelas | [`results/figures/new763_rgbd4_per_class_delta.png`](../results/figures/new763_rgbd4_per_class_delta.png) |
| Script builder/training/eval/CI | [`scripts/`](../scripts/) dengan nama `*new763_rgbd4*` |

Checkpoint besar tidak dimasukkan ke GitHub. Best model sudah diunggah
langsung ke bucket Hugging Face:

- `hf://buckets/ULM-DS-Lab/project-expertise-backup/runs/new763_rgbd4/rfdetr_l_rgbd4_s42_i1280_fair_v2_checkpoint_best_total.pth`
- `hf://buckets/ULM-DS-Lab/project-expertise-backup/runs/new763_rgbd4/yolo26l_rgbd4_s42_i1280_best.pt`

Hash dan ukuran kedua model tercantum dalam
[`results/new763_rgbd4/new763_rgbd4_summary.json`](../results/new763_rgbd4/new763_rgbd4_summary.json).
Dataset TIFF RGBD4 sekitar 6,9 GiB juga sengaja tidak dimasukkan ke GitHub;
manifest, konfigurasi, metrics training, prediction dump VALID, dan seluruh
JSON hasil sudah dicatat di repository.

## Keputusan

Dengan protokol ini, depth belum layak menggantikan baseline RGB pada
`new763`: YOLO26l pada dasarnya imbang dan RF-DETR-L v2 memiliki point estimate
lebih rendah. Ini bukan bukti bahwa depth tidak berguna secara umum; cakupan
valid sensor hanya sekitar 0,286–0,288 dari grid warna dan eksperimen ini
baru menguji early 4-channel fusion dengan depth stem nol. Eksperimen fusion
lain harus tetap diperlakukan sebagai ablation baru dan tidak boleh membuka
TEST secara diam-diam.
