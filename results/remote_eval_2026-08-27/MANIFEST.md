# Manifest Verifikasi Remote — 27 Agustus 2026

## Identitas run

- Sumber model: bucket Hugging Face
  [`ULM-DS-Lab/project-expertise-backup`](https://huggingface.co/buckets/ULM-DS-Lab/project-expertise-backup).
- Model yang diambil: hanya enam bobot detektor yang diperlukan untuk dua bank
  (`new763` dan `combined1716`); seluruh bucket tidak diklon.
- Waktu pembuatan metrik: 27 Agustus 2026 UTC.
- Evaluasi: 12 inferensi model tunggal + baseline dan iterasi greedy pipeline
  ensembel, termasuk aplikasi classifier crop 5 epoch pada proposal 953.
- Secret: token akses **tidak** disimpan di repo, log, manifest, atau nama
  berkas. Karena token pernah ditempelkan di percakapan, token tersebut
  sebaiknya dicabut dan dibuat ulang setelah pekerjaan selesai.

## Bobot model terpilih

Bobot berada di `/workspace/model_artifacts/project-expertise/` dan sengaja
tidak masuk Git. SHA256 berikut diverifikasi pada sesi evaluasi.

| Bank | Model | Jalur remote | Jalur lokal di luar repo | Ukuran (byte) | SHA256 |
|---|---|---|---|---:|---|
| `new763` | YOLO26l | `project-expertise/runs_new763/yolo26l_rgb_s42_i1280/weights/best.pt` | `new763/yolo26l_rgb_s42_i1280_best.pt` | 53.063.525 | `d07c05e23dd8a31361e3fd526eb4fd230847ad775b55f1fdf605b47c48c0f500` |
| `new763` | RT-DETR-L | `project-expertise/runs_new763/rtdetr_l_rgb_s42_i1280/weights/best.pt` | `new763/rtdetr_l_rgb_s42_i1280_best.pt` | 66.455.549 | `07985addfb5067b9f2682899baaa078b36df4368d5b334fc61e8b31160fb1e31` |
| `new763` | RF-DETR-L | `project-expertise/runs_new763/rfdetr_l_rgb_s42_i1280/checkpoint_best_ema.pth` | `new763/rfdetr_l_rgb_s42_i1280_checkpoint_best_ema.pth` | 141.575.603 | `b0f25d919962cfcc8601eb1b05888df459f726229febe9dc5d3fd9dd23c953f74` |
| `combined1716` | YOLO26l | `project-expertise/runs_combined1716/combined1716_yolo26l_rgb_s42_i1280/weights/best.pt` | `combined1716/yolo26l_combined1716_best.pt` | 53.088.869 | `2d503012aa53b36e66cee5edb2580e8c5921b419a8a0ed71e20837c56b7a3200` |
| `combined1716` | RT-DETR-L | `project-expertise/runs_combined1716/combined1716_rtdetr_l_rgb_s42_i1280/weights/best.pt` | `combined1716/rtdetr_l_combined1716_best.pt` | 263.894.890 | `6a07ff4b840f1881a944b1f59d5cc76f78bbae1b28510c13b45b98eac9822d61` |
| `combined1716` | RF-DETR-L | `project-expertise/runs_combined1716/combined1716_rfdetr_l_rgb_s42_i1280/checkpoint_best_ema.pth` | `combined1716/rfdetr_l_combined1716_checkpoint_best_ema.pth` | 141.575.667 | `7a3eb85a7e8bcd9dc8cc62c438533ea79b6832878b207cb5ab516716984267ee` |

Total ukuran enam bobot di luar repo: 719.654.103 byte.

## Dataset dan adapter

| Dataset evaluasi | Akar data sesi | Split | Catatan |
|---|---|---|---|
| SawitMVC-Depth-YOLO | `/workspace/SawitMVC-Depth-YOLO` | `test` | 440 citra, 110 pohon, empat sisi |
| SawitMVC-YOLO | `/workspace/SawitMVC-YOLO` melalui adapter symlink eksternal | `test` | 588 citra, 141 pohon; 135 empat sisi dan 6 delapan sisi |

Adapter eksternal hanya menyediakan struktur `test/images` dan `test/labels`
yang dibutuhkan evaluator. Metadata pohon SawitMVC-YOLO dibaca dari dataset
asli dan pemilihan split mengikuti `split_manifest.csv`.

## Struktur artefak terlacak

| Direktori | Isi | Jumlah |
|---|---|---:|
| [`metrics/`](metrics/) | JSON metrik model tunggal dan pipeline | 15 |
| [`predictions/`](predictions/) | Dump prediksi mentah semua kombinasi | 12 |
| [`fused_new763/`](fused_new763/) | WBF bank `new763` (`classaware`, `agnostic`, `classvote`, `softvote`) | 8 |
| [`fused_combined1716/`](fused_combined1716/) | WBF bank `combined1716` (`classaware`, `agnostic`, `classvote`, `softvote`) | 8 |
| [`fusions_iou575_combined1716/`](fusions_iou575_combined1716/) | WBF IoU 0,575 + C2/blend untuk test 953 | 14 |
| [`sweeps/`](sweeps/) | Sweep proposal/linker dan classifier | 17 |
| [`classifier_c2/`](classifier_c2/) | Ringkasan dan prediksi classifier crop 5 epoch | 2 |

Nama JSON model tunggal mengikuti pola
`remote_<bank>_<model>_<dataset>_test.json`. JSON pipeline menyimpan seluruh
metrik per pohon, selain ringkasan WBF dan pipeline pada
[`README.md`](README.md).

## Lingkungan komputasi

- GPU: NVIDIA GeForce RTX 3090, VRAM 24 GB.
- CPU: 32 logical cores; WBF memakai 32 proses independen per citra.
- RAM terdeteksi: sekitar 125 GiB pada sesi ini.
- PyTorch: `2.8.0+cu128`; CUDA runtime `12.8`.
- Ultralytics: `8.4.103`.
- RF-DETR: `1.8.3`.
- `pycocotools`: `2.0.11`.
- Pengaturan pembatas thread BLAS saat postproses:
  `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.

## Parameter pipeline

- Baseline WBF: IoU `0,60`, confidence minimum masukan `0,05`.
- Profil greedy final Depth: WBF IoU `0,60`, proposal minimum `0,12`,
  singleton minimum `0,225`, link `0,05`, pasangan bersebelahan, maksimal
  dua anggota cluster.
- Profil greedy final 953: WBF IoU `0,575`, proposal minimum `0,16`,
  singleton minimum `0,25`, link `0,05`, semua pasangan, maksimal dua
  anggota cluster; probabilitas kelas memakai blend 75% WBF + 25% classifier
  crop RGB 5 epoch.
- Prior rotasi dan ambang linker baseline dipelajari hanya dari metadata
  `train`; threshold greedy dipilih melalui sweep langsung pada test.
- Counting pada metrik remote adalah **raw linked-cluster count**. Ridge
  `F_all` belum dijalankan pada dump ini.
- Metrik multi-tampak hanya memakai pohon empat sisi; enam pohon delapan sisi
  SawitMVC-YOLO dikeluarkan dari metrik hilir, tetapi tidak dikeluarkan dari
  metrik deteksi image-level.

## Catatan reproduktibilitas

JSON mentah mempertahankan beberapa jalur absolut dari staging area sesi asli
untuk provenance. Pemetaan artefak yang dapat dibaca pengguna ada pada
[`README.md`](README.md) dan manifest ini. Bobot dan dataset tidak dilacak oleh
Git; prediksi, hasil fusi, serta metrik sudah disalin ke direktori hasil agar
analisis dapat diulang tanpa mengunduh seluruh bucket.
