# Eksperimen pipeline lanjutan — V3

Tanggal: 27 Agustus 2026
Tujuan: menaikkan klasifikasi empat kelas ke minimal 75% dan lokalisasi
class-agnostic ke sekitar 90%, tanpa memilih konfigurasi berdasarkan test.

## Ringkasan keputusan

Sweep besar sudah mencakup perubahan citra, model crop, fusi prediksi,
penaut empat sisi, dan fine-tuning detector. Tidak ada konfigurasi yang
mencapai dua target sekaligus. Hasil terbaik yang masih dapat diklaim sebagai
evaluasi test yang wajar adalah:

| Sasaran | Dataset | Hasil test | Catatan |
|---|---|---:|---|
| Lokalisasi agnostik AP50 | SawitMVC-Depth | **87,64%** | WBF tiga detector |
| Lokalisasi agnostik AP50 | SawitMVC-YOLO 953 | **83,50%** | WBF tiga detector |
| Akurasi kelas pada match | SawitMVC-Depth | **80,31%** | pipeline empat sisi |
| Akurasi kelas pada match | SawitMVC-YOLO 953 | **71,67%** | Small RGB head, profil standar |
| Akurasi kelas, profil class-priority | SawitMVC-YOLO 953 | **71,81%** | precision/counting turun |
| F1 fisik pipeline | SawitMVC-YOLO 953 | **80,43%** | profil standar |
| MAE counting | SawitMVC-YOLO 953 | **1,393** | tandan per pohon |

Angka 83,50% adalah AP50 lokalisasi class-agnostic. Itu bukan akurasi empat
kelas, bukan F1 counting, dan bukan bukti bahwa seluruh pipeline mencapai 83%.
Pada test 953, konfigurasi validasi dengan akurasi kelas 74,06% turun menjadi
71,81% pada test; karena itu angka 75% belum aman untuk diklaim.

## Konfigurasi yang dibandingkan

Semua sweep utama memakai split train/validation untuk pemilihan konfigurasi.
Test dijalankan setelah kandidat dipilih. Profil standar memakai proposal
minimum 0,125, link threshold 0,30, singleton 0,15, maksimum ukuran klaster 3,
ranking `max_member`, voting mean, dan head RGB Small dengan bobot head yang
dipilih dari validation.

| Keluarga eksperimen | Variasi yang diuji | Keputusan |
|---|---|---|
| Photometric | hue/color MLP, CLAHE, sharpening, gamma 0,95/1,05, brightness/contrast | ditolak; tidak stabil atau tidak mengungguli baseline |
| Color correction | gray-world dan mild white-balance | ditolak; E2E terbaik hanya 71,62% dan 71,88% pada validation |
| Crop head | RGB ConvNeXt Tiny 224/256, ConvNeXt Small 224, Swin Tiny | Tiny/Small paling kompetitif; tidak mencapai 75% pada test |
| Training objective | class weighting, focal fine-tune, ordinal BCE, GT-pretrain | proposal head membaik sedikit, E2E tidak |
| Multimodal | RGB + depth/segmentation/metadata feature | tidak memberi kenaikan E2E yang berarti |
| Ensemble | ordinal + regular head, model-vote classwise, geometric/mean/max/median | tidak mengungguli head Small regular secara konsisten |
| TTA | photometric, context 1,25/2,0, horizontal/vertical flip, rotasi | AP/labelling turun atau sama; ditolak |
| Detector | YOLO/RT-DETR local fine-tune dan YOLO resolusi 1600 | validation lebih rendah atau terlalu mahal tanpa gain |
| Postprocess | WBF threshold/weight, class profile, class-confidence rank, support rank | trade-off precision vs class accuracy; tidak semua target tercapai |
| Counting | count blending dan cluster-quality reranking | counting/kelas tidak membaik secara simultan |

## Hasil validation yang paling informatif

Angka berikut dipakai untuk memilih atau menolak keluarga eksperimen; angka
validation tidak boleh dipresentasikan sebagai hasil generalisasi final.

| Kandidat | Akurasi kelas match | F1 fisik | MAE | Macro-F1 E2E | Putusan |
|---|---:|---:|---:|---:|---|
| Small RGB natural, profil standar | 72,40% | 80,87% | 1,253 | — | kandidat all-round terbaik |
| Small RGB + `class_conf` | **74,06%** | 78,93% | 1,352 | — | terlalu banyak false positive |
| Small + profil all-round + count blend | 72,62% | **81,51%** | 1,330 | **55,98%** | trade-off terbaik, masih <75% |
| Ordinal + regular ensemble | **72,40%** | 80,87% | 1,253 | 55,38% | tidak lebih baik dari regular |
| ConvNeXt Tiny RGB natural | 72,01% | 80,87% | 1,253 | — | sedikit di bawah ordinal mix |
| Focal fine-tune | 71,88% | 80,87% | 1,253 | — | ditolak |
| Gray-world | 71,62% | 80,87% | 1,253 | — | ditolak |
| Mild white-balance | 71,88% | 80,87% | 1,253 | — | ditolak |
| Context 1,25 | 72,40% | 80,87% | 1,253 | 55,28% | sama, tidak perlu biaya inferensi |
| Cluster-quality reranker | 70,71% | 80,55% | — | — | ditolak |

`class_conf` dapat menaikkan akurasi kelas pada validation, tetapi precision
turun dan F1 fisik menjadi 78,93%. Untuk klaim all-round, profil ini tidak
dipakai sebagai pilihan utama.

## Hasil test yang dipilih sebelum melihat test

### Profil standar — Small RGB natural

| Model head | Akurasi kelas match | F1 fisik | MAE | Akurasi ±1 |
|---|---:|---:|---:|---:|
| Small train-only, bobot head validation | **71,67%** | **80,43%** | **1,393** | **61,48%** |
| Small final-fit train+validation | 71,48% | 80,43% | 1,393 | 61,48% |
| Ordinal + regular mix | 71,30% | 80,43% | 1,393 | 61,48% |

Final-fit tidak memberi peningkatan; train-only dipertahankan sebagai hasil
utama karena lebih konservatif terhadap generalisasi.

### Profil class-priority — hanya sebagai trade-off

| Model head | Akurasi kelas match | F1 fisik | MAE |
|---|---:|---:|---:|
| Small RGB natural | **71,81%** | 78,57% | 1,415 |
| Small final-fit train+validation | 71,62% | 78,57% | 1,415 |
| Ordinal + regular mix | 71,72% | 78,57% | 1,415 |

Profil ini memprioritaskan keputusan kelas, tetapi mengorbankan kualitas
deteksi fisik. Ia bukan peningkatan menyeluruh.

## Diagnosis teknis

1. Proposal/lokalisasi sudah cukup kuat, tetapi masih di bawah target 90% pada
   test 953 (`83,50%` AP50).
2. Kesalahan kelas bersifat ordinal: B2 sering tertukar dengan B3, dan B3
   dengan B4. Ini bukan masalah yang dapat diselesaikan stabil dengan hue atau
   sharpening global.
3. Head crop memang belajar sinyal tambahan—proposal validation terbaik sekitar
   65,5% macro-F1—tetapi sinyal tersebut tidak cukup konsisten lintas tampak
   untuk mencapai 75% pada test.
4. Validation class-priority 74,06% memiliki precision/F1 fisik yang lebih
   rendah dan turun pada test. Ini indikasi overfitting pada profil validasi,
   bukan bukti kemampuan general.
5. Oracle disagreement antara baseline dan head menunjukkan masih ada ruang
   teoritis, tetapi gate yang diuji (logistic, ExtraTrees, margin, support,
   confidence, dan cluster quality) tidak dapat menemukan pemilih yang
   general.

## Klaim akhir yang aman

- Sistem ini sudah layak disebut pipeline proposal/lokalisasi multi-model dan
  asosiasi empat sisi.
- Lokalisasi class-agnostic terbaik yang terukur adalah 87,64% pada Depth dan
  83,50% pada YOLO 953.
- Untuk empat kelas pada test 953, performa realistis saat ini sekitar 71,7%
  akurasi kelas pada match, dengan F1 fisik sekitar 80,4% dan MAE counting
  sekitar 1,39.
- Belum aman mengklaim 75% klasifikasi empat kelas atau 90% lokalisasi pada
  kedua domain.

## Langkah yang masih bernilai tinggi

Eksperimen filter/TTA tambahan sebaiknya dihentikan. Untuk melewati batas ini,
investasi yang paling masuk akal adalah:

- audit dan perbaikan label B2/B4 serta hard-example mining yang dikunci pada
  split lintas lokasi/kamera;
- fine-tuning detector class-aware dengan data tambahan/oversampling B4,
  bukan hanya postprocess proposal;
- menambah capture empat sisi yang benar-benar baru untuk validasi eksternal;
- menilai ulang dengan split tree/site yang bebas overlap sebelum publikasi.

Tanpa data/label baru, eksperimen yang sudah dilakukan menunjukkan bahwa
menambah layer warna, sharpening, atau reranker hanya menggeser trade-off dan
tidak memberi dasar kuat untuk klaim 75% + 90%.

## Artefak dan reproduksi

Skrip eksperimen tersimpan di [`../../scripts/`](../../scripts/). Dump model
dan prediksi besar berada di luar Git pada `/workspace/model_artifacts/` agar
repo tetap ringan. Artefak utama:

- [`README.md`](README.md) — baseline dan protokol awal;
- [`MANIFEST.md`](MANIFEST.md) — sumber model dan checksum;
- [`Small RGB natural`](/workspace/model_artifacts/project-expertise/proposal_head_953_rgb_small_natural224) — head Small RGB natural;
- [`test_final_standard_head.json`](/workspace/model_artifacts/project-expertise/test_final_standard_head.json) — hasil test utama;
- [`test_final_classpriority.json`](/workspace/model_artifacts/project-expertise/test_final_classpriority.json) — trade-off class-priority;
- [`profile_sweep_regsmall_rankcount_val.json`](../../../model_artifacts/project-expertise/eval_2026-08-27/profile_sweep_regsmall_rankcount_val.json) — sweep profile/count/rank;
- [`crop_ctx125_e2e_val.json`](../../../model_artifacts/project-expertise/crop_ctx125_e2e_val.json) — context TTA;
- [`test_final_fit_standard.json`](../../../model_artifacts/project-expertise/test_final_fit_standard.json) — diagnostic final-fit.
