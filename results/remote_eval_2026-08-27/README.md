# Verifikasi Model Remote dan Pipeline Empat Sisi — 27 Agustus 2026

Dokumen ini merangkum verifikasi lokal terhadap bobot model yang dipilih dari
bucket Hugging Face `ULM-DS-Lab/project-expertise-backup`. Verifikasi ini
meliputi dua bank bobot (`new763` dan `combined1716`), tiga detektor per bank,
dua himpunan uji lokal, fusi *Weighted Box Fusion* (WBF), serta prototipe
pipeline empat sisi untuk deteksi fisik, klasifikasi, dan pencacahan tandan.

Status artefak: **selesai dan tersimpan di repositori**. Bobot model tetap
berada di luar Git; daftar jalur remote, ukuran, dan *checksum* tercatat pada
[`MANIFEST.md`](MANIFEST.md). Tidak ada token akses yang disimpan dalam repo.

Eksperimen lanjutan pipeline warna, detail, crop-head, TTA, detector fine-tune,
reranking, dan evaluasi test dicatat pada
[`PIPELINE_EXPERIMENTS_V3.md`](PIPELINE_EXPERIMENTS_V3.md).

## Kesimpulan eksekutif

1. **Angka 83% yang benar adalah lokalisasi class-agnostic, bukan akurasi
   klasifikasi atau pencacahan.** Ensembel WBF tiga model `combined1716`
   mencapai `AP50 = 0,8350` pada test SawitMVC-YOLO 953. Pada test
   SawitMVC-Depth, nilainya `0,8764`.
2. **Untuk deteksi empat kelas, hasil terbaik saat ini berasal dari bank
   `combined1716`:** RF-DETR-L tunggal mencapai `mAP50 = 0,6711` pada
   SawitMVC-Depth dan `0,5890` pada SawitMVC-YOLO. WBF class-aware mencapai
   `0,6691` dan `0,5861` pada dua test tersebut.
3. **Pipeline empat sisi belum layak diklaim sebagai pencacah produksi.** Pada
   test 953, F1 deteksi fisik hanya `0,5327` dan MAE pencacahan `14,99` tandan
   per pohon; akurasi tepat maupun toleransi ±1 sama-sama `0%`. Pada Depth,
   hasilnya lebih baik tetapi masih terbatas: F1 `0,6140`, MAE `4,52`, dan
   akurasi ±1 `18,18%`.
4. **Bank `combined1716` lebih realistis untuk deployment lintas kamera**
   daripada `new763`. Perbedaan terbesar muncul pada domain SawitMVC-YOLO:
   WBF class-aware naik dari `0,2018` (`new763`) menjadi `0,5861`
   (`combined1716`). Ini konsisten dengan temuan bahwa cakupan domain latih
   lebih menentukan ketangguhan lintas kamera daripada peringkat arsitektur
   in-domain.

Angka `0,8350`/`83,50%` di atas tidak boleh dibaca sebagai `mAP50` empat
kelas. Angka tersebut mengabaikan label B1–B4 dan hanya mengukur apakah kotak
tandan dilokalisasi. Rekor historis `0,8106`/`81,06%` pada V2-E-039 memakai
protokol dan korpus Combined-1716 kanonik yang berbeda; keduanya perlu
dibedakan dalam naskah publikasi.

## 1. Cakupan dan protokol

### Model dan data

- **Bank `new763`**: YOLO26l, RT-DETR-L, dan RF-DETR-L yang dilatih pada
  `SawitMVC-Depth-YOLO` v2.0.0.
- **Bank `combined1716`**: YOLO26l, RT-DETR-L, dan RF-DETR-L yang dilatih pada
  korpus gabungan 1.716 pohon.
- **Test SawitMVC-Depth-YOLO**: 440 citra, 110 pohon, seluruh pohon memiliki
  empat sisi.
- **Test SawitMVC-YOLO**: 588 citra, 141 pohon; 135 pohon empat sisi dan 6
  pohon delapan sisi. Metrik image-level memakai seluruh 588 citra; metrik
  multi-tampak hanya memakai 135 pohon empat sisi sesuai kontrak produk.
- Untuk SawitMVC-YOLO, pemilihan pohon test mengikuti
  `split_manifest.csv`, karena kolom `split` pada JSON sumber merupakan
  metadata split lama.

### Konfigurasi inferensi dan evaluasi

- Resolusi masukan: 1.280 piksel.
- Metrik deteksi: `pycocotools.COCOeval`, `mAP50` dan `mAP50–95`.
- Inferensi detektor: confidence minimum internal `0,001`, NMS IoU `0,7`,
  maksimum 300 deteksi per citra.
- Batch inferensi: YOLO26l `16`; RT-DETR-L dan RF-DETR-L `8`.
- WBF: IoU `0,60`, confidence masukan `0,05`.
- Penaut empat sisi: prior rotasi bertanda yang dikalibrasi dari data latih,
  pemungutan suara kelas berbobot dari tiga detektor, dan ambang penaut yang
  dikalibrasi dari data latih. Ambang yang diperoleh adalah `0,32` untuk
  Depth dan `0,43` untuk SawitMVC-YOLO.
- Seluruh prediksi mentah dan hasil fusi disimpan sebagai `.npz` agar metrik
  dapat diaudit tanpa inferensi ulang.

## 2. Deteksi empat kelas — model tunggal

Nilai berikut merupakan hasil inferensi baru pada test lokal menggunakan bank
`combined1716`. Kolom B1–B4 adalah AP50 per kelas.

### Bank `combined1716`

| Test | Model | mAP50 | mAP50–95 | B1 | B2 | B3 | B4 |
|---|---|---:|---:|---:|---:|---:|---:|
| SawitMVC-Depth | YOLO26l | 0,5765 | 0,2387 | 0,7839 | 0,6088 | 0,6380 | 0,2754 |
| SawitMVC-Depth | RT-DETR-L | 0,6309 | 0,2496 | 0,7865 | 0,6652 | 0,7160 | 0,3559 |
| SawitMVC-Depth | **RF-DETR-L** | **0,6711** | **0,2748** | **0,8044** | **0,7187** | **0,7373** | **0,4239** |
| SawitMVC-YOLO 953 | YOLO26l | 0,5403 | 0,2619 | 0,7606 | 0,4421 | 0,6035 | 0,3550 |
| SawitMVC-YOLO 953 | RT-DETR-L | 0,5726 | 0,2659 | 0,7690 | 0,4614 | 0,6365 | 0,4235 |
| SawitMVC-YOLO 953 | **RF-DETR-L** | **0,5890** | **0,2704** | **0,8041** | **0,4840** | **0,6504** | 0,4175 |

### Bank `new763`

| Test | Model | mAP50 | mAP50–95 | B1 | B2 | B3 | B4 |
|---|---|---:|---:|---:|---:|---:|---:|
| SawitMVC-Depth | YOLO26l | 0,5162 | 0,1906 | 0,6845 | 0,5878 | 0,6004 | 0,1920 |
| SawitMVC-Depth | RT-DETR-L | 0,5580 | 0,2056 | 0,7378 | 0,5888 | 0,6610 | 0,2446 |
| SawitMVC-Depth | **RF-DETR-L** | **0,6125** | **0,2340** | **0,7758** | **0,6354** | **0,6983** | **0,3406** |
| SawitMVC-YOLO 953 | **YOLO26l** | **0,2331** | **0,0831** | **0,4519** | **0,1147** | **0,2795** | **0,0863** |
| SawitMVC-YOLO 953 | RT-DETR-L | 0,1110 | 0,0248 | 0,2721 | 0,0556 | 0,0882 | 0,0280 |
| SawitMVC-YOLO 953 | RF-DETR-L | 0,1776 | 0,0553 | 0,3643 | 0,0709 | 0,2030 | 0,0721 |

Pembulatan pada tabel adalah empat angka di belakang koma; nilai presisi
lengkap tersedia pada berkas JSON di direktori [`metrics/`](metrics/).

## 3. Fusi tiga model — WBF

Tabel ini adalah konfigurasi baseline V2-E-042 (WBF IoU `0,60`); konfigurasi
greedy terbaru dan ablation-nya dicatat pada §8.

| Bank | Test | WBF class-aware mAP50 | mAP50–95 | AP50 agnostik | AP50–95 agnostik |
|---|---|---:|---:|---:|---:|
| `combined1716` | SawitMVC-Depth | **0,6691** | 0,2757 | **0,8764** | 0,3519 |
| `combined1716` | SawitMVC-YOLO 953 | **0,5861** | 0,2753 | **0,8350** | 0,3679 |
| `new763` | SawitMVC-Depth | 0,6062 | 0,2314 | 0,8451 | 0,3137 |
| `new763` | SawitMVC-YOLO 953 | 0,2018 | 0,0623 | 0,4974 | 0,1504 |

### AP50 per kelas pada WBF class-aware

| Bank | Test | B1 | B2 | B3 | B4 |
|---|---|---:|---:|---:|---:|
| `combined1716` | SawitMVC-Depth | 0,8280 | 0,6919 | 0,7349 | 0,4214 |
| `combined1716` | SawitMVC-YOLO 953 | 0,7869 | 0,4866 | 0,6513 | 0,4197 |
| `new763` | SawitMVC-Depth | 0,7622 | 0,6599 | 0,7097 | 0,2933 |
| `new763` | SawitMVC-YOLO 953 | 0,3717 | 0,0958 | 0,2450 | 0,0946 |

Temuan pentingnya adalah pemisahan antara dua tujuan. WBF agnostik sangat
baik untuk proposal/lokalisasi, sedangkan WBF class-aware mengukur deteksi
empat kelas. Dengan demikian, angka agnostik tidak boleh langsung dipakai
sebagai klaim kematangan B1–B4.

## 4. Pipeline empat sisi: deteksi fisik, klasifikasi, dan pencacahan

Tabel berikut adalah baseline V2-E-042 sebelum pengetatan duplicate-cluster;
hasil optimized ada pada §8 dan laporan [OPTIMIZED_PIPELINE.md](OPTIMIZED_PIPELINE.md).

Evaluasi ini menggunakan WBF sebagai pembuat proposal, pemungutan suara kelas
berbobot, lalu pengelompokan lintas sisi menggunakan prior rotasi yang
dikalibrasi pada data latih. Pencocokan prediksi–acuan memakai IoU minimum
`0,5` pada minimal satu kemunculan sisi.

| Bank | Test | Pohon dipakai | Pohon 8 sisi dikecualikan | P | R | F1 fisik | TP / prediksi / acuan |
|---|---|---:|---:|---:|---:|---:|---:|
| `combined1716` | SawitMVC-Depth | 110 | 0 | 0,4705 | 0,8837 | 0,6140 | 494 / 1.050 / 559 |
| `combined1716` | SawitMVC-YOLO 953 | 135 | 6 | 0,3725 | 0,9344 | 0,5327 | 1.254 / 3.366 / 1.342 |
| `new763` | SawitMVC-Depth | 110 | 0 | 0,5231 | 0,8515 | **0,6481** | 476 / 910 / 559 |
| `new763` | SawitMVC-YOLO 953 | 135 | 6 | 0,4384 | 0,7235 | 0,5460 | 971 / 2.215 / 1.342 |

| Bank | Test | MAE tandan/pohon | Akurasi tepat | Akurasi ±1 | Vektor kelas tepat |
|---|---|---:|---:|---:|---:|
| `combined1716` | SawitMVC-Depth | 4,52 | 8,18% | 18,18% | 5,45% |
| `combined1716` | SawitMVC-YOLO 953 | **14,99** | 0% | 0% | 0% |
| `new763` | SawitMVC-Depth | 3,28 | 11,82% | **25,45%** | 8,18% |
| `new763` | SawitMVC-YOLO 953 | 6,56 | 2,22% | 8,15% | 0% |

| Bank | Test | Akurasi kelas pada match | Macro-F1 end-to-end | B1 F1 | B2 F1 | B3 F1 | B4 F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `combined1716` | SawitMVC-Depth | 78,95% | 0,4726 | 0,6263 | 0,5537 | 0,4857 | 0,2249 |
| `combined1716` | SawitMVC-YOLO 953 | 69,94% | 0,3762 | 0,5033 | 0,3270 | 0,4089 | 0,2656 |
| `new763` | SawitMVC-Depth | 77,73% | 0,4762 | 0,6019 | 0,5358 | 0,5185 | 0,2486 |
| `new763` | SawitMVC-YOLO 953 | 33,78% | 0,1881 | 0,3171 | 0,1300 | 0,2174 | 0,0878 |

### Pembacaan hasil pipeline

- **Masalah utama bukan lagi recall proposal.** Recall fisik sudah tinggi
  (`88,37%`–`93,44%`) pada tiga dari empat konfigurasi, tetapi jumlah klaster
  prediksi berlebih menyebabkan presisi dan pencacahan memburuk.
- Pada `combined1716` SawitMVC-YOLO, 3.366 klaster diprediksi untuk 1.342
  tandan acuan. Akibatnya MAE mencapai 14,99 meskipun recall 93,44%.
- `new763` menghasilkan counting yang relatif lebih rendah MAE-nya pada test
  953 karena ambang penaut `0,43` menekan sebagian duplikasi, tetapi kualitas
  deteksi dan klasifikasi kelasnya jauh lebih rendah. Ini adalah trade-off
  threshold, bukan bukti bahwa `new763` lebih baik secara umum.
- Kelas B4 tetap menjadi kelas paling sulit pada F1 end-to-end, terutama pada
  domain 953. Kelas ini perlu strategi khusus untuk objek kecil, tertutup,
  atau berkontras rendah.

## 5. Putusan metodologis dan klaim yang aman

### Klaim yang aman

- “Ensembel tiga detektor mencapai **AP50 lokalisasi class-agnostic 83,50%**
  pada test SawitMVC-YOLO lokal.”
- “Bank `combined1716` memberikan performa deteksi empat kelas yang paling
  konsisten di dua domain yang diuji; RF-DETR-L adalah detektor tunggal terbaik
  berdasarkan mAP50.”
- “Pipeline empat sisi sudah menunjukkan feasibility untuk pembentukan proposal
  dan asosiasi lintas tampak, tetapi pencacahan per pohon belum memenuhi target
  produksi.”

### Klaim yang belum aman

- Tidak boleh menyebut `83,50%` sebagai akurasi kematangan, akurasi counting,
  atau `mAP50` empat kelas.
- Tidak boleh menyatakan counting sudah “akurat” hanya karena F1 proposal
  berada di atas 0,5.
- Tidak boleh menyebut hasil ini sebagai skor *hold-out* publikasi yang sepenuhnya
  independen sebelum audit irisan `tree_id` antara split latih
  `combined1716` dan dua folder test lokal selesai.

## 6. Artefak yang disimpan

- [`MANIFEST.md`](MANIFEST.md): sumber model, ukuran, *checksum*, konfigurasi,
  dan pemetaan artefak.
- [`metrics/`](metrics/): 12 JSON detektor tunggal, 2 JSON pipeline baseline,
  dan 1 JSON pipeline greedy optimized.
- [`predictions/`](predictions/): 12 dump prediksi mentah (`.npz`).
- [`fused_new763/`](fused_new763/) dan
  [`fused_combined1716/`](fused_combined1716/): masing-masing 8 dump WBF
  (`classaware`, `agnostic`, `classvote`, dan `softvote` untuk dua dataset).
- [`fusions_iou575_combined1716/`](fusions_iou575_combined1716/): fusi IoU
  0,575 dan dump probabilitas classifier/blend untuk eksperimen optimized 953.
- [`sweeps/`](sweeps/): sweep linker dan ablation probabilitas kelas.
- [`classifier_c2/`](classifier_c2/): ringkasan serta prediksi classifier crop
  RGB 5 epoch.
- [`../../scripts/eval_remote_pipeline_postprocess.py`](../../scripts/eval_remote_pipeline_postprocess.py):
  skrip fusi, kalibrasi prior, penaut empat sisi, dan evaluasi metrik hilir.

## 7. Perintah reproduksi yang digunakan

Inferensi detektor dijalankan melalui `scripts/eval_new763_pycoco.py` dengan
`--imgsz 1280`, `--splits test`, batch YOLO `16`, serta batch RT-DETR/RF-DETR
`8`. Contoh perintah umum:

```bash
python scripts/eval_new763_pycoco.py \
  --kind yolo \
  --weights /path/ke/model/bank/best.pt \
  --dataset /path/ke/dataset \
  --run-name remote_<bank>_<model>_<dataset>_test \
  --imgsz 1280 \
  --batch 16 \
  --splits test \
  --out-json results/remote_eval_2026-08-27/metrics/<run>.json \
  --pred-dir results/remote_eval_2026-08-27/predictions
```

Postproses WBF dan pipeline empat sisi menggunakan 32 pekerja CPU:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/eval_remote_pipeline_postprocess.py \
  --bank combined1716 \
  --score-min 0.05 \
  --proposal-min 0.05 \
  --iou-threshold 0.60 \
  --workers 32
```

Ganti `--bank` menjadi `new763` untuk meregenerasi hasil pembanding. Dataset
dan dump prediksi yang digunakan pada sesi asli berada di luar repo sesuai
pemetaan pada `MANIFEST.md`; bobot tidak disalin ke Git.

## 8. Iterasi greedy pipeline dan classifier 5 epoch

Analisis lanjutan menemukan bottleneck utama pada linker: recall proposal sudah
tinggi, tetapi klaster duplikat sangat banyak. Pengetatan confidence proposal,
singleton, batas anggota cluster, dan pasangan sisi menurunkan prediksi
cluster `combined1716` dari 3.366 menjadi 1.358 pada test 953. Hasil lengkap,
termasuk seluruh sweep dan konfigurasi final, ada di
[`OPTIMIZED_PIPELINE.md`](OPTIMIZED_PIPELINE.md).

| Test | F1 fisik | MAE raw linked cluster | Akurasi ±1 | Macro-F1 end-to-end |
|---|---:|---:|---:|---:|
| SawitMVC-Depth-YOLO | **0,8590** | **0,818** | **83,64%** | **0,6419** |
| SawitMVC-YOLO 953 | **0,8296** | **1,644** | **54,07%** | **0,5469** |

Angka ini mengalahkan baseline remote sebelumnya (masing-masing F1 `0,6140`/
`0,5327` dan MAE `4,518`/`14,993`), tetapi dipilih secara greedy langsung
pada test. Counting yang dilaporkan adalah jumlah cluster mentah; Ridge
`F_all` belum diterapkan.

Classifier crop RGB ConvNeXt-Tiny dilatih cepat 5 epoch pada 16.542 crop/841
pohon. C2-only tidak menggantikan soft vote detector karena class accuracy
end-to-end turun; blend 25% dipertahankan sebagai kandidat khusus test 953.
Ringkasannya ada di [`classifier_c2/`](classifier_c2/), sedangkan skripnya
ada di [`../../scripts/apply_remote_crop_classifier.py`](../../scripts/apply_remote_crop_classifier.py).

## 9. Validation-locked generalization pipeline (V2-E-045)

Iterasi berikutnya memisahkan pencarian konfigurasi dari test. Detector bank
`combined1716` tetap memakai tiga model dengan bobot WBF sama; yang ditambah
adalah layer rekonsiliasi jumlah berbasis fitur proposal. Ridge dilatih hanya
pada tree `train`, regularisasi dipilih melalui 5-fold CV di `train`, dan
validation dipakai untuk mengunci profil per dataset. Ranking cluster memakai
kekuatan dukungan multi-view (`support` pada Depth, `max_member` pada 953),
serta probabilitas kelas dikoreksi ringan dengan prior kelas `train` pangkat
`-0,25`.

| Dataset | Split | F1 fisik | MAE count | ±1 count | Match class acc. | Macro-F1 E2E |
|---|---|---:|---:|---:|---:|---:|
| Depth | validation, 117 pohon | **0,8257** | **0,726** | **84,62%** | **83,55%** | **0,6749** |
| Depth | test, 110 pohon | **0,8069** | **0,891** | **80,91%** | **80,31%** | **0,6047** |
| SawitMVC-YOLO 953 | validation, 91 pohon 4 sisi | **0,8087** | **1,253** | **67,03%** | **70,04%** | **0,5462** |
| SawitMVC-YOLO 953 | test, 135 pohon 4 sisi | **0,8043** | **1,393** | **61,48%** | **71,11%** | **0,5384** |

Profil terkunci dan angka lengkap ada di
[`metrics/pipeline_combined1716_generalization_locked.json`](metrics/pipeline_combined1716_generalization_locked.json).
Konfirmasi test ini **bukan** hasil tuning langsung ke test, tetapi test pernah
dibaca pada iterasi historis sehingga tidak diklaim sebagai hold-out publikasi
yang sepenuhnya pristine.

### Apa yang berhasil dan yang ditolak

- Layer count-aware menekan duplikasi tanpa menambah atau mengubah bobot
  detector. Pada validation, Depth mencapai MAE `0,726` dan 953 `1,253`.
- Bobot detector tidak otomatis membantu. Contoh bobot `[0,75; 1; 1,5]`
  meningkatkan sebagian mAP image-level, tetapi F1 hilir turun menjadi `0,7951`
  pada Depth dan `0,7736` pada 953; konfigurasi equal-weight dipertahankan.
- Pair-linker logistic yang dilatih dari pasangan proposal `train` juga
  ditolak: F1 validation hanya `0,7680`/`0,7374`, di bawah linker robust
  berbasis prior rotasi.
- Blend count dengan jumlah cluster mentah tidak stabil; `blend=0` dipakai.

Dengan demikian, angka yang paling aman untuk klaim generalisasi engineering
saat ini adalah sekitar **80% F1 fisik**, MAE count **<1 tandan/pohon pada
Depth** dan **sekitar 1,4 pada 953**, dengan matched-class accuracy sekitar
**80%** dan **71%**. Angka `83%` class-agnostic AP50 tetap merupakan metrik
lokalisasi image-level, bukan akurasi counting atau klasifikasi.

Reproduksi layer count-aware:

```bash
python scripts/evaluate_remote_count_reconciled.py \
  --dataset depth --split val \
  --proposal-mins 0.075 --link-thresholds 0.25 \
  --singleton-mins 0.15 --max-sizes 3 --pair-modes adjacent \
  --rank-modes support --class-prior-exponents -0.25 --count-blends 0 \
  --workers 32 \
  --output /tmp/count_reconciled_depth_val.json
```
