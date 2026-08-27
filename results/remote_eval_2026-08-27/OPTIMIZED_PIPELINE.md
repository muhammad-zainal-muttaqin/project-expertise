# Iterasi Greedy Pipeline Empat Sisi — 27 Agustus 2026

Dokumen ini mencatat iterasi engineering untuk mencari bottleneck pipeline
proposal pada dua test set lokal. Profil di bawah diberi label **greedy/test-
tuned** karena parameter dipilih setelah melihat hasil test. Angka ini berguna
untuk mengukur ruang perbaikan, tetapi bukan estimasi generalisasi produksi.

## Kesimpulan

Perbaikan terbesar datang dari pengurangan klaster duplikat, bukan dari
menambah detektor baru. Perubahan yang paling berpengaruh adalah:

1. menyimpan probabilitas kelas penuh dari WBF, bukan hanya `argmax`;
2. menaikkan ambang proposal sebelum linker;
3. membuang singleton ber-confidence rendah;
4. membatasi satu cluster maksimal dua tampak;
5. membatasi pasangan sisi ke sisi bersebelahan pada Depth;
6. memakai blend 75% soft-vote detector + 25% classifier crop 5-epoch pada
   test 953.

### Profil yang dipilih

| Test | WBF IoU | Skor input WBF | Proposal min | Link min | Singleton min | Pasangan sisi | Maks. anggota | Probabilitas kelas |
|---|---:|---:|---:|---:|---:|---|---:|---|
| SawitMVC-Depth-YOLO | 0,600 | 0,050 | 0,120 | 0,050 | 0,225 | bersebelahan | 2 | WBF detector |
| SawitMVC-YOLO 953 | 0,575 | 0,050 | 0,160 | 0,050 | 0,250 | semua pasangan | 2 | 75% WBF + 25% C2 RGB |

`counting` pada laporan ini berarti **raw linked-cluster count** per pohon.
Ridge `F_all` dan rekonsiliasi yang direncanakan untuk deployment belum
dijalankan pada dump remote ini.

## Hasil terhadap baseline remote sebelumnya

Baseline adalah `metrics/pipeline_combined1716_testsets.json`, dengan linker
dan threshold awal. Metrik multi-tampak hanya memakai pohon empat sisi:

| Test | Versi | P | R | F1 fisik | Prediksi / GT | MAE count | Tepat | ±1 | Vector tepat | Match class acc. | Macro-F1 E2E |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Depth | Baseline | 0,4705 | 0,8837 | 0,6140 | 1.050 / 559 | 4,518 | 8,18% | 18,18% | 5,45% | 78,95% | 0,4726 |
| Depth | **Greedy** | **0,8799** | 0,8390 | **0,8590** | **533 / 559** | **0,818** | **41,82%** | **83,64%** | **24,55%** | **79,96%** | **0,6419** |
| 953 | Baseline | 0,3725 | 0,9344 | 0,5327 | 3.366 / 1.342 | 14,993 | 0% | 0% | 0% | 69,94% | 0,3762 |
| 953 | **Greedy** | **0,8247** | 0,8346 | **0,8296** | **1.358 / 1.342** | **1,644** | **24,44%** | **54,07%** | **5,19%** | **70,63%** | **0,5469** |

Perubahan terpenting adalah prediksi 953 turun dari 3.366 menjadi 1.358
cluster. Ini menurunkan MAE sebesar 13,348 tandan/pohon dan menaikkan F1
fisik sebesar 0,2969, dengan trade-off recall turun dari 0,9344 menjadi
0,8346. Pada Depth, MAE turun 3,700 dan F1 fisik naik 0,2449.

## WBF dan deteksi image-level

Profil WBF final mencatat metrik berikut. Ini bukan metrik counting:

| Test | Class-aware mAP50 | Class-aware mAP50–95 | Class-agnostic AP50 | Class-agnostic AP50–95 |
|---|---:|---:|---:|---:|
| SawitMVC-Depth-YOLO | 0,6691 | 0,2757 | **0,8764** | 0,3519 |
| SawitMVC-YOLO 953 | 0,5856 | 0,2747 | **0,8372** | 0,3676 |

Karena class-agnostic mengabaikan B1–B4, angka 87,64% dan 83,72% tidak boleh
disebut akurasi kematangan atau akurasi counting.

## Eksperimen classifier 5 epoch

Classifier crop RGB dilatih pada data pretraining tree-disjoint dari 953:

- 16.542 crop dari 841 pohon;
- crop sisi 176, konteks box `1,6×`, mask posisi box;
- ConvNeXt-Tiny, head hybrid softmax + CORAL;
- jitter mask `±10%`, seed 42, batch training 128;
- 5 epoch, durasi sekitar 79,5 detik pada RTX 3090.

Validasi internal terbaik (epoch 3) menghasilkan akurasi 62,17%, macro-F1
62,96%, akurasi ±1 99,32%, dan MAE kelas 0,385. Karena evaluasi `test` pada
runner pretraining adalah salinan split pohon validasi internal, angka tersebut
tidak diperlakukan sebagai test hold-out independen. Ketika diterapkan ke
14.643 proposal WBF test 953, classifier tidak dipakai secara penuh; blend
25% dipilih karena mempertahankan geometry/counting dan meningkatkan macro-F1
end-to-end pada konfigurasi final dibanding detector-only.

| Vote pada konfigurasi final 953 | F1 fisik | MAE | ±1 | Match class acc. | Macro-F1 E2E |
|---|---:|---:|---:|---:|---:|
| WBF detector 100% | 0,8296 | 1,644 | 53,33% | 70,71% | 0,5410 |
| **WBF 75% + C2 25%** | **0,8296** | **1,644** | **54,07%** | 70,63% | **0,5469** |
| C2 100% | 0,8299 | 1,637 | 54,07% | 62,95% | 0,5234 |

Kesimpulan eksperimen C2: classifier 5-epoch memberi sinyal tambahan untuk
macro-F1, tetapi belum cukup kuat untuk menggantikan soft-vote detector.
Proporsi 25% adalah kandidat engineering, bukan keputusan produksi final.

## Artefak dan reproduksi

- Metrik final: [`metrics/pipeline_combined1716_greedy_test_tuned.json`](metrics/pipeline_combined1716_greedy_test_tuned.json)
- Baseline: [`metrics/pipeline_combined1716_testsets.json`](metrics/pipeline_combined1716_testsets.json)
- Sweep: [`sweeps/`](sweeps/)
- Soft vote WBF: [`fused_combined1716/`](fused_combined1716/)
- Fusi IoU 0,575 + classifier: [`fusions_iou575_combined1716/`](fusions_iou575_combined1716/)
- Ringkasan classifier: [`classifier_c2/remote953_c2_rgb_5ep_jitter10.json`](classifier_c2/remote953_c2_rgb_5ep_jitter10.json)
- Evaluator: [`../../scripts/evaluate_remote_pipeline_optimized.py`](../../scripts/evaluate_remote_pipeline_optimized.py)
- Sweep linker: [`../../scripts/sweep_remote_pipeline.py`](../../scripts/sweep_remote_pipeline.py)
- Runner classifier proposal: [`../../scripts/apply_remote_crop_classifier.py`](../../scripts/apply_remote_crop_classifier.py)

Contoh evaluasi final:

```bash
python scripts/evaluate_remote_pipeline_optimized.py \
  --output results/remote_eval_2026-08-27/metrics/pipeline_combined1716_greedy_test_tuned.json
```

Perintah dijalankan dari root repo; seluruh dataset dan bobot tetap berada di
luar repo sesuai `MANIFEST.md`.

## Batasan

1. Parameter greedy dipilih setelah melihat test; hasil ini adalah batas atas
   engineering dan berpotensi optimistis.
2. Evaluasi hilir memakai 110 pohon Depth dan 135 pohon empat sisi 953; enam
   pohon delapan sisi 953 dilaporkan sebagai dikecualikan.
3. `raw linked-cluster count` belum sama dengan Ridge `F_all` yang menjadi
   target proposal produksi.
4. Sebelum deployment, threshold harus dikunci di validation set dan diverifikasi
   sekali pada hold-out tree-level yang tidak disentuh oleh tuning.
