# Handoff — project-expertise

Dokumen ini adalah ringkasan GitHub dari progres pipeline. Catatan kerja
lengkap berada di `/workspace/HANDOFF.md` pada mesin eksperimen; artifact
model besar berada di `/workspace/model_artifacts/project-expertise/` dan
tidak disimpan di git.

## Tujuan

Membangun pipeline yang menerima empat foto dari empat sisi pohon yang sama,
lalu mendeteksi tandan, menggabungkan deteksi lintas-sisi tanpa
double-counting, menghitung tandan fisik, dan mengklasifikasikan setiap tandan
ke B1/B2/B3/B4 dengan confidence yang dapat diaudit.

Target engineering adalah sekitar 75% untuk klasifikasi empat kelas dan 90%
untuk lokalisasi/kemampuan agnostik. Target tersebut bukan hasil yang boleh
diasumsikan tercapai. Claim final harus memakai test tree-level yang dijalankan
setelah model dan profile dikunci dari train/validation.

## Boundary dan aturan

- Model inference tidak boleh menerima ground-truth box, count, class, atau
  identity tandan.
- Semua fitting dilakukan pada train; pemilihan threshold/profile dilakukan
  pada validation; test hanya dijalankan sekali setelah lock.
- Split dilakukan pada level tree, bukan image/crop, agar empat sisi pohon
  yang sama tidak bocor antar split.
- Metrik crop/head tidak boleh disebut sebagai metrik pipeline end-to-end.
- `matched_class_accuracy`, macro-F1, detector mAP50, physical F1, dan
  counting accuracy adalah metrik berbeda dan harus dilaporkan bersama
  denominator/jumlah match.
- Tidak boleh memakai oracle, memilih member berdasarkan GT, atau membuang
  kasus sulit tanpa dokumentasi.
- Secret/token tidak boleh masuk source, JSON, atau git.

## Arsitektur proposal

Dasar desain adalah [`PROPOSAL-Pipeline.md`](PROPOSAL-Pipeline.md):

```text
4 foto berurutan -> quality check -> YOLO26l/RT-DETR-L/RF-DETR-L
-> WBF class-agnostic -> cross-view linker -> crop classifier
-> ordinal/class aggregation -> count reconciliation -> laporan confidence
```

## Progres aktual

| Modul | Status |
|---|---|
| Tiga detector dan dump prediksi | Selesai |
| WBF class-aware/agnostic/soft-vote | Selesai dan dievaluasi |
| Signed rotation prior lintas-sisi | Selesai |
| Baseline linker dan count reconciliation | Selesai |
| Learned detector-space edge linker | Berhasil di validation, belum test-locked |
| Crop classifier/ordinal head | Banyak varian diuji, belum konsisten memberi gain |
| Quality gate dan retake recommendation | Belum menjadi modul production |
| Confidence/UI deployment | Belum final |
| Test untuk learned linker terbaru | Belum dijalankan |

## Baseline test yang sudah tercatat

- Depth: physical F1 `0.806859`, count MAE `0.8909`, ±1 `0.8091`, matched
  class accuracy `0.8031`, macro-F1 `0.6047`.
- 953: physical F1 `0.804348`, count MAE `1.3926`, ±1 `0.6148`; baseline
  matched class accuracy sekitar `0.7111`.
- WBF agnostic AP50: Depth `0.8764`, 953 `0.8350`; target 0.90 belum tercapai.

Baseline metrics dan eksperimen sebelumnya tersimpan di
`results/remote_eval_2026-08-27/`, terutama:

- `metrics/pipeline_combined1716_generalization_locked.json`;
- `PIPELINE_EXPERIMENTS_V3.md`;
- `README.md` dan `MANIFEST.md`.

## Learned edge linker terbaru

Source: `scripts/train_detection_edge_linker.py`.

Metode melatih classifier pasangan pada proposal WBF nyata. Label train
dibuat dari assignment proposal–GT IoU `>= 0.5` one-to-one per sisi; label GT
tidak masuk inference. Fitur berjumlah 65 dan meliputi geometry box,
signed-rotation residual, area/shape, detector score, soft-class similarity,
entropy/confidence, dan rank lokal dalam sisi.

### SawitMVC-YOLO 953

- Train adjacent: `94,165` pasangan, `4,894` positif.
- Validation adjacent: `13,205` pasangan, `593` positif.
- ExtraTrees validation edge AUC `0.94846`, AP `0.59636`.
- Candidate validation all-round: ExtraTrees, `link=.15`,
  `singleton=.15`, `max_size=4`, `rank=score`:
  physical F1 `0.8232`, count MAE `1.2527`, ±1 `0.6703`, matched class
  accuracy `0.7542`, macro-F1 `0.6014`.
- Artifact: `/workspace/model_artifacts/project-expertise/detection_edge_linker_953_v2/`.

### SawitMVC-Depth-YOLO

- Train adjacent: `12,517` pasangan, `1,245` positif.
- Validation adjacent: `2,504` pasangan, `250` positif.
- ExtraTrees best physical validation grid: F1 `0.8471`, MAE `0.9487`,
  matched class accuracy `0.8359`.
- Artifact: `/workspace/model_artifacts/project-expertise/detection_edge_linker_depth_v1/`.

Angka learned linker di atas adalah validation-only. Candidate 953 `max_size=4`
berasal dari sweep tambahan dan harus direproduksi serta disimpan sebagai JSON
versioned sebelum dipakai pada test.

## Eksperimen yang ditolak

Hue/color, sharpening, CLAHE, gray-world, white balance, gamma/contrast,
photometric TTA, crop 224/256/320, ConvNeXt/Swin variants, class weighting,
focal/ordinal/mixed loss, model-vote WBF, count/reranker, local detector
fine-tuning, group selector, dan agnostic NMS tidak memberi improvement
all-round yang cukup atau turun pada metrik penting. Aglostic NMS quick test
sendiri turun ke mAP50 `0.175860` pada 953 dan `0.482666` pada Depth.

## Resume checklist

1. Reproduce candidate 953 `max_size=4` dari validation dan simpan JSON.
2. Review feature order/checkpoint compatibility; ubah feature berarti retrain.
3. Buat evaluator test terpisah untuk dump
   `fused_combined1716_test_rebuilt`.
4. Jalankan test sekali dengan profile terkunci.
5. Simpan physical/count/class metrics, confusion matrix, matched denominator,
   dan tree-level uncertainty.

Last known repository commit sebelum handoff: `cd6c809`. Model besar dan dump
regenerable tetap berada di artifact storage eksternal.
