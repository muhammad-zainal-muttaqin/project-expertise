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

### Pemetaan revisi

| Revisi | Komponen | Dokumen hasil |
|---|---|---|
| **V1/original** | WBF proposal, prior rotasi, linker awal, classifier per tandan, counting/reconciliation | [`PIPELINE_EXPERIMENTS_V3.md`](results/remote_eval_2026-08-27/PIPELINE_EXPERIMENTS_V3.md), [`experiments/EKSPERIMEN.md`](experiments/EKSPERIMEN.md) |
| **V2** | Deep-tail proposal, `p_tp` re-ranker, learned edge linker, GSP MILP, V2 count/class composition | [`GSP_LINKER.md`](results/remote_eval_2026-08-28/GSP_LINKER.md), [`MAP_BOOST.md`](results/remote_eval_2026-08-28/MAP_BOOST.md), [`WAVE2_RECAP.md`](results/remote_eval_2026-08-28/validation_wave/WAVE2_RECAP.md) |
| **RGB+D4 follow-up** | Ablasi modalitas dan fixed late fusion pada `new763` | [`NEW763_RGBD4_RESULTS.md`](docs/NEW763_RGBD4_RESULTS.md) |

V2 dan RGB+D4 follow-up berstatus validation-only untuk konfigurasi terbaru;
hasil test-locked sebelumnya tidak ditimpa.

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

## Validation wave 2026-08-28

Wave lanjutan tetap mematuhi pemisahan TRAIN/VALIDATION/TEST. Kandidat
terpilih untuk 953 adalah opini DINOv2-Large dengan bobot `0,15`, opini
logistic anggota dengan bobot `0,05`, serta bias logit B2 `+0,15`; opini
detektor tetap menjadi jalur *skip*. Pada validation, matched-class accuracy
meningkat dari `0,7542` menjadi `0,7684`, sedangkan macro-F1 meningkat dari
`0,6014` menjadi `0,6164`. Bootstrap berpasangan pada 5.000 resampling
pohon menghasilkan selang delta matched `[+0,0026; +0,0268]` dan selang delta
macro-F1 `[+0,0007; +0,0303]`; keduanya tidak mencakup nilai nol.

Perubahan tersebut belum dijalankan pada TEST dan tidak mengubah topology
fisik maupun kepala pencacahan. OOF stacking, agregasi per sisi, kepala
ordinal, KNN/prototype, attention GPU, selector Hungarian–GSP, serta regresor
pencacahan kaya fitur dicatat sebagai ablasi; tidak ada yang memenuhi kriteria
all-round pada validation. Rincian angka, konfigurasi, skrip reproduksi, dan
checksum tersedia di
[`PERFORMANCE_WAVE_2026-08-28`](results/remote_eval_2026-08-28/PERFORMANCE_WAVE_2026-08-28.md).

Follow-up ConvNeXt/Swin/EfficientNet independen hanya memberi fusion nominal
`0,7697` matched dan `0,6166` macro-F1 pada validation 953—satu pohon lebih
baik dari anchor tanpa CI independen—sehingga tidak dipromosikan. Selector
Depth antara original GSP dan V2 geo/count juga ditolak karena pertukaran
physical F1/macro-F1 versus MAE. Kedua hasil ini tersimpan sebagai ablasi
TRAIN/VAL-only; original Depth GSP dan kandidat 953 robust di atas tetap
menjadi keputusan kerja.

### Cross-layer composition follow-up

Komposisi empat cabang pada VAL menemukan kandidat yang lebih kuat daripada
sekadar V2-only: topology original GSP dipertahankan, tetapi target count
diganti dengan Ridge V2 geo yang dilatih TRAIN, kemudian class head
`scale_macro` diterapkan. Pada Depth VAL, physical F1 berubah
`0,852641 → 0,854225`, MAE `0,931624 → 0,914530`, matched class
`0,845652 → 0,850000`, dan macro-F1 `0,680685 → 0,689013`; ±1 tetap
`0,786325`. Ini adalah kandidat validation-only terbaik sementara dan
menutupi kompromi antar-layer tanpa test tuning.

Paired bootstrap 5.000 pohon tidak memberi CI yang mengecualikan nol (F1
`[-0,006160; +0,009540]`, MAE `[-0,085470; +0,051282]`, matched
`[-0,011744; +0,021558]`, macro `[-0,015425; +0,033033]`), sehingga angka
tersebut belum boleh disebut klaim signifikan. Artefak lengkap ada di
`results/remote_eval_2026-08-28/validation_wave/reports/`.

Head-aware ranking juga diuji sebagai layer berikutnya; ia menaikkan akurasi
kelas tetapi menurunkan physical F1, sehingga tetap menjadi ablasi dan skor
linker tetap menjadi ranking utama.

Fresh composition-aware retraining juga sudah diaudit: member head yang
dilatih ulang pada label komposisi TRAIN tidak mengalahkan head yang sudah
ada (macro-F1 `0,684983` vs `0,689013`, matched sama `0,850000`). Branch ini
disimpan sebagai negative control; tidak menggantikan kandidat utama.

## Validation wave 2 — cross-layer audit (2026-08-28)

Wave kedua menjalankan **2.893 baris evaluasi** pada TRAIN/VAL: Pipeline V2
stage 1+2 (172 per dataset), cross-layer 953 (815 detector + 176 class
frontier), composition-aware head (378), count meta-ensemble (58 untuk 953,
30 untuk Depth), edge ensemble→GSP (1.080), dan GPU group-attention (6 per
dataset). Semua branch memiliki jalur TEST yang ditolak atau tidak tersedia.

Hasil paling penting:

| Dataset / branch | F1 fisik | MAE | ±1 | matched | macro-F1 | Keputusan |
|---|---:|---:|---:|---:|---:|---|
| 953 anchor | 0,823216 | 1,252747 | 0,670330 | 0,754204 | 0,601394 | Referensi |
| 953 robust class calibration | 0,823216 | 1,252747 | 0,670330 | **0,769728** | **0,617081** | Kandidat terbaik; count invariant |
| 953 cross-layer best macro | 0,839396 | 1,527473 | 0,582418 | 0,758667 | **0,631103** | Exploratory; count turun |
| 953 count-meta + calibration | 0,825994 | 1,318681 | 0,626374 | 0,768531 | 0,622456 | Exploratory; ±1 turun |
| Depth anchor original GSP | 0,852641 | 0,931624 | 0,786325 | 0,845652 | 0,680685 | Profil kerja |
| Depth topology + V2 geo count + scale macro | **0,854225** | **0,914530** | 0,786325 | **0,850000** | **0,689013** | Kandidat VAL; CI inconclusive |
| Depth GPU group-attention | 0,852641 | 0,931624 | 0,786325 | 0,845652 | 0,680688 | Flat control |

Cross-layer 953 mengulang persis kandidat robust sebagai best-by-matched dan
tidak menghasilkan all-rounder baru. Skor macro tertinggi berasal dari
topology original GSP + target count V2 geo Ridge + `scale_macro`, tetapi
fragmentasi count menurunkan MAE dan ±1. Pada Depth, kandidat topology/count/
class lebih baik secara point estimate, namun bootstrap 5.000 pohon memberi
CI delta yang seluruhnya melintasi nol: F1 `[-0,006160; +0,009540]`, MAE
`[-0,085470; +0,051282]`, matched `[-0,011744; +0,021558]`, dan macro-F1
`[-0,015425; +0,033034]`.

Bootstrap class calibration 953 untuk kandidat robust memberi matched delta
`+0,015410` dengan CI `[+0,003736; +0,027883]`, serta macro delta `+0,015575`
dengan CI `[+0,001266; +0,030756]`. Anchor gate train/VAL tetap lulus dengan
selisih maksimum kurang dari `5×10⁻⁵`. Rincian tabel, jumlah percobaan,
konfigurasi, dan artefak ada di
[`WAVE2_RECAP.md`](results/remote_eval_2026-08-28/validation_wave/WAVE2_RECAP.md).

Keputusan operasional tidak berubah: jangan menimpa hasil test-locked dengan
kandidat VAL baru. Kandidat macro/matched yang mengorbankan count tetap
disimpan sebagai exploratory, sedangkan Depth topology+count+class menunggu
validasi independen sebelum dipromosikan.
