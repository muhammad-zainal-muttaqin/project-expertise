# Skema JSON Per-Pohon (kontrak untuk adaptor detektor baru)

Sumber: `/workspace/Baseline-SawitMVC/predictions/y26mv2_per_tree/*.json`
(953 berkas, satu per pohon SawitMVC). Diverifikasi langsung dari isi berkas,
bukan dari dokumentasi.

## Struktur

```json
{
  "tree_name": "DAMIMAS_A21B_0001",
  "split": "test",
  "detector": "y26mv2",
  "images": {
    "side_1": {
      "side_index": 0,
      "annotations": [
        {
          "class_name": "B2",
          "bbox_yolo": [0.6173592209815979, 0.4416269361972809, 0.10798104852437973, 0.08652486652135849],
          "conf": 0.4842214584350586
        }
      ]
    },
    "side_2": { "side_index": 1, "annotations": [ ... ] }
  }
}
```

- `tree_name`: id pohon, cocok dengan `ground_truth/split_manifest.csv`.
- `split`: `train` / `val` / `test`, ikut split resmi 716/96/141.
- `detector`: nama detektor pembuat file (untuk adaptor baru: `yolo26l_v2repro`,
  `rtdetr_l_v2repro`, `rfdetr_l_v2repro`, dst).
- `images`: dict per sisi (`side_1`, `side_2`, ...), key bebas tapi harus
  konsisten dengan urutan `side_index` (0-based).
- `annotations`: list per bbox. `class_name` &isin; {B1,B2,B3,B4}. `bbox_yolo`
  = [cx, cy, w, h] ternormalisasi (format YOLO standar). `conf` = confidence
  deteksi (0-1).

**Adaptor RT-DETR-L / RF-DETR-L wajib menghasilkan struktur ini persis** —
nama field, nesting, dan normalisasi bbox harus sama, supaya
`pipeline/build_counting_features.py` dan `experiments/exp_counting_v3.py`
bisa membacanya tanpa modifikasi.

## Jalur reproduksi angka yang benar (koreksi asumsi rencana awal)

Angka yang dikutip di `docs/REKAP.md` (Ridge + F_all 67-dim: 77,48% / 32,62% /
1,036) **tidak** dihasilkan oleh `scripts/report_metrics.py` atau
`pipeline/run_e2e_pipeline.py`. Kedua skrip itu (jalur "Track B" resmi di
`pipeline/README.md`) hanya memakai fitur 13-dim dan counter {svm, rf, lr,
m01} — tidak ada Ridge di daftar itu.

Angka Ridge+F_all yang dikutip dihasilkan oleh
**`experiments/exp_counting_v3.py`** (80 konfigurasi: 8 feature set × 5 model
× 2 strategi training), yang mengimpor `_load_gt`/`_load_splits`/`CLASSES`
dari `pipeline/build_counting_features.py`. Skrip ini butuh `xgboost` dan
`lightgbm` (tidak ada di `requirements.txt` Baseline-SawitMVC maupun di venv
research-pipeline) — dipasang di venv terpisah
`/workspace/Baseline-SawitMVC/.venv` agar tidak mengubah venv research-pipeline
yang sudah pinned. Sudah diverifikasi: menjalankan
`.venv/bin/python experiments/exp_counting_v3.py` mereproduksi persis
`F_all, Ridge, train, 67dim → macro_acc 0.774823, joint_acc 0.326241,
macro_mae 1.035461` (baris teratas "TOP 15 OVERALL").

**Untuk detektor baru (YOLO26l/RT-DETR-L/RF-DETR-L retrain E-021), jalur
counting Fase 1 yang benar mengikuti pola `exp_counting_v3.py`** (fit Ridge
segar pada fitur F_all dari split train detektor tsb, evaluasi di test),
bukan `run_e2e_pipeline.py`'s Track B. Ini juga konsisten dengan
`run_counting_regularized.py`: defaultnya `pipe.fit(X_tr, y_tr)` — fit segar
per detektor, bukan load model tersimpan — kecuali `--load-model` diberikan
eksplisit.

## Kompatibilitas ground truth SawitMVC vs SawitMVC-Depth (verifikasi Fase 0.5)

Dibandingkan langsung: `/workspace/SawitMVC/data/json/{tree}.json` vs
`/workspace/SawitMVC-Depth-YOLO/{train,valid,test}/linked/{tree}.json`.

**Skemanya identik** — sama-sama punya top-level key `version, tree_id,
tree_name, split, metadata, images, bunches, summary, _confirmedLinks`, dan
struktur `bunches[i]` (`bunch_id, class, class_mismatch, appearance_count,
appearances[]`) serta `summary` (`total_unique_bunches, total_detections,
duplicates_linked, by_class, by_side`) sama persis. **Tidak perlu shim
terjemahan** — `build_counting_features.py`/`exp_counting_v3.py` dari
Baseline-SawitMVC bisa langsung membaca kedua sumber tanpa modifikasi.

Satu-satunya beda: field `split` di JSON SawitMVC-Depth-YOLO bernilai
`"field"` (bukan `train`/`val`/`test`) — pembagian split yang benar untuk
352 pohon ini datang dari **lokasi folder** (`train/`, `valid/`, `test/`) dan
`split_stats.json`, bukan dari field internal ini. Field internal `split` di
JSON SawitMVC (953 pohon) sudah benar berisi `train`/`val`/`test`. Versi
Depth juga punya field tambahan per-sisi (`rgb_sha256`, `capture_origin`,
`depth_required`) yang bersifat aditif, tidak mengganggu kompatibilitas.
