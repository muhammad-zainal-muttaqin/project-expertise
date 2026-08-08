# Catatan teknis Fase 1 (untuk kelanjutan otomatis)

## Bug ditemukan & diperbaiki: resolusi path `data_rgb.yaml`

`research-pipeline/reproduce/experiments/config/data_rgb.yaml` punya
`path: ../../../evidence/experiments/splits_rgb` (relatif). Ultralytics 8.4.103
me-resolve `path:` relatif terhadap **CWD proses**, bukan terhadap lokasi file
yaml. Kalau training dijalankan dari `reproduce/experiments/` (cara "wajar"),
tiga `..` melenceng ke `/workspace/evidence/...` (tidak ada), lalu ultralytics
fallback ke `DATASETS_DIR`-relative dan menghasilkan path rusak
`/evidence/experiments/splits_rgb/val.txt` -> crash langsung di awal training.

**Perbaikan (tanpa mengubah file research-pipeline):** jalankan skrip training
dengan **CWD = `research-pipeline/reproduce/experiments/config/`** (folder
tempat `data_rgb.yaml` berada), dan panggil skrip train via path absolut.
Contoh yang benar (dipakai untuk retrain YOLO26l):

```bash
cd /workspace/research-pipeline/reproduce/experiments/config
/workspace/research-pipeline/reproduce/experiments/.venv/bin/python \
  /workspace/research-pipeline/reproduce/experiments/train/train_yolo26l.py \
  --weights /workspace/research-pipeline/reproduce/experiments/yolo26l.pt \
  --imgsz 1280 --epochs 60 --batch 4 --name yolo26l_e60_i1280_v2repro
```

Pola CWD yang sama berlaku untuk `train_rtdetr.py` (pakai `data_rgb.yaml` yang
sama). `train_rfdetr.py` beda jalur (pakai `rfdetr_ds/`, bukan yaml ultralytics)
-- perlu dicek terpisah apakah punya masalah serupa sebelum dijalankan.

## Cache dataset ke disk lokal (2026-08-08) — hasil terukur

Dataset di-cache dari `/workspace` (network mount, moosefs) ke disk lokal
overlay (`/home/claudeuser/data-cache/`): `SawitMVC` (2,4G), `SawitMVC-Depth-YOLO`
(1,6G), `depth_png_v2` (0,2G). Split file lokal + `data_rgb_local.yaml` dibuat
di `/home/claudeuser/data-cache/`; symlink `rfdetr_ds/*/images/*` dialihkan ke
cache lokal. Skrip pemicu training lokal: `scripts/train_ultra_local.py`
(replikasi `train_yolo26l.py`/`train_rtdetr.py`, hanya `--data` yang beda,
tidak mengedit file research-pipeline apa pun).

**Hasil setelah RT-DETR-L direstart dengan cache lokal:** GPU utilization naik
jadi 97% konsisten (sebelumnya osilasi 33-99%), tapi **kecepatan wall-clock per
epoch nyaris sama** (~326 detik vs ~334 detik sebelumnya). Kesimpulan:
hipotesis awal (I/O jaringan sebagai bottleneck utama) **salah** — GPU compute
RTX A4500 sendiri yang jadi batas kecepatan untuk beban kerja ini, bukan
storage. Osilasi utilization yang teramati sebelumnya kemungkinan sampling
sesaat, bukan pola I/O-wait yang konsisten.

**Keputusan:** cache lokal tetap dipertahankan (tidak merugikan, dan akan
berguna di Fase 2/3/5 yang membaca ulang dataset 352-pohon berkali-kali untuk
banyak percobaan pendek — di situ overhead scan/setup per-run yang berulang
baru benar-benar terasa). Tapi untuk mempercepat Fase 1 secara berarti,
upgrade GPU (kandidat: L4, tensor core gen-4, per log asli E-021 mencatat ~1
jam/60 epoch di L4 vs ~4-4,4 jam di A4500) lebih relevan daripada optimasi
storage lebih lanjut.

## Urutan Fase 1 (status берjalan)

1. [running] YOLO26l retrain -> `evidence/experiments/runs/yolo26l_e60_i1280_v2repro/`
2. [belum] RT-DETR-L retrain -> `..._rtdetr_l_e60_i1280_v2repro/`
3. [belum] RF-DETR-L retrain (override config lihat docs/RENCANA.md Fase 1.3)
4. [belum] Eval pycocotools ketiganya vs target E-021 (0,5300/0,5784/0,6038)
5. [belum] Inference + adaptor per-pohon (scripts/adapters/) + counting Baseline-SawitMVC
   (ikuti pola exp_counting_v3.py: fit Ridge segar pada F_all per detektor,
   BUKAN run_e2e_pipeline.py -- lihat docs/SCHEMA-PERTREE.md)
6. [belum] Tulis entri V2-E-001 (validasi reproduksi) dan V2-E-002 (mAP vs counting)
   di experiments/EKSPERIMEN.md
