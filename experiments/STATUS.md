# Status Eksperimen

## Fase saat ini: 0 — Persiapan (hampir selesai)

Lihat [docs/RENCANA.md](../docs/RENCANA.md) untuk rencana kerja lengkap dan
[EKSPERIMEN.md](EKSPERIMEN.md) untuk log append-only per hipotesis.

**Temuan penting:** bobot model E-021 (YOLO26l/RT-DETR-L/RF-DETR-L hasil
Volume 1) hilang — hanya bobot pretrained COCO yang tersisa. Fase 1 kini
butuh retrain ketiga detektor dari nol sebelum counting bisa diisi (bukan
inference-saja seperti asumsi awal RENCANA.md). Estimasi total naik dari
~7-10 hari menjadi ~9-14 hari.

### Progres Fase 0 (selesai penuh)

| # | Tugas | Status |
|---|---|---|
| 0.1 | Clone Baseline-SawitMVC | Selesai — `/workspace/Baseline-SawitMVC` |
| 0.2 | Reproduksi angka Baseline-SawitMVC (Ridge+F_all) | Selesai — persis 77,48%/32,62%/1,0355 via `experiments/exp_counting_v3.py` |
| 0.3 | Skema JSON per-pohon didokumentasikan | Selesai — `docs/SCHEMA-PERTREE.md` |
| 0.4 | Split SawitMVC-Depth | Selesai — reuse `/workspace/SawitMVC-Depth-YOLO/split_stats.json` |
| 0.5 | Verifikasi kompatibilitas GT | Selesai — skema identik, tidak perlu shim |
| 0.6 | Adaptor RT-DETR-L/RF-DETR-L → per-pohon | Selesai (smoke test) — `scripts/adapters/` |
| 0.7 | Reprojeksi depth (1.408 file) | Selesai — Z 0,8-15,0m, cakupan valid 71% |
| 0.8 | Freeze Fase 0 | Selesai, ter-commit+push |

### Progres Fase 1 (953 pohon, retrain + counting)

Catatan teknis penting: `data_rgb.yaml` di research-pipeline punya bug resolusi
path relatif — perbaikan (tanpa mengubah file research-pipeline) di
`docs/CATATAN-TEKNIS-FASE1.md`.

| Model | Retrain | Eval pycocotools (test) | Target E-021 | Verdict |
|---|---|---|---|---|
| YOLO26l | Selesai (4,4 jam, 60 epoch) | mAP50=0,5435 / mAP50-95=0,2565 | 0,5300 / 0,2568 | Reproduksi OK (+0,0135) |
| RT-DETR-L | Sedang berjalan | — | 0,5784 / 0,2707 | — |
| RF-DETR-L | Belum mulai | — | 0,6038 / 0,2770 | — |

Catatan waktu: retraining di RTX A4500 jauh lebih lambat dari referensi L4
(4,4 jam vs ~1 jam untuk YOLO26l) — diduga karena storage `/workspace`
di-mount lewat jaringan (moosefs), bukan NVMe lokal. Estimasi total direvisi
naik lagi dari perkiraan sebelumnya.

## Matriks target

| Dataset | YOLO26l | RT-DETR-L | RF-DETR-L |
|---|---|---|---|
| RGB 953 pohon | Det: 0,5300 / Count: — | Det: 0,5784 / Count: — | Det: 0,6038 / Count: — |
| RGB 352 pohon | — / — | — / — | — / — |
| RGB+D 352 pohon | — / — | — / — | — / — |

Format sel: `Det: mAP50 / Count: Class ±1 Acc`

Tanda `—` berarti belum diukur.
