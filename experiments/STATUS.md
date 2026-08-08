# Status Eksperimen

## Fase saat ini: 0 — Persiapan (hampir selesai)

Lihat [docs/RENCANA.md](../docs/RENCANA.md) untuk rencana kerja lengkap dan
[EKSPERIMEN.md](EKSPERIMEN.md) untuk log append-only per hipotesis.

**Temuan penting:** bobot model E-021 (YOLO26l/RT-DETR-L/RF-DETR-L hasil
Volume 1) hilang — hanya bobot pretrained COCO yang tersisa. Fase 1 kini
butuh retrain ketiga detektor dari nol sebelum counting bisa diisi (bukan
inference-saja seperti asumsi awal RENCANA.md). Estimasi total naik dari
~7-10 hari menjadi ~9-14 hari.

### Progres Fase 0

| # | Tugas | Status |
|---|---|---|
| 0.1 | Clone Baseline-SawitMVC | Selesai — `/workspace/Baseline-SawitMVC` |
| 0.2 | Reproduksi angka Baseline-SawitMVC (Ridge+F_all) | Selesai — persis 77,48%/32,62%/1,0355 via `experiments/exp_counting_v3.py` |
| 0.3 | Skema JSON per-pohon didokumentasikan | Selesai — `docs/SCHEMA-PERTREE.md` |
| 0.4 | Split SawitMVC-Depth | Selesai — reuse `/workspace/SawitMVC-Depth-YOLO/split_stats.json` |
| 0.5 | Verifikasi kompatibilitas GT | Selesai — skema identik, tidak perlu shim |
| 0.6 | Adaptor RT-DETR-L/RF-DETR-L → per-pohon | Selesai (smoke test) — `scripts/adapters/` |
| 0.7 | Reprojeksi depth (1.408 file) | Sedang berjalan |
| 0.8 | Freeze Fase 0 | Menyusul setelah 0.7 |

## Matriks target

| Dataset | YOLO26l | RT-DETR-L | RF-DETR-L |
|---|---|---|---|
| RGB 953 pohon | Det: 0,5300 / Count: — | Det: 0,5784 / Count: — | Det: 0,6038 / Count: — |
| RGB 352 pohon | — / — | — / — | — / — |
| RGB+D 352 pohon | — / — | — / — | — / — |

Format sel: `Det: mAP50 / Count: Class ±1 Acc`

Tanda `—` berarti belum diukur.
