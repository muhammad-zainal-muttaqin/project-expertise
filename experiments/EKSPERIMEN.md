# Log Eksperimen — Volume 2 (append-only)

Aturan (dari `CLAUDE.md`): satu entri = satu hipotesis falsifiable. Append-only
— entri lama tidak pernah diedit; koreksi ditulis sebagai entri baru yang
mereferensikan entri yang dikoreksi. Hasil negatif dicatat dengan bobot yang
sama dengan hasil positif. Setiap angka harus terlacak ke skrip/JSON/log.

Penomoran: **`V2-E-0xx`**, mulai dari `V2-E-001`. Sengaja terpisah dari
`E-0xx`/`F-0xx` Volume 1 (repo/riwayat berbeda; `E-021` sudah punya arti
spesifik di Volume 1, `V2-E-0xx` menghindari tabrakan referensi silang).

Bukti mentah per entri disimpan di `results/V2-E-0xx/`.

## Template

```markdown
## V2-E-0xx — <satu kalimat hipotesis falsifiable>

**Tanggal:** YYYY-MM-DD
**Hipotesis:** <pernyataan falsifiable, satu saja>
**Dataset & split:** <dataset persis + path split>
**Metode:** <skrip + invokasi CLI persis + identitas bobot/commit>
**Hasil:** <angka apa adanya, tanpa dibungkus>
**Sumber:** <path JSON/CSV/log yang membuktikan tiap angka>
**Verdict:** CONFIRMED | FALSIFIED | INCONCLUSIVE
```

---

<!-- Entri berikutnya ditambahkan di bawah baris ini, tidak pernah menyisip di atas. -->

## V2-E-001 — Reproduksi deteksi E-021 dengan tiga arsitektur pada 953 pohon SawitMVC-YOLO

**Tanggal:** 2026-08-09
**Hipotesis:** Retrain tiga arsitektur (YOLO26l, RT-DETR-L, RF-DETR-L) dengan
konfigurasi identik dengan E-021 Volume 1 menghasilkan test mAP50 dalam ±0,02 dari
angka asli, mengonfirmasi reprodusibilitas.
**Dataset & split:** SawitMVC-YOLO, 953 pohon (716 train / 96 val / 141 test),
resolusi 960×1280, evaluasi `pycocotools`.
**Metode:**
- YOLO26l: `YOLO('yolo26l.pt').train(epochs=60, imgsz=1280, batch=4, seed=42, cos_lr=True, patience=60)`
  — bobot: `models/yolo26l_e60_i1280_v2repro/best.pt`
- RT-DETR-L: `RTDETR('rtdetr-l.pt').train(...)` — config identik
  — bobot: `runs/rtdetr_l_e60_i1280_v2repro/weights/best.pt`
- RF-DETR-L: `RFDETRLarge(resolution=1280, gradient_checkpointing=True).train(epochs=60, batch_size=4, grad_accum_steps=4, seed=42)`
  — bobot: `runs/rfdetr_l_e60_i1280_v2repro/checkpoint_best_ema.pth`
  — Catatan: peak mAP50 di epoch 8, overfitting setelahnya; best EMA checkpoint otomatis tersimpan.
- Evaluasi: `scripts/eval_all_pycoco_v2repro.py`

**Hasil (test split, pycocotools):**

| Model | Params | test mAP50 | Target E-021 | Gap | test mAP50-95 |
|---|---|---|---|---|---|
| YOLO26l | 26,3jt | 0,5435 | 0,5300 | +0,014 | 0,2564 |
| RT-DETR-L | 33,0jt | 0,5781 | 0,5784 | −0,000 | 0,2629 |
| RF-DETR-L | 35,7jt | 0,6012 | 0,6038 | −0,003 | 0,2747 |

Per-kelas AP50 (test):

| Model | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| YOLO26l | 0,7705 | 0,4479 | 0,6050 | 0,3506 |
| RT-DETR-L | 0,7874 | 0,4614 | 0,6371 | 0,4266 |
| RF-DETR-L | 0,8150 | 0,5184 | 0,6553 | 0,4160 |

**Sumber:** `results/perkelas_pycoco_v2repro.json`
**Verdict:** CONFIRMED — ketiga model mereproduksi E-021 dalam ±0,014 mAP50.

---

## V2-E-002 — Counting tiga detektor v2repro pada 953 pohon (Ridge + F_all)

**Tanggal:** 2026-08-09
**Hipotesis:** Mengganti detektor YOLO26m (baseline DiB) dengan tiga arsitektur
yang lebih besar (YOLO26l, RT-DETR-L, RF-DETR-L) masing-masing meningkatkan
Class ±1 Acc counting di atas baseline 77,48%.
**Dataset & split:** SawitMVC-YOLO, 953 pohon (812 train+val / 141 test).
**Metode:**
- Inference conf=0,25 pada seluruh split via `scripts/adapters/{yolo,rtdetr,rfdetr}_to_pertree.py`
- Counting: `scripts/run_counting_v2repro.py` — Ridge + F_all (67 dim), strategy train+val,
  pola identik `exp_counting_v3.py` Baseline-SawitMVC.

**Hasil (test, 141 pohon):**

| Detektor | Class ±1 Acc | Tree ±1 Acc | Macro MAE |
|---|---|---|---|
| YOLO26m (baseline DiB) | 77,48% | 32,62% | 1,036 |
| YOLO26l v2repro | 72,16% | 30,50% | 1,090 |
| RT-DETR-L v2repro | 76,24% | 34,04% | 0,997 |
| RF-DETR-L v2repro | 76,24% | 36,17% | 0,993 |

Per-kelas (±1 Acc / MAE / bias):

| Detektor | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| YOLO26l | 94,3% / 0,43 / −0,07 | 78,7% / 1,01 / −0,15 | 48,2% / 1,70 / −0,18 | 67,4% / 1,22 / +0,10 |
| RT-DETR-L | 96,5% / 0,40 / +0,08 | 81,6% / 0,94 / −0,11 | 58,2% / 1,45 / −0,13 | 68,8% / 1,18 / +0,05 |
| RF-DETR-L | 95,7% / 0,38 / +0,03 | 82,3% / 0,94 / −0,13 | 60,3% / 1,47 / −0,02 | 66,7% / 1,18 / +0,01 |

**Sumber:** `results/counting_v2repro.json`
**Verdict:** FALSIFIED — tidak ada satu pun detektor baru yang melampaui baseline
Class ±1 Acc 77,48%. Namun RF-DETR-L memiliki Tree ±1 Acc terbaik (36,17% vs 32,62%),
Macro MAE terendah (0,993 vs 1,036), dan bias paling seimbang.
**Catatan:** YOLO26l justru lebih buruk dari YOLO26m — kemungkinan karena perbedaan
konfigurasi training (batch=4 vs 32, imgsz=1280 vs 640).
B3 tetap menjadi kelas terlemah di semua detektor (48–60% ±1 Acc).
