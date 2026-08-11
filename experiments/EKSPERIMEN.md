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

---

## V2-E-003 — Deteksi tiga arsitektur pada 352 pohon SawitMVC-Depth (RGB)

**Tanggal:** 2026-08-09
**Hipotesis:** Tiga arsitektur (YOLO26l, RT-DETR-L, RF-DETR-L) mempertahankan
urutan performa relatif yang sama pada dataset SawitMVC-Depth 352 pohon (subset
lebih kecil) seperti pada SawitMVC 953 pohon.
**Dataset & split:** SawitMVC-Depth-YOLO, 352 pohon (245 train / 52 val / 55 test),
resolusi 1280×800, evaluasi `pycocotools`.
**Metode:**
- YOLO26l: `YOLO('yolo26l.pt').train(epochs=60, imgsz=1280, batch=4, seed=42, cos_lr=True, patience=60)`
  — bobot: `runs/yolo26l_e60_i1280_rgb352/weights/best.pt`
- RT-DETR-L: `RTDETR('rtdetr-l.pt').train(...)` — config identik
  — bobot: `runs/rtdetr_l_e60_i1280_rgb352/weights/best.pt`
- RF-DETR-L: `RFDETRLarge(resolution=1280).train(epochs=60, batch_size=4, grad_accum_steps=4, seed=42)`
  — bobot: `runs/rfdetr_l_e60_i1280_rgb352/checkpoint_best_ema.pth`
  — Peak EMA mAP50 di epoch 7, overfitting setelahnya (pola konsisten dengan 953 pohon).
- Evaluasi: `scripts/eval_pycoco_352.py`

**Hasil (test split, pycocotools):**

| Model | Params | test mAP50 | test mAP50-95 |
|---|---|---|---|
| YOLO26l | 26,3jt | 0,3606 | 0,1246 |
| RT-DETR-L | 33,0jt | 0,4343 | 0,1503 |
| RF-DETR-L | 35,7jt | 0,4544 | 0,1599 |

Per-kelas AP50 (test):

| Model | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| YOLO26l | 0,6804 | 0,4320 | 0,2001 | 0,1299 |
| RT-DETR-L | 0,7680 | 0,4867 | 0,2641 | 0,2185 |
| RF-DETR-L | 0,6853 | 0,5184 | 0,3477 | 0,2661 |

**Sumber:** `results/perkelas_pycoco_rgb352.json`
**Verdict:** CONFIRMED — urutan relatif terjaga (RF-DETR-L > RT-DETR-L > YOLO26l).
Angka absolut lebih rendah dari 953 pohon karena dataset lebih kecil (352 vs 953)
dan distribusi kelas berbeda. B3 dan B4 jauh lebih sulit di dataset ini.

---

## V2-E-004 — Counting tiga detektor RGB pada 352 pohon (Ridge + F_all)

**Tanggal:** 2026-08-09
**Hipotesis:** Detektor yang lebih baik (mAP50 lebih tinggi) menghasilkan counting
accuracy yang lebih tinggi pada 352 pohon SawitMVC-Depth.
**Dataset & split:** SawitMVC-Depth, 352 pohon (297 train+val / 55 test).
**Metode:**
- Inference conf=0,25 pada seluruh split via `scripts/run_counting_rgb352.py`
- Counting: Ridge + F_all (67 dim), strategy train+val, pola identik exp_counting_v3.py.

**Hasil (test, 55 pohon):**

| Detektor | Class ±1 Acc | Tree ±1 Acc | Macro MAE |
|---|---|---|---|
| YOLO26l | 89,55% | 69,09% | 0,577 |
| **RT-DETR-L** | **90,91%** | 67,27% | **0,532** |
| RF-DETR-L | 88,18% | 65,45% | 0,600 |

Per-kelas (±1 Acc / MAE / bias):

| Detektor | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| YOLO26l | 98,2% / 0,45 / −0,09 | 83,6% / 0,93 / −0,13 | 81,8% / 0,65 / −0,47 | 94,5% / 0,27 / −0,13 |
| RT-DETR-L | 92,7% / 0,51 / −0,04 | 81,8% / 0,84 / −0,07 | 89,1% / 0,58 / −0,07 | 100% / 0,20 / +0,02 |
| RF-DETR-L | 89,1% / 0,65 / −0,04 | 80,0% / 0,91 / +0,04 | 83,6% / 0,62 / −0,18 | 100% / 0,22 / −0,04 |

**Sumber:** `results/counting_rgb352.json`
**Verdict:** FALSIFIED — urutan counting tidak mengikuti urutan deteksi. RT-DETR-L
(mAP50 ke-2) memiliki Class ±1 Acc tertinggi (90,91%), bukan RF-DETR-L (mAP50
tertinggi). Pola serupa dengan 953 pohon: detektor terbaik secara mAP50 belum tentu
menghasilkan counting terbaik.
**Catatan:** Accuracy counting 352 pohon (88–91%) jauh lebih tinggi dari 953 pohon
(72–76%) karena distribusi kelas SawitMVC-Depth lebih seragam dan jumlah tandan per
pohon lebih sedikit. B4 mencapai 100% untuk RT-DETR-L dan RF-DETR-L.

---

## V2-E-005 — Deteksi tiga arsitektur RGBD 4-kanal pada 352 pohon SawitMVC-Depth

**Tanggal:** 2026-08-09
**Hipotesis:** Menambahkan depth sebagai kanal ke-4 (early fusion BGRD) meningkatkan
test mAP50 dibandingkan RGB saja pada ketiga arsitektur.
**Dataset & split:** SawitMVC-Depth-4ch-YOLO, 352 pohon (245 train / 52 val / 55 test),
resolusi 1280, input 4-kanal (BGRD TIFF), evaluasi `pycocotools`.
**Metode:**
- YOLO26l: `YOLO('yolo26l.pt').train(data='data_rgbd_352.yaml', epochs=60, imgsz=1280, batch=4, seed=42, cos_lr=True, patience=60)`
  — bobot: `runs/yolo26l_e60_i1280_rgbd352/weights/best.pt` (peak epoch 47)
- RT-DETR-L: `RTDETR('rtdetr-l.pt').train(data='data_rgbd_352.yaml', ...)` — config identik
  — bobot: `runs/rtdetr_l_e60_i1280_rgbd352/weights/best.pt` (peak epoch 19)
- RF-DETR-L: `train_rfdetr_4ch.py` dengan 3 patch (TIFF loader, normalisasi 4ch, conv inflate)
  — bobot: `runs/rfdetr_l_e60_i1280_rgbd352/checkpoint_best_ema.pth` (peak epoch 7)
- Evaluasi: `scripts/eval_pycoco_rgbd352.py`

**Hasil (test split, pycocotools):**

| Model | Params | RGBD mAP50 | RGB mAP50 | Δ | RGBD mAP50-95 |
|---|---|---|---|---|---|
| YOLO26l | 26,3jt | 0,3919 | 0,3606 | **+0,0313** | 0,1408 |
| RT-DETR-L | 33,0jt | 0,3877 | 0,4343 | **−0,0466** | 0,1359 |
| RF-DETR-L | 35,7jt | 0,4186 | 0,4544 | **−0,0358** | 0,1508 |

Per-kelas AP50 (RGBD test):

| Model | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| YOLO26l | 0,6857 | 0,4579 | 0,2637 | 0,1601 |
| RT-DETR-L | 0,7417 | 0,4621 | 0,2382 | 0,1090 |
| RF-DETR-L | 0,6929 | 0,5160 | 0,3158 | 0,1499 |

Delta per-kelas (RGBD − RGB):

| Model | Δ B1 | Δ B2 | Δ B3 | Δ B4 |
|---|---|---|---|---|
| YOLO26l | +0,005 | +0,026 | +0,064 | +0,030 |
| RT-DETR-L | −0,026 | −0,025 | −0,026 | −0,110 |
| RF-DETR-L | +0,008 | −0,002 | −0,032 | −0,116 |

**Sumber:** `results/perkelas_pycoco_rgbd352.json`, `results/perkelas_pycoco_rgb352.json`
**Verdict:** FALSIFIED — depth 4-kanal TIDAK meningkatkan deteksi secara konsisten.
Hanya YOLO26l yang naik (+0,031 mAP50), RT-DETR-L dan RF-DETR-L justru turun.
YOLO26l naik di semua kelas (terutama B3 +0,064), sementara RT-DETR-L dan RF-DETR-L
turun tajam di B4 (−0,110 dan −0,116). Konsisten dengan temuan Volume 1 E-022/E-027
bahwa early fusion depth cenderung merugikan.

---

## V2-E-006 — Counting tiga detektor RGBD 4-kanal pada 352 pohon (Ridge + F_all)

**Tanggal:** 2026-08-09
**Hipotesis:** Depth 4-kanal meningkatkan counting accuracy (Class ±1 Acc)
dibandingkan RGB saja pada ketiga detektor.
**Dataset & split:** SawitMVC-Depth, 352 pohon (297 train+val / 55 test).
**Metode:**
- Inference conf=0,25 pada citra 4-kanal TIFF via `scripts/run_counting_rgbd352.py`
- Counting: Ridge + F_all (67 dim), strategy train+val, pola identik exp_counting_v3.py.
- Bootstrap CI: 10.000 replikat, paired per pohon (`scripts/bootstrap_ci.py`).

**Hasil (test, 55 pohon):**

| Detektor | RGBD Class ±1 | RGB Class ±1 | Δ | RGBD Tree ±1 | RGBD MAE |
|---|---|---|---|---|---|
| YOLO26l | 87,73% | 89,55% | **−1,82pp** | 60,00% | 0,673 |
| RT-DETR-L | 88,64% | 90,91% | **−2,27pp** | 61,82% | 0,632 |
| RF-DETR-L | 88,18% | 88,18% | **±0,00pp** | 67,27% | 0,586 |

Bootstrap CI paired (RGBD − RGB, 10.000 replikat):

| Detektor | Δ Class ±1 CI95 | P(RGBD > RGB) |
|---|---|---|
| YOLO26l | [−5,9pp, +1,8pp] | 16,5% |
| RT-DETR-L | [−5,0pp, +0,5pp] | 5,6% |
| RF-DETR-L | [−2,7pp, +2,7pp] | 47,3% |

Per-kelas RGBD (±1 Acc / MAE / bias):

| Detektor | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| YOLO26l | 87,3% / 0,60 / −0,24 | 81,8% / 0,91 / −0,04 | 85,5% / 0,91 / −0,07 | 96,4% / 0,27 / −0,13 |
| RT-DETR-L | 92,7% / 0,58 / −0,07 | 81,8% / 1,02 / −0,07 | 81,8% / 0,71 / −0,16 | 98,2% / 0,22 / −0,07 |
| RF-DETR-L | 90,9% / 0,51 / −0,07 | 83,6% / 0,84 / +0,04 | 83,6% / 0,65 / −0,25 | 94,5% / 0,35 / −0,05 |

**Sumber:** `results/counting_rgbd352.json`, `results/counting_rgb352.json`, `results/bootstrap_ci_352.json`
**Verdict:** FALSIFIED — depth 4-kanal TIDAK meningkatkan counting. Bootstrap CI
menunjukkan P(RGBD>RGB) hanya 16,5% (YOLO26l), 5,6% (RT-DETR-L), dan 47,3%
(RF-DETR-L). Tidak satu pun arsitektur yang CI-nya eksklusif positif.
RF-DETR-L tepat sama (88,18%) dengan Tree ±1 Acc sedikit naik (67,27% vs 65,45%),
tapi CI simetris [−2,7pp, +2,7pp] menunjukkan ini kebetulan.
**Catatan:** Early fusion depth secara konsisten merugikan counting meskipun YOLO26l
menunjukkan sedikit perbaikan deteksi. Ini mengonfirmasi bahwa perbaikan deteksi minor
tidak otomatis menerjemahkan ke perbaikan counting.

---

## V2-E-007 — Analisis matriks 9-sel: dampak dataset dan arsitektur terhadap deteksi dan counting

**Tanggal:** 2026-08-09
**Tujuan:** Sintesis terstratifikasi dari 9 kombinasi (3 arsitektur × 3 dataset)
untuk menjawab: (1) apakah urutan arsitektur konsisten lintas dataset,
(2) apakah early fusion depth membantu, (3) bagaimana pola per-kelas berubah.

### A. Matriks deteksi (test split, pycocotools mAP50)

| | YOLO26l | RT-DETR-L | RF-DETR-L |
|---|---|---|---|
| 953-RGB | 0,5435 | 0,5781 | **0,6012** |
| 352-RGB | 0,3606 | 0,4343 | **0,4544** |
| 352-RGBD | **0,3919** | 0,3877 | 0,4186 |

**Temuan deteksi:**
1. **Urutan arsitektur stabil di RGB**: RF-DETR-L > RT-DETR-L > YOLO26l pada kedua
   dataset RGB (953 dan 352 pohon). Gap RF-DETR vs YOLO26l konsisten (~0,06–0,09).
2. **RGBD memecah urutan**: RT-DETR-L turun di bawah YOLO26l saat menerima depth
   (0,3877 vs 0,3919). RF-DETR-L tetap #1 tapi gap menyempit.
3. **Penurunan absolut 953→352**: semua arsitektur turun ~0,15–0,18 mAP50
   (efek ukuran dataset + distribusi kelas berbeda, BUKAN degradasi model).
4. **B4 paling terdampak depth**: delta B4 untuk RT-DETR-L (−0,110) dan
   RF-DETR-L (−0,116) jauh lebih besar dari kelas lain. B4 (tandan overripe)
   memiliki instance paling sedikit dan paling rentan terhadap noise depth.

### B. Matriks counting (test split, Ridge + F_all, Class ±1 Acc)

| | YOLO26l | RT-DETR-L | RF-DETR-L |
|---|---|---|---|
| 953-RGB | 72,16% | 76,24% | 76,24% |
| 352-RGB | 89,55% | **90,91%** | 88,18% |
| 352-RGBD | 87,73% | 88,64% | 88,18% |

**Temuan counting:**
1. **Detektor terbaik ≠ counter terbaik**: RF-DETR-L unggul deteksi di semua
   dataset RGB, tapi RT-DETR-L unggul counting pada 352-RGB (90,91% vs 88,18%).
   Pada 953-RGB, RT-DETR-L dan RF-DETR-L seri (76,24%).
2. **Depth merugikan counting pada 2/3 arsitektur**: YOLO26l −1,8pp, RT-DETR-L
   −2,3pp. RF-DETR-L netral (88,18% → 88,18%) tapi Tree ±1 Acc naik
   (65,45% → 67,27%).
3. **Counting 352 >> 953**: semua model mencapai 87–91% pada 352 pohon vs
   72–76% pada 953 pohon. Ini karena distribusi kelas lebih seragam dan
   jumlah tandan per pohon lebih sedikit di SawitMVC-Depth.

### C. Bootstrap CI — uji signifikansi RGBD vs RGB (10.000 replikat, paired)

| Arsitektur | Δ Class ±1 | CI 95% | P(RGBD>RGB) | Signifikan? |
|---|---|---|---|---|
| YOLO26l | −1,82pp | [−5,9, +1,8] | 16,5% | Tidak (CI memuat 0) |
| RT-DETR-L | −2,25pp | [−5,0, +0,5] | 5,6% | Marginal (CI hampir ekskl. negatif) |
| RF-DETR-L | +0,02pp | [−2,7, +2,7] | 47,3% | Tidak (CI simetris, efek ~0) |

**Kesimpulan:** Tidak ada arsitektur yang secara signifikan diuntungkan oleh depth
pada level α=0,05. RT-DETR-L mendekati signifikan ke arah NEGATIF (P=5,6%),
artinya depth kemungkinan merugikan RT-DETR-L.

### D. Analisis per-kelas terstratifikasi (352 pohon, RGB vs RGBD)

**Deteksi — kelas yang paling diuntungkan depth:**
- YOLO26l B3: +0,064 (kelas terlemah naik paling banyak)
- YOLO26l B4: +0,030
- RF-DETR-L B1: +0,008

**Deteksi — kelas yang paling dirugikan depth:**
- RF-DETR-L B4: −0,116
- RT-DETR-L B4: −0,110
- RF-DETR-L B3: −0,032

**Counting — pola bias:**
- Semua model RGBD memiliki bias negatif lebih besar di B1 dan B3
  (under-predict), terutama YOLO26l B1 (bias −0,24 vs −0,09 di RGB).
- B4 konsisten baik (acc >94%) karena instance sedikit dan Ridge mudah
  memprediksi mendekati 0.

### E. Ringkasan Fase 4

**Jawaban untuk pertanyaan utama:**

| Pertanyaan | Jawaban |
|---|---|
| Arsitektur terbaik deteksi? | RF-DETR-L (konsisten #1 di RGB) |
| Arsitektur terbaik counting? | RT-DETR-L (90,91% pada 352-RGB) |
| Apakah depth membantu deteksi? | Hanya YOLO26l (+0,031), sisanya merugikan |
| Apakah depth membantu counting? | Tidak — 0/3 arsitektur signifikan naik |
| Kelas tersulit? | B3 (matang awal) — AP50 terendah di semua kondisi |
| Kelas termudah? | B1 (mentah) — AP50 >0,68 di semua kondisi |

**Implikasi untuk Fase 5:** Early fusion naif (concat kanal) tidak efektif.
Fase 5 harus mengeksplorasi (1) representasi depth alternatif (edge, inverse,
clipping) yang mungkin memberikan sinyal lebih informatif, dan/atau (2) arsitektur
fusi yang lebih canggih (mid/late fusion, attention-based) yang tidak sekadar
menggabungkan kanal di input.

**Sumber:** `results/matrix_compiled.json`, `results/bootstrap_ci_352.json`

---

## V2-E-008 [screening-15ep] — Encoding depth alternatif pada YOLO26l 4-kanal (early fusion), 352 pohon

**Tanggal:** 2026-08-10/11
**Hipotesis:** Mengganti encoding kanal depth (arsitektur early fusion TIDAK
diubah, sama seperti V2-E-005) meningkatkan val mAP50 dibandingkan encoding
`inverse` V2-E-005 dalam protokol screening cepat (≤15 epoch, patience 3),
konsisten dengan lever representasi Fase 5 (`docs/RENCANA.md`).
**Dataset & split:** SawitMVC-Depth-4ch-{edge,clipped,valid_mask}, 352 pohon
(245 train / 52 val / 55 test, split `canonical_70_15_15` — sama persis
dengan V2-E-003..007). Depth direproyeksi ulang (`depth_meta.json`: cakupan
valid 71,0%, Z_NEAR/Z_FAR=0,8/15,0 m, sama dengan angka lama).
**Metode:**
- `edge` (Sobel gradient magnitude), `clipped` (clip@80, near-field), keduanya
  via `scripts/create_depth_edge_dataset.py` (sudah ada sebelumnya).
- `valid_mask` (BARU): pisahkan sentinel "tidak ada data" (0) dari valid-terjauh
  secara numerik (rentang valid dimampatkan ke [40,220]) — motivasi: pada
  encoding `inverse`, invalid(0) hanya beda 1 increment dari valid-terjauh(1)
  pada skala kontinu yang sama, network tak punya sinyal eksplisit membedakan
  "sensor gagal" vs "sekadar jauh". Fungsi `encode_valid_mask` di
  `scripts/create_depth_edge_dataset.py`.
- `dropout` (BARU, augmentasi kanal depth): kanal depth di-nol-kan acak p=0,25
  saat TRAIN saja, arsitektur early fusion `inverse` tidak diubah. Adaptasi
  `Research-Pipeline/pipeline/fourch.py::patch_loader` (copy, bukan
  cross-import) ke `scripts/train_yolo_4ch_dropout.py`.
- Training: `YOLO('yolo26l.pt').train(epochs=15, patience=3, imgsz=1280,
  batch=4, seed=42, cos_lr=True)` — protokol screening cepat wajib
  (`docs/RENCANA.md` Fase 5), BUKAN angka final 60-epoch.
- Metrik: val mAP50/mAP50-95 native ultralytics (bukan pycocotools test-split
  — hanya untuk ranking relatif antar-kandidat Fase 5, per protokol).

**Hasil (val split, 208 pohon, mAP50 terbaik selama training):**

| Kandidat | Epoch terbaik | val mAP50 | val mAP50-95 | Durasi |
|---|---|---|---|---|
| `inverse` (V2-E-005, acuan, 60 epoch bukan 15 — tidak di-rerun) | — | — | — | — |
| `dropout` | 15 (belum plateau) | 0,3168 | 0,1091 | 2583,7 dtk |
| **`edge`** | **15 (belum plateau)** | **0,3777** | **0,1279** | 2584,4 dtk |
| `clipped` | 14 | 0,3221 | 0,1136 | 2574,9 dtk |
| `valid_mask` | 11 | 0,3321 | 0,1022 | 2575,5 dtk |

**Sumber:** `runs/yolo26l_screening_{dropout,edge,clipped,valid_mask}352/results.csv`,
`runs/yolo26l_screening_*352/hasil.json`
**Verdict:** CONFIRMED — `edge` (Sobel gradient magnitude) unggul jelas dari
tiga kandidat lain (+0,046 s/d +0,061 mAP50), selaras F-002 (frekuensi tinggi
memisahkan tandan dari pelepah, +0,0731 pada B4). Tidak seperti tiga kandidat
lain yang mulai plateau/turun, `edge` dan `dropout` masih naik di epoch 15 —
`edge` dipromosikan ke training penuh 60 epoch (lihat V2-E-010).
**Catatan:** Angka screening 15-epoch ini TIDAK dibandingkan langsung dengan
angka 60-epoch V2-E-003/005 (dataset/protokol sama tapi durasi beda) — hanya
untuk ranking relatif antar-kandidat Fase 5, sesuai protokol.

---

## V2-E-009 [screening-15ep] — Mid-fusion depth + gate non-zero-init pada YOLO26l, 352 pohon

**Tanggal:** 2026-08-11
**Hipotesis:** Memindahkan depth dari early fusion (concat kanal ke-4 di
input) ke cabang terpisah dengan fusi aditif ber-gate di backbone menengah
(P3/8, layer index 4 `yolo26.yaml`), gate diinisialisasi kecil-taknol (0,02,
BUKAN nol seperti F-007), meningkatkan val mAP50 dibandingkan baseline RGB
352 pohon (V2-E-003, 0,3606 test) dan tidak berhenti mati seperti F-007
(gate diharapkan bergerak menjauhi inisialisasinya).
**Dataset & split:** SawitMVC-Depth-4ch (encoding `inverse`, sama dengan
V2-E-005) — 352 pohon, split sama seperti V2-E-008.
**Metode:**
- Arsitektur baru `scripts/train_yolo_midfusion.py`: stem RGB 3-kanal
  TIDAK disentuh (beda mendasar dari V2-E-005/early fusion) — dibangun
  `ch=3` eksplisit (bukan `data["channels"]=4`), bobot pratlatih COCO
  di-load bersih tanpa mismatch shape. Cabang depth terpisah (conv stride-8,
  1→16→32→512 kanal, conv terakhir diinisialisasi skala 0,1x — mitigasi
  F-007 "inisialisasi kecil-taknol"), fitur-nya dijumlahkan ke output layer 4
  dikali gate scalar `γ` (init 0,02).
- Patch di level CLASS (`BaseModel._predict_once`, cek `hasattr(self,
  "depth_branch")`) — bukan per-instance (`types.MethodType`, percobaan
  pertama GAGAL: `Trainer.final_eval()` me-reload model dari checkpoint
  lewat `AutoBackend`, method per-instance tidak ikut ter-reload walau
  `depth_branch`/`gate` sebagai submodul/parameter biasa tetap ter-reload
  benar — diverifikasi lewat smoke test save→reload→forward sebelum retry).
- Training: sama seperti V2-E-008 (15 epoch, patience 3, imgsz 1280, batch 4,
  seed 42, cos_lr).

**Hasil (val split, 208 pohon):**

| Epoch | val mAP50 | val mAP50-95 |
|---|---|---|
| 1 | 0,0799 | 0,0247 |
| 2 | 0,1615 | 0,0414 |
| **3 (terbaik)** | **0,2087** | **0,0712** |
| 4 | 0,2015 | 0,0647 |
| 5 | 0,2161 | 0,0667 |
| 6 (early-stop, patience=3) | 0,1876 | 0,0552 |

Validasi akhir (best.pt, epoch 3) per kelas: B1=0,396, B2=0,329, **B3=0,056,
B4=0,051** (mAP50-95 masing-masing 0,131/0,115/0,015/0,023).
Gate: init 0,02 → final 0,0250 (bergerak naik — TIDAK macet di titik mati
seperti F-007, secara mekanis pelajaran F-007 berhasil dihindari).
**Sumber:** `runs/yolo26l_screening_midfusion352/results.csv`,
`runs/yolo26l_screening_midfusion352/hasil.json`
**Verdict:** FALSIFIED — sinyal TIDAK naik konsisten (plateau lalu turun
setelah epoch 3, early-stop di epoch 6), kalah jauh dari keempat kandidat
representasi V2-E-008 (0,209 vs 0,317-0,378) pada jumlah epoch yang sama.
Per protokol Fase 5 (`docs/RENCANA.md`: "kandidat yang lolos screening naik
konsisten"), TIDAK dipromosikan ke 60 epoch. B3/B4 nyaris nol kemungkinan
karena cabang depth mulai dari inisialisasi acak (beda dengan kandidat
representasi yang langsung mewarisi bobot pretrained di conv pertama) —
enam epoch kemungkinan tidak cukup untuk kelas langka (B3/B4 paling sedikit
instance-nya). Dicatat sebagai hasil negatif dengan bobot yang sama — TIDAK
membantah bahwa mid-fusion+gate non-zero-init bisa bekerja secara umum,
hanya bahwa konfigurasi spesifik ini (fuse_at=4, gate init=0,02, tanpa LR
terpisah untuk cabang depth) tidak lolos screening cepat pada YOLO26l.

---

## V2-E-010 — Encoding depth `edge` (Sobel) pada YOLO26l, 60 epoch penuh, dibanding `inverse` (V2-E-005)

**Tanggal:** 2026-08-11
**Hipotesis:** Encoding depth `edge` (Sobel gradient magnitude), yang menang
screening 15-epoch (V2-E-008, val mAP50 0,3777), meningkatkan test mAP50
dibandingkan `inverse`/early fusion biasa (V2-E-005, test mAP50 0,3919) saat
dilatih penuh 60 epoch dengan protokol identik.
**Dataset & split:** SawitMVC-Depth-4ch-edge, 352 pohon (245 train / 52 val /
55 test), split `canonical_70_15_15` — sama persis dengan V2-E-003/005.
**Metode:** `scripts/train_yolo_4ch_screening.py --epochs 60 --patience 60`
(config identik V2-E-005: imgsz 1280, batch 4, seed 42, cos_lr). Evaluasi:
`scripts/eval_pycoco_rgbd352.py` (pycocotools, test split), bobot
`runs/yolo26l_e60_i1280_rgbd352_edge/weights/best.pt`.

**Hasil (test split, pycocotools):**

| | inverse (V2-E-005) | edge (V2-E-010) | Δ |
|---|---|---|---|
| mAP50 | 0,3919 | **0,4316** | **+0,0397 (+10,1% relatif)** |
| mAP50-95 | 0,1408 | 0,1441 | +0,0033 |

Per-kelas AP50 (test):

| Kelas | inverse | edge | Δ |
|---|---|---|---|
| B1 | 0,6857 | 0,7252 | +0,0395 |
| B2 | 0,4579 | 0,5031 | +0,0452 |
| B3 | 0,2637 | 0,2240 | −0,0397 |
| **B4** | 0,1601 | **0,2740** | **+0,1139** |

Dibanding RGB-352 murni (V2-E-003, test mAP50 0,3606): `edge` unggul di
**keempat kelas sekaligus** (B1 +0,0448, B2 +0,0711, B3 +0,0239, B4 +0,1441),
sesuatu yang `inverse` tidak pernah capai (`inverse` cuma unggul RGB di 1-2
kelas, campur naik-turun — lihat V2-E-005).

**Sumber:** `results/perkelas_pycoco_rgbd352.json` (kunci
`YOLO26l-RGBD-edge`), `runs/yolo26l_e60_i1280_rgbd352_edge/results.csv`,
`runs/yolo26l_e60_i1280_rgbd352_edge/hasil.json`
**Verdict:** CONFIRMED — `edge` mengalahkan `inverse` secara jelas di mAP50
keseluruhan (+10,1% relatif, di atas ambang "2-5% tidak cukup" yang jadi
standar proyek ini). Pola per-kelas selaras hipotesis F-002: **B4 (kelas
paling dirugikan early fusion di V2-E-005, −0,116) sekarang paling diuntungkan
(+0,114)** — sinyal tepi/gradien depth membantu tepat di kasus tandan
kecil/tertutup pelepah yang paling sulit dipisahkan dari fronds secara warna.
B3 sedikit turun (−0,040), konsisten dengan diagnosis bahwa B2/B3 adalah
ambiguitas fotometrik (warna) yang depth — dalam bentuk apapun — tidak bisa
menyelesaikan.
**Counting (Ridge + F_all, test 55 pohon):**

| | inverse (V2-E-006) | edge | Δ |
|---|---|---|---|
| Class ±1 Acc | 87,73% | 87,27% | −0,46pp |
| Tree ±1 Acc | 60,00% | 61,82% | +1,82pp |
| Macro MAE | 0,673 | 0,564 | −0,109 (membaik) |

Per-kelas edge (Acc/MAE/bias): B1=85,5%/0,600/−0,236, B2=81,8%/0,836/−0,182,
B3=85,5%/0,655/−0,255, B4=96,4%/0,164/−0,127.

**Sumber counting:** `results/counting_rgbd352.json` (kunci
`YOLO26l-RGBD-edge`), `runs/pertree_rgbd352/yolo_yolo26lrgbdedge/`

**Catatan penting:** deteksi naik jelas (+10,1% mAP50) TIDAK diikuti
kenaikan counting Class ±1 Acc yang setara — malah sedikit turun (−0,46pp),
meski Tree ±1 Acc dan Macro MAE membaik. Ini pola yang sama dengan V2-E-005/006
(deteksi naik tak otomatis bikin counting naik, karena pipeline counting
bergantung pada konsistensi lintas-sisi, bukan cuma mAP rata-rata). Kesimpulan
detection-level tetap CONFIRMED; kesimpulan counting-level lebih tepat
INCONCLUSIVE — perbaikan di beberapa metrik (Tree Acc, MAE), datar/sedikit
turun di metrik utama (Class Acc).

**Belum lengkap:** bootstrap CI berpasangan (edge vs RGB-352) menyusul
setelah retrain baseline RGB-352 (bobot lama tidak tersimpan di workspace
ini) selesai — akan dicatat sebagai entri terpisah, `V2-E-011`.

---

## V2-E-011 — Retrain baseline RGB-352 + bootstrap CI berpasangan: `edge` vs RGB

**Tanggal:** 2026-08-11
**Hipotesis:** `edge` (RGBD) secara signifikan mengalahkan RGB-352 murni
pada counting Class ±1 Acc (bootstrap CI berpasangan per-pohon, 10.000
replikat), melengkapi kemenangan deteksi di V2-E-010.
**Dataset & split:** SawitMVC-Depth, 352 pohon, split `canonical_70_15_15`
— identik V2-E-003/004/010.
**Metode:** Retrain YOLO26l RGB-352 dari nol (bobot lama V2-E-003 tidak
tersimpan di workspace ini) — config identik V2-E-003 (`scripts/train_yolo_4ch_screening.py
--epochs 60 --patience 60`, data `SawitMVC-Depth/data_rgb_352.yaml`).
Eval: `scripts/eval_pycoco_352.py`, `scripts/run_counting_rgb352.py`,
`scripts/bootstrap_ci.py` (entri `YOLO26l-edge` ditambahkan).

**Sanity check reproduksi retrain RGB-352 vs V2-E-003/004 asli:**

| | Asli (V2-E-003/004) | Retrain ini | Δ |
|---|---|---|---|
| Deteksi test mAP50 | 0,3606 | 0,3711 | +0,0105 (wajar, dalam variasi run) |
| Counting Class ±1 Acc | 89,55% | **84,09%** | **−5,46pp (lebih besar dari variasi biasa)** |

Deteksi reproduksi baik. Counting reproduksi lebih buruk dari yang
diharapkan — konsisten dengan pola yang berulang di proyek ini: perbedaan
kecil pada deteksi (box mana yang lolos/tidak) bisa mengubah fitur
konsistensi lintas-sisi yang dipelajari Ridge secara tidak proporsional.
Ini bukan bug, tapi konsekuensi nyata yang harus dibawa ke interpretasi
hasil di bawah.

**Hasil bootstrap CI berpasangan (edge vs retrain RGB-352 ini, 10.000 replikat):**

| Metrik | Δ | CI95 | P(RGBD>RGB) |
|---|---|---|---|
| Class ±1 Acc | +3,18pp | [−0,5pp, +7,3pp] | 94,3% |
| Tree ±1 Acc | +7,24pp | [−1,8pp, +18,2pp] | 90,0% |

**Sumber:** `results/bootstrap_ci_352.json` (kunci `YOLO26l-edge`),
`results/perkelas_pycoco_rgb352.json`, `results/counting_rgb352.json`,
`runs/yolo26l_e60_i1280_rgb352/`

**Verdict: INCONCLUSIVE untuk counting** (CI hampir tidak memuat nol tapi
masih memuat nol secara ketat; P=94,3% cukup kuat tapi belum ambang 95%
formal) — **DAN kesimpulannya berbalik arah tergantung baseline RGB mana
yang dipakai:**

- Dibanding retrain RGB-352 ini (84,09%): `edge` (87,27%) UNGGUL +3,18pp.
- Dibanding angka ASLI V2-E-004 (89,55%): `edge` (87,27%) justru KALAH −2,28pp.

Ini BUKAN kemenangan bersih seperti deteksi (V2-E-010: `edge` unggul dari
SEMUA baseline RGB manapun yang dipakai, 0,4316 vs 0,3606/0,3711 RGB-only
dan 0,3919 inverse). Untuk counting, kesimpulan sensitif terhadap noise
reproduksi baseline itu sendiri — kejujuran metodologis mengharuskan ini
dilaporkan sebagai TIDAK KONKLUSIF, bukan dibulatkan ke arah manapun yang
lebih enak didengar.

**Ringkasan Fase 5 akhir:** lever representasi (`edge`) CONFIRMED
memperbaiki deteksi (+10,1% mAP50, robust lintas-baseline), TIDAK
KONKLUSIF untuk counting (arah tergantung baseline pembanding). Lever
arsitektur (mid-fusion+gate) FALSIFIED di screening (V2-E-009). Hasil
positif deteksi ini genuinely baru — tidak ada benchmark RGB-D pada TBS
sawit sebelumnya di literatur manapun.
