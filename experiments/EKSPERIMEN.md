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

---

# Fase 6 — Diagnostik ulang dan pipeline dua-tahap

Konteks: pengguna meminta terobosan yang bisa dipertanggungjawabkan secara
matematis supaya depth benar-benar menaikkan metrik, dengan pelonggaran scope
eksplisit — boleh berat, boleh multi-tahap, tidak harus YOLO, tidak harus satu
pipeline. Sebelum melatih apa pun, dijalankan lima probe read-only; hasilnya
mengubah rumusan masalahnya. Jalan penemuan lengkap: `docs/DIAGNOSIS-DEPTH.md`.
Semua probe reproducible via `scripts/probe_depth_signal.py`.

---

## V2-E-012 — Gap mAP50 antara 953 dan 352 pohon disebabkan kelangkaan label B3/B4, bukan kanal depth

**Tanggal:** 2026-08-11
**Hipotesis:** Selisih test mAP50 953-vs-352 dapat dijelaskan sepenuhnya oleh
perbedaan komposisi kelas, bukan oleh kehadiran kanal depth. Falsifikasi:
kalau gap tersebar merata di keempat kelas, hipotesis ini salah.
**Dataset & split:** SawitMVC-YOLO (953) dan SawitMVC-Depth (352), seluruh
split, hitung ulang langsung dari file label.
**Metode:** `scripts/probe_depth_signal.py --probe distribusi`

**Hasil:**

| Split | citra | instance | /citra | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|---|---|
| 953-train | 3.000 | 14.041 | 4,68 | 11,2% | 18,6% | 52,2% | 17,9% |
| 953-test | 588 | 2.612 | 4,44 | 9,6% | 19,0% | 53,9% | 17,4% |
| 352-train | 980 | 1.517 | 1,55 | 35,8% | 43,6% | 14,2% | 6,5% |
| 352-test | 220 | 410 | 1,86 | 35,9% | 42,4% | 15,4% | 6,3% |

B3 train 7.333 → 215 instance (34× lebih sedikit); B4 2.513 → 98 (26×).
AP50 per kelas (YOLO26l test): B1 0,7705→0,6804, B2 0,4479→0,4320,
**B3 0,6050→0,2001, B4 0,3506→0,1299**.

**Sumber:** `results/perkelas_pycoco_v2repro.json`,
`results/perkelas_pycoco_rgb352.json`, hitung ulang label via probe.
**Verdict: CONFIRMED** — gap terkonsentrasi persis di dua kelas yang
instance-nya menghilang; B1/B2 nyaris tidak berubah.
**Konsekuensi:** perbandingan lintas dataset 953-vs-352 tidak sah dan tidak
boleh dipakai lagi untuk menilai depth. Memotong dataset 953 jadi 25% tetap
menyisakan ~1.800 instance B3 dengan komposisi kelas yang sama, jadi "RGB 25%
tetap menang" adalah hasil yang diharapkan dan tidak menguji depth.

---

## V2-E-013 — Sebagian besar kehilangan mAP50 berasal dari salah kelas, bukan gagal lokalisasi

**Tanggal:** 2026-08-11
**Hipotesis:** Pada bobot RGB-352 yang sudah ada, AP50 class-agnostic jauh di
atas mAP50 class-aware. Falsifikasi: kalau keduanya berdekatan, yang rusak
adalah lokalisasi dan pemisahan dua-tahap tidak akan menolong.
**Dataset & split:** SawitMVC-Depth 352, split kanonik, test (410 box).
**Metode:** inference `runs/yolo26l_e60_i1280_rgb352/weights/best.pt`
(conf 0,001, IoU-NMS 0,7), lalu AP50 gaya COCO dihitung dua kali — sekali
per kelas, sekali dengan seluruh kelas dilipat jadi satu. Implementasi
divalidasi lebih dulu: mAP50 hasil hitung sendiri 0,3707 vs pycocotools
0,3711 (selisih 0,0004).

**Hasil:**

| Besaran | Nilai |
|---|---|
| mAP50 class-aware | 0,3707 |
| AP50 class-agnostic (lokalisasi murni) | **0,6677** |
| Hilang karena salah kelas | **0,2970 (44,5%)** |

Konfusi pada box yang sudah benar lokasinya (IoU≥0,5, conf≥0,25):

| | →B1 | →B2 | →B3 | →B4 | recall |
|---|---|---|---|---|---|
| B1 | 92 | 26 | 0 | 0 | 78,0% |
| B2 | 13 | 83 | 12 | 0 | 76,9% |
| B3 | 0 | 21 | 11 | 4 | 30,6% |
| B4 | 0 | 1 | 3 | 5 | 55,6% |

Akurasi klasifikasi 70,5% (n=271). Seluruh kesalahan jatuh ke kelas
bertetangga — nol kasus B1→B3/B4 — jadi ini masalah **ordinal**.
Catatan kejujuran: 70,5% itu bersyarat pada box yang berhasil dideteksi
(271 dari 410); atas seluruh GT akurasinya 191/410 = **46,6%**.

**Sumber:** `scripts/eval_twostage.py` (fungsi `ap50`), log sesi 2026-08-11.
**Verdict: CONFIRMED** — plafon mAP50 pipeline ini adalah 0,6677, dan 44,5%
kemampuan yang sudah ada terbuang di tahap penamaan kelas.

---

## V2-E-014 — Sinyal depth yang tersedia adalah relief lokal ordinal, bukan skala metrik, dan sub-kuantum per piksel

**Tanggal:** 2026-08-11
**Hipotesis (A):** depth memberi skala metrik (`D = d·Z/f`) yang memisahkan
kelas lebih baik daripada ukuran piksel.
**Hipotesis (B):** depth memberi kontras kedalaman lokal antara tandan dan
sekelilingnya yang monoton terhadap kematangan.
**Dataset & split:** 2.299 box GT SawitMVC-Depth + `depth_png_352/`.
**Metode:** `scripts/probe_depth_signal.py --probe depth`

**Hasil A — FALSIFIED.** Z median per kelas nyaris konstan:
B1 1,36 m / B2 1,33 / B3 1,31 / B4 1,20. Protokol foto jarak tetap, jadi
mengalikan dengan Z hanya menggeser skala. (Temuan sampingan: depth **95,1%
valid DI DALAM box** — angka "29% invalid" yang selama ini dikutip itu latar,
bukan objek.)

**Hasil B — CONFIRMED.** Relief = median Z(cincin) − median Z(box):

| | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| relief median | **+2,8 cm** | 0,0 cm | −1,5 cm | **−5,1 cm** |
| lebih dekat dari sekitar | 61,3% | 50,7% | 41,4% | 26,4% |

Kruskal-Wallis 4 kelas: **H = 99,8, p = 1,7×10⁻²¹**. Monoton sempurna.

**Kenapa sinyal sekuat itu tidak terpakai:** encoding uint8 inverse-depth
`[0,8; 15,0]` m punya step `dZ/dv = Z²·(1/Z_NEAR − 1/Z_FAR)/254` = **2,91 cm
per level di Z=2,5 m** (median Z per citra dataset ini 2,49 m). Sinyal relief
median 0,8 cm = **0,27 level**; B4 (5,1 cm) = 1,8 level. Dengan derau sensor
~1% Z ≈ 2,5 cm, **SNR per piksel ≈ 0,3**. Rentang dinamis kanal justru habis
untuk ramp global adegan (entropi 7,68 dari 8 bit) yang **nuisance** — median
Z per citra std 0,82 m, rentang 0,80–6,44 m, mengikuti posisi operator.

Pooling memulihkan sinyalnya (AUC B1-vs-B4):

| piksel di-pool | AUC train+val | AUC test |
|---|---|---|
| 1 | 0,592 | 0,577 |
| 16 | 0,724 | 0,650 |
| 256 | 0,728 | 0,593 |
| 4.096 | 0,730 | 0,621 |

**Sumber:** `scripts/probe_depth_signal.py`, `depth_png_352/depth_meta.json`.
**Verdict: A FALSIFIED, B CONFIRMED.**
**Konsekuensi:** depth harus dikonsumsi **setelah pooling wilayah**, pada jalur
**klasifikasi**. Early fusion di stem adalah rezim terburuk (resolusi penuh,
pooling minimum) — menjelaskan kegagalan berulang E-022/E-027/E-032/V2-E-005/006,
sekaligus meretrodiksi kenapa `edge` (Sobel = high-pass yang membuang ramp
global) satu-satunya yang pernah menang (V2-E-008/010).
**Koreksi terhadap pemahaman lama:** rentang `[0,8; 15,0]` dipilih di Volume 1
dengan memaksimalkan entropi SELURUH CITRA — objektif yang keliru untuk tugas
ini, karena mengoptimalkan deskripsi langit dan pohon jauh, bukan resolusi pada
skala objek.

---

## V2-E-015 — Classifier kematangan pada crop mengalahkan klasifikasi detektor satu-tahap

**Tanggal:** 2026-08-11
**Hipotesis:** Memisahkan klasifikasi kematangan menjadi model crop tersendiri
(dengan pretraining dari 846 pohon 953 yang bebas bocor, sampling seimbang
kelas, dan mask box target) menaikkan akurasi kematangan di atas 46,6% yang
dicapai detektor Fase 1-5 atas seluruh GT.
**Dataset & split:** crop GT SawitMVC-Depth 352, split kanonik
(1.517 train / 372 val / 410 test); pretraining dari 16.542 crop 846 pohon 953
(`splits_fase6/pretrain953_*`, irisan nol dengan val/test-352 diverifikasi).
**Metode:** `scripts/build_crop_dataset.py` + `scripts/train_crop_classifier.py`,
backbone `convnext_tiny.fb_in22k_ft_in1k` (in_chans=4: RGB + mask box), head
hybrid (CE + CORAL), 45 epoch, batch 32.

**Hasil (akurasi kematangan, test split 410 crop):**

| Pendekatan | test akurasi |
|---|---|
| Tebak kelas terbanyak (B2) | 0,4244 |
| Histogram warna + regresi logistik | 0,4780 |
| **Detektor Fase 1-5 atas seluruh GT** | **0,4659** (191/410) |
| **Classifier crop (rata-rata 3 seed)** | **0,6309 ± 0,0203** |

**Dua bug sendiri yang sempat menahan hasil** (dicatat karena keduanya generik
dan mudah terulang):
1. Crop diperluas ctx=1,6 supaya cincin ikut masuk, tapi di kanopi padat sering
   ada >1 tandan per crop — tanpa penanda, model tidak tahu tandan mana yang
   dinilai. Ditambahkan kanal **mask box**.
2. Augmentasi fotometrik awal (brightness ±25%, saturasi 0,6–1,4) menghapus
   label: kematangan tandan DIDEFINISIKAN oleh warna. Diturunkan ke ±7%.
Setelah keduanya diperbaiki, pretrain 953 naik dari akurasi 0,471 → 0,648.

**Sumber:** `runs_fase6/sd{101,202,303}_rgb/hasil.json`,
`runs_fase6/pre953v2/hasil.json`.
**Verdict: CONFIRMED** — +16,5pp absolut di atas klasifikasi detektor.
**Catatan:** run dengan `in_chans=3` di `runs_fase6/` (ft_rgb_coral,
ft_rgb_hybrid, ft_rgbd_hybrid) berasal dari kode sebelum kedua bug diperbaiki
dan **tidak sebanding** — sengaja tidak dihapus, tapi tidak dipakai di angka
manapun.

---

## V2-E-016 — Informasi kematangan yang dibawa depth REDUNDAN secara kondisional terhadap RGB

**Tanggal:** 2026-08-11
**Hipotesis:** Kanal relief depth menaikkan akurasi klasifikasi kematangan di
atas RGB saja. Falsifikasi: kalau delta-nya nol atau negatif lintas seed,
hipotesis gugur.
**Dataset & split:** sama dengan V2-E-015.

### Bagian A — cabang CNN depth, 3 seed

Cabang depth terpisah (2 kanal: relief + mask valid), difusikan setelah global
pooling, gate init 0,1 (taknol, pelajaran F-007), plus loss auxiliary RGB-only.

| seed | val rgb | val rgbd | Δ | test rgb | test rgbd | Δ |
|---|---|---|---|---|---|---|
| 101 | 0,6505 | 0,6075 | −0,0430 | 0,6146 | 0,5805 | −0,0341 |
| 202 | 0,6640 | 0,6398 | −0,0242 | 0,6537 | 0,6073 | −0,0463 |
| 303 | 0,6290 | 0,6532 | +0,0242 | 0,6244 | 0,6439 | +0,0195 |

Rata-rata **Δval = −0,0143** (t=−0,72, p=0,55), **Δtest = −0,0203**
(t=−1,01, p=0,42). Gate berhenti di 0,110–0,114 dari init 0,100 — model
praktis tidak membuka jalur depth.

Catatan penting: satu seed tunggal sempat memberi **+5,9pp** — persis besaran
yang, kalau dilaporkan sendirian, akan terbaca sebagai kemenangan depth.
Multi-seed menunjukkan itu derau.

### Bagian B — statistik depth terpool secara analitik

Bagian A bisa dibantah: desain cabang CNN melanggar temuan V2-E-014 sendiri
(pooling ditaruh di akhir, sesudah 4 conv ber-stride bekerja pada medan
ber-SNR ~0,3). Jadi diuji lagi dengan depth diberi kondisi paling
menguntungkan — 8 statistik yang SUDAH terpool (relief cincin−box, median,
std, cakupan valid, rentang persentil), ditempel ke fitur penultimate
classifier RGB terlatih, dibandingkan lewat regresi logistik yang sama.

Sinyal relief terverifikasi masih utuh di crop: B1 +1,34 cm, B2 −0,24,
B3 −2,60, B4 −4,29 — tetap monoton.

| Fitur | val akurasi | test akurasi |
|---|---|---|
| statistik depth saja (8 dim) | 0,3468 | 0,3756 |
| RGB saja (768 dim) | 0,6774 | 0,6415 |
| RGB + statistik depth (776 dim) | 0,6720 | 0,6415 |

**Kontribusi depth: −0,0054 val, +0,0000 test.**

**Sumber:** `runs_fase6/sd*/hasil.json`, `results/probe_fitur_depth.json`,
`scripts/probe_fitur_depth.py`.
**Verdict: FALSIFIED.**

**Interpretasi — ini temuan utamanya.** Depth membawa informasi kematangan bila
berdiri sendiri (`I(Y;D) > 0`: relief monoton, Kruskal-Wallis p=1,7×10⁻²¹ di
V2-E-014; dan sendirian ia mencapai 0,3756 vs tebakan acak 0,25). Tetapi
informasi itu **redundan secara kondisional terhadap RGB** (`I(Y;D|RGB) ≈ 0`).
Penjelasan fisiknya sederhana: tandan yang menonjol keluar dari pelepah (B1)
juga *terlihat* besar dan matang di RGB — relief adalah **akibat** dari
variabel laten yang sama (kematangan/ukuran tandan), bukan pengukuran
independen atasnya.

**Konsekuensinya bersifat batas, bukan kegagalan implementasi.** Tidak ada
arsitektur fusi yang bisa mengekstrak informasi yang tidak ada: kalau
`I(Y;D|RGB) ≈ 0`, maka risiko Bayes model RGB-D sama dengan model RGB, dan
setiap parameter tambahan hanya menambah error estimasi. Ini menjelaskan
seluruh rangkaian hasil nol RGB-D di kedua volume (E-022, E-027, E-032,
V2-E-005/006, V2-E-009) dengan satu pernyataan, dan memprediksi bahwa
percobaan fusi berikutnya juga akan nol.

**Batas klaim ini — jangan digeneralisasi berlebihan:**
- Berlaku untuk **klasifikasi kematangan** pada dataset ini. Kontribusi depth
  untuk **lokalisasi** (menemukan tandan tertutup) belum diuji terpisah —
  seluruh eksperimen sebelumnya mencampur kedua tugas.
- Berlaku untuk protokol pengambilan data ini: jarak standoff hampir tetap
  (Z per kelas 1,20–1,36 m), depth uint8, 352 pohon. Sensor dengan presisi
  lebih tinggi atau protokol jarak bervariasi bisa memberi hasil berbeda.

---

## V2-E-017 — Lokalisasi (deteksi 1 kelas) sudah mentok di plafon dataset, bukan kurang kapasitas

**Tanggal:** 2026-08-12
**Hipotesis:** AP50 lokalisasi pada 352 pohon masih jauh di bawah yang bisa
dicapai kalau data lebih banyak. Falsifikasi: kalau dataset 953 (9,8x lebih
banyak box latih) mencapai AP50 lokalisasi yang setara, berarti keduanya sudah
menyentuh plafon resep ini dan menambah kapasitas/model tidak akan menolong.
**Dataset & split:** `agnostic352` dan `agnostic953` (label dilipat jadi 1 kelas
"tandan", `scripts/make_agnostic_dataset.py`), split kanonik yang sama dengan
Fase 1-5. Pretraining memakai 846 pohon 953 yang sudah dibersihkan dari
kebocoran (`splits_fase6/pretrain953_*`, irisan nol terverifikasi).
**Metode:** `scripts/train_yolo_4ch_screening.py` (YOLO26l) dan RTDETR untuk
pembanding arsitektur; evaluasi `scripts/eval_detector_agnostic.py`.

**Hasil — training (val split masing-masing):**

| Run | Epoch | best val AP50 | @ep | P | R |
|---|---|---|---|---|---|
| `agn953_pre-2` (pretrain dipotong) | 4 | 0,7604 | 4 | 0,7467 | 0,6976 |
| `agn953_full` (pretrain cosine utuh) | 12 | **0,8101** | 11 | 0,8044 | 0,7279 |
| `agn352_ft` (dari pretrain dipotong) | 50 | **0,7522** | 39 | 0,8105 | 0,6909 |
| `agn352_ft2` (dari pretrain utuh, patience 10) | 11 | 0,6413 | 1 | 0,7015 | 0,6075 |
| `agn352_ft3` (dari pretrain utuh, patience 45) | 60 | 0,7473 | 42 | 0,7620 | 0,6969 |
| `agn352_rtdetr` (RT-DETR-L) | 36 | 0,7157 | 26 | 0,7634 | 0,6504 |

**Hasil — pengukuran plafon, keduanya di split TEST dan bebas kebocoran:**

| Dataset | box latih | AP50 lokalisasi (test) |
|---|---|---|
| SawitMVC 953 (`v2repro`, dilatih pada split 953 yang benar) | 14.859 | **0,7374** |
| SawitMVC-Depth 352 (`agn352_ft`) | 1.517 | **0,7330** |

**Sumber:** `results/fase6_ringkas.json`, `runs/agn*/results.csv`.
**Verdict: CONFIRMED** — selisihnya hanya **0,0044** padahal dataset 953 punya
9,8x lebih banyak box latih. Lokalisasi sudah menyentuh plafon resep ini.
**Konsekuensi:** rencana memperbesar model (`yolo26x`, 59,0jt vs 26,3jt param)
DIBATALKAN sebelum dijalankan — hambatannya bukan kapasitas detektor. Sebagai
akibat lain, **mAP50 di dataset ini tidak mungkin melewati ~0,733**, karena
mAP50 <= AP50 lokalisasi secara definisi. Target 0,80 berada di atas plafon.

---

## V2-E-018 — Pretrain yang lebih baik di 953 TIDAK berpindah ke 352, dan patience bisa membunuh run di puncak palsu

**Tanggal:** 2026-08-12
**Hipotesis:** pretrain 953 yang lebih baik (0,8101 vs 0,7604, +5,0 poin)
menghasilkan finetune 352 yang lebih baik pula.
**Metode:** dua finetune dari pretrain utuh — `agn352_ft2` (patience 10) dan
`agn352_ft3` (patience 45) — dibandingkan dengan `agn352_ft` dari pretrain
yang dipotong.

**Hasil:**
- `agn352_ft2` berhenti di epoch 11 dengan best di **epoch 1** (0,6413).
  Transfer yang kuat membuat epoch 1 mencetak nilai tinggi, itu tercatat sebagai
  "best", lalu patience=10 memicu justru saat kurva sedang mendaki lagi
  (ep9 0,5924 -> ep10 0,6060 -> ep11 0,6063). Pembandingnya, `agn352_ft`, baru
  mencapai puncak di **epoch 39**. Run ini **cacat protokol**, bukan hasil.
- `agn352_ft3` (patience 45, jalan penuh 60 epoch): puncak **0,7473 @ep42**,
  vs `agn352_ft` **0,7522 @ep39**. Perbandingan epoch-per-epoch: ft3 unggul di
  14 dari 31 epoch pertama — pada dasarnya **seri**.

**Verdict: FALSIFIED** — keunggulan +5,0 poin pada domain 953 tidak berpindah
ke 352. Masuk akal: dua kamera berbeda (960x1280 HP vs 1280x800 Orbbec) dan
kepadatan objek berbeda (4,64 vs 1,55 per citra).
**Pelajaran protokol:** memotong jadwal cosine di tengah berbeda dari
early-stop saat plateau — `agn953_pre-2` yang dihentikan di epoch 4 dari 25
kehilangan seluruh fase anneal (LR masih di puncak 0,00193), dan pretrain utuh
menaikkannya +5,0 poin. Sebaliknya, patience yang terlalu ketat pada finetune
ber-transfer kuat bisa membunuh run sebelum kurva sebenarnya dimulai.

---

## V2-E-019 — WBF antar-detektor dan sweep konfigurasi inference menaikkan lokalisasi tanpa training tambahan

**Tanggal:** 2026-08-12
**Hipotesis:** menggabungkan beberapa detektor dan menyetel konfigurasi
inference menaikkan AP50 lokalisasi di atas detektor tunggal terbaik.
**Metode:** `scripts/pilih_detektor.py` (WBF, seluruh kombinasi) dan
`scripts/sweep_inferensi.py` (imgsz x NMS IoU). **Pemilihan dilakukan di split
val**, tidak pernah di test.

**Hasil (AP50 val):**

| Kombinasi | AP50 |
|---|---|
| **`agn352_ft` + `agn352_ft3` (WBF)** | **0,7577** |
| `agn352_ft` + `agn352_ft3` + `agn352_rtdetr` | 0,7443 |
| `agn352_ft` sendiri | 0,7370 |
| `agn352_ft3` sendiri | 0,7250 |
| `agn352_rtdetr` sendiri | 0,7135 |

Sweep memilih **imgsz 1280, NMS IoU 0,5** (bukan 0,7 default).
TTA deteksi (`augment=True`) diuji dan memberi **nol** perubahan — diabaikan
ultralytics untuk YOLO26.

**Verdict: CONFIRMED** — +2,1 poin dari 0,7370 ke 0,7577, tanpa training.
**Catatan penting:** `agn352_ft3` **kalah** sendirian (0,7250 vs 0,7370) tapi
gabungannya **melampaui keduanya**. Menambah RT-DETR justru menurunkan. Jadi
nilai sebuah model dalam ensemble tidak bisa dinilai dari performa tunggalnya.

---

## V2-E-020 — Pipeline dua-tahap mencapai mAP50 0,4500, setara model terbaik proyek

**Tanggal:** 2026-08-12
**Hipotesis:** memisahkan lokalisasi (detektor 1 kelas) dari klasifikasi
kematangan (classifier crop) menghasilkan mAP50 lebih tinggi daripada detektor
4-kelas satu-tahap.
**Dataset & split:** test 352 (410 box), sama persis dengan Fase 1-5.
**Metode:** `scripts/eval_twostage.py` — kelas + confidence tahap-2 ditempel ke
box tahap-1, skor = `conf_det x P(kelas)`, **multi-kelas** (tiap box memancarkan
4 deteksi), TTA 8 arah, ensemble classifier.

**Hasil:**

| Versi | Classifier | mAP50 | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|---|
| v1 | 6 | 0,4192 | 0,7188 | 0,4474 | 0,2734 | 0,2375 |
| v2 | 6 | 0,4395 | 0,7314 | 0,4689 | 0,3138 | 0,2440 |
| v3 | 3 (gabungan) | 0,4102 | 0,7358 | 0,4658 | 0,2681 | 0,1713 |
| **v4** | **9 (semua)** | **0,4500** | 0,7366 | 0,4683 | **0,3212** | **0,2738** |

Pembanding Fase 1-5 (test 352 yang sama):

| Model | mAP50 |
|---|---|
| YOLO26l RGB | 0,3711 |
| YOLO26l RGB+D `inverse` | 0,3919 |
| YOLO26l RGB+D `edge` | 0,4316 |
| RT-DETR-L RGB | 0,4343 |
| **Dua-tahap v4** | **0,4500** |
| RF-DETR-L RGB (rekor) | 0,4544 |

**Verdict: CONFIRMED terhadap satu-tahap YOLO26l** (+0,0789 absolut, +21,3%
relatif dari 0,3711), dan melampaui `edge` serta RT-DETR-L. **Belum melampaui
RF-DETR-L** — selisih 0,0044.
**Catatan:** dua-tahap unggul di B3 (0,3212 vs 0,2641 RT-DETR-L) dan B4
(0,2738 vs 0,2661 RF-DETR-L) — dua kelas yang paling langka.
**Rasio panen:** 0,4500 / 0,7330 = 0,614 dari plafon lokalisasi. Model lama
0,3711 / 0,6677 = 0,556. Jadi perbaikan datang dari KEDUA faktor.

---

## V2-E-021 — Training gabungan 953+352 menurunkan mAP50 tapi menaikkan counting

**Tanggal:** 2026-08-12
**Hipotesis:** melatih classifier pada gabungan crop 953+352 (B3: 215 -> 8.780,
B4: 98 -> 3.013) mengalahkan skema pretrain-lalu-finetune, karena tahap akhir
skema lama hanya melihat 215 crop B3 dan 98 B4 sehingga menghapus pengetahuan
kelas langka.
**Metode:** `--tahap gabung` di `scripts/train_crop_classifier.py`, 3 seed,
`convnext_small` @176; evaluasi tetap di val/test 352.

**Hasil (akurasi crop GT, rata-rata 3 seed):**

| Skema | val | test | test macro-F1 |
|---|---|---|---|
| `ftS` pretrain->finetune | 0,6729 | **0,6837** | 0,6105 |
| `ftJ` + jitter mask | 0,6900 | 0,6829 | 0,6065 |
| `ftG` gabungan | **0,6953** | 0,6724 | 0,5318 |

**Hasil hilir (test 352):**

| Konfigurasi | mAP50 | Counting Class ±1 |
|---|---|---|
| v2 (6 classifier lama) | 0,4395 | 86,82% |
| v3 (3 classifier gabungan) | **0,4102** | **88,18%** |
| v4 (9 classifier semua) | **0,4500** | 85,91% |

**Verdict: FALSIFIED untuk mAP50, CONFIRMED untuk counting.**
Gabungan menang di val tapi kalah di test — pola overfit ke domain yang salah:
dari 18.059 crop latih, **92% berasal dari 953** (kamera berbeda). Sampling
menyeimbangkan KELAS tapi tidak menyeimbangkan DOMAIN.

**Divergensi metrik yang penting dicatat:** konfigurasi terbaik untuk mAP50
(v4, 0,4500) BUKAN yang terbaik untuk counting (v3, 88,18%). Ini konsisten
secara matematis: mAP hanya peduli **urutan** deteksi di dalam tiap kelas,
sementara counting memakai **argmax** dan karenanya sensitif terhadap
kalibrasi prior kelas. Menyetel satu metrik bisa mengorbankan yang lain —
mis. NMS IoU 0,5 hasil sweep menaikkan mAP50 tapi menurunkan counting
(85,45% -> 83,18% pada v1).

**Sumber:** `results/fase6_ringkas.json`, `results/twostage_final*.json`,
`results/counting_twostage.json`, `runs_fase6/ft{S,J,G}_*/hasil.json`.

---

## V2-E-022 — Dataset 953 dan 352 adalah dua sesi akuisisi terpisah ~80 hari, bukan dua "view" pohon yang sama

**Tanggal:** 2026-08-12
**Hipotesis yang diuji (asumsi implisit seluruh Volume 2):** karena 352 pohon
DAMIMAS memakai tree ID yang sama di kedua dataset, keduanya adalah pohon yang
sama — satu direkam dengan depth, satu tanpa — sehingga 953 sah dipakai sebagai
korpus pretraining untuk 352.
**Metode:** `scripts/probe_pergeseran_temporal.py` — read-only, membandingkan
label kedua dataset pada citra ber-ID sama, dan membaca tanggal akuisisi dari
sidecar JSON 953 serta `MERGE_VERIFICATION.json` 352.

**Hasil — 1.408 citra ber-ID sama, dua himpunan label:**

| Sumber label | Total kotak | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|
| SawitMVC-YOLO (953) | 6.523 | 566 (8,7%) | 1.098 (16,8%) | **3.604 (55,3%)** | 1.255 (19,2%) |
| SawitMVC-Depth (352) | 2.299 | 829 (36,1%) | 1.001 (43,5%) | **321 (14,0%)** | 148 (6,4%) |

Rasio jumlah kotak 2,84x; B3 **11,2x**; B4 **8,5x**.

**Hasil — tanggal akuisisi:**

| Dataset | Akuisisi |
|---|---|
| SawitMVC-YOLO (953) | 30 April – 16 Mei 2026 |
| SawitMVC-Depth (352) | 28 – 29 Juli 2026 |

Jeda **~80 hari**. Rotasi panen sawit 7–15 hari, jadi ada 5–11 putaran panen di
antara kedua sesi. Citranya sendiri juga bukan berkas yang sama (953 potret
960x1280, 352 lanskap 1280x800).

**Sumber:** `results/pergeseran_temporal.json`.
**Verdict: asumsi FALSIFIED.** Tandan yang difoto Mei bukan tandan yang difoto
Juli. Kohort B3 yang dominan pada Mei sudah matang menjadi B1/B2 pada Juli, dan
sebagian sudah dipanen — konsisten dengan turunnya total kotak 6.523 ke 2.299
dan bergesernya distribusi dari 55% B3 menjadi 80% B1+B2.

**Koreksi terhadap V2-E-012.** Angka "B3 34x lebih langka di dataset 352" benar,
tetapi sebabnya salah. Itu bukan artefak dataset yang lebih kecil, melainkan
fase kematangan kebun yang berbeda **pada pohon yang sama**. Perbandingan
lintas-dataset 953-vs-352 tidak sah bukan hanya karena ketimpangan kelas, tapi
karena keduanya mengukur populasi buah yang berbeda.

**Konsekuensi untuk seluruh Fase 6.** Rangkaian pretrain 953 → finetune 352
bukan transfer di dalam satu domain, melainkan transfer melintasi pergeseran
domain temporal dengan distribusi kematangan nyaris terbalik. Ini menjelaskan
tiga hal yang sebelumnya tidak terjelaskan:

1. Recall B3 classifier hanya 0,254 dengan 36 dari 63 bocor ke B2, meskipun B3
   adalah kelas terbanyak dalam training gabungan (8.780 dari 18.059 crop).
2. `ftG` (training gabungan) mencatat val tertinggi tapi test terendah — bobot
   953 yang 8x lebih besar mendominasi prior yang salah untuk domain target.
3. Empat skema classifier (ftS/ftJ/ftG/ftH) tidak terbedakan satu sama lain:
   semuanya melawan celah domain yang sama, dan celah itu ada di data.

**Implikasi utama — mengapa detektor dan classifier timpang.** Label lokalisasi
("ada tandan di sini") bertahan melintasi jeda 80 hari karena posisi tandan di
kanopi relatif stabil. Label kematangan ("ini B3") tidak bertahan karena benda
fisiknya berubah. Itulah sebabnya AP50 class-agnostic mencapai 0,7330 sementara
mAP50 class-aware berhenti di ~0,45. Ketimpangan itu sifat pasangan data yang
dipakai, bukan cacat arsitektur.

---

## V2-E-023 — Split test 352 tidak punya daya statistik untuk membedakan konfigurasi Fase 6

**Tanggal:** 2026-08-12
**Hipotesis:** urutan konfigurasi Fase 6 berdasarkan titik estimasi mAP50
(0,4102 → 0,4192 → 0,4395 → 0,4500 → 0,4544) mencerminkan perbedaan nyata.
**Metode:** `scripts/bootstrap_map.py` — resampling pada tingkat CITRA (bukan
kotak), 500 ulangan, seed 42; selisih antar-sumber dihitung **berpasangan**
(sampel citra yang sama untuk kedua model) supaya korelasi antar-model tidak
menggelembungkan selang.

**Hasil (split test 352: 220 citra, 410 kotak GT):**

| Sumber | mAP50 | CI 95% | Lebar |
|---|---|---|---|
| YOLO26l-RGBD `edge` | 0,4270 | [0,3771; 0,4938] | **0,1167** |
| YOLO26l-RGB | 0,3677 | [0,3286; 0,4417] | **0,1130** |

Selisih berpasangan `edge` − RGB: **+0,0593**, CI 95% **[−0,0013; +0,1168]**,
P(Δ>0) = 0,972 → **tidak signifikan**, selang masih memuat nol.

**Sumber:** `results/bootstrap_map_awal.json`.
**Verdict: hipotesis FALSIFIED.** Lebar CI ~0,117 sementara jarak antara
dua-tahap terbaik (0,4500) dan rekor proyek RF-DETR-L (0,4544) hanya **0,0044**
— **26x lebih kecil dari lebar selangnya**. Seluruh urutan konfigurasi Fase 6
jatuh di dalam satu selang kepercayaan yang sama dan tidak terbedakan.

**Kegagalan metodologis yang diakui.** V2-E-011 (Fase 5) memakai bootstrap CI
dan berani menyimpulkan INCONCLUSIVE. Fase 6 meninggalkan praktik itu dan
mengurutkan konfigurasi berdasarkan titik estimasi selama enam versi
rekomposisi (v1–v6) serta empat skema classifier (12 training). Bukti bahwa
selisih-selisih itu derau sebenarnya sudah tersedia lebih awal: sebaran
akurasi test antar-seed (0,6512–0,7049) lebih lebar daripada sebaran
antar-metode (0,6707–0,6837).

**Konsekuensi.** Angka Fase 6 hanya boleh dilaporkan dengan selangnya. Setiap
pekerjaan lanjutan pada dataset ini harus menghitung daya statistik lebih dulu:
dengan 410 kotak GT, efek di bawah ~0,10 mAP50 tidak terdeteksi.

---

## V2-E-024 — Depth menaikkan LOKALISASI dan menembus plafon yang diklaim V2-E-017, tapi belum signifikan di split ini

**Tanggal:** 2026-08-12
**Hipotesis:** kanal depth menaikkan AP50 **lokalisasi** (deteksi 1 kelas).
Ini satu-satunya perbandingan RGB vs RGB+D pada proyek ini yang **tidak bisa
dikotori pergeseran temporal** (V2-E-022), karena class-agnostic membuang label
kematangan sepenuhnya dan menyisakan hanya "ada tandan atau tidak" — label yang
bertahan melintasi jeda 80 hari karena posisi tandan di kanopi relatif stabil.

**Rancangan berpasangan.** Resep, inisialisasi (`agn953_full`), seed (42),
jadwal (60 epoch, patience 45, cosine), resolusi (1280), dan batch (4)
**identik**. Satu-satunya yang berbeda: jumlah kanal masukan. Kanal ke-4 memakai
encoding `edge` (Sobel gradien depth), pemenang screening V2-E-008/010 —
diverifikasi identik dengan `SawitMVC-Depth-4ch-edge`. Bobot stem diinflasi
3→4 kanal (1092/1092 item tertransfer, terverifikasi sebelum run).

**Hasil — training (val):**

| Run | Kanal | best val AP50 | @ep | Durasi |
|---|---|---|---|---|
| `agn352_4ch` | 4 (RGB + `edge`) | **0,7893** | 33 | 165 mnt |
| `agn352_ft3` | 3 (RGB) | 0,7473 | 42 | 168 mnt |

`agn352_4ch` unggul di 21 dari 26 epoch pertama dan menyamai puncak
seumur-hidup kontrol RGB pada epoch 26.

**Hasil — split TEST, AP50 lokasi murni, dengan CI bootstrap berpasangan
(1.000 ulangan, resampling tingkat citra, seed 42):**

| Model | AP50 test | CI 95% | Lebar | n prediksi |
|---|---|---|---|---|
| `agn352_4ch` (RGB+D) | **0,7636** | [0,7144; 0,8123] | 0,0979 | 1.660 |
| `agn352_ft3` (RGB) | 0,7358 | [0,6820; 0,7917] | 0,1097 | 1.226 |

Selisih berpasangan: **+0,0278**, CI 95% **[−0,0121; +0,0648]**,
P(Δ>0) = **0,921** → **belum signifikan pada taraf 95%**.

**Sumber:** `results/bootstrap_lokalisasi.json`,
`results/pred_agn4ch_test.npz`, `results/pred_agnrgb_test.npz`,
`runs/agn352_4ch/results.csv`.

**Verdict: POSITIF TAPI BELUM KONKLUSIF.** Arah efeknya konsisten di val
(+0,0420) dan test (+0,0278), dan ini sinyal positif terkuat untuk depth di
seluruh Volume 2 — satu-satunya yang muncul dari perbandingan yang benar-benar
bersih. Tetapi selangnya masih memuat nol.

**Ketidaksignifikanan di sini TIDAK boleh dibaca sebagai "tidak ada efek".**
V2-E-023 sudah menetapkan bahwa split test ini tidak mampu memisahkan efek di
bawah ~0,10. Efek terukur 0,0278 berada jauh di bawah ambang itu, jadi hasil
"tidak signifikan" memang sudah bisa diramalkan sebelum eksperimen dijalankan
dan tidak membawa informasi tentang ada-tidaknya efek. Yang kurang adalah data,
bukan efeknya.

**Koreksi terhadap V2-E-017.** Entri itu menyimpulkan "mAP50 di dataset ini
tidak mungkin melewati ~0,733" karena AP50 lokalisasi test-352 (0,7330) praktis
sama dengan test-953 (0,7374) meski 953 punya 9,8× lebih banyak kotak latih.
Kesimpulan itu benar **sebagai pernyataan tentang masukan RGB**, tapi ditulis
seolah berlaku umum untuk dataset. Dengan kanal depth, titik estimasi lokalisasi
mencapai **0,7636** — di atas kedua angka tersebut. Plafon itu ternyata sifat
dari **modalitas masukan**, bukan sifat dataset. Perlu ditegaskan: 0,7636 masih
di dalam CI 0,7330, jadi ini pembalikan **titik estimasi**, bukan pembalikan
yang terbukti signifikan.

**Konsekuensi.** Ini menajamkan rekomendasi §10 laporan. Depth tampaknya
menolong di tempat yang persis diprediksi teori V2-E-022 — lokalisasi, bukan
kematangan. Akuisisi berikutnya sebaiknya dirancang untuk menguji **itu**,
dengan test split ≈4.000 kotak supaya efek berukuran 0,03 bisa dipisahkan.

---

## V2-E-025 — Angka test class-agnostic untuk `agn953_full`, dan besarnya efek kontaminasi pretraining

**Tanggal:** 2026-08-12
**Lubang yang ditutup:** `agn953_full` selama ini hanya punya AP50 **val**
(0,8101). `make_agnostic_dataset.py` memang hanya membuat split train+val untuk
`agnostic953` (baris `p953 = {"train": [], "val": []}`), sehingga angka test-nya
tidak pernah ada. Angka "test-953 = 0,7374" yang sempat dikutip berasal dari
model **berbeda** — detektor class-aware `v2repro` yang prediksinya dilipat jadi
satu kelas.
**Metode:** `scripts/buat_test_953_bersih.py`. Karena `pretrain953_images.txt`
mengambil semua 846 pohon bebas-bocor tanpa menghormati split kanonik 953, dari
141 pohon test kanonik hanya **19 pohon (76 citra, 316 kotak)** yang benar-benar
tak tersentuh training. Dua set dilaporkan supaya efek kontaminasi terlihat,
bukan disembunyikan.

**Hasil:**

| Set evaluasi | Pohon | Citra | Kotak | AP50 agnostik |
|---|---|---|---|---|
| **test bersih** (tak tersentuh) | 19 | 76 | 316 | **0,7702** |
| test penuh (122/141 pohon terpakai saat training) | 141 | 588 | 2.612 | 0,8090 |
| val (pembanding, dilaporkan selama ini) | — | 364 | — | 0,8101 |

**Sumber:** `results/pred_agn953_bersih.npz`, `results/pred_agn953_penuh.npz`,
`results/test953_bersih.json`.
**Verdict:** angka yang sah untuk `agn953_full` adalah **0,7702**, bukan 0,8101.
Selisih 0,0388 antara set bersih dan set penuh adalah besarnya optimisme akibat
kontaminasi — dan angka val (0,8101) hampir identik dengan set terkontaminasi
(0,8090), persis seperti yang diharapkan kalau keduanya berbagi pohon dengan
training.
**Peringatan:** set bersih hanya 316 kotak, jadi CI-nya lebih lebar lagi
daripada split test 352 (yang sudah ±0,058 pada 410 kotak). Angka 0,7702 harus
dibaca sebagai indikasi, bukan pengukuran presisi.

---

## V2-E-026 — CI untuk angka utama Fase 6: dua-tahap 0,4500 tidak terbedakan dari pembandingnya

**Tanggal:** 2026-08-12
**Metode:** konfigurasi v4 dijalankan ulang di test (9 classifier, WBF
`agn352_ft`+`agn352_ft3`, imgsz 1280, NMS IoU 0,5, TTA, multi-kelas) dengan dump
prediksi, lalu bootstrap 1.000 ulangan berpasangan.

**Hasil:** reproduksi persis — mAP50 = **0,44999** vs 0,4500 yang dilaporkan
V2-E-020, per kelas identik (B1 0,7366 / B2 0,4683 / B3 0,3212 / B4 0,2738).

| Model | mAP50 | CI 95% | Lebar |
|---|---|---|---|
| Dua-tahap v4 | 0,4500 | [0,4054; 0,5188] | 0,1133 |
| YOLO26l-RGBD `edge` | 0,4270 | [0,3836; 0,4984] | 0,1148 |

Selisih berpasangan: **+0,0230**, CI 95% **[−0,0286; +0,0663]**, P(Δ>0) = 0,789
→ **tidak signifikan**.

**Sumber:** `results/bootstrap_map.json`, `results/twostage_v4_ulang.json`.
**Verdict:** menegaskan V2-E-023. Angka utama Fase 6 tidak terbedakan dari
detektor satu-tahap yang jauh lebih sederhana, apalagi dari RF-DETR-L (0,4544)
yang bahkan lebih tinggi titik estimasinya. Seluruh kerja rekomposisi enam versi
tidak menghasilkan perbedaan yang bisa dibuktikan pada split ini.

---

## V2-E-027 — Monocular-depth sebagai kanal ke-4 pada SawitMVC 953: turun −0,0475 mAP50 di test

**Tanggal:** 2026-08-15
**Hipotesis:** peta monocular-depth (`yolo26l-depth.pt`, ukuran L) yang
ditambahkan sebagai kanal ke-4 lewat early fusion menaikkan deteksi kelas-sadar
pada SawitMVC 953 dibandingkan RGB murni. Ini sel 6 dari matriks
mono-depth; satu-satunya sel yang punya daya statistik memadai (test 2.612
kotak, bukan 410) dan bebas pergeseran temporal 80 hari.

**Data:** `/workspace/d953_rgbmono` — TIFF 4 kanal `[B,G,R,mono]`, dibangun
`scripts/buat_dataset_nch.py --dataset 953 --kanal mono`. Kanal mono = PNG uint8
inverse-depth pada `[z_near, z_far] = [0,8; 15,0] m`, di-encode dengan
`encode_inverse()` yang sama persis dengan kanal depth sensor (diimpor dari
Research-Pipeline, bukan ditulis ulang). Split kanonik 716/96/141 pohon =
3.000/404/588 citra, 14.041/1.887/**2.612** kotak.

**Resep:** identik dengan sel 5 (`yolo26l_e60_i1280_v2repro`) — `yolo26l.pt`
COCO init, imgsz 1280, batch 4, seed 42, `cos_lr`, `close_mosaic` 10,
optimizer auto, lr0 0,01. Stem di-inflate 3 -> 4 kanal oleh ultralytics.

### Metrik training (val split, 404 citra, evaluator native ultralytics)

| | nilai | epoch |
|---|---|---|
| val mAP50 terbaik | **0,5012** | ep17 |
| val mAP50-95 terbaik | 0,2295 | ep23 |
| val terakhir (ep31) | 0,4870 / 0,2231 | ep31 |

Perbandingan val-lawan-val dengan sel 5 pada epoch yang sama (kurva sel 5
dipulihkan dari kunci `train_results` di dalam `best.pt`, disimpan ke
`results/val_curve_sel5_953_rgb_v2repro.csv`):

| ep | sel 6 RGB+Mono | sel 5 RGB | selisih |
|---|---|---|---|
| 7 | 0,4612 | 0,4407 | +0,0205 |
| 17 | 0,5012 | 0,5003 | +0,0009 |
| 24 | 0,4738 | 0,5181 | −0,0444 |
| 28 | 0,4760 | 0,5219 | −0,0459 |
| 31 | 0,4870 | 0,5195 | −0,0325 |

Sel 6 tertinggal di 21 dari 31 epoch, dan di **setiap** epoch sejak ep18. Puncak
sel 5 sendiri 0,5373 @ep34. **Peringatan:** kurva val sel 6 di atas dihitung
atas 394 citra, bukan 404 — 10 citra val korup dan dilewati diam-diam oleh
ultralytics (lihat V2-E-028). Baris "selisih" di atas karena itu tidak
sepenuhnya sebanding dan tidak boleh dikutip sebagai angka; ia hanya
menunjukkan arah.

### Metrik test (pycocotools, 588 citra, 2.612 kotak GT — setelah perbaikan citra korup)

| | sel 6 RGB+Mono | sel 5 RGB | selisih |
|---|---|---|---|
| **mAP50** | **0,4960** | **0,5436** | **−0,0475** |
| mAP50-95 | 0,2322 | 0,2565 | −0,0243 |
| AP75 | 0,186 | — | — |

Per kelas AP50:

| Kelas | sel 6 | sel 5 | selisih |
|---|---|---|---|
| B1 | 0,6902 | 0,7708 | −0,0806 |
| B2 | 0,4097 | 0,4479 | −0,0382 |
| B3 | 0,5635 | 0,6051 | −0,0416 |
| B4 | 0,3206 | 0,3506 | −0,0300 |

Turun di **keempat** kelas, terbesar di B1. AP per ukuran objek (sel 6):
small 0,017 / medium 0,134 / large 0,270; AR@100 = 0,527.

### Batas yang melekat pada angka ini

1. **Dihentikan di 31 dari 60 epoch** atas keputusan pengguna, setelah kurva val
   konsisten tertinggal. `best.pt` = checkpoint ep17; `last.pt` = ep31, masih
   bisa di-resume. Sel 5 dilatih 60 epoch penuh dan puncaknya jatuh di ep34,
   **di luar jangkauan** run ini. Perbandingan ini karena itu **timpang dan
   condong menguntungkan sel 5**: cukup untuk menyimpulkan "mono tidak memberi
   keunggulan yang terlihat", tidak cukup untuk mengukur besar kerugiannya
   secara adil.
2. Satu seed. Tidak ada replikasi.
3. Training berjalan di atas 2.999 dari 3.000 citra train (satu korup).
4. CI bootstrap berpasangan sedang dihitung; hasilnya menyusul di entri
   terpisah. Sebelum itu, selisih −0,0475 belum boleh disebut signifikan.

**Sumber:** `results/eval_sel6_953_rgbmono_test.json`,
`results/pred_sel6_953_rgbmono_test.npz`, `runs/sel6_953_rgbmono/results.csv`,
`runs/sel6_953_rgbmono/DIHENTIKAN_LEBIH_AWAL`,
`results/val_curve_sel5_953_rgb_v2repro.csv`, `results/eval_sel5_953_rgb_test.json`.

**Verdict:** hipotesis **tidak didukung**. Menambahkan monocular-depth sebagai
kanal ke-4 menurunkan mAP50 sebesar 0,0475 di test, konsisten di keempat kelas,
dan konsisten pula dengan catatan lama repo ini bahwa early fusion depth adalah
regresi (E-022, E-027 Volume 1: −0,0230 pada YOLO26n). Yang **belum** bisa
dipisahkan: apakah kerugian ini berasal dari isi peta mono, atau semata dari
biaya menambah kanal pada stem yang bobot COCO-nya 3 kanal. Kontrol M_shuf
lintas-pohon adalah uji yang memisahkan keduanya dan tetap layak dijalankan
meski arahnya negatif.

---

## V2-E-028 — 39 citra TIFF korup di dataset turunan, dilewati diam-diam oleh ultralytics

**Tanggal:** 2026-08-15
**Bukan hipotesis** — catatan cacat data yang memengaruhi cara membaca V2-E-027.

**Temuan:** eval sel 6 gagal dengan `gagal membaca ...tiff`. Berkasnya ada dan
berukuran 8,5 MB tapi tidak bisa didekode oleh `cv2.imread`, pembaca
ultralytics, maupun `cv2.imdecodemulti`. Pemindaian penuh
(`scripts/perbaiki_tiff_korup.py`) menemukan 39 berkas korup:

| Dataset | split | total | korup |
|---|---|---|---|
| d953_rgbmono | train | 3.000 | 1 |
| d953_rgbmono | **val** | 404 | **10** |
| d953_rgbmono | test | 588 | 22 |
| d352_rgbmono | train | 980 | 6 |
| d352_rgbmono | val / test | 208 / 220 | 0 |
| d352_rgbedgemono (5 kanal) | semua | 1.408 | 0 |

Dua tanda tangan galat: `TIFFReadRGBAStrip` gagal (data terpotong) dan
`TIFFGetField PHOTOMETRIC` gagal (header rusak) — keduanya khas penulisan yang
terputus. Menariknya berkas 5 kanal yang ditulis `cv2.imwritemulti` justru
bersih seluruhnya; yang rusak hanya yang ditulis `cv2.imwrite`.

**Kenapa ini berbahaya, dan ini pelajaran utamanya:** ultralytics **melewati**
citra korup dengan peringatan lalu tetap menyelesaikan training. Tidak ada
kegagalan, tidak ada jejak di metrik akhir. Akibat konkretnya, metrik val sel 6
selama 31 epoch dihitung atas **394 citra** sementara baseline sel 5 dihitung
atas **404** — perbandingan yang tampak sah sepanjang malam sebenarnya
dilakukan di atas himpunan data yang berbeda. Cacat semacam ini tidak akan
pernah terlihat dari angkanya sendiri.

**Tindakan:** berkas korup dihapus (citra turunan, regenerable dalam hitungan
menit; ATURAN #1 diperiksa — nol `.pt`/`.pth`/`.ckpt` di sasaran), dibangun
ulang dengan `buat_dataset_nch.py`, cache label dibuang supaya ultralytics
memindai ulang. Verifikasi setelahnya: **0 korup** di ketiga dataset, jumlah
kanal terkonfirmasi 4/4/5, jumlah kotak kembali ke angka kanonik (test 953 =
2.612, test 352 = 410). Eval test sel 6 di V2-E-027 dijalankan **setelah**
perbaikan ini, jadi angka test-nya sah; yang tidak sepenuhnya sah hanya kurva
val-nya.

**Sumber:** `scripts/perbaiki_tiff_korup.py`, `results/tiff_korup.json`,
`results/tiff_korup_setelah_perbaikan.json`.

**Aturan yang lahir dari sini:** setiap dataset turunan diperiksa
keterbacaannya sebelum dipakai melatih, bukan sesudah. Satu pemindaian penuh
memakan ~3 menit; kalau dilewati, biayanya adalah seluruh run tidak bisa
dibandingkan dan baru ketahuan berjam-jam kemudian.

---

## V2-E-029 — CI berpasangan sel 6 vs sel 5: penurunan −0,0476 mAP50 SIGNIFIKAN

**Tanggal:** 2026-08-15
**Metode:** bootstrap berpasangan 2.000 ulangan atas citra test (seed 42), dari
dump prediksi yang disimpan saat evaluasi — `pred_sel6_953_rgbmono_test.npz` dan
`pred_sel5_953_rgb_test.npz`. GT diambil dari dataset asli
`/workspace/SawitMVC-YOLO` supaya kedua lengan dibandingkan terhadap sumber yang
sama. 588 citra, 2.612 kotak.

| Model | mAP50 | CI 95% | Lebar |
|---|---|---|---|
| Sel 6 — RGB+Mono (4 kanal) | 0,4960 | [0,4729; 0,5225] | 0,0496 |
| Sel 5 — RGB (3 kanal) | 0,5436 | [0,5206; 0,5712] | 0,0506 |

**Selisih berpasangan: −0,0476, CI 95% [−0,0671; −0,0274], P(Δ>0) = 0,000
→ SIGNIFIKAN pada 95%.**

CI selisih tidak memuat nol, dan tidak satu pun dari 2.000 ulangan menghasilkan
Δ positif. Lebar CI 0,0496 sesuai perkiraan daya statistik untuk 2.612 kotak
(bandingkan split 352 dengan 410 kotak, lebar CI ~0,11 — di sana selisih sebesar
ini tidak akan bisa dibedakan dari nol).

**Sumber:** `results/boot_sel6_vs_sel5.json`.

**Verdict:** ini hasil negatif yang **tegas**, bukan sekadar tidak terbukti.
Menambahkan monocular-depth sebagai kanal ke-4 pada SawitMVC 953 menurunkan
mAP50 secara signifikan. Perlu diingat run sel 6 berhenti di 31 dari 60 epoch
(V2-E-027 butir 1), sehingga besar penurunannya kemungkinan dilebih-lebihkan —
tapi arahnya tidak diragukan, dan konsisten di keempat kelas serta di seluruh
2.000 ulangan bootstrap. Menjalankan sel 6 sampai 60 epoch bisa memperkecil
angkanya, tidak masuk akal membalikkan tandanya.

Konsisten dengan catatan lama repo: early fusion depth adalah regresi (E-022,
E-027 Volume 1, −0,0230 pada YOLO26n). Yang masih terbuka: apakah kerugian
berasal dari isi peta mono atau dari biaya menambah kanal pada stem COCO
3-kanal. M_shuf lintas-pohon memisahkan keduanya.

---

## V2-E-030 — Sel 3 (352 RGB+Mono): naik +0,0266 atas RGB tapi tidak signifikan, dan urutan val terbalik dari test

**Tanggal:** 2026-08-15
**Hipotesis:** monocular-depth sebagai kanal ke-4 menaikkan deteksi kelas-sadar
pada SawitMVC-Depth 352, dataset yang sama tempat depth sensor terbukti menang.

**Data:** `/workspace/d352_rgbmono`, TIFF 4 kanal `[B,G,R,mono]`, split kanonik
`canonical_70_15_15` = 980/208/220 citra, 1.517/372/**410** kotak. Resep identik
dengan sel 1 dan sel 2.

**Training:** dihentikan atas keputusan pengguna di **54 dari 60 epoch** setelah
plateau terkonfirmasi. `best.pt` = ep41 (val mAP50 0,3888). Biaya
komparabilitasnya kecil — pembandingnya juga checkpoint tengah (sel 1 @ep45,
sel 2 @ep38) dan tujuh epoch yang dilewatkan seluruhnya di fase `close_mosaic`
yang menurunkan val di ketiga run. Detail: `runs/sel3_352_rgbmono/DIHENTIKAN_LEBIH_AWAL`.

### Val vs test — urutannya TERBALIK, untuk ketiga sel

| Sel | Input | ch | val puncak | test mAP50 |
|---|---|---|---|---|
| 1 | RGB | 3 | **0,4111** @ep45 | 0,3677 |
| 3 | RGB+Mono | 4 | 0,3888 @ep41 | 0,3943 |
| 2 | RGB+Depth `edge` | 4 | 0,3856 @ep38 | **0,4270** |

Peringkat val (1 > 3 > 2) adalah **kebalikan persis** peringkat test (2 > 3 > 1).
Ini pengulangan kedua dari pembalikan yang sudah terlihat pada sel 2 di V2-E-005
dan sekarang terbukti berlaku untuk seluruh trio. Val 352 hanya 208 citra;
**val split ini tidak boleh dipakai memeringkat model.** Bandingkan 953, di mana
val 404 citra dan urutannya sejalan dengan test (sel 5 val 0,5373 -> test 0,5436;
sel 6 val 0,5012 -> test 0,4960).

### Test (pycocotools, 220 citra, 410 kotak)

| | sel 3 RGB+Mono | sel 1 RGB | sel 2 RGB+Depth |
|---|---|---|---|
| mAP50 | **0,3943** | 0,3677 | 0,4270 |
| mAP50-95 | 0,1360 | — | — |

Per kelas AP50 sel 3: B1 0,7232 / B2 0,4698 / B3 0,2546 / B4 0,1295.

### CI bootstrap berpasangan (2.000 ulangan, seed 42, dari dump .npz)

| Perbandingan | Selisih | CI 95% | P(Δ>0) | Signifikan |
|---|---|---|---|---|
| sel 3 − sel 1 (mono vs RGB) | **+0,0266** | [−0,0270; +0,0739] | 0,830 | **tidak** |
| sel 3 − sel 2 (mono vs depth sensor) | **−0,0327** | [−0,0756; +0,0074] | 0,057 | **tidak** |

Lebar CI 0,099–0,116. Ketidaksignifikanan ini **sudah diperkirakan sebelum
eksperimen dijalankan**: dengan 410 kotak, selisih di bawah ~0,06 memang tidak
bisa dibedakan dari nol. Ini batas daya statistik split-nya, bukan bukti bahwa
efeknya nol.

Catatan angka: titik estimasi sel 1 dan sel 2 di sini (0,3677 dan 0,4270)
dihitung ulang dari `.npz` lewat jalur kode yang sama dengan sel 3, sehingga
perbandingan berpasangannya konsisten secara internal. Angka historis
0,3711/0,4316 berasal dari skrip eval lama; selisihnya ~0,004 dan tidak
mengubah kesimpulan apa pun.

**Sumber:** `results/eval_sel3_352_rgbmono_test.json`,
`results/pred_sel3_352_rgbmono_test.npz`, `results/boot_sel3_vs_sel1.json`,
`results/boot_sel3_vs_sel2.json`, `runs/sel3_352_rgbmono/results.csv`.

**Verdict:** arahnya **berlawanan dengan sel 6**. Di 953 mono menurunkan mAP50
secara signifikan (−0,0476, V2-E-029); di 352 mono justru menaikkannya
(+0,0266), meski tidak signifikan. Mono juga berada di antara RGB dan depth
sensor pada dataset ini — konsisten dengan probe V2-E-0xx yang menemukan mono
mereproduksi relief ordinal B1->B4 yang sama dengan sensor tapi dengan amplitudo
lebih lemah (−4,08 cm vs −5,14 cm).

Penjelasan yang tersisa dan belum diuji: (a) mono berguna pada dataset dengan
citra dekat/terkendali (352, median 1,91 m) tapi merugikan pada citra lapangan
yang lebih beragam (953, median 1,31 m); (b) perbedaannya berasal dari ukuran
data latih (980 vs 3.000 citra); (c) sel 6 dihentikan di 31 epoch sehingga
kerugiannya dilebih-lebihkan. Ketiganya bisa dibedakan, tapi butuh eksperimen
tambahan yang belum dijadwalkan.

---

## V2-E-031 — Sel 4 (352 RGB+Depth+Mono, 5 kanal): mono DI ATAS depth sensor merugikan −0,0504, signifikan

**Tanggal:** 2026-08-15
**Hipotesis:** menambahkan monocular-depth sebagai kanal kelima di atas RGB +
depth sensor menaikkan deteksi dibandingkan RGB+Depth 4 kanal (sel 2). Ini sel
terakhir matriks mono-depth, dan satu-satunya yang tuntas **60 epoch penuh**.

**Data:** `/workspace/d352_rgbedgemono`, TIFF 5 kanal `[B,G,R,edge,mono]`,
disimpan sebagai 5 halaman satu-kanal (`cv2.imwritemulti`) karena `cv2.imwrite`
menolak 5 kanal. Split kanonik 980/208/220 citra, 410 kotak test. Stem model
diverifikasi `(64, 5, 3, 3)`.

**Training:** 60/60 epoch tuntas, batch 4 utuh (nol `Reducing to batch`).
`best.pt` = **ep50**, val mAP50 0,4281.

### Val — sel 4 memuncaki SEMUA sel di 352

| Sel | Input | ch | val puncak | epoch |
|---|---|---|---|---|
| **4** | **RGB+Depth+Mono** | **5** | **0,4281** | **ep50** |
| 1 | RGB | 3 | 0,4111 | ep45 |
| 3 | RGB+Mono | 4 | 0,3888 | ep41 |
| 2 | RGB+Depth `edge` | 4 | 0,3856 | ep38 |

Sel 4 juga satu-satunya dari empat run 352 yang **naik** saat `close_mosaic`
menyala di ep51 — puncaknya justru tercapai di ep50, sementara tiga run lain
melandai turun di fase itu.

### Test (pycocotools, 220 citra, 410 kotak) — urutannya TERBALIK lagi

| Sel | Input | ch | test mAP50 | mAP50-95 |
|---|---|---|---|---|
| 2 | RGB+Depth `edge` | 4 | **0,4270** | — |
| 3 | RGB+Mono | 4 | 0,3943 | 0,1360 |
| **4** | **RGB+Depth+Mono** | **5** | **0,3766** | 0,1290 |
| 1 | RGB | 3 | 0,3677 | — |

Peringkat val (4 > 1 > 3 > 2) kembali hampir kebalikan peringkat test
(2 > 3 > 4 > 1). Ini pembalikan **keempat** berturut-turut di split 352 dan
menutup kasusnya: **val 208 citra tidak boleh dipakai memeringkat model di
dataset ini, titik.** Sel 4 memuncaki val dan tetap kalah dari sel 2 di test.

Per kelas AP50 sel 4: B1 0,7014 / B2 0,4560 / B3 0,2138 / B4 0,1351.

### CI bootstrap berpasangan (2.000 ulangan, seed 42)

| Perbandingan | Selisih | CI 95% | P(Δ>0) | Signifikan |
|---|---|---|---|---|
| sel 4 − sel 2 (mono di atas depth) | **−0,0504** | [−0,1038; −0,0015] | 0,022 | **ya** |
| sel 4 − sel 3 (5 kanal vs 4 kanal mono) | −0,0177 | [−0,0672; +0,0323] | 0,243 | tidak |

Signifikansinya tipis — batas atas CI −0,0015, nyaris menyentuh nol — jadi
sebaiknya dibaca sebagai "bukti cukup kuat untuk menolak bahwa mono membantu di
atas depth", bukan sebagai pengukuran presisi atas besarnya kerugian.

**Sumber:** `results/eval_sel4_352_rgbedgemono_test.json`,
`results/pred_sel4_352_rgbedgemono_test.npz`, `results/boot_sel4_vs_sel2.json`,
`results/boot_sel4_vs_sel3.json`, `results/riwayat_epoch/sel4_*`.

**Verdict:** hipotesis **ditolak**. Mono tidak menambah apa pun di atas depth
sensor; ia mengurangi −0,0504, dan itu signifikan meski di split yang cuma 410
kotak. Kanal kelima bukan cuma sia-sia, ia mengencerkan sinyal yang sudah
dibawa kanal depth.

---

## V2-E-032 — Matriks mono-depth lengkap: mono tidak pernah menang, dan dua kali kalah signifikan

**Tanggal:** 2026-08-15
**Ringkasan enam sel.** Semua memakai resep identik (`yolo26l.pt` COCO init,
60 epoch, batch 4, imgsz 1280, seed 42, `cos_lr`), evaluator pycocotools pada
split test, dump prediksi `.npz` disimpan saat evaluasi.

| # | Dataset | Input | ch | test mAP50 | Epoch dijalankan |
|---|---|---|---|---|---|
| 1 | 352 | RGB | 3 | 0,3677 | 60 |
| 2 | 352 | RGB+Depth `edge` | 4 | **0,4270** | 60 |
| 3 | 352 | RGB+Mono | 4 | 0,3943 | 54 (dihentikan) |
| 4 | 352 | RGB+Depth+Mono | 5 | 0,3766 | 60 |
| 5 | 953 | RGB | 3 | **0,5436** | 60 |
| 6 | 953 | RGB+Mono | 4 | 0,4960 | 31 (dihentikan) |

**Semua perbandingan berpasangan:**

| Perbandingan | Selisih | CI 95% | Signifikan |
|---|---|---|---|
| sel 6 − sel 5 — mono vs RGB, 953 | **−0,0476** | [−0,0671; −0,0274] | **YA** |
| sel 4 − sel 2 — mono di atas depth, 352 | **−0,0504** | [−0,1038; −0,0015] | **YA** |
| sel 3 − sel 2 — mono vs depth, 352 | −0,0327 | [−0,0756; +0,0074] | tidak |
| sel 4 − sel 3 — 5ch vs 4ch mono, 352 | −0,0177 | [−0,0672; +0,0323] | tidak |
| sel 3 − sel 1 — mono vs RGB, 352 | +0,0266 | [−0,0270; +0,0739] | tidak |

**Kesimpulan: monocular-depth tidak pernah menang secara signifikan di satu
pun dari lima perbandingan, dan kalah signifikan di dua.** Satu-satunya selisih
positifnya (+0,0266, sel 3 vs sel 1) tidak signifikan dan lebih kecil daripada
lebar CI-nya sendiri.

**Depth sensor tetap kanal keempat terbaik.** Sel 2 (0,4270) mengungguli sel 3
(0,3943) dan sel 4 (0,3766). Mono mereproduksi struktur yang sama dengan sensor
tapi lebih lemah (Spearman dalam kotak 0,676; relief B1->B4 −4,08 cm vs sensor
−5,14 cm), dan pelemahan itu tampaknya cukup untuk membalik manfaatnya jadi
kerugian.

**Dua batas yang harus ikut dikutip:**

1. **Daya statistik split 352 tidak memadai.** 410 kotak memberi lebar CI
   ~0,10; selisih di bawah ~0,06 memang tidak bisa dibedakan dari nol. Tiga
   dari lima perbandingan di atas berada di zona itu. Hanya sel 6 vs sel 5
   (2.612 kotak, lebar CI 0,050) yang punya daya memadai.
2. **Sel 3 dan sel 6 dihentikan lebih awal** (54 dan 31 dari 60 epoch) atas
   keputusan pengguna setelah kurva val menunjukkan arah yang jelas. Sel 6
   paling terdampak: pembandingnya memuncak di ep34, di luar jangkauan run itu,
   sehingga −0,0476 kemungkinan dilebih-lebihkan. Arahnya tidak diragukan
   (nol dari 2.000 ulangan bootstrap positif), besarannya diragukan.

**Catatan angka sel 1 dan sel 2 — jangan bingung dengan STATUS.md.** Tabel di
atas memakai estimasi titik dari *resampler* bootstrap (0,3677 dan 0,4270),
bukan dari pycocotools (0,3711 dan 0,4316 di STATUS.md / V2-E-010/011).
Selisih ~0,004 itu murni beda implementasi mAP antar-evaluator, bukan model
atau data yang berbeda: keduanya membaca `.npz` prediksi yang sama
(`pred_rgb352_test.npz`, `pred_edge_test.npz`). Semua selisih dan CI di tabel
perbandingan dihitung di dalam satu evaluator yang sama, jadi internal
konsisten — tapi **jangan campur** angka pycocotools dengan angka bootstrap
dalam satu pengurangan.

**Yang belum terjawab, dan sengaja tidak ditebak:** apakah kerugian mono
berasal dari isi petanya atau dari biaya menambah kanal pada stem COCO 3-kanal.
Kontrol M_shuf lintas-pohon memisahkan keduanya dan belum dijalankan.

---

## V2-E-033 — Dua kebocoran split yang membatasi cara membaca angka lama

**Tanggal:** 2026-08-15
**Konteks:** dua temuan sampingan yang muncul saat menelusuri daya statistik
matriks mono-depth. Keduanya **tidak** mengubah satu pun angka yang sudah
tercatat, tapi mengubah cara angka-angka itu boleh dikutip. Diverifikasi
langsung dari berkas split, bukan dari ingatan.

### 1. Pretraining agnostik Fase 6 bocor ke `test_penuh`

Split pretraining `agnostic953` (train 3.200 + val 364 = 3.564 citra,
846 pohon) berpotongan besar dengan split evaluasi `agnostic953_test_penuh`:

| Himpunan uji | Citra | Pohon | Citra bocor | Pohon bocor |
|---|---|---|---|---|
| `test_penuh` | 588 | 141 | **512/588 (87%)** | **122/141 (87%)** |
| `test_bersih` | 76 | 19 | **0/76** | **0/19** |

Jadi 87% citra `test_penuh` **secara harfiah ikut dilatih** saat pretraining
agnostik — bukan cuma pohon yang sama dari sudut lain, tapi berkas citra yang
identik. Angka apa pun dari `test_penuh` untuk model yang melewati pretraining
agnostik adalah angka **train-on-test** dan tidak boleh dikutip sebagai
performa generalisasi.

`test_bersih` (76 citra, 19 pohon) benar-benar bersih dan memang dibuat untuk
alasan ini. Itu satu-satunya himpunan yang sah untuk menilai jalur agnostik —
dengan konsekuensi 19 pohon terlalu sedikit untuk CI yang berguna.

Perbandingan yang dilaporkan di V2-E-0xx Fase 6 memakai `pred_agn953_bersih.npz`
maupun `pred_agn953_penuh.npz`; yang boleh dibaca sebagai hasil hanya yang
`bersih`.

### 2. 44 dari 55 pohon test-352 ada di dalam train-953

Split kanonik 953 (`SawitMVC-YOLO`, train 716 pohon) memuat **44 dari 55
pohon** di split test 352 (`SawitMVC-Depth-YOLO/test`).

Ini **tidak** mencemari matriks mono-depth: keenam sel dilatih dari
`yolo26l.pt` COCO, bukan dari bobot yang pernah melihat 953, jadi sel 1-4
tidak pernah bersinggungan dengan train-953. Yang tercemar adalah **rantai
transfer apa pun yang memakai bobot 953 sebagai inisialisasi untuk model 352** —
di situ 80% pohon test-352 sudah pernah dilihat. Kalau nanti ada eksperimen
finetune 953→352, hasilnya wajib dilaporkan dengan catatan ini, atau memakai
subset 11 pohon yang bersih (yang lagi-lagi terlalu kecil untuk CI).

**Verifikasi:** kedua angka dihitung dengan mencocokkan identitas pohon
(`DAMIMAS_A21B_<id>`, sufiks nomor tampilan dibuang) langsung dari isi
`splits/*.txt` dan direktori `images/`, 2026-08-15.

**Verdict:** tidak ada angka lama yang ditarik, tapi dua pembatas kutipan
ditambahkan: (a) hasil agnostik hanya sah dari `test_bersih`; (b) transfer
953→352 tidak punya split test yang bersih.

---

## V2-E-034 — Baseline seed-42 pada rilis SawitMVC-Depth-YOLO v2.0.0 (763 pohon): urutan RF-DETR-L > RT-DETR-L > YOLO26l bertahan, tapi budget training tidak setara

**Tanggal:** 2026-08-22
**Konteks.** Rilis dataset baru `SawitMVC-Depth-YOLO` v2.0.0 menggabungkan tiga
kampanye akuisisi (DAMIMAS, MARIHAT, TOPAZ) jadi 763 pohon dengan split bawaan
536/117/110 pohon (lihat `docs/NEW763_BASELINE.md` untuk resep lengkap). Ini
baseline pertama pada rilis ini, seed 42 saja — seed 1337 dan 2026 dibatalkan
atas keputusan pengguna untuk memprioritaskan campaign lain
(`combined1716`, lihat V2-E-035 kalau sudah ditulis).

**Resep.** RGB, COCO-pretrained, resolusi 1280, batch 4, deterministic,
`cos_lr`. YOLO26l dan RT-DETR-L: maksimum 60 epoch/patience 15. RF-DETR-L:
maksimum 20 epoch/patience 5 (sengaja lebih pendek, berdasar temuan lama
bahwa RF-DETR overfit dini di korpus kecil — lihat `docs/NEW763_BASELINE.md`).
Evaluator `pycocotools.COCOeval`, prediksi val+test didump ke `.npz` saat
evaluasi, riwayat per-epoch disalin ke `results/riwayat_epoch_new763/`.

**Hasil keseluruhan (test):**

| Model | Epoch aktual (dari budget) | Test mAP50 | Test mAP50-95 |
|---|---|---|---|
| RF-DETR-L | 14/20 (early-stop) | **0,6129** | **0,2335** |
| RT-DETR-L | 50/60 (early-stop) | 0,5580 | 0,2055 |
| YOLO26l | 55/60 (early-stop) | 0,5163 | 0,1906 |

Urutan sama dengan angka lama E-021 di `CLAUDE.md` (RF-DETR-L > RT-DETR-L >
YOLO26l), dan RF-DETR-L kembali jadi detektor terbaik.

**Stratifikasi kampanye (test mAP50), wajib per catatan README dataset:**

| Model | DAMIMAS (n=52 pohon) | MARIHAT (n=11 pohon) | TOPAZ (n=47 pohon) |
|---|---|---|---|
| RF-DETR-L | 0,4460 | 0,5182 | 0,6369 |
| RT-DETR-L | 0,4366 | 0,5380 | 0,5494 |
| YOLO26l | 0,4019 | 0,4179 | 0,5044 |

Ketiga model konsisten terlemah di DAMIMAS dan terkuat di TOPAZ. RF-DETR-L
menang di DAMIMAS dan TOPAZ tapi RT-DETR-L sedikit lebih baik di MARIHAT
(0,5380 vs 0,5182) — MARIHAT cuma 11 pohon/44 citra test, jadi selisih ini
kemungkinan besar di dalam noise, belum dihitung CI-nya.

**Kaveat penting — budget training TIDAK setara.** Ketiganya early-stop,
tapi RF-DETR-L berhenti di 70% dari budget-nya (14/20) sementara YOLO26l dan
RT-DETR-L masing-masing di 92% (55/60) dan 83% (50/60) dari budget mereka.
Lebih penting lagi: jadwal `cos_lr` RF-DETR-L didesain untuk 20 epoch, jadi
LR-nya sudah habis meluruh di epoch 14 — sementara jadwal YOLO/RT-DETR
didesain untuk 60 epoch dan baru berhenti di 50-55. Ini bukan perbandingan
"tiga model dilatih sama lama lalu dibandingkan"; ini "tiga model dilatih
sampai konvergen menurut jadwal masing-masing yang sengaja beda". Urutan
akhirnya kemungkinan tetap valid (konsisten dengan E-021 lama yang memakai
resep berbeda pula), tapi keunggulan RF-DETR-L di atas tidak boleh dibaca
sebagai "menang meski dilatih lebih singkat" tanpa catatan ini.

**Diagnosis performa infrastruktur (tidak mengubah angka, tapi menjelaskan
kenapa training RF-DETR terasa lambat):** RF-DETR-L CPU-bound, bukan
GPU-bound — lihat `CLAUDE.md` bagian "RF-DETR CPU-bound, YOLO/RT-DETR tidak"
untuk bukti (GPU util 0-1% vs YOLO 19-100%, CPU proses utama 1613% vs 57%).
Diukur langsung di RTX 4090 saat run RF-DETR-L seed 42 di atas sedang jalan.

**Yang belum dikerjakan:** replikasi seed 1337/2026 (dibatalkan, lihat di
atas), CI berpasangan antar-arsitektur, dan counting end-to-end untuk ketiga
detektor pada rilis v2.0.0 ini (angka counting yang ada di `CLAUDE.md` masih
dari YOLO26m rilis lama).

**Sumber:** `results/new763/{rfdetr,rtdetr,yolo26l}_l?_rgb_s42_i1280.json`
(pycocotools), `results/new763_campaigns.json` (stratifikasi kampanye),
`results/new763_summary.json` (agregat), `results/riwayat_epoch_new763/`
(riwayat per-epoch), manifest `results/new763/matrix_manifest.json`.

---

## V2-E-035 — Baseline seed-42 pada korpus gabungan SawitMVC-Combined-1716-RGB: RF-DETR-L tetap terbaik

**Tanggal:** 2026-08-23
**Konteks.** Korpus baru menggabungkan dataset lama 953 pohon (`SawitMVC-YOLO`)
dan dataset Depth 763 pohon (`SawitMVC-Depth-YOLO`) jadi satu: **1.716 tree
record / 7.044 gambar**, dengan **352 tree-ID yang sama di kedua sumber**
sehingga group pohon unik cuma **1.364**. Nama file diberi prefix `SAWIT_`
dan `DEPTH_` untuk mencegah tabrakan; dataset asli tidak disentuh. Split
group-safe: **train 5.184 / val 808 / test 1.052 citra**, tidak ada group
pohon yang menyeberang split (mencegah kebocoran seperti yang didokumentasikan
di V2-E-033).

**Resep.** Sama seperti V2-E-034: RGB, COCO-pretrained, resolusi 1280,
batch 4, deterministic, `cos_lr`. Ketiga model kali ini **budget training
disamakan**: 60 epoch/patience 15 untuk semua tiga arsitektur (RF-DETR-L
sebelumnya dibatasi 20/5 di V2-E-034 — diperbaiki di sini atas permintaan
pengguna supaya perbandingan lebih adil).

**Hasil (test), ketiganya selesai:**

| Model | Epoch aktual | Test mAP50 | Test mAP50-95 |
|---|---|---|---|
| **RF-DETR-L** | 24/60 (early-stop otomatis) | **0,5960** | **0,2522** |
| RT-DETR-L | 43/60 (dihentikan manual) | 0,5745 | 0,2458 |
| YOLO26l | 51/60 (early-stop otomatis) | 0,5389 | 0,2395 |

Urutan **RF-DETR-L > RT-DETR-L > YOLO26l** — identik dengan V2-E-034 (new763)
dan E-021 lama. Konsisten di dua korpus berbeda dengan protokol training yang
kini setara (60 epoch/patience 15 untuk ketiganya).

**Catatan penghentian RT-DETR-L.** Bukan early-stop otomatis Ultralytics —
dihentikan manual atas keputusan pengguna di epoch 43/60 setelah plateau
14 epoch tanpa perbaikan sejak best di epoch 29 (patience terkonfigurasi 15,
nyaris habis). `best.pt` Ultralytics sudah otomatis menunjuk ke checkpoint
epoch 29 (val mAP50-95 0,2447) — checkpoint yang dievaluasi di sini identik
dengan yang akan tersimpan seandainya early-stop resmi terjadi di epoch ~44.
Jadi angka di atas bukan hasil training yang dipotong prematur, hanya
penghentian observasi lebih awal dari titik konvergensinya.

**Catatan operasional yang berpengaruh ke validitas run, bukan ke angka:**
1. **Bug path relatif Ultralytics.** `--project` yang diberikan sebagai path
   relatif menyebabkan Ultralytics diam-diam menaruh output di
   `runs/detect/<project>/...`, bukan di `<project>/...` yang diharapkan
   skrip orkestrasi (`run_combined1716_matrix.py`). Ini membuat proses
   finalisasi YOLO26l crash (exception tak tertangani saat menulis
   `baseline_args.json`) dan menyeret mati seluruh proses runner sebelum
   sempat menjadwalkan RT-DETR-L. **Training YOLO26l sendiri sukses penuh**
   (val mAP50 0,548 di epoch 51) — hanya langkah finalisasi yang gagal.
   Diperbaiki dengan memindah direktori run secara manual dan menjalankan
   eval langsung; RT-DETR-L kemudian di-start manual dengan path absolut.
   **Pelajaran:** selalu pakai path absolut untuk `--project` saat
   menjalankan `train_baseline_new763.py` di luar skrip matrix bawaannya.
2. **Runner otomatis (`run_combined1716_matrix.py`) tidak dipakai lagi**
   setelah crash di atas — sesuai aturan repo (runner yang gagal sekali
   tidak diperbaiki lagi untuk langkah itu), sisa orkestrasi (start RT-DETR,
   eval RF-DETR setelah early-stop) dijalankan manual.

**Sumber:** `results/combined1716/{combined1716_yolo26l,combined1716_rfdetr_l}_rgb_s42_i1280.json`
(pycocotools), `results/combined1716/campaign_manifest.json`,
`results/combined1716/predictions/` (dump `.npz` val+test), log training di
`results/combined1716/logs/`.

---

## V2-E-036 — Rekor AP50 class-agnostic baru (0,7951) dari model sesi ini, dihitung tanpa re-inferensi

**Tanggal:** 2026-08-23
**Konteks.** V2-E-013/017/025 menunjukkan mAP50 tertinggi di project ini
selalu datang dari deteksi **class-agnostic** (kelas dilipat jadi 1
"tandan", murni lokalisasi), jauh di atas mAP50 4-kelas B1-B4. Pertanyaan:
apakah pola itu bertahan pada enam model baru sesi ini (V2-E-034/035,
new763 dan combined1716)?

**Metode.** `scripts/eval_agnostic_from_npz.py`. **Tidak ada inferensi ulang
dan tidak butuh GPU** — script memuat dump `.npz` prediksi test yang sudah
disimpan saat eval V2-E-034/035, melipat kategori GT dan prediksi jadi satu
kelas, lalu menghitung ulang AP50 lewat `pycocotools.COCOeval` dari nol.
Reproduksi: `python3 scripts/eval_agnostic_from_npz.py` (hasil lengkap +
metadata tersimpan di `results/agnostic_ap50_sesi2026-08.json`).

**Hasil (test, class-agnostic):**

| # | Model | Korpus | AP50 agnostik | AP50-95 agnostik | n_images |
|---|---|---|---|---|---|
| 1 | RF-DETR-L | new763 | **0,7951** | 0,3003 | 440 |
| 2 | RF-DETR-L | combined1716 | 0,7850 | 0,3245 | 1.052 |
| 3 | RT-DETR-L | new763 | 0,7712 | 0,2801 | 440 |
| 4 | RT-DETR-L | combined1716 | 0,7577 | 0,3168 | 1.052 |
| 5 | YOLO26l | combined1716 | 0,7250 | 0,3156 | 1.052 |
| 6 | YOLO26l | new763 | 0,7161 | 0,2580 | 440 |

**Verdict: CONFIRMED, dan ini rekor AP50 tertinggi baru di seluruh project.**
0,7951 (RF-DETR-L, new763) mengalahkan seluruh angka agnostik lama yang sah:
- 0,7702 — V2-E-025, test bersih 953, tapi cuma N=19 pohon/316 kotak (CI
  sangat lebar, "indikasi bukan pengukuran presisi").
- 0,7374 — V2-E-017, plafon lama di split kanonik 953 (test lengkap 141
  pohon), model v2repro lama.
- (0,8101 val / 0,8090 test-penuh dari V2-E-025 **BUKAN pembanding yang
  sah** — sudah ditarik karena kebocoran pretraining, lihat V2-E-025/033.
  Jangan disandingkan dengan angka di entry ini.)

Angka baru ini dievaluasi di split test kanonik **440 citra new763**
(bukan subset kecil, bukan tercemar) — jadi lebih dipercaya daripada
0,7702 sekaligus lebih tinggi.

**Pola yang bertahan (konsisten dengan V2-E-013).** Urutan arsitektur di
agnostik **identik** dengan class-aware (RF-DETR-L > RT-DETR-L > YOLO26l) di
kedua korpus, tapi jarak mengecil drastis: gap class-aware new763
(0,6129 vs 0,5163 = 0,0966) menyusut jadi 0,079 di agnostik (0,7951 vs
0,7161). Konsisten dengan temuan lama: sebagian besar kekalahan mAP50
4-kelas berasal dari **salah kelas pada kotak yang sudah benar**, bukan
gagal mendeteksi.

**Kaveat.** new763 (440 citra test) dan combined1716 (1.052 citra test)
punya ukuran dan komposisi kampanye berbeda (lihat V2-E-034/035) — tabel di
atas bukan perbandingan langsung satu populasi, melainkan dua pengukuran
plafon lokalisasi yang terpisah per korpus.

**Sumber:** `scripts/eval_agnostic_from_npz.py`,
`results/agnostic_ap50_sesi2026-08.json`, enam `.npz` prediksi test yang
sudah ter-commit di V2-E-034/035.

---

## V2-E-037 — Confusion analysis pada 6 model sesi ini: kehilangan ke salah-kelas jauh lebih kecil dari V2-E-013 (44,5%)

**Tanggal:** 2026-08-23
**Konteks.** V2-E-013 (2026-08-11, model lama RGB-352) menunjukkan 44,5% dari
kegagalan mAP50 class-aware berasal dari salah kelas pada kotak yang sudah
benar lokasinya, bukan gagal deteksi. Pertanyaan: apakah proporsi ini
bertahan pada enam model baru sesi ini?

**Metode.** `scripts/eval_confusion_from_npz.py` — replikasi persis
metodologi V2-E-013. Untuk tiap kotak GT, dicari prediksi dengan skor >=0,25
yang IoU-nya >=0,5 terhadap kotak itu (pencocokan class-agnostic, greedy per
skor tertinggi, satu prediksi dan satu GT hanya dipakai sekali). Kelas GT vs
kelas prediksi ditabulasi jadi confusion matrix. **Tidak re-infer** — pakai
dump `.npz` test yang sama dengan V2-E-034/035/036. Angka "hilang karena
salah kelas" dihitung terpisah, murni aritmetika dari angka yang sudah
tercatat (AP50 agnostik V2-E-036 dikurangi mAP50 class-aware V2-E-034/035),
tanpa komputasi baru.

**Hasil 1 — hilang karena salah kelas (aritmetika dari V2-E-034/035/036):**

| Model | Korpus | mAP50 class-aware | AP50 agnostik | Hilang | % dari plafon |
|---|---|---|---|---|---|
| YOLO26l | new763 | 0,5163 | 0,7161 | 0,1998 | 27,9% |
| RT-DETR-L | new763 | 0,5580 | 0,7712 | 0,2132 | 27,7% |
| **RF-DETR-L** | new763 | 0,6129 | 0,7951 | 0,1822 | **22,9%** |
| YOLO26l | combined1716 | 0,5389 | 0,7250 | 0,1861 | 25,7% |
| RT-DETR-L | combined1716 | 0,5745 | 0,7577 | 0,1832 | 24,2% |
| RF-DETR-L | combined1716 | 0,5960 | 0,7850 | 0,1890 | 24,1% |

**Semua enam model kehilangan 23-28% ke salah-kelas — jauh lebih rendah
dari 44,5% di V2-E-013.** Bukan perbandingan apel-ke-apel (dataset, model,
protokol training berbeda), jadi ini bukan bukti "klasifikasi membaik 2x
lipat" — kemungkinan besar dataset baru (new763/combined1716, campuran tiga
kampanye) punya distribusi kematangan yang lebih mudah dipisahkan daripada
RGB-352 tunggal yang dipakai V2-E-013. Perlu perbandingan langsung dengan
protokol sama untuk klaim yang lebih kuat.

**Hasil 2 — confusion matrix representatif (RF-DETR-L, new763, model
terbaik, IoU>=0,5, conf>=0,25):**

| GT\\Pred | →B1 | →B2 | →B3 | →B4 |
|---|---|---|---|---|
| B1 | 113 | 45 | 1 | 0 |
| B2 | 10 | 238 | 16 | 4 |
| B3 | 3 | 50 | 214 | 8 |
| B4 | 0 | 6 | 22 | 15 |

Akurasi klasifikasi bersyarat (di antara 745 box yang terdeteksi dari 891
GT): **77,85%**. Akurasi atas seluruh GT (termasuk gagal deteksi): **65,10%**.
Pola ordinal dari V2-E-013 bertahan: nyaris nol kesalahan B1↔B4 (lompat dua
tingkat), mayoritas kesalahan ke kelas bertetangga.

**Recall per kelas (bersyarat pada terdeteksi), keenam model:**

| Model | Korpus | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|
| YOLO26l | new763 | 77,8% | 81,3% | 85,3% | 34,6% |
| RT-DETR-L | new763 | 74,2% | 76,2% | 75,1% | 56,8% |
| RF-DETR-L | new763 | 71,1% | 88,8% | 77,8% | 34,9% |
| YOLO26l | combined1716 | 79,6% | 56,9% | 89,8% | 53,1% |
| RT-DETR-L | combined1716 | 68,8% | 58,6% | 87,5% | 57,8% |
| RF-DETR-L | combined1716 | 79,7% | 58,0% | 85,8% | 59,4% |

**B4 konsisten paling lemah di new763** (34,6-56,8%) untuk semua tiga
arsitektur — pola sistematis, bukan kebetulan satu model. Di combined1716
B4 justru lebih baik (53-59%) tapi B2 yang melemah (56,9-58,6%) — arah
kelemahan berbeda antar korpus, kemungkinan terkait komposisi kampanye yang
berbeda (lihat stratifikasi kampanye di V2-E-034/035).

**Verdict: CONFIRMED (pola V2-E-013 bertahan), tapi besaran gap berbeda.**
Urutan konsisten: lokalisasi >> klasifikasi kematangan sebagai sumber
kegagalan utama, di kedua korpus baru dan keenam model. Kesalahan tetap
ordinal (nyaris nol lompat dua tingkat) di semua model yang diperiksa.

**Sumber:** `scripts/eval_confusion_from_npz.py`,
`results/confusion_analysis_sesi2026-08.json` (confusion matrix lengkap
keenam model), angka pembanding dari V2-E-013/034/035/036.

---

## V2-E-038 — Bootstrap CI mAP50: urutan RF-DETR-L > RT-DETR-L > YOLO26l SIGNIFIKAN di kedua korpus

**Tanggal:** 2026-08-23
**Konteks.** V2-E-023 (2026-08-12) menunjukkan lebar CI bootstrap bisa jauh
lebih besar dari selisih titik estimasi antar-konfigurasi, membuat urutan
Fase 6 tidak terbedakan secara statistik. Pertanyaan: apakah urutan
RF-DETR-L > RT-DETR-L > YOLO26l di V2-E-034/035 juga sekadar derau, atau
memang signifikan?

**Metode.** `scripts/bootstrap_map_from_npz.py` — replikasi metodologi
V2-E-023: resampling **citra** (bukan kotak) dengan pengembalian, 500
replikasi, seed 42, **berpasangan** (sampel citra yang sama dipakai untuk
ketiga arsitektur dalam satu korpus, supaya korelasi antar-model tidak
menggelembungkan selang selisihnya). mAP50 dihitung ulang dari nol tiap
replikasi (rata-rata makro 4 kelas, AP50 gaya COCO interpolasi 101 titik).
**Tidak re-infer** — dari dump `.npz` test V2-E-034/035. Titik estimasi dari
implementasi sendiri sedikit berbeda dari pycocotools (mis. 0,5163 vs 0,5163
new763-yolo, 0,558 vs 0,5580 new763-rtdetr) — selisih <0,0004, sama seperti
yang sudah divalidasi di V2-E-013, bukan bug.

**Hasil — CI95 per model:**

| Korpus | Model | mAP50 | CI95 | Lebar CI |
|---|---|---|---|---|
| new763 (440 citra, 891 kotak) | YOLO26l | 0,5163 | [0,4853; 0,5572] | 0,0719 |
| new763 | RT-DETR-L | 0,5580 | [0,5261; 0,6067] | 0,0806 |
| new763 | RF-DETR-L | 0,6129 | [0,5788; 0,6614] | 0,0826 |
| combined1716 (1.052 citra, 3.513 kotak) | YOLO26l | 0,5389 | [0,5204; 0,5611] | 0,0407 |
| combined1716 | RT-DETR-L | 0,5746 | [0,5558; 0,5984] | 0,0426 |
| combined1716 | RF-DETR-L | 0,5960 | [0,5780; 0,6208] | 0,0428 |

**Hasil — selisih berpasangan, keduanya korpus:**

| Perbandingan | Δ titik | CI95 Δ | P(Δ>0) | Signifikan |
|---|---|---|---|---|
| new763: YOLO − RT-DETR | −0,0417 | [−0,0747; −0,0144] | 0,002 | **YA** |
| new763: YOLO − RF-DETR | −0,0966 | [−0,1269; −0,0662] | 0,000 | **YA** |
| new763: RT-DETR − RF-DETR | −0,0549 | [−0,0826; −0,0261] | 0,000 | **YA** |
| combined1716: YOLO − RT-DETR | −0,0357 | [−0,0517; −0,0219] | 0,000 | **YA** |
| combined1716: YOLO − RF-DETR | −0,0571 | [−0,0721; −0,0420] | 0,000 | **YA** |
| combined1716: RT-DETR − RF-DETR | −0,0214 | [−0,0377; −0,0064] | 0,004 | **YA** |

**Verdict: CONFIRMED — SEMUA ENAM perbandingan berpasangan signifikan pada
α=0,05, di kedua korpus.** Berbeda dari V2-E-023 (Fase 6, semua konfigurasi
TIDAK terbedakan): urutan tiga arsitektur di sini nyata, bukan derau. Lebar
CI di combined1716 (~0,04) jauh lebih sempit dari new763 (~0,08) — konsisten
dengan jumlah kotak GT yang 3,9x lebih banyak (3.513 vs 891), memberi daya
statistik lebih tinggi.

**Kaitan dengan V2-E-023.** Perbedaan hasil ini BUKAN kontradiksi metodologi:
V2-E-023 menguji konfigurasi *dalam satu arsitektur* (variasi resep 4-6
kanal) dengan selisih titik kecil (~0,004-0,01) pada split kecil (410 kotak);
di sini yang dibandingkan adalah *arsitektur berbeda* dengan selisih jauh
lebih besar (0,02-0,10) pada split yang jauh lebih besar (891-3.513 kotak).
Daya statistik dan besaran efek sama-sama mendukung signifikansi di sini.

**Sumber:** `scripts/bootstrap_map_from_npz.py`,
`results/bootstrap_map_sesi2026-08.json`.

---

## V2-E-039 — Precision/Recall/F1 standar + sweep threshold + WBF ensemble: rekor AP50 agnostik baru (0,8106), tapi WBF menurunkan mAP50 class-aware

**Tanggal:** 2026-08-23
**Konteks.** Melengkapi V2-E-034/035/037 dengan metrik standar yang belum
dihitung: Precision/Recall/F1 per kelas pada satu ambang confidence (bukan
recall bersyarat V2-E-037), titik operasi optimal per model, dan replikasi
WBF ensemble V2-E-019 (yang dulu cuma diuji pada detektor agnostik) ke
skenario class-aware 4-kelas.

**Metode.** `scripts/eval_extra_metrics_from_npz.py`, tiga bagian, semua dari
dump `.npz` test V2-E-034/035, **tanpa re-infer**:
1. P/R/F1 per kelas pada conf=0,25 (sama seperti ambang V2-E-013), pencocokan
   IoU≥0,5 **di dalam kelas yang sama** (beda dari V2-E-037 yang class-agnostic).
2. Sweep conf 0,05–0,95 (step 0,05) untuk cari titik macro-F1 terbaik.
3. WBF ensemble 3 detektor per korpus (fungsi `wbf` dari `eval_twostage.py`):
   fusi per-kelas untuk mAP50 class-aware, fusi lintas-kelas untuk AP50 agnostik.

**Hasil 1 — P/R/F1 @ conf=0,25 (macro) vs titik optimal dari sweep:**

| Model | Korpus | F1@0,25 | Conf optimal | F1 optimal |
|---|---|---|---|---|
| YOLO26l | new763 | 0,4967 | 0,20 | 0,5121 |
| RT-DETR-L | new763 | 0,4943 | 0,45 | 0,5676 |
| RF-DETR-L | new763 | 0,5790 | 0,35 | 0,6042 |
| YOLO26l | combined1716 | 0,5163 | 0,20 | 0,5407 |
| RT-DETR-L | combined1716 | 0,4745 | 0,45 | 0,5888 |
| RF-DETR-L | combined1716 | 0,5536 | 0,35 | 0,6028 |

**Conf=0,25 BUKAN terlalu tinggi untuk RT-DETR/RF-DETR — titik optimalnya
malah lebih tinggi (0,35-0,45).** Kurva sweep menunjukkan di conf rendah
(0,05) recall tinggi (0,87-0,91) tapi precision hancur (0,10-0,13),
menjatuhkan F1 ke 0,19-0,22 — jauh dari optimal. Hanya YOLO26l yang
optimalnya sedikit di bawah 0,25 (0,20), selisih tipis.

**Hasil 2 — WBF ensemble (3 detektor digabung per korpus):**

| Korpus | mAP50 class-aware (ensemble) | AP50 agnostik (ensemble) | AP50 agnostik terbaik tunggal |
|---|---|---|---|
| new763 | 0,5631 | 0,8039 | 0,7951 (RF-DETR) |
| combined1716 | 0,5538 | **0,8106** | 0,7850 (RF-DETR) |

**0,8106 adalah rekor AP50 tertinggi baru di seluruh project** — melampaui
V2-E-036 (0,7951, RF-DETR tunggal) dan seluruh angka lama yang sah (0,7702
V2-E-025, 0,7374 V2-E-017). Diukur di split test kanonik penuh (1.052 citra
combined1716, 440 citra new763), bukan subset kecil, bukan tercemar.

**Temuan yang tidak terduga dan harus dicatat jujur: WBF MENURUNKAN mAP50
class-aware dibanding detektor tunggal terbaik.** 0,5631/0,5538 (ensemble)
lebih rendah dari RF-DETR sendirian (0,6129/0,5960). Ini **berlawanan** arah
dengan V2-E-019 (ensemble AGNOSTIK menang atas semua anggota tunggal) karena
skenarionya beda: V2-E-019 menggabung box tanpa peduli kelas; di sini fusi
dilakukan **per-kelas**, jadi tiga detektor yang menebak kelas berbeda untuk
objek fisik yang sama akan terpecah ke tiga kelompok kelas terpisah alih-alih
saling menguatkan — WBF class-aware naif memecah suara, bukan memperkuatnya.
Detektor terbaik (RF-DETR) tetap pilihan lebih baik daripada ensemble kalau
tugasnya klasifikasi kematangan, bukan cuma lokalisasi.

**Verdict: CONFIRMED untuk plafon lokalisasi (rekor baru), FALSIFIED untuk
manfaat WBF class-aware naif** — ensembling per-kelas butuh strategi lebih
cermat (mis. voting kelas terpisah dari fusi lokasi) kalau mau dipakai untuk
tugas 4-kelas, bukan sekadar WBF per-kelas independen.

**Sumber:** `scripts/eval_extra_metrics_from_npz.py`,
`results/extra_metrics_sesi2026-08.json`.

---

## V2-E-040 — Cross-dataset: model sesi ini gagal total ke domain 953 kalau tak pernah melihatnya, tapi urutan arsitektur ikut berubah

**Tanggal:** 2026-08-23
**Konteks.** Enam model sesi ini (V2-E-034/035) hanya pernah dievaluasi pada
split test dari korpus tempat mereka dilatih. Pertanyaan: bagaimana mereka
tampil di dua dataset lama yang tak pernah dilihat sama sekali saat
training — SawitMVC-YOLO (953 pohon, kampanye DAMIMAS+LONSUM) dan
SawitMVC-Depth v1.1.0 (352 pohon, DAMIMAS saja, diunduh dari revisi HF
sebelum digabung jadi v2.0.0/763 pohon)? **Tidak ada training ulang** — cuma
inferensi dengan bobot yang sudah ada.

**Metode.** `scripts/eval_new763_pycoco.py` dijalankan langsung (tanpa
runner) untuk 12 kombinasi (6 model × 2 target), split test saja. Dataset
352 diunduh dari commit HF `80dcbae` (v1.1.0, sebelum merge 763) karena
rilis v2.0.0 sudah menimpa split kanonik lama di tempat.

**Pemeriksaan kebocoran (WAJIB sebelum baca hasil) — dua dari empat
pasangan TERKONTAMINASI:**

| Pasangan | Pohon test | Tumpang tindih train+val | Status |
|---|---|---|---|
| new763 vs 352-test | 55 | **47/55 (85%)** | **TERKONTAMINASI** |
| combined1716 vs 352-test | 55 | **46/55 (84%)** | **TERKONTAMINASI** |
| new763 vs 953-test | 141 | 50/141 (35%), sesi capture beda ~80 hari | Bersih (lihat catatan) |
| combined1716 vs 953-test | 141 | **0/141 (0%)** | Bersih |

**Kenapa 352 tercemar:** dataset 352 versi mandiri (v1.1.0) adalah persis
korpus DAMIMAS Juli 2026 yang **kemudian digabung mentah-mentah** jadi
bagian dari `SawitMVC-Depth-YOLO` v2.0.0 (basis new763) — split v2.0.0
dihitung ulang dari nol, jadi pohon yang masuk *test* di 352 lama bisa saja
masuk *train* di new763. Ini rantai kontaminasi yang sama persis dengan
yang diperingatkan V2-E-033 ("353→953: 44 dari 55 pohon test-352 ada di
train-953") — cuma arahnya dibalik di sini. **Angka new763→352 dan
combined1716→352 di bawah TIDAK BOLEH dikutip sebagai bukti generalisasi.**

**Kenapa 953 (untuk new763) tetap sah dipakai meski tree-ID tumpang
tindih 35%:** V2-E-033 sudah membuktikan 953 dan 352 adalah **dua sesi
akuisisi berbeda ~80 hari**, kamera/resolusi berbeda (953: HP RGB
960×1280; sumber new763/352: sensor Depth 1280×800) — jadi pohon yang sama
difoto ulang dengan kondisi visual yang genuinely berbeda (buah matang
berubah, sudut, pencahayaan, kamera). Tumpang tindih identitas pohon bukan
tumpang tindih piksel.

**Hasil — test mAP50, dibandingkan dengan in-domain (V2-E-034/035):**

| Model | In-domain | → 953 (bersih) | → 352 (TERCEMAR, referensi saja) |
|---|---|---|---|
| new763 YOLO26l | 0,5163 | 0,2331 | ~~0,5572~~ |
| new763 RT-DETR-L | 0,5580 | **0,1110** | ~~0,6378~~ |
| new763 RF-DETR-L | 0,6129 | 0,1774 | ~~0,6072~~ |
| combined1716 YOLO26l | 0,5389 | 0,5402 | ~~0,5646~~ |
| combined1716 RT-DETR-L | 0,5745 | 0,5723 | ~~0,5729~~ |
| combined1716 RF-DETR-L | 0,5960 | **0,5894** | ~~0,6621~~ |

**Temuan 1 — combined1716 nyaris tidak kehilangan performa di 953, new763
runtuh total.** combined1716 (0,54-0,59, turun cuma 0,001-0,007 dari
in-domain) hampir tidak berbeda dari performa aslinya; new763 (0,11-0,23)
kehilangan 0,39-0,45 poin mAP50 — **runtuh ke 20-38% dari performa
in-domain-nya**. Penjelasannya bukan misteri: `combined1716` memasukkan
sebagian pohon 953 lain (bukan yang di split test-nya) ke training, jadi
model ini sudah pernah melihat domain kamera/resolusi 953 walau bukan
pohon spesifiknya. new763 tidak pernah sekalipun melihat domain itu.

**Temuan 2 — di bawah pergeseran domain, urutan arsitektur new763
TERBALIK dari in-domain.** In-domain: RF-DETR-L (0,6129) > RT-DETR-L
(0,5580) > YOLO26l (0,5163). Ke domain 953: **YOLO26l (0,2331) > RF-DETR-L
(0,1774) > RT-DETR-L (0,1110)** — RT-DETR-L, runner-up in-domain, jadi
**paling buruk** menggeneralisasi; YOLO26l, yang terlemah in-domain, jadi
**paling tangguh**. Untuk combined1716 urutan tetap sama (RF-DETR>RT-DETR>
YOLO) karena ketiganya sudah pernah melihat domain 953 saat training, jadi
bukan murni soal generalisasi arsitektur.
**Bacaan yang benar: model terbaik in-domain BUKAN jaminan model paling
robust ke domain baru** — kalau prioritasnya deployment ke kondisi capture
yang belum diketahui, kemampuan generalisasi (bukan cuma mAP50 in-domain)
harus diukur terpisah.

**Verdict: CONFIRMED untuk keduanya** — (a) komposisi data training
menentukan robustness lintas-domain jauh lebih kuat daripada pilihan
arsitektur; (b) ranking arsitektur in-domain tidak transitif ke ranking
generalisasi domain-shift.

**Sumber:** `results/cross_eval/*.json` (12 hasil), skrip pemeriksaan
kebocoran dijalankan interaktif (tidak disimpan sebagai file terpisah —
logikanya didokumentasikan di sini: cocokkan `tree_id` = nama berkas tanpa
suffix `_<nomor_sisi>`, prefix `SAWIT_`/`DEPTH_` dilucuti untuk
combined1716). Dataset 352 (revisi pre-merge): HF `ULM-DS-Lab/SawitMVC-Depth`
commit `80dcbae6ca5521515db84038dabc2ead96fa007e`.
