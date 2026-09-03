# Atlas Metrik: Evaluasi Sistem Pipeline Terpadu (*End-to-End*)

Dokumen ini memuat rangkuman performa evaluasi sistem pipeline multi-tahap terpadu yang mengintegrasikan seluruh tahapan kerja: **Lokalisasi Bebas-Kelas $\to$ Penaut Asosiasi Multi-Tampak $\to$ Klasifikasi Kematangan Tandan $\to$ Rekonsiliasi Cacah per Pohon**.

---

## 1. Panduan Pembacaan & Definisi Metrik

1. **Presisi / Recall / F1 Deteksi Fisik (*Physical Cluster Metrics*)**: Evaluasi ketepatan klaster spasial gabungan multi-sisi yang berhasil dipasangkan secara *one-to-one* dengan tandan buah fisik acuan riil pada pohon.
2. **Counting MAE**: Rata-rata galat absolut pencacahan jumlah tandan per pohon yang dihasilkan langsung oleh sistem pipeline.
3. **Tree $\pm 1$ Acc**: Persentase pohon dengan estimasi cacah yang meleset paling banyak 1 tandan.
4. **Matched Class Accuracy**: Akurasi klasifikasi tingkat kematangan 4 kelas yang dievaluasi **hanya pada klaster tandan fisik yang berhasil dipasangkan** dengan nilai acuan kebenaran.
5. **Macro-F1 End-to-End**: Nilai Macro-F1 klasifikasi kematangan di mana deteksi positif palsu (*false positive*) dan negatif palsu (*false negative*) ikut dihitung sebagai penalti.

---

## 2. Tabel Master Evaluasi Pipeline Terpadu (*End-to-End*)

| ID Simpul | Konfigurasi Sistem Pipeline | Dataset & Pohon Uji | Presisi Fisik | Recall Fisik | F1 Fisik | Counting MAE | Tree $\pm 1$ Acc | Matched Class Acc | Macro-F1 E2E | Status Bukti | Rujukan Artefak |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `V2-E-020` | Two-Stage (YOLO26l Edge + ResNet18) | 352 Test (55 pohon) | − | − | − | − | − | ⚠ TIDAK-BISA-DIVERIFIKASI | $mAP50=$**0,4500** *(bukan macro-F1)* | `VALID` | Entri asli hanya mengukur $mAP50$ deteksi 4-kelas, bukan pipeline asosiasi/klasifikasi; "Matched Class Acc" (`0,6415`) tidak ditemukan; kolom "Macro-F1 E2E" lama sebenarnya nilai $mAP50$. |
| `PT-E-003` | Pipeline Utuh Awal (Hungarian + C1) | DAMIMAS (Test, n=1.269 pool) | **0,3342** | **0,1200** | **0,1766** | ⚠ TBD | ⚠ TBD | **0,7124** *(R4, bukan "matched")* | ⚠ tidak diukur | `FALSIFIED` | `pipeline-pertandan/results/pt_e_003_endtoend.json` — Presisi/Recall/F1 fisik riil JAUH lebih rendah dari klaim lama (ruang deteksi penuh galat linker); Putusan asli DIPALSUKAN (G2 GUGUR: 0,7124 vs oracle 0,7360), bukan SUPERSEDED. |
| `PT-E-019` | Pipeline (penaut lama × ensemble, terbaik) | DAMIMAS (Test) | ⚠ tidak diukur | ⚠ tidak diukur | ⚠ tidak diukur | ⚠ tidak diukur | ⚠ tidak diukur | ⚠ tidak diukur | R4=**0,7311** *(bukan macro-F1)* | `FALSIFIED` | `pipeline-pertandan/results/pt_e_019_gabungan.json` — sumber asli HANYA mengukur R4 pada 4 konfigurasi, TIDAK memuat metrik fisik/counting/matched-class/macro-F1 sama sekali; Putusan asli DIPALSUKAN pada klaim berlipat, bukan SUPERSEDED. |
| `PT-E-025` | Global DAMIMAS 1-to-1 Pipeline | DAMIMAS (Test) | **0,8530** | **0,8116** | **0,8318** *(dihitung dari P/R)* | **1,638** | ⚠ TBD | **0,7322** | **0,5867** | `VALID` | `pipeline-pertandan/results/damimas_endtoend_global.json` (`.test.*`) — MAE lama `0,734` salah besar (riil `1,638`); seluruh nilai lain dikoreksi. |
| `V2-E-042` | Baseline Remote (WBF + Hungarian) | Depth Test (110 pohon) | **0,4705** | **0,8837** | **0,6140** | **4,518** | **18,18%** | ⚠ TBD | **0,4726** | `SUPERSEDED` | `experiments/EKSPERIMEN.md` (V2-E-043, baris Baseline) — seluruh nilai lama salah kecuali MAE (mendekati). |
| `V2-E-042` | Baseline Remote (WBF + Hungarian) | 953 Test (135 pohon) | **0,3725** | **0,9344** | **0,5327** | **14,993** | **0%** | ⚠ TBD | **0,3762** | `SUPERSEDED` | Idem — seluruh nilai lama salah; baseline 953 sangat buruk (Tree$\pm1$ 0%, bukan 4,44%). |
| `V2-E-043` | Optimized Greedy Pipeline | Depth Test (110 pohon) | **0,8799** | **0,8390** | **0,8590** | **0,818** | **83,64%** | ⚠ tidak dilaporkan | **0,6419** | `VALID` | `experiments/EKSPERIMEN.md` (V2-E-043) — Presisi/Recall/Macro-F1 dikoreksi; "Matched Class Acc" tidak ada dengan nama itu di entri ini. |
| `V2-E-043` | Optimized Greedy Pipeline | 953 Test (135 pohon) | **0,8247** | **0,8346** | **0,8296** | **1,644** | **54,07%** | ⚠ tidak dilaporkan | **0,5469** | `VALID` | Presisi/Recall/Macro-F1 dikoreksi. |
| `V2-E-044` | Pipeline + Classifier C2 25% Blend | Depth Test (110 pohon) | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TBD | ⚠ TBD | ⚠ TBD | ⚠ TBD | ⚠ TBD | ⚠ TBD | `VALID` | **Baris ini tidak ditemukan di sumber** — entri asli V2-E-044 HANYA melaporkan hasil pada 953 Test, tidak ada baris Depth sama sekali; kemungkinan disalin dari V2-E-043. |
| `V2-E-044` | Pipeline + Classifier Blend (WBF 75%+C2 25%) | 953 Test (135 pohon) | ⚠ tidak dilaporkan | ⚠ tidak dilaporkan | **0,8296** | **1,644** | **54,07%** | **0,7063** | **0,5469** | `FALSIFIED` (untuk penggantian penuh) | `results/remote_eval_2026-08-27/classifier_c2/` — Matched/Macro-F1 dikoreksi (`0,5910/0,5280`→`0,7063/0,5469`); Verdict asli: FALSIFIED untuk penggantian penuh C2-only (turunkan match −7,76pp), blend 25% dipertahankan sebagai kandidat *engineering* saja. |
| `V2-E-045` | Validation-Locked Pipeline | Depth Test (110 pohon) | ⚠ tidak dilaporkan | ⚠ tidak dilaporkan | **0,8069** | **0,891** | **80,91%** | **0,8031** | **0,6047** | `VALID` | `experiments/EKSPERIMEN.md` (STATUS.md §7) — seluruh nilai lama (`0,8450/.../83,64%/0,6710/0,6120`) sebenarnya milik V2-E-043 (*greedy/test-tuned*), bukan V2-E-045 (*validation-locked*). |
| `V2-E-045` | Validation-Locked Pipeline | 953 Test (135 pohon) | ⚠ tidak dilaporkan | ⚠ tidak dilaporkan | **0,8043** | **1,393** | **61,48%** | **0,7111** | **0,5384** | `VALID` | Idem — nilai lama tertukar dengan V2-E-043. |
| `Wave-V2` | Locked GSP Pipeline (profil terkunci: GSP) | Depth Test (110 pohon, *test-locked*) | **0,8926** | **0,8175** | **0,8534** | **0,7727** | **85,45%** | **0,8162** | **0,6519** | `VALID` | `results/remote_eval_2026-08-28/GSP_LINKER.md` — Matched/Macro-F1/F1 fisik dikoreksi signifikan (Matched `0,6840`→`0,8162`). |
| `Wave-V2` | Locked Hungarian+UF Pipeline (profil terkunci 953: **Hungarian**) | 953 Test (135 pohon, *test-locked*) | **0,8444** | **0,8331** | **0,8387** | **1,3630** | **63,70%** | **0,7442** | **0,6034** | `VALID` | Matched dikoreksi signifikan (`0,6010`→`0,7442`); profil terkunci 953 adalah Hungarian, bukan GSP. |
