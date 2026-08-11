# Status Eksperimen

## Fase saat ini: 5 — Loop perbaikan RGB+D (SELESAI, semua metrik terisi)

Lihat [docs/RENCANA.md](../docs/RENCANA.md) untuk rencana kerja lengkap dan
[EKSPERIMEN.md](EKSPERIMEN.md) untuk log append-only per hipotesis.

**Fase 0-4: SELESAI** (V2-E-001..V2-E-007, ter-commit). Fase 5 dimulai
2026-08-10/11: screening lever representasi (4 kandidat encoding depth) dan
lever arsitektur (mid-fusion+gate) pada YOLO26l — lihat V2-E-008/009.
`edge` (Sobel gradient depth) menang screening, dipromosikan ke training
penuh 60 epoch:

- **Deteksi: CONFIRMED.** Test mAP50 0,4316 vs `inverse` 0,3919 — **+10,1%
  relatif**, robust terhadap baseline RGB manapun (V2-E-010).
- **Counting: INCONCLUSIVE.** Bootstrap CI vs retrain RGB-352 baru
  (Class Acc 84,09%, underperform 5,46pp dari angka asli V2-E-004 89,55%)
  menunjukkan `edge` unggul +3,18pp (P=94,3%, CI hampir tapi belum
  eksklusif positif) — TAPI dibanding angka RGB asli V2-E-004, `edge`
  malah kalah −2,28pp. Kesimpulan berbalik arah tergantung baseline mana
  yang dipakai — dilaporkan tidak konklusif, bukan dibulatkan (V2-E-011).
- **Arsitektur (mid-fusion+gate): FALSIFIED** di screening, tidak
  dipromosikan (V2-E-009).

### Progres Fase 0-4 (selesai penuh, ter-commit)

Semua retrain (YOLO26l/RT-DETR-L/RF-DETR-L) dan evaluasi deteksi+counting
pada RGB 953 pohon, RGB 352 pohon, dan RGB+D 352 pohon (early fusion) selesai
— lihat matriks di bawah dan `V2-E-001` s/d `V2-E-007` untuk detail metode.

### Progres Fase 5 (loop perbaikan RGB+D)

| Lever | Kandidat | val mAP50 (screening 15ep) | Verdict |
|---|---|---|---|
| Representasi | `dropout` | 0,3168 | tidak menang |
| Representasi | **`edge`** | **0,3777** | **menang → promosi 60ep** |
| Representasi | `clipped` | 0,3221 | tidak menang |
| Representasi | `valid_mask` (baru) | 0,3321 | tidak menang |
| Arsitektur | mid-fusion+gate (fuse_at=4, gate init=0,02) | 0,2087 (epoch 3, early-stop) | **FALSIFIED** — tidak lolos |

## Matriks hasil (test split, pycocotools mAP50 / Ridge+F_all Class ±1 Acc)

| Dataset | YOLO26l | RT-DETR-L | RF-DETR-L |
|---|---|---|---|
| RGB 953 pohon | Det: 0,5435 / Count: 72,16% | Det: 0,5781 / Count: 76,24% | Det: 0,6012 / Count: 76,24% |
| RGB 352 pohon (asli, V2-E-003/004) | Det: 0,3606 / Count: 89,55% | Det: 0,4343 / Count: 90,91% | Det: 0,4544 / Count: 88,18% |
| RGB 352 pohon (retrain, V2-E-011) | Det: 0,3711 / Count: 84,09% | — | — |
| RGB+D 352 pohon (early fusion, `inverse`) | Det: 0,3919 / Count: 87,73% | Det: 0,3877 / Count: 88,64% | Det: 0,4186 / Count: 88,18% |
| RGB+D 352 pohon (`edge`, Fase 5) | **Det: 0,4316 / Count: 87,27%** | — | — |

Format sel: `Det: mAP50 / Count: Class ±1 Acc`. Sumber: `results/*.json`
(V2-E-001..011), `EKSPERIMEN.md` untuk detail metode tiap sel.

**Baris `edge` — baca dengan konteks, jangan dikutip sepotong:**
- Deteksi: menang jelas dari SEMUA baris RGB/RGBD lain di atas (CONFIRMED).
- Counting: 87,27% ada DI ANTARA dua angka RGB-352 (84,09% retrain vs 89,55%
  asli) — menang atau kalah tergantung mana yang jadi pembanding. Bootstrap
  CI vs retrain: +3,18pp (P=94,3%, hampir signifikan). Vs angka asli:
  −2,28pp. Dilaporkan INCONCLUSIVE (V2-E-011), bukan salah satu arah saja.
