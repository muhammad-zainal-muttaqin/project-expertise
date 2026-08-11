# Status Eksperimen

## Fase saat ini: 6 — Diagnostik ulang + pipeline dua-tahap (BERJALAN)

Scope dilonggarkan pengguna: boleh berat/multi-tahap, tidak harus YOLO, tidak
harus satu pipeline — target metrik setinggi mungkin. Lima probe read-only
(tanpa training) mengubah rumusan masalahnya; jalan penemuannya lengkap di
[../docs/DIAGNOSIS-DEPTH.md](../docs/DIAGNOSIS-DEPTH.md), entri
`V2-E-012` s.d. `V2-E-014`.

**Tiga temuan yang mengoreksi pemahaman Fase 1–5:**

1. **Gap 953-vs-352 bukan efek depth** (V2-E-012) — B3 34× dan B4 26× lebih
   langka di dataset depth; gap terkonsentrasi persis di dua kelas itu
   (B3 AP50 0,605→0,200, B4 0,351→0,130), B1/B2 nyaris sama. Perbandingan
   lintas dataset 953-vs-352 **tidak sah** dan tidak dipakai lagi.
2. **44,5% kemampuan detektor hangus karena salah kelas** (V2-E-013) — AP50
   class-agnostic 0,6677 vs mAP50 class-aware 0,3707. Mencari tandan sudah
   baik; menamainya yang rusak, dan konfusinya selalu ke kelas bertetangga
   (masalah ordinal).
3. **Sinyal depth = relief lokal, bukan skala metrik** (V2-E-014) — relief
   B1 +2,8 cm → B4 −5,1 cm, monoton, Kruskal-Wallis p=1,7×10⁻²¹; tapi
   SNR per-piksel ≈0,3 (satu level uint8 = 2,91 cm di Z=2,5 m, sinyalnya
   0,8 cm), jadi hanya terbaca setelah pooling wilayah (AUC 0,592→0,724).
   Depth **95,1% valid di dalam box** — "29% invalid" itu latar, bukan objek.

Konsekuensi desain: pisahkan lokalisasi dari klasifikasi, dan konsumsi depth
setelah pooling di jalur klasifikasi — bukan early fusion di stem.

### Status pengerjaan Fase 6

| Komponen | Status |
|---|---|
| Probe diagnostik (`probe_depth_signal.py`) | selesai — V2-E-012/013/014 |
| Split 953 bebas bocor (846 pohon) | selesai — irisan nol terverifikasi |
| Dataset crop + relief depth + mask box | selesai — 16.542 crop (953) + 2.299 (352) |
| Classifier kematangan crop | selesai tahap awal; ablasi depth multi-seed berjalan |
| Detektor class-agnostic | berjalan (pretrain 953 → finetune 352) |
| Rekomposisi dua-tahap + counting | belum |

---

## Fase 5 — Loop perbaikan RGB+D (SELESAI, semua metrik terisi)

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
