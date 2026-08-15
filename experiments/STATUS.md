# Status Eksperimen

> **Ringkasan akhir proyek ada di [../docs/LAPORAN-AKHIR.md](../docs/LAPORAN-AKHIR.md).**
> Pengumpulan metrik dihentikan 2026-08-12 setelah dua temuan menunjukkan
> pertanyaan RGB-vs-RGB+D tidak bisa dijawab dengan pasangan data ini:
> pergeseran akuisisi ~80 hari antara kedua dataset (`V2-E-022`) dan daya
> statistik split test yang tidak memadai (`V2-E-023`). Bagian di bawah
> dipertahankan apa adanya sebagai riwayat; baca bersama kedua entri itu.

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
   **DIKOREKSI oleh V2-E-022:** angkanya benar, sebabnya salah. Kelangkaan itu
   bukan artefak dataset yang lebih kecil, melainkan fase kematangan berbeda
   pada pohon yang sama — kedua dataset direkam terpisah ~80 hari (Mei vs Juli
   2026). Pada 1.408 citra ber-ID sama, B3 berbanding 3.604 lawan 321.
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
| Ablasi depth pada classifier (3 seed + statistik terpool) | selesai — FALSIFIED (V2-E-016) |
| Detektor class-agnostic (YOLO26l, RT-DETR-L) | selesai — V2-E-017/018 |
| WBF antar-detektor + sweep inference | selesai — V2-E-019 |
| Rekomposisi dua-tahap + counting | selesai — V2-E-020/021 |
| Classifier crop resolusi 256 @224 (`ftH`) | selesai — tidak menolong (test 0,6569, grup terlemah) |
| Fusi lintas-jalur dua-tahap + detektor class-aware | selesai — nihil (+0,0004), V2-E-023 |
| Probe pergeseran temporal 953 vs 352 | selesai — **V2-E-022** |
| Bootstrap CI seluruh angka Fase 6 | selesai — **V2-E-023** |
| Test split bersih untuk `agn953_full` | selesai — AP50 **0,7702** (19 pohon), V2-E-025 |
| Metadata split pada 6 berkas hasil | selesai — `_meta` ditambahkan, integritas terverifikasi |
| **Depth untuk lokalisasi (`agn352_4ch`)** | **selesai — 0,7636 vs 0,7358 RGB, V2-E-024** |
| CI angka utama Fase 6 | selesai — 0,4500 CI [0,4054; 0,5188], V2-E-026 |

### Hasil Fase 6 (test split 352, sebanding dengan Fase 1–5)

| Model | mAP50 | B1 | B2 | B3 | B4 | Counting ±1 |
|---|---|---|---|---|---|---|
| YOLO26l RGB | 0,3711 | 0,6842 | 0,4184 | 0,2301 | 0,1516 | 84,09% |
| YOLO26l RGB+D `edge` | 0,4316 | 0,7252 | 0,5031 | 0,2240 | 0,2740 | 87,27% |
| RT-DETR-L RGB | 0,4343 | 0,7680 | 0,4867 | 0,2641 | 0,2185 | **90,91%** |
| **Dua-tahap v4 (Fase 6)** | **0,4500** | 0,7366 | 0,4683 | **0,3212** | 0,2738 | 85,91% |
| RF-DETR-L RGB | **0,4544** | 0,6853 | 0,5184 | **0,3477** | 0,2661 | 88,18% |
| *Dua-tahap v3 (counting terbaik)* | 0,4102 | — | — | — | — | *88,18%* |

**Peringatan utama (V2-E-023):** CI 95% untuk mAP50 di split ini selebar
**±0,058** (220 citra, 410 kotak GT). Selisih antar-baris di tabel atas
sebagian besar **lebih kecil dari lebar selangnya** dan tidak terbedakan.
Jarak dua-tahap (0,4500) ke RF-DETR-L (0,4544) adalah 0,0044 — 26× lebih kecil
dari lebar CI. Jangan mengurutkan baris-baris ini sebagai peringkat.

**Tiga hal yang harus dibaca bersama angka di atas:**

1. **Plafon mAP50 dataset ini ≈ 0,733.** mAP50 ≤ AP50 lokalisasi secara
   definisi, dan AP50 lokalisasi test-352 = 0,7330 — praktis sama dengan
   test-953 = 0,7374 yang punya 9,8× lebih banyak box latih (V2-E-017).
   Target mAP50 0,80 berada **di atas plafon**, bukan sekadar belum tercapai.
   **DIKOREKSI oleh V2-E-024:** angka 0,733 itu plafon untuk masukan **RGB**,
   bukan plafon dataset. Dengan kanal depth, AP50 lokalisasi mencapai **0,7636**
   (kontrol RGB berpasangan: 0,7358; selisih +0,0278, P(Δ>0)=0,921, CI masih
   memuat nol). Plafon itu sifat modalitas masukan.
2. **Konfigurasi terbaik untuk mAP50 bukan yang terbaik untuk counting.**
   v4 menang mAP50 (0,4500), v3 menang counting (88,18%). mAP peduli urutan
   deteksi dalam kelas; counting memakai argmax sehingga sensitif kalibrasi
   prior (V2-E-021).
3. **Depth tetap tidak berkontribusi** pada klasifikasi kematangan:
   `I(Y;D) > 0` tapi `I(Y;D|RGB) ≈ 0` (V2-E-016). Kontribusi depth untuk
   lokalisasi belum diisolasi — ditunda atas permintaan pengguna.

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

---

## Fase 7 — Matriks monocular-depth (SELESAI 2026-08-15, enam sel terisi)

Pertanyaan tunggal: **apakah monocular-depth menaikkan performa deteksi?**
Enam sel, resep identik (`yolo26l.pt` COCO init, 60 epoch, patience 60,
batch 4, imgsz 1280, seed 42, `cos_lr`), evaluator pycocotools pada split
test, prediksi di-dump ke `.npz` saat evaluasi.

| # | Dataset | Input | ch | test mAP50 | val puncak | Epoch |
|---|---|---|---|---|---|---|
| 1 | 352 | RGB | 3 | 0,3677 | 0,4111 @ep45 | 60 |
| 2 | 352 | RGB+Depth `edge` | 4 | **0,4270** | 0,3856 @ep38 | 60 |
| 3 | 352 | RGB+Mono | 4 | 0,3943 | 0,3888 @ep41 | 54 (dihentikan) |
| 4 | 352 | RGB+Depth+Mono | 5 | 0,3766 | **0,4281** @ep50 | 60 |
| 5 | 953 | RGB | 3 | **0,5436** | 0,5373 @ep34 | 60 |
| 6 | 953 | RGB+Mono | 4 | 0,4960 | 0,5012 @ep17 | 31 (dihentikan) |

Angka sel 1 dan 2 di tabel ini dari *resampler* bootstrap; padanan
pycocotools-nya 0,3711 dan 0,4316 (baris matriks Fase 5 di atas). Beda ~0,004
itu implementasi mAP, bukan model — jangan campur antar-evaluator dalam satu
pengurangan (V2-E-032).

### Bootstrap CI berpasangan (2.000 ulangan, seed 42)

| Perbandingan | Selisih | CI 95% | Signifikan |
|---|---|---|---|
| sel 6 − sel 5 — mono vs RGB, 953 | **−0,0476** | [−0,0671; −0,0274] | **YA** |
| sel 4 − sel 2 — mono di atas depth, 352 | **−0,0504** | [−0,1038; −0,0015] | **YA** |
| sel 3 − sel 2 — mono vs depth sensor, 352 | −0,0327 | [−0,0756; +0,0074] | tidak |
| sel 4 − sel 3 — 5ch vs 4ch mono, 352 | −0,0177 | [−0,0672; +0,0323] | tidak |
| sel 3 − sel 1 — mono vs RGB, 352 | +0,0266 | [−0,0270; +0,0739] | tidak |

**Jawaban: TIDAK.** Monocular-depth tidak menang signifikan di satu pun dari
lima perbandingan, dan kalah signifikan di dua. Satu-satunya selisih positifnya
(+0,0266) lebih kecil daripada lebar CI-nya sendiri. **Depth sensor tetap kanal
keempat terbaik** (sel 2 = 0,4270, tertinggi di antara semua varian 352).

**Tiga batas yang wajib ikut dikutip:**
1. Split test 352 hanya 410 kotak → lebar CI ~0,10; selisih di bawah ~0,06
   memang tidak bisa dibedakan dari nol. Hanya sel 6 vs sel 5 (2.612 kotak,
   lebar CI 0,050) yang punya daya statistik memadai.
2. Sel 3 dan sel 6 dihentikan lebih awal (54 dan 31 dari 60 epoch) atas
   keputusan pengguna. Sel 6 paling terdampak — pembandingnya memuncak di
   ep34, di luar jangkauan run itu, jadi −0,0476 kemungkinan dilebih-lebihkan.
   Arahnya tidak diragukan (0 dari 2.000 ulangan positif), besarannya iya.
3. **Val 208 citra di split 352 tidak boleh dipakai memeringkat model.**
   Peringkat val (4 > 1 > 3 > 2) hampir persis kebalikan peringkat test
   (2 > 3 > 4 > 1) — terjadi pada keempat sel 352. Di 953 (404 citra val)
   val dan test sepakat. Lihat V2-E-030/031.

**Belum dijalankan:** kontrol M_shuf lintas-pohon, yang memisahkan "isi peta
mono" dari "biaya menambah kanal pada stem COCO 3-kanal". Selama itu belum
ada, penyebab kerugiannya tetap tidak diketahui.

Detail per sel: V2-E-027 (sel 6), V2-E-030 (sel 3), V2-E-031 (sel 4),
V2-E-032 (sintesis matriks). Riwayat per-epoch tiap run ada di
`results/riwayat_epoch/`, log training ringkas di `results/logs_ringkas/`.

### Dua pembatas kutipan yang berlaku lintas fase (V2-E-033)

1. **Hasil jalur agnostik Fase 6 hanya sah dari `test_bersih`.** 512 dari 588
   citra `agnostic953_test_penuh` (87%) ikut dilatih saat pretraining agnostik
   — berkas citra yang identik, bukan sekadar pohon yang sama. Angka dari
   `test_penuh` adalah train-on-test. `test_bersih` bersih total (0/76) tapi
   cuma 19 pohon.
2. **Tidak ada split test bersih untuk transfer 953→352.** 44 dari 55 pohon
   test-352 ada di train-953. Matriks mono-depth tidak terdampak (keenam sel
   init dari COCO, bukan dari bobot 953), tapi eksperimen finetune 953→352
   apa pun wajib mencantumkan catatan ini.
