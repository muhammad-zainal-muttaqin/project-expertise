# Rencana Kerja — 4 Fase

## Gambaran Umum

```
Fase 0   Persiapan infrastruktur          (tanpa GPU)
Fase 1   RGB 953 pohon — counting         (GPU: inference saja)
Fase 2   RGB 352 pohon — train + eval     (GPU: training)
Fase 3   RGB+D 352 pohon — train + eval   (GPU: training + modifikasi arsitektur)
Fase 4   Evaluasi dan pelaporan           (tanpa GPU)
```

Setiap fase menghasilkan angka yang mengisi matriks di README.

---

## Fase 0 — Persiapan Infrastruktur

**Tujuan:** Pastikan pipeline counting bisa menerima output dari detektor
mana pun, dan dataset SawitMVC-Depth siap dipakai.

| # | Tugas | Detail | Status |
|---|---|---|---|
| 0.1 | Adaptor format deteksi | Buat konverter output RT-DETR-L dan RF-DETR-L ke format JSON per-pohon yang sama dengan `predictions/y26mv2_per_tree/` di Baseline-SawitMVC | Belum |
| 0.2 | Siapkan split SawitMVC-Depth | Buat split train/val/test per pohon untuk 352 pohon, konsisten dengan konvensi SawitMVC (per pohon, bukan per citra) | **Selesai** — sudah ada siap pakai di `/workspace/SawitMVC-Depth-YOLO/` (70/15/15, seed 10, tree-stratified, 245/52/55 pohon), tidak perlu dibuat ulang |
| 0.3 | Verifikasi ground truth | Pastikan anotasi SawitMVC-Depth kompatibel dengan pipeline counting (format JSON, kelas B1–B4, identitas tandan) | **Selesai** — skema identik dengan SawitMVC (lihat `docs/SCHEMA-PERTREE.md`), tidak perlu shim terjemahan |
| 0.4 | Siapkan depth yang sudah diproyeksikan | Gunakan `reproject_depth.py` dari Research-Pipeline untuk memproyeksikan seluruh depth ke bidang RGB | Belum |
| 0.5 | Setup environment | `requirements.txt`, versi pustaka (ultralytics, rfdetr, pycocotools, scikit-learn) | Belum |

**Keluaran:** Infrastruktur siap, split terdefinisi, depth terproyeksi.

---

## Fase 1 — RGB 953 Pohon (SawitMVC): Counting

**Tujuan:** Isi sel counting untuk ketiga detektor pada dataset 953 pohon.
Angka deteksi sudah ada dari E-021 — yang kurang hanya counting.

| # | Tugas | Detail | Status |
|---|---|---|---|
| 1.1 | Inference YOLO26l | Jalankan YOLO26l (bobot E-021) pada 3.992 citra, simpan JSON per pohon | Belum |
| 1.2 | Inference RT-DETR-L | Jalankan RT-DETR-L (bobot E-021) pada 3.992 citra, simpan JSON per pohon | Belum |
| 1.3 | Inference RF-DETR-L | Jalankan RF-DETR-L (bobot E-021) pada 3.992 citra, simpan JSON per pohon | Belum |
| 1.4 | Counting ketiga model | Jalankan pipeline Ridge + F_all untuk ketiga set prediksi | Belum |
| 1.5 | Bandingkan | Apakah detektor dengan mAP lebih tinggi juga memberi counting lebih baik? | Belum |

**Referensi pembanding:** YOLO26m sudah ada di Baseline-SawitMVC (77,48%).

**Keluaran Fase 1:**

| Model | Test mAP50 | Class &plusmn;1 Acc | Tree &plusmn;1 Acc | Macro MAE |
|---|---|---|---|---|
| YOLO26l | 0,5300 (ada) | ? | ? | ? |
| RT-DETR-L | 0,5784 (ada) | ? | ? | ? |
| RF-DETR-L | 0,6038 (ada) | ? | ? | ? |

**Estimasi waktu:** ~1 jam GPU (inference saja), ~30 menit CPU (counting).

---

## Fase 2 — RGB 352 Pohon (SawitMVC-Depth, tanpa depth)

**Tujuan:** Bangun baseline RGB pada dataset yang sama yang nantinya akan
dibandingkan dengan RGB+D. Ini jembatan perbandingan apple-to-apple.

| # | Tugas | Detail | Status |
|---|---|---|---|
| 2.1 | Latih YOLO26l | Pada RGB 352 pohon, konfigurasi setara E-021 | Belum |
| 2.2 | Latih RT-DETR-L | Pada RGB 352 pohon, konfigurasi setara E-021 | Belum |
| 2.3 | Latih RF-DETR-L | Pada RGB 352 pohon, konfigurasi setara E-021 | Belum |
| 2.4 | Evaluasi deteksi | pycocotools, per kelas | Belum |
| 2.5 | Inference + counting | Pipeline Ridge + F_all | Belum |

**Catatan:** Dataset ini lebih kecil (352 vs 953 pohon) dan distribusi
kelasnya terbalik (B2 dominan, B4 langka). Angka pasti lebih rendah dari
Fase 1 — itu wajar, bukan kegagalan.

**Keluaran Fase 2:**

| Model | Test mAP50 | Class &plusmn;1 Acc | Tree &plusmn;1 Acc | Macro MAE |
|---|---|---|---|---|
| YOLO26l | ? | ? | ? | ? |
| RT-DETR-L | ? | ? | ? | ? |
| RF-DETR-L | ? | ? | ? | ? |

**Estimasi waktu:** ~1–2 hari GPU (3× training).

---

## Fase 3 — RGB+D 352 Pohon (4-kanal)

**Tujuan:** Ini inti eksperimen. Latih ketiga arsitektur dengan input 4-kanal
(RGB + depth), bandingkan dengan baseline RGB Fase 2.

| # | Tugas | Detail | Status |
|---|---|---|---|
| 3.1 | Modifikasi YOLO26l | Ubah stem dari 3 → 4 kanal input | Belum |
| 3.2 | Modifikasi RT-DETR-L | Ubah stem HGNetV2 dari 3 → 4 kanal | Belum |
| 3.3 | Modifikasi RF-DETR-L | **Paling sulit.** DINOv2 backbone beku, stem 3-kanal. Opsi: (a) tambah proyeksi 1-kanal → 3-kanal lalu concat, (b) tulis stem baru 4→embed_dim, (c) fusi menengah dengan cabang terpisah | Belum |
| 3.4 | Latih ketiga model | Konfigurasi identik Fase 2, hanya input berubah | Belum |
| 3.5 | Evaluasi deteksi | pycocotools, per kelas, per strata (oklusi, ukuran) | Belum |
| 3.6 | Inference + counting | Pipeline Ridge + F_all | Belum |
| 3.7 | Uji Target 1 | RGB+D &ge; RGB? Per arsitektur, per kelas | Belum |
| 3.8 | Uji Target 2 | RGB+D > RGB secara signifikan? | Belum |

### Risiko dan mitigasi

| Risiko | Bukti | Mitigasi |
|---|---|---|
| Early fusion menyebabkan regresi | E-022, E-027 | Coba pendekatan yang berbeda dari concat naif; misalnya proyeksi depth terpisah sebelum concat |
| Backbone beku menolak kanal ke-4 | Seri F (gate init-nol) | Jangan gunakan gate init-nol; gunakan warmup atau inisialisasi kecil-taknol |
| Dataset terlalu kecil (352 pohon) | — | Evaluasi dengan bootstrap CI; laporkan apakah selisih signifikan |
| Distribusi kelas terbalik | B4 hanya 148 bbox | Stratifikasi evaluasi per kelas |

**Keluaran Fase 3:**

| Model | Test mAP50 | Delta vs RGB | Class &plusmn;1 Acc | Delta vs RGB |
|---|---|---|---|---|
| YOLO26l + D | ? | ? | ? | ? |
| RT-DETR-L + D | ? | ? | ? | ? |
| RF-DETR-L + D | ? | ? | ? | ? |

**Estimasi waktu:** ~3–5 hari (termasuk debugging modifikasi arsitektur).

---

## Fase 4 — Evaluasi dan Pelaporan

| # | Tugas | Detail |
|---|---|---|
| 4.1 | Kompilasi matriks detection | mAP50, mAP50-95, P, R per kelas, semua 9 sel |
| 4.2 | Kompilasi matriks counting | Class &plusmn;1 Acc, Tree &plusmn;1 Acc, MAE, semua 9 sel |
| 4.3 | Analisis terstratifikasi | Per kelas, per strata oklusi/ukuran — di mana depth membayar? |
| 4.4 | Uji signifikansi | Bootstrap CI per pohon, 10.000 replikat |
| 4.5 | Tulis laporan | Ringkasan untuk naskah/sidang |

---

## Fase 5 — Loop Perbaikan RGB+D (dibatasi eksplisit oleh pengguna)

**Ditambahkan 2026-08-08.** Setelah Fase 3+4 menghasilkan angka RGB+D
pertama, fase ini mencari cara menaikkannya lebih jauh vs RGB. Pengguna
membatasi eksplisit: **hanya dua jenis intervensi yang boleh dicoba.**

1. **Mengubah representasi dataset** — mis. encoding depth (inverse-depth
   vs linear, HHA, surface normal, edge map turunan depth), preprocessing/
   normalisasi, augmentasi khusus kanal depth, resolusi/kanal tambahan.
2. **Mengubah arsitektur model** — mis. lokasi fusi (awal/menengah/akhir),
   desain cabang, modifikasi stem, mekanisme gating (dengan pelajaran dari
   F-007: hindari init-nol).

**Yang TIDAK boleh dicoba di fase ini** (di luar dua kategori di atas):
tuning hyperparameter (sudah terbukti mentok — lihat "Hal yang sudah dicoba
dan GAGAL" di `CLAUDE.md`), teknik inference/post-processing siap pakai
(SAHI sudah gagal), ensembling, atau trik training lain.

Setiap percobaan tetap wajib mengikuti aturan eksperimen `CLAUDE.md`: satu
hipotesis falsifiable per entri, dicatat di `experiments/EKSPERIMEN.md`
dengan verdict CONFIRMED/FALSIFIED/INCONCLUSIVE, hasil negatif dicatat
dengan bobot sama. Kandidat awal yang konsisten dengan kedua lever di atas
dan pelajaran Volume 1 (lihat `docs/REKAP.md` §4-5):

- Representasi: ganti encoding inverse-depth linear saat ini dengan encoding
  yang menonjolkan kontras dekat (di mana tandan berada), bukan seluruh
  rentang 0,8-15,0 m secara linear.
- Arsitektur: cabang depth terpisah dengan fusi menengah, inisialisasi
  kecil-taknol (bukan gate init-nol seperti F-007), dibangun di atas sinyal
  indikatif E-032 (mid-fusion 3/3 seed positif, CI masih memuat nol).

**Protokol iterasi cepat (ditambahkan 2026-08-08):** setiap percobaan Fase 5
dijalankan dengan **maksimal 15 epoch, patience 3 epoch** (bukan 60 epoch
seperti Fase 1-3) — screening cepat, bukan angka final. Alasan pengguna:
sinyal menjanjikan atau tidak sudah terlihat di 15 epoch pertama. Konsekuensi:
- Angka dari screening 15-epoch **tidak dibandingkan langsung** dengan angka
  60-epoch Fase 1-3 — hanya dipakai untuk ranking relatif antar-percobaan
  Fase 5 (mana yang layak dilanjutkan).
- Kandidat yang lolos screening (naik konsisten, bukan derau) baru dijalankan
  penuh 60 epoch untuk angka yang bisa dikutip dan dibandingkan RGB vs RGB+D.
- Tetap dicatat di `EKSPERIMEN.md` sebagai entri sendiri (mis. tag
  `[screening-15ep]` di judul), verdict tetap CONFIRMED/FALSIFIED/INCONCLUSIVE
  berdasar sinyal screening, bukan diklaim sebagai hasil final.

### Status pelaksanaan (2026-08-11)

Screening kedua lever sudah dijalankan pada YOLO26l (prioritas pertama sesuai
keputusan pengguna — paling mudah dimodifikasi, satu-satunya yang naik di
early fusion naif V2-E-005). Lihat `experiments/EKSPERIMEN.md` V2-E-008/009
untuk detail lengkap.

- **Lever representasi — 4 kandidat di-screening**: `edge` (Sobel gradient
  magnitude) menang jelas (val mAP50 0,3777 vs 0,3168/0,3221/0,3321 untuk
  dropout/clipped/valid_mask). Dipromosikan ke 60 epoch penuh — hasil test
  split: **deteksi CONFIRMED naik +10,1% mAP50** dari `inverse` (0,4316 vs
  0,3919), robust terhadap baseline RGB manapun. **Counting INCONCLUSIVE**
  — bootstrap CI berpasangan berbalik arah tergantung baseline RGB-352 yang
  dipakai (menang +3,18pp vs retrain baru, kalah −2,28pp vs angka asli
  V2-E-004) — lihat V2-E-010/011 untuk detail lengkap kenapa ini dilaporkan
  tidak konklusif, bukan dibulatkan ke arah manapun.
- **Lever arsitektur — mid-fusion + gate non-zero-init**: gate berhasil
  dihindarkan dari titik mati F-007 (bergerak 0,02→0,025), tapi sinyal
  keseluruhan TIDAK naik konsisten (plateau epoch 3, early-stop epoch 6, val
  mAP50 0,209) — kalah jauh dari kandidat representasi pada jumlah epoch
  sama. **TIDAK dipromosikan** ke 60 epoch — hasil negatif, dicatat apa
  adanya. RT-DETR-L/RF-DETR-L untuk lever arsitektur tidak dikerjakan
  (kondisional pada lever arsitektur YOLO26l lolos screening, yang tidak
  terjadi).

**Fase 5 SELESAI** (2026-08-11) — semua metrik terisi (deteksi, counting,
bootstrap CI). Lihat `experiments/EKSPERIMEN.md` V2-E-008 s/d V2-E-011.

---

## Ringkasan Estimasi

| Fase | Kebutuhan GPU | Estimasi waktu |
|---|---|---|
| 0 | Tidak | 1 hari |
| 1 | Ya (inference) | 0,5 hari |
| 2 | Ya (training) | 1–2 hari |
| 3 | Ya (training) | 3–5 hari |
| 4 | Tidak | 1 hari |
| **Total** | | **~7–10 hari** |

---

## Fase 6 — Diagnostik ulang + pipeline dua-tahap (2026-08-11)

**Pelonggaran scope oleh pengguna.** Fase 5 dibatasi ke dua lever
(representasi, arsitektur) dengan larangan eksplisit terhadap tuning
hyperparameter, SAHI, dan ensembling. Untuk Fase 6 pengguna melonggarkan:
target akhirnya aplikasi mobile (foto 4 sisi → deteksi → hasil), tapi selama
pengembangan **boleh lambat/berat, boleh dipecah multi-tahap, boleh ada
preprocessing, tidak harus YOLO, tidak harus satu pipeline** — yang penting
metriknya naik. Ditambah arahan: **tanpa rerun eksperimen lama, tanpa audit
history dulu; kejar hasil terbaik, baru trace back**. Cabang yang turun sedikit
saja atau buang waktu langsung dibuang.

Larangan Fase 5 diperlakukan **tergantikan** untuk fase ini. Kalau itu keliru,
cabang yang bersangkutan tinggal dibuang.

### Kenapa fase ini ada

Lima probe read-only (tanpa training, hitungan menit) menunjukkan rumusan
masalah Fase 1–5 keliru di tiga titik. Lengkapnya di `docs/DIAGNOSIS-DEPTH.md`
dan entri `V2-E-012` s.d. `V2-E-014`:

1. Gap 953-vs-352 adalah kelangkaan label B3/B4 (34×/26×), bukan efek depth.
2. 44,5% kemampuan detektor hangus karena salah kelas (AP50 lokalisasi 0,6677
   vs mAP50 0,3707), dan konfusinya selalu antar-kelas-tetangga → ordinal.
3. Sinyal depth adalah relief lokal ordinal (p=1,7×10⁻²¹) ber-SNR ~0,3 per
   piksel — hanya terbaca setelah pooling wilayah. Early fusion di stem adalah
   rezim terburuk untuk sinyal seperti itu.

### Rancangan

| Tahap | Isi | Alasan dari diagnostik |
|---|---|---|
| 0 | Split 953 bebas bocor (846 pohon, buang 107 pohon val/test-352) | 44 dari 55 pohon test-352 ada di train-953 — tanpa dibersihkan, pretraining tidak sah |
| 1 | Classifier kematangan pada crop, RGB vs RGB+relief-depth | menyerang headroom 0,2970; crop melakukan pooling wilayah by-construction |
| 2 | Detektor class-agnostic (1 kelas "tandan"), pretrain 953 → finetune 352 | lokalisasi lihat 2.299 positif, bukan terpecah jadi 215 (B3) / 98 (B4) |
| 3 | Rekomposisi: kelas tahap-2 ditempel ke box tahap-1 → mAP50 sebanding Fase 1–5, lalu Ridge+F_all | angka tetap bisa dibandingkan dengan seluruh riwayat |

Kanal depth yang dipakai bukan inverse-depth absolut (itu nuisance: standoff
per citra std 0,82 m) melainkan **relief** `R = Z − median lokal`, di-scale
±10 cm → 0..255 (step 0,08 cm/level vs 2,91 cm sebelumnya). Cabang depth
difusikan **setelah global pooling**, gate init taknol (F-007), plus loss
auxiliary RGB-only supaya jalur RGB tidak bisa dirusak jalur depth.

### Batas yang harus jujur disebut

Test split 352 cuma 410 box, dengan B4 = 26. Selisih kecil pada mAP50 tidak
bisa dibedakan dari derau — pada Fase 5 bahkan val dan test berlawanan arah
(RGB unggul di val 0,4111 vs 0,3856; `edge` unggul di test). Karena itu ablasi
depth Fase 6 dijalankan **multi-seed**, dan klaim tanpa pemisahan yang jelas
dari derau dilaporkan INCONCLUSIVE, bukan dibulatkan.

Soal target ~90%: yang realistis menyentuh itu adalah **Class ±1 Acc counting**
(sekarang 89,55%) dan akurasi klasifikasi kematangan per-crop. **mAP50 tidak
bisa 90%** di dataset ini — plafonnya AP50 lokalisasi (0,6677 sekarang).
