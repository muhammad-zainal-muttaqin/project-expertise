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
