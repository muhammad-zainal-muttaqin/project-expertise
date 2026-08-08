# Rekap Volume 1 — Research-Pipeline

Seluruh angka, percobaan, dan pelajaran dari repo
[Research-Pipeline](https://github.com/muhammad-zainal-muttaqin/Research-Pipeline)
yang relevan untuk eksperimen baru ini. Disusun agar sesi Claude baru bisa
langsung bekerja tanpa membaca seluruh Volume 1.

---

## 1. Angka Deteksi yang Berlaku (E-021, SawitMVC 953 pohon)

Keempat model dievaluasi dengan satu protokol `pycocotools` pada split
716/96/141 (per pohon, irisan nol).

| Model | Param | Resolusi | Val mAP50 | Val mAP50-95 | Test mAP50 | Test mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26m | 21,9 jt | 640 | 0,5195 | 0,2411 | 0,5165 | 0,2452 |
| YOLO26l | 26,3 jt | 1280 | 0,5270 | 0,2526 | 0,5300 | 0,2568 |
| RT-DETR-L | 33,0 jt | 1280 | 0,5459 | 0,2555 | 0,5784 | 0,2707 |
| **RF-DETR-L** | **35,7 jt** | **1280** | **0,5695** | **0,2604** | **0,6038** | **0,2770** |

Sumber: `experiments/results/E-021/perkelas_pycoco.json` di Research-Pipeline.

### Per kelas (RF-DETR-L, test)

| Kelas | AP50 | AP50-95 |
|---|---|---|
| B1 | tertinggi | tertinggi |
| B2 | rendah | rendah |
| B3 | sedang | sedang |
| B4 | **terendah** | **terendah** |

B4 sulit karena kecil, tertanam di pelepah, kontras rendah.
B2 sulit karena ambigu secara fotometrik dengan B3.

## 2. Angka Counting yang Berlaku (Baseline-SawitMVC, SawitMVC 953 pohon)

Detektor: YOLO26m (`y26mv2`, 60 epoch, batch 32, imgsz 640, seed 42).

### Counting dengan deteksi YOLO26m (Track B)

| Counter | Fitur | Class &plusmn;1 Acc | Tree &plusmn;1 Acc | Macro MAE |
|---|---|---:|---:|---:|
| **Ridge** | **F_all 67-dim** | **77,48%** | **32,62%** | **1,036** |
| ElasticNet | F0+spatial 21-dim | 76,77% | 31,21% | 1,039 |
| ElasticNet | F0 13-dim | 76,42% | 29,79% | 1,043 |
| LR | F0 13-dim | 75,71% | 30,50% | 1,048 |

### Counting dengan deteksi sempurna (Track C, batas atas)

| Counter | Class &plusmn;1 Acc | Tree &plusmn;1 Acc | Macro MAE |
|---|---|---:|---:|
| ElasticNet | 98,05% | 92,20% | 0,277 |
| SVM | 97,87% | 91,49% | 0,266 |
| Ridge | 97,70% | 90,78% | 0,275 |

**Gap Track B → C = 20,57 pp.** Sumber utama error adalah detektor, bukan
counter. Memperbaiki detektor adalah cara paling efektif menaikkan counting.

### Angka counting yang BELUM ADA

- YOLO26l pada 953 pohon → counting
- RT-DETR-L pada 953 pohon → counting
- RF-DETR-L pada 953 pohon → counting
- Semua model pada 352 pohon (RGB dan RGB+D) → counting

## 3. Angka dari Publikasi Baseline (DiB 2026)

Sumber: Indriani dkk., *Data in Brief* 67 (2026) 112990. Angka diverifikasi
langsung dari PDF (Tabel 3–4).

Detektor: YOLO26m, `epochs=60, batch=32, imgsz=640, patience=60, seed=42`.
**Sengaja tidak di-tuning** — titik acuan, bukan plafon.

| | AP50 | Precision | Recall |
|---|---|---|---|
| Overall | 0,531 | 0,508 | 0,571 |
| B1 | 0,739 | 0,602 | 0,776 |
| B2 | 0,433 | 0,482 | 0,441 |
| B3 | 0,599 | 0,515 | 0,674 |
| B4 | 0,354 | 0,432 | 0,393 |

Counting (test 141 pohon):

| Deteksi | Counter | Class &plusmn;1 | Tree &plusmn;1 | MAE |
|---|---|---:|---:|---:|
| GT | SVR | 96,81% | 88,65% | 0,303 |
| YOLO26m | SVR | 75,35% | 33,33% | 1,027 |

## 4. Percobaan Gagal (Jangan Diulang)

### 4.1 Early fusion depth (E-022, E-027)

**Apa:** Menambahkan depth sebagai kanal ke-4 langsung ke input YOLO.
**Hasil:** E-027 menunjukkan depth **merugikan** YOLO26n sebesar −0,0230 mAP
rerata; dua dari tiga seed signifikan negatif. E-022 kesimpulannya dicabut
setelah audit.
**Pelajaran:** Concat naif tidak bekerja. Depth pada early fusion menambah
derau, bukan sinyal.

### 4.2 Fusi menengah/akhir dari nol (E-032)

**Apa:** 15 run (5 lengan × 3 seed), 150 epoch dari nol, uji fusi
awal/menengah/akhir.
**Hasil:** Seluruh 12 CI95 memuat nol. Mid-fusion konsisten positif 3/3 seed
(rerata +0,0139) tetapi berstatus INDIKASI, bukan temuan.
**Pelajaran:** Efek depth terlalu kecil untuk diukur pada rezim dari-nol
dengan YOLO26n.

### 4.3 Gate init-nol (F-007)

**Apa:** Cabang frekuensi tinggi dengan γ = 0 saat inisialisasi, disuntik ke
RF-DETR-L.
**Hasil:** γ akhir ≈ 0 (dwt +0,0003, laplacian −6e-5). Gate tidak pernah
terbuka.
**Pelajaran:** `γ = 0` memberikan no-op sempurna sekaligus **titik mati**
sempurna — cabang samping tidak menerima gradien (dikali γ = 0), dan γ sendiri
hanya menerima derau. Setiap rancangan "cabang samping ber-gate init-nol"
menabrak masalah ini kecuali gate-nya diberi warmup, LR terpisah, inisialisasi
kecil-taknol, atau tugas pendamping.

### 4.4 Konsistensi lintas-sisi (F-003)

**Apa:** Mengukur plafon konsistensi prediksi antar sisi pohon.
**Hasil:** 0,2794 < ambang 0,30. 72% galat kelas salah di semua sisi. B4
hanya 0,1038.
**Pelajaran:** Galat kelas bukan masalah satu-sisi — model konsisten salah
di semua sisi. Menambahkan konsistensi lintas-sisi tidak akan membantu.

### 4.5 Tuning hyperparameter

**Status:** Sudah habis dijalankan oleh pengguna (batch, imgsz, augmentasi,
lr, dll). Ditegaskan dua kali. **Jangan disarankan lagi.**

### 4.6 SAHI dan teknik siap-pakai

**Status:** Sudah dicoba langsung oleh pengguna. Tidak satu pun menaikkan mAP.
**Jangan diusulkan ulang.**

## 5. Percobaan Berhasil (Boleh Dibangun di Atasnya)

### 5.1 RF-DETR-L sebagai detektor terbaik (E-021)

NMS-free, DINOv2 backbone, test mAP50 0,6038. Melewati sasaran 0,60.
Mengalahkan YOLO26l dan RT-DETR-L pada semua metrik.

### 5.2 Frekuensi tinggi memisahkan tandan dari pelepah (F-002)

DWT high-high +0,0731 pada B4 (ambang +0,02). Laplacian +0,0721 praktis seri.
Monoton B1 < B2 < B3 < B4 — semakin muda, semakin mudah dibedakan dari
pelepah via frekuensi.

### 5.3 Pipeline counting modular (Baseline-SawitMVC)

Ridge + F_all 67-dim. Hanya butuh JSON deteksi per pohon sebagai input.
Ganti detektor, evaluasi counting langsung jalan.

### 5.4 Reproyeksi depth tervalidasi (E-022)

`reproject_depth.py` memproyeksikan depth Orbbec ke bidang RGB dengan kalibrasi
per berkas. Sudah diverifikasi — bukan resize naif (yang meleset median 29 px).

### 5.5 Varians seed RF-DETR-L terukur (F-004)

SD test mAP50 = 0,0049 (rentang 0,0097). Jauh lebih kecil dari yang
diasumsikan. Rerata 3 seed: 0,5949 (jalur `run_test`).

## 6. Diagnosa yang Sudah Disepakati

1. **Bottleneck ada di detektor, bukan counter.** Counter hampir sempurna
   bila diberi deteksi bersih (98,05% vs 77,48%).
2. **Kegagalan deteksi terbelah dua:**
   - **(A) Geometris** — B4 kecil/tertanam/tertutup pelepah. Depth relevan
     di sini.
   - **(B) Fotometrik** — ambiguitas B2↔B3. Depth **tidak** membantu di sini.
3. **Kelas paling ambigu adalah B2** (0,434), bukan B4 (0,234). B4 gagal
   karena deteksi, bukan kebingungan kelas (E-028).

## 7. Caveat yang Wajib Dibawa

- Pseudo-depth berasal dari RGB yang sama → error-nya berkorelasi.
- **Tidak ada benchmark RGB-D pada TBS sawit di literatur 182 makalah.**
  "Depth menaikkan angka" = hipotesis yang falsifiable.
- Hasil naik di B4/crowded tapi datar di B2/B3 = **konfirmasi teori**, bukan
  kegagalan.
- Angka di SawitMVC-Depth (352 pohon) **tidak sebanding** dengan angka di
  SawitMVC (953 pohon) — distribusi kelas terbalik, kepadatan lebih rendah.
