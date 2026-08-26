# Panduan Reproduksi: Pipeline Dua-Tahap dan Audit Statistik (Fase 6)

Dokumen ini memuat prosedur eksekusi langkah-demi-langkah untuk merekonstruksi seluruh hasil eksperimen Fase 6 secara deterministik dari nol. Setiap tahapan merujuk langsung ke berkas kode sumber eksekutif dan entri simpul pembuktian terkait.

Seluruh perintah dieksekusi dari direktori kerja utama repositori dengan lingkungan Python terisolasi (`.venv`).

---

## 1. Prasyarat Lingkungan dan Integritas Data

```bash
python3 -m venv .venv --system-site-packages
.venv/bin/pip install -r requirements-freeze.txt
```

> [!NOTE]
> Lingkungan komputasi acuan menggunakan **Python 3.12.3** dengan 181 pustaka ter-pin pada [`requirements-freeze.txt`](file:///D:/Work/Assisten-Dosen/project-expertise/requirements-freeze.txt).

---

## 2. Diagnostik Sifat Sinyal Kedalaman (Simpul V2-E-012 s.d. V2-E-014)

Eksekusi probe analitik *read-only* (tanpa beban GPU, durasi $\approx 5\text{ menit}$):

```bash
.venv/bin/python scripts/probe_depth_signal.py --probe semua
```

Menghasilkan kalkulasi distribusi frekuensi kelas, validitas spasial piksel kedalaman di dalam kotak objek, analisis relief lokal ordinal ($H = 99,8$, $p = 1,7 \times 10^{\minus 21}$), tabel resolusi kuantisasi, dan kurva efektivitas agregasi spasial (*pooling*).

---

## 3. Pembangunan Partisi Bebas Bocor & Dataset Turunan

```bash
.venv/bin/python scripts/make_pretrain_split.py        # 846 pohon 953 bebas kebocoran
.venv/bin/python scripts/make_agnostic_dataset.py      # Pembangkitan dataset 1-kelas agnostic953 & agnostic352
.venv/bin/python scripts/build_crop_dataset.py --src 352 --workers 8
.venv/bin/python scripts/build_crop_dataset.py --src 953 --workers 8
```

Skrip `make_pretrain_split.py` dan `make_agnostic_dataset.py` menerapkan asersi otomatis (*zero-leakage assertion*) terhadap partisi validasi dan uji dataset 352 pohon.

---

## 4. Pelatihan Detektor Lokalisasi Murni 1-Kelas (Simpul V2-E-017 & V2-E-018)

```bash
# 1. Prapelatihan 953 Pohon (Jadwal Cosine Lengkap)
.venv/bin/python scripts/train_yolo_4ch_screening.py \
  --data /workspace/agnostic953/data.yaml --epochs 12 --patience 12 \
  --imgsz 1280 --batch 4 --weights yolo26l.pt --name agn953_full

# 2. Penyesuaian Terarah 352 Pohon (Toleransi Early Stopping Longgar)
.venv/bin/python scripts/train_yolo_4ch_screening.py \
  --data /workspace/agnostic352/data.yaml --epochs 60 --patience 45 \
  --imgsz 1280 --batch 4 --weights runs/agn953_full/weights/best.pt --name agn352_ft3

# 3. Pelatihan RT-DETR-L Sebagai Anggota Ensembel
.venv/bin/python -c "from ultralytics import RTDETR; RTDETR('rtdetr-l.pt').train(data='/workspace/agnostic352/data.yaml', epochs=60, patience=10, imgsz=1280, batch=4, seed=42, cos_lr=True, project='/workspace/project-expertise/runs', name='agn352_rtdetr')"
```

Evaluasi plafon lokalisasi murni:
```bash
.venv/bin/python scripts/eval_detector_agnostic.py --detektor runs/agn352_ft/weights/best.pt --split test
```

---

## 5. Pelatihan Pengklasifikasi Kematangan pada Citra Terpotong (Simpul V2-E-015, V2-E-016, V2-E-021)

```bash
# Prapelatihan pada Korpus 953 Bebas Bocor
.venv/bin/python scripts/train_crop_classifier.py --tahap pretrain --mode rgb \
  --head hybrid --backbone convnext_small.fb_in22k_ft_in1k --img 176 \
  --epochs 14 --batch 24 --name pre953s

# Penyesuaian Terarah pada 352 Pohon (3 Seed untuk Ensembel)
for s in 42 101 202; do
  .venv/bin/python scripts/train_crop_classifier.py --tahap finetune --mode rgb \
    --head hybrid --backbone convnext_small.fb_in22k_ft_in1k --img 176 \
    --epochs 50 --batch 24 --seed $s --init runs_fase6/pre953s/best.pt --name ftS_$s
done

# Studi Ablasi Kontribusi Statistik Kedalaman (V2-E-016)
.venv/bin/python scripts/probe_fitur_depth.py --model runs_fase6/ftS_202/best.pt
```

---

## 6. Penyelarasan Ambang Inferensi, Ensembel WBF, & Rekomposisi (Simpul V2-E-019 & V2-E-020)

> [!IMPORTANT]
> Penyetelan ambang inferensi dan kombinasi ensembel **wajib dilakukan pada partisi validasi (`--split val`)**, bukan partisi uji.

```bash
# 1. Pemilihan Kombinasi Detektor pada Validasi
.venv/bin/python scripts/pilih_detektor.py --split val \
  --kandidat runs/agn352_ft/weights/best.pt runs/agn352_ft3/weights/best.pt runs/agn352_rtdetr/weights/best.pt \
  --out results/detektor_pilihan.json

# 2. Penelusuran Kombinasi Resolusi & Ambang NMS
.venv/bin/python scripts/sweep_inferensi.py --split val \
  --detektor runs/agn352_ft/weights/best.pt runs/agn352_ft3/weights/best.pt \
  --out results/sweep_inferensi.json

# 3. Evaluasi Akhir Rekomposisi Dua-Tahap pada Partisi Uji
.venv/bin/python scripts/eval_twostage.py --split test --conf 0.005 \
  --detektor runs/agn352_ft/weights/best.pt runs/agn352_ft3/weights/best.pt \
  --classifier runs_fase6/ftS_*/best.pt runs_fase6/ftJ_*/best.pt runs_fase6/ftG_*/best.pt \
  --tta --multi-kelas --imgsz 1280 --det-iou 0.5 \
  --out results/twostage_final_v4.json

# 4. Pipeline Pencacahan Ridge + F_all
.venv/bin/python scripts/run_counting_twostage.py --tta --imgsz 1280 --det-iou 0.5 \
  --detektor runs/agn352_ft/weights/best.pt runs/agn352_ft3/weights/best.pt \
  --classifier runs_fase6/ftS_*/best.pt runs_fase6/ftJ_*/best.pt runs_fase6/ftG_*/best.pt \
  --label TwoStage-FINAL_v4 --out results/counting_twostage.json
```

---

## 7. Audit Validitas Metodologis & Evaluasi Bootstrap (Simpul V2-E-022 s.d. V2-E-025)

```bash
# 1. Audit Pergeseran Temporal Antar-Dataset (V2-E-022)
.venv/bin/python scripts/probe_pergeseran_temporal.py --out results/pergeseran_temporal.json

# 2. Evaluasi Selang Kepercayaan Bootstrap mAP50 (V2-E-023)
.venv/bin/python scripts/dump_classaware.py \
  --bobot runs/yolo26l_e60_i1280_rgbd352_edge/weights/best.pt \
  --data /workspace/SawitMVC-Depth-4ch-edge-YOLO --split test \
  --out results/pred_edge_test.npz

.venv/bin/python scripts/bootstrap_map.py --split test --n-boot 1000 \
  --sumber results/pred_edge_test.npz results/pred_rgb352_test.npz \
  --nama edge_rgbd yolo_rgb --out results/bootstrap_map.json

# 3. Uji Lokalisasi Murni Modalitas Depth 4-Kanal vs RGB (V2-E-024)
.venv/bin/python scripts/train_yolo_4ch_screening.py \
  --data /workspace/agnostic352_4ch/data.yaml \
  --epochs 60 --patience 45 --imgsz 1280 --batch 4 --seed 42 \
  --weights runs/agn953_full/weights/best.pt --name agn352_4ch

.venv/bin/python scripts/bootstrap_map.py --split test --agnostik --n-boot 1000 \
  --sumber results/pred_agn352_4ch_test.npz results/pred_agn352_ft3_test.npz \
  --nama agn352_4ch agn352_ft3_rgb --out results/bootstrap_lokalisasi.json

# 4. Evaluasi Partisi Uji Bersih agn953_full (V2-E-025)
.venv/bin/python scripts/buat_test_953_bersih.py
```

---

## 8. Katalog 9 Jebakan Operasional (*Silent Failures*)

| No. | Jebakan Operasional | Dampak Kritis Jika Terabaikan |
|---|---|---|
| 1 | **Pemotongan Jadwal Cosine Learning Rate di Tengah** | Fase peluruhan laju belajar tidak terjadi; kehilangan $\approx 5,0\text{ pp } AP50$. |
| 2 | **Toleransi Penghentian Dini Terlalu Ketat pada Penyesuaian Terarah** | Puncak performa semu pada epoch 1 mematikan pelatihan sebelum konvergensi sejati dimulai ($0,6413$ vs $0,7473$). |
| 3 | **Penomoran Otomatis Direktori Ultralytics (`run`, `run2`)** | Skrip hilir membaca direktori lama secara diam-diam. Jalur direktori wajib di-resolve secara dinamis. |
| 4 | **Pemuatan Checkpoint RT-DETR Menggunakan Kelas `YOLO`** | Model dibangun sebagai model konvolusi standar tanpa pesan galat, merusak integritas inferensi. Wajib menggunakan kelas `RTDETR()`. |
| 5 | **Resolusi Citra Terpotong (*Crop*) Tidak Konsisten** | *Upscaling* citra kecil saat inferensi mengaburkan tekstur warna dan mereduksi akurasi klasifikasi. |
| 6 | **Penyetelan Ambang Model Langsung pada Partisi Uji** | Terjadi kebocoran metodologis (*test peeking*); metrik uji menjadi tidak valid. |
| 7 | **Augmentasi Fotometrik Terlalu Ekstrem** | Distorsi warna menghapus label kematangan alami buah ($0,648 \to 0,471$). |
| 8 | **Ketiadaan Kanal Mask Kotak Pembatas pada Citra Terpotong** | Model tidak mampu membedakan tandan target saat terdapat $>1$ tandan dalam satu *crop*. |
| 9 | **Asumsi Peningkatan Deteksi via Test-Time Augmentation (TTA)** | Opsi `augment=True` pada Ultralytics tidak berpengaruh pada arsitektur tertentu; tidak boleh diasumsikan membawa peningkatan tanpa verifikasi. |
