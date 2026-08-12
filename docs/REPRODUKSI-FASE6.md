# Cara Mereproduksi Fase 6

Urutan persis untuk membangun ulang seluruh hasil Fase 6 dari nol. Setiap
langkah menyebutkan keluaran yang dihasilkan dan entri `EKSPERIMEN.md` yang
mengutipnya, supaya tiap angka bisa ditelusuri ke perintah yang membuatnya.

Semua dijalankan dari `/workspace/project-expertise` dengan `.venv` yang
dibangun dari `Research-Pipeline/experiments/code/requirements.txt`
(+ `timm`, lihat §0).

## 0. Prasyarat

```bash
python3 -m venv .venv --system-site-packages
.venv/bin/pip install -r /workspace/Research-Pipeline/experiments/code/requirements.txt
.venv/bin/pip install timm            # untuk classifier crop
```

Data mentah: `SawitMVC-YOLO/` (953 pohon) dan `SawitMVC-Depth/` (352 pohon)
dari HuggingFace, plus `depth_png_352/` hasil `reproject_depth.py` (Volume 1).

## 1. Diagnostik (read-only, ~5 menit, tanpa GPU)

```bash
.venv/bin/python scripts/probe_depth_signal.py --probe semua
```

Menghasilkan seluruh angka di `docs/DIAGNOSIS-DEPTH.md` dan entri
**V2-E-012/013/014**: distribusi kelas, cakupan depth dalam box, relief per
kelas + Kruskal-Wallis, tabel kuantisasi, AUC vs pooling.

## 2. Split bebas kebocoran dan dataset turunan

```bash
.venv/bin/python scripts/make_pretrain_split.py        # 846 pohon 953, irisan nol
.venv/bin/python scripts/make_agnostic_dataset.py      # agnostic953 + agnostic352 (1 kelas)
.venv/bin/python scripts/build_crop_dataset.py --src 352 --workers 8
.venv/bin/python scripts/build_crop_dataset.py --src 953 --workers 8
```

`make_pretrain_split.py` dan `make_agnostic_dataset.py` **assert** irisan nol
terhadap `val_trees.txt`/`test_trees.txt` 352 — kalau bocor, keduanya berhenti.

## 3. Detektor class-agnostic (V2-E-017/018)

```bash
# pretrain 953 — jadwal cosine harus SELESAI, jangan dipotong di tengah
.venv/bin/python scripts/train_yolo_4ch_screening.py \
  --data /workspace/agnostic953/data.yaml --epochs 12 --patience 12 \
  --imgsz 1280 --batch 4 --weights yolo26l.pt --name agn953_full

# finetune 352 — patience LONGGAR; transfer kuat bisa membuat epoch 1 jadi
# puncak palsu dan patience ketat membunuh run sebelum kurva sebenarnya mulai
.venv/bin/python scripts/train_yolo_4ch_screening.py \
  --data /workspace/agnostic352/data.yaml --epochs 60 --patience 45 \
  --imgsz 1280 --batch 4 --weights runs/agn953_full/weights/best.pt --name agn352_ft3

# RT-DETR-L sebagai anggota ensemble (arsitektur berbeda -> galat berbeda)
.venv/bin/python -c "
from ultralytics import RTDETR
RTDETR('rtdetr-l.pt').train(data='/workspace/agnostic352/data.yaml', epochs=60,
    patience=10, imgsz=1280, batch=4, seed=42, cos_lr=True,
    project='/workspace/project-expertise/runs', name='agn352_rtdetr')"
```

Ukur plafon lokalisasi:

```bash
.venv/bin/python scripts/eval_detector_agnostic.py \
  --detektor runs/agn352_ft/weights/best.pt --split test
```

## 4. Classifier kematangan pada crop (V2-E-015/016/021)

```bash
# pretrain 953
.venv/bin/python scripts/train_crop_classifier.py --tahap pretrain --mode rgb \
  --head hybrid --backbone convnext_small.fb_in22k_ft_in1k --img 176 \
  --epochs 14 --batch 24 --name pre953s

# finetune 352, 3 seed untuk ensemble
for s in 42 101 202; do
  .venv/bin/python scripts/train_crop_classifier.py --tahap finetune --mode rgb \
    --head hybrid --backbone convnext_small.fb_in22k_ft_in1k --img 176 \
    --epochs 50 --batch 24 --seed $s --init runs_fase6/pre953s/best.pt --name ftS_$s
done
```

Ablasi depth (V2-E-016) — ganti `--mode rgb` jadi `--mode rgbd`, dan uji
statistik depth terpool:

```bash
.venv/bin/python scripts/probe_fitur_depth.py --model runs_fase6/ftS_202/best.pt
```

## 5. Pemilihan, sweep, dan rekomposisi (V2-E-019/020)

**Pemilihan selalu di `--split val`.** Test hanya dipakai untuk angka akhir.

```bash
.venv/bin/python scripts/pilih_detektor.py --split val \
  --kandidat runs/agn352_ft/weights/best.pt runs/agn352_ft3/weights/best.pt \
             runs/agn352_rtdetr/weights/best.pt \
  --out results/detektor_pilihan.json

.venv/bin/python scripts/sweep_inferensi.py --split val \
  --detektor runs/agn352_ft/weights/best.pt runs/agn352_ft3/weights/best.pt \
  --out results/sweep_inferensi.json

# angka akhir (mAP50 sebanding Fase 1-5)
.venv/bin/python scripts/eval_twostage.py --split test --conf 0.005 \
  --detektor runs/agn352_ft/weights/best.pt runs/agn352_ft3/weights/best.pt \
  --classifier runs_fase6/ftS_*/best.pt runs_fase6/ftJ_*/best.pt runs_fase6/ftG_*/best.pt \
  --tta --multi-kelas --imgsz 1280 --det-iou 0.5 \
  --out results/twostage_final_v4.json

# counting, memakai fungsi Ridge+F_all yang SAMA dengan Fase 1-5
.venv/bin/python scripts/run_counting_twostage.py --tta --imgsz 1280 --det-iou 0.5 \
  --detektor runs/agn352_ft/weights/best.pt runs/agn352_ft3/weights/best.pt \
  --classifier runs_fase6/ftS_*/best.pt runs_fase6/ftJ_*/best.pt runs_fase6/ftG_*/best.pt \
  --label TwoStage-FINAL_v4 --out results/counting_twostage.json
```

## 6. Rangkuman seluruh angka

`results/fase6_ringkas.json` memuat metrik tiap detektor, tiap classifier,
tiap versi rekomposisi, dan tiap counting dalam satu berkas.

---

## Hal yang WAJIB diperhatikan saat mereproduksi

| Jebakan | Akibat kalau diabaikan |
|---|---|
| **Jangan potong jadwal cosine di tengah.** `agn953_pre-2` dihentikan di epoch 4 dari 25 sehingga fase anneal tidak pernah terjadi. | Kehilangan ~5,0 poin AP50 pretrain. |
| **Patience longgar untuk finetune ber-transfer kuat.** `agn352_ft2` mati di epoch 11 karena epoch 1 jadi puncak palsu. | Run berhenti sebelum kurva sebenarnya dimulai (0,6413 vs 0,7473). |
| **Ultralytics auto-increment nama run** (`agn953_pre` → `agn953_pre-2`). | Tahap berikutnya menunjuk direktori kosong dan diam-diam memakai bobot default. Selalu resolve direktori secara dinamis. |
| **Muat RT-DETR dengan kelas `RTDETR`, bukan `YOLO`.** `YOLO()` menerima bobotnya tanpa error tapi membangunnya sebagai `DetectionModel` biasa. | Hasil inference tidak bisa dipercaya, tanpa pesan error apa pun. |
| **Resolusi crop saat inference harus ≥ resolusi saat training.** | Crop kecil diperbesar ke ukuran input → detail warna hilang, gain lenyap tanpa error. |
| **Pemilihan detektor/konfigurasi di `val`, bukan `test`.** | Angka test menjadi tidak sah (mengepaskan model ke angka laporan). |
| **Augmentasi fotometrik harus RINGAN.** Kematangan didefinisikan oleh warna; jitter brightness ±25% dan saturasi 0,6–1,4 menghapus label. | Akurasi classifier turun ~18 poin (0,648 → 0,471). |
| **Crop butuh kanal mask box.** Dengan ctx=1,6 di kanopi padat sering ada >1 tandan per crop. | Model tidak tahu tandan mana yang dinilai. |
| **`augment=True` (TTA deteksi) tidak berpengaruh** pada YOLO26 di ultralytics 8.4. | Mengira ada gain padahal nol. |
