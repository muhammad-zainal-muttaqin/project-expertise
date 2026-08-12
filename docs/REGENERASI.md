# Regenerasi Data Turunan

Sebagian data turunan **dihapus dari `/workspace` dan dari bucket backup pada
2026-08-12** saat proyek ditutup, karena semuanya bisa dibangun ulang dari
skrip. Dokumen ini adalah resep lengkapnya: apa yang hilang, perintah persis
untuk membuatnya kembali, dan cara memastikan hasilnya benar.

Tidak ada satu pun bobot terlatih, log training, atau berkas hasil yang
dihapus. Yang dibuang hanya masukan antara yang bisa dihitung ulang.

**Prasyarat:** lingkungan Python sesuai [`../requirements-freeze.txt`](../requirements-freeze.txt)
(Python 3.12.3; torch `+cu128` butuh `--extra-index-url`, lihat kepala berkas
itu), plus kedua dataset sumber di `/workspace/SawitMVC-YOLO` dan
`/workspace/SawitMVC-Depth`.

## Apa yang masih ada dan tidak perlu dibangun ulang

| Folder | Kenapa disimpan |
|---|---|
| `SawitMVC-YOLO/`, `SawitMVC-Depth/` | Dataset sumber. Juga ada di HF Hub (`ULM-DS-Lab/*`, privat). **Split kanonik 245/52/55 ada di `SawitMVC-Depth/splits/canonical_70_15_15{,_tiff}`** — dasar semua hasil 352 pohon |
| `depth_png_352/` | Perantara wajib untuk seluruh dataset 4-kanal. 0,22 GB, murah disimpan. Parameter yang dibekukan ada di `depth_png_352/depth_meta.json` (`z_near=0,8`, `z_far=15,0`) |
| `SawitMVC-Depth-4ch-edge/` | Encoding pemenang Fase 5, dasar `V2-E-010` dan `V2-E-024` |
| `agnostic*/`, `SawitMVC-Depth-YOLO/`, `SawitMVC-Depth-4ch-edge-YOLO/` | Rak symlink, ≤11 MB di disk. Dikeluarkan dari backup lewat `--exclude`, tapi tidak dihapus dari `/workspace` |
| `project-expertise/runs/`, `runs_fase6/`, semua `*.pt` | **Bobot dan log training — jangan pernah dihapus.** 6,45 GB, mahal dihasilkan ulang |

## Urutan ketergantungan

```
SawitMVC-Depth/data/*.raw
  └─ reproject_depth.py ─────────────► depth_png_352/            [ADA]
       │
SawitMVC-Depth/images + depth_png_352/
  └─ build_4ch_dataset.py ───────────► SawitMVC-Depth-4ch/       [DIHAPUS]  ~5 mnt
       └─ create_depth_edge_dataset.py --encoding <enc>
                                     ► SawitMVC-Depth-4ch-<enc>/ [edge ADA; clipped & valid_mask DIHAPUS]  ~10 mnt/varian
       └─ materialize_split_dirs.py ─► SawitMVC-Depth-4ch*-YOLO/ [4ch-YOLO DIHAPUS]  <1 mnt

SawitMVC-YOLO + SawitMVC-Depth
  ├─ make_agnostic_dataset.py ───────► agnostic953/, agnostic352/          [ADA]
  ├─ buat_agnostic352_4ch.py ────────► agnostic352_4ch/                    [ADA]
  ├─ buat_test_953_bersih.py ────────► agnostic953_test_{bersih,penuh}/    [ADA]
  └─ build_crop_dataset.py ──────────► crops_fase6/, crops_fase6_256/      [DIHAPUS]  ~15 mnt masing-masing
```

`create_depth_edge_dataset.py` membaca dari `SawitMVC-Depth-4ch/images`, jadi
**basis `inverse` harus dibangun lebih dulu** sebelum varian encoding mana pun.

## Perintah

Semua dijalankan dari `/workspace/project-expertise`.

### 1. `SawitMVC-Depth-4ch/` — basis 4-kanal, encoding `inverse` (~5 mnt, 3,9 GB)

Dipakai `V2-E-005`/`V2-E-006` (early fusion). Wajib ada sebelum varian lain.

```bash
.venv/bin/python scripts/build_4ch_dataset.py \
  --rgb-dir   /workspace/SawitMVC-Depth/images \
  --depth-dir /workspace/depth_png_352 \
  --out-dir   /workspace/SawitMVC-Depth-4ch/images \
  --workers 8
```

Verifikasi: `ls /workspace/SawitMVC-Depth-4ch/images/*.tiff | wc -l` → **1408**.
Tiap TIFF 4 kanal `[B,G,R,D]`, ukuran sama dengan RGB-nya.

### 2. Varian encoding depth (~10 mnt per varian, 3,8–4,0 GB masing-masing)

```bash
for enc in clipped valid_mask; do
  .venv/bin/python scripts/create_depth_edge_dataset.py \
    --encoding "$enc" \
    --src /workspace/SawitMVC-Depth-4ch/images \
    --dst /workspace/SawitMVC-Depth-4ch-$enc/images \
    --workers 8
done
```

`edge` tidak perlu dibangun ulang — masih ada. Kalau toh hilang, ganti
`$enc` di atas dengan `edge`.

Keduanya **kalah screening Fase 5** (val mAP50 `clipped` 0,3221,
`valid_mask` 0,3321, vs `edge` 0,3777) dan hanya perlu dibangun kalau
screening itu mau diulang. Angkanya sudah tercatat di
[`../experiments/STATUS.md`](../experiments/STATUS.md).

### 3. `SawitMVC-Depth-4ch-YOLO/` — struktur `{split}/images,labels` (<1 mnt)

Dibutuhkan `eval_pycoco_rgbd352.py` dan `run_counting_rgbd352.py`, yang
mengharapkan pola direktori lama, bukan direktori flat + `train/val/test.txt`.

```bash
.venv/bin/python scripts/materialize_split_dirs.py \
  --src-images /workspace/SawitMVC-Depth-4ch/images \
  --src-labels /workspace/SawitMVC-Depth/labels \
  --splits-dir /workspace/SawitMVC-Depth/splits/canonical_70_15_15_tiff \
  --out-root   /workspace/SawitMVC-Depth-4ch-YOLO
```

Symlink saja, tidak menyalin byte. Verifikasi: 2.816 entri
(1.408 citra + 1.408 label).

### 4. `crops_fase6/` — crop tandan + relief depth + mask (~15 mnt, 3,3 GB)

Masukan classifier kematangan Fase 6 (`V2-E-015`..`V2-E-021`).

```bash
.venv/bin/python scripts/build_crop_dataset.py --src 352 --workers 8
.venv/bin/python scripts/build_crop_dataset.py --src 953 --workers 8
```

Verifikasi terhadap angka yang tercatat: 2.299 crop dari 352 pohon dan
16.542 crop dari 953 pohon; `crops352_ringkas.json` harus melaporkan
`per_split` = 1.517 train / 372 val / 410 test.

**Deterministik** — diverifikasi 2026-08-12 dengan membangun ulang varian 352
ke direktori terpisah, lalu membandingkan SHA-256 `crops352_{rgb,dep,msk}.npy`
terhadap berkas aslinya: identik byte-per-byte.

### 5. `crops_fase6_256/` — varian resolusi 256 (~15 mnt, 7,0 GB)

Hanya dipakai eksperimen `ftH` (classifier crop 256 @224), yang **tidak
menolong** (test 0,6569, grup terlemah — lihat `STATUS.md`). Bangun ulang cuma
kalau eksperimen itu mau diulang.

```bash
.venv/bin/python scripts/build_crop_dataset.py --src 352 --workers 8 \
  --sisi 256 --out /workspace/crops_fase6_256
.venv/bin/python scripts/build_crop_dataset.py --src 953 --workers 8 \
  --sisi 256 --out /workspace/crops_fase6_256
```

### 6. Dataset class-agnostic (detik, kalau sewaktu-waktu hilang)

Ketiganya masih ada di `/workspace`; perintahnya dicatat di sini supaya
lengkap.

```bash
.venv/bin/python scripts/make_agnostic_dataset.py     # agnostic953 + agnostic352
.venv/bin/python scripts/buat_agnostic352_4ch.py      # agnostic352_4ch  (butuh 4ch-edge)
.venv/bin/python scripts/buat_test_953_bersih.py      # test bersih 19 pohon + test penuh
```

`make_agnostic_dataset.py` memakai `RandomState(42)` untuk memecah 846 pohon
pretraining jadi train/val — deterministik, dan ia **assert** irisan nol
terhadap val/test 352.

`buat_agnostic352_4ch.py` bisa memverifikasi dirinya sendiri terhadap direktori
acuan:

```bash
.venv/bin/python scripts/buat_agnostic352_4ch.py \
  --out /tmp/uji --periksa /workspace/agnostic352_4ch
```

### 7. `.venv/` — lingkungan Python (~5 mnt)

```bash
cd /workspace/project-expertise
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-freeze.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

181 paket ter-pin. Tanpa `--extra-index-url`, `torch==2.8.0+cu128` dan
`torchvision==0.23.0+cu128` gagal dipasang — keduanya tidak ada di PyPI biasa.

## Kalau semuanya harus dibangun ulang dari nol

Urutan minimal untuk mengembalikan seluruh yang dihapus, ~45 menit:

```bash
cd /workspace/project-expertise
.venv/bin/python scripts/build_4ch_dataset.py \
  --rgb-dir /workspace/SawitMVC-Depth/images --depth-dir /workspace/depth_png_352 \
  --out-dir /workspace/SawitMVC-Depth-4ch/images --workers 8
for enc in clipped valid_mask; do
  .venv/bin/python scripts/create_depth_edge_dataset.py --encoding "$enc" \
    --src /workspace/SawitMVC-Depth-4ch/images \
    --dst /workspace/SawitMVC-Depth-4ch-$enc/images --workers 8
done
.venv/bin/python scripts/materialize_split_dirs.py \
  --src-images /workspace/SawitMVC-Depth-4ch/images \
  --src-labels /workspace/SawitMVC-Depth/labels \
  --splits-dir /workspace/SawitMVC-Depth/splits/canonical_70_15_15_tiff \
  --out-root /workspace/SawitMVC-Depth-4ch-YOLO
.venv/bin/python scripts/build_crop_dataset.py --src 352 --workers 8
.venv/bin/python scripts/build_crop_dataset.py --src 953 --workers 8
```

Kalau `depth_png_352/` ikut hilang, ia dibangun oleh
`Research-Pipeline/experiments/code/build/reproject_depth.py` dari
`SawitMVC-Depth/data/*.raw`, dengan parameter yang dibekukan di
`depth_png_352/depth_meta.json`: `--z-near 0.8 --z-far 15.0`. Nilai itu
dipilih dari histogram split **train** saja (menghitungnya atas test adalah
kebocoran) — jangan pilih ulang, pakai yang beku.

## Catatan untuk sesi berikutnya

`hf buckets list` melaporkan ukuran **logis**. Sebelum pemangkasan, bucket
tampak 57,64 GB padahal isinya 41,24 GB — sisanya duplikat byte-identik
(rak symlink yang di-dereference `hf sync`) yang sudah di-dedup `xet`.
Menghapus duplikat semacam itu tidak membebaskan storage sama sekali, hanya
memperkecil jumlah berkas. Kalau suatu saat ingin memangkas lagi, ukur dulu
byte eksklusif per folder — jangan mengurutkan berdasarkan ukuran logis.
