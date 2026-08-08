# CLAUDE.md — Panduan Kerja

Baca seluruhnya sebelum mengubah apa pun.

## Bahasa

Seluruh isi repo dan percakapan memakai **Bahasa Indonesia**.
Istilah teknis asing ditulis apa adanya tanpa diterjemahkan.

## Apa Ini

Repo eksperimen untuk membandingkan tiga arsitektur detektor (YOLO26l,
RT-DETR-L, RF-DETR-L) pada dataset RGB dan RGB+Depth (4-kanal), lalu
mengukur dampaknya terhadap deteksi, klasifikasi kematangan, dan counting
tandan sawit per pohon.

**Bukan** repo tinjauan pustaka — itu ada di
[Research-Pipeline](https://github.com/muhammad-zainal-muttaqin/Research-Pipeline).

## Dataset

Dua dataset, keduanya CC BY-NC 4.0, dari ULM-DS-Lab:

| | SawitMVC | SawitMVC-Depth |
|---|---|---|
| Sumber | [HuggingFace](https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-YOLO) | [HuggingFace](https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-Depth) |
| Pohon | 953 | 352 |
| Citra | 3.992 | 1.408 |
| Resolusi | 960 x 1280 | 1.280 x 800 |
| Bbox | 18.540 | 2.299 |
| Depth | Tidak | Ya (Orbbec, uint16 mm) |
| Split | 716 / 96 / 141 | Perlu dibuat |

Detail lengkap: [docs/DATASET.md](docs/DATASET.md).

## Metrik yang berlaku saat ini

Satu-satunya angka deteksi yang boleh dikutip (dari E-021, SawitMVC 953 pohon):

| Model | Test mAP50 | Test mAP50-95 |
|---|---|---|
| YOLO26l @ 1280 | 0,5300 | 0,2568 |
| RT-DETR-L @ 1280 | 0,5784 | 0,2707 |
| RF-DETR-L @ 1280 | 0,6038 | 0,2770 |

Angka counting terbaik (Baseline-SawitMVC, YOLO26m):

| Counter | Class &plusmn;1 Acc | Tree &plusmn;1 Acc | Macro MAE |
|---|---|---|---|
| Ridge + F_all | 77,48% | 32,62% | 1,036 |

Angka counting untuk YOLO26l, RT-DETR-L, RF-DETR-L **belum ada**.

## Aturan eksperimen

- Satu entri = satu hipotesis yang falsifiable.
- Append-only. Jangan edit entri lama.
- Hasil negatif wajib dicatat dengan bobot yang sama.
- Angka apa adanya. Jangan dibungkus.
- Setiap klaim numerik harus terlacak ke sumber (skrip, JSON, log).
- Evaluasi deteksi: `pycocotools` (mengikat dari E-025).
- Evaluasi counting: pipeline dari Baseline-SawitMVC.
- Setiap angka menyebut dataset dan split.

## Hal yang sudah dicoba dan GAGAL (jangan diulang)

Daftar lengkap: [docs/REKAP.md](docs/REKAP.md) bagian "Percobaan Gagal".
Ringkasan singkat:

1. **Early fusion (depth sebagai kanal ke-4 langsung)** — regresi, bukan
   peningkatan (E-022, E-027). Depth merugikan YOLO26n sebesar −0,0230 mAP.
2. **Tuning hyperparameter** — sudah habis dijalankan, tidak naik lagi.
3. **SAHI dan teknik siap-pakai** — sudah dicoba sendiri oleh pengguna, tidak
   satu pun menaikkan mAP.
4. **Gate init-nol pada cabang samping** — gate tidak pernah terbuka, γ ≈ 0 (F-007).
5. **Konsistensi lintas-sisi** — plafon hanya 0,2794 (F-003).
6. **Fusi menengah/akhir dari nol** — tidak konklusif, semua CI memuat nol (E-032).

## Hal yang BERHASIL (boleh dibangun di atasnya)

1. **RF-DETR-L** adalah detektor terbaik saat ini (mAP50 0,6038).
2. **Pipeline counting Ridge + F_all** sudah modular dan established.
3. **Reproyeksi depth** ke RGB sudah tervalidasi untuk SawitMVC-Depth.
4. **Frekuensi tinggi memisahkan tandan dari pelepah** (+0,0731 B4, F-002).

## Cara kerja

- Paralelisme dibatasi VRAM, bukan slot tetap. Baca `nvidia-smi` sebelum
  menyalakan run berikutnya.
- Jangan mengarang eksperimen tambahan untuk "mengisi GPU".
- Laporkan hasil apa adanya.

## Status sesi & handoff (2026-08-08) — BACA INI DULU kalau melanjutkan sesi

Sesi sebelumnya dihentikan sengaja karena user pindah ke GPU yang lebih kuat
(GPU sebelumnya, RTX A4500, terbukti compute-bound — lihat
`docs/CATATAN-TEKNIS-FASE1.md`, ~4-5 jam/60-epoch, jauh lebih lambat dari
referensi L4 ~1 jam). **Sesi berikutnya: minta ke user GitHub personal
access token (untuk push ke repo ini & clone `research-pipeline` kalau perlu)
dan HuggingFace token (untuk `ULM-DS-Lab/SawitMVC-Depth`, repo private).**

### Link penting

- Repo ini: https://github.com/muhammad-zainal-muttaqin/project-expertise
- Volume 1 (toolbox training/eval, read-only, JANGAN diedit langsung):
  https://github.com/muhammad-zainal-muttaqin/research-pipeline
- Pipeline counting: https://github.com/ULM-SawitMVC/Baseline-SawitMVC
- Dataset: https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-YOLO (publik),
  https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-Depth (privat, butuh token)

### Status Fase 0 — SELESAI

Semua sudah di-commit (lihat riwayat git). Ringkasan: `Baseline-SawitMVC`
di-clone ke `/workspace/Baseline-SawitMVC`, Ridge+F_all tereproduksi persis
(77,48%/32,62%/1,0355 via `experiments/exp_counting_v3.py`, BUKAN
`run_e2e_pipeline.py` — lihat `docs/SCHEMA-PERTREE.md`), skema GT SawitMVC vs
SawitMVC-Depth identik (tidak perlu shim), split SawitMVC-Depth-YOLO
(70/15/15, seed 10) di-reuse langsung, adaptor `scripts/adapters/` lolos
smoke test, depth tereproyeksi penuh (1408 file, Z 0,8-15,0m).

### Status Fase 1 — SEBAGIAN (retrain 953 pohon)

Bobot E-021 asli (Volume 1) hilang — semua di bawah ini adalah retrain ulang
("_v2repro"), bukan bobot asli.

| Model | Status | Hasil |
|---|---|---|
| YOLO26l | **Selesai**, lolos reproduksi | test mAP50=0,5435 (target 0,5300), mAP50-95=0,2565 (target 0,2568) |
| RT-DETR-L | **Dihentikan sengaja di epoch 6/60** untuk pindah GPU | checkpoint ada di `research-pipeline/evidence/experiments/runs/rtdetr_l_e60_i1280_v2repro/weights/{best,last}.pt` (kalau volume network `/workspace` persisten ke pod baru) — kalau tidak persisten, ULANGI dari nol, jangan coba resume |
| RF-DETR-L | Belum mulai | Config override wajib: lihat `docs/RENCANA.md` Fase 1.3 (epochs 60, resolution 1280, batch 8, grad-accum 2, seed 42 — default CLI skrip TIDAK sama) |

**Bobot YOLO26l_v2repro SUDAH aman** — di-commit ke git repo ini di
`models/yolo26l_e60_i1280_v2repro/best.pt` (51MB, memakai pengecualian
`!models/**/*.pt` di `.gitignore`). Tinggal `git pull`/`git clone`, tidak
perlu retrain ulang. RT-DETR-L (cuma 6/60 epoch saat dihentikan) dan
RF-DETR-L (belum mulai) TIDAK disimpan — retrain dari nol di GPU baru.

**Cara lanjut (di GPU baru):**
1. Cek dulu apakah `/workspace` (mount `mfs#euro-3.runpod.net:9421`) masih
   ada isinya dari sesi lalu (`ls /workspace/research-pipeline`,
   `ls /workspace/SawitMVC`). Kalau iya, repo dan dataset sudah lengkap,
   tinggal `git pull` di `project-expertise`. Kalau tidak, clone ulang dari
   link di atas dan unduh dataset dari HuggingFace pakai token. Bobot
   YOLO26l ikut ter-clone otomatis (ada di `models/` repo ini), tidak perlu
   diambil dari `/workspace` lagi.
2. Cache dataset ke disk lokal **kalau I/O ternyata jadi bottleneck lagi** —
   tapi jangan asumsikan otomatis membantu; ukur dulu (lihat
   `docs/CATATAN-TEKNIS-FASE1.md`, di RTX A4500 ternyata GPU compute yang
   jadi batas, bukan I/O). Skrip pembantu: `scripts/train_ultra_local.py`.
3. Lanjut retrain RT-DETR-L dari nol (bukan resume) dengan
   `scripts/train_ultra_local.py --arch rtdetr` (lihat docstring skrip).
4. Retrain RF-DETR-L (config di atas), lalu eval pycocotools ketiganya
   (`research-pipeline/reproduce/experiments/eval/eval_all_pycoco_v2repro.py`
   — SUDAH ada, arahkan ke bobot `_v2repro`, JANGAN edit `eval_all_pycoco.py`
   asli).
5. Inference + counting 953 pohon pakai adaptor `scripts/adapters/` + pola
   `exp_counting_v3.py` (fit Ridge segar per detektor, BUKAN
   `run_e2e_pipeline.py`).
6. Tulis `V2-E-001`/`V2-E-002` di `experiments/EKSPERIMEN.md`, commit+push.
7. Fase 2 (RGB 352 pohon): data yaml lokal sudah disiapkan template-nya —
   `path: <cache>/SawitMVC-Depth-YOLO` (yaml `SawitMVC-Depth-YOLO/data.yaml`
   sudah portable, `path: .`, tinggal override).
8. Fase 3 (RGB+D 352 pohon) lalu **Fase 5 (loop perbaikan RGB+D)** — baca
   `docs/RENCANA.md` bagian Fase 5: **HANYA dua lever yang boleh dicoba,
   representasi dataset ATAU arsitektur model** (dilarang tuning
   hyperparameter/SAHI/ensembling). Screening cepat: **maks 15 epoch,
   patience 3**, baru full 60 epoch kalau kandidat lolos screening.
9. Fase 4: kompilasi matriks final + laporan.

### Catatan lain

- `research-pipeline` (repo terpisah) punya perubahan uncommitted milik user
  dari sebelum sesi ini (`eval_e022_paired.py`, `eval_e022_pycoco.py`,
  `eval_rfdetr_e022.py`, `train_rfdetr_4ch.py` modified) — **bukan dari sesi
  ini, jangan di-commit tanpa konfirmasi user**, itu kemungkinan kerjaan lain
  yang belum selesai.
- `research-pipeline` juga punya artefak baru dari sesi ini yang belum
  di-push ke remote-nya (masih di working tree lokal, di `/workspace`):
  `eval_all_pycoco_v2repro.py`, folder run `yolo26l_e60_i1280_v2repro` &
  `rtdetr_l_e60_i1280_v2repro` (JSON/CSV saja, bobot gitignored),
  `docs/experiments/KOREKSI-SIDECAR-SAWITMVC-DEPTH.md`, beberapa skrip build
  baru (`materialize_yolo_split.py`, `patch_sidecar_metadata.py`,
  `push_hf_v110.py`, `push_hf_yolo.py`, `repack_jpeg_padding.py`). Kalau
  `/workspace` tidak persisten ke pod baru, ini semua hilang kecuali
  di-push manual — bukan keputusan sesi ini untuk push otomatis karena
  bercampur dengan perubahan user yang belum dikonfirmasi.
