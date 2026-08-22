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
- **Dump prediksi test ke `.npz` pada saat evaluasi, bukan belakangan.**
  Pakai `scripts/dump_classaware.py` (atau pola `np.savez_compressed` di
  `eval_twostage.py`). Alasannya konkret: bobot RT-DETR-L dan RF-DETR-L
  Volume 2 hilang antara 9 dan 10 Agustus 2026 sebelum sempat di-backup, dan
  karena prediksinya tidak pernah disimpan, **CI untuk kedua arsitektur itu
  tidak bisa dihitung selamanya** (lihat `docs/LAPORAN-AKHIR.md` §Ancaman
  validitas butir 3). Satu berkas `.npz` ~90 KB menutup risiko itu.
- **Riwayat per-epoch WAJIB ikut ter-commit ke git.** `runs/` ada di
  `.gitignore`, dan `results.csv` hidup di dalamnya — jadi kalau tidak disalin
  keluar, riwayat per-epoch hilang bersama direktori run. Setiap run yang
  masuk matriks harus menyalin ke `results/riwayat_epoch/` dengan pola
  `<nama_run>__<berkas>`:

  | Berkas | Isi | Ukuran khas |
  |---|---|---|
  | `results.csv` | mAP50/mAP50-95/loss/lr per epoch | 2–7 KB |
  | `args.yaml` | resep yang BENAR-BENAR dijalankan | 1,7 KB |
  | `DIHENTIKAN_LEBIH_AWAL` | alasan + epoch, kalau ada | <1 KB |

  Log training juga ikut, diringkas ke `logs_ringkas/` (buang progress bar:
  `tr '\r' '\n' | sed 's/\x1b\[[0-9;]*[A-Za-z]//g'` lalu saring baris
  `^ *[0-9]+/60 +[0-9.]+G` — 12 MB jadi ~2 MB). Yang berharga di sana adalah
  peringatan citra korup, baris `Scanning`, dan konfirmasi batch.

  **Kenapa wajib:** `results.csv` run sel 5 (`yolo26l_e60_i1280_v2repro`) hilang
  bersama direktori run-nya. Kurva val 60 epoch-nya baru bisa dipulihkan
  2026-08-15 karena kebetulan ultralytics menyimpannya di kunci `train_results`
  di dalam `best.pt`. Kalau bobotnya ikut hilang — dan bobot RT-DETR-L serta
  RF-DETR-L Volume 2 memang sudah hilang — riwayat itu hilang permanen. Total
  biayanya 19 KB per tiga run; tidak ada alasan melewatkannya.

- **Bobot tidak boleh dihapus** — `*.pt`, `*.pth`, `runs*/`, `models/`, di
  lokal maupun di bucket backup. Lihat ATURAN #1 di `/workspace/CLAUDE.md`.
  Git tidak bisa menampungnya (`runs/` di-gitignore, ukurannya ratusan MB per
  run), jadi **bobot butuh jalur backup terpisah ke bucket HF** — dan loop sync
  otomatis sudah dihentikan permanen sejak 2026-08-12, artinya sync itu
  sekarang manual dan harus dijalankan sadar. Commit git saja TIDAK melindungi
  bobot.

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
7. **Mengganti detektor untuk menyelesaikan masalah hilir** — tidak pernah jadi
   pengungkit besar, dan sekarang ada ukurannya (PT-E-011, 2026-08-17). Dua
   detektor YOLO26l yang dilatih di korpus berbeda ternyata **mutunya setara**:
   presisi deteksi 0,584 (953) lawan 0,639 (352), dan yang 953 justru **lebih
   baik recall-nya** (0,823 lawan 0,739). Yang tampak seperti "detektor 953 lebih
   kotor" — 6,26 deteksi/citra lawan 2,15 — hampir seluruhnya cuma **kepadatan
   objek**: 4,44 kotak GT/citra lawan 1,86.
   Konsekuensinya untuk tugas hilir apa pun yang bergantung pada kandidat:
   **mengganti backbone tidak mengubah berapa banyak tandan ada di satu pohon.**
   Yang berubah cuma marginnya, dan pemilik data sudah memperkirakan 1-2% —
   perkiraan itu konsisten dengan angka di atas. Kalau sebuah masalah hilir
   ternyata kombinatorik (lihat butir BERHASIL #5), detektor bukan obatnya.
8. **Penampilan tangan (histogram HSV, warna, ketajaman) untuk mencocokkan
   tandan lintas-sisi** — menambah nol (PT-E-002a: AUC 0,9301 → 0,9307).
   Penyebabnya fisik: negatif yang harus dikalahkan semuanya dari **pohon yang
   sama**, jadi warnanya nyaris identik, sementara tandan yang sama justru
   berubah rupa dari sudut 90° berbeda.

## Hal yang BERHASIL (boleh dibangun di atasnya)

1. **RF-DETR-L** adalah detektor terbaik saat ini (mAP50 0,6038).
2. **Pipeline counting Ridge + F_all** sudah modular dan established.
3. **Reproyeksi depth** ke RGB sudah tervalidasi untuk SawitMVC-Depth.
4. **Frekuensi tinggi memisahkan tandan dari pelepah** (+0,0731 B4, F-002).
5. **Prior yang MEMANGKAS RUANG KANDIDAT** — pengungkit terkuat yang ditemukan
   sejauh ini untuk tugas pencocokan, dan cara berpikir yang layak dibawa ke
   masalah lain (PT-E-008, 2026-08-17).

   Foto diambil **memutari pohon searah jarum jam**. Seluruh fitur geometri
   sebelumnya memakai `|dx|` — nilai mutlak — sehingga arah pergeseran dibuang.
   Memakai pergeseran **bertanda** menaikkan F1 penautan **0,398 → 0,649** dan
   membalikkan dua gerbang dari gugur menjadi lolos. Konsistensi arahnya
   98,6%/99,7% di korpus 953 dan **98,4%/99,0% di korpus 352** — sesi akuisisi
   terpisah ~80 hari, kamera dan orientasi citra berbeda. Jadi ini sifat
   protokol pengambilan, bukan kebetulan satu sesi.

   **Kenapa ini bekerja, dan kenapa detektor tidak.** Mencocokkan tandan di 953
   berarti mencari ~10 pasangan benar di antara ~235 pasangan lintas-sisi per
   pohon (prevalensi ~4%); di 352, ~6 di antara ~28 (~21%). Masalahnya
   **kombinatorik**. Prior arah tidak mempertajam penilaian per pasangan — ia
   membuang sebagian besar kandidat sebelum dinilai. Itu obat yang tepat.
   Detektor yang lebih baik tidak mengurangi jumlah tandan per pohon.

   **Yang belum dicoba dan paling menjanjikan berikutnya:** prior lain yang
   memangkas kandidat, terutama **depth di korpus 352** — jarak fisik tandan
   dari kamera, digabung arah putar, praktis menetapkan posisi 3D-nya. Kaveat
   jujur: E-007 (Volume 1) sudah pernah memalsukan penautan berbasis depth,
   tetapi **tanpa prior arah dan tanpa penilai terlatih**.

6. **Agregasi multi-tampak menaikkan akurasi kelas** (PT-E-001/003). Pada tandan
   yang terlihat di >=2 sisi: **+4,36 pp** dengan tautan sempurna (CI95
   [+2,33; +6,25]) dan **+5,32 pp** saat pipeline berjalan tanpa GT sama sekali
   (CI95 [+2,09; +8,42]), replikasi val→test. Aturan yang menang: **ekspektasi
   ordinal** dengan ambang dilatih di val — bukan voting (85,5% tandan yang bisa
   di-pool hanya punya DUA sisi, jadi voting selalu seri) dan bukan rerata
   softmax biasa.

   Kaveat: butuh split val yang cukup besar. Di 352 (52 pohon val) ambang
   ordinalnya overfit dan aturan ini kalah dari rerata softmax di test.

## Sub-proyek `pipeline-pertandan/` (2026-08-17)

Satuan inferensi dipindah dari kotak-per-citra ke **tandan fisik per pohon**:
deteksi per sisi → penautan lintas-sisi → satu keputusan kelas per tandan.
Dua belas eksperimen (`PT-E-000`…`PT-E-011`) di dua dataset, penomoran terpisah
dari deret `V2-E-*` supaya tidak tabrakan.

Baca [`pipeline-pertandan/STATUS.md`](pipeline-pertandan/STATUS.md) dulu kalau
melanjutkan. Ringkasnya:

| Gerbang | Putusan | Angka (test 953) |
|---|---|---|
| G0 agregasi berguna | **LOLOS** | +4,36 pp, CI95 [+2,33; +6,25] |
| G1 penaut cukup baik | **LOLOS** | val F1 0,6718 / ARI 0,6139 |
| G2 pipeline utuh | **LOLOS** | −1,81 pp dari plafon oracle |
| G3 counting | **GUGUR** | macro MAE 3,66 lawan 1,0542 (Ridge+F_all) |

**Yang belum dikerjakan dan paling layak berikutnya:** modul **C3**, classifier
multi-tampak — satu model melihat SEMUA foto tandan itu sekaligus dan
memutuskan sendiri cara menggabungkannya, alih-alih tiap foto dinilai terpisah
lalu digabung rumus di luar model. Ini bentuk yang paling setia terhadap sketsa
asal (`pipeline-pertandan/docs/sketsa-asal-2026-07-22.png`), sah dikerjakan
karena G0 lolos, dan biayanya ~4 jam GPU. Batasnya jujur: ia hanya memperbesar
nilai dari tandan yang **berhasil** disatukan (22-29% di korpus 953, 60%+ di
352) dan tidak menyentuh masalah kepadatan.

**Untuk counting, jangan pakai jalur pool.** Ridge+F_all (1,0542) dan M01
Baseline-SawitMVC yang dikalibrasi ulang di deteksi (1,18) sudah mendekati batas
yang ditentukan detektor; menghitung pool mewarisi seluruh positif palsunya.

**Kaveat pengutipan yang penting.** Angka `Acc±1 87,62% / macro MAE 0,3746` di
`ULM-SawitMVC/Baseline-SawitMVC` `algorithms/README.md` adalah angka **kotak
GT**, bukan end-to-end — tereproduksi persis sampai empat desimal (PT-E-006).
Dengan deteksi nyata dari detektor repo itu sendiri, M01 memberi 1,1826. Jangan
mengutipnya berdampingan dengan angka end-to-end.

## Cara kerja

- Paralelisme dibatasi VRAM, bukan slot tetap. Baca `nvidia-smi` sebelum
  menyalakan run berikutnya.
- Jangan mengarang eksperimen tambahan untuk "mengisi GPU".
- Laporkan hasil apa adanya.

### Eval dan langkah pendek: JALANKAN DI DEPAN, jangan lewat runner

- **Evaluasi, bootstrap, pembangunan dataset, dan apa pun yang selesai dalam
  hitungan menit dijalankan langsung di depan**, bukan lewat
  `jalankan_matriks.sh` atau proses latar. Tujuannya satu: angkanya keluar
  sekarang, dan kalau gagal, galatnya terlihat saat itu juga.
- **Runner hanya untuk training panjang** (puluhan menit sampai berjam-jam),
  di mana menunggu di depan memang tidak masuk akal.
- **Kalau runner sudah gagal SEKALI, jangan dipakai lagi untuk langkah itu.**
  Jalankan perintahnya langsung. Jangan berhenti untuk memperbaiki runner,
  memburu penyebabnya, atau menulis pembungkus baru — itu membakar waktu untuk
  perkakas, bukan untuk hasil.
- Alasannya konkret, 2026-08-15: eval sel 6 gagal tiga kali berturut-turut
  lewat runner (sekali karena citra korup, dua kali karena proses runner mati
  tanpa menulis satu baris pun log). Dijalankan langsung di depan, eval yang
  sama selesai dalam ~4 menit dan langsung mengeluarkan angkanya. Total waktu
  yang hilang gara-gara memakai runner untuk langkah pendek: lebih dari
  setengah jam, tanpa satu pun angka dihasilkan.
- Catatan teknis kalau tetap perlu proses latar: `setsid ... &` **tidak
  mem-fork** kalau pemanggilnya sudah process group leader, jadi prosesnya
  tidak benar-benar terlepas dan ikut mati bersama shell-nya. Pakai
  `setsid --fork`. Ini penyebab dua kematian runner di atas.
- Jangan pernah memakai `pkill -f <pola>` untuk membunuh proses: baris perintah
  shell pemanggil ikut memuat polanya, jadi perintah itu membunuh dirinya
  sendiri (dan pernah membunuh training sel 6 yang sah). Pakai PID langsung,
  atau pindai `/proc/<pid>/cmdline` dan pastikan `argv[0]` benar-benar
  interpreter yang menjalankan skrip itu.

### RF-DETR CPU-bound, YOLO/RT-DETR tidak — jangan asal upgrade GPU

Didiagnosis 2026-08-22 di RTX 4090 (`rfdetr_l_rgb_s42_i1280`, batch 4,
grad-accum 4, 1280px): **GPU-nya nganggur nunggu CPU, bukan CPU nunggu GPU.**
Buktinya, disampel langsung saat training aktif:

| | RF-DETR-L | YOLO26-L (dataset lebih besar, batch sama) |
|---|---|---|
| GPU util | 0-1%, sesekali nyembul ke 24-68% | 19-100%, rata-rata ~55% |
| Power draw | 60-100 W dari 450 W | 120-171 W |
| CPU proses utama | **1613%** (16 core) | **57%** (<1 core) |
| CPU per dataloader worker (×8) | ~1-8% | ~9-10%, rata |
| Kecepatan iterasi | ~0,5 it/s | ~1,7-1,9 it/s |

**Sebabnya:** augmentasi dan target-matching (Hungarian matching, khas
arsitektur DETR) RF-DETR jalan **single-threaded di proses utama** (GIL
Python), bukan tersebar ke dataloader worker seperti Ultralytics
(YOLO26-L/RT-DETR-L punya pipeline augmentasi C/numpy yang jauh lebih
paralel). Ini juga menjelaskan kenapa training RF-DETR di RTX 4090 dan di
RTX 2000 Ada (GPU jauh lebih lemah) sama-sama ~17-19 menit/epoch: kompute
ekstra 4090 tidak pernah terpakai karena GPU-nya memang menunggu CPU, bukan
sebaliknya. **Jangan simpulkan "GPU kurang kuat" dari epoch time RF-DETR
yang lambat — cek `nvidia-smi` dulu sebelum ganti GPU.**

Container ini juga dibatasi **~27,2 core CPU** (`cat /sys/fs/cgroup/cpu.max`),
meski `nproc` melaporkan 256 (angka host, bukan alokasi kontainer). Proses
utama RF-DETR sendirian sudah memakai >16 core dari kuota itu.

**Konsekuensi praktis untuk paralelisasi:**
- **YOLO26-L dan RT-DETR-L aman dipasangkan dengan RF-DETR** di GPU yang sama
  — RF-DETR menyisakan VRAM (~10,7 GB dari 24,6 GB di resep batch 4) dan tidak
  memperebutkan GPU compute (util-nya nyaris nol), jadi kompetisi nyata cuma
  di VRAM dan sebagian CPU. Sudah teruji langsung: RF-DETR + YOLO26-L
  berbarengan mencapai peak 23,5/24,6 GB, stabil, tanpa OOM.
- **YOLO26-L + RT-DETR-L tetap tidak boleh dipasangkan** (aturan lama di
  `run_new763_parallel.py`/`run_combined1716_matrix.py` masih berlaku) — dua
  model yang sama-sama GPU-bound memperebutkan compute yang sama, beda kasus
  dari RF-DETR yang justru idle.
- Kalau butuh RF-DETR lebih cepat (bukan sekadar iterasi paralel): coba
  turunkan `--workers` (worker ekstra tidak menolong kalau bottleneck-nya
  proses utama, bukan I/O) atau naikkan `--batch`/turunkan `--grad-accum`
  (batch lebih besar per step mengurangi rasio overhead-Python-per-gambar).
  **Belum diuji** — kalau dicoba, catat epoch time sebelum/sesudah di sini,
  bukan diasumsikan membantu.

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
