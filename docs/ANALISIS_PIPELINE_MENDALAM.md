# Analisis Mendalam Eksperimen Pipeline Pencacahan Tandan per Pohon

Dokumen ini menganalisis seluruh eksperimen yang dirujuk oleh
[`PROPOSAL-Pipeline.md`](../PROPOSAL-Pipeline.md) pada tiga jalur: **V1
(*baseline*/original)**, **V2 (*learned*/*re-ranked*)**, dan ***follow-up*
modalitas RGB+D4 pada `new763`**. Analisis disusun dari berkas log dan artefak
JSON di dalam repositori. Setiap angka kuantitatif disertai berkas sumbernya.
Tidak ada angka yang dihitung ulang atau diperkirakan di dokumen ini.

**Tanggal penyusunan:** 30 Agustus 2026.

---

## 0. Ruang lingkup penelusuran dan satu temuan administratif

### 0.1 Berkas yang dibaca

| Kategori | Berkas |
|---|---|
| Arsitektur kanonik | `PROPOSAL-Pipeline.md`, `HANDOFF.md` |
| Status agregat | `experiments/STATUS.md` |
| Log eksperimen | `experiments/EKSPERIMEN.md` (entri `V2-E-042` s.d. `V2-E-045`) |
| Jalur V1 | `results/remote_eval_2026-08-27/PIPELINE_EXPERIMENTS_V3.md`, `.../OPTIMIZED_PIPELINE.md` |
| Jalur V2 | `results/remote_eval_2026-08-28/GSP_LINKER.md`, `.../MAP_BOOST.md`, `.../PERFORMANCE_WAVE_2026-08-28.md`, `.../validation_wave/WAVE2_RECAP.md` |
| Artefak statistik | `results/remote_eval_2026-08-28/ci_artifacts/CI_SUMMARY.md`, `.../ci_artifacts/e2e_paired_test.json` |
| Modalitas | `docs/NEW763_RGBD4_RESULTS.md`, `results/new763_rgbd4/*.json` |
| Konteks pembanding | `PIPELINE_DAMIMAS.md`, `pipeline-pertandan/EKSPERIMEN.md` (entri `PT-E-026`) |

### 0.2 Temuan administratif: dua ID eksperimen yang dirujuk tidak ada

Permintaan analisis menyebut rentang `V2-E-042` sampai `V2-E-048`. Penelusuran
menyeluruh terhadap repositori memberi hasil berikut.

- `V2-E-042`, `V2-E-043`, `V2-E-044`, dan `V2-E-045` **ada** sebagai entri penuh
  di `experiments/EKSPERIMEN.md` (baris 2350, 2450, 2499, dan 2549). Entri
  terakhir pada berkas tersebut adalah `V2-E-045`.
- `V2-E-047` **hanya muncul sebagai judul bagian** pada `experiments/STATUS.md`
  baris 162 ("Komposisi lintas-layer dan head-aware ranking"). Tidak ada entri
  padanannya di `experiments/EKSPERIMEN.md`.
- `V2-E-046` dan `V2-E-048` **tidak ditemukan di berkas mana pun** dalam
  repositori.

Hal ini konsisten dengan pernyataan status pada kedua lembar bukti V2 sendiri.
`GSP_LINKER.md` dan `MAP_BOOST.md` menyatakan secara eksplisit bahwa keduanya
berada di *staging mirror* dan "belum mendapat ID eksperimen resmi
(`V2-E-###`/`PT-E-###`); penomoran dan integrasi ke `experiments/` serta
`docs/LAPORAN-AKHIR.md` menjadi keputusan pemelihara repo yang sebenarnya".

**Konsekuensi bagi analisis ini:** jalur V2 dianalisis dari lembar buktinya
langsung, bukan dari entri `EKSPERIMEN.md`, karena entri tersebut belum ditulis.
Penomoran `V2-E-046`/`V2-E-048` yang dipakai di percakapan bukan rujukan yang
dapat ditelusuri, dan sebaiknya tidak dipakai dalam naskah publikasi sebelum
entri resminya dibuat.

---

## 1. Kerangka evaluasi bersama

Ketiga jalur berbagi definisi metrik yang sama, tetapi **tidak berbagi partisi
data yang sama**. Pembacaan silang antarjalur hanya sah apabila partisi
disebutkan.

| Kumpulan uji | Ukuran | Dipakai oleh |
|---|---|---|
| `SawitMVC-Depth-YOLO` (test) | 110 pohon / 440 citra, seluruhnya empat sisi | V1 (`V2-E-042` s.d. `V2-E-045`), V2 (GSP, *map boost*) |
| `SawitMVC-YOLO` 953 (test) | 141 pohon / 588 citra; metrik pipeline memakai 135 pohon empat sisi | idem |
| `SawitMVC-Depth-YOLO` (validation) | 117 pohon | seleksi profil V1 dan V2 |
| `SawitMVC-YOLO` 953 (validation) | 91 pohon empat sisi | idem |
| `new763` RGBD4 | 536 pohon / 2.144 citra TRAIN; 117 pohon / 468 citra VALID; TEST tidak dimaterialkan | *follow-up* modalitas |

Sumber ukuran: `experiments/EKSPERIMEN.md` (`V2-E-042`, `V2-E-045`),
`results/remote_eval_2026-08-28/GSP_LINKER.md`, dan
`results/new763_rgbd4/new763_rgbd4_summary.json`.

Metrik yang dipakai berulang dan sering tertukar:

- **`AP50` agnostik** — kualitas lokalisasi tanpa label kelas. Bukan akurasi
  kematangan dan bukan akurasi pencacahan (`OPTIMIZED_PIPELINE.md` menegaskan
  hal ini secara eksplisit).
- **`mAP50` sadar-kelas** — deteksi empat kelas B1–B4 pada tingkat citra.
- **F1 fisik** — kualitas asosiasi lintas sisi: satu klaster prediksi
  dipasangkan dengan satu tandan acuan pada tingkat pohon.
- **MAE / akurasi ±1** — pencacahan tandan per pohon.
- **`matched_class_accuracy` / makro-F1 E2E** — klasifikasi kematangan pada
  klaster yang berhasil dipasangkan saja.

---

## 2. Jalur V1 — garis dasar pembanding/original

Jalur ini memakai WBF proposal *class-agnostic*, penaut Hungarian dan
*union-find* dengan prior rotasi, pengklasifikasi per tandan, serta lapisan
pencacahan Ridge dengan rekonsiliasi.

### 2.1 `V2-E-042` — verifikasi bobot remote dan garis dasar pipeline empat sisi

**Rancangan.** Enam bobot detektor (YOLO26l, RT-DETR-L, RF-DETR-L dari bank
`new763` dan `combined1716`) diuji ulang pada dua kumpulan uji lokal dengan
`imgsz = 1.280`, `pycocotools.COCOeval`, *confidence* inferensi `0,001`, NMS IoU
`0,7`, dan maksimum 300 deteksi per citra. Dua belas evaluasi model tunggal
dijalankan, lalu difusikan dengan WBF tiga detektor (IoU `0,60`, skor masukan
minimum `0,05`) dan ditautkan dengan prior rotasi bertanda yang dikalibrasi dari
data latih saja.

**Hasil deteksi model tunggal** (`experiments/EKSPERIMEN.md`, `V2-E-042`):

| Kumpulan uji | YOLO26l | RT-DETR-L | RF-DETR-L |
|---|---:|---:|---:|
| Depth `mAP50` (`combined1716`) | 0,5765 | 0,6309 | **0,6711** |
| 953 `mAP50` (`combined1716`) | 0,5403 | 0,5726 | **0,5890** |
| Depth `mAP50` (`new763`) | 0,5162 | 0,5580 | **0,6125** |
| 953 `mAP50` (`new763`) | 0,2331 | 0,1110 | 0,1776 |

**Hasil fusi WBF.** Bank `combined1716`: sadar-kelas `0,6691` (Depth) dan
`0,5861` (953); agnostik `0,8764` dan `0,8350`. Bank `new763`: sadar-kelas
`0,6062` dan `0,2018`; agnostik `0,8451` dan `0,4974`.

**Hasil pipeline empat sisi** (garis dasar, sebelum pengetatan):

| Bank | Uji | F1 fisik | MAE | Tepat | ±1 | Akurasi kelas | Makro-F1 E2E |
|---|---|---:|---:|---:|---:|---:|---:|
| `combined1716` | Depth | 0,6140 | 4,52 | 8,18% | 18,18% | 78,95% | 0,4726 |
| `combined1716` | 953 | 0,5327 | 14,99 | 0% | 0% | 69,94% | 0,3762 |
| `new763` | Depth | 0,6481 | 3,28 | 11,82% | 25,45% | 77,73% | 0,4762 |
| `new763` | 953 | 0,5460 | 6,56 | 2,22% | 8,15% | 33,78% | 0,1881 |

**Diagnosis inti.** *Recall* deteksi fisik tinggi (`0,8515`–`0,9344`), tetapi
presisi menurun tajam menjadi `0,3725`–`0,5231`: 910–3.366 klaster diprediksi untuk
559–1.342 tandan acuan. Pada konfigurasi `combined1716`/953, 3.366 klaster
diprediksi untuk 1.342 tandan, dan kelebihan klaster inilah yang menjelaskan MAE
`14,99` meskipun *recall* mencapai `0,9344`.

**Penilaian.** Eksperimen ini menetapkan masalah utama dengan benar: kegagalan
pipeline berasal dari **duplikasi klaster**, bukan dari kelemahan detektor. Ini
menjadi hipotesis kerja untuk tiga eksperimen berikutnya. Bank `combined1716`
ditetapkan sebagai kandidat detektor utama karena performa empat kelas paling
konsisten lintas domain; degradasi performa `new763` pada domain 953 (`mAP50` RT-DETR-L
`0,1110`) menegaskan bahwa cakupan domain data latih adalah faktor ketangguhan
yang dominan.

### 2.2 `V2-E-043` — pengetatan proposal dan penaut (*greedy*/*test-tuned*)

**Rancangan.** *Sweep* CPU pada *dump* WBF melalui `scripts/sweep_remote_pipeline.py`
yang mencakup ambang proposal, ambang tautan, ambang *singleton*, mode pasangan
sisi (`all` atau `adjacent`), dan ukuran klaster maksimum (2 atau 3). Parameter
dipilih **secara *greedy* langsung dari kumpulan uji** untuk mencari batas atas
rekayasa.

**Hasil** (`experiments/EKSPERIMEN.md`, `V2-E-043`;
`results/remote_eval_2026-08-27/OPTIMIZED_PIPELINE.md`):

| Uji | Versi | P | R | F1 fisik | Klaster prediksi/acuan | MAE | Tepat | ±1 | Makro-F1 E2E |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Depth | Garis dasar | 0,4705 | 0,8837 | 0,6140 | 1.050 / 559 | 4,518 | 8,18% | 18,18% | 0,4726 |
| Depth | *Optimized* | 0,8799 | 0,8390 | **0,8590** | 533 / 559 | **0,818** | 41,82% | 83,64% | 0,6419 |
| 953 | Garis dasar | 0,3725 | 0,9344 | 0,5327 | 3.366 / 1.342 | 14,993 | 0% | 0% | 0,3762 |
| 953 | *Optimized* | 0,8247 | 0,8346 | **0,8296** | 1.358 / 1.342 | **1,644** | 24,44% | 54,07% | 0,5469 |

Profil Depth: WBF IoU `0,60`, proposal `0,12`, tautan `0,05`, *singleton* `0,225`,
pasangan bersebelahan, maksimum dua anggota. Profil 953: WBF IoU `0,575`,
proposal `0,16`, tautan `0,05`, *singleton* `0,25`, semua pasangan sisi, maksimum
dua anggota.

**Penilaian.** Hipotesis duplikasi klaster **terkonfirmasi secara mekanistis**:
jumlah klaster 953 turun dari 3.366 menjadi 1.358 dan MAE turun `13,348`
tandan per pohon. Namun status statistiknya lemah menurut standar proyek sendiri.
Entri ini diberi putusan "CONFIRMED pada test yang dipakai untuk engineering
sweep. Ini bukan estimasi hold-out; threshold wajib dikunci ulang pada validation
set." Angka `0,8590`/`0,8296` karena itu adalah **batas atas rekayasa**, bukan
estimasi generalisasi.

### 2.3 `V2-E-044` — pengklasifikasi *crop* RGB lima *epoch*

**Rancangan.** Prapelatihan *tree-disjoint* dari SawitMVC-YOLO/953: 16.542 *crop*
dari 841 pohon, ConvNeXt-Tiny, kepala hibrida *softmax* + CORAL, *seed* 42.
Hasilnya diterapkan pada 14.643 proposal WBF uji 953, lalu diuji sebagai
pengganti penuh dan sebagai *blend* 10/25/50/75% terhadap *soft-vote* detektor.

**Hasil pengklasifikasi.** *Epoch* terbaik menurut makro-F1 validasi internal
adalah *epoch* 3: akurasi 62,17%, makro-F1 62,96%, akurasi ordinal ±1 99,32%, dan
MAE kelas 0,385.

**Hasil penerapan pada konfigurasi final 953:**

| Sumber probabilitas kelas | F1 fisik | MAE | ±1 | Akurasi kelas | Makro-F1 E2E |
|---|---:|---:|---:|---:|---:|
| WBF detektor 100% | 0,8296 | 1,644 | 53,33% | **70,71%** | 0,5410 |
| C2 pengklasifikasi 100% | 0,8299 | 1,637 | 54,07% | 62,95% | 0,5234 |
| WBF 75% + C2 25% | 0,8296 | 1,644 | 54,07% | 70,63% | **0,5469** |

**Penilaian.** Putusan resmi pada log adalah **FALSIFIED** — gugur secara empiris — untuk penggantian penuh:
akurasi kelas turun 7,76 poin persentase dan makro-F1 E2E turun `0,0176`.
Interpretasi yang wajar: pengklasifikasi *crop* yang dilatih lima *epoch* pada
kotak acuan tidak dapat menandingi probabilitas *soft-vote* tiga detektor yang
sudah beroperasi pada distribusi kotak yang sama dengan waktu inferensi. *Blend*
25% dipertahankan hanya sebagai kandidat rekayasa dengan margin sangat tipis
(makro-F1 naik `0,0059`, ±1 naik 0,74 poin persentase) pada kumpulan uji yang
sama yang dipakai untuk memilihnya — sehingga margin tersebut tidak dapat
dibedakan dari derau seleksi.

### 2.4 `V2-E-045` — lapisan *count-aware* terkunci validasi (jangkar test-locked V1)

**Rancangan.** Pertanyaan yang diuji adalah apakah kenaikan `V2-E-043` bertahan
ketika konfigurasi dikunci dari TRAIN/VAL, bukan dipilih dari uji. Prior rotasi
dan seluruh model pencacahan dilatih dari TRAIN saja. Kepala pencacahan berupa
Ridge terstandardisasi dengan fitur proposal per sisi; *alpha* dipilih melalui
validasi silang lima lipatan di TRAIN (`10` untuk Depth, `100` untuk 953).
Validasi dipakai untuk mengunci ambang proposal, ambang tautan, ambang
*singleton*, peringkat klaster, dan prior kelas. Uji hanya dipakai sebagai
konfirmasi.

**Profil terkunci:**

| Kumpulan | Proposal | Tautan | *Singleton* | Pasangan | Maks. | Peringkat | Prior kelas |
|---|---:|---:|---:|---|---:|---|---:|
| Depth | 0,075 | 0,25 | 0,15 | bersebelahan | 3 | *support* | −0,25 |
| 953 | 0,125 | 0,30 | 0,15 | bersebelahan | 3 | *max member* | −0,25 |

**Hasil validasi dan konfirmasi uji:**

| Kumpulan | Partisi | F1 fisik | MAE | Tepat | ±1 | Akurasi kelas | Makro-F1 E2E |
|---|---|---:|---:|---:|---:|---:|---:|
| Depth | val, 117 pohon | 82,57% | 0,726 | 44,44% | 84,62% | 83,55% | 0,6749 |
| Depth | uji, 110 pohon | 80,69% | 0,891 | 33,64% | 80,91% | 80,31% | 0,6047 |
| 953 | val, 91 pohon | 80,87% | 1,253 | 28,57% | 67,03% | 70,04% | 0,5462 |
| 953 | uji, 135 pohon | 80,43% | 1,393 | 25,93% | 61,48% | 71,11% | 0,5384 |

Sebagai pemeriksaan terhadap *overfitting* kepala pencacahan, MAE validasi silang
lima lipatan di TRAIN adalah `0,813` (Depth) dan `1,251` (953), sangat dekat
dengan MAE validasi `0,726` dan `1,253`.

**Empat cabang yang ditolak** pada eksperimen yang sama:

1. **WBF berbobot `[0,75; 1; 1,5]`** — ditolak. Meskipun sebagian `mAP` tingkat
   citra naik, F1 hilir validasi turun menjadi `0,7951` (Depth) dan `0,7736` (953).
2. **Penaut pasangan logistik TRAIN-only** — ditolak. F1 validasi `0,7680` (Depth)
   dan `0,7374` (953), di bawah prior rotasi manual.
3. ***Blend* jumlah prediksi dengan jumlah klaster mentah** — ditolak. Profil final
   memakai *blend* `0` (Ridge murni).
4. **WBF IoU `0,50`–`0,70`** — IoU `0,60` dipertahankan.

**Penilaian.** Inilah kontribusi metodologis paling penting pada jalur V1. Jarak
antara `V2-E-043` (*greedy*: F1 `0,8590`/`0,8296`) dan `V2-E-045` (terkunci
validasi: `0,8069`/`0,8043`) sebesar 5,2 dan 2,5 poin persentase mengukur besar
bias optimisme akibat pemilihan parameter dari kumpulan uji. Angka `V2-E-045`
menjadi jangkar test-locked resmi jalur V1 dan menjadi garis dasar pembanding
bagi seluruh eksperimen V2.

Kaveat yang dinyatakan sendiri oleh entri tersebut penting: "Test lokal tetap
pernah dibaca dalam eksperimen historis, jadi konfirmasi ini tidak disebut
*hold-out* publikasi yang sepenuhnya pristine."

### 2.5 `PIPELINE_EXPERIMENTS_V3` — penelusuran parameter keluarga eksperimen V1

Dokumen `results/remote_eval_2026-08-27/PIPELINE_EXPERIMENTS_V3.md` mencatat
penelusuran parameter (*sweep*) sepuluh keluarga eksperimen terhadap dua target rekayasa: klasifikasi
empat kelas ≥ 75% dan lokalisasi agnostik ≈ 90%.

Keluarga yang **ditolak seluruhnya**: fotometrik (*hue*/MLP warna, CLAHE,
*sharpening*, *gamma*, kecerahan/kontras); koreksi warna (*gray-world*,
*white-balance* ringan — E2E validasi terbaik hanya 71,62% dan 71,88%); TTA
(fotometrik, konteks 1,25/2,0, *flip*, rotasi); *fine-tuning* detektor lokal dan
YOLO resolusi 1.600 (validasi lebih rendah atau terlalu mahal); *ensemble*
kepala; serta *reranking* kualitas klaster.

Hasil uji terbaik yang dinilai sah: akurasi kelas pada *match* **71,67%** (953,
profil standar) dan **80,31%** (Depth); F1 fisik **80,43%**; MAE **1,393**;
lokalisasi agnostik **87,64%** (Depth) dan **83,50%** (953).

**Diagnosis penting dari dokumen ini:** konfigurasi validasi dengan akurasi kelas
74,06% (`class_conf`) **turun menjadi 71,81% pada uji**, dengan F1 fisik ikut
turun ke 78,57%. Dokumen menyebut ini "indikasi *overfitting* pada profil
validasi, bukan bukti kemampuan general". Kesalahan kelas dinyatakan bersifat
ordinal: B2 sering tertukar dengan B3, dan B3 dengan B4.

Kesimpulan yang dinyatakan: target 75% klasifikasi dan 90% lokalisasi **belum
tercapai**, dan eksperimen filter/TTA tambahan sebaiknya dihentikan.

---

## 3. Jalur V2 — *learned*/*re-ranked*

Jalur V2 mengganti dua komponen: penaut Hungarian + *union-find* diganti *Global
Set-Partition* (GSP) berbasis MILP, dan proposal WBF diberi lapisan
*re-ranker* `p_tp` terlatih.

### 3.1 GSP *linker* — partisi global per pohon

**Motivasi teknis.** Inspeksi kode menemukan bahwa *constraint* "maksimal satu
proposal per sisi fisik dalam satu klaster" pada kelas `UF` di
`sweep_remote_pipeline.py` bersifat ***vacuous***: `self.sides[i]` diinisialisasi
dengan indeks larik proposal (`i`), bukan `dets[i]["side"]`, sehingga uji irisan
sisi tidak pernah aktif. Pembanding pada `eval_remote_pipeline_postprocess.UnionFind`
mengimplementasikannya dengan benar, sehingga cacat ini spesifik pada modul yang
justru dipakai seluruh jalur evaluasi terkunci.

**Metode.** Model log-*odds* tepi (`ExtraTreesClassifier`, fitur 65 dimensi)
menghasilkan probabilitas $p$ untuk setiap pasangan proposal lintas sisi
bersebelahan. Kandidat klaster dienumerasi sebagai *connected subset* pada graf
berambang ($p \geq p_{floor} = 0,02$) dengan enumerasi kanonis, dan kandidat yang
memuat dua anggota dari sisi fisik yang sama **ditolak secara struktural**.
Partisi optimal per pohon diselesaikan melalui `scipy.optimize.milp`.

**Gerbang jangkar.** Empat profil Hungarian+UF yang sudah terkunci direproduksi
lebih dahulu dan dibandingkan dengan angka acuan (toleransi `±0,003`); selisih
aktual ≈ 10⁻⁵. Gerbang **lulus** untuk kedua kumpulan data.

**Hasil validasi.**

Pada Depth (117 pohon), profil GSP terbaik-menurut-fisik (`extra`,
$\tau_{prob} = 0,10$, *singleton* `0,20`, `max_size` 3, peringkat *support*)
**mendominasi jangkar Hungarian C pada kelima metrik secara bersamaan**:

| Metrik Depth VAL | Hungarian (Jangkar C) | GSP | Δ |
|---|---:|---:|---:|
| F1 fisik | 0,8471 | **0,8526** | +0,0055 |
| MAE | 0,9487 | **0,9316** | −0,0171 |
| ±1 | 0,7692 | **0,7863** | +0,0171 |
| Akurasi kelas | 0,8359 | **0,8457** | +0,0098 |
| Makro-F1 | 0,6700 | **0,6807** | +0,0107 |

Pada 953 (91 pohon), **tidak ada satu pun profil GSP yang unggul menyeluruh**.
Kandidat GSP terbaik-menurut-kelas unggul tipis pada akurasi kelas
(`0,7555` vs `0,7542`; Δ = +0,0013) tetapi mengalami penurunan performa tajam
pada pencacahan: MAE memburuk dari `1,2527` menjadi `1,7473` (Δ = +0,4946) dan
akurasi ±1 memburuk dari `0,6703` menjadi `0,5055` (Δ = −0,1648).

**Keputusan.** Depth dikunci memakai GSP; 953 **tetap memakai profil Hungarian
*incumbent*** (Jangkar A: tautan `0,15`, *singleton* `0,15`, `max_size` 4,
peringkat *score*).

**Hasil TEST-LOCKED** (`results/remote_eval_2026-08-28/GSP_LINKER.md`):

| Kumpulan | Profil | F1 fisik | MAE | Tepat | ±1 | Akurasi kelas | Makro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 953 (135 pohon) | Hungarian Jangkar A | 0,8387 | 1,3630 | 0,2741 | 0,6370 | 0,7442 (832/1.118) | 0,6034 |
| Depth (110 pohon) | GSP | 0,8534 | 0,7727 | 0,4455 | 0,8545 | 0,8162 (373/457) | 0,6519 |

Selang kepercayaan 95% *bootstrap* (2.000 resampel pohon, `RandomState(42)`) —
953: F1 [0,8174; 0,8587]; MAE [1,1630; 1,5852]; ±1 [0,5556; 0,7185]; akurasi
kelas [0,7112; 0,7735]; makro-F1 [0,5655; 0,6382]. Depth: F1 [0,8301; 0,8761];
MAE [0,6091; 0,9455]; ±1 [0,7818; 0,9182]; akurasi kelas [0,7765; 0,8556];
makro-F1 [0,6046; 0,6918].

*Solver* pada profil Depth terkunci: `milp` = 109, `empty` = 1,
`greedy_fallback` = 0 dari 110 pohon.

**Catatan kritis yang mudah terlewat.** Peningkatan pada 953 (F1 `0,8043` →
`0,8387`) **bukan hasil GSP**, melainkan hasil profil Hungarian yang
*dikunci ulang* pada sesi tersebut (Jangkar A) yang berbeda dari profil
`V2-E-045` (proposal `0,125`, tautan `0,30`, `max_size` 3, peringkat
*max member*). Menyebut kenaikan 953 sebagai "kemenangan V2" akan salah
atribusi: yang menang pada 953 adalah profil *baseline* yang disetel ulang,
sedangkan metode V2 justru ditolak di sana.

### 3.2 *Map boost* — proposal *deep-tail* dan *re-ranker* `p_tp`

**Metode tiga lapis.** (1) Fusi WBF *deep-tail* dengan `iou_threshold = 0,60` pada
tiga ambang skor minimum {`0,05`; `0,02`; `0,01`}; (2) `HistGradientBoostingClassifier`
(`max_iter` 300, `learning_rate` 0,05, `max_leaf_nodes` 31, `l2_regularization`
10,0, `random_state` 42) dilatih **hanya pada TRAIN** dengan 30 fitur (953) atau
37 fitur (Depth); (3) skor gabungan
$\text{skor}_{\text{WBF}}^{a} \times p_{tp}^{b} \times p_c^{\gamma}$ dengan
$(a, b, \gamma)$ dipilih dari VAL.

**AUC *re-ranker* VAL:** 953 `0,9544`–`0,9867`; Depth `0,9706`–`0,9900`, naik
seiring *floor* yang makin longgar.

**Hasil TEST-LOCKED** (`MAP_BOOST.md` dan `ci_artifacts/CI_SUMMARY.md`, 500
resampel berpasangan tingkat citra):

| Kumpulan | Jenis | Terkunci | Garis dasar | Δ | CI95 Δ | Signifikan |
|---|---|---:|---:|---:|---|---|
| 953 | agnostik `AP50` | 0,8419 | 0,8350 | +0,0070 | [+0,0027; +0,0135] | **ya** |
| 953 | sadar-kelas `mAP50` | 0,5970 | 0,5861 | +0,0108 | [+0,0030; +0,0194] | **ya** |
| Depth | agnostik `AP50` | 0,8783 | 0,8764 | +0,0018 | [−0,0016; +0,0045] | tidak |
| Depth | sadar-kelas `mAP50` | 0,6552 | 0,6691 | **−0,0139** | [−0,0284; +0,0016] | tidak; arah regresi |

**Penilaian.** Dua hal patut diapresiasi secara metodologis. Pertama, keunggulan
VAL pada profil sadar-kelas Depth (`0,6623` vs `0,6595`, yaitu +0,0028)
**berbalik arah pada uji** menjadi −0,0139, dan tim memutuskan mempertahankan
profil apa adanya serta mencatatnya sebagai temuan generalisasi negatif alih-alih
memilih ulang dari uji. Kedua, besar efek yang benar-benar didukung secara
statistik sangat kecil: +0,0070 dan +0,0108 `mAP` pada 953. Ini bukan lompatan
performa, melainkan perbaikan marginal yang kebetulan dapat dideteksi karena
*bootstrap* berpasangan tingkat citra memiliki daya yang relatif tinggi
(588 citra).

Batasan yang dinyatakan sendiri: skor $p_{tp}$ **tidak dikalibrasi**; AUC tinggi
mengukur diskriminasi peringkat, bukan kalibrasi probabilitas, sehingga tidak sah
ditafsirkan sebagai "kotak ini 92% kemungkinan benar". Data *depth* mentah juga
belum direproyeksi ke bidang RGB pada jalur ini; pengubahan ukuran
*nearest-neighbor* meleset median 29 piksel, diterapkan identik pada TRAIN/VAL/TEST
sehingga tidak menimbulkan kebocoran, tetapi nilai kedalaman absolut per kotak
harus dibaca sebagai perkiraan kasar.

### 3.3 Signifikansi *end-to-end* pada uji terkunci

Artefak `ci_artifacts/e2e_paired_test.json` memuat *bootstrap* berpasangan 5.000
resampel pohon atas ringkasan per-pohon yang sudah terkunci (tanpa inferensi
ulang dan tanpa seleksi):

| Kumpulan | Metrik | Garis dasar | Terkunci | Perbaikan | CI95 | Mencakup nol? |
|---|---|---:|---:|---:|---|---|
| 953 | F1 fisik | 0,804348 | 0,838710 | +0,034362 | [+0,020939; +0,047690] | tidak |
| 953 | MAE | 1,392593 | 1,362963 | +0,029630 | [−0,014815; +0,081481] | **ya** |
| 953 | tepat | 0,259259 | 0,274074 | +0,014815 | [−0,014815; +0,044444] | **ya** |
| 953 | ±1 | 0,614815 | 0,637037 | +0,022222 | [0,000000; +0,051852] | **ya** (batas) |
| Depth | F1 fisik | 0,806859 | 0,853408 | +0,046549 | [+0,025733; +0,069025] | tidak |
| Depth | MAE | 0,890909 | 0,772727 | +0,118182 | [−0,036364; +0,263636] | **ya** |
| Depth | tepat | 0,336364 | 0,445455 | +0,109091 | [+0,008864; +0,218182] | tidak |
| Depth | ±1 | 0,809091 | 0,854545 | +0,045455 | [−0,018182; +0,109091] | **ya** |

Akurasi kelas dan makro-F1 **tidak memiliki CI berpasangan** karena garis dasar
lama hanya menyimpan hasil kelas agregat, bukan hitungan benar per pohon. Kedua
metrik itu dilaporkan sebagai estimasi titik saja — keputusan yang tepat, dan
dinyatakan eksplisit di `PERFORMANCE_WAVE_2026-08-28.md`.

**Pembacaan yang benar:** dari delapan perbandingan E2E, hanya **tiga** yang
selang kepercayaannya tidak mencakup nol — F1 fisik pada kedua kumpulan dan
akurasi pencacahan tepat pada Depth. Seluruh metrik MAE dan ±1 tetap belum konklusif.
Dengan kata lain, bukti yang kokoh adalah **perbaikan asosiasi/deteksi fisik**,
bukan perbaikan pencacahan.

### 3.4 *Validation wave* dan *Wave 2* — 2.893 baris tanpa kandidat unggul menyeluruh

Kedua gelombang ini dijalankan sepenuhnya pada TRAIN/VAL dan **tidak membuka
TEST**.

**Kandidat yang dipromosikan (satu-satunya).** *Stack* opini DINOv2-Large
(`0,15`) dan logistik anggota (`0,05`) dengan bias logit B2 `+0,15` menaikkan
akurasi kelas validasi 953 dari `0,7542` menjadi `0,7684` dan makro-F1 dari
`0,6014` menjadi `0,6164`. *Bootstrap* berpasangan 5.000 pohon memberi selang
delta [+0,0026; +0,0268] untuk akurasi kelas dan [+0,0007; +0,0303] untuk
makro-F1; keduanya tidak mencakup nol. Metrik fisik dan pencacahan invarian
karena hanya keputusan kelas yang diganti.

**Kandidat VAL Depth terbaik (belum signifikan).** Komposisi lintas-lapis
(topologi GSP original + target pencacahan V2 geo Ridge + kalibrasi kelas
`scale_macro`) memperbaiki kelima metrik sekaligus:

| Metrik Depth VAL | Garis dasar | Kandidat | Δ |
|---|---:|---:|---:|
| F1 fisik | 0,852641 | 0,854225 | +0,001583 |
| MAE | 0,931624 | 0,914530 | −0,017094 |
| ±1 | 0,786325 | 0,786325 | 0 |
| Akurasi kelas | 0,845652 | 0,850000 | +0,004348 |
| Makro-F1 | 0,680685 | 0,689013 | +0,008328 |

Namun *bootstrap* berpasangan 5.000 pohon atas 117 pohon VAL menghasilkan seluruh
CI delta yang mencakup nol: F1 [−0,006160; +0,009540]; MAE [−0,085470; +0,051282];
akurasi kelas [−0,011744; +0,021558]; makro-F1 [−0,015425; +0,033034]. Kandidat
ini disimpan sebagai *validation candidate*, bukan klaim signifikansi.

**Cabang yang ditolak** (seluruhnya didokumentasikan sebagai ablasi):

| Cabang | Alasan penolakan |
|---|---|
| OOF *stacking* tingkat pohon | Akurasi kelas `0,754204`, makro `0,607873`; tidak lebih baik dari *stack* terpilih |
| Agregasi berbasis sisi, kepala ordinal, KNN/prototipe DINOv2-Large, atensi GPU | Tidak memperbaiki profil terpilih |
| Selektor adaptif Hungarian↔GSP | Model batas atas teoretis (*oracle*) pada TRAIN (F1 `0,892258`, MAE `0,936951`) hanya diagnostik; kebijakan terlatih gagal |
| Regresor pencacahan kaya fitur / nonlinear | Galat CV TRAIN turun, tetapi kompromi performa (*trade-off*) pada VAL memburuk |
| *Count meta-ensemble* 953 | Akurasi kelas `0,756443` tetapi MAE `1,472527`, melewati pagar `1,35` |
| *Composition-aware retraining* | Akurasi kelas tetap `0,850000`, makro-F1 turun ke `0,684983` dari `0,689013` |
| *Head-aware truncation* | Akurasi kelas naik ke `0,7718` (953) dan `0,8603` (Depth), tetapi F1 fisik turun `0,0064` dan `0,0037` |
| Kerangka utama (*backbone*) independen: ConvNeXt-Small, Swin-Tiny, EfficientNetV2-S | Fusi nominal `0,7697`/`0,6166` hanya menambah satu pohon benar atas jangkar `0,7684`/`0,6164`, tanpa CI independen |
| Selektor kompromi V2-only Depth | MAE turun `0,9316` → `0,7607` dan akurasi kelas naik `0,8457` → `0,8495`, tetapi F1 fisik turun ke `0,8341` dan makro-F1 ke `0,6667` |

**Penilaian.** Ini adalah bagian paling kuat dari seluruh berkas: 2.893 baris
konfigurasi dievaluasi, dan keputusannya adalah "tidak ada konfigurasi baru yang
memenuhi seluruh pagar 953 atau Depth sekaligus lebih baik daripada jangkar yang
relevan". Hasil negatif disimpan sebagai kontrol, bukan dibuang. Pola yang
berulang dan konsisten: **setiap cabang yang menaikkan akurasi kelas disertai
penurunan MAE atau F1 fisik**. Ini menandakan pipeline sudah berada pada permukaan
kompromi performa (*trade-off*), bukan pada wilayah yang masih dapat diperbaiki bebas biaya.

---

## 4. *Follow-up* modalitas — RGB+D4 pada `new763`

### 4.1 *Early fusion* empat kanal

**Desain kontrol.** Pembanding RGB dan RGB+D4 memakai 468 citra VALID yang sama,
label yang sama, ukuran 1.280, *seed* 42, dan evaluator `pycocotools.COCOeval`.
Unit *split* diwariskan dari `SawitMVC-Depth-YOLO v2.0.0`; direktori TEST **tidak
dibaca, tidak dimaterialkan, dan evaluator tidak memiliki opsi TEST**
(`results/new763_rgbd4/new763_rgbd4_summary.json`: `"test_policy": "test
directory was intentionally not read or materialized"`).

*Depth* mentah Y16 848×480 mm diproyeksikan ke grid warna 1.280×800 memakai
*calibration sidecar* per citra, intrinsik kedua kamera, ekstrinsik, distorsi
Brown–Conrady, dan *z-buffer*, dengan batas fisik tetap 0,3–20 m. Perlu
diperhatikan: fraksi *depth* valid hanya `0,2859` (rerata TRAIN) dan `0,2882`
(rerata VALID) dari grid warna.

**Hasil** (`docs/NEW763_RGBD4_RESULTS.md`):

| Arsitektur | RGB `mAP50` | RGB+D4 `mAP50` | Δ | CI95 Δ berpasangan | Kesimpulan |
|---|---:|---:|---:|---|---|
| YOLO26l | 0,529357 | 0,529523 | +0,000166 | [−0,024195; +0,028892] | Tidak ada peningkatan |
| RF-DETR-L v2 | 0,608233 | 0,597070 | −0,011163 | [−0,037049; +0,018074] | *Point* turun; belum signifikan |
| RT-DETR-L | 0,577766 | 0,584088 | +0,006322 | [−0,026770; +0,039670] | *Point* naik; belum signifikan |

Fraksi resampel dengan Δ positif: YOLO26l `0,558`; RF-DETR-L v2 `0,232`;
RT-DETR-L `0,650`. Seluruh selang kepercayaan mencakup nol.

Per kelas, pola gainnya saling meniadakan: pada YOLO26l depth membantu B3
(+0,026175) tetapi merugikan B1 (−0,018841), B2 (−0,002292), dan B4 (−0,004377).

### 4.2 *Late fusion* dengan resep tetap

Karena *early fusion* tidak konsisten, prediksi RGB dan RGB+D4 yang sudah dibekukan
diuji sebagai dua sumber pelengkap dengan **satu resep tetap tanpa penelusuran parameter (*sweep*)**:
*union class-aware* NMS pada IoU `0,60`, dengan *union*-WBF *mean-score* IoU `0,60`
sebagai kontrol.

| Arsitektur | RGB | RGB+D4 | *Union*-NMS | *Union*-WBF |
|---|---:|---:|---:|---:|
| YOLO26l | 0,529357 | 0,529523 | 0,561982 | **0,567718** |
| RF-DETR-L v2 | **0,608233** | 0,597070 | 0,606856 | 0,528041 |
| RT-DETR-L | 0,577766 | 0,584088 | **0,606368** | 0,407071 |

NMS sumber tunggal sebagai kontrol hanya menghasilkan `0,550878` (YOLO),
`0,610903` (RF), dan `0,579054` (RT), sehingga kenaikan *union* bukan sekadar
efek penekanan duplikasi dalam satu model.

*Paired bootstrap* tingkat citra: YOLO *union*-WBF Δ `+0,037912`, CI95
[+0,016060; +0,059120] (500 resampel, seluruh resampel positif); RT *union*-NMS
Δ `+0,028492`, CI95 [+0,009231; +0,047236] (200 resampel sebagai penyaringan awal (*screening*) cepat,
seluruh resampel positif).

**Penilaian.** Kedua hasil signifikan pada VAL, tetapi karena resep ditemukan
melalui penyaringan awal (*screening*) pada VAL, keduanya berstatus eksploratif dan
terpilih dari validasi (*validation-selected*), bukan klaim generalisasi. Yang paling menarik secara ilmiah adalah **asimetri arsitektur
yang tajam**: WBF *class-aware* naif menaikkan YOLO (+0,038) tetapi merusak RF
(−0,080) dan RT (−0,171). Dokumen menyimpulkan dengan benar bahwa WBF tidak boleh
dipakai sebagai modul umum tanpa validasi terpisah.

### 4.3 Audit yang memperkuat kredibilitas

`results/new763_rgbd4/rfdetr_l_rgbd4_failed_run_audit.json` mencatat *run*
RF-DETR-L v1 yang **dikeluarkan dari perbandingan**: `patch_projection` dibangun
tiga kanal, lalu adaptor mengganti modul menjadi empat kanal *setelah* grup
parameter *optimizer* dibuat, sehingga parameter kanal *depth* tidak pernah masuk
*optimizer*. Buktinya adalah `checkpoint_depth_weight_norm = 0.0` berbanding
`checkpoint_rgb_weight_norm = 4.430938`. *Run* tersebut sempat mencatat validasi
`mAP50 = 0,595464`, angka yang tampak wajar dan mudah dilaporkan tanpa audit.
Menemukan dan mengeluarkannya adalah praktik yang benar.

Anomali RT-DETR juga dicatat: kolom *validation loss* menjadi `NaN` pada sejumlah
*epoch* akhir, meskipun metrik COCO dan seluruh tensor *checkpoint* tetap
terhingga.

---

## 5. Analisis kritis

### 5.1 Apa yang benar-benar terbukti

Hanya empat klaim yang didukung selang kepercayaan yang tidak mencakup nol pada
data uji:

1. **Perbaikan F1 deteksi fisik** pada kedua kumpulan uji: 953 +0,034362
   [+0,020939; +0,047690] dan Depth +0,046549 [+0,025733; +0,069025]
   (`e2e_paired_test.json`).
2. **Akurasi pencacahan tepat pada Depth**: +0,109091 [+0,008864; +0,218182]
   (sumber yang sama).
3. **Perbaikan `mAP` pada 953** dari lapisan *re-ranker*: agnostik +0,0070
   [+0,0027; +0,0135] dan sadar-kelas +0,0108 [+0,0030; +0,0194]
   (`CI_SUMMARY.md`).
4. **Superioritas RF-DETR-L sebagai model tunggal** dan **peran dominan cakupan
   domain data latih**, terlihat dari degradasi performa bank `new763` pada domain 953
   (`mAP50` RT-DETR-L `0,1110` vs `0,5726` untuk `combined1716`; `V2-E-042`).

Di luar keempat hal ini, seluruh klaim lain berstatus estimasi titik,
terpilih dari validasi (*validation-selected*), atau belum konklusif.

### 5.2 Apa yang ditolak dan mengapa

Penolakan pada berkas ini berkualitas tinggi karena hampir selalu disertai alasan
kuantitatif, bukan sekadar "tidak membantu". Tiga pola penolakan yang berulang:

**Pola A — keunggulan validasi yang berbalik arah pada uji.** `class_conf` pada
V1 (74,06% VAL → 71,81% TEST) dan profil sadar-kelas Depth pada *map boost*
(+0,0028 VAL → −0,0139 TEST). Keduanya menunjukkan bahwa kumpulan validasi
proyek ini terlalu kecil untuk membedakan profil yang berdekatan.

**Pola B — pertukaran metrik yang tidak dapat dihindari.** Setiap cabang yang
menaikkan akurasi kelas menurunkan F1 fisik atau MAE: *head-aware truncation*,
selektor kompromi V2-only Depth, GSP pada 953, *count meta-ensemble*, dan profil
*class-priority* V1. Konsistensi pola ini lintas sembilan cabang independen
adalah bukti struktural bahwa pipeline sudah berada di permukaan Pareto pada
kapasitas data saat ini.

**Pola C — kapasitas model tambahan yang tidak berpindah menjadi performa.**
DINOv2-Large, ConvNeXt-Small, Swin-Tiny, EfficientNetV2-S, atensi GPU, dan
regresor pencacahan nonlinear seluruhnya gagal mengungguli *stack* sederhana
yang sudah ada. Pada `PT-E-026` di subproyek `pipeline-pertandan`, pola serupa
terjadi pada pencacahan: makro-MAE VAL `0,8110` tidak berpindah ke uji, yang
justru memburuk menjadi `1,0374` terhadap jangkar `1,0039`.

### 5.3 Konsistensi lintas eksperimen

**Konsisten.** (a) Urutan arsitektur RF-DETR-L > RT-DETR-L > YOLO26l bertahan
pada `V2-E-042` dan pada *baseline* RGB `new763`. (b) Duplikasi klaster sebagai
penyebab utama galat pencacahan terkonfirmasi berulang dari `V2-E-042` sampai
`V2-E-045`. (c) Kesalahan kelas bersifat ordinal (B2↔B3, B3↔B4) muncul di
`PIPELINE_EXPERIMENTS_V3` dan tetap terlihat pada matriks konfusi uji terkunci
GSP — pada 953, 101 prediksi B3 berpasangan dengan acuan B2 dan 79 prediksi B3
berpasangan dengan acuan B4. (d) B4 secara konsisten menjadi kelas terlemah:
F1 `0,5114` (953) dan `0,4176` (Depth) pada uji terkunci; `AP50` B4 `0,3693`
pada *map boost* Depth; `AP50` B4 `0,240195` pada *baseline* RGB YOLO26l `new763`.

**Tidak konsisten.** (a) Manfaat *depth*: `V2-E-024` mencatat *depth* menaikkan
lokalisasi murni (`AP50` 0,7636 vs 0,7358, $P(\Delta > 0) = 92,1\%$), tetapi
`V2-E-027`/`V2-E-029` mencatat penurunan signifikan −0,0476 [−0,0671; −0,0274]
untuk *depth* monokular pada 953, dan `new763` RGB+D4 tidak menemukan efek
signifikan pada ketiga arsitektur. Kesimpulan yang konsisten dengan seluruh bukti
adalah: **manfaat *depth* bergantung pada jenis *depth* (sensor vs monokular),
tugasnya (lokalisasi vs klasifikasi), dan cara penggabungannya (*early* vs
*late*)** — bukan properti umum modalitas. (b) Arah efek WBF *class-aware*
berbalik antararsitektur pada `new763`.

### 5.4 Ketidaksesuaian dokumentasi yang perlu diperbaiki

Empat hal berikut bukan kesalahan eksperimen, melainkan risiko salah baca.

**(1) Status "V2 belum menggantikan hasil test-locked" sudah tidak akurat.**
`PROPOSAL-Pipeline.md` menyatakan "V2 terbaru belum menggantikan hasil
test-locked" dan `experiments/STATUS.md` §0 menyatakan V2 "sudah diimplementasikan
dan diaudit pada TRAIN/VAL; belum menggantikan hasil test-locked". Kenyataannya,
`GSP_LINKER.md` dan `MAP_BOOST.md` mendokumentasikan **pembukaan TEST pada
28 Agustus 2026 dengan profil terkunci**, dan `WAVE2_RECAP.md` sendiri
menampilkan baris test-locked yang sudah diperbarui (953 F1 `0,804348` →
`0,838710`; Depth F1 `0,806859` → `0,853408`). Pernyataan status tersebut benar
untuk kandidat *class head* dan *Wave 2*, tetapi salah untuk topologi GSP Depth
dan lapisan `mAP` 953. `STATUS.md` §5–§11 belum memuat baris test-locked terbaru
ini sama sekali.

**(2) Kenaikan 953 salah atribusi apabila disebut "GSP".** Seperti diuraikan pada
§3.1, GSP **ditolak** untuk 953; kenaikan F1 di sana berasal dari profil Hungarian
Jangkar A yang dikunci ulang.

**(3) Klaim "pembukaan TEST tepat satu kali" berlaku per sesi, bukan per proyek.**
`GSP_LINKER.md` menyatakan "Data test dibuka tepat satu kali per dataset pada
sesi ini". Namun kedua kumpulan uji yang sama sudah dibaca pada `V2-E-042`
(garis dasar), `V2-E-043` (*sweep greedy* langsung pada uji), `V2-E-044` (*blend*
pada uji), dan `V2-E-045` (konfirmasi). Akumulasi paparan uji karena itu jauh
lebih besar daripada satu kali. `V2-E-045` sendiri mengakui hal ini. Pengaman (*guard*)
teknis `SystemExit` hanya memeriksa keberadaan berkas keluaran pada `--output-root`
yang sama, dan dapat dilewati sepenuhnya dengan `--output-root` berbeda — hal yang
juga dinyatakan sendiri oleh kedua lembar bukti.

**(4) Jangkar Ridge `75,79%` / `1,0039` pada `PROPOSAL-Pipeline.md` §5 berasal
dari korpus yang berbeda.** Kedua angka itu bersumber dari `PT-E-026` pada
subproyek `pipeline-pertandan` (`PIPELINE_DAMIMAS.md`), yang memakai korpus
`SawitMVC-YOLO-Damimas` dengan partisi 641 pohon latih / 86 validasi / 127 uji
dan mengeksklusikan 99 pohon varietas LONSUM. Angka tersebut **tidak sebanding
langsung** dengan MAE `1,363` (953) atau `0,773` (Depth) pada jalur V1/V2 karena
korpus, partisi, dan definisi metriknya berbeda (macro-MAE per kelas berbanding
MAE total per pohon). Rujukan korpus perlu ditulis eksplisit di proposal.

### 5.5 Risiko teknis yang masih terbuka

- **Cacat *side constraint* pada `sweep_remote_pipeline.UF` belum diperbaiki.**
  Cacat ini tidak aktif (*dormant*) untuk `pair_mode="adjacent"` dan `max_size ≤ 4` karena
  menutup siklus yang mengulang satu sisi memerlukan minimal lima anggota. Namun
  modul ini dipakai oleh seluruh jalur evaluasi terkunci, dan profil 953 pada
  `V2-E-043` memakai `pair_mode` "semua pasangan sisi". Perbaikan satu baris
  (`{dets[i]["side"]}` menggantikan `{i}`) beserta uji regresi perlu dijadwalkan.
- **Skor $p_{tp}$ tidak terkalibrasi**, sehingga tidak dapat dipakai langsung
  sebagai `link_confidence` yang ditampilkan ke pengguna sebagaimana diminta
  `PROPOSAL-Pipeline.md` §6.
- **Data *depth* pada jalur *map boost* belum direproyeksi** (meleset median
  29 piksel), berbeda dari jalur `new763` RGBD4 yang sudah memakai reproyeksi
  terkalibrasi penuh.

---

## 6. Keterbatasan validitas

**Daya statistik.** Kumpulan uji berisi 135 dan 110 pohon; kumpulan validasi
91 dan 117 pohon. Pada ukuran ini, selang kepercayaan MAE Depth membentang
[0,6091; 0,9455] — lebar `0,336` tandan per pohon, yaitu sekitar 43% dari nilai
*estimasi titiknya. Temuan `V2-E-023` bahwa "split test 352 tidak punya daya
statistik untuk membedakan konfigurasi" berlaku juga di sini. Konsekuensinya,
seluruh perbandingan profil yang selisihnya di bawah ≈0,02 pada metrik apa pun
tidak dapat dibedakan dari derau.

**Paparan berulang terhadap kumpulan uji.** Sebagaimana diuraikan pada §5.4 poin
(3), kedua kumpulan uji sudah dibaca berkali-kali. Tidak ada satu pun angka pada
jalur V1 atau V2 yang memenuhi definisi *hold-out* pristine. Klaim publikasi
memerlukan pohon baru yang belum pernah disentuh.

**Kebocoran partisi historis.** `experiments/STATUS.md` §4 mencatat dua pembatas
yang masih berlaku: (a) 87% citra pada `test_penuh` `agn953_full` beririsan dengan
data prapelatihan, sehingga nilai generalisasi yang sah adalah `test_bersih`
(19 pohon / 316 kotak, `AP50 = 0,7702`); (b) 44 dari 55 pohon uji dataset 352
termuat dalam partisi latih dataset 953. Audit irisan `tree_id` antara partisi
latih `combined1716` dan kedua kumpulan uji lokal **belum selesai** menurut
`V2-E-042` batasan poin 1 — ini merupakan butir audit terpenting yang belum
tuntas sebelum publikasi.

**Pergeseran domain temporal.** Dataset 953 direkam Mei 2026 dan Depth 352 pada
Juli 2026, berjarak ≈80 hari. Proporsi kelas B3 pada pohon yang sama menyusut dari
55,3% menjadi 14,0% (`V2-E-022`). Perbandingan deteksi empat kelas lintas-dataset
tidak valid secara metodologis.

**Cakupan sensor *depth* terbatas.** Fraksi *depth* valid hanya ≈0,286–0,288 dari
grid warna pada `new763`. Kesimpulan "*depth* belum layak menggantikan RGB" berlaku
untuk kondisi cakupan ini, bukan untuk modalitas *depth* secara umum.

**Metrik `counting` yang tidak seragam.** Pada `V2-E-042` s.d. `V2-E-044`,
"counting" berarti jumlah klaster tertaut mentah. Pada `V2-E-045` dan seterusnya,
angka tersebut berasal dari lapisan Ridge *count-aware* yang dilatih dari TRAIN.
Keduanya tidak sama dengan Ridge `F_all` + rekonsiliasi yang dispesifikasikan
untuk *deployment* pada `PROPOSAL-Pipeline.md` §5, yang **belum pernah dijalankan**
pada *dump* remote.

**Metrik yang tidak memiliki CI.** Akurasi kelas dan makro-F1 E2E dilaporkan
sebagai estimasi titik pada uji terkunci karena garis dasar lama tidak menyimpan
dekomposisi per pohon. Kedua metrik ini justru yang paling dekat dengan target
rekayasa 75%, sehingga ketiadaan CI-nya membatasi klaim.

---

## 7. Sintesis: status pipeline saat ini

### 7.1 Status per jalur

| Jalur | Status bukti | Angka rujukan uji terkunci |
|---|---|---|
| **V1** | Referensi yang dapat dibandingkan langsung; profil terkunci validasi pada `V2-E-045` | 953: F1 `0,8043`, MAE `1,3926`, ±1 `61,48%`, akurasi kelas `71,11%`. Depth: F1 `0,8069`, MAE `0,8909`, ±1 `80,91%`, akurasi kelas `80,31%` |
| **V2** | Sebagian sudah terkunci pada uji dan didukung CI untuk F1 fisik; sisanya validasi saja | 953: F1 `0,8387`, MAE `1,3630`, ±1 `63,70%`, akurasi kelas `74,42%`, makro-F1 `0,6034` (profil Hungarian dikunci ulang). Depth: F1 `0,8534`, MAE `0,7727`, ±1 `85,45%`, akurasi kelas `81,62%`, makro-F1 `0,6519` (GSP) |
| **RGB+D4 `new763`** | Validasi saja; TEST tidak pernah dimaterialkan | VALID `mAP50`: RGB `0,6082` (RF), *union*-WBF YOLO `0,5677`, *union*-NMS RT `0,6064` |

### 7.2 Penilaian menyeluruh

**Yang sudah dicapai.** Sistem ini layak disebut pipeline proposal/lokalisasi
multi-model dengan asosiasi empat sisi yang berfungsi. Lokalisasi
*class-agnostic* mencapai `0,8783` (Depth) dan `0,8419` (953) pada uji terkunci
dengan lapisan *re-ranker*. Asosiasi fisik mencapai F1 `0,85` (Depth) dan `0,84`
(953), dengan perbaikan atas garis dasar yang didukung selang kepercayaan pada
kedua kumpulan. Pencacahan Depth berada pada MAE `0,77` dengan akurasi ±1
`85,45%`.

**Yang belum dicapai.** Kedua target rekayasa pada `HANDOFF.md` — 75% klasifikasi
empat kelas dan 90% lokalisasi agnostik — belum terpenuhi secara bersamaan pada
kedua domain. Klasifikasi 953 berada pada `74,42%` (mendekati target, tetapi tanpa
CI) dan lokalisasi 953 pada `84,19%`. Modul *quality gate*, rekomendasi
pengambilan ulang, *confidence*/UI, dan *deployment* belum ada implementasinya —
`PROPOSAL-Pipeline.md` sendiri menegaskan bahwa modul-modul ini "belum dianggap
selesai hanya karena proposal arsitekturnya sudah terdokumentasi". Ridge `F_all`
+ rekonsiliasi yang menjadi spesifikasi *deployment* belum pernah dijalankan pada
*dump* remote.

**Penilaian atas kualitas metodologi.** Disiplin eksperimen pada berkas ini di
atas rata-rata: gerbang jangkar dengan toleransi eksplisit sebelum setiap grid,
*guard* teknis terhadap pembukaan uji berulang, pelaporan regresi apa adanya
(profil sadar-kelas Depth −0,0139), penolakan untuk memfabrikasi CI ketika data
per pohon tidak tersedia, audit *run* yang gagal secara diam-diam (RF-DETR v1
dengan `depth_weight_norm = 0.0`), dan penyimpanan hasil negatif sebagai kontrol.
Kelemahan utamanya bukan pada pelaksanaan eksperimen, melainkan pada **sinkronisasi
dokumentasi**: `STATUS.md` dan `PROPOSAL-Pipeline.md` tertinggal dari lembar bukti
V2, dan dua lembar bukti terpenting belum memiliki ID eksperimen resmi.

---

## 8. Rekomendasi berdasarkan bukti

Diurutkan menurut rasio nilai terhadap biaya.

**Prioritas 1 — Menuntaskan butir audit dan sinkronisasi dokumentasi (biaya rendah,
menghalangi publikasi).**

1. Selesaikan **audit irisan `tree_id`** antara partisi latih `combined1716` dan
   kedua kumpulan uji lokal, sesuai batasan yang dinyatakan sendiri pada
   `V2-E-042`. Tanpa ini, seluruh angka uji jalur V1 dan V2 memiliki risiko
   kontaminasi yang belum terkuantifikasi.
2. Terbitkan entri `EKSPERIMEN.md` resmi untuk GSP *linker*, *map boost*, dan
   gelombang validasi, lalu perbarui `STATUS.md` §5–§11 dan tabel status pada
   `PROPOSAL-Pipeline.md` agar mencerminkan hasil test-locked 28 Agustus 2026.
   Sertakan koreksi atribusi 953 (Hungarian, bukan GSP).
3. Tambahkan keterangan korpus pada jangkar Ridge `75,79%`/`1,0039` di
   `PROPOSAL-Pipeline.md` §5 (korpus DAMIMAS, partisi 641/86/127).
4. Perbaiki cacat *side constraint* `sweep_remote_pipeline.UF` beserta uji regresi
   yang membuktikan hasil terkunci tidak berubah pada konfigurasi
   `adjacent`/`max_size ≤ 4`.

**Prioritas 2 — Menambah daya statistik (biaya sedang, satu-satunya jalan menuju
klaim publikasi).**

5. Kumpulkan **pohon empat sisi baru** sebagai *hold-out* eksternal yang belum
   pernah disentuh, lalu jalankan profil V2 terkunci **satu kali**. Ini adalah
   satu-satunya cara memenuhi syarat yang dinyatakan `V2-E-045` dan
   `PROPOSAL-Pipeline.md`. Semua bukti pada §5.2 menunjukkan bahwa penambahan
   model atau lapisan baru tidak lagi produktif tanpa data baru; kumpulan validasi
   91–117 pohon sudah terbukti tidak mampu membedakan profil yang berdekatan.
6. Simpan **dekomposisi per pohon untuk akurasi kelas dan makro-F1** pada setiap
   evaluasi mendatang, agar CI berpasangan untuk kedua metrik ini dapat dihitung.
   Ketiadaannya saat ini menghalangi klaim atas metrik yang paling dekat dengan
   target 75%.

**Prioritas 3 — Arah teknis yang masih menjanjikan (berbasis bukti yang ada).**

7. **Kualitas label B2/B4.** Bukti konvergen dari tiga sumber independen —
   kekeliruan ordinal B2↔B3 dan B3↔B4 pada matriks konfusi uji terkunci, `AP50`
   B4 terendah pada setiap evaluasi, dan kegagalan seluruh cabang fotometrik —
   menunjukkan bahwa batas saat ini kemungkinan besar berada pada definisi dan
   konsistensi label, bukan pada kapasitas model. Audit label yang dikunci pada
   partisi lintas lokasi/kamera adalah investasi yang paling mungkin memindahkan
   plafon.
8. **Lapisan rekonsiliasi jumlah berbasis statistik klaster GSP.** Diusulkan
   sendiri pada `GSP_LINKER.md` batasan poin 1: mempelajari target hitung dari
   statistik klaster GSP (jumlah klaster berskor positif per ukuran, distribusi
   $\operatorname{logit}(p)$ terpilih per pohon) alih-alih hanya dari fitur
   proposal WBF mentah. Ini menyasar satu-satunya kelemahan GSP yang teridentifikasi
   pada 953, yaitu fragmentasi yang merusak pencacahan.
9. **Jalankan Ridge `F_all` + rekonsiliasi pada *dump* remote.** Spesifikasi
   *deployment* pada `PROPOSAL-Pipeline.md` §5 belum pernah diuji pada jalur ini;
   status "jumlah klaster mentah" atau "Ridge fitur proposal" bukan yang
   dispesifikasikan.
10. **Kalibrasi $p_{tp}$** (*reliability diagram* atau *Brier score*) apabila
    `link_confidence` akan ditampilkan ke pengguna sesuai `PROPOSAL-Pipeline.md` §6.

**Yang sebaiknya dihentikan.** Berdasarkan 2.893 baris evaluasi pada gelombang
validasi dan sepuluh keluarga eksperimen pada `PIPELINE_EXPERIMENTS_V3`, hal
berikut tidak lagi produktif tanpa data atau label baru: penambahan kerangka utama (*backbone*)
yang lebih besar, TTA, koreksi fotometrik/warna, *reranker* kualitas klaster, dan
regresor pencacahan nonlinear. Seluruhnya sudah diuji dan ditolak dengan alasan
kuantitatif.

**Mengenai jalur RGB+D4.** Bukti saat ini tidak mendukung promosi *early fusion*
empat kanal. Yang layak dilanjutkan adalah *late fusion* — tetapi **per arsitektur**,
tidak sebagai modul umum, mengingat WBF *class-aware* naif merusak RF (−0,080) dan
RT (−0,171) sementara menaikkan YOLO (+0,038). Verifikasi memerlukan evaluasi
*held-out* baru karena resep saat ini dipilih melalui penyaringan awal (*screening*) pada VAL.

---

## Lampiran — Peta rujukan angka utama

| Angka | Nilai | Berkas sumber |
|---|---|---|
| F1 fisik uji terkunci 953 | 0,838710 | `results/remote_eval_2026-08-28/GSP_LINKER.md`; `ci_artifacts/e2e_paired_test.json` |
| F1 fisik uji terkunci Depth | 0,853408 | idem |
| CI Δ F1 fisik 953 | [+0,020939; +0,047690] | `ci_artifacts/e2e_paired_test.json` |
| CI Δ F1 fisik Depth | [+0,025733; +0,069025] | idem |
| MAE uji terkunci 953 / Depth | 1,3630 / 0,7727 | `GSP_LINKER.md` |
| Akurasi kelas uji terkunci 953 / Depth | 0,7442 / 0,8162 | idem |
| `AP50` agnostik uji terkunci 953 / Depth | 0,8419 / 0,8783 | `MAP_BOOST.md`; `ci_artifacts/CI_SUMMARY.md` |
| Regresi sadar-kelas Depth | −0,0139 [−0,0284; +0,0016] | idem |
| Jangkar V1 uji 953 / Depth (F1) | 0,8043 / 0,8069 | `experiments/EKSPERIMEN.md` `V2-E-045` |
| Batas atas *greedy* `V2-E-043` | 0,8590 / 0,8296 | `experiments/EKSPERIMEN.md`; `OPTIMIZED_PIPELINE.md` |
| Kandidat kelas VAL 953 | 0,7684 / makro 0,6164 | `PERFORMANCE_WAVE_2026-08-28.md`; `STATUS.md` §8 |
| Kandidat lintas-lapis VAL Depth | 0,854225 / 0,914530 / 0,850000 / 0,689013 | `WAVE2_RECAP.md`; `STATUS.md` §9 |
| `new763` RGB vs RGB+D4 | 0,529357 / 0,529523 / 0,608233 / 0,597070 / 0,577766 / 0,584088 | `docs/NEW763_RGBD4_RESULTS.md`; `results/new763_rgbd4/new763_rgbd4_summary.json` |
| `new763` *union*-WBF YOLO Δ | +0,037912 [+0,016060; +0,059120] | `docs/NEW763_RGBD4_RESULTS.md` |
| `new763` *union*-NMS RT Δ | +0,028492 [+0,009231; +0,047236] | idem |
| Fraksi *depth* valid `new763` | 0,2859 (TRAIN) / 0,2882 (VALID) | `results/new763_rgbd4/new763_rgbd4_summary.json` |
| Jangkar Ridge DAMIMAS | 1,0039 / 75,79% | `PIPELINE_DAMIMAS.md`; `pipeline-pertandan/EKSPERIMEN.md` `PT-E-026` |
