# Log Eksperimen — Audit Forensik Data dan Pipeline (6 September 2026)

Berkas ini bersifat *append-only* dan memakai penomoran `AF-E-###` agar tidak
bertabrakan dengan rangkaian `V2-E-###` maupun `PT-E-###` yang sudah ada. Tidak
ada satu pun entri lama yang diubah oleh audit ini.

**Lingkungan eksekusi.** NVIDIA RTX 3090 (24 GB), 64 vCPU, 503 GB RAM,
`torch 2.14.0+cu130`, `ultralytics 8.4.142`, Python 3.12. Seluruh angka pada
berkas ini dihitung ulang dari data mentah; tidak ada angka yang dikutip dari
laporan lama kecuali disebut secara eksplisit sebagai pembanding.

**Korpus.** `ULM-DS-Lab/SawitMVC-YOLO` (953 pohon, Mei 2026) dan
`ULM-DS-Lab/SawitMVC-Depth-YOLO` v2.0.0 (763 pohon = 352 pohon Juli 2026 +
411 pohon Agustus 2026). Kedua korpus diunduh utuh; 26.838 dan 38.928 berkas.

**Artefak.** Metrik pada `results/audit_forensik_2026-09-06/`, log eksekusi pada
`logs_ringkas/audit_forensik_2026-09-06/`, skrip pada `scripts/audit_forensik/`.
Bobot model, dump prediksi, dan citra terpotong berada di bucket
`ULM-DS-Lab/project-expertise-backup` pada awalan `audit_forensik_2026-09-06/`.

---

## AF-E-001 — Perbandingan tingkat pohon antar-kampanye pada 352 pohon identik

**Rancangan.** Membaca `bunches` tingkat pohon pada kedua rilis untuk 352 pohon
fisik yang sama, lalu membandingkan jumlah tandan unik dan komposisi kelasnya.
Tidak melibatkan model apa pun.
Skrip: `scripts/audit_forensik/an1_overlap.py`, `an3_framing.py`, `an4_bunches.py`.

**Temuan empiris terukur.**

| Besaran | Mei (953) | Juli (Depth) | Perubahan |
|---|---:|---:|---:|
| Tandan unik per pohon | 9,89 | 3,99 | −60% |
| Kotak mentah per citra | 4,63 | 1,63 | −65% |
| B1 (matang, siap panen) | 275 | 457 | **+66%** |
| B2 (peralihan) | 584 | 620 | +6% |
| B3 (mentah) | 1.883 | 212 | **−89%** |
| B4 (sangat mentah) | 739 | 111 | **−85%** |
| Sisi tanpa satu kotak pun | 1,1% | 14,2% | — |

Laju sisi kosong Juli memburuk secara berurutan menurut nomor sisi:
`4% → 13% → 19% → 20%` untuk sisi 1 sampai 4, sedangkan pada Mei keempat sisi
seragam pada `1%`. Kampanye Agustus (411 pohon, blok MARIHAT/TOPAZ) mencatat
5,76 tandan unik per pohon.

**Keputusan metodologis.** Dua arah perubahan bertentangan dengan fenologi
sekaligus: panen berulang seharusnya menurunkan stok B1, sedangkan inisiasi
tandan yang berlangsung terus seharusnya mempertahankan stok B4. Penjelasan
"pergeseran temporal 80 hari" pada `V2-E-022` karena itu **tidak memadai
sebagai penjelasan tunggal**; terdapat komponen perbedaan protokol pengamatan
atau kelengkapan anotasi yang belum terkuantifikasi.

**Batasan validitas.** Analisis ini menunjukkan ketidakcocokan, bukan
menetapkan penyebabnya. Penentuan penyebab memerlukan verifikasi manual
berstrata oleh anotator, yang belum dilakukan.

---

## AF-E-002 — Konsistensi label lintas-tampak untuk tandan fisik yang sama

**Rancangan.** Untuk setiap tandan multi-tampak, membandingkan `class_id` pada
seluruh tampak tempat ia muncul. Skrip: `scripts/audit_forensik/an5_labelnoise.py`.

**Temuan empiris terukur.** Korpus 953: 7.328 tandan multi-tampak,
**0 (0,00%)** memiliki tampak yang berselisih kelas. Juli: 2 dari 841 (0,24%).
Agustus: 0 dari 1.184.

**Keputusan metodologis.** Angka nol tersebut bukan indikator mutu anotasi,
melainkan bukti bahwa satu kelas ditetapkan per tandan fisik lalu disalin ke
seluruh tampak oleh perkakas anotasi. Konsekuensinya, **label kematangan adalah
properti tandan fisik, bukan properti tampak**. Pengklasifikasi per-tampak
dipaksa memprediksi atribut yang tidak selalu teramati pada masukannya, dan
`mAP50` sadar-kelas per citra karena itu menghukum model atas informasi yang
definisi labelnya sendiri tidak sediakan.

---

## AF-E-003 — Struktur posisi tandan dalam pohon sebagai prediktor kematangan

**Rancangan.** Menghitung peringkat vertikal dan peringkat ukuran setiap tandan
di dalam pohonnya, lalu melatih pengklasifikasi tanpa satu piksel pun.
Partisi tingkat pohon kanonik. Skrip: `an6_structure.py`, `an7_monotone.py`.

**Temuan empiris terukur.**

- Korelasi Spearman antara indeks kelas dan peringkat vertikal dalam pohon:
  **−0,616**; terhadap peringkat ukuran: **+0,431**.
- Pengklasifikasi geometri murni (8 fitur, tanpa piksel), test 953:
  akurasi `0,5713`, makro-F1 `0,4729`, akurasi ordinal ±1 `0,9429`
  (garis dasar kelas mayoritas `0,5300`).
- Penataan monoton dengan komposisi kelas pohon diketahui (*oracle*),
  diurutkan menurut posisi vertikal: akurasi `0,6912`, makro-F1 **`0,6237`**.
  Dengan komposisi dari prior global saja: akurasi `0,5207`.

**Keputusan metodologis.** Makro-F1 `0,6237` tanpa piksel melampaui makro-F1
*end-to-end* `0,6034` yang dicapai seluruh tumpukan visual pada 953. Sinyal
struktur layak dimasukkan ke tahap klasifikasi; pengujiannya ada pada `AF-E-009`.

**Batasan validitas.** Angka `0,6912` memakai komposisi kelas acuan, sehingga
merupakan batas atas, bukan performa yang dapat dicapai saat inferensi.

---

## AF-E-004 — Plafon lapisan pencacahan dengan deteksi *oracle*

**Rancangan.** Mengganti seluruh tahap deteksi dengan kotak acuan, lalu
menjalankan Ridge dari cacah kotak per kelas menuju cacah tandan unik per kelas.
Skrip: `scripts/audit_forensik/an8_counting.py`.

**Temuan empiris terukur.** Test 953, 136 pohon.

| Besaran | MAE | Tepat | ±1 | Tampak tunggal |
|---|---:|---:|---:|---:|
| B1 | 0,101 | 0,899 | **1,000** | 11,6% |
| B2 | 0,239 | 0,768 | **0,993** | 23,4% |
| B3 | 0,638 | 0,428 | **0,942** | 22,7% |
| B4 | 0,268 | 0,739 | **0,993** | **40,5%** |
| Total per pohon | 1,058 | **0,290** | **0,754** | — |

Makro-MAE `0,312`, berdekatan dengan jalur *oracle* historis `0,275`–`0,277`
pada `docs/REKAP.md` §2, sehingga pengukuran ini terkalibrasi terhadap angka
proyek sendiri. Faktor duplikasi per pohon `k = 1,905` dengan simpangan baku
`0,384`; *estimator* "kotak dibagi k" hanya mencapai `0,304` tepat persis
walaupun kotaknya sempurna.

**Keputusan metodologis.** Target pencacahan **total** yang tepat persis tidak
dapat dicapai. Target pencacahan **per kelas dengan toleransi ±1** dapat
dicapai. Laju tampak-tunggal yang bergantung kelas (B4 `40,5%` berbanding B1
`11,6%`) berarti aturan "konfirmasi minimal dua tampak" menghapus B4 secara
sistematis; ambang *singleton* global tidak sesuai untuk tugas ini.

---

## AF-E-005 — Plafon `mAP50` dengan lokalisasi sempurna

**Rancangan.** Membentuk 18.540 citra terpotong tandan (cincin konteks `1,6×`,
sesuai `PROPOSAL-Pipeline.md` §4), melatih ConvNeXt-Tiny 10 *epoch*, lalu
menyusun prediksi deteksi yang kotaknya **identik dengan kotak acuan** sehingga
lokalisasi bernilai sempurna dan hanya kelas yang diprediksi.
Skrip: `exp_crops.py`, `exp_train.py`, `exp_ceiling.py`, `exp_sensitivity.py`.

**Temuan empiris terukur.**

- Akurasi validasi per-*crop* `0,6635` — berada di dalam pita `0,62`–`0,70`
  yang dicapai berulang oleh ConvNeXt, Swin, EfficientNetV2, dan DINOv2 pada
  repositori ini.
- Plafon `mAP50` empat kelas dengan kotak sempurna: **`0,6569`**
  (B1 `0,7653`, B2 `0,4912`, B3 `0,7625`, B4 `0,6088`).
  Uji kewarasan dengan kelas *oracle* menghasilkan `1,0000`.
- Taksonomi dua kelas, pengklasifikasi yang sama: B1 lawan sisanya
  **`0,8766`**; B1+B2 lawan B3+B4 **`0,8891`**.
- Kurva sensitivitas (galat disalurkan ke kelas bertetangga): akurasi
  kematangan `0,661 → 0,587`; `0,80 → 0,735`; **`0,90 → 0,847`**; `0,95 → 0,927`.

**Keputusan metodologis.** Hasil test terkunci proyek `0,5970` berada pada
`91%` dari plafon `0,6569`. Seluruh ruang perbaikan yang tersisa untuk semua
detektor digabung adalah sekitar `6` poin `mAP`. Target `0,85` pada empat kelas
menuntut akurasi kematangan `≈0,90`, dua puluh poin di atas segala yang pernah
dicapai; target tersebut **dinyatakan tidak terjangkau pada taksonomi empat
kelas**, dan terjangkau pada taksonomi dua kelas.

---

## AF-E-006 — Perbandingan taksonomi dengan detektor nyata

**Rancangan.** Tiga pelatihan YOLO26s yang identik kecuali taksonomi labelnya,
`imgsz 960`, 30 *epoch*, `patience 8`, `batch 16`, `workers 32`, `cache ram`,
*seed* 42, partisi kanonik 953. Skrip: `build_ds.py`, `run_exp.py`.

**Temuan empiris terukur.** Test 953 (588 citra).

| Taksonomi | `mAP50` | Presisi | Daya tangkap |
|---|---:|---:|---:|
| Empat kelas B1–B4 | 0,5433 | 0,5087 | 0,5856 |
| Dua kelas (siap panen / belum) | **0,7754** | 0,7662 | 0,7021 |
| Satu kelas (agnostik) | **0,8057** | 0,7965 | 0,7087 |

**Kalibrasi terhadap angka proyek.** YOLO26**l** pada tugas empat kelas
tercatat `0,5435` (`experiments/STATUS.md` §1). Replikasi ini dengan YOLO26**s**
pada resolusi lebih rendah menghasilkan `0,5433`. Selisihnya `0,0002`, sehingga
perbandingan taksonomi di atas sah dibaca sebagai perbandingan setara.

**Keputusan metodologis.** Perubahan definisi target menaikkan `mAP50` sebesar
`+0,2321` tanpa mengubah model, resolusi, maupun data.

---

## AF-E-007 — Matriks generalisasi lintas-kampanye dan dekomposisi galatnya

**Rancangan.** Dua detektor *class-agnostic* dilatih terpisah — satu pada Mei
(953), satu pada 763 (Juli+Agustus) — lalu disilangkan. Evaluasi kampanye pada
model 763 memakai partisi uji miliknya sendiri untuk mencegah kontaminasi.
Skrip: `run_exp.py`, `e1b_fp.py`, `e1c_fpkind.py`, `e1d_merge.py`.

**Temuan empiris terukur — `mAP50` lokalisasi agnostik.**

| Dilatih pada | Mei 953 | Juli 352 | Agustus 411 |
|---|---:|---:|---:|
| Mei (953) | **0,8057** (P 0,797 · R 0,709) | 0,4720 (P **0,581** · R 0,470) | 0,5955 (P 0,663 · R 0,539) |
| 763 (Juli+Agustus) | 0,6243 (P 0,707 · R **0,609**) | **0,7691** (P 0,782 · R 0,724) | **0,8522** (P 0,852 · R 0,818) |

Pada taksonomi empat kelas, model 763 mencapai `mAP50 = 0,1898` di test Mei —
mereproduksi rentang bencana `0,1776`–`0,2018` yang dilaporkan `V2-E-042`.

**Dekomposisi positif palsu** (model Mei, `conf ≥ 0,50`):

| Kumpulan | TP | FP | *nested* | *shifted* | *orphan* | sisi kotak model / acuan |
|---|---:|---:|---:|---:|---:|---:|
| Mei 953 test | 1.123 | 79 | 74,7% | 5,1% | 20,3% | 1,003 |
| Juli 352 | 657 | 193 | **95,9%** | 0,5% | 3,6% | 0,869 |
| Agustus 411 | 1.135 | 195 | **96,4%** | 0,5% | 3,1% | 0,922 |

*nested* berarti titik pusat deteksi berada **di dalam** sebuah kotak acuan
tetapi `IoU < 0,5`. Melonggarkan kriteria menjadi `IoU ≥ 0,3` menaikkan presisi
Juli dari `0,612` menjadi `0,867` — melampaui presisi Mei sendiri (`0,839`) —
dan daya tangkapnya dari `0,454` menjadi `0,644`.

**Hipotesis penggabungan kotak diuji dan ditolak.** Bila Juli membingkai satu
gerombol tandan sebagai satu kotak, sebuah kotak acuan akan memuat beberapa
deteksi yakin. Terukur: `0,2%` pada Mei, `0,2%` pada Juli, `0,0%` pada Agustus.

**Keputusan metodologis.** Terdapat **dua perbedaan protokol yang terpisah**:
(a) konvensi kotak — Juli/Agustus membingkai objek tunggal yang sama secara
lebih longgar, dan ini menjelaskan sebagian besar keruntuhan `AP` lintas-korpus;
(b) kelengkapan anotasi tingkat pohon (`AF-E-001`), yang tidak dapat dijelaskan
oleh konvensi kotak maupun penggabungan. Keduanya membuat pelatihan gabungan
`combined1716` dan evaluasi lintas-korpus **tidak mengukur mutu model**.

**Koreksi terhadap hipotesis awal audit ini.** Dugaan awal bahwa positif palsu
Juli sebagian besar adalah tandan nyata yang tidak dilabeli **tidak didukung**:
hanya `3,6%` yang benar-benar berada di lokasi tanpa kotak acuan. Dugaan itu
dicatat di sini sebagai hipotesis yang gugur.

---

## AF-E-008 — Pencacahan tandan siap panen ujung ke ujung

**Rancangan.** Detektor dua kelas `AF-E-006` dijalankan pada seluruh partisi
953, lalu fitur bergaya `F_all` (cacah deteksi per kelas pada sepuluh ambang
keyakinan, ditambah statistik geometri dan jumlah sisi) dimasukkan ke
`RidgeCV`. Target diambil dari `bunches` acuan. Skrip: `run_e345.py` bagian E3.

**Temuan empiris terukur.** Test 953, 141 pohon.

| Besaran | Ridge di-*fit* pada | MAE | Tepat | ±1 |
|---|---|---:|---:|---:|
| Tandan siap panen | TRAIN | 0,369 | 0,674 | **0,957** |
| Tandan siap panen | VAL | 0,418 | 0,624 | **0,965** |
| Total tandan | TRAIN | 1,383 | 0,227 | 0,610 |
| Total tandan | VAL | 1,518 | 0,213 | 0,553 |

**Kalibrasi terhadap angka proyek.** Total tandan ±1 pada replikasi ini
`0,610`; angka test terkunci `V2-E-045` adalah `0,6148`. Kesesuaian tersebut
membuat angka `0,957`–`0,965` sah dibaca sebagai perbandingan setara.

**Keputusan metodologis.** Target konsistensi `≥95%` **tercapai** apabila
besaran yang dilaporkan adalah cacah tandan siap panen per pohon dengan
toleransi ±1, dan **tidak tercapai** untuk cacah total.

---

## AF-E-009 — Fusi penampilan dan struktur pada deteksi nyata

**Rancangan.** Model struktur (`HistGradientBoosting`, 8 fitur geometri)
dilatih pada deteksi partisi TRAIN; pengklasifikasi *crop* dijalankan pada
deteksi yang berpasangan dengan kotak acuan (`IoU ≥ 0,5`); satu skalar bobot
fusi `w` ditala pada VAL; TEST dibuka satu kali.
Skrip: `e4b_fuse.py`. Deteksi berasal dari detektor agnostik `AF-E-006`.

**Temuan empiris terukur.** 2.466 deteksi berpasangan pada test 953, `w = 0,8`.

| Konfigurasi | Akurasi | Makro-F1 | ±1 |
|---|---:|---:|---:|
| Penampilan saja | 0,6951 | 0,6470 | 0,9943 |
| Struktur saja | 0,5669 | 0,3973 | 0,9152 |
| **Penampilan + struktur** | **0,6963** | **0,6528** | 0,9935 |

Pada kotak acuan (bukan deteksi nyata), fusi yang sama memberi akurasi
`0,6745 → 0,6809` dan makro-F1 `0,6368 → 0,6588`.

**Kontrol negatif yang dilaporkan apa adanya.** Percobaan pertama melatih model
struktur pada VAL dan menala `w` pada VAL yang sama. Protokol bocor itu memilih
`w = 1,3` dan menghasilkan **regresi** pada test: akurasi `0,6468`, makro-F1
`0,5934`. Hasilnya disimpan pada `results/audit_forensik_2026-09-06/e345.json`
bagian `E4` sebagai kontrol, dan tidak dipakai sebagai klaim.

**Keputusan metodologis.** Manfaat struktur pada deteksi nyata **nyata tetapi
kecil** (`+0,0058` makro-F1), jauh di bawah manfaatnya pada kotak acuan
(`+0,0220`). Penyebab yang paling mungkin: peringkat dalam-pohon dihitung dari
himpunan deteksi yang memuat positif palsu dan kehilangan objek, sehingga
urutannya lebih berderau. Komponen ini layak dipertahankan karena biayanya nol,
tetapi **tidak boleh dipasarkan sebagai perbaikan besar**.

**Batasan validitas.** Belum ada selang kepercayaan berpasangan untuk selisih
`+0,0058`; dengan 2.466 deteksi, selisih sekecil itu belum tentu dapat
dibedakan dari derau seleksi.

---

## AF-E-010 — Verifikasi cacat kendala sisi pada `sweep_remote_pipeline.UF`

**Rancangan.** Kedua varian `UF` — versi repositori dan versi yang diperbaiki —
diberi daftar tepi yang sama dari proposal nyata detektor agnostik pada test
953, lalu jumlah klaster yang memuat dua deteksi atau lebih dari sisi fisik yang
sama dihitung. Skrip: `run_e345.py` bagian E5.

**Temuan empiris terukur.**

| Varian | Klaster | Memuat ≥2 deteksi dari sisi yang sama |
|---|---:|---:|
| Versi repositori (`self.sides = [{i} …]`) | 1.191 | **540 (45,3%)** |
| Versi diperbaiki (`{dets[i]["side"]}`) | 1.249 | **0 (0,0%)** |

**Keputusan metodologis.** Cacat yang dicatat pada
`docs/ANALISIS_PIPELINE_MENDALAM.md` §5.5 masih ada dan **tidak dorman**:
`45,3%` klaster melanggar kendala "maksimal satu proposal per sisi fisik".
Karena modul ini yang memilih ambang proposal, ambang tautan, ambang
*singleton*, dan ukuran klaster maksimum, seluruh profil parameter terpilih
terhadap klasterer yang mengizinkan penghitungan ganda dalam satu tampak, lalu
diterapkan pada evaluator yang melarangnya. Perbaikannya satu baris, tetapi
seluruh penelusuran parameter wajib dijalankan ulang sesudahnya.

**Batasan validitas.** Daftar tepi pada uji ini memakai skor geometri sederhana,
bukan penaut terlatih milik proyek. Angka `45,3%` karena itu mengukur besarnya
celah kendala, bukan besarnya perubahan metrik akhir. Besaran yang terakhir
hanya dapat diketahui setelah *sweep* dijalankan ulang dengan kode yang benar.

---

## AF-E-011 — Detektor agnostik berkapasitas lebih tinggi dan kepala ordinal CORN

**Rancangan.** Dua komponen dilatih untuk menguji rekomendasi `AF-E-005` dan
`AF-E-002` secara ujung ke ujung, bukan sebagai plafon.

1. Detektor **class-agnostic** YOLO26m pada `imgsz 1280`, 40 *epoch*,
   `patience 10`, `batch 12`, `workers 32`, `cache ram`, *seed* 42; berhenti
   dini pada *epoch* 36. Skrip: `scripts/audit_forensik/panen_det.py`.
2. Kepala **ordinal CORN** (ConvNeXt-Small, tiga logit terkondisi untuk K=4)
   pada 18.540 citra terpotong, 18 *epoch*, masukan 176 piksel. Keluarannya
   satu skor kontinu $s \in [0;3]$, bukan `argmax`. Skrip: `panen_ordinal.py`.

**Temuan empiris terukur.**

| Komponen | Metrik | Nilai |
|---|---|---:|
| Detektor agnostik | `AP50` test 953 | **0,8104** |
| | presisi / daya tangkap | 0,7942 / 0,7271 |
| Kepala ordinal | MAE skor validasi terbaik | **0,3474** |

Sebagai pembanding, ansambel WBF tiga detektor proyek mencapai `AP50` agnostik
`0,8350`, dan `0,8419` setelah lapisan *re-ranker*. Satu model tunggal pada
audit ini mendekati angka tersebut tanpa ansambel.

**Keputusan metodologis.** Skor ordinal tunggal menggantikan keputusan empat
kelas. Keputusan kasar matang/belum dan keputusan halus di dalam tiap kelompok
keduanya menjadi **ambang pada skor yang sama**, sehingga tandan yang keliru di
batas B2\|B3 tidak hilang permanen sebagaimana pada hierarki keras.

---

## AF-E-012 — Pipeline Panen ujung ke ujung

**Rancangan.** Deteksi agnostik → penaut tepi terlatih → `UF` berkendala sisi
(versi diperbaiki `AF-E-010`) → skor kematangan tingkat tandan (rerata berbobot
keyakinan atas seluruh tampak anggota klaster) → ambang. Penaut memakai 13 fitur
termasuk pergeseran horizontal **bertanda** menurut arah rotasi kamera dan
selisih skor kematangan. Seluruh ambang ditala pada VALIDATION; TEST dibuka
satu kali. Metrik dihitung pada pohon empat sisi saja (132 pohon test),
sejalan dengan konvensi metrik proyek.
Skrip: `panen_pipeline.py`, `panen_eval.py`, `panen_final.py`.

**Temuan empiris terukur.** Penaut versi pertama dilatih pada pasangan dengan
`conf ≥ 0,10` sedangkan inferensi memakai `conf ≥ 0,30`; ketidakcocokan
distribusi itu memberi VAL AUC `0,9064` dan AP `0,3609`. Setelah dilatih ulang
pada ambang yang sama dengan inferensi: AUC `0,9185`, AP **`0,5562`**.
Sebagai pembanding, penaut tepi proyek mencatat AUC `0,94846` dan AP `0,59636`.

| Metrik test 953 | Pipeline Panen | `V2-E-045` | GSP terkunci |
|---|---:|---:|---:|
| F1 fisik | 0,7619 | 0,8043 | **0,8387** |
| Presisi / daya tangkap | 0,8538 / 0,6878 | — | — |
| Akurasi kelas empat | 0,7161 | 0,7111 | **0,7442** |
| Makro-F1 kelas empat | **0,6692** | — | 0,6034 |
| Akurasi ordinal ±1 | **0,9946** | — | — |
| Akurasi dua kelas (matang/belum) | **0,8678** (F1 `0,9072`) | — | — |

**Keputusan metodologis.** Dua hasil berlawanan arah dan keduanya dilaporkan.
Keputusan kelas di tingkat tandan dengan skor ordinal **mengungguli** hasil
terkunci proyek pada makro-F1 (`+0,0658`) dan memberi metrik ordinal ±1 yang
praktis sempurna. Sebaliknya, penaut audit ini **lebih lemah** daripada GSP MILP
proyek: F1 fisik `0,7619` berbanding `0,8387`, terutama karena daya tangkap
`0,6878`. Klaim yang sah karena itu terbatas pada tahap klasifikasi, bukan pada
tahap asosiasi.

**Batasan validitas.** Proposal berasal dari satu detektor, bukan WBF tiga
detektor; sebagian selisih F1 fisik dapat berasal dari situ dan belum
dipisahkan. Tidak ada selang kepercayaan berpasangan untuk selisih makro-F1.

---

## AF-E-013 — Lapisan pencacahan Ridge per kelas dan cacah tandan siap panen

**Rancangan.** Cacah klaster mentah diganti lapisan Ridge yang memetakan
statistik klaster dan statistik deteksi multi-ambang (37 fitur, gaya `F_all`)
menuju empat target cacah per pohon: total, B1, B1+B2, dan B3+B4. Dilatih pada
TRAIN, `alpha` melalui `RidgeCV`, TEST dibuka sekali. Skrip: `panen_count.py`,
`panen_final.py`.

**Temuan empiris terukur.** Test 953, 132 pohon empat sisi.

| Besaran | MAE | Tepat | ±1 | Rerata acuan |
|---|---:|---:|---:|---:|
| **B1 — siap panen** | **0,402** | **0,629** | **0,970** | 0,86/pohon |
| B1+B2 — matang | 1,068 | 0,379 | 0,765 | 2,72/pohon |
| B3+B4 — belum matang | 1,636 | 0,235 | 0,545 | 7,45/pohon |
| Total tandan | 1,402 | 0,227 | 0,568 | 10,17/pohon |

Perbandingan langsung terhadap cacah klaster mentah pada profil yang sama:
total MAE `2,288 → 1,402`, ±1 `0,371 → 0,568`; B3+B4 MAE `2,189 → 1,636`.
Pembanding proyek untuk cacah total: `V2-E-045` MAE `1,393` dengan ±1 `0,6148`;
GSP MAE `1,363` dengan ±1 `0,6370`.

**Keputusan metodologis.** Target `≥95%` **tercapai untuk cacah tandan siap
panen** (`±1 = 0,970`) pada pipeline ujung ke ujung tanpa *oracle* apa pun.
Target yang sama **tidak tercapai** untuk cacah total (`0,568`) maupun untuk
gabungan B1+B2 (`0,765`). Karena kartu dataset menetapkan B1 sebagai
*optimal harvest stage* sedangkan B2 masih *transitioning*, besaran operasional
yang benar adalah B1, bukan B1+B2 — dan besaran itulah yang justru memenuhi
target.

**Batasan validitas.** Cacah B1 memiliki rerata acuan hanya `0,86` per pohon,
sehingga ambang toleransi ±1 relatif longgar terhadap besarannya; angka ini
tidak sebanding langsung dengan ±1 pada cacah total yang reratanya `10,17`.
Cacah total masih di bawah hasil terkunci proyek, konsisten dengan penaut yang
lebih lemah pada `AF-E-012`.

---

## AF-E-014 — Sweep dijalankan ulang dengan `UF` yang benar, dan koreksi atas `AF-E-010`

**Rancangan.** Setelah perbaikan `AF-E-010` diterapkan, seluruh penelusuran
parameter dijalankan ulang dengan **grid yang persis sama** dengan berkas hasil
lama (`proposal_min` 9 nilai × `link_threshold` 10 nilai × `singleton_min`
7 nilai = 630 konfigurasi, `max_size` 3, `pair_mode` "all", `vote_mode`
softvote, split test), memakai dump WBF `combined1716` yang sudah tersimpan di
repositori. Skrip: `scripts/sweep_remote_pipeline.py` (versi diperbaiki) dan
`scripts/audit_forensik/uf_impact.py`.

**Temuan empiris terukur.**

| Kumpulan | Profil terbaik | F1 lama → baru | MAE lama → baru | Konfigurasi berubah |
|---|---|---|---|---:|
| 953 | proposal `0,10`, tautan `0,20`, singleton `0,25` | `0,8068 → 0,8068` | `1,719 → 1,719` | **0 dari 630** |
| Depth | idem berkas lama | `0,8231 → 0,8231` | `1,091 → 1,091` | **0 dari 630** |

Pengukuran pelanggaran kendala pada daftar tepi yang **sebenarnya dipakai**
sweep — yaitu setelah `linear_sum_assignment` per pasangan sisi:

| `pair_mode` | `max_size` | Klaster melanggar (versi cacat) | Δ jumlah klaster setelah diperbaiki |
|---|---:|---:|---:|
| `all` | 2 | 0,00% | 0 |
| `all` | 3 | 0,00% | 0 |
| `all` | **4** | **7,95%** | **+60** |
| `adjacent` | 2, 3, 4 | 0,00% | 0 |

**Keputusan metodologis — koreksi terhadap `AF-E-010`.** Angka `45,3%` pada
`AF-E-010` **melebih-lebihkan dampak operasional cacat tersebut**. Angka itu
diukur pada daftar tepi geometri sederhana tanpa penugasan Hungarian,
sedangkan jalur sweep yang sebenarnya menerapkan `linear_sum_assignment` pada
setiap pasangan sisi lebih dahulu. Penugasan satu-lawan-satu itu membuat satu
deteksi hanya dapat memiliki satu tepi per pasangan sisi, sehingga membentuk
klaster dengan dua anggota dari sisi yang sama memerlukan **tiga pasangan sisi
berbeda** — mustahil bila anggotanya paling banyak tiga. Cacat itu karena itu
**dorman untuk seluruh profil yang pernah dikunci proyek**: `V2-E-043` memakai
maksimum dua anggota, `V2-E-045` memakai tiga anggota bersebelahan.

Perbaikannya tetap dipertahankan karena ia mencegah kegagalan nyata pada
`max_size ≥ 4` dengan `pair_mode` "all" — konfigurasi yang ada di dalam ruang
pencarian dan dipakai oleh jangkar Hungarian A pada `GSP_LINKER` (`max_size` 4).
Namun perbaikan ini **tidak mengubah satu pun angka test terkunci**, dan
penilaian `docs/ANALISIS_PIPELINE_MENDALAM.md` §5.5 bahwa cacat tersebut dorman
terbukti benar.

**Batasan validitas.** Verifikasi ini memakai bank `combined1716` dengan
`vote_mode` softvote pada split test. Jangkar Hungarian A pada `GSP_LINKER`
memakai `max_size` 4; profil itu berada di wilayah tempat cacat aktif dan
**belum** dijalankan ulang di sini karena dump serta jalur evaluasinya berbeda.
Itu adalah satu-satunya profil terkunci yang masih perlu diperiksa ulang.

---

## AF-E-015 — Harga presisi untuk memulihkan tandan yang punya kandidat

**Konteks.** `results/audit_2026-09-06/recovery_budget_val.json` (sesi lain,
6 September 2026) menunjukkan bahwa 227 dari 287 tandan yang tidak terpasangkan
sudah memiliki kandidat mentah, dan menyimpulkan adanya peluang pemulihan
sebelum keputusan penyaringan menjadi permanen. Berkas tersebut menyatakan
sendiri bahwa cakupan itu dibantu GT dan **belum** membuktikan pemulihan tanpa
tambahan positif palsu. Entri ini mengukur biaya tersebut.

**Rancangan.** Ambang detektor dan ambang *singleton* diturunkan bertahap pada
91 pohon VALIDATION memakai cache kandidat, `edge_model_v2.pkl`, dan
`link_thr`/`max_size` Pipeline Panen yang tetap. Yang dicatat adalah kurva
operasi identitas fisik, bukan metrik citra. Batas atas dihitung dengan
mempertahankan seluruh kandidat longgar dan membuang seluruh positif palsu
secara *oracle*. Skrip: `scripts/audit_forensik/recovery_price.py`.

**Temuan empiris terukur.** VAL, 91 pohon, 936 tandan acuan.

| `det_conf` | *singleton* | Daya tangkap | Presisi | F1 | TP | FP |
|---:|---:|---:|---:|---:|---:|---:|
| 0,30 | 0,45 | 0,6934 | 0,8374 | 0,7586 | 649 | 126 |
| **0,25** | **0,40** | 0,7361 | 0,8106 | **0,7716** | 689 | 161 |
| 0,20 | 0,35 | 0,7714 | 0,7568 | 0,7640 | 722 | 232 |
| 0,15 | 0,30 | 0,8141 | 0,6773 | 0,7394 | 762 | 363 |
| 0,12 | 0,25 | 0,8526 | 0,6009 | 0,7049 | 798 | 530 |
| 0,10 | 0,20 | 0,8803 | 0,5192 | 0,6532 | 824 | 763 |
| 0,10 | 0,00 | **0,9156** | 0,3598 | 0,5166 | 857 | 1.525 |

Memulihkan `+208` tandan menuntut `+1.399` positif palsu, yaitu **`6,7` positif
palsu per tandan yang dipulihkan**. Batas atas daya tangkap dengan kandidat
longgar adalah `0,9156`; **`79` tandan (`8,4%`) tetap tak terjangkau** oleh
pengelompokan kandidat apa pun.

**Keputusan metodologis — palang keputusan untuk decoder yang diusulkan.**
Peluang pemulihan itu nyata, tetapi bukan daya tangkap gratis. Pada titik
operasi longgar, penyaring terpelajar harus:

- membuang **≥ 69%** dari `1.525` klaster palsu hanya untuk **impas** pada F1
  terhadap profil sekarang (FP turun ke bawah `466`);
- membuang **≈ 89%** untuk sekaligus mempertahankan presisi `0,8374`
  (FP turun ke bawah `166`);
- membuang hanya `50%` menghasilkan F1 `0,672`, yaitu **lebih buruk** daripada
  profil sekarang.

Angka-angka ini menjadi kriteria yang dapat diperiksa untuk usulan pada
`docs/research_2026-09-06/USULAN-PERBAIKAN.md`, menggantikan penilaian kualitatif
"peluang besar".

**Dua konsekuensi praktis.**

1. Garis dasar pembanding yang adil bukan `F1 0,7586`, melainkan **`0,7716`**
   pada profil `det_conf 0,25` / *singleton* `0,40` — kenaikan yang diperoleh
   tanpa model baru. Membandingkan decoder terhadap `0,7586` akan mengatributkan
   hasil penalaan ambang sebagai kemajuan arsitektur.
2. Tahap "pulihkan objek tanpa kandidat" memang layak ditunda: imbalannya
   terbatas pada `8,4%` tandan sedangkan biayanya paling besar. Urutan tahap
   pada usulan tersebut didukung data ini.

**Batasan validitas.** Kurva ini memakai penaut dan `link_thr` yang tetap;
penaut yang lebih baik dapat menggeser seluruh kurva. Pengukuran hanya pada
VALIDATION 953. Batas `0,9156` bergantung pada cache kandidat `conf ≥ 0,10` dan
bukan seluruh keluaran detektor sebelum NMS, sehingga merupakan batas atas untuk
cache tersebut, bukan untuk detektor secara umum.

---

## AF-E-016 — Jangkar Hungarian A diperiksa: cacat `UF` tidak memengaruhinya

**Konteks.** `AF-E-014` menyisakan satu butir terbuka: jangkar Hungarian A
pada `GSP_LINKER` adalah satu-satunya profil test-locked yang memakai
`max_size = 4`, yaitu wilayah tempat cacat `UF` berpotensi aktif (`7,95%`
pelanggaran pada `pair_mode="all"`).

**Penelusuran jalur kode.** Jalur terkunci Anchor A memang melewati `UF` yang
cacat: `results/remote_eval_2026-08-28/scripts/rank_and_emit.py` →
`scripts/evaluate_remote_class_head.evaluate_payload` → `sweep.clusters(...)` →
`sweep_remote_pipeline.UF`. Modul `train_detection_edge_linker` hanya memakai
`sweep.iou` dan `sweep.build_edges`, bukan `UF`. Jalur GSP tidak mewarisi cacat
ini karena menjamin ≤ 1 proposal per sisi secara struktural pada
`enumerate_candidates`.

**Rancangan.** Kedua varian `UF` diberi daftar tepi yang identik pada profil
Anchor A yang persis — proposal `0,125`, `pair_mode` bersebelahan, tautan
`0,15`, *singleton* `0,15`, `max_size` `4` — memakai dump WBF `combined1716`
softvote dan prior rotasi yang dilatih dari TRAIN.
Skrip: `scripts/audit_forensik/anchor_a.py`.

**Temuan empiris terukur.** Split TEST 953, 135 pohon empat sisi.

| Varian | Klaster | Melanggar kendala sisi | Pohon dengan partisi berbeda |
|---|---:|---:|---:|
| Versi repositori (cacat) | 1.586 | 0 | — |
| Versi diperbaiki | 1.586 | 0 | **0 dari 135** |

**Keputusan metodologis.** Partisi klaster identik pohon demi pohon, sehingga
seluruh metrik hilir Anchor A — F1 fisik `0,8387`, MAE `1,3630`, ±1 `0,6370`,
akurasi kelas `0,7442`, makro-F1 `0,6034` — **tidak berubah** oleh perbaikan
`AF-E-010`. Alasannya sesuai argumen `GSP_LINKER.md`: dengan hanya pasangan
sisi bersebelahan, menutup siklus yang mengulang satu sisi memerlukan jalur
0→1→2→3→0, yaitu lima anggota, sedangkan `max_size = 4` sudah memblokirnya.

Dengan ini seluruh profil test-locked proyek telah diperiksa terhadap cacat
`AF-E-010`, dan **tidak satu pun angka terkunci yang terpengaruh**.

**Batasan validitas.** Pemeriksaan pada split VALIDATION menghasilkan nol
klaster karena dump `fused_combined1716` yang tersimpan di repositori hanya
memuat split TEST; baris VAL karena itu kosong, bukan lulus. Yang benar-benar
terukur adalah TEST, yaitu split tempat angka Anchor A dikunci.
