# Proposal — Pipeline per-tandan: deteksi → penautan lintas-sisi → klasifikasi terkumpul

Status: **proposal awal disetujui 2026-08-17; menjadi landasan eksperimen V1/V2
yang kemudian dijalankan dan dicatat di log hasil.**
ID: `PT-E-001` … `PT-E-005`.
Asal-usul: sketsa tangan 22 Juli 2026 —
[`sketsa-asal-2026-07-22.png`](sketsa-asal-2026-07-22.png).

> **Posisi dokumen ini.** Ini adalah *design record* historis untuk proposal
> per-tandan, bukan satu-satunya sumber status implementasi. Peta versi dan
> arsitektur terkini ada di [`PROPOSAL-Pipeline.md`](../../PROPOSAL-Pipeline.md).
> V1 berarti jalur baseline/original; V2 berarti jalur learned proposal,
> re-ranking, dan GSP. Hasil rinci tetap berada di
> [`experiments/EKSPERIMEN.md`](../../experiments/EKSPERIMEN.md) dan laporan
> wave yang ditautkan dari handoff.

Semua angka "bukti awal" di §4 terlacak ke
[`results/probe_penautan_953.json`](../results/probe_penautan_953.json),
dihasilkan oleh [`scripts/probe_penautan_953.py`](../scripts/probe_penautan_953.py).
Konvensi kerja folder ini: [`../CLAUDE.md`](../CLAUDE.md).
Log append-only: [`../EKSPERIMEN.md`](../EKSPERIMEN.md).

---

## 1. Ringkasan

Yang diusulkan mengubah **satuan inferensi**: dari "kotak di dalam sebuah citra"
menjadi "tandan fisik di sebuah pohon". Detektor tetap bekerja per citra, tetapi
keluarannya tidak lagi dinilai per citra — deteksi dari 4 (atau 8) sisi pohon
yang sama dikumpulkan ke dalam satu *pool* per tandan, lalu satu keputusan kelas
dikeluarkan untuk tandan itu.

Empat hal yang membuat ini layak dikerjakan sekarang:

1. **Kebenaran acuannya sudah ada di dalam dataset vanilla.** `json/*.json`
   memuat `bunches[]` dan `_confirmedLinks` — graf identitas tandan lintas-sisi
   hasil anotasi manusia, 9.823 tandan unik dari 18.540 kotak. Penaut bisa
   dilatih **dan** diuji, bukan cuma ditebak.
2. **Detektor dasarnya sudah ada dan bersih.** `models/yolo26l_e60_i1280_v2repro/best.pt`
   (sel 5, YOLO26l @1280) dilatih pada split kanonik 716 pohon, test mAP50
   0,5436. Dump prediksi test-nya juga sudah ada. Tidak perlu training ulang
   untuk memulai.
3. **Keuntungan terbesarnya sudah terukur dan gratis.** Sekadar mengganti satuan
   evaluasi menaikkan recall **+14,49 pp** (63,36% per-kemunculan → 77,85%
   per-tandan, conf 0,25, test). Tanpa model baru, tanpa penaut, tanpa apa pun.
4. **Kaveatnya juga sudah terukur.** Penaut geometri-saja hanya mencapai F1
   pasangan 0,4282 di test. Penaut adalah bottleneck sebenarnya, dan itulah
   satu-satunya bagian yang benar-benar perlu diteliti.

Tiga hasil negatif terdahulu (E-007, F-003, E-016) menyentuh wilayah ini dan
**semuanya harus dijawab lebih dulu** — §3.

---

## 2. Yang diminta (dari sketsa 22 Juli 2026)

Sketsanya: empat kotak sisi di kiri, masing-masing berisi beberapa tandan;
panah dari semua sisi menuju satu corong; di dalamnya sebuah ruang berlabel
**T1** yang memuat potongan **V1** dan **V2** dari tandan yang sama; keluar
sebuah panah ke **B1 / B2 / B3**.

Diformalkan menjadi empat modul:

```
  sisi 1 ┐
  sisi 2 ┤   ┌───┐  kotak    ┌───┐   pool    ┌───┐  p_v   ┌───┐
  sisi 3 ┼──▶│ D │──+skor───▶│ L │──per──────│ C │───────▶│ A │──▶ 1 kelas / tandan
  sisi 4 ┘   └───┘           └───┘  tandan   └───┘        └───┘        │
                               │                                       ▼
                               └─ T1 = {crop@V1, crop@V2, crop@V3}   inventaris
                                  T2 = {crop@V2, crop@V4}            per pohon
                                  T3 = {crop@V1}
```

| Modul | Isi | Status |
|---|---|---|
| **D** detektor | kotak + skor kelas per sisi | **sudah ada** (sel 5) |
| **L** penaut | partisi kotak → pool, satu pool = satu tandan fisik | **belum ada** ← inti riset |
| **C** pengklasifikasi | skor kelas per anggota pool | sebagian ada (Fase 6) |
| **A** agregator | satu keputusan kelas per pool | **belum ada** ← pertanyaan Anda |

Keluaran akhir bukan lagi daftar kotak, melainkan **inventaris per pohon**:
daftar tandan, masing-masing dengan satu kelas. Hitungan per kelas per pohon
mengikuti gratis (= jumlah pool per kelas) — itu menyambung langsung ke jalur
counting yang sudah ada.

### 2.1 Intuisi Anda soal keterlihatan — dikonfirmasi GT, persis

> "4 sisi maksimal tandan yang sama terlihat di 3 posisi, 8 sisi maksimal 5,
> melar dikit ke 6 masih mungkin"

| Konfigurasi | `appearance_count` maksimum yang benar-benar muncul di 9.823 tandan |
|---|---|
| pohon 4 sisi (908 pohon) | **3** — tidak pernah 4 |
| pohon 8 sisi (45 pohon) | **6** — sebarannya 4 (8 pohon), 5 (28 pohon), 6 (9 pohon) |

Tepat seperti yang Anda katakan. Ini dipakai sebagai **kendala keras** di modul
L (plafon ukuran pool), bukan sekadar catatan — lihat kaveat sirkularitasnya di
§9.2.

---

## 3. Kenapa ini bukan pengulangan yang sudah gagal

`docs/LAPORAN-AKHIR.md` §10 mendaftar "konsistensi lintas-sisi (F-003)" sebagai
hal yang **tidak perlu diulang**, dan `CLAUDE.md` mendaftar hal yang sama di
"Percobaan Gagal" butir 5. Proposal ini harus lewat pintu itu dulu.

| Hasil terdahulu | Yang sebenarnya diukur | Kenapa tidak mematikan proposal ini |
|---|---|---|
| **E-007** — penautan geometris **DIPALSUKAN**. Ablasi A/B/C semuanya kalah dari koreksi global k=1,8905 (macro MAE 1,392 vs 0,356) | Kualitas **counting** dari penautan. Penautnya geometri tangan: kelas + ukuran kotak (A), depth (B), pose DA3 (C) | (a) target di sini **kelas per tandan**, bukan jumlah; (b) E-007 tidak pernah memakai **embedding penampilan terlatih** — varian "hanya penampilan" isinya kelas + ukuran kotak saja, bukan piksel; (c) E-007 tidak pernah melaporkan mutu penautannya sendiri (presisi/recall pasangan) — hanya galat counting hilirnya |
| **F-003** — plafon konsistensi lintas-sisi **GUGUR**, fraksi 0,2794, CI95 [0,2353; 0,3235] | Hanya **transfer kelas**: dari kemunculan yang salah kelas, berapa yang punya saudara yang benar | Berkas hasil F-003 yang sama juga mencatat **plafon deteksi terlewat = 0,4946** (CI [0,4583; 0,5309]) — hampir separuh kemunculan yang terlewat punya saudara yang terdeteksi. **Itu** yang dipanen pipeline ini, dan F-003 tidak pernah memakainya. Tambahan: F-003 memakai proksi **yolo26n**, dan berkas hasilnya sendiri menulis bahwa ia menjawab "apakah ada ruang", bukan "berapa besar ruangnya" |
| **E-016** — `multiview_vote` dengan tautan oracle: per-sisi 67,89% → per-tandan 68,55% (val), naik **+0,66 pp** saja | Satu aturan fusi saja: **argmax dari rerata softmax**, ConvNeXt-tiny, potongan GT | Aturan itu membuang dua hal yang diketahui benar tentang label ini: **ordinalitas** (E-012/SR-009: akurasi ±1 = 99,5%) dan **mutu tampak yang tidak seragam**. §5.4 mengusulkan tangga aturan yang memakai keduanya. Kalau tangga itu pun cuma menghasilkan +0,66 pp, gerbang G0 mematikan separuh proposal ini — dan itu memang jawabannya |

**Ringkasnya:** yang dipalsukan sebelumnya adalah *penautan geometris untuk
counting* (E-007) dan *transfer kelas antar sisi* (F-003). Yang belum pernah
diuji adalah *penautan berbasis penampilan terlatih* dan *agregasi yang sadar
ordinalitas* — dan keuntungan terbesar yang terukur (§4.3) tidak lewat mekanisme
F-003 sama sekali.

---

## 4. Bukti awal yang sudah diukur

Semua dari `results/probe_penautan_953.json`. Dataset vanilla, tanpa modifikasi.

### 4.1 Struktur graf identitas

| Split | Pohon | 4-sisi / 8-sisi | Kotak | Tandan unik | Multi-sisi |
|---|---|---|---|---|---|
| train | 716 | 682 / 34 | 14.041 | 7.427 | 5.546 (74,7%) |
| val | 96 | 91 / 5 | 1.887 | 992 | 760 (76,6%) |
| test | 141 | 135 / 6 | 2.612 | 1.404 | 1.022 (72,8%) |
| **total** | **953** | 908 / 45 | **18.540** | **9.823** | **7.328 (74,6%)** |

Sebaran `appearance_count` global: 1 → 2.495 · 2 → **6.264** · 3 → 834 ·
4 → 147 · 5 → 71 · 6 → 12.

Dua konsekuensi langsung:

- **25,4% tandan hanya terlihat di satu sisi.** Pooling secara struktural tidak
  bisa menolong mereka. Batas atas pengaruh pipeline ini = 74,6% tandan.
- **Dari yang bisa di-pool, 85,5% punya TEPAT DUA sisi.** Ini mematikan voting
  mayoritas sebagai aturan keputusan: dua suara yang berbeda selalu seri. Aturan
  agregasi **wajib kontinu** (skor), bukan voting. Lihat §5.4.

### 4.2 Label kelas konsisten sempurna antar sisi

`class_mismatch = 0` untuk **seluruh** 9.823 tandan, dan 100,00% pasangan kotak
setandan punya kelas identik. Artinya: **setiap ketidaksepakatan kelas antar
sisi yang muncul nanti adalah galat model murni, bukan derau label.** Ini
membuat metrik pooling bisa dibaca lurus. (Konsisten dengan E-001, yang
memalsukan `class_mismatch` sebagai ukuran ambiguitas justru karena nilainya
nol.)

### 4.3 Keuntungan gratis: recall per-tandan ≫ recall per-kemunculan

Detektor sel 5 (YOLO26l @1280), test 141 pohon, IoU 0,5, tanpa penaut apa pun
(memakai tautan GT hanya untuk mengelompokkan — jadi ini plafon):

| conf | recall per-KEMUNCULAN | recall per-TANDAN (≥1 sisi) | selisih |
|---|---|---|---|
| 0,15 | 75,77% | 86,61% | **+10,84 pp** |
| **0,25** | **63,36%** | **77,85%** | **+14,49 pp** |
| 0,35 | 50,31% | 66,03% | **+15,72 pp** |

Kelas benar juga ikut: 46,78% per-kemunculan → 60,75% tandan yang kelasnya benar
di ≥1 sisi (conf 0,25).

Ini bukan trik metrik. Untuk pemakaian di lapangan — menghitung dan menilai
tandan **pada sebuah pohon** — tandan yang terlewat di tiga sisi tetapi tertangkap
di satu sisi memang sudah ketemu. Satuan per-citra menghukum tiga kali;
satuan per-tandan menghukum nol kali, dan yang kedua itulah yang benar untuk
tugasnya.

### 4.4 Ruang untuk aturan keputusan: 23,3%

Di antara tandan yang terdeteksi di ≥2 sisi, prediksi kelas antar sisi
**berbeda** pada:

| conf | tandan ≥2 sisi terdeteksi | kelasnya tidak sepakat |
|---|---|---|
| 0,15 | 659 | 169 (25,6%) |
| 0,25 | 490 | 114 (23,3%) |
| 0,35 | 332 | 71 (21,4%) |

Jadi modul A punya sesuatu untuk diputuskan pada ~seperempat pool. Bukan nol —
tapi juga bukan setengah. Ini yang membatasi keuntungan pooling kelas, dan
angkanya sejalan dengan F-003.

### 4.5 Bottleneck sebenarnya: penaut

Probe penaut **tanpa piksel** — 12 fitur geometri + kelas
(`gap_sisi`, `|Δcx|`, `|Δcy|`, rasio area/w/h, Δaspek, `cy` rerata, kelas sama,
log rasio area), HistGradientBoosting dilatih di 121.891 pasangan train:

| Bentuk | Split | ROC-AUC | Presisi | Recall | F1 | ARI |
|---|---|---|---|---|---|---|
| D1 ambang per-pasangan | val | **0,9301** | 0,3669 | 0,7155 | **0,4850** | — |
| D2 penugasan global berkendala | val | — | 0,3909 | 0,5352 | 0,4518 | 0,4385 |
| D2 dikunci ke test (ambang 0,15) | test | — | 0,3742 | 0,5003 | **0,4282** | 0,3912 |

Bacaannya penting dan agak berlawanan intuisi:

- **AUC 0,93 tapi F1 0,48.** Sinyal peringkatnya kuat, tetapi prevalensi positif
  cuma 6,62% (10.538 dari 159.198 pasangan lintas-sisi) — jadi ambang
  per-pasangan selalu tenggelam oleh negatif. Ini menegaskan bahwa penaut harus
  dirumuskan sebagai **penugasan**, bukan klasifikasi pasangan independen.
- **Tetapi penugasan berkendala pun tidak menolong** (0,4518 vs 0,4850). Hungarian
  per pasangan-sisi + plafon ukuran pool tidak menambal kekurangan sinyalnya.
  Kesimpulannya tegas: **geometri saja tidak cukup untuk menaut.** Ini
  mereplikasi E-007 lewat jalur berbeda dan dengan metrik penautan langsung
  (yang E-007 sendiri tidak pernah laporkan).

Prior geometri yang tetap berguna, dari prevalensi positif menurut jarak sisi
melingkar (val): 4-sisi gap 1 = 8,79% vs gap 2 = **1,60%**; 8-sisi gap 1/2/3/4 =
14,47% / 8,33% / 3,74% / 1,54%. Sisi berseberangan hampir tidak pernah berbagi
tandan — fisiknya masuk akal, dan itu fitur yang murah.

### 4.6 Biaya galat-gabung: sebagian, tidak seluruhnya, murah

Kalau penaut keliru menggabung dua kotak yang bukan tandan sama, peluang keduanya
kebetulan **sekelas = 37,74%** — pada kasus itu galat penautan tidak mengubah
kelas keluaran sama sekali (hanya merusak hitungan). Sisanya, **62,26%, memang
merusak.**

Jadi argumen "untuk klasifikasi, penaut boleh lebih jelek daripada untuk
counting" **berlaku sebagian saja**. Angka 37,74% itu untuk gabungan acak;
penaut nyata menggabung yang *mirip*, dan yang mirip lebih sering sekelas, jadi
angka efektifnya akan lebih tinggi — tapi itu **belum diukur** dan tidak boleh
diklaim sebelum PT-E-002.

---

## 5. Rancangan

### 5.1 Modul D — detektor

**Rekomendasi: pakai ulang sel 5** (`models/yolo26l_e60_i1280_v2repro/best.pt`).
Alasan: dilatih pada split kanonik 716 pohon (jadi test 141 pohon bersih),
bobotnya ada di git, dump prediksi test-nya sudah ada, dan Anda sendiri bilang
arsitekturnya tidak masalah. Melatih YOLO26m baru menambah ~2 jam GPU tanpa
menjawab pertanyaan apa pun di proposal ini; kalau tetap diinginkan, taruh
sebagai ablasi di akhir, bukan di awal.

> ⚠ **Jangan pakai detektor/klasifikasi Fase 6 di split test.** `agn953_full`
> dan turunannya melihat **122 dari 141 pohon test** saat training
> (LAPORAN-AKHIR §9.2: test bersih hanya 19 pohon, AP50 0,7702 vs test penuh
> 0,8090). Semua model baru di sub-proyek ini dilatih **hanya** di 716 pohon train.

### 5.2 Modul L — penaut lintas-sisi ← inti riset

**Titik berangkatnya bukan nol.** Deduplikasi lintas-sisi sudah ada dan sudah
berjalan — graf `_confirmedLinks` di dalam dataset adalah hasilnya, dan
README dataset menyebut metodenya "expert agronomists using multi-view
cross-referencing". Algoritma dedup yang sudah dibuat sebelumnya itu **wajib
menjadi baseline PT-E-002**, bukan disaingi dari nol.

Konsekuensinya untuk rancangan: yang diukur pertama bukan "apakah penaut bisa
dibangun", melainkan **berapa angka penaut yang sudah ada pada metrik penautan
langsung** (presisi/recall/F1 pasangan, ARI) — angka yang, sejauh yang bisa
ditelusuri di workspace ini, belum pernah dicatat dalam bentuk itu. Dua
kemungkinan hasilnya sama-sama berguna:

| Kalau baseline itu | Maka |
|---|---|
| sudah ≥ ambang G1 (F1 0,65 / ARI 0,55) | modul L **selesai**. Langsung ke PT-E-003; embedding re-ID jadi opsional, dan seluruh sisa anggaran GPU pindah ke modul C/A |
| di bawah ambang G1 | baru embedding re-ID dikerjakan, dan gainnya diukur sebagai **delta terhadap baseline itu**, bukan terhadap 0,4282 |

> ⚠ **Yang saya butuhkan dari Anda:** lokasi algoritma dedup itu (repo, berkas,
> atau alat anotasinya). Ia tidak ada di `/workspace` — yang ada di sini cuma
> keluarannya (graf tautan di dalam JSON) dan penaut geometris E-007 yang sudah
> dipalsukan. Tanpa kodenya, PT-E-002 terpaksa mulai dari baseline geometri
> 0,4282 yang jauh lebih rendah, dan angka gain-nya akan menyesatkan.

Kalau baseline itu perlu diperkuat, yang belum pernah dicoba di repo ini adalah
**embedding re-ID tandan yang dilatih dari graf identitas GT.**

- **Data latih**: 716 pohon train → 7.427 tandan, 5.546 di antaranya multi-sisi
  → **8.034 pasangan positif** dan 113.857 pasangan negatif *dalam pohon yang
  sama*. Negatif dalam-pohon inilah yang berharga; negatif lintas-pohon terlalu
  mudah dan tidak mengajarkan apa pun.
- **Backbone**: ResNet-18 atau ConvNeXt-tiny, potongan 224×224 dengan padding
  25% (pola `build_crop_dataset.py` yang sudah ada).
- **Loss**: supervised contrastive / InfoNCE, positif = `bunch_id` sama,
  negatif = tandan lain di pohon yang sama (hard negative).
- **Skor pasangan**: `s(i,j)` = HistGradientBoosting atas 12 fitur geometri §4.5
  **+ cosine embedding** sebagai fitur ke-13. Bentuk ini menjaga agar penambahan
  penampilan bisa diukur sebagai delta yang bersih terhadap baseline 0,4282.
- **Penugasan**: per pohon, Hungarian per pasangan-sisi, lalu union-find serakah
  dengan dua kendala keras:
  1. satu kotak per sisi per tandan (sebuah tandan tidak muncul dua kali di satu
     foto);
  2. plafon ukuran pool: **3** untuk 4-sisi, **6** untuk 8-sisi (§2.1).

Kalau versi serakah ini jadi pembatas, naik ke correlation clustering; tapi
jangan mulai dari sana.

### 5.3 Modul C — pengklasifikasi

Tiga varian, naik urutan biaya:

| Varian | Isi | Biaya |
|---|---|---|
| **C1** | pakai skor kelas detektor apa adanya | nol |
| **C2** | classifier potongan khusus (ConvNeXt, pola Fase 6) dilatih ulang **hanya di 716 pohon train** | ~2 jam GPU + regenerasi crops (~15 mnt, `REGENERASI.md`) |
| **C3** | classifier **multi-tampak**: satu model melihat seluruh pool sekaligus (attention antar tampak), bukan per-sisi lalu digabung | ~4 jam GPU |

C1 dulu — ia sudah cukup untuk melewati gerbang G0, dan C3 baru masuk akal
kalau G0 lolos. C3 adalah bentuk paling setia terhadap sketsa Anda (satu ruang
T1 masuk, satu kelas keluar).

### 5.4 Modul A — aturan keputusan ← pertanyaan Anda

Untuk pool `P` berisi tampak `v` dengan softmax `p_v` atas {B1,B2,B3,B4} dan
bobot mutu `w_v`:

| ID | Aturan | Catatan |
|---|---|---|
| **R0** | tampak tunggal berkeyakinan tertinggi | **baseline "tanpa pipeline"** — pembanding wajib |
| **R1** | confidence tertinggi (usulan Anda) | rapuh: satu tampak berkeyakinan-tinggi-tapi-salah menang telak |
| **R2** | argmax rerata softmax | **inilah yang E-016 sudah coba: +0,66 pp** |
| **R3** | argmax rerata **berbobot mutu** | `w_v` dari luas bbox, jarak ke tepi citra (keterpotongan), conf detektor, ketajaman (varians Laplacian) |
| **R4** | **ekspektasi ordinal** (rekomendasi) | `ĝ = Σ_v w_v Σ_k k·p_vk / Σ_v w_v`, lalu potong dengan ambang `τ1,τ2,τ3` yang **dilatih di val** |
| **R5** | agregator terlatih (DeepSets/attention) | kapasitas tertinggi, risiko overfit tertinggi (7.427 tandan train) |

**Kenapa R4 yang saya rekomendasikan, bukan R1 atau R2:**

1. **Voting mati di sini.** 85,5% pool yang bisa di-pool punya tepat dua tampak
   (§4.1). Aturan apa pun yang bersandar pada suara mayoritas akan seri pada
   mayoritas kasus. R4 kontinu, jadi tidak pernah seri.
2. **Kelasnya ordinal, dan itu bukan tafsiran.** README dataset menyatakannya
   eksplisit: *"Biological order: B1 → B2 → B3 → B4 from most ripe to least
   ripe"* — B1 matang (merah, besar, bulat), B2 transisi, B3 mentah (hitam,
   berduri, lonjong), B4 sangat mentah. Jadi encoding integer 0–3 memang sumbu
   kematangan yang monoton (menurun), dan ekspektasi di atasnya punya arti
   fisik. Ini diperkuat dua hasil repo: E-012/SR-009 (kebingungan bersifat
   ordinal) dan E-016 (akurasi ±1 = 99,5%). Kalau tampak A
   bilang B2 (0,6) dan tampak B bilang B4 (0,6), R1 dan R2 sama-sama harus
   memilih **B2 atau B4** — padahal dengan galat yang 99% berjarak satu langkah,
   kebenaran yang paling mungkin justru **B3**, yaitu di antaranya. Hanya R4
   yang bisa mengeluarkan B3. Ini bukan detail: itu tepat kasus 23,3% di §4.4.
3. **Keluarannya kontinu**, jadi hitungan per kelas per pohon bisa dijumlahkan
   sebagai soft count untuk jalur counting hilir tanpa membuang informasi.

**Risiko R4 yang harus ikut dilaporkan:** pembulatan ke tengah menguntungkan
B2/B3 dan menekan B1/B4 — padahal B1/B4 justru kelas yang recall-nya paling baik
(E-012: B1 70,2%, B4 62,9% vs B2 42,4%, B3 41,6%), dan B3 sudah mendominasi
51,6% tandan. Mitigasi: `τ1,τ2,τ3` **dicari di val**, bukan dipatok di 0,5/1,5/2,5;
dan recall per kelas wajib dilaporkan berdampingan dengan akurasi agregat. Kalau
B1/B4 turun sementara akurasi total naik, itu **bukan** kemenangan.

Tangga R0→R4 dijalankan **dua kali**: sekali dengan tautan oracle (GT) dan
sekali dengan penaut nyata. Selisihnya yang memisahkan "aturannya lemah" dari
"penautnya lemah".

---

## 6. Metrik dan protokol

| Tingkat | Metrik |
|---|---|
| Penaut | presisi/recall/F1 pasangan, ARI per pohon, galat jumlah pool (bias + MAE) |
| Tandan | akurasi, macro-F1, **recall per kelas**, MAE ordinal, akurasi ±1 |
| Pohon | MAE hitungan per kelas, joint accuracy — sebanding dengan 77,48% / 32,62% / 1,036 |

Aturan yang mengikat:

- **Pemilihan konfigurasi selalu di val. Test dievaluasi SEKALI** (pola
  `fuse_final.py`).
- **Bootstrap di tingkat POHON, bukan tandan** — tandan dalam satu pohon
  berkorelasi kuat (satu pohon cenderung satu tingkat kematangan). Pakai
  `scripts/bootstrap_ci.py`.
- **Dekomposisi oracle vs end-to-end wajib** di setiap tabel. Tanpa itu, hasil
  buruk tidak bisa dilokalisasi ke L atau ke A.
- **Dump prediksi ke `.npz` saat evaluasi**, bukan belakangan (aturan repo;
  alasannya di `CLAUDE.md`).
- Pencocokan pool→tandan GT: sebuah pool cocok dengan tandan GT bila mayoritas
  anggotanya adalah deteksi yang ter-IoU≥0,5 ke kemunculan tandan itu. Presisi
  dan recall pool dilaporkan terpisah dari akurasi kelas.

---

## 7. Gerbang falsifikasi (pra-daftar — ditulis sebelum ada hasil)

| Gerbang | Di | Syarat lolos | Kalau gugur |
|---|---|---|---|
| **G0** nilai pooling | PT-E-001 | Aturan terbaik dengan **tautan oracle** mengalahkan R0 ≥ **+2,0 pp** akurasi per-tandan di val, CI95 bootstrap-pohon tidak memuat nol | Separuh-kelas dari proposal **mati**. Berhenti; pertahankan hanya perubahan satuan evaluasi (§4.3), yang sudah terbukti tanpa perlu eksperimen apa pun |
| **G1** penaut | PT-E-002 | F1 pasangan val ≥ **0,65** DAN ARI ≥ **0,55** — diuji lebih dulu pada **algoritma dedup yang sudah ada** (§5.2), baru pada embedding re-ID kalau yang pertama belum lolos | Penampilan terlatih pun tidak menyelesaikan penautan → catat sebagai pemalsuan kedua E-007 (kali ini dengan metrik penautan langsung), berhenti di analisis oracle |
| **G2** end-to-end | PT-E-003 | akurasi per-tandan dengan penaut nyata ≥ akurasi oracle − **2,0 pp** | Bottleneck-nya L, bukan A — hasil negatif yang terlokalisasi dan tetap layak dilaporkan |
| **G3** counting | PT-E-004 | macro MAE hitungan pohon < **0,356** (rekor koreksi global k=1,8905, `E-007/report_test.json`) | Konfirmasi ulang E-007: penautan eksplisit tetap kalah dari pembagian global untuk counting. Opsional, tidak memblokir G0–G2 |

Asal angkanya, supaya tidak terlihat sembarangan:

- **+2,0 pp (G0)**: E-016 mendapat +0,66 pp dengan R2. Kalau tangga R3/R4 tidak
  bisa melipattigakan itu, mekanismenya memang tidak ada.
- **0,65 / 0,55 (G1)**: geometri-saja sudah di 0,4518 / 0,4385 (val). Ambang ini
  menuntut lompatan yang jelas di atas derau, bukan perbaikan marginal.
- **0,356 (G3)**: angka yang tercatat di `E-007/report_test.json` untuk koreksi
  global; penautan terbaik E-007 waktu itu 1,392 — kalah 4×.

---

## 8. Rencana eksperimen dan biaya

| ID | Isi | Butuh | Biaya | Gerbang |
|---|---|---|---|---|
| **PT-E-001** | Rangka evaluasi per-tandan + plafon oracle. Pemetaan deteksi→tandan, R0–R4 di atas tautan GT, CI bootstrap-pohon. Memakai `pred_sel5_953_rgb_test.npz` + inferensi val | — | **CPU, ~1 jam** | **G0** |
| **PT-E-002a** | **Ukur algoritma dedup yang sudah ada** pada metrik penautan langsung (P/R/F1 pasangan, ARI) — baseline sebenarnya, §5.2 | PT-E-001 lolos + kode dedup-nya | CPU, ~1 jam | **G1** |
| **PT-E-002b** | Penaut re-ID: latih embedding, gabung dengan geometri, penugasan berkendala. Kunci ambang di val, test sekali. **Hanya kalau 002a di bawah G1** | PT-E-002a | GPU ~2–4 jam | **G1** |
| **PT-E-003** | Tangga aturan keputusan R0–R5 di atas penaut nyata; bandingkan ke oracle | PT-E-002 | CPU (R5: ~1 jam GPU) | **G2** |
| **PT-E-004** | Counting dari jumlah pool vs k=1,8905 vs Ridge+F_all | PT-E-003 | CPU | G3 |
| **PT-E-005** | (opsional) classifier multi-tampak C3 — bentuk paling setia ke sketsa | PT-E-003 lolos | GPU ~4 jam | — |

Total di bawah satu hari GPU pada RTX 2000 Ada 16 GB yang terpasang.
**PT-E-001 sengaja CPU-saja dan jalan lebih dulu**: ia gerbang termurah yang bisa
membatalkan sisanya, sesuai aturan "eval dan langkah pendek dijalankan di depan".

---

## 9. Risiko dan ancaman validitas

1. **Kontaminasi Fase 6.** `agn953_full` dan turunannya melihat 122/141 pohon
   test. Semua model sub-proyek ini dilatih hanya di 716 pohon train. (§5.1)
2. **Sirkularitas plafon keterlihatan.** Plafon 3/6 diambil **dari GT yang sama**
   yang dipakai menilai. Kalau anotator melewatkan tautan, plafon nyata bisa
   lebih besar dan kendala itu menjadi bias yang menguntungkan diri sendiri.
   Mitigasi: laporkan hasil dengan dan tanpa kendala plafon.
3. **Pohon 8-sisi cuma 45** (train 34 / val 5 / test 6). Tidak boleh ada
   kesimpulan terpisah untuk 8-sisi; angkanya digabung dan jumlahnya disebut.
4. **25,4% tandan sisi-tunggal** tidak bisa dibantu pipeline ini sama sekali.
   Setiap angka agregat harus juga dipecah menurut `appearance_count`.
5. **Recall anotasi graf tautan tidak diketahui.** Kalau anotator melewatkan
   tautan yang benar, positif kurang tercatat → presisi penaut **diremehkan**.
   Arah biasnya diketahui, besarnya tidak.
6. **Test hanya 141 pohon / 1.404 tandan.** LAPORAN-AKHIR §10 sudah memperingatkan
   bahwa efek berukuran ~0,03 butuh ~10× data untuk terdeteksi dengan daya 80%.
   Efek yang diharapkan di sini lebih besar, tetapi CI tetap akan lebar.
7. **GT-nya sudah diselaraskan di tingkat tandan** (`class_mismatch = 0`), jadi
   label memang didefinisikan per tandan, bukan per tampak. Ini membuat evaluasi
   per-tandan *cocok dengan cara GT dibuat*. Itu argumen yang sah, tapi harus
   dinyatakan terbuka — bukan disembunyikan sebagai kemenangan gratis.

---

## 10. Keputusan yang saya ambil sendiri (bisa dibatalkan)

| Keputusan | Pilihan | Alasan |
|---|---|---|
| Detektor dasar | pakai ulang sel 5 (YOLO26l @1280) | bersih terhadap test, bobot + dump sudah ada, Anda bilang arsitektur bebas |
| Classifier awal | C1 (skor detektor) | cukup untuk melewati G0 tanpa training apa pun |
| Aturan utama | R4 ekspektasi ordinal, ambang dilatih di val | §5.4 |
| Urutan | G0 dulu, CPU-saja | gerbang termurah yang bisa membatalkan sisanya |

Sudah diputuskan oleh Anda (2026-08-17): proposal **disetujui**, dan pekerjaan
ini berdiri sebagai **sub-proyek sendiri** di `pipeline-pertandan/`, bukan bab
tambahan Volume 2. Karena itu penomorannya `PT-E-*` dengan log append-only
sendiri — memakai deret `V2-E-*` dari dua log berbeda mengundang tabrakan ID.

Yang masih menunggu jawaban Anda:

1. **Lokasi algoritma dedup yang sudah ada** (§5.2) — ini memblokir PT-E-002a,
   dan tanpanya angka gain penaut akan menyesatkan.
2. **Sampai mana?** Berhenti di kelas per tandan (G0–G2), atau lanjut sampai
   counting per pohon (G3) yang menantang E-007 secara langsung?
3. **YOLO26m dilatih dari nol atau tidak?** Saya sarankan tidak dulu — tapi
   kalau ini penting untuk narasi naskah, taruh sebagai ablasi setelah G0.

Konsekuensi yang belum dikerjakan: `docs/LAPORAN-AKHIR.md` §10 di repo induk
masih mendaftar "konsistensi lintas-sisi (F-003)" sebagai hal yang tidak perlu
diulang. Selama sub-proyek ini berjalan, kalimat itu perlu diberi rujukan silang
ke §3 di sini — **belum saya ubah**, karena mengedit laporan akhir Volume 2
bukan keputusan yang boleh diambil sambil lalu.

---

## 11. Reproduksi

```bash
cd /workspace/project-expertise
.venv/bin/python pipeline-pertandan/scripts/probe_penautan_953.py
```

CPU-saja, ~2 menit, deterministik (`seed=0`). Semua tabel di §4 keluar dari
berkas itu.

Jebakan yang sudah ketahuan dan sudah ditangani di dalam skrip:

| Jebakan | Akibat kalau kena |
|---|---|
| `json/*.json` punya field `split` sendiri yang **berbeda dari split kanonik pada 465 dari 953 pohon** (610/177/166 vs 716/96/141) | bocoran train↔test yang senyap. **Pakai `split_manifest.csv`** — ia identik dengan tata letak folder `images/`, nol beda |
| `split_manifest.csv` ber-BOM | kolom pertama terbaca `﻿tree_id`; buka dengan `encoding="utf-8-sig"` |
| urutan baris `labels/*.txt` ↔ `box_index` di JSON | terverifikasi selaras 504/504 sisi yang diperiksa, nol beda — pemetaan kotak→`bunch_id` gratis, tidak perlu pencocokan IoU untuk GT |
