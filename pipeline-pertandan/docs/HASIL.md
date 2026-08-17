# Hasil — pipeline per-tandan pada SawitMVC 953

Dijalankan 2026-08-17. Detektor: `yolo26l_e60_i1280_v2repro` (sel 5), dipakai
ulang apa adanya, tanpa training detektor baru.

Setiap angka di bawah terlacak ke berkas JSON di `../results/` dan ke skrip
yang menghasilkannya. Status per langkah ada di [`../STATUS.md`](../STATUS.md).

---

## 0. Ringkasan

| Gerbang | Pertanyaan | Putusan | Angka penentu (test) |
|---|---|---|---|
| validasi dump | apakah prediksi yang dipakai memang prediksi yang sama? | **LOLOS** | mAP50 0,5435 vs 0,5436 tercatat |
| **G0** | apakah menggabungkan tampak menaikkan akurasi kelas? | **LOLOS** | +4,36 pp, CI95 [+2,33; +6,25] |
| **G1** | apakah penaut bisa dibangun cukup baik? | **LOLOS** ¹ | val F1 0,6718 / ARI 0,6139 |
| **G2** | apakah pipeline utuh mendekati plafon oracle? | **LOLOS** ¹ | −1,81 pp vs toleransi −2,0 |
| **G3** | apakah menghitung pool mengalahkan penghitung lama? | **GUGUR** | macro MAE 3,66 vs 1,0542 |

¹ setelah **PT-E-008** (§12). Sebelum fitur arah putar dipakai, keduanya gugur
(F1 0,398 dan −2,36 pp). §5–§7 di bawah mendahului temuan itu dan menggambarkan
kondisi *sebelum* arah putar — sengaja tidak disunting, karena urutan penemuannya
sendiri adalah bagian dari hasilnya.

Kesimpulan satu kalimat: **mekanismenya nyata dan terukur, penautnya yang
belum cukup.** Penggabungan menaikkan akurasi kelas +4,36 pp dengan tautan
sempurna, dan tetap +4,85 pp saat pipeline berjalan tanpa GT sama sekali —
keduanya dengan CI95 yang tidak memuat nol. Yang membatasi: penaut hanya
menyatukan 29% tandan.

Dua kemungkinan perbaikan diuji dan **dua-duanya gugur**: algoritma dedup
Baseline-SawitMVC tidak bisa menggantikan penaut karena ia menghitung tanpa
mencocokkan (§9), dan memakainya sebagai rem penggabungan justru menurunkan
akurasi (§10). §10 mempersempit diagnosisnya: yang salah bukan kapan penaut
berhenti, melainkan **urutan skornya**.

**Dan §12 memperbaiki urutan itu.** Foto diambil memutari pohon searah jarum jam
— informasi dari pemilik data yang tidak pernah dipakai. Menambahkan arah
pergeseran (bukan sekadar jaraknya) menaikkan F1 penaut 0,398 → 0,649 dan
membalikkan G1 serta G2 menjadi lolos. §13 kemudian menutup jalur terakhir yang
tersisa: menaikkan ambang deteksi tidak menolong.

**Angka jujur akhir (test, konfigurasi terkunci):** akurasi kelas **0,6474** atas
seluruh 1.404 tandan GT dengan cakupan 90%; atau **0,7163** kalau dihitung hanya
atas tandan yang terdeteksi (plafon oracle 0,7360).

---

## 1. Yang dijalankan

```
SawitMVC-YOLO (vanilla, 953 pohon)
        │
        ▼  infer_skor_penuh.py           ← dump VEKTOR 4-KELAS penuh, bukan top-1
   pred_skorpenuh_{train,val,test}.npz
        │
        ├─▶ validasi_dump.py             ← wajib lolos sebelum apa pun dipakai
        │
        ├─▶ eval_pertandan.py   PT-E-001 ← plafon dengan tautan ORACLE      (G0)
        ├─▶ penaut_pertandan.py PT-E-002 ← penaut nyata, 5 varian           (G1)
        ├─▶ eval_endtoend.py    PT-E-003 ← pipeline utuh tanpa GT           (G2)
        └─▶ eval_counting.py    PT-E-004 ← counting vs pipeline lama        (G3)
```

Tiga hal yang tidak jalan di percobaan pertama dan harus diperbaiki dulu —
semuanya gagal **diam-diam**, tanpa pesan error:

| Jebakan | Gejalanya | Perbaikan |
|---|---|---|
| Tensor mentah `(1,300,6)` dipakai apa adanya | mAP50 anjlok ke **0,1342**; kotaknya benar (cocok ~0,4 px dengan dump lama) tapi banyak baris duplikat jadi positif palsu | ambil kotak dari `predict()`, ambil vektor kelas lewat forward hook |
| Satu anchor memancarkan beberapa deteksi berbeda kelas | argmax vektor ≠ kelas resmi pada **41%** deteksi | simpan `conf`/`cls` resmi DAN vektornya; satukan per anchor sebelum di-pool |
| Baseline satu-tampak diundi acak | selisih R4−R0 bergeser dari 2,38 pp ke 1,03 pp hanya karena undian berbeda | baseline dihitung sebagai **ekspektasi** atas seluruh tampak |

---

## 2. Validasi dump — gerbang kewarasan

`results/validasi_dump_test.json`

| | terukur | acuan `eval_sel5_953_rgb_test.json` |
|---|---|---|
| mAP50 | **0,5435** | 0,5436 |
| mAP50-95 | **0,2565** | 0,2565 |

Selisih 0,0001. Prediksi yang dipakai seluruh pipeline ini memang prediksi yang
sama dengan yang sudah tercatat di repo induk.

---

## 3. G0 — apakah penggabungan benar-benar menolong?

`results/pt_e_001_oracle.json` · tautan **oracle** (dari GT), jadi ini plafon
aturan agregasi, terlepas dari mutu penaut.

### 3.1 Koreksi metodologis yang mengubah jawabannya

Versi pertama membandingkan R4 (ekspektasi ordinal) langsung ke R0 (satu
tampak) dan mendapat **+4,9 pp** di val. Angka itu menyesatkan: R4 juga
menaikkan akurasi pool yang cuma punya SATU tampak (0,6917 → 0,7222), padahal
di sana tidak ada penggabungan apa pun. Sebagian "gain" ternyata **rekalibrasi
ambang kelas**, yang bisa didapat tanpa pipeline ini sama sekali.

Selisihnya karena itu dipecah dua, dan gerbang dinilai pada suku kedua saja:

```
R0cal − R0    = untung dari REKALIBRASI      (bukan klaim pipeline ini)
R4    − R0cal = untung dari PENGGABUNGAN     (klaim pipeline ini)
```

### 3.2 Hasilnya

Konfigurasi dikunci di val: `conf 0,10`, bobot `conf × √luas`, ambang ordinal
τ = (0,6; 1,7; 2,6).

| Suku | val | test |
|---|---|---|
| rekalibrasi (R0cal − R0) | +0,47 pp, CI95 [−1,26; +2,31] | −0,20 pp, CI95 [−1,90; +1,51] |
| **penggabungan, pool multi-tampak** (R4 − R0cal) | **+7,13 pp, CI95 [+5,21; +9,24]** | **+4,36 pp, CI95 [+2,33; +6,25]** |
| total (R4 − R0) | +4,99 pp, CI95 [+2,92; +7,02] | +2,42 pp, CI95 [+0,38; +4,46] |
| gaya E-016 (R2 − R0) | +1,62 pp, CI95 [+0,29; +3,01] | +1,91 pp, CI95 [+0,98; +2,92] |

Dua hal penting terbaca di sini:

1. **Suku rekalibrasi nol di kedua split** (CI memuat nol dua-duanya). Jadi
   gain-nya memang dari penggabungan, bukan dari menggeser ambang kelas.
2. **Suku penggabungan replikasi di test**, CI-nya tidak memuat nol. Ini bukan
   hasil val yang kebetulan.

Bootstrap di tingkat **pohon**, 2.000 ulangan.

### 3.3 Tangga aturan keputusan (test, semua pool)

| Aturan | Akurasi | macro-F1 | MAE ordinal | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|---|---|
| R0 satu tampak, argmax | 0,7122 | 0,6851 | 0,2766 | 0,774 | 0,378 | 0,848 | 0,709 |
| R0cal satu tampak, ambang ordinal | 0,7100 | 0,6971 | 0,2734 | 0,800 | 0,479 | 0,828 | 0,654 |
| R1 keyakinan tertinggi | 0,7273 | 0,6851 | 0,2766 | 0,774 | 0,378 | 0,848 | 0,709 |
| R2 argmax rerata softmax | 0,7313 | 0,6866 | 0,2742 | 0,774 | 0,361 | 0,856 | 0,722 |
| R3 rerata berbobot mutu | 0,7352 | 0,6938 | 0,2679 | 0,783 | 0,392 | 0,858 | 0,714 |
| **R4 ekspektasi ordinal** | **0,7360** | **0,7084** | **0,2656** | 0,783 | **0,542** | 0,834 | 0,624 |

**Risiko pra-daftar yang benar-benar terjadi.** Proposal §5.4 sudah menulis:
"kalau B1/B4 turun sementara akurasi total naik, itu bukan kemenangan". Itu
persis yang terjadi — R4 menaikkan B2 dari 0,378 ke 0,542 tetapi menurunkan B4
dari 0,709 ke 0,624. Yang menyelamatkannya: **macro-F1 juga naik** (0,6851 →
0,7084) dan MAE ordinal turun, jadi ini bukan sekadar memindahkan galat dari
satu kelas ke kelas lain. Tetap harus dilaporkan berdampingan, bukan
disembunyikan di balik akurasi agregat.

### 3.4 Menurut jumlah tampak dalam pool (test)

| Tampak | n | R0 | R0cal | R4 |
|---|---|---|---|---|
| 1 | 511 | 0,6869 | 0,6869 | 0,6869 |
| 2 | 661 | 0,7277 | 0,7284 | **0,7685** |
| 3 | 77 | 0,7403 | 0,7013 | **0,7662** |
| 4+ | 20 | 0,7375 | 0,7250 | **0,8000** |

Pool bersisi-tunggal identik di ketiga aturan — seperti seharusnya, karena di
sana tidak ada yang bisa digabung. Seluruh gain datang dari pool ≥2 tampak.

---

## 4. Perbandingan terhadap pipeline yang sudah ada

Satuan lama = kotak per citra. Satuan baru = tandan fisik per pohon. Keduanya
dari **detektor dan deteksi yang sama** (conf 0,10, test 141 pohon).

| | pipeline lama (per citra) | pipeline ini (per tandan) | selisih |
|---|---|---|---|
| **Recall** | 0,8227 | **0,9038** | **+8,11 pp** |
| **Akurasi kelas** | 0,7203 | **0,7360** (tautan sempurna) · 0,7124 (penaut nyata) | +1,57 pp · −0,79 pp |
| **Akurasi kelas**, hanya tandan yang tersatukan (n=371) | 0,6655 | **0,7143** | **+4,88 pp** |

Recall naik karena satuannya berubah: tandan yang terlewat di tiga sisi tetapi
tertangkap di satu sisi **sudah ketemu** kalau yang dihitung tandan, sementara
satuan per-citra menghukumnya tiga kali. Itu bukan trik metrik — untuk menilai
tandan pada sebuah pohon, satuan per-tandan yang benar.

Akurasi kelas naik jauh lebih kecil, dan dengan penaut nyata praktis hilang.
Penyebabnya di §5, §6, dan §10.

---

## 5. G1 — penaut: satu-satunya modul yang gagal

`results/pt_e_002_penaut.json` · diuji di atas **kotak GT**, supaya mutu
penautan terisolasi dari galat deteksi.

| Varian | val F1 | val ARI | test F1 | test ARI |
|---|---|---|---|---|
| A geometri + `kelas_sama` (kelas **GT**) | 0,4518 | 0,4385 | 0,4282 | 0,3912 |
| B + penampilan tangan (histogram HSV, ketajaman) | 0,4485 | 0,4260 | 0,4290 | 0,3937 |
| B2 tanpa fitur kelas sama sekali | 0,2557 | 0,2243 | 0,2552 | 0,2123 |
| D kelas **prediksi**, lunak | 0,3732 | 0,3228 | 0,3651 | 0,3069 |
| **E = D + embedding re-ID out-of-fold** | **0,4323** | **0,3623** | **0,3979** | **0,3292** |

Ambang G1: F1 ≥ 0,65 dan ARI ≥ 0,55. **Tidak satu pun mendekati — GUGUR.**

### 5.1 Penampilan tangan tidak menolong

B vs A: −0,0033 F1. Penjelasannya fisik dan konsisten dengan datanya: negatif
yang harus dikalahkan **semuanya berasal dari pohon yang sama**, jadi warna,
pencahayaan, dan kematangannya nyaris identik — sementara tandan yang *sama*
justru berubah rupa saat dilihat dari 90° berbeda. Histogram warna global
karena itu hampir tidak membawa daya pisah di sini.

### 5.2 Kelas GT dan kelas prediksi tidak boleh dipertukarkan

Aturan "beda kelas berarti bukan tandan yang sama" **benar secara fisik** dan
berlaku 100% di GT (`class_mismatch` = 0 untuk seluruh 9.823 tandan). Tetapi
penaut dilatih di kotak GT lalu dipakai atas kelas **prediksi**, dan di sana
aturan itu hanya benar ~77% — 23,3% tandan multi-sisi punya prediksi kelas yang
berbeda antar sisi (§3, PT-E-000).

Akibatnya terukur dan besar:

- `kelas_sama` menurunkan AUC **0,375** saat dipermutasi — lima kali lipat
  fitur berikutnya (`abs_dcx`, 0,072);
- **100,0%** pool multi-anggota jadi homogen kelasnya (467 dari 467 diperiksa);
- sehingga agregasi tidak punya apa pun untuk diperbaiki. Ini terlihat langsung
  di PT-E-003 varian B: R1 = R2 = R3 = R0 **persis** (0,7116), yang mustahil
  kecuali setiap anggota pool memang sekelas.

Membuang fiturnya juga salah — B2 anjlok ke 0,2557. Yang benar: bentuk **lunak**
atas distribusi prediksi (kemiripan Bhattacharyya + selisih ekspektasi ordinal),
dipakai identik saat latih dan inferensi. Itu varian D, dan ia memulihkan
sebagian besar jarak yang hilang (0,2557 → 0,3732) **sambil** mengizinkan pool
berisi tampak berbeda kelas.

### 5.3 Embedding re-ID: menghafal, tapi tidak kosong

AUC memakai cosine embedding saja, tanpa fitur lain:

| Split | AUC |
|---|---|
| train | **1,0000** |
| val | 0,7564 |
| test | 0,7195 |

Model menghafal identitas split train sempurna, tetapi tetap membawa daya pisah
nyata di pohon yang belum pernah dilihat (0,72–0,76 ≫ 0,50).

Melatih penaut dengan embedding yang sudah menghafal pohon-pohon itu
**meruntuhkan** hasilnya: AUC val 0,578, F1 test 0,1801. Perbaikannya standar —
dua model re-ID dilatih, masing-masing menahan separuh pohon train, sehingga
pasangan train memakai embedding yang genuinely unseen (14.041 dari 18.540
potongan). Dengan itu varian E naik ke **F1 test 0,3979**, melewati D
(+0,033 F1, +0,022 ARI).

**Jadi ide inti proposal — embedding re-ID dari graf identitas — memang
menambah sinyal nyata. Hanya belum cukup untuk melewati gerbang.**

## 6. G2 — pipeline utuh, tanpa GT sama sekali

`results/pt_e_003_endtoend.json` · penaut E, ambang 0,25, kelas GT **tidak**
dipakai.

| Test | R0 | R0cal | R1 | R2 | R3 | **R4** |
|---|---|---|---|---|---|---|
| semua pool (n=1269) | 0,6999 | 0,6981 | 0,7053 | 0,7053 | 0,7053 | **0,7124** |
| pool ≥2 tampak (n=371) | — | 0,6655 | — | — | — | **0,7143** |
| pembanding: tautan oracle | 0,7122 | — | — | — | — | 0,7360 |

**Dua hal yang berubah dibanding varian B (yang memakai kelas GT):**

1. **R1, R2, R3 kini berbeda dari R0.** Dengan varian B ketiganya identik
   (0,7116 persis) karena setiap pool homogen kelasnya — tidak ada yang bisa
   diperbaiki. Blokade itu hilang.
2. **Penggabungan bekerja end-to-end**: pada pool ≥2 tampak, R4 mengalahkan
   R0cal **+4,85 pp, CI95 [+2,03; +7,81]** (val: +3,15 pp, CI [−0,03; +6,21]).
   CI test tidak memuat nol — jadi mekanismenya bukan artefak tautan oracle.

**Yang membatasi: penaut terlalu pelit menggabung di ruang deteksi.**

| | val | test |
|---|---|---|
| presisi pasangan | 0,4272 | 0,3342 |
| **recall pasangan** | **0,1760** | **0,1200** |
| F1 | 0,2493 | 0,1766 |
| pool yang seluruhnya positif palsu | 38,7% | 39,9% |

Akibatnya hanya **371 dari 1.269** tandan (29%) punya pool ≥2 tampak. Sisanya
71% tidak tersentuh agregasi sama sekali, jadi gain +4,85 pp itu terdilusi
menjadi +1,27 pp pada angka agregat (CI95 [−0,49; +3,03], memuat nol).

**Putusan G2: GUGUR** — 0,7124 vs oracle 0,736 = −2,36 pp, di luar toleransi
−2,0 pp.

> **Kaveat yang harus ikut dibaca.** Ambang penaut 0,25 diwarisi dari sapuan
> di atas **kotak GT**, lalu dipakai di atas **deteksi** yang distribusi
> skornya berbeda (banyak positif palsu). Itu penyebab langsung recall
> pasangan 0,120. Menyetel ulang ambangnya di val-deteksi adalah perbaikan
> yang sah menurut protokol (val memang split pemilihan) dan belum dijalankan;
> opsinya sudah ada di `eval_endtoend.py --sapu-ambang`.

---

## 7. G3 — counting per pohon

`results/pt_e_004_counting.json` · seluruh penghitung memakai **deteksi yang
sama**, dipas di train (716 pohon), dievaluasi di test.

| Penghitung | macro MAE | class ±1 | tree ±1 | bias total/pohon |
|---|---|---|---|---|
| C1 naif (hitung deteksi apa adanya) | 4,3582 | 0,3741 | 0,0142 | +16,13 |
| C2 bagi k global 1,8905 | 1,6696 | 0,5000 | 0,0780 | +3,84 |
| C3 bagi k per kelas (dipas di train) | 1,1854 | 0,5709 | 0,1064 | −0,08 |
| **C4 hitung pool** ← pipeline ini | **3,3422** | 0,4273 | 0,0426 | **+11,96** |
| **C5 Ridge + F_all** ← jalur repo | **1,0542** | 0,6064 | 0,1489 | −0,04 |

**Putusan G3: GUGUR, dan telak** — 3,3422 vs 1,0542.

Penyebabnya sama persis dengan §6: penaut yang tidak menggabung membuat satu
tandan pecah menjadi beberapa pool, jadi jumlah pool **melebihi** jumlah tandan
sebenarnya sekitar **12 per pohon**. Bandingkan dengan C1 (bias +16,13):
penautan hanya memangkas seperempat kelebihan hitungnya.

Catatan kewarasan: C5 mencapai macro MAE 1,0542, dekat dengan angka repo induk
untuk YOLO26m (1,036) — jadi implementasi F_all di sini wajar, bukan salah pas.

Ini juga **mengonfirmasi ulang E-007** lewat jalur berbeda: penautan eksplisit
tetap kalah dari koreksi statistik untuk tugas counting. Bedanya, sekarang
sebabnya diketahui dan terukur (recall pasangan 0,12), bukan sekadar tercatat
sebagai kegagalan.

---

## 8. Kesimpulan

**Mekanismenya terbukti; penautnya belum cukup.** Satu kalimat itu didukung
tiga pengukuran yang saling bebas:

| Bukti | Angka |
|---|---|
| Menemukan tandan naik hanya karena satuannya berubah | 0,8227 → **0,9038** (+8,11 pp) |
| Penggabungan menaikkan akurasi kelas, tautan sempurna | **+4,36 pp**, CI95 [+2,33; +6,25] |
| Penggabungan tetap bekerja **tanpa GT sama sekali** | **+4,85 pp**, CI95 [+2,03; +7,81] |

Dan satu kegagalan tunggal yang menjelaskan semua angka agregat yang datar:
penaut hanya menyatukan **29%** tandan (recall pasangan 0,12), sehingga 71%
tandan tidak pernah tersentuh agregasi.

Aritmetika kasarnya: kalau penaut bisa menyatukan 8 dari 10 tandan alih-alih
3 dari 10, gain +4,85 pp yang sudah terbukti itu akan terasa pada ~80% tandan
alih-alih 29% — akurasi kelas naik dari 72,0% ke sekitar 75–76%, dan galat
counting turun mendekati C3/C5.

---

## 9. Penghitung Baseline-SawitMVC — apa yang sebenarnya diukur

`results/pt_e_006_baseline_counting.json` · repo
[`ULM-SawitMVC/Baseline-SawitMVC`](https://github.com/ULM-SawitMVC/Baseline-SawitMVC)

Repo itu mencatat **Acc±1 87,62% / macro MAE 0,3746** untuk algoritma M01 pada
953 pohon — jauh lebih baik daripada apa pun di §7. Algoritma yang sama
dijalankan di sini dengan tiga masukan berbeda:

| Masukan (141 pohon test) | macro MAE | class ±1 | tree ±1 |
|---|---|---|---|
| kotak GT | **0,3404** | 0,9592 | 0,8723 |
| deteksi `y26mv2` (detektor repo itu sendiri) | **1,1826** | 0,7057 | 0,2411 |
| deteksi YOLO26l @conf 0,10 | 1,8298 | 0,5745 | 0,1206 |

Angka acuan repo tereproduksi **persis sampai empat desimal** di kotak GT 953
pohon (0,3746 / 0,8762 / 1,3305). Jadi 87,62% itu **angka kotak GT**, bukan
end-to-end — sah dan benar, tetapi tidak sebanding dengan angka end-to-end mana
pun. README utama repo itu sendiri sudah memisahkan keduanya (98,05% dengan
deteksi GT vs 77,48% dengan YOLO26m); yang belum, halaman `algorithms/README.md`
tidak menyebut masukannya.

Pada titik kerja yang sama-sama wajar, semua jalur mendarat berdekatan:
M01 **1,1826** · k-per-kelas 1,1854 · Ridge+F_all 1,0542. Selisih 0,34 → 1,18
adalah **efek detektor murni**.

**Konsekuensi untuk sub-proyek ini:** M01–M05 tidak bisa menjadi modul L.
Diperiksa langsung di seluruh repo — nol hasil untuk `linear_sum_assignment`,
`hungarian`, `iou`, `bunch_id`. Kelima algoritma menjawab "ada berapa", bukan
"kotak mana milik tandan mana".

---

## 10. Penghitung itu sebagai REM — dan diagnosis ulang yang mengikat

`results/pt_e_007_rem_hitung.json`

Penaut terlalu pelit (§6). M01 tahu hal yang tidak diketahui penaut: berapa
banyak tandan di pohon itu. Gagasannya, pakai angka itu sebagai target — gabung
terus dari skor tertinggi sampai jumlah kelompok turun ke target.

| Split | Mode | Kelompok / asli | Tersatukan | R0 | R4 |
|---|---|---|---|---|---|
| val | A ambang tetap | 2.141 / 992 (2,16×) | 32,9% | 0,6973 | **0,7180** |
| val | B rem M01 | 1.490 / 992 (1,50×) | 62,1% | 0,6382 | 0,6551 |
| val | C rem **cacah sempurna** | 1.104 / 992 (1,11×) | 76,5% | 0,5974 | 0,6101 |
| test | A ambang tetap | 3.095 / 1.404 (2,20×) | 29,2% | 0,6998 | **0,7139** |
| test | B rem M01 | 2.098 / 1.404 (1,49×) | 59,3% | 0,6577 | 0,6872 |
| test | C rem **cacah sempurna** | 1.582 / 1.404 (1,13×) | 75,9% | 0,6132 | 0,6454 |

Remnya bekerja persis seperti dirancang — porsi tersatukan 29% → 76%, rasio
kelompok mendekati 1,0. **Akurasi tetap turun monoton di kedua split.**

Mode C adalah kuncinya: cacahnya diambil dari GT, jadi **sempurna benar**, dan
akurasinya justru turun paling dalam. Maka:

> Masalah penaut **bukan berhenti terlalu cepat, melainkan urutan skornya
> salah.** Kalau peringkat pasangannya benar, memaksa menggabung lebih banyak —
> dari skor tertinggi ke bawah — akan menggabung yang benar lebih dulu dan
> akurasi naik. Yang terjadi sebaliknya: pasangan berskor tertinggi yang belum
> tergabung ternyata mayoritas SALAH.

**Dua dugaan di §6 karena itu dibatalkan:** recall pasangan 0,12 **bukan**
artefak ambang warisan, dan menyetel ambang di ruang deteksi **tidak** berguna.
Ambang berapa pun tidak menolong kalau peringkatnya keliru.

**Yang bertahan:** aturan agregasi terus memberi nilai tambah di semua mode, dan
justru menguat saat pool makin kotor (test R4−R0: +1,44 → +2,96 → +3,24 pp).
Modul A sehat; masukannya yang rusak.

---

## 11. Yang paling berdampak berikutnya

Urutan ini sudah dua kali berubah. Yang dulu di puncak — "ukur algoritma dedup
yang sudah ada" (§9–10) dan "naikkan mutu peringkat skor" (§12) — keduanya sudah
dikerjakan dan terjawab. Yang tersisa ada di §13.

---

## 12. PT-E-008 — arah putar pengambilan foto

`results/harapan_geser.json` · `results/pt_e_002_penaut.json`

Foto diambil **memutari pohon searah jarum jam**. Seluruh fitur geometri
sebelumnya memakai `abs_dcx` — nilai mutlak — sehingga arah pergeseran dibuang.

| Offset sisi (4-sisi) | Pasangan BENAR | Konsistensi | Pasangan SALAH |
|---|---|---|---|
| +1 | **+0,241** (σ 0,116) | **98,6% ke kanan** | −0,024 (54,9% kiri, σ 0,213) |
| +3 | **−0,260** (σ 0,109) | **99,7% ke kiri** | +0,019 (46,6%) |

Konstanta yang dipas di train memperlihatkan tanda tangan putaran melingkar pada
pohon 8-sisi — naik ke puncak lalu berbalik tanda, simetris:
`+0,120 · +0,255 · +0,347 · +0,308 · −0,355 · −0,261 · −0,152`.

| Penaut (kotak GT) | val F1 | val ARI | test F1 | test ARI |
|---|---|---|---|---|
| E tanpa arah | 0,4323 | 0,3623 | 0,3979 | 0,3292 |
| **E dengan arah** | **0,6718** | **0,6139** | **0,6486** | **0,5904** |

**G1 LOLOS** dan **G2 LOLOS** (−1,81 pp). Bias jumlah kelompok −0,96 → −0,22.

> **Jebakan yang sudah ditutup.** Tabel konstanta hidup sebagai global modul yang
> semula hanya diisi di `main()` skrip penaut. Skrip end-to-end meng-import-nya
> **kosong**, sehingga fitur arah tidak aktif dan hasilnya diam-diam kembali ke
> fitur lama tanpa pesan galat. Gejalanya khas: F1 end-to-end 0,1766 → 0,1761,
> identik sampai tiga desimal, padahal di kotak GT melonjak 0,3651 → 0,6486.
> Perbaikan akar: di-cache ke berkas, dimuat otomatis saat import, plus
> `RuntimeWarning` kalau kosong.

**Catatan jujur.** Lompatan ini bukan hasil pemodelan. Warna, tekstur, embedding
terlatih, out-of-fold, dan rem cacah semuanya mentok atau memperburuk. Yang
membuka jalan adalah satu kalimat pemilik data tentang **cara foto diambil** —
informasi yang seharusnya ditanyakan sebelum eksperimen pertama.

---

## 13. PT-E-009 — sapuan ambang deteksi, dan jalur terakhir yang tertutup

`results/pt_e_009_sapu_conf.json`

Setelah penaut diperbaiki, 40% kelompok di ruang deteksi masih seluruhnya positif
palsu. Dugaan: naikkan `conf`. Diuji 6 nilai × 3 ambang penaut, semua dipilih di
val, `tau` disetel ulang per conf.

| conf | Cakupan | F1 penaut | R4 tandan terdeteksi | **R4 seluruh tandan** |
|---|---|---|---|---|
| **0,10** | 90% | 0,251 | 0,7146 | **0,6411** |
| 0,20 | 83% | 0,284 | 0,7242 | 0,6008 |
| 0,30 | 74% | 0,353 | 0,7425 | 0,5464 |
| 0,40 | 62% | 0,406 | 0,7423 | 0,4587 |
| 0,50 | 46% | 0,397 | 0,7636 | 0,3549 |
| 0,60 | 24% | 0,538 | 0,8107 | 0,1986 |

**DIPALSUKAN.** `conf = 0,10` sudah optimal; menaikkannya memperburuk monoton.

> **Jebakan penyebut — ketiga kalinya di proyek ini.** Versi pertama sapuan
> menilai akurasi hanya pada tandan yang TERDETEKSI, dan memilih conf 0,60 dengan
> R4 0,8107 — kelihatan seperti lompatan besar. Palsu: penyebutnya menyusut dari
> 890 tandan ke 243. Menaikkan conf membuang tandan sulit lebih dulu, jadi soal
> ujiannya yang jadi mudah. Dengan penyebut tetap (seluruh tandan GT, tak
> terdeteksi = salah), jawabannya berbalik.
>
> Tiga kali kesalahan penyebut hampir menghasilkan klaim palsu di proyek ini
> (§3.1 rekalibrasi, §4 recall, dan di sini). **Periksa penyebut setiap kali dua
> angka dibandingkan.**

Dugaannya separuh benar: membuang positif palsu memang melipatgandakan F1 penaut
(0,25 → 0,54) dan menaikkan akurasi pada tandan yang tersisa (0,715 → 0,811).
Salah: kurva presisi-recall detektor terlalu curam — tiap sampah yang dibuang
menyeret beberapa tandan asli. **Perbaikannya harus di detektornya, bukan di
ambangnya.** Ini juga menutup jalur counting: cacah pool akan terus mewarisi
seluruh positif palsu detektor, dan ambang bukan alatnya.

---

## 14. Yang tersisa

> **KOREKSI (PT-E-011).** Butir 1 versi pertama berbunyi "detektor adalah
> satu-satunya yang membatasi". Itu **dipalsukan**: presisi deteksi 0,584 (953)
> vs 0,639 (352) — beda 5,5 pp saja — dan recall 953 justru lebih baik
> (0,823 vs 0,739). Kedua detektor setara. Yang berbeda adalah **kepadatan
> adegan**: ~235 pasangan lintas-sisi per pohon di 953 dengan prevalensi benar
> ~4%, lawan ~28 dengan ~21% di 352. Mengganti backbone tidak mengubah itu.

1. **Prior yang memangkas ruang kandidat.** Inilah obat untuk masalah
   kombinatorik, dan justru itu yang menjelaskan kenapa arah putar (§12) memberi
   lompatan terbesar: ia tidak memperbaiki diskriminasi per pasangan, ia
   memangkas kandidatnya. Kandidat berikutnya: **depth** di korpus 352. Kaveat
   jujur: E-007 Volume 1 sudah memalsukan penautan berbasis depth — tetapi tanpa
   prior arah dan tanpa penilai terlatih.
2. **PT-E-005 — classifier multi-tampak (C3).** Bentuk paling setia terhadap
   sketsa asli; sah dikerjakan karena G0 lolos. Tidak memperbaiki penaut, tetapi
   memperbesar nilai dari tandan yang berhasil disatukan.
3. **Counting: pakai jalur yang sudah ada.** §7, §9, dan §13 sepakat — Ridge+F_all
   (1,0542) dan M01 terkalibrasi-ulang (1,18) sudah mendekati batas detektor.
   Jalur pool tidak akan mengalahkannya, dan keunggulan pipeline ini memang ada
   di klasifikasi per tandan, bukan di cacah.
