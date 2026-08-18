# EKSPERIMEN — `pipeline-pertandan`

Log **append-only**. Satu entri = satu hipotesis falsifiable. Jangan sunting
entri lama; kalau sebuah entri keliru, tulis entri baru yang mengoreksinya dan
sebutkan nomornya. Hasil negatif dicatat dengan bobot yang sama dengan hasil
positif.

Penomoran `PT-E-*` — terpisah dari deret `V2-E-*` di `../experiments/EKSPERIMEN.md`.
Alasannya di [`CLAUDE.md`](CLAUDE.md) §1.

## Template entri

```markdown
## PT-E-0NN — <judul> (YYYY-MM-DD)

**Hipotesis** — <satu kalimat, falsifiable>

**Yang memalsukan** — <ditulis SEBELUM melihat hasil>

**Data & split** — <dataset, split, jumlah pohon/tandan/kotak>

**Cara** — <perintah persis + skrip>

**Hasil** — <tabel angka; setiap angka menyebut split>

**Putusan** — DIKONFIRMASI / DIPALSUKAN / TIDAK KONKLUSIF

**Sumber** — <skrip, JSON, log>
```

---

## Rencana (belum dijalankan)

Diambil dari [`docs/PROPOSAL.md`](docs/PROPOSAL.md) §8. Baris ini **bukan**
entri — ia dihapus dari daftar dan diganti entri sungguhan begitu eksperimennya
selesai.

| ID | Isi | Gerbang | Biaya | Status |
|---|---|---|---|---|
| `PT-E-001` | Rangka evaluasi per-tandan + plafon oracle (R0–R4 di atas tautan GT, CI bootstrap-pohon) | **G0** | CPU ~1 jam | belum |
| `PT-E-002a` | Ukur algoritma dedup yang sudah ada pada metrik penautan langsung | **G1** | CPU ~1 jam | **terblokir** — menunggu lokasi kodenya |
| `PT-E-002b` | Penaut re-ID (hanya kalau 002a di bawah G1) | **G1** | GPU 2–4 jam | belum |
| `PT-E-003` | Tangga aturan keputusan R0–R5 di atas penaut nyata | **G2** | CPU (+1 jam GPU untuk R5) | belum |
| `PT-E-004` | Counting dari jumlah pool vs k=1,8905 vs Ridge+F_all | G3 | CPU | belum |
| `PT-E-005` | (opsional) classifier multi-tampak C3 | — | GPU ~4 jam | belum |

**Urutannya mengikat**: PT-E-001 lebih dulu, dan tidak ada training GPU sebelum
G0 punya angka.

---

## PT-E-000 — Probe kelayakan (2026-08-17)

Bukan eksperimen berhipotesis; pengukuran dasar yang menjadi landasan proposal.
Dicatat di sini supaya angkanya punya nomor yang bisa dirujuk.

**Data & split** — SawitMVC-YOLO vanilla, 953 pohon (716/96/141 kanonik),
18.540 kotak, 9.823 tandan unik. Verifikasi integritas: seluruh 8.945 berkas
dataset byte-identik dengan hash yang dicatat HuggingFace saat unduh (nol beda).

**Cara** —
```bash
cd /workspace/project-expertise
.venv/bin/python pipeline-pertandan/scripts/probe_penautan_953.py
```

**Hasil** —

| Blok | Temuan |
|---|---|
| A struktur GT | 7.328 dari 9.823 tandan (74,6%) multi-sisi; **85,5% di antaranya tepat 2 sisi**; plafon keterlihatan 3 (4-sisi) dan 6 (8-sisi); `class_mismatch` = 0 untuk seluruh 9.823 tandan |
| B integritas split | field `split` di JSON berbeda dari split kanonik pada **465 dari 953 pohon**; `split_manifest.csv` nol beda terhadap tata letak folder |
| C biaya galat-gabung | penggabungan keliru tidak merusak kelas pada **37,74%** kasus (sisanya, 62,26%, merusak) |
| D penaut geometri-saja | val ROC-AUC 0,9301, F1 pasangan 0,4850; penugasan global berkendala val F1 0,4518 / ARI 0,4385; **dikunci ke test: F1 0,4282 / ARI 0,3912** |
| E selisih recall | conf 0,25: recall per-kemunculan 63,36% → per-tandan **77,85%** (**+14,49 pp**); 23,3% pool yang terdeteksi ≥2 sisi punya prediksi kelas yang tidak sepakat |

**Putusan** — Tidak ada. Ini pengukuran, bukan uji hipotesis. Yang diputuskan
olehnya: (a) satuan per-tandan memberi keuntungan besar tanpa model baru;
(b) geometri saja tidak cukup untuk menaut; (c) voting mayoritas tidak bisa
dipakai sebagai aturan agregasi.

**Kaveat** — Angka D adalah **batas bawah** penautan (geometri tanpa piksel),
bukan kemampuan algoritma dedup yang sudah ada. Lihat `CLAUDE.md` §4.

**Sumber** — `scripts/probe_penautan_953.py` · `results/probe_penautan_953.json`

---

## PT-E-001 — Plafon penggabungan dengan tautan oracle (2026-08-17)

**Hipotesis** — Dengan tautan lintas-sisi yang sempurna, menggabungkan tampak
dari tandan fisik yang sama menaikkan akurasi kelas per tandan dibanding
melihat satu foto saja.

**Yang memalsukan** (ditulis sebelum melihat hasil) — suku PENGGABUNGAN
(R4 vs R0cal pada pool multi-tampak) < +2,0 pp di val, atau CI95 bootstrap-pohon
memuat nol.

**Data & split** — SawitMVC-YOLO vanilla, val 96 / test 141 pohon. Detektor
`yolo26l_e60_i1280_v2repro` (sel 5). Dump prediksi tervalidasi lebih dulu:
mAP50 test **0,5435** vs 0,5436 tercatat (selisih 0,0001).

**Cara** —
```bash
.venv/bin/python pipeline-pertandan/scripts/infer_skor_penuh.py --split train val test
.venv/bin/python pipeline-pertandan/scripts/validasi_dump.py --split test
.venv/bin/python pipeline-pertandan/scripts/eval_pertandan.py
```
Terkunci di val: conf 0,10 · bobot `conf × √luas` · τ = (0,6; 1,7; 2,6).

**Koreksi rancangan yang mengubah jawaban** — versi pertama membandingkan R4
langsung ke R0 dan mendapat +4,9 pp di val. Angka itu menyesatkan: R4 juga
menaikkan akurasi pool bersisi-TUNGGAL (0,6917 → 0,7222), tempat tidak ada
penggabungan sama sekali. Sebagian gain ternyata rekalibrasi ambang kelas.
Selisih karena itu dipecah, dan gerbang dinilai pada suku penggabungan saja.
Baseline satu-tampak juga diubah dari undian acak menjadi EKSPEKTASI atas
seluruh tampak — satu undian menggeser selisih dari 2,38 pp ke 1,03 pp, derau
sebesar efeknya sendiri.

**Hasil** —

| Suku | val | test |
|---|---|---|
| rekalibrasi (R0cal−R0) | +0,47 pp, CI95 [−1,26; +2,31] | −0,20 pp, CI95 [−1,90; +1,51] |
| **penggabungan, pool ≥2 tampak** | **+7,13 pp, CI95 [+5,21; +9,24]** | **+4,36 pp, CI95 [+2,33; +6,25]** |
| total (R4−R0) | +4,99 pp, CI95 [+2,92; +7,02] | +2,42 pp, CI95 [+0,38; +4,46] |
| gaya E-016 (R2−R0) | +1,62 pp | +1,91 pp |

Tangga aturan di test (semua pool): R0 0,7122 · R0cal 0,7100 · R1 0,7273 ·
R2 0,7313 · R3 0,7352 · **R4 0,7360** (macro-F1 0,7084, MAE ordinal 0,2656).

Menurut jumlah tampak (test): 1 tampak 0,6869 (ketiga aturan identik, seperti
seharusnya) · 2 tampak 0,7277 → **0,7685** · 3 tampak 0,7403 → **0,7662** ·
4+ 0,7375 → **0,8000**.

**Putusan** — **DIKONFIRMASI.** Gerbang G0 LOLOS. Suku rekalibrasi nol di kedua
split; suku penggabungan replikasi dari val ke test dengan CI tidak memuat nol.

**Kaveat wajib** — risiko yang sudah dipra-daftar di proposal §5.4 memang
terjadi: R4 menaikkan recall B2 (0,378 → 0,542) tetapi menurunkan B4
(0,709 → 0,624). Yang menyelamatkannya, macro-F1 ikut naik (0,6851 → 0,7084)
dan MAE ordinal turun — jadi bukan sekadar memindahkan galat antar kelas.

**Sumber** — `scripts/eval_pertandan.py` · `results/pt_e_001_oracle.json` ·
`results/validasi_dump_test.json`

---

## PT-E-002 — Penaut lintas-sisi nyata (2026-08-17)

**Hipotesis** — Penaut berbasis penampilan terlatih mencapai F1 pasangan val
≥ 0,65 dan ARI ≥ 0,55, cukup untuk menyalurkan gain PT-E-001 ke pipeline utuh.

**Yang memalsukan** — F1 val < 0,65 atau ARI val < 0,55.

**Data & split** — kotak GT (bukan deteksi), supaya mutu penautan terisolasi
dari galat deteksi. Train 716 pohon → 121.891 pasangan lintas-sisi, 8.034
positif (6,6%).

**Hasil** —

| Varian | val F1 | val ARI | test F1 | test ARI |
|---|---|---|---|---|
| A geometri + `kelas_sama` (kelas **GT**) | 0,4518 | 0,4385 | 0,4282 | 0,3912 |
| B + penampilan tangan (histogram HSV dll) | 0,4485 | 0,4260 | 0,4290 | 0,3937 |
| B2 tanpa fitur kelas sama sekali | 0,2557 | 0,2243 | 0,2552 | 0,2123 |
| D kelas **prediksi**, lunak | 0,3732 | 0,3228 | 0,3651 | 0,3069 |
| **E = D + embedding re-ID out-of-fold** | **0,4323** | **0,3623** | **0,3979** | **0,3292** |

**Putusan** — **DIPALSUKAN.** Gerbang G1 GUGUR di semua varian; yang tertinggi
dan sah (E) hanya 0,4323 lawan ambang 0,65.

**Tiga temuan yang mengikat pekerjaan berikutnya:**

1. **Penampilan tangan tidak menolong** (B vs A: −0,0033 F1). Negatif yang harus
   dikalahkan semuanya dari pohon yang sama — warna, pencahayaan, dan kematangan
   nyaris identik, sementara tandan yang sama berubah rupa dari 90° berbeda.

2. **Kelas GT dan kelas prediksi tidak boleh dipertukarkan.** Aturan "beda kelas
   berarti bukan tandan yang sama" BENAR secara fisik dan berlaku 100% di GT
   (`class_mismatch` = 0). Tetapi penaut dilatih di kotak GT lalu dipakai atas
   kelas PREDIKSI, tempat aturan itu cuma benar ~77% (23,3% tandan multi-sisi
   punya prediksi berbeda antar sisi). Akibatnya: `kelas_sama` menurunkan AUC
   **0,375** saat dipermutasi — lima kali lipat fitur berikutnya — dan **100,0%**
   pool multi-anggota jadi homogen kelasnya, sehingga agregasi tidak punya apa
   pun untuk diperbaiki (terlihat sebagai R1 = R2 = R3 = R0 persis di PT-E-003
   varian B). Membuang fiturnya juga salah: B2 anjlok ke 0,2557. Yang benar
   adalah bentuk LUNAK atas distribusi prediksi (varian D).

3. **Embedding re-ID menghafal, tapi tidak kosong.** AUC cosine-saja: train
   **1,0000** / val 0,7564 / test 0,7195. Melatih penaut memakai embedding yang
   sudah menghafal pohon-pohon itu meruntuhkan AUC val ke 0,578 dan F1 test ke
   0,1801. Dengan embedding **out-of-fold** (dua model, masing-masing menahan
   separuh pohon train), varian E naik ke F1 test 0,3979 — melewati D
   (+0,033 F1, +0,022 ARI). Jadi ide inti proposal menambah sinyal nyata,
   hanya belum cukup.

**Sumber** — `scripts/penaut_pertandan.py` · `scripts/reid_pertandan.py` ·
`results/pt_e_002_penaut.json` · `results/pt_e_002_penaut_kontaminasi_fold.json`
(versi terkontaminasi, disimpan sebagai bukti butir 3)

---

## PT-E-003 — Pipeline utuh tanpa GT (2026-08-17)

**Hipotesis** — Akurasi kelas per tandan dengan penaut nyata >= akurasi dengan
tautan oracle - 2,0 pp.

**Data & split** — val 96 / test 141 pohon, deteksi conf>=0,10, penaut varian E
(ambang 0,25, kelas GT tidak dipakai).

**Hasil (test)** — R0 0,6999 · R0cal 0,6981 · R1 0,7053 · R2 0,7053 ·
R3 0,7053 · **R4 0,7124**. Oracle R4 0,7360.

Pada pool >=2 tampak (n=371): R0cal 0,6655 -> R4 **0,7143**, selisih
**+4,85 pp, CI95 [+2,03; +7,81]** (val +3,15 pp, CI [-0,03; +6,21]).

Mutu penautan di ruang DETEKSI: presisi 0,3342 · **recall 0,1200** · F1 0,1766 ·
ARI 0,1549. 39,9% pool seluruhnya positif palsu.

**Putusan** — **DIPALSUKAN.** G2 GUGUR: 0,7124 vs 0,7360 = -2,36 pp.

**Yang penting justru bukan putusannya:**

1. R1, R2, R3 akhirnya **berbeda** dari R0. Dengan varian B (kelas GT) ketiganya
   identik 0,7116 persis — bukti langsung bahwa pool homogen kelasnya. Blokade
   itu hilang setelah kelas prediksi dipakai secara lunak.
2. Penggabungan **bekerja end-to-end tanpa GT sama sekali**, CI95 tidak memuat
   nol. Jadi +4,36 pp di PT-E-001 bukan artefak tautan oracle.
3. Yang datar adalah angka AGREGAT, karena penaut hanya menyatukan 371 dari
   1.269 tandan (29%). Sisanya 71% tidak tersentuh agregasi.

**Kaveat** — ambang 0,25 diwarisi dari sapuan di atas kotak GT lalu dipakai di
atas deteksi yang distribusi skornya berbeda. Menyetelnya ulang di val-deteksi
sah menurut protokol dan belum dijalankan (`--sapu-ambang`).

**Sumber** — `scripts/eval_endtoend.py` · `results/pt_e_003_endtoend.json` ·
`results/pt_e_003_endtoend_varianB_kelasGT.json` (pembanding varian bocor)

---

## PT-E-004 — Counting per pohon vs penghitung yang sudah ada (2026-08-17)

**Hipotesis** — Menghitung POOL mengalahkan penaksiran statistik dari hitungan
deteksi mentah.

**Data & split** — deteksi yang sama untuk kelima penghitung; C3 dan C5 dipas di
train (716 pohon), dievaluasi di test (141 pohon).

**Hasil (test)** —

| Penghitung | macro MAE | class +-1 | tree +-1 | bias/pohon |
|---|---|---|---|---|
| C1 naif | 4,3582 | 0,3741 | 0,0142 | +16,13 |
| C2 k global 1,8905 | 1,6696 | 0,5000 | 0,0780 | +3,84 |
| C3 k per kelas | 1,1854 | 0,5709 | 0,1064 | -0,08 |
| **C4 hitung pool** | **3,3422** | 0,4273 | 0,0426 | **+11,96** |
| **C5 Ridge + F_all** | **1,0542** | 0,6064 | 0,1489 | -0,04 |

**Putusan** — **DIPALSUKAN.** G3 GUGUR telak: 3,3422 vs 1,0542.

**Sebabnya sama dengan PT-E-003**: penaut yang tidak menggabung memecah satu
tandan menjadi beberapa pool, jadi jumlah pool melebihi jumlah tandan sekitar
12 per pohon. Penautan hanya memangkas seperempat kelebihan hitung C1
(+16,13 -> +11,96).

**Kewarasan** — C5 mencapai 1,0542, dekat angka repo induk untuk YOLO26m
(1,036), jadi implementasi F_all di sini wajar.

**Hubungan dengan E-007** — ini mengonfirmasi ulang E-007 lewat jalur berbeda:
penautan eksplisit tetap kalah dari koreksi statistik untuk counting. Bedanya,
sekarang sebabnya terukur (recall pasangan 0,12), bukan sekadar tercatat
sebagai kegagalan.

**Sumber** — `scripts/eval_counting.py` · `results/pt_e_004_counting.json`

---

## PT-E-006 — Penghitung Baseline-SawitMVC (M01-M05) di deteksi nyata (2026-08-17)

**Latar** — User menunjuk `github.com/ULM-SawitMVC/Baseline-SawitMVC` sebagai
lokasi algoritma dedup miliknya. Repo itu memuat lima algoritma heuristik dengan
angka tercatat **Acc+-1 87,62% / macro MAE 0,3746 pada 953 pohon** — jauh lebih
baik daripada apa pun di PT-E-004 (terbaik Ridge+F_all, macro MAE 1,0542).

**Hipotesis** — Selisih itu berasal dari MASUKAN, bukan dari algoritmanya:
angka 0,375 diukur di kotak GT, bukan di deteksi detektor.

**Cara** — algoritma yang sama, dua masukan, korpus dan split disebut eksplisit.

**Hasil** —

| Masukan | Himpunan | Algoritma | macro MAE | class ±1 | tree ±1 | total MAE |
|---|---|---|---|---|---|---|
| kotak GT | 953 pohon | **M01** | **0,3746** | 0,9541 | **0,8762** | **1,3305** |
| kotak GT | 141 test | M01 | 0,3404 | 0,9592 | 0,8723 | 1,1915 |
| deteksi YOLO26l | 141 test | M01 | **1,8298** | 0,5745 | 0,1206 | 5,1348 |
| deteksi YOLO26l | 141 test | M05 (terbaik dari lima) | 1,8085 | 0,5975 | 0,1489 | 4,9929 |
| deteksi YOLO26l | 141 test | naif tanpa dedup | 4,3582 | 0,3741 | 0,0142 | 16,1560 |

**Putusan** — **DIKONFIRMASI.** Angka acuan repo tereproduksi **persis sampai
empat desimal** (0,3746 / 0,8762 / 1,3305) di kotak GT 953 pohon. Di deteksi
nyata, algoritma yang sama turun ke macro MAE 1,8298.

**Perbandingan yang adil, seluruhnya di deteksi yang sama (test 141 pohon):**

| Penghitung | macro MAE |
|---|---|
| Ridge + F_all (PT-E-004 C5) | **1,0542** |
| k per kelas dipas di train (C3) | 1,1854 |
| k global 1,8905 (C2) | 1,6696 |
| **M05 Baseline-SawitMVC** | 1,8085 |
| **M01 Baseline-SawitMVC (juara di GT)** | 1,8298 |
| hitung pool (C4, pipeline ini) | 3,3422 |
| naif (C1) | 4,3582 |

**Kenapa M01 turun** — konstantanya (`BASE_FACTORS` 1,986 / 1,786 / 1,795 /
1,655 dan `dup_rate` adaptif) diturunkan dari rasio duplikasi **kotak GT**.
Detektor melewatkan ~18% kemunculan, jadi rasio duplikasi nyatanya berbeda dan
konstanta itu jadi salah kalibrasi. Bias totalnya +4,92 tandan per pohon di
deteksi vs +0,41 di GT. **Ini kalibrasi, bukan cacat gagasan**: menurunkan ulang
faktor yang sama dari train-deteksi (C3) langsung memberi 1,1854.

**Konsekuensi untuk sub-proyek ini** — algoritma M01-M05 **tidak bisa dipakai
sebagai modul L**. Ia menjawab "berapa banyak tandan per kelas", bukan "kotak
mana milik tandan mana". Pipeline per-tandan butuh identitas, bukan cacah.
Pemblokir G1 karena itu masih terbuka.

**Catatan kehati-hatian** — README repo itu sendiri sudah memisahkan kedua
setelan (98,05% Class+-1 dengan deteksi GT vs 77,48% dengan YOLO26m). Angka
87,62% di `algorithms/README.md` tidak menyebut masukannya, jadi mudah terbaca
sebagai angka end-to-end padahal bukan. Layak diberi keterangan di sana.

**Sumber** — `scripts/eval_counting_baseline.py` ·
`results/pt_e_006_baseline_counting.json`

### PT-E-006 — tambahan: masukan C, detektor repo itu sendiri

Ditambahkan setelah entri di atas ditulis (bukan menyunting yang lama). Repo
Baseline-SawitMVC menyertakan prediksi ter-cache detektornya sendiri
(`predictions/y26mv2_per_tree/`, YOLO26m `y26mv2`, 953 pohon). Memakainya
menutup kemungkinan bahwa penurunan di masukan B cuma efek detektor berbeda.

**M01 pada 141 pohon test, tiga masukan:**

| Masukan | macro MAE | class ±1 | tree ±1 | total MAE |
|---|---|---|---|---|
| A kotak GT | **0,3404** | 0,9592 | 0,8723 | 1,1915 |
| C deteksi `y26mv2` (detektor repo itu) | **1,1826** | 0,7057 | 0,2411 | 2,7447 |
| B deteksi YOLO26l @conf 0,10 (sel 5) | 1,8298 | 0,5745 | 0,1206 | 5,1348 |

**Bacaannya:**

1. Selisih A → C (0,34 → 1,18, **3,5x**) adalah **efek detektor murni**:
   algoritma sama, korpus sama, hanya kotaknya yang berganti dari GT ke deteksi.
   Ini sejalan dengan README repo itu sendiri (98,05% Class+-1 dengan deteksi GT
   vs 77,48% dengan YOLO26m).
2. Selisih C → B (1,18 → 1,83) **bukan** kelemahan algoritmanya melainkan
   ambang keyakinan: PT-E-00x memakai conf 0,10 karena disetel untuk memaksimalkan
   manfaat penggabungan pada tugas KLASIFIKASI, bukan untuk counting. Di ambang
   serendah itu deteksi jauh lebih banyak (naif 4,36 vs 2,45 macro MAE), jadi
   penghitung apa pun yang berbasis pembagian ikut membengkak. Angka M01 yang
   adil untuk dikutip adalah **1,1826**, bukan 1,8298.
3. Pada titik kerja yang sama-sama wajar, ketiga jalur mendarat berdekatan:
   M01 1,1826 · k-per-kelas 1,1854 · Ridge+F_all 1,0542 (dan 1,036 di catatan
   repo itu untuk detektornya sendiri). Tidak ada yang mendekati 0,37.

**Kesimpulan yang mengikat** — angka 87,62% / macro MAE 0,3746 di
`algorithms/README.md` adalah angka **kotak GT**, bukan end-to-end. Ia sah dan
tereproduksi persis, tetapi tidak sebanding dengan angka end-to-end mana pun.
Yang membatasi counting bukan algoritma dedup-nya, melainkan detektornya —
persis kesimpulan yang sudah ditulis README utama repo itu.

---

## PT-E-007 — Penghitung Baseline-SawitMVC sebagai REM penggabungan (2026-08-17)

**Latar** — PT-E-003 menunjukkan penaut **terlalu pelit**: 3.095 kelompok untuk
1.404 tandan asli (2,2x), hanya 29% tandan punya pool >=2 tampak. Algoritma
M01 dari Baseline-SawitMVC tahu hal yang tidak diketahui penaut — berapa banyak
tandan di pohon itu. Gagasannya: pakai angka itu sebagai target, gabungkan terus
dari pasangan berskor tertinggi sampai jumlah kelompok turun ke target.

**Hipotesis** — Memaksa penggabungan sampai cacah target menaikkan porsi tandan
yang tersatukan, sehingga gain +4,9 pp dari PT-E-001/003 menyentuh lebih banyak
tandan dan akurasi kelas agregat naik melewati pipeline lama (0,7203).

**Yang memalsukan** — akurasi kelas TIDAK naik saat porsi tersatukan naik.

**Cara** — deteksi, penaut, dan aturan yang sama persis di ketiga mode; hanya
kriteria berhentinya berbeda. Kendala keras tetap: satu kotak per sisi per
tandan, plafon ukuran 3 (4-sisi) / 6 (8-sisi).

**Hasil** —

| Split | Mode | Kelompok / asli | Tersatukan | R0 | R4 | F1 penaut |
|---|---|---|---|---|---|---|
| val | A ambang tetap | 2.141 / 992 (2,16x) | 32,9% | 0,6973 | **0,7180** | 0,2529 |
| val | B rem M01 | 1.490 / 992 (1,50x) | 62,1% | 0,6382 | 0,6551 | 0,2393 |
| val | C rem **cacah sempurna** | 1.104 / 992 (1,11x) | 76,5% | 0,5974 | 0,6101 | 0,2232 |
| test | A ambang tetap | 3.095 / 1.404 (2,20x) | 29,2% | 0,6998 | **0,7139** | 0,1731 |
| test | B rem M01 | 2.098 / 1.404 (1,49x) | 59,3% | 0,6577 | 0,6872 | 0,1897 |
| test | C rem **cacah sempurna** | 1.582 / 1.404 (1,13x) | 75,9% | 0,6132 | 0,6454 | 0,1875 |

**Putusan** — **DIPALSUKAN, tegas.** Remnya bekerja persis seperti dirancang —
porsi tersatukan naik 29% → 76% dan rasio kelompok mendekati 1,0 — tetapi
akurasi kelas **turun monoton** di kedua split (test 0,7139 → 0,6872 → 0,6454).

**Diagnosis ulang yang mengikat pekerjaan berikutnya.** Mode C adalah kuncinya:
di sana cacahnya **sempurna benar** (diambil dari GT), jadi tidak ada ruang
menyalahkan taksiran M01. Akurasinya tetap turun paling dalam. Artinya:

> Masalah penaut **bukan berhenti terlalu cepat**, melainkan **urutan skornya
> salah**. Kalau peringkat pasangannya benar, memaksa menggabung lebih banyak
> — dari skor tertinggi ke bawah — akan menggabung pasangan yang benar lebih
> dulu dan akurasi naik. Yang terjadi sebaliknya: pasangan berskor tertinggi
> yang belum tergabung ternyata mayoritas SALAH.

Ini membatalkan dua dugaan sebelumnya yang tercatat di PT-E-003:
1. "recall pasangan 0,12 sebagian artefak ambang warisan kotak GT" — **salah**.
   Menurunkan ambang (mode B/C) memperburuk, bukan memperbaiki.
2. "menyetel ambang di ruang deteksi adalah perbaikan murah" — **tidak berguna**.
   Ambang berapa pun tidak menolong kalau peringkatnya yang keliru.

**Yang tetap bertahan** — aturan agregasi terus memberi nilai tambah di semua
mode, bahkan justru menguat saat pool makin kotor (test R4−R0: +1,44 → +2,96 →
+3,24 pp). Jadi modul A sehat; yang rusak masukannya.

**Konsekuensi** — algoritma dedup Baseline-SawitMVC **tidak bisa menolong
pipeline ini**, baik sebagai pengganti penaut (ia tidak mencocokkan kotak,
PT-E-006) maupun sebagai rem (dicoba di sini, memperburuk). Jalan yang tersisa
adalah menaikkan mutu **peringkat pasangan** itu sendiri — bukan menyetel ulang
kapan berhenti.

**Sumber** — `scripts/eval_rem_hitung.py` · `results/pt_e_007_rem_hitung.json`

---

## PT-E-008 — Arah putar pengambilan foto (2026-08-17)

**Latar** — Pemilik data mengonfirmasi: foto diambil **memutari pohon searah
jarum jam**, urutan sisi 1→2→3→4 konsisten. Informasi ini tidak pernah dipakai;
seluruh fitur geometri sebelumnya memakai `abs_dcx` — **nilai mutlak** — sehingga
arah pergeseran dibuang.

**Hipotesis** — Tandan yang sama bergeser ke arah yang konsisten antar sisi
berurutan, dan tanda pergeseran itu memisahkan pasangan benar dari salah jauh
lebih baik daripada jaraknya saja.

**Bukti dasar** (pohon 4-sisi, seluruh korpus):

| Offset sisi | Pasangan BENAR: dx rerata | Konsistensi arah | Pasangan SALAH |
|---|---|---|---|
| +1 | **+0,241** (simpangan 0,116) | **98,6% ke kanan** | −0,024 (54,9% kiri, simpangan 0,213) |
| +2 | +0,088 (simpangan 0,331) | 64,0% ke kanan | −0,000 (50,1%) |
| +3 | **−0,260** (simpangan 0,109) | **99,7% ke kiri** | +0,019 (46,6%) |

Konstanta yang dipas di train (`results/harapan_geser.json`) memperlihatkan
tanda tangan putaran melingkar yang rapi pada pohon 8-sisi:

    offset  1      2      3      4      5      6      7
    dx    +0,120 +0,255 +0,347 +0,308 -0,355 -0,261 -0,152

Naik sampai puncak di offset 3–4 (~90–135°), lalu berbalik tanda, dan simetris
hampir sempurna terhadap titik balik. Ini persis yang diharapkan dari
pengambilan foto memutar satu arah.

**Fitur yang ditambahkan** — `offset_bertanda`, `dx_bertanda`, `dy_bertanda`,
`sisa_dx` (dx dikurangi harapan untuk offset itu), `abs_sisa_dx`. Urutan pasangan
dikanonikkan (selalu dari sisi ber-indeks kecil ke besar) supaya tandanya
bermakna tetap. Konstanta dipas **hanya di split train**.

**Hasil — penaut di kotak GT:**

| Varian | val F1 | val ARI | test F1 | test ARI |
|---|---|---|---|---|
| D tanpa arah | 0,3732 | 0,3228 | 0,3651 | 0,3069 |
| **D dengan arah** | **0,6550** | **0,6252** | **0,6170** | **0,5694** |
| E tanpa arah | 0,4323 | 0,3623 | 0,3979 | 0,3292 |
| **E dengan arah** | **0,6718** | **0,6139** | **0,6486** | **0,5904** |

Bias jumlah kelompok ikut membaik: −0,96 → **−0,22**.

**Putusan — gerbang G1 LOLOS** (val F1 0,6718 ≥ 0,65 dan ARI 0,6139 ≥ 0,55),
setelah gugur di semua varian sebelumnya.

**Hasil — pipeline utuh (test):**

| | tanpa arah | dengan arah |
|---|---|---|
| F1 penautan di deteksi | 0,1766 | 0,1957 |
| R4 akurasi kelas | 0,7124 | **0,7179** |
| R4 vs oracle | −2,36 pp | **−1,81 pp** |
| penggabungan, pool ≥2 tampak | +4,76 pp | **+5,32 pp**, CI95 [+2,09; +8,42] |

**Putusan — gerbang G2 LOLOS** (−1,81 pp, toleransi −2,0), setelah gugur
sebelumnya.

**JEBAKAN yang sempat memakan korban dan sudah ditutup.** Tabel konstanta hidup
sebagai global modul yang semula hanya diisi di dalam `main()` skrip penaut.
Skrip end-to-end meng-import modul itu dan mendapatkannya **kosong**, sehingga
fitur arah tidak aktif dan hasilnya diam-diam kembali ke fitur lama — tanpa satu
pun pesan galat. Gejalanya khas: F1 end-to-end 0,1766 → 0,1761, **identik sampai
tiga desimal**, padahal di kotak GT melonjak 0,3651 → 0,6486. Angka yang tidak
berubah sama sekali itu tandanya. Perbaikan akar: tabel di-cache ke
`results/harapan_geser.json` dan dimuat **otomatis saat modul di-import**, plus
`RuntimeWarning` keras kalau kosong.

**Catatan jujur** — lompatan ini bukan hasil pemodelan. Warna, tekstur, embedding
terlatih, out-of-fold, dan rem cacah semuanya mentok atau memperburuk. Yang
membuka jalan adalah satu kalimat dari pemilik data tentang **cara foto diambil**
— informasi yang seharusnya ditanyakan di awal, bukan setelah tujuh eksperimen.

**Sumber** — `scripts/penaut_pertandan.py` (fitur + `hitung_harapan_geser`) ·
`results/harapan_geser.json` · `results/pt_e_002_penaut.json` ·
`results/pt_e_003_endtoend.json` · pembanding tanpa arah disimpan di
`results/pt_e_00{2,3,4}_*_tanpa_arah.json`

### PT-E-008 — tambahan: sapuan ambang di ruang deteksi, dan counting diulang

Ditambahkan setelah entri di atas (bukan menyunting yang lama).

**Sapuan ambang penaut di val-DETEKSI** (kini sah dicoba: pembatalan di PT-E-007
berlaku saat PERINGKAT skornya buruk; setelah arah putar, AUC pasangan 0,95):

| Ambang | R4 val | pool multi | F1 pasangan |
|---|---|---|---|
| 0,10 | 0,7067 | 290 | 0,2680 |
| 0,18 | 0,7067 | 278 | 0,2689 |
| 0,25 | 0,7079 | 272 | 0,2668 |
| 0,35 | 0,7079 | 261 | 0,2596 |
| **0,45** | **0,7112** | 246 | 0,2595 |

Polanya konsisten: **ambang lebih tinggi lebih baik** — lebih sedikit kelompok
tapi lebih bersih. Dikunci 0,45. Perbaikannya marginal (+0,22 pp di val, nol di
test: 0,7179 di kedua ambang; jumlah pool multi berubah 335 → 309, jadi runnya
memang berbeda).

**Counting diulang dengan penaut baru (test):**

| Penghitung | sebelum arah | sesudah arah |
|---|---|---|
| C4 hitung pool | 3,3422 (bias +11,96) | **3,4610** (bias +12,54) |
| C5 Ridge + F_all | 1,0542 | 1,0542 |

**G3 tetap GUGUR**, dan sedikit memburuk — konsekuensi langsung ambang 0,45 yang
menghasilkan lebih banyak kelompok.

**Bottleneck sudah BERGESER, dan ini temuan operasionalnya.** Di kotak GT penaut
mencapai F1 0,649; di ruang deteksi hanya 0,19. Selisih itu bukan lagi soal
peringkat skor (sudah beres) melainkan **positif palsu detektor**: 40% kelompok
di ruang deteksi seluruhnya berisi positif palsu.

Penyebabnya bisa ditunjuk: ambang keyakinan deteksi **conf 0,10** dipilih di
PT-E-001 saat penggabungan masih lemah, ketika memaksimalkan recall memang
menguntungkan. Setelah penaut bekerja, pilihan itu jadi merugikan — deteksi
sampah ikut membentuk kelompok palsu, merusak baik klasifikasi maupun cacah.
**Menyapu ulang conf belum dikerjakan** dan merupakan langkah termurah berikutnya.

**Sumber** — `results/pt_e_003_endtoend.json` (memuat `sapuan_ambang_val_deteksi`) ·
`results/pt_e_004_counting.json` · pembanding tanpa arah di
`results/pt_e_00{3,4}_*_tanpa_arah.json`

---

## PT-E-009 — Sapuan ulang ambang keyakinan deteksi (2026-08-17)

**Hipotesis** — `conf = 0,10` dikunci di PT-E-001 saat penggabungan masih lemah.
Setelah PT-E-008 memperbaiki penaut, 40% kelompok di ruang deteksi seluruhnya
positif palsu; menaikkan `conf` seharusnya membuang sampah itu dan menaikkan
akurasi kelas maupun cacah.

**Yang memalsukan** — akurasi kelas atas SELURUH tandan GT tidak naik saat
`conf` dinaikkan.

**Cara** — 6 conf x 3 ambang penaut, `tau` disetel ulang per conf, semua dipilih
di val. Bagian mahal (citra, deskriptor, embedding, skor seluruh pasangan)
dihitung sekali di conf terendah lalu disaring — deteksi di ambang tinggi adalah
himpunan bagian dari ambang rendah. Penghitung pembanding dipas ulang di train
pada tiap conf.

**KOREKSI METRIK yang mengubah jawaban.** Versi pertama menilai akurasi hanya
pada tandan yang TERDETEKSI, dan memilih `conf = 0,60` dengan R4 0,8107. Angka
itu palsu: penyebutnya menyusut dari 890 tandan (conf 0,10) menjadi 243 (conf
0,60). Menaikkan conf membuang tandan yang sulit lebih dulu, jadi soal ujiannya
yang jadi mudah, bukan pipeline-nya yang membaik. Metrik pemilih diganti menjadi
akurasi atas **seluruh tandan GT**, dengan tandan tak terdeteksi dihitung salah.

**Hasil (val, ambang penaut terbaik per conf):**

| conf | Cakupan | F1 penaut | R4 tandan terdeteksi | **R4 seluruh tandan** |
|---|---|---|---|---|
| **0,10** | 90% | 0,251 | 0,7146 | **0,6411** |
| 0,20 | 83% | 0,284 | 0,7242 | 0,6008 |
| 0,30 | 74% | 0,353 | 0,7425 | 0,5464 |
| 0,40 | 62% | 0,406 | 0,7423 | 0,4587 |
| 0,50 | 46% | 0,397 | 0,7636 | 0,3549 |
| 0,60 | 24% | 0,538 | 0,8107 | 0,1986 |

**Putusan** — **DIPALSUKAN.** `conf = 0,10` sudah optimal; menaikkannya
memperburuk secara monoton. Terkunci dari val: conf 0,10, ambang penaut 0,65
(marginal di atas 0,45: 0,6411 vs 0,6391).

**Test sekali, konfigurasi terkunci:**

| | nilai |
|---|---|
| R4 atas seluruh 1.404 tandan GT | **0,6474** (cakupan 90%) |
| R4 atas tandan terdeteksi | 0,7163 (oracle 0,7360) |
| C4 hitung pool | macro MAE 3,6571 |
| C5 Ridge + F_all | macro MAE **1,0542** |

**Yang dipelajari** — dugaannya separuh benar dan separuh salah. Benar: positif
palsu memang merusak penautan, dan membuangnya menaikkan F1 penaut dua kali
lipat (0,25 → 0,54) serta akurasi pada tandan yang tersisa (0,715 → 0,811).
Salah: itu tidak bisa ditebus dengan menaikkan ambang, karena kurva
presisi-recall detektor terlalu curam — setiap tandan sampah yang dibuang
menyeret beberapa tandan asli. **Perbaikannya harus di detektornya, bukan di
ambangnya.**

Ini juga menutup jalur counting: C4 tidak akan mengalahkan C5 selama cacah pool
mewarisi seluruh positif palsu detektor, dan ambang bukan alat untuk itu.

**Sumber** — `scripts/sapu_conf.py` · `results/pt_e_009_sapu_conf.json`

---

## PT-E-010 — Konfigurasi terbaik diuji di SawitMVC-Depth 352 (2026-08-17)

**Hipotesis** — Temuan di korpus 953 bertahan di sesi akuisisi yang berbeda.

**Kenapa uji ini kuat** — SawitMVC-Depth direkam terpisah ~80 hari setelah korpus
953 (`../results/pergeseran_temporal.json`), kamera berbeda, citra landscape
1280x800 alih-alih portrait 960x1280, dan 352 pohon alih-alih 953. Kalau temuan
intinya bertahan, ia sifat protokol pengambilan — bukan kebetulan satu sesi.

**Konfigurasi** — dikunci dari 953, TIDAK disetel ulang: detektor
`runs/yolo26l_e60_i1280_rgb352` (detektor 352 sendiri), penaut varian E
(geometri + arah putar + kelas prediksi lunak + re-ID), aturan R4, conf 0,10.
Bobot re-ID **dipindah apa adanya dari 953** (uji transfer). Konstanta arah putar
dan `tau` dipas ulang di train 352 — kalibrasi, bukan penyetelan konfigurasi.

### Hasil 1 — arah putar: REPLIKASI KUAT

| Offset (4-sisi) | 352 | 953 |
|---|---|---|
| +1 | +0,163 · **98,4% ke kanan** | +0,241 · 98,6% |
| +3 | −0,175 · **99,0% ke kiri** | −0,260 · 99,7% |
| pasangan salah | ~0,000 · ~50/50 | sama |

Arah dan konsistensi identik. Besaran lebih kecil karena citra lebih lebar, jadi
pergeseran sudut yang sama menutupi fraksi lebar yang lebih kecil. Konstanta yang
dipas di train 352: `+0,168 · +0,209 · −0,164`.

### Hasil 2 — penaut: REPLIKASI, dan JAUH LEBIH BAIK

| | 953 | **352** |
|---|---|---|
| F1 kotak GT (test) | 0,6486 | **0,6847** |
| ARI kotak GT (test) | 0,5904 | **0,6643** |
| **F1 di ruang DETEKSI (test)** | **0,1957** | **0,7083** |
| ARI di ruang deteksi | 0,1727 | **0,6044** |
| pool seluruhnya positif palsu | 40,1% | **33,9%** |
| bias jumlah kelompok | −0,22 | +0,35 (MAE 0,60) |

Selisih di ruang deteksi itu **3,6 kali lipat** dan menjelaskan sisanya: detektor
352 menghasilkan 2,7 deteksi/citra pada conf 0,10, sedangkan detektor 953
menghasilkan 8,2 — jadi beban positif palsu jauh lebih ringan. **Ini konfirmasi
langsung diagnosis PT-E-009**: yang membatasi pipeline di 953 memang detektornya,
dan begitu detektornya lebih bersih, penaut langsung bekerja mendekati mutunya di
kotak GT (0,708 vs 0,685 — praktis sama).

### Hasil 3 — recall: REPLIKASI

| | 953 | 352 |
|---|---|---|
| recall per-kemunculan | 0,8227 | 0,7390 |
| recall per-tandan | 0,9038 | 0,8354 |
| **selisih** | **+8,11 pp** | **+9,64 pp** |

### Hasil 4 — penggabungan kelas: TIDAK REPLIKASI SECARA SIGNIFIKAN

| Split | G0 (R4 vs R0cal, pool >=2) |
|---|---|
| val (52 pohon, 84 pool multi) | **+6,70 pp**, CI95 [+2,03; +11,59] |
| **test (55 pohon, 95 pool multi)** | **+2,85 pp**, CI95 [−2,00; +8,24] |
| pembanding 953 test (758 pool multi) | +4,36 pp, CI95 [+2,33; +6,25] |

Arahnya positif di kedua split, tetapi CI test **memuat nol**. Lebar CI-nya 10,2
pp lawan 3,9 pp di 953 — konsekuensi langsung ukuran sampel: 95 pool multi-tampak
lawan 758. **Uji ini tidak punya daya untuk memutuskan**, bukan bukti efeknya
hilang.

### Hasil 5 — R4 tidak menang di test 352

| Test 352, tautan oracle | akurasi |
|---|---|
| R0 satu tampak | 0,6970 |
| R0cal | 0,6814 |
| **R2 rerata softmax** | **0,7094** |
| R4 ekspektasi ordinal | 0,6946 |

Di val 352 R4 menang telak (0,7846 vs R0 0,7017, +8,3 pp), di test ia kalah dari
R2. Penyebab paling mungkin: `tau` dicari di **52 pohon val** — terlalu sedikit,
jadi ambangnya overfit. Di 953 (96 pohon val, 890 pool) R4 menang konsisten di
kedua split. **Pelajaran: pemilihan ambang ordinal butuh split val yang lebih
besar daripada yang tersedia di 352.**

### Perbandingan terhadap pipeline lama (test 352)

| | lama (per citra) | baru (per tandan) |
|---|---|---|
| Recall | 0,7390 | **0,8354** |
| Akurasi kelas | **0,7063** | 0,6946 |

Pola yang sama dengan 953: recall menang, kelas imbang-sedikit-di-bawah.

**Putusan** — **SEBAGIAN DIKONFIRMASI.** Tiga dari lima temuan replikasi kuat
(arah putar, mutu penaut, pergeseran recall), dan yang keempat — penaut jauh
lebih baik di detektor yang lebih bersih — **mengonfirmasi diagnosis PT-E-009
secara independen**. Yang tidak replikasi adalah keuntungan kelas dari
penggabungan, tetapi test 352 terlalu kecil untuk memutuskannya (CI 10,2 pp).

**Sumber** — `scripts/uji_352.py` · `results/pt_e_010_uji_352.json`

---

## PT-E-011 — KOREKSI: hambatannya bukan mutu detektor, melainkan kepadatan adegan (2026-08-17)

**Latar** — PT-E-009 dan PT-E-010 sama-sama menyimpulkan "yang membatasi adalah
detektor". Pemilik data menolak kesimpulan itu: menurutnya mengganti detektor
paling naik 1-2%, dan kalau memang seberpengaruh itu, efeknya akan terlihat
lepas dari arsitektur backbone-nya. Klaim itu diuji langsung.

**Yang tidak pernah saya periksa** — kepadatan objek per citra. Seluruh argumen
"detektor 352 lebih bersih" bersandar pada 2,7 deteksi/citra lawan 8,2, tanpa
membandingkannya dengan jumlah objek yang memang ada di sana.

**Hasil (test, conf 0,10, IoU 0,5):**

| | 953 | 352 |
|---|---|---|
| kotak GT / citra | 4,44 | 1,86 |
| deteksi / citra | 6,26 | 2,15 |
| rasio deteksi:GT | 1,41x | 1,16x |
| **presisi deteksi** | **0,584** | **0,639** |
| **recall deteksi** | **0,823** | 0,739 |

**Putusan** — **KESIMPULAN PT-E-009/010 DIPALSUKAN.** Kedua detektor mutunya
setara: presisi beda 5,5 pp, dan detektor 953 justru **lebih baik recall-nya**
(0,823 vs 0,739). Deteksi/citra 2,90x lebih banyak di 953 hampir seluruhnya
dijelaskan oleh objek yang 2,39x lebih padat, bukan oleh positif palsu berlebih.

**Diagnosis pengganti — kepadatan adegan, bukan mutu detektor:**

| | 953 | 352 |
|---|---|---|
| deteksi per pohon | ~25 | ~8,6 |
| pasangan lintas-sisi per pohon | ~235 | ~28 |
| **prevalensi pasangan benar** | **~4%** | **~21%** |

Mencari 10 pasangan benar di antara 235 versus 6 di antara 28. Tugas penautan di
953 **secara kombinatorik ~5x lebih sulit**, dan itu sifat korpusnya (10,3 tandan
per pohon lawan 6,5), bukan sifat detektornya. Mengganti backbone tidak mengubah
kepadatan adegan.

**Apa yang ini batalkan:**
- "Perbaikannya harus di detektornya, bukan di ambangnya" (PT-E-009) — separuh
  benar: ambang memang bukan alatnya, tetapi detektor juga bukan.
- "Selisih 3,6x di ruang deteksi mengonfirmasi diagnosis PT-E-009 secara
  independen" (PT-E-010) — **tidak sah**. Selisih itu dijelaskan kepadatan.

**Apa yang ini KUATKAN:** kenapa fitur arah putar (PT-E-008) memberi lompatan
terbesar sepanjang sub-proyek ini. Ia bekerja bukan dengan memperbaiki
diskriminasi per pasangan, melainkan dengan **memangkas ruang kandidat** —
persis obat untuk masalah kombinatorik. Prior lain yang mempersempit kandidat
lebih jauh (mis. depth di korpus 352) karena itu lebih menjanjikan daripada
detektor baru. Kaveat: E-007 Volume 1 sudah pernah memalsukan penautan berbasis
depth, tetapi tanpa prior arah dan tanpa penilai terlatih.

**Pelajaran metodologis** — ini kesalahan penyebut yang **keempat** di sub-proyek
ini, dalam bentuk baru: membandingkan hitungan mentah (deteksi/citra) antar
korpus tanpa menormalkannya terhadap jumlah objek yang ada. Aturan di
`docs/HASIL.md` §13 diperluas: **periksa penyebut setiap kali dua angka
dibandingkan — termasuk saat penyebutnya adalah "berapa banyak yang seharusnya
ada".**

**Sumber** — diukur langsung dari `results/pred_skorpenuh{,_352}_test.npz` dan
GT kedua korpus; perintahnya ada di riwayat percakapan sesi 2026-08-17.

---

## PT-E-013 — Depth + arah putar → rekonstruksi 3D (352) (2026-08-17)

**Hipotesis** — PT-E-011 menyimpulkan hambatannya kombinatorik, dan obatnya
prior yang memangkas ruang kandidat. Korpus 352 punya depth metrik. Digabung
arah putar yang sudah terbukti, seharusnya bisa merekonstruksi posisi 3D tiap
tandan di kerangka berpusat-pohon — dan tandan yang sama harus mendarat di titik
yang sama dari sisi mana pun ia dilihat. Itu prior pemangkas kandidat yang jauh
lebih kuat daripada pergeseran horizontal saja.

**Yang memalsukan** — jarak 3D antar pasangan setandan tidak lebih kecil
daripada antar pasangan beda-tandan.

**Cara** — depth mentah `.raw` (848x480, uint16 mm) direproyeksi penuh ke bidang
warna 1280x800 memakai kalibrasi di sidecar tiap berkas: intrinsik depth ->
ekstrinsik (`mTrans` = [-23,67; 0,07; 0,14] mm) -> intrinsik warna. Metadata
dataset **eksplisit memperingatkan** bahwa me-resize buffer secara langsung
meleset median 29 px, jadi reproyeksi penuh memang wajib. Distorsi diabaikan
(koefisiennya kecil).

Rekonstruksi: `Z` = median depth di dalam kotak; titik kamera
`(Xc, Yc, Zc) = ((u-cx)Z/fx, (v-cy)Z/fy, Z)`; lalu diputar ke kerangka
berpusat-pohon dengan `theta_i = ±i·2π/n` dan `R` = median depth per pohon.
Kedua tanda putaran diuji.

**Hasil (kotak GT, 120 pohon 4-sisi):**

| Fitur | AUC |
|---|---|
| **rekonstruksi 3D, arah +** | **0,4511** |
| **rekonstruksi 3D, arah −** | **0,5083** |
| jarak depth mentah (ΔZ) | 0,4797 |
| tinggi metrik (ΔY) | 0,6027 |
| jarak horizontal MUTLAK | 0,2542 |

Kewarasan: median depth per citra 2.524 mm (p10 1.122, p90 3.337) — reproyeksinya
benar, ini bukan bug.

**Putusan** — **DIPALSUKAN.** Rekonstruksi 3D tidak lebih baik daripada acak,
untuk kedua arah putaran. Jarak 3D pasangan setandan (median 416 mm) justru
**lebih besar** daripada pasangan beda-tandan (383 mm).

**Kenapa gagal, dan ini penjelasan yang E-007 tidak punya.** Rekonstruksinya
mengandaikan kamera mengorbit pada jari-jari tetap, selalu membidik sumbu pohon,
dengan langkah azimut persis 90°. Pengambilannya handheld: jarak berubah, arah
bidik berubah, sudutnya tidak persis. Rekonstruksi **memperbesar** galat itu
alih-alih meniadakannya — sebab tiap galat pose masuk sebagai pergeseran
sistematis seluruh titik dari citra tersebut.

Ini mereproduksi pemalsuan E-007 (Volume 1) lewat jalur berbeda, **dengan
sebabnya**: bukan "depth tidak informatif", melainkan "geometri pengambilan
tidak terkendali sehingga tidak bisa direkonstruksi".

**Satu yang bertahan, tapi tidak cukup.** Tinggi metrik (ΔY) mencapai AUC 0,6027,
sedikit di atas tinggi-citra ternormalkan yang sudah dipakai (0,5926). Diuji
gabungan pada held-out: 0,5648 → **0,5766**, naik 0,012 AUC. Terlalu kecil untuk
berarti ketika penaut sudah di AUC 0,95. **Tidak dimasukkan ke fitur produksi.**

**Konsekuensi** — depth **bukan** prior pemangkas kandidat yang dicari. Prior
yang bekerja (arah putar) bekerja justru karena ia **tidak** memerlukan
rekonstruksi: ia hanya butuh urutan sisi, bukan pose. Kandidat prior berikutnya
sebaiknya punya sifat yang sama — bergantung pada hal yang terkendali dalam
protokol pengambilan, bukan pada geometri yang tidak diukur.

**Sumber** — diukur langsung dari `/workspace/SawitMVC-Depth/depth/*.{raw,json}`;
skrip probe ada di riwayat percakapan sesi 2026-08-17.

---

## PT-E-012 — Modul C3, classifier multi-tampak (2026-08-17)

**Hipotesis** — Sketsa asal menunjukkan seluruh potongan tandan masuk KE DALAM
model, keluar satu label. Yang dibangun selama ini bukan itu: tiap potongan
dinilai sendiri, digabung rumus R4 di luar model. Rumus itu buta konteks — tidak
bisa tahu satu foto buram atau dua tampak beda kelas karena satu dari sisi
bayangan. Model yang melihat semuanya sekaligus mestinya bisa mempelajarinya.

**Yang memalsukan** — C3 tidak mengalahkan C2 (yang mengisolasi "melihat banyak
tampak sekaligus" dari "punya classifier khusus").

**Cara** — ketiganya dinilai pada POTONGAN GT dan TAUTAN ORACLE, himpunan tandan
yang sama, supaya galat deteksi dan galat penautan tidak ikut campur.

| Jalur | Isi |
|---|---|
| C1 | distribusi kelas detektor dipetakan ke kotak GT, digabung R4 (tanpa training) |
| C2 | ResNet-18 dilatih di potongan, per-tampak, digabung R4 |
| C3 | ResNet-18 + attention antar-tampak, seluruh tampak sekaligus, satu keluaran |

**Hasil (test, 1.404 tandan / 1.022 multi-tampak):**

| Jalur | Akurasi | Pada pool >=2 tampak |
|---|---|---|
| **C1 skor detektor + R4** | **0,7208** | **0,7583** |
| C2 classifier per-tampak + R4 | 0,7087 | 0,7397 |
| C3 multi-tampak (backbone sebagian beku) | 0,6781 | 0,7006 |
| C3 multi-tampak (backbone **beku penuh**) | 0,6467 | 0,6820 |

**Putusan** — **DIPALSUKAN.** C3 kalah dari C2 sebesar 3,06 pp dan dari C1
sebesar 4,27 pp. Membekukan backbone penuh — dugaan perbaikan untuk overfit —
**memperburuk lagi** menjadi 0,6467.

**Dua temuan, dan yang kedua lebih luas dari yang pertama:**

1. **C3 < C2.** Melihat seluruh tampak sekaligus tidak menolong pada skala data
   ini. Penyebabnya overfit: C2 mencapai loss latih **0,0018** (praktis
   menghafal), dan varian beku penuh tetap menunjukkan train 0,878 lawan val
   0,643 meski kepalanya kecil dengan dropout 0,5 dan weight decay 1e-2.
   Contoh latihnya cuma 7.427 tandan; C3 punya parameter lebih banyak dan
   contoh lebih sedikit daripada C2 (yang berlatih per-tampak, ~14 ribu).

2. **C2 < C1** — ini yang lebih penting. Classifier potongan khusus **kalah dari
   skor kelas detektor yang sudah ada**. Detektor dilatih pada tugas deteksi
   penuh di 3.000 citra dengan sinyal supervisi dan augmentasi jauh lebih kaya;
   classifier potongan di 716 pohon tidak bisa menandinginya. **Jadi yang
   tertutup bukan cuma C3, melainkan seluruh jalur "tingkatkan modul C".**

**Batas klaim yang jujur** — ini memalsukan C3 **pada skala data ini**, bukan
gagasan classifier multi-tampak secara umum. Kalau nanti korpusnya jauh lebih
besar, pertanyaannya layak dibuka lagi. Yang TIDAK boleh disimpulkan dari sini:
bahwa agregasi multi-tampak tidak berguna — R4 di atas C1 tetap memberi
+4,36 pp (PT-E-001), dan itu bertahan.

**Sumber** — `scripts/c3_multitampak.py` · `results/pt_e_012_c3.json`; varian
backbone beku penuh ada di riwayat percakapan sesi 2026-08-17.

---

## PT-E-014 — Backbone lain untuk modul C (2026-08-18)

**Hipotesis** — PT-E-012 memalsukan modul C memakai SATU backbone (ResNet-18).
`IDEA.md` sec.4 butir 1 meminta backbone lebih kuat. Kalau ConvNeXt-Tiny membalik
C2<C1 atau C3<C2, penutupan jalur modul C oleh PT-E-012 terlalu dini.

**Yang memalsukan** — ConvNeXt tidak menaikkan C2/C3 di atas ResNet-18.

**Cara** — protokol identik PT-E-012 baris per baris: potongan GT, tautan oracle,
himpunan tandan sama (train 7.427 / val 992 / test 1.404, multi-tampak
5.546/760/1.022), 25 epoch, `tau` dipas di val per jalur. Yang berubah hanya
backbone. ConvNeXt dibekukan `features.0..3`, analog dengan `conv1..layer2` di
ResNet-18. Dua seed (0, 1) supaya efeknya bisa dibandingkan dengan derau.

**Reproduksi lebih dulu** — sel kontrol `resnet18+ce` menghasilkan **C1 R4 test
0,7208** dan **C1 R4_multi 0,7583**, keduanya PERSIS sama dengan PT-E-012. Jalur
tanpa training tereproduksi penuh, jadi pipa data, pembentukan pool, pemasangan
`tau`, dan evaluasi tervalidasi. Jalur terlatih TIDAK tereproduksi: C2 0,6823
lawan 0,7087 di PT-E-012 (-2,64 pp), sebabnya urutan konsumsi RNG saat
inisialisasi berbeda karena struktur modul sedikit berbeda.

**Hasil (test 953, akurasi R4; C1 = 0,7208 di semua sel):**

| backbone | loss | seed | C2 R4 | C3 | C2-C1 | C3-C1 |
|---|---|---|---|---|---|---|
| resnet18 | ce | 0 | 0,6823 | 0,6667 | -3,85 | -5,41 |
| resnet18 | ce | 1 | 0,6980 | 0,6859 | -2,28 | -3,49 |
| convnext_tiny | ce | 0 | 0,7009 | 0,6994 | -1,99 | -2,14 |
| convnext_tiny | ce | 1 | 0,7115 | **0,7187** | -0,93 | **-0,21** |

**Putusan** — **SEBAGIAN DIKONFIRMASI.** ConvNeXt menaikkan C2 (+1,86 pp pada ce,
seed 0) dan C3 tajam (+3,27 pp pada ce, seed 0), tetapi tidak satu pun sel
mengalahkan C1. Arah PT-E-012 bertahan; magnitudonya tidak.

**Temuan yang lebih luas: PT-E-012 tidak punya error bar.** Rentang antar-seed
untuk konfigurasi yang SAMA adalah 1,06-1,99 pp (C2) dan 0,43-1,93 pp (C3).
Putusan PT-E-012 bersandar pada selisih C2-C1 = -1,21 pp, yang lebih KECIL
daripada rentang seed di tiga dari empat konfigurasi. Selisih sebesar itu tidak
bisa dipisahkan dari derau inisialisasi dengan satu seed.

**Sumber** — `scripts/c_backbone_ordinal.py` · `results/pt_e_014_c_*.json` ·
dump `results/pt_e_014_prob_*.npz` · bobot `runs/c_*/best.pt`

---

## PT-E-015 — Ordinal loss (CORAL) untuk modul C (2026-08-18)

**Hipotesis** — `IDEA.md` sec.4 butir 2. B1<B2<B3<B4 berurutan; cross-entropy
memperlakukan galat B1-vs-B2 sama mahal dengan B1-vs-B4. Aturan agregasi R4
sendiri sudah ordinal, jadi loss ordinal menyelaraskan latih dengan hilir.

**Yang memalsukan** — CORAL tidak menaikkan akurasi di atas cross-entropy.

**Cara** — faktor kedua dari skrip yang sama, backbone dipatok. CORAL memodelkan
K-1 ambang kumulatif `P(y>k)` dengan bobot BERSAMA dan bias dipaksa menurun
(`b_k = b0 - cumsum(softplus(delta))`). Monotonisitas itu wajib, bukan kosmetik:
tanpanya selisih kumulatif bisa negatif dan vektor kelasnya tidak sah disuap ke R4.

**Hasil (test, akurasi; selisih coral - ce pada seed 0):**

| backbone | C2 ce | C2 coral | delta | C3 ce | C3 coral | delta |
|---|---|---|---|---|---|---|
| resnet18 | 0,6823 | 0,7058 | **+2,35 pp** | 0,6667 | 0,6766 | +0,99 pp |
| convnext_tiny | 0,7009 | 0,7037 | +0,28 pp | 0,6994 | **0,7130** | +1,36 pp |

**Putusan** — **SEBAGIAN DIKONFIRMASI.** CORAL menaikkan akurasi di keempat
pasangan (C2 dan C3, dua backbone) pada seed 0, terbesar +2,35 pp. Tetapi
seed 1 membalik sebagian (resnet18 C2: ce 0,6980 lawan coral 0,6859), dan
seluruh selisihnya berada di dalam rentang seed yang diukur PT-E-014. **Arahnya
konsisten, magnitudonya tidak terpisahkan dari derau pada n seed = 2.**

Digabung dengan PT-E-014: sel terbaik `convnext_tiny+coral` seed 0 memberi C3
0,7130 lawan 0,6781 di PT-E-012, memperkecil jarak C3-C1 dari -4,27 pp menjadi
**-0,78 pp**. Jalur modul C tidak seburuk yang PT-E-012 simpulkan, tapi tetap
belum mengalahkan C1 secara tunggal.

**Sumber** — sama dengan PT-E-014.

---

## PT-E-016 — Penaut GNN di ruang kotak GT (2026-08-18)

**Hipotesis** — `IDEA.md` sec.4 butir 3. PT-E-007 menyimpulkan urutan skor penaut
yang salah, bukan ambangnya. Urutan skor yang salah adalah gejala khas penilaian
INDEPENDEN: kalau kotak `a` sangat cocok dengan `b`, itu semestinya menurunkan
skor `a`-dengan-`c`, tapi `HistGradientBoosting` yang menilai satu pasangan
sekaligus tidak punya jalan untuk tahu. GNN dengan attention per-simpul membawa
persaingan antar-kandidat ke DALAM skor.

**Yang memalsukan** — GNN tidak mengalahkan penilai independen pada F1/ARI klaster.

**Cara** — fitur pasangan SAMA PERSIS (varian E: geometri + arah putar +
penampilan + re-ID + prob prediksi) dan perakit klaster SAMA PERSIS (Hungarian
per pasangan-sisi, lalu union-find serakah, batasan sisi-unik dan ukuran maks
3/6). Yang berbeda hanya cara skor sisi dihitung.

**Cacat metodologis di run pertama, dicatat karena mengubah putusan.** Grid
ambang `[0,10 .. 0,50]` diwarisi dari baseline. Baseline memuncak di 0,25
(optimum interior, sah); GNN naik MONOTON sampai 0,50 lalu grid habis. Delta yang
tercatat -3,36 pp mengukur GNN yang dilumpuhkan. Skor GNN memang terkalibrasi
lebih tinggi karena `pos_weight` 14,2. Disapu ulang di grid sampai 0,975,
optimum GNN ada di **0,90** (interior) dan putusannya berbalik.

**Hasil (test 953, 138 pohon, ambang dikunci dari val):**

| | baseline | GNN |
|---|---|---|
| AUC pasangan val | 0,9508 | **0,9585** |
| F1 | 0,6243 | 0,6349 |
| ARI | 0,5702 | 0,6047 |
| presisi | 0,6346 | 0,6718 |
| cakupan tandan | 0,6595 | 0,6497 |

Bootstrap tingkat pohon (2.000 resample):

| | delta | CI95 | P(delta>0) |
|---|---|---|---|
| F1 | +1,06 pp | [-1,46 ; +3,83] | 0,787 |
| ARI | +3,45 pp | [-0,05 ; +7,38] | 0,971 |
| cakupan | -0,98 pp | [-3,99 ; +2,14] | 0,257 |

**Putusan** — **TIDAK KONKLUSIF.** Yang bertahan tanpa kaveat hanya AUC pasangan
(+0,0077), yang bebas ambang dan bebas perakit. F1 tidak terpisahkan dari nol,
dan val justru memilih baseline (0,6692 lawan 0,6535) sementara test memilih GNN
— ranking yang berbalik antar-split adalah tanda selisihnya sebanding derau.
ARI nyaris lolos (batas bawah -0,05 pp) tetapi menyebutnya signifikan berarti
menggeser ambang setelah melihat angka, yang dilarang CLAUDE.md sec.2.

**Koreksi penyebut yang penting.** `IDEA.md` menargetkan cakupan penaut
29% -> >70%. Angka 29% itu hidup di ruang DETEKSI. Di ruang kotak GT — tempat
eksperimen ini berjalan dan tempat gerbang G1 diukur — baseline SUDAH 0,6595.
Target IDEA.md tidak bisa dijawab di sini. Lihat PT-E-017.

**Sumber** — `scripts/gnn_penaut.py` · `scripts/sapu_ambang_gnn.py` ·
`scripts/ci_gnn.py` · `results/pt_e_016_gnn.json` ·
`results/pt_e_016b_sapu_ambang.json` · `results/pt_e_016c_ci.json` ·
dump `results/pt_e_016_skor_test.npz` · bobot `runs/gnn_penaut/best.pt`

---

## PT-E-017 — Penaut dilatih di RUANG DETEKSI, bukan kotak GT (2026-08-18)

**Hipotesis** — Sejak PT-E-002 sampai PT-E-010, penaut SELALU dilatih di pasangan
kotak GT (`eval_endtoend.py`: "melatih ulang penaut di pasangan kotak GT split
train") lalu dipakai di atas deteksi. Kotak GT bersih: tepat satu per tandan
nyata, nol positif palsu. Deteksi tidak — PT-E-003 mencatat 39,9% pool seluruhnya
positif palsu. Penaut yang tak pernah melihat positif palsu saat latihan tidak
punya cara belajar menolaknya. Kalau benar, sebagian dari "cakupan 29%" bukan
kombinatorik melainkan domain shift.

**Yang memalsukan** — melatih di pasangan deteksi tidak menaikkan F1 penautan
di ruang deteksi.

**Cara** — tiga lengan, fitur/conf/perakit sama persis, `conf` 0,10 dikunci dari
PT-E-001:

| lengan | penilai | dilatih di |
|---|---|---|
| A | HistGradientBoosting | pasangan KOTAK GT (cara repo sekarang) |
| B | HistGradientBoosting | pasangan DETEKSI |
| C | GNN (PT-E-016) | pasangan DETEKSI |

**Hasil (test 953):**

| lengan | AUC val | F1 | presisi | recall | ARI | cakupan* | pool palsu |
|---|---|---|---|---|---|---|---|
| A latih kotak GT | **0,5868** | 0,1492 | 0,1793 | 0,1278 | 0,1223 | 0,1425 | 0,128 |
| B latih deteksi | 0,9015 | 0,3080 | 0,2771 | 0,3466 | 0,2707 | 0,3799 | 0,147 |
| C GNN di deteksi | **0,9422** | **0,3788** | 0,3915 | 0,3669 | 0,3221 | 0,3839 | **0,040** |

\* `cakupan_atas_terdeteksi`, penyebut = 758 tandan GT multi-sisi yang punya >=2
deteksi terpetakan. Penyebut kedua dilaporkan juga di JSON
(`cakupan_atas_semua`, penyebut 1.022 = seluruh tandan multi-sisi termasuk yang
detektornya lewatkan): A 0,1057 · B 0,2818 · C 0,2847. Dua penyebut ini sengaja
dipisah — CLAUDE.md sec.8.

**Putusan** — **DIKONFIRMASI, kuat.** Domain shift (B-A) = **+15,88 pp F1**.
Penalaran bersama di atasnya (C-B) = **+7,08 pp F1**. Total F1 naik 2,5x.

**Angka yang paling telak: AUC lengan A = 0,5868**, nyaris tebak-tebakan. Penaut
yang sama mencetak AUC 0,9508 di pasangan kotak GT. Artinya seluruh hasil
penautan ruang deteksi di sub-proyek ini — F1 0,1766 (PT-E-003), cakupan 29%,
dan gerbang G3 yang gugur — diproduksi oleh penaut yang praktis acak di domain
tempat ia sebenarnya dipakai.

**Konsekuensi untuk diagnosis yang berlaku (CLAUDE.md sec.6).** Diagnosis
"hambatannya kepadatan adegan, dan itu kombinatorik" tidak salah tapi **tidak
lengkap**: 15,88 pp bisa diambil tanpa satu pun ide baru, hanya dengan
memindahkan data latih ke domain yang benar. Kombinatorik tetap nyata — F1 0,3788
masih jauh dari 0,65 — tetapi ia bukan satu-satunya penjelasan, dan bukan yang
termurah diperbaiki.

**Dan GNN baru menunjukkan nilainya di sini.** Di ruang kotak GT ia menambah
+0,0077 AUC (tidak konklusif, PT-E-016); di ruang deteksi +0,0407 AUC dan
+7,08 pp F1. Pool yang seluruhnya positif palsu turun 0,147 -> **0,040**. Masuk
akal secara mekanis: persaingan antar-kandidat baru berguna kalau ada kandidat
sampah untuk dikalahkan, dan kotak GT tidak punya satu pun.

**Sumber** — `scripts/gnn_deteksi.py` · `results/pt_e_017_gnn_deteksi.json` ·
dump `results/pt_e_017_skor_test.npz` · bobot `runs/gnn_deteksi/best.pt`

---

## PT-E-018 — C1/C2/C3 sebagai ANGGOTA ENSEMBLE, bukan pesaing (2026-08-18)

**Hipotesis** — PT-E-012 mengadu C1/C2/C3 satu lawan satu, menyimpulkan tidak ada
yang mengalahkan C1, lalu menutup "seluruh jalur tingkatkan modul C". Pengukuran
itu benar; inferensinya melompat. Yang tidak pernah ditanyakan: apakah galat
mereka TERDEKORELASI. C1 adalah kepala klasifikasi detektor (tugas deteksi penuh,
3.000 citra, augmentasi mosaic/hsv, supervisi kotak+kelas). C2 adalah classifier
potongan (7.427 potongan sudah-terpotong, flip+brightness, kelas murni). Dua
rezim latih yang nyaris tidak beririsan.

**Yang memalsukan** — ensemble tidak mengalahkan C1 sendirian.

**Cara** — nol training baru; hanya kombinasi dump PT-E-014/015. Subset dipilih
SERAKAH maju di val (`tau` dipatok selama pencarian supaya seleksi tidak memilih
anggota yang cocok dengan ambang tertentu — jebakan yang sama dengan rekalibrasi
tersamar sebagai agregasi di PT-E-001), lalu bobot dan `tau` dipas di val. Test
disentuh sekali.

**Hasil (test 953, akurasi R4, potongan GT + tautan oracle):**

| | test | test multi-tampak |
|---|---|---|
| C1 sendiri | 0,7208 | 0,7583 |
| C2 resnet18+coral | 0,7058 | 0,7397 |
| C2 convnext+coral | 0,7037 | 0,7319 |
| C2 convnext+ce | 0,7009 | 0,7299 |
| C2 resnet18+ce | 0,6823 | 0,7084 |
| **Ensemble** C1 0,6 + convnext-coral 0,2 + convnext-ce 0,2 | **0,7464** | **0,7789** |

vs C1: **+2,56 pp, CI95 [+0,52 ; +4,53], P(delta>0) = 0,992** (bootstrap 138 pohon).
val 0,7470 lawan test 0,7464 — stabil, tidak seperti PT-E-016 yang rankingnya
berbalik antar-split.

**Putusan** — **DIKONFIRMASI.** CI tidak memuat nol.

**Dua konsekuensi yang lebih besar dari angkanya:**

1. **Setiap anggota C2 kalah dari C1, gabungannya menang.** Semua C2 ada di
   0,682-0,706, C1 di 0,7208. Ini membantah INFERENSI PT-E-012, bukan
   pengukurannya: yang tertutup adalah jalur MENGGANTI C1, bukan jalur
   MELENGKAPI C1. Yang terakhir tidak pernah diuji.

2. **Plafon 73,60% bukan plafon.** PT-E-001 menetapkan plafon oracle R4 di atas
   skor detektor 0,7360, dan `IDEA.md` menutup dengan "potensi maksimal ide ini
   melalui Oracle R4 adalah 73,60% bila tetap mengandalkan skor detektor YOLO".
   Ensemble mendarat di **0,7464** pada protokol yang sama, melewatinya +1,04 pp.
   Klausa "bila tetap mengandalkan skor detektor" ternyata menanggung seluruh
   beban: 73,60% adalah sifat probabilitas C1, bukan sifat pendekatan agregasi.

**Batas klaim** — diukur pada potongan GT dan tautan oracle, sama seperti
PT-E-012, jadi ia mengukur plafon modul C, bukan pipeline utuh. Anggota C2
dilatih di potongan GT; memakainya di potongan DETEKSI adalah domain shift
tersendiri yang belum diuji (bandingkan PT-E-017, di mana shift serupa merugikan
0,35 AUC). Itu pekerjaan PT-E-019.

**Sumber** — `scripts/ensemble_c.py` · `results/pt_e_018_ensemble.json`

---

## PT-E-019 — Pipeline utuh: penaut PT-E-017 + ensemble PT-E-018 (2026-08-18)

**Hipotesis** — PT-E-017 (penaut ruang deteksi, F1 0,1492 -> 0,3788) dan PT-E-018
(ensemble kelas, +2,56 pp) menyentuh pipeline lewat jalur berbeda: penaut
menentukan BERAPA BANYAK tandan tersentuh agregasi, ensemble menentukan akurasi
TIAP tandan yang tersentuh. Kalau keduanya nyata, efeknya berlipat.

**Yang memalsukan** — gabungan tidak melebihi jumlah kontribusi masing-masing.

**Cara** — faktorial 2x2 (penaut lama/baru x kelas C1/ensemble), test 953,
139 pohon, 1.268 tandan. Ambang penaut DIKUNCI dari PT-E-017 (lama 0,05; baru
0,90) supaya tidak disetel terhadap metrik hilir. `tau` dipas di val per sel.

**Hasil (test):**

| sel | val R4 | test R4 | test multi | n multi |
|---|---|---|---|---|
| penaut lama x C1 (kontrol) | 0,7247 | 0,7200 | 0,7124 | 372 |
| penaut lama x ensemble | 0,7213 | **0,7311** | 0,6989 | 372 |
| penaut baru x C1 | 0,7315 | 0,7263 | 0,7319 | **649** |
| penaut baru x ensemble | 0,7303 | 0,7287 | 0,7242 | **649** |

Acuan: PT-E-003 pipeline utuh 0,7124 · pipeline lama per-citra 0,7203 ·
plafon oracle C1 0,7360.

**Putusan** — **DIPALSUKAN pada klaim berlipat.** Kontribusi penaut +0,63 pp,
kontribusi kelas +1,11 pp, jumlah seharusnya +1,74 pp; gabungan hanya **+0,87 pp**
(CI95 [-1,23; +2,93], P=0,778). Sel terbaik BUKAN gabungannya melainkan
`penaut lama x ensemble` = 0,7311.

**Kenapa mereka saling menggantikan, bukan menambah.** Kolom `test multi`
membacakan mekanismenya: ensemble MENURUNKAN akurasi pada tandan multi-tampak
(0,7124 -> 0,6989 dengan penaut lama) tetapi menaikkan akurasi total. Jadi
ensemble menolong terutama pada tandan SATU-tampak, tempat tidak ada agregasi
yang bisa memperbaiki galat dan mutu probabilitas per-tampak menentukan
segalanya. Penaut baru memindahkan tandan dari satu-tampak ke multi-tampak --
yaitu ke wilayah tempat R4 sudah bekerja dan keunggulan ensemble encer. Keduanya
menyembuhkan penyakit yang sama lewat pintu berbeda.

**Cakupan: target IDEA.md bergerak, separuh jalan.** `n_multi` naik 372 -> 649
dari 1.268 tandan, yaitu **29,3% -> 51,2%**. Angka 29,3% mereproduksi "29%"
PT-E-003 tepat, jadi penyebutnya kali ini sebanding. Target IDEA.md >70% belum
tercapai.

**Kaveat yang membatasi seluruh entri ini:**

1. **Sel kontrol meleset +0,76 pp** dari PT-E-003 (0,7200 lawan 0,7124). Penyebab
   paling mungkin `results/harapan_geser.json` yang diperbaiki di sesi ini:
   penaut lama di sini memakai prior arah putar yang benar, termasuk untuk 34
   pohon 8-sisi yang sebelumnya mendapat nol. Artinya sebagian kenaikan sudah ada
   di baseline yang diperbaiki, bukan dari intervensi. Ini BUKAN reproduksi persis.
2. **Tidak ada sel yang signifikan** terhadap kontrol pada n=137 pohon.
3. Ensemble memakai C2 yang dilatih di potongan KOTAK GT dan dipakai di potongan
   DETEKSI. Shift itu ternyata TIDAK merusak seperti pada penaut (+1,11 pp masih
   positif), jadi risiko yang dikhawatirkan di PT-E-018 tidak terwujud -- tapi
   juga tidak diuji terpisah.

**Yang bertahan** — tiga sel non-kontrol melewati 0,7124 (PT-E-003), dan tiga
dari empat melewati 0,7203 (pipeline lama per-citra). Sel terbaik 0,7311 tinggal
0,49 pp dari plafon oracle 0,7360 **dengan tautan nyata, bukan oracle**.

**Arah berikutnya yang ditunjukkan hasil ini** — karena ensemble menolong di
tandan satu-tampak dan penaut menolong dengan memindahkan tandan keluar dari
sana, keduanya bersaing memperebutkan populasi yang sama. Pengungkit yang belum
tersentuh adalah menaikkan plafon pada tandan MULTI-tampak: di sana R4 di atas
C1 sudah 0,7319 sementara oracle C1 0,7360, jadi ruangnya tipis. Menembusnya
menuntut aturan agregasi yang lebih baik daripada R4, bukan probabilitas per
tampak yang lebih baik.

**Sumber** — `scripts/endtoend_gabungan.py` · `results/pt_e_019_gabungan.json`

---

## PT-E-020 — Penaut global khusus DAMIMAS dengan konteks kompetisi (2026-08-18)

**Tujuan engineering** — menaikkan mutu association di adegan padat tanpa
mengubah detektor: model sisi dilatih langsung pada pasangan DETEKSI DAMIMAS,
fiturnya membawa residual gerak bertanda, rank kandidat per sisi, margin
kompetitor, distribusi kelas lunak, dan embedding re-ID. Skor kemudian dirakit
dengan Hungarian, average/min/top-2 linkage, atau correlation-clustering ILP.

**Protokol** — prior gerak dan lima model sisi dipasang di 641 pohon TRAIN;
model/ensemble/perakit/batas ukuran/ambang dipilih di 86 pohon VAL; 127 pohon
TEST baru dibuat graf dan dinilai setelah konfigurasi terkunci. Checkpoint
re-ID juga DAMIMAS-only. Detektor tetap dump C1 yang sama pada `conf=0,10`.

**Sinyal model sisi.** Ada 225.918 pasangan train, 6.242 positif (2,763%), 52
fitur. AUC val lima kandidat = 0,9309--0,9435. Konfigurasi utility memilih
rerata tiga HistGradientBoosting, average-link, batas klaster observasi, ambang
0,70.

| kepala (semua dikunci dari VAL) | F1 test | presisi | recall | cakupan terdeteksi | cakupan semua | MAE pool |
|---|---:|---:|---:|---:|---:|---:|
| utility | 0,4631 | 0,4359 | 0,4940 | 0,5628 | 0,4206 | 9,584 |
| **F1** | **0,4704** | **0,4721** | 0,4688 | 0,5241 | 0,3918 | 10,656 |
| **cakupan** | 0,3561 | 0,2568 | **0,5806** | **0,6400** | **0,4784** | 4,160 |
| hitung-pool | 0,2600 | 0,1721 | 0,5312 | 0,5614 | 0,4196 | **2,880** |

**Putusan** — association membaik material dan kepala tugas memang harus
dipisahkan. Sebagai acuan historis, PT-E-017 pada seluruh varietas mencatat F1
0,3788, cakupan-terdeteksi 0,3839, dan cakupan-semua 0,2847; angka itu bukan
pembanding kausal yang persis karena scope sekarang hanya DAMIMAS. Walau
demikian, kenaikan absolut pada korpus yang tetap sangat padat cukup besar untuk
dipakai sebagai komponen pipeline berikutnya.

**Batas yang tetap keras.** Target cakupan semua >70% belum tercapai. Lebih
penting, jumlah pool bukan estimator counting yang baik: bahkan kepala khusus
MAE-pool masih 2,880, jauh di atas regresor counting sekitar 1,00. Karena itu
linker dipakai untuk identitas/agregasi dan sebagai fitur counting, bukan
dipaksa menjadi hasil counting akhir.

**Sumber** — `scripts/linker_global_damimas.py` ·
`scripts/laporkan_kepala_linker_damimas.py` ·
`results/damimas_linker_global.json`

---

## PT-E-021 — Kepala proposal fisik dan relabel probabilistik DAMIMAS (2026-08-18)

**Tujuan engineering** — memisahkan dua keputusan yang sebelumnya bercampur:
proposal fisik dibuat unik lintas kelas untuk lokalisasi/linker, sedangkan
kepala mAP boleh memancarkan empat hipotesis kelas berperingkat untuk satu
proposal. Semua ambang NMS, temperatur, smoothing ordinal, eksponen skor, dan
routing dipilih di 86 pohon VAL; TEST dibuka setelah konfigurasi terkunci.

**Proposal fisik.** NMS class-agnostic IoU 0,60 dengan kotak dari baris skor
tertinggi menang di VAL. Ia menghasilkan test AP50 lokalisasi **0,8381**,
AP50-95 **0,3662**, dan titik operasi P/R/F1 **0,8017 / 0,7952 / 0,7984**.
Dengan tautan oracle pada `conf=0,01`, recall fisik test **0,9620**, R4
**0,7464**, macro-F1 **0,7162**, dan R4 pada 883 tandan multi-tampak **0,7724**.
Utility `accuracy x recall` 0,7180 belum mengalahkan baseline 0,7204, sehingga
proposal ini menjadi kepala lokalisasi/linker dan tidak menggusur kepala recall.

**Relabel probabilistik.** Konfigurasi VAL-locked memakai classifier hibrida,
campuran label asal 0,5, `T=0,6`, gamma 2, top-4, eksponen lokalisasi 1,25, dan
smoothing ordinal 0,2. Dibanding fusion YOLO PT-E-019/awal, hasil test berubah:

| metrik | fusion YOLO | relabel | routing mAP akhir |
|---|---:|---:|---:|
| mAP50 | 0,5839 | 0,5880 | **0,5881** |
| mAP50-95 | 0,2711 | 0,2721 | **0,2723** |
| macro-F1 operasi | 0,5752 | **0,5773** | 0,5759 |
| AP50 B4 | 0,3983 | **0,4106** | **0,4106** |

Keempat AP50 kelas naik pada relabel: **0,7923 / 0,4942 / 0,6548 / 0,4106**.
Routing akhir hanya mengganti skor B3 dengan WBF original+relabel dan menjadi
kepala mAP; relabel murni tetap kepala titik-operasi karena macro-F1-nya lebih
baik. Ekspansi multi-kelas hanya dipakai evaluator deteksi; counting dan linker
tetap menerima satu proposal fisik, sehingga satu tandan tidak dihitung empat
kali.

**Sumber** — `../scripts/relabel_detektor_damimas.py` ·
`../scripts/fusi_proposal_damimas.py` · `../scripts/fusi_detektor_damimas.py` ·
`../results/damimas_relabel_classifier.json` ·
`../results/damimas_fusi_yolo_relabel.json`

---

## PT-E-022 — Linker global di atas proposal unik (2026-08-18)

**Hipotesis** — deduplikasi class-agnostic sebelum membentuk graf mengurangi
kompetitor palsu tanpa membuang distribusi kelas lunak. Pair model dipasang pada
proposal unik TRAIN dari C1; model/perakit/ambang dipilih pada proposal fusion
VAL; TEST proposal fusion baru dinilai setelah lock.

Pasangan train turun **225.918 -> 144.277**, sementara prevalensi pasangan benar
naik **2,763% -> 4,237%**. Average-link HGB ambang 0,70 menang di VAL.

| kepala VAL-locked | F1 test | presisi | recall | cakupan terdeteksi | cakupan semua | MAE pool |
|---|---:|---:|---:|---:|---:|---:|
| utility/F1 | **0,5171** | **0,5000** | 0,5354 | 0,6229 | 0,4546 | 3,672 |
| cakupan | 0,4977 | 0,4014 | **0,6549** | **0,7062** | **0,5155** | **1,864** |
| hitung-pool | 0,4816 | 0,3877 | 0,6356 | 0,6977 | 0,5093 | 1,880 |

Terhadap PT-E-020 dengan protokol DAMIMAS yang sama, kepala utility naik F1
**0,4631 -> 0,5171** dan cakupan-terdeteksi **0,5628 -> 0,6229**. Kepala
coverage melewati target 70% bila penyebutnya tandan yang terdeteksi; cakupan
atas seluruh tandan masih 51,55%, jadi target global belum tercapai. MAE pool
1,864 juga masih kalah dari regresor counting 1,004 dan tetap hanya menjadi
fitur, bukan hasil counting final.

**Sumber** — `scripts/linker_global_damimas.py` ·
`scripts/laporkan_kepala_linker_damimas.py` ·
`results/damimas_linker_global_proposal_yolo.json`

---

## PT-E-023 — Mixture-of-experts strict DAMIMAS (2026-08-18)

Empat classifier per-tandan strict mempunyai oracle-disagreement sekitar 83%
di VAL, tetapi meta-model 120 fitur tidak dapat memprediksi anggota yang benar
secara stabil. OOF GroupKFold per pohon memilih classifier klasik saja;
hasil test 0,7234 / macro-F1 0,7055, di bawah champion ConvNeXt 0,7378 / 0,7166.
Kepala per-tandan ini **DITOLAK**.

Pada tugas per-view, meta ordinal berbasis classifier klasik, ConvNeXt-224,
jumlah sisi, arah view, dan konteks pohon memberi test akurasi **0,7111** dan
macro-F1 **0,6894**. Akurasi hanya +0,08 pp dari klasik 0,7103, tetapi macro-F1
naik +1,01 pp; ia diterima sebagai kepala per-view sementara. Semua meta-model,
blend, dan ambang dipilih dari prediksi OOF VAL sebelum TEST dibaca.

**Sumber** — `scripts/moe_classifier_damimas.py` · `scripts/moe_view_damimas.py`
· `results/damimas_moe_classifier.json` · `results/damimas_moe_view.json`

---

## PT-E-024 — Propagasi confidence kelas lintas-view (2026-08-18)

**Hipotesis** — linker proposal-unik tidak hanya berguna untuk laporan
per-tandan. Evidence kelas dari view lain pada klaster prediksi yang sama dapat
dipropagasikan kembali ke confidence deteksi per-citra, sehingga memperbaiki
ranking COCO tanpa menciptakan kotak baru.

**Protokol** — baris deteksi routing PT-E-021 dipetakan ke proposal fisik unik.
Kepala linker, agregasi, kekuatan campuran, eksponen objectness/kelas, dan
score-blend disapu hanya pada 86 pohon VAL. Setelah konfigurasi global terkunci,
router per kelas mengambil konfigurasi terbaik untuk masing-masing AP kelas
(AP COCO memang separabel per kategori). B1/B3/B4 memilih kepala utility,
sedangkan B2 memilih kepala coverage. TEST baru dibuka setelah keempat rute
tetap.

| metrik test | PT-E-021 | propagasi | perubahan |
|---|---:|---:|---:|
| mAP50 | 0,5881 | **0,5965** | **+0,84 pp** |
| mAP50-95 | 0,2723 | **0,2743** | **+0,20 pp** |
| macro-F1 operasional | 0,5759 | **0,5906** | **+1,47 pp** |
| AP50 B1/B2/B3/B4 | 0,7923/0,4942/0,6554/0,4106 | **0,8042/0,5035/0,6570/0,4214** | semua naik |

Validation juga bergerak searah: mAP50 **0,5881 -> 0,6024** dan mAP50-95
**0,2716 -> 0,2774**. Audit invariant atas 44.926 baris VAL dan 66.539 baris
TEST membuktikan delta koordinat maksimum 0, delta label maksimum 0, dan jumlah
baris identik. Hanya 5.714/7.751 confidence yang berubah. Dengan demikian gain
ini benar-benar berasal dari evidence multi-view, bukan penambahan proposal.

**Putusan** — **DITERIMA** sebagai kepala deteksi class-aware sementara. Ia
memberi gain pada seluruh kelas dan tiga metrik utama sekaligus. Jalur proposal
fisik/linker/counting tetap terpisah agar empat hipotesis kelas tidak pernah
menjadi empat objek.

**Sumber** — `../scripts/propagasi_multiview_damimas.py` ·
`../results/damimas_propagasi_multiview.json` ·
`../results/pred_damimas_propagasi_multiview_{val,test}.npz`

---

## PT-E-025 — Evaluasi end-to-end global satu-ke-satu (2026-08-18)

**Tujuan** — mengganti angka classifier strict/oracle dengan evaluasi deploy
yang benar-benar memakai proposal dan cluster prediksi. Pool dipasangkan ke GT
secara Hungarian satu-ke-satu hanya pada lapisan evaluator. Pool tak terpasang
menjadi FP dan tandan tak terpasang menjadi FN; satu pool tidak boleh mengklaim
dua tandan.

Seluruh sumber probabilitas, kepala linker, threshold pool, aturan agregasi,
skema bobot, dan tau ordinal dipilih pada VAL. Konfigurasi final memakai
probabilitas PT-E-024, kepala linker coverage, threshold 0,15, serta R4 berbobot
confidence dengan tau `(0,40; 1,75; 2,50)`. TEST baru dibuka setelah lock.

| metrik fisik | VAL | TEST |
|---|---:|---:|
| precision pool | 0,8550 | **0,8530** |
| recall pool | 0,8215 | **0,8116** |
| akurasi kelas pada pool terpasang | 0,7576 | **0,7322** |
| macro-F1 kelas pada pool terpasang | 0,7356 | **0,7028** |
| correct-class recall atas seluruh GT | 0,6224 | **0,5942** |
| macro-F1 end-to-end (miss+FP dihitung) | 0,6240 | **0,5867** |
| MAE jumlah pool per pohon | 1,558 | **1,638** |

F1 end-to-end per kelas test adalah **0,7124 / 0,4823 / 0,6741 / 0,4781**
untuk B1--B4. Ini sengaja lebih rendah daripada angka strict karena sekaligus
memuat kegagalan deteksi, pemecahan/penggabungan identitas, klasifikasi, dan
pool palsu. Probabilitas hasil propagasi mengalahkan routing lokal dan proposal
C1 pada objective VAL, sehingga gain PT-E-024 bertahan setelah seluruh lapisan
disambungkan.

**Sumber** — `scripts/eval_endtoend_global_damimas.py` ·
`results/damimas_endtoend_global.json`

---

## PT-E-026 — Counting multi-bank anchor + proposal + linker (2026-08-18)

**Hipotesis** — statistik proposal unik dan klaster linker membawa informasi
counting yang tidak ada pada dump anchor 1.683-dim. Menggabungkan bank anchor,
proposal, dan 321 fitur linker seharusnya menurunkan macro-MAE terhadap kepala
anchor 1,0039 tanpa memakai jumlah pool sebagai hitungan langsung.

**Protokol** — fitur TRAIN/VAL dibangun lebih dulu. Seluruh 13 keluarga model
baseline ditambah PLS dijalankan pada ruang anchor 1.683, proposal 1.683,
concat 3.366, dan concat+linker 3.687 dimensi. Kepala per kelas, kalibrasi,
kepala total, dan rekonsiliasi dipilih di 86 pohon VAL. TEST 127 pohon baru
dibuka setelah lock tercetak. Fitur linker hanya memakai cluster prediksi;
`bid` dan kelas GT cache tidak pernah menjadi fitur. Run full memakai source
yang tersimpan pada commit `0de9fa0`; generalisasi multi-bank berikutnya
dikerjakan sesudah hasil ini ditutup.

| konfigurasi | split | macro-MAE | class ±1 | tree ±1 | total-MAE |
|---|---|---:|---:|---:|---:|
| anchor champion lama | VAL | 0,8459 | 0,7849 | 0,4186 | 1,6163 |
| compact multi-bank | VAL | 0,8256 | 0,7936 | **0,4535** | **1,3488** |
| full multi-bank | VAL | **0,8110** | **0,8081** | **0,4535** | 1,4767 |
| anchor champion lama | TEST | **1,0039** | **0,7579** | **0,3228** | 1,8583 |
| compact multi-bank | TEST | 1,0433 | **0,7579** | 0,3150 | **1,7795** |
| full multi-bank | TEST | 1,0374 | **0,7579** | 0,3071 | 1,8504 |

Full search mereproduksi kandidat anchor lama sampai empat desimal sebelum
menguji bank baru, sehingga regresi bukan akibat baseline yang hilang. Meski
lock full terlihat kuat di VAL, gain macro tidak bertransfer: 0,8110 menjadi
1,0374 di TEST. B3 tetap bottleneck (MAE 1,5276). Fitur tambahan membantu
seleksi validation tetapi meningkatkan variance pada hanya 641 pohon train.

**Putusan** — hipotesis gain macro **DIPALSUKAN**; kepala macro tetap ensemble
anchor 1,0039. Varian compact **DITERIMA HANYA sebagai kepala total khusus**
karena total-MAE turun 1,8583 → 1,7795, sementara angka macro/tree tidak boleh
diatribusikan kepadanya. Full dan compact disimpan terpisah agar hasil negatif
tidak tertimpa.

**Sumber** — `scripts/counting_multibank_damimas.py` ·
`results/damimas_counting_multibank_{compact,full}.json` ·
`runs/counting_multibank_damimas/ensemble_{compact,full}.joblib`

---

## PT-E-028 — CatBoost regularized dengan seleksi OOF+VAL (2026-08-18)

**Hipotesis** — regularisasi kuat dan pemilihan kepala dari prediksi OOF TRAIN
ditambah VAL bersih dapat mengurangi variance PT-E-026 dan menurunkan
macro-MAE counting terhadap champion anchor 1,0039. Resep CatBoost ditetapkan
sebelum run: MultiRMSE 500 iterasi, depth 5, learning rate 0,035,
`l2_leaf_reg=20`, `rsm=0,25`, dan 5 fold.

**Protokol** — empat ruang fitur (`anchor`, `proposal`, `concat`, dan
`concat_linker`) menghasilkan prediksi OOF TRAIN serta prediksi VAL. Kepala per
kelas, kalibrasi, kepala jumlah-total, dan rekonsiliasi dikunci pada gabungan
OOF+VAL. Konfigurasi tercetak sebelum cache, label, serta fitur TEST dibuka.
Model terpakai kemudian dipasang ulang pada TRAIN+VAL tanpa mengubah lock.

| keluaran empat kelas | OOF TRAIN+VAL | VAL bersih | TEST |
|---|---:|---:|---:|
| macro-MAE | 0,9006 | 0,9244 | 1,0236 |
| class ±1 | 0,7861 | 0,7791 | 0,7480 |
| tree ±1 | 0,3920 | 0,4186 | 0,3386 |
| MAE jumlah empat kepala | 1,5915 | 1,5581 | 1,7323 |

CatBoost tidak mengganti champion macro 1,0039 ataupun class ±1 0,7579.
Tree ±1 naik dari ensemble anchor 0,3228 menjadi 0,3386, tetapi masih di bawah
single-model 0,3780. Hipotesis gain macro karena itu **DIPALSUKAN**; dump dan
model tetap disimpan sebagai kandidat diversity untuk stacker final.

**Koreksi terminologi PT-E-026** — rekonsiliasi seluruh run terkunci pada mode
`raw`. Akibatnya kolom `total-MAE` PT-E-026 dan angka CatBoost 1,7323 adalah MAE
penjumlahan empat kepala kelas, bukan metrik regresor jumlah-total yang sudah
dipilih terpisah. Audit inference-only atas model yang telah terkunci memberi:

| kepala jumlah-total langsung | VAL saat seleksi | TEST audit |
|---|---:|---:|
| baseline anchor | 1,3837 | 1,5669 |
| compact multi-bank | 1,3605 | **1,4882** |
| full multi-bank | **1,3140** | 1,5276 |
| CatBoost | 1,3721 | 1,5512 |

Compact 1,4882 adalah hasil TEST terbaik yang teramati, tetapi ranking audit
TEST tidak dijadikan lock final. Full menang pada VAL, sementara stacker
counting final akan dipilih dari OOF/VAL setelah bank RF-DETR/RT-DETR lengkap.
Prediksi per-pohon dan hash model disimpan agar koreksi dapat dihitung ulang
tanpa fitting atau pemilihan ulang.

**Sumber** — source run `edfeb5c` ·
`scripts/counting_catboost_damimas.py` ·
`scripts/audit_counting_total_damimas.py` ·
`results/damimas_counting_catboost.json` ·
`results/damimas_counting_total_head_audit.json` ·
`results/damimas_counting_total_head_audit_pred.npz`
