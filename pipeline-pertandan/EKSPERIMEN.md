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
