# Diagnosis: Kenapa RGB+D Tidak Menaikkan mAP

Dokumen ini mencatat **jalan penemuannya**, bukan cuma kesimpulannya — supaya
tiap langkah bisa diperiksa ulang dan dibantah. Semua angka dihasilkan tanpa
melatih apa pun (probe read-only, hitungan menit) dan bisa direproduksi dengan:

```bash
.venv/bin/python scripts/probe_depth_signal.py --probe semua
```

Ditulis 2026-08-11, sebagai dasar Fase 6.

---

## 0. Premis yang mau diuji

Premis yang dipegang sebelum ini: *"dataset SawitMVC tanpa depth, walau dipotong
jadi 25%, tetap jauh di atas SawitMVC+Depth — jadi ada yang salah dengan depth."*

Angka yang memicunya nyata: YOLO26l pada 953 pohon dapat test mAP50 **0,5435**,
sementara pada 352 pohon RGB+D cuma **0,3919**. Selisihnya besar dan konsisten.

Pertanyaannya: apakah selisih itu benar-benar disebabkan depth?

---

## 1. Probe pertama: bandingkan isi kedua dataset, bukan cuma jumlah pohonnya

Yang biasa dikutip adalah jumlah pohon (953 vs 352, rasio 2,7×). Tapi mAP
dihitung dari **instance**, bukan pohon. Jadi saya hitung ulang seluruh file
label:

| Split | citra | instance | /citra | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|---|---|
| 953-train | 3.000 | 14.041 | 4,68 | 11,2% | 18,6% | **52,2%** | 17,9% |
| 953-val | 404 | 1.887 | 4,67 | 10,7% | 20,6% | 50,8% | 18,0% |
| 953-test | 588 | 2.612 | 4,44 | 9,6% | 19,0% | **53,9%** | 17,4% |
| 352-train | 980 | 1.517 | 1,55 | 35,8% | 43,6% | **14,2%** | **6,5%** |
| 352-val | 208 | 372 | 1,79 | 37,4% | 44,6% | 11,6% | 6,5% |
| 352-test | 220 | 410 | 1,86 | 35,9% | 42,4% | **15,4%** | **6,3%** |

Dua hal langsung terlihat:

1. Rasio instance bukan 2,7× tapi **9,3×** (14.041 vs 1.517) — kepadatan objek
   per citra turun dari 4,68 ke 1,55.
2. Komposisi kelasnya **terbalik**. B3 turun dari 7.333 ke 215 instance
   (**34× lebih sedikit**), B4 dari 2.513 ke 98 (**26×**).

Sekarang lihat mAP50 dipecah per kelas (YOLO26l, test split):

| | B1 | B2 | B3 | B4 | mAP50 |
|---|---|---|---|---|---|
| 953-RGB | 0,7705 | 0,4479 | **0,6050** | **0,3506** | 0,5435 |
| 352-RGB | 0,6804 | 0,4320 | **0,2001** | **0,1299** | 0,3606 |

B1 dan B2 nyaris sama. **Seluruh gap ada di B3 dan B4** — persis dua kelas yang
instance-nya menghilang. Karena mAP50 itu rata-rata makro empat kelas, dua kelas
yang kelaparan langsung menyeret separuh skor.

**Kesimpulan probe 1:** gap 953-vs-352 adalah efek kelangkaan label, bukan efek
depth. Dan konsekuensinya untuk premis awal: memotong dataset 953 jadi 25% tetap
menyisakan ~1.800 instance B3 (vs 215 di dataset depth) dengan komposisi kelas
yang sama — jadi "RGB 25% tetap menang" adalah hasil yang **diharapkan** dan
tidak menguji apa pun tentang depth.

Perbandingan yang sah cuma di dalam split 352 yang sama. Di situ, untuk YOLO26l,
RGB+D justru **di atas** RGB: 0,3919 (`inverse`) dan 0,4316 (`edge`) vs 0,3606.

---

## 2. Probe kedua: sebenarnya yang rusak itu mencari tandan, atau menamainya?

mAP50 mencampur dua kemampuan berbeda: menemukan objek (lokalisasi) dan memberi
kelas yang benar. Keduanya bisa dipisah dengan mengevaluasi bobot yang sama dua
kali — sekali normal, sekali dengan semua kelas dilipat jadi satu.

Diukur pada `runs/yolo26l_e60_i1280_rgb352/weights/best.pt`, test split.
(Reimplementasi divalidasi dulu: mAP50 saya 0,3707 vs pycocotools 0,3711.)

```
mAP50 class-aware      = 0,3707
AP50  class-agnostic   = 0,6677     <- lokalisasi murni
selisih                = 0,2970     = 44,5% dari kemampuan lokalisasi
```

Detektornya **menemukan tandan dengan baik**. Yang hangus adalah penamaan kelas.

Konfusi pada box yang sudah benar lokasinya (IoU≥0,5, conf≥0,25):

| | →B1 | →B2 | →B3 | →B4 | recall |
|---|---|---|---|---|---|
| B1 | 92 | 26 | 0 | 0 | 78,0% |
| B2 | 13 | 83 | 12 | 0 | 76,9% |
| B3 | 0 | 21 | 11 | 4 | **30,6%** |
| B4 | 0 | 1 | 3 | 5 | 55,6% |

Akurasi klasifikasi 70,5%. Perhatikan polanya: **semua kesalahan jatuh ke kelas
bertetangga**, nol kasus B1→B3 atau B1→B4. Kematangan itu kontinum, jadi ini
masalah **ordinal**, bukan klasifikasi 4-arah sembarang.

Catatan kejujuran: 70,5% itu bersyarat pada box yang berhasil dideteksi
(271 dari 410). Kalau yang tidak terdeteksi dihitung salah, akurasi atas seluruh
GT = 191/410 = **46,6%**.

---

## 3. Probe ketiga: apakah depth membawa informasi kelas sama sekali?

Ini pertanyaan intinya. Saya uji dua hipotesis berbeda.

### Hipotesis A — skala metrik (GAGAL)

Geometri pinhole itu eksak: `d_piksel = f · D_metrik / Z`. Citra tunggal tidak
bisa memulihkan ukuran fisik (ambiguitas skala monokuler), tapi dengan depth
bisa: `D = d · Z / f`. Kalau tandan B1..B4 beda ukuran fisik, ukuran metrik
harus memisahkan kelas lebih baik daripada ukuran piksel.

Hasil (2.299 box):

| | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| ukuran piksel (median) | 153,9 | 136,7 | 122,1 | 108,8 |
| Z (m, median) | 1,36 | 1,33 | 1,31 | 1,20 |

**Z hampir konstan lintas kelas.** Protokol pengambilan foto memang jarak tetap
(operator berdiri di jarak yang mirip), jadi mengalikan dengan Z cuma menggeser
skala — tidak menambah daya pisah. Hipotesis A gugur.

Bonus temuan: cakupan depth **di dalam box = 95,1% valid**. Angka "29% piksel
invalid" yang selama ini dikutip itu **latar** (langit, pohon jauh), bukan objek.
Jadi narasi "depth rusak karena banyak lubang di tandan" tidak berlaku.

### Hipotesis B — relief lokal (BERHASIL)

Kalau jarak absolut tidak informatif, mungkin yang informatif adalah **kontras
kedalaman antara tandan dan sekelilingnya**: tandan matang menonjol keluar dari
pelepah, tandan muda tertanam ke dalam.

Diukur sebagai `relief = median Z(cincin sekitar) − median Z(dalam box)`:

| | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| relief (median) | **+2,8 cm** | 0,0 cm | −1,5 cm | **−5,1 cm** |
| lebih dekat dari sekitar | 61,3% | 50,7% | 41,4% | 26,4% |

**Monoton sempurna terhadap kematangan.** Kruskal-Wallis 4 kelas:
**H = 99,8, p = 1,7×10⁻²¹**.

Ini konsisten dengan hipotesis F-002 yang sudah tercatat di Volume 1: depth
membantu untuk pembedaan **geometris** (B4 kecil/tertanam/tertutup pelepah),
bukan untuk ambiguitas **fotometrik** (warna).

---

## 4. Probe keempat: kenapa sinyal sekuat itu tidak terpakai?

Sinyalnya ada (p≈10⁻²¹), tapi amplitudonya perlu dibandingkan dengan resolusi
kanal yang dipakai untuk mengangkutnya.

Encoding yang dipakai sejak Volume 1: uint8, inverse depth, rentang tetap
`[Z_NEAR=0,8; Z_FAR=15,0]` m. Turunkan besar satu level dalam meter:

```
v = 1 + 254 · (1/Z − 1/Z_FAR) / (1/Z_NEAR − 1/Z_FAR)
dZ/dv = Z² · (1/Z_NEAR − 1/Z_FAR) / 254
```

| Z (m) | 1 level uint8 |
|---|---|
| 1,0 | 0,5 cm |
| 1,5 | 1,0 cm |
| 2,0 | 1,9 cm |
| **2,5** | **2,9 cm** |
| 3,0 | 4,2 cm |
| 4,0 | 7,5 cm |

Median Z per citra di dataset ini 2,49 m. Jadi satu level ≈ **2,9 cm**,
sementara sinyal relief median cuma **0,8 cm** — yaitu **0,27 level**. Bahkan
B4 (5,1 cm) hanya 1,8 level. Ditambah derau sensor Orbbec (~1% dari Z ≈ 2,5 cm),
**SNR per-piksel ≈ 0,3**.

Yang menarik: kanal depth-nya sendiri *terpakai penuh* (entropi 7,68 dari 8 bit,
p25–p75 membentang 114 level). Rentang dinamisnya habis untuk menggambarkan
**ramp global adegan** — tanah, batang, latar 0,8–6,4 m — yang justru
**nuisance**: median Z per citra bervariasi mean 2,49 m, std 0,82 m, rentang
0,80–6,44 m, mengikuti di mana operator berdiri, bukan mengikuti kematangan.

Jadi kanal depth membawa: amplitudo besar yang tidak relevan + sinyal relevan
yang sub-kuantum. Itu resep bagus untuk model belajar korelasi semu.

---

## 5. Probe kelima: sinyalnya bisa diselamatkan dengan pooling

Derau turun ~√N saat dirata-rata atas N piksel. Karena itu sinyal yang tenggelam
per-piksel bisa muncul setelah pooling wilayah. Diuji dengan AUC memisahkan
B1 vs B4 memakai relief yang dihitung dari N piksel acak:

| piksel di-pool | AUC train+val | AUC test |
|---|---|---|
| 1 | 0,592 | 0,577 |
| 16 | **0,724** | 0,650 |
| 256 | 0,728 | 0,593 |
| 4.096 | 0,730 | 0,621 |

Naik tajam dari 1 ke 16 piksel, lalu jenuh.

---

## 6. Sintesis: satu penjelasan yang menutup semuanya

> Sinyal depth di dataset ini bersifat **relatif, lokal, dan hanya terbaca
> setelah pooling wilayah**.

Konsekuensinya:

- **Early fusion di stem adalah tempat paling buruk.** Di sana resolusinya
  penuh dan poolingnya minimum — persis rezim ber-SNR 0,3. Ini menjelaskan
  kenapa E-022, E-027, E-032, V2-E-005/006 semuanya gagal dengan pola yang sama.
- **Kenapa `edge` (Sobel depth) satu-satunya yang menang** (V2-E-008/010):
  operator turunan **membuang ramp global** — yang nuisance — dan menonjolkan
  relief lokal. Jadi teorinya bukan cuma menjelaskan kegagalan, tapi juga
  meretrodiksi satu-satunya keberhasilan.
- **Sinyalnya cocok dengan lubang yang ada.** Probe 2 bilang yang rusak adalah
  klasifikasi ordinal; probe 3 bilang depth membawa sinyal ordinal. Diagnosis
  dan obatnya nyambung.

### Yang keliru dalam pemahaman sebelumnya

| Pemahaman lama | Hasil pengukuran |
|---|---|
| "Gap 953-vs-352 karena depth" | Karena B3/B4 34×/26× lebih langka |
| "29% piksel invalid merusak sinyal di tandan" | Di dalam box, depth 95,1% valid |
| "Depth memberi skala metrik" | Z hampir konstan (1,20–1,36 m), tidak memisahkan |
| "Depth harus di-fusi lebih pintar di backbone" | Harus dikonsumsi setelah pooling, di jalur klasifikasi |

Catatan: rentang `[0,8; 15,0]` dipilih di Volume 1 dengan memaksimalkan
**entropi seluruh citra**. Itu objektif yang keliru untuk tugas ini — ia
mengoptimalkan deskripsi langit dan pohon jauh, padahal yang dibutuhkan adalah
resolusi pada skala objek.

---

## 7. Uji akhir: ternyata sinyalnya nyata tapi REDUNDAN

Diagnosis di atas memberi resep jelas — konsumsi depth sebagai relief lokal
setelah pooling wilayah, di jalur klasifikasi. Itu dikerjakan (`V2-E-015/016`),
dan hasilnya menutup pertanyaannya dengan cara yang tidak diduga.

**Uji 1 — cabang CNN depth, 3 seed** (relief + mask valid, difusikan setelah
global pooling, gate init taknol, plus loss auxiliary RGB-only):

| seed | Δ val | Δ test |
|---|---|---|
| 101 | −0,0430 | −0,0341 |
| 202 | −0,0242 | −0,0463 |
| 303 | +0,0242 | +0,0195 |

Rata-rata −1,4pp val (p=0,55), −2,0pp test (p=0,42). Gate berhenti di ~0,11
dari init 0,10 — model tidak membuka jalur depth. Catatan: **satu seed sempat
memberi +5,9pp**, persis besaran yang kalau dilaporkan sendirian akan terbaca
sebagai kemenangan.

**Uji 2 — depth diberi kondisi paling menguntungkan.** Uji 1 bisa dibantah:
cabang CNN menaruh pooling di paling akhir, padahal §5 bilang pooling-lah yang
menyelamatkan sinyal. Jadi diuji ulang dengan 8 statistik depth yang **sudah
terpool secara analitik** (relief cincin−box, median, std, cakupan, rentang
persentil), ditempel langsung ke fitur penultimate classifier RGB terlatih:

| Fitur | val | test |
|---|---|---|
| statistik depth saja (8 dim) | 0,3468 | 0,3756 |
| RGB saja (768 dim) | 0,6774 | 0,6415 |
| RGB + statistik depth | 0,6720 | 0,6415 |

Sinyal relief terverifikasi masih utuh di crop (B1 +1,34 cm → B4 −4,29 cm,
monoton). Depth sendirian jelas di atas tebakan acak. **Kontribusi di atas RGB:
−0,5pp val, +0,0pp test.**

### Pernyataan akhirnya

> `I(Y;D) > 0` **tetapi** `I(Y;D | RGB) ≈ 0`.

Depth membawa informasi kematangan, tapi informasi itu sudah seluruhnya
terkandung di RGB. Penjelasan fisiknya sederhana: tandan yang menonjol keluar
dari pelepah juga **terlihat** besar dan matang di RGB — relief adalah *akibat*
dari variabel laten yang sama (kematangan/ukuran), bukan pengukuran independen
atasnya.

Ini **batas informasi, bukan kegagalan implementasi**. Kalau `I(Y;D|RGB) ≈ 0`,
risiko Bayes model RGB-D sama dengan model RGB, dan setiap parameter tambahan
hanya menambah error estimasi. Satu pernyataan ini menjelaskan seluruh
rangkaian hasil nol RGB-D di kedua volume — E-022, E-027, E-032, V2-E-005/006,
V2-E-009 — dan memprediksi percobaan fusi berikutnya juga akan nol.

**Batas klaim (jangan digeneralisasi):** ini berlaku untuk **klasifikasi
kematangan** pada protokol data ini (standoff hampir tetap, Z per kelas
1,20–1,36 m; depth uint8; 352 pohon). Kontribusi depth untuk **lokalisasi**
diuji terpisah — seluruh eksperimen sebelumnya mencampur kedua tugas sehingga
tidak pernah bisa menjawabnya.

## 8. Batas alat ukur

Test split 352 cuma 410 box, dengan **B4 = 26**. Selisih kecil pada mAP50 tidak
bisa dibedakan dari derau: pada Fase 5, val dan test bahkan berlawanan arah
(RGB unggul di val 0,4111 vs 0,3856; `edge` unggul di test). Multi-seed di §7
menunjukkan hal yang sama pada classifier — variasi antar-seed ±2-3pp, cukup
untuk memalsukan "kemenangan" apa pun yang dilaporkan dari satu run.

Angka apa pun dari split ini harus dibaca dengan itu di kepala.

---

## 9. Koreksi (2026-08-12): sebab yang saya tulis di Probe 1 salah

Seluruh dokumen di atas dibiarkan apa adanya karena jalan penemuannya memang
begitu. Bagian ini mengoreksi satu kesimpulan yang ternyata keliru, dan
koreksinya mengubah arti hampir semua yang menyusul.

**Yang saya tulis di Probe 1.** B3 turun dari 7.333 ke 215 instance (34×) dan B4
dari 2.513 ke 98 (26×) ketika berpindah dari dataset 953 ke 352, lalu saya
simpulkan itu **efek kelangkaan label** akibat dataset yang lebih kecil.

**Yang sebenarnya terjadi.** Angkanya benar, sebabnya salah. Kedua dataset
memakai tree ID yang sama untuk 352 pohon DAMIMAS, jadi saya bandingkan
labelnya langsung pada 1.408 citra ber-ID sama
(`scripts/probe_pergeseran_temporal.py`):

| Sumber label | Total kotak | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|
| SawitMVC-YOLO (953) | 6.523 | 566 (8,7%) | 1.098 (16,8%) | 3.604 (55,3%) | 1.255 (19,2%) |
| SawitMVC-Depth (352) | 2.299 | 829 (36,1%) | 1.001 (43,5%) | 321 (14,0%) | 148 (6,4%) |

Pada **pohon yang sama**, B3 berbeda 11,2× dan B4 8,5×. Sebabnya ada di
metadata akuisisi: dataset 953 direkam **30 April – 16 Mei 2026**, dataset 352
direkam **28–29 Juli 2026**. Jeda ~80 hari, dengan rotasi panen sawit 7–15 hari.

Jadi ini bukan dataset kecil versus dataset besar. Ini **kebun yang sama pada
fase kematangan yang berbeda**. Kohort B3 yang dominan pada Mei sudah matang
jadi B1/B2 pada Juli, sebagian sudah dipanen — konsisten dengan turunnya total
kotak dan bergesernya distribusi ke 80% B1+B2.

**Kenapa ini mengubah arti seluruh Fase 6.** Rangkaian pretrain 953 → finetune
352 yang jadi tulang punggung Fase 6 bukan transfer di dalam satu domain,
melainkan transfer melintasi pergeseran domain temporal. Model belajar dunia
yang 55% B3, lalu diuji di dunia yang 14% B3.

**Dan ini menjawab pertanyaan yang menggantung sejak Probe 2:** kenapa mencari
tandan berhasil (AP50 agnostik 0,7330) tapi menamainya rusak (mAP50 0,45)?
Karena label lokalisasi bertahan melintasi 80 hari — posisi tandan di kanopi
relatif stabil — sementara label kematangan tidak, karena benda fisiknya
berubah. Ketimpangan antara detektor dan classifier adalah sifat dari pasangan
data yang dipakai, bukan cacat arsitektur yang bisa diperbaiki dengan model
atau loss yang lebih baik.

Detail lengkap dan konsekuensinya: `experiments/EKSPERIMEN.md` entri
**V2-E-022**. Batas daya statistik yang membuat seluruh perbandingan Fase 6
tidak terbedakan: entri **V2-E-023**.
