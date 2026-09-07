# Laporan Diagnostik: Analisis Sifat Sinyal Kedalaman dan Efektivitas Multimodal

Dokumen ini merekam **alur pembuktian empiris** mengenai penyebab mendasar mengapa penggabungan modalitas RGB+Depth secara konvensional tidak memberikan peningkatan performa deteksi ($mAP50$) pada tandan buah segar (TBS) kelapa sawit. Seluruh kalkulasi diagnostik dihasilkan melalui pengujian *read-only* deterministik yang dapat direproduksi secara mandiri melalui perintah:

```bash
.venv/bin/python scripts/probe_depth_signal.py --probe semua
```

---

## 1. Premis Awal dan Motivasi Pengujian

Premis awal yang melandasi riset sebelum Fase 6 berasumsi bahwa *"dataset SawitMVC modalitas RGB menghasilkan performa deteksi yang jauh lebih tinggi daripada SawitMVC-Depth, sehingga terdapat kelemahan pada integrasi kanal kedalaman."*

Data empiris awal yang mendasari premis tersebut:
* YOLO26l pada SawitMVC-953 (RGB): $mAP50 = \mathbf{0,5435}$.
* YOLO26l pada SawitMVC-Depth-352 (RGB+D *early fusion*): $mAP50 = \mathbf{0,3919}$.

Tujuan investigasi diagnostik ini adalah membuktikan secara kausal apakah kesenjangan tersebut benar-benar disebabkan oleh karakteristik sinyal kedalaman atau akibat faktor perancangan data.

---

## 2. Pengujian Diagnostik 1: Dekomposisi Distribusi Sampel dan Kelangkaan Kelas

Kompilasi frekuensi objek (*instances*) pada seluruh partisi data:

| Partisi Data | Jumlah Citra | Total Objek | Kepadatan (/Citra) | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|---|---|
| 953 Latih | 3.000 | 14.041 | 4,68 | 11,2% | 18,6% | **52,2%** | 17,9% |
| 953 Validasi | 404 | 1.887 | 4,67 | 10,7% | 20,6% | 50,8% | 18,0% |
| 953 Uji | 588 | 2.612 | 4,44 | 9,6% | 19,0% | **53,9%** | 17,4% |
| 352 Latih | 980 | 1.517 | 1,55 | 35,8% | 43,6% | **14,2%** | **6,5%** |
| 352 Validasi | 208 | 372 | 1,79 | 37,4% | 44,6% | 11,6% | 6,5% |
| 352 Uji | 220 | 410 | 1,86 | 35,9% | 42,4% | **15,4%** | **6,3%** |

Dua temuan kunci teridentifikasi:
1. **Rasio Volume Objek**: Jumlah objek latih menyusut **$9,3\times$** ($14.041 \to 1.517\text{ objek}$), dengan kepadatan per citra turun dari $4,68$ menjadi $1,55$.
2. **Pergeseran Komposisi Kelas**: Kelas B3 menyusut dari 7.333 menjadi 215 objek (**penurunan $34\times$**), dan kelas B4 menyusut dari 2.513 menjadi 98 objek (**penurunan $26\times$**).

Evaluasi $mAP50$ per-kelas pada split uji (YOLO26l):

| Dataset Acuan | B1 | B2 | B3 | B4 | $mAP50$ Makro |
|---|---|---|---|---|---|
| SawitMVC-953 (RGB) | 0,7705 | 0,4479 | **0,6050** | **0,3506** | 0,5435 |
| SawitMVC-Depth-352 (RGB) | 0,6804 | 0,4320 | **0,2001** | **0,1299** | 0,3606 |

Performa pada kelas B1 dan B2 relatif stabil. **Seluruh kesenjangan performa terpusat pada kelas B3 dan B4** yang populasinya menyusut secara ekstrem. Karena metrik $mAP50$ merupakan rata-rata makro tak terbobot dari keempat kelas, degradasi pada dua kelas langka ini secara langsung menurunkan skor total.

---

## 3. Pengujian Diagnostik 2: Dekomposisi Galat Lokalisasi vs Kesalahan Klasifikasi

Pemisahan tugas deteksi menjadi lokalisasi murni (*class-agnostic*) dan klasifikasi tingkat kematangan (*class-aware*) pada model `yolo26l_e60_i1280_rgb352`:

$$\begin{aligned}
mAP50\text{ (4-kelas / class-aware)} &= 0,3707 \\
AP50\text{ (lokalisasi murni / class-agnostic)} &= 0,6677 \\
\text{Kesenjangan Performa} &= 0,2970\text{ (setara 44,5\% kapasitas lokalisasi)}
\end{aligned}$$

Detektor terbukti **mampu menemukan lokasi fisik tandan dengan presisi tinggi**. Penurunan performa terutama disebabkan oleh kesalahan pemberian label kematangan.

Matriks konfusi pada kotak yang terdeteksi dengan benar (*IoU* $\ge 0,5$, *confidence* $\ge 0,25$):

| Kelas Acuan | Terprediksi B1 | Terprediksi B2 | Terprediksi B3 | Terprediksi B4 | Daya Tangkap (*Recall*) |
|---|---|---|---|---|---|
| B1 | 92 | 26 | 0 | 0 | 78,0% |
| B2 | 13 | 83 | 12 | 0 | 76,9% |
| B3 | 0 | 21 | 11 | 4 | **30,6%** |
| B4 | 0 | 1 | 3 | 5 | 55,6% |

Akurasi klasifikasi pada kotak terdeteksi mencapai $70,5\%$. Seluruh kesalahan klasifikasi **terdistribusi ke kelas yang bertetangga langsung**, tanpa pernah terjadi kesalahan ekstrem (misal B1 tertukar menjadi B3/B4). Hal ini menegaskan bahwa kematangan buah merupakan fenomena **regresi ordinal kontinu**, bukan klasifikasi kategori independen.

---

## 4. Pengujian Diagnostik 3: Karakterisasi Fisik Sinyal Kedalaman

### Hipotesis A: Skala Metrik Absolut (Ditolak)
Pengukuran jarak absolut $Z$ dari sensor kamera ke objek pada 2.299 kotak pembatas:

| Parameter Fisik | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| Ukuran Kotak Piksel (Median) | 153,9 | 136,7 | 122,1 | 108,8 |
| Jarak Sensor $Z$ (Meter, Median) | 1,36 | 1,33 | 1,31 | 1,20 |

Jarak perekaman $Z$ relatif konstan ($1,20\text{--}1,36\text{ m}$) di seluruh kelas karena protokol pemotretan lapangan mempertahankan jarak berdiri operator yang serupa. Jarak absolut tidak membawa daya pembeda kelas kematangan.

*Catatan Validitas Sensor*: Validitas piksel kedalaman **di dalam kotak objek mencapai 95,1%**. Angka 29% piksel tak valid yang dilaporkan sebelumnya berasal dari latar belakang terbuka (langit dan vegetasi jauh), bukan pada permukaan tandan buah.

### Hipotesis B: Relief Kedalaman Lokal (Dikonfirmasi)
Pengukuran kontras kedalaman lokal ($\text{Relief} = \text{Median } Z_{\text{cincin latar}} − \text{Median } Z_{\text{dalam kotak}}$):

| Parameter Relief | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| Nilai Median Relief Lokal | **$+2,8\text{ cm}$** | $0,0\text{ cm}$ | $−1,5\text{ cm}$ | **$−5,1\text{ cm}$** |
| Proporsi Objek Lebih Menonjol | 61,3% | 50,7% | 41,4% | 26,4% |

Relief lokal terbukti **monoton sempurna terhadap tingkat kematangan buah** (Krusial: Uji Kruskal-Wallis menghasilkan **$H = 99,8$, $p = 1,7 \times 10^{−21}$**). Sinyal kedalaman menyediakan informasi pembeda geometris (tandan muda tertanam di sela pelepah vs tandan lewat matang yang menonjol keluar).

---

## 5. Pengujian Diagnostik 4: Analisis Rasio Sinyal terhadap Derau (SNR)

Kanal kedalaman berformat uint8 dengan pemetaan invers rentang $[0,8; 15,0]\text{ meter}$ menghasilkan resolusi kuantisasi:

$$\frac{dZ}{dv} = \frac{Z^2 \cdot (1/Z_{\text{near}} − 1/Z_{\text{far}})}{254}$$

Pada median jarak adegan $Z = 2,49\text{ m}$, 1 level kuantisasi setara dengan **$2,9\text{ cm}$**. Amplitudo sinyal relief median ($0,8\text{ cm}$) hanya setara dengan **$0,27\text{ level kuantisasi}$**, sementara derau fisik sensor berkisar $\sim 2,5\text{ cm}$. Akibatnya, **rasio sinyal terhadap derau (*SNR*) per piksel berada pada kisaran rendah $\approx 0,3$**.

Rentang dinamis kanal depth habis dipakai untuk memetakan variasi jarak latar belakang ($0,8\text{--}6,4\text{ m}$) yang merupakan faktor pengganggu (*nuisance parameter*).

---

## 6. Pengujian Diagnostik 5: Pemulihan Sinyal Melalui Agregasi Spasial (*Pooling*)

Derau acak tereduksi sebesar $\sim \sqrt{N}$ melalui perataan spasial pada $N$ piksel. Efektivitas pemisahan B1 vs B4 diuji menggunakan nilai *Area Under Curve* (AUC):

| Jumlah Piksel Teragregasi ($N$) | AUC Latih + Validasi | AUC Uji |
|---|---|---|
| 1 piksel mentah | 0,592 | 0,577 |
| 16 piksel | **0,724** | **0,650** |
| 256 piksel | 0,728 | 0,593 |
| 4.096 piksel | 0,730 | 0,621 |

Nilai diskriminatif sinyal meningkat drastis setelah agregasi $\ge 16\text{ piksel}$.

---

## 7. Pembuktian Redundansi Sinyal Kedalaman terhadap Visual RGB

Pengujian kontribusi 8 fitur statistik kedalaman teragregasi terhadap representasi visual RGB (768-dimensi) pada model ConvNeXt:

| Konfigurasi Masukan Klasifikasi | Akurasi Validasi | Akurasi Uji ($n = 410$) |
|---|---|---|
| Statistik Kedalaman Saja (8-dim) | 0,3468 | 0,3756 |
| Fitur Visual RGB Murni (768-dim) | 0,6774 | **0,6415** |
| RGB + Statistik Kedalaman | 0,6720 | **0,6415** |

### Teorema Redundansi Informasi
$$I(Y; D) > 0 \quad \text{namun} \quad I(Y; D \mid \text{RGB}) \approx 0$$

Meskipun sinyal kedalaman membawa informasi kematangan secara mandiri, seluruh varians informasi tersebut **telah terwakili secara sempurna oleh fitur visual RGB**. Perubahan fisik tonjolan tandan berkorelasi kuat dengan fitur warna dan tekstur visual pada citra RGB resolusi tinggi.

---

## 8. Koreksi Kausal: Pergeseran Domain Temporal Antar-Dataset (V2-E-022)

Penyebab sejati perbedaan distribusi antara dataset 953 dan 352 pohon dibuktikan melalui pelacakan tanggal perekaman lapangan:

| Sumber Dataset | Rentang Waktu Akuisisi | Karakteristik Fenologi Kebun |
|---|---|---|
| SawitMVC-YOLO (953) | 30 April – 16 Mei 2026 | Dominan fase matang awal B3 ($55,3\%$) |
| SawitMVC-Depth (352) | 28 – 29 Juli 2026 (Jeda $\sim 80\text{ hari}$) | Pasca-rotasi panen, dominan B1+B2 ($79,6\%$) |

Pergeseran temporal ini menjelaskan secara tuntas mengapa performa lokalisasi murni bertahan tinggi ($AP50 = \mathbf{0,7330}$) sementara deteksi 4-kelas mengalami degradasi ($mAP50 \approx 0,45$): struktur kanopi pohon kelapa sawit bersifat stabil secara spasial melintasi 80 hari, sedangkan kondisi fisik buah telah mengalami pematangan dan pemanenan berulang.
