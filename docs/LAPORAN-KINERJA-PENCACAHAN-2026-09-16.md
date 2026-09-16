# Laporan Kinerja Pencacahan: Kalibrasi Koefisien Pengali per Kelas

## 1. Identitas Eksperimen

| Parameter | Nilai |
|---|---|
| Identitas simpul | `V2-E-050` |
| Tanggal pelaksanaan | 16 September 2026 |
| Modalitas masukan | RGB, kelas kematangan B1–B4 |
| Model pencacahan | $\hat{y}_c(t) = \operatorname{round}\left(k_c \cdot n_c(t)\right)$, dengan $n_c(t)$ sebagai jumlah deteksi kelas $c$ pada seluruh sisi pohon $t$ yang memenuhi skor keyakinan (*confidence*) $\ge \tau_c$ |
| Fungsi koefisien | Koreksi bias sistematis pencacahan: menaikkan hitungan kelas yang mengalami pencacahan kurang (*under-count*) dan menurunkan hitungan kelas yang mengalami pencacahan berlebih (*over-count*) |
| Pelatihan ulang | Tidak dilakukan. Koefisien dipasang langsung dari *dump* prediksi `.npz` yang sudah terlacak |
| Partisi pemasangan | Partisi validasi (`953-val` sebanyak 96 pohon; `763-val` sebanyak 117 pohon) |
| Partisi pelaporan | Partisi uji (`953-test` sebanyak 141 pohon; `763-test` sebanyak 110 pohon, dengan irisan setara 66 pohon) |
| Ruang pencarian | $\tau \in [0,05; 0,90]$ langkah $0,05$; $k \in [0,20; 3,00]$ langkah $0,01$ |
| Kriteria pemasangan | $MAE$ minimum pada partisi validasi |
| Metrik peringkat | $MAE$ makro, yakni rerata $MAE$ per kelas B1–B4 |
| Detektor | YOLO26l, RT-DETR-L, RF-DETR-L, masing-masing satu bobot terbaik per arsitektur per korpus latih |
| Korpus latih | `763` (SawitMVC-Depth-YOLO v2.0.0) dan `1716` (gabungan 953 dan 763) |
| Acuan kebenaran korpus 953 | `Baseline-SawitMVC/ground_truth/split_manifest.csv` |
| Acuan kebenaran korpus 763 | `SawitMVC-Depth-YOLO/{train,valid,test}/linked/*.json`, kunci `summary.by_class` |
| Skrip | [`scripts/kalibrasi_koefisien_pencacahan.py`](../scripts/kalibrasi_koefisien_pencacahan.py), [`scripts/ringkas_koefisien_pencacahan.py`](../scripts/ringkas_koefisien_pencacahan.py) |
| Artefak angka | [`results/counting_koefisien_2026-09-16/koefisien_pencacahan.json`](../results/counting_koefisien_2026-09-16/koefisien_pencacahan.json), memuat 72 baris hasil |

## 2. Konfigurasi Terbaik di Antara yang Terbaik

| Korpus uji | Detektor | Korpus latih | Kalibrasi | Metode | $MAE$ makro | $RMSE$ makro | Akurasi ±1 makro | $MAE$ total per pohon | $mAP50$ |
|---|---|---|---|---|---|---|---|---|---|
| `953-test` ($n = 141$) | **RT-DETR-L** | 1716 | `953-val` | $k + \tau$ per kelas | **1,0301** | 1,4728 | 0,7358 | 2,1064 | 0,5723 |
| `763-test` ($n = 66$) | **RF-DETR-L** | 1716 | `763-val` | $k + \tau$ per kelas | **0,5871** | 1,0388 | 0,8902 | 0,9545 | 0,5960 |

## 3. Lima Konfigurasi Teratas pada Korpus Uji `953-test` ($n = 141$ pohon)

| # | Detektor | Latih | Kalibrasi | Metode | $MAE$ makro | $RMSE$ makro | Akurasi ±1 makro | $MAE$ total | Bias mutlak makro | $mAP50$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | RT-DETR-L | 1716 | `953-val` | $k + \tau$ per kelas | **1,0301** | 1,4728 | 0,7358 | 2,1064 | 0,2819 | 0,5723 |
| 2 | RF-DETR-L | 1716 | `953-val` | $k + \tau$ per kelas | 1,0337 | 1,4757 | **0,7447** | **1,8227** | **0,1082** | 0,5894 |
| 3 | RT-DETR-L | 1716 | `953-val` | $k$ per kelas | 1,0567 | 1,5177 | 0,7340 | 2,0709 | 0,2624 | 0,5723 |
| 4 | YOLO26l | 1716 | `953-val` | $k$ per kelas | 1,0621 | 1,5313 | 0,7394 | 2,0496 | 0,2784 | 0,5402 |
| 5 | RF-DETR-L | 1716 | `953-val` | $k$ per kelas | 1,0691 | 1,5162 | 0,7429 | 1,8794 | 0,1294 | 0,5894 |

## 4. Lima Konfigurasi Teratas pada Korpus Uji `763-test` (basis irisan setara, $n = 66$ pohon)

| # | Detektor | Latih | Kalibrasi | Metode | $MAE$ makro | $RMSE$ makro | Akurasi ±1 makro | $MAE$ total | Bias mutlak makro | $mAP50$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | RF-DETR-L | 1716 | `763-val` | $k + \tau$ per kelas | **0,5871** | **1,0388** | **0,8902** | 0,9545 | 0,1174 | 0,5960 |
| 2 | RF-DETR-L | 763 | `763-val` | $k + \tau$ per kelas | 0,5947 | 1,1041 | 0,8788 | 1,0152 | 0,2159 | 0,6129 |
| 3 | RF-DETR-L | 1716 | `763-val` | $k$ per kelas | 0,6061 | 1,0836 | 0,8826 | **0,8485** | **0,1061** | 0,5960 |
| 4 | RT-DETR-L | 1716 | `763-val` | $k + \tau$ per kelas | 0,6136 | 1,1573 | 0,8864 | 0,9091 | 0,1439 | 0,5745 |
| 5 | RF-DETR-L | 763 | `763-val` | $k$ global | 0,6288 | 1,1017 | 0,8598 | 1,0606 | 0,2273 | 0,6129 |

Tabel berikut memuat hasil pada basis penuh `763-test` ($n = 110$ pohon), yang hanya berlaku untuk detektor berkorpus latih `763`.

| Detektor | Metode | $MAE$ makro | $RMSE$ makro | Akurasi ±1 makro |
|---|---|---|---|---|
| RF-DETR-L | $k + \tau$ per kelas | 0,5909 | 1,1011 | 0,8750 |
| YOLO26l | $k$ per kelas | 0,6250 | 1,1051 | 0,8636 |
| RT-DETR-L | $k + \tau$ per kelas | 0,6545 | 1,1495 | 0,8386 |

## 5. Koefisien Pengali dan Ambang Konfigurasi Unggulan

| Detektor | Latih | Kalibrasi ke uji | Metode | $k_{B1}$ | $k_{B2}$ | $k_{B3}$ | $k_{B4}$ | $\tau_{B1}$ | $\tau_{B2}$ | $\tau_{B3}$ | $\tau_{B4}$ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RT-DETR-L | 1716 | `953-val` ke `953-test` | $k + \tau$ per kelas | 0,32 | 0,20 | 0,29 | 0,28 | 0,40 | 0,35 | 0,35 | 0,35 |
| RF-DETR-L | 1716 | `953-val` ke `953-test` | $k + \tau$ per kelas | 0,59 | 0,26 | 0,29 | 0,20 | 0,65 | 0,25 | 0,25 | 0,20 |
| RT-DETR-L | 1716 | `953-val` ke `953-test` | $k$ per kelas | 0,32 | 0,26 | 0,33 | 0,32 | 0,40 | 0,40 | 0,40 | 0,40 |
| RF-DETR-L | 1716 | `763-val` ke `763-test` | $k + \tau$ per kelas | 0,55 | 0,46 | 0,69 | 0,25 | 0,40 | 0,30 | 0,40 | 0,20 |
| RF-DETR-L | 763 | `763-val` ke `763-test` | $k + \tau$ per kelas | 0,63 | 0,41 | 0,71 | 0,51 | 0,50 | 0,25 | 0,45 | 0,40 |

Seluruh koefisien terpilih berada di bawah $1,00$, sehingga arah koreksi adalah penurunan hitungan. Sumber pencacahan berlebih (*over-count*) bersifat struktural: penjumlahan naif mencacah satu tandan berulang kali karena kemunculan objek (*appearance*) yang sama terekam pada empat sisi pohon.

## 6. Koreksi Bias per Kelas dari Metode Naif ke Metode Terkalibrasi

| Detektor | Latih | Korpus uji | Metode | Bias B1 | Bias B2 | Bias B3 | Bias B4 | $MAE$ makro |
|---|---|---|---|---|---|---|---|---|
| RT-DETR-L | 1716 | `953-test` | Naif ($k = 1$; $\tau = 0,25$) | +2,560 | +9,362 | +15,887 | +8,142 | 8,9876 |
| RT-DETR-L | 1716 | `953-test` | $k + \tau$ per kelas | −0,028 | −0,312 | −0,433 | −0,355 | **1,0301** |
| RF-DETR-L | 1716 | `763-test` | Naif ($k = 1$; $\tau = 0,25$) | +0,939 | +2,712 | +3,333 | +0,439 | 1,9773 |
| RF-DETR-L | 1716 | `763-test` | $k + \tau$ per kelas | −0,121 | +0,015 | −0,030 | −0,303 | **0,5871** |

Akurasi ±1 per kelas RF-DETR-L berkorpus latih `1716` pada `763-test` setelah kalibrasi tercatat sebesar $0,939$ untuk B1, $0,803$ untuk B2, $0,864$ untuk B3, dan $0,955$ untuk B4.

## 7. Efek Kalibrasi pada Seluruh Skenario

| Detektor | Latih | Kalibrasi ke uji | $MAE$ naif | $MAE$ terkalibrasi | Perubahan | Akurasi ±1 naif | Akurasi ±1 terkalibrasi |
|---|---|---|---|---|---|---|---|
| YOLO26l | 763 | `763-val` ke `763-test` | 1,038 | 0,682 | −34,3% | 0,746 | 0,852 |
| YOLO26l | 763 | `763-val` ke `953-test` | 2,887 | 2,059 | −28,7% | 0,394 | 0,456 |
| YOLO26l | 1716 | `953-val` ke `953-test` | 2,066 | 1,062 | −48,6% | 0,521 | 0,739 |
| YOLO26l | 1716 | `763-val` ke `953-test` | 2,066 | 1,301 | −37,0% | 0,521 | 0,670 |
| YOLO26l | 1716 | `763-val` ke `763-test` | 0,955 | 0,716 | −25,0% | 0,788 | 0,841 |
| YOLO26l | 1716 | `953-val` ke `763-test` | 0,955 | 0,871 | −8,7% | 0,788 | 0,795 |
| RT-DETR-L | 763 | `763-val` ke `763-test` | 2,652 | 0,716 | −73,0% | 0,424 | 0,826 |
| RT-DETR-L | 763 | `763-val` ke `953-test` | 5,161 | 2,466 | −52,2% | 0,246 | 0,468 |
| RT-DETR-L | 1716 | `953-val` ke `953-test` | 8,988 | 1,030 | **−88,5%** | 0,122 | 0,736 |
| RT-DETR-L | 1716 | `763-val` ke `953-test` | 8,988 | 1,427 | −84,1% | 0,122 | 0,670 |
| RT-DETR-L | 1716 | `763-val` ke `763-test` | 3,348 | 0,614 | −81,7% | 0,394 | 0,886 |
| RT-DETR-L | 1716 | `953-val` ke `763-test` | 3,348 | 0,826 | −75,3% | 0,394 | 0,799 |
| RF-DETR-L | 763 | `763-val` ke `763-test` | 1,875 | 0,595 | −68,3% | 0,602 | 0,879 |
| RF-DETR-L | 763 | `763-val` ke `953-test` | 5,117 | 2,250 | −56,0% | 0,280 | 0,436 |
| RF-DETR-L | 1716 | `953-val` ke `953-test` | 6,131 | 1,034 | −83,1% | 0,207 | 0,745 |
| RF-DETR-L | 1716 | `763-val` ke `953-test` | 6,131 | 1,324 | −78,4% | 0,207 | 0,684 |
| RF-DETR-L | 1716 | `763-val` ke `763-test` | 1,977 | 0,587 | −70,3% | 0,591 | 0,890 |
| RF-DETR-L | 1716 | `953-val` ke `763-test` | 1,977 | 0,864 | −56,3% | 0,591 | 0,788 |

Kolom $MAE$ terkalibrasi memuat metode terbaik pada setiap skenario. Seluruh baris dengan korpus uji `763-test` memakai basis irisan setara sebanyak 66 pohon.

## 8. Kalibrasi Domain Sama dan Kalibrasi Silang (Metode $k$ per Kelas)

| Detektor | Latih | Korpus uji | Kalibrasi domain sama | Kalibrasi silang | Selisih relatif |
|---|---|---|---|---|---|
| YOLO26l | 1716 | `953-test` | 1,0621 (`953-val`) | 1,3014 (`763-val`) | +22,5% |
| RT-DETR-L | 1716 | `953-test` | 1,0567 (`953-val`) | 1,5142 (`763-val`) | +43,3% |
| RF-DETR-L | 1716 | `953-test` | 1,0691 (`953-val`) | 1,3582 (`763-val`) | +27,0% |
| YOLO26l | 1716 | `763-test` | 0,7235 (`763-val`) | 0,8712 (`953-val`) | +20,4% |
| RT-DETR-L | 1716 | `763-test` | 0,6402 (`763-val`) | 0,8674 (`953-val`) | +35,5% |
| RF-DETR-L | 1716 | `763-test` | 0,6061 (`763-val`) | 0,8636 (`953-val`) | +42,5% |

Selisih relatif bertanda positif menunjukkan bahwa kalibrasi silang menaikkan $MAE$, sehingga performa pencacahan mengalami penurunan.

## 9. Generalisasi Detektor Berkorpus Latih `763` ke Korpus Uji `953`

Seluruh nilai $MAE$ pada tabel ini memakai metode $k$ per kelas dengan kalibrasi `763-val`.

| Detektor | $mAP50$ pada `763-test` | $mAP50$ pada `953-test` | $MAE$ makro `763-test` ($n = 66$) | $MAE$ makro `953-test` ($n = 141$) |
|---|---|---|---|---|
| YOLO26l | 0,5163 | 0,2331 | 0,682 | 2,059 |
| RT-DETR-L | 0,5580 | 0,1110 | 0,762 | 2,466 |
| RF-DETR-L | 0,6129 | 0,1774 | 0,659 | 2,250 |

Kalibrasi koefisien tidak mengompensasi penurunan kualitas deteksi lintas korpus. Pada `953-test`, detektor berkorpus latih `763` tetap berada pada $MAE$ $2,06$–$2,47$, sedangkan detektor berkorpus latih `1716` mencapai $1,03$–$1,07$.

## 10. Perbandingan Metode Kalibrasi (Rerata 18 Skenario)

| Metode | Parameter bebas | $MAE$ makro rerata | Akurasi ±1 makro rerata |
|---|---|---|---|
| Naif ($k = 1$; $\tau = 0,25$) | 0 | 3,6477 | 0,4411 |
| $k$ global | 2 | 1,1857 | 0,7231 |
| **$k$ per kelas** | 5 | **1,1562** | **0,7263** |
| $k + \tau$ per kelas | 8 | 1,1887 | 0,7205 |

Penambahan ambang per kelas dengan 8 parameter menghasilkan konfigurasi tunggal terbaik, tetapi tidak menunjukkan keunggulan pada nilai rerata karena lebih rentan terhadap penyesuaian berlebih (*overfitting*) pada partisi validasi.

## 11. Batasan Validitas dan Audit

| No. | Batasan |
|---|---|
| 1 | Koefisien dipasang pada partisi validasi saja. *Dump* prediksi partisi latih belum terlacak pada repositori, sehingga penyertaan partisi latih memerlukan inferensi ulang. |
| 2 | Partisi uji `763-test` pada korpus latih `1716` hanya beririsan 66 dari 110 pohon karena skema partisi kedua korpus berbeda. Seluruh perbandingan lintas korpus latih pada `763-test` memakai irisan 66 pohon tersebut. |
| 3 | Nilai $mAP50$ untuk korpus latih `1716` pada `763-test` berasal dari partisi uji gabungan (DEPTH dan SAWIT), bukan dari 66 pohon irisan, sehingga nilai tersebut hanya berlaku sebagai konteks kualitas deteksi. |
| 4 | Detektor `v2repro` yang dilatih pada korpus 953 tidak disertakan karena *dump* prediksi partisi validasinya belum tersedia. Kalibrasi yang bebas kebocoran partisi data (*data leakage*) untuk bobot tersebut memerlukan inferensi ulang pada `953-val`. |
| 5 | Dengan $n = 141$ dan $n = 66$ pohon, selisih $MAE$ di bawah sekitar $0,05$ tidak terpisahkan dari variasi acak (*noise*) pencuplikan. Peringkat 1 sampai 5 pada Bagian 3 dan Bagian 4 berada di dalam rentang tersebut dan tidak dapat dibaca sebagai keunggulan yang pasti antarbaris berdekatan. |
| 6 | Deteksi berindeks kelas 4 pada *dump* RF-DETR tidak disertakan. Seluruh deteksi tersebut memiliki skor keyakinan (*confidence*) sangat rendah dan tidak memiliki padanan kelas B1–B4. |
