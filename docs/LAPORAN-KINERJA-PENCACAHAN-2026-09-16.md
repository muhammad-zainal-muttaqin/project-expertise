# Laporan Kinerja per Korpus Latih: Deteksi dan Pencacahan

## 1. Ringkasan Eksekutif

| Korpus | Detektor terbaik | $mAP50$ | F1 | Metode kalibrasi | $MAE$ | $MAE$ relatif | Akurasi ±1 | Penurunan $MAE$ |
|---|---|---|---|---|---|---|---|---|
| 953 | RF-DETR-L | 0,5964 | 0,5874 | $k$ per kelas | 1,0408 | 0,4803 | 0,7429 | 84,5% |
| 763 | RF-DETR-L | 0,6101 | 0,6046 | $k + \tau$ per kelas | 0,6091 | 0,5648 | 0,8614 | 65,9% |
| 1716 | RF-DETR-L | 0,5961 | 0,6039 | $k$ per kelas | 0,8473 | 0,4847 | 0,7938 | 79,8% |

| Temuan utama | Angka pendukung |
|---|---|
| RF-DETR-L terbaik pada ketiga korpus | $mAP50$ 0,5961–0,6101 |
| Kalibrasi menurunkan galat pada 27 kombinasi dalam domain | $MAE$ turun 28,0–88,6%; \|bias\| makro turun 34,8–98,1% |
| Peringkat korpus berbalik pada basis relatif | $MAE$ 0,6091 (`763`) berbanding 1,0408 (`953`); $MAE$ relatif 0,5648 berbanding 0,4803 |
| Deteksi menurun saat `953` dan `763` bertukar korpus uji | $mAP50$ 0,1109–0,2724 |
| Korpus latih `1716` terbaik pada uji `1716` | $MAE$ 0,9287–0,9964 berbanding 1,1256–1,4879 (latih `953` dan `763`, 207 pohon) |

## 2. Identitas Eksperimen

| Parameter | Nilai |
|---|---|
| Identitas simpul | `V2-E-050` sampai `V2-E-050g` |
| Tanggal | 16–17 September 2026 |
| Korpus latih | `953` (SawitMVC-YOLO), `763` (SawitMVC-Depth-YOLO v2.0.0), `1716` (gabungan) |
| Partisi uji | `953`: 141 pohon; `763`: 110 pohon; `1716`: 257 pohon, basis sama 207 pohon |
| Detektor | YOLO26l, RT-DETR-L, RF-DETR-L |
| Metrik deteksi | Presisi, *recall*, F1, $AP50$, dan $AP50\text{--}95$ per kelas serta makro |
| Model pencacahan | $\hat{y}_c(t) = \operatorname{round}(k_c \cdot n_c(t))$ |
| Kalibrasi | Validasi silang 5 lipatan tingkat pohon, tanpa pelatihan ulang |
| Metrik pencacahan | $MAE$ makro, $MAE$ relatif, $RMSE$ makro, \|bias\| makro, akurasi ±1 makro |
| Skrip | [`susun_laporan_pencacahan.py`](../scripts/susun_laporan_pencacahan.py), [`metrik_deteksi_perkorpus.py`](../scripts/metrik_deteksi_perkorpus.py), [`kalibrasi_pencacahan_perkorpus.py`](../scripts/kalibrasi_pencacahan_perkorpus.py), [`permutasi_koefisien_pencacahan.py`](../scripts/permutasi_koefisien_pencacahan.py), [`inferensi_953_ke_763.py`](../scripts/inferensi_953_ke_763.py), [`gabung_dump_1716.py`](../scripts/gabung_dump_1716.py) |
| Artefak angka | [`metrik_deteksi_perkorpus.json`](../results/counting_koefisien_2026-09-16/metrik_deteksi_perkorpus.json), [`pencacahan_perkorpus.json`](../results/counting_koefisien_2026-09-16/pencacahan_perkorpus.json), [`permutasi_koefisien.json`](../results/counting_koefisien_2026-09-16/permutasi_koefisien.json), [`gabungan_1716_manifest.json`](../results/cross_eval/predictions/gabungan_1716_manifest.json) |

## 3. Karakteristik Partisi Uji

| Korpus | Pohon | Tandan | Tandan per pohon | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|---|---|
| 953 | 141 | 1.397 | 9,91 | 0,83 | 1,82 | 5,26 | 1,99 |
| 763 | 110 | 558 | 5,07 | 0,85 | 1,81 | 1,95 | 0,45 |
| 1716 | 257 | 1.972 | 7,67 | 0,88 | 1,77 | 3,76 | 1,26 |
| 1716, basis sama | 207 | 1.768 | 8,54 | 0,76 | 1,78 | 4,49 | 1,52 |

| Pasangan partisi | Pohon beririsan | Konsekuensi |
|---|---|---|
| Uji `1716` dan uji `953` | 141 dari 141 pohon SAWIT | Hasil tidak saling bebas |
| Uji `1716` dan uji `763` | 66 dari 110 pohon | Basis baris `1716` ke `763` |
| Uji `1716` dan latih/validasi `763` | 50 pohon DEPTH, citra identik | Dikeluarkan dari baris `763` ke `1716` |
| Uji `953` dan uji `763` | 8 ID pohon | Sesi akuisisi berbeda |
| Uji `953` dan latih/validasi `763` | 50 dari 141 ID pohon | Sesi akuisisi berbeda, sah menurut V2-E-040 |
| Uji `763` dan latih/validasi `953` | 44 dari 110 ID pohon | Sesi akuisisi berbeda |

## 4. Deteksi per Korpus Latih

![Metrik deteksi makro per korpus](assets/laporan-pencacahan-2026-09-16/deteksi_makro.png)

![F1 per kelas, tiga korpus](assets/laporan-pencacahan-2026-09-16/deteksi_perkelas.png)

Presisi, *recall*, dan F1 diukur pada ambang $\text{conf}^{*}$. Tebal: nilai tertinggi tiap korpus.

| Korpus | Detektor | Citra | Objek | $\text{conf}^{*}$ | Presisi | Recall | F1 | $mAP50$ | $mAP50\text{--}95$ | $mAP50$ `pycocotools` |
|---|---|---|---|---|---|---|---|---|---|---|
| 953 | YOLO26l | 588 | 2.612 | 0,20 | 0,5426 | 0,5681 | 0,5515 | 0,5434 | 0,2565 | 0,5435 |
| 953 | RT-DETR-L | 588 | 2.612 | 0,45 | 0,5316 | **0,6136** | 0,5618 | 0,5725 | 0,2687 | 0,5718 |
| **953** | **RF-DETR-L** | 588 | 2.612 | 0,37 | **0,5824** | 0,6017 | **0,5874** | **0,5964** | **0,2756** | 0,5965 |
| 763 | YOLO26l | 440 | 891 | 0,17 | 0,4778 | 0,5650 | 0,5173 | 0,5143 | 0,1902 | 0,5163 |
| 763 | RT-DETR-L | 440 | 891 | 0,48 | 0,5604 | 0,5893 | 0,5712 | 0,5563 | 0,2056 | 0,5580 |
| **763** | **RF-DETR-L** | 440 | 891 | 0,34 | **0,6089** | **0,6103** | **0,6046** | **0,6101** | **0,2335** | 0,6129 |
| 1716 | YOLO26l | 1.052 | 3.513 | 0,20 | 0,5604 | 0,5354 | 0,5406 | 0,5386 | 0,2393 | 0,5389 |
| 1716 | RT-DETR-L | 1.052 | 3.513 | 0,46 | 0,5630 | 0,6256 | 0,5898 | 0,5742 | 0,2456 | 0,5745 |
| **1716** | **RF-DETR-L** | 1.052 | 3.513 | 0,36 | **0,5846** | **0,6269** | **0,6039** | **0,5961** | **0,2522** | 0,5960 |

### 4.1 Rincian Kelas Korpus `953`

| Detektor | Kelas | Objek | Presisi | Recall | F1 | $AP50$ | $AP50\text{--}95$ |
|---|---|---|---|---|---|---|---|
| YOLO26l | B1 | 252 | 0,6555 | 0,7778 | 0,7114 | 0,7704 | 0,4022 |
| YOLO26l | B2 | 496 | 0,5312 | 0,4113 | 0,4636 | 0,4475 | 0,2110 |
| YOLO26l | B3 | 1.409 | 0,5741 | 0,6458 | 0,6079 | 0,6050 | 0,2748 |
| YOLO26l | B4 | 455 | 0,4095 | 0,4374 | 0,4230 | 0,3505 | 0,1379 |
| YOLO26l | **Makro** | 2.612 | **0,5426** | **0,5681** | **0,5515** | **0,5434** | **0,2565** |
| RT-DETR-L | B1 | 252 | 0,5730 | 0,8254 | 0,6764 | 0,7739 | 0,4183 |
| RT-DETR-L | B2 | 496 | 0,5538 | 0,4153 | 0,4747 | 0,4810 | 0,2270 |
| RT-DETR-L | B3 | 1.409 | 0,5782 | 0,7083 | 0,6367 | 0,6323 | 0,2776 |
| RT-DETR-L | B4 | 455 | 0,4212 | 0,5055 | 0,4595 | 0,4028 | 0,1517 |
| RT-DETR-L | **Makro** | 2.612 | **0,5316** | **0,6136** | **0,5618** | **0,5725** | **0,2687** |
| RF-DETR-L | B1 | 252 | 0,6933 | 0,8254 | 0,7536 | 0,8196 | 0,4383 |
| RF-DETR-L | B2 | 496 | 0,5251 | 0,4637 | 0,4925 | 0,5010 | 0,2364 |
| RF-DETR-L | B3 | 1.409 | 0,6068 | 0,7175 | 0,6576 | 0,6632 | 0,2869 |
| RF-DETR-L | B4 | 455 | 0,5042 | 0,4000 | 0,4461 | 0,4018 | 0,1408 |
| RF-DETR-L | **Makro** | 2.612 | **0,5824** | **0,6017** | **0,5874** | **0,5964** | **0,2756** |

### 4.2 Rincian Kelas Korpus `763`

| Detektor | Kelas | Objek | Presisi | Recall | F1 | $AP50$ | $AP50\text{--}95$ |
|---|---|---|---|---|---|---|---|
| YOLO26l | B1 | 173 | 0,5821 | 0,6763 | 0,6257 | 0,6847 | 0,2708 |
| YOLO26l | B2 | 321 | 0,5695 | 0,6511 | 0,6076 | 0,5877 | 0,2074 |
| YOLO26l | B3 | 334 | 0,4988 | 0,6467 | 0,5632 | 0,5916 | 0,2146 |
| YOLO26l | B4 | 63 | 0,2609 | 0,2857 | 0,2727 | 0,1929 | 0,0680 |
| YOLO26l | **Makro** | 891 | **0,4778** | **0,5650** | **0,5173** | **0,5143** | **0,1902** |
| RT-DETR-L | B1 | 173 | 0,6029 | 0,7110 | 0,6525 | 0,7377 | 0,2947 |
| RT-DETR-L | B2 | 321 | 0,5674 | 0,6822 | 0,6195 | 0,5889 | 0,2124 |
| RT-DETR-L | B3 | 334 | 0,7117 | 0,5988 | 0,6504 | 0,6542 | 0,2362 |
| RT-DETR-L | B4 | 63 | 0,3594 | 0,3651 | 0,3622 | 0,2445 | 0,0791 |
| RT-DETR-L | **Makro** | 891 | **0,5604** | **0,5893** | **0,5712** | **0,5563** | **0,2056** |
| RF-DETR-L | B1 | 173 | 0,7294 | 0,7168 | 0,7230 | 0,7758 | 0,3249 |
| RF-DETR-L | B2 | 321 | 0,5777 | 0,7414 | 0,6494 | 0,6353 | 0,2376 |
| RF-DETR-L | B3 | 334 | 0,6718 | 0,6497 | 0,6606 | 0,6893 | 0,2537 |
| RF-DETR-L | B4 | 63 | 0,4565 | 0,3333 | 0,3853 | 0,3401 | 0,1179 |
| RF-DETR-L | **Makro** | 891 | **0,6089** | **0,6103** | **0,6046** | **0,6101** | **0,2335** |

### 4.3 Rincian Kelas Korpus `1716`

| Detektor | Kelas | Objek | Presisi | Recall | F1 | $AP50$ | $AP50\text{--}95$ |
|---|---|---|---|---|---|---|---|
| YOLO26l | B1 | 443 | 0,6803 | 0,7111 | 0,6954 | 0,7298 | 0,3455 |
| YOLO26l | B2 | 808 | 0,4952 | 0,5062 | 0,5006 | 0,4764 | 0,2095 |
| YOLO26l | B3 | 1.749 | 0,5762 | 0,6398 | 0,6063 | 0,6065 | 0,2657 |
| YOLO26l | B4 | 513 | 0,4899 | 0,2846 | 0,3600 | 0,3419 | 0,1364 |
| YOLO26l | **Makro** | 3.513 | **0,5604** | **0,5354** | **0,5406** | **0,5386** | **0,2393** |
| RT-DETR-L | B1 | 443 | 0,7110 | 0,6885 | 0,6995 | 0,7308 | 0,3533 |
| RT-DETR-L | B2 | 808 | 0,4734 | 0,6176 | 0,5360 | 0,5120 | 0,2169 |
| RT-DETR-L | B3 | 1.749 | 0,5949 | 0,7421 | 0,6604 | 0,6466 | 0,2683 |
| RT-DETR-L | B4 | 513 | 0,4726 | 0,4542 | 0,4632 | 0,4074 | 0,1438 |
| RT-DETR-L | **Makro** | 3.513 | **0,5630** | **0,6256** | **0,5898** | **0,5742** | **0,2456** |
| RF-DETR-L | B1 | 443 | 0,6892 | 0,7607 | 0,7232 | 0,7688 | 0,3729 |
| RF-DETR-L | B2 | 808 | 0,5443 | 0,5854 | 0,5641 | 0,5369 | 0,2266 |
| RF-DETR-L | B3 | 1.749 | 0,6125 | 0,7130 | 0,6589 | 0,6645 | 0,2687 |
| RF-DETR-L | B4 | 513 | 0,4925 | 0,4483 | 0,4694 | 0,4141 | 0,1407 |
| RF-DETR-L | **Makro** | 3.513 | **0,5846** | **0,6269** | **0,6039** | **0,5961** | **0,2522** |

## 5. Pencacahan per Korpus Latih

![Galat dan akurasi pencacahan per korpus latih](assets/laporan-pencacahan-2026-09-16/pencacahan_makro.png)

Setiap baris memakai varian per kelas dengan $MAE$ terendah. Tebal: $MAE$ terendah tiap korpus.

| Korpus | Detektor | Metode | $MAE$ makro | $MAE$ relatif | $RMSE$ makro | \|Bias\| makro | Akurasi ±1 makro |
|---|---|---|---|---|---|---|---|
| 953 | YOLO26l | $k$ per kelas | 1,0816 | 0,5031 | 1,5408 | 0,1206 | 0,7447 |
| 953 | RT-DETR-L | $k + \tau$ per kelas | 1,0621 | 0,5102 | 1,5733 | 0,2784 | 0,7429 |
| **953** | **RF-DETR-L** | **$k$ per kelas** | **1,0408** | 0,4803 | 1,5087 | 0,1720 | 0,7429 |
| 763 | YOLO26l | $k$ per kelas | 0,6477 | 0,5817 | 1,1638 | 0,2205 | 0,8591 |
| 763 | RT-DETR-L | $k$ per kelas | 0,6886 | 0,6141 | 1,2506 | 0,2614 | 0,8455 |
| **763** | **RF-DETR-L** | **$k + \tau$ per kelas** | **0,6091** | 0,5648 | 1,1023 | 0,1273 | 0,8614 |
| 1716 | YOLO26l | $k + \tau$ per kelas | 0,9348 | 0,5324 | 1,4794 | 0,2519 | 0,7685 |
| 1716 | RT-DETR-L | $k$ per kelas | 0,8677 | 0,5015 | 1,4495 | 0,2490 | 0,7928 |
| **1716** | **RF-DETR-L** | **$k$ per kelas** | **0,8473** | 0,4847 | 1,4006 | 0,0982 | 0,7938 |

## 6. Koefisien Terbaik RF-DETR-L

![Koefisien pengali dan ambang per kelas](assets/laporan-pencacahan-2026-09-16/koefisien.png)

Metode dipilih dari varian per kelas ber-$MAE$ terendah, dan nilainya merupakan rerata lima lipatan.

| Korpus | Metode | $k_{B1}$ | $k_{B2}$ | $k_{B3}$ | $k_{B4}$ | $\tau_{B1}$ | $\tau_{B2}$ | $\tau_{B3}$ | $\tau_{B4}$ |
|---|---|---|---|---|---|---|---|---|---|
| 953 | $k$ per kelas | 0,30 | 0,28 | 0,33 | 0,38 | 0,30 | 0,30 | 0,30 | 0,30 |
| 763 | $k + \tau$ per kelas | 0,42 | 0,42 | 0,68 | 0,62 | 0,32 | 0,27 | 0,40 | 0,31 |
| 1716 | $k$ per kelas | 0,37 | 0,47 | 0,46 | 0,51 | 0,35 | 0,35 | 0,35 | 0,35 |

### 6.1 Rincian per Kelas RF-DETR-L

![MAE, bias, dan akurasi ±1 per kelas, RF-DETR-L](assets/laporan-pencacahan-2026-09-16/pencacahan_perkelas.png)

Konfigurasi mengikuti tabel §6, dan nilainya dihitung dari prediksi luar-lipatan.

| Korpus | Metode | Metrik | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|---|
| 953 | $k$ per kelas | $MAE$ | 0,376 | 1,014 | 1,539 | 1,234 |
| 953 | $k$ per kelas | Bias | −0,050 | −0,248 | −0,163 | −0,227 |
| 953 | $k$ per kelas | Akurasi ±1 | 0,986 | 0,766 | 0,560 | 0,660 |
| 763 | $k + \tau$ per kelas | $MAE$ | 0,436 | 0,845 | 0,745 | 0,409 |
| 763 | $k + \tau$ per kelas | Bias | −0,127 | +0,027 | −0,236 | −0,118 |
| 763 | $k + \tau$ per kelas | Akurasi ±1 | 0,900 | 0,782 | 0,827 | 0,936 |
| 1716 | $k$ per kelas | $MAE$ | 0,412 | 0,949 | 1,276 | 0,751 |
| 1716 | $k$ per kelas | Bias | −0,163 | −0,125 | 0,000 | −0,105 |
| 1716 | $k$ per kelas | Akurasi ±1 | 0,938 | 0,747 | 0,661 | 0,829 |

## 7. Efek Kalibrasi

![Penurunan MAE dan kenaikan akurasi ±1, sembilan kombinasi](assets/laporan-pencacahan-2026-09-16/efek_kalibrasi_penuh.png)

Penurunan pada gambar dan §7.1 dihitung terhadap garis dasar naif berikut.

| Korpus | Detektor | $MAE$ naif | $RMSE$ naif | \|Bias\| makro naif | Akurasi ±1 naif |
|---|---|---|---|---|---|
| 953 | YOLO26l | 2,1560 | 2,9507 | 1,4149 | 0,5213 |
| 953 | RT-DETR-L | 9,2837 | 10,5769 | 9,2766 | 0,1401 |
| 953 | RF-DETR-L | 6,7039 | 7,8535 | 6,6755 | 0,2092 |
| 763 | YOLO26l | 0,9591 | 1,5413 | 0,5545 | 0,7659 |
| 763 | RT-DETR-L | 2,5227 | 3,4071 | 2,4364 | 0,4432 |
| 763 | RF-DETR-L | 1,7864 | 2,5912 | 1,6773 | 0,6068 |
| 1716 | YOLO26l | 1,5370 | 2,3869 | 1,0117 | 0,6449 |
| 1716 | RT-DETR-L | 6,3103 | 8,0470 | 6,2967 | 0,2412 |
| 1716 | RF-DETR-L | 4,1858 | 5,7300 | 4,1060 | 0,3784 |

### 7.1 Perbandingan Metode

![Perbandingan metode kalibrasi](assets/laporan-pencacahan-2026-09-16/metode_kalibrasi.png)

Kolom penurunan dan kenaikan dihitung terhadap garis dasar naif pada §7.

| Korpus | Detektor | Metode | $MAE$ | Penurunan $MAE$ | $RMSE$ | Penurunan $RMSE$ | \|Bias\| makro | Penurunan \|bias\| | Akurasi ±1 | Kenaikan akurasi ±1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 953 | YOLO26l | **$k$ global** | **1,0691** | **50,4%** | 1,5512 | 47,4% | 0,1826 | 87,1% | 0,7606 | +23,9 pp |
| 953 | YOLO26l | $k$ per kelas | 1,0816 | 49,8% | 1,5408 | 47,8% | 0,1206 | 91,5% | 0,7447 | +22,3 pp |
| 953 | YOLO26l | $k + \tau$ per kelas | 1,1046 | 48,8% | 1,5605 | 47,1% | 0,1543 | 89,1% | 0,7323 | +21,1 pp |
| 953 | RT-DETR-L | $k$ global | 1,1489 | 87,6% | 1,6590 | 84,3% | 0,2163 | 97,7% | 0,7074 | +56,7 pp |
| 953 | RT-DETR-L | $k$ per kelas | 1,1294 | 87,8% | 1,6305 | 84,6% | 0,1720 | 98,1% | 0,7181 | +57,8 pp |
| 953 | RT-DETR-L | **$k + \tau$ per kelas** | **1,0621** | **88,6%** | 1,5733 | 85,1% | 0,2784 | 97,0% | 0,7429 | +60,3 pp |
| 953 | RF-DETR-L | **$k$ global** | **1,0372** | **84,5%** | 1,5037 | 80,9% | 0,1791 | 97,3% | 0,7465 | +53,7 pp |
| 953 | RF-DETR-L | $k$ per kelas | 1,0408 | 84,5% | 1,5087 | 80,8% | 0,1720 | 97,4% | 0,7429 | +53,4 pp |
| 953 | RF-DETR-L | $k + \tau$ per kelas | 1,0833 | 83,8% | 1,5616 | 80,1% | 0,1649 | 97,5% | 0,7216 | +51,2 pp |
| 763 | YOLO26l | **$k$ global** | **0,6159** | **35,8%** | 1,1467 | 25,6% | 0,3614 | 34,8% | 0,8568 | +9,1 pp |
| 763 | YOLO26l | $k$ per kelas | 0,6477 | 32,5% | 1,1638 | 24,5% | 0,2205 | 60,2% | 0,8591 | +9,3 pp |
| 763 | YOLO26l | $k + \tau$ per kelas | 0,6909 | 28,0% | 1,1992 | 22,2% | 0,1864 | 66,4% | 0,8386 | +7,3 pp |
| 763 | RT-DETR-L | $k$ global | 0,7250 | 71,3% | 1,2871 | 62,2% | 0,2523 | 89,6% | 0,8432 | +40,0 pp |
| 763 | RT-DETR-L | **$k$ per kelas** | **0,6886** | **72,7%** | 1,2506 | 63,3% | 0,2614 | 89,3% | 0,8455 | +40,2 pp |
| 763 | RT-DETR-L | $k + \tau$ per kelas | 0,7023 | 72,2% | 1,2610 | 63,0% | 0,1932 | 92,1% | 0,8386 | +39,5 pp |
| 763 | RF-DETR-L | $k$ global | 0,6205 | 65,3% | 1,1234 | 56,6% | 0,2477 | 85,2% | 0,8523 | +24,5 pp |
| 763 | RF-DETR-L | $k$ per kelas | 0,6386 | 64,2% | 1,1698 | 54,9% | 0,2250 | 86,6% | 0,8432 | +23,6 pp |
| 763 | RF-DETR-L | **$k + \tau$ per kelas** | **0,6091** | **65,9%** | 1,1023 | 57,5% | 0,1273 | 92,4% | 0,8614 | +25,5 pp |
| 1716 | YOLO26l | $k$ global | 0,9387 | 38,9% | 1,4665 | 38,6% | 0,1683 | 83,4% | 0,7714 | +12,6 pp |
| 1716 | YOLO26l | $k$ per kelas | 0,9582 | 37,7% | 1,5115 | 36,7% | 0,2364 | 76,6% | 0,7685 | +12,4 pp |
| 1716 | YOLO26l | **$k + \tau$ per kelas** | **0,9348** | **39,2%** | 1,4794 | 38,0% | 0,2519 | 75,1% | 0,7685 | +12,4 pp |
| 1716 | RT-DETR-L | $k$ global | 0,9270 | 85,3% | 1,5134 | 81,2% | 0,3064 | 95,1% | 0,7753 | +53,4 pp |
| 1716 | RT-DETR-L | **$k$ per kelas** | **0,8677** | **86,2%** | 1,4495 | 82,0% | 0,2490 | 96,0% | 0,7928 | +55,2 pp |
| 1716 | RT-DETR-L | $k + \tau$ per kelas | 0,8842 | 86,0% | 1,4608 | 81,8% | 0,2617 | 95,8% | 0,7879 | +54,7 pp |
| 1716 | RF-DETR-L | $k$ global | 0,8765 | 79,1% | 1,4178 | 75,3% | 0,2208 | 94,6% | 0,7899 | +41,1 pp |
| 1716 | RF-DETR-L | **$k$ per kelas** | **0,8473** | **79,8%** | 1,4006 | 75,6% | 0,0982 | 97,6% | 0,7938 | +41,5 pp |
| 1716 | RF-DETR-L | $k + \tau$ per kelas | 0,8959 | 78,6% | 1,4513 | 74,7% | 0,1313 | 96,8% | 0,7840 | +40,6 pp |

Tebal: $MAE$ terendah tiap pasangan korpus dan detektor.

## 8. Uji Silang Detektor

![Dalam domain berbanding lintas korpus, RF-DETR-L](assets/laporan-pencacahan-2026-09-16/uji_silang.png)

Detektor diuji pada partisi uji korpus lain, dan koefisiennya dipasang ulang pada partisi sasaran. Tebal: nilai terbaik tiap pasangan korpus uji dan detektor.

| Korpus uji | Detektor | Korpus latih | Pohon | $mAP50$ | F1 | $MAE$ makro | $MAE$ relatif | Akurasi ±1 |
|---|---|---|---|---|---|---|---|---|
| 953 | YOLO26l | 953 (dalam domain) | 141 | **0,5434** | **0,5515** | **1,0816** | 0,5031 | **0,7447** |
| 953 | YOLO26l | 763 | 141 | 0,2332 | 0,2706 | 1,2961 | 0,5876 | 0,6702 |
| 953 | YOLO26l | 1716 | 141 | 0,5399 | 0,5342 | 1,0851 | **0,4945** | 0,7287 |
| 953 | RT-DETR-L | 953 (dalam domain) | 141 | **0,5725** | 0,5618 | 1,0621 | 0,5102 | **0,7429** |
| 953 | RT-DETR-L | 763 | 141 | 0,1109 | 0,1408 | 1,5142 | 0,6846 | 0,6277 |
| 953 | RT-DETR-L | 1716 | 141 | 0,5723 | **0,5855** | **1,0567** | **0,4793** | 0,7411 |
| 953 | RF-DETR-L | 953 (dalam domain) | 141 | **0,5964** | 0,5874 | **1,0408** | **0,4803** | 0,7429 |
| 953 | RF-DETR-L | 763 | 141 | 0,1767 | 0,2062 | 1,2518 | 0,5959 | 0,6702 |
| 953 | RF-DETR-L | 1716 | 141 | 0,5894 | **0,5935** | 1,0426 | 0,4866 | **0,7642** |
| 763 | YOLO26l | 953 | 110 | 0,2373 | 0,3018 | 1,1136 | 0,8766 | 0,7477 |
| 763 | YOLO26l | 763 (dalam domain) | 110 | 0,5143 | 0,5173 | **0,6477** | **0,5817** | **0,8591** |
| 763 | YOLO26l | 1716 | 66 | **0,5435** | **0,5466** | 0,7538 | 0,6138 | 0,8371 |
| 763 | RT-DETR-L | 953 | 110 | 0,1200 | 0,1596 | 1,2273 | 0,9485 | 0,7273 |
| 763 | RT-DETR-L | 763 (dalam domain) | 110 | 0,5563 | **0,5712** | **0,6886** | **0,6141** | 0,8455 |
| 763 | RT-DETR-L | 1716 | 66 | **0,5727** | 0,5689 | 0,7008 | 0,6279 | **0,8598** |
| 763 | RF-DETR-L | 953 | 110 | 0,2724 | 0,3023 | 1,1250 | 0,8759 | 0,7705 |
| 763 | RF-DETR-L | 763 (dalam domain) | 110 | 0,6101 | 0,6046 | **0,6091** | 0,5648 | 0,8614 |
| 763 | RF-DETR-L | 1716 | 66 | **0,6302** | **0,6348** | 0,6098 | **0,5489** | **0,8674** |
| 1716 | YOLO26l | 953 | 257 | 0,4411 | 0,4801 | 1,1148 | 0,6292 | 0,7267 |
| 1716 | YOLO26l | 763 | 207 | 0,2669 | 0,3123 | 1,1969 | 0,6289 | 0,6993 |
| 1716 | YOLO26l | 1716 (dalam domain) | 257 | **0,5386** | **0,5406** | **0,9348** | **0,5324** | **0,7685** |
| 1716 | RT-DETR-L | 953 | 257 | 0,4339 | 0,4720 | 1,1274 | 0,6486 | 0,7325 |
| 1716 | RT-DETR-L | 763 | 207 | 0,1745 | 0,2138 | 1,4879 | 0,7706 | 0,6184 |
| 1716 | RT-DETR-L | 1716 (dalam domain) | 257 | **0,5742** | **0,5898** | **0,8677** | **0,5015** | **0,7928** |
| 1716 | RF-DETR-L | 953 | 257 | 0,4851 | 0,5105 | 1,1274 | 0,6372 | 0,7286 |
| 1716 | RF-DETR-L | 763 | 207 | 0,2502 | 0,2666 | 1,1800 | 0,6410 | 0,6896 |
| 1716 | RF-DETR-L | 1716 (dalam domain) | 257 | **0,5961** | **0,6039** | **0,8473** | **0,4847** | **0,7938** |

Basis pohon dalam satu kelompok berbeda pada uji `763` dan `1716`; perbandingan setara ada di §8.1. Baris latih `763` pada uji `1716` tidak memuat 50 pohon yang citranya dipakai untuk melatih atau memvalidasi detektor `763`.

### 8.1 Basis Sama 207 Pohon

Ketiga korpus latih diuji pada 207 pohon uji `1716` yang sama: 141 SAWIT dan 66 DEPTH. Tebal: nilai terbaik tiap detektor.

| Detektor | Korpus latih | $mAP50$ | F1 | $MAE$ makro | $MAE$ relatif | Akurasi ±1 |
|---|---|---|---|---|---|---|
| YOLO26l | 953 | 0,4705 | 0,5024 | 1,1703 | 0,6054 | 0,7017 |
| YOLO26l | 763 | 0,2669 | 0,3123 | 1,1969 | 0,6289 | 0,6993 |
| YOLO26l | **1716** | **0,5436** | **0,5392** | **0,9964** | **0,5251** | **0,7597** |
| RT-DETR-L | 953 | 0,4792 | 0,5023 | 1,1256 | 0,5989 | 0,7379 |
| RT-DETR-L | 763 | 0,1745 | 0,2138 | 1,4879 | 0,7706 | 0,6184 |
| RT-DETR-L | **1716** | **0,5770** | **0,5894** | **0,9287** | **0,5044** | **0,7826** |
| RF-DETR-L | 953 | 0,5174 | 0,5351 | 1,1461 | 0,6031 | 0,7101 |
| RF-DETR-L | 763 | 0,2502 | 0,2666 | 1,1800 | 0,6410 | 0,6896 |
| RF-DETR-L | **1716** | **0,6008** | **0,6040** | **0,9420** | **0,4993** | **0,7717** |

## 9. Uji Silang Koefisien

![Matriks permutasi koefisien](assets/laporan-pencacahan-2026-09-16/permutasi_koefisien.png)

Diagonal tebal: koefisien milik kombinasi itu sendiri.

| Bagian | Yang dipindahkan | Pertanyaan yang dijawab |
|---|---|---|
| 8 | Detektor, ke citra korpus lain | Ketangguhan detektor lintas korpus |
| 9 | Koefisien; detektor dan citra tetap milik sasaran | Kekhususan koefisien terhadap arsitektur |

Tebal: nilai terendah tiap kolom.

| Jenis permutasi | Jumlah sel | $MAE$ rerata | Terendah | Tertinggi |
|---|---|---|---|---|
| **Koefisien sendiri** | 9 | **0,8778** | 0,6386 | **1,1294** |
| Korpus sama, detektor berbeda | 18 | 1,8369 | 0,7227 | 5,5284 |
| Detektor sama, korpus berbeda | 18 | 0,9572 | **0,6295** | 1,3972 |
| Korpus dan detektor berbeda | 36 | 1,8381 | 0,7295 | 5,8564 |

## 10. Glosarium

| Simbol atau istilah | Arti |
|---|---|
| B1–B4 | Kelas kematangan; B1 lewat matang, B4 mentah |
| SAWIT, DEPTH | Awalan citra `1716` yang berasal dari `953` dan `763` |
| $n_c(t)$ | Jumlah deteksi kelas $c$ lintas sisi pohon $t$ yang lolos ambang $\tau_c$ |
| $k_c$ | Pengali hitungan kelas $c$; nilai di bawah $1,00$ menurunkan hitungan |
| $\tau_c$ | Ambang skor keyakinan kelas $c$ |
| $\hat{y}_c(t)$ | Hitungan akhir kelas $c$ pada pohon $t$ |
| $\text{conf}^{*}$ | Ambang skor keyakinan yang memaksimalkan F1 makro |
| Presisi dan *recall* | Proporsi deteksi yang cocok dengan anotasi acuan, dan proporsi anotasi acuan yang terdeteksi |
| F1 | Rerata harmonik presisi dan *recall* |
| IoU | Rasio irisan terhadap gabungan dua kotak pembatas (*bounding box*) |
| $AP50$ dan $AP50\text{--}95$ | Luas kurva presisi-*recall* pada IoU $0,50$; rerata atas sepuluh ambang IoU $0,50$–$0,95$ |
| $mAP50$ dan $mAP50\text{--}95$ | Rerata $AP50$ dan $AP50\text{--}95$ atas empat kelas |
| $MAE$ | Rerata galat absolut per pohon, dalam satuan tandan |
| $MAE$ relatif | $MAE$ dibagi rerata jumlah tandan acuan kelas yang sama |
| $RMSE$ | Akar rerata kuadrat galat; lebih peka terhadap galat besar |
| Bias | Rerata selisih hitungan terhadap acuan; negatif berarti hitungan di bawah acuan |
| \|Bias\| makro | Rerata nilai mutlak bias per kelas atas B1–B4 |
| Akurasi ±1 | Proporsi pohon dengan selisih paling banyak satu tandan |
| pp | Poin persentase |
| Makro | Rerata tanpa bobot atas B1–B4 |
| Naif | Tanpa kalibrasi: $k = 1$, $\tau = 0,25$ |
| $k$ global | Satu $k$ dan satu $\tau$ untuk semua kelas |
| $k$ per kelas | $k$ per kelas dengan satu $\tau$ bersama |
| $k + \tau$ per kelas | $k$ dan $\tau$ per kelas |
| Validasi silang 5 lipatan | Koefisien dipasang pada empat lipatan, lalu diuji pada lipatan kelima |
| Prediksi luar-lipatan (*out-of-fold*) | Hitungan pohon dari koefisien yang dipasang tanpa pohon itu |
| Basis sama 207 pohon | Uji `1716` tanpa 50 pohon DEPTH latih/validasi `763` |
| ID pohon sama, sesi berbeda | Pohon fisik yang sama, difoto pada sesi berselang sekitar 80 hari dengan kamera berbeda |
| Y, RT, RF | YOLO26l, RT-DETR-L, RF-DETR-L |
