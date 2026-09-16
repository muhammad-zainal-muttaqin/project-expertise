# Laporan Kinerja per Korpus Latih: Deteksi dan Pencacahan

## 1. Identitas Eksperimen

| Parameter | Nilai |
|---|---|
| Identitas simpul | `V2-E-050`, `V2-E-050b`, `V2-E-050c`, dan `V2-E-050d` |
| Tanggal | 16 September 2026 |
| Korpus latih | `953` (SawitMVC-YOLO), `763` (SawitMVC-Depth-YOLO v2.0.0), `1716` (gabungan) |
| Partisi uji | `953`: 141 pohon; `763`: 110 pohon; `1716`: 257 pohon (141 SAWIT + 116 DEPTH) |
| Detektor | YOLO26l, RT-DETR-L, RF-DETR-L |
| Metrik deteksi | Presisi, Recall, F1, $AP50$, dan $AP50\text{--}95$ per kelas serta makro |
| Model pencacahan | $\hat{y}_c(t) = \operatorname{round}(k_c \cdot n_c(t))$, $n_c(t)$ = jumlah deteksi kelas $c$ lintas sisi pohon dengan skor keyakinan $\ge \tau_c$ |
| Kalibrasi | Lipat-silang 5 lipatan pada tingkat pohon, tanpa pelatihan ulang detektor |
| Metrik pencacahan | $MAE$ makro, $RMSE$ makro, bias mutlak makro, akurasi ±1 makro |
| Skrip | [`metrik_deteksi_perkorpus.py`](../scripts/metrik_deteksi_perkorpus.py), [`kalibrasi_pencacahan_perkorpus.py`](../scripts/kalibrasi_pencacahan_perkorpus.py), [`permutasi_koefisien_pencacahan.py`](../scripts/permutasi_koefisien_pencacahan.py), [`kalibrasi_koefisien_pencacahan.py`](../scripts/kalibrasi_koefisien_pencacahan.py) |
| Artefak angka | [`metrik_deteksi_perkorpus.json`](../results/counting_koefisien_2026-09-16/metrik_deteksi_perkorpus.json) (9 baris), [`pencacahan_perkorpus.json`](../results/counting_koefisien_2026-09-16/pencacahan_perkorpus.json) (48 baris), [`permutasi_koefisien.json`](../results/counting_koefisien_2026-09-16/permutasi_koefisien.json) (81 baris), [`koefisien_pencacahan.json`](../results/counting_koefisien_2026-09-16/koefisien_pencacahan.json) (72 baris) |

## 2. Deteksi per Korpus Latih

Presisi, Recall, dan F1 dilaporkan pada ambang skor keyakinan yang memaksimalkan F1 makro, kolom $\text{conf}^{*}$.

| Korpus latih | Detektor | Citra | Objek | $\text{conf}^{*}$ | Presisi | Recall | F1 | $mAP50$ | $mAP50\text{--}95$ | $mAP50$ `pycocotools` |
|---|---|---|---|---|---|---|---|---|---|---|
| 953 | YOLO26l | 588 | 2.612 | 0,20 | 0,5426 | 0,5681 | 0,5515 | 0,5434 | 0,2565 | 0,5435 |
| 953 | RT-DETR-L | 588 | 2.612 | 0,45 | 0,5316 | 0,6136 | 0,5618 | 0,5725 | 0,2687 | 0,5718 |
| **953** | **RF-DETR-L** | 588 | 2.612 | 0,37 | **0,5824** | **0,6017** | **0,5874** | **0,5964** | **0,2756** | 0,5965 |
| 763 | YOLO26l | 440 | 891 | 0,17 | 0,4778 | 0,5650 | 0,5173 | 0,5143 | 0,1902 | 0,5163 |
| 763 | RT-DETR-L | 440 | 891 | 0,48 | 0,5604 | 0,5893 | 0,5712 | 0,5563 | 0,2056 | 0,5580 |
| **763** | **RF-DETR-L** | 440 | 891 | 0,34 | **0,6089** | **0,6103** | **0,6046** | **0,6101** | **0,2335** | 0,6129 |
| 1716 | YOLO26l | 1.052 | 3.513 | 0,20 | 0,5604 | 0,5354 | 0,5406 | 0,5386 | 0,2393 | 0,5389 |
| 1716 | RT-DETR-L | 1.052 | 3.513 | 0,46 | 0,5630 | 0,6256 | 0,5898 | 0,5742 | 0,2456 | 0,5745 |
| **1716** | **RF-DETR-L** | 1.052 | 3.513 | 0,36 | **0,5846** | **0,6269** | **0,6039** | **0,5961** | **0,2522** | 0,5960 |

### 2.1 Rincian per Kelas, Korpus `953`

| Detektor | Kelas | Objek | Presisi | Recall | F1 | $AP50$ | $AP50\text{--}95$ |
|---|---|---|---|---|---|---|---|
| YOLO26l | B1 | 252 | 0,6555 | 0,7778 | 0,7114 | 0,7704 | 0,4022 |
| YOLO26l | B2 | 496 | 0,5312 | 0,4113 | 0,4636 | 0,4475 | 0,2110 |
| YOLO26l | B3 | 1.409 | 0,5741 | 0,6458 | 0,6079 | 0,6050 | 0,2748 |
| YOLO26l | B4 | 455 | 0,4095 | 0,4374 | 0,4230 | 0,3505 | 0,1379 |
| YOLO26l | **Semua** | 2.612 | **0,5426** | **0,5681** | **0,5515** | **0,5434** | **0,2565** |
| RT-DETR-L | B1 | 252 | 0,5730 | 0,8254 | 0,6764 | 0,7739 | 0,4183 |
| RT-DETR-L | B2 | 496 | 0,5538 | 0,4153 | 0,4747 | 0,4810 | 0,2270 |
| RT-DETR-L | B3 | 1.409 | 0,5782 | 0,7083 | 0,6367 | 0,6323 | 0,2776 |
| RT-DETR-L | B4 | 455 | 0,4212 | 0,5055 | 0,4595 | 0,4028 | 0,1517 |
| RT-DETR-L | **Semua** | 2.612 | **0,5316** | **0,6136** | **0,5618** | **0,5725** | **0,2687** |
| RF-DETR-L | B1 | 252 | 0,6933 | 0,8254 | 0,7536 | 0,8196 | 0,4383 |
| RF-DETR-L | B2 | 496 | 0,5251 | 0,4637 | 0,4925 | 0,5010 | 0,2364 |
| RF-DETR-L | B3 | 1.409 | 0,6068 | 0,7175 | 0,6576 | 0,6632 | 0,2869 |
| RF-DETR-L | B4 | 455 | 0,5042 | 0,4000 | 0,4461 | 0,4018 | 0,1408 |
| RF-DETR-L | **Semua** | 2.612 | **0,5824** | **0,6017** | **0,5874** | **0,5964** | **0,2756** |

### 2.2 Rincian per Kelas, Korpus `763`

| Detektor | Kelas | Objek | Presisi | Recall | F1 | $AP50$ | $AP50\text{--}95$ |
|---|---|---|---|---|---|---|---|
| YOLO26l | B1 | 173 | 0,5821 | 0,6763 | 0,6257 | 0,6847 | 0,2708 |
| YOLO26l | B2 | 321 | 0,5695 | 0,6511 | 0,6076 | 0,5877 | 0,2074 |
| YOLO26l | B3 | 334 | 0,4988 | 0,6467 | 0,5632 | 0,5916 | 0,2146 |
| YOLO26l | B4 | 63 | 0,2609 | 0,2857 | 0,2727 | 0,1929 | 0,0680 |
| YOLO26l | **Semua** | 891 | **0,4778** | **0,5650** | **0,5173** | **0,5143** | **0,1902** |
| RT-DETR-L | B1 | 173 | 0,6029 | 0,7110 | 0,6525 | 0,7377 | 0,2947 |
| RT-DETR-L | B2 | 321 | 0,5674 | 0,6822 | 0,6195 | 0,5889 | 0,2124 |
| RT-DETR-L | B3 | 334 | 0,7117 | 0,5988 | 0,6504 | 0,6542 | 0,2362 |
| RT-DETR-L | B4 | 63 | 0,3594 | 0,3651 | 0,3622 | 0,2445 | 0,0791 |
| RT-DETR-L | **Semua** | 891 | **0,5604** | **0,5893** | **0,5712** | **0,5563** | **0,2056** |
| RF-DETR-L | B1 | 173 | 0,7294 | 0,7168 | 0,7230 | 0,7758 | 0,3249 |
| RF-DETR-L | B2 | 321 | 0,5777 | 0,7414 | 0,6494 | 0,6353 | 0,2376 |
| RF-DETR-L | B3 | 334 | 0,6718 | 0,6497 | 0,6606 | 0,6893 | 0,2537 |
| RF-DETR-L | B4 | 63 | 0,4565 | 0,3333 | 0,3853 | 0,3401 | 0,1179 |
| RF-DETR-L | **Semua** | 891 | **0,6089** | **0,6103** | **0,6046** | **0,6101** | **0,2335** |

### 2.3 Rincian per Kelas, Korpus `1716`

| Detektor | Kelas | Objek | Presisi | Recall | F1 | $AP50$ | $AP50\text{--}95$ |
|---|---|---|---|---|---|---|---|
| YOLO26l | B1 | 443 | 0,6803 | 0,7111 | 0,6954 | 0,7298 | 0,3455 |
| YOLO26l | B2 | 808 | 0,4952 | 0,5062 | 0,5006 | 0,4764 | 0,2095 |
| YOLO26l | B3 | 1.749 | 0,5762 | 0,6398 | 0,6063 | 0,6065 | 0,2657 |
| YOLO26l | B4 | 513 | 0,4899 | 0,2846 | 0,3600 | 0,3419 | 0,1364 |
| YOLO26l | **Semua** | 3.513 | **0,5604** | **0,5354** | **0,5406** | **0,5386** | **0,2393** |
| RT-DETR-L | B1 | 443 | 0,7110 | 0,6885 | 0,6995 | 0,7308 | 0,3533 |
| RT-DETR-L | B2 | 808 | 0,4734 | 0,6176 | 0,5360 | 0,5120 | 0,2169 |
| RT-DETR-L | B3 | 1.749 | 0,5949 | 0,7421 | 0,6604 | 0,6466 | 0,2683 |
| RT-DETR-L | B4 | 513 | 0,4726 | 0,4542 | 0,4632 | 0,4074 | 0,1438 |
| RT-DETR-L | **Semua** | 3.513 | **0,5630** | **0,6256** | **0,5898** | **0,5742** | **0,2456** |
| RF-DETR-L | B1 | 443 | 0,6892 | 0,7607 | 0,7232 | 0,7688 | 0,3729 |
| RF-DETR-L | B2 | 808 | 0,5443 | 0,5854 | 0,5641 | 0,5369 | 0,2266 |
| RF-DETR-L | B3 | 1.749 | 0,6125 | 0,7130 | 0,6589 | 0,6645 | 0,2687 |
| RF-DETR-L | B4 | 513 | 0,4925 | 0,4483 | 0,4694 | 0,4141 | 0,1407 |
| RF-DETR-L | **Semua** | 3.513 | **0,5846** | **0,6269** | **0,6039** | **0,5961** | **0,2522** |

## 3. Pencacahan per Korpus Latih

Setiap baris memuat varian koefisien per kelas terbaik untuk detektor tersebut.

| Korpus latih | Detektor | Metode | $MAE$ makro | $RMSE$ makro | Bias mutlak makro | Akurasi ±1 makro |
|---|---|---|---|---|---|---|
| 953 | YOLO26l | $k$ per kelas | 1,0816 | 1,5408 | 0,1206 | 0,7447 |
| 953 | RT-DETR-L | $k + \tau$ per kelas | 1,0621 | 1,5733 | 0,2784 | 0,7429 |
| **953** | **RF-DETR-L** | **$k$ per kelas** | **1,0408** | **1,5087** | 0,1720 | 0,7429 |
| 763 | YOLO26l | $k$ per kelas | 0,6477 | 1,1638 | 0,2205 | 0,8591 |
| 763 | RT-DETR-L | $k$ per kelas | 0,6886 | 1,2506 | 0,2614 | 0,8455 |
| **763** | **RF-DETR-L** | **$k + \tau$ per kelas** | **0,6091** | **1,1023** | **0,1273** | **0,8614** |
| 1716 | YOLO26l | $k + \tau$ per kelas | 0,9348 | 1,4794 | 0,2519 | 0,7685 |
| 1716 | RT-DETR-L | $k$ per kelas | 0,8677 | 1,4495 | 0,2490 | 0,7928 |
| **1716** | **RF-DETR-L** | **$k$ per kelas** | **0,8473** | **1,4006** | **0,0982** | **0,7938** |

## 4. Rekap Terbaik per Korpus Latih

Seluruh metrik pencacahan bersifat makro atas kelas B1–B4.

| Korpus | Detektor | P | R | F1 | $mAP50$ | $mAP50\text{--}95$ | Metode | $MAE$ | $RMSE$ | Akurasi ±1 | $MAE$ total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 953 | RF-DETR-L | 0,5824 | 0,6017 | 0,5874 | 0,5964 | 0,2756 | $k$ per kelas | 1,0408 | 1,5087 | 0,7429 | 1,9787 |
| 763 | RF-DETR-L | 0,6089 | 0,6103 | 0,6046 | 0,6101 | 0,2335 | $k + \tau$ per kelas | 0,6091 | 1,1023 | 0,8614 | 1,0909 |
| 1716 | RF-DETR-L | 0,5846 | 0,6269 | 0,6039 | 0,5961 | 0,2522 | $k$ per kelas | 0,8473 | 1,4006 | 0,7938 | 1,7471 |

## 5. Koefisien Konfigurasi Pencacahan Terbaik

Nilai adalah rerata lima lipatan.

| Korpus latih | Metode | $k_{B1}$ | $k_{B2}$ | $k_{B3}$ | $k_{B4}$ | $\tau_{B1}$ | $\tau_{B2}$ | $\tau_{B3}$ | $\tau_{B4}$ |
|---|---|---|---|---|---|---|---|---|---|
| 953 | $k$ per kelas | 0,30 | 0,28 | 0,33 | 0,38 | 0,30 | 0,30 | 0,30 | 0,30 |
| 763 | $k + \tau$ per kelas | 0,42 | 0,42 | 0,68 | 0,62 | 0,32 | 0,27 | 0,40 | 0,31 |
| 1716 | $k$ per kelas | 0,37 | 0,47 | 0,46 | 0,51 | 0,35 | 0,35 | 0,35 | 0,35 |

## 6. Rincian per Kelas pada Konfigurasi Terbaik

| Korpus latih | Metrik | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|
| 953 | $MAE$ | 0,376 | 1,014 | 1,539 | 1,234 |
| 953 | Bias | −0,050 | −0,248 | −0,163 | −0,227 |
| 953 | Akurasi ±1 | 0,986 | 0,766 | 0,560 | 0,660 |
| 763 | $MAE$ | 0,436 | 0,845 | 0,745 | 0,409 |
| 763 | Bias | −0,127 | +0,027 | −0,236 | −0,118 |
| 763 | Akurasi ±1 | 0,900 | 0,782 | 0,827 | 0,936 |
| 1716 | $MAE$ | 0,412 | 0,949 | 1,276 | 0,751 |
| 1716 | Bias | −0,163 | −0,125 | 0,000 | −0,105 |
| 1716 | Akurasi ±1 | 0,938 | 0,747 | 0,661 | 0,829 |

## 7. Efek Kalibrasi pada RF-DETR-L

| Korpus latih | $MAE$ naif | $MAE$ terkalibrasi | Perubahan | $RMSE$ naif | $RMSE$ terkalibrasi | Akurasi ±1 naif | Akurasi ±1 terkalibrasi |
|---|---|---|---|---|---|---|---|
| 953 | 6,7039 | 1,0408 | −84,5% | 7,8535 | 1,5087 | 0,2092 | 0,7429 |
| 763 | 1,7864 | 0,6091 | −65,9% | 2,5912 | 1,1023 | 0,6068 | 0,8614 |
| 1716 | 4,1858 | 0,8473 | −79,8% | 5,7300 | 1,4006 | 0,3784 | 0,7938 |

## 8. Perbandingan Metode Kalibrasi ($MAE$ makro, rerata tiga detektor)

| Korpus latih | Naif | $k$ global | $k$ per kelas | $k + \tau$ per kelas |
|---|---|---|---|---|
| 953 | 6,0479 | 1,0851 | 1,0839 | **1,0833** |
| 763 | 1,7561 | **0,6538** | 0,6583 | 0,6674 |
| 1716 | 4,0110 | 0,9141 | **0,8911** | 0,9050 |

## 9. Varian Koefisien pada RF-DETR-L

| Korpus latih | $k$ global | $k$ per kelas | $k + \tau$ per kelas |
|---|---|---|---|
| 953 | 1,0372 | **1,0408** | 1,0833 |
| 763 | 0,6205 | 0,6386 | **0,6091** |
| 1716 | 0,8765 | **0,8473** | 0,8959 |

Angka bercetak tebal adalah varian yang dipakai pada Bagian 3 sampai Bagian 7. Pada korpus 953, koefisien tunggal menghasilkan $MAE$ $0,0036$ lebih rendah, yakni di bawah batas keterpisahan $0,05$, sehingga varian per kelas tetap dipilih agar seluruh korpus memakai bentuk koefisien yang sama.

## 10. Uji Silang: Detektor Latih `763` pada Partisi Uji `953` ($n = 141$)

| Detektor | $mAP50$ pada 953 | Metode terbaik | $MAE$ makro | $RMSE$ makro | Bias mutlak makro | Akurasi ±1 makro |
|---|---|---|---|---|---|---|
| YOLO26l | 0,2331 | $k + \tau$ per kelas | 1,2961 | 1,7660 | 0,3316 | 0,6702 |
| RT-DETR-L | 0,1110 | $k + \tau$ per kelas | 1,5142 | 2,1154 | 0,4291 | 0,6277 |
| RF-DETR-L | 0,1774 | $k + \tau$ per kelas | 1,2518 | 1,8023 | 0,3475 | 0,6702 |

Pembanding dalam domain pada partisi uji yang sama: $MAE$ makro $1,0408$ sampai $1,0816$.

## 11. Matriks Permutasi Koefisien

Koefisien dari setiap kombinasi diterapkan ke seluruh kombinasi lain. Nilai sel adalah $MAE$ makro pada kombinasi sasaran. Label kolom memakai singkatan Y untuk YOLO26l, RT untuk RT-DETR-L, dan RF untuk RF-DETR-L.

| Sumber koefisien | 953 Y | 953 RT | 953 RF | 763 Y | 763 RT | 763 RF | 1716 Y | 1716 RT | 1716 RF |
|---|---|---|---|---|---|---|---|---|---|
| 953 Y | **1,0816** | 5,5284 | 4,4202 | 0,7023 | 1,3432 | 1,3364 | 0,9300 | 3,6955 | 2,7198 |
| 953 RT | 1,7961 | **1,1294** | 1,2394 | 0,9409 | 0,7432 | 0,7295 | 1,4864 | 0,9601 | 0,9981 |
| 953 RF | 1,5585 | 1,3599 | **1,0408** | 0,8773 | 0,8523 | 0,7455 | 1,3230 | 1,0700 | 0,9037 |
| 763 Y | 1,1011 | 4,6507 | 3,3227 | **0,6477** | 1,3114 | 0,9182 | 0,9446 | 2,9465 | 1,8959 |
| 763 RT | 1,8848 | 1,2801 | 1,7553 | 1,0136 | **0,6886** | 0,7227 | 1,5642 | 1,0263 | 1,1936 |
| 763 RF | 1,4486 | 2,0904 | 1,3972 | 0,8045 | 0,8227 | **0,6386** | 1,2374 | 1,5331 | 1,0399 |
| 1716 Y | 1,1135 | 5,8564 | 4,5691 | 0,7636 | 1,4068 | 1,3841 | **0,9582** | 3,7140 | 2,5798 |
| 1716 RT | 1,9344 | 1,1525 | 1,6613 | 1,0364 | 0,7091 | 0,7523 | 1,5885 | **0,8677** | 1,1605 |
| 1716 RF | 1,4309 | 1,6915 | 1,0869 | 0,8182 | 0,7364 | 0,6295 | 1,2656 | 1,2597 | **0,8473** |

Angka bercetak tebal pada diagonal adalah koefisien yang dipasang pada kombinasi itu sendiri.

| Jenis permutasi | Jumlah sel | $MAE$ rerata | Terendah | Tertinggi |
|---|---|---|---|---|
| Koefisien sendiri | 9 | 0,8778 | 0,6386 | 1,1294 |
| Korpus sama, detektor berbeda | 18 | 1,8369 | 0,7227 | 5,5284 |
| Detektor sama, korpus berbeda | 18 | 0,9572 | 0,6295 | 1,3972 |
| Korpus dan detektor berbeda | 36 | 1,8381 | 0,7295 | 5,8564 |

## 12. Glosarium Simbol dan Istilah

| Simbol atau istilah | Arti |
|---|---|
| B1 sampai B4 | Kelas kematangan tandan. B1 berarti lewat matang atau siap panen, B4 berarti mentah |
| $n_c(t)$ | Jumlah deteksi kelas $c$ pada seluruh sisi pohon $t$ yang lolos ambang skor keyakinan |
| $k_c$ | Koefisien pengali kelas $c$. Nilai di bawah $1,00$ menurunkan hitungan mentah, nilai di atas $1,00$ menaikkannya |
| $\tau_c$ | Ambang skor keyakinan kelas $c$. Deteksi dengan skor di bawah ambang ini tidak ikut dihitung |
| $\hat{y}_c(t)$ | Hitungan akhir kelas $c$ pada pohon $t$ setelah kalibrasi |
| $\text{conf}^{*}$ | Ambang skor keyakinan yang memaksimalkan F1 makro pada satu model |
| Presisi | Proporsi deteksi yang cocok dengan anotasi acuan |
| Recall | Proporsi anotasi acuan yang berhasil terdeteksi |
| F1 | Rerata harmonik presisi dan recall |
| IoU | Rasio luas irisan terhadap luas gabungan dua kotak pembatas (*bounding box*) |
| $AP50$ | Luas di bawah kurva presisi-recall satu kelas pada ambang IoU $0,50$ |
| $AP50\text{--}95$ | Rerata $AP$ pada sepuluh ambang IoU, dari $0,50$ sampai $0,95$ |
| $mAP50$ dan $mAP50\text{--}95$ | Rerata $AP50$ dan $AP50\text{--}95$ atas empat kelas |
| $MAE$ | Rerata galat absolut hitungan per pohon, bersatuan tandan |
| $RMSE$ | Akar rerata kuadrat galat. Memberi bobot lebih besar pada galat yang besar |
| Bias | Rerata selisih bertanda. Nilai negatif berarti hitungan lebih rendah daripada acuan |
| Akurasi ±1 | Proporsi pohon dengan selisih hitungan paling banyak satu tandan |
| Makro | Rerata tanpa bobot atas empat kelas B1–B4 |
| Naif | Pencacahan tanpa kalibrasi, yakni $k = 1$ dan $\tau = 0,25$ |
| $k$ global | Satu nilai $k$ dan satu nilai $\tau$ berlaku untuk seluruh kelas |
| $k$ per kelas | Nilai $k$ berbeda pada tiap kelas, dengan $\tau$ tunggal |
| $k + \tau$ per kelas | Nilai $k$ dan $\tau$ berbeda pada tiap kelas |
| Lipat-silang 5 lipatan | Pohon uji dibagi lima kelompok. Koefisien untuk satu kelompok dipasang pada empat kelompok lain, lalu diterapkan pada kelompok yang ditahan |
| Korpus 953, 763, dan 1716 | Korpus latih SawitMVC-YOLO, SawitMVC-Depth-YOLO, dan gabungan keduanya |
| $n$ | Jumlah pohon pada partisi uji yang dilaporkan |
| P dan R | Singkatan Presisi dan Recall pada Bagian 4 |
| Y, RT, RF | Singkatan detektor YOLO26l, RT-DETR-L, dan RF-DETR-L pada Bagian 11 |
