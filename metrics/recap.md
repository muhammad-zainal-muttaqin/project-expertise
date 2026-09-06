# Rekapitulasi Papan Peringkat Hasil Eksperimen

Menyusun ulang angka [atlas metrik](README.md) menjadi satu papan peringkat
(*leaderboard*) per tugas. Seluruh korpus dan sistem digabung dalam satu tabel,
diurutkan dari terbaik ke terburuk menurut metrik utama tugas tersebut, dengan
metrik pendampingnya disertakan pada tabel yang sama.

Cakupan: `V2-E-001`–`V2-E-045`, `PT-E-000`–`PT-E-036`, verifikasi remote Agustus
2026, dan audit forensik `AF-E-001`–`AF-E-016`. Aditif; tidak ada angka baru dan
tidak ada berkas atlas lama yang diubah.

---

## 1. Daftar tugas dan metrik

| Tugas | Arti | Satuan | Metrik utama | Metrik pendamping | Tabel |
|---|---|---|---|---|---|
| Deteksi empat kelas | Kotak pembatas (*bounding box*) dan label kematangan sekaligus, dari satu detektor satu tahap | per citra | $mAP50$ | $mAP50\text{--}95$, $AP$ per kelas B1–B4 | [3](#3-deteksi-empat-kelas) |
| Deteksi agnostik | Menemukan kotak pembatas tandan tanpa label kelas, yaitu lokalisasi murni | per citra | $AP50_{agn}$ | $AP50\text{--}95_{agn}$, presisi, daya tangkap, F1 | [4](#4-deteksi-agnostik-lokalisasi-murni) |
| Deduplikasi multi-tampak | Satu tandan yang tampak di beberapa sisi dihitung satu kali | per pohon | F1 fisik | Presisi fisik, daya tangkap fisik | [5](#5-deduplikasi-multi-tampak) |
| Klasifikasi kematangan | Ketepatan kelas pada tandan yang sudah tertaut benar | per pohon | Akurasi *matched-class* | Makro-F1 ujung ke ujung, F1 per kelas B1–B4 | [6](#6-klasifikasi-kematangan) |
| Pencacahan per pohon | Selisih cacah tandan tiap pohon terhadap acuannya | per pohon | MAE cacah total | Akurasi $\pm 1$, akurasi tepat persis | [7](#7-pencacahan-per-pohon) |
| Pencacahan kohort | Jumlah tandan per kelas B1–B4 untuk seluruh partisi | per kohort | Bias per kelas | Makro-rerata nilai mutlak bias | [8](#8-pencacahan-kohort-per-kelas) |

Sistem bekerja berurutan: detektor menemukan tandan per citra, penaut
menggabungkan kemunculan tandan yang sama dari empat sisi, pengklasifikasi
menentukan kematangannya, pencacah menjumlahkan per kelas. Penyebut tiap tugas
berbeda, sehingga angka antar-tabel tidak dapat dibandingkan.

### 1.1 Metrik yang sering tertukar

| Metrik | Mengukur | Bukan |
|---|---|---|
| $AP50_{agn}$ $0,8350$ | Presisi rerata lokalisasi $83,50\%$, kelas dilipat jadi satu | Akurasi kematangan, akurasi pencacahan |
| F1 fisik | Ketepatan klaster tandan gabungan multi-sisi | F1 pasangan (relasi dua kotak antar-sisi), F1 titik operasi deteksi |
| Akurasi *matched-class* | Ketepatan kelas pada klaster yang berpasangan saja | Makro-F1 ujung ke ujung, yang ikut menghitung klaster palsu dan tandan terlewat |
| Bias kohort | Selisih total prediksi dan total acuan per kelas, galat antar-pohon saling meniadakan | MAE per pohon, yang menghitung galat tiap pohon |

---

## 2. Korpus

| Korpus | Modalitas | Ukuran | Citra uji | Pohon uji empat sisi | Cakupan tugas |
|---|---|---:|---:|---:|---|
| `combined1716` | RGB | 1.716 pohon | 1.052 | — | Deteksi saja |
| `763-depth` | RGB + kedalaman Y16 | 763 pohon | 440 | 110 | Keenam tugas |
| `953` | RGB | 953 pohon | 588 | 135 | Keenam tugas |

`combined1716` adalah gabungan korpus 953 dan 763 yang berperan sebagai bank
pelatihan detektor. Pool 1.716 pohonnya tidak memiliki nilai acuan kebenaran
(*ground truth*) multi-sisi tingkat pohon, sehingga tidak muncul pada tabel 5
sampai 8.

Baris berlabel "bank `combined1716`" adalah bobot yang dilatih pada korpus itu
lalu diuji pada korpus lain. Baris berlabel "native" dilatih dan diuji pada
korpus yang sama.

### 2.1 Selisih protokol anotasi antar-korpus (`AF-E-001`)

Diukur pada 352 pohon fisik yang sama persis di kedua rilis.

| Besaran | 953 (Mei 2026) | 763-depth (Juli–Agustus 2026) | Perubahan |
|---|---:|---:|---:|
| Tandan unik per pohon | 9,89 | 3,99 | −59,7% |
| Proporsi sisi tanpa anotasi | 1,1% | 14,2% | +13,1 pp |
| Proporsi kelas B1 | acuan | | +66% |
| Proporsi kelas B4 | acuan | | −85% |

Selisih lintas korpus karena itu sebagian besar mengukur selisih protokol
pengamatan, dan bukan kegagalan generalisasi lintas domain.

---

## 3. Deteksi empat kelas

Metrik utama $mAP50$ sadar-kelas; pendamping $mAP50\text{--}95$ dan $AP50$ per
kelas. Evaluasi `pycocotools.COCOeval` pada partisi uji. Diurutkan dari $mAP50$
tertinggi.

| Sistem | Korpus | $mAP50$ | $mAP50\text{--}95$ | $AP_{B1}$ | $AP_{B2}$ | $AP_{B3}$ | $AP_{B4}$ | ID | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| RF-DETR-L, bank `combined1716` | 763-depth | **0,6711** | 0,2748 | 0,8044 | **0,7187** | **0,7373** | 0,4239 | `V2-E-042` | uji |
| WBF [YOLO26l+RT-DETR-L+RF-DETR-L], bank `combined1716` | 763-depth | 0,6691 | **0,2757** | 0,8280 | 0,6919 | 0,7349 | 0,4214 | `V2-E-042` | uji |
| Plafon lokalisasi sempurna + ConvNeXt-Tiny | 953 | 0,6569 | N/A¹ | 0,7653 | 0,4912 | 0,7625 | **0,6088** | `AF-E-005` | plafon |
| WBF + *re-ranker* | 763-depth | 0,6552 | 0,2701 | **0,8288** | 0,6863 | 0,7362 | 0,3693 | `MAP_BOOST` | uji |
| RT-DETR-L, bank `combined1716` | 763-depth | 0,6309 | 0,2496 | 0,7865 | 0,6652 | 0,7160 | 0,3559 | `V2-E-042` | uji |
| RF-DETR-L native | 763-depth | 0,6129 | 0,2335 | 0,7758 | 0,6353 | 0,6997 | 0,3406 | `V2-E-034` | uji |
| RF-DETR-L native | 953 | 0,6012 | 0,2747 | 0,8150 | 0,5184 | 0,6553 | 0,4160 | `V2-E-001` | uji |
| WBF + *re-ranker* | 953 | 0,5970 | 0,2743 | 0,8042 | 0,4942 | 0,6566 | 0,4328 | `MAP_BOOST` | uji |
| RF-DETR-L native | `combined1716` | 0,5960 | 0,2522 | 0,7654 | 0,5394 | 0,6652 | 0,4141 | `V2-E-035` | uji |
| WBF, bank `combined1716` | 953 | 0,5861 | 0,2753 | 0,7869 | 0,4866 | 0,6513 | 0,4197 | `V2-E-042` | uji |
| RT-DETR-L native | 953 | 0,5781 | 0,2629 | 0,7874 | 0,4614 | 0,6371 | 0,4266 | `V2-E-001` | uji |
| YOLO26l, bank `combined1716` | 763-depth | 0,5765 | 0,2387 | 0,7839 | 0,6088 | 0,6380 | 0,2754 | `V2-E-042` | uji |
| RT-DETR-L native | `combined1716` | 0,5745 | 0,2458 | 0,7308 | 0,5120 | 0,6465 | 0,4089 | `V2-E-035` | uji |
| RT-DETR-L native | 763-depth | 0,5580 | 0,2055 | 0,7377 | 0,5889 | 0,6607 | 0,2445 | `V2-E-034` | uji |
| WBF native | `combined1716` | 0,5538 | N/A² | 0,7286 | 0,4732 | 0,6372 | 0,3760 | `V2-E-039` | uji |
| YOLO26l native | 953 | 0,5435 | 0,2564 | 0,7705 | 0,4479 | 0,6050 | 0,3506 | `V2-E-001` | uji |
| YOLO26s 960 px, replikasi audit | 953 | 0,5433 | N/A² | 0,7588 | 0,4495 | 0,6095 | 0,3553 | `AF-E-006` | uji |
| YOLO26l native | `combined1716` | 0,5389 | 0,2395 | 0,7298 | 0,4766 | 0,6075 | 0,3419 | `V2-E-035` | uji |
| YOLO26l native | 763-depth | 0,5163 | 0,1906 | 0,6847 | 0,5877 | 0,6005 | 0,1920 | `V2-E-034` | uji |

¹ `N/A: bukan metrik uji; baris ini adalah batas atas teoretis, bukan detektor.`
² `N/A: tidak dilaporkan pada artefak sumber.`

Baris plafon `AF-E-005` menempatkan kotak acuan sebagai prediksi, sehingga galat
lokalisasi dihilangkan menurut konstruksi. Nilai $0,6569$ hanya sah dibaca
sebagai batas atas. Capaian nyata tertinggi pada korpus 953 adalah $0,5970$,
yaitu $91\%$ dari plafon itu.

Urutan **RF-DETR-L > RT-DETR-L > YOLO26l** konsisten pada ketiga korpus. Baris
`AF-E-006` menggunakan detektor dan resolusi berbeda; kedekatannya dengan
`V2-E-001` ($0,5433$ berbanding $0,5435$) adalah kalibrasi replikasi audit, bukan
perbandingan arsitektur.

Baris korpus 352 dan varian kedalaman monokular tidak dimuat di sini; lihat
[`01_deteksi_dan_lokalisasi.md`](01_deteksi_dan_lokalisasi.md).

---

## 4. Deteksi agnostik (lokalisasi murni)

Metrik utama $AP50$ dengan keempat kelas dilipat menjadi satu kategori;
pendamping $AP50\text{--}95$ serta presisi, daya tangkap, dan F1 pada titik
operasi ambang kepercayaan $0,25$ dan $IoU = 0,5$. Diurutkan dari $AP50$
tertinggi.

| Sistem | Korpus | $AP50_{agn}$ | $AP50\text{--}95_{agn}$ | Presisi | Daya tangkap | F1 | Citra | ID |
|---|---|---:|---:|---:|---:|---:|---:|---|
| WBF + *re-ranker* | 763-depth | **0,8783** | 0,3523 | N/A¹ | N/A¹ | N/A¹ | 440 | `MAP_BOOST` |
| WBF, bank `combined1716` | 763-depth | 0,8764 | 0,3519 | 0,9296 | 0,6813 | **0,7863** | 440 | `V2-E-042` |
| WBF + *re-ranker* | 953 | 0,8419 | **0,3717** | N/A¹ | N/A¹ | N/A¹ | 588 | `MAP_BOOST` |
| WBF, bank `combined1716` | 953 | 0,8350 | 0,3679 | 0,8825 | 0,6443 | 0,7449 | 588 | `V2-E-042` |
| WBF native | `combined1716` | 0,8104 | 0,3363 | **0,9371** | 0,0763² | 0,1411² | 1.052 | `V2-E-039` |
| YOLO26m 1.280 px tunggal | 953 | 0,8104 | N/A³ | 0,7942 | 0,7271 | N/A³ | 588 | `AF-E-011` |
| YOLO26s 960 px tunggal | 953 | 0,8057 | N/A³ | 0,7965 | 0,7087 | N/A³ | 588 | `AF-E-006` |
| RF-DETR-L native | 763-depth | 0,7951 | 0,3003 | 0,5748 | 0,8361 | 0,6813 | 440 | `V2-E-036` |
| RF-DETR-L native | `combined1716` | 0,7850 | 0,3245 | 0,4820 | **0,8497** | 0,6151 | 1.052 | `V2-E-036` |
| YOLO26l, kelas dilipat | 953 | 0,7388 | 0,3312 | 0,7547 | 0,6336 | 0,6889 | 588 | `V2-E-017` |

¹ `N/A: artefak MAP_BOOST tidak menyimpan metrik titik operasi.`
² Kalibrasi skor WBF pada korpus ini berbeda, sehingga ambang tetap $0,25$
menyisakan daya tangkap $7,63\%$. Nilai $AP50$ tidak terpengaruh karena dihitung
melintasi seluruh ambang.
³ `N/A: tidak dilaporkan pada artefak sumber.`

Baris `AF-E-011` menunjukkan satu detektor besar mendekati ansambel tiga detektor
($0,8104$ berbanding $0,8350$), tetapi tidak melampauinya.

---

## 5. Deduplikasi multi-tampak

Metrik utama F1 fisik pada pencocokan $IoU \ge 0,5$; pendamping presisi dan daya
tangkap tingkat klaster. Diurutkan dari F1 tertinggi.

| Sistem | Korpus | Partisi | $n$ pohon | F1 fisik | Presisi | Daya tangkap | ID |
|---|---|---|---:|---:|---:|---:|---|
| WBF + penaut *greedy strict* | 763-depth | uji | 110 | **0,8590** | 0,8799 | **0,8390** | `V2-E-043`⁴ |
| WBF + penaut GSP MILP + pencacah Ridge | 763-depth | uji | 110 | 0,8534 | 0,8926 | 0,8175 | `Wave-V2` |
| WBF + penaut GSP MILP + pencacah Ridge | 763-depth | validasi | 117 | 0,8526 | **0,9055** | 0,8056 | `Wave-V2` |
| WBF + penaut Hungarian *Anchor A* + pencacah Ridge | 953 | uji | 135 | 0,8387 | 0,8444 | 0,8331 | `Wave-V2` |
| WBF + penaut *greedy strict* | 953 | uji | 135 | 0,8296 | 0,8247 | 0,8346 | `V2-E-043`⁴ |
| WBF + penaut prior rotasi + pencacah Ridge | 763-depth | validasi | 117 | 0,8257 | 0,8431 | 0,8091 | `V2-E-045` |
| WBF + penaut Hungarian *Anchor A* + pencacah Ridge | 953 | validasi | 91 | 0,8232 | 0,8206 | 0,8259 | `Wave-V2` |
| WBF + penaut prior rotasi + pencacah Ridge | 953 | validasi | 91 | 0,8087 | 0,8044 | 0,8130 | `V2-E-045` |
| WBF + penaut prior rotasi + pencacah Ridge | 763-depth | uji | 110 | 0,8069 | 0,8142 | 0,7996 | `V2-E-045` |
| WBF + penaut prior rotasi + pencacah Ridge | 953 | uji | 135 | 0,8043 | 0,8092 | 0,7996 | `V2-E-045` |
| YOLO26m + penaut terlatih (Pipeline Panen) | 953 | uji | 132 | 0,7619 | 0,8538 | 0,6878 | `AF-E-012` |
| YOLO26m + penaut terlatih (Pipeline Panen) | 953 | validasi | 91 | 0,7586 | 0,8374 | 0,6934 | `AF-E-012` |

⁴ Parameter `V2-E-043` dipilih melalui sapuan *greedy* langsung pada partisi uji,
sehingga bukan angka generalisasi. Profil `Wave-V2` dikunci pada partisi uji;
profil `V2-E-045` dikunci pada partisi validasi.

Selang kepercayaan kedua profil terkunci uji bertumpang tindih (bagian 7.1),
sehingga selisih $0,8534$ berbanding $0,8387$ belum dapat dipisahkan secara
statistik.

---

## 6. Klasifikasi kematangan

Metrik utama akurasi *matched-class* pada klaster yang berpasangan dengan acuan;
pendamping makro-F1 ujung ke ujung yang ikut menghitung klaster palsu dan tandan
terlewat, serta F1 per kelas. Diurutkan dari akurasi tertinggi.

| Sistem | Korpus | Partisi | *Matched-class* | Makro-F1 E2E | $F1_{B1}$ | $F1_{B2}$ | $F1_{B3}$ | $F1_{B4}$ | ID |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| WBF + GSP MILP | 763-depth | validasi | **84,57%** | **0,6807** | **0,7778** | **0,7244** | **0,7500** | 0,4706 | `Wave-V2` |
| WBF + prior rotasi | 763-depth | validasi | 83,55% | 0,6749 | N/A¹ | N/A¹ | N/A¹ | N/A¹ | `V2-E-045` |
| WBF + GSP MILP | 763-depth | uji | 81,62% | 0,6519 | 0,7578 | 0,7230 | 0,7092 | 0,4176 | `Wave-V2` |
| WBF + prior rotasi | 763-depth | uji | 80,31% | 0,6047 | N/A¹ | N/A¹ | N/A¹ | N/A¹ | `V2-E-045` |
| WBF + Hungarian *Anchor A* | 953 | validasi | 75,42% | 0,6014 | 0,7059 | 0,4698 | 0,6737 | **0,5561** | `Wave-V2` |
| WBF + Hungarian *Anchor A* | 953 | uji | 74,42% | 0,6034 | 0,7465 | 0,4706 | 0,6850 | 0,5114 | `Wave-V2` |
| YOLO26m + skor ordinal (Pipeline Panen) | 953 | validasi | 71,65% | 0,6683⁵ | N/A¹ | N/A¹ | N/A¹ | N/A¹ | `AF-E-012` |
| YOLO26m + skor ordinal (Pipeline Panen) | 953 | uji | 71,61% | 0,6692⁵ | N/A¹ | N/A¹ | N/A¹ | N/A¹ | `AF-E-012` |
| WBF + prior rotasi | 953 | uji | 71,11% | 0,5384 | N/A¹ | N/A¹ | N/A¹ | N/A¹ | `V2-E-045` |
| WBF + prior rotasi | 953 | validasi | 70,04% | 0,5462 | N/A¹ | N/A¹ | N/A¹ | N/A¹ | `V2-E-045` |

¹ `N/A: artefak sumber tidak menyimpan rincian per kelas.`
⁵ Makro-F1 Pipeline Panen dihitung hanya pada klaster yang berpasangan, sehingga
tidak setara dengan kolom yang sama pada baris lain. Nilai ujung ke ujungnya
$0,5201$.

Kolom F1 per kelas memperlihatkan sumber kelemahan pada korpus 953: kelas B2
hanya mencapai $0,47$, sedangkan tiga kelas lain berada pada pita $0,51$–$0,75$.
Pada korpus 763-depth, kelas B4 yang menjadi kelemahannya ($0,42$).

---

## 7. Pencacahan per pohon

Metrik utama MAE cacah total per pohon; pendamping akurasi toleransi $\pm 1$ dan
akurasi tepat persis. Diurutkan dari MAE terendah.

| Sistem | Korpus | Partisi | $n$ pohon | MAE total | $\pm 1$ | Tepat persis | ID |
|---|---|---|---:|---:|---:|---:|---|
| WBF + prior rotasi + Ridge | 763-depth | validasi | 117 | **0,726** | 84,62% | 44,44% | `V2-E-045` |
| WBF + GSP MILP + Ridge | 763-depth | uji | 110 | 0,773 | **85,45%** | **44,55%** | `Wave-V2` |
| WBF + *greedy strict*, cacah klaster mentah | 763-depth | uji | 110 | 0,818 | 83,64% | 41,82% | `V2-E-043`⁴ |
| WBF + prior rotasi + Ridge | 763-depth | uji | 110 | 0,891 | 80,91% | 33,64% | `V2-E-045` |
| WBF + GSP MILP + Ridge | 763-depth | validasi | 117 | 0,932 | 78,63% | 34,19% | `Wave-V2` |
| Plafon: kotak acuan + Ridge per kelas | 953 | uji | 136 | 1,058 | 75,40% | 29,00% | `AF-E-004` |
| WBF + Hungarian / prior rotasi + Ridge | 953 | validasi | 91 | 1,253 | 67,03% | 28,57% | `Wave-V2`, `V2-E-045`⁶ |
| WBF + Hungarian *Anchor A* + Ridge | 953 | uji | 135 | 1,363 | 63,70% | 27,41% | `Wave-V2` |
| YOLO26m + penaut terlatih + Ridge | 953 | validasi | 91 | 1,374 | 63,74% | 26,37% | `AF-E-013` |
| WBF + prior rotasi + Ridge | 953 | uji | 135 | 1,393 | 61,48% | 25,93% | `V2-E-045` |
| YOLO26m + penaut terlatih + Ridge | 953 | uji | 132 | 1,402 | 56,82% | 22,73% | `AF-E-013` |
| WBF + *greedy strict*, cacah klaster mentah | 953 | uji | 135 | 1,644 | 54,07% | N/A² | `V2-E-043`⁴ |

⁶ Lapisan pencacah Ridge bekerja pada fitur proposal, terpisah dari penaut,
sehingga kedua profil menghasilkan MAE identik pada partisi validasi 953.

Baris plafon `AF-E-004` mengganti seluruh tahap deteksi dengan kotak acuan.
Bahkan dengan lokalisasi sempurna, cacah total tepat persis hanya mencapai
$29,00\%$, sedangkan capaian nyata sudah $27,41\%$. Rinciannya per kelas ada di
bagian 9.3.

### 7.1 Selang kepercayaan 95% profil terkunci uji

*Bootstrap* berpasangan 2.000 ulangan, *seed* 42, pada seluruh metrik pipeline.

| Metrik | 953, Hungarian *Anchor A* (135 pohon) | 763-depth, GSP MILP (110 pohon) |
|---|---|---|
| F1 fisik | $0,8387$ $[0,8174; 0,8587]$ | $0,8534$ $[0,8301; 0,8761]$ |
| Akurasi *matched-class* | $0,7442$ $[0,7112; 0,7735]$ | $0,8162$ $[0,7765; 0,8556]$ |
| Makro-F1 ujung ke ujung | $0,6034$ $[0,5655; 0,6382]$ | $0,6519$ $[0,6046; 0,6918]$ |
| MAE cacah total | $1,363$ $[1,163; 1,585]$ | $0,773$ $[0,609; 0,945]$ |
| Cacah $\pm 1$ | $0,6370$ $[0,5556; 0,7185]$ | $0,8545$ $[0,7818; 0,9182]$ |

---

## 8. Pencacahan kohort per kelas

Metrik penilaian akhir tugas pencacahan, dari profil penaut terkunci `Wave-V2`,
dengan menjumlahkan seluruh pohon dalam satu partisi. Bias absolut bersatuan
tandan; bias relatif adalah bias absolut dibagi total acuan kelasnya.

Skala kelas menurun dari paling matang ke paling mentah
([`docs/DATASET.md`](../docs/DATASET.md) §1).

| Kelas | Tingkat kematangan | Ciri visual | Ukuran kotak median, 953 |
|---|---|---|---:|
| B1 | Lewat matang, siap panen | Jingga-kemerahan cerah, posisi terbawah | 133 px |
| B2 | Matang optimal | Oranye kemerahan bersemburat ungu kehitaman | 120 px |
| B3 | Matang awal, mengkal | Ungu kemerahan kehitaman | 107 px |
| B4 | Mentah, muda | Hitam kehijauan pekat, tertanam di sela pelepah | 93 px |

### 8.1 Korpus 953, partisi uji (Hungarian *Anchor A*, 135 pohon)

| Kelas | Total prediksi | Total acuan | Bias absolut | Bias relatif |
|---|---:|---:|---:|---:|
| B1 | 104 | 113 | −9 | −7,96% |
| B2 | 145 | 246 | −101 | −41,06% |
| B3 | 824 | 706 | +118 | +16,71% |
| B4 | 251 | 277 | −26 | −9,39% |
| **Total** | **1.324** | **1.342** | **−18** | **−1,34%** |

Makro-rerata nilai mutlak bias relatif $18,78\%$.

### 8.2 Korpus 953, partisi validasi (Hungarian *Anchor A*, 91 pohon)

| Kelas | Total prediksi | Total acuan | Bias absolut | Bias relatif |
|---|---:|---:|---:|---:|
| B1 | 84 | 86 | −2 | −2,33% |
| B2 | 112 | 186 | −74 | −39,78% |
| B3 | 560 | 476 | +84 | +17,65% |
| B4 | 186 | 188 | −2 | −1,06% |
| **Total** | **942** | **936** | **+6** | **+0,64%** |

Makro-rerata nilai mutlak bias relatif $15,21\%$.

### 8.3 Korpus 763-depth, partisi uji (GSP MILP, 110 pohon)

| Kelas | Total prediksi | Total acuan | Bias absolut | Bias relatif |
|---|---:|---:|---:|---:|
| B1 | 67 | 94 | −27 | −28,72% |
| B2 | 227 | 199 | +28 | +14,07% |
| B3 | 177 | 215 | −38 | −17,67% |
| B4 | 41 | 50 | −9 | −18,00% |
| **Total** | **512** | **558** | **−46** | **−8,24%** |

Makro-rerata nilai mutlak bias relatif $19,62\%$.

### 8.4 Korpus 763-depth, partisi validasi (GSP MILP, 117 pohon)

| Kelas | Total prediksi | Total acuan | Bias absolut | Bias relatif |
|---|---:|---:|---:|---:|
| B1 | 66 | 96 | −30 | −31,25% |
| B2 | 234 | 205 | +29 | +14,15% |
| B3 | 176 | 216 | −40 | −18,52% |
| B4 | 32 | 53 | −21 | −39,62% |
| **Total** | **508** | **570** | **−62** | **−10,88%** |

Makro-rerata nilai mutlak bias relatif $25,88\%$.

### 8.5 Catatan pembacaan

Pada korpus 953, B2 diestimasi kurang sekitar $40\%$ dan B3 diestimasi berlebih
sekitar $17\%$, konsisten di kedua partisi. Arah berlawanan dengan besaran
sebanding itu menunjukkan perpindahan massa prediksi dari B2 ke B3. Matriks
konfusi `AF-E-009` menguatkannya: batas B2 berbanding B3 menyumbang 195 galat,
sedangkan batas B1 berbanding B2 hanya 57. Tabel 6 menunjukkan hal yang sama dari
sisi lain, dengan $F1_{B2}$ pada korpus 953 hanya $0,47$.

Baris `Total` menunjukkan mengapa bias kohort tidak dapat dibaca sendirian. Pada
uji 953, bias totalnya $−18$ tandan dari 1.342 acuan, padahal MAE per pohon
$1,363$. Audit
[`count_error_cancellation.json`](../results/audit_2026-09-06/count_error_cancellation.json)
mengukurnya: dari 37 pohon dengan cacah total tepat, 24 tepat karena kesalahan
positif dan negatif saling meniadakan.

Angka diturunkan dari matriks konfusi `confusion_prediction_rows` berukuran
$5 \times 5$; barisnya kelas prediksi (baris kelima untuk tandan acuan tanpa
prediksi), kolomnya kelas acuan (kolom kelima untuk prediksi tanpa pasangan).
Jumlah baris pertama sampai keempat sudah diverifikasi sama dengan
`pred_clusters`, jumlah kolomnya sama dengan `gt_bunches`, kecuali selisih satu
tandan pada 763-depth yang belum ditelusuri: 558 berbanding 559 pada uji, 570
berbanding 571 pada validasi. Profil `V2-E-045` tidak menyimpan rincian per
kelas, sehingga tabel ini memakai `Wave-V2`.

---

## 9. Plafon dan target rekayasa

### 9.1 Vonis atas target

| Target | Capaian kini | Plafon terukur | Vonis |
|---|---:|---:|---|
| $mAP50$ deteksi empat kelas $\ge 0,85$ | 0,5970 | 0,6569 | **Tidak terjangkau** |
| $mAP50$ dua kelas siap panen berbanding belum $\ge 0,85$ | 0,7754 | 0,8766 | Terjangkau |
| Pencacahan total per pohon, tepat persis | 0,2741 | 0,2900 | **Tidak terjangkau** |
| Pencacahan siap panen per pohon, $\pm 1 \ge 95\%$ | 0,957–0,965 | 1,000 | **Tercapai** |

### 9.2 Akurasi kematangan yang dibutuhkan untuk mencapai target (`AF-E-005`)

| Akurasi kematangan per citra terpotong | $mAP50$ empat kelas |
|---:|---:|
| 0,661 (capaian kini) | 0,587 |
| 0,80 | 0,735 |
| 0,90 | **0,847** |
| 0,95 | 0,927 |

Target $mAP50 = 0,85$ menuntut akurasi kematangan sekitar $0,90$, yaitu dua puluh
poin di atas segala yang pernah dicapai repositori ini.

### 9.3 Plafon pencacahan per kelas dengan deteksi *oracle* (`AF-E-004`, uji 953, 136 pohon)

| Kelas | MAE | Tepat persis | $\pm 1$ | Tampak tunggal |
|---|---:|---:|---:|---:|
| B1 | **0,101** | **0,899** | **1,000** | 11,6% |
| B2 | 0,239 | 0,768 | 0,993 | 23,4% |
| B3 | 0,638 | 0,428 | 0,942 | 22,7% |
| B4 | 0,268 | 0,739 | 0,993 | **40,5%** |
| Total per pohon | 1,058 | 0,290 | 0,754 | — |

Makro-MAE $0,312$, berdekatan dengan jalur *oracle* historis $0,275$–$0,277$ pada
[`docs/REKAP.md`](../docs/REKAP.md) §2. Faktor duplikasi per pohon $k = 1,905$
dengan simpangan baku $0,384$.

---

## 10. Kaveat

| Kaveat | Dampak terhadap pembacaan | Rujukan |
|---|---|---|
| Makro-F1 `AF-E-012` punya dua nilai yang sama-sama benar: $0,6692$ (klaster berpasangan saja) dan $0,5201$ (ujung ke ujung) | Hanya $0,5201$ yang setara dengan makro-F1 $0,6034$ profil terkunci | [`EVIDENCE.md`](../docs/research_2026-09-06/EVIDENCE.md) |
| Pipeline Panen memakai 132 pohon, YOLO26m tunggal, dan penaut milik audit | Tidak setara dengan baris terkunci pada tabel yang sama; unggul pada cacah B1 siap panen $\pm 1$ ($0,970$), akurasi ordinal $\pm 1$ ($0,9946$), dan akurasi dua kelas ($0,8678$) | `AF-E-012`, `AF-E-013` |
| `AF-E-010` melaporkan $45,3\%$ klaster melanggar kendala satu tandan satu sisi | Dikoreksi `AF-E-014`: hanya berlaku pada jalur sapuan; profil terkunci `max_size` $\le 3$ pelanggarannya $0,00\%$, 0 dari 630 konfigurasi berubah. `AF-E-016`: *Anchor A* (`max_size` 4) juga aman, 0 dari 135 pohon berbeda | `AF-E-014`, `AF-E-016` |
| Seluruh `AF-E` memakai detektor lebih kecil, penaut sendiri, atau arah taksonomi berlawanan | Tidak ada yang menggantikan angka terkunci; statusnya eksperimen tambahan | `metrics/07` |
| Irisan `tree_id` antara partisi latih `combined1716` dan kedua kumpulan uji lokal belum diaudit | Kemungkinan kebocoran partisi data (*data leakage*) belum terkuantifikasi | [`ANALISIS_PIPELINE_MENDALAM.md`](../docs/ANALISIS_PIPELINE_MENDALAM.md) |
| Partisi uji pernah dibaca pada iterasi historis | Angka uji adalah konfirmasi profil terkunci validasi, bukan partisi terisolasi yang belum tersentuh | `V2-E-045` |
| Plafon $0,6569$ bergantung pada mutu satu pengklasifikasi ($0,6635$ per citra terpotong) | Terkalibrasi terhadap pita $0,62$–$0,70$ repositori ini, bukan batas teoretis-informasi | `AF-E-005` |

---

## 11. Sumber angka

| Tabel | Artefak |
|---|---|
| 3, deteksi empat kelas | [`combined1716`](../results/combined1716), [`new763`](../results/new763), medan `splits.test.mAP50`; [`perkelas_pycoco_v2repro.json`](../results/perkelas_pycoco_v2repro.json); [`extra_metrics_sesi2026-08.json`](../results/extra_metrics_sesi2026-08.json); [`map_boost_artifacts`](../results/remote_eval_2026-08-28/map_boost_artifacts), medan `test_metrics.classaware`; [`detector_matrix.json`](../results/audit_forensik_2026-09-06/detector_matrix.json) |
| 4, deteksi agnostik | [`class_agnostic_metrics_audit_2026-09-03.json`](../results/class_agnostic_metrics_audit_2026-09-03.json), medan `rows[]`; [`map_boost_artifacts`](../results/remote_eval_2026-08-28/map_boost_artifacts), medan `test_metrics.agnostic` |
| 5, 6, 7, profil `V2-E-045` | [`pipeline_combined1716_generalization_locked.json`](../results/remote_eval_2026-08-27/metrics/pipeline_combined1716_generalization_locked.json) |
| 5, 6, 7, profil `Wave-V2` dan selang kepercayaan | [`gsp_artifacts/953`](../results/remote_eval_2026-08-28/gsp_artifacts/953/results_test_locked.json), [`gsp_artifacts/depth`](../results/remote_eval_2026-08-28/gsp_artifacts/depth/results_test_locked.json), [`anchor_check.json`](../results/remote_eval_2026-08-28/validation_wave/reports/anchor_check.json) |
| 5, 6, 7, Pipeline Panen | [`panen/panen_final.json`](../results/audit_forensik_2026-09-06/panen/panen_final.json) |
| 8, bias per kelas | Diturunkan dari medan `metrics.classification.confusion_prediction_rows` pada artefak `Wave-V2` |
| 9, plafon dan target | [`experiments/AUDIT-FORENSIK-2026-09-06.md`](../experiments/AUDIT-FORENSIK-2026-09-06.md), [`docs/AUDIT-FORENSIK-2026-09-06.md`](../docs/AUDIT-FORENSIK-2026-09-06.md) |

Rujukan pelengkap: [`README.md`](README.md) sebagai gerbang atlas dan tujuh
berkas spesialisasi `01`–`07` untuk penelusuran per eksperimen.
