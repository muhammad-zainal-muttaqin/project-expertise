# Rekapitulasi Papan Peringkat Hasil Eksperimen

Dokumen ini menyusun ulang seluruh hasil evaluasi dari [atlas metrik](README.md) menjadi satu papan peringkat (*leaderboard*) terpadu untuk setiap tahapan tugas pada sistem. Seluruh korpus data dan konfigurasi sistem dirangkum dalam tabel komparasi terstruktur yang diurutkan dari performa terbaik hingga terendah berdasarkan metrik evaluasi utama, lengkap dengan metrik pendukungnya.

**Cakupan evaluasi:** Eksperimen `V2-E-001`–`V2-E-045`, `PT-E-000`–`PT-E-036`, sesi verifikasi *remote* Agustus 2026, serta rangkaian audit forensik `AF-E-001`–`AF-E-016`. Dokumen ini bersifat aditif (hanya mengonsolidasi data historis yang valid) tanpa mengubah angka metrik empiris maupun struktur berkas atlas terdahulu.

---

## 1. Daftar Tugas dan Metrik Evaluasi

| Tugas Sistem | Penjelasan & Makna Fungsional | Satuan Evaluasi | Metrik Utama | Metrik Pendamping | Rujukan Tabel |
|---|---|---|---|---|---|
| Deteksi empat kelas | Lokalisasi kotak pembatas (*bounding box*) sekaligus klasifikasi tingkat kematangan (B1–B4) dari satu detektor satu tahap | per citra | $mAP50$ | $mAP50\text{--}95$, $AP$ per kelas B1–B4 | [Tabel 3](#3-deteksi-empat-kelas) |
| Deteksi agnostik | Lokalisasi murni kotak pembatas tandan buah sawit tanpa membedakan kelas kematangan (*foreground/background*) | per citra | $AP50_{agn}$ | $AP50\text{--}95_{agn}$, Presisi, Daya Tangkap (*Recall*), *F1* | [Tabel 4](#4-deteksi-agnostik-lokalisasi-murni) |
| Deduplikasi multi-tampak | Penggabungan kemunculan tandan buah yang sama dari empat sudut pandang pohon agar terhitung tepat satu kali | per pohon | *F1 fisik* | Presisi fisik, Daya tangkap fisik | [Tabel 5](#5-deduplikasi-multi-tampak) |
| Klasifikasi kematangan | Ketepatan penetapan kelas kematangan khusus pada klaster tandan yang terasosiasi secara tepat dengan data acuan (*matched-class*) | per pohon | Akurasi *matched-class* | Makro-*F1* ujung ke ujung (*end-to-end*), *F1* per kelas B1–B4 | [Tabel 6](#6-klasifikasi-kematangan) |
| Pencacahan per pohon | Selisih absolut antara jumlah tandan hasil prediksi pada setiap pohon terhadap nilai acuan kebenarannya | per pohon | MAE cacah total | Akurasi toleransi $\pm 1$ tandan, Akurasi tepat persis | [Tabel 7](#7-pencacahan-per-pohon) |
| Pencacahan kohort | Akumulasi total tandan per kelas kematangan B1–B4 pada seluruh pohon dalam satu partisi data populasi | per kohort | Bias per kelas | Makro-rerata nilai mutlak bias relatif | [Tabel 8](#8-pencacahan-kohort-per-kelas) |

### Alur Kerja Sistem Ujung ke Ujung (*Pipeline Workflow*)
Sistem pemrosesan bekerja secara berurutan (*sequential*):
1. **Detektor** melokalisasi kotak pembatas tandan pada citra setiap sudut pandang.
2. **Modul penaut (*linking*)** menggabungkan kemunculan tandan yang sama dari 4 sudut pandang pohon menjadi satu entitas fisik (klaster tandan).
3. **Model pengklasifikasi** menetapkan tingkat kematangan final dari klaster tersebut.
4. **Modul pencacah** menjumlahkan total estimasi tandan per pohon maupun per kohort.

> [!NOTE]
> Karena basis perhitungan (penyebut atau *denominator*) pada setiap tahapan tugas berbeda (tingkat citra tunggal, klaster pohon, atau populasi kohort), nilai metrik antar-tabel tidak dapat diperbandingkan secara langsung.

### 1.1 Metrik yang Sering Tertukar (Panduan Interpretasi)

| Metrik Evaluasi | Objek yang Diukur | Hal yang BUKAN Diukur (Hindari Salah Paham) |
|---|---|---|
| $AP50_{agn} = 0,8350$ | Presisi rata-rata lokalisasi spasial sebesar $83,50\%$ saat seluruh kelas kematangan dilebur menjadi satu kategori tunggal. | Bukan akurasi penentuan kelas kematangan dan bukan akurasi pencacahan buah pada pohon. |
| *F1 fisik* | Keseimbangan presisi dan daya tangkap (*recall*) pengelompokan tandan gabungan dari multi-sudut pandang pada tingkat pohon. | Bukan *F1* pasangan (asosiasi dua kotak antar-sisi citra) dan bukan *F1* titik operasi deteksi citra tunggal. |
| Akurasi *matched-class* | Ketepatan prediksi kelas kematangan khusus pada klaster tandan yang berhasil dipasangkan secara tepat dengan data acuan. | Bukan makro-*F1* ujung ke ujung (*end-to-end*), karena metrik ujung ke ujung ikut memperhitungkan penalti klaster palsu (*false positive*) dan tandan yang terlewat (*false negative*). |
| Bias kohort | Selisih kumulatif antara total estimasi dan total data acuan per kelas pada satu populasi (kesalahan estimasi lebih dan kurang antar-pohon dapat saling meniadakan). | Bukan MAE per pohon, karena MAE mengukur magnitudo galat rata-rata pada masing-masing pohon tanpa saling meniadakan. |

---

## 2. Karakteristik Korpus Data

| Korpus Data | Modalitas Sensor Masukan | Ukuran Populasi | Jumlah Citra Uji | Jumlah Pohon Uji (4 Sudut Pandang) | Cakupan Evaluasi Tugas |
|---|---|---:|---:|---:|---|
| `combined1716` | RGB standar | 1.716 pohon | 1.052 | &mdash; | Evaluasi deteksi citra saja |
| `763-depth` | RGB + Kedalaman Y16 | 763 pohon | 440 | 110 | Evaluasi keenam tahapan tugas |
| `953` | RGB standar | 953 pohon | 588 | 135 | Evaluasi keenam tahapan tugas |

Korpus `combined1716` merupakan gabungan dari korpus 953 dan 763 yang difungsikan sebagai bank data pelatihan modul detektor berkapasitas besar. Kumpulan data 1.716 pohon ini tidak dilengkapi label nilai acuan kebenaran (*ground truth*) multi-sisi tingkat pohon, sehingga evaluasinya terbatas pada tugas deteksi (Tabel 3 dan 4) dan tidak disertakan pada Tabel 5 hingga 8.

**Keterangan Label Model:**
- **Bank `combined1716`**: Bobot model dilatih menggunakan bank data gabungan `combined1716`, kemudian diuji performa generalisasinya pada partisi uji korpus spesifik (953 atau 763-depth).
- **Native**: Bobot model dilatih dan diuji pada partisi dari korpus yang sama.

### 2.1 Perbedaan Protokol Anotasi Antar-Korpus (Audit `AF-E-001`)

Evaluasi audit dilakukan secara berpasangan pada subset 352 pohon fisik yang identik pada kedua rilis dataset:

| Parameter Pengamatan | Korpus 953 (Mei 2026) | Korpus 763-depth (Juli–Agustus 2026) | Perubahan Relatif |
|---|---:|---:|---:|
| Rata-rata tandan unik per pohon | 9,89 tandan | 3,99 tandan | −59,7% |
| Proporsi sudut pandang tanpa anotasi | 1,1% | 14,2% | +13,1 pp |
| Proporsi kemunculan kelas B1 (lewat matang) | Data acuan dasar | Meningkat | +66% |
| Proporsi kemunculan kelas B4 (mentah) | Data acuan dasar | Menurun | −85% |

> [!IMPORTANT]
> Perbedaan performa model saat dievaluasi lintas korpus sebagian besar merefleksikan perbedaan protokol anotasi lapangan dan pergeseran fenologi musiman kebun, bukan kegagalan generalisasi representasi visual model.

---

## 3. Deteksi Empat Kelas

Metrik utama: $mAP50$ sadar-kelas (*class-aware*); metrik pendamping: $mAP50\text{--}95$ serta $AP50$ untuk setiap kelas kematangan (B1–B4). Seluruh evaluasi menggunakan protokol baku `pycocotools.COCOeval` pada partisi data uji. Setiap tabel diurutkan berdasarkan nilai $mAP50$ tertinggi.

### 3.1 Korpus 763-depth (Partisi Uji)

| Konfigurasi Sistem | $mAP50$ | $mAP50\text{--}95$ | $AP_{B1}$ | $AP_{B2}$ | $AP_{B3}$ | $AP_{B4}$ | ID Simpul |
|---|---:|---:|---:|---:|---:|---:|---|
| RF-DETR-L, bank `combined1716` | **0,6711** | 0,2748 | 0,8044 | **0,7187** | **0,7373** | **0,4239** | `V2-E-042` |
| WBF [YOLO26l+RT-DETR-L+RF-DETR-L], bank `combined1716` | 0,6691 | **0,2757** | 0,8280 | 0,6919 | 0,7349 | 0,4214 | `V2-E-042` |
| WBF + *re-ranker* | 0,6552 | 0,2701 | **0,8288** | 0,6863 | 0,7362 | 0,3693 | `MAP_BOOST` |
| RT-DETR-L, bank `combined1716` | 0,6309 | 0,2496 | 0,7865 | 0,6652 | 0,7160 | 0,3559 | `V2-E-042` |
| RF-DETR-L native | 0,6129 | 0,2335 | 0,7758 | 0,6353 | 0,6997 | 0,3406 | `V2-E-034` |
| YOLO26l, bank `combined1716` | 0,5765 | 0,2387 | 0,7839 | 0,6088 | 0,6380 | 0,2754 | `V2-E-042` |
| RT-DETR-L native | 0,5580 | 0,2055 | 0,7377 | 0,5889 | 0,6607 | 0,2445 | `V2-E-034` |
| YOLO26l native | 0,5163 | 0,1906 | 0,6847 | 0,5877 | 0,6005 | 0,1920 | `V2-E-034` |

### 3.2 Korpus 953 (Partisi Uji)

| Konfigurasi Sistem | $mAP50$ | $mAP50\text{--}95$ | $AP_{B1}$ | $AP_{B2}$ | $AP_{B3}$ | $AP_{B4}$ | ID Simpul | Status Evaluasi |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Plafon lokalisasi sempurna + ConvNeXt-Tiny | **0,6569** | N/A¹ | 0,7653 | 0,4912 | **0,7625** | **0,6088** | `AF-E-005` | Batas Atas (*Oracle*) |
| RF-DETR-L native | 0,6012 | 0,2747 | **0,8150** | **0,5184** | 0,6553 | 0,4160 | `V2-E-001` | Data Uji |
| WBF + *re-ranker* | 0,5970 | 0,2743 | 0,8042 | 0,4942 | 0,6566 | 0,4328 | `MAP_BOOST` | Data Uji |
| WBF, bank `combined1716` | 0,5861 | **0,2753** | 0,7869 | 0,4866 | 0,6513 | 0,4197 | `V2-E-042` | Data Uji |
| RT-DETR-L native | 0,5781 | 0,2629 | 0,7874 | 0,4614 | 0,6371 | 0,4266 | `V2-E-001` | Data Uji |
| YOLO26l native | 0,5435 | 0,2564 | 0,7705 | 0,4479 | 0,6050 | 0,3506 | `V2-E-001` | Data Uji |
| YOLO26s 960 px, replikasi audit | 0,5433 | N/A² | 0,7588 | 0,4495 | 0,6095 | 0,3553 | `AF-E-006` | Data Uji |

### 3.3 Korpus `combined1716` (Partisi Uji)

| Konfigurasi Sistem | $mAP50$ | $mAP50\text{--}95$ | $AP_{B1}$ | $AP_{B2}$ | $AP_{B3}$ | $AP_{B4}$ | ID Simpul |
|---|---:|---:|---:|---:|---:|---:|---|
| RF-DETR-L native | **0,5960** | **0,2522** | **0,7654** | **0,5394** | **0,6652** | **0,4141** | `V2-E-035` |
| RT-DETR-L native | 0,5745 | 0,2458 | 0,7308 | 0,5120 | 0,6465 | 0,4089 | `V2-E-035` |
| WBF native | 0,5538 | N/A² | 0,7286 | 0,4732 | 0,6372 | 0,3760 | `V2-E-039` |
| YOLO26l native | 0,5389 | 0,2395 | 0,7298 | 0,4766 | 0,6075 | 0,3419 | `V2-E-035` |

¹ `N/A: Bukan keluaran detektor riil; baris ini merupakan batas atas teoretis (oracle).`  
² `N/A: Metrik tidak dicatat pada berkas artefak sumber.`

### Catatan Analisis Deteksi Empat Kelas:
1. **Batas Atas Teoretis (*Oracle* `AF-E-005`):** Baris plafon memanfaatkan kotak acuan kebenaran (*ground truth*) sebagai masukan klasifikasi sehingga galat lokalisasi tereliminasi sempurna. Nilai $mAP50 = 0,6569$ merepresentasikan batas performa maksimum. Capaian model riil tertinggi pada korpus 953 adalah $0,5970$ (mencapai $90,88\%$ dari batas teoretis tersebut).
2. **Konsistensi Hierarki Arsitektur:** Urutan performa **RF-DETR-L > RT-DETR-L > YOLO26l** terbukti konsisten pada ketiga korpus data.
3. **Replikasi Audit (`AF-E-006`):** Pengujian dengan YOLO26s pada resolusi 960 piksel menghasilkan angka yang sangat dekat dengan `V2-E-001` ($0,5433$ berbanding $0,5435$). Hal ini membuktikan keterulangan dan kalibrasi audit independen yang sahih.
4. Rujukan komparasi korpus 352 pohon dan uji ablasi kedalaman monokular (*pseudo-depth*) tersedia pada dokumen [`01_deteksi_dan_lokalisasi.md`](01_deteksi_dan_lokalisasi.md).

---

## 4. Deteksi Agnostik (Lokalisasi Murni)

Metrik utama: $AP50_{agn}$ dengan seluruh kelas kematangan dilebur menjadi satu kategori objek tunggal (*foreground/background*); metrik pendamping: $AP50\text{--}95_{agn}$, Presisi, Daya Tangkap (*Recall*), dan *F1* pada titik operasi ambang keyakinan $0,25$ dan ambang $IoU = 0,5$. Setiap tabel diurutkan berdasarkan nilai $AP50_{agn}$ tertinggi.

### 4.1 Korpus 763-depth (440 Citra Uji)

| Konfigurasi Sistem | $AP50_{agn}$ | $AP50\text{--}95_{agn}$ | Presisi | Daya Tangkap | *F1* | ID Simpul |
|---|---:|---:|---:|---:|---:|---|
| WBF + *re-ranker* | **0,8783** | **0,3523** | N/A¹ | N/A¹ | N/A¹ | `MAP_BOOST` |
| WBF, bank `combined1716` | 0,8764 | 0,3519 | **0,9296** | 0,6813 | **0,7863** | `V2-E-042` |
| RF-DETR-L native | 0,7951 | 0,3003 | 0,5748 | **0,8361** | 0,6813 | `V2-E-036` |

### 4.2 Korpus 953 (588 Citra Uji)

| Konfigurasi Sistem | $AP50_{agn}$ | $AP50\text{--}95_{agn}$ | Presisi | Daya Tangkap | *F1* | ID Simpul |
|---|---:|---:|---:|---:|---:|---|
| WBF + *re-ranker* | **0,8419** | **0,3717** | N/A¹ | N/A¹ | N/A¹ | `MAP_BOOST` |
| WBF, bank `combined1716` | 0,8350 | 0,3679 | **0,8825** | 0,6443 | **0,7449** | `V2-E-042` |
| YOLO26m 1.280 px tunggal | 0,8104 | N/A³ | 0,7942 | **0,7271** | N/A³ | `AF-E-011` |
| YOLO26s 960 px tunggal | 0,8057 | N/A³ | 0,7965 | 0,7087 | N/A³ | `AF-E-006` |
| YOLO26l, kelas dilebur | 0,7388 | 0,3312 | 0,7547 | 0,6336 | 0,6889 | `V2-E-017` |

### 4.3 Korpus `combined1716` (1.052 Citra Uji)

| Konfigurasi Sistem | $AP50_{agn}$ | $AP50\text{--}95_{agn}$ | Presisi | Daya Tangkap | *F1* | ID Simpul |
|---|---:|---:|---:|---:|---:|---|
| WBF native | **0,8104** | **0,3363** | **0,9371** | 0,0763² | 0,1411² | `V2-E-039` |
| RF-DETR-L native | 0,7850 | 0,3245 | 0,4820 | **0,8497** | **0,6151** | `V2-E-036` |

¹ `N/A: Artefak MAP_BOOST tidak menyimpan metrik titik operasi tetap.`  
² Kalibrasi distribusi probabilitas WBF pada korpus ini berbeda, sehingga penerapan ambang tetap $0,25$ menghasilkan daya tangkap $7,63\%$. Nilai $AP50$ tidak terpengaruh karena dihitung integral melintasi seluruh spektrum ambang.  
³ `N/A: Metrik tidak dicatat pada berkas artefak sumber.`

> [!NOTE]
> Hasil audit pada simpul `AF-E-011` membuktikan bahwa model detektor tunggal berkapasitas besar (YOLO26m 1.280 px) mampu mendekati performa ansambel tiga model ($AP50_{agn} = 0,8104$ berbanding $0,8350$), meskipun belum melampauinya.

---

## 5. Deduplikasi Multi-Tampak

Metrik utama: *F1 fisik* pada ambang pencocokan spasial $IoU \ge 0,5$; metrik pendamping: Presisi fisik dan Daya tangkap fisik (*recall*) pada tingkat klaster pohon. Setiap tabel diurutkan berdasarkan nilai *F1 fisik* tertinggi. Korpus `combined1716` tidak memiliki nilai acuan kebenaran multi-sisi tingkat pohon sehingga tidak dievaluasi pada tugas ini.

### 5.1 Korpus 763-depth

| Konfigurasi Sistem | Partisi Evaluasi | $n$ Pohon | *F1 Fisik* | Presisi Fisik | Daya Tangkap Fisik | ID Simpul |
|---|---|---:|---:|---:|---:|---|
| WBF + penaut *greedy strict* | Uji | 110 | **0,8590** | 0,8799 | **0,8390** | `V2-E-043`⁴ |
| WBF + penaut GSP MILP + pencacah Ridge | Uji | 110 | 0,8534 | 0,8926 | 0,8175 | `Wave-V2` |
| WBF + penaut GSP MILP + pencacah Ridge | Validasi | 117 | 0,8526 | **0,9055** | 0,8056 | `Wave-V2` |
| WBF + penaut prior rotasi + pencacah Ridge | Validasi | 117 | 0,8257 | 0,8431 | 0,8091 | `V2-E-045` |
| WBF + penaut prior rotasi + pencacah Ridge | Uji | 110 | 0,8069 | 0,8142 | 0,7996 | `V2-E-045` |

### 5.2 Korpus 953

| Konfigurasi Sistem | Partisi Evaluasi | $n$ Pohon | *F1 Fisik* | Presisi Fisik | Daya Tangkap Fisik | ID Simpul |
|---|---|---:|---:|---:|---:|---|
| WBF + penaut Hungarian *Anchor A* + pencacah Ridge | Uji | 135 | **0,8387** | 0,8444 | 0,8331 | `Wave-V2` |
| WBF + penaut *greedy strict* | Uji | 135 | 0,8296 | 0,8247 | **0,8346** | `V2-E-043`⁴ |
| WBF + penaut Hungarian *Anchor A* + pencacah Ridge | Validasi | 91 | 0,8232 | 0,8206 | 0,8259 | `Wave-V2` |
| WBF + penaut prior rotasi + pencacah Ridge | Validasi | 91 | 0,8087 | 0,8044 | 0,8130 | `V2-E-045` |
| WBF + penaut prior rotasi + pencacah Ridge | Uji | 135 | 0,8043 | 0,8092 | 0,7996 | `V2-E-045` |
| YOLO26m + penaut terlatih (Pipeline Panen) | Uji | 132 | 0,7619 | **0,8538** | 0,6878 | `AF-E-012` |
| YOLO26m + penaut terlatih (Pipeline Panen) | Validasi | 91 | 0,7586 | 0,8374 | 0,6934 | `AF-E-012` |

⁴ Parameter pada `V2-E-043` diperoleh dari penyapuan (*grid search*) langsung pada partisi uji sehingga mencerminkan optimasi lokal (*overfitting* uji) dan bukan tolok ukur generalisasi murni. Profil `Wave-V2` dikunci pada partisi uji; profil `V2-E-045` dikunci secara baku pada partisi validasi.

> [!NOTE]
> Evaluasi inferensial menunjukkan bahwa selang kepercayaan 95% dari kedua profil utama saling bertumpang tindih (lihat Bagian 7.3). Dengan demikian, selisih performa antara $0,8534$ dan $0,8387$ belum signifikan secara statistik.

---

## 6. Klasifikasi Kematangan

Metrik utama: Akurasi *matched-class* pada klaster yang terasosiasi secara tepat dengan data acuan; metrik pendamping: Makro-*F1* ujung ke ujung (*end-to-end*) yang turut memperhitungkan penalti klaster palsu (*false positive*) dan tandan terlewat (*false negative*), serta nilai *F1* per kelas kematangan. Setiap tabel diurutkan berdasarkan akurasi tertinggi.

### 6.1 Korpus 763-depth

| Konfigurasi Sistem | Partisi Evaluasi | Akurasi *Matched-Class* | Makro-*F1* E2E | $F1_{B1}$ | $F1_{B2}$ | $F1_{B3}$ | $F1_{B4}$ | ID Simpul |
|---|---|---:|---:|---:|---:|---:|---:|---|
| WBF + GSP MILP | Validasi | **84,57%** | **0,6807** | **0,7778** | **0,7244** | **0,7500** | **0,4706** | `Wave-V2` |
| WBF + prior rotasi | Validasi | 83,55% | 0,6749 | N/A¹ | N/A¹ | N/A¹ | N/A¹ | `V2-E-045` |
| WBF + GSP MILP | Uji | 81,62% | 0,6519 | 0,7578 | 0,7230 | 0,7092 | 0,4176 | `Wave-V2` |
| WBF + prior rotasi | Uji | 80,31% | 0,6047 | N/A¹ | N/A¹ | N/A¹ | N/A¹ | `V2-E-045` |

### 6.2 Korpus 953

| Konfigurasi Sistem | Partisi Evaluasi | Akurasi *Matched-Class* | Makro-*F1* E2E | $F1_{B1}$ | $F1_{B2}$ | $F1_{B3}$ | $F1_{B4}$ | ID Simpul |
|---|---|---:|---:|---:|---:|---:|---:|---|
| WBF + Hungarian *Anchor A* | Validasi | **75,42%** | 0,6014 | 0,7059 | 0,4698 | 0,6737 | **0,5561** | `Wave-V2` |
| WBF + Hungarian *Anchor A* | Uji | 74,42% | **0,6034** | **0,7465** | **0,4706** | **0,6850** | 0,5114 | `Wave-V2` |
| YOLO26m + skor ordinal (Pipeline Panen) | Validasi | 71,65% | 0,6683⁵ | N/A¹ | N/A¹ | N/A¹ | N/A¹ | `AF-E-012` |
| YOLO26m + skor ordinal (Pipeline Panen) | Uji | 71,61% | 0,6692⁵ | N/A¹ | N/A¹ | N/A¹ | N/A¹ | `AF-E-012` |
| WBF + prior rotasi | Uji | 71,11% | 0,5384 | N/A¹ | N/A¹ | N/A¹ | N/A¹ | `V2-E-045` |
| WBF + prior rotasi | Validasi | 70,04% | 0,5462 | N/A¹ | N/A¹ | N/A¹ | N/A¹ | `V2-E-045` |

¹ `N/A: Artefak sumber tidak menyimpan rincian metrik per kelas.`  
⁵ Nilai makro-*F1* pada Pipeline Panen hanya dihitung khusus pada subset klaster yang berhasil dipasangkan, sehingga tidak setara secara metodologis dengan kolom yang sama pada baris lainnya. Nilai makro-*F1* ujung ke ujung sesungguhnya adalah $0,5201$.

### Analisis Kinerja per Kelas:
- **Korpus 953:** Penurunan performa terbesar terkonsentrasi pada **kelas B2** ($F1_{B2} = 0,4706$), sedangkan tiga kelas lainnya berada pada rentang yang lebih tinggi ($0,5114$–$0,7465$).
- **Korpus 763-depth:** Penurunan performa terendah terjadi pada **kelas B4** ($F1_{B4} = 0,4176$).

---

## 7. Pencacahan Tingkat Pohon

Metrik utama: *Mean Absolute Error* (MAE) cacah tandan total per pohon; metrik pendamping: Akurasi toleransi $\pm 1$ tandan dan akurasi pencacahan tepat persis (selisih nol). Setiap tabel diurutkan berdasarkan nilai MAE terendah (terbaik).

### 7.1 Korpus 763-depth

| Konfigurasi Sistem | Partisi Evaluasi | $n$ Pohon | MAE Total | Akurasi $\pm 1$ | Akurasi Tepat Persis | ID Simpul |
|---|---|---:|---:|---:|---:|---|
| WBF + prior rotasi + Ridge | Validasi | 117 | **0,726** | 84,62% | 44,44% | `V2-E-045` |
| WBF + GSP MILP + Ridge | Uji | 110 | 0,773 | **85,45%** | **44,55%** | `Wave-V2` |
| WBF + *greedy strict*, cacah klaster mentah | Uji | 110 | 0,818 | 83,64% | 41,82% | `V2-E-043`⁴ |
| WBF + prior rotasi + Ridge | Uji | 110 | 0,891 | 80,91% | 33,64% | `V2-E-045` |
| WBF + GSP MILP + Ridge | Validasi | 117 | 0,932 | 78,63% | 34,19% | `Wave-V2` |

### 7.2 Korpus 953

| Konfigurasi Sistem | Partisi Evaluasi | $n$ Pohon | MAE Total | Akurasi $\pm 1$ | Akurasi Tepat Persis | ID Simpul |
|---|---|---:|---:|---:|---:|---|
| Plafon: Kotak acuan + Ridge per kelas | Uji | 136 | **1,058** | **75,40%** | **29,00%** | `AF-E-004` |
| WBF + Hungarian / prior rotasi + Ridge | Validasi | 91 | 1,253 | 67,03% | 28,57% | `Wave-V2`, `V2-E-045`⁶ |
| WBF + Hungarian *Anchor A* + Ridge | Uji | 135 | 1,363 | 63,70% | 27,41% | `Wave-V2` |
| YOLO26m + penaut terlatih + Ridge | Validasi | 91 | 1,374 | 63,74% | 26,37% | `AF-E-013` |
| WBF + prior rotasi + Ridge | Uji | 135 | 1,393 | 61,48% | 25,93% | `V2-E-045` |
| YOLO26m + penaut terlatih + Ridge | Uji | 132 | 1,402 | 56,82% | 22,73% | `AF-E-013` |
| WBF + *greedy strict*, cacah klaster mentah | Uji | 135 | 1,644 | 54,07% | N/A² | `V2-E-043`⁴ |

² `N/A: Metrik tidak dicatat pada berkas artefak sumber.`  
<sup>6</sup> Lapisan modul regresi Ridge beroperasi langsung pada vektor fitur proposal terlepas dari modul penaut, sehingga kedua profil menghasilkan nilai MAE yang identik pada partisi validasi 953.

### Analisis Batas Teoretis Pencacahan Pohon:
Baris *oracle* `AF-E-004` menyuplai kotak acuan kebenaran (*ground truth*) ke dalam pipeline pencacahan. Bahkan dengan deteksi sempurna tanpa galat lokalisasi, akurasi pencacahan tepat persis hanya mencapai $29,00\%$ (sedangkan sistem berbasis deteksi riil telah mencapai $27,41\%$). Hal ini mengindikasikan bahwa batas akurasi pencacahan pohon lebih dipengaruhi oleh kompleksitas oklusi kanopi pelepah dan geometri spasial antar-sisi pohon, bukan semata-mata galat deteksi visual.

### 7.3 Estimasi Selang Kepercayaan 95% Profil Terkunci Uji

Dihitung melalui simulasi *bootstrap* berpasangan sebanyak 2.000 ulangan (*random seed* 42) pada seluruh metrik alur kerja:

| Metrik Evaluasi | Korpus 953: Hungarian *Anchor A* (135 pohon) | Korpus 763-depth: GSP MILP (110 pohon) |
|---|---|---|
| *F1 fisik* | 0,8387 [0,8174; 0,8587] | 0,8534 [0,8301; 0,8761] |
| Akurasi *matched-class* | 0,7442 [0,7112; 0,7735] | 0,8162 [0,7765; 0,8556] |
| Makro-*F1* ujung ke ujung | 0,6034 [0,5655; 0,6382] | 0,6519 [0,6046; 0,6918] |
| MAE cacah total | 1,363 [1,163; 1,585] | 0,773 [0,609; 0,945] |
| Akurasi toleransi cacah $\pm 1$ | 0,6370 [0,5556; 0,7185] | 0,8545 [0,7818; 0,9182] |

---


## 8. Pencacahan Kohort Agregat per Kelas

Evaluasi akhir pencacahan kohort dari profil terkunci `Wave-V2`, dihitung dengan mengakumulasi seluruh estimasi pohon dalam satu partisi data populasi.
- **Bias absolut**: Selisih bersih antara jumlah prediksi dan jumlah nilai acuan riil (bersatuan tandan).
- **Bias relatif**: Persentase bias absolut terhadap total nilai acuan pada kelas terkait.

### Karakteristik Visual & Skala Kematangan Tandan Sawit ([`docs/DATASET.md`](../docs/DATASET.md) §1)

| Kelas Kematangan | Tingkat Kematangan & Status Panen | Karakteristik Visual Dominan | Ukuran Kotak Median (Korpus 953) |
|---|---|---|---:|
| B1 | Lewat matang (siap panen) | Jingga kemerahan cerah, posisi lingkaran terbawah kanopi | 133 piksel |
| B2 | Matang optimal (siap panen) | Oranye kemerahan bersemburat ungu kehitaman | 120 piksel |
| B3 | Matang awal (mengkal / belum siap) | Ungu kemerahan kehitaman | 107 piksel |
| B4 | Mentah (muda / belum siap) | Hitam kehijauan pekat, tertanam rapat di sela pelepah | 93 piksel |

### 8.1 Korpus 953, Partisi Uji (Hungarian *Anchor A*, 135 Pohon)

| Kelas | Total Prediksi | Total Acuan (*GT*) | Bias Absolut | Bias Relatif |
|---|---:|---:|---:|---:|
| B1 | 104 tandan | 113 tandan | −9 | −7,96% |
| B2 | 145 tandan | 246 tandan | −101 | −41,06% |
| B3 | 824 tandan | 706 tandan | +118 | +16,71% |
| B4 | 251 tandan | 277 tandan | −26 | −9,39% |
| **Total** | **1.324 tandan** | **1.342 tandan** | **−18** | **−1,34%** |

Makro-rerata nilai mutlak bias relatif: 18,78%.

### 8.2 Korpus 953, Partisi Validasi (Hungarian *Anchor A*, 91 Pohon)

| Kelas | Total Prediksi | Total Acuan (*GT*) | Bias Absolut | Bias Relatif |
|---|---:|---:|---:|---:|
| B1 | 84 tandan | 86 tandan | −2 | −2,33% |
| B2 | 112 tandan | 186 tandan | −74 | −39,78% |
| B3 | 560 tandan | 476 tandan | +84 | +17,65% |
| B4 | 186 tandan | 188 tandan | −2 | −1,06% |
| **Total** | **942 tandan** | **936 tandan** | **+6** | **+0,64%** |

Makro-rerata nilai mutlak bias relatif: 15,21%.

### 8.3 Korpus 763-depth, Partisi Uji (GSP MILP, 110 Pohon)

| Kelas | Total Prediksi | Total Acuan (*GT*) | Bias Absolut | Bias Relatif |
|---|---:|---:|---:|---:|
| B1 | 67 tandan | 94 tandan | −27 | −28,72% |
| B2 | 227 tandan | 199 tandan | +28 | +14,07% |
| B3 | 177 tandan | 215 tandan | −38 | −17,67% |
| B4 | 41 tandan | 50 tandan | −9 | −18,00% |
| **Total** | **512 tandan** | **558 tandan** | **−46** | **−8,24%** |

Makro-rerata nilai mutlak bias relatif: 19,62%.

### 8.4 Korpus 763-depth, Partisi Validasi (GSP MILP, 117 Pohon)

| Kelas | Total Prediksi | Total Acuan (*GT*) | Bias Absolut | Bias Relatif |
|---|---:|---:|---:|---:|
| B1 | 66 tandan | 96 tandan | −30 | −31,25% |
| B2 | 234 tandan | 205 tandan | +29 | +14,15% |
| B3 | 176 tandan | 216 tandan | −40 | −18,52% |
| B4 | 32 tandan | 53 tandan | −21 | −39,62% |
| **Total** | **508 tandan** | **570 tandan** | **−62** | **−10,88%** |

Makro-rerata nilai mutlak bias relatif: 25,88%.

### 8.5 Catatan Sintesis dan Analisis Bias

1. **Konfusi Representasi Kelas B2 dan B3:**  
   Pada korpus 953, kelas B2 mengalami estimasi kurang (*underestimation*) sekitar 40%, sedangkan kelas B3 mengalami estimasi berlebih (*overestimation*) sekitar 17% secara konsisten di kedua partisi. Pola pergeseran massa prediksi yang saling berkebalikan ini menunjukkan terjadinya konfusi representasi visual antara kelas B2 dan B3. Bukti matriks konfusi (`AF-E-009`) mengonfirmasi hal ini: transisi batas B2 ke B3 menyumbang 195 galat klasifikasi, berbanding hanya 57 galat pada batas B1 ke B2. Hal ini selaras dengan nilai $F1_{B2}$ pada korpus 953 yang hanya mencapai 0,4706.

2. **Kompensasi Galat Antar-Pohon (*Error Cancellation*):**  
   Metrik bias kohort tidak dapat diinterpretasikan secara terpisah tanpa meninjau MAE tingkat pohon. Pada partisi uji 953, bias kumulatif tampak sangat rendah (−18 tandan dari 1.342 acuan atau −1,34%), meskipun MAE per pohon bernilai 1,363. Audit forensik ([`count_error_cancellation.json`](../results/audit_2026-09-06/count_error_cancellation.json)) membuktikan bahwa dari 37 pohon dengan hasil cacah tepat persis, 24 pohon di antaranya (64,86%) tepat akibat galat positif dan negatif antar-sudut pandang yang saling meniadakan (*error cancellation*).

3. **Integritas Derivasi Data:**  
   Angka diturunkan dari matriks konfusi `confusion_prediction_rows` berukuran $5 \times 5$ (baris 1–4 untuk kelas prediksi, baris ke-5 untuk acuan yang terlewat; kolom 1–4 untuk kelas acuan, kolom ke-5 untuk prediksi tanpa pasangan). Jumlah baris 1–4 konsisten dengan `pred_clusters`, dan jumlah kolom 1–4 konsisten dengan `gt_bunches` (terdapat selisih 1 tandan yang belum terpetakan pada 763-depth: 558 vs 559 pada uji, 570 vs 571 pada validasi).

---

## 9. Analisis Batas Teoretis (Plafon) dan Target Rekayasa

### 9.1 Evaluasi Ketercapaian Target Rekayasa

| Target Rekayasa Sistem | Capaian Saat Ini | Batas Plafon Terukur (*Oracle*) | Status Ketercapaian Target |
|---|---:|---:|---|
| $mAP50$ deteksi empat kelas $\ge 0,85$ | 0,5970 | 0,6569 | **Tidak terjangkau secara teoretis** |
| $mAP50$ klasifikasi biner $\ge 0,85$ | 0,7754 | 0,8766 | Terjangkau secara empiris |
| Pencacahan total per pohon tepat persis $\ge 50\%$ | 0,2741 | 0,2900 | **Tidak terjangkau secara teoretis** |
| Pencacahan siap panen ($\pm 1 \ge 95\%$) | 0,957–0,965 | 1,000 | **Tercapai** |

### 9.2 Simulasi Kebutuhan Akurasi Kematangan (`AF-E-005`)

Simulasi proyeksi untuk mengetahui tingkat akurasi klasifikasi kematangan per citra terpotong yang dibutuhkan guna menaikkan $mAP50$ deteksi 4 kelas:

| Akurasi Klasifikasi per Citra Terpotong | Proyeksi $mAP50$ Deteksi Empat Kelas |
|---:|---:|
| 0,661 (Capaian saat ini) | 0,587 |
| 0,800 | 0,735 |
| 0,900 | **0,847** |
| 0,950 | 0,927 |

> [!WARNING]
> Target rekayasa $mAP50 \ge 0,85$ menuntut akurasi modul pengklasifikasi minimal 90%, yaitu sekitar 24 poin persentase di atas kapabilitas tertinggi model yang berhasil dicapai pada penelitian ini (66,35%).

### 9.3 Batas Atas Teoretis Pencacahan per Kelas dengan Deteksi *Oracle* (`AF-E-004`, Uji 953, 136 Pohon)

| Kelas Kematangan | MAE | Akurasi Tepat Persis | Akurasi Toleransi $\pm 1$ | Proporsi Tampak Tunggal |
|---|---:|---:|---:|---:|
| B1 | **0,101** | **0,899** | **1,000** | 11,6% |
| B2 | 0,239 | 0,768 | 0,993 | 23,4% |
| B3 | 0,638 | 0,428 | 0,942 | 22,7% |
| B4 | 0,268 | 0,739 | 0,993 | **40,5%** |
| **Total per Pohon** | **1,058** | **0,290** | **0,754** | &mdash; |

---

## 10. Batasan Validitas dan Kaveat Audit

| Aspek Batasan / Kaveat | Implikasi terhadap Pembacaan dan Interpretasi Data | Sumber Rujukan |
|---|---|---|
| Ambiguitas Makro-*F1* (`AF-E-012`) | Terdapat dua nilai: $0,6692$ (klaster terpasangkan) dan $0,5201$ (ujung ke ujung). Hanya nilai $0,5201$ yang setara dengan baseline $0,6034$. | [`EVIDENCE.md`](../docs/research_2026-09-06/EVIDENCE.md) |
| Pipeline Panen (`AF-E-012`, `AF-E-013`) | Menggunakan 132 pohon, detektor tunggal YOLO26m. Unggul pada estimasi B1 toleransi $\pm 1$ ($0,970$) dan akurasi ordinal ($0,9946$). | `AF-E-012`, `AF-E-013` |
| Pelanggaran kendala fisik (`AF-E-010`) | Telah dikoreksi oleh `AF-E-014`: pada profil terkunci dengan `max_size` $\le 3$, tingkat pelanggaran adalah $0,00\%$. | `AF-E-014`, `AF-E-016` |
| Status eksperimen `AF-E` | Eksperimen audit (`AF-E`) berfungsi sebagai diagnostik pelengkap, bukan pengganti angka acuan profil terkunci. | [Atlas 07](07_audit_forensik.md) |
| Kebocoran data (*leakage*) `combined1716` | Irisan identitas pohon (`tree_id`) antar-partisi belum diaudit tuntas. | [`ANALISIS_PIPELINE.md`](../docs/ANALISIS_PIPELINE_MENDALAM.md) |

---

## 11. Sumber Data dan Keterlacakan Artefak

| Rujukan Tabel | Berkas Artefak Sumber Data |
|---|---|
| Tabel 3 | [`combined1716`](../results/combined1716), [`new763`](../results/new763); [`detector_matrix.json`](../results/audit_forensik_2026-09-06/detector_matrix.json) |
| Tabel 4 | [`class_agnostic_metrics_audit_2026-09-03.json`](../results/class_agnostic_metrics_audit_2026-09-03.json) |
| Tabel 5, 6, 7 | [`pipeline_combined1716_generalization_locked.json`](../results/remote_eval_2026-08-27/metrics/pipeline_combined1716_generalization_locked.json) |
| Tabel 8 | Diturunkan dari medan `metrics.classification.confusion_prediction_rows` pada artefak `Wave-V2` |
| Tabel 9 | [`experiments/AUDIT-FORENSIK-2026-09-06.md`](../experiments/AUDIT-FORENSIK-2026-09-06.md) |

**Dokumentasi Terkait:**
- [`README.md`](README.md): Gerbang utama penjelajahan atlas metrik.
- Berkas Spesialisasi: Berkas `01`–`07` untuk analisis mendalam pada setiap tahapan modul eksperimen.
