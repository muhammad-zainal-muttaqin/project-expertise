# Rekapitulasi Papan Peringkat Hasil Eksperimen

Dokumen ini menyusun ulang seluruh angka pada [atlas metrik](README.md) ke dalam
bentuk papan peringkat (*leaderboard*) per korpus. Tujuh berkas atlas lainnya
disusun per domain teknis dan per ID eksperimen, sehingga menjawab pertanyaan
"eksperimen `V2-E-034` menghasilkan angka berapa". Berkas ini menjawab pertanyaan
yang berbeda: pada satu korpus, sistem mana yang terbaik untuk tiap tugas, dan
berapa angkanya.

Setiap tugas dijelaskan lengkap: arti operasionalnya, satuan penyebut evaluasi,
metrik penilaian, serta eksperimen dan nama sistem yang menghasilkan angkanya.
Kode eksperimen tidak dipakai sebagai pengganti penjelasan.

Dokumen ini bersifat aditif. Tidak ada angka, entri eksperimen, atau berkas atlas
lama yang diubah. Cakupan angkanya meliputi `V2-E-001` sampai `V2-E-045`,
`PT-E-000` sampai `PT-E-036`, gelombang verifikasi remote Agustus 2026, dan audit
forensik `AF-E-001` sampai `AF-E-016` tertanggal 6 September 2026.

---

## 1. Daftar tugas dan metrik

| Kolom | Arti | Satuan | Metrik |
|---|---|---|---|
| `detection` | Menemukan kotak pembatas (*bounding box*) tandan tanpa label kelas, yaitu lokalisasi murni | per citra | $AP50$ agnostik |
| `detection + classification` | Kotak pembatas dan label kematangan sekaligus, dihasilkan satu detektor satu tahap | per citra | $mAP50$ sadar-kelas empat kelas |
| `deduplication` | Satu tandan fisik yang tampak pada beberapa dari empat sisi pemotretan dihitung satu kali, tanpa penghitungan ganda | per pohon | F1 fisik |
| `classification` | Ketepatan label kelas pada tandan yang sudah tertaut benar terhadap acuannya | per pohon | Akurasi *matched-class* |
| `counting` | Jumlah tandan per kelas B1–B4 pada tingkat kohort, yaitu sasaran akhir sistem | per kohort (partisi validasi atau partisi uji) | Bias per kelas dan makro-rerata nilai mutlak bias |

### 1.1 `detection`

Tugas ini mengukur kemampuan detektor menemukan posisi tandan tanpa
mempertimbangkan kelas kematangannya. Seluruh kategori B1–B4 dilipat menjadi satu
kategori "tandan" sebelum evaluasi, sehingga angkanya merupakan plafon lokalisasi
spasial sistem.

Penyebut evaluasi adalah citra tunggal. Metrik penilaiannya adalah $AP50$
agnostik yang dihitung dengan `pycocotools.COCOeval` pada ambang $IoU = 0,5$.
Metrik titik operasi (presisi, daya tangkap, dan F1) dihitung pada ambang
kepercayaan $0,25$ dengan pencocokan *greedy* menurun berdasarkan skor.

Eksperimen penghasilnya adalah `V2-E-017` (YOLO26l dengan kelas dilipat),
`V2-E-024` (YOLO26l RGB+Sobel `edge` empat kanal), `V2-E-036` (RF-DETR-L native),
`V2-E-039` dan `V2-E-042` (ansambel WBF tiga detektor), `MAP_BOOST` (lapisan
*re-ranker* di atas WBF), serta `AF-E-006`, `AF-E-007`, dan `AF-E-011` dari audit
forensik.

### 1.2 `detection + classification`

Tugas ini mengukur detektor satu tahap yang menghasilkan kotak pembatas beserta
label kematangannya sekaligus, tanpa pengklasifikasi terpisah di hilir. Angkanya
menggabungkan kualitas lokalisasi dan kualitas pelabelan dalam satu bilangan.

Penyebut evaluasi tetap citra tunggal. Metrik penilaiannya adalah $mAP50$
sadar-kelas, yaitu rerata $AP50$ pada empat kelas B1–B4. Nilai per kelas
($AP_{B1}$ sampai $AP_{B4}$) tercatat pada
[`01_deteksi_dan_lokalisasi.md`](01_deteksi_dan_lokalisasi.md).

Eksperimen penghasilnya adalah `V2-E-001` (tiga arsitektur pada korpus 953),
`V2-E-034` (korpus 763), `V2-E-035` (korpus Combined-1716), `V2-E-039` dan
`V2-E-042` (ansambel WBF), `MAP_BOOST`, serta `AF-E-005` untuk plafon teoretis
dan `AF-E-006` untuk replikasi silang.

### 1.3 `deduplication`

Setiap pohon dipotret dari empat sisi. Satu tandan fisik dapat muncul pada dua
sampai empat citra sekaligus. Modul penaut (*linker*) menggabungkan kemunculan
tersebut menjadi satu klaster tandan fisik, sehingga jumlah tandan tidak terhitung
berlipat.

Penyebut evaluasi adalah pohon, dengan unit penilaian berupa klaster tandan
fisik. Metrik penilaiannya adalah F1 fisik: klaster prediksi dipasangkan dengan
tandan acuan pada ambang $IoU \ge 0,5$, lalu presisi dan daya tangkap dihitung
pada tingkat klaster. Metrik ini berbeda dari F1 pasangan (*pair F1*) yang menilai
relasi biner antara dua kotak dari sisi berbeda, dan berbeda pula dari F1 titik
operasi pada tugas `detection`.

Eksperimen penghasilnya adalah `V2-E-043` (penaut *greedy strict*), `V2-E-045`
(profil terkunci validasi), `Wave-V2` untuk dua profil terkunci uji yaitu GSP MILP
pada korpus 763 dan Hungarian *Anchor A* pada korpus 953, `PT-E-008` (prior rotasi
bertanda), `PT-E-020` (penaut global DAMIMAS), serta `AF-E-012`, `AF-E-014`, dan
`AF-E-016`.

### 1.4 `classification`

Tugas ini mengukur ketepatan label kematangan, tetapi hanya pada klaster yang
sudah berhasil dipasangkan dengan tandan acuan. Klaster palsu dan tandan acuan
yang terlewat tidak masuk hitungan, sehingga metrik ini memisahkan mutu pelabelan
dari mutu penautan di hulunya.

Penyebut evaluasi adalah pohon, dengan unit penilaian berupa klaster yang
berpasangan. Metrik penilaiannya adalah akurasi *matched-class*. Makro-F1 ujung ke
ujung dilaporkan terpisah karena penyebutnya berbeda: makro-F1 memperhitungkan
klaster palsu dan tandan terlewat, sedangkan akurasi *matched-class* tidak.

Eksperimen penghasilnya adalah `V2-E-044` (pengklasifikasi *crop* lima *epoch*),
`V2-E-045`, `Wave-V2`, `PT-E-001` (aturan keputusan R0–R4), serta `AF-E-003`
(pengklasifikasi berbasis struktur tanpa piksel), `AF-E-009` (fusi kemunculan
objek dan struktur), dan `AF-E-012`.

### 1.5 `counting`

Tugas ini merupakan sasaran akhir sistem: memperkirakan jumlah tandan pada tiap
kelas B1–B4 untuk keseluruhan kohort, bukan untuk pohon tunggal. Seluruh pohon
pada satu partisi dijumlahkan, lalu total prediksi dibandingkan dengan total acuan
pada tiap kelas.

Penyebut evaluasi adalah kohort, yaitu satu partisi utuh (validasi atau uji) pada
satu korpus. Metrik penilaiannya adalah bias per kelas dengan dua bentuk
pelaporan. Bias absolut adalah selisih total prediksi dikurangi total acuan pada
kelas tersebut, bersatuan tandan. Bias relatif adalah bias absolut dibagi total
acuan kelas itu, dinyatakan dalam persen. Nilai negatif menunjukkan estimasi
kurang (*under-predict*), nilai positif menunjukkan estimasi berlebih
(*over-predict*). Ringkasan satu bilangan untuk seluruh kohort adalah makro-rerata
nilai mutlak bias relatif pada empat kelas.

Metrik ini berbeda dari metrik pencacahan per pohon yang tercatat pada
[`03_pencacahan_per_pohon.md`](03_pencacahan_per_pohon.md). Macro MAE, Total MAE,
Class $\pm 1$ Acc, dan Tree $\pm 1$ Acc dihitung per pohon lalu dirata-ratakan,
sehingga estimasi berlebih pada satu pohon dan estimasi kurang pada pohon lain
tetap terhitung keduanya. Bias tingkat kohort membiarkan kedua galat itu saling
meniadakan, sehingga nilainya selalu lebih kecil. Kedua sudut pandang dilaporkan
karena keduanya menjawab pertanyaan yang berbeda.

Eksperimen penghasilnya adalah `V2-E-002` (Ridge dengan fitur $F_{all}$ 67
dimensi), `V2-E-045`, `Wave-V2`, serta `AF-E-004` (plafon dengan model batas atas
teoretis atau *oracle*), `AF-E-008`, dan `AF-E-013`.

---

## 2. Aturan pembacaan

Tiga aturan normalisasi pada [`README.md`](README.md) bagian 3 berlaku penuh di
sini dan diulang karena sering menjadi sumber salah tafsir.

1. $AP50$ agnostik adalah plafon lokalisasi spasial tanpa label kelas. Nilai
   $0,8350$ pada korpus 953 berarti $83,50\%$ presisi rerata lokalisasi, bukan
   akurasi kematangan dan bukan akurasi pencacahan.
2. Class $\pm 1$ Acc pada modul pencacahan mengukur persentase sel pohon-kelas
   dengan galat cacah paling banyak satu tandan. Metrik itu bukan akurasi
   klasifikasi citra.
3. Tanda minus tipografis `−` menunjukkan penurunan atau estimasi kurang, bukan
   tanda hubung teks.

Sel bertanda `—` berarti metrik tidak berlaku menurut definisi untuk baris
tersebut. Sel bertanda `N/A` disertai catatan kaki beralasan berarti metrik
berlaku, tetapi tidak tersedia pada artefak dengan protokol yang sama.

> [!WARNING]
> **Arah definisi kelas B1–B4 belum diselaraskan.** Pembuka
> [`02_klasifikasi_kematangan.md`](02_klasifikasi_kematangan.md) mendefinisikan B1
> sebagai mentah dan B4 sebagai lewat matang. Kartu dataset resmi
> `ULM-DS-Lab/SawitMVC-YOLO`, [`docs/DATASET.md`](../docs/DATASET.md), dan dua
> sinyal data (ukuran kotak median serta relief kedalaman) menunjukkan arah
> sebaliknya, yaitu B1 sebagai tandan matang siap panen. Seluruh baris `AF-E-###`
> memakai arah kartu dataset resmi. Tabel bias pada bagian 5 dibaca per kelas,
> sehingga penyelarasan istilah ini perlu diputuskan sebelum publikasi.

---

## 3. Korpus dan partisi evaluasi

| Korpus | Modalitas | Ukuran | Partisi uji | Peran pada papan peringkat |
|---|---|---|---|---|
| `combined1716` | RGB | 1.716 pohon | 1.052 citra | Daya statistik terkuat untuk tugas deteksi; juga berperan sebagai bank pelatihan detektor |
| `763-depth` | RGB dan kedalaman sensor Y16 | 763 pohon (352 Juli dan 411 Agustus 2026) | 440 citra, 110 pohon empat sisi | Satu-satunya korpus dengan modalitas kedalaman fisik |
| `953` | RGB | 953 pohon | 588 citra, 135 pohon empat sisi | Korpus rujukan historis, terhubung ke publikasi *Data in Brief* 2026 |

Korpus `combined1716` memiliki partisi uji terbesar, sehingga selang kepercayaan
metrik deteksinya paling sempit. Keunggulan itu hanya berlaku untuk tugas
`detection` dan `detection + classification`. Untuk tiga tugas hilir, korpus ini
tidak memiliki angka sama sekali; alasannya dijelaskan pada bagian 4.1.

**Dua korpus, dua protokol anotasi.** Temuan `AF-E-001` menunjukkan bahwa korpus
953 (Mei 2026) dan korpus 763 (Juli sampai Agustus 2026) tidak memakai protokol
anotasi yang sama. Pada 352 pohon fisik yang identik di kedua rilis, jumlah tandan
unik per pohon turun dari $9,89$ menjadi $3,99$, sedangkan proporsi sisi tanpa
anotasi naik dari $1,1\%$ menjadi $14,2\%$. Perbedaan itu terukur pada dua sumbu
terpisah, yaitu konvensi kotak pembatas dan kelengkapan anotasi tingkat pohon.
Akibatnya, sebagian besar selisih lintas korpus mengukur selisih protokol
pengamatan, dan tidak dapat dibaca sebagai kegagalan generalisasi lintas domain.

---

## 4. Papan peringkat per korpus

Baris diurutkan menurut kelengkapan sistem, dari detektor tunggal sampai pipeline
empat sisi utuh, agar progresi penambahan komponen terbaca langsung. Kolom
`counting (diagnostik)` memuat MAE pencacahan total per pohon dan akurasi
toleransi $\pm 1$ sebagai indikator pendukung. Metrik penilaian akhir tugas
`counting`, yaitu bias per kelas, berada pada bagian 5.

### 4.1 `combined1716` (RGB, 1.052 citra uji)

| Metode/sistem | `detection` | `det+class` | `dedup` | `classification` | `counting` (diagnostik) | Status |
|---|---:|---:|---:|---:|---:|---|
| YOLO26l (`V2-E-035`) | N/A¹ | 0,5389 | N/A² | N/A² | N/A² | uji |
| RT-DETR-L (`V2-E-035`) | N/A¹ | 0,5745 | N/A² | N/A² | N/A² | uji |
| RF-DETR-L (`V2-E-036`) | 0,7850 | **0,5960** | N/A² | N/A² | N/A² | uji |
| WBF [YOLO26l+RT-DETR-L+RF-DETR-L] (`V2-E-039`) | **0,8104** | 0,5538 | N/A² | N/A² | N/A² | uji |

¹ `N/A: audit agnostik 2026-09-03 tidak menghitung baris ini.`
² `N/A: pipeline empat sisi tidak dievaluasi pada partisi uji korpus ini.`

Korpus Combined-1716 adalah gabungan korpus 953 dan korpus 763, dan berperan
sebagai bank pelatihan tiga detektor. Bobot yang dilatih pada korpus ini kemudian
diuji pada partisi uji 953 dan 763-depth. Pool 1.716 pohon itu sendiri tidak
memiliki nilai acuan kebenaran (*ground truth*) multi-sisi tingkat pohon fisik,
sehingga F1 fisik, akurasi *matched-class*, dan bias pencacahan per kelas tidak
dapat dihitung padanya. Pemeriksaan terhadap
[`03_pencacahan_per_pohon.md`](03_pencacahan_per_pohon.md),
[`04_pengaitan_multi_tampak.md`](04_pengaitan_multi_tampak.md), dan
[`05_pipeline_end_to_end.md`](05_pipeline_end_to_end.md) memastikan ketiganya
tidak memuat satu pun baris Combined-1716.

Satu perbedaan mudah tertukar perlu dicatat. Nilai $mAP50 = 0,5861$ yang beredar
pada beberapa dokumen adalah hasil bank `combined1716` yang dievaluasi pada
partisi uji 953, bukan hasil pada partisi uji Combined-1716 sendiri. Angka yang
benar untuk baris WBF pada tabel di atas adalah $0,5538$. Koreksi tersebut
tercatat pada [`01_deteksi_dan_lokalisasi.md`](01_deteksi_dan_lokalisasi.md).

### 4.2 `763-depth` (RGB+D, 440 citra dan 110 pohon uji)

| Metode/sistem | `detection` | `det+class` | `dedup` | `classification` | `counting` (diagnostik) | Status |
|---|---:|---:|---:|---:|---:|---|
| YOLO26l native (`V2-E-034`) | N/A¹ | 0,5163 | — | — | — | uji |
| RT-DETR-L native (`V2-E-034`) | N/A¹ | 0,5580 | — | — | — | uji |
| RF-DETR-L native (`V2-E-034`) | 0,7951 | 0,6129 | — | — | — | uji |
| RF-DETR-L bank `combined1716` (`V2-E-042`) | N/A¹ | 0,6711 | — | — | — | uji |
| WBF bank `combined1716` (`V2-E-042`) | 0,8764 | **0,6691** | — | — | — | uji |
| WBF + *re-ranker* (`MAP_BOOST`) | **0,8783** | 0,6552 | — | — | — | uji |
| WBF + penaut prior rotasi + pencacah Ridge (`V2-E-045`) | — | — | 0,8257 | 83,55% | 0,726 / 84,62% | validasi (117 pohon) |
| WBF + penaut prior rotasi + pencacah Ridge (`V2-E-045`) | — | — | 0,8069 | 80,31% | 0,891 / 80,91% | uji |
| WBF + penaut GSP MILP + pencacah Ridge (`Wave-V2`) | — | — | 0,8526 | 84,57% | 0,932 / 78,63% | validasi (117 pohon) |
| WBF + penaut GSP MILP + pencacah Ridge (`Wave-V2`) | — | — | **0,8534** | **81,62%** | **0,773 / 85,45%** | uji |

¹ `N/A: audit agnostik 2026-09-03 tidak menghitung baris ini.`

Lapisan *re-ranker* menaikkan lokalisasi agnostik dari $0,8764$ menjadi $0,8783$,
tetapi menurunkan $mAP50$ sadar-kelas dari $0,6691$ menjadi $0,6552$. Arah kedua
perubahan itu berlawanan, dan berbeda dari perilakunya pada korpus 953 tempat
kedua metrik naik. Lapisan tersebut karena itu belum layak diterapkan seragam
lintas korpus.

Profil `Wave-V2` GSP MILP dikunci pada partisi uji dan memberi F1 fisik $0,8534$
dengan selang kepercayaan 95% $[0,8301; 0,8761]$, serta akurasi *matched-class*
$81,62\%$ dengan selang kepercayaan 95% $[0,7765; 0,8556]$. Profil `V2-E-045`
dikunci pada partisi validasi dan lebih rendah pada partisi uji, tetapi angkanya
merupakan konfirmasi generalisasi karena parameternya tidak dipilih dari partisi
uji.

### 4.3 `953` (RGB, 588 citra dan 135 pohon uji)

| Metode/sistem | `detection` | `det+class` | `dedup` | `classification` | `counting` (diagnostik) | Status |
|---|---:|---:|---:|---:|---:|---|
| YOLO26l (`V2-E-001`) | N/A¹ | 0,5435 | — | — | — | uji |
| RT-DETR-L (`V2-E-001`) | N/A¹ | 0,5781 | — | — | — | uji |
| RF-DETR-L (`V2-E-001`) | N/A¹ | 0,6012 | — | — | — | uji |
| YOLO26m tunggal agnostik (`AF-E-011`) | 0,8104 | — | — | — | — | uji |
| WBF bank `combined1716` (`V2-E-042`) | 0,8350 | 0,5861 | — | — | — | uji |
| WBF + *re-ranker* (`MAP_BOOST`) | **0,8419** | **0,5970** | — | — | — | uji |
| WBF + penaut prior rotasi + pencacah Ridge (`V2-E-045`) | — | — | 0,8087 | 70,04% | 1,253 / 67,03% | validasi (91 pohon) |
| WBF + penaut prior rotasi + pencacah Ridge (`V2-E-045`) | — | — | 0,8043 | 71,11% | 1,393 / 61,48% | uji |
| WBF + penaut Hungarian *Anchor A* + pencacah Ridge (`Wave-V2`) | — | — | 0,8232 | 75,42% | 1,253 / 67,03% | validasi (91 pohon) |
| WBF + penaut Hungarian *Anchor A* + pencacah Ridge (`Wave-V2`) | — | — | **0,8387** | **74,42%** | **1,363 / 63,70%** | uji |
| Pipeline Panen: YOLO26m + penaut terlatih + skor ordinal (`AF-E-012`, `AF-E-013`) | — | — | 0,7586 | 71,65% | 1,374 / 63,74% | validasi (91 pohon) |
| Pipeline Panen: YOLO26m + penaut terlatih + skor ordinal (`AF-E-012`, `AF-E-013`) | — | — | 0,7619 | 71,61% | 1,402 / 56,82% | uji (132 pohon) |

¹ `N/A: audit agnostik 2026-09-03 tidak menghitung baris ini.`

Dua baris Pipeline Panen memakai penyebut dan konfigurasi yang berbeda dari baris
lain pada tabel yang sama, sehingga perbandingan langsung tidak setara. Penyebut
ujinya 132 pohon empat sisi, sedangkan baris lain memakai 135 pohon. Detektornya
YOLO26m tunggal, sedangkan baris lain memakai ansambel WBF tiga detektor, dan
daya tangkap fisiknya hanya $0,6878$. Penaut yang dipakai adalah model tepi
HistGradientBoosting terlatih milik audit, sedangkan baris terkunci memakai
Hungarian *Anchor A* atau GSP MILP.

Pipeline Panen mengungguli profil terkunci pada tiga besaran yang tidak pernah
dilaporkan sebelumnya: cacah B1 siap panen dengan toleransi $\pm 1$ mencapai
$0,970$, akurasi ordinal $\pm 1$ mencapai $0,9946$, dan akurasi dua kelas matang
berbanding belum matang mencapai $0,8678$. Ketiganya berasal dari satu keputusan
rancangan yang sama, yaitu penentuan kelas pada tingkat tandan fisik melalui skor
ordinal kontinu. Kesimpulan `AF-E-012` menyatakan bahwa langkah lanjutan yang
tepat adalah menggabungkan penaut GSP proyek dengan tahap kelas ordinal audit,
karena keunggulan keduanya berada pada komponen yang berbeda.

---

## 5. Pencacahan kohort per kelas

Tabel berikut memuat metrik penilaian akhir tugas `counting`, yaitu bias per
kelas. Nilainya dihitung terpisah untuk tiap partisi pada tiap korpus, dengan
menjumlahkan seluruh pohon dalam partisi tersebut.

### 5.1 Korpus 953, partisi uji (penaut Hungarian *Anchor A*, 135 pohon)

| Kelas | Total prediksi | Total acuan | Bias absolut | Bias relatif |
|---|---:|---:|---:|---:|
| B1 | 104 | 113 | −9 | −7,96% |
| B2 | 145 | 246 | −101 | −41,06% |
| B3 | 824 | 706 | +118 | +16,71% |
| B4 | 251 | 277 | −26 | −9,39% |
| **Total** | **1.324** | **1.342** | **−18** | **−1,34%** |

Makro-rerata nilai mutlak bias relatif mencapai $18,78\%$.

### 5.2 Korpus 953, partisi validasi (penaut Hungarian *Anchor A*, 91 pohon)

| Kelas | Total prediksi | Total acuan | Bias absolut | Bias relatif |
|---|---:|---:|---:|---:|
| B1 | 84 | 86 | −2 | −2,33% |
| B2 | 112 | 186 | −74 | −39,78% |
| B3 | 560 | 476 | +84 | +17,65% |
| B4 | 186 | 188 | −2 | −1,06% |
| **Total** | **942** | **936** | **+6** | **+0,64%** |

Makro-rerata nilai mutlak bias relatif mencapai $15,21\%$.

### 5.3 Korpus 763-depth, partisi uji (penaut GSP MILP, 110 pohon)

| Kelas | Total prediksi | Total acuan | Bias absolut | Bias relatif |
|---|---:|---:|---:|---:|
| B1 | 67 | 94 | −27 | −28,72% |
| B2 | 227 | 199 | +28 | +14,07% |
| B3 | 177 | 215 | −38 | −17,67% |
| B4 | 41 | 50 | −9 | −18,00% |
| **Total** | **512** | **558** | **−46** | **−8,24%** |

Makro-rerata nilai mutlak bias relatif mencapai $19,62\%$.

### 5.4 Korpus 763-depth, partisi validasi (penaut GSP MILP, 117 pohon)

| Kelas | Total prediksi | Total acuan | Bias absolut | Bias relatif |
|---|---:|---:|---:|---:|
| B1 | 66 | 96 | −30 | −31,25% |
| B2 | 234 | 205 | +29 | +14,15% |
| B3 | 176 | 216 | −40 | −18,52% |
| B4 | 32 | 53 | −21 | −39,62% |
| **Total** | **508** | **570** | **−62** | **−10,88%** |

Makro-rerata nilai mutlak bias relatif mencapai $25,88\%$.

### 5.5 Catatan metodologis tabel bias

**Cara penurunan angka.** Keempat tabel diturunkan dari matriks konfusi
`confusion_prediction_rows` berukuran $5 \times 5$ pada artefak profil terkunci.
Baris menyatakan kelas prediksi, dengan baris kelima menampung tandan acuan yang
tidak memperoleh prediksi. Kolom menyatakan kelas acuan, dengan kolom kelima
menampung prediksi yang tidak memperoleh pasangan acuan. Total prediksi kelas
ke-$k$ adalah jumlah baris ke-$k$, sedangkan total acuan kelas ke-$k$ adalah
jumlah kolom ke-$k$.

**Hasil pemeriksaan konsistensi.** Jumlah baris pertama sampai keempat
diverifikasi sama dengan medan `pred_clusters`, dan jumlah kolom pertama sampai
keempat sama dengan medan `gt_bunches`. Pada korpus 953 keduanya cocok persis.
Pada korpus 763-depth terdapat selisih satu tandan: jumlah kolom acuan bernilai
558 sedangkan `gt_bunches` tercatat 559 pada partisi uji, dan 570 berbanding 571
pada partisi validasi. Satu tandan acuan tidak terwakili pada matriks konfusi.
Selisih tersebut dilaporkan apa adanya dan belum ditelusuri penyebabnya.

**Profil sumber berbeda dari kolom `dedup` dan `classification`.** Keempat tabel
berasal dari profil penaut terkunci `Wave-V2`, sedangkan kolom `dedup` dan
`classification` pada bagian 4 juga memuat baris `V2-E-045`. Artefak
[`pipeline_combined1716_generalization_locked.json`](../results/remote_eval_2026-08-27/metrics/pipeline_combined1716_generalization_locked.json)
milik `V2-E-045` menyimpan metrik agregat saja dan tidak menyimpan rincian per
kelas, sehingga bias per kelas untuk profil itu tidak dapat dihitung tanpa
menjalankan ulang inferensi. Baris `Wave-V2` dicantumkan pada bagian 4 agar tabel
bias memiliki baris induk yang cocok.

**Pola galat yang konsisten.** Pada korpus 953, kelas B2 diestimasi kurang secara
substansial ($−41,06\%$ pada partisi uji dan $−39,78\%$ pada partisi validasi)
sedangkan kelas B3 diestimasi berlebih ($+16,71\%$ dan $+17,65\%$). Arah dan
besaran yang berlawanan itu menunjukkan perpindahan massa prediksi dari B2 ke B3,
dan bukan galat pencacahan acak. Temuan tersebut sejalan dengan matriks konfusi
`AF-E-009`, yang mencatat batas B2 berbanding B3 sebagai batas tersulit pada data
ini dengan 195 galat, berbanding 57 galat pada batas B1 berbanding B2. Pada korpus
763-depth polanya berbeda: B1 dan B4 diestimasi kurang, sedangkan B2 diestimasi
berlebih.

**Efek saling meniadakan.** Bias tingkat kohort jauh lebih kecil daripada galat
tingkat pohon karena estimasi berlebih pada satu pohon meniadakan estimasi kurang
pada pohon lain. Pada korpus 953 partisi uji, bias total hanya $−18$ tandan dari
1.342 tandan acuan, sedangkan MAE pencacahan per pohon mencapai $1,363$. Audit
[`count_error_cancellation.json`](../results/audit_2026-09-06/count_error_cancellation.json)
mengukur efek tersebut secara langsung: dari 37 pohon dengan cacah total tepat, 24
pohon di antaranya tepat karena kesalahan positif dan kesalahan negatif saling
meniadakan, dan bukan karena himpunan tandannya benar. Bias kohort karena itu
tidak dapat dibaca sendirian sebagai ukuran ketepatan sistem.

---

## 6. Batasan validitas dan audit

**Plafon aritmetis taksonomi empat kelas.** Eksperimen `AF-E-005` mengukur plafon
$mAP50$ dengan lokalisasi dibuat sempurna, yaitu setiap prediksi adalah kotak
acuan sehingga presisi dan daya tangkap bernilai $1,0$. Pada kondisi itu $mAP50$
empat kelas hanya mencapai $0,6569$, sedangkan capaian nyata sistem terbaik adalah
$0,5970$. Ruang perbaikan yang tersisa untuk seluruh tumpukan detektor karena itu
sekitar $6$ poin $mAP$, dan bukan $25$ poin. Target rekayasa $mAP50 \ge 0,85$ pada
taksonomi empat kelas dinyatakan tidak terjangkau. Pada taksonomi dua kelas,
plafonnya $0,8766$ sehingga target itu terjangkau.

**Makro-F1 `AF-E-012` memiliki dua penyebut.** Nilai $0,6692$ yang dilaporkan
`AF-E-012` dihitung hanya pada klaster yang berpasangan dengan acuan. Sesi
verifikasi paralel pada
[`docs/research_2026-09-06/EVIDENCE.md`](../docs/research_2026-09-06/EVIDENCE.md)
menghitung ulang besaran itu secara ujung ke ujung, dengan klaster palsu dan
tandan terlewat ikut dihitung, lalu memperoleh $0,5201$ pada 132 pohon. Kedua
angka benar menurut definisinya masing-masing, tetapi hanya angka ujung ke ujung
yang setara dengan makro-F1 $0,6034$ milik profil terkunci.

**Koreksi terhadap temuan cacat kendala sisi.** Eksperimen `AF-E-010` melaporkan
bahwa $45,3\%$ klaster melanggar kendala satu tandan satu sisi. Angka itu
dikoreksi oleh `AF-E-014`: pelanggaran tersebut hanya muncul pada jalur sapuan
parameter, sedangkan pada seluruh profil terkunci dengan `max_size` paling banyak
3 pelanggarannya bernilai $0,00\%$, dan 0 dari 630 konfigurasi berubah setelah
perbaikan. Eksperimen `AF-E-016` mengonfirmasi bahwa profil Hungarian *Anchor A*
yang memakai `max_size` bernilai 4 juga tidak terpengaruh, dengan 0 dari 135 pohon
berbeda. Seluruh angka terkunci pada dokumen ini karena itu tetap berlaku.

**Tidak ada hasil audit forensik yang menggantikan angka terkunci.** Seluruh
eksperimen `AF-E-001` sampai `AF-E-016` memakai protokol yang berbeda, yaitu
detektor berkapasitas lebih kecil, penaut milik audit sendiri, atau arah taksonomi
yang berlawanan. Hasilnya berada di bawah profil terkunci pada metrik yang setara.
Status validitas pada atlas karena itu tidak berubah, dan baris `AF-E` masuk
sebagai eksperimen tambahan.

**Audit silsilah partisi belum tuntas.** Pemeriksaan irisan `tree_id` antara
partisi latih `combined1716` dan kedua kumpulan uji lokal belum selesai menurut
[`docs/ANALISIS_PIPELINE_MENDALAM.md`](../docs/ANALISIS_PIPELINE_MENDALAM.md).
Selama audit itu belum tuntas, seluruh angka bank `combined1716` yang diuji pada
partisi 953 dan 763-depth menyimpan kemungkinan kebocoran partisi data
(*data leakage*) yang belum terkuantifikasi.

**Partisi uji pernah dibaca pada iterasi historis.** Artefak `V2-E-045` mencatat
kaveat ini secara eksplisit. Angka uji pada dokumen ini merupakan konfirmasi
terhadap profil yang dikunci pada partisi validasi, dan tidak dapat disebut
sebagai hasil partisi terisolasi yang sepenuhnya belum tersentuh.

---

## 7. Peta artefak sumber

| Kolom papan peringkat | Artefak sumber |
|---|---|
| `detection` | [`class_agnostic_metrics_audit_2026-09-03.json`](../results/class_agnostic_metrics_audit_2026-09-03.json) pada medan `rows[].ap50_agnostic` |
| `detection` setelah *re-ranker* | [`map_boost_artifacts/953`](../results/remote_eval_2026-08-28/map_boost_artifacts/953/results_test_locked.json) dan [`map_boost_artifacts/depth`](../results/remote_eval_2026-08-28/map_boost_artifacts/depth/results_test_locked.json) pada medan `test_metrics.agnostic.AP50` |
| `det + class` | [`combined1716`](../results/combined1716) dan [`new763`](../results/new763) pada medan `splits.test.mAP50`; [`perkelas_pycoco_v2repro.json`](../results/perkelas_pycoco_v2repro.json); [`extra_metrics_sesi2026-08.json`](../results/extra_metrics_sesi2026-08.json) |
| `dedup` dan `classification` profil `V2-E-045` | [`pipeline_combined1716_generalization_locked.json`](../results/remote_eval_2026-08-27/metrics/pipeline_combined1716_generalization_locked.json) |
| `dedup` dan `classification` profil `Wave-V2` | [`gsp_artifacts/953`](../results/remote_eval_2026-08-28/gsp_artifacts/953/results_test_locked.json), [`gsp_artifacts/depth`](../results/remote_eval_2026-08-28/gsp_artifacts/depth/results_test_locked.json), dan [`anchor_check.json`](../results/remote_eval_2026-08-28/validation_wave/reports/anchor_check.json) |
| Bias pencacahan per kelas | Diturunkan dari medan `metrics.classification.confusion_prediction_rows` pada tiga artefak `Wave-V2` di atas |
| Baris `AF-E` | [`detector_matrix.json`](../results/audit_forensik_2026-09-06/detector_matrix.json) dan [`panen/panen_final.json`](../results/audit_forensik_2026-09-06/panen/panen_final.json) |

Rujukan pelengkap: [`README.md`](README.md) sebagai gerbang atlas, tujuh berkas
spesialisasi `01` sampai `07`, log *append-only*
[`experiments/AUDIT-FORENSIK-2026-09-06.md`](../experiments/AUDIT-FORENSIK-2026-09-06.md),
serta sintesis [`docs/AUDIT-FORENSIK-2026-09-06.md`](../docs/AUDIT-FORENSIK-2026-09-06.md).
