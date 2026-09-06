# Rekapitulasi Papan Peringkat Hasil Eksperimen

Menyusun ulang angka [atlas metrik](README.md) ke bentuk papan peringkat
(*leaderboard*) per korpus. Cakupan: `V2-E-001`–`V2-E-045`, `PT-E-000`–`PT-E-036`,
verifikasi remote Agustus 2026, dan audit forensik `AF-E-001`–`AF-E-016`. Aditif;
tidak ada angka baru dan tidak ada berkas atlas lama yang diubah.

---

## 1. Tugas dan metrik

| Kolom | Arti | Satuan | Metrik | Eksperimen utama |
|---|---|---|---|---|
| `detection` | Menemukan kotak pembatas (*bounding box*) tandan tanpa label kelas | per citra | $AP50$ agnostik | `V2-E-036`, `V2-E-039`, `V2-E-042`, `MAP_BOOST`, `AF-E-011` |
| `detection + classification` | Kotak pembatas dan label kematangan sekaligus, dari satu detektor satu tahap | per citra | $mAP50$ sadar-kelas empat kelas | `V2-E-001`, `V2-E-034`, `V2-E-035`, `V2-E-042`, `MAP_BOOST`, `AF-E-005`, `AF-E-006` |
| `deduplication` | Satu tandan yang tampak di beberapa sisi dihitung satu kali | per pohon | F1 fisik, pencocokan $IoU \ge 0,5$ | `V2-E-043`, `V2-E-045`, `Wave-V2`, `PT-E-008`, `AF-E-012` |
| `classification` | Ketepatan kelas pada tandan yang sudah tertaut benar | per pohon | Akurasi *matched-class* | `V2-E-044`, `V2-E-045`, `Wave-V2`, `PT-E-001`, `AF-E-009`, `AF-E-012` |
| `counting` | Jumlah tandan per kelas B1–B4 untuk seluruh kohort | per kohort | Bias per kelas, makro-rerata nilai mutlak bias | `V2-E-002`, `V2-E-045`, `Wave-V2`, `AF-E-004`, `AF-E-013` |

Sistem bekerja berurutan: detektor menemukan tandan per citra, penaut
menggabungkan kemunculan tandan yang sama dari empat sisi, pengklasifikasi
menentukan kematangannya, pencacah menjumlahkan per kelas. Penyebut tiap tugas
berbeda, sehingga angkanya tidak dapat saling dibandingkan.

### 1.1 Metrik yang sering tertukar

| Metrik | Mengukur | Bukan |
|---|---|---|
| $AP50$ agnostik $0,8350$ | Presisi rerata lokalisasi $83,50\%$, kelas dilipat jadi satu | Akurasi kematangan, akurasi pencacahan |
| F1 fisik | Ketepatan klaster tandan gabungan multi-sisi | F1 pasangan (relasi dua kotak antar-sisi), F1 titik operasi deteksi |
| Akurasi *matched-class* | Ketepatan kelas pada klaster yang berpasangan saja | Makro-F1 ujung ke ujung, yang ikut menghitung klaster palsu dan tandan terlewat |
| Bias kohort | Selisih total prediksi dan total acuan per kelas, galat antar-pohon saling meniadakan | Macro MAE atau Tree $\pm 1$ Acc, yang menghitung galat tiap pohon |

---

## 2. Korpus

| Korpus | Modalitas | Ukuran | Citra uji | Pohon uji empat sisi | Cakupan tugas |
|---|---|---:|---:|---:|---|
| `combined1716` | RGB | 1.716 pohon | 1.052 | — | Deteksi saja |
| `763-depth` | RGB + kedalaman Y16 | 763 pohon | 440 | 110 | Kelima tugas |
| `953` | RGB | 953 pohon | 588 | 135 | Kelima tugas |

`combined1716` adalah gabungan korpus 953 dan 763 yang berperan sebagai bank
pelatihan detektor. Pool 1.716 pohonnya tidak memiliki nilai acuan kebenaran
(*ground truth*) multi-sisi tingkat pohon, sehingga tiga tugas hilir tidak dapat
dihitung padanya.

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

## 3. Papan peringkat per korpus

Baris diurutkan dari detektor tunggal sampai pipeline empat sisi utuh. Kolom
`counting (diagnostik)` memuat MAE cacah total per pohon dan akurasi $\pm 1$;
penilaian akhir tugas `counting` ada di bagian 4. Sel `—` berarti metrik tidak
berlaku menurut definisi; sel `N/A` berarti berlaku tetapi tidak tersedia.

### 3.1 `combined1716` (RGB, 1.052 citra uji)

| Metode/sistem | `detection` | `det+class` | `dedup` | `classification` | `counting` (diagnostik) | Status |
|---|---:|---:|---:|---:|---:|---|
| YOLO26l (`V2-E-035`) | N/A¹ | 0,5389 | N/A² | N/A² | N/A² | uji |
| RT-DETR-L (`V2-E-035`) | N/A¹ | 0,5745 | N/A² | N/A² | N/A² | uji |
| RF-DETR-L (`V2-E-036`) | 0,7850 | **0,5960** | N/A² | N/A² | N/A² | uji |
| WBF [YOLO26l+RT-DETR-L+RF-DETR-L] (`V2-E-039`) | **0,8104** | 0,5538 | N/A² | N/A² | N/A² | uji |

¹ `N/A: audit agnostik 2026-09-03 tidak menghitung baris ini.`
² `N/A: pipeline empat sisi tidak dievaluasi pada partisi uji korpus ini.`

Nilai $mAP50 = 0,5861$ yang beredar di beberapa dokumen adalah bank
`combined1716` yang dievaluasi pada partisi uji 953, bukan hasil pada partisi uji
`combined1716` sendiri. Angka yang benar untuk baris WBF adalah $0,5538$
([`01_deteksi_dan_lokalisasi.md`](01_deteksi_dan_lokalisasi.md)).

### 3.2 `763-depth` (RGB+D, 440 citra dan 110 pohon uji)

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

Lapisan *re-ranker* di korpus ini bekerja ke dua arah berlawanan: lokalisasi
agnostik naik $0,8764 \to 0,8783$, sedangkan $mAP50$ sadar-kelas turun
$0,6691 \to 0,6552$. Pada korpus 953 keduanya naik.

### 3.3 `953` (RGB, 588 citra dan 135 pohon uji)

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

### 3.4 Rincian pipeline empat sisi

| Sistem | Korpus | Partisi | $n$ pohon | P fisik | R fisik | F1 fisik | MAE cacah | $\pm 1$ | Tepat | *Matched-class* | Makro-F1 E2E |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Wave-V2` GSP | 763-depth | validasi | 117 | **0,9055** | 0,8056 | 0,8526 | 0,932 | 78,63% | 34,19% | 84,57% | **0,6807** |
| `Wave-V2` GSP | 763-depth | uji | 110 | 0,8926 | 0,8175 | **0,8534** | 0,773 | **85,45%** | **44,55%** | 81,62% | 0,6519 |
| `V2-E-045` | 763-depth | validasi | 117 | 0,8431 | 0,8091 | 0,8257 | 0,726 | 84,62% | 44,44% | 83,55% | 0,6749 |
| `V2-E-045` | 763-depth | uji | 110 | 0,8142 | 0,7996 | 0,8069 | 0,891 | 80,91% | 33,64% | 80,31% | 0,6047 |
| `Wave-V2` Hungarian | 953 | validasi | 91 | 0,8206 | 0,8259 | 0,8232 | 1,253 | 67,03% | 28,57% | 75,42% | 0,6014 |
| `Wave-V2` Hungarian | 953 | uji | 135 | 0,8444 | **0,8331** | 0,8387 | 1,363 | 63,70% | 27,41% | 74,42% | 0,6034 |
| `V2-E-045` | 953 | validasi | 91 | 0,8044 | 0,8130 | 0,8087 | 1,253 | 67,03% | 28,57% | 70,04% | 0,5462 |
| `V2-E-045` | 953 | uji | 135 | 0,8092 | 0,7996 | 0,8043 | 1,393 | 61,48% | 25,93% | 71,11% | 0,5384 |
| Pipeline Panen | 953 | validasi | 91 | 0,8374 | 0,6934 | 0,7586 | 1,374 | 63,74% | 26,37% | 71,65% | 0,6683³ |
| Pipeline Panen | 953 | uji | 132 | 0,8538 | 0,6878 | 0,7619 | 1,402 | 56,82% | 22,73% | 71,61% | 0,6692³ |

³ Makro-F1 Pipeline Panen dihitung hanya pada klaster yang berpasangan. Nilai
ujung ke ujung yang setara dengan kolom ini pada baris lain adalah $0,5201$
(lihat bagian 6).

### 3.5 Selang kepercayaan 95% profil terkunci uji (*bootstrap* 2.000 ulangan)

| Besaran | 953, Hungarian *Anchor A* | 763-depth, GSP MILP |
|---|---|---|
| F1 fisik | $0,8387$ $[0,8174; 0,8587]$ | $0,8534$ $[0,8301; 0,8761]$ |
| Akurasi *matched-class* | $0,7442$ $[0,7112; 0,7735]$ | $0,8162$ $[0,7765; 0,8556]$ |
| Makro-F1 E2E | $0,6034$ $[0,5655; 0,6382]$ | $0,6519$ $[0,6046; 0,6918]$ |
| MAE cacah | $1,363$ $[1,163; 1,585]$ | $0,773$ $[0,609; 0,945]$ |
| Cacah $\pm 1$ | $0,6370$ $[0,5556; 0,7185]$ | $0,8545$ $[0,7818; 0,9182]$ |

### 3.6 Perbandingan Pipeline Panen (`AF-E-012`, `AF-E-013`)

Penyebut ujinya 132 pohon, bukan 135. Detektornya YOLO26m tunggal dengan daya
tangkap fisik $0,6878$, bukan ansambel WBF tiga detektor. Penautnya model tepi
HistGradientBoosting milik audit. Perbandingan langsung karena itu tidak setara.

| Besaran, uji 953 | Pipeline Panen | Profil terkunci `Wave-V2` |
|---|---:|---:|
| Cacah B1 siap panen, $\pm 1$ | **0,970** | tidak dilaporkan |
| Akurasi ordinal $\pm 1$ | **0,9946** | tidak dilaporkan |
| Akurasi dua kelas matang/belum | **0,8678** | tidak dilaporkan |
| Makro-F1 kelas empat | 0,6692³ | 0,6034 |
| F1 fisik | 0,7619 | **0,8387** |
| MAE cacah total | 1,402 | **1,363** |
| Cacah total $\pm 1$ | 0,5682 | **0,6370** |
| Akurasi kelas empat | 0,7161 | **0,7442** |

Ketiga keunggulan pertama berasal dari satu keputusan rancangan, yaitu penentuan
kelas pada tingkat tandan fisik melalui skor ordinal kontinu. Kesimpulan
`AF-E-012` menyarankan penggabungan penaut GSP proyek dengan tahap kelas ordinal
audit.

---

## 4. Pencacahan per kelas untuk tiap kohort

Penilaian akhir tugas `counting`, dari profil penaut terkunci `Wave-V2`, dengan
menjumlahkan seluruh pohon dalam satu partisi.

> [!WARNING]
> Arah definisi kelas B1–B4 belum diselaraskan. Pembuka
> [`02_klasifikasi_kematangan.md`](02_klasifikasi_kematangan.md) menempatkan B1
> sebagai mentah; kartu dataset `ULM-DS-Lab/SawitMVC-YOLO`,
> [`docs/DATASET.md`](../docs/DATASET.md), ukuran kotak median ($133/120/107/93$
> piksel pada 953), dan relief kedalaman ($+2,8$ cm untuk B1 berbanding $−5,1$ cm
> untuk B4) menempatkan B1 sebagai tandan matang siap panen. Seluruh baris
> `AF-E-###` memakai arah kartu dataset.

### 4.1 Korpus 953, partisi uji (Hungarian *Anchor A*, 135 pohon)

| Kelas | Total prediksi | Total acuan | Bias absolut | Bias relatif |
|---|---:|---:|---:|---:|
| B1 | 104 | 113 | −9 | −7,96% |
| B2 | 145 | 246 | −101 | −41,06% |
| B3 | 824 | 706 | +118 | +16,71% |
| B4 | 251 | 277 | −26 | −9,39% |
| **Total** | **1.324** | **1.342** | **−18** | **−1,34%** |

Makro-rerata nilai mutlak bias relatif $18,78\%$.

### 4.2 Korpus 953, partisi validasi (Hungarian *Anchor A*, 91 pohon)

| Kelas | Total prediksi | Total acuan | Bias absolut | Bias relatif |
|---|---:|---:|---:|---:|
| B1 | 84 | 86 | −2 | −2,33% |
| B2 | 112 | 186 | −74 | −39,78% |
| B3 | 560 | 476 | +84 | +17,65% |
| B4 | 186 | 188 | −2 | −1,06% |
| **Total** | **942** | **936** | **+6** | **+0,64%** |

Makro-rerata nilai mutlak bias relatif $15,21\%$.

### 4.3 Korpus 763-depth, partisi uji (GSP MILP, 110 pohon)

| Kelas | Total prediksi | Total acuan | Bias absolut | Bias relatif |
|---|---:|---:|---:|---:|
| B1 | 67 | 94 | −27 | −28,72% |
| B2 | 227 | 199 | +28 | +14,07% |
| B3 | 177 | 215 | −38 | −17,67% |
| B4 | 41 | 50 | −9 | −18,00% |
| **Total** | **512** | **558** | **−46** | **−8,24%** |

Makro-rerata nilai mutlak bias relatif $19,62\%$.

### 4.4 Korpus 763-depth, partisi validasi (GSP MILP, 117 pohon)

| Kelas | Total prediksi | Total acuan | Bias absolut | Bias relatif |
|---|---:|---:|---:|---:|
| B1 | 66 | 96 | −30 | −31,25% |
| B2 | 234 | 205 | +29 | +14,15% |
| B3 | 176 | 216 | −40 | −18,52% |
| B4 | 32 | 53 | −21 | −39,62% |
| **Total** | **508** | **570** | **−62** | **−10,88%** |

Makro-rerata nilai mutlak bias relatif $25,88\%$.

### 4.5 F1 ujung ke ujung per kelas, profil `Wave-V2`

| Korpus dan partisi | B1 | B2 | B3 | B4 | Makro |
|---|---:|---:|---:|---:|---:|
| 953, validasi | 0,7059 | 0,4698 | 0,6737 | 0,5561 | 0,6014 |
| 953, uji | 0,7465 | 0,4706 | 0,6850 | 0,5114 | 0,6034 |
| 763-depth, validasi | 0,7778 | 0,7244 | 0,7500 | 0,4706 | 0,6807 |
| 763-depth, uji | 0,7578 | 0,7230 | 0,7092 | 0,4176 | 0,6519 |

### 4.6 Catatan pembacaan

Pada korpus 953, B2 diestimasi kurang sekitar $40\%$ dan B3 diestimasi berlebih
sekitar $17\%$, konsisten di kedua partisi. Arah berlawanan dengan besaran
sebanding itu menunjukkan perpindahan massa prediksi dari B2 ke B3. Matriks
konfusi `AF-E-009` menguatkannya: batas B2 berbanding B3 menyumbang 195 galat,
sedangkan batas B1 berbanding B2 hanya 57. Tabel 4.5 menunjukkan hal yang sama
dari sisi lain, dengan F1 B2 pada korpus 953 hanya $0,47$.

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

## 5. Plafon dan target rekayasa

### 5.1 Vonis atas target (`AF-E-004`, `AF-E-005`, `AF-E-006`, `AF-E-008`)

| Target | Capaian kini | Plafon terukur | Vonis |
|---|---:|---:|---|
| $mAP50$ deteksi + klasifikasi empat kelas $\ge 0,85$ | 0,5970 | 0,6569 | **Tidak terjangkau** |
| $mAP50$ dua kelas siap panen berbanding belum $\ge 0,85$ | 0,7754 | 0,8766 | Terjangkau |
| Pencacahan total per pohon, tepat persis | 0,2741 | 0,2900 | **Tidak terjangkau** |
| Pencacahan siap panen per pohon, $\pm 1 \ge 95\%$ | 0,957–0,965 | 1,000 | **Tercapai** |

Capaian $0,5970$ berada pada $91\%$ dari plafon $0,6569$. Ruang perbaikan tersisa
untuk seluruh tumpukan detektor sekitar $6$ poin $mAP$, bukan $25$.

### 5.2 Akurasi kematangan yang dibutuhkan untuk mencapai target (`AF-E-005`)

| Akurasi kematangan per citra terpotong | $mAP50$ empat kelas |
|---:|---:|
| 0,661 (capaian kini) | 0,587 |
| 0,80 | 0,735 |
| 0,90 | **0,847** |
| 0,95 | 0,927 |

Target $mAP50 = 0,85$ menuntut akurasi kematangan sekitar $0,90$, yaitu dua puluh
poin di atas segala yang pernah dicapai repositori ini.

### 5.3 Plafon lapisan pencacahan dengan deteksi *oracle* (`AF-E-004`, uji 953, 136 pohon)

Seluruh tahap deteksi diganti kotak acuan, lalu Ridge memetakan cacah kotak per
kelas menuju cacah tandan unik per kelas.

| Besaran | MAE | Tepat | $\pm 1$ | Tampak tunggal |
|---|---:|---:|---:|---:|
| B1 | 0,101 | 0,899 | **1,000** | 11,6% |
| B2 | 0,239 | 0,768 | 0,993 | 23,4% |
| B3 | 0,638 | 0,428 | 0,942 | 22,7% |
| B4 | 0,268 | 0,739 | 0,993 | **40,5%** |
| Total per pohon | 1,058 | 0,290 | 0,754 | — |

Makro-MAE $0,312$, berdekatan dengan jalur *oracle* historis $0,275$–$0,277$ pada
[`docs/REKAP.md`](../docs/REKAP.md) §2. Faktor duplikasi per pohon $k = 1,905$
dengan simpangan baku $0,384$.

---

## 6. Kaveat

| Kaveat | Dampak terhadap pembacaan | Rujukan |
|---|---|---|
| Makro-F1 `AF-E-012` punya dua nilai yang sama-sama benar: $0,6692$ (klaster berpasangan saja) dan $0,5201$ (ujung ke ujung) | Hanya $0,5201$ yang setara dengan makro-F1 $0,6034$ profil terkunci | [`EVIDENCE.md`](../docs/research_2026-09-06/EVIDENCE.md) |
| `AF-E-010` melaporkan $45,3\%$ klaster melanggar kendala satu tandan satu sisi | Dikoreksi `AF-E-014`: hanya berlaku pada jalur sapuan; profil terkunci `max_size` $\le 3$ pelanggarannya $0,00\%$, 0 dari 630 konfigurasi berubah. `AF-E-016`: *Anchor A* (`max_size` 4) juga aman, 0 dari 135 pohon berbeda | `AF-E-014`, `AF-E-016` |
| Seluruh `AF-E` memakai detektor lebih kecil, penaut sendiri, atau arah taksonomi berlawanan | Tidak ada yang menggantikan angka terkunci; statusnya eksperimen tambahan | `metrics/07` |
| Irisan `tree_id` antara partisi latih `combined1716` dan kedua kumpulan uji lokal belum diaudit | Kemungkinan kebocoran partisi data (*data leakage*) belum terkuantifikasi | [`ANALISIS_PIPELINE_MENDALAM.md`](../docs/ANALISIS_PIPELINE_MENDALAM.md) |
| Partisi uji pernah dibaca pada iterasi historis | Angka uji adalah konfirmasi profil terkunci validasi, bukan partisi terisolasi yang belum tersentuh | `V2-E-045` |
| Plafon $0,6569$ bergantung pada mutu satu pengklasifikasi ($0,6635$ per citra terpotong) | Terkalibrasi terhadap pita $0,62$–$0,70$ repositori ini, bukan batas teoretis-informasi | `AF-E-005` |

---

## 7. Sumber angka

| Kolom | Artefak |
|---|---|
| `detection` | [`class_agnostic_metrics_audit_2026-09-03.json`](../results/class_agnostic_metrics_audit_2026-09-03.json), medan `rows[].ap50_agnostic` |
| `detection` setelah *re-ranker* | [`map_boost_artifacts/953`](../results/remote_eval_2026-08-28/map_boost_artifacts/953/results_test_locked.json), [`depth`](../results/remote_eval_2026-08-28/map_boost_artifacts/depth/results_test_locked.json), medan `test_metrics.agnostic.AP50` |
| `det + class` | [`combined1716`](../results/combined1716), [`new763`](../results/new763), medan `splits.test.mAP50`; [`perkelas_pycoco_v2repro.json`](../results/perkelas_pycoco_v2repro.json); [`extra_metrics_sesi2026-08.json`](../results/extra_metrics_sesi2026-08.json) |
| Pipeline `V2-E-045` | [`pipeline_combined1716_generalization_locked.json`](../results/remote_eval_2026-08-27/metrics/pipeline_combined1716_generalization_locked.json) |
| Pipeline `Wave-V2`, CI95, F1 per kelas | [`gsp_artifacts/953`](../results/remote_eval_2026-08-28/gsp_artifacts/953/results_test_locked.json), [`gsp_artifacts/depth`](../results/remote_eval_2026-08-28/gsp_artifacts/depth/results_test_locked.json), [`anchor_check.json`](../results/remote_eval_2026-08-28/validation_wave/reports/anchor_check.json) |
| Bias pencacahan per kelas | Diturunkan dari medan `metrics.classification.confusion_prediction_rows` pada tiga artefak `Wave-V2` di atas |
| Baris `AF-E` | [`detector_matrix.json`](../results/audit_forensik_2026-09-06/detector_matrix.json), [`panen/panen_final.json`](../results/audit_forensik_2026-09-06/panen/panen_final.json), [`experiments/AUDIT-FORENSIK-2026-09-06.md`](../experiments/AUDIT-FORENSIK-2026-09-06.md) |

Rujukan pelengkap: [`README.md`](README.md), tujuh berkas spesialisasi `01`–`07`,
dan [`docs/AUDIT-FORENSIK-2026-09-06.md`](../docs/AUDIT-FORENSIK-2026-09-06.md).
