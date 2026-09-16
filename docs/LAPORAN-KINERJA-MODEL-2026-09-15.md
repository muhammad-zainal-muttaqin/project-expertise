# Laporan Kinerja Model Deteksi Pencacahan dan Klasifikasi Tandan Kelapa Sawit

| Butir | Isi |
|---|---|
| Penelitian | Pengembangan Perangkat Mobile dengan Teknologi Depth Sensor untuk Penghitungan dan Klasifikasi Tandan Kelapa Sawit Berbasis Deep Learning |
| Penyusun | Muhammad Zainal Muttaqin |
| Tanggal penyusunan | 15 September 2026 |
| Revisi laporan | 16 September 2026 |
| Cakupan | Hasil deteksi, pencocokan tandan antarfoto, klasifikasi, pencacahan, dan pengujian manfaat kedalaman dalam repositori `project-expertise` sampai 10 September 2026 |

Laporan ini menilai kemampuan model menemukan tandan pada foto, mengenali tandan yang sama dari beberapa sisi, menentukan kelas kematangan, dan menghitung jumlahnya. Konfigurasi acuan mencapai *F1* deteksi tandan 0,8387 pada dataset RGB 953 dan 0,8534 pada dataset Depth 763. Akurasi kelas pada tandan yang berhasil dicocokkan mencapai 74,42% dan 81,62%, sedangkan makro-*F1* seluruh sistem mencapai 0,6034 dan 0,6519. Ketepatan jumlah per kelas masih menjadi keterbatasan utama. Manfaat tambahan sensor kedalaman belum terkonfirmasi secara konsisten pada pengujian yang tersedia.

Angka dalam laporan berasal dari berkas hasil eksperimen yang tersimpan di repositori. Penyusunan laporan ini tidak mencakup pelatihan ulang atau pengujian ulang model. Pemeriksaan aritmetika dan sumber gambar tersedia pada [berkas verifikasi](assets/monev-2026-09-15/verifikasi_angka.json). Sebutan "terbaik" mengacu pada nilai tertinggi yang tercatat untuk tugas dan kelompok data yang sebanding. Nilai tersebut belum tentu lebih tinggi secara signifikan daripada seluruh konfigurasi lain dan belum membuktikan kesiapan penggunaan di lapangan.

Dataset **Depth 763** berisi data dari kamera yang merekam citra warna dan kedalaman. Namun, detektor `combined1716` yang digunakan dalam konfigurasi acuan hanya menerima citra warna RGB. Karena itu, hasil pada dataset ini tidak dengan sendirinya menunjukkan manfaat sensor kedalaman. Percobaan yang menggunakan kanal sensor kedalaman dibahas terpisah pada §6. Hasil eksperimen terdahulu memiliki keterbatasan karena penggunaan ulang data uji, catatan pelatihan yang belum lengkap, dan kendala pengulangan eksperimen (§10).

---

## Ringkasan Kinerja

| Pertanyaan | Jawaban | Bagian |
|---|---|---|
| Apakah sistem sudah berjalan? | Sudah; diuji pada 135 pohon (RGB 953) dan 110 pohon (Depth 763). | §3–4 |
| Berapa kinerja terbaik? | F1 deteksi tandan 0,8387–0,8534; akurasi kelas pada tandan yang ditemukan 74,42–81,62%; galat jumlah ≤ 1 pada 63,70–85,45% pohon; B1–B4 tepat sekaligus pada 5,19–27,27% pohon. | §4 |
| Apa kelemahan utama? | Kesalahan utama berupa salah klasifikasi pada tingkat kematangan yang berdekatan dan tandan yang tidak terdeteksi. Pada 953, 41,06% tandan acuan B2 diprediksi B3. Secara terpisah, bias jumlah B2 adalah −41,06%, sementara bias total −1,34%. Kesamaan kedua persentase tidak berarti keduanya mengukur hal yang sama. | §5 |
| Apakah sensor kedalaman bermanfaat? | Manfaatnya belum konsisten. Penambahan kedalaman sebagai masukan model belum meningkatkan kinerja secara signifikan. Penggabungan prediksi model RGB dan RGB+kedalaman meningkatkan skor pada data validasi, tetapi belum dikonfirmasi pada data baru. | §6 |
| Apakah sudah mencapai batas? | Belum terbukti. Dari prediksi yang sama, simulasi yang memakai label acuan untuk memilih prediksi benar mencapai mAP50 0,9752; metode untuk memilih prediksi benar secara otomatis belum tersedia. | §7–8 |
| Apa yang sudah tercapai? | Pencacahan B1 meleset paling banyak satu tandan pada 95,7–97,0% pohon dan mAP50 dua kelas 0,7754 (uji 953). Target empat kelas dan konsistensi pencacahan total > 95% belum tercapai. | §8–9 |
| Apakah hasil membuktikan keandalan lapangan? | Belum. Hasil mencakup populasi uji historis dan belum membuktikan konsistensi pada periode pengambilan data baru dengan kriteria keberhasilan penggunaan lapangan. | §9–11 |

---

## 1. Tahapan Sistem dan Arti Ukuran Kinerja

![Rangkaian sistem](assets/monev-2026-09-15/00_rangkaian_sistem.png)

Sistem mengolah empat foto dari sisi berbeda pada setiap pohon. Sistem kemudian mencocokkan hasil deteksi yang merujuk pada tandan yang sama agar satu tandan tidak dihitung berulang. Beberapa eksperimen menambahkan kedalaman sebagai masukan detektor, informasi untuk klasifikasi dan pemeringkatan deteksi, atau koordinat untuk pencocokan antarfoto. Diagram menunjukkan tahapan sistem; penggunaan kedalaman berbeda pada setiap konfigurasi.

| Metrik | Mengukur | Tingkat |
|---|---|---|
| AP50 deteksi tanpa kelas | Posisi tandan, tanpa kelas | Citra |
| mAP50 empat kelas | Posisi dan kelas B1–B4 sekaligus | Citra |
| *F1* deteksi tandan | Keseimbangan antara ketepatan hasil deteksi dan proporsi tandan acuan yang ditemukan setelah pencocokan antarfoto | Tandan, dikelompokkan per pohon |
| Akurasi kelas pada tandan yang ditemukan | Proporsi kelas yang benar di antara tandan hasil deteksi yang berhasil dicocokkan dengan tandan acuan | Tandan |
| Makro-F1 seluruh sistem | Rerata F1 B1–B4, termasuk tandan terlewat dan prediksi berlebih | Tandan |
| MAE jumlah | Selisih mutlak rata-rata jumlah total per pohon | Pohon |
| Galat ≤ 1 / tepat / B1–B4 tepat | Proporsi pohon dengan jumlah total meleset paling banyak satu, tepat, atau tepat pada keempat kelas | Pohon |
| Bias per kelas | Selisih relatif antara jumlah prediksi dan jumlah acuan untuk setiap kelas | Seluruh pohon yang dievaluasi |

*AP50* dan *mAP50* mengevaluasi pengurutan skor deteksi pada ambang tumpang tindih kotak (*IoU*) 0,50. Keduanya bukan persentase tandan benar pada satu ambang keyakinan. Akurasi kelas pada tandan yang ditemukan mengevaluasi pasangan yang berhasil dicocokkan; objek terlewat dan prediksi tanpa pasangan tidak masuk pembaginya. Makro-*F1* seluruh sistem menyertakan kesalahan tersebut.

Bias kelas dihitung sebagai `(jumlah prediksi − jumlah acuan) / jumlah acuan` untuk seluruh pohon dalam satu partisi. Rerata mutlak bias per kelas dihitung dengan mengabaikan tanda positif atau negatif pada bias B1–B4, kemudian merata-ratakan keempat nilainya. Nilai ini berbeda dari MAE jumlah total per pohon. Kelebihan hitungan pada satu kelas dapat menutupi kekurangan pada kelas lain saat dijumlahkan. MAE tetap menghitung besar kesalahan pada setiap pohon tanpa saling menghapus kesalahan positif dan negatif.

Label *test-locked* berarti pengaturan model telah ditetapkan sebelum pengujian pada eksperimen tersebut. Meskipun demikian, data uji yang sama mungkin telah digunakan dalam eksperimen sebelumnya. Selang kepercayaan menunjukkan rentang ketidakpastian berdasarkan data dan metode pengambilan sampel ulang yang digunakan. Rentang ini belum mencakup risiko penurunan kinerja pada lokasi atau waktu pengambilan data yang berbeda.

---

## 2. Dataset dan Populasi Pengujian

| Dataset | Kamera | Akuisisi | Pohon | Citra | Kotak | Latih/val/uji (pohon) |
|---|---|---|---:|---:|---:|---|
| SawitMVC-YOLO (953) | Ponsel, RGB 960 × 1.280 | 30 April–16 Mei 2026 | 953 | 3.992 | 18.540 | 716/96/141 |
| SawitMVC-Depth (352) | Orbbec, RGB 1.280 × 800 + kedalaman 848 × 480 | 28–29 Juli 2026 | 352 | 1.408 | 2.299 | 245/52/55 |
| SawitMVC-Depth-YOLO v2.0.0 (763) | Orbbec, pengambilan data Juli (352) dan Agustus (411) | Juli–Agustus 2026 | 763 | 3.052 | 5.872 | 536/117/110 |
| Combined-1716 | Gabungan 953 dan 763 | Mei–Agustus 2026 | 1.364 unik | 7.044 | 24.412 | 1.005/160/199 |

Perbandingan hasil perlu memperhitungkan tiga perbedaan data berikut.

1. **Komposisi kelas berbeda.** B3 mencakup 52,3% kotak pada 953, sedangkan Depth didominasi B2 (35,0%) dan B3 (38,3%) dengan B4 hanya 7,3%.
2. **Kepadatan berbeda.** Rerata tandan per pohon uji adalah 9,94 (953) dan 5,08 (763). Penautan pada 953 lebih sulit: sekitar 235 pasangan kandidat per pohon (4% benar) dibanding 28 pasangan (21% benar) pada 352 (`PT-E-011`).
3. **Jumlah label, komposisi kelas, dan ukuran kotak pembatas berbeda antarperiode pengambilan data.** Audit pada 352 pohon yang difoto Mei dan Juli mencatat perubahan jumlah tandan per pohon dari 9,89 menjadi 3,99, proporsi sisi tanpa kotak dari 1,1% menjadi 14,2%, kenaikan B1 sekitar 66%, dan penurunan B4 sekitar 85%. Angka ini merupakan bukti perbedaan data. Pengaruh pedoman anotasi, kelengkapan label, perubahan biologis, kamera, dan cara pengambilan gambar belum dapat ditentukan secara terpisah. Karena itu, selisih hasil antardataset belum dapat dianggap sebagai akibat penggunaan sensor kedalaman. [Audit dan koreksi interpretasi](research_2026-09-06/report-source.md).

Dataset Combined-1716 memuat 1.716 entri pohon dari dua sumber, dengan 352 identitas yang berulang antarsesi. Jumlah pohon fisik unik adalah 953 + 763 − 352 = **1.364**. Pada pembagian dataset gabungan, rekaman dengan identitas pohon yang sama dikelompokkan bersama. Namun, saat model diuji kembali pada dataset asal, daftar data pelatihannya tetap perlu diperiksa untuk memastikan pohon uji tidak pernah digunakan dalam pelatihan.

Evaluasi detektor RGB 953 mencakup 141 pohon dan 588 citra. Konfigurasi empat sisi memakai 135 pohon, sehingga populasinya tidak identik dengan evaluasi detektor. Pipeline Panen historis memakai 132 pohon; audit menemukan tiga pohon kosong yang tidak disertakan. Evaluasi Depth 763 mencakup 110 pohon dan 440 citra. Dataset 352 adalah rilis lebih awal dengan 55 pohon uji, bukan nama lain untuk partisi uji 763. [Spesifikasi dataset 763](NEW763_BASELINE.md), [komposisi gabungan](EDA-COMBINED1716.md), [audit populasi](research_2026-09-06/report-source.md).

---

## 3. Komponen Model dan Status Pengukuran

| Komponen | Status | Bukti |
|---|---|---|
| Data RGB dan kedalaman Orbbec + kalibrasi | Sudah | Pada diagnosis dataset 352, 95,1% piksel di dalam kotak tandan memiliki nilai kedalaman yang valid ([DIAGNOSIS-DEPTH.md](DIAGNOSIS-DEPTH.md)) |
| Reproyeksi kedalaman ke citra RGB | Sudah | Pergeseran median 29 piksel dikoreksi ([NEW763_RGBD4_RESULTS.md](NEW763_RGBD4_RESULTS.md)) |
| Tiga detektor | Sudah | RF-DETR-L mempunyai *mAP50* tertinggi pada tabel perbandingan utama; perbandingan berpasangan pada dataset 763 dan gabungan dilaporkan signifikan (`V2-E-038`) |
| Detektor RGB+D dan fusi prediksi | Sudah | Manfaat belum konsisten (§6) |
| WBF + *re-ranker* | Sudah | +0,0108 mAP50 signifikan (953); −0,0139 tidak signifikan (763) ([CI_SUMMARY.md](../results/remote_eval_2026-08-28/ci_artifacts/CI_SUMMARY.md)) |
| Pencocokan tandan antarfoto | Sudah | *Prior* arah putar: F1 0,398 → 0,649 (`PT-E-008`) |
| Kelas per tandan | Sudah, belum memadai | Akurasi pada tandan yang ditemukan 74,42–81,62% |
| Pencacahan per kelas | Sudah, bias besar | Rerata mutlak bias 18,78–19,62% |
| Evaluasi uji + selang kepercayaan | Sudah | Bootstrap 2.000 ulangan ([recap.md §5](../metrics/recap.md)) |
| Pemeriksaan kualitas foto | Belum | [HANDOFF.md](../HANDOFF.md) |
| Pelaporan tingkat keyakinan prediksi | Belum final | [HANDOFF.md](../HANDOFF.md) |
| Uji pada pohon baru | Belum | Partisi uji sudah dibaca berulang (§10) |
| Pengulangan lengkap konfigurasi terbaik | Sebagian | Solver GSP MILP tidak ditemukan; `V2-E-046` tidak tereproduksi |
| Waktu pemrosesan pada perangkat sasaran | Belum terukur dalam sumber yang diperiksa | Kinerja komputasi pada perangkat mobile tidak dinilai dalam laporan ini |

---

## 4. Kinerja Detektor dan Sistem Secara Keseluruhan

### 4.1 Kemampuan deteksi pada setiap citra

| Kemampuan | RGB 953 (588 citra uji) | Depth 763 (440 citra uji) | Combined-1716 (1.052 citra uji) |
|---|---|---|---|
| AP50 deteksi tanpa kelas | 0,8419 [0,8270; 0,8595], WBF + *re-ranker* | 0,8783 [0,8541; 0,9009], WBF + *re-ranker* | 0,8104, WBF |
| mAP50, model tunggal | 0,6012, RF-DETR-L | 0,6711, RF-DETR-L `combined1716` | 0,5960, RF-DETR-L |
| mAP50, ansambel | 0,5970 [0,5751; 0,6239], WBF + *re-ranker* | 0,6691 [0,6334; 0,7108], WBF | 0,5538, WBF |

| AP50 per kelas | B1 | B2 | B3 | B4 | mAP50 |
|---|---:|---:|---:|---:|---:|
| YOLO26l, uji 953 ([boot_sel6_vs_sel5.json](../results/boot_sel6_vs_sel5.json)) | 0,7708 | 0,4479 | 0,6051 | 0,3506 | 0,5436 |
| YOLO26s, uji 763 ([detector_matrix.json](../results/audit_forensik_2026-09-06/detector_matrix.json)) | 0,7268 | 0,6057 | 0,6480 | 0,2476 | 0,5570 |

Pada uji RGB 953 dan Depth 763, skor lokalisasi tertinggi masing-masing 0,8419 dan 0,8783, sementara *mAP50* empat kelas tertinggi 0,6012 dan 0,6711. Nilai terbaik pada kedua tugas berasal dari konfigurasi yang berbeda. Selisih kedua skor tersebut tidak dapat digunakan untuk menghitung besarnya kesalahan klasifikasi pada satu model. Pola kesalahan kelas yang lebih spesifik disajikan melalui matriks kesalahan klasifikasi konfigurasi tetap pada §5.

Angka RF-DETR-L 953 sebesar 0,6012 merupakan hasil historis. Bobot aslinya tidak tersedia dalam audit reproduksi; pelatihan ulang menghasilkan 0,5965. Nilai historis dipertahankan sebagai rekaman, tanpa disamakan dengan hasil bobot pengganti. [Hasil pelatihan ulang](../results/rfdetr_l_v2repro_953_retrain_2026-09-07.json).

### 4.2 Perbandingan tiga arsitektur pada dataset masing-masing

| Dataset dan masukan | YOLO26l | RT-DETR-L | RF-DETR-L |
|---|---:|---:|---:|
| RGB 953, hasil uji historis | 0,5435 | 0,5781 | 0,6012 |
| RGB 352, hasil uji historis | 0,3606 | 0,4343 | 0,4544 |
| RGB+D 352, representasi invers | 0,3919 | 0,3877 | 0,4186 |
| RGB 763, pelatihan pada dataset 763 | 0,5163 | 0,5580 | 0,6129 |
| RGB Combined-1716, uji gabungan | 0,5389 | 0,5745 | 0,5960 |

Seluruh nilai merupakan *mAP50* empat kelas. Perbandingan arsitektur dibaca dalam satu baris; perbedaan antarbaris mencakup perbedaan pohon yang dievaluasi, jumlah data, atau jenis masukan model. Sumber daya dan pengaturan pelatihan tidak sama pada seluruh arsitektur. Sumber: [hasil awal 953](../results/perkelas_pycoco_v2repro.json), [hasil awal 352](LAPORAN-AKHIR.md), [hasil 763](NEW763_BASELINE.md), dan [rekap gabungan](../metrics/recap.md).

Pada dataset 763, RF-DETR-L menghasilkan *mAP50* 0,4460 pada DAMIMAS (52 pohon), 0,5182 pada MARIHAT (11 pohon), dan 0,6369 pada TOPAZ (47 pohon). RT-DETR-L mencatat 0,5380 pada MARIHAT. RF-DETR-L memperoleh skor tertinggi pada keseluruhan data, tetapi RT-DETR-L lebih tinggi pada MARIHAT. Selang kepercayaan untuk selisih skor per kelompok pengambilan data belum tersedia. [Hasil per kelompok pengambilan data](NEW763_BASELINE.md).

### 4.3 Kinerja seluruh sistem pada setiap pohon

Konfigurasi acuan tanggal 28 Agustus 2026 menggunakan tiga detektor `combined1716`. Metode WBF menggabungkan kotak pembatas hasil deteksi. Metode Hungarian Anchor A pada dataset 953 dan GSP MILP pada dataset 763 kemudian mencocokkan tandan antarfoto. Sistem menentukan kelas dan jumlah tandan dari kelompok hasil pencocokan tersebut. Tabel berikut melaporkan hasil akhir konfigurasi ini.

| Metrik | RGB 953 (135 pohon) | Depth 763 (110 pohon) |
|---|---|---|
| F1 deteksi tandan | 0,8387 [0,8174; 0,8587] | 0,8534 [0,8301; 0,8761] |
| Presisi / daya tangkap | 84,44% / 83,31% | 89,26% / 81,75% |
| Akurasi kelas pada tandan yang ditemukan | 74,42% [71,12%; 77,35%], *n* = 1.118 | 81,62% [77,65%; 85,56%], *n* = 457 |
| Makro-F1 seluruh sistem | 0,6034 [0,5655; 0,6382] | 0,6519 [0,6046; 0,6918] |
| F1 B1 / B2 / B3 / B4 | 0,7465 / 0,4706 / 0,6850 / 0,5114 | 0,7578 / 0,7230 / 0,7092 / 0,4176 |
| MAE jumlah | 1,363 [1,163; 1,585] | 0,773 [0,609; 0,945] |
| Jumlah tepat | 27,41% (37 pohon) | 44,55% (49 pohon) |
| Galat ≤ 1 | 63,70% [55,56%; 71,85%] (86 pohon) | 85,45% [78,18%; 91,82%] (94 pohon) |
| B1–B4 tepat sekaligus | 5,19% (7 pohon) | 27,27% (30 pohon) |
| Rerata mutlak bias per kelas | 18,78% | 19,62% |
| Tandan acuan per pohon | 9,94 | 5,08 |

Selang kepercayaan 95% pada konfigurasi ini dihitung dengan bootstrap tingkat pohon sebanyak 2.000 ulangan ([953](../results/remote_eval_2026-08-28/gsp_artifacts/953/results_test_locked.json), [Depth](../results/remote_eval_2026-08-28/gsp_artifacts/depth/results_test_locked.json), [recap.md §5](../metrics/recap.md)).

![Ketepatan pencacahan](assets/monev-2026-09-15/01_ketepatan_pencacahan.png)

*Gambar 1. Ketepatan pencacahan per pohon.*

Perbedaan hasil pada kedua dataset belum menunjukkan pengaruh sensor kedalaman. Keduanya menggunakan detektor RGB, sedangkan rata-rata jumlah tandan per pohon pada dataset 763 hanya sekitar separuh jumlah pada dataset 953.

### 4.4 Konfigurasi alternatif

| Konfigurasi | Dataset | F1 deteksi tandan | Akurasi | MAE | ≤ 1 | Bias kelas |
|---|---|---:|---:|---:|---:|---:|
| Hungarian Anchor A (acuan) | 953 | **0,8387** | **74,4%** | **1,363** | **63,70%** | 18,78% |
| *Prior* rotasi + Ridge (`V2-E-045`) | 953 | 0,8043 | 71,1% | 1,393 | 61,48% | **16,82%** |
| Pipeline Panen, 132 pohon (`AF-E-012`) | 953 | 0,7619 | 71,6% | 1,402 | 56,82% | 23,22% |
| GSP MILP (acuan) | 763 | **0,8534** | **81,6%** | **0,773** | **85,45%** | 19,62% |
| *Prior* rotasi + Ridge (`V2-E-045`) | 763 | 0,8069 | 80,3% | 0,891 | 80,91% | **13,93%** |

Konfigurasi acuan mempunyai skor *F1* deteksi tandan yang lebih tinggi dan MAE yang lebih rendah daripada konfigurasi lain pada tabel. Namun, hasil penghitungan ulang `V2-E-045` mempunyai bias jumlah per kelas yang lebih kecil. Pipeline Panen menggunakan jumlah pohon evaluasi yang berbeda. Sebagian nilai bias juga sedikit berbeda dari catatan awal karena dihitung ulang. Oleh karena itu, tabel ini belum membuktikan keunggulan satu konfigurasi pada semua ukuran kinerja. [Pemeriksaan ulang bias jumlah](../results/class_bias_v2e045_2026-09-07.json).

Pengujian berpasangan tingkat pohon memberikan bukti peningkatan yang lebih spesifik:

| Perbandingan terhadap `V2-E-045` | Perubahan yang menguntungkan | Selang kepercayaan 95% | Kesimpulan |
|---|---:|---|---|
| *F1* deteksi tandan, 953 | +0,0344 | [+0,0209; +0,0477] | Signifikan |
| Pengurangan MAE, 953 | +0,0296 tandan | [−0,0148; +0,0815] | Belum signifikan |
| *F1* deteksi tandan, Depth | +0,0465 | [+0,0257; +0,0690] | Signifikan |
| Pengurangan MAE, Depth | +0,1182 tandan | [−0,0364; +0,2636] | Belum signifikan |
| Proporsi jumlah tepat, Depth | +10,91 poin persentase | [+0,89; +21,82] poin persentase | Signifikan |

Analisis perbedaan menggunakan 5.000 pengambilan sampel ulang. Pada setiap ulangan, kedua konfigurasi dinilai menggunakan sampel pohon yang sama. Analisis ini berbeda dari penghitungan selang kepercayaan masing-masing konfigurasi dengan 2.000 ulangan pada tabel sebelumnya. Selang kepercayaan 95% untuk penurunan MAE mencakup nilai nol, sehingga penurunan tersebut belum signifikan secara statistik. [Hasil bootstrap berpasangan](../results/remote_eval_2026-08-28/ci_artifacts/e2e_paired_test.json).

### 4.5 Hasil kandidat lain dan keterbatasan pengujiannya

| Kandidat | Hasil | Keterbatasan |
|---|---|---|
| *Stacking* DINOv2-Large, 953 (`V2-E-046`) | Akurasi 76,8% (validasi) | Pengulangan menghasilkan akurasi 71,64% |
| Komposisi lintas lapis, 763 (`V2-E-047`) | Akurasi 85,0% (validasi) | Selisih tidak signifikan; kode penyelesaian optimasi tidak ditemukan |
| *Greedy linker* (`V2-E-043`) | ≤ 1: 83,64% (763), 54,07% (953) | Parameter dipilih pada data uji |
| Pipeline Panen (`AF-E-012`) | Makro-F1 0,6692 | Hanya tandan yang ditemukan; seluruh sistem 0,5201 |

---

## 5. Kesalahan dalam Deteksi Klasifikasi dan Pencacahan

### 5.1 Tandan yang ditemukan terlewat dan salah diklasifikasikan

![Jumlah tandan yang ditemukan dan diklasifikasikan dengan benar](assets/monev-2026-09-15/03_alur_kehilangan_tandan.png)

*Gambar 2. Jumlah tandan per tahap terhadap tandan acuan.*

- Pada **dataset 953**, sistem menemukan 1.118 dari 1.342 tandan acuan (83,3%). Dari seluruh tandan acuan, 832 tandan ditemukan sekaligus diklasifikasikan dengan benar (62,0%). Sistem menghasilkan 206 prediksi tambahan yang tidak cocok dengan tandan acuan. Jumlah tandan yang salah diklasifikasikan (286) lebih banyak daripada tandan yang terlewat (224).
- Pada **dataset 763**, sistem menemukan 457 dari 559 tandan acuan (81,8%). Dari seluruh tandan acuan, 373 tandan ditemukan sekaligus diklasifikasikan dengan benar (66,7%). Sistem menghasilkan 55 prediksi tambahan yang tidak cocok dengan tandan acuan. Jumlah tandan yang terlewat (102) sedikit lebih banyak daripada tandan yang salah diklasifikasikan (84).

### 5.2 Kesalahan penentuan kelas kematangan

![Matriks kesalahan klasifikasi per tandan](assets/monev-2026-09-15/04_matriks_konfusi.png)

*Gambar 3. Matriks kesalahan klasifikasi per tandan acuan.*

1. **Pada dataset 953, ketepatan klasifikasi B2 dan B4 paling rendah.** Model memberikan kelas yang benar pada 37,40% tandan B2 dan 48,74% tandan B4. Model salah mengklasifikasikan 41,06% tandan B2 dan 28,52% tandan B4 sebagai B3.
2. **Pada dataset 763, ketepatan klasifikasi B4 paling rendah.** Model memberikan kelas yang benar pada 38,00% tandan B4, salah mengklasifikasikan 28,00% sebagai B3, dan melewatkan 32,00%. Untuk B1, model memberikan kelas yang benar pada 64,89% tandan dan salah mengklasifikasikan 28,72% sebagai B2.
3. **Kesalahan terutama terjadi pada tingkat kematangan yang berdekatan.** Kesalahan antara B1 dan B2, B2 dan B3, serta B3 dan B4 mencakup 99,3% kesalahan kelas pada dataset 953 dan 96,4% pada dataset 763.
4. **Pada dataset 953, kesalahan antara B3 dan B4 paling banyak.** Jika kedua arah kesalahan dijumlahkan, terdapat 123 kesalahan B2↔B3, 126 kesalahan B3↔B4, dan 35 kesalahan B1↔B2. Angka 195 dan 57 pada `AF-E-009` berasal dari eksperimen lain dan tidak digunakan untuk menjelaskan matriks konfigurasi ini.
5. **Konsistensi penilaian antarpetugas anotasi belum diukur.** Petugas mencatat satu kelas untuk setiap tandan, kemudian menyalin kelas tersebut ke seluruh foto yang menampilkan tandan yang sama (`AF-E-002`).

### 5.3 Jumlah total yang tepat belum menjamin ketepatan setiap kelas

![Bias jumlah per kelas](assets/monev-2026-09-15/02_bias_kelas_rgb.png)

*Gambar 4. Bias jumlah per kelas, RGB 953.*

| Kelas | 953: prediksi / acuan | 953: bias | 763: prediksi / acuan | 763: bias |
|---|---:|---:|---:|---:|
| B1 | 104 / 113 | −7,96% | 67 / 94 | −28,72% |
| B2 | 145 / 246 | −41,06% | 227 / 199 | +14,07% |
| B3 | 824 / 706 | +16,71% | 177 / 215 | −17,67% |
| B4 | 251 / 277 | −9,39% | 41 / 50 | −18,00% |
| **Total** | **1.324 / 1.342** | **−1,34%** | **512 / 558** | **−8,24%** |

Pada dataset 763, jumlah tandan acuan dalam matriks kesalahan klasifikasi adalah 558. Jumlah ini lebih kecil satu tandan daripada 559 yang digunakan dalam evaluasi deteksi tandan. Penyebab perbedaan tersebut belum diketahui.

Jumlah total yang tepat belum menjamin bahwa sistem menghitung tandan yang benar. Pada 24 dari 37 pohon dengan jumlah tepat dalam dataset 953 dan 17 dari 49 pohon dalam dataset 763, tandan yang terlewat tergantikan oleh prediksi berlebih sehingga jumlah total tampak tepat. Galat ≥ 2 terjadi pada 49 dari 135 pohon (953) dan 16 dari 110 pohon (763). Pada dataset 763, sistem menghitung terlalu sedikit tandan pada 46 pohon dan terlalu banyak tandan pada 15 pohon ([`count_error_cancellation.json`](../results/audit_2026-09-06/count_error_cancellation.json)).

---

## 6. Kontribusi Sensor Kedalaman

![Pengaruh penambahan kedalaman terhadap kinerja](assets/monev-2026-09-15/05_bukti_depth.png)

*Gambar 5. Selisih berpasangan terhadap kontrol. Titik terisi menandakan selang kepercayaan 95% tidak mencakup nol.*

### 6.1 Rancangan dan hasil pengujian

RGB+D berarti citra warna RGB ditambah data kedalaman; RGB+D4 menggunakan empat kanal masukan. Kedalaman dapat berupa nilai jarak atau peta tepi perubahan kedalaman (*edge*). Pada penggabungan prediksi, keluaran model RGB dan RGB+D disatukan menggunakan WBF atau NMS, yaitu metode untuk menggabungkan atau menghapus deteksi yang bertumpang tindih. Simbol Δ pada tabel menunjukkan selisih skor terhadap model pembanding.

| Uji | Rancangan | Hasil | Kesimpulan |
|---|---|---|---|
| Lokalisasi, 352 (`V2-E-024`) | Detektor satu kelas, RGB+D `edge` terhadap RGB | AP50 0,7636 terhadap 0,7358; Δ = +0,0278 [−0,0121; +0,0648] | Belum signifikan |
| mAP50, 352 (`V2-E-011`) | YOLO26l RGB+D `edge` terhadap RGB | 0,4270 terhadap 0,3677; Δ = +0,0593 [−0,0013; +0,1168] | Belum signifikan |
| Kedalaman sebagai masukan keempat, 763 (validasi) | Tiga arsitektur, 468 citra | Δ: YOLO26l +0,0002; RT-DETR-L +0,0063; RF-DETR-L −0,0112 | Tidak signifikan |
| Penggabungan prediksi RGB dan RGB+D4, 763 (validasi) | Union-WBF (YOLO26l), union-NMS (RT-DETR-L) | Δ = +0,0379 [+0,0161; +0,0591]; +0,0285 [+0,0092; +0,0472] | Signifikan; pengaturan dipilih pada validasi |
| Fitur kematangan, 352 (`V2-E-016`) | RGB terhadap RGB + statistik kedalaman; cabang CNN diuji terpisah | Akurasi pengklasifikasi pembanding: 0,6415 dan 0,6415. Rerata tiga pelatihan dengan CNN: 0,6309 dan 0,6106. Kedalaman saja: 0,3756 | Belum ada peningkatan pada representasi yang diuji |
| Pencocokan tandan dengan koordinat 3D, 352 (`PT-E-013`) | Kedalaman + arah putar | AUC 0,45–0,51 | Mendekati tebakan acak |
| Depth monokular, 953 (`V2-E-029`) | RGB + estimasi kedalaman | Δ = −0,0476 [−0,0671; −0,0274] | Turun signifikan |
| Gabungan kedalaman sensor dan estimasi monokular, 352 (`V2-E-031`) | Lima kanal terhadap RGB+D `edge` | Δ = −0,0504 [−0,1038; −0,0015] | Turun signifikan |

### 6.2 Mutu data kedalaman dan keterbatasan pengukurannya

1. **Bentuk permukaan yang terukur berkaitan dengan kelas kematangan.** Selisih kedalaman tandan terhadap sekitarnya berubah secara berurutan menurut kelas: +2,8 cm (B1) sampai −5,1 cm (B4); Kruskal–Wallis *H* = 99,8; *p* = 1,7 × 10⁻²¹.
2. **Ketelitian nilai kedalaman per piksel terbatas.** Pada jarak 2,49 m, perubahan satu tingkat nilai kedalaman dalam format uint8 setara dengan 2,9 cm. Derau sensor berada di sekitar 2,5 cm, sedangkan rasio sinyal terhadap derau per piksel sekitar 0,3.
3. **Persentase piksel valid dihitung pada wilayah yang berbeda.** Eksperimen RGB+D4 dataset 763 melaporkan piksel kedalaman valid sekitar 28,6–28,8% dari seluruh piksel citra warna. Diagnosis dataset 352 melaporkan 95,1% piksel valid di dalam kotak objek. Karena wilayah penghitungan dan datasetnya berbeda, kedua persentase tersebut tidak dapat dibandingkan langsung untuk menilai mutu sensor. [Protokol 763](NEW763_RGBD4_RESULTS.md), [diagnosis 352](DIAGNOSIS-DEPTH.md).
4. **Belum terbukti bahwa informasi kedalaman seluruhnya sudah tersedia pada citra RGB.** Penambahan ringkasan statistik kedalaman menghasilkan akurasi yang sama dengan RGB saja pada pengklasifikasi yang diuji. Uji itu tidak membuktikan bahwa seluruh informasi kedalaman sudah terkandung dalam RGB atau tidak dapat dimanfaatkan oleh metode lain.

### 6.3 Kesimpulan pengujian sensor kedalaman

Bukti yang tersedia belum mendukung klaim peningkatan kinerja yang konsisten akibat penambahan sensor kedalaman. Hasilnya dapat dirangkum sebagai berikut:

- Data kedalaman telah diselaraskan dengan citra RGB dan berhasil digunakan sebagai masukan oleh tiga detektor.
- Perbedaan bentuk permukaan tandan yang diukur dari data kedalaman berkaitan secara signifikan dengan kelas kematangan.
- Fusi prediksi RGB dan RGB+D menghasilkan selisih positif pada validasi; konfirmasi pada data baru belum tersedia.
- Penambahan depth monokular pada dua konfigurasi yang disebut menghasilkan penurunan performa signifikan; hasil tersebut tidak membuktikan ketidakmampuan seluruh metode monokular untuk setiap tugas.

Uji 352 memuat 410 kotak pada 220 citra dari 55 pohon. Selang *mAP50* pada salah satu konfigurasi mempunyai lebar sekitar 0,12. Pengambilan sampel ulang pada eksperimen awal dilakukan per citra. Cara ini belum sepenuhnya memperhitungkan bahwa beberapa foto berasal dari pohon yang sama dan saling berkaitan. Laporan lama memperkirakan kebutuhan sekitar 4.000 kotak anotasi untuk mendeteksi selisih kinerja sebesar 0,03. Perkiraan tersebut bergantung pada asumsi perhitungan dan belum menjamin peluang 80% untuk mendeteksi peningkatan pada semua rancangan pengujian.

Pada fusi prediksi 763, selisih estimasi titik YOLO26l adalah +0,0384 dan RT-DETR-L +0,0286. Nilai +0,0379 dan +0,0285 pada tabel merupakan rerata selisih hasil bootstrap. Selang kepercayaan 95% yang dilaporkan tidak mencakup nilai nol. Namun, pengaturan penggabungan prediksi dipilih menggunakan data validasi yang sama, sehingga hasilnya masih perlu dikonfirmasi pada data baru. Perbandingan dengan gabungan model RGB yang memakai sumber daya setara juga diperlukan. Tanpa pembanding tersebut, peningkatan akibat penggabungan beberapa model belum dapat dipisahkan dari manfaat kedalaman. [Hasil fusi](NEW763_RGBD4_RESULTS.md).

---

## 7. Riwayat Pendekatan dan Hasil Eksperimen

Riwayat eksperimen mencakup seri `V2-E`, `PT-E`, dan `AF-E`, serta 2.893 baris evaluasi pada gelombang validasi. Nomor ID tertinggi tidak diperlakukan sebagai jumlah eksperimen unik karena terdapat celah dan penggunaan ulang ID. Jumlah baris evaluasi juga tidak sama dengan jumlah replikasi independen. [Buku besar eksperimen](../metrics/07_buku_besar_eksperimen.md), [status gelombang validasi](../experiments/STATUS.md).

| Pendekatan | Varian | Hasil | ID eksperimen |
|---|---|---|---|
| Detektor | YOLO26l/s/m, RT-DETR-L, RF-DETR-L; 960–1.280 px | RF-DETR-L tertinggi pada tabel perbandingan utama; keunggulan statistik tercatat pada dataset 763 dan gabungan, bukan semua kondisi | `V2-E-001`, `017`, `034`–`038` |
| Fusi kedalaman | Invers, Sobel `edge`, *mid-fusion*, monokular, lima kanal, RGBD4, *late fusion* | Belum terbukti meningkatkan kinerja secara konsisten pada data uji (§6) | `V2-E-005`–`011`, `024`, `027`–`032` |
| Pascaproses deteksi | WBF, *re-ranker* 30–37 fitur, pengujian berbagai ambang dan resolusi | +0,0108 mAP50 signifikan (953); −0,0139 tidak signifikan (763) | `V2-E-019`, `039`; [MAP_BOOST.md](../results/remote_eval_2026-08-28/MAP_BOOST.md) |
| Pengklasifikasi | ConvNeXt, Swin, EfficientNetV2, ResNet-18, DINOv2; CE, CORAL, CORN; multi-tampak; model khusus untuk kelas kematangan berdekatan; MoE; *stacking*; ansambel | Akurasi per tandan 0,734–0,744; CORAL 0,3305 dan CORN 0,6983 | `PT-E-012`, `014`, `015`, `018`, `023`, `029`–`036`; `V2-E-015`, `044`, `046` |
| Praproses warna | Hue, *sharpening*, CLAHE, *gray-world*, *white balance*, gamma, TTA | Tidak ada perbaikan menyeluruh | [HANDOFF.md](../HANDOFF.md) |
| Resolusi citra terpotong (*crop*) | 32–320 px | Tidak ada peningkatan signifikan di atas acuan pada pengklasifikasi linear tanpa pelatihan ulang model pengekstraksi fitur | `V2-E-048` ([Pengujian pengaruh resolusi citra](ABLASI-ANGGARAN-PIKSEL.md)) |
| Pencocokan tandan antarfoto | Geometri, re-ID, GNN, *prior* arah putar, ExtraTrees, Hungarian, GSP MILP, *greedy*, 3D | *Prior* arah putar: F1 0,398 → 0,649; pelatihan menggunakan hasil deteksi: 0,1492 → 0,3788; metode 3D tidak menunjukkan manfaat | `PT-E-002`, `008`, `013`, `016`, `017`, `020`, `022`; `V2-E-043` |
| Pencacahan | Ridge, CatBoost, meta-ansambel, rekonsiliasi | Perbaikan bergantung pada dataset, definisi jumlah, dan konfigurasi; MAE total tidak disamakan dengan makro-MAE kelas | `PT-E-004`, `026`, `028`; `V2-E-045` |
| Data dan domain | Latih gabungan 953 + 352, Combined-1716, antarperiode pengambilan data | Model 763 pada uji 953: mAP50 0,1776 | `V2-E-021`, `035`, `040`–`042` |
| Struktur pohon | Peringkat vertikal tandan (Spearman −0,616) | +0,0058 makro-F1 | `AF-E-003`, `009` |
| Gelombang validasi | Pipeline V2, lintas lapis, *head* komposisi, *attention* | Tidak ada kandidat unggul di semua metrik | `V2-E-046`–`048` |
| Pengujian ulang tanpa pelatihan tambahan | Pemeriksaan ulang kandidat pada potongan 384 px | mAP50 0,4470 terhadap 0,4647 (12 citra) | [val_12images.json](../results/verifikasi_lokal_beku_2026-09-08/val_12images.json) |

Penggunaan informasi arah pengambilan foto meningkatkan skor *F1* pencocokan tandan antarfoto sekitar 0,25 pada eksperimen terkait. Fusi dan pemeringkatan ulang memperbaiki sebagian metrik lokalisasi atau deteksi. Penggabungan empat kelas menjadi dua kelas menghasilkan tugas evaluasi yang lebih sederhana; selisih +0,2321 *mAP50* tidak dinilai sebagai peningkatan pada tugas empat kelas yang sama.

Sejumlah variasi pengekstraksi fitur, pengklasifikasi, praproses warna, dan ansambel tidak menghasilkan perbaikan yang konsisten dalam protokol yang dicoba. Hasil negatif tersebut tidak membuktikan bahwa penambahan data, kapasitas, atau metode baru tidak dapat membantu. Perbandingan 0,7374 dan 0,7330 dari laporan lama berbeda kondisi serta mempunyai koreksi evaluator, sehingga tidak digunakan sebagai bukti batas kapasitas dataset.

### 7.1 Pengaruh resolusi citra terhadap ketepatan klasifikasi

Percobaan pada 10 September menguji apakah citra tandan dengan resolusi lebih tinggi meningkatkan ketepatan penentuan kelas kematangan. ConvNeXt-Tiny yang telah dilatih sebelumnya digunakan untuk mengekstraksi ciri visual dari citra. Bobot ConvNeXt-Tiny tidak dilatih ulang; hanya pengklasifikasi regresi logistik yang dilatih menggunakan ciri visual tersebut. Pengujian ini disebut *probe* linear.

Percobaan pertama mengubah resolusi citra tandan, kemudian menyamakan ukuran masukan model menjadi 224 × 224 piksel. Contohnya, citra diturunkan menjadi 32 × 32 piksel lalu diperbesar ke ukuran masukan tersebut. Pembesaran ini tidak memulihkan detail yang telah hilang. Percobaan kedua mengubah ukuran citra langsung menjadi 288 × 288 atau 320 × 320 piksel, tanpa tahap penurunan resolusi tambahan.

Pengaturan regularisasi, yang membatasi kerumitan pengklasifikasi, dipilih melalui validasi silang lima kelompok pada data latih. Seluruh foto dari satu pohon ditempatkan dalam kelompok yang sama. Hasil dilaporkan pada 894 objek dari 117 pohon validasi; data uji tidak digunakan.

| Resolusi citra dan ukuran masukan model | Akurasi | Makro-*F1* | Hasil dibandingkan acuan |
|---|---:|---:|---|
| Diturunkan ke 32 × 32, lalu diperbesar ke 224 × 224 piksel | 0,6174 | 0,5537 | Akurasi turun signifikan |
| Diturunkan ke 48 × 48, lalu diperbesar ke 224 × 224 piksel | 0,6141 | 0,5363 | Akurasi turun signifikan |
| Diturunkan ke 96 × 96, lalu diperbesar ke 224 × 224 piksel | 0,6477 | 0,5837 | Selisih akurasi belum signifikan |
| Diturunkan ke 176 × 176, lalu diperbesar ke 224 × 224 piksel | 0,6510 | 0,5852 | Acuan pembanding |
| Diubah langsung ke 224 × 224 piksel | 0,6488 | 0,5650 | Peningkatan akurasi tidak signifikan |
| Diubah langsung ke 288 × 288 piksel | 0,6510 | 0,6044 | Peningkatan makro-*F1* belum signifikan |
| Diubah langsung ke 320 × 320 piksel | 0,6421 | 0,6047 | Peningkatan makro-*F1* belum signifikan |

Selisih akurasi antara kondisi 96 dan 176 piksel adalah −0,0034, dengan selang kepercayaan 95% [−0,0240; +0,0184]. Selang ini mencakup nilai nol, sehingga selisihnya belum signifikan secara statistik. Hasil tersebut belum membuktikan bahwa kedua resolusi setara atau bahwa resolusi di atas 96 piksel tidak lagi bermanfaat. Kesimpulan hanya berlaku pada pengklasifikasi yang memakai ciri visual dari ConvNeXt-Tiny tanpa pelatihan ulang. Pengaruh resolusi ketika seluruh bobot model dilatih ulang belum diuji. [Laporan pengujian resolusi](ABLASI-ANGGARAN-PIKSEL.md), [data hasil pengujian](../results/ablasi_piksel_2026-09-10/ablasi_anggaran_piksel.json).

---

## 8. Peluang Perbaikan Model dan Capaian pada Tugas yang Lebih Sederhana

![Hasil model dan simulasi perbaikan dengan label acuan](assets/monev-2026-09-15/06_capaian_dan_diagnostik.png)

*Gambar 6. mAP50 pada data uji 953. Batang berarsir menunjukkan hasil simulasi yang menggunakan label acuan untuk memperbaiki atau memilih prediksi. Nilainya belum dapat dicapai oleh sistem secara otomatis.*

| Klaim | Bukti | Status |
|---|---|---|
| mAP50 empat kelas maksimal 0,6569 | Satu pengklasifikasi ConvNeXt (akurasi per foto 0,6612) pada kotak acuan ([exp_ceiling.log](../logs_ringkas/audit_forensik_2026-09-06/exp_ceiling.log)) | Belum terbukti |
| Target mAP50 0,85 memerlukan akurasi klasifikasi 0,90 | Simulasi dengan skor buatan | Hanya berlaku pada simulasi |
| Resolusi citra terpotong lebih tinggi membantu | Peningkatan tidak signifikan ketika bobot pengekstraksi fitur tidak dilatih ulang (§7.1) | Belum didukung dalam protokol ini |
| Prediksi yang benar terdapat di antara kandidat hasil deteksi | Pemilihan prediksi menggunakan label acuan 0,9752; penggantian kelas 0,7944 ([BUKTI-BATAS-KOREKSI-MAP50.md](BUKTI-BATAS-KOREKSI-MAP50.md)) | Didukung; pemilihan otomatis belum tersedia |
| Pengujian ulang tanpa pelatihan tambahan membantu | 12 citra validasi: 0,4470 terhadap 0,4647 | Tidak lebih baik |
| Ansambel dapat mencapai 0,80 per tandan | Rentang akurasi 0,734–0,744; *oracle* 0,8739; korelasi keyakinan +0,1185 | Label acuan dapat digunakan untuk memilih model yang jawabannya benar; cara pemilihan otomatis belum terbukti |

Beberapa hasil yang lebih tinggi diperoleh pada tugas dengan kategori lebih sedikit.

- **mAP50 dua kelas 0,7754**, dibanding 0,5433 untuk empat kelas dengan model dan data yang sama (YOLO26s, `AF-E-006`).
- **Jumlah B1 dengan kesalahan paling banyak satu tandan sebesar 95,7–96,5%** (`AF-E-008`) **dan 97,0%** (`AF-E-013`). Jumlah tandan B1 per pohon kecil. Metode pembanding yang selalu menghasilkan jumlah tetap sudah memenuhi toleransi satu tandan pada 92,91% pohon. Namun, MAE model tetap lebih rendah, yaitu 0,3688 dibandingkan 0,8582.
- Pada konfigurasi Panen yang dirujuk, jumlah gabungan B1+B2 meleset paling banyak satu tandan pada sekitar 76,5% pohon. Hasil ini belum memenuhi sasaran > 95%.

Target empat kelas belum terbukti mustahil, tetapi jalur untuk mencapainya juga belum terbukti.

---

## 9. Posisi Kinerja terhadap Sasaran Internal

Target resmi dalam proposal hibah tidak ditemukan pada dokumen repositori yang diperiksa. Tabel berikut menggunakan sasaran internal yang tercatat dalam dokumentasi proyek.

| Target | Sumber | Capaian | Status |
|---|---|---|---|
| mAP50 empat kelas ≥ 0,75 dan ≥ 0,85 | [BUKTI-BATAS-KOREKSI-MAP50.md](BUKTI-BATAS-KOREKSI-MAP50.md), [report-source.md](research_2026-09-06/report-source.md) | 0,6012 (953); 0,6711 (763) | Belum |
| Akurasi kelas sekitar 75% | [HANDOFF.md](../HANDOFF.md) | 74,42% [71,12%; 77,35%] (953); 81,62% (763); hanya tandan yang ditemukan | 953 belum pasti; 763 tercapai |
| Lokalisasi sekitar 90% | [HANDOFF.md](../HANDOFF.md) | AP50 0,8419 (953); 0,8783 (763) | Belum |
| Akurasi per tandan 0,80 | [pipeline-pertandan/STATUS.md](../pipeline-pertandan/STATUS.md) | 0,7439 | Belum |
| Konsistensi pencacahan > 95% | [report-source.md](research_2026-09-06/report-source.md) | Total ±1: 63,70% (953), 85,45% (763); B1 ±1: 97,0% | Hanya B1 |

Akurasi kelas pada tandan yang ditemukan dalam dataset 953 berada di bawah 75%, tetapi selang kepercayaannya mencakup ambang tersebut. Karena itu, belum dapat dipastikan secara statistik apakah akurasi pada populasi yang diwakili data berada di atas atau di bawah 75%. Pada dataset Depth 763, nilai yang tercatat melampaui 75%, dengan keterbatasan data uji yang dijelaskan pada §10. Kedua angka hanya menilai tandan yang berhasil dicocokkan dengan acuan. Repositori tidak memuat kriteria penerimaan industri atau target resmi hibah yang dapat digunakan untuk menyatakan kelulusan sistem.

---

## 10. Keterbatasan Data Pengujian dan Keandalan Kesimpulan

1. **Data uji digunakan berulang.** Hasil menggambarkan kinerja pada pohon yang telah dievaluasi. Penggunaan data uji berulang dalam pengembangan membatasi kekuatannya sebagai bukti pada data yang belum pernah digunakan. Kinerja pada periode pengambilan data baru belum dipastikan.
2. **Sebagian pohon uji Depth berpotensi masuk ke data pengembangan.** Pada skrip penyusunan dataset gabungan yang diperiksa, 39 pohon dari data uji Depth dimasukkan ke data latih dan 5 ke data validasi. Konfigurasi terbaik pada Depth menggunakan detektor `combined1716`. Namun, belum dapat dipastikan apakah bobot detektor yang dievaluasi dilatih menggunakan pohon-pohon tersebut ([report-source.md §4](research_2026-09-06/report-source.md)).
3. **Sebagian data latih dan data uji pada eksperimen lama memuat pohon yang sama.** Sebanyak 122 dari 141 pohon uji 953 terpakai pada prapelatihan `agn953_full`; evaluasi lanjutan kemudian menggunakan 19 pohon yang tidak termasuk dalam prapelatihan tersebut. Sebanyak 44 dari 55 pohon uji 352 juga berada dalam partisi latih 953. Evaluasi lanjutan pada sebagian data tidak menghapus masalah kebocoran pada hasil pengujian sebelumnya (`V2-E-025`, `V2-E-033`).
4. **Penyebab perbedaan antarperiode pengambilan data belum dipastikan.** Jumlah label, komposisi kelas, dan ukuran kotak berubah. Pengaruh pedoman anotasi, kelengkapan label, perubahan biologis, dan cara pengambilan gambar belum diukur secara terpisah (§2).
5. **Program evaluasi masih memiliki keterbatasan.** Skor F1 deteksi tandan belum memeriksa apakah seluruh kotak dalam satu kelompok berasal dari tandan yang sama. Program evaluasi Pipeline Panen tidak menyertakan tiga pohon yang tercatat kosong. Program evaluasi juga mengalami galat ketika tidak ada prediksi.
6. **Dampak kesalahan implementasi belum diukur.** Pemeriksaan kode menemukan masalah pada normalisasi skor WBF, pemeringkatan kelompok deteksi tanpa probabilitas dari modul klasifikasi, pemeriksaan identitas kandidat deteksi, penerapan bobot kelas ExtraTrees dua kali, serta normalisasi citra BGR menggunakan statistik RGB ([`implementation_probes.json`](../results/audit_2026-09-06/implementation_probes.json)).
7. **Sebagian hasil belum dapat diulang dengan lengkap.** Bobot asli RT-DETR-L dan RF-DETR-L pada dataset 953 tidak tersedia. Pelatihan ulang menghasilkan 0,5718 dan 0,5965, sedangkan hasil awalnya 0,5781 dan 0,6012. `link_global_setpartition.py` tidak ditemukan, dan `V2-E-046` tidak tereproduksi.
8. **Definisi kelas tidak konsisten.** `DATASET.md` menyebut B1 lewat matang dan B2 matang optimal, sedangkan audit mengutip kartu dataset yang menyebut B1 tahap panen optimal dan B2 transisi. Definisi ini perlu dikonfirmasi ke pemilik data.
9. **Beberapa dokumen memuat angka atau kesimpulan yang perlu dikoreksi.**
   - [LAPORAN-AKHIR.md](LAPORAN-AKHIR.md) dan [README.md](../README.md) menulis "terbukti meningkatkan lokalisasi", padahal selang kepercayaan 95% untuk selisihnya mencakup nilai nol, sehingga peningkatannya belum signifikan secara statistik.
   - [AUDIT-FORENSIK-2026-09-06.md §1](AUDIT-FORENSIK-2026-09-06.md) menulis target empat kelas "tidak terjangkau", padahal belum terbukti.
   - [`metrics/07_buku_besar_eksperimen.md`](../metrics/07_buku_besar_eksperimen.md) mencantumkan selang `V2-E-030` dan `V2-E-031` yang berbeda dari sumber: [−0,0270; +0,0739] ([`results/boot_sel3_vs_sel1.json`](../results/boot_sel3_vs_sel1.json)) dan [−0,1038; −0,0015] ([`results/boot_sel4_vs_sel2.json`](../results/boot_sel4_vs_sel2.json)).
   - ID `V2-E-048` dipakai dua kali (28 Agustus dan 10 September 2026).
   - [README.md](../README.md) mencantumkan AP50 0,8372; nilai terkoreksi 0,8350.

---

## 11. Kesimpulan Kinerja

### 11.1 Capaian yang didukung hasil pengujian

Model telah menghasilkan deteksi, pengaitan empat sisi, kelas kematangan, dan jumlah tandan yang dapat dievaluasi. Pada dua konfigurasi acuan, *F1* deteksi tandan tercatat 0,8387 dan 0,8534. Pengujian berpasangan terhadap `V2-E-045` mendukung peningkatan *F1* deteksi tandan pada kedua dataset. Hasil tersebut menunjukkan peningkatan pada konfigurasi yang diuji. Pengaruh setiap komponen dan kinerja pada kondisi pengambilan data lain masih perlu dibedakan.

Pada dataset RGB 953, 832 dari 1.342 tandan acuan ditemukan dan diklasifikasikan dengan benar menurut program evaluasi yang digunakan. Pada Depth 763, jumlahnya 373 dari 559. Rasio terhadap seluruh tandan acuan adalah sekitar 62,0% dan 66,7%; rasio ini berbeda dari akurasi kelas pada tandan yang ditemukan 74,42% dan 81,62%, dan tidak digunakan sebagai pengganti makro-*F1*.

### 11.2 Kelemahan utama yang belum teratasi

Keandalan pencacahan per kelas belum setara dengan kemampuan melokalisasi tandan. Jumlah total tepat pada 27,41% pohon RGB 953 dan 44,55% pohon Depth, sedangkan jumlah pada masing-masing kelas B1–B4 tepat sekaligus pada 5,19% dan 27,27%. Sebagian pohon dengan jumlah tepat masih mempunyai objek terlewat dan prediksi berlebih yang saling meniadakan.

Manfaat tambahan kedalaman belum konsisten. Penggabungan prediksi meningkatkan skor pada data validasi, tetapi manfaat kedalaman belum dikonfirmasi pada data baru yang tidak digunakan selama pengembangan. Pengujian penambahan kedalaman dan perubahan resolusi citra juga belum membuktikan bahwa informasi dalam dataset telah dimanfaatkan sepenuhnya atau bahwa metode lain tidak dapat meningkatkan kinerja.

### 11.3 Kesimpulan keseluruhan

Penelitian telah menghasilkan sistem deteksi, klasifikasi, dan pencacahan beserta catatan eksperimennya. Hasil pengujian belum membuktikan kesiapan pencacahan empat kelas untuk penggunaan lapangan. Namun, hasil tersebut juga belum membuktikan bahwa target empat kelas mustahil dicapai. Kesimpulan ini berlaku pada data dan kondisi pengujian yang dilaporkan.

---

## Lampiran A Glosarium

| Istilah | Arti |
|---|---|
| Dataset | Kumpulan citra, label, dan data pendukung yang digunakan dalam penelitian |
| Label acuan atau anotasi | Penandaan posisi dan kelas tandan oleh petugas sebagai pembanding hasil model |
| Konfigurasi | Susunan model, metode pemrosesan, dan pengaturan yang digunakan dalam satu pengujian |
| Presisi / daya tangkap (*recall*) | Presisi mengukur proporsi prediksi yang benar. Daya tangkap mengukur proporsi tandan acuan yang ditemukan |
| Ansambel | Penggabungan keluaran beberapa model untuk menghasilkan satu prediksi akhir |
| Kotak pembatas (*bounding box*) | Persegi panjang yang membatasi wilayah objek dalam citra |
| WBF (*weighted boxes fusion*) | Penggabungan kotak dari beberapa detektor |
| Pemeringkat ulang (*re-ranker*) | Model yang memberi skor baru untuk mengurutkan kandidat deteksi. Skor yang tinggi tidak langsung menunjukkan persentase peluang prediksi benar |
| Pencocokan tandan antarfoto (*linker*) | Pencocokan kotak dari empat foto yang mewakili tandan yang sama |
| Hungarian / GSP MILP | Metode untuk mencocokkan atau mengelompokkan hasil deteksi antarfoto agar tandan yang sama dihitung satu kali |
| *Prior* arah putar | Urutan foto searah jarum jam untuk memperkirakan pergeseran posisi tandan |
| Bootstrap berpasangan | Pengambilan sampel ulang dengan sampel yang sama untuk kedua model pada setiap ulangan, untuk menghitung selang kepercayaan perbedaan kinerjanya |
| Data validasi / data uji | Data validasi digunakan untuk memilih pengaturan model. Data uji digunakan untuk mengukur hasil akhir setelah pengaturan ditetapkan |
| Model batas atas teoretis (*oracle*) | Simulasi yang menggunakan label acuan untuk memilih jawaban benar; hasilnya menunjukkan potensi perbaikan dengan informasi acuan, bukan kemampuan otomatis model |
| *Probe* linear | Pengujian dengan melatih pengklasifikasi linear pada ciri visual yang dihasilkan model pengekstraksi fitur; bobot model pengekstraksi fitur tidak dilatih ulang |
| Penggabungan masukan / prediksi | Kedalaman dapat ditambahkan sebagai masukan model (*early fusion*), atau digunakan dalam model terpisah yang prediksinya digabungkan dengan model RGB (*late fusion*) |
| Depth monokular | Estimasi kedalaman dari citra RGB tanpa sensor |

## Lampiran B Kelas Kematangan

| Kelas | Tingkat | Ciri visual | Median kotak (953) |
|---|---|---|---:|
| B1 | Lewat matang (siap panen) | Jingga kemerahan cerah, posisi terbawah | 133 px |
| B2 | Matang optimal (siap panen) | Oranye kemerahan bersemburat ungu kehitaman | 120 px |
| B3 | Matang awal | Ungu kemerahan kehitaman | 107 px |
| B4 | Mentah | Hitam kehijauan, di sela pelepah | 93 px |

Sumber: [DATASET.md](DATASET.md), [recap.md §4](../metrics/recap.md). Definisi B1 dan B2 perlu dikonfirmasi (§10 butir 8).

## Lampiran C Sumber Data dan Dokumen Pendukung

| Sumber | Isi |
|---|---|
| [Konfigurasi acuan RGB 953](../results/remote_eval_2026-08-28/gsp_artifacts/953/results_test_locked.json) | Metrik, matriks kesalahan klasifikasi, dan ringkasan 135 pohon |
| [Konfigurasi acuan Depth 763](../results/remote_eval_2026-08-28/gsp_artifacts/depth/results_test_locked.json) | Metrik, matriks kesalahan klasifikasi, dan ringkasan 110 pohon |
| [Rekap seluruh konfigurasi](../metrics/recap.md) | Perbandingan, bias kelas, dan audit reproduksi |
| [Bootstrap konfigurasi berpasangan](../results/remote_eval_2026-08-28/ci_artifacts/e2e_paired_test.json) | Selisih terhadap V2-E-045; 5.000 ulangan |
| [Bootstrap pemeringkat ulang](../results/remote_eval_2026-08-28/ci_artifacts/CI_SUMMARY.md) | Selisih lokalisasi dan deteksi empat kelas |
| [Kesalahan yang saling meniadakan pada jumlah total](../results/audit_2026-09-06/count_error_cancellation.json) | Pohon dengan jumlah tepat tetapi identitas tidak tepat |
| [Bootstrap lokalisasi kedalaman](../results/bootstrap_lokalisasi.json) | Uji berpasangan pada 352 |
| [Bootstrap deteksi kedalaman](../results/bootstrap_map_awal.json) | Deteksi empat kelas pada 352 |
| [Uji monokular 953](../results/boot_sel6_vs_sel5.json) | RGB dengan kedalaman monokular terhadap RGB |
| [Uji monokular 352](../results/boot_sel3_vs_sel1.json) | RGB dengan kedalaman monokular terhadap RGB |
| [Uji lima kanal](../results/boot_sel4_vs_sel2.json) | Penambahan estimasi kedalaman dari RGB pada masukan yang sudah memakai sensor |
| [Hasil RGB+D4](NEW763_RGBD4_RESULTS.md) | Fusi masukan dan fusi prediksi pada validasi 763 |
| [Diagnosis kedalaman](DIAGNOSIS-DEPTH.md) | Karakteristik relief dan fitur pada dataset 352 |
| [Matriks audit detektor](../results/audit_forensik_2026-09-06/detector_matrix.json) | Hasil empat, dua, dan satu kelas |
| [Koreksi berbantuan anotasi](BUKTI-BATAS-KOREKSI-MAP50.md) | Simulasi perbaikan pada kumpulan prediksi yang sama dan keterbatasannya |
| [Pengujian ulang tanpa pelatihan tambahan](../results/verifikasi_lokal_beku_2026-09-08/val_12images.json) | Pengujian terbatas pada 12 citra validasi |
| [Pengujian pengaruh resolusi citra](ABLASI-ANGGARAN-PIKSEL.md) | Klasifikasi dengan variasi resolusi citra tanpa melatih ulang ConvNeXt-Tiny |
| [Pemeriksaan metode evaluasi dan implementasi](research_2026-09-06/report-source.md) | Koreksi kesimpulan audit serta batas evaluator |
| [Log eksperimen utama](../experiments/EKSPERIMEN.md) | Rekaman eksperimen V2-E |
| [Log eksperimen per tandan](../pipeline-pertandan/EKSPERIMEN.md) | Rekaman eksperimen PT-E |
| [Buku besar eksperimen](../metrics/07_buku_besar_eksperimen.md) | Indeks seri V2-E, PT-E, dan AF-E |
| [Verifikasi angka laporan](assets/monev-2026-09-15/verifikasi_angka.json) | Aritmetika, hash sumber, dan selisih jumlah antarringkasan evaluasi |
