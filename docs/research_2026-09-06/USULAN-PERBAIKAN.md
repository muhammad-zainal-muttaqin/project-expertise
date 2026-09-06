# Usulan perbaikan berdasarkan lokasi kehilangan informasi

Tanggal: 6 September 2026. Status: diagnosis telah dijalankan; arsitektur di bawah merupakan hipotesis yang belum dilatih. Sasaran mAP50 empat kelas 0,85 dan konsistensi pencacahan 95% belum tercapai.

## Rancangan eksperimen diagnostik

Skrip `scripts/measure_recovery_budget.py` mengevaluasi 91 pohon VALIDATION dengan empat tampak, memakai cache kandidat Panen, `edge_model_v2.pkl`, dan profil final yang tetap. TRAIN tidak dilatih ulang dan TEST tidak digunakan untuk pemilihan. Kandidat mentah berasal dari cache dengan ambang detektor 0,10; ini bukan semua keluaran sebelum NMS.

Reproduksi dari akar repositori:

```bash
.venv-audit/bin/python scripts/measure_recovery_budget.py
```

## Temuan empiris terukur

| Tahap | Tandan GT tercakup atau terpasangkan | Kehilangan pada tahap |
|---|---:|---:|
| GT | 936 | — |
| Memiliki kandidat mentah dengan IoU ≥ 0,50 pada tampak yang sama | 876 | 60 |
| Setelah ambang kepercayaan detektor 0,30 | 775 | 101 |
| Anggota kelompok yang bertahan setelah penyaringan kandidat tunggal 0,45 | 701 | 74 |
| Setelah pengelompokan dan pencocokan evaluator | 649 | 52 |

Sebanyak 227 dari 287 tandan yang tidak terpasangkan, atau 79,09%, mempunyai kandidat mentah. Ini menunjukkan peluang pemulihan sebelum keputusan penyaringan menjadi permanen. Ini **bukan** bukti bahwa 227 tandan tersebut dapat dipulihkan tanpa tambahan positif palsu. Kolom cakupan bersifat optimistis karena satu kotak dapat bertumpang tindih dengan beberapa identitas GT; baris terakhir memakai pencocokan satu-ke-satu evaluator Panen.

Penurunan cakupan dari kandidat mentah ke hasil terpasangkan berbeda menurut kelas: B1 81→72, B2 174→138, B3 450→332, B4 171→107. Perbedaan tersebut mendukung pengujian bias penyaringan menurut kelas, bukan membuktikan satu penyebab visual tertentu.

Dengan bantuan GT untuk mengumpulkan kandidat setiap identitas, rata-rata skor ordinal menghasilkan kelas benar pada 612/876 tandan (69,86%). Memilih kandidat mana pun yang kelas prediksinya benar dengan bantuan GT menghasilkan 703/876 (80,25%). Ukuran kedua adalah batas bersyarat untuk memilih prediksi kelas kandidat yang sudah tersedia, **bukan batas semua metode fusi, akurasi model terbaik, atau mAP**. Pembelajaran fitur baru masih dapat memperbaiki kesalahan bersama antartampak.

Hanya 50/91 pohon memiliki cakupan kandidat mentah yang lengkap; 78/91 kehilangan paling banyak satu identitas. Karena itu, model yang hanya mengelompokkan kandidat tetap belum memiliki bahan yang cukup untuk memulihkan semua identitas pada 95% pohon. Jumlah numerik dapat kebetulan benar ketika positif palsu mengimbangi objek hilang; hal tersebut berbeda dari pemulihan identitas yang benar.

## Keputusan metodologis

Pilih satu arah: **prediksi himpunan tandan fisik dari seluruh tampak, dengan kandidat deteksi sebagai masukan awal**. Pelajari keberadaan, identitas, dan kelas sebelum membuang kandidat berkepercayaan rendah. Pertahankan satu sumber keluaran untuk kotak, kelas, dan pencacahan.

### Keluaran dan supervisi

Satu slot objek memprediksi probabilitas keberadaan, distribusi kelas B1–B4, serta kotak dan visibilitas pada masing-masing dari empat tampak. Jumlah total merupakan banyaknya slot aktif; jumlah per kelas berasal dari kelas slot yang sama. Dengan demikian, pelatihan mengawasi identitas yang menghasilkan jumlah, bukan hanya menyesuaikan angka akhir.

Gunakan `bunch_id` dan anotasi kemunculan antartampak untuk pencocokan bipartit slot–GT. Fungsi objektif mencakup keberadaan/tanpa objek, kelas empat kategori, lokalisasi dan visibilitas per tampak, serta keanggotaan kandidat. Kehilangan pasangan harus menyertakan negatif sulit dari **pohon dan kelas yang sama**. Regularisasi ordinal bersifat tambahan; simpan distribusi empat kelas, bukan hanya satu skor harapan ordinal.

### Implementasi bertahap dalam satu eksperimen

1. **Uji pemulihan dengan kandidat tetap.** Gunakan kandidat asli berkepercayaan ≥ 0,10, termasuk positif palsu; ekstrak fitur visual dari kotak prediksi. Decoder kecil melihat seluruh kandidat empat tampak sebelum menetapkan keanggotaan. Izinkan objek tampak pada satu hingga empat sisi dan sediakan keadaan tanpa objek. Jangan mensyaratkan pasangan kuat agar tandan satu-tampak dapat bertahan. Bekukan backbone dahulu untuk menguji apakah informasi yang ada cukup berguna.
2. **Perbarui representasi jika tahap pertama layak.** Latih fitur visual dengan supervisi identitas fisik dan kelas, termasuk ketahanan terhadap tampak hilang serta perubahan urutan awal sisi. Pertahankan relasi siklik sisi; jangan mengandalkan nomor sisi absolut. Perubahan warna harus dibatasi karena warna merupakan sinyal kematangan, sehingga invariansi warna yang terlalu kuat dapat merusak tugas.
3. **Pulihkan objek tanpa kandidat.** Tambahkan akses decoder ke peta fitur citra dan slot bebas yang dapat memprediksi kotak baru. Tahap ini diperlukan untuk menangani 60 identitas tanpa kandidat; pengelompokan kandidat saja tidak dapat menciptakan bukti citra yang hilang. Ukur deteksi per citra pada tahap ini, selain metrik tandan fisik.

Tahap-tahap ini bukan tiga pencarian arsitektur yang terpisah. Tahap pertama menguji alasan melakukan investasi pada tahap berikutnya. Hindari melatih backbone besar dari awal pada dataset ini.

### Perbedaan dari percobaan terdahulu

| Percobaan yang telah ditinjau | Perbedaan yang harus benar-benar diterapkan |
|---|---|
| C3/Set Transformer pada citra terpotong atau fitur dengan kelompok GT/tetap | Model baru menerima kandidat nyata beserta gangguannya, dan mempelajari keanggotaan sebelum kelompok dikunci. Sekadar mengganti agregator akan mengulang hipotesis lama. |
| Re-ID dan pengklasifikasi pasangan | Supervisi mengutamakan tandan berbeda yang berdekatan dalam pohon dan kelas sama; keluaran pasangan turut dilatih bersama keberadaan dan kelas objek fisik. |
| GSP/Hungarian dengan fitur geometri dan skor kelas | Fitur visual dan distribusi kelas penuh tersedia sebelum keputusan asosiasi. Decoder dapat menolak kandidat dan, pada tahap lanjut, memperbaiki kotak. |
| Ridge/ElasticNet untuk jumlah | Jumlah berasal dari objek fisik yang sama dengan keluaran deteksi. Kesalahan hitung dapat ditelusuri ke objek, bukan hanya dikoreksi sebagai angka agregat. |

Prinsip kandidat dari detektor pralatih untuk membantu pembelajaran asosiasi mempunyai landasan pada [MOTRv2, CVPR 2023](https://arxiv.org/abs/2211.09791). Adaptasi ke empat tampak sawit adalah usulan di sini; hasil pelacakan video pada makalah tersebut tidak menjamin keberhasilan untuk perubahan sudut pandang yang besar.

### Protokol keputusan sebelum pelatihan

- TRAIN melatih, VAL memilih, dan prediksi per pohon disimpan. Untuk kandidat TRAIN, utamakan prediksi di luar lipatan pelatihan detektor; kandidat pada citra yang digunakan melatih detektor dapat terlalu bersih dan menciptakan ketidaksesuaian dengan VAL. Periksa silsilah bobot sebelum membuat cache baru agar pekerjaan terdahulu dapat dipakai kembali.
- Bandingkan Panen final, kandidat rendah dengan pengaitan lama, dan decoder baru pada **pohon dan evaluator VAL yang sama**. Model terkunci terdahulu yang lebih kuat juga perlu direproduksi pada VAL sebelum menyebut model baru sebagai kemajuan proyek.
- Ablasi minimal: fitur visual versus geometri/skor saja, serta keanggotaan terpelajar versus kelompok tetap. Ini menguji unsur baru, bukan melakukan sapuan puluhan model.
- Catat mAP50 agnostik dan empat kelas **per citra**, presisi/recall/F1 identitas fisik, akurasi kelas pada objek terpasangkan, kesalahan jumlah total, akurasi tepat, toleransi ±1, dan ketepatan vektor empat kelas. Jangan menukar nama metrik tersebut.
- Periksa kemurnian identitas kelompok, bukan hanya kecocokan maksimum satu anggota; catat pula pohon dengan hitungan tepat yang masih mempunyai positif palsu dan negatif palsu.
- Gunakan selisih metrik berpasangan dan bootstrap per pohon. Lanjutkan ke backbone/peta fitur hanya jika pemulihan recall tidak diimbangi kerusakan presisi dan kelas, serta pencacahan membaik pada pohon yang sama. Laporkan hasil tidak signifikan sebagai demikian, tanpa memilih berdasarkan TEST.
- Klaim generalisasi memerlukan pemisahan ID pohon fisik lintas sumber dan waktu. Pisahkan klaim perubahan waktu pada pohon yang sama dari klaim pohon atau kebun baru. TEST lama yang sudah sering dilihat merupakan pembanding historis; validasi konfirmatori membutuhkan kelompok yang belum dipakai memilih metode.

## Batasan validitas dan status

Diagnosis ini memilih arah yang memiliki peluang terukur, bukan membuktikan terobosan. Belum ada decoder baru, pelatihan baru, atau peningkatan mAP dari usulan ini. Metrik cakupan kandidat tidak dapat dikonversi langsung menjadi mAP50 atau konsistensi jumlah 95%. Artefak angka lengkap: `results/audit_2026-09-06/recovery_budget_val.json`.
