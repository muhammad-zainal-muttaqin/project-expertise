# Peluang koreksi mAP50 dengan kandidat deteksi tetap

Tanggal: 8 September 2026. Status: **diagnosis CPU selesai; target mAP50
empat kelas 0,75 belum dicapai oleh pipeline otomatis baru**. Seluruh angka
koreksi berbantuan anotasi di bawah merupakan konstruksi diagnostik. Angka
tersebut tidak boleh dilaporkan sebagai hasil inferensi model.

## Rancangan Eksperimen

Audit memeriksa apakah kotak prediksi yang sudah tersedia masih memungkinkan
mAP50 melampaui 0,75. Skrip
[`audit_batas_koreksi_map50.py`](../scripts/audit_batas_koreksi_map50.py)
membaca tiga dump RF-DETR tanpa pelatihan atau inferensi GPU. Angka lengkap,
versi perangkat lunak, lokasi masukan, dan hash SHA-256 disimpan dalam
[`audit_batas_koreksi_map50_2026-09-08.json`](../results/audit_batas_koreksi_map50_2026-09-08.json).

Dataset acuan adalah SawitMVC RGB 953 pohon. Partisi berasal dari
`Baseline-SawitMVC/ground_truth/split_manifest.csv`; anotasi berasal dari
berkas JSON per pohon pada direktori yang sama. Medan `split` lama dalam
JSON tidak dipakai. VALIDATION mencakup 96 pohon, 404 citra, dan 1.887 kotak;
TEST mencakup 141 pohon, 588 citra, dan 2.612 kotak. Semua tampak dalam
manifest dipertahankan, termasuk pohon dengan delapan tampak dan citra tanpa
objek. Audit menolak perbedaan himpunan identitas citra antara dump dan
anotasi. Dataset Depth tidak dicampurkan dalam perhitungan ini.

Evaluator NumPy memakai pencocokan serakah satu-ke-satu pada *IoU* ≥ 0,50,
101 titik daya tangkap (*recall*), dan maksimum 100 prediksi per citra per
kelas. Implementasi dibatasi pada kotak tanpa anotasi `crowd` atau `ignore`.
Definisi interpolasi mengikuti
[kode evaluator resmi COCO](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).
Hasil aktual dibandingkan dengan dua artefak historis sebelum koreksi
dihitung. Skrip juga memeriksa kasus deteksi sempurna, keluaran kosong,
duplikat, pertukaran kelas, positif palsu berperingkat tinggi, *IoU* tepat
0,50, pembatasan 100 prediksi, dan daya tangkap parsial.

Dump memuat sejumlah baris dengan ID kelas 4, sedangkan B1–B4 memakai ID
0–3. Baris di luar keempat kelas tersebut dikeluarkan **sebelum seluruh
evaluasi aktual dan konstruksi koreksi**, sesuai cakupan kategori evaluator
historis. Jumlahnya dicatat dalam JSON; baris tersebut tidak diubah menjadi
kandidat B1–B4 untuk memperbesar hasil koreksi.

Empat konstruksi diperiksa:

1. **Aktual:** kotak, kelas, dan skor asli pada empat kelas yang dievaluasi.
2. **Penggantian kelas berbantuan anotasi:** untuk setiap kotak dengan *IoU*
   ≥ 0,50 terhadap suatu objek acuan, kelas diganti dengan kelas objek acuan
   yang memiliki *IoU* terbesar. Koordinat, skor, dan seluruh baris pada empat
   kelas tetap tersedia. Latar belakang serta duplikat tidak dihapus.
3. **Penyaringan berbantuan anotasi:** hanya baris yang dinyatakan positif
   benar (*true positive*, TP) oleh evaluator aktual dipertahankan. Kelas,
   koordinat, dan skor baris tersebut tetap asli.
4. **Penggantian kelas dan penyaringan:** konstruksi kedua diikuti
   penyaringan TP dari hasil evaluasinya sendiri.

Penggantian kelas pada butir kedua adalah satu aturan diagnostik yang
eksplisit, bukan pencarian penggantian kelas terbaik. Nilainya bukan batas
atas semua cara memperbaiki kelas. Pendekatan analisis dengan koreksi
berbantuan anotasi sejalan dengan prinsip diagnosis kesalahan dalam
[TIDE, ECCV 2020](https://arxiv.org/abs/2008.08115); audit ini tidak menjalankan
perangkat lunak TIDE ataupun mengklaim dekomposisi kesalahan TIDE yang lengkap.

Reproduksi dari akar repositori dengan NumPy dan SciPy tersedia:

```powershell
python scripts/audit_batas_koreksi_map50.py --out results/audit_batas_koreksi_map50_2026-09-08.json
python -m compileall -q scripts pipeline-pertandan/scripts
```

Lokasi anotasi dapat ditetapkan melalui `--ground-truth-root`. Eksekusi audit
pada sesi ini menggunakan Python 3.14.5, NumPy 2.4.2, dan SciPy 1.17.1.

## Temuan Empiris Terukur

Semua kolom metrik berikut adalah **mAP50 empat kelas per citra**, dengan
populasi yang sama di dalam setiap baris.

| Dump dan partisi | Aktual | Penggantian kelas dengan anotasi | Penyaringan TP dengan anotasi | Kelas dan penyaringan dengan anotasi |
|---|---:|---:|---:|---:|
| RF-DETR `combined1716`, VALIDATION | 0,5727 | 0,8017 | 0,9703 | 0,9777 |
| RF-DETR `combined1716`, TEST | 0,5890 | 0,7944 | 0,9752 | 0,9777 |
| RF-DETR pada artefak pelatihan ulang 7 September, TEST | 0,5965 | 0,7837 | 0,9802 | 0,9827 |

Hasil aktual TEST pertama mereproduksi mAP50 dan AP50 per kelas dalam
[`remote_combined1716_rfdetr_l_953_test.json`](../results/remote_eval_2026-08-27/metrics/remote_combined1716_rfdetr_l_953_test.json)
dengan galat absolut di bawah 0,000001. Hasil aktual TEST kedua mereproduksi
nilai 0,596470002851234 dalam
[`rfdetr_l_v2repro_953_retrain_2026-09-07.json`](../results/rfdetr_l_v2repro_953_retrain_2026-09-07.json).
Artefak kedua mencatat checkpoint `best_ema` epoch 5 saat pelatihan historis
masih berjalan. Dump ini **bukan** dump asli yang menghasilkan angka
historis 0,6012. Sesi audit ini tidak melanjutkan pelatihan tersebut.

Pada dump kedua, penyaringan TP mempertahankan 2.564 dari 2.612 objek acuan
yang sudah terdeteksi dengan kelas benar di antara banyak hipotesis mentah.
Hal ini membuktikan keberadaan himpunan bagian prediksi dengan mAP50 0,9802
ketika pemilihnya boleh membaca anotasi. Hal ini belum membuktikan bahwa
identitas himpunan bagian tersebut dapat dipelajari dari masukan inferensi.

Sebagai pemeriksaan geometri, audit menghitung pencocokan bipartit maksimum
antara kotak kandidat dan objek acuan per kelas pada *IoU* ≥ 0,50. Penggunaan
proposal lintas kelas dilonggarkan sehingga hasilnya merupakan batas atas
optimistis untuk keluarga keluaran dengan geometri tetap. Pada ketiga dump
ini, konstruksi penggantian kelas dan penyaringan mencapai nilai batas
optimistis tersebut. Batas ini tidak berlaku bagi seluruh algoritme yang
bisa membuat kotak baru, dan bukan batas informasi dataset.

Temuan merupakan perhitungan deterministik pada dump tetap. Tidak dilakukan
bootstrap atau uji signifikansi karena belum ada perbandingan dua pipeline
inferensi baru yang dapat digunakan untuk klaim peningkatan populasi.

## Keputusan Metodologis

**Satu hipotesis pipeline yang dipertahankan adalah verifikasi visual lokal
atas pasangan kandidat–kelas.** Detektor yang tersedia menjadi penghasil
kandidat; modul tambahan menilai kelayakan setiap pasangan kotak dan kelas
untuk menjadi deteksi akhir. Sasaran tambahannya adalah memperbaiki kelas
serta pengurutan TP terhadap positif palsu, dengan koordinat kandidat tetap.

Rancangan konkretnya sebagai berikut:

1. Pertahankan kandidat dan alternatif kelas dari keluaran detektor sebelum
   pemangkasan agresif. Simpan indeks kandidat agar setiap keluaran dapat
   ditelusuri. Pengurangan kandidat demi biaya mengubah cakupan dan wajib
   dinilai kembali; hasil audit seluruh kandidat tidak otomatis berlaku
   pada himpunan yang telah dipangkas.
2. Ambil potongan lokal dari citra RGB asli di dalam kandidat, lalu gunakan
   ekstraktor fitur pralatih yang dibekukan. Pertahankan representasi lokal
   sebelum agregasi menjadi satu vektor objek. Tujuannya ialah menyediakan
   detail permukaan buah yang mungkin hilang pada representasi seluruh
   kotak. Keberadaan sinyal pembeda yang cukup pada detail tersebut masih
   merupakan hipotesis, bukan hasil audit ini.
3. Pelajari skor pasangan kandidat–kelas dari fitur lokal, skor detektor,
   dan hubungan dengan kandidat yang bertumpang tindih. Supervisi TRAIN
   harus mencakup kelas salah, latar belakang, serta duplikat. Tetapkan
   kandidat positif secara satu-ke-satu per objek acuan; jangan menganggap
   semua kotak yang beririsan dengan objek sebagai deteksi akhir yang sama
   baiknya. Anotasi hanya digunakan untuk supervisi dan evaluasi, bukan
   sebagai masukan inferensi.
4. Gunakan skor tersebut untuk menetapkan kelas, urutan, dan penanganan
   duplikat. Ukur langsung mAP50 per citra pada seluruh VALIDATION. Hasil
   pengklasifikasi pada potongan objek yang sudah cocok dengan anotasi tidak
   menggantikan evaluasi tersebut. TEST historis tidak dipakai memilih
   parameter atau merancang aturan per citra.

Perbedaan yang harus benar-benar diterapkan dapat ditelusuri pada kode lama.
[`train_proposal_crop_head.py`](../scripts/train_proposal_crop_head.py)
menyaring sampel pelatihan dan validasi menjadi `label >= 0`, yaitu proposal
yang cocok dengan objek acuan. Sementara itu, fitur pemeringkat untuk
dataset 953 dalam
[`rank_and_emit.py`](../results/remote_eval_2026-08-28/scripts/rank_and_emit.py)
berasal dari skor, geometri, dan dukungan lintas tampak, tanpa representasi
lokal RGB sebagai masukan langsung. Pemeringkat historis tersebut telah
menghasilkan mAP50 TEST 0,5970 menurut
[`MAP_BOOST.md`](../results/remote_eval_2026-08-28/MAP_BOOST.md).
Usulan di sini mensyaratkan bukti visual lokal dan supervisi kelayakan
deteksi akhir hadir dalam modul yang sama. Menambahkan pemeringkat skor
biasa akan mengulang arah yang sudah diuji. Audit ini tidak membuktikan
kebaruan metode terhadap seluruh literatur atau seluruh berkas repositori.

Pelatihan ulang detektor tidak diperlukan oleh rancangan ini. Ekstraksi
fitur dan pelatihan modul kecil dapat dijalankan pada CPU, tetapi biaya
waktunya belum diukur. Cache vektor agregat tidak dapat dianggap memuat
kembali token lokal yang telah dibuang. Karena itu, audit ini belum
membenarkan pengeluaran GPU untuk pelatihan MIL atau modul baru: yang telah
diimplementasikan dan dijalankan hanya diagnosis, bukan verifikator.

### Syarat cukup yang dapat dibuktikan

Misalkan pada keluaran akhir suatu kelas $c$, sebuah ambang skor menghasilkan
presisi terukur $p_c$ dan daya tangkap terukur $r_c > 0$, dengan pencocokan
dan populasi evaluator yang sama. Definisikan kisi COCO:

$$
G=\{0;\ 0{,}01;\ \ldots;\ 1\},\qquad
k(r_c)=\left|\{u\in G:u\le r_c\}\right|.
$$

Presisi terinterpolasi pada setiap titik kisi $u\le r_c$ sedikitnya $p_c$:
titik operasi $(r_c,p_c)$ sendiri tersedia dalam maksimum presisi di kanan
titik $u$. Titik kisi lainnya menyumbang nilai tidak negatif. Oleh sebab itu,

$$
\mathrm{AP50}_c\ \ge\ p_c\frac{k(r_c)}{101},\qquad
\mathrm{mAP50}\ \ge\ \frac{1}{4}\sum_{c=1}^{4}
p_c\frac{k(r_c)}{101}.
$$

Untuk $0\le r_c\le1$, secara aritmetika eksak
$k(r_c)=1+\lfloor100r_c\rfloor$. Implementasi menghitung banyaknya titik
kisi secara langsung agar mengikuti representasi numerik evaluator.

Dengan demikian, dua contoh syarat cukup adalah:

$$
\begin{aligned}
\forall c:\ p_c\ge0{,}90,\ r_c\ge0{,}85
&\implies \mathrm{mAP50}\ge0{,}90\frac{86}{101}
=0{,}7663366\ldots,\\
\forall c:\ p_c\ge0{,}95,\ r_c\ge0{,}90
&\implies \mathrm{mAP50}\ge0{,}95\frac{91}{101}
=0{,}8559405\ldots.
\end{aligned}
$$

Ini adalah pembuktian syarat cukup pada **keluaran yang dievaluasi**.
Presisi terukur 0,90 tidak sama dengan skor kepercayaan model 0,90, dan
tidak sama dengan akurasi pengklasifikasi 0,90. Syarat tersebut bukan syarat
wajib: mAP50 dapat melampaui target dengan bentuk kurva presisi–daya tangkap
lain. Yang belum terbukti adalah kemampuan verifikator untuk memenuhi
syarat itu dan mempertahankannya pada pohon baru.

## Batasan Validitas & Audit

Kesimpulan bahwa model atau dataset telah mencapai batas informasinya belum
terbukti. Angka 0,6569 dari
[`exp_ceiling.py`](../scripts/audit_forensik/exp_ceiling.py)
berasal dari pengklasifikasi dan prosedur tertentu pada kotak acuan, sehingga
bukan batas semua pengklasifikasi. Hubungan akurasi 0,90 dengan mAP sekitar
0,847 dalam
[`exp_sensitivity.py`](../scripts/audit_forensik/exp_sensitivity.py)
bergantung pada simulasi kesalahan dan skor buatan. Keduanya tidak
memberikan jaminan mAP50 pipeline yang diusulkan. Ambang akurasi 0,90 dan
makro-AP 0,88 yang sebelumnya diajukan dalam percakapan juga tidak memiliki
kedudukan sebagai syarat matematis penjamin mAP50 deteksi.

Nilai mAP50 0,7837 dari penggantian kelas hanya membuktikan satu keluaran
berbantuan anotasi yang melampaui 0,75. Nilai 0,9802 dari penyaringan hanya
membuktikan keberadaan kandidat benar. Keduanya tidak mengukur kemampuan
modul otomatis untuk mengenali kandidat tersebut. Jarak antara kedua
angka juga tidak boleh ditafsirkan sebagai efek komponen yang aditif,
karena penggantian kelas mengubah pencocokan dan urutan di dalam tiap kelas.

Pipeline yang hanya menerima ringkasan skor dan koordinat tidak mendapat
akses ke detail visual yang tidak tersimpan dalam ringkasan tersebut.
Keputusan dan pemeringkatan dari ringkasan masih dapat diperbaiki, tetapi
keberhasilan ekstraksi informasi kelas tidak mengikuti hanya dari
penambahan lapisan komputasi. Rancangan lokal di atas mencoba memakai
informasi citra yang mungkin belum dimanfaatkan; tidak ada bukti dalam
audit ini bahwa informasi tambahan itu cukup untuk target.

TEST historis dibuka kembali secara eksplisit untuk diagnosis. Tidak ada
pemilihan model atau parameter dalam audit, tetapi penggunaan TEST
berulang tetap membatasi klaim konfirmatori penelitian berikutnya.
Hasil pada himpunan yang sudah diamati tidak memberikan jaminan
deterministik atas data masa depan. Karena itu, status penelitian tetap:
**peluang perbaikan terukur tersedia pada kandidat sekarang; pipeline
otomatis dengan mAP50 sedikitnya 0,75 belum terbukti tersedia**.
