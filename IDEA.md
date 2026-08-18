# IDEA — Pipeline Multi-View Per-Tandan

Ide awal untuk menggabungkan prediksi *multi-view* secara teoretis dan
matematis terbukti valid, tetapi performa agregat totalnya saat ini tertahan
oleh efisiensi penaut (*linker*) di adegan padat.

## 1. Ide Awal dan Pergeseran Paradigma

- **Konsep:** mengubah satuan evaluasi dari kotak per-foto menjadi objek tandan
  fisik per-pohon dengan menggabungkan 4–8 sudut kamera.
- **Tujuan utama:** memangkas galat klasifikasi kematangan sawit dengan
  memanfaatkan sudut pandang yang lebih kaya.

## 2. Temuan Utama dan Keberhasilan

- **Hadiah gratis — recall fisik naik 8,11 pp.** Tanpa melatih detektor baru,
  recall fisik pada *test set* melonjak dari **82,27% menjadi 90,38%**. Tandan
  yang terlewat di satu foto tetap berhasil ditemukan bila tertangkap dari sisi
  lain.
- **Mekanisme agregasi valid — naik 4,85 pp.** Aturan Ekspektasi Ordinal (R4)
  menaikkan akurasi klasifikasi dari **66,55% menjadi 71,43%** pada
  sub-populasi tandan multi-sisi yang berhasil disatukan.
- **Fitur penaut kunci.** Informasi arah putaran kamera (`+dx`) menaikkan F1
  penaut dari **0,3979 menjadi 0,6486**.

## 3. Akar Bottleneck

- **Cakupan penaut rendah — 29%.** Pada adegan padat, penaut hanya mampu
  menyatukan 29% tandan. Sisanya, 71%, terpecah menjadi pool kecil berderau.
- **Dampak agregat.** Efek positif +4,85 pp tenggelam oleh tandan yang gagal
  disatukan. Akurasi total pipeline turun tipis dari **72,03% menjadi 71,24%**,
  sementara galat counting membengkak menjadi MAE **3,34**, dibandingkan
  Baseline Ridge **1,05**.

## 4. Solusi yang Belum Dicoba untuk Mengejar Target $\geq 80\%$

- **Modul C2/C3 — classifier terpisah.** Mengganti skor detektor YOLO (C1)
  dengan *backbone* khusus seperti ConvNeXt atau Transformer Multi-View untuk
  menilai potongan tandan.
- **Ordinal loss.** Mengganti Cross-Entropy biasa dengan Ordinal Regression
  Loss agar galat antarkelas bersebelahan, terutama B2 versus B3, makin kecil.
- **Visual Re-ID dan GNN penaut.** Memperkuat penaut agar cakupan penyatuan
  tandan naik dari 29% menjadi lebih dari 70%.

Potensi maksimal ide awal, *Oracle R4*, adalah **73,60%** selama tetap
mengandalkan skor detektor YOLO apa adanya. Karena itu kombinasi peningkatan
cakupan penaut dan classifier khusus C2/C3 merupakan jalur untuk melampaui
plafon tersebut dan mengejar 80%.

## 5. Scope Eksperimen Aktif

Mulai eksperimen lanjutan, pengembangan dan pemilihan konfigurasi difokuskan
pada varietas **DAMIMAS** menggunakan dataset turunan
`/workspace/SawitMVC-YOLO-Damimas`. Split pohon tetap mengikuti manifest
kanonik; pohon LONSUM tidak boleh masuk ke train, validation, ataupun test run
DAMIMAS. Strategi pengembangan bersifat *greedy gain*: setiap kepala tugas
(deteksi, klasifikasi per-tampak/per-tandan, penautan, dan counting) boleh
berlapis atau memakai ensemble selama tidak menggunakan label test untuk
pemilihan konfigurasi.
