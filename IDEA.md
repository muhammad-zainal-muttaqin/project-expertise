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

## 6. Pembaruan Eksperimental DAMIMAS — 18 Agustus 2026

Angka 73,60% pada §4 adalah plafon historis **C1 YOLO lama**, bukan plafon
pendekatan multi-view. Eksperimen strict DAMIMAS dan pipeline proposal baru
telah mempersempit bottleneck secara lebih tepat:

- propagasi confidence lintas-view yang dikunci di validation menaikkan kepala
  deteksi class-aware menjadi test mAP50 **0,5965**, mAP50-95 **0,2743**, dan
  macro-F1 titik-operasi **0,5906**, tanpa mengubah kotak atau jumlah deteksi;
- setelah kelas dilipat, proposal fisik yang sama mencapai AP50 **0,8381** dan
  F1 operasi **0,7984**. Selisih sekitar 25 pp terhadap mAP class-aware adalah
  bukti bahwa klasifikasi kematangan, bukan pencarian objek, kini menjadi
  pengungkit terbesar;
- relabel probabilistik berbasis classifier crop menaikkan keempat AP50 kelas,
  terutama B4 0,3983 -> **0,4106**, tanpa menggandakan objek di jalur fisik;
- propagasi multi-view berikutnya kembali menaikkan keempat AP50 kelas menjadi
  **0,8042 / 0,5035 / 0,6570 / 0,4214** untuk B1--B4;
- proposal unik sebelum linker menaikkan F1 association **0,4631 -> 0,5171**.
  Kepala coverage mencapai **70,62% atas tandan terdeteksi** dan 51,55% atas
  seluruh tandan;
- evaluasi deploy satu-ke-satu tanpa kotak/link GT memilih probabilitas hasil
  propagasi + linker coverage + R4 dan mencapai precision/recall pool fisik
  **85,30% / 81,16%**, akurasi kelas pada pool terpasang **73,22%**, serta
  macro-F1 fisik end-to-end **58,67%** saat miss dan pool palsu ikut dihitung;
- classifier strict terbaik pada kotak dan tautan GT masih ConvNeXt residual:
  akurasi per-tandan **0,7378**, macro-F1 **0,7166**, dan akurasi multi-tampak
  **0,7753**. Mixture-of-experts per-tandan tidak mengalahkannya;
- counting macro terbaik tetap regresor multi-ambang dengan macro-MAE
  **1,0039**. Full multi-bank overfit (VAL 0,8110, TEST 1,0374), tetapi kepala
  compact khusus jumlah total menurunkan total-MAE **1,8583 menjadi 1,7795**.
  MAE jumlah pool linker 1,864 tetap tidak dipakai sebagai counter akhir.

Konsekuensinya, pipeline final memakai kepala berbeda atas bank kandidat yang
sama: skor multi-label untuk mAP, proposal unik untuk identitas fisik, agregasi
ordinal untuk kelas tandan, dan regresi multi-bank untuk counting. RF-DETR-L,
RT-DETR-L, serta detektor agnostik DAMIMAS sedang/akan ditambahkan sebagai
anggota bank; konfigurasi tetap dipilih di validation sebelum test dibuka.

Satu pengungkit kelas tambahan kini punya dasar data yang kuat: classifier
lama belajar dari crop kotak GT, padahal saat deploy ia menerima kotak prediksi.
Pada proposal nyata TRAIN terdapat 26.403 sampel positif, 16.252 hard false
positive (IoU <= 0,15), dan cakupan GT 99,61% pada IoU >= 0,4. Karena itu modul
residual lima kelas `B1--B4 + background` disiapkan untuk dilatih pada proposal
fusion final. Ia sekaligus dapat memperbaiki label kematangan dan menekan false
positive tanpa mengubah koordinat proposal; TEST tetap tidak masuk training
atau pemilihan checkpoint.
