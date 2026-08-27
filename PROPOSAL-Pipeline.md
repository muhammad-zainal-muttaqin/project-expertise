# Proposal Pipeline Pencacahan Tandan per Pohon

Menurut saya, pipeline terbaik harus memisahkan tiga tugas:

1. menemukan tandan;
2. menggabungkan tandan yang sama dari empat foto;
3. menentukan kelas dan menghitung jumlahnya.

Alur yang disarankan:

```text
4 foto terarah
→ pemeriksaan kualitas
→ deteksi 3 model
→ proposal fisik class-agnostic
→ penautan lintas-sisi
→ klasifikasi per tandan
→ agregasi ordinal
→ counting per pohon
→ laporan + confidence
```

## 1. Spesifikasi pengambilan foto

- Wajib empat foto: `view_1`, `view_2`, `view_3`, `view_4`.
- Urutan harus konsisten searah jarum jam. Temuan proyek menunjukkan prior arah ini sangat penting untuk penautan.
- Perpindahan antarfoto kira-kira 90°.
- Kamera, jarak, tinggi, dan kemiringan relatif dijaga tetap.
- Pohon harus berada di tengah dan seluruh area tajuk yang relevan terlihat.
- Gunakan resolusi asli kamera; backend melakukan letterbox ke input 1.280 piksel.
- Hindari blur, foto terlalu gelap/terang, digital zoom, filter, dan kompresi berlebihan.
- Aplikasi harus menolak atau meminta pengambilan ulang jika satu sisi buram atau urutannya salah.
- Simpan metadata: ID pohon, ID sesi, nomor sisi, waktu, perangkat, orientasi kamera.

Tidak semua tandan akan terlihat di empat sisi. Berdasarkan data, satu tandan dapat terlihat hanya satu, dua, atau tiga sisi. Jadi pipeline tidak boleh mengasumsikan setiap tandan harus muncul empat kali.

## 2. Deteksi dan proposal fisik

Untuk mode akurasi maksimum:

- Jalankan YOLO26l, RT-DETR-L, dan RF-DETR-L.
- Simpan seluruh vektor probabilitas B1–B4, bukan hanya kelas tertinggi.
- Gunakan WBF secara class-agnostic untuk membuat proposal tandan fisik.
- Proposal ini hanya menjawab “di mana tandannya?”, bukan kelas kematangannya.

Angka 81,06% dan 83,81% yang pernah tercantum di proposal adalah hasil
historis dengan protokol berbeda. Pada verifikasi remote 27 Agustus 2026,
WBF tiga model `combined1716` mencapai AP50 lokalisasi class-agnostic 87,64%
di SawitMVC-Depth-YOLO dan 83,72% di SawitMVC-YOLO. Angka ini tetap hanya
mengukur lokasi kotak, bukan klasifikasi kematangan atau counting.

## 3. Penautan lintas-sisi

Modul linker harus:

- bekerja pada hasil deteksi nyata, bukan hanya kotak ground truth;
- memakai arah pergeseran bertanda akibat gerakan searah jarum jam;
- lebih mengutamakan pasangan sisi bersebelahan;
- menggunakan fitur posisi, ukuran, luas, kemiripan probabilitas kelas, dan bila tersedia embedding Re-ID;
- memakai kompatibilitas kelas secara lunak, bukan aturan keras “kelas berbeda pasti tandan berbeda”;
- membatasi satu tandan maksimal satu deteksi per sisi;
- memakai batas ukuran cluster yang dikalibrasi; baseline memakai maksimal
  tiga tampak, sedangkan iterasi greedy remote menunjukkan maksimal dua
  anggota lebih efektif pada test yang diuji;
- menghasilkan `link_confidence` dan menandai pool yang meragukan.

Jangan menghitung jumlah tandan hanya dengan menghitung kotak atau jumlah pool. Cara itu terbukti menghasilkan kesalahan besar.

## 4. Klasifikasi kematangan

Untuk setiap tandan fisik yang sudah ditautkan:

- gunakan crop persegi berisi box dan cincin konteks `1,6×` sisi box;
- masukkan RGB/BGR sesuai preprocessing, ditambah mask posisi box;
- gunakan backbone `ConvNeXt-Tiny` dengan head hybrid softmax + CORAL untuk
  memanfaatkan urutan ordinal B1–B4;
- lakukan augmentasi fotometrik ringan serta jitter posisi/skala mask `±10%`
  agar crop training menyerupai box detektor;
- split berdasarkan pohon, bukan crop, untuk mencegah kebocoran antar-sisi;
- simpan seluruh probabilitas kelas, entropy, dan confidence, bukan hanya
  `argmax`;
- jangan mengaktifkan depth otomatis. Eksperimen proyek menunjukkan early
  fusion depth tidak konsisten; cabang depth harus lulus ablation terkontrol.

Iterasi cepat yang telah dijalankan pada 27 Agustus 2026 adalah pretraining
RGB selama 5 epoch pada 16.542 crop/841 pohon. Hasil validasi terbaiknya
akurasi 62,17%, macro-F1 62,96%, akurasi ordinal ±1 99,32%, dan MAE kelas
0,385. Model ini diuji sebagai blend 25% pada probabilitas WBF di test 953;
hasilnya dicatat sebagai kandidat engineering, bukan model produksi yang
sudah tervalidasi independen.

## 5. Counting

- Untuk pipeline produksi penuh, gunakan Ridge Regression dengan fitur `F_all`
  multi-ambang dan rekonsiliasi agar jumlah per kelas konsisten dengan jumlah
  total.
- Pada verifikasi remote, Ridge `F_all` belum dijalankan karena dump yang
  tersedia adalah keluaran detector/linker. Angka `counting` pada laporan
  remote berarti jumlah **raw linked clusters** per pohon, bukan klaim Ridge.
- Setiap cluster harus memiliki `link_confidence`, kelas/probabilitas agregat,
  daftar sisi yang mendukung, dan status `low_confidence`.

Jumlah pool mentah tidak boleh dijadikan hasil counting final untuk deployment.
Benchmark historis terbaik yang tercatat untuk Ridge adalah:

- akurasi pencacahan dengan toleransi ±1: 75,79%;
- Macro-MAE: 1,0039.

## 6. Hasil yang ditampilkan ke pengguna

Untuk setiap pohon, aplikasi sebaiknya menampilkan:

- estimasi total tandan;
- jumlah B1, B2, B3, dan B4;
- daftar setiap tandan fisik;
- kelas dan probabilitasnya;
- jumlah sisi tempat tandan terlihat;
- confidence penautan;
- kotak tandan pada masing-masing foto;
- indikator kualitas empat foto;
- peringatan jika terlalu banyak tandan hanya terlihat dari satu sisi;
- rekomendasi mengambil ulang sisi tertentu bila kualitas rendah.

## Spesifikasi deployment yang paling realistis

Kandidat arsitektur deployment adalah:

> Model dilatih pada `combined1716`; YOLO26l, RT-DETR-L, dan RF-DETR-L
> menghasilkan proposal class-agnostic melalui WBF; linker memakai prior arah
> putar dan batas satu deteksi per sisi; classifier crop mengeluarkan
> probabilitas B1–B4; kemudian Ridge `F_all` dan rekonsiliasi menghitung total
> per pohon.

Profil parameter greedy pada laporan remote (`proposal_min`, `link_threshold`,
`singleton_min`, mode pasangan sisi, dan ukuran cluster maksimum) adalah alat
diagnostik untuk menunjukkan ruang perbaikan. Sebelum dipakai ke pengguna,
semua parameter harus dikunci pada validation set, lalu diuji sekali pada
hold-out tree-level yang tidak disentuh saat tuning.
