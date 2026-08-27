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

WBF tiga model menghasilkan AP50 lokalisasi 81,06%. Pipeline proposal fisik DAMIMAS mencapai 83,81%, tetapi angka tersebut bukan akurasi klasifikasi akhir.

## 3. Penautan lintas-sisi

Modul linker harus:

- bekerja pada hasil deteksi nyata, bukan hanya kotak ground truth;
- memakai arah pergeseran bertanda akibat gerakan searah jarum jam;
- lebih mengutamakan pasangan sisi bersebelahan;
- menggunakan fitur posisi, ukuran, luas, kemiripan probabilitas kelas, dan bila tersedia embedding Re-ID;
- memakai kompatibilitas kelas secara lunak, bukan aturan keras “kelas berbeda pasti tandan berbeda”;
- membatasi satu tandan maksimal satu deteksi per sisi;
- memakai batas ukuran pool maksimal tiga tampak untuk konfigurasi empat sisi, sesuai data yang tersedia;
- menghasilkan `link_confidence` dan menandai pool yang meragukan.

Jangan menghitung jumlah tandan hanya dengan menghitung kotak atau jumlah pool. Cara itu terbukti menghasilkan kesalahan besar.

## 4. Klasifikasi kematangan

Untuk setiap tandan fisik yang sudah ditautkan:

## 5. Counting

- Ridge Regression dengan fitur `F_all` multi-ambang;
- rekonsiliasi agar jumlah per kelas konsisten dengan jumlah total.

Jumlah pool mentah tidak boleh dijadikan hasil counting final. Benchmark terbaik yang tercatat adalah:

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

Saya akan memilih:

> Model dilatih pada combined1716, tiga detektor digunakan untuk proposal class-agnostic, linker berbasis arah putar digunakan untuk identitas tandan, classifier terpisah digunakan untuk kelas kematangan, dan Ridge F_all digunakan untuk counting.
