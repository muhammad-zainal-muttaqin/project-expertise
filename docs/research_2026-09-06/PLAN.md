# Rencana audit dan riset pemulihan proyek

Tanggal: 6 September 2026. Pelaksana: satu agen, berurutan, tanpa subagen.

## Ruang lingkup dan asumsi

- Menelusuri berkas sumber, evaluator, pembentukan dataset, pelatihan, fusi,
  pengaitan multi-tampak, klasifikasi, dan pencacahan; menghubungkan temuan
  dengan artefak asli dan penelitian primer.
- Sasaran pengguna: mAP50 deteksi empat kelas sekurang-kurangnya 0,85 dan
  konsistensi pencacahan lebih dari 95%; definisi konsistensi sedang diklarifikasi.
  Akurasi tepat, toleransi ±1, serta jumlah per kelas dilaporkan terpisah.
- Pengguna telah mengonfirmasi kelanjutan setelah pengunduhan dan eksperimen
  tambahan selesai. Audit boleh membaca dataset serta cache yang tersedia.
- Tidak mengubah anotasi, hasil historis, atau partisi pengujian. Perbaikan
  kode harus disertai kasus reproduksi dan uji regresi yang relevan.
- Angka target merupakan kriteria penerimaan, bukan janji hasil.
- Arahan lanjutan pengguna: prioritaskan review implementasi dan sejarah
  `project_expertise_experiment_map`; jangan mengulang pendekatan yang telah
  diuji. Selama tahap riset, perubahan produksi ditunda sampai temuan dipetakan
  ke eksperimen terdahulu. Kasus sintetis hanya untuk reproduksi cacat kode.
- Sumber: kode dan artefak lokal; makalah asli; dokumentasi resmi implementasi.
- Perkakas update_plan tidak tersedia pada sesi ini; berkas ini menyimpan
  rencana dan status penggantinya.

## Tahap

1. **SELESAI untuk cakupan review** — Inventaris berkas utama, definisi metrik,
   silsilah data, dan audit evaluator. Cakupan bukan seluruh isi repositori.
2. **SELESAI untuk cakupan review** — Reproduksi sintetis dan evaluasi ulang
   konfigurasi tetap Panen awal serta Panen final; tanpa pelatihan/GPU.
3. **SELESAI untuk cakupan review** — Penelitian primer berdasarkan kesenjangan bukti;
   membandingkan hipotesis penyebab dan alternatif rancangan.
4. **SELESAI untuk cakupan review** — Laporan DOCX, cakupan pembacaan,
   matriks bukti, dua skrip diagnostik, serta JSON hasil dan pemeriksaan
   struktur dokumen tersedia. Perbaikan produksi belum diimplementasikan.
5. **BELUM DIJALANKAN** — Implementasi produksi dan ablasi TRAIN/VAL setelah
   review; bukan bagian dari pekerjaan diagnostik yang sudah selesai.
6. **BELUM DIJALANKAN** — Kunci model dan protokol; verifikasi generalisasi hanya pada partisi yang
   sesuai status paparan datanya. Laporkan kesenjangan terhadap target.

## Pembaruan setelah jeda

- Commit `a9766c3` menambah AF-E-001–010; `e6bddc9` memperbaiki UF.
- AF-E-011–013 dan berkas Panen final muncul selama review dilanjutkan.
  Perubahan pengguna dipertahankan; audit baru ditulis terpisah.
- Klaim plafon universal, perbandingan makro-F1, definisi B1/B1+B2, dan
  populasi 132/135/141 diverifikasi ulang pada kode dan artefak.

## Aturan keputusan

Pembaruan 6 September setelah pertanyaan tentang solusi: diagnosis kehilangan
kandidat pada VAL telah dijalankan melalui `measure_recovery_budget.py`.
`USULAN-PERBAIKAN.md` menetapkan hipotesis prediksi himpunan tandan sebelum
penyaringan permanen, perbedaannya dari eksperimen lama, dan protokol pengujian.
Implementasi serta pelatihan arsitektur tersebut belum dijalankan.

- Bedakan cacat terkonfirmasi, hipotesis, hasil negatif, dan bukti tidak tersedia.
- Jangan menafsirkan AP agnostik sebagai mAP empat kelas atau akurasi pencacahan.
- Jangan menganggap seluruh hipotesis lama tertolak secara umum ketika resep,
  anggaran pelatihan, evaluator, atau sumber data tidak sebanding.
- Berhenti menambah pencarian yang berulang bila tidak mengubah keputusan;
  lanjutkan pekerjaan kode, pembuktian, atau dokumentasi yang masih diperlukan.
