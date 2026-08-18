# Pipeline Greedy DAMIMAS

Dokumen ini adalah rancangan hidup untuk satu sistem berlapis yang mengejar
nilai terbaik per tugas. Ia bukan matriks ablation. Setiap cabang boleh memakai
kandidat, model, atau aturan keputusan yang berbeda selama seluruh pemilihan
dilakukan pada train/validation dan label test tidak menjadi fitur inferensi.

## Scope Data

- Dataset: `/workspace/SawitMVC-YOLO-Damimas`
- Split pohon kanonik: **641 train / 86 validation / 127 test**
- Citra: **2.700 / 364 / 532**
- Kotak: **13.227 / 1.757 / 2.461**
- Kelas train B1/B2/B3/B4: **1.569 / 2.504 / 6.764 / 2.390**
- Tidak ada pohon LONSUM di ketiga split.

## Prinsip Sistem

Satu bank kandidat dibaca oleh empat kepala tugas. Tidak ada alasan matematis
untuk memaksa keputusan yang optimal bagi mAP menjadi keputusan yang sama untuk
counting atau klasifikasi fisik.

1. **Kepala deteksi per-citra** memilih/menyatukan YOLO, RT-DETR, dan RF-DETR
   secara per kelas. Keluaran ini dipakai untuk mAP50, mAP50-95, precision, dan
   recall operasional.
2. **Kepala klasifikasi** menggabungkan distribusi C1 detektor, classifier crop,
   dan statistik multi-view. Aturan ordinal dipasang di validation.
3. **Kepala identitas fisik** memakai geometri putaran, visual Re-ID, skor kelas
   lunak, GNN kompetitif, dan perakit klaster global berkendala.
4. **Kepala counting** menerima statistik multi-ambang dari seluruh sisi dan
   seluruh detektor. Jumlah pool linker hanya menjadi salah satu fitur; ia tidak
   dipaksa menjadi hitungan akhir.

## Leaderboard DAMIMAS Saat Ini

Angka di bawah adalah checkpoint antara dari artefak lama yang difilter ke
DAMIMAS dan kepala baru yang sudah dipilih tanpa test.

| Tugas | Baseline DAMIMAS | Kepala greedy saat ini | Perubahan |
|---|---:|---:|---:|
| Deteksi test mAP50 | 0,5503 | training khusus DAMIMAS berjalan | — |
| Deteksi test mAP50-95 | 0,2604 | training khusus DAMIMAS berjalan | — |
| Recall fisik oracle-link @ conf 0,01 | 0,9704 | 0,9704 | baseline plafon |
| Klasifikasi per-tandan (kotak GT, link oracle) | 0,7242 | **0,7462** | **+2,20 pp** |
| Macro-F1 per-tandan (kotak GT, link oracle) | 0,7014 | **0,7270** | **+2,56 pp** |
| Akurasi tandan multi-tampak (link oracle) | — | **0,7825** | bottleneck mendekati 80% |
| Akurasi tandan satu-tampak (link oracle) | — | 0,6445 | bottleneck utama kelas |
| Counting macro MAE | Ridge historis 1,0542 (semua varietas) | **1,0039** | konteks, bukan selisih berpasangan |
| Counting class ±1 | Ridge historis 60,64% (semua varietas) | **75,79%** | konteks, bukan selisih berpasangan |

Catatan: angka counting historis memakai semua varietas sehingga tidak boleh
dipakai sebagai klaim efek berpasangan. Ia hanya memberi skala. Pembanding
DAMIMAS yang identik akan dihitung dari fitur yang sama.

## Konfigurasi Klasifikasi yang Sedang Dikunci

Pemilihan dilakukan dengan GroupKFold berdasarkan pohon pada validation.
Campuran saat ini:

- 54% C1 multi-view berbobot keyakinan;
- 36% rerata seluruh probabilitas C1 dan delapan C2;
- 10% HistGradientBoosting atas 235 statistik per-tandan;
- keputusan ordinal dengan ambang `(0,55; 1,70; 2,50)`.

Di test, recall B1/B2/B3/B4 menjadi **75,63 / 61,13 / 82,46 / 66,54%**.
Semua kelas naik terhadap kepala C1 lama; gain tidak dibeli dengan merusak B2.
Angka modul ini belum boleh disebut end-to-end: potongannya berasal dari kotak
GT dan pengelompokan view memakai identitas oracle. Evaluasi deploy baru sah
setelah detektor dan linker DAMIMAS baru dipasang.

## Urutan Kerja Berikutnya

1. Selesaikan YOLO26l khusus DAMIMAS dan dump skor penuh semua split.
2. Latih RT-DETR-L dan RF-DETR-L khusus DAMIMAS; cari fusion per kelas di val.
3. Fine-tune classifier residual C1+crop pada skor detektor baru.
4. Latih ulang GNN penaut di pasangan deteksi DAMIMAS dan ganti union-find
   serakah dengan pemilihan hipotesis klaster global.
5. Masukkan statistik seluruh detektor dan linker sebagai fitur counting,
   kemudian kunci ensemble dari validation.
6. Setelah semua konfigurasi tetap, jalankan laporan test final dan bootstrap
   berkelompok pada tingkat pohon.
