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

Angka di bawah adalah checkpoint antara. Konfigurasi kepala baru dipilih di
validation; test baru dihitung setelah konfigurasi terkunci.

| Tugas | Baseline DAMIMAS | Kepala greedy saat ini | Perubahan |
|---|---:|---:|---:|
| Deteksi test mAP50 | 0,5503 | **0,5839** | **+3,36 pp** |
| Deteksi test mAP50-95 | 0,2604 | **0,2711** | **+1,06 pp** |
| Deteksi macro-F1 operasional | 0,5557 | **0,5752** | **+1,96 pp** |
| Recall fisik oracle-link @ conf 0,01 | 0,9704 | 0,9704 | baseline plafon |
| Klasifikasi per-tandan, strict DAMIMAS (kotak/link GT) | 0,7242 | **0,7378** | **+1,36 pp** |
| Macro-F1 per-tandan, strict DAMIMAS | 0,7014 | **0,7166** | **+1,52 pp** |
| Akurasi tandan multi-tampak, strict DAMIMAS | — | **0,7753** | mendekati 80% |
| Akurasi tandan satu-tampak, strict DAMIMAS | — | 0,6329 | bottleneck utama kelas |
| Akurasi per-view, classifier klasik strict DAMIMAS | — | **0,7103** | anggota komplementer |
| Counting macro MAE | 1,0236 (single model, dipilih di val) | **1,0039** | **−0,0197** |
| Counting class ±1 | 74,61% (single model, dipilih di val) | **75,79%** | **+1,18 pp** |
| Counting tree ±1 | **37,80%** (single model) | 32,28% | ensemble belum menang di metrik ini |

Pembanding counting di tabel memakai fitur 1.683-dim dan protokol split yang
sama. Ensemble menang pada macro-MAE dan akurasi sel-kelas, tetapi belum pada
akurasi gabungan empat kelas per pohon. Karena itu counting belum dianggap
selesai dan akan menerima fitur dari detektor kedua/ketiga.

Kepala deteksi saat ini adalah routing per kelas yang dipilih di validation:
B1 memakai WBF fine-tune+inferensi 1536, B2 WBF baseline+fine-tune, B3 WBF
baseline+1536, dan B4 WBF seluruh bank termasuk tile. AP50 test B1/B2/B3/B4
semuanya naik menjadi **0,7912 / 0,4916 / 0,6546 / 0,3983**. Jadi kenaikan mAP
tidak dibeli dengan mengorbankan satu kelas kecil.

## Status Kepala Klasifikasi

Stacker lama tetap menjadi angka referensi tertinggi, 0,7462, tetapi delapan
anggota C2-nya pernah dilatih pada dua varietas. Karena scope aktif mewajibkan
tinkering DAMIMAS-only, angka itu **tidak dipakai sebagai champion strict**.
Hasil model yang dipasang ulang hanya dengan DAMIMAS adalah:

- ConvNeXt residual 128: 0,7378 / macro-F1 0,7166 (strict terbaik);
- classifier klasik: per-view 0,7103, tetapi per-tandan 0,7234;
- ConvNeXt crop native 224: 0,7242;
- Set Transformer: 0,7310;
- stacking seluruh model strict: 0,7272.

Semua kandidat non-pemenang tetap disimpan sebagai bank probabilitas, tetapi
tidak dipaksa masuk champion. Angka modul ini belum boleh disebut end-to-end:
potongannya berasal dari kotak GT dan pengelompokan view memakai identitas
oracle. Evaluasi deploy baru sah setelah detektor dan linker DAMIMAS dipasang.

## Urutan Kerja Berikutnya

1. Tuntaskan linker DAMIMAS dan ganti union-find
   serakah dengan pemilihan hipotesis klaster global.
2. Latih RT-DETR-L dan RF-DETR-L khusus DAMIMAS; perluas fusion per kelas.
3. Relabel crop deteksi dengan classifier dan pilih campuran di validation.
4. Masukkan statistik seluruh detektor dan linker sebagai fitur counting,
   kemudian kunci ensemble dari validation.
5. Setelah semua konfigurasi tetap, jalankan laporan test final dan bootstrap
   berkelompok pada tingkat pohon.
