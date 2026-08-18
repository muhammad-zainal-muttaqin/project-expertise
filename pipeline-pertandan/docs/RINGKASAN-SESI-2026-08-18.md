# Ringkasan Sesi 2026-08-18

Dokumen ini merangkum SELURUH temuan sesi, termasuk yang tidak punya entri
`PT-E-*` sendiri. Ditulis atas permintaan pemilik repo agar tidak ada informasi
yang hanya hidup di riwayat percakapan.

## 1. Hasil terhadap target

**Target `IDEA.md` 0,80 TIDAK tercapai.** Angka akhir klasifikasi per-tandan
DAMIMAS: **0,7439** (selisih -5,6 pp).

| | |
|---|---|
| champion terdokumentasi sebelum sesi | 0,7378 |
| **dicapai, terkunci VAL** | **0,7439** |
| plafon rata-rata berbobot (bobot dipas di TEST, curang) | 0,7523 |
| oracle pilih-anggota | 0,8739 |
| target | 0,8000 |

## 2. Kenapa berhenti di situ — jaraknya terkuantifikasi

Populasi test terbelah dua:

| wilayah | porsi | akurasi |
|---|---|---|
| seluruh anggota SEPAKAT | 64,7% | 0,8192 |
| anggota BERSELISIH | 35,3% | 0,6121 |

Seluruh kesenjangan ke oracle ada di wilayah berselisih. Di sana:

| strategi | akurasi |
|---|---|
| oracle (ada anggota yang benar) | 0,9741 |
| rata-rata probabilitas | 0,6121 |
| pilih anggota paling YAKIN | 0,5711 |
| pilih anggota ACAK | 0,5435 |

Untuk total 0,80: `0,647 x 0,8192 + 0,353 x X = 0,80` -> **X = 0,765**, yaitu
**+15,3 pp** di wilayah berselisih.

Tiga cara membaca "siapa yang benar" diuji, ketiganya gagal:

1. **bobot global** (PT-E-034) -- plafon curang 0,7523, hanya +0,84 pp di atas hasil jujur
2. **keyakinan** (PT-E-035) -- korelasi confidence-vs-benar **+0,1185**; conf saat
   benar 0,6905 lawan saat salah 0,6502; memilih yang paling yakin LEBIH BURUK
   daripada merata-ratakan
3. **pola perselisihan** (PT-E-036) -- gerbang gradient boosting **-3,59 pp** di CV

**Kesimpulan:** sinyalnya ADA (oracle 0,9741) tetapi tidak terbaca dari keluaran
anggota saja. Gerbang yang bisa membacanya harus melihat CITRA dan dilatih pada
prediksi out-of-fold (tiap anggota dilatih ulang K kali). Itu biaya GPU dan belum
dijalankan.

## 3. Temuan yang berlaku lintas-korpus (bukan cuma DAMIMAS)

### 3.1 Penaut selama ini dilatih di domain yang salah (PT-E-017, korpus 953)

Sejak PT-E-002, penaut dilatih di pasangan kotak GT lalu dipakai di atas deteksi.
AUC-nya **0,9508 di kotak GT tetapi 0,5868 di deteksi** -- praktis acak di domain
tempat ia dipakai. Melatih di pasangan deteksi: F1 **0,1492 -> 0,3080**
(+15,88 pp). GNN di atasnya -> **0,3788** (+7,08 pp). Pool yang seluruhnya positif
palsu turun 0,147 -> 0,040.

Konsekuensi: diagnosis "hambatannya kepadatan adegan/kombinatorik"
(`CLAUDE.md` sec.6) **tidak lengkap**.

### 3.2 Plafon 73,60% bukan plafon (PT-E-018, korpus 953)

`IDEA.md` menutup dengan "potensi maksimal 73,60% bila tetap mengandalkan skor
detektor YOLO". Ensemble C1+C2 memberi **0,7464**. Setiap anggota C2 KALAH dari
C1 (0,682-0,706 lawan 0,7208) tetapi gabungannya menang. Yang PT-E-012 tutup
adalah jalur MENGGANTI C1, bukan MELENGKAPI C1.

### 3.3 Penaut dan classifier saling menggantikan, bukan berlipat (PT-E-019)

Kontribusi penaut +0,63 pp, kontribusi kelas +1,11 pp, gabungan hanya +0,87 pp.
Sebabnya terbaca dari kolom multi-tampak: ensemble menolong terutama di tandan
SATU-tampak, sementara penaut baru memindahkan tandan KELUAR dari sana.

### 3.4 CORAL runtuh di mana CORN bekerja (PT-E-030)

Resep identik, hanya loss berbeda: CORAL test **0,3305**, CORN **0,6983**
(+36,8 pp). Sebabnya struktural: weight-sharing CORAL membuat
`P(y=tengah) = sigma(s+b0) - sigma(s+b1)` terkurung jarak antar-bias; terukur
maks P(B2)=0,291, maks P(B3)=0,301. Jangan digeneralisasi -- paper CORN sendiri
melaporkan gain modest di dataset seimbang.

### 3.5 RF-DETR DAMIMAS memuncak di epoch 5 (PT-E-032)

val ema_mAP50 puncak **epoch 5 = 0,5830**; epoch 59 = 0,4885 (**-9,46 pp**),
menurun monoton. ~6,5 dari 7 jam GPU tidak menghasilkan checkpoint yang dipakai.
Untuk run berikutnya: maksimal ~15 epoch + patience, JANGAN pilih checkpoint
terakhir.

### 3.6 `harapan_geser.json` yang ter-commit SALAH

Berkas cache konstanta arah putar tidak memuat satu pun entri 8-sisi padahal
split train berisi 34 pohon 8-sisi. Karena `penaut_pertandan` memuatnya otomatis
saat di-import, pohon 8-sisi selama ini tidak dapat prior arah putar sama sekali
(`HARAP.get(...)` jatuh ke 0,0). Sudah dihitung ulang di split kanonik.

### 3.7 PT-E-012 tidak punya error bar

Rentang antar-seed untuk konfigurasi yang SAMA adalah 1,06-1,99 pp (C2) dan
0,43-1,93 pp (C3), lebih besar daripada selisih C2-C1 = -1,21 pp yang jadi dasar
putusannya. Arah PT-E-012 bertahan; magnitudonya tidak.

### 3.8 `moe` dan `klasik` di DAMIMAS adalah sinyal yang sama

val 0,7301 / test 0,7166 / 1-view 0,6098 / multi 0,7546 -- identik di semua kolom.
Seleksi ensemble akan memungutnya sebagai anggota "baru" dan menghitung satu
sinyal dua kali kalau tidak dibuang lebih dulu.

## 4. Pelajaran metodologis yang mengikat

### 4.1 Memilih aturan keputusan dari fit VAL itu bocor halus

`tau` ordinal per-nview memenangkan fit VAL (0,7595 lawan 0,7508 dan 0,7410) lalu
jatuh ke **0,7318 di TEST**. Aturan berparameter lebih banyak SELALU menang di
data tempat parameternya dipas. Perbaikannya: pilih aturan lewat **CV di dalam
VAL**, yang langsung membalik urutannya (per-nview jadi TERBURUK, 0,7399).

### 4.2 VAL 86 pohon terlalu kecil untuk seleksi -- tetapi bukan penyebab plafon

Empat kali dalam sesi ini VAL menyesatkan dengan pola sama. Namun PT-E-034
menunjukkan plafon 0,74 BUKAN akibat VAL kecil: bahkan dengan bobot dipas
langsung di TEST, rata-rata berbobot mentok 0,7523. Dua hal ini terpisah dan
sering tertukar.

### 4.3 Test peeking berulang

Seleksi ensemble dijalankan beberapa kali dengan kumpulan anggota berbeda dan
TEST dilihat setiap kali. **Angka bersih adalah konfigurasi terkunci PERTAMA,
0,7439**; varian dengan `corn224` (0,7409) eksploratif. PT-E-036 memperbaiki
kebiasaan ini: TEST tidak dibuka karena CV tidak menunjukkan keunggulan.

### 4.4 Cari dasar literatur SEBELUM training

Diminta pemilik repo. Bukti kenapa ini penting muncul di sesi yang sama: setelah
membaca, ternyata (a) CORAL punya kelemahan terdokumentasi yang persis kita alami,
(b) "spesialis batas" yang dirancang dari analisis galat sudah ada bentuk lebih
baiknya sejak 2021 (task bersyarat CORN), dan (c) risiko overconfidence pada data
fine-grained kecil sudah dipetakan (Pairwise Confusion, ECCV 2018).

## 5. Acuan literatur yang dipakai

- CORN -- Shi, Cao & Raschka, arXiv:2111.08851 (rank-consistent ordinal via
  conditional probabilities)
- CORAL -- Cao, Mirjalili & Raschka, arXiv:1901.07884
- Ensemble Selection -- Caruana, Niculescu-Mizil, Crew & Ksikes, ICML 2004
- Dynamic Classifier Selection -- Cruz, Sabourin & Cavalcanti, Information Fusion 2018
- Pairwise Confusion -- Dubey et al., ECCV 2018
- Hierarchy of Alternating Specialists -- ECCV 2018

Kalibrasi target: literatur FFB melaporkan 90-99%, tetapi untuk tandan HASIL
PANEN difoto dekat dengan pencahayaan terkendali dan sering 2-3 kelas. Setting
kita -- tandan di pohon, terhalang pelepah, 4 tingkat ordinal, per tandan fisik
lintas-tampak -- tidak sebanding.

## 6. Catatan operasional

- Loop sync 60 menit aktif (`~/.config/pe-sync/sync.sh`, PID tercatat di
  `sync.log`): commit+push GitHub dan unggah bobot ke
  `mz-muttaqin/project-expertise-bobot`. Memakai `git add -A`, jadi ia ikut
  men-commit pekerjaan sesi lain yang sedang berjalan.
- `runs/rfdetr_l_damimas_s42/` berukuran ~9 GB. Yang perlu dipertahankan:
  `checkpoint_best_ema.pth`, `checkpoint_best_regular.pth`,
  `checkpoint_best_total.pth`.
- `scripts/spesialis_batas_damimas.py` ditulis tetapi **tidak pernah dijalankan**
  -- task bersyarat CORN sudah menutupi sebagian besar idenya.
- PAT GitHub sempat dikirim lewat chat; sebaiknya dirotasi.
