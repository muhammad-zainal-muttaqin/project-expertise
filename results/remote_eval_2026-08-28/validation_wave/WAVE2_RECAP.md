# Validation Wave 2 — Recap dan keputusan

Tanggal eksekusi: 2026-08-28 UTC. Seluruh eksperimen baru pada wave ini
memakai TRAIN untuk fitting dan VAL untuk seleksi. Tidak ada skrip wave ini
yang menerima atau membaca split TEST. Angka TEST yang disebut di bawah hanya
rujukan hasil test-locked yang sudah ada; wave ini tidak mengubahnya.

## Ringkasan jumlah percobaan

| Jalur | Dataset | Baris evaluasi | Hasil utama |
|---|---:|---:|---|
| Pipeline V2 stage 1 + stage 2 | 953 | 172 | Tidak mengalahkan anchor end-to-end |
| Pipeline V2 stage 1 + stage 2 | Depth | 172 | Beberapa trade-off membaik, belum all-round terhadap anchor |
| Cross-layer topology/count/class frontier | 953 | 815 detector + 176 class = 991 | Mengulang kandidat robust; tidak menemukan all-rounder baru |
| Composition-aware member head | 953 | 378 | Matched/macro naik pada komposisi tertentu, count turun |
| Count meta-ensemble | 953 | 58 | Kandidat macro/matched, MAE dan ±1 turun |
| Count meta-ensemble | Depth | 30 | Tidak all-round |
| Edge ensemble → GSP | 953 | 1.080 | Tidak mengalahkan profil robust |
| GPU group-attention control | 953 + Depth | 6 + 6 | 953 turun; Depth flat |

Total yang dievaluasi: **2.893 baris konfigurasi**. Ini jumlah baris
evaluator, bukan 2.893 model independen; beberapa konfigurasi menghasilkan
output identik karena batas ukuran atau ranking tidak aktif pada topology
tertentu.

## Anchor VAL dan kandidat terbaik

Metrik: F1 fisik makin besar makin baik; MAE makin kecil makin baik; ±1,
matched class accuracy, dan macro-F1 makin besar makin baik.

| Dataset / profil | F1 fisik | MAE | ±1 | Matched | Macro-F1 | Putusan |
|---|---:|---:|---:|---:|---:|---|
| 953 anchor Hungarian | 0,823216 | 1,252747 | 0,670330 | 0,754204 | 0,601394 | Baseline yang dipertahankan |
| 953 robust class calibration | 0,823216 | 1,252747 | 0,670330 | **0,769728** | **0,617081** | Kandidat terbaik all-round; count/physical invariant |
| 953 cross-layer, best matched | 0,823216 | 1,252747 | 0,670330 | **0,769728** | **0,617081** | Reproduksi kandidat robust |
| 953 cross-layer, best macro | **0,839396** | 1,527473 | 0,582418 | 0,758667 | **0,631103** | Exploratory; count/±1 turun |
| 953 count-meta + calibration | 0,825994 | 1,318681 | 0,626374 | 0,768531 | 0,622456 | Exploratory; count/±1 turun |
| 953 composition-aware, best macro | 0,825175 | 1,307692 | 0,626374 | 0,758801 | 0,618633 | Ditolak sebagai produksi |
| 953 edge ensemble, best matched | 0,823467 | 1,263736 | 0,659341 | 0,760363 | 0,608231 | Ditolak |
| Depth anchor original GSP | 0,852641 | 0,931624 | 0,786325 | 0,845652 | 0,680685 | Profil kerja |
| Depth topology + V2 geo count + scale macro | **0,854225** | **0,914530** | 0,786325 | **0,850000** | **0,689013** | Kandidat VAL; CI inconclusive |
| Depth count-meta, best target MAE | 0,854495 | 0,914530 | 0,786325 | 0,839479 | 0,672771 | Ditolak; matched/macro turun |
| Depth V2-only, best F1 | 0,845601 | 0,820513 | 0,837607 | 0,845011 | 0,686110 | Trade-off, bukan pengganti |
| Depth GPU group-attention | 0,852641 | 0,931624 | 0,786325 | 0,845652 | 0,680688 | Flat control |

### Statistik VAL

- Anchor gate 953 dan Depth lulus dengan perbedaan kurang dari `5×10⁻⁵`
  pada metrik utama dan jumlah match sama persis.
- Kandidat robust 953 dibanding anchor, bootstrap berpasangan 5.000 pohon:
  matched delta `+0,015410`, CI95 `[+0,003736; +0,027883]`, `P>0=0,9918`;
  macro delta `+0,015575`, CI95 `[+0,001266; +0,030756]`, `P>0=0,9842`.
- Kandidat Depth topology+count+class dibanding anchor, bootstrap 5.000
  pohon: CI delta F1 `[-0,006160; +0,009540]`, MAE
  `[-0,085470; +0,051282]`, matched `[-0,011744; +0,021558]`, dan
  macro-F1 `[-0,015425; +0,033034]`. Jadi point estimate-nya baik, tetapi
  belum layak disebut peningkatan signifikan.

## Hasil TEST-LOCKED sebagai rujukan, tidak dibuka ulang

| Dataset | Profil test-locked | F1 fisik | MAE | ±1 | Matched | Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| 953 | Baseline → mAP/GSP hasil terkunci | 0,804348 → **0,838710** | 1,392593 → **1,362963** | 0,614815 → **0,637037** | 0,7111 → **0,7442** | — |
| Depth | Baseline → GSP hasil terkunci | 0,806859 → **0,853408** | 0,890909 → **0,772727** | 0,809091 → **0,854545** | 0,8031 → **0,8162** | 0,6047 → **0,6519** |

CI test yang sudah tersimpan menunjukkan F1 fisik signifikan positif untuk
953 (`[+0,020939; +0,047689]`) dan Depth (`[+0,025733; +0,069025]`).
CI mAP image-level juga tersedia di
[`ci_artifacts/CI_SUMMARY.md`](ci_artifacts/CI_SUMMARY.md). Tidak ada hasil
wave ini yang dijalankan ke TEST, sehingga angka TEST di atas tetap single-look
dan tidak tercampur dengan seleksi baru.

## Keputusan

1. Tidak ada konfigurasi baru yang memenuhi semua guardrail 953 atau Depth
   sekaligus lebih baik daripada anchor yang relevan.
2. Profil robust 953 tetap menjadi kandidat class layer VAL terbaik; profil
   test-locked yang sudah dilaporkan tidak diubah.
3. Depth topology+count+class disimpan sebagai kandidat VAL untuk eksperimen
   berikutnya, bukan diklaim sebagai kemenangan statistik dan bukan dipakai
   untuk menulis ulang hasil test.
4. Semua branch yang menaikkan macro atau matched dengan mengorbankan MAE/±1
   disimpan sebagai exploratory/negative control agar komprominya dapat
   ditelusuri, bukan disembunyikan.

## Artefak reproduksi

Script wave ada di `scripts/`; laporan numerik ada di `reports/`; checksum
seluruh artefak ada di [`SHA256SUMS.validation_wave.txt`](../SHA256SUMS.validation_wave.txt).
Laporan besar Pipeline V2 menyimpan fitur VAL mentah sebagai bagian dari
diagnostik; ia tidak mengandung TEST prediction.
