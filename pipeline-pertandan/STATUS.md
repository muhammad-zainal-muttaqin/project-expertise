# STATUS — lacakan kerja `pipeline-pertandan`

Diperbarui: 2026-08-18 (PT-E-012..018 masuk; lihat bagian baru di bawah). **Berkas ini hidup** — dicentang saat langkah selesai.
Aturan mainnya di [`CLAUDE.md`](CLAUDE.md), rencananya di
[`docs/PROPOSAL.md`](docs/PROPOSAL.md), catatan hasilnya di
[`EKSPERIMEN.md`](EKSPERIMEN.md).

## Tujuan

Mengubah satuan inferensi dari kotak-per-citra menjadi tandan-fisik-per-pohon,
lalu **membandingkannya dengan pipeline yang sudah ada**. Detektor dasar:
`yolo26l_e60_i1280_v2repro` (sel 5), dipakai ulang apa adanya.

## Daftar kerja

| | Langkah | Status | Hasil |
|---|---|---|---|
| ✅ | **PT-E-000** probe kelayakan | selesai | `results/probe_penautan_953.json` |
| ✅ | Inferensi + dump **vektor 4-kelas penuh** (train/val/test) | selesai | `results/pred_skorpenuh_*.npz` |
| ✅ | Validasi dump vs mAP tercatat | **LOLOS** | mAP50 0,5435 vs 0,5436 acuan (selisih 0,0001) |
| ✅ | **PT-E-001** plafon oracle + gerbang **G0** | **LOLOS** | penggabungan +4,36 pp test, CI95 [+2,33; +6,25] |
| ✅ | **PT-E-002a** penaut geometri + penampilan tangan, **G1** | **GUGUR** | penampilan tangan menambah ~0 (F1 val 0,4518 → 0,4485) |
| ✅ | **PT-E-003** end-to-end percobaan 1 (varian B, kelas GT) | LOLOS semu (−1,5 pp) | pool 100% homogen → agregasi tidak bekerja sama sekali |
| 🔧 | **Ketidakcocokan kelas GT vs kelas prediksi di penaut** | diperbaiki, sedang diuji | 100,0% pool homogen kelasnya; AUC-drop fitur `kelas_sama` 0,375 |
| ✅ | **PT-E-002b** embedding re-ID (ResNet-18, SupCon) | selesai | loss 4,62 → 0,317 (30 epoch, 433 s) |
| ✅ | **PT-E-002** 5 varian (A / B / B2 / D / E) | selesai | tabel di bawah; **G1 tetap GUGUR** |
| ✅ | **PT-E-002** varian E ulang dengan embedding **out-of-fold** | selesai | F1 test 0,1801 → **0,3979** |
| ✅ | **PT-E-003** ulang dengan penaut sah (E) | **G2 GUGUR** | −2,36 pp vs oracle; tapi penggabungan +4,85 pp CI [+2,03; +7,81] |
| ✅ | **PT-E-004** counting: pool vs k-global vs Ridge+F_all | **G3 GUGUR** | 3,3422 vs 1,0542 (Ridge) |
| ✅ | `docs/HASIL.md` §0–8 + `EKSPERIMEN.md` PT-E-000…004 | selesai | |
| ✅ | **PT-E-006** uji penghitung Baseline-SawitMVC di deteksi yang sama | selesai | angka repo tereproduksi persis; 0,375 ternyata angka kotak GT |
| ✅ | **PT-E-007** penghitung itu sebagai REM penggabungan | **DIPALSUKAN** | porsi tersatukan 29% → 76%, tapi akurasi turun monoton |
| ❌ | ~~Setel ambang penaut di ruang deteksi~~ | **dibatalkan** | PT-E-007 membuktikan ambang bukan masalahnya |
| ✅ | **PT-E-012** classifier multi-tampak (C3) | **DIPALSUKAN** | C3 -3,06 pp vs C2; C2 -1,21 pp vs C1 |
| ✅ | **PT-E-008** fitur **arah putar** pengambilan foto | **G1 & G2 kini LOLOS** | F1 penaut 0,398 → 0,649; end-to-end −2,36 → −1,81 pp |
| 🔧 | Bug: konstanta arah tidak aktif saat modul di-import | ditutup di akar | di-cache ke berkas + dimuat otomatis + `RuntimeWarning` |
| ✅ | Setel ambang penaut di ruang deteksi | selesai | 0,45 terpilih; perbaikan marginal (+0,22 pp val, 0 di test) |
| ✅ | **PT-E-004** counting diulang dengan penaut baru | **G3 tetap GUGUR** | 3,4610 vs 1,0542 (Ridge) |
| ✅ | **PT-E-009** sapu ulang `conf` deteksi | **DIPALSUKAN** | conf 0,10 sudah optimal; menaikkannya memburuk monoton |
| ✅ | **PT-E-010** konfigurasi terbaik diuji di **SawitMVC-Depth 352** | **SEBAGIAN DIKONFIRMASI** | penaut di ruang deteksi 0,196 → **0,708** dengan detektor lebih bersih |
| ✅ | **PT-E-011** uji klaim "hambatannya detektor" | **DIPALSUKAN** | presisi 0,584 vs 0,639, recall 953 justru lebih baik |
| ❌ | ~~Perbaiki detektor~~ | **dibatalkan** | kedua detektor setara; yang beda kepadatan adegan |
| ✅ | **PT-E-013** prior depth -> rekonstruksi 3D (352) | **DIPALSUKAN** | AUC 0,45/0,51; geometri pengambilan tak terkendali |

## Angka yang sudah final

### Perbandingan terhadap pipeline lama (test, 141 pohon, detektor sama)

| | lama (per citra) | baru (per tandan) | selisih |
|---|---|---|---|
| Recall | 0,8227 | **0,9038** | **+8,11 pp** |
| Akurasi kelas, tautan sempurna | 0,7203 | **0,7360** | +1,57 pp |
| Akurasi kelas, pipeline utuh | 0,7203 | 0,7124 | −0,79 pp |
| Akurasi kelas, **pada tandan yang berhasil disatukan** (n=371) | 0,6655 | **0,7143** | **+4,88 pp** |

### G0 — nilai penggabungan (tautan oracle), dipisah dari rekalibrasi

| Suku | val | test |
|---|---|---|
| rekalibrasi (R0cal−R0) | +0,47 pp, CI [−1,26; +2,31] | −0,20 pp, CI [−1,90; +1,51] |
| **penggabungan, pool ≥2 tampak** | **+7,13 pp, CI [+5,21; +9,24]** | **+4,36 pp, CI [+2,33; +6,25]** |
| gaya E-016 (R2−R0) | +1,62 pp | +1,91 pp |

### Ringkasan gerbang

| Gerbang | Putusan | Angka penentu (test) |
|---|---|---|
| validasi dump | **LOLOS** | mAP50 0,5435 vs 0,5436 |
| **G0** nilai penggabungan | **LOLOS** | +4,36 pp, CI95 [+2,33; +6,25] |
| **G1** penaut | **LOLOS** (setelah PT-E-008) | val F1 0,6718 / ARI 0,6139 vs ambang 0,65 / 0,55 |
| **G2** end-to-end | **LOLOS** (setelah PT-E-008) | −1,81 pp vs toleransi −2,0 |
| **G3** counting | **GUGUR** | macro MAE 3,4610 vs 1,0542 (Ridge+F_all) |

Satu penyebab menjelaskan ketiga kegagalan: penaut hanya menyatukan **29%**
tandan (recall pasangan 0,120 di ruang deteksi). Pada 29% yang berhasil,
penggabungan tetap menaikkan akurasi **+4,85 pp** (CI95 [+2,03; +7,81]) —
tanpa GT sama sekali.

**PT-E-008 mengubah gambaran ini.** Setelah fitur **arah putar** ditambahkan
(foto diambil memutari pohon searah jarum jam — informasi dari pemilik data yang
tidak pernah dipakai), penaut melonjak dari F1 0,398 ke **0,649** dan G1 serta G2
sama-sama lolos. Paragraf di bawah ini mendahului temuan itu dan tetap berlaku
untuk kondisi *sebelum* arah putar dipakai.

**PT-E-007 mempersempit penyebabnya (kondisi tanpa arah putar).** Memaksa penaut menggabung lebih
banyak — sampai jumlah kelompok mendekati cacah SEBENARNYA — justru menurunkan
akurasi (0,7139 → 0,6454 di test). Jadi penaut bukan berhenti terlalu cepat;
**urutan skornya yang salah**. Pasangan berskor tertinggi yang belum tergabung
mayoritas keliru. Menyetel ambang tidak akan menolong.

### Penaut (PT-E-002) — G1 GUGUR di semua varian

| Varian | val F1 | val ARI | test F1 | catatan |
|---|---|---|---|---|
| A geometri + `kelas_sama` (kelas **GT**) | 0,4518 | 0,4385 | 0,4282 | **bocor**: kelas GT tak ada saat inferensi |
| B + penampilan tangan | 0,4485 | 0,4260 | 0,4290 | penampilan tangan menambah ~0 |
| B2 tanpa fitur kelas | 0,2557 | 0,2243 | 0,2552 | membuang sinyal kelas merusak berat |
| **D kelas prediksi, lunak** | **0,3732** | **0,3228** | **0,3651** | varian sah terbaik |
| E + re-ID, fold terkontaminasi | 0,2257 | 0,1542 | 0,1801 | bukti kebocoran fold |
| **E + re-ID out-of-fold** | **0,4323** | **0,3623** | **0,3979** | **terbaik yang sah** |

Ambang G1 = F1 0,65 / ARI 0,55. Tidak satu pun mendekati.

### PT-E-010 — replikasi di sesi akuisisi berbeda (352 pohon, ~80 hari terpisah)

| Temuan | Replikasi? | 953 | 352 |
|---|---|---|---|
| Arah putar konsisten | **YA, kuat** | 98,6% / 99,7% | **98,4% / 99,0%** |
| Penaut di kotak GT (F1 test) | **YA, lebih baik** | 0,6486 | **0,6847** |
| **Penaut di ruang DETEKSI (F1 test)** | **YA, jauh lebih baik** | 0,1957 | **0,7083** |
| Pergeseran recall | **YA** | +8,11 pp | **+9,64 pp** |
| Keuntungan kelas dari penggabungan | **tidak konklusif** | +4,36 pp CI [+2,33; +6,25] | +2,85 pp CI [−2,00; +8,24] |

~~Selisih 3,6x di ruang deteksi mengonfirmasi diagnosis PT-E-009~~ — **DIBATALKAN
oleh PT-E-011.** Kedua detektor mutunya setara (presisi 0,584 vs 0,639; recall
953 malah lebih baik, 0,823 vs 0,739). Selisihnya dijelaskan **kepadatan adegan**:
953 punya ~235 pasangan lintas-sisi per pohon dengan prevalensi benar ~4%,
352 punya ~28 dengan prevalensi ~21%. Tugasnya ~5x lebih sulit secara
kombinatorik, dan itu sifat korpus, bukan sifat detektor.

Yang tidak konklusif bukan karena efeknya hilang, melainkan karena test 352 cuma
95 pool multi-tampak lawan 758 di 953: lebar CI 10,2 pp lawan 3,9 pp. Uji itu
tidak punya daya untuk memutuskan.

Catatan tambahan: R4 kalah dari R2 di test 352 (0,6946 vs 0,7094) padahal menang
+8,3 pp di val — `tau` dicari di 52 pohon val, terlalu sedikit, jadi overfit.

### Embedding re-ID — menghafal, tapi tidak kosong

| Split | AUC cosine saja |
|---|---|
| train | 1,0000 |
| val | 0,7564 |
| test | 0,7195 |

## Temuan yang sudah mengikat (jangan diulang)

1. **Dump top-1 saja tidak cukup.** Aturan R2/R3/R4 butuh distribusi 4 kelas.
   Diambil dari cabang `one2one` YOLO26 lewat forward hook.
2. **Tensor mentah `(1,300,6)` tidak boleh dipakai langsung** — memuat baris
   duplikat, mAP50 anjlok ke 0,1342. Pakai `predict()`.
3. **Satu anchor bisa memancarkan beberapa deteksi berbeda kelas** (41% kasus),
   dan kelas terpilih belum tentu argmax anchor-nya. Simpan `conf`/`cls` resmi
   DAN vektornya; satukan per anchor sebelum di-pool.
4. **Baseline satu-tampak harus ekspektasi, bukan undian.** Satu undian acak
   menggeser selisih R4−R0 dari 2,38 pp ke 1,03 pp — derau sebesar efeknya.
5. **Rekalibrasi ambang harus dipisahkan dari penggabungan.** Versi pertama
   gerbang G0 mengklaim +4,9 pp padahal sebagian besar berasal dari
   rekalibrasi yang bisa didapat tanpa pipeline ini.
6. **Kelas GT dan kelas prediksi tidak boleh dipertukarkan di penaut.**
   Aturan "beda kelas berarti bukan tandan yang sama" memang **benar secara
   fisik** dan berlaku 100% di GT (`class_mismatch` = 0). Tetapi penaut dilatih
   di kotak GT lalu dipakai atas kelas PREDIKSI, dan di sana aturan itu cuma
   benar ~77% — 23,3% tandan multi-sisi punya prediksi kelas berbeda antar
   sisi. Memberi veto absolut kepada bukti berderau membuat penaut memecah pool
   tepat pada 23% kasus yang ingin diperbaiki agregasi (terukur: `kelas_sama`
   menurunkan AUC 0,375 saat dipermutasi, dan 100,0% pool multi-anggota jadi
   homogen kelasnya). Perbaikannya **bukan** membuang sinyalnya — itu sinyal
   sah — melainkan memakai besaran yang sama di latih dan inferensi, dalam
   bentuk lunak: kemiripan Bhattacharyya antar distribusi prediksi dan selisih
   ekspektasi ordinalnya (varian D).

## Yang masih menunggu jawaban user

- Lokasi algoritma dedup yang sudah ada. Tanpa itu, baseline penaut terpaksa
  dibangun dari nol dan angka gain-nya tidak sebanding dengan milik Anda.


---

# Lanjutan 2026-08-18 — implementasi `IDEA.md` sec.4

`IDEA.md` mengusulkan tiga solusi. Dua di antaranya sudah dipalsukan sebelum
berkas itu ditulis (butir 1 oleh PT-E-012, varian depth butir 3 oleh PT-E-013).
Ketiganya tetap dijalankan atas permintaan, dengan backbone/loss berbeda untuk
yang sudah pernah dicoba.

| | Langkah | Status | Hasil |
|---|---|---|---|
| ✅ | **PT-E-014** backbone ConvNeXt untuk modul C | **SEBAGIAN** | C3 -5,41 -> -2,14 pp vs C1; tetap tidak mengalahkan C1 |
| ✅ | **PT-E-015** ordinal loss CORAL | **SEBAGIAN** | +2,35 pp C2 (resnet18); arah konsisten, magnitudo dalam derau seed |
| ✅ | **PT-E-016** penaut GNN di ruang kotak GT | **TIDAK KONKLUSIF** | AUC +0,0077; F1 +1,06 pp CI [-1,46; +3,83] |
| ✅ | **PT-E-017** penaut dilatih di RUANG DETEKSI | **DIKONFIRMASI** | **F1 0,1492 -> 0,3788** (domain shift +15,88 pp, GNN +7,08 pp) |
| ✅ | **PT-E-018** ensemble C1+C2 | **DIKONFIRMASI** | **0,7208 -> 0,7464** (+2,56 pp, CI [+0,52; +4,53]) |
| ⏳ | **PT-E-019** gabungan end-to-end (penaut baru + ensemble) | belum | bayaran dari kedua di atas |

## Dua koreksi yang mengikat

**1. Penaut selalu dilatih di domain yang salah (PT-E-017).** Sejak PT-E-002,
penaut dilatih di pasangan kotak GT lalu dipakai di atas deteksi. AUC-nya
**0,9508 di pasangan kotak GT, 0,5868 di pasangan deteksi** — praktis acak di
domain tempat ia dipakai. Seluruh angka penautan ruang deteksi (F1 0,1766,
cakupan 29%, G3 gugur) bersandar pada itu. Melatih di pasangan deteksi menaikkan
F1 **+15,88 pp** tanpa satu pun ide baru.

Konsekuensi untuk diagnosis di `CLAUDE.md` sec.6: "hambatannya kepadatan adegan,
dan itu kombinatorik" **tidak lengkap**. Kombinatorik nyata (F1 0,3788 masih jauh
dari 0,65) tapi bukan satu-satunya, dan bukan yang termurah diperbaiki.

**2. Plafon 73,60% bukan plafon (PT-E-018).** PT-E-001 menetapkannya sebagai
maksimum "bila tetap mengandalkan skor detektor YOLO". Ensemble C1+C2 mendarat di
**0,7464** pada protokol yang sama. Setiap anggota C2 KALAH dari C1 (0,682-0,706
lawan 0,7208) tetapi gabungannya menang — jadi yang PT-E-012 tutup adalah jalur
*mengganti* C1, bukan jalur *melengkapi* C1.

## Peringatan berkas: `results/harapan_geser.json` ter-commit SALAH

Berkas cache konstanta arah putar yang ter-commit sebelum 2026-08-18 berisi:

    {"4|1": 0.16783, "4|2": 0.20851, "4|3": -0.16404}

Dihitung ulang dengan `hitung_harapan_geser` di split train kanonik (716 pohon):

    {"4|1": 0.23073, "4|2": 0.23125, "4|3": -0.25,
     "8|1": 0.11979, ... "8|7": -0.15208}

Dua masalah. Nilai 4-sisinya tidak cocok, dan berkas lama **tidak punya satu pun
entri 8-sisi** padahal split train berisi 34 pohon 8-sisi (682 pohon 4-sisi).
Karena `penaut_pertandan` memuat berkas ini OTOMATIS saat di-import, skrip yang
tidak menghitung ulang secara eksplisit memakai konstanta salah, dan untuk pohon
8-sisi prior arah putarnya **nol sama sekali** (`HARAP.get(...)` jatuh ke 0,0).
Nilai baru lebih dekat ke angka yang didokumentasikan di docstring
`hitung_harapan_geser` sendiri (+0,241 / -0,260). Berkasnya sudah diperbarui.

Eksperimen 2026-08-18 tidak terdampak: semuanya memanggil `hitung_harapan_geser`
eksplisit, dan modul C tidak memakai fitur geometri.

## Bobot BELUM di-backup

`runs/` di-gitignore (~1,5 GB: 8 sel modul C, 3 re-ID, 2 GNN). Sesuai
`../CLAUDE.md` ATURAN #1 bobot butuh jalur backup terpisah ke bucket HF, dan sync
otomatis sudah dihentikan permanen sejak 2026-08-12 — **jadi ini harus dijalankan
manual dan belum dilakukan.** Yang sudah aman di git: seluruh dump probabilitas
(`pt_e_014_prob_*.npz`, `pt_e_016/017_skor_test.npz`), sehingga angka bisa
dihitung ulang tanpa bobot.
