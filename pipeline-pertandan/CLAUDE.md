# CLAUDE.md — konvensi kerja `pipeline-pertandan/`

Berlaku **tambahan** di atas `../CLAUDE.md` (repo induk) dan `/workspace/CLAUDE.md`
(ops). Kalau bertabrakan, yang di repo induk menang — kecuali empat butir di
bawah, yang memang khusus folder ini.

## 1. Penomoran: `PT-E-*`, bukan `V2-E-*`

Deret `V2-E-*` hidup di `../experiments/EKSPERIMEN.md` dan sudah sampai
`V2-E-033`. Sub-proyek ini punya log sendiri
([`EKSPERIMEN.md`](EKSPERIMEN.md)), jadi memakai deret yang sama dari dua berkas
append-only berbeda hanya menunggu tabrakan ID. Entri di sini dinomori
`PT-E-001` dan seterusnya.

Aturan lain dari repo induk tetap berlaku apa adanya: satu entri = satu hipotesis
falsifiable, append-only, hasil negatif dicatat dengan bobot yang sama, setiap
klaim numerik terlacak ke skrip/JSON/log, setiap angka menyebut dataset dan split.

## 2. Gerbang G0–G3 mengikat

Ambangnya sudah ditulis di [`docs/PROPOSAL.md`](docs/PROPOSAL.md) §7 **sebelum**
ada satu pun hasil. Kalau sebuah gerbang gugur, putusannya gugur — jangan
menggeser ambangnya setelah melihat angkanya. Kalau ambangnya ternyata salah
pilih, tulis itu sebagai entri baru berisi alasannya, jangan sunting yang lama.

Urutannya juga mengikat: **PT-E-001 (CPU, ~1 jam) jalan lebih dulu**, karena ia
gerbang termurah yang bisa membatalkan sisanya. Jangan mulai training GPU apa pun
sebelum G0 punya angka.

## 3. Kebersihan split — tiga jebakan yang sudah terbukti

| Jebakan | Aturannya di sini |
|---|---|
| Field `split` di dalam `json/*.json` **beda dari split kanonik pada 465 dari 953 pohon** (610/177/166 vs 716/96/141) | Split selalu dari `SawitMVC-YOLO/split_manifest.csv` kolom `new_split`. Ia identik dengan tata letak folder `images/`, nol beda. **Jangan pernah** pakai field `split` di dalam JSON |
| `split_manifest.csv` ber-BOM | buka dengan `encoding="utf-8-sig"`, kalau tidak kolom pertamanya terbaca `﻿tree_id` |
| Kontaminasi Fase 6 | `agn953_full` dan turunannya melihat **122 dari 141 pohon test** (`../docs/LAPORAN-AKHIR.md` §9.2). **Jangan pakai bobot `runs_fase6/*` atau apa pun yang dilatih di `agnostic953` untuk mengevaluasi test.** Model baru di sini dilatih hanya di 716 pohon train |

Yang aman dipakai: `../models/yolo26l_e60_i1280_v2repro/best.pt` (sel 5) — dilatih
di split kanonik, jadi test 141 pohon bersih.

## 4. Deduplikasi bukan masalah terbuka

Graf `_confirmedLinks` di dalam dataset adalah **hasil kerja dedup yang sudah
ada**, bukan sekadar anotasi mentah. Karena itu:

- Angka **0,4282** (F1 penaut geometri-saja) adalah **batas bawah**, bukan
  "kemampuan penautan saat ini". Jangan mengutipnya seolah itu state of the art
  proyek ini.
- Setiap klaim gain penaut diukur sebagai **delta terhadap algoritma dedup yang
  sudah ada**, begitu kodenya tersedia — bukan terhadap 0,4282.
- Kalau algoritma itu ternyata sudah melewati G1, modul L **selesai** dan sisa
  anggaran GPU pindah ke modul C/A. Itu hasil yang bagus, bukan antiklimaks.

## 5. Bobot dan dump

ATURAN #1 di `/workspace/CLAUDE.md` berlaku penuh: `*.pt`, `*.pth`, `runs*/`,
`models/` tidak boleh dihapus, di lokal maupun di bucket. Dan seperti di repo
induk: **dump prediksi test ke `.npz` pada saat evaluasi, bukan belakangan** —
alasannya (bobot Volume 2 yang hilang sebelum sempat di-backup) ada di
`../CLAUDE.md`.

Bobot baru dari sub-proyek ini masuk ke `pipeline-pertandan/runs/`, dan
riwayat per-epoch disalin ke `pipeline-pertandan/results/riwayat_epoch/`
dengan pola `<nama_run>__<berkas>` — sama seperti repo induk.

## 6. Diagnosis yang berlaku (dikoreksi 2026-08-17)

Jangan mulai dari diagnosis lama. Urutan penemuannya begini, dan yang **terakhir**
yang berlaku:

1. ~~"Penaut berhenti terlalu cepat, tinggal setel ambangnya"~~ — **dipalsukan**
   PT-E-007. Memaksa menggabung lebih banyak, bahkan sampai cacah yang
   **sempurna benar**, menurunkan akurasi secara monoton (0,7139 → 0,6454).
   Yang salah urutan skornya, bukan kapan berhenti.
2. ~~"Urutan skornya lemah, tidak ada yang bisa menolong"~~ — **dipatahkan**
   PT-E-008. Fitur arah putar menaikkan F1 0,398 → 0,649.
3. ~~"Hambatannya mutu detektor"~~ — **dipalsukan** PT-E-011. Presisi deteksi
   0,584 (953) lawan 0,639 (352), dan recall 953 justru lebih baik. Kedua
   detektor setara.
4. **YANG BERLAKU: hambatannya kepadatan adegan, dan itu kombinatorik.**
   953 punya ~235 pasangan lintas-sisi per pohon dengan prevalensi benar ~4%;
   352 punya ~28 dengan ~21%. Mencari 10 jarum di 235 jerami lawan 6 di 28.
   **Mengganti backbone tidak mengubah berapa banyak tandan ada di satu pohon.**

Konsekuensi praktis: obat yang tepat adalah **prior yang memangkas ruang
kandidat**, bukan penilai pasangan yang lebih pintar dan bukan detektor yang
lebih bagus. Arah putar bekerja persis karena itu.

## 7. Dua pekerjaan berikutnya, berurutan

**(a) Modul C3 — classifier multi-tampak.** Bentuk paling setia terhadap sketsa
asal, dan satu-satunya bagian rancangan yang belum pernah dibangun. Sekarang:
tiap foto dinilai sendiri lalu digabung **rumus di luar model**. C3: satu model
menerima seluruh foto tandan itu sekaligus (attention antar-tampak, panjang
variabel 1-6) dan memutuskan sendiri cara menggabungkannya. Rumus buta konteks —
ia tidak bisa tahu satu foto buram atau dua tampak beda kelas karena satu dari
sisi bayangan; model yang melihat semuanya bisa mempelajarinya.

Latih di 7.427 tandan split train 953 (5.546 multi-sisi), backbone sebagian
dibekukan karena datanya tipis. ~4 jam GPU. **Batasnya:** hanya memperbesar
nilai dari tandan yang berhasil disatukan — 22-29% di 953, 60%+ di 352 — dan
tidak menyentuh masalah kepadatan.

**(b) Prior pemangkas kandidat berikutnya: depth di korpus 352.** Jarak fisik
tandan dari kamera, digabung arah putar, praktis menetapkan posisi 3D-nya.
Kaveat jujur: E-007 Volume 1 sudah memalsukan penautan berbasis depth — tetapi
tanpa prior arah dan tanpa penilai terlatih, dua hal yang sekarang ada.

**Yang TIDAK perlu dicoba lagi:** ganti detektor (butir 3 di atas), setel ambang
penaut (PT-E-007), deskriptor penampilan tangan (PT-E-002a), algoritma dedup
Baseline-SawitMVC sebagai penaut maupun sebagai rem (PT-E-006/007), menaikkan
ambang deteksi (PT-E-009).

## 8. Periksa penyebut. Setiap kali.

Empat kali di sub-proyek ini kesalahan penyebut hampir menghasilkan klaim palsu:

1. Rekalibrasi ambang tersamar sebagai keuntungan agregasi (+4,9 pp palsu)
2. Recall per-citra dibandingkan dengan recall per-tandan
3. Sapuan conf: akurasi 0,8107 di 243 tandan mudah, bukan 890
4. Deteksi/citra dibandingkan antar korpus **tanpa menormalkan terhadap jumlah
   objek yang seharusnya ada** — ini yang melahirkan diagnosis detektor yang salah

Bentuk keempat paling licin karena penyebutnya tidak kelihatan sebagai penyebut.
Aturannya: **setiap kali dua angka dibandingkan, tanyakan dulu keduanya dibagi
apa** — termasuk saat pembaginya "berapa banyak yang seharusnya ada".

## 9. Bahasa

Bahasa Indonesia, istilah teknis asing ditulis apa adanya. Sama seperti repo
induk.
