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

## 6. Bahasa

Bahasa Indonesia, istilah teknis asing ditulis apa adanya. Sama seperti repo
induk.
