# pipeline-pertandan

Sub-proyek `project-expertise`. **Perjalanan baru**, dimulai 2026-08-17.

Satu kalimat: mengubah satuan inferensi dari **kotak di dalam citra** menjadi
**tandan fisik di sebuah pohon** — deteksi per sisi, penautan lintas-sisi,
lalu satu keputusan kelas per tandan.

Asal-usulnya sketsa tangan 22 Juli 2026
([`docs/sketsa-asal-2026-07-22.png`](docs/sketsa-asal-2026-07-22.png)): empat
sisi masuk ke sebuah corong, ruang `T1` memuat potongan `V1` dan `V2` dari
tandan yang sama, keluar satu label `B1/B2/B3`.

## Status

| | |
|---|---|
| Proposal | disetujui 2026-08-17 |
| Eksperimen dijalankan | **PT-E-000 … PT-E-011, dua dataset** |
| Penomoran | deret `PT-E-*` sendiri, bukan `V2-E-*` |
| Gerbang | G0/G1/G2 **LOLOS**, G3 **GUGUR** — lihat [STATUS.md](STATUS.md) |
| Diagnosis yang berlaku | **kepadatan adegan (kombinatorik)**, bukan mutu detektor — [CLAUDE.md §6](CLAUDE.md) |
| Berikutnya | modul **C3** classifier multi-tampak, lalu prior depth — [CLAUDE.md §7](CLAUDE.md) |

## Sesi 2026-08-18

Target `IDEA.md` 0,80 **tidak tercapai**; angka akhir 0,7439. Jaraknya sudah
terkuantifikasi dan jalurnya menyempit -- baca
[`docs/RINGKASAN-SESI-2026-08-18.md`](docs/RINGKASAN-SESI-2026-08-18.md) sebelum
melanjutkan, khususnya bagian 2 (kenapa berhenti) dan 4 (pelajaran metodologis).

## Mulai dari mana

1. Baca [`docs/PROPOSAL.md`](docs/PROPOSAL.md) — argumen lengkap, gerbang
   falsifikasi, rencana, risiko.
2. Baca [`CLAUDE.md`](CLAUDE.md) — konvensi kerja khusus folder ini.
3. Jalankan probe-nya untuk memastikan angkanya masih keluar:

   ```bash
   cd /workspace/project-expertise
   .venv/bin/python pipeline-pertandan/scripts/probe_penautan_953.py
   ```

   CPU-saja, ~2 menit, `seed=0`. Menulis
   [`results/probe_penautan_953.json`](results/probe_penautan_953.json).

4. Langkah pertama yang sebenarnya: **PT-E-001**, CPU-saja ~1 jam, gerbang G0.
   Ia bisa membatalkan separuh proposal sebelum satu jam GPU pun terpakai.

## Empat angka yang jadi dasarnya

| Angka | Arti |
|---|---|
| **+14,49 pp** | recall naik hanya karena satuannya berubah: 63,36% per-kemunculan → 77,85% per-tandan (detektor sel 5, test, conf 0,25) |
| **9.823** | tandan unik dengan identitas lintas-sisi sudah ada di GT — hasil deduplikasi yang sudah dikerjakan sebelumnya |
| **0,4282** | F1 penautan **geometri-saja** di test — batas bawah, bukan batas kemampuan penaut yang sudah ada |
| **85,5%** | dari tandan yang bisa di-pool punya tepat 2 sisi → voting mayoritas selalu seri, aturan agregasi wajib kontinu |

## Isi folder

```
pipeline-pertandan/
├── README.md            ← Anda di sini
├── CLAUDE.md            konvensi kerja folder ini
├── EKSPERIMEN.md        log append-only PT-E-*
├── docs/
│   ├── PROPOSAL.md      proposal lengkap
│   └── sketsa-asal-2026-07-22.png
├── scripts/
│   └── probe_penautan_953.py
└── results/
    └── probe_penautan_953.json
```

## Yang dipakai dari repo induk (jangan disalin ke sini)

| Aset | Lokasi | Untuk |
|---|---|---|
| Detektor dasar sel 5 | `../models/yolo26l_e60_i1280_v2repro/best.pt` | modul D — bersih terhadap test, dilatih di split kanonik 716 pohon |
| Dump prediksi test | `../results/pred_sel5_953_rgb_test.npz` | PT-E-001 tanpa perlu inferensi ulang |
| venv | `../.venv` | sklearn 1.9.0, torch 2.8.0+cu128, CUDA aktif |
| Bootstrap CI | `../scripts/bootstrap_ci.py` | CI tingkat pohon |
| Dataset | `/workspace/SawitMVC-YOLO` | vanilla, 953 pohon, terverifikasi byte-identik dengan HF |
