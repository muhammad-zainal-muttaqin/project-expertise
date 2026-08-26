# Repository Guidelines

## Struktur Proyek

- `scripts/` berisi skrip Python dan shell untuk membangun dataset, training,
  inferensi, evaluasi, counting, dan bootstrap.
- `docs/` menyimpan spesifikasi dataset, prosedur regenerasi, reproduksi, dan
  laporan. Mulai dari `docs/LAPORAN-AKHIR.md` sebelum mengubah arah eksperimen.
- `experiments/` berisi log eksperimen append-only dan status fase; `results/`
  berisi JSON, dump prediksi, dan artefak ringkas yang terlacak.
- `pipeline-pertandan/` adalah subproyek mandiri dengan skrip, dokumentasi,
  dan hasilnya sendiri. `splits_fase6/` berisi daftar split kecil yang dilacak.
  Dataset besar, cache, dan sebagian bobot berada di luar repo.

## Perintah Build, Test, dan Development

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-freeze.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m compileall scripts pipeline-pertandan/scripts
.venv/bin/python scripts/probe_depth_signal.py --probe semua
```

Dua perintah terakhir adalah pemeriksaan sintaks dan probe diagnostik
read-only. Regenerasi dataset mengikuti urutan di `docs/REGENERASI.md`;
training dan evaluasi mengikuti `docs/REPRODUKSI-FASE6.md`. Jalankan evaluasi
atau pekerjaan singkat langsung di foreground; runner hanya untuk training
panjang. Periksa kapasitas GPU dengan `nvidia-smi` sebelum training.

## Gaya Kode, Penamaan, dan Standar Bahasa
 
Gunakan Python 4 spasi per indentasi, fungsi kecil dengan nama deskriptif, dan
`snake_case` untuk modul, fungsi, serta variabel. Pertahankan istilah teknis
dan komentar berbahasa Indonesia. Eksperimen memakai ID `V2-E-###` atau
`PT-E-###`; jangan mengubah entri eksperimen lama.

Seluruh dokumentasi narasi wajib mematuhi **EYD Edisi V / PUEBI**, prinsip **anti-*calque***
(misal: "penurunan performa yang signifikan", "selang kepercayaan 95% mencakup nilai nol"),
notasi matematika desimal koma (0,6012), simbol minus tipografis asli `−`, dan format
selang kepercayaan `[min; max]`. Rujukan lengkap: `.agents/skills/perbaikan-bahasa-penyampaian/SKILL.md`.

## Panduan Pengujian

Repo ini tidak memiliki test suite otomatis atau target coverage. Minimal,
jalankan `compileall` setelah perubahan Python dan probe/evaluasi yang relevan.
Simpan angka, konfigurasi resep, dump prediksi test (`.npz`), riwayat epoch,
dan log ringkas sesuai aturan di `CLAUDE.md`; setiap angka harus dapat dilacak
ke skrip, dataset, split, dan berkas hasilnya.

## Commit dan Pull Request

Riwayat memakai subjek singkat dan spesifik, biasanya diawali ID eksperimen,
misalnya `PT-E-031 ...` atau `V2-E-026 ...`. Pull request harus menjelaskan
tujuan, dataset/split, perintah reproduksi, hasil atau regresi, serta tautan
ke dokumen dan artefak terkait. Sertakan caveat validitas dan alasan untuk
hasil negatif; jangan memasukkan dataset, cache besar, rahasia, atau bobot
yang seharusnya dicadangkan melalui jalur terpisah.
