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

### Standar Bahasa & Penulisan Ilmiah Baku

Seluruh teks narasi, judul, kesimpulan, temuan teknis node, dan label antarmuka (UI) wajib mematuhi kaidah penulisan ilmiah formal (EYD Edisi V / PUEBI):

1. **Prinsip Anti-Calque (Pencegahan Terjemahan Harfiah / Mesin):**
   - Gunakan **"penurunan performa yang signifikan"** atau **"degradasi performa"** (bukan *"kerugian signifikan"* atau *"loss"*).
   - Gunakan **"selang kepercayaan 95% mencakup nilai nol (tidak signifikan secara statistik)"** (bukan *"CI95 memuat nol"*).
   - Gunakan **"tidak menunjukkan keunggulan performa"** atau **"mengalami penurunan"** (bukan *"tidak pernah menang"* atau *"kalah"*).
   - Gunakan **"disimpulkan sebagai peningkatan"** atau **"terbukti meningkatkan"** (bukan *"menyebut kenaikan"*).
   - Gunakan **"kemunculan objek (*appearance*)"** (bukan *"appearance"* mentah).

2. **Notasi Matematika, Statistika, dan Angka:**
   - **Tanda Desimal & Ribuan:** Gunakan tanda koma (`,`) untuk desimal (misal `0,6038`) dan tanda titik (`.`) untuk pemisah ribuan (misal `3.992 citra`, `2.612 objek`).
   - **Tanda Minus Matematis:** Gunakan simbol minus asli `−` (*Unicode U+2212*), bukan tanda hubung keyboard biasa `-`. Contoh: `−0,0476`.
   - **Selang Kepercayaan (*Confidence Interval*):** Tuliskan dengan format `[min; max]` menggunakan kurung siku dan titik koma, contoh: `[−0,0270; +0,0739]`.
   - **Simbol Variabel:** Cetak miring simbol matematis/variabel seperti *$p$-value*, *$n$ sampel*, *IoU*, *$\Delta$ mAP*, *$M_{shuf}$*.
   - **Rentang Satuan:** Gunakan *en dash* (`–`) untuk rentang: `B1–B4`, `10–11 Agu 2026`.

3. **Taksonomi Padanan Istilah Teknis Baku (EYD V / KBBI):**
   - `detector` → **detektor**
   - `monocular / monocular-depth` → **depth monokular / monokular**
   - `classifier` → **pengklasifikasi / model pengklasifikasi**
   - `counting` → **pencacahan (*counting*)**
   - `screening` → **penyaringan awal (*screening*)**
   - `early stopping / early stop` → **penghentian dini (*early stopping*)**
   - `data leakage` → **kebocoran data (*data leakage*) / kebocoran partisi data**
   - `ground truth (GT)` → **nilai acuan kebenaran (*ground truth*) / data acuan riil**
   - `oracle` → **model batas atas teoretis (*oracle*)**
   - `ablation study` → **studi ablasi / uji eliminasi komponen**
   - `baseline` → **garis dasar pembanding (*baseline*) / model acuan**
   - `bounding box` → **kotak pembatas (*bounding box*)**
   - `fine-tuning` → **penyesuaian terarah (*fine-tuning*) / adaptasi model**
   - `spatial pooling` → **agregasi spasial (*spatial pooling*)**
   - `temporal shift` → **pergeseran temporal (*temporal shift*)**
   - `booster detector` → **modul penguat (*booster*) detektor**
   - `crop` → **citra terpotong (*crop*) / pemotongan objek**
   - `noise` → **variasi acak (*noise*) / derau**

4. **Konvensi Terminologi Antarmuka (UI):**
   - *Akar data* → **Dataset acuan**
   - *Node / jejak* → **Simpul eksperimen**
   - *Status bukti* → **Status validitas bukti**
   - *Filter bukti* → **Penyaringan bukti**
   - *Alasan lineage* → **Rasional relasi silsilah**
   - *Kesimpulan singkat* → **Kesimpulan eksekutif**
   - *Cerita kerja* → **Narasi metodologi & pembuktian**
   - *Yang dikerjakan* → **Rancangan eksperimen**
   - *Bukti yang ditemukan* → **Temuan empiris terukur**
   - *Keputusan setelahnya* → **Keputusan metodologis**
   - *Batas pembacaan* → **Batasan validitas & audit**
   - *Angka utama* → **Metrik kuantitatif utama**
   - *Penjelasan teknis* → **Catatan sintesis teknis**
   - *Arti istilah* → **Glosarium istilah teknis**
   - *File pendukung* → **Artefak data pendukung**

5. **Struktur Narasi Empat Bagian (Lembar Bukti):**
   - **Rancangan Eksperimen:** Ringkasan desain eksperimen, konfigurasi input/model, dan komparasi yang dijalankan.
   - **Temuan Empiris Terukur:** Ringkasan kuantitatif terukur dengan signifikansi statistik (*confidence interval*, *p-value*, *bootstrap*).
   - **Keputusan Metodologis:** Implikasi terhadap kelanjutan arah riset.
   - **Batasan Validitas & Audit:** Peringatan audit, asumsi kontrol yang belum tuntas, atau batasan generalisasi.

`docs/MAINTENANCE.md` memuat prosedur rinci: template node siap salin, aturan lineage, penambahan dataset, penghapusan node, dan checklist publikasi.

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
