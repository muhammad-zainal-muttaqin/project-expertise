# Sintesis Temuan Metodologis & Batasan Optimasi: Sesi Evaluasi DAMIMAS (18 Agustus 2026)

Dokumen ini mendokumentasikan sintesis menyeluruh hasil evaluasi, analisis batas optimasi, dan pembelajaran metodologis pada sub-populasi varietas **DAMIMAS**.

---

## 1. Evaluasi Kinerja Terhadap Target

Eksperimen klasifikasi per-tandan pada varietas DAMIMAS menghasilkan akurasi uji akhir sebesar **$74,39\%$** (selisih $\minus 5,6\text{ pp}$ dari target awal $80,00\%$).

| Tingkatan Model / Konfigurasi | Akurasi Uji Terukur | Catatan Status |
|---|---|---|
| Model Tunggal Acuan (*Baseline*) | 0,7378 | ConvNeXt Residual 128 |
| **Ensembel Terbobot Terkunci Validasi** | **0,7439** | **Hasil Sah Terverifikasi (CI95 $[\minus 0,15; +3,55]$)** |
| Plafon Rerata Terbobot Teoretis (*Fitted on Test*) | 0,7523 | Batas atas komputasi bobot linier |
| Model Batas Atas Teoretis (*Oracle Model Selection*) | 0,8739 | Terdapat setidaknya 1 anggota ensembel yang benar |
| Target Desain Awal | 0,8000 | Selisih $\minus 5,6\text{ pp}$ |

---

## 2. Kuantifikasi Kesenjangan Menuju Target

Dekomposisi populasi data uji memetakan sumber hambatan secara presisi:

| Karakteristik Wilayah Prediksi | Porsi Sampel | Akurasi Rerata |
|---|---|---|
| Wilayah Seluruh Anggota **Sepakat** | 64,7% | **81,92%** |
| Wilayah Anggota **Berselisih Pendapat** | 35,3% | **61,21%** |

Evaluasi strategi pemungutan suara pada wilayah berselisih:
* **Model Batas Atas Teoretis (*Oracle*)**: Akurasi mencapai **$97,41\%$** (membuktikan bahwa sinyal informasi kelas yang benar eksis di dalam bank model).
* **Rata-rata Probabilitas**: Akurasi $61,21\%$.
* **Pemilihan Berbasis Keyakinan Tertinggi (*Max Confidence*)**: Akurasi **$57,11\%$** (lebih rendah daripada rata-rata sederhana).
* **Tebakan Acak Proporsional**: Akurasi $54,35\%$.

Untuk mencapai target akurasi $80,0\%$, akurasi pada wilayah berselisih wajib dinaikkan menjadi **$76,5\%$** ($+15,3\text{ pp}$). Tiga metode eksplorasi pemilihan model diuji dan menghasilkan kesimpulan:
1. **Optimasi Bobot Global (PT-E-034)**: Plafon linier mentok pada $75,23\%$ (hanya $+0,84\text{ pp}$ di atas rata-rata sederhana).
2. **Seleksi Berbasis Keyakinan (PT-E-035)**: Korelasi antara tingkat keyakinan (*confidence*) dan kebenaran prediksi sangat lemah ($r = \mathbf{+0,1185}$).
3. **Pola Perselisihan Graf (PT-E-036)**: Model *gradient boosting* mengalami penurunan performa $\minus 3,59\text{ pp}$ pada validasi silang.

---

## 3. Temuan Metodologis Lintas-Korpus

### 3.1 Resolusi Pergeseran Domain Pelatihan Penaut (PT-E-017)
Model penaut (*linker*) yang dilatih pada pasangan kotak data acuan kebenaran (*ground truth*) mengalami *domain shift* ekstrem saat diuji pada kotak deteksi nyata ($AUC = 0,9508 \to 0,5868$, mendekati tebakan acak). Pelatihan ulang langsung pada pasangan deteksi nyata meningkatkan skor $F1$ dari $0,1492$ menjadi **$0,3080$** ($+15,88\text{ pp}$), dan penambahan GNN meningkatkan $F1$ menjadi **$0,3788$** ($+7,08\text{ pp}$).

### 3.2 Sifat Komplementer Detektor dan Pengklasifikasi (PT-E-018 & PT-E-019)
Penggabungan ensembel pengklasifikasi C1+C2 menghasilkan akurasi **$74,64\%$**, mematahkan asumsi batas lama $73,60\%$. Penaut multi-tampak dan pengklasifikasi ensembel bersifat saling melengkapi (*complementary*): ensembel memperbaiki tandan satu-tampak, sedangkan penaut memindahkan tandan ke wilayah multi-tampak yang ditangani oleh aturan ordinal $R4$.

### 3.3 Superioritas Loss Ordinal CORN terhadap CORAL (PT-E-030)
Pada resep pelatihan yang identik, fungsi *loss* CORAL mengalami keruntuhan struktural ($33,05\%$) akibat keterbatasan pembagian bobot (*weight-sharing*), sementara **CORN mencapai akurasi uji $69,83\%$** ($+36,8\text{ pp}$).

### 3.4 Karakteristik Konvergensi RF-DETR-L DAMIMAS (PT-E-032)
Model RF-DETR-L mencapai puncak performa validasi pada **epoch 5 ($ema\_mAP50 = 0,5830$)**, lalu mengalami degradasi konsisten hingga epoch 59 ($0,4885$, penurunan $\minus 9,46\text{ pp}$). Disimpulkan bahwa pelatihan RF-DETR-L pada korpus spesifik cukup dibatasi $\approx 15\text{ epoch}$ dengan penghentian dini ketat.

---

## 4. Kaidah Metodologis yang Mengikat

1. **Pencegahan Kebocoran Validasi Halus (*Subtle Validation Leakage*)**: Parameter aturan keputusan (seperti ambang $\tau$) wajib dipilih melalui validasi silang internal di dalam partisi validasi untuk mencegah *overfitting* parameter.
2. **Keterbatasan Partisi Validasi 86 Pohon**: Kenaikan performa di bawah $\sim 2\text{ pp}$ pada validasi 86 pohon tidak dapat dibedakan dari variasi acak seleksi model.
3. **Protokol Uji Tertutup (*Strict Test Evaluation*)**: Himpunan data uji hanya dievaluasi setelah konfigurasi ensembel terkunci penuh dari partisi validasi.
