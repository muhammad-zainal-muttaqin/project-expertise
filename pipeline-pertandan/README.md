# Subproyek: Pipeline Inferensi Per-Tandan Buah Fisik (Multi-View Linking)

Subproyek mandiri dari `project-expertise` yang mengalihkan paradigma inferensi komputer visi: dari **pendeteksian kotak pembatas per-citra** menjadi **identifikasi entitas tandan buah fisik per pohon kelapa sawit**.

Proses kerja mencakup: pendeteksian per sisi pandang pohon, penautan asosiasi lintas-sisi (*multi-view linking*), dan penetapan keputusan klasifikasi kematangan tunggal per entitas buah fisik.

---

## 1. Status Gerbang Kelayakan Ilmiah (Gates G0–G3)

| Gerbang Verifikasi | Deskripsi Pengujian | Status Hasil | Metrik Penentu |
|---|---|---|---|
| **Gerbang G0** | Evaluasi Nilai Penggabungan (*Oracle Linking*) | **LOLOS** | Peningkatan akurasi $+4,36\text{ pp}$ (CI95 $[+2,33; +6,25]$). |
| **Gerbang G1** | Evaluasi Kualitas Model Penaut (*Linker*) | **LOLOS** | Skor $F1 = \mathbf{0,6486}$ (setelah penemuan prior arah putar searah jarum jam / *clockwise*, PT-E-008). |
| **Gerbang G2** | Evaluasi *End-to-End* Tanpa Data Acuan Kebenaran | **LOLOS** | Akurasi klasifikasi end-to-end $\minus 1,81\text{ pp}$ vs oracle (memenuhi toleransi $\le 2,0\text{ pp}$). |
| **Gerbang G3** | Pencacahan Berbasis Klaster Graf Murni | **GUGUR** | Macro-$MAE = 3,4610$ vs $1,0542$ pada *Ridge +* $F_{\text{all}}$ (pencacahan klaster graf kalah presisi dari estimator regresi). |

---

## 2. Fondasi Empiris Subproyek

1. **Peningkatan Daya Tangkap (*Recall*) Fisik (+14,49 pp)**:
   Perubahan satuan evaluasi ke tingkat tandan fisik meningkatkan daya tangkap dari $63,36\%$ per kemunculan citra menjadi **$77,85\%$ per entitas tandan** (pada detektor YOLO26l sel 5, ambang keyakinan $0,25$).
2. **Ketersediaan 9.823 Tandan Teranotasi Lintas-Sisi**:
   Data acuan kebenaran (*ground truth*) memuat 9.823 tandan unik dengan identitas terhubung antar-sudut pandang.
3. **Karakteristik Agregasi Spasial**:
   Sebanyak $85,5\%$ tandan yang terhubung melintasi tepat 2 sudut pandang. Kondisi ini menyebabkan aturan pemungutan suara mayoritas diskret sering mengalami hasil seri, sehingga **aturan agregasi probabilitas kontinu ($R4$) wajib diterapkan**.
4. **Terobosan Prior Rotasi Searah Jarum Jam (*Clockwise*, PT-E-008)**:
   Fotografer lapangan merekam 4 sisi pohon secara konsisten searah jarum jam. Pemanfaatan ekspektasi pergeseran posisi horizontal bertanda memangkas ruang pencarian pasangan kandidat dan melipatgandakan performa model penaut.

---

## 3. Struktur Direktori dan Skrip Terkait

```
pipeline-pertandan/
├── README.md                  ← Dokumen pengantar ini
├── CLAUDE.md                  Pedoman operasional subproyek
├── EKSPERIMEN.md              Log append-only PT-E-000 s.d. PT-E-036
├── STATUS.md                  Status verifikasi dan rekapitulasi gerbang
├── docs/
│   ├── HASIL.md               Sintesis komprehensif temuan per-tandan
│   ├── PROPOSAL.md            Proposal metodologis awal dan gerbang falsifikasi
│   └── RINGKASAN-SESI-2026-08-18.md  Sintesis batas optimasi DAMIMAS
├── scripts/
│   └── probe_penautan_953.py  Probe verifikasi asosiasi topologis
└── results/
    └── probe_penautan_953.json
```

---

## 4. Aset Bersama dari Repositori Utama

* **Model Detektor Dasar**: [`../models/yolo26l_e60_i1280_v2repro/best.pt`](file:///D:/Work/Assisten-Dosen/project-expertise/models/yolo26l_e60_i1280_v2repro/best.pt) (Sel 5, dilatih pada 716 pohon kanonik).
* **Dump Prediksi Uji**: [`../results/pred_sel5_953_rgb_test.npz`](file:///D:/Work/Assisten-Dosen/project-expertise/results/pred_sel5_953_rgb_test.npz).
* **Lingkungan Python & Skrip Evaluasi**: [`../scripts/bootstrap_ci.py`](file:///D:/Work/Assisten-Dosen/project-expertise/scripts/bootstrap_ci.py).
