# Atlas Metrik: Klasifikasi Tingkat Kematangan Citra Terpotong (*Crop*)

Dokumen ini merangkum seluruh hasil eksperimen klasifikasi kematangan tandan buah segar kelapa sawit 4 kelas ordinal (**B1**: Mentah, **B2**: Mengkal/Matang Awal, **B3**: Matang, **B4**: Lewat Matang) menggunakan model pengklasifikasi terpisah pada wilayah objek terpotong (*bounding box crop*), fungsi rugi ordinal khusus (CORAL, CORN), evaluasi multi-tampak (C1–C3), serta aturan keputusan per tandan (R0–R4).

---

## 1. Panduan Pembacaan & Definisi Metrik

1. **Akurasi (*Accuracy*)**: Proporsi prediksi kelas yang tepat sama dengan label acuan kebenaran (*ground truth*).
2. **Akurasi $\pm 1$ (*Class $\pm 1$ Acc*)**: Proporsi prediksi dengan selisih tingkat kematangan maksimal 1 tingkat ($|\hat{y} - y| \le 1$).
3. **MAE Ordinal**: Rata-rata galat absolut jarak tingkatan kelas ordinal: $\frac{1}{N}\sum |\hat{y} - y|$.
4. **Macro-F1**: Rata-rata harmonik tidak berbobot dari F1-score keempat kelas.
5. **Aturan Keputusan Per Tandan Fisik (R0 s.d. R4)**:
   - **R0**: Prediksi mentah kemunculan tunggal (*single-view baseline*).
   - **R0cal**: Prediksi kemunculan tunggal setelah rekalibrasi probabilitas Isotonik / Platt.
   - **R1**: Rata-rata probabilitas antar-sudut tampak pohon.
   - **R2**: Pemilihan kelas mayoritas (*majority voting*) antar-sudut tampak.
   - **R3**: Aturan keyakinan maksimum (*maximum confidence rule*).
   - **R4**: Agregasi ekspektasi ordinal berbobot (*ordinal expectation pooling*).

---

## 2. Tabel Komparasi Pengklasifikasi Kematangan Citra Terpotong (*Crop Classifier*)

| ID Eksperimen | Metode / Model Pengklasifikasi | Modalitas Masukan | Dataset & Partisi | Akurasi | Akurasi $\pm 1$ | MAE Ordinal | Macro-F1 | Status Bukti | Rujukan Artefak & Skrip |
|---|---|---|---|---:|---:|---:|---:|---|---|
| `V2-E-015` | Classifier crop (rata-rata 3 seed) | RGB | SawitMVC-Depth (352, Test) | **0,6309** | ⚠ TBD | ⚠ TBD | ⚠ TBD | `VALID` | `runs_fase6/sd{101,202,303}_rgb/hasil.json` — nilai lama tidak cocok sumber; ±1/MAE/Macro-F1 tidak dilaporkan entri asli. |
| `V2-E-016` | ResNet18 (Crop Head) | RGB+Depth Invers (4-ch) | SawitMVC-Depth (352, Test) | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TBD | ⚠ TBD | ⚠ TBD | `FALSIFIED` | `runs_fase6/sd*/hasil.json` — entri asli hanya mencatat Verdict FALSIFIED tanpa tabel angka mandiri. |
| `PT-E-000` | Probe kelayakan (bukan klasifikasi) | RGB Crop | DAMIMAS 953 (Val) | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TBD | ⚠ TBD | ⚠ TBD | `VALID` | `results/probe_penautan_953.json` — entri asli TIDAK mengukur akurasi klasifikasi crop sama sekali (murni probe kelayakan penautan); baris ini salah kategori, bukan hanya salah angka. |
| `PT-E-001` | R4 (agregasi ordinal, seluruh pool) | RGB Crop Acuan GT | DAMIMAS 953 (Test, n=1.269) | **0,7360** | **0,9984** | **0,2656** | **0,7084** | `VALID` | `pipeline-pertandan/results/pt_e_001_oracle.json` (`split.test.semua_pool.R4`) — seluruh 4 nilai lama diganti nilai JSON asli. |
| `PT-E-012` | Pengklasifikasi Multi-Tampak C3 | RGB 4 Sudut Tampak | DAMIMAS (Test, n=1.404) | **0,6781** | ⚠ TIDAK-BISA-DIVERIFIKASI | **0,3369** | **0,6451** | `FALSIFIED` | `pipeline-pertandan/results/pt_e_012_c3.json` (`split.test.C3`) — Putusan asli DIPALSUKAN (C3 kalah dari C1/C2), bukan SUPERSEDED. |
| `PT-E-014` | ConvNeXt-Tiny + CE (seed 1, terbaik) | RGB Crop | DAMIMAS (Test, C3) | **0,7187** | ⚠ TBD | ⚠ TBD | ⚠ TBD | `VALID` | `pipeline-pertandan/results/pt_e_014_c_convnext_tiny_ce_s1.json` — nilai lama `0,7120` mendekati tapi tidak presisi; Putusan asli SEBAGIAN DIKONFIRMASI (belum kalahkan C1=0,7208). |
| `PT-E-014` | EfficientNetV2-RW-S | RGB Crop | DAMIMAS (Test) | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TBD | ⚠ TBD | ⚠ TBD | `VALID` | **Tidak ditemukan bukti arsitektur ini pernah diuji** — entri asli PT-E-014 hanya menguji backbone `resnet18` dan `convnext_tiny`; baris ini tampak fabrikasi murni (arsitektur yang tidak pernah dijalankan). |
| `PT-E-014` | Swin-Tiny Transformer | RGB Crop | DAMIMAS (Test) | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TBD | ⚠ TBD | ⚠ TBD | `VALID` | **Tidak ditemukan bukti arsitektur ini pernah diuji** — sama seperti baris EfficientNetV2 di atas; PT-E-014 asli hanya menguji `resnet18`/`convnext_tiny`. |
| `PT-E-015` | Loss Ordinal CORAL (C2, resnet18) | RGB Crop | DAMIMAS (Test, n=1.404) | **0,5686** | **0,9679** | **0,5271** | **0,4865** | `FALSIFIED` | `pipeline-pertandan/results/pt_e_014_c_resnet18_coral.json` (`split.test.C2.R0`) — CE pembanding test 0,6522 (CORAL −8,36pp di TEST; val seed0 sempat +2,35pp — arah val/test berlawanan). Seluruh 4 nilai lama tidak cocok sumber. |
| `PT-E-018` | Stacking Ensemble C1+C2+C3 | Multi-Representation | DAMIMAS (Test, n=138 pohon) | **0,7464** | ⚠ TBD | ⚠ TBD | ⚠ TBD | `VALID` | `pipeline-pertandan/results/pt_e_018_ensemble.json` — Akurasi dikoreksi dari `0,7240`; $\Delta+2,56$pp CI95 $[+0,52;+4,53]$ vs C1 tunggal. |
| `PT-E-023` | Mixture-of-Experts Strict | RGB Crop + Konteks Pohon | DAMIMAS (Val OOF, bukan Test) | **0,7552** | ⚠ TBD | **0,2514** | **0,7434** | `VALID` | `pipeline-pertandan/results/damimas_moe_classifier.json` (`.val_oof.*`) — hanya angka VAL/OOF yang ditemukan di JSON level atas, bukan TEST seperti diklaim; seluruh 4 nilai lama tidak cocok. |
| `PT-E-024` | Propagasi Keyakinan Lintas-View | Softmax Lintas-View | DAMIMAS (Test) | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TBD | ⚠ TBD | ⚠ TBD | `VALID` | `results/damimas_propagasi_multiview.json` — Putusan asli "DITERIMA" (kualitatif, gain di semua kelas & 3 metrik), tapi angka `0,7350` tidak ditemukan pada pemeriksaan awal JSON; perlu telusur lebih dalam. |
| `PT-E-029` | Weighted Average Ensemble | Stacking vs Soft-Vote | DAMIMAS (Test, n=1.316) | **0,7439** | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | **0,7048** | `VALID` | `pipeline-pertandan/results/pt_e_029_ensemble_kelas_damimas.json` — Akurasi cocok sumber; Macro-F1 dikoreksi `0,7290`→`0,7048`. |
| `PT-E-030` | Loss Ordinal CORN | RGB Crop | DAMIMAS (Test, n=1.316) | **0,6983** | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | ⚠ TIDAK-BISA-DIVERIFIKASI | `VALID` | `pipeline-pertandan/results/damimas_classifier_corn_s42.json` (`test_akurasi`) — Akurasi dikoreksi `0,7100`→`0,6983` (val 0,7095); ±1/MAE/Macro-F1 belum ditemukan di level JSON yang dicek. |
| `PT-E-031` | Model Spesialis Per-Batas Kelas | Binary Boundary Heads | DAMIMAS (Test, n=1.316) | **0,7340** | ⚠ TBD | ⚠ TBD | **0,6988** | `FALSIFIED` | `pipeline-pertandan/results/pt_e_031_spesialis_batas.json` — Akurasi dikoreksi `0,7140`→`0,7340` (typo digit); acuan PT-E-029 0,7356, $\Delta-0,15$pp. |
| `PT-E-033` | Bagged Ensemble Selection | Multi-Model Subsets | DAMIMAS (Test, n=1.316) | 0,7394 | ⚠ TBD | ⚠ TBD | ⚠ TBD | `FALSIFIED` | `pipeline-pertandan/results/pt_e_033_bagged.json` — Akurasi `0,7390` mendekati sumber (`0,7394`, selisih pembulatan, dianggap OK). |
| `PT-E-035` | Dynamic Ensemble Selection (DES) | Keyakinan Spasial | DAMIMAS (Test) | **0,7340** | ⚠ TBD | ⚠ TBD | ⚠ TBD | `FALSIFIED` | `pipeline-pertandan/results/pt_e_035_des.json` — Akurasi dikoreksi `0,7210`→`0,7340` ($\Delta-0,68$pp vs PT-E-029). |
| `PT-E-036` | Gerbang Pola Perselisihan | Gating Dispute Network | DAMIMAS (Val, CV — TEST TIDAK DIBUKA) | **0,7062** | ⚠ TBD | ⚠ TBD | ⚠ TBD | `FALSIFIED` | `pipeline-pertandan/results/pt_e_036_gate.json` — Akurasi dikoreksi `0,7280`→`0,7062` (vs rata-rata probabilitas CV 0,7421); kolom Dataset lama "Test" keliru, TEST sengaja tidak dibuka. |
| `V2-E-044` | Pengklasifikasi C2 5-Epoch (val internal) | RGB Crop (Jitter 10%) | SawitMVC-YOLO 953 (Val internal) | **0,6217** | **0,9932** | **0,385 (MAE kelas)** | **0,6296** | `VALID` | `results/remote_eval_2026-08-27/classifier_c2/remote953_c2_rgb_5ep_jitter10.json` — seluruh 4 nilai lama diganti angka validasi internal epoch terbaik yang benar. |

---

## 3. Evaluasi Aturan Keputusan Tandan Fisik Multi-Tampak (R0–R4)

*Evaluasi aturan perataan multi-sisi pada klaster tandan fisik (Simpul `PT-E-001` & `PT-E-012`).*

| Aturan Keputusan | Deskripsi Mekanisme | Akurasi Tandan | Akurasi $\pm 1$ | MAE Ordinal | $\Delta$ vs R0 | Keterangan Ilmiah |
|---|---|---:|---:|---:|---:|---|
| **R0** | *Single-view baseline* tanpa kalibrasi | 0,7122 | 0,9961 | 0,2766 | Garis Dasar | Prediksi citra tunggal tanpa memanfaatkan sudut tampak lain |
| **R0cal** | Rekalibrasi probabilitas Isotonik | 0,7100 | 0,9984 | 0,2734 | $-0,22\text{ pp}$ | Rekalibrasi TIDAK menaikkan akurasi tandan pada agregasi test penuh (nilai lama seluruh baris tabel ini tertukar/salah — lihat catatan di bawah) |
| **R1** | Rata-rata probabilitas linier lintas-sisi | 0,7273 | 0,9961 | 0,2766 | $+1,51\text{ pp}$ | Mengurangi derau oklusi daun pada sudut pandang tertentu |
| **R2** | *Majority Voting* kelas prediksi | 0,7313 | 0,9945 | 0,2742 | $+1,91\text{ pp}$ | Rentan terhadap perselisihan (*dispute*) imbang |
| **R3** | Pemilihan sudut dengan keyakinan maksimum | 0,7352 | 0,9968 | 0,2679 | $+2,30\text{ pp}$ | Terdistorsi oleh prediksi positif palsu berkeyakinan tinggi |
| **R4** | **Ekspektasi ordinal berbobot (*ordinal pooling*)** | **0,7360** | **0,9984** | **0,2656** | **$+2,38\text{ pp}$** | **Aturan terbaik:** memperhitungkan jarak topologi kelas ordinal |

> **Koreksi menyeluruh tabel ini**: seluruh baris R0–R4 lama tidak cocok satu sumber JSON pun yang bisa ditemukan (angka R0 lama `0,6820` sebenarnya milik baris `PT-E-000` yang salah kategori, dan R4 lama `0,7439` sebenarnya milik `PT-E-029`, eksperimen berbeda). Nilai pengganti di atas diambil langsung dari `pipeline-pertandan/results/pt_e_001_oracle.json` (`split.test.semua_pool.*`, n=1.269 pool, PT-E-001), progresi R0→R4 yang benar-benar koheren dari satu eksperimen yang sama. Kolom "$\Delta$ vs R0" dihitung ulang dari akurasi terkoreksi.
