# Atlas Metrik: Klasifikasi Tingkat Kematangan Citra Terpotong (*Crop*)

Dokumen ini merangkum seluruh hasil eksperimen klasifikasi kematangan tandan buah segar kelapa sawit 4 kelas ordinal (**B1**: Lewat Matang / Siap Panen, **B2**: Matang Optimal, **B3**: Matang Awal / Mengkal, **B4**: Mentah / Muda) menggunakan model pengklasifikasi terpisah pada wilayah objek terpotong (*bounding box crop*), fungsi rugi ordinal khusus (CORAL, CORN), evaluasi multi-tampak (C1–C3), serta aturan keputusan per tandan (R0–R4).

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

Sel `N/A — ...` berarti metrik tersebut tidak menjadi keluaran artefak yang cocok
untuk unit evaluasi pada baris itu (bukan angka nol dan bukan angka yang boleh
diisi dari eksperimen lain).

---

## 2. Tabel Komparasi Pengklasifikasi Kematangan Citra Terpotong (*Crop Classifier*)

| ID Eksperimen | Metode / Model Pengklasifikasi | Modalitas Masukan | Dataset & Partisi | Akurasi | Akurasi $\pm 1$ | MAE Ordinal | Macro-F1 | Status Bukti | Rujukan Artefak & Skrip |
|---|---|---|---|---:|---:|---:|---:|---|---|
| `PT-E-023` | Mixture-of-Experts Strict | RGB Crop + Konteks Pohon | DAMIMAS (Val OOF, bukan Test) | **0,7552** | **0,9935** | **0,2514** | **0,7434** | `VALID` | `pipeline-pertandan/results/damimas_moe_classifier.json` + `damimas_moe_classifier_pred.npz` — seluruh metrik baris ini adalah VAL/OOF (n=919); jangan dibaca sebagai TEST. ±1 dihitung dari label/prediksi OOF. |
| `PT-E-018` | Stacking Ensemble C1+C2+C3 | Multi-Representation | DAMIMAS (Test, n=1.404) | **0,7464** | **0,9979** | **0,2557** | **0,7098** | `VALID` | `pipeline-pertandan/results/pt_e_018_ensemble.json` (`ensemble.test`) — Akurasi dikoreksi dari `0,7240`; tiga metrik ordinal dan Macro-F1 diambil dari objek TEST yang sama. |
| `PT-E-029` | Weighted Average Ensemble | Stacking vs Soft-Vote | DAMIMAS (Test, n=1.316) | **0,7409** | **0,9970** | **0,2622** | **0,7048** | `VALID` | `pipeline-pertandan/results/pt_e_029_ensemble_kelas_damimas_pred.npz` + JSON — Akurasi JSON lama `0,7439` tidak cocok dengan dump TEST yang tersedia; tabel memakai dump n=1.316 secara konsisten. |
| `PT-E-033` | Bagged Ensemble Selection | Multi-Model Subsets | DAMIMAS (Test, n=1.316) | 0,7394 | **0,9962** | **0,2644** | **0,7192** | `FALSIFIED` | `pipeline-pertandan/results/pt_e_033_bagged_pred.npz` — metrik ordinal dihitung ulang dari dump TEST; Akurasi `0,7394` cocok pembulatan. |
| `PT-E-001` | R4 (agregasi ordinal, seluruh pool) | RGB Crop Acuan GT | DAMIMAS 953 (Test, n=1.269) | **0,7360** | **0,9984** | **0,2656** | **0,7084** | `VALID` | `pipeline-pertandan/results/pt_e_001_oracle.json` (`split.test.semua_pool.R4`) — seluruh 4 nilai lama diganti nilai JSON asli. |
| `PT-E-031` | Model Spesialis Per-Batas Kelas | Binary Boundary Heads | DAMIMAS (Test, n=1.316) | **0,7340** | **0,9954** | **0,2713** | **0,6988** | `FALSIFIED` | `pipeline-pertandan/results/pt_e_031_spesialis_batas_pred.npz` — Akurasi dikoreksi `0,7140`→`0,7340`; metrik ordinal dihitung dari dump TEST. |
| `PT-E-035` | Dynamic Ensemble Selection (DES) | Keyakinan Spasial | DAMIMAS (Test) | **0,7340** | **0,9962** | **0,2698** | **0,6968** | `FALSIFIED` | `pipeline-pertandan/results/pt_e_035_des_pred.npz` — Akurasi dikoreksi `0,7210`→`0,7340`; metrik ordinal dihitung dari dump TEST. |
| `PT-E-014` | ConvNeXt-Tiny + CE (seed 1, terbaik) | RGB Crop | DAMIMAS (Test, C3, n=1.404) | **0,7187** | **0,9957** | **0,2863** | **0,6857** | `VALID` | `pipeline-pertandan/results/pt_e_014_c_convnext_tiny_ce_s1.json` (`split.test.C3`) — nilai lama `0,7120` dikoreksi; Putusan asli SEBAGIAN DIKONFIRMASI. |
| `PT-E-036` | Gerbang Pola Perselisihan | Gating Dispute Network | DAMIMAS (Val, CV — TEST TIDAK DIBUKA) | **0,7062** | N/A — TEST tidak dibuka | N/A — TEST tidak dibuka | N/A — TEST tidak dibuka | `FALSIFIED` | `pipeline-pertandan/results/pt_e_036_gate.json` — hanya akurasi CV yang tersedia; TEST sengaja tidak dibuka. |
| `PT-E-030` | Loss Ordinal CORN | RGB Crop | DAMIMAS (Test, n=1.316) | **0,6983** | **0,9954** | **0,3062** | **0,6554** | `VALID` | `pipeline-pertandan/results/damimas_classifier_corn_s42_pred.npz` + JSON — ±1/MAE/Macro-F1 dihitung dari dump TEST; akurasi dikoreksi `0,7100`→`0,6983`. |
| `PT-E-012` | Pengklasifikasi Multi-Tampak C3 | RGB 4 Sudut Tampak | DAMIMAS (Test, n=1.404) | **0,6781** | N/A — tidak dilaporkan | **0,3369** | **0,6451** | `FALSIFIED` | `pipeline-pertandan/results/pt_e_012_c3.json` (`split.test.C3`) — C3 tidak menyimpan metrik ±1; Putusan asli DIPALSUKAN (C3 kalah dari C2 dan C1). |
| `V2-E-015` | Classifier crop (rata-rata 3 seed) | RGB | SawitMVC-Depth (352, Test) | **0,6309** | **0,9569** | **0,4163** | **0,5525** | `VALID` | `results/fase6_classifier.json` (`runs.sd{101,202,303}_rgb.test`, rata-rata 3 seed) — ±1, MAE, dan Macro-F1 dihitung dari 3 keluaran test; akurasi adalah rata-rata `.6309`. |
| `V2-E-044` | Pengklasifikasi C2 5-Epoch (val internal) | RGB Crop (Jitter 10%) | SawitMVC-YOLO 953 (Val internal) | **0,6217** | **0,9932** | **0,385 (MAE kelas)** | **0,6296** | `VALID` | `results/remote_eval_2026-08-27/classifier_c2/remote953_c2_rgb_5ep_jitter10.json` — seluruh 4 nilai lama diganti angka validasi internal epoch terbaik yang benar. |
| `V2-E-016` | ResNet18 (Crop Head) | RGB+Depth Invers (4-ch) | SawitMVC-Depth (352, Test) | **0,6106** | **0,9561** | **0,4374** | **0,5432** | `FALSIFIED` | `results/fase6_classifier.json` (`ablasi_depth_multiseed.test`, rata-rata RGB+Depth 3 seed) — angka sekarang terverifikasi langsung; depth tidak mengungguli RGB (0,6106 vs 0,6309). |
| `PT-E-015` | Loss Ordinal CORAL (C2, resnet18) | RGB Crop | DAMIMAS (Test, n=1.404) | **0,5686** | **0,9679** | **0,5271** | **0,4865** | `FALSIFIED` | `pipeline-pertandan/results/pt_e_014_c_resnet18_coral.json` (`split.test.C2.R0`) — CE pembanding test 0,6522 (CORAL −8,36pp di TEST; val seed0 sempat +2,35pp — arah val/test berlawanan). Seluruh 4 nilai lama tidak cocok sumber. |
| `PT-E-000` | Probe kelayakan (bukan klasifikasi) | RGB Crop | DAMIMAS 953 (Val) | N/A — bukan classifier | N/A — bukan classifier | N/A — bukan classifier | N/A — bukan classifier | `INVALID` | `pipeline-pertandan/results/probe_penautan_953.json` — artefak hanya mengukur kelayakan penautan, bukan akurasi klasifikasi crop. |
| `PT-E-014` | EfficientNetV2-RW-S | RGB Crop | DAMIMAS (Test) | N/A — bukan crop classifier/test | N/A — bukan crop classifier/test | N/A — bukan crop classifier/test | N/A — bukan crop classifier/test | `INVALID` | `results/remote_eval_2026-08-28/validation_wave/reports/953_timm_efficientnetv2_rw_s_results_val.json` — hanya laporan pipeline VAL/matched-class; tidak ada evaluasi crop-classifier TEST yang cocok dengan baris ini. |
| `PT-E-014` | Swin-Tiny Transformer | RGB Crop | DAMIMAS (Test) | N/A — bukan crop classifier/test | N/A — bukan crop classifier/test | N/A — bukan crop classifier/test | N/A — bukan crop classifier/test | `INVALID` | `results/remote_eval_2026-08-28/validation_wave/reports/953_timm_swin_tiny_results_val.json` — hanya laporan pipeline VAL/matched-class; tidak ada evaluasi crop-classifier TEST yang cocok dengan baris ini. |
| `PT-E-024` | Propagasi Keyakinan Lintas-View | Softmax Lintas-View | DAMIMAS (Test) | N/A — metrik deteksi | N/A — metrik deteksi | N/A — metrik deteksi | N/A — metrik deteksi | `INVALID` | `results/damimas_propagasi_multiview.json` — artefak mengukur mAP deteksi/propagasi (`test.mAP50=0,5965`), bukan crop-classifier; baris dipertahankan hanya sebagai catatan audit salah kategori. |

---

## 3. Evaluasi Aturan Keputusan Tandan Fisik Multi-Tampak (R0–R4)

*Evaluasi aturan perataan multi-sisi pada klaster tandan fisik (Simpul `PT-E-001` & `PT-E-012`).*

| Aturan Keputusan | Deskripsi Mekanisme | Akurasi Tandan | Akurasi $\pm 1$ | MAE Ordinal | $\Delta$ vs R0 | Keterangan Ilmiah |
|---|---|---:|---:|---:|---:|---|
| **R4** | **Ekspektasi ordinal berbobot (*ordinal pooling*)** | **0,7360** | **0,9984** | **0,2656** | **$+2,38\text{ pp}$** | **Aturan terbaik:** memperhitungkan jarak topologi kelas ordinal |
| **R3** | Pemilihan sudut dengan keyakinan maksimum | 0,7352 | 0,9968 | 0,2679 | $+2,30\text{ pp}$ | Terdistorsi oleh prediksi positif palsu berkeyakinan tinggi |
| **R2** | *Majority Voting* kelas prediksi | 0,7313 | 0,9945 | 0,2742 | $+1,91\text{ pp}$ | Rentan terhadap perselisihan (*dispute*) imbang |
| **R1** | Rata-rata probabilitas linier lintas-sisi | 0,7273 | 0,9961 | 0,2766 | $+1,51\text{ pp}$ | Mengurangi derau oklusi daun pada sudut pandang tertentu |
| **R0** | *Single-view baseline* tanpa kalibrasi | 0,7122 | 0,9961 | 0,2766 | Garis Dasar | Prediksi citra tunggal tanpa memanfaatkan sudut tampak lain |
| **R0cal** | Rekalibrasi probabilitas Isotonik | 0,7100 | 0,9984 | 0,2734 | $-0,22\text{ pp}$ | Rekalibrasi TIDAK menaikkan akurasi tandan pada agregasi test penuh (nilai lama seluruh baris tabel ini tertukar/salah — lihat catatan di bawah) |

> **Koreksi menyeluruh tabel ini**: seluruh baris R0–R4 lama tidak cocok satu sumber JSON pun yang bisa ditemukan (angka R0 lama `0,6820` sebenarnya milik baris `PT-E-000` yang salah kategori, dan R4 lama `0,7439` sebenarnya milik `PT-E-029`, eksperimen berbeda). Nilai pengganti di atas diambil langsung dari `pipeline-pertandan/results/pt_e_001_oracle.json` (`split.test.semua_pool.*`, n=1.269 pool, PT-E-001), progresi R0→R4 yang benar-benar koheren dari satu eksperimen yang sama. Kolom "$\Delta$ vs R0" dihitung ulang dari akurasi terkoreksi.

---

## Tambahan — Audit Forensik 6 September 2026 (`AF-E-003`, `AF-E-005`, `AF-E-009`, `AF-E-012`)

Rujukan penuh: `experiments/AUDIT-FORENSIK-2026-09-06.md`.

| ID Eksperimen | Metode / Model Pengklasifikasi | Modalitas Masukan | Dataset & Partisi | Akurasi | Akurasi $\pm 1$ | MAE Ordinal | Macro-F1 | Status Bukti | Rujukan Artefak & Skrip |
|---|---|---|---|---:|---:|---:|---:|---|---|
| `AF-E-003` | Geometri dalam pohon saja, **tanpa satu piksel pun** | Peringkat vertikal & ukuran (8 fitur) | 953 Uji (1.402 tandan) | 0,5713 | 0,9429 | N/A — tidak dilaporkan | 0,4729 | `VALID` | `scripts/audit_forensik/an6_structure.py`; garis dasar kelas mayoritas `0,5300` |
| `AF-E-003` | Penataan monoton + komposisi kelas *oracle* | Urutan vertikal dalam pohon | 953 Uji (1.402 tandan) | 0,6912 | N/A — tidak dilaporkan | N/A — tidak dilaporkan | **0,6237** | `VALID` | `an7_monotone.py`; memakai komposisi acuan, jadi batas atas |
| `AF-E-005` | ConvNeXt-Tiny, citra terpotong cincin `1,6×` | RGB | 953 Uji (2.612 *crop*) | 0,6612 | N/A — tidak dilaporkan | N/A — tidak dilaporkan | N/A — tidak dilaporkan | `VALID` | `exp_train.py`; berada di dalam pita `0,62`–`0,70` proyek |
| `AF-E-009` | Penampilan saja, tingkat tandan, deteksi nyata | RGB | 953 Uji (2.466 deteksi terpasangkan) | 0,6951 | 0,9943 | N/A — tidak dilaporkan | 0,6470 | `VALID` | `e4b_fuse.py` |
| `AF-E-009` | Penampilan + struktur dalam pohon | RGB + geometri | 953 Uji (2.466 deteksi terpasangkan) | **0,6963** | 0,9935 | N/A — tidak dilaporkan | **0,6528** | `VALID` | Idem; `w = 0,8` ditala pada VAL; kenaikan `+0,0058` tanpa selang kepercayaan |
| `AF-E-012` | Kepala ordinal CORN, agregasi antartampak | RGB, skor kontinu | 953 Uji (132 pohon empat sisi) | 0,7161 | **0,9946** | N/A — tidak dilaporkan | **0,6692** | `VALID` | `panen_final.py`; makro-F1 melampaui GSP terkunci `0,6034` |
