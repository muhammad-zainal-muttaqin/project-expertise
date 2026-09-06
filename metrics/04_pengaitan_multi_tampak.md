# Atlas Metrik: Pengaitan Multi-Tampak (*Association & Clustering Linker*)

Dokumen ini merangkum seluruh eksperimen penaut spasial (*spatial linker*) untuk mengaitkan kemunculan kotak pembatas dari 4 sudut pandang kanonik (Utara, Selatan, Barat, Timur) menjadi klaster tandan fisik yang utuh. Modul yang dievaluasi mencakup algoritma Hungarian dengan *prior* rotasi, *Union-Find*, *Relational Graph Neural Network* (GNN), *Global Linker* khusus DAMIMAS, serta *Graph Shortest Path* (GSP) Linker.

---

## 1. Panduan Pembacaan & Definisi Metrik

1. **Presisi / Recall / F1 Pasangan (*Pair Metrics*)**: Menilai kebenaran hubungan biner antara dua kotak deteksi dari dua sudut pandang berbeda apakah berasal dari objek fisik yang sama.
2. **Adjusted Rand Index (ARI)**: Tingkat kesamaan partisi pengelompokan (*clustering partition similarity*) antara klaster prediksi dengan klaster nilai acuan kebenaran (*ground truth*), terkoreksi peluang acak ($[-1; +1]$).
3. **Cakupan Tandan Multi-Sisi (*Multi-View Coverage*)**: Persentase tandan fisik yang terlihat dari $>1$ sudut pandang yang berhasil dipersatukan ke dalam satu klaster secara utuh.
4. **Fraksi Pool Palsu (*False Pool Fraction*)**: Proporsi klaster multi-anggota yang seluruh anggotanya merupakan deteksi positif palsu (*false positive*).
5. **AUC Val (*Linker ROC-AUC*)**: Kemampuan pembeda model relasional dalam mengklasifikasikan pasangan benar versus salah.

Sel `N/A — ...` berarti metrik tidak dilaporkan atau tidak cocok dengan unit
baris. Angka berlabel `(fisik)` adalah metrik *physical cluster* yang berasal
dari pipeline end-to-end, bukan metrik pasangan/ARI; keduanya tidak boleh
dibandingkan langsung.

---

## 2. Tabel Master Komparasi Modul Penaut Spasial

| ID Eksperimen | Algoritma / Modul Penaut | Ruang Masukan | Presisi Pasangan | Recall Pasangan | F1 Pasangan | ARI (*Rand Index*) | Cakupan Multi-Sisi | Fraksi Pool Palsu | AUC Val | Status Bukti | Rujukan Artefak |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `PT-E-001` | Oracle Association Linker (tautan GT) | Kotak Acuan (*Ground Truth*) | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 100,0% | N/A — oracle | N/A — oracle | `VALID` | `pipeline-pertandan/results/pt_e_001_oracle.json` — trivial benar secara definisi (tautan = GT); false-pool dan AUC tidak bermakna untuk oracle. |
| `V2-E-043` | Greedy Strict Linker (F1 fisik cluster) | WBF Proposal Gabungan, *Depth* | **0,8799 (fisik)** | **0,8390 (fisik)** | **0,8590 (fisik)** | N/A — bukan pair metric | N/A — tidak dilaporkan | N/A — tidak dilaporkan | N/A — tidak dilaporkan | `CORRECTED` | `experiments/EKSPERIMEN.md` (V2-E-043) — angka P/R/F1 adalah **physical cluster**, bukan pairing/ARI murni. |
| `Wave-V2 Depth` | GSP Linker (profil terkunci *depth*) | Multi-View, *test-locked* | **0,8926 (fisik)** | **0,8175 (fisik)** | **0,8534 (fisik)** | N/A — bukan pair metric | N/A — tidak dilaporkan | N/A — tidak dilaporkan | N/A — tidak dilaporkan | `CORRECTED` | `results/remote_eval_2026-08-28/gsp_artifacts/depth/results_test_locked.json` — P/R/F1 berasal dari `metrics.physical_detection`, bukan pair metrics. |
| `Wave-V2 953` | Hungarian+UF Anchor (profil terkunci 953) | Multi-View, *test-locked* | **0,8444 (fisik)** | **0,8331 (fisik)** | **0,8387 (fisik)** | N/A — bukan pair metric | N/A — tidak dilaporkan | N/A — tidak dilaporkan | N/A — tidak dilaporkan | `CORRECTED` | `results/remote_eval_2026-08-28/gsp_artifacts/953/results_test_locked.json` — profil terkunci 953 adalah Hungarian+union-find; P/R/F1 adalah physical detection. |
| `PT-E-010` | Konfigurasi Terbaik pada 352 | Korpus SawitMVC-Depth | **0,7261** | **0,6477** | **0,6847** | **0,6643** | N/A — cakupan multi-sisi tidak dilaporkan | N/A — tidak dilaporkan | N/A — tidak dilaporkan | `SEBAGIAN` | `pipeline-pertandan/results/pt_e_010_uji_352.json` (`penaut_kotak_GT.test`) — P/R/F1/ARI diambil dari test; `recall_per_tandan=0,8354` tidak disalahlabeli sebagai coverage. |
| `PT-E-008` | Rotation-Aware Signed Prior (varian E) | 4 Sisi Berurutan | **0,6679** | **0,6303** | **0,6486** | **0,5904** | N/A — tidak dilaporkan | N/A — tidak dilaporkan | **0,9502** | `VALID` | `pipeline-pertandan/results/pt_e_002_penaut.json` (`E_reid_plus_kelas_prediksi.test_sekali`) — P/R/F1/ARI/AUC diambil dari varian E yang sama; cakupan tidak dilaporkan. |
| `PT-E-016` | Relational GNN Linker | Kotak Acuan (*Ground Truth*) | **0,5204** | **0,6831** | **0,5907** | **0,5586** | **69,08% (coverage tandan)** | N/A — tidak dilaporkan | **0,9585** | `INCONCLUSIVE` | `pipeline-pertandan/results/pt_e_016_gnn.json` (`gnn.test`) — nilai lama berasal dari tabel lain; P/R/F1/ARI/cakupan/AUC sekarang memakai keluaran GNN TEST yang sama. |
| `PT-E-022` | Linker Global di atas Proposal | Multi-View Fusion | **0,5000** | **0,5354** | **0,5171** | **0,4684** | **62,29%** | **0,0383 (frac. pool palsu)** | **0,9391** | `VALID` | `pipeline-pertandan/results/damimas_linker_global_proposal_yolo.json` (`.test.*`, AUC val `hist_l31`) — angka pairing/fraction/AUC berasal dari konfigurasi yang sama. |
| `PT-E-020` | Global Linker Khusus DAMIMAS | Matriks Afinitas Global | **0,4359** | **0,4940** | **0,4631** | **0,4228** | **56,28%** *(cakupan\_atas\_terdeteksi)* | **0,0772** | N/A — ensemble mean_top3 | `VALID` | `pipeline-pertandan/results/damimas_linker_global.json` (`.test.*`) — false-pool diisi dari test; ensemble VAL memilih mean_top3 sehingga tidak punya satu AUC tunggal (kandidat terbaik `0,9435`). |
| `PT-E-002` | Varian A: geometri + `kelas_sama` (GT) | Kotak Acuan (*Ground Truth*) | **0,3742** | **0,5003** | **0,4282** | **0,3912** | N/A — tidak dilaporkan | N/A — tidak dilaporkan | **0,9301** | `FALSIFIED` | `pipeline-pertandan/results/pt_e_002_penaut.json` (`A_geometri_saja.test_sekali`, AUC val) — P/R yang sebelumnya kosong kini diambil dari varian A yang sama. |
| `PT-E-017` | Relational GNN Linker (varian C, di deteksi) | Ruang Prediksi Deteksi | **0,3915** | **0,3669** | **0,3788** | **0,3221** | **38,39%** | **0,040 (frac. pool palsu)** | **0,9422** | `VALID` | `pipeline-pertandan/results/pt_e_017_gnn_deteksi.json` (test 953) — seluruh 8 nilai lama diganti; nilai riil sekitar separuh dari klaim lama. |
| `PT-E-009` | Sapuan Ambang Keyakinan Deteksi | conf $= 0,10$ (terkunci) | **0,4672** | **0,1171** | **0,1873** | **0,1677** | N/A — `0,9038` adalah recall deteksi | N/A — tidak dilaporkan | N/A — tidak dilaporkan | `FALSIFIED` | `pipeline-pertandan/results/pt_e_009_sapu_conf.json` (`test.penautan`) — seluruh pair metrics dan ARI kini memakai test; cakupan yang tersedia adalah cakupan deteksi, bukan multi-view coverage. |
| `PT-E-011` | Koreksi Kepadatan Adegan (bukan linker) | Diagnostik Detektor, bukan Pairing | N/A — deteksi: 0,584 (953) / 0,639 (352) | N/A — deteksi: 0,823 (953) / 0,739 (352) | N/A — bukan pair metric | N/A — bukan pair metric | N/A — bukan pair metric | N/A — tidak dilaporkan | N/A — tidak dilaporkan | `CORRECTED` | `results/pred_skorpenuh{,_352}_test.npz` — angka deteksi dipertahankan di deskripsi, tetapi semua kolom pair dikosongkan secara eksplisit karena eksperimen ini bukan linker. |
| `PT-E-013` | Depth + Rekonstruksi 3D | Koordinat Spasial 3D | N/A — bukan linker | N/A — bukan linker | N/A — bukan linker | N/A — bukan linker | N/A — bukan linker | N/A — bukan linker | N/A — AUC fitur `0,6027` | `FALSIFIED` | Diukur langsung dari `/workspace/SawitMVC-Depth/depth/` — entri mengukur AUC per-fitur (termasuk ΔY `0,6027`), bukan F1/ARI linker; semua sel pair sengaja N/A. |
| `PT-E-021` | Kepala Proposal Fisik | Proposal Unik Spasial | N/A — tidak ada metrik pairing | N/A — tidak ada metrik pairing | N/A — tidak ada metrik pairing | N/A — tidak ada metrik pairing | N/A — tidak ada metrik pairing | N/A — tidak ada metrik pairing | N/A — tidak ada metrik pairing | `INVALID` | `results/damimas_relabel_classifier.json`, `results/damimas_fusi_yolo_relabel.json` — artefak relabel/proposal tidak menyediakan tabel pairing yang cocok; baris ini salah kategori, bukan angka yang belum dihitung. |

---

## Tambahan — Audit Forensik 6 September 2026 (`AF-E-010`, `AF-E-012`, `AF-E-014`, `AF-E-016`)

Kolom presisi/daya tangkap/F1 pada berkas ini bersifat **tingkat pasangan**;
audit mengukur pada **tingkat klaster fisik**, sehingga selnya ditulis
`N/A — definisi berbeda`. Rujukan penuh: `experiments/AUDIT-FORENSIK-2026-09-06.md`.

| ID Eksperimen | Algoritma / Modul Penaut | Ruang Masukan | Presisi Pasangan | Recall Pasangan | F1 Pasangan | ARI (*Rand Index*) | Cakupan Multi-Sisi | Fraksi Pool Palsu | AUC Val | Status Bukti | Rujukan Artefak |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `AF-E-012` | Penaut tepi `HistGradientBoosting`, 13 fitur | Proposal detektor agnostik, `conf ≥ 0,10` | N/A — definisi berbeda | N/A — definisi berbeda | N/A — definisi berbeda | N/A — tidak dilaporkan | N/A — tidak dilaporkan | N/A — tidak dilaporkan | 0,9064 | `SUPERSEDED` | `panen_pipeline.py`; AP `0,3609`; distribusi latih tidak cocok dengan inferensi |
| `AF-E-012` | Idem, dilatih pada ambang inferensi `conf ≥ 0,30` | Proposal detektor agnostik | N/A — definisi berbeda | N/A — definisi berbeda | N/A — definisi berbeda | N/A — tidak dilaporkan | N/A — tidak dilaporkan | N/A — tidak dilaporkan | **0,9185** | `VALID` | `panen_final.py`; AP `0,5562`. Pembanding penaut proyek: AUC `0,94846`, AP `0,59636`. F1 klaster fisik uji `0,7619` |
| `AF-E-010` | Kendala satu proposal per sisi pada `UF` | Proposal nyata, tepi geometri | N/A — bukan metrik pasangan | N/A — bukan metrik pasangan | N/A — bukan metrik pasangan | N/A — tidak dilaporkan | N/A — tidak dilaporkan | N/A — tidak dilaporkan | N/A — bukan metrik pasangan | `CORRECTED` | `run_e345.py`; klaim awal `45,3%` klaster melanggar **dikoreksi** oleh `AF-E-014` |
| `AF-E-014` | Idem, diukur pada jalur *sweep* yang sebenarnya | Tepi setelah `linear_sum_assignment` | N/A — bukan metrik pasangan | N/A — bukan metrik pasangan | N/A — bukan metrik pasangan | N/A — tidak dilaporkan | N/A — tidak dilaporkan | N/A — tidak dilaporkan | N/A — bukan metrik pasangan | `VALID` | `uf_impact.py`; pelanggaran `0,00%` untuk `max_size ≤ 3`, `7,95%` pada `max_size 4` + `pair_mode` "all"; **0 dari 630** konfigurasi berubah |
| `AF-E-016` | Jangkar Hungarian A (`max_size 4`, bersebelahan) | Dump WBF `combined1716` softvote | N/A — bukan metrik pasangan | N/A — bukan metrik pasangan | N/A — bukan metrik pasangan | N/A — tidak dilaporkan | N/A — tidak dilaporkan | N/A — tidak dilaporkan | N/A — bukan metrik pasangan | `VALID` | `anchor_a.py`; 1.586 klaster, **0 dari 135 pohon** berbeda; angka terkunci Anchor A tidak berubah |
