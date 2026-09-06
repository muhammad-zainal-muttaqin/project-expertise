# Cakupan review berurutan

Review ini tidak menyatakan bahwa seluruh berkas repositori sudah dibaca.
Daftar berikut membatasi klaim terhadap bagian yang benar-benar diperiksa.

## Jalur produksi dan eksperimen yang dibaca penuh

- Dataset: `build_combined_rgb_dataset.py`, `build_new763_rgbd4.py`,
  `build_crop_dataset.py`.
- Detektor dan RGBD: `train_baseline_new763.py`, `train_new763_rgbd4.py`,
  `train_rfdetr_4ch.py`, `eval_new763_rgbd4_val.py`.
- Evaluator: `eval_new763_pycoco.py`, `eval_agnostic_from_npz.py`,
  `eval_twostage.py`, `eval_remote_pipeline_postprocess.py`.
- Pipeline remote: `sweep_remote_pipeline.py`,
  `evaluate_remote_count_reconciled.py`, `train_detection_edge_linker.py`,
  `evaluate_remote_class_head.py`, `fit_fused_probability_calibrator.py`,
  `train_crop_classifier.py`.
- Subproyek pertandan: `audit_counting_total_damimas.py`,
  `reid_pertandan.py`, `c3_multitampak.py`, `c_backbone_ordinal.py`,
  `set_transformer_damimas.py`.
- Audit forensik terbaru: `an1_overlap.py`, `an5_labelnoise.py`,
  `an8_counting.py`, `build_ds.py`, `exp_crops.py`, `exp_train.py`,
  `exp_ceiling.py`, `exp_sensitivity.py`, `e1c_fpkind.py`, `e1d_merge.py`,
  `e4b_fuse.py`.
- Pipeline Panen terbaru: `panen_det.py`, `panen_ordinal.py`,
  `panen_pipeline.py`, `panen_eval.py`, `panen_count.py`, `panen_final.py`.

Nama skrip tanpa direktori mengacu pada `scripts/`,
`pipeline-pertandan/scripts/`, atau `scripts/audit_forensik/` sesuai kelompok.

## Pembacaan sebagian dan penelusuran rujukan

- `scripts/audit_forensik/run_e345.py`: persiapan data, pencacahan E3,
  ekstraksi deteksi E4, dan seluruh reproduksi UF E5.
- `penaut_pertandan.py`, `gnn_penaut.py`: antarmuka dan penelusuran fungsi;
  tidak menjadi dasar klaim bahwa seluruh implementasinya telah diaudit.
- project_expertise_experiment_map: README, bagian naratif empat dossier
  audit repositori, serta bagian awal `experimentData.ts`. Bukan audit
  seluruh antarmuka React atau seluruh simpul TypeScript.
- Dokumen hasil, ledger metrik, hasil terkunci GSP/reranker, serta JSON/log
  AF dan Panen dibaca untuk memeriksa silsilah dan definisi. Lampiran
  inventaris yang panjang tidak seluruhnya dibaca baris demi baris.

## Verifikasi eksekusi

- `audit_implementation_contracts.py`: kendala sisi terkini, kasus IoU,
  kemurnian identitas, normalisasi WBF, bobot kelas, ranking kepala,
  penyelarasan baris, prediksi kosong, C3/BatchNorm, dan kanal crop.
- `audit_latest_artifacts.py`: konstanta dipilih TRAIN, transisi split,
  reproduksi hasil Panen awal/final, matriks konfusi ujung ke ujung,
  evaluasi 135 pohon dengan profil tetap, serta diagnosis cache CORN.
- Lima kasus regresi UF dari pengguna lulus.
- Pemeriksaan sintaks `compileall` pada scripts dan subproyek lulus.
- DOCX dibuka ulang; 2 tabel, 35 hyperlink, dan 27 rujukan lokal diperiksa.
  Render visual tidak tersedia.

Tidak ada pelatihan baru, inferensi GPU, penalaan pada TEST, perubahan
dataset, perubahan angka historis, atau penggunaan subagen dalam review ini.
