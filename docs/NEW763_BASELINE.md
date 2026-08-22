# Baseline SawitMVC-Depth v2.0.0 — 763 pohon

Dokumen ini mencatat matriks baseline baru yang dijalankan pada rilis
`SawitMVC-Depth-YOLO` v2.0.0. Dataset sumber tetap di
`/workspace/SawitMVC-Depth-YOLO`; training RGB memakai salinan turunan
`/workspace/SawitMVC-Depth-YOLO-RGB` supaya loader yang memperbaiki JPEG tidak
memutasi clone kanonik.

## Pemilihan tiga anggota

Yang dibawa dari `project-expertise` adalah tiga anggota detektor dengan bukti
training penuh dan metrik puncak, bukan probe, dry run, audit, atau klaim
plafon:

1. **RF-DETR-L** — deteksi RGB tertinggi pada matriks RGB lama.
2. **RT-DETR-L** — anggota dengan counting terbaik pada split RGB-352 dan
   deteksi kedua pada matriks RGB.
3. **YOLO26l** — baseline acuan yang juga menjadi backbone jalur dua-tahap.

Urutan ini tidak diperlakukan sebagai peringkat final; dataset baru memiliki
dua kampanye akuisisi yang berbeda dan seluruh keputusan akhir akan memakai
validation, lalu test dibuka sekali.

## Resep

- split bawaan rilis v2.0.0: 536/117/110 pohon, unit split pohon;
- RGB, 4 kelas B1–B4, COCO-pretrained;
- resolusi 1280, batch 4, cosine LR, deterministic training;
- tiga seed: 42, 1337, 2026;
- YOLO26l/RT-DETR-L: maksimum 60 epoch, patience 15;
- RF-DETR-L: maksimum 20 epoch, early stopping patience 5 karena eksperimen
  sebelumnya menunjukkan overfit sangat dini pada korpus kecil;
- evaluator: `pycocotools.COCOeval`, prediksi val dan test didump ke `.npz`;
- semua riwayat per-epoch disalin ke `results/riwayat_epoch_new763/`.

## Artefak runtime

- audit dataset: `results/new763_dataset_audit.json`;
- manifest sequence: `results/new763/matrix_manifest.json`;
- log: `results/new763/logs/`;
- prediksi: `results/new763/predictions/`;
- run/bobot: `runs_new763/`.

Karena README dataset memperingatkan perbedaan kampanye Juli DAMIMAS dan
Agustus MARIHAT/TOPAZ, laporan akhir wajib menyertakan metrik keseluruhan dan
stratifikasi kampanye untuk model/seed yang dipilih dari validation.
