# Manifest Artefak — Audit Forensik 6 September 2026

Rujukan naratif: [`docs/AUDIT-FORENSIK-2026-09-06.md`](../../docs/AUDIT-FORENSIK-2026-09-06.md).
Log eksperimen: [`experiments/AUDIT-FORENSIK-2026-09-06.md`](../../experiments/AUDIT-FORENSIK-2026-09-06.md).

## 1. Lingkungan

| Komponen | Versi |
|---|---|
| GPU | NVIDIA RTX 3090, 24 GB |
| CPU / RAM | 64 vCPU / 503 GB |
| Python | 3.12 |
| `torch` | 2.14.0+cu130 |
| `ultralytics` | 8.4.142 |
| `scikit-learn` | 1.9.0 |

Korpus: `ULM-DS-Lab/SawitMVC-YOLO` (26.838 berkas, 2,4 GB) dan
`ULM-DS-Lab/SawitMVC-Depth-YOLO` (38.928 berkas, 3,4 GB), diunduh utuh
6 September 2026.

## 2. Berkas metrik

| Berkas | Ukuran | SHA-256 (16 karakter awal) | Entri |
|---|---|---|---|
| `detector_matrix.json` | 2,7K | `ff0ef71cc449e521` | `AF-E-006`, `AF-E-007` |
| `e1b_precision_vs_conf.json` | 418 B | `b4253d5e28e9f540` | `AF-E-007` |
| `e1c_fp_kind.json` | 532 B | `bd95523c041bdd4f` | `AF-E-007` |
| `e1d_merge.json` | 466 B | `b4fedf1c5577c3b6` | `AF-E-007` |
| `e345.json` | 1,2K | `8e05ff9a24f1ff33` | `AF-E-008`, `AF-E-010`, kontrol negatif `AF-E-009` |
| `e4b.json` | 433 B | `5b24327237f1adc6` | `AF-E-009` |
| `figures/pair_0_DAMIMAS_A21B_0268_4.jpg` | 703K | `b90b8b84c6ea442b` | `AF-E-001` |
| `figures/pair_2_DAMIMAS_A21B_0053_4.jpg` | 725K | `107cf8730aef4b20` | `AF-E-001` |
| `figures/fp_july.jpg` | 408K | `680c28ad78da9d06` | `AF-E-007` |

Angka `AF-E-001` s.d. `AF-E-005` tidak menghasilkan JSON tersendiri; keluarannya
tercatat penuh pada log di `logs_ringkas/audit_forensik_2026-09-06/`.

## 3. Peta skrip terhadap entri

| Skrip | Entri | Keluaran |
|---|---|---|
| `an1_overlap.py` | `AF-E-001` | `logs_ringkas/.../an1_overlap.log` |
| `an3_framing.py` | `AF-E-001` | `.../an3_framing.log` |
| `an4_bunches.py` | `AF-E-001` | `.../an4_bunches.log` |
| `an2_visual.py` | `AF-E-001` | `figures/pair_*.jpg` |
| `an5_labelnoise.py` | `AF-E-002` | `.../an5_labelnoise.log` |
| `an6_structure.py`, `an7_monotone.py` | `AF-E-003` | `.../an7_monotone.log` |
| `an8_counting.py` | `AF-E-004` | `.../an8_counting.log` |
| `exp_crops.py`, `exp_train.py` | `AF-E-005` | bobot + `P_*.npy` (bucket) |
| `exp_ceiling.py`, `exp_sensitivity.py` | `AF-E-005` | `.../exp_ceiling.log`, `.../exp_sensitivity.log` |
| `exp_fuse.py`, `exp_fuse2.py` | `AF-E-003`, `AF-E-009` | `.../exp_fuse2.log` |
| `build_ds.py`, `run_exp.py` | `AF-E-006`, `AF-E-007` | `detector_matrix.json` |
| `e1b_fp.py`, `e1c_fpkind.py`, `e1d_merge.py` | `AF-E-007` | `e1b_*`, `e1c_*`, `e1d_*.json` |
| `run_e345.py` | `AF-E-008`, `AF-E-010` | `e345.json` |
| `e4b_fuse.py` | `AF-E-009` | `e4b.json` |

## 4. Artefak besar di bucket

Bucket `ULM-DS-Lab/project-expertise-backup`, awalan
`audit_forensik_2026-09-06/`:

| Awalan | Isi | Ukuran |
|---|---|---|
| `runs_audit/` | Lima pelatihan YOLO26s (`may4`, `may2`, `may1`, `dep1`, `dep4`): `weights/best.pt`, `weights/last.pt`, `results.csv`, `args.yaml`, serta sembilan direktori evaluasi silang | ±195 MB |
| `crops953/` | `convnext.pt`, `index.json`, `P_train.npy`, `P_val.npy`, `P_test.npy` | ±360 MB |

Citra terpotong mentah (`crops953/{train,val,test}/*.jpg`) dapat dibentuk ulang
secara deterministik melalui `scripts/audit_forensik/exp_crops.py`.

## 5. Prosedur reproduksi

```bash
# 0) prasyarat: dua korpus berada di /workspace
python3.12 -m venv .venv-af
.venv-af/bin/pip install ultralytics scikit-learn

# 1) forensik data — tanpa GPU, tanpa model  (AF-E-001 … AF-E-004)
for s in an1_overlap an3_framing an4_bunches an5_labelnoise an7_monotone an8_counting; do
  .venv-af/bin/python scripts/audit_forensik/$s.py
done

# 2) plafon mAP50 dan kurva sensitivitas          (AF-E-005)
.venv-af/bin/python scripts/audit_forensik/exp_crops.py
.venv-af/bin/python scripts/audit_forensik/exp_train.py
.venv-af/bin/python scripts/audit_forensik/exp_ceiling.py
.venv-af/bin/python scripts/audit_forensik/exp_sensitivity.py

# 3) detektor dan matriks lintas-kampanye         (AF-E-006, AF-E-007)
.venv-af/bin/python scripts/audit_forensik/build_ds.py
.venv-af/bin/python scripts/audit_forensik/run_exp.py yolo26s.pt
.venv-af/bin/python scripts/audit_forensik/e1b_fp.py
.venv-af/bin/python scripts/audit_forensik/e1c_fpkind.py
.venv-af/bin/python scripts/audit_forensik/e1d_merge.py

# 4) pencacahan, fusi struktur, cacat UF          (AF-E-008 … AF-E-010)
.venv-af/bin/python scripts/audit_forensik/run_e345.py
.venv-af/bin/python scripts/audit_forensik/e4b_fuse.py
```

Seluruh pelatihan memakai `seed 42`, `deterministic=True`, `imgsz 960`,
30 *epoch*, `patience 8`, `batch 16`, `workers 32`, `cache=ram`.

## 6. Batasan yang wajib dibawa

1. Detektor audit adalah **YOLO26s pada 960 piksel**, bukan YOLO26l pada 1.280
   piksel. Kesetaraannya diverifikasi pada satu titik (`mAP50` empat kelas
   `0,5433` berbanding `0,5435`), bukan pada seluruh rentang konfigurasi.
2. Angka `AF-E-005` mengandaikan lokalisasi sempurna; ia adalah plafon, bukan
   performa yang dapat dicapai.
3. Selisih `+0,0058` makro-F1 pada `AF-E-009` **belum memiliki selang
   kepercayaan berpasangan** dan belum tentu dapat dibedakan dari derau.
4. `AF-E-010` mengukur besarnya celah kendala pada daftar tepi geometri
   sederhana, bukan perubahan metrik akhir pipeline.
5. Partisi uji 953 sudah pernah dibaca berkali-kali pada sejarah proyek.
   Tidak ada angka pada audit ini yang memenuhi definisi *hold-out* pristine.
