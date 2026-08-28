# Status Evaluasi dan Matriks Hasil Eksperimen

> [!NOTE]
> **Rujukan Utama Riset:**  
> Sintesis akhir komprehensif, evaluasi pergeseran temporal ($\sim 80\text{ hari}$, `V2-E-022`), dan audit daya statistik (`V2-E-023`) dirangkum lengkap pada [docs/LAPORAN-AKHIR.md](../docs/LAPORAN-AKHIR.md) dan alur waktu kronologis pada [docs/WORKFLOW_KRONOLOGIS.md](../docs/WORKFLOW_KRONOLOGIS.md).

---

## 1. Matriks Hasil Deteksi dan Pencacahan Utama

Split Uji: $mAP50$ pycocotools / $\text{Class }\pm 1\text{ Acc}$ Ridge+$F_{\text{all}}$

| Korpus & Modalitas Data | YOLO26l | RT-DETR-L | RF-DETR-L |
|---|---|---|---|
| SawitMVC 953 Pohon (RGB Murni) | Det: 0,5435 / Count: 72,16% | Det: 0,5781 / Count: 76,24% | **Det: 0,6012** / Count: 76,24% |
| SawitMVC-Depth 352 Pohon (RGB Murni) | Det: 0,3606 / Count: 89,55% | Det: 0,4343 / Count: **90,91%** | **Det: 0,4544** / Count: 88,18% |
| SawitMVC-Depth 352 Pohon (RGB+D Invers Mentah) | Det: 0,3919 / Count: 87,73% | Det: 0,3877 / Count: 88,64% | Det: 0,4186 / Count: 88,18% |
| SawitMVC-Depth 352 Pohon (Sobel `edge`, Fase 5) | **Det: 0,4316** / Count: 87,27% | — | — |
| **Pipeline Dua-Tahap Modular (Fase 6)** | **Det: 0,4500** / Count: 85,91% | — | — |
| **SawitMVC-Depth v2.0.0 (763 Pohon)** | Det: 0,5163 | Det: 0,5580 | **Det: 0,6129** |
| **SawitMVC-Combined-1716 (1.716 Pohon)** | Det: 0,5389 | Det: 0,5746 | **Det: 0,5960** |

---

## 2. Sintesis Temuan Kritis Lintas-Fase

1. **Pergeseran Domain Temporal (Simpul V2-E-022)**:
   Dataset SawitMVC 953 direkam pada Mei 2026 dan SawitMVC-Depth 352 pada Juli 2026 ($\sim 80\text{ hari}$ jeda / $5\text{--}11$ siklus panen). Pada pohon yang sama, proporsi kelas B3 menyusut drastis dari $55,3\%$ menjadi $14,0\%$. Perbandingan deteksi 4-kelas lintas-dataset **tidak valid secara metodologis**.
2. **Dekomposisi Galat Detektor (Simpul V2-E-013)**:
   Lokalisasi murni mencapai $AP50 = \mathbf{0,6677}$, sementara deteksi 4-kelas hanya $0,3707$. Sebanyak $44,5\%$ kapasitas model tereduksi akibat kesalahan klasifikasi kelas ordinal.
3. **Efektivitas Sinyal Kedalaman (Simpul V2-E-024)**:
   Modalitas kedalaman terbukti meningkatkan performa **lokalisasi murni** ($AP50 = \mathbf{0,7636}$ vs kontrol RGB $0,7358$, $\Delta = +0,0278$, $P(\Delta > 0) = 92,1\%$), namun bersifat redundan terhadap fitur visual RGB untuk klasifikasi kematangan ($I(Y; D \mid \text{RGB}) \approx 0$).
4. **Rekor Plafon Lokalisasi Agnostik (Simpul V2-E-039)**:
   Ensembel WBF 3-detektor pada korpus Combined-1716 mencetak rekor tertinggi proyek dengan **$AP50 = \mathbf{0,8106}$ ($81,06\%$)** pada 1.052 citra uji kanonik.

---

## 3. Matriks Evaluasi Depth Monokular (Fase 7)

Protokol terkontrol pada 6 sel komparasi:

| No. Sel | Dataset | Modalitas Masukan | Kanal | Uji $mAP50$ | Validasi Puncak | Putusan Ilmiah |
|---|---|---|---|---|---|---|
| 1 | 352 | RGB Murni | 3 | 0,3677 | 0,4111 | Garis Dasar Pembanding 352 |
| 2 | 352 | RGB + Depth Fisik (`edge`) | 4 | **0,4270** | 0,3856 | **Kanal Masukan Terbaik 352** |
| 3 | 352 | RGB + Depth Monokular | 4 | 0,3943 | 0,3888 | Tidak Signifikan ($\Delta = +0,0266$, CI95 $[\minus 0,0270; +0,0739]$) |
| 4 | 352 | RGB + Depth Fisik + Depth Mono | 5 | 0,3766 | 0,4281 | **Penurunan Signifikan** ($\Delta = \minus 0,0504$, CI95 $[\minus 0,1038; \minus 0,0015]$) |
| 5 | 953 | RGB Murni | 3 | **0,5436** | 0,5373 | Garis Dasar Pembanding 953 |
| 6 | 953 | RGB + Depth Monokular | 4 | 0,4960 | 0,5012 | **Penurunan Signifikan** ($\Delta = \minus 0,0476$, CI95 $[\minus 0,0671; \minus 0,0274]$) |

---

## 4. Dua Pembatas Audit Silsilah Partisi Data (Simpul V2-E-033)

1. **Evaluasi Partisi Bersih `agn953_full`**: Sebanyak $87\%$ citra pada `test_penuh` beririsan dengan data prapelatihan. Nilai generalisasi yang sah adalah evaluasi pada partisi bersih (**`test_bersih`, 19 pohon / 316 kotak**) dengan skor **$AP50 = \mathbf{0,7702}$**.
2. **Keterbatasan Partisi Transfer 953 ke 352**: Sebanyak 44 dari 55 pohon uji dataset 352 termuat di dalam partisi latih dataset 953. Seluruh klaim adaptasi lintas-dataset wajib menyertakan kaveat silsilah ini.

## 5. Verifikasi Bobot Remote dan Pipeline Empat Sisi (V2-E-042)

Pada 27 Agustus 2026, enam bobot detektor terpilih dari bucket Hugging Face
(`new763` dan `combined1716`) diuji ulang pada test lokal
SawitMVC-Depth-YOLO dan SawitMVC-YOLO. Rincian lengkap dan seluruh artefak
tersedia di [laporan V2-E-042](../results/remote_eval_2026-08-27/README.md).

| Bank | Test | RF-DETR-L mAP50 | WBF class-aware mAP50 | WBF agnostik AP50 | MAE pipeline |
|---|---|---:|---:|---:|---:|
| `combined1716` | Depth | **0,6711** | **0,6691** | **0,8764** | 4,52 |
| `combined1716` | SawitMVC-YOLO 953 | **0,5890** | **0,5861** | **0,8350** | 14,99 |
| `new763` | Depth | 0,6125 | 0,6062 | 0,8451 | 3,28 |
| `new763` | SawitMVC-YOLO 953 | 0,1776 | 0,2018 | 0,4974 | 6,56 |

Nilai agnostik adalah lokalisasi tanpa label kelas. Pipeline empat sisi belum
ditetapkan sebagai pencacah produksi karena duplikasi klaster masih
menghasilkan akurasi tepat/±1 yang rendah, khususnya pada domain 953.

## 6. Iterasi greedy pipeline dan classifier 5 epoch (V2-E-043/V2-E-044)

Iterasi 27 Agustus 2026 berhasil menurunkan duplikasi cluster pada bank
`combined1716`:

| Test | F1 fisik | MAE raw cluster | Akurasi ±1 | Macro-F1 E2E |
|---|---:|---:|---:|---:|
| SawitMVC-Depth-YOLO | **0,8590** | **0,818** | **83,64%** | **0,6419** |
| SawitMVC-YOLO 953 | **0,8296** | **1,644** | **54,07%** | **0,5469** |

Baseline masing-masing adalah F1 `0,6140`/`0,5327` dan MAE
`4,518`/`14,993`. Counting di sini adalah raw linked-cluster count, bukan
Ridge `F_all`. Parameter dipilih melalui greedy sweep langsung pada test,
sehingga belum merupakan angka generalisasi.

Classifier crop RGB 5 epoch menghasilkan validasi internal terbaik akurasi
`0,6217` dan macro-F1 `0,6296`. C2-only ditolak karena class accuracy
end-to-end turun; blend 25% dengan soft-vote detector dipakai sebagai kandidat
khusus test 953. Rincian dan artefak ada di
[`V2-E-043/V2-E-044`](EKSPERIMEN.md#v2-e-043-pengetatan-proposal-dan-linker-mengurangi-duplikasi-cluster-pada-pipeline-empat-sisi)
dan [laporan optimasi](../results/remote_eval_2026-08-27/OPTIMIZED_PIPELINE.md).

## 7. Pipeline generalisasi validation-locked (V2-E-045)

Konfigurasi baru tidak dipilih langsung dari test. Layer Ridge count-aware
dilatih dari fitur proposal `train`, alpha dipilih dengan 5-fold CV train, lalu
parameter pipeline dikunci pada validation. Profil final mempertahankan WBF
equal-weight dan linker prior rotasi robust, dengan ranking cluster yang
memanfaatkan dukungan multi-view serta koreksi prior kelas ringan.

| Dataset | F1 fisik test | MAE count | ±1 count | Match class acc. | Macro-F1 E2E |
|---|---:|---:|---:|---:|---:|
| SawitMVC-Depth-YOLO | **0,8069** | **0,891** | **80,91%** | **80,31%** | **0,6047** |
| SawitMVC-YOLO 953 | **0,8043** | **1,393** | **61,48%** | **71,11%** | **0,5384** |

Angka ini adalah konfirmasi test atas profil validation-locked, bukan angka
greedy/test-tuned V2-E-043. Test pernah dibaca pada iterasi historis, sehingga
tetap tidak disebut hold-out publikasi yang sepenuhnya pristine. Rincian,
validation, profil, dan eksperimen yang ditolak ada di
[V2-E-045](EKSPERIMEN.md#v2-e-045-layer-count-aware-validation-locked-meningkatkan-generalisasi-pipeline-empat-sisi)
dan [JSON metrik](../results/remote_eval_2026-08-27/metrics/pipeline_combined1716_generalization_locked.json).

## 8. Validation wave 2026-08-28

Eksperimen lanjutan yang tidak membuka TEST menemukan satu kandidat yang
layak dipertahankan. Stack opini DINOv2-Large (`0,15`) dan logistic anggota
(`0,05`) dengan bias logit B2 `+0,15` meningkatkan matched-class accuracy
validation 953 dari `0,7542` menjadi `0,7684` dan macro-F1 dari `0,6014`
menjadi `0,6164`. Bootstrap berpasangan pada 5.000 pohon memberi selang
delta `[+0,0026; +0,0268]` untuk matched accuracy dan `[+0,0007; +0,0303]`
untuk macro-F1; kedua selang tidak mencakup nilai nol.

OOF stacking, agregasi fitur per sisi, kepala ordinal, KNN/prototype,
attention GPU, selector adaptif Hungarian–GSP, dan regresor pencacahan kaya
fitur tidak memperbaiki seluruh metrik utama secara bersamaan. Semua hasil
tersebut disimpan sebagai studi ablasi pada
[`PERFORMANCE_WAVE_2026-08-28`](../results/remote_eval_2026-08-28/PERFORMANCE_WAVE_2026-08-28.md).
Kandidat baru masih berupa hasil validation-lock dan belum menggantikan
angka TEST yang telah dikunci sebelumnya.

Follow-up backbone independen memakai ConvNeXt-Small, Swin-Tiny, dan
EfficientNetV2-S sebagai opini tambahan. Head terbaik tunggal 953 mencapai
matched `0,7594` / macro-F1 `0,6055`; fusion nominal mencapai `0,7697` /
`0,6166`, tetapi hanya menambah satu pohon benar dibanding anchor `0,7684` /
`0,6164` dan belum memiliki CI independen. Karena itu branch tersebut dicatat
sebagai ablasi, bukan kandidat produksi.

Selector TRAIN-fitted untuk memilih original GSP versus V2 geo/count pada
Depth juga ditolak sebagai kompromi: V2-only menurunkan MAE menjadi `0,7607`
dan menaikkan matched menjadi `0,8495`, tetapi menurunkan physical F1 menjadi
`0,8341` dan macro-F1 menjadi `0,6667`; policy terbaik count-oriented masih
menukar physical F1 (`0,8421`) demi MAE (`0,7265`). Original GSP tetap
menjadi referensi. Seluruh follow-up ini TRAIN/VAL-only dan tidak membuka
TEST.

## 9. Komposisi lintas-layer dan head-aware ranking (V2-E-047)

Dengan topology, target count, dan class head sebagai layer terpisah, seluruh
komposisi VAL yang telah dideklarasikan diuji ulang. Kandidat terbaik menjaga
original GSP, memakai target count V2 geo (Ridge fit TRAIN), lalu memakai
class calibration `scale_macro` yang sudah dipilih dari VAL:

| Metrik Depth VAL | Baseline | Kandidat |
|---|---:|---:|
| physical F1 | 0,852641 | **0,854225** |
| MAE | 0,931624 | **0,914530** |
| ±1 | 0,786325 | **0,786325** |
| matched class | 0,845652 | **0,850000** |
| macro-F1 | 0,680685 | **0,689013** |

Point estimate ini unggul secara umum, tetapi paired bootstrap 5.000
resampling pohon (117 pohon VAL) masih inconclusive; seluruh CI delta
melintasi nol. Kandidat disimpan sebagai `validation candidate`, bukan klaim
test/signifikansi. Head-aware truncation juga tidak dipromosikan karena
kenaikan matched class dibayar dengan penurunan physical F1. Script dan JSON
ada di `results/remote_eval_2026-08-28/validation_wave/`.

## 10. Composition-aware retraining audit

Head anggota baru yang dilatih pada komposisi persis `original GSP + V2 geo
count` tidak memberi kenaikan: matched tetap `0,850000`, tetapi macro-F1
turun ke `0,684983` dari `0,689013`. Branch ini ditolak dan disimpan sebagai
negative control; kandidat lintas-layer yang sudah ada tetap menjadi hasil
terbaik sementara.
