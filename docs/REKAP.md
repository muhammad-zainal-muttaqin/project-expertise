# Rekapitulasi Pembelajaran & Sintesis Temuan Volume 1 (Research-Pipeline)

Dokumen ini merangkum seluruh metrik acuan, hasil eksperimen, percobaan yang berhasil, dan kegagalan yang tidak boleh diulang dari repositori pendahulu ([Research-Pipeline](https://github.com/muhammad-zainal-muttaqin/Research-Pipeline)).

---

## 1. Metrik Deteksi Acuan Historis (SawitMVC 953 Pohon, E-021)

Evaluasi empat model detektor menggunakan protokol standar `pycocotools` pada partisi kanonik bebas kebocoran (716 latih / 96 validasi / 141 uji):

| Arsitektur Detektor | Jumlah Parameter | Resolusi Masukan | Validasi $mAP50$ | Validasi $mAP50\text{--}95$ | Uji $mAP50$ | Uji $mAP50\text{--}95$ |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26m | 21,9 juta | $640\text{ px}$ | 0,5195 | 0,2411 | 0,5165 | 0,2452 |
| YOLO26l | 26,3 juta | $1.280\text{ px}$ | 0,5270 | 0,2526 | 0,5300 | 0,2568 |
| RT-DETR-L | 33,0 juta | $1.280\text{ px}$ | 0,5459 | 0,2555 | 0,5784 | 0,2707 |
| **RF-DETR-L** | **35,7 juta** | **$1.280\text{ px}$** | **0,5695** | **0,2604** | **0,6038** | **0,2770** |

---

## 2. Metrik Pencacahan Acuan Historis (Baseline-SawitMVC)

Evaluasi modul pencacahan (*counting*) pada 141 pohon uji SawitMVC:

### Evaluasi Pencacahan dengan Deteksi Nyata YOLO26m (Jalur B)
| Model Pencacah | Representasi Fitur | $\text{Class }\pm 1\text{ Acc}$ | $\text{Tree }\pm 1\text{ Acc}$ | Macro-$MAE$ |
|---|---|---:|---:|---:|
| **Ridge Regression** | **$F_{\text{all}}$ (67-dimensi)** | **77,48%** | **32,62%** | **1,036** |
| ElasticNet | $F_0 + \text{spasial}$ (21-dimensi) | 76,77% | 31,21% | 1,039 |
| ElasticNet | $F_0$ (13-dimensi) | 76,42% | 29,79% | 1,043 |
| Regresi Linier | $F_0$ (13-dimensi) | 75,71% | 30,50% | 1,048 |

### Batas Atas Teoretis dengan Deteksi Sempurna (Jalur C / Oracle)
| Model Pencacah | $\text{Class }\pm 1\text{ Acc}$ | $\text{Tree }\pm 1\text{ Acc}$ | Macro-$MAE$ |
|---|---:|---:|---:|
| ElasticNet | 98,05% | 92,20% | 0,277 |
| SVM | 97,87% | 91,49% | 0,266 |
| Ridge Regression | 97,70% | 90,78% | 0,275 |

Kesenjangan performa dari Jalur B ke Jalur C mencapai **$20,57\text{ pp}$**. Sumber galat utama pencacahan berasal dari presisi detektor di hulu, bukan keterbatasan algoritma regresi pencacahan.

---

## 3. Metrik Acuan Publikasi Ilmiah (*Data in Brief* 2026)

Rujukan: Indriani dkk., *Data in Brief* 67 (2026) 112990 (Tabel 3–4):
* Detektor: YOLO26m ($60\text{ epoch}$, *batch size* $32$, resolusi $640\text{ px}$, *seed* $42$).
* Evaluasi Deteksi: $AP50\text{ total} = \mathbf{0,531}$, Presisi $= 0,508$, Daya Tangkap (*Recall*) $= 0,571$.
* Evaluasi Pencacahan (141 pohon uji): SVR pada deteksi YOLO26m mencatat $\text{Class }\pm 1\text{ Acc} = 75,35\%$, $\text{Tree }\pm 1\text{ Acc} = 33,33\%$, $MAE = 1,027$.

---

## 4. Evaluasi Eksperimen Gagal (Larangan Pengulangan)

1. **Early Fusion Depth Konvensional (E-022, E-027)**:
   Penambahan kanal kedalaman mentah langsung ke *stem* konvolusi mendegradasi performa deteksi sebesar $−0,0230\text{ mAP}$. Penyatuan naif (*naive concatenation*) hanya menyuntikkan variasi derau kuantisasi.
2. **Fusi Menengah/Akhir dari Nol Tanpa Pretraining (E-032)**:
   Pelatihan 150 *epoch* dari bobot acak tidak menghasilkan signifikansi statistik (seluruh selang kepercayaan 95% memuat nilai nol).
3. **Modul Gating Inisialisasi Nol ($\gamma = 0$, F-007)**:
   Inisialisasi skalar $\gamma = 0$ membentuk **titik mati gradien (*zero-gradient trap*)**, sehingga bobot cabang samping tidak pernah terbarui selama proses optimasi.
4. **Penyetelan Hiperparameter Minor**:
   Variasi *batch size*, *learning rate*, dan augmentasi standar telah dieksplorasi penuh dan tidak menghasilkan lompatan performa signifikan.
5. **Metode SAHI (*Slicing Aided Hyper Inference*)**:
   Pemotongan citra parsial (*tiling*) tidak meningkatkan $mAP50$ karena memecah konteks global pohon kelapa sawit.

---

## 5. Eksperimen Berhasil yang Menjadi Fondasi Riset

1. **Keunggulan Arsitektur Transformer RF-DETR-L (E-021)**:
   Mencapai $mAP50 = \mathbf{0,6038}$, melampaui seluruh varian YOLO dan RT-DETR-L pada evaluasi visual RGB.
2. **Pemisahan Sinyal Frekuensi Tinggi (F-002)**:
   DWT *high-high* ($+0,0731$) terbukti memisahkan tandan dari pelepah secara geometris.
3. **Pipa Pencacahan Modular Ridge Regression + $F_{\text{all}}$**:
   Mencatat akurasi pencacahan tertinggi pada representasi 67-dimensi.
4. **Reproyeksi Piksel-ke-Piksel Sensor Kedalaman (E-022)**:
   Skrip `reproject_depth.py` mengoreksi pergeseran fisik median 29 piksel antara sensor inframerah Orbbec dan kamera RGB.

---

## 6. Prinsip Metodologis yang Mengikat

1. **Hambatan Utama Terletak pada Detektor**: Memperbaiki detektor di hulu merupakan tuas pengungkit paling efektif untuk meningkatkan pencacahan di hilir.
2. **Dua Ranah Kegagalan Deteksi**:
   * **Ranah Geometris**: Tandan kecil/tertutup pelepah (B4) — dapat dibantu oleh modalitas kedalaman lokalisasi.
   * **Ranah Fotometrik**: Ambiguitas visual kematangan (B2 vs B3) — tidak dapat diselesaikan oleh informasi kedalaman.
3. **Ketiadaan Komparabilitas Lintas-Dataset**: Dataset 953 pohon dan 352 pohon terpisah oleh jeda waktu 80 hari dan tidak boleh diperbandingkan secara langsung.
