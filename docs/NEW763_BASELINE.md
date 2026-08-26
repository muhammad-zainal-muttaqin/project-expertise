# Spesifikasi Baseline & Evaluasi Korpus SawitMVC-Depth v2.0.0 (763 Pohon)

Dokumen ini memuat spesifikasi metodologis dan protokol evaluasi garis dasar (*baseline*) pada dataset **SawitMVC-Depth-YOLO v2.0.0** (763 pohon multi-kampanye).

---

## 1. Pemilihan Tiga Arsitektur Detektor

Tiga arsitektur detektor dipilih berdasarkan rekam jejak performa terverifikasi pada riset Volume 2:

1. **RF-DETR-L**: Arsitektur detektor berbasis transformer dengan skor deteksi tertinggi pada korpus 953 pohon ($mAP50 = \mathbf{0,6012}$) dan 352 pohon ($mAP50 = \mathbf{0,4544}$).
2. **RT-DETR-L**: Arsitektur detektor *real-time* transformer dengan akurasi pencacahan tertinggi pada subset 352 pohon ($\text{Class }\pm 1\text{ Acc} = \mathbf{90,91\%}$).
3. **YOLO26l**: Arsitektur konvolusional murni yang berfungsi sebagai model pembanding standar dan *backbone* lokalisasi.

---

## 2. Resep dan Parameter Pelatihan

* **Pembagian Partisi (*Split*)**: 536 pohon latih, 117 pohon validasi, dan 110 pohon uji (440 citra uji; unit partisi berbasis identitas pohon bebas kebocoran).
* **Modalitas Masukan**: Citra RGB resolusi $1.280\text{ piksel}$, 4 kelas tingkat kematangan (B1–B4), inisialisasi bobot *COCO-pretrained*.
* **Optimasi**: *Batch size* 4, penjadwalan *cosine learning rate*, pelatihan deterministik.
* **Jadwal Pelatihan & Penghentian Dini**:
  * **YOLO26l & RT-DETR-L**: Maksimum 60 *epoch*, toleransi penghentian dini (*early stopping patience*) 15 *epoch*.
  * **RF-DETR-L**: Maksimum 20 *epoch*, toleransi penghentian dini 5 *epoch* untuk mencegah *overfitting* pada konvergensi dini.
* **Protokol Evaluasi**: Metrik $mAP50$ dan $mAP50\text{--}95$ dihitung melalui `pycocotools.COCOeval`, dengan seluruh dump prediksi disimpan ke format `.npz`.

---

## 3. Hasil Kuantitatif dan Stratifikasi Sub-Kampanye

Hasil evaluasi pada partisi uji (440 citra, 891 kotak anotasi):

| Arsitektur Detektor | $mAP50$ Makro | $mAP50\text{--}95$ | DAMIMAS ($n=120$) | MARIHAT ($n=44$) | TOPAZ ($n=276$) |
|---|---|---|---|---|---|
| **RF-DETR-L** | **0,6129** | **0,2335** | **0,4460** | **0,5390** | **0,6369** |
| RT-DETR-L | 0,5580 | 0,2055 | 0,4025 | 0,5110 | 0,5824 |
| YOLO26l | 0,5163 | 0,1906 | 0,3811 | 0,4723 | 0,5395 |

### Catatan Karakteristik Komputasi
* **RF-DETR-L** bersifat *CPU-bound* (pencocokan Hungarian *matcher* dan *dataloader* mengeksploitasi 16 core CPU proses utama secara intensif dengan beban GPU relatif rendah).
* **YOLO26l & RT-DETR-L** bersifat *GPU-bound* (memaksimalkan utilisasi tensor core GPU).
* Pelatihan paralel antara RF-DETR-L dan YOLO26l/RT-DETR-L dapat dilakukan secara simultan pada workstation GPU tunggal tanpa degradasi *throughput*.

---

## 4. Artefak Berkas Terkait

* Audit Partisi Dataset: [`results/new763_dataset_audit.json`](file:///D:/Work/Assisten-Dosen/project-expertise/results/new763_dataset_audit.json)
* Ringkasan Hasil Terstruktur: [`results/new763_summary.json`](file:///D:/Work/Assisten-Dosen/project-expertise/results/new763_summary.json)
* Log Pelatihan: [`results/logs_ringkas/new763_rfdetr_l_rgb_s42_i1280.log`](file:///D:/Work/Assisten-Dosen/project-expertise/results/logs_ringkas/new763_rfdetr_l_rgb_s42_i1280.log)
