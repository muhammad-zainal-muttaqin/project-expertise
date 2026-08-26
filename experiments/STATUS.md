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
