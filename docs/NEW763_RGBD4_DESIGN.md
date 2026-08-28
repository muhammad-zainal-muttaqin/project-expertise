# Rancangan Eksperimen new763 RGB+D Empat Kanal

## Rancangan Eksperimen

Eksperimen ini menjawab satu pertanyaan terisolasi: apakah citra `new763`
dengan depth terukur meningkatkan detektor RGB yang sudah tersedia? Dataset
yang digunakan adalah `SawitMVC-Depth-YOLO v2.0.0` dengan 763 pohon. Unit
pemisahan tetap pohon, sehingga empat sudut pandang satu pohon tidak tersebar
ke partisi berbeda.

Perbandingan ditetapkan sebelum membaca hasil eksperimen baru:

| Arsitektur | Model acuan RGB, mAP50 validasi | Model RGB+D, mAP50 validasi | Δ RGB+D − RGB |
|---|---:|---:|---:|
| YOLO26l | 0,529357 | 0,529523 | +0,000166 |
| RT-DETR-L | 0,577766 | 0,584088 | +0,006322 |
| RF-DETR-L v2 | 0,608233 | 0,597070 | −0,011163 |

Model acuan RGB tidak dilatih ulang dalam eksperimen ini. YOLO26l dan RT-DETR-L
mengikuti resep acuan: 1.280 piksel, *batch* 4, *seed* 42, *cosine learning
rate*, 60 *epoch*, dan *patience* 15. RF-DETR-L mengikuti resep acuan:
1.280 piksel, *batch* 4, akumulasi gradien 4, 20 *epoch*, dan *patience* 5.
Eksperimen RGB+D RF-DETR menggunakan berkas pralatih generik resmi
`rf-detr-large-2026.pth` yang sama dengan sumber baseline RGB. Hash MD5 berkas
tersebut adalah `5cb72153541cbcb9aa6efa26222acc75`; head COCO diganti menjadi
empat kelas pada kedua recipe. Dengan demikian perbandingan RGB versus RGB+D
RF-DETR memakai sumber inisialisasi yang sama.

## Konstruksi depth

Hanya `train` dan `valid` yang dimaterialkan; direktori `test` tidak dibaca
oleh builder. Setiap citra memiliki pasangan berdasarkan *stem* yang sama:
JPEG RGB, label YOLO, berkas raw Y16, dan *sidecar* kalibrasi.

Depth raw berukuran 848×480 dalam milimeter, dengan nilai 0 sebagai nilai
invalid. Untuk setiap citra, `cameraParamDump` milik citra itu sendiri
diparsing ulang. Titik depth diproyeksikan ke grid warna 1.280×800 dengan
intrinsik depth dan warna, rotasi-translasi depth-ke-warna, distorsi
Brown–Conrady, serta *z-buffer* untuk konflik piksel. Tidak ada kalibrasi
global yang di-hard-code.

Payload TIFF lossless disimpan dalam urutan OpenCV `[B,G,R,D]`. Adapter model
mengubah tepat tiga kanal pertama menjadi `[R,G,B,D]`; kanal keempat tidak
dibalik. Ini diperlukan karena loader Ultralytics hanya melakukan pembalikan
otomatis untuk citra tiga kanal. Nilai depth dikodekan menggunakan batas fisik
tetap 0,3–20,0 meter:

\[
d_8=1+\operatorname{round}\left(254\operatorname{clip}\left(
\frac{1/z-1/z_{far}}{1/z_{near}-1/z_{far}},0,1\right)\right).
\]

Nilai 0 dicadangkan untuk lubang/invalid; nilai 1–255 adalah depth valid.
Batas ini ditetapkan sebelum statistik partisi dan bukan hasil optimasi
validasi. Statistik normalisasi RF-DETR diambil dari sampel deterministik
TRAIN saja.

## Pengendalian kanal dan augmentasi

Pada YOLO26l dan RT-DETR-L, model target dibangun dengan `ch=4`. Stem RGB
disalin ke tiga kanal pertama, kanal keempat diinisialisasi nol, dan bobot
lainnya dimuat dari checkpoint RGB. Untuk RF-DETR, patch embedding DINO
diubah dari 3 menjadi 4 kanal sebelum optimizer dan parameter groups dibuat,
dengan tiga bobot RGB tetap dan kanal depth nol. Guard juga memiliki fallback
di *forward* pertama untuk mencegah input 4-kanal lolos ke stem 3-kanal.

Transformasi spasial diterapkan ke empat kanal secara serentak. Padding RGB
memakai 114 seperti resep RGB; padding depth memakai 0 agar tidak menciptakan
depth sintetis. HSV hanya diterapkan pada tiga kanal warna. Evaluator
validasi melewatkan array RGBD secara langsung dan tidak menggunakan pembacaan
halaman pertama TIFF yang dapat menghilangkan kanal keempat.

## Temuan empiris terukur

Builder menghasilkan 2.144 citra TRAIN dan 468 citra VALID. Seluruh 2.612
TIFF dibuka ulang oleh reader produksi sebagai `uint8 (800, 1280, 4)`, dan
seluruh 2.612 label identik dengan label sumber. Dua varian kalibrasi teramati
dan dihitung per *sidecar*. Selang valid depth setelah reproyeksi rata-rata
sekitar 0,286 pada TRAIN dan 0,288 pada VALID; nilai ini adalah diagnostik
cakupan sensor, bukan skor detektor.

Hasil akhir VAL-only dicatat pada
[`NEW763_RGBD4_RESULTS.md`](NEW763_RGBD4_RESULTS.md) dan JSON di direktori
`results/`. YOLO26l menghasilkan mAP50 0,529523 versus RGB 0,529357;
bootstrap berpasangan 500 resample memberi CI95 Δ [-0,024195; 0,028892],
sehingga tidak ada bukti peningkatan. RF-DETR-L v2 menghasilkan mAP50 0,597070
versus RGB 0,608233; CI95 paired Δ [-0,037049; 0,018074], juga melintasi nol.
Metrik mAP50:95 RF v2 adalah 0,226946 versus 0,227471 pada RGB. Angka RF v1
tidak dipakai karena stem depth-nya dibuat setelah optimizer sehingga tidak
trainable melalui optimizer yang sudah ada. RT-DETR-L menghasilkan mAP50
0,584088 versus RGB 0,577766 dan mAP50:95 0,212043 versus 0,211041; paired
bootstrap memberi CI95 Δ [-0,026770; 0,039670], sehingga point gain RT juga
belum signifikan. Selisih semuanya dihitung
oleh `pycocotools.COCOeval` pada 468 citra validasi yang sama. Uji pada TEST
tidak termasuk dalam eksperimen ini.

## Keputusan metodologis

Model RGB+D hanya dapat disebut memberikan peningkatan apabila mAP validasi
dan analisis per kelas mendukungnya tanpa membuka TEST. Jika hasilnya tidak
meningkat, depth tidak akan dipaksakan masuk ke jalur produksi. Jika ada
peningkatan, checkpoint terbaik dipilih berdasarkan VALIDASI, lalu baru dapat
diajukan sebagai pembukaan TEST terpisah dengan label protokol yang eksplisit.

## Batasan validitas dan audit

1. Depth sensor hanya valid pada sebagian grid warna; nilai invalid tidak
   diimputasi dengan nilai acuan dan tidak dianggap sebagai objek.
2. TIFF menambah penyimpanan sekitar 6,9 GiB, tetapi berkas sumber RGB tidak
   diubah dan dataset gabungan `combined1716` tidak digunakan.
3. RT-DETR-L sudah selesai dilatih RGB+D dan evaluasi independen VALID sudah
   terkunci; paired bootstrap tetap menjadi syarat sebelum klaim signifikansi.
   Untuk YOLO26l, RT-DETR-L, dan RF-DETR-L,
   sumber bobot RGB dan RGB+D dicatat eksplisit dan sama dalam pasangan
   arsitektur masing-masing. Run RF-DETR v1 yang memperluas stem terlambat
   dikeluarkan dari perbandingan; hanya run v2 dengan guard sebelum optimizer
   yang sah.
4. Evaluator dan konfigurasi baru tidak memiliki opsi TEST. Pembukaan TEST,
   jika disetujui kemudian, harus menjadi artefak dan keputusan metodologis
   baru, bukan bagian dari pemilihan checkpoint saat ini.

## Reproduksi

```bash
python3 scripts/build_new763_rgbd4.py \
  --source /workspace/SawitMVC-Depth-YOLO \
  --output /workspace/new763_rgbd4 --workers 12

python3 scripts/train_new763_rgbd4.py --arch yolo26l \
  --data /workspace/new763_rgbd4 --name yolo26l_rgbd4_s42_i1280
python3 scripts/train_new763_rgbd4.py --arch rtdetr_l \
  --data /workspace/new763_rgbd4 --name rtdetr_l_rgbd4_s42_i1280
python3 scripts/train_new763_rgbd4.py --arch rfdetr_l \
  --data /workspace/new763_rgbd4 --name rfdetr_l_rgbd4_s42_i1280

python3 scripts/eval_new763_rgbd4_val.py --arch yolo26l \
  --weights /workspace/project-expertise/runs_new763_rgbd4/yolo26l_rgbd4_s42_i1280/weights/best.pt \
  --dataset /workspace/new763_rgbd4 --run-name yolo26l_rgbd4_s42_i1280_best
python3 scripts/eval_new763_rgbd4_val.py --arch rtdetr_l \
  --weights /workspace/project-expertise/runs_new763_rgbd4/rtdetr_l_rgbd4_s42_i1280_fair/weights/best.pt \
  --dataset /workspace/new763_rgbd4 --run-name rtdetr_l_rgbd4_s42_i1280_fair_best
python3 scripts/eval_new763_rgbd4_val.py --arch rfdetr_l \
  --weights /workspace/project-expertise/runs_new763_rgbd4/rfdetr_l_rgbd4_s42_i1280_fair_v2/checkpoint_best_total.pth \
  --dataset /workspace/new763_rgbd4 --run-name rfdetr_l_rgbd4_s42_i1280_fair_v2_best
```

Evaluasi dilakukan terpisah, selalu dengan `eval_new763_rgbd4_val.py`, yang
hanya menerima split validasi.
