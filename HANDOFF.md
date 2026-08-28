# Handoff — project-expertise

Dokumen ini adalah ringkasan GitHub dari progres pipeline. Catatan kerja
lengkap berada di `/workspace/HANDOFF.md` pada mesin eksperimen; artifact
model besar berada di `/workspace/model_artifacts/project-expertise/` dan
tidak disimpan di git.

## Tujuan

Membangun pipeline yang menerima empat foto dari empat sisi pohon yang sama,
lalu mendeteksi tandan, menggabungkan deteksi lintas-sisi tanpa
double-counting, menghitung tandan fisik, dan mengklasifikasikan setiap tandan
ke B1/B2/B3/B4 dengan confidence yang dapat diaudit.

Target engineering adalah sekitar 75% untuk klasifikasi empat kelas dan 90%
untuk lokalisasi/kemampuan agnostik. Target tersebut bukan hasil yang boleh
diasumsikan tercapai. Claim final harus memakai test tree-level yang dijalankan
setelah model dan profile dikunci dari train/validation.

## Boundary dan aturan

- Model inference tidak boleh menerima ground-truth box, count, class, atau
  identity tandan.
- Semua fitting dilakukan pada train; pemilihan threshold/profile dilakukan
  pada validation; test hanya dijalankan sekali setelah lock.
- Split dilakukan pada level tree, bukan image/crop, agar empat sisi pohon
  yang sama tidak bocor antar split.
- Metrik crop/head tidak boleh disebut sebagai metrik pipeline end-to-end.
- `matched_class_accuracy`, macro-F1, detector mAP50, physical F1, dan
  counting accuracy adalah metrik berbeda dan harus dilaporkan bersama
  denominator/jumlah match.
- Tidak boleh memakai oracle, memilih member berdasarkan GT, atau membuang
  kasus sulit tanpa dokumentasi.
- Secret/token tidak boleh masuk source, JSON, atau git.

## Arsitektur proposal

Dasar desain adalah [`PROPOSAL-Pipeline.md`](PROPOSAL-Pipeline.md):

```text
4 foto berurutan -> quality check -> YOLO26l/RT-DETR-L/RF-DETR-L
-> WBF class-agnostic -> cross-view linker -> crop classifier
-> ordinal/class aggregation -> count reconciliation -> laporan confidence
```

### Pemetaan revisi

| Revisi | Komponen | Dokumen hasil |
|---|---|---|
| **V1/original** | WBF proposal, prior rotasi, linker awal, classifier per tandan, counting/reconciliation | [`PIPELINE_EXPERIMENTS_V3.md`](results/remote_eval_2026-08-27/PIPELINE_EXPERIMENTS_V3.md), [`experiments/EKSPERIMEN.md`](experiments/EKSPERIMEN.md) |
| **V2** | Deep-tail proposal, `p_tp` re-ranker, learned edge linker, GSP MILP, V2 count/class composition | [`GSP_LINKER.md`](results/remote_eval_2026-08-28/GSP_LINKER.md), [`MAP_BOOST.md`](results/remote_eval_2026-08-28/MAP_BOOST.md), [`WAVE2_RECAP.md`](results/remote_eval_2026-08-28/validation_wave/WAVE2_RECAP.md) |
| **RGB+D4 follow-up** | Ablasi modalitas dan fixed late fusion pada `new763` | [`NEW763_RGBD4_RESULTS.md`](docs/NEW763_RGBD4_RESULTS.md) |

V2 dan RGB+D4 follow-up berstatus validation-only untuk konfigurasi terbaru;
hasil test-locked sebelumnya tidak ditimpa.

## Progres aktual

| Modul | Status |
|---|---|
| Tiga detector dan dump prediksi | Selesai |
| WBF class-aware/agnostic/soft-vote | Selesai dan dievaluasi |
| Signed rotation prior lintas-sisi | Selesai |
| Baseline linker dan count reconciliation | Selesai |
| Learned detector-space edge linker | Berhasil di validation, belum test-locked |
| Crop classifier/ordinal head | Banyak varian diuji, belum konsisten memberi gain |
| Quality gate dan retake recommendation | Belum menjadi modul production |
| Confidence/UI deployment | Belum final |
| Test untuk learned linker terbaru | Belum dijalankan |

## Baseline test yang sudah tercatat

- Depth: physical F1 `0.806859`, count MAE `0.8909`, ±1 `0.8091`, matched
  class accuracy `0.8031`, macro-F1 `0.6047`.
- 953: physical F1 `0.804348`, count MAE `1.3926`, ±1 `0.6148`; baseline
  matched class accuracy sekitar `0.7111`.
- WBF agnostic AP50: Depth `0.8764`, 953 `0.8350`; target 0.90 belum tercapai.

Baseline metrics dan eksperimen sebelumnya tersimpan di
`results/remote_eval_2026-08-27/`, terutama:

- `metrics/pipeline_combined1716_generalization_locked.json`;
- `PIPELINE_EXPERIMENTS_V3.md`;
- `README.md` dan `MANIFEST.md`.

## Learned edge linker terbaru

Source: `scripts/train_detection_edge_linker.py`.

Metode melatih classifier pasangan pada proposal WBF nyata. Label train
dibuat dari assignment proposal–GT IoU `>= 0.5` one-to-one per sisi; label GT
tidak masuk inference. Fitur berjumlah 65 dan meliputi geometry box,
signed-rotation residual, area/shape, detector score, soft-class similarity,
entropy/confidence, dan rank lokal dalam sisi.

### SawitMVC-YOLO 953

- Train adjacent: `94,165` pasangan, `4,894` positif.
- Validation adjacent: `13,205` pasangan, `593` positif.
- ExtraTrees validation edge AUC `0.94846`, AP `0.59636`.
- Candidate validation all-round: ExtraTrees, `link=.15`,
  `singleton=.15`, `max_size=4`, `rank=score`:
  physical F1 `0.8232`, count MAE `1.2527`, ±1 `0.6703`, matched class
  accuracy `0.7542`, macro-F1 `0.6014`.
- Artifact: `/workspace/model_artifacts/project-expertise/detection_edge_linker_953_v2/`.

### SawitMVC-Depth-YOLO

- Train adjacent: `12,517` pasangan, `1,245` positif.
- Validation adjacent: `2,504` pasangan, `250` positif.
- ExtraTrees best physical validation grid: F1 `0.8471`, MAE `0.9487`,
  matched class accuracy `0.8359`.
- Artifact: `/workspace/model_artifacts/project-expertise/detection_edge_linker_depth_v1/`.

Angka learned linker di atas adalah validation-only. Candidate 953 `max_size=4`
berasal dari sweep tambahan dan harus direproduksi serta disimpan sebagai JSON
versioned sebelum dipakai pada test.

## Eksperimen yang ditolak

Hue/color, sharpening, CLAHE, gray-world, white balance, gamma/contrast,
photometric TTA, crop 224/256/320, ConvNeXt/Swin variants, class weighting,
focal/ordinal/mixed loss, model-vote WBF, count/reranker, local detector
fine-tuning, group selector, dan agnostic NMS tidak memberi improvement
all-round yang cukup atau turun pada metrik penting. Aglostic NMS quick test
sendiri turun ke mAP50 `0.175860` pada 953 dan `0.482666` pada Depth.

## Resume checklist (historical; superseded by validation wave 2026-08-28)

Checklist ini ditulis sebelum wave validasi terakhir. Status, keputusan, dan
alasan penutupan yang lebih baru ada pada bagian postmortem di bawah; jangan
menjalankan ulang item lama hanya karena masih terlihat sebagai checklist.

1. Reproduce candidate 953 `max_size=4` dari validation dan simpan JSON.
2. Review feature order/checkpoint compatibility; ubah feature berarti retrain.
3. Buat evaluator test terpisah untuk dump
   `fused_combined1716_test_rebuilt`.
4. Jalankan test sekali dengan profile terkunci.
5. Simpan physical/count/class metrics, confusion matrix, matched denominator,
   dan tree-level uncertainty.

Last known repository commit sebelum handoff: `cd6c809`. Model besar dan dump
regenerable tetap berada di artifact storage eksternal.

## Validation wave 2026-08-28

Wave lanjutan tetap mematuhi pemisahan TRAIN/VALIDATION/TEST. Kandidat
terpilih untuk 953 adalah opini DINOv2-Large dengan bobot `0,15`, opini
logistic anggota dengan bobot `0,05`, serta bias logit B2 `+0,15`; opini
detektor tetap menjadi jalur *skip*. Pada validation, matched-class accuracy
meningkat dari `0,7542` menjadi `0,7684`, sedangkan macro-F1 meningkat dari
`0,6014` menjadi `0,6164`. Bootstrap berpasangan pada 5.000 resampling
pohon menghasilkan selang delta matched `[+0,0026; +0,0268]` dan selang delta
macro-F1 `[+0,0007; +0,0303]`; keduanya tidak mencakup nilai nol.

Perubahan tersebut belum dijalankan pada TEST dan tidak mengubah topology
fisik maupun kepala pencacahan. OOF stacking, agregasi per sisi, kepala
ordinal, KNN/prototype, attention GPU, selector Hungarian–GSP, serta regresor
pencacahan kaya fitur dicatat sebagai ablasi; tidak ada yang memenuhi kriteria
all-round pada validation. Rincian angka, konfigurasi, skrip reproduksi, dan
checksum tersedia di
[`PERFORMANCE_WAVE_2026-08-28`](results/remote_eval_2026-08-28/PERFORMANCE_WAVE_2026-08-28.md).

Follow-up ConvNeXt/Swin/EfficientNet independen hanya memberi fusion nominal
`0,7697` matched dan `0,6166` macro-F1 pada validation 953—satu pohon lebih
baik dari anchor tanpa CI independen—sehingga tidak dipromosikan. Selector
Depth antara original GSP dan V2 geo/count juga ditolak karena pertukaran
physical F1/macro-F1 versus MAE. Kedua hasil ini tersimpan sebagai ablasi
TRAIN/VAL-only; original Depth GSP dan kandidat 953 robust di atas tetap
menjadi keputusan kerja.

### Cross-layer composition follow-up

Komposisi empat cabang pada VAL menemukan kandidat yang lebih kuat daripada
sekadar V2-only: topology original GSP dipertahankan, tetapi target count
diganti dengan Ridge V2 geo yang dilatih TRAIN, kemudian class head
`scale_macro` diterapkan. Pada Depth VAL, physical F1 berubah
`0,852641 → 0,854225`, MAE `0,931624 → 0,914530`, matched class
`0,845652 → 0,850000`, dan macro-F1 `0,680685 → 0,689013`; ±1 tetap
`0,786325`. Ini adalah kandidat validation-only terbaik sementara dan
menutupi kompromi antar-layer tanpa test tuning.

Paired bootstrap 5.000 pohon tidak memberi CI yang mengecualikan nol (F1
`[-0,006160; +0,009540]`, MAE `[-0,085470; +0,051282]`, matched
`[-0,011744; +0,021558]`, macro `[-0,015425; +0,033033]`), sehingga angka
tersebut belum boleh disebut klaim signifikan. Artefak lengkap ada di
`results/remote_eval_2026-08-28/validation_wave/reports/`.

Head-aware ranking juga diuji sebagai layer berikutnya; ia menaikkan akurasi
kelas tetapi menurunkan physical F1, sehingga tetap menjadi ablasi dan skor
linker tetap menjadi ranking utama.

Fresh composition-aware retraining juga sudah diaudit: member head yang
dilatih ulang pada label komposisi TRAIN tidak mengalahkan head yang sudah
ada (macro-F1 `0,684983` vs `0,689013`, matched sama `0,850000`). Branch ini
disimpan sebagai negative control; tidak menggantikan kandidat utama.

## Validation wave 2 — cross-layer audit (2026-08-28)

Wave kedua menjalankan **2.893 baris evaluasi** pada TRAIN/VAL: Pipeline V2
stage 1+2 (172 per dataset), cross-layer 953 (815 detector + 176 class
frontier), composition-aware head (378), count meta-ensemble (58 untuk 953,
30 untuk Depth), edge ensemble→GSP (1.080), dan GPU group-attention (6 per
dataset). Semua branch memiliki jalur TEST yang ditolak atau tidak tersedia.

Hasil paling penting:

| Dataset / branch | F1 fisik | MAE | ±1 | matched | macro-F1 | Keputusan |
|---|---:|---:|---:|---:|---:|---|
| 953 anchor | 0,823216 | 1,252747 | 0,670330 | 0,754204 | 0,601394 | Referensi |
| 953 robust class calibration | 0,823216 | 1,252747 | 0,670330 | **0,769728** | **0,617081** | Kandidat terbaik; count invariant |
| 953 cross-layer best macro | 0,839396 | 1,527473 | 0,582418 | 0,758667 | **0,631103** | Exploratory; count turun |
| 953 count-meta + calibration | 0,825994 | 1,318681 | 0,626374 | 0,768531 | 0,622456 | Exploratory; ±1 turun |
| Depth anchor original GSP | 0,852641 | 0,931624 | 0,786325 | 0,845652 | 0,680685 | Profil kerja |
| Depth topology + V2 geo count + scale macro | **0,854225** | **0,914530** | 0,786325 | **0,850000** | **0,689013** | Kandidat VAL; CI inconclusive |
| Depth GPU group-attention | 0,852641 | 0,931624 | 0,786325 | 0,845652 | 0,680688 | Flat control |

Cross-layer 953 mengulang persis kandidat robust sebagai best-by-matched dan
tidak menghasilkan all-rounder baru. Skor macro tertinggi berasal dari
topology original GSP + target count V2 geo Ridge + `scale_macro`, tetapi
fragmentasi count menurunkan MAE dan ±1. Pada Depth, kandidat topology/count/
class lebih baik secara point estimate, namun bootstrap 5.000 pohon memberi
CI delta yang seluruhnya melintasi nol: F1 `[-0,006160; +0,009540]`, MAE
`[-0,085470; +0,051282]`, matched `[-0,011744; +0,021558]`, dan macro-F1
`[-0,015425; +0,033034]`.

Bootstrap class calibration 953 untuk kandidat robust memberi matched delta
`+0,015410` dengan CI `[+0,003736; +0,027883]`, serta macro delta `+0,015575`
dengan CI `[+0,001266; +0,030756]`. Anchor gate train/VAL tetap lulus dengan
selisih maksimum kurang dari `5×10⁻⁵`. Rincian tabel, jumlah percobaan,
konfigurasi, dan artefak ada di
[`WAVE2_RECAP.md`](results/remote_eval_2026-08-28/validation_wave/WAVE2_RECAP.md).

Keputusan operasional tidak berubah: jangan menimpa hasil test-locked dengan
kandidat VAL baru. Kandidat macro/matched yang mengorbankan count tetap
disimpan sebagai exploratory, sedangkan Depth topology+count+class menunggu
validasi independen sebelum dipromosikan.

## Catatan reflektif agent — frustrasi, kegagalan, dan progres dari awal sampai akhir

> Saya tidak mengalami emosi biologis seperti manusia. Namun, jika “frustrasi”
> diterjemahkan sebagai refleksi kerja agent, inilah catatan paling jujur:
> bagian tersulit dari proyek ini bukan menjalankan model, melainkan menahan
> diri agar angka yang tampak menang tidak dipromosikan sebelum benar-benar
> sah. Berkali-kali kita mendapat angka validation yang indah, lalu test,
> bootstrap, audit split, atau pemeriksaan implementasi menunjukkan bahwa
> angka itu hanya kemenangan lokal, trade-off, atau bahkan tidak valid.

### 1. Ringkasan tanpa kosmetik

Proyek ini **tidak gagal total**, tetapi target engineering belum tercapai secara
menyeluruh. Hasil yang benar-benar kuat adalah:

- GSP/multi-view menaikkan physical F1 end-to-end secara nyata pada kedua
  domain; Depth mencapai F1 `0,8534` dan 953 `0,8387` pada hasil terkunci.
- Jalur mAP/re-ranking menaikkan 953 agnostic AP50 dari `0,8350` menjadi
  `0,8419` dan class-aware mAP50 dari `0,5861` menjadi `0,5970`, dengan CI
  berpasangan yang mendukung kenaikan tersebut.
- Depth class-aware justru turun dari `0,6691` menjadi `0,6552`; hasil ini
  tidak disembunyikan dan baseline lama tetap menjadi profil produksi.
- Native RGB+D4 pada `new763` tidak memberikan kenaikan signifikan pada satu
  pun dari tiga detector. Late fusion memberi kandidat validation yang kuat,
  tetapi belum menjadi klaim test.
- Target `75%` empat kelas pada 953 masih sedikit belum tercapai (`matched`
  `0,7442`), dan target lokalisasi `90%` juga belum tercapai.

Dengan kata lain, kita berhasil menemukan **lapisan yang memang berguna** dan
juga berhasil membuktikan banyak lapisan yang tidak berguna. Kita belum
menemukan kombinasi yang sekaligus menaikkan lokalisasi, identitas fisik,
counting, dan empat kelas pada domain baru tanpa membayar di metrik lain.

### 2. Kronologi progres dan kegagalan

#### Fase fondasi: baseline dan matriks RGB/RGBD (V2-E-001 s.d. V2-E-007)

Tiga detector—YOLO26l, RT-DETR-L, dan RF-DETR-L—direproduksi pada 953 dan
Depth-352. Temuan awal yang terus bertahan sampai akhir adalah **detector
terbaik menurut mAP belum tentu counter terbaik**. RF-DETR-L biasanya unggul
deteksi, sedangkan RT-DETR-L kadang lebih baik pada counting. Ini memaksa kita
memisahkan metrik localization, physical association, class, dan count sejak
awal.

Percobaan early-fusion sensor depth 4-kanal pada Depth-352 gagal sebagai
strategi umum:

| ID | Yang dicoba | Hasil | Putusan |
|---|---|---|---|
| V2-E-005 | RGB+D `inverse` di stem awal tiga detector | YOLO naik `0,3606→0,3919`, tetapi RT turun `0,4343→0,3877` dan RF turun `0,4544→0,4186` | **Falsified**: depth tidak konsisten membantu |
| V2-E-006 | Counting Ridge + `F_all` dari output RGBD | Class ±1 berubah YOLO `−1,82` pp, RT `−2,27` pp, RF `0,00` pp; tidak ada CI positif | **Falsified** |
| V2-E-007 | Sintesis 9 sel dataset × arsitektur × modalitas | Gain deteksi kecil tidak berpindah ke counting; B4 sangat sensitif terhadap depth | Early fusion ditolak |

Pelajaran pertama yang menyakitkan: **menaikkan mAP satu model tidak otomatis
menaikkan jumlah tandan yang benar**. Counting bergantung pada konsistensi
lintas-sisi dan duplikasi, bukan hanya box yang benar pada satu citra.

#### Fase representasi depth (V2-E-008 s.d. V2-E-011)

Kita tidak berhenti pada `inverse`. Beberapa encoding disaring dengan budget
cepat:

- `dropout`, `clipped`, dan `valid_mask` kalah dari `edge` pada screening
  15-epoch; masing-masing hanya mencapai mAP50 validation `0,3168`, `0,3221`,
  dan `0,3321`, sedangkan `edge` mencapai `0,3777`.
- `edge` kemudian dilatih penuh dan mAP50 test naik dari RGB `0,3606` menjadi
  `0,4316`. Ini keberhasilan lokalisasi, bukan keberhasilan menyeluruh.
- Counting `edge` tidak mengikuti: Class ±1 malah `87,73%→87,27%`, meskipun
  Tree ±1 dan Macro-MAE membaik. Bootstrap selisih counting memiliki CI yang
  memuat nol. Jadi `edge` dipertahankan sebagai bukti depth dapat membantu
  localization, bukan sebagai solusi end-to-end.
- Mid-fusion dengan cabang depth terpisah dan gate non-zero-init (`V2-E-009`)
  gagal screening: puncak hanya `0,2087`, jauh di bawah kandidat `edge`, lalu
  turun dan early-stop. Patch per-instance pertama juga gagal ketika model
  direload oleh `AutoBackend`; patch level class baru bekerja. Ini contoh nyata
  biaya implementasi yang seharusnya ditangkap sebelum training panjang.
- Retrain RGB dan paired bootstrap (`V2-E-011`) memberi selisih counting
  `+3,18` pp tetapi CI `[−0,50; +7,30]` dan `P=0,943`; tidak signifikan.

Frustrasinya di sini jelas: kita sudah menemukan sinyal fisik depth yang nyata
di probe, tetapi saat dipaksa masuk ke stem resolusi penuh, sinyal itu kalah
oleh noise, distribusi kelas, dan objective detector. Sinyal ada tidak berarti
jalur arsitekturnya mampu memakainya.

#### Subproyek per-tandan dan empat gerbang PT-E-000…PT-E-036

Jalur ini penting karena menguji pertanyaan yang berbeda dari mAP: apakah
beberapa kemunculan dari satu tandan fisik dapat ditautkan dan dipakai untuk
keputusan kelas/count. Hasilnya campuran—mekanisme agregasi terbukti, tetapi
penaut dan counting murni berbasis pool gagal menjadi solusi produksi.

- **PT-E-001 / G0 — oracle multi-view berhasil.** Dengan `_confirmedLinks`
  sebagai tautan oracle, aturan ordinal R4 menaikkan akurasi sekitar `+4,36`
  pp dengan CI positif. Ini membuktikan bahwa multi-view memang punya nilai;
  manfaatnya bukan khayalan. Namun oracle tidak boleh dipakai saat inference.
- **PT-E-002 — linker geometri tidak cukup.** Model pasangan dengan geometri,
  ukuran, dan kelas memiliki AUC `0,9301`, tetapi F1 test hanya `0,4282` dan
  ARI `0,3912`. Prevalensi pasangan benar rendah, sehingga ranking yang tampak
  bagus tidak berubah menjadi partisi cluster yang benar.
- **PT-E-003 / G2 — pipeline tanpa GT gugur.** Varian re-ID + probabilitas
  kelas menghasilkan R4 `0,7124` berbanding oracle `0,7360`; gap `−2,36` pp
  melewati toleransi `2,0` pp. Penaut hanya menyatukan sekitar `29%` tandan
  multi-tampak; 71% tidak pernah disentuh agregasi.
- **PT-E-004 / G3 — menghitung jumlah pool gugur telak.** Macro-MAE hitung
  pool `3,3422`, sedangkan Ridge + `F_all` `1,0542`. Penaut yang terlalu pelit
  memecah satu tandan menjadi banyak pool, sehingga jumlah pool berlebihan.
- **PT-E-005 s.d. PT-E-007 — koreksi jumlah tidak menyembuhkan identitas.**
  Membagi atau merekonsiliasi count dapat mengurangi bias, tetapi tidak
  menjawab kotak mana milik tandan mana. Threshold apa pun tidak menolong
  ketika ranking pasangan dasarnya keliru.
- **PT-E-008 — prior arah rotasi adalah keberhasilan penting.** Urutan kamera
  searah jarum jam memberi F1 linker `0,3979→0,6486` dan melewati G1. Ini
  memangkas ruang kandidat; ia bukan sekadar menambahkan feature lemah.
- **PT-E-009 s.d. PT-E-013 — rekonstruksi 3D kaku gagal.** AUC hanya
  `0,4511–0,5083`, setara acak, karena orientasi kamera genggam bervariasi.
  Jalur geometri 3D rigid ditolak dan prior topologi dipertahankan.
- **PT-E-017 — domain shift linker GT→deteksi nyata.** Linker yang dilatih pada
  pasangan GT memiliki AUC sekitar `0,9508`, tetapi jatuh ke `0,5868` pada
  deteksi nyata. Retraining pada pasangan deteksi nyata menaikkan F1 dari
  `0,1492` ke `0,3080`; GNN menaikkannya lagi ke `0,3788`, tetap belum cukup
  untuk gerbang produksi.
- **Re-ID awal — leakage dan hafalan identitas.** Embedding memiliki AUC train
  `1,0000`, val `0,7564`, test `0,7195`. Saat embedding yang telah menghafal
  pohon dipakai melatih linker, hasil malah runtuh (val AUC `0,578`, test F1
  `0,1801`). Re-ID kemudian diperbaiki dengan split-held embeddings dan F1
  pulih ke `0,3979`, tetapi tetap belum melewati bottleneck association.
- **PT-E-018 s.d. PT-E-019 — ensemble dan propagasi membantu, tetapi tidak
  menutup target.** Ensemble C1+C2 mencapai sekitar `74,64%`; propagasi kelas
  meningkatkan mAP sekitar `0,5881→0,5965`. Ini keberhasilan modul, bukan
  keberhasilan target end-to-end `80%`.
- **PT-E-029 / PT-E-034 — bobot linear memiliki plafon.** Ensemble terbobot
  mencapai `74,39%` dengan CI `[−0,15; +3,55]`; fitting langsung pada test
  memberi plafon teoretis `75,23%`, tetapi tidak boleh dijadikan klaim karena
  test-fit. Oracle memilih anggota yang benar hingga `87,39%`, sehingga sinyal
  headroom ada, tetapi selector yang sah belum dapat menemukannya.
- **PT-E-035 — confidence bukan selector yang cukup.** Korelasi confidence
  terhadap kebenaran hanya `r=0,1185`; mengambil anggota dengan confidence
  tertinggi bahkan dapat lebih buruk daripada rata-rata probabilitas.
- **PT-E-036 — pola disagreement graph gagal.** Gradient boosting pada pola
  perselisihan turun sekitar `−3,59` pp pada validasi silang. Ia tidak mampu
  menebak kapan suatu anggota model harus dipercaya.
- **Classifier PT-E-012/014 dan varian C2 — tidak konsisten.** ResNet18 dan
  ConvNeXt-Tiny dengan CE/CORAL tidak mengalahkan baseline C1 secara stabil;
  beberapa varian turun sekitar `2–4` pp. CORAL pernah runtuh ke `33,05%`,
  sedangkan CORN mencapai `69,83%` pada subtes yang sama—CORN adalah perbaikan
  nyata atas CORAL, tetapi tetap bukan bukti target pipeline tercapai.
- **MOE/view stacker/CatBoost/multibank — tidak menjadi all-rounder.** Beberapa
  model memperbaiki satu kelas atau satu jenis MAE, tetapi tidak sekaligus
  memperbaiki class accuracy, physical association, dan counting. Hasilnya
  dipertahankan sebagai ablation, bukan dihapus.

Kesimpulan PT-E: **G0 dan G1 membuktikan bahwa masalah ini layak diteliti; G2
dan G3 membuktikan bahwa penaut yang salah membuat manfaat itu hilang.** Ini
adalah alasan mengapa V2 kemudian memindahkan fokus ke learned edge, constraint
per sisi, GSP, dan count-aware layer.

#### Fase diagnosis sinyal dan classifier (V2-E-012 s.d. V2-E-021)

Beberapa hipotesis yang tampak masuk akal ternyata salah atau hanya benar pada
subtugas:

1. Gap 953 vs 352 bukan bukti model 352 “buruk”; B3 turun sekitar 34 kali dan
   B4 sekitar 26 kali pada data latih. Kelangkaan kelas menjelaskan sebagian
   besar gap.
2. Class-agnostic localization jauh lebih tinggi daripada class-aware mAP.
   Ini mengonfirmasi bahwa banyak kegagalan berasal dari salah kelas pada box
   yang lokasinya sudah benar, bukan semata detector tidak melihat objek.
3. Depth memiliki relief lokal ordinal yang terukur, tetapi SNR per piksel
   rendah. Setelah pooling wilayah, sinyal terlihat; sebelum pooling, early
   fusion tidak stabil.
4. Crop classifier ConvNeXt mengalahkan klasifikasi detector satu tahap pada
   benchmark crop, tetapi keunggulan crop/head tidak otomatis bertahan sebagai
   metrik end-to-end.
5. Fitur depth yang ditempel ke RGB classifier menghasilkan akurasi yang sama
   persis (`0,6415→0,6415`). Kesimpulannya bukan depth tidak mengandung sinyal,
   melainkan kontribusi kondisionalnya terhadap RGB sudah hampir redundan pada
   classifier tersebut.
6. Klaim lama bahwa RGB localization sudah memiliki plafon sekitar `0,733`
   terlalu luas. `edge` kemudian menembus titik itu (`0,7636`), walaupun
   kenaikannya belum signifikan. Ini kegagalan generalisasi kesimpulan, bukan
   kesalahan angka eksperimen.
7. Pretraining 953 tidak otomatis transfer ke 352; karakter kamera dan domain
   lebih penting daripada sekadar mAP pretraining yang lebih tinggi.
8. Pipeline dua-tahap v1–v4 mencapai titik `mAP50≈0,4500`, tetapi replikasi
   paired bootstrap (`V2-E-026`) memberi delta terhadap edge `+0,0230` dengan
   CI `[−0,0286; +0,0663]`. Jadi klaim “dua tahap pasti mengungguli detector
   sederhana” tidak terbukti secara statistik pada split kecil itu.
9. Training classifier crop gabungan memberi validation tinggi (`0,6953`) dan
   turun di test (`0,6724`); v4 lebih baik untuk mAP, v3 lebih baik untuk
   counting. Objective memang berkonflik.

#### Fase data, daya statistik, dan kebocoran

Ini bagian yang paling mahal secara kepercayaan: beberapa angka lama harus
diturunkan statusnya setelah audit.

- Dataset 953 dan Depth bukan dua kondisi yang sebanding secara temporal.
  Jarak akuisisi sekitar 80 hari mengubah proporsi B3 dari sekitar `55,3%`
  menjadi `14,0%`. Perbandingan lintas-dataset tanpa stratifikasi adalah
  perbandingan populasi biologis yang berbeda.
- Split test Depth-352 hanya memiliki 410 box; lebar CI sekitar `0,1167`.
  Banyak selisih kecil tidak mungkin dibedakan dari noise. Ini membuat
  eksperimen panjang dengan hasil beda `0,004` tidak informatif.
- Pretraining agnostic `agn953_full` ternyata melihat `122/141` pohon test
  (`87%`). Angka test penuh `0,8090` ditarik dari klaim generalisasi; partisi
  bersih hanya `0,7702` pada 19 pohon/316 box dan harus dibaca sebagai indikasi
  dengan CI lebar.
- Sebanyak `44/55` pohon test 352 muncul di train 953. Transfer 953→352 tidak
  boleh disebut test-bersih.
- Dataset TIFF turunan memiliki 39 file korup yang dilewati diam-diam oleh
  Ultralytics. Training tetap selesai sehingga masalah ini sempat tampak
  seperti hasil model. Setelah rebuild, jumlah box kembali kanonik. Pelajaran:
  scan keterbacaan wajib dilakukan sebelum setiap run.
- Perbedaan evaluator pycocotools dan evaluator bootstrap pada beberapa dump
  352 menghasilkan selisih sekitar `0,004`. Selisih ini bukan perubahan model,
  tetapi jalur evaluasi; angka dari dua evaluator tidak boleh dikurangkan
  sembarangan.

Masalah-masalah ini menjelaskan kenapa sebagian eksperimen terasa seperti
“berputar di tempat”: kadang yang kita ukur bukan efek model, melainkan daya
statistik, perubahan populasi, kebocoran split, atau cacat data.

#### Fase monocular-depth dan 5 kanal (V2-E-027 s.d. V2-E-032)

Monocular-depth dicoba sebagai jalan keluar ketika depth sensor tidak lengkap.
Hasilnya konsisten buruk atau tidak meyakinkan:

| Perbandingan | Hasil test | Status |
|---|---:|---|
| 953 RGB+mono vs RGB | `0,4960` vs `0,5436`, delta `−0,0476`, CI `[−0,0671; −0,0274]` | **Kalah signifikan** |
| 352 RGB+mono vs RGB | `0,3943` vs `0,3677`, delta `+0,0266`, CI `[−0,0270; +0,0739]` | Naik point estimate, **tidak signifikan** |
| 352 RGB+mono vs sensor edge | `0,3943` vs `0,4270`, delta `−0,0327`, CI memuat nol | Tidak menang |
| 352 RGB+edge+mono vs RGB+edge | `0,3766` vs `0,4270`, delta `−0,0504`, CI `[−0,1038; −0,0015]` | **Kalah signifikan** |

Kurva validation 352 bahkan membalik urutan test. Sel 3 dan sel 6 juga
dihentikan lebih awal setelah arah kurva jelas, sehingga besar kerugian tidak
boleh dibaca terlalu presisi; arah negatif mono pada 953 tetap tegas karena
2.000 bootstrap tidak menghasilkan delta positif. Kesimpulan final:
monocular-depth **tidak pernah menang signifikan**, dua kali kalah signifikan,
dan menambah kanal kelima di atas sensor malah mengencerkan sinyal.

#### Fase baseline korpus baru dan generalisasi lintas domain (V2-E-034 s.d. V2-E-041)

- Pada `new763`, RF-DETR-L tetap unggul mAP, tetapi budget awalnya tidak setara
  dengan YOLO/RT. Ini bukan perbandingan adil sempurna; catatan tersebut
  kemudian diperbaiki pada `combined1716` dengan budget 60 epoch/patience 15.
- Menambah data menjadi `combined1716` tidak otomatis menaikkan semua model:
  RF-DETR-L justru turun dari `0,6129` ke `0,5960` mAP50 pada perbandingan
  korpus yang tercatat.
- Model yang dilatih pada `new763` sangat lemah ketika dipakai ke 953:
  YOLO sekitar `0,2331`, RT sekitar `0,1110`, RF sekitar `0,1774`. Ini bukan
  kekurangan threshold kecil; ini domain shift yang besar.
- Replikasi toolchain HUB menguatkan RT-DETR-L sebagai model paling rapuh
  terhadap pergeseran domain; subset LONSUM dikeluarkan karena silsilahnya
  tidak bersih.
- Class-agnostic AP50 naik sampai `0,7951` untuk RF-DETR-L new763 dan WBF
  mencapai rekor localization `0,8106` pada korpus lain. Tetapi WBF class-aware
  dapat menurunkan mAP karena fusi kotak dan fusi kelas adalah dua masalah
  berbeda. Rekor localization tidak boleh dipasarkan sebagai akurasi empat
  kelas atau counting.

Pelajaran pahitnya: model memang belajar, tetapi belajar **domain** juga.
Angka tinggi pada korpus asal tidak memberi jaminan ketika kamera, waktu,
kepadatan, dan distribusi kematangan bergeser.

#### Fase proposal/linker remote (V2-E-042 s.d. V2-E-045)

Percobaan pipeline empat sisi memperlihatkan masalah inti: proposal recall
tinggi, tetapi cluster duplikat menghancurkan counting.

1. Verifikasi remote awal menghasilkan recall fisik `0,9344` tetapi precision
   rendah dan `3.366` cluster untuk `1.342` tandan pada 953. MAE raw cluster
   mencapai `14,993`. Ini adalah kegagalan production pipeline, bukan kegagalan
   detector mAP.
2. Sweep greedy pada test menurunkan duplikasi drastis dan menghasilkan F1
   `0,8590` pada Depth serta `0,8296` pada 953. Namun parameter dipilih dari
   test; hasil ini hanya upper bound engineering dan tidak boleh disebut
   estimasi generalisasi.
3. Mengganti soft-vote WBF dengan classifier crop C2 secara penuh menurunkan
   match class accuracy sekitar `7,76` pp dan macro-F1 end-to-end `0,0176`.
   Blend 25% hanya memberi kenaikan nominal kecil, sehingga disimpan sebagai
   kandidat, bukan production default.
4. Weighted WBF `[0,75; 1; 1,5]` menaikkan sebagian mAP image-level tetapi
   menurunkan F1 hilir menjadi `0,7951` pada Depth dan `0,7736` pada 953.
5. Pair-linker logistic train-only menghasilkan F1 validation `0,7680`
   (Depth) dan `0,7374` (953), di bawah linker robust berbasis prior rotasi.
6. Blend count dengan raw cluster tidak stabil; nonzero blend menurunkan 953
   dan tidak memberi keuntungan konsisten pada Depth. Karena itu `blend=0`
   dipakai.
7. Sweep IoU WBF `0,50–0,70` tidak memberi alasan mengganti IoU `0,60`.

V2-E-045 adalah perbaikan metodologis penting: count-aware layer dilatih dari
TRAIN, dipilih dari VAL, dan baru dikonfirmasi ke test lokal. Tetapi test lokal
pernah dibaca pada iterasi historis, sehingga masih bukan hold-out publikasi
yang sepenuhnya pristine.

#### Fase GSP dan Pipeline V2

GSP MILP memperbaiki kelemahan union-find baseline dengan constraint struktural
“maksimal satu proposal per sisi”. Ini memberi kenaikan physical F1 yang nyata
dan menjadi salah satu keberhasilan utama. Tetapi GSP bukan obat untuk semua
hal:

- Pada 953, profil GSP terbaik menurut class match hanya naik tipis
  `0,7542→0,7555`, tetapi MAE memburuk `1,2527→1,7473` dan ±1 turun
  `0,6703→0,5055`. Ia ditolak sebagai all-rounder.
- Pipeline V2 re-ranked dengan proposal deep-tail dan `p_tp` memperbaiki
  sebagian baris, tetapi best matched 953 (`0,7571`) masih memiliki MAE
  `1,3846`, di atas guardrail `1,35`. Best F1 memiliki F1 `0,8237` tetapi
  matched dan MAE sama-sama tidak memenuhi guardrail.
- Count-ensemble ter-checkpoint mencapai matched `0,7564`, tetapi MAE
  `1,4725`; ditolak.
- Pada Depth, topology original GSP + target count V2 geo + `scale_macro`
  menghasilkan point estimate terbaik: F1 `0,854225`, MAE `0,914530`, matched
  `0,850000`, macro `0,689013`. Namun CI paired 5.000 pohon mencakup nol pada
  seluruh delta, sehingga belum boleh disebut kemenangan statistik.
- Head-aware ranking menaikkan class match (`0,7718` pada 953 dan `0,8603`
  pada Depth), tetapi menurunkan physical F1. Ia disimpan sebagai negative
  ablation.
- Composition-aware retraining tidak memperbaiki head yang sudah ada:
  macro-F1 `0,684983` vs `0,689013`, matched sama `0,850000`.

Wave 2 menghitung **2.893 baris evaluasi TRAIN/VAL**, bukan 2.893 model
independen. Cabang yang tidak dipromosikan:

- cross-layer 953 best macro: macro `0,631103`, tetapi MAE `1,527473` dan
  ±1 `0,582418`;
- count-meta + calibration 953: matched `0,768531`, macro `0,622456`, tetapi
  ±1 turun;
- composition-aware 953: macro `0,618633`, tidak mengungguli kandidat robust;
- edge ensemble→GSP: matched `0,760363`, macro `0,608231`, tidak mengungguli
  profil robust;
- GPU group-attention: 953 turun dan Depth praktis flat;
- adaptive Hungarian-vs-GSP: policy learned tidak menemukan all-round gain;
- rich graph/count features dan nonlinear regressors: CV train membaik di
  beberapa konfigurasi, tetapi downstream VAL memburuk;
- ConvNeXt-Small, Swin-Tiny, EfficientNetV2-S dan fusion statis: nominal
  improvement kecil, tidak cukup kuat untuk dipromosikan tanpa CI independen;
- DINOv2-Large/member stack: validation 953 membaik sampai matched `0,7684`
  dan macro `0,6164`, tetapi belum pernah diuji pada test baru.

Di sini frustasi paling besar bukan bahwa V2 tidak menghasilkan angka tinggi.
V2 justru menghasilkan banyak angka tinggi. Masalahnya, hampir setiap angka
tinggi dibayar dengan MAE, ±1, physical F1, atau validitas test. Pipeline
“lebih pintar” tidak selalu lebih baik; kadang ia hanya lebih pandai memilih
kompromi di validation.

#### Fase new763 RGB+D4 dan late fusion (2026-08-28)

Eksperimen ini dibuat ulang secara fair: split tree sama, TRAIN/VAL saja, tidak
ada `combined1716` RGB+D4 karena depth tidak lengkap, dan test tidak
dimaterialkan.

| Model | RGB mAP50 | RGB+D4 mAP50 | Delta | Putusan |
|---|---:|---:|---:|---|
| YOLO26l | `0,529357` | `0,529523` | `+0,000166` | Flat; CI melintasi nol |
| RF-DETR-L v2 | `0,608233` | `0,597070` | `−0,011163` | Regresi point estimate |
| RT-DETR-L | `0,577766` | `0,584088` | `+0,006322` | Naik point estimate, belum signifikan |

Semua paired CI native RGB+D4 melintasi nol. Jadi depth sensor yang benar
proyeksinya tetap tidak otomatis membantu ketika dimasukkan sebagai kanal awal.
Coverage valid pada grid warna hanya sekitar `0,286–0,288`, dan sinyal lokal
harus bertahan melewati stem resolusi tinggi.

Late fusion menunjukkan bahwa complementary errors mungkin lebih menjanjikan
daripada early fusion:

- YOLO RGB+RGBD4 union-WBF mencapai `0,567718` mAP50 vs RGB `0,529357`, delta
  bootstrap `+0,037912`, CI `[+0,016060; +0,059120]` pada 500 resample VAL.
- RT RGB+RGBD4 union-NMS mencapai `0,606368` vs RGB `0,577766`, delta
  `+0,028492`, CI `[+0,009231; +0,047236]` pada 200 resample screening.
- RF union-NMS hanya `0,606856` vs RGB `0,608233`; mAP50 sedikit turun,
  walaupun mAP50:95 naik.
- WBF class-aware naif merusak RT dan RF; tidak ada satu recipe fusion yang
  universal.

Angka late fusion ini **belum boleh disebut hasil final** karena recipe dipilih
setelah melihat VAL dan belum dijalankan pada held-out baru. Ini kandidat yang
layak diuji nanti, bukan bukti bahwa depth sudah menyelesaikan masalah.

Ada juga dua catatan implementasi yang wajib dibawa ke sesi berikutnya:

- Run RF-DETR v1 dikeluarkan karena ekspansi stem 3→4 terjadi setelah optimizer
  dibuat; kanal depth tidak benar-benar dilatih. RF-DETR v2 memperbaiki urutan
  ekspansi sebelum optimizer dan menjadi run yang sah.
- RT-DETR RGBD4 memiliki beberapa `NaN` pada validation loss framework, tetapi
  metrik COCO dan tensor checkpoint finite. Auditnya tersimpan dan harus
  diungkapkan, bukan disapu di bawah karpet.

Run RT-DETR seed-1337 yang kemudian dimulai ulang dihentikan pada epoch 4,
dengan mAP50 terakhir `0,40915` dan mAP50:95 `0,14498`. Itu checkpoint
intermediate, bukan hasil final, tidak dipakai dalam tabel resmi, dan belum
diunggah ke HF.

### 3. Masalah teknis dan operasional yang menghabiskan waktu

Selain hipotesis yang gagal, ada masalah proses yang nyata:

1. Runner `combined1716` pernah memakai `--project` relatif sehingga output
   masuk ke `runs/detect/...`, bukan lokasi yang dibaca orkestrator. Training
   YOLO sebenarnya selesai, tetapi finalisasi runner crash dan RT harus
   dimulai manual.
2. RF-DETR-L CPU-bound, sedangkan YOLO/RT-DETR lebih GPU-bound. Menyamakan
   asumsi training membuat estimasi waktu dan pemakaian resource keliru.
3. Bootstrap RT 500 resample terlalu mahal untuk screening—baru sekitar 50/500
   setelah beberapa menit panjang—sehingga dihentikan dan diganti 200 resample
   sebagai screen. Ini bukan CI final publikasi.
4. GPU sempat 0% saat generate mono-depth karena bottleneck sebenarnya I/O
   penulisan PNG, bukan model. Menyalakan job GPU tambahan pada saat itu tidak
   otomatis mempercepat pipeline.
5. CPU sempat oversubscribed oleh banyak worker dan agent paralel. Paralelisme
   tanpa pembagian resource yang benar tidak sama dengan throughput lebih
   tinggi.
6. Beberapa agent terputus karena API/session limit, tetapi proses latarnya
   tetap hidup. Ini membuat status “agent gagal” tidak identik dengan “job
   berhenti” dan memerlukan verifikasi PID/GPU secara manual.
7. Ada false alarm seolah dataset/checkpoint hilang ketika disk justru
   membebaskan ruang dari proses di luar sesi. Audit ukuran dan isi direktori
   diperlukan sebelum mengambil tindakan destruktif.
8. Utilitas bootstrap mAP lama hard-coded ke dataset lain dan tidak menerima
   submission dict arbitrary. CI mAP wave harus memakai glue/evaluator yang
   dikunci khusus; keterbatasan ini dicatat, bukan dipalsukan sebagai CI.
9. Disk/pod volume sempat hampir penuh. Checkpoint epoch intermediate yang
   aman dipangkas, tetapi best/last dan artefak reproducibility dipertahankan.
10. Push GitHub sempat gagal karena kredensial lama ditolak. Setelah token baru
    diberikan, commit dokumentasi dan hasil berhasil disinkronkan; token tidak
    disimpan di source atau git.

Hal-hal ini bukan alasan untuk membesar-besarkan hasil, tetapi juga bukan
detail remeh. Satu path salah, satu file TIFF korup, atau satu optimizer yang
dibuat terlalu dini dapat menghabiskan berjam-jam dan menghasilkan cerita
ilmiah yang salah bila tidak diaudit.

### 4. Daftar ringkas semua keluarga yang gagal atau tidak dipromosikan

Daftar ini sengaja panjang supaya sesi berikutnya tidak mengulang ide yang
sudah dijawab:

- early fusion RGB+D `inverse` pada tiga detector;
- counting RGB+D `inverse` dengan Ridge + `F_all`;
- encoding depth `dropout`, `clipped`, dan `valid_mask`;
- mid-fusion depth dengan gate, termasuk patch per-instance yang gagal reload;
- klaim bahwa depth sensor otomatis memperbaiki counting;
- depth sebagai fitur tambahan pada classifier RGB ketika kontribusinya
  kondisional redundan;
- monocular-depth sebagai kanal ke-4 pada 953;
- monocular-depth sebagai kanal ke-4 pada 352 sebagai klaim signifikan;
- monocular-depth di atas sensor depth sebagai kanal kelima;
- penggabungan depth sensor + mono untuk memperbaiki kelas;
- perluasan model ke YOLO26x berdasarkan asumsi kapasitas detector adalah
  bottleneck;
- transfer pretraining 953→352 sebagai solusi domain;
- training gabungan sebagai jaminan mAP lebih tinggi;
- weighted WBF detector;
- WBF class-aware naif sebagai recipe universal;
- WBF IoU di luar operating point `0,60`;
- pair-linker logistic train-only;
- geometry-only linker sebagai solusi association;
- Hungarian/Union-Find tanpa constraint sisi yang benar;
- GSP 953 yang memperbaiki class match tetapi memecahkan count;
- V2 deep-tail/re-ranker yang tidak memenuhi guardrail all-round 953;
- count-meta ensemble yang menurunkan MAE/±1 atau melampaui guardrail lain;
- nonzero blend raw cluster count;
- cluster-quality reranker;
- C2 crop classifier sebagai pengganti penuh soft-vote detector;
- blend classifier yang hanya memberi kenaikan nominal tanpa validasi baru;
- class weighting, focal fine-tune, ordinal/mixed loss sebagai jaminan E2E;
- ConvNeXt/Swin/EfficientNet sebagai opini tambahan yang tidak mengalahkan
  stack robust;
- DINOv2/member stack sebagai klaim test—masih VAL-only;
- OOF expert stacking yang tidak melampaui kandidat terpilih;
- side-aware aggregation dan ordinal logistic head;
- KNN/prototype opinion;
- residual MLP/meta-stack yang tidak memberi all-round gain;
- GPU group-attention yang flat/turun;
- adaptive Hungarian-vs-GSP selector;
- rich graph/count features dan nonlinear count regressor yang hanya membaik
  di CV TRAIN tetapi memburuk di downstream VAL;
- head-aware ranking yang menaikkan class match dengan mengorbankan physical
  F1;
- composition-aware retraining yang lebih rendah dari head lama;
- photometric TTA, flip, rotasi, context TTA;
- hue/color MLP, CLAHE, sharpening, gamma, brightness/contrast;
- gray-world dan mild white-balance;
- detector fine-tuning lokal dan resolusi 1600 sebagai solusi otomatis;
- pipeline test-tuned V2-E-043 sebagai klaim publikasi;
- semua konfigurasi yang menaikkan satu metrik sambil menyembunyikan penurunan
  MAE, ±1, precision, atau physical F1.

### 5. Apa yang benar-benar berhasil dan dipertahankan

Kegagalan di atas tidak berarti semua usaha sia-sia. Komponen berikut punya
nilai dan tetap menjadi fondasi:

- prior arah rotasi bertanda; F1 linker naik dari `0,3979` ke `0,6486` pada
  probe dan melewati gerbang awal;
- crop pooling/classification sebagai pemisahan tugas, walaupun head terbaik
  belum selalu generalisasi;
- WBF class-agnostic sebagai pembuat proposal localization;
- learned edge linker pada proposal nyata, bukan hanya GT box;
- GSP MILP dengan constraint satu proposal per sisi;
- count-aware layer train/VAL-locked untuk menekan duplikasi;
- robust 953 class calibration pada VAL: matched `0,769728`, macro-F1
  `0,617081`, dengan bootstrap VAL yang mendukung—tetapi belum test baru;
- GSP original Depth sebagai profil production/reference, karena physical F1
  dan end-to-end test sudah terbukti kuat;
- mAP deep-tail/re-ranking untuk 953, dengan kenaikan test yang didukung CI;
- calibrated sensor-depth reprojection: pergeseran naive vs calibrated sekitar
  `28,36` px, konsisten dengan dokumentasi misalignment `29` px;
- aturan anchor gate, split tree-level, test guard, checksum, dan pelaporan
  negative result.

### 6. Diagnosis akhir: mengapa target belum terlewati

Urutan bottleneck yang paling didukung bukti adalah:

1. **Kualitas observasi dan domain** — kamera, waktu, jarak, kepadatan, dan
   distribusi kelas berubah. Model tidak bisa memulihkan informasi yang tidak
   muncul atau berubah distribusinya.
2. **Association lintas-sisi** — recall proposal cukup tinggi, tetapi false
   merge dan duplicate cluster merusak counting. Ini sebab GSP membantu F1,
   tetapi threshold/cluster size tidak boleh dipilih dari test.
3. **Ambiguitas kelas bertetangga** — B2↔B3 dan B3↔B4 dominan; hue atau
   sharpening global tidak memberi informasi baru yang stabil.
4. **Depth belum menjadi informasi kondisional yang kuat** — depth memiliki
   relief lokal, tetapi coverage rendah, SNR pixel rendah, dan early stem tidak
   cocok untuk mengonsumsinya.
5. **Daya statistik dan protokol** — validation kecil dan test historis yang
   pernah dibaca membuat banyak “kemenangan” tidak cukup untuk klaim publikasi.

Ini bukan bukti bahwa ruang riset habis. Ini bukti bahwa **ruang optimasi murah
di data dan hipotesis saat ini sudah mengalami plateau lokal**. Untuk melampaui
hasil sekarang, sumber keuntungan berikutnya kemungkinan harus berupa data
berlabel/hard-example yang lebih baik, capture lintas waktu/lokasi, association
berbasis penampilan yang lebih kuat, dan hold-out eksternal yang bersih—bukan
sekadar menambah channel, threshold, atau head lain.

### 7. Status handoff final

- Hasil resmi test-locked tetap tidak berubah.
- Native RGB+D4 tidak dipromosikan menggantikan RGB.
- Late fusion YOLO/RT adalah kandidat validation-only.
- V2 cross-layer Depth adalah kandidat point estimate, CI inconclusive.
- Checkpoint RT seed-1337 yang dihentikan adalah intermediate dan bukan hasil
  resmi.
- Seluruh hasil, failure ledger, artefak, dan keputusan ada di repository;
  laporan rinci berada di [`experiments/EKSPERIMEN.md`](experiments/EKSPERIMEN.md),
  [`results/remote_eval_2026-08-28/PERFORMANCE_WAVE_2026-08-28.md`](results/remote_eval_2026-08-28/PERFORMANCE_WAVE_2026-08-28.md),
  dan [`results/remote_eval_2026-08-28/validation_wave/WAVE2_RECAP.md`](results/remote_eval_2026-08-28/validation_wave/WAVE2_RECAP.md).

Catatan terakhir saya: kita tidak kehabisan ide; kita kehabisan alasan untuk
mempercayai setiap angka validation tanpa audit. Itu frustrasi yang sehat untuk
riset, tetapi juga garis batas yang harus dijaga. Progress yang sebenarnya
bukan hanya angka tertinggi—melainkan mengetahui dengan tepat angka mana yang
boleh dipercaya, komponen mana yang benar-benar membantu, dan kegagalan mana
yang tidak perlu kita ulangi.
