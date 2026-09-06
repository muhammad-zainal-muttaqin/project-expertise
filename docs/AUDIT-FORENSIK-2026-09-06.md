# Audit Forensik Data dan Pipeline — 6 September 2026

Dokumen ini memuat sintesis, kronologi kerja, dan rekomendasi dari audit
independen terhadap korpus dan pipeline `project-expertise`. Seluruh angka
dihitung ulang dari data mentah; log eksperimen berformat *append-only* berada
pada [`experiments/AUDIT-FORENSIK-2026-09-06.md`](../experiments/AUDIT-FORENSIK-2026-09-06.md)
dengan penomoran `AF-E-001` sampai `AF-E-014`.

> [!NOTE]
> Audit ini bersifat **aditif**. Tidak ada entri eksperimen, metrik, atau
> dokumen lama yang diubah. Angka test terkunci `V2-E-045`, `GSP_LINKER`, dan
> `MAP_BOOST` tetap berlaku sebagaimana adanya dan dipakai di sini hanya sebagai
> pembanding.

---

## 1. Ringkasan eksekutif

Pipeline ini **sudah berada pada `91%` dari plafon aritmetis** yang dimungkinkan
oleh kualitas label kematangannya. Dengan lokalisasi dibuat sempurna — setiap
prediksi adalah kotak acuan, presisi dan daya tangkap bernilai `1,0` — `mAP50`
empat kelas hanya mencapai `0,6569`, sedangkan hasil test terkunci proyek dengan
detektor nyata adalah `0,5970`. Ruang perbaikan yang tersisa untuk seluruh
tumpukan detektor karena itu sekitar `6` poin `mAP`, bukan `25`.

Selain itu, dua korpus yang dipakai melatih pipeline ini **tidak memakai
protokol anotasi yang sama**. Perbedaannya terukur pada dua sumbu yang terpisah:
konvensi kotak pembatas, dan kelengkapan anotasi tingkat pohon. Akibatnya,
sebagian besar angka yang selama ini dibaca sebagai kegagalan generalisasi
lintas-domain sesungguhnya mengukur selisih protokol.

### 1.1 Vonis atas target rekayasa

| Target | Capaian kini | Plafon terukur | Vonis |
|---|---:|---:|---|
| `mAP50` deteksi + klasifikasi empat kelas `≥ 0,85` | `0,5970` | `0,6569` | **Tidak terjangkau** |
| `mAP50` dua kelas (siap panen / belum) `≥ 0,85` | `0,7754` | `0,8766` | Terjangkau |
| Pencacahan total per pohon, tepat persis | `0,2741` | `0,2900` | **Tidak terjangkau** |
| Pencacahan siap panen per pohon, ±1 `≥ 95%` | `0,957`–`0,965` | `1,000` | **Tercapai** |

Kolom "capaian kini" untuk baris kedua dan keempat berasal dari replikasi audit
ini (`AF-E-006`, `AF-E-008`), bukan dari eksperimen proyek terdahulu.

### 1.2 Kalibrasi replikasi terhadap angka proyek

Agar perbandingan sah, dua angka replikasi diuji terhadap angka proyek yang
sudah terkunci:

| Besaran | Angka proyek | Replikasi audit | Selisih |
|---|---:|---:|---:|
| `mAP50` empat kelas, test 953 | `0,5435` (YOLO26l, 1.280 px) | `0,5433` (YOLO26s, 960 px) | `0,0002` |
| Pencacahan total ±1, test 953 | `0,6148` (`V2-E-045`) | `0,6100` | `0,0048` |
| Makro-MAE pencacahan *oracle* | `0,275`–`0,277` (`REKAP` §2) | `0,312` | `0,035` |

Ketiganya berdekatan, sehingga angka baru pada dokumen ini dapat dibaca sebagai
perbandingan setara terhadap hasil proyek, bukan sebagai skala yang berbeda.

---

## 2. Kronologi kerja

| Tahap | Yang dikerjakan | Keluaran |
|---|---|---|
| 1 | Pengunduhan `SawitMVC-YOLO` (2,4 GB) dan `SawitMVC-Depth-YOLO` (3,4 GB) dari Hugging Face dengan mekanisme coba-ulang untuk batas laju akun | 65.766 berkas |
| 2 | Pembacaan 27 dokumen repositori (7.910 baris): `HANDOFF`, `ANALISIS_PIPELINE_MENDALAM`, `LAPORAN-AKHIR`, `DIAGNOSIS-DEPTH`, `REKAP`, `STATUS`, `PROPOSAL-Pipeline`, seluruh `metrics/` | peta klaim dan angka rujukan |
| 3 | Forensik data tingkat pohon tanpa model (`AF-E-001` … `AF-E-004`) | ketidakcocokan antar-kampanye, sifat label, sinyal struktur, plafon pencacahan |
| 4 | Pembentukan 18.540 citra terpotong dan pelatihan pengklasifikasi kematangan | akurasi per-*crop* `0,6635` |
| 5 | Perhitungan plafon `mAP50` dan kurva sensitivitas (`AF-E-005`) | plafon `0,6569`; ambang akurasi yang dibutuhkan `≈0,90` |
| 6 | Pelatihan lima detektor YOLO26s dan penyilangan antar-kampanye (`AF-E-006`, `AF-E-007`) | matriks generalisasi dan dekomposisi positif palsu |
| 7 | Pencacahan siap panen ujung ke ujung (`AF-E-008`) | ±1 `0,957`–`0,965` |
| 8 | Fusi penampilan dan struktur pada deteksi nyata (`AF-E-009`) | `+0,0058` makro-F1; satu kontrol negatif |
| 9 | Verifikasi cacat kendala sisi (`AF-E-010`) | `45,3%` klaster melanggar |

Dua hipotesis audit ini sendiri gugur sepanjang pengerjaan dan dicatat sebagai
hasil negatif, bukan dihapus: dugaan bahwa positif palsu Juli sebagian besar
adalah tandan tak berlabel (`AF-E-007`), dan dugaan bahwa dekode monoton per
pohon akan memperbaiki klasifikasi (`AF-E-009` serta
`logs_ringkas/audit_forensik_2026-09-06/exp_fuse2.log`).

---

## 3. Temuan utama

### 3.1 Dua korpus, dua protokol — bukan dua musim

Pada 352 pohon fisik yang sama, jumlah tandan unik per pohon turun dari `9,89`
(Mei) menjadi `3,99` (Juli). Rinciannya bertentangan dengan fenologi pada dua
arah sekaligus: B1 naik `+66%` walaupun kebun mengalami 5–11 rotasi panen,
sedangkan B4 turun `−85%` walaupun inisiasi tandan berlangsung terus sepanjang
tahun. Laju sisi tanpa satu kotak pun naik dari `1,1%` menjadi `14,2%`, dan
memburuk berurutan menurut nomor sisi (`4% → 13% → 19% → 20%`), pola yang tidak
mungkin dihasilkan oleh proses biologis.

Penyilangan detektor memperlihatkan bahwa degradasinya **asimetris menurut jenis
galat**: arah Mei → Juli kehilangan presisi (`0,797 → 0,581`), sedangkan arah
763 → Mei kehilangan daya tangkap (`0,818 → 0,609`). Jurang visual murni akan
menurunkan keduanya bersamaan pada kedua arah.

Dekomposisi positif palsu menunjukkan penyebab utamanya adalah **konvensi kotak
pembatas**: `95,9%` positif palsu pada Juli berpusat di dalam sebuah kotak
acuan tetapi tidak mencapai `IoU 0,5`, dan hanya `3,6%` yang berada di lokasi
tanpa kotak acuan. Pada pasangan yang cocok, sisi kotak model berbanding kotak
acuan bernilai `1,003` di Mei tetapi `0,869` di Juli. Melonggarkan kriteria ke
`IoU ≥ 0,3` menaikkan presisi Juli dari `0,612` menjadi `0,867`. Hipotesis bahwa
Juli menggabungkan beberapa tandan menjadi satu kotak diuji dan ditolak (`0,2%`).

Konvensi kotak tidak menjelaskan selisih jumlah tandan per pohon. Karena itu
terdapat **dua perbedaan protokol yang terpisah**, dan keduanya membuat korpus
`combined1716` melatih detektor dengan supervisi yang tidak konsisten.

### 3.2 Target empat kelas tertutup secara aritmetika

Dengan kotak sempurna dan pengklasifikasi yang mutunya setara dengan seluruh
kerangka utama yang pernah diuji pada repositori ini (`0,62`–`0,70`), `mAP50`
empat kelas berhenti di `0,6569`. Untuk mencapai `0,85` dibutuhkan akurasi
kematangan `≈0,90`. Runtuhkan taksonominya, dan plafon yang sama menjadi
`0,8766` (B1 lawan sisanya) atau `0,8891` (B1+B2 lawan B3+B4).

Pelatihan detektor nyata mengonfirmasi arah dan besarannya: `0,5433` pada empat
kelas menjadi `0,7754` pada dua kelas, dengan model, resolusi, resep, dan data
yang identik.

### 3.3 Label kematangan adalah properti pohon

Nol dari 7.328 tandan multi-tampak pada korpus 953 memiliki tampak yang
berselisih kelas. Kelas ditetapkan satu kali per tandan fisik lalu disalin.
Konsekuensinya, keputusan kelas harus diambil **setelah** penautan lintas-sisi,
dan `mAP50` sadar-kelas per citra bukan metrik yang tepat untuk pipeline ini.

### 3.4 Sinyal struktur nyata, tetapi manfaat praktisnya kecil

Korelasi Spearman antara kelas dan peringkat vertikal dalam pohon `−0,616`.
Penataan monoton dengan komposisi *oracle* mencapai makro-F1 `0,6237` tanpa satu
piksel pun. Namun pada deteksi nyata, fusi penampilan dan struktur hanya
memberi `+0,0058` makro-F1 (`0,6470 → 0,6528`), jauh di bawah `+0,0220` yang
diperoleh pada kotak acuan. Komponen ini layak dipertahankan karena biayanya
nol, tetapi bukan pengungkit besar.

### 3.5 Cacat kendala sisi masih ada, tetapi dorman (dikoreksi pada `AF-E-014`)

`scripts/sweep_remote_pipeline.py:123` mengisi himpunan sisi dengan indeks
proposal, bukan sisi fisik. Cacat itu nyata dan sudah diperbaiki.

Namun besarnya dampak yang semula saya laporkan **keliru dan sudah dikoreksi
pada `AF-E-014`**. Angka `45,3%` pada `AF-E-010` diukur pada daftar tepi
geometri sederhana tanpa penugasan Hungarian. Pada jalur sweep yang sebenarnya —
yang menerapkan `linear_sum_assignment` per pasangan sisi lebih dahulu —
pelanggarannya `0,00%` untuk `max_size ≤ 3` pada kedua mode pasangan, dan baru
muncul (`7,95%`) pada `max_size 4` dengan `pair_mode` "all". Menjalankan ulang
seluruh grid 630 konfigurasi pada 953 dan Depth menghasilkan **nol perubahan**:
profil terbaik, F1 fisik, dan MAE seluruhnya identik. Penilaian
`docs/ANALISIS_PIPELINE_MENDALAM.md` §5.5 bahwa cacat ini dorman terbukti benar,
dan tidak ada angka test terkunci yang berubah karenanya.

---

## 4. Rekomendasi

Diurutkan menurut rasio dampak terhadap biaya.

| # | Tindakan | Biaya | Dasar bukti |
|---|---|---|---|
| 1 | ~~Perbaiki `UF.sides`, lalu jalankan ulang penelusuran parameter~~ **sudah dikerjakan**; hasilnya nol perubahan pada 630 konfigurasi, kecuali profil `max_size 4` yang masih perlu diperiksa | selesai | `AF-E-010`, `AF-E-014` |
| 2 | Definisikan ulang target rekayasa ke taksonomi dua kelas; laporkan empat kelas sebagai metrik ordinal ±1 | jam | `AF-E-005`, `AF-E-006` |
| 3 | Ganti besaran pencacahan utama menjadi cacah tandan siap panen per pohon dengan toleransi ±1 | jam | `AF-E-004`, `AF-E-008` |
| 4 | Audit kelengkapan dan konvensi anotasi lintas-kampanye oleh anotator, berstrata | hari | `AF-E-001`, `AF-E-007` |
| 5 | Ganti protokol evaluasi menjadi *leave-one-campaign-out*; simpan dekomposisi per pohon agar selang kepercayaan berpasangan dapat dihitung | hari | `AF-E-007` |
| 6 | Pindahkan keputusan kelas ke tingkat tandan fisik setelah penautan, dengan fitur struktur sebagai *prior* lunak | hari | `AF-E-002`, `AF-E-009` |
| 7 | Ganti ambang *singleton* global dengan pengklasifikasi terlatih, karena laju tampak tunggal bergantung kelas | minggu | `AF-E-004` |
| 8 | Latih ulang detektor sepenuhnya *class-agnostic*; keluarkan keputusan kematangan dari detektor | minggu | `AF-E-006`, `AF-E-007` |
| 9 | Selaraskan konvensi kotak antar-kampanye, atau anotasi ulang ke satu standar | bulan | `AF-E-007` |

### Yang sebaiknya dihentikan

Sejalan dengan `docs/ANALISIS_PIPELINE_MENDALAM.md` §8 dan diperkuat oleh
`AF-E-005`: kerangka utama yang lebih besar, TTA, koreksi fotometrik dan warna,
*reranker* kualitas klaster, serta regresor pencacahan nonlinear tidak lagi
produktif. Plafonnya tidak berada di sana.

---

## 5. Nilai ilmiah yang paling kuat

Kontribusi yang paling layak diterbitkan dari pekerjaan ini bukan "RGB+D
menaikkan `mAP`", melainkan bahwa **dua korpus multi-tampak atas pohon yang
identik dapat berbeda dua kali lipat dalam kelengkapan anotasi dan berbeda
sistematis dalam konvensi kotak, dan bahwa perbedaan itu terbaca sebagai
pergeseran domain apabila tidak diaudit di tingkat pohon.** Repositori ini sudah
memegang seluruh buktinya, termasuk 352 pohon fisik yang difoto dua kali —
kondisi kontrol yang jarang tersedia pada set data lapangan.

---

## 6. Tindak lanjut yang sudah dijalankan (`AF-E-011` … `AF-E-013`)

Setelah cacat `AF-E-010` diperbaiki, rekomendasi §4 nomor 2, 3, dan 6 diuji
secara ujung ke ujung, bukan sebagai plafon. Hasilnya terbelah dan keduanya
dilaporkan.

### 6.1 Yang mengungguli hasil terkunci

| Metrik test 953 | Pipeline Panen | Pembanding proyek |
|---|---:|---:|
| Cacah **B1 siap panen**, toleransi ±1 | **0,970** | tidak pernah dilaporkan |
| Makro-F1 kelas empat | **0,6692** | `0,6034` (GSP) |
| Akurasi ordinal ±1 | **0,9946** | tidak pernah dilaporkan |
| Akurasi dua kelas matang/belum | **0,8678** | tidak pernah dilaporkan |

Ketiganya berasal dari satu keputusan rancangan: kelas ditentukan di tingkat
tandan fisik melalui skor ordinal kontinu, dan keputusan kasar maupun halus
menjadi ambang pada skor yang sama.

### 6.2 Yang masih tertinggal

| Metrik test 953 | Pipeline Panen | GSP terkunci |
|---|---:|---:|
| F1 fisik | 0,7619 | **0,8387** |
| Cacah total, MAE | 1,402 | **1,363** |
| Cacah total, ±1 | 0,568 | **0,6370** |
| Akurasi kelas empat | 0,7161 | **0,7442** |

Penyebabnya teridentifikasi: penaut audit ini memakai proposal satu detektor,
bukan WBF tiga detektor, dan daya tangkap fisiknya hanya `0,6878`. GSP MILP
proyek tetap merupakan penaut yang lebih baik. **Rekomendasi yang benar karena
itu adalah menggabungkan keduanya** — penaut GSP proyek dengan tahap kelas
ordinal tingkat tandan dari audit ini — bukan mengganti salah satunya.

### 6.3 Koreksi terhadap usulan taksonomi

Menggabungkan B1+B2 sebagai "matang" tidak didukung data maupun kartu dataset.
Kartu `SawitMVC-YOLO` menyebut B1 sebagai *optimal harvest stage* sedangkan B2
masih *transitioning*. Secara empiris, cacah B1 mencapai ±1 `0,970` sedangkan
cacah B1+B2 hanya `0,765`. Besaran operasional yang benar adalah **B1**.

Selain itu, batas B2\|B3 adalah batas tersulit dalam data ini — pada matriks
konfusi `AF-E-009`, B2↔B3 menyumbang 195 galat berbanding 57 untuk B1↔B2 —
sehingga menaruhnya sebagai akar hierarki keras akan mengunci sekitar 15% tandan
pada jalur yang salah. Skor ordinal dengan dua ambang menghindari hal itu.

---

## 7. Artefak

| Lokasi | Isi |
|---|---|
| `experiments/AUDIT-FORENSIK-2026-09-06.md` | Log *append-only* `AF-E-001` … `AF-E-010` |
| `results/audit_forensik_2026-09-06/` | Seluruh metrik JSON dan bukti citra |
| `results/audit_forensik_2026-09-06/panen/` | Metrik Pipeline Panen (`AF-E-011` … `AF-E-013`) |
| `logs_ringkas/audit_forensik_2026-09-06/` | Log eksekusi, `results.csv`, dan `args.yaml` tiap pelatihan |
| `scripts/audit_forensik/` | 21 skrip analisis dan eksperimen |
| Bucket `ULM-DS-Lab/project-expertise-backup`, awalan `audit_forensik_2026-09-06/` | Bobot lima detektor, bobot pengklasifikasi *crop*, dump probabilitas |

Prosedur reproduksi tercantum pada
[`results/audit_forensik_2026-09-06/MANIFEST.md`](../results/audit_forensik_2026-09-06/MANIFEST.md).
