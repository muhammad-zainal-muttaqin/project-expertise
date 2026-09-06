---
name: perbaikan-bahasa-penyampaian
description: >-
  Standardizes scientific writing, Indonesian formal grammar (EYD Edisi V / PUEBI), anti-calque translations, mathematical/statistical notations, negative constraints (Do's and Don'ts), and clear explanatory narrative structure for academic and technical documentation (.md files, reports, summaries). Activate when the user requests language improvement, rewriting in standard Indonesian, fixing calque translations, organizing chronological evidence sheets, or formatting mathematical/statistical symbols.
---

# Standar Penulisan Ilmiah Baku, Diksi Ilmiah, & Perbaikan Bahasa Penyampaian (EYD V / PUEBI)

Skill ini berfungsi sebagai pedoman baku penulisan ilmiah formal Bahasa Indonesia (EYD Edisi V / PUEBI) untuk laporan teknis, publikasi akademik, ringkasan eksekutif, dan dokumentasi riset di bidang Kecerdasan Buatan, Komputer Visi, dan Sains Data.

---

## 1. Prinsip Diksi Ilmiah & Anti-Calque (Pencegahan Terjemahan Harfiah Mesin)

### Rasional Pemilihan Kata
Terjemahan harfiah (*calque*) dari bahasa Inggris sering kali menghasilkan struktur kalimat yang janggal, ambigu, atau salah makna (misalnya kata *"loss"* diterjemahkan sebagai *"kerugian"* alih-alih *"fungsi rugi / degradasi performa"*, atau *"appearance"* diterjemahkan *"penampilan"* alih-alih *"kemunculan objek"*). 

Bahasa ilmiah baku mengutamakan **presisi makna**, **kebakuan istilah serapan**, dan **kejelasan relasi sebab-akibat**.

### Tabel Komprehensif: Padanan Dilarang vs Wajib Digunakan

#### A. Ranah Machine Learning, Optimasi, & Arsitektur Model
| Bentuk Dilarang / Terjemahan Harfiah (Hindari) | Bentuk Ilmiah Baku EYD V (Wajib Digunakan) | Alasan & Rasional Pemilihan Kata |
|---|---|---|
| *loss model besar / rugi performa* | **nilai fungsi rugi (*loss*) / degradasi performa** | *"Rugi"* bermakna finansial; dalam optimasi gunakan *fungsi rugi* atau *penurunan performa*. |
| *train on test* | **pelatihan pada data uji / kebocoran partisi (*train-on-test*)** | Istilah formal untuk kontaminasi partisi evaluasi. |
| *holdout set* | **partisi data terisolasi / himpunan uji terpisah** | Menjelaskan fungsi partisi data yang tidak tersentuh pelatihan. |
| *best observed* | **nilai terbaik yang teramati** | Diksi formal dalam observasi empiris. |
| *fine-tuning / finetune* | **penyesuaian terarah (*fine-tuning*) / adaptasi model** | Menjelaskan proses transfer pembelajaran secara spesifik. |
| *early stopping* | **penghentian dini (*early stopping*)** | Menghentikan pelatihan sebelum jadwal maksimum tercapai. |
| *freeze weight / freeze layer* | **pembekuan bobot layer (*freezing*)** | Menjaga bobot parameter agar tidak terbarui saat *backpropagation*. |
| *backbone* | **kerangka utama (*backbone*) / pengekstraksi fitur** | Jaringan ekstraktor fitur dasar sebelum kepala prediksi. |
| *head / classifier head* | **kepala klasifikasi / modul kepala (*head*)** | Lapisan proyeksi keluaran tugas spesifik. |
| *bottleneck* | **hambatan struktural (*bottleneck*) / leher botol** | Titik konvergensi atau penyempitan kapasitas representasi. |
| *gating mechanism / gate* | **mekanisme penapisan ber-gerbang (*gating*)** | Modul pembobotan fitur dinamis. |
| *weight sharing* | **pembagian bobot parameter (*weight sharing*)** | Penggunaan matriks bobot yang sama antar-cabang. |
| *data leakage / leak* | **kebocoran data (*data leakage*)** | Kontaminasi informasi masa depan/uji ke partisi latih. |
| *trade-off* | **kompromi performa (*trade-off*) / pertukaran timbal-balik** | Keseimbangan antara dua metrik yang saling bertolak belakang. |

#### B. Ranah Komputer Visi & Pengolahan Citra
| Bentuk Dilarang / Terjemahan Harfiah (Hindari) | Bentuk Ilmiah Baku EYD V (Wajib Digunakan) | Alasan & Rasional Pemilihan Kata |
|---|---|---|
| *appearance feature / penampilan objek* | **kemunculan objek (*appearance*) / fitur visual** | *"Penampilan"* merujuk pada performa panggung; gunakan *kemunculan visual*. |
| *bounding box / bbox* | **kotak pembatas (*bounding box*)** | Koordinat persegi penanda lokalisasi objek. |
| *crop / crop citra* | **citra terpotong (*crop*) / pemotongan wilayah objek** | Potongan spasial citra fokus target. |
| *ground truth (GT)* | **nilai acuan kebenaran (*ground truth*) / label acuan riil** | Label anotasi yang diverifikasi sebagai acuan faktual. |
| *pseudo-depth / depth palsu* | **estimasi kedalaman semu (*pseudo-depth*)** | Estimasi kedalaman non-sensorik fisik. |
| *screening* | **penyaringan awal (*screening*)** | Evaluasi eliminasi kandidat model secara cepat. |
| *spatial pooling* | **agregasi spasial (*spatial pooling*)** | Perataan atau reduksi dimensi matriks piksel secara spasial. |
| *temporal domain shift* | **pergeseran domain temporal (*temporal shift*)** | Perubahan distribusi statistik akibat perbedaan rentang waktu perekaman. |
| *noise / berderau* | **variasi acak (*noise*) / derau sensor** | Gangguan stokastik non-sinyal pada data masukan. |
| *letterboxing / pad* | **penambahan batas tepi (*padding / letterboxing*)** | Penyesuaian rasio aspek citra tanpa distorsi geometri. |
| *occlusion / kehalang* | **oklusi objek / objek terhalang pelepah** | Kondisi target tertutup parsial oleh objek latar depan. |

#### C. Ranah Statistika, Metrik, & Evaluasi
| Bentuk Dilarang / Terjemahan Harfiah (Hindari) | Bentuk Ilmiah Baku EYD V (Wajib Digunakan) | Alasan & Rasional Pemilihan Kata |
|---|---|---|
| *CI95 memuat nol / overlap nol* | **selang kepercayaan 95% mencakup nilai nol (tidak signifikan secara statistik)** | Rumusan baku uji signifikansi inferensial. |
| *tidak pernah menang / kalah* | **tidak menunjukkan keunggulan performa / mengalami penurunan** | Bahasa ilmiah bersifat objektif, bukan kompetitif informal. |
| *menyebut kenaikan / klaim naik* | **disimpulkan sebagai peningkatan / terbukti meningkatkan** | Diksi verifikasi berbasis pembuktian data. |
| *baseline* | **garis dasar pembanding (*baseline*) / model acuan** | Titik tolak komparasi kinerja algoritma. |
| *oracle* | **model batas atas teoretis (*oracle*)** | Estimasi performa maksimum dengan asumsi informasi sempurna. |
| *counting* | **pencacahan (*counting*)** | Penghitungan kuantitas diskret objek fisik. |
| *ablation study* | **studi ablasi / uji eliminasi komponen** | Eksperimen pelepasan modul untuk mengukur kontribusi marginal. |
| *point estimate* | **estimasi titik** | Nilai tunggal metrik tanpa interval ketidakpastian. |
| *statistically significant* | **signifikan secara statistik** | Hasil uji hipotesis dengan $p$-value $< \alpha$. |
| *inconclusive* | **belum konklusif / belum dapat diputuskan** | Bukti empiris belum cukup untuk menolak hipotesis nol. |
| *falsified / dipalsukan* | **gugur secara empiris (*falsified*) / tertolak** | *"Dipalsukan"* berarti manipulasi curang (*fake*); gunakan *gugur/tertolak*. |

#### D. Ranah Umum (Calque Non-Teknis, di Luar ML/CV/Statistika)
| Bentuk Dilarang / Terjemahan Harfiah (Hindari) | Bentuk Ilmiah Baku EYD V (Wajib Digunakan) | Alasan & Rasional Pemilihan Kata |
|---|---|---|
| *secara praktis* | **dalam praktiknya** | Padanan alami, bukan terjemahan literal *practically*. |
| *bekerja baik / bekerja ampuh* | **efektif / berfungsi dengan baik** | Hindari calque *works well*. |
| *ada untuk / hadir untuk* | **dirancang untuk** | Calque *exists to*; nyatakan tujuan rancangannya. |
| *secara independen* (non-statistik) | **secara mandiri** | Calque *independently* di luar konteks uji statistik. |
| *menunjuk ke* (non-spasial) | **mengacu pada** | *Point to* diterjemahkan literal padahal maknanya rujukan. |
| *deliverable* | **luaran** | Istilah manajemen proyek, ada padanan baku. |
| *di-cover* | **dibahas / tercakup** | Hindari verba hibrida asing-Indonesia. |
| *resource eksternal* | **sumber rujukan / sumber belajar** | *Resource* diterjemahkan sesuai konteks, bukan dipertahankan mentah. |
| *aturan jempol* | **aturan praktis / kaidah umum** | Calque literal *rule of thumb*. |
| *kerangka mental* | **kerangka berpikir** | Calque literal *mental framework*. |

#### E. Ranah Metafora Teknis (Kiasan Fisik/Sosial untuk Konsep Abstrak)
Kiasan fisik atau sosial untuk operasi komputasional terdengar naratif, bukan ilmiah. Sebut nama operasi atau objek teknisnya secara langsung.

| Kiasan (Hindari) | Istilah Langsung (Wajib) |
|---|---|
| *"bahan baku"* untuk data/input | **data, dataset, atau masukan (*input*)** |
| *"pabrik"* untuk model | **model atau arsitektur** |
| *"gerbang"* untuk fungsi aktivasi | **fungsi aktivasi (mis. ReLU)** |
| *"sinyal galat mengalir/merambat"* | **gradien dihitung pada lintasan mundur (*backward pass*)** |
| *"data mengalir ke depan"* | **lintasan maju (*forward pass*) menghitung keluaran dari masukan** |
| *"model menyepakati"* | **model memuat/menghasilkan** |

Uji: "apakah ini istilah teknis baku yang dipakai di dokumentasi/paper, atau kiasan buatan sendiri?" Jika kiasan, ganti dengan istilah teknisnya.

---

## 2. Katalog Larangan Khusus (Negative Constraints / Hal yang Dilarang vs Wajib)

### A. Larangan Antropomorfisme Model (Larangan Menjadikan AI/Model Seperti Manusia)
Model pembelajaran mesin tidak "berpikir", "bingung", "tahu", atau "melihat". Model memetakan tensor numerik, menghitung probabilitas, atau mengalami tumpang tindih representasi.

* ❌ **Dilarang**: *"Model bingung membedakan antara B2 dan B3 karena warnanya mirip."*
* ✅ **Wajib**: *"Terjadi konfusi representasi antara kelas B2 dan B3 akibat tingginya kemiripan fotometrik pada ruang warna RGB."*
* ❌ **Dilarang**: *"Detektor tahu posisi tandan tapi tidak tahu kelasnya."*
* ✅ **Wajib**: *"Detektor berhasil melokalisasi koordinat spasial tandan dengan presisi tinggi, namun mengalami galat pada penentuan label kematangan."*
* ❌ **Dilarang**: *"Model memutuskan untuk berhenti belajar di epoch 15."*
* ✅ **Wajib**: *"Mekanisme penghentian dini (*early stopping*) menghentikan pelatihan pada epoch 15 setelah metrik validasi mengalami stagnasi."*

---

### B. Larangan Bahasa Percakapan, Slang, & Diksi Informal
Dokumen ilmiah wajib bebas dari kata-kata informal dan emotif.

| Kata Informal (DILARANG) | Padanan Ilmiah Baku (WAJIB) |
|---|---|
| *banget / sangat amat* | **sangat / secara signifikan / secara substansial** |
| *cuma / cuma dapat* | **hanya / tercatat sebesar** |
| *lumayan / lumayan bagus* | **cukup memadai / moderat / mencatat peningkatan terukur** |
| *kayak / seperti kayak* | **sebagaimana / serupa dengan** |
| *nggak / ndak / tidak pernah ada* | **tidak terdapat / ketiadaan bukti empiris** |
| *curang (pada evaluasi)* | **mengalami kebocoran informasi (*data leakage*) / evaluasi tidak terisolasi** |
| *hancur / jeblok / anjlok parah* | **mengalami penurunan drastis / mengalami degradasi performa substansial** |
| *mentok* | **mencapai batas saturasi / menyentuh batas teoretis** |

---

### C. Larangan Klaim Kausalitas Palsu & Generalisasi Berlebihan
Jangan menyatakan klaim superioritas jika selang kepercayaan masih memuat nilai nol atau ukuran sampel tidak mencukupi.

* ❌ **Dilarang**: *"Model baru kami membuktikan bahwa penambahan depth selalu meningkatkan counting tandan sawit."*
* ✅ **Wajib**: *"Uji empiris menunjukkan bahwa penambahan kanal kedalaman meningkatkan estimasi lokalisasi ($AP50 = 0,7636$ vs $0,7358$), namun peningkatan pencacahan belum mencapai signifikansi statistik formal pada split uji 352 pohon ($P = 0,921$, selang kepercayaan 95% mencakup nilai nol)."*
* ❌ **Dilarang**: *"Dataset 352 pohon jelek karena mAP-nya rendah."*
* ✅ **Wajib**: *"Rendahnya nilai $mAP50$ pada dataset 352 pohon disebabkan oleh pergeseran fenologi temporal kebun ($\sim 80\text{ hari}$) yang menyebabkan kelangkaan ekstrem sampel latih kelas B3 dan B4, bukan karena kelemahan arsitektur model."*

---

### D. Larangan Pengubahan Angka Historis Log Append-Only
* ❌ **Dilarang**: Mengubah, membulatkan secara sepihak, atau menghapus angka metrik riil pada berkas *log append-only* (`experiments/EKSPERIMEN.md`, `pipeline-pertandan/EKSPERIMEN.md`).
* ✅ **Wajib**: Memperbaiki redaksi pengantar, tata bahasa, dan keterbacaan penjelasan tanpa memutasi satu pun angka empiris yang menjadi rekaman eksperimen.

---

### E. Larangan Pembenaran "Sudah Konvensi/Konsisten dengan Teks Lama"
Pola tidak baku yang sudah terlanjur ada pada bagian lain dokumen bukan alasan sah untuk mengulanginya pada teks baru, atau untuk membiarkannya saat ditemukan pada audit berikutnya. Setiap kemunculan pola terlarang pada tabel Bagian 1 dan katalog Bagian 2 wajib diperbaiki secara independen, termasuk yang sudah "menjadi konvensi" pada dokumen yang sama.

* ❌ **Dilarang**: Membiarkan *"CI95 mencakup nol"* pada baris tabel karena baris lain di dokumen yang sama sudah memakai bentuk itu lebih dulu.
* ✅ **Wajib**: Memperbaiki seluruh kemunculan pola terlarang, termasuk yang sudah ada sebelum revisi berjalan. Argumen "supaya konsisten dengan yang lama" tidak menggugurkan kewajiban perbaikan pada Bagian 1/2; yang sah justru sebaliknya, memperbaiki semuanya agar konsisten dengan tabel, bukan mereplikasi kekeliruan yang sudah ada.

---

### F. Label Ringkas Berulang (Legenda, Lencana/*Pill*, Sel Tabel) vs Prosa Naratif
Rumusan WAJIB pada tabel Bagian 1 dan Bagian 2 (mis. *"selang kepercayaan 95% mencakup nilai nol (tidak signifikan secara statistik)"*) disusun untuk prosa naratif, satu kali per klaim, di dalam paragraf atau Lembar Bukti Bagian 4. Rumusan itu tidak ditempel apa adanya secara berulang pada elemen ringkas yang berulang (legenda diagram, lencana status pada sel tabel, label sumbu): hasilnya jargon berulang yang tetap melanggar semangat "pembaca dipercaya, bukan diceramahi", meski setiap kata sudah lolos tabel anti-calque secara harfiah.

* ❌ **Dilarang**: Menempel kalimat lengkap "selang kepercayaan 95% mencakup nilai nol (belum signifikan)" pada setiap lencana tabel dan setiap entri legenda diagram yang berulang.
* ✅ **Wajib**: Pada elemen ringkas berulang, nyatakan kesimpulan langsung ("signifikan secara statistik" / "belum signifikan secara statistik"). Jelaskan mekanismenya (selang kepercayaan mencakup/tidak mencakup nilai nol) hanya sekali, pada keterangan gambar atau paragraf pengantar terdekat.

---

### G. Larangan Personifikasi Umum, Bahasa Moral, dan Bahasa Dramatis untuk Hal Teknis
Bagian A membahas antropomorfisme khusus model AI ("model bingung", "model tahu"). Larangan berikut lebih luas: benda mati apa pun (bukan hanya model), termasuk tensor, keputusan, atau proses, tidak melakukan kata kerja manusia atau bertindak sendiri tanpa pelaku yang disebut.

* ❌ **Dilarang**: *"Tensor mengalami transformasi."* / *"Keputusan itu muncul dengan sendirinya."*
* ✅ **Wajib**: *"Tensor ditransformasi oleh lapisan konvolusi."* / *"Tim peneliti memutuskan ... berdasarkan ..."* (sebutkan pelakunya)

Bahasa moral/administratif untuk fakta teknis dihindari karena menyiratkan penilaian etis, bukan deskripsi teknis:

| Hindari | Pakai |
|---|---|
| *"jujur"* (maksud teknis) | **sesuai / konsisten dengan** |
| *"dapat dipertanggungjawabkan"* | **dapat diverifikasi ulang / memiliki catatan lengkap** |
| *"dengan alasan"* | **yang sesuai / yang menjelaskan pemilihan X** |

Kata kerja dan kata sifat dramatis untuk fenomena teknis (termasuk kesalahan operasional) diganti bentuk deskriptif dan tenang:

| Drama (Hindari) | Deskriptif (Wajib) |
|---|---|
| *"mematikan", "fatal"* (untuk galat) | **utama, signifikan, mendasar** |
| *"liar", "brutal", "sembrono"* | **tidak terstruktur, acak, keliru** |
| *"menabrak", "menginfeksi", "menjangkiti"* | **memengaruhi, mengubah, tercampur** |
| *"jeblok", "anjlok", "ambruk"* (untuk metrik) | **menurun, lebih rendah dari** |

---

### H. Larangan Intensifier Kosong dan Pembuka Berbasa-basi (*Throat-Clearing*)
Kata penegas generik yang tidak menambah makna, dan kalimat pembuka yang mengumumkan pentingnya sesuatu alih-alih langsung menyatakannya, keduanya melemahkan bobot argumen ilmiah.

| Hindari | Pakai / Tindakan |
|---|---|
| *"benar-benar", "sungguhan", "sebenarnya"* (kosong) | hapus jika makna kalimat tidak berubah tanpanya |
| *"dataset nyata", "kondisi nyata"* tanpa lawan eksplisit | hapus penegasnya, cukup "dataset", "kondisi" (pertahankan hanya bila ada kontras eksplisit dengan data sintetis/*toy*) |
| *"kritis"* sebagai penegas umum | **penting, menentukan** |
| *"intuitif"* untuk penjelasan | **mudah dipahami** |
| *"Perlu ditekankan bahwa ..."*, *"Perlu dicatat bahwa ..."* di awal kalimat | hapus, langsung nyatakan isinya |
| *"Ingat,"*, *"Perlu diingat,"* di awal kalimat | hapus; jika isinya penting, ia berdiri sendiri tanpa penanda itu |
| *"inilah bagian paling krusial"* tanpa isi konkret | sebutkan bagian dan alasannya secara langsung |

Uji cepat: hapus frasa itu dari kalimat. Jika makna kalimat tidak berubah, frasa itu filler dan wajib dibuang.

---

### I. Larangan Kontras Retoris sebagai Pemanis Kalimat
Struktur "bukan X, melainkan Y" atau "bukan sekadar X" sering dipakai sebagai hiasan retoris, bukan untuk membedakan dua hal yang benar-benar perlu dibedakan.

* ❌ **Dilarang**: *"Ini bukan sekadar penurunan performa, melainkan kegagalan mendasar arsitektur."*
* ✅ **Wajib**: *"Penurunan performa ini menunjukkan kegagalan mendasar arsitektur."* (nyatakan poinnya langsung)
* Pengecualian sah: kontras yang informatif dan spesifik, mis. *"logit mentah, bukan probabilitas"*, tetap dipertahankan karena membedakan dua besaran yang berbeda secara teknis.

---

## 3. Standar Notasi Matematika, Statistika, & Tipografi Baku

```text
[Pedoman Cepat Notasi Angka & Simbol]
1. Desimal              : koma (,)              -> 0,6012  (BUKAN 0.6012)
2. Ribuan               : titik (.)             -> 3.992 citra (BUKAN 3,992 atau 3992)
3. Tanda Minus          : simbol asli − / $\minus$ -> −0,0476 (BUKAN tanda hubung -0.0476)
4. Selang Kepercayaan   : [min; max]            -> CI95 [−0,0671; −0,0274] (BUKAN [-0.06, -0.02])
5. Rentang Nilai/Waktu  : en dash (–)           -> B1–B4, 10–11 Agu 2026 (BUKAN B1-B4)
6. Simbol Variabel      : cetak miring (italic) -> p-value, n sampel, IoU, Δ mAP, F1, mAP50
```

### Tanda Baca Umum (Non-Numerik)
* **Tidak ada tanda pisah panjang (*em dash*, `—`, U+2014)** di mana pun dalam prosa. Ganti dengan koma, titik dua, tanda kurung, atau spasi-hyphen-spasi (` - `). Pengecualian: simbol data untuk sel tabel kosong/tidak berlaku (mis. `&mdash;` sebagai isi sel tabel numerik) tetap sah karena itu notasi data, bukan tanda baca kalimat.
* En dash (`–`) hanya untuk rentang numerik atau tanggal (`B1–B4`, `10–11 Agu 2026`), tidak untuk memisahkan klausa kalimat.
* Tanda kutip ASCII lurus (`"..."`), bukan kutip melengkung tipografis.
* Setiap kalimat memiliki subjek dan predikat eksplisit (SPOK), dapat dibaca lantang tanpa tersendat; tidak ada fragmen berlabel (`"Kekuatan: ..."`, `"Asumsi: tidak ada struktur khusus."`).

### Tabel Komparasi Notasi: Dilarang vs Wajib

| Ranah Notasi | Penulisan Dilarang (Salah) | Penulisan Baku (Wajib) |
|---|---|---|
| **Nilai Desimal** | `mAP50 = 0.5435`, `acc = 74.39%` | **$mAP50 = 0,5435$**, **$\text{Akurasi} = 74,39\%$** |
| **Kuantitas Ribuan** | `18,540 boxes`, `3992 images` | **$18.540\text{ kotak pembatas}$**, **$3.992\text{ citra}$** |
| **Nilai Negatif / Selisih** | `-0.0476`, `delta = -2.3%` | **$\minus 0,0476$** atau **`−0,0476`**, **$\Delta = \minus 2,3\%$** |
| **Selang Kepercayaan** | `95% CI: (-0.02, 0.07)`, `CI95 [-0.05 - 0.05]` | **CI95 $[\minus 0,0270; +0,0739]$** |
| **Simbol Statistik** | `p-value < 0.05`, `N=410`, `mAP50-95` | ***$p$-value* $< 0,05$**, **$n = 410$**, **$mAP50\text{--}95$** |
| **Rentang Kategori** | `kelas B1-B4`, `tanggal 12-15 Agustus` | **kelas B1–B4**, **tanggal 12–15 Agustus 2026** |

---

## 4. Struktur Lembar Bukti Empiris Empat Bagian

Setiap simpul eksperimen (`V2-E-###` atau `PT-E-###`) dan bab dokumentasi evaluasi wajib disusun ke dalam 4 bagian sistematis:

```mermaid
graph TD
    B1["1. Rancangan Eksperimen\n(Desain komparasi, input, parameter, kontrol)"] --> B2["2. Temuan Empiris Terukur\n(Metrik kuantitatif, p-value, CI95 berpasangan)"]
    B2 --> B3["3. Keputusan Metodologis\n(Putusan ilmiah: terkonfirmasi / gugur / pivot)"]
    B3 --> B4["4. Batasan Validitas & Audit\n(Silsilah data, ancaman validitas, kaveat)"]
```

### Contoh Penerapan Paragraf Baku (Sebelum vs Sesudah)

#### ❌ Contoh Buruk (Informal, Calque, Notasi Salah):
> *"Kita coba train yolo 4ch pake depth mono di 953 pohon. Hasilnya mAP50 dapet 0.4960 dibanding rgb yang 0.5436, jadi loss -0.0476. CI95 memuat nol (-0.067, -0.027) dan model kalah telak. Jadi mono depth jelek banget dan gak guna, mending di drop aja."*

#### ✅ Contoh Baku Sesuai Skill (Formal, Anti-Calque, Presisi Ilmiah):
> **Simpul V2-E-027: Evaluasi Pengaruh Estimasi Kedalaman Monokular pada Korpus 953 Pohon**
> 1. **Rancangan Eksperimen**: Pengujian komparasi terkontrol mengevaluasi penambahan kanal kedalaman estimasi monokular (`yolo26l-depth.pt`) sebagai masukan 4-kanal pada arsitektur YOLO26l beresolusi 1.280 piksel (60 *epoch*, *cosine learning rate*) terhadap garis dasar pembanding RGB 3-kanal pada split uji SawitMVC (2.612 kotak anotasi).
> 2. **Temuan Empiris Terukur**: Penambahan depth monokular menyebabkan **penurunan performa yang signifikan** sebesar $\Delta = \mathbf{\minus 0,0476}$ ($mAP50 = \mathbf{0,4960}$ berbanding kontrol RGB $\mathbf{0,5436}$). Evaluasi bootstrap 2.000 ulangan berpasangan menghasilkan selang kepercayaan 95% **$[\minus 0,0671; \minus 0,0274]$** ($P(\Delta > 0) = 0,000$, terbukti signifikan secara statistik).
> 3. **Keputusan Metodologis**: Hipotesis keunggulan depth monokular dinyatakan **gugur secara empiris**. Jalur integrasi kedalaman monokular dihentikan dari pipeline utama.
> 4. **Batasan Validitas & Audit**: Penurunan performa terjadi konsisten di seluruh kelas kematangan (B1–B4). Berkas log eksekusi tersimpan pada [`logs_ringkas/eval_sel6_953_rgbmono.log`](file:///D:/Work/Assisten-Dosen/project-expertise/logs_ringkas/eval_sel6_953_rgbmono.log).

---

## 5. Daftar Periksa Mandiri (Checklist Audit Sebelum Menyimpan Berkas)

Sebelum menyelesaikan penulisan dokumen markdown:

- [ ] **Kepatuhan Anti-Calque**: Tidak ada kata *loss* (untuk penurunan performa), *appearance* mentah, *CI memuat nol*, *ground truth* mentah, dll.
- [ ] **Bebas Antropomorfisme**: Tidak ada kalimat *model bingung*, *model tahu*, atau *model berpikir*.
- [ ] **Format Angka**: Seluruh desimal menggunakan koma (`,`) dan ribuan menggunakan titik (`.`).
- [ ] **Simbol Minus**: Seluruh bilangan negatif menggunakan minus asli `−` atau `$\minus$`.
- [ ] **Selang Kepercayaan**: Menggunakan format $[\text{min}; \text{max}]$ dengan pemisah titik koma.
- [ ] **Keterlacakan Berkas**: Seluruh rujukan skrip (`.py`), log (`.log`), dan data (`.json`) memiliki tautan markdown aktif yang valid.
- [ ] **Integritas Log**: Angka riil historis tetap dipertahankan tanpa manipulasi.
- [ ] **Verifikasi tersistematis, bukan sampel**: setiap frasa yang memakai singkatan atau pola pada tabel Bagian 1 dan katalog Bagian 2 dicocokkan satu per satu lewat pencarian teks (bukan dibaca sekilas), termasuk teks yang sudah ada sebelum revisi berjalan pada dokumen yang sama. "Sudah konsisten dengan bagian lain" bukan alasan untuk melewati pengecekan (lihat Bagian 2.E).
- [ ] **Label ringkas vs prosa**: rumusan lengkap dari tabel Bagian 1/2 dipakai satu kali dalam prosa/keterangan gambar; elemen ringkas berulang (legenda, lencana, sel tabel) memakai kesimpulan langsung, bukan kalimat lengkap yang ditempel berulang (lihat Bagian 2.F).
- [ ] **Tanda baca umum**: tidak ada tanda pisah panjang (*em dash*) di luar sel tabel data kosong; kutip lurus, bukan kutip melengkung; setiap kalimat SPOK penuh tanpa fragmen berlabel (lihat Bagian 3).
- [ ] **Bebas metafora teknis, personifikasi umum, bahasa dramatis/moral, dan intensifier kosong**: lihat Bagian 1.E dan Bagian 2.G/H.
- [ ] **Bebas kontras retoris berlebihan**: "bukan X, melainkan Y" dan "bukan sekadar" hanya dipakai bila kontrasnya informatif dan spesifik, bukan hiasan (lihat Bagian 2.I).
