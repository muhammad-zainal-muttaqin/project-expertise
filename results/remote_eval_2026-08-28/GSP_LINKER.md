# GSP Linker — Lembar Bukti Eksperimen

- **Tanggal pencatatan:** 2026-08-28 (eksekusi val: 2026-08-27; eksekusi test-locked: 2026-08-28)
- **Skrip:** `scripts/link_global_setpartition.py` (disalin dari `/workspace/gsp_linker/link_global_setpartition.py`; berkas asli di luar repo karena `/workspace/project-expertise` bersifat *read-only*)
- **Artefak numerik:** `results/remote_eval_2026-08-28/gsp_artifacts/{953,depth}/results_val.json`, `results/remote_eval_2026-08-28/gsp_artifacts/{953,depth}/results_test_locked.json`
- **Status:** catatan ini berada di *staging mirror* (`/workspace/repo_staging/project-expertise/`) dan belum mendapat ID eksperimen resmi (`V2-E-###`/`PT-E-###`); penomoran dan integrasi ke `experiments/` serta `docs/LAPORAN-AKHIR.md` menjadi keputusan pemelihara repo yang sebenarnya.

---

## Rancangan Eksperimen

**Masalah pada garis dasar pembanding (*baseline*).** Pipeline empat sisi saat ini menautkan proposal lintas sisi memakai penugasan Hungarian per pasangan sisi berdekatan (`train_detection_edge_linker.build_edges`), lalu menggabungkannya secara serakah lewat *union-find* (`sweep_remote_pipeline.UF`/`sweep_remote_pipeline.clusters`). Inspeksi kode sumber menemukan bahwa *constraint* "maksimal satu proposal per sisi fisik dalam satu klaster" pada kelas `UF` bersifat **vacuous** (tidak pernah benar-benar aktif):

```python
# sweep_remote_pipeline.py, baris 119–124
class UF:
    def __init__(self, n: int, max_size: int):
        self.parent = list(range(n))
        self.size = [1] * n
        self.sides = [{i} for i in range(n)]   # << bug: indeks proposal, bukan dets[i]["side"]
        self.max_size = max_size
```

`self.sides[i]` diinisialisasi dengan indeks larik proposal itu sendiri (`i`), bukan `dets[i]["side"]` (sisi fisik kamera). Akibatnya, uji `self.sides[a] & self.sides[b]` pada `union()` tidak pernah mendeteksi dua anggota dari sisi fisik yang sama, karena indeks larik selalu unik per proposal. Pembanding pada `eval_remote_pipeline_postprocess.UnionFind`/`link_clusters` mengimplementasikan pelacakan sisi dengan benar (`side_sets = {i: {dets[i]["side"]} for i in range(len(dets))}`), sehingga cacat ini spesifik pada `sweep_remote_pipeline.py` — modul yang justru dipakai oleh seluruh jalur evaluasi terkunci (`evaluate_remote_class_head.evaluate_payload` → `evaluate_remote_count_reconciled.selected_clusters` → `sweep_remote_pipeline.clusters`).

Secara empiris cacat ini *dormant* pada konfigurasi yang dipakai di seluruh eksperimen ini (`pair_mode="adjacent"`, `max_size ≤ 4`, `n_sides=4`): menutup siklus yang mengulang satu sisi memerlukan minimal lima anggota (satu putaran penuh siklus 0–1–2–3–0 ditambah satu ulangan), dan batas ukuran klaster (`max_size ≤ 4`) sudah memblokir penggabungan kelima secara independen dari uji sisi yang cacat tersebut. Cacat ini tidak diperbaiki pada sesi ini karena `sweep_remote_pipeline.py` berada di repo *read-only*; ia dicatat sebagai motivasi konseptual sekaligus item audit (lihat *Batasan Validitas & Audit*).

**Metode yang diusulkan: *Global Set-Partition* (GSP).** Alih-alih penugasan Hungarian per pasangan sisi diikuti union serakah, GSP menyelesaikan **partisi global optimal per pohon** lewat *mixed-integer linear programming* (MILP) eksak:

1. Model log-odds tepi (`ExtraTreesClassifier` terlatih terpisah per dataset, fitur 65 dimensi dari `train_detection_edge_linker.pair_features`) menghasilkan probabilitas $p$ untuk setiap pasangan proposal lintas sisi berdekatan dalam satu pohon, dihitung satu kali lewat `model.predict_proba` per pohon (`tree_pair_probs`).
2. Kandidat klaster dienumerasi sebagai *connected subset* di atas graf berambang ($p \geq p_{floor}=0,02$), dengan enumerasi kanonis per simpul awal $v$ (menaik) yang hanya meluas ke tetangga berindeks $> v$ — menghindari penghitungan ganda tanpa memerlukan *union-find*. Kandidat ditolak apabila memuat dua anggota dari sisi fisik yang sama (*constraint* ≤ 1 proposal/sisi dijamin secara struktural pada tahap enumerasi, bukan lewat pemeriksaan *union-find* yang rentan cacat seperti pada garis dasar pembanding). Ukuran kandidat dibatasi 2–4 anggota; jika jumlah kandidat mentah per pohon melampaui 20.000, ambang $p_{floor}$ digandakan dan enumerasi diulang (dicatat sebagai eskalasi).
3. Skor kandidat $= \sum_{\text{pasangan berdekatan} \in \text{klaster}} \big[\operatorname{logit}(p_{\text{pasangan}}) - \tau\big]$, memakai **seluruh** pasangan berdekatan di dalam kandidat (termasuk yang di bawah ambang enumerasi — pasangan tersebut tetap dikenai penalti lewat $\operatorname{logit}(p)$ yang rendah).
4. Partisi optimal per pohon diselesaikan lewat `scipy.optimize.milp` (maksimasi jumlah skor kandidat terpilih dengan *constraint* setiap proposal muncul di ≤ 1 klaster terpilih); solusi cadangan serakah deterministik (urut skor menurun, ambil bila belum bertumpang tindih) dipakai hanya jika `milp` gagal/*exception*. Klaster terpilih diteruskan ke evaluator terkunci lewat "tepi keputusan" berbobot $1,0$ (`(1,0; anggota[0]; anggota[k])`), sehingga ambang tautan $0,5$ pada `evaluate_remote_class_head.evaluate_payload` bersifat *vacuous by design* (seluruh tepi keputusan bernilai $1,0 \geq 0,5$), dan singleton/batas ukuran/peringkat/agregasi kelas tetap identik-bit dengan evaluator baku.

**Konfigurasi tetap:** `proposal_min=0,125`; `pair_mode=adjacent`; `p_floor=0,02`; ukuran enumerasi maksimum $=4$. **Grid validasi (VAL):** $\tau_{prob} \in \{0,05;\ 0,10;\ 0,15;\ 0,20;\ 0,25;\ 0,35;\ 0,50\}$ ($\tau=\operatorname{logit}(\tau_{prob})$); `max_size` $\in \{3;4\}$; `singleton` $\in \{0,10;\ 0,15;\ 0,20;\ 0,25\}$; `rank` $\in \{$*score*, *support*, *max\_member*$\}$. Model terlatih: dataset 953 → `extra` saja; dataset *depth* → `extra` dan `hist_deep`.

**Gerbang anchor (*anchor gate*).** Sebelum grid GSP dijalankan, empat profil Hungarian+*union-find* yang sudah terkunci sebelumnya direproduksi lewat `edge.build_edges` + `evaluate_payload` dan dibandingkan dengan angka acuan (toleransi $\pm 0,003$) sebagai syarat gerbang wajib lulus. Kegagalan pada tahap ini akan menghentikan seluruh sesi tanpa menjalankan grid.

---

## Temuan Empiris Terukur

### Gerbang anchor — seluruh 4 profil eksak (toleransi ±0,003, diff aktual ≈10⁻⁵)

| Anchor | Dataset/model | Profil (*link*/singleton/max\_size/rank) | Metrik | Acuan | Aktual |
|---|---|---|---|---|---|
| A | 953 / `extra` | 0,15 / 0,15 / 4 / *score* | F1 fisik | 0,8232 | 0,8232 |
| A | | | MAE | 1,2527 | 1,2527 |
| A | | | ±1 | 0,6703 | 0,6703 |
| A | | | *matched\_class\_accuracy* | 0,7542 | 0,7542 |
| B | 953 / `extra` | 0,20 / 0,25 / 3 / *max\_member* | F1 fisik | 0,8307 | 0,8307 |
| B | | | *matched\_class\_accuracy* | 0,7538 | 0,7538 |
| C | *depth* / `extra` | 0,10 / 0,20 / 3 / *support* | F1 fisik | 0,8471 | 0,8471 |
| C | | | MAE | 0,9487 | 0,9487 |
| C | | | *matched\_class\_accuracy* | 0,8359 | 0,8359 |
| D | *depth* / `hist_deep` | 0,85 / 0,15 / 3 / *support* | F1 fisik | 0,8007 | 0,8007 |
| D | | | MAE | 0,8034 | 0,8034 |
| D | | | ±1 | 0,8376 | 0,8376 |
| D | | | *matched\_class\_accuracy* | 0,8622 | 0,8622 |

Gerbang **lulus** untuk kedua dataset; grid GSP dijalankan penuh.

### Grid VAL — dataset 953 (91 pohon)

- Baris grid: 168 (model `extra` saja; $2 \times 7$ kombinasi `max_size`×$\tau_{prob}$, masing-masing dievaluasi pada $4\times3=12$ kombinasi *singleton*×*rank*).
- Solver per pohon: 1.274 percobaan (91 pohon × 14 kombinasi `max_size`×$\tau_{prob}$) → `milp`=1.268, `empty`=6 (tidak ada kandidat berskor positif — pohon jatuh ke seluruh *singleton*), `greedy_fallback`=0.
- Kandidat mentah terenumerasi (dijumlah seluruh pohon, model `extra`): 14.802. **Tidak ada eskalasi ambang** ($p_{floor}$ tetap 0,02 di seluruh pohon).
- **Referensi Hungarian+UF Anchor A:** F1=0,8232, presisi=0,8206, *recall*=0,8259, MAE=1,2527, ±1=0,6703, *matched*=0,7542 (*matched*=773), makro-F1=0,6014.
- **Referensi Hungarian+UF Anchor B:** F1=0,8307, presisi=0,8871, *recall*=0,7810, MAE=1,7363, ±1=0,5055, *matched*=0,7538 (*matched*=731), makro-F1=0,6061.
- **GSP terbaik-menurut-kelas** (`extra`, $\tau_{prob}=0,20$, *singleton*=0,25, `max_size`=3, *rank*=*max\_member*; baris identik pada `max_size`=4): F1=0,8313, presisi=0,8873, *recall*=0,7821, MAE=1,7473, ±1=0,5055, *matched*=0,7555 (*matched*=732), makro-F1=0,6079.
- **GSP terbaik-menurut-fisik** (`extra`, $\tau_{prob}=0,05$, *singleton*=0,25, `max_size`=3, *rank*=*score*): F1=0,8393, presisi=0,8785, *recall*=0,8034, MAE=1,5385, ±1=0,5604, *matched*=0,7447 (*matched*=752), makro-F1=0,6062.

### Grid VAL — dataset *depth* (117 pohon)

- Baris grid: 336 (2 model × 168 baris).
- Solver `extra`: 1.638 percobaan (117 pohon × 14) → `milp`=1.546, `empty`=92, `greedy_fallback`=0; kandidat mentah=2.602; tidak ada eskalasi.
- Solver `hist_deep`: 1.638 percobaan → `milp`=1.554, `empty`=84, `greedy_fallback`=0; kandidat mentah=9.716; tidak ada eskalasi.
- **Referensi Hungarian+UF Anchor C** (`extra`): F1=0,8471, presisi=0,8996, *recall*=0,8004, MAE=0,9487, ±1=0,7692, *matched*=0,8359 (*matched*=457), makro-F1=0,6700.
- **Referensi Hungarian+UF Anchor D** (`hist_deep`): F1=0,8007, presisi=0,8137, *recall*=0,7881, MAE=0,8034, ±1=0,8376, *matched*=0,8622 (*matched*=450), makro-F1=0,6572.
- **GSP terbaik-menurut-kelas** (`extra`, $\tau_{prob}=0,50$, *singleton*=0,25, `max_size`=3, *rank*=*support*; baris identik pada `max_size`=4 dan *rank*=*max\_member*): F1=0,8054, presisi=0,8951, *recall*=0,7320, MAE=1,0940, ±1=0,7521, *matched*=0,8636 (*matched*=418), makro-F1=0,6503.
- **GSP terbaik-menurut-fisik** (`extra`, $\tau_{prob}=0,10$, *singleton*=0,20, `max_size`=3, *rank*=*support*): F1=0,8526, presisi=0,9055, *recall*=0,8056, MAE=0,9316, ±1=0,7863, *matched*=0,8457 (*matched*=460), makro-F1=0,6807.

### Hasil TEST-LOCKED — dataset 953 (profil terkunci: Hungarian, `extra`, *link*=0,15, *singleton*=0,15, `max_size`=4, *rank*=*score*)

$n_{pohon}=135$; dump vote: `fused_combined1716_test_rebuilt/SawitMVC_YOLO__wbf_softvote.npz`; model: `detection_edge_linker_953_v2/extra.joblib`.

- **Deteksi fisik:** presisi=0,8444, *recall*=0,8331, F1=0,8387 (*tp*=1.118, *pred\_clusters*=1.324, *gt\_bunches*=1.342).
- **Pencacahan (*counting*):** MAE=1,3630, akurasi eksak=0,2741, ±1=0,6370, akurasi vektor eksak=0,0519 (denominator $n=135$ pohon).
- **Klasifikasi:** *matched\_class\_accuracy*=0,7442 = 832/1.118 (*matched*=1.118); makro-F1=0,6034.
- **F1 per kelas:** B1=0,7465; B2=0,4706; B3=0,6850; B4=0,5114.
- **Matriks konfusi** (baris = prediksi, kolom = [B1, B2, B3, B4, tidak cocok/*unmatched*]):

  | Prediksi \\ Acuan | B1 | B2 | B3 | B4 | Tak cocok |
  |---|---|---|---|---|---|
  | B1 | 81 | 18 | 0 | 0 | 5 |
  | B2 | 17 | 92 | 22 | 0 | 14 |
  | B3 | 2 | 101 | 524 | 79 | 118 |
  | B4 | 0 | 0 | 47 | 135 | 69 |
  | Tak cocok (GT) | 13 | 35 | 113 | 63 | — |

- **Selang kepercayaan 95% *bootstrap*** (2.000 resampel pohon dengan pengembalian, `RandomState(seed=42)`): F1 fisik [0,8174; 0,8587]; MAE [1,1630; 1,5852]; ±1 [0,5556; 0,7185]; *matched\_class\_accuracy* [0,7112; 0,7735]; makro-F1 [0,5655; 0,6382].
- **Pemeriksaan agregasi per pohon vs. jalur penuh:** F1 fisik, MAE, ±1, *matched\_class\_accuracy*, dan makro-F1 seluruhnya `match: true` (kesetaraan eksak, bukan pendekatan).
- *Solver tag counts*: tidak berlaku (*linker* Hungarian, bukan GSP).

**Dibandingkan dengan baseline test-locked yang dirujuk orkestrator** (F1 0,8043 / MAE 1,3926 / ±1 0,6148 / *matched* ≈0,7111): F1 0,8387 vs 0,8043 (Δ=+0,0344); MAE 1,3630 vs 1,3926 (Δ=−0,0296); ±1 0,6370 vs 0,6148 (Δ=+0,0222); *matched* 0,7442 vs ≈0,7111 (Δ=+0,0331).

### Hasil TEST-LOCKED — dataset *depth* (profil terkunci: GSP, `extra`, $\tau_{prob}=0,10$, *singleton*=0,20, `max_size`=3, *rank*=*support*)

$n_{pohon}=110$; dump vote: `fused_combined1716_test_rebuilt/SawitMVC_Depth_YOLO__wbf_softvote.npz`; model: `detection_edge_linker_depth_v1/extra.joblib`.

- **Deteksi fisik:** presisi=0,8926, *recall*=0,8175, F1=0,8534 (*tp*=457, *pred\_clusters*=512, *gt\_bunches*=559).
- **Pencacahan:** MAE=0,7727, akurasi eksak=0,4455, ±1=0,8545, akurasi vektor eksak=0,2727 (denominator $n=110$ pohon).
- **Klasifikasi:** *matched\_class\_accuracy*=0,8162 = 373/457 (*matched*=457); makro-F1=0,6519.
- **F1 per kelas:** B1=0,7578; B2=0,7230; B3=0,7092; B4=0,4176.
- **Matriks konfusi:**

  | Prediksi \\ Acuan | B1 | B2 | B3 | B4 | Tak cocok |
  |---|---|---|---|---|---|
  | B1 | 61 | 4 | 0 | 0 | 2 |
  | B2 | 27 | 154 | 21 | 1 | 24 |
  | B3 | 0 | 8 | 139 | 14 | 16 |
  | B4 | 0 | 2 | 7 | 19 | 13 |
  | Tak cocok (GT) | 6 | 31 | 48 | 16 | — |

- **Selang kepercayaan 95% *bootstrap*** (2.000 resampel, seed=42): F1 fisik [0,8301; 0,8761]; MAE [0,6091; 0,9455]; ±1 [0,7818; 0,9182]; *matched\_class\_accuracy* [0,7765; 0,8556]; makro-F1 [0,6046; 0,6918].
- **Pemeriksaan agregasi per pohon vs. jalur penuh:** seluruh lima metrik `match: true` (kesetaraan eksak).
- ***Solver tag counts*** (110 pohon, satu percobaan `solve_partition` per pohon pada profil terkunci): `milp`=109, `empty`=1, `greedy_fallback`=0.

**Dibandingkan dengan baseline test-locked yang dirujuk orkestrator** (F1 0,8069 / MAE 0,8909 / ±1 0,8091 / *matched* 0,8031 / makro-F1 0,6047): F1 0,8534 vs 0,8069 (Δ=+0,0465); MAE 0,7727 vs 0,8909 (Δ=−0,1182); ±1 0,8545 vs 0,8091 (Δ=+0,0454); *matched* 0,8162 vs 0,8031 (Δ=+0,0131); makro-F1 0,6519 vs 0,6047 (Δ=+0,0472).

---

## Keputusan Metodologis

1. **Dataset *depth* memakai GSP sebagai profil terkunci** (`extra`, $\tau_{prob}=0,10$, *singleton*=0,20, `max_size`=3, *rank*=*support*). Pada grid VAL, profil ini mendominasi referensi Hungarian+UF Anchor C pada **kelima** metrik utama secara bersamaan (bukan hanya rata-rata gabungan): F1 0,8526 vs 0,8471 (Δ=+0,0055); MAE 0,9316 vs 0,9487 (Δ=−0,0171); ±1 0,7863 vs 0,7692 (Δ=+0,0171); *matched\_class\_accuracy* 0,8457 vs 0,8359 (Δ=+0,0098); makro-F1 0,6807 vs 0,6700 (Δ=+0,0107). Karena tidak ada satu pun metrik yang mengalami penurunan performa, keputusan penguncian pada profil ini konsisten dengan pedoman *all-rounder* orkestrator.
2. **Dataset 953 tetap memakai profil *incumbent* Hungarian+UF (Anchor A)**, bukan hasil GSP terbaik. Kandidat GSP terbaik-menurut-kelas (`extra`, $\tau_{prob}=0,20$, *singleton*=0,25, `max_size`=3, *rank*=*max\_member*) memang unggul tipis pada *matched\_class\_accuracy* (0,7555 vs 0,7542; Δ=+0,0013), tetapi **gagal memenuhi pedoman *all-rounder* pra-deklarasi orkestrator** karena mengalami penurunan performa tajam pada pencacahan: MAE memburuk dari 1,2527 menjadi 1,7473 (Δ=+0,4946) dan akurasi ±1 memburuk dari 0,6703 menjadi 0,5055 (Δ=−0,1648). Tidak satu pun profil pada grid GSP 953 menunjukkan keunggulan performa yang konsisten pada seluruh metrik sekaligus dibanding Anchor A; profil *incumbent* dipertahankan.
3. **Data *test* dibuka tepat satu kali per dataset** pada sesi ini (2026-08-28), masing-masing lewat `--stage test` dengan profil yang telah dikunci di langkah 1–2. Skrip menegakkan disiplin ini secara teknis: `stage_test` menolak (`SystemExit`) menjalankan ulang apabila `results_test_locked.json` untuk kombinasi `{output-root}/{dataset}/` sudah ada, dan untuk *linker* GSP menolak menjalankan profil yang tidak terdaftar persis di `gsp_grid_results` pada `results_val.json` (mencegah penguncian profil yang tidak pernah diuji di VAL). Kedua guard ini diverifikasi lulus sebelum eksekusi: untuk 953, cabang Hungarian pada `stage_test` diverifikasi *call-identical* dengan jalur referensi VAL yang lulus gerbang anchor (`edge.build_edges(...,"adjacent")` → `head_eval.evaluate_payload(...,LINK,singleton,max_size,rank,0,class_prior,0,None,"mean",0)` dengan `LINK=tau_prob`); untuk *depth*, profil GSP terkunci dikonfirmasi ada persis satu kali di `gsp_grid_results` (F1=0,8526413345690453, MAE=0,9316239316239316, cocok dengan angka yang dikutip orkestrator).

---

## Batasan Validitas & Audit

1. **953/GSP: fragmentasi merusak pencacahan.** Seluruh baris grid GSP 953 yang mencapai *matched\_class\_accuracy* setara atau lebih tinggi dari *incumbent* (≈0,75) membayar harga pada MAE dan ±1 (mis. MAE naik ke 1,7473–1,7473 pada dua baris teratas-menurut-kelas, dibanding 1,2527 pada *incumbent*). Pola ini konsisten dengan hipotesis bahwa ambang $\tau_{prob}$ yang tinggi (0,20) membuat MILP lebih sering memilih klaster kecil atau *singleton* murni (presisi tinggi, 0,8871–0,8873, tetapi *recall* rendah, ≈0,782), sehingga jumlah klaster terprediksi tersebar berlebihan (*fragmentasi*) relatif terhadap jumlah kelompok fisik acuan. **Kandidat penutup untuk sesi berikutnya (belum diimplementasikan):** lapisan penutup jumlah (*count reconciliation*) yang mempelajari target hitung dari statistik klaster GSP itu sendiri (mis. jumlah klaster ber-skor positif per ukuran, distribusi $\operatorname{logit}(p)$ terpilih per pohon) — bukan hanya dari fitur proposal WBF mentah seperti model *ridge* pencacahan yang dipakai saat ini (`evaluate_remote_count_reconciled.feature_vector`) — untuk mengurangi kesenjangan antara jumlah klaster GSP dan jumlah kelompok fisik acuan tanpa mengorbankan presisi klasifikasi yang sudah diperoleh GSP.
2. ***Constraint* sisi vacuous pada `sweep_remote_pipeline.UF` (lihat *Rancangan Eksperimen*):** dormant secara empiris untuk seluruh konfigurasi yang dipakai di sesi ini (`pair_mode=adjacent`, `max_size ≤ 4`), tetapi belum diperbaiki di kode sumber karena `/workspace/project-expertise` bersifat *read-only*. Apabila di masa depan pipeline Hungarian+UF dijalankan dengan `pair_mode="all"` (memungkinkan tepi antar-sisi berseberangan) atau `max_size > 4`, cacat ini berpotensi meloloskan klaster dengan dua proposal dari sisi fisik yang sama tanpa terdeteksi, karena `self.sides` melacak indeks larik proposal, bukan `dets[i]["side"]`. Metode GSP pada catatan ini tidak bergantung pada `union-find` tersebut dan sudah menjamin ≤ 1 proposal/sisi secara struktural pada tahap enumerasi kandidat (`enumerate_candidates`), sehingga tidak mewarisi cacat ini.
3. **Data test 953 dan *depth* kini sudah terbuka** (masing-masing dibuka tepat satu kali, "pembukaan pertama", pada 2026-08-28 melalui sesi ini). **Setiap eksekusi `--stage test` berikutnya terhadap dataset yang sama — baik dengan profil yang identik maupun berbeda — wajib diberi label eksplisit sebagai "pembukaan kedua" (atau seterusnya) di catatan eksperimen**, karena pembukaan berulang tanpa pelabelan melanggar protokol *single-look* pada data test. Guard teknis pada skrip (`SystemExit` bila `results_test_locked.json` sudah ada di `{output-root}/{dataset}/`) mencegah penimpaan tidak sengaja pada `--output-root` yang sama, tetapi menjalankan dengan `--output-root` berbeda akan melewati guard ini sepenuhnya — disiplin pelabelan pembukaan berulang tetap menjadi tanggung jawab manual, bukan tanggung jawab skrip.
4. **Cakupan grid tidak lengkap secara kombinatorial.** GSP hanya diuji dengan `pair_mode="adjacent"` (sama seperti *incumbent*); `pair_mode="all"` tidak dievaluasi pada sesi ini dan berada di luar cakupan. Ukuran enumerasi kandidat dibatasi maksimum 4 anggota (sama dengan $n_{sisi}=4$); klaster berukuran >4 secara definisi tidak mungkin terjadi pada pohon empat sisi tanpa mengulang sisi fisik, sehingga pembatasan ini tidak membuang ruang solusi yang valid.
5. **Reproduksibilitas *bootstrap*.** Selang kepercayaan memakai *resampling* pada tingkat pohon (bukan pada tingkat proposal atau pasangan), `numpy.random.RandomState(seed=42)`, 2.000 resampel dengan pengembalian. Makro-F1 pada tiap resampel dihitung ulang dari matriks konfusi kumulatif hasil penjumlahan pohon-pohon terresampel (bukan rata-rata makro-F1 per pohon), konsisten dengan definisi makro-F1 pada evaluator penuh.
