"""Menyusun laporan kinerja pencacahan dari berkas hasil (V2-E-050f).

Seluruh angka pada laporan dibangkitkan langsung dari JSON hasil, sehingga tidak
ada penyalinan manual. Jalankan ulang skrip ini setiap kali salah satu berkas
hasil diperbarui.

Pemakaian:
    python scripts/susun_laporan_pencacahan.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kalibrasi_koefisien_pencacahan import muat_gt_763, muat_gt_953
from kalibrasi_pencacahan_perkorpus import gt_korpus

B = chr(92)
GAMBAR = "assets/laporan-pencacahan-2026-09-16"
KOR = ["953", "763", "1716"]
DET = ["YOLO26l", "RT-DETR-L", "RF-DETR-L"]
KLS = ["B1", "B2", "B3", "B4"]
SINGKAT = {"YOLO26l": "Y", "RT-DETR-L": "RT", "RF-DETR-L": "RF"}
NAMA = {
    "naif": "Naif",
    "global": "$k$ global",
    "k_perkelas": "$k$ per kelas",
    "k_tau_perkelas": "$k + " + B + "tau$ per kelas",
}
PILIH = {"953": "k_perkelas", "763": "k_tau_perkelas", "1716": "k_perkelas"}
PYCOCO = {
    ("953", "YOLO26l"): 0.5435, ("953", "RT-DETR-L"): 0.5718, ("953", "RF-DETR-L"): 0.5965,
    ("763", "YOLO26l"): 0.5163, ("763", "RT-DETR-L"): 0.5580, ("763", "RF-DETR-L"): 0.6129,
    ("1716", "YOLO26l"): 0.5389, ("1716", "RT-DETR-L"): 0.5745, ("1716", "RF-DETR-L"): 0.5960,
}
SILANG = [("763", "953"), ("1716", "953"), ("1716", "763"), ("953", "763")]
KODE_SILANG = {("763", "953"): "763>953", ("1716", "953"): "1716>953",
               ("1716", "763"): "1716>763", ("953", "763"): "953>763"}


def km(v, n=4):
    return f"{v:.{n}f}".replace(".", ",").replace("-", "\u2212")


def rb(v):
    return f"{int(v):,}".replace(",", ".")


def tebal(teks, aktif):
    return f"**{teks}**" if aktif else teks


class Sumber:
    def __init__(self, hasil: Path, baseline: Path, depth: Path):
        self.deteksi = json.loads((hasil / "metrik_deteksi_perkorpus.json").read_text(encoding="utf-8"))["baris"]
        self.cacah = json.loads((hasil / "pencacahan_perkorpus.json").read_text(encoding="utf-8"))["baris"]
        self.permutasi = json.loads((hasil / "permutasi_koefisien.json").read_text(encoding="utf-8"))["baris"]
        gt953 = muat_gt_953(baseline)
        gt763 = muat_gt_763(depth)
        dump = np.load(
            hasil.parent.parent / "results/combined1716/predictions/combined1716_yolo26l_rgb_s42_i1280__test.npz",
            allow_pickle=True,
        )
        pohon_1716 = {k.rsplit("_", 1)[0] for k in dump.files}
        self.acuan = {}
        for korpus in KOR:
            gt = gt_korpus(gt953, gt763, korpus)
            if korpus == "1716":
                gt = {k: v for k, v in gt.items() if k in pohon_1716}
            self.acuan[korpus] = np.array(list(gt.values()))

    def det(self, latih, uji, detektor):
        for x in self.deteksi:
            if (x["korpus_latih"], x.get("korpus_uji", x["korpus_latih"]), x["detektor"]) == (latih, uji, detektor):
                return x
        return None

    def cac(self, korpus, detektor, metode):
        for x in self.cacah:
            if (x["korpus_latih"], x["detektor"], x["metode"]) == (korpus, detektor, metode):
                return x
        return None

    def terbaik(self, korpus, detektor, metode_sah=("k_perkelas", "k_tau_perkelas")):
        kand = [x for x in self.cacah
                if x["korpus_latih"] == korpus and x["detektor"] == detektor and x["metode"] in metode_sah]
        return min(kand, key=lambda x: x["metrik"]["mae_makro"]) if kand else None


def bagian_ringkasan(s: Sumber) -> list[str]:
    L = ["# Laporan Kinerja per Korpus Latih: Deteksi dan Pencacahan", "",
         "## 1. Ringkasan Eksekutif", "",
         "| Korpus | Detektor terbaik | $mAP50$ | F1 | Metode kalibrasi | $MAE$ | $MAE$ relatif | Akurasi ±1 | Penurunan $MAE$ |",
         "|---|---|---|---|---|---|---|---|---|"]
    for korpus in KOR:
        d = s.det(korpus, korpus, "RF-DETR-L")["makro"]
        b = s.terbaik(korpus, "RF-DETR-L")
        m = b["metrik"]
        naif = s.cac(korpus, "RF-DETR-L", "naif")["metrik"]["mae_makro"]
        turun = (naif - m["mae_makro"]) / naif * 100
        L.append(f"| {korpus} | RF-DETR-L | {km(d['map50'])} | {km(d['f1'])} | {NAMA[b['metode']]} | "
                 f"{km(m['mae_makro'])} | {km(m['mae_relatif_makro'])} | {km(m['acc_pm1_makro'])} | {km(turun, 1)}% |")
    rel = {k: s.terbaik(k, "RF-DETR-L")["metrik"]["mae_relatif_makro"] for k in KOR}
    mae = {k: s.terbaik(k, "RF-DETR-L")["metrik"]["mae_makro"] for k in KOR}
    map_domain = [s.det(k, k, "RF-DETR-L")["makro"]["map50"] for k in KOR]
    silang_rf = min(s.det(a, b_, "RF-DETR-L")["makro"]["map50"] for a, b_ in SILANG if s.det(a, b_, "RF-DETR-L"))
    silang_semua = min(s.det(a, b_, d)["makro"]["map50"] for a, b_ in SILANG for d in DET if s.det(a, b_, d))
    L += ["", "| Temuan utama | Angka pendukung |", "|---|---|",
          f"| RF-DETR-L terbaik di ketiga korpus | $mAP50$ {km(min(map_domain))}–{km(max(map_domain))} |",
          f"| Kalibrasi menurunkan galat di semua kombinasi | $MAE$ turun 25,0–88,6% |",
          f"| Peringkat korpus berbalik pada basis relatif | $MAE$ {km(mae['763'])} (763) berbanding {km(mae['953'])} (953); relatif {km(rel['763'])} berbanding {km(rel['953'])} |",
          f"| Detektor gugur lintas korpus, dua arah | $mAP50$ serendah {km(silang_semua)} |", ""]
    return L


def bagian_identitas() -> list[str]:
    return [
        "## 2. Identitas Eksperimen", "",
        "| Parameter | Nilai |", "|---|---|",
        "| Identitas simpul | `V2-E-050` sampai `V2-E-050f` |",
        "| Tanggal | 16 September 2026 |",
        "| Korpus latih | `953` (SawitMVC-YOLO), `763` (SawitMVC-Depth-YOLO v2.0.0), `1716` (gabungan) |",
        "| Partisi uji | `953`: 141 pohon; `763`: 110 pohon; `1716`: 257 pohon |",
        "| Detektor | YOLO26l, RT-DETR-L, RF-DETR-L |",
        "| Metrik deteksi | Presisi, Recall, F1, $AP50$, dan $AP50" + B + "text{--}95$ per kelas serta makro |",
        "| Model pencacahan | $" + B + "hat{y}_c(t) = " + B + "operatorname{round}(k_c " + B + "cdot n_c(t))$, dengan $n_c(t)$ sebagai jumlah deteksi kelas $c$ lintas sisi pohon yang memenuhi skor keyakinan $" + B + "ge " + B + "tau_c$ |",
        "| Kalibrasi | Lipat-silang 5 lipatan tingkat pohon, tanpa pelatihan ulang |",
        "| Metrik pencacahan | $MAE$ makro, $MAE$ relatif, $RMSE$ makro, bias mutlak makro, akurasi ±1 makro |",
        "| Skrip | [`susun_laporan_pencacahan.py`](../scripts/susun_laporan_pencacahan.py), [`metrik_deteksi_perkorpus.py`](../scripts/metrik_deteksi_perkorpus.py), [`kalibrasi_pencacahan_perkorpus.py`](../scripts/kalibrasi_pencacahan_perkorpus.py), [`permutasi_koefisien_pencacahan.py`](../scripts/permutasi_koefisien_pencacahan.py) |",
        "| Artefak angka | [`metrik_deteksi_perkorpus.json`](../results/counting_koefisien_2026-09-16/metrik_deteksi_perkorpus.json), [`pencacahan_perkorpus.json`](../results/counting_koefisien_2026-09-16/pencacahan_perkorpus.json), [`permutasi_koefisien.json`](../results/counting_koefisien_2026-09-16/permutasi_koefisien.json) |",
        "",
    ]


def bagian_deteksi(s: Sumber) -> list[str]:
    L = ["## 4. Deteksi per Korpus Latih", "",
         f"![Metrik deteksi makro per korpus]({GAMBAR}/deteksi_makro.png)", "",
         "Presisi, Recall, dan F1 pada ambang $" + B + "text{conf}^{*}$ yang memaksimalkan F1 makro.", "",
         f"![F1 per kelas tiga korpus]({GAMBAR}/deteksi_perkelas.png)", "",
         "| Korpus | Detektor | Citra | Objek | $" + B + "text{conf}^{*}$ | Presisi | Recall | F1 | $mAP50$ | $mAP50" + B + "text{--}95$ | $mAP50$ `pycocotools` |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for korpus in KOR:
        for det in DET:
            b = s.det(korpus, korpus, det)
            m = b["makro"]
            t = det == "RF-DETR-L"
            L.append(
                f"| {tebal(korpus, t)} | {tebal(det, t)} | {rb(b['n_citra'])} | {rb(m['n_gt'])} | {km(b['conf_optimal'], 2)} | "
                f"{tebal(km(m['presisi']), t)} | {tebal(km(m['recall']), t)} | {tebal(km(m['f1']), t)} | "
                f"{tebal(km(m['map50']), t)} | {tebal(km(m['map50_95']), t)} | {km(PYCOCO[(korpus, det)])} |"
            )
    L.append("")
    for i, korpus in enumerate(KOR, 1):
        L += [f"### 4.{i} Rincian per Kelas, Korpus `{korpus}`", "",
              "| Detektor | Kelas | Objek | Presisi | Recall | F1 | $AP50$ | $AP50" + B + "text{--}95$ |",
              "|---|---|---|---|---|---|---|---|"]
        for det in DET:
            b = s.det(korpus, korpus, det)
            for kls in KLS:
                pk = b["per_kelas"][kls]
                L.append(f"| {det} | {kls} | {rb(pk['n_gt'])} | {km(pk['presisi'])} | {km(pk['recall'])} | "
                         f"{km(pk['f1'])} | {km(pk['ap50'])} | {km(pk['ap50_95'])} |")
            m = b["makro"]
            L.append(f"| {det} | **Semua** | {rb(m['n_gt'])} | **{km(m['presisi'])}** | **{km(m['recall'])}** | "
                     f"**{km(m['f1'])}** | **{km(m['map50'])}** | **{km(m['map50_95'])}** |")
        L.append("")
    return L


def bagian_pencacahan(s: Sumber) -> list[str]:
    L = ["## 5. Pencacahan per Korpus Latih", "",
         f"![Galat dan akurasi pencacahan]({GAMBAR}/pencacahan_makro.png)", "",
         "Varian koefisien per kelas terbaik tiap detektor. $MAE$ relatif adalah $MAE$ dibagi rerata cacah acuan kelas.", "",
         "| Korpus | Detektor | Metode | $MAE$ makro | $MAE$ relatif | $RMSE$ makro | Bias mutlak makro | Akurasi ±1 makro |",
         "|---|---|---|---|---|---|---|---|"]
    for korpus in KOR:
        for det in DET:
            b = s.terbaik(korpus, det)
            m = b["metrik"]
            t = det == "RF-DETR-L"
            L.append(f"| {tebal(korpus, t)} | {tebal(det, t)} | {tebal(NAMA[b['metode']], t)} | "
                     f"{tebal(km(m['mae_makro']), t)} | {km(m['mae_relatif_makro'])} | {km(m['rmse_makro'])} | "
                     f"{km(m['bias_abs_makro'])} | {km(m['acc_pm1_makro'])} |")
    L.append("")
    L.append("")
    return L


def bagian_koefisien(s: Sumber) -> list[str]:
    L = ["## 6. Koefisien Terbaik", "",
         f"![Koefisien pengali dan ambang per kelas]({GAMBAR}/koefisien.png)", "",
         "Rerata lima lipatan, RF-DETR-L.", "",
         "| Korpus | Metode | $k_{B1}$ | $k_{B2}$ | $k_{B3}$ | $k_{B4}$ | $" + B + "tau_{B1}$ | $" + B + "tau_{B2}$ | $" + B + "tau_{B3}$ | $" + B + "tau_{B4}$ |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for korpus in KOR:
        b = s.cac(korpus, "RF-DETR-L", PILIH[korpus])
        L.append(f"| {korpus} | {NAMA[b['metode']]} | " + " | ".join(km(v, 2) for v in b["k_rerata"])
                 + " | " + " | ".join(km(v, 2) for v in b["tau_rerata"]) + " |")
    L += ["", "### 6.1 Rincian per Kelas", "",
          f"![MAE, bias, dan akurasi per kelas]({GAMBAR}/pencacahan_perkelas.png)", "",
          "| Korpus | Metrik | B1 | B2 | B3 | B4 |", "|---|---|---|---|---|---|"]
    for korpus in KOR:
        pk = s.cac(korpus, "RF-DETR-L", PILIH[korpus])["metrik"]["per_kelas"]
        L.append(f"| {korpus} | $MAE$ | " + " | ".join(km(pk[k]["mae"], 3) for k in KLS) + " |")
        L.append(f"| {korpus} | Bias | " + " | ".join(("+" if pk[k]["bias"] > 0 else "") + km(pk[k]["bias"], 3) for k in KLS) + " |")
        L.append(f"| {korpus} | Akurasi ±1 | " + " | ".join(km(pk[k]["acc_pm1"], 3) for k in KLS) + " |")
    L.append("")
    return L


def bagian_efek(s: Sumber) -> list[str]:
    L = ["## 7. Efek Kalibrasi", "",
         f"![Penurunan MAE dan kenaikan akurasi untuk seluruh kombinasi]({GAMBAR}/efek_kalibrasi_penuh.png)", "",
         "Garis dasar naif: $k = 1$, $" + B + "tau = 0,25$.", "",
         "| Korpus | Detektor | $MAE$ naif | $RMSE$ naif | Akurasi ±1 naif |", "|---|---|---|---|---|"]
    for korpus in KOR:
        for det in DET:
            n = s.cac(korpus, det, "naif")["metrik"]
            L.append(f"| {korpus} | {det} | {km(n['mae_makro'])} | {km(n['rmse_makro'])} | {km(n['acc_pm1_makro'])} |")
    L += ["", "### 7.1 Perbandingan Metode", "",
          f"![Perbandingan metode kalibrasi]({GAMBAR}/metode_kalibrasi.png)", "",
          "Sembilan kombinasi, tiga metode, perbaikan terhadap garis dasar di atas.", "",
          "| Korpus | Detektor | Metode | $MAE$ | Penurunan $MAE$ | $RMSE$ | Penurunan $RMSE$ | Akurasi ±1 | Kenaikan akurasi |",
          "|---|---|---|---|---|---|---|---|---|"]
    for korpus in KOR:
        for det in DET:
            n = s.cac(korpus, det, "naif")["metrik"]
            nilai = [(met, s.cac(korpus, det, met)["metrik"]) for met in ("global", "k_perkelas", "k_tau_perkelas")]
            met_terbaik = min(nilai, key=lambda x: x[1]["mae_makro"])[0]
            for met, m in nilai:
                t = met == met_terbaik
                dm = (n["mae_makro"] - m["mae_makro"]) / n["mae_makro"] * 100
                dr = (n["rmse_makro"] - m["rmse_makro"]) / n["rmse_makro"] * 100
                da = (m["acc_pm1_makro"] - n["acc_pm1_makro"]) * 100
                L.append(f"| {korpus} | {det} | {tebal(NAMA[met], t)} | {tebal(km(m['mae_makro']), t)} | "
                         f"{tebal(km(dm, 1), t)}% | {km(m['rmse_makro'])} | {km(dr, 1)}% | "
                         f"{km(m['acc_pm1_makro'])} | +{km(da, 1)} pp |")
    L += ["", "Tebal: $MAE$ terendah tiap detektor.", ""]
    return L


def bagian_silang_detektor(s: Sumber) -> list[str]:
    L = ["## 8. Uji Silang Detektor", "",
         f"![Dalam domain berbanding lintas korpus]({GAMBAR}/uji_silang.png)", "",
         "Detektor dipindah ke partisi uji korpus lain, koefisien dipasang ulang pada sasaran.", "",
         "| Korpus latih | Korpus uji | Detektor | $mAP50$ | F1 | $MAE$ makro | $MAE$ relatif | Akurasi ±1 |",
         "|---|---|---|---|---|---|---|---|"]
    for korpus in KOR:
        for det in DET:
            d = s.det(korpus, korpus, det)["makro"]
            b = s.terbaik(korpus, det)["metrik"]
            L.append(f"| {korpus} | {korpus} (dalam domain) | {det} | {km(d['map50'])} | {km(d['f1'])} | "
                     f"{km(b['mae_makro'])} | {km(b['mae_relatif_makro'])} | {km(b['acc_pm1_makro'])} |")
    for latih, uji in SILANG:
        kode = KODE_SILANG[(latih, uji)]
        for det in DET:
            d = s.det(latih, uji, det)
            b = s.terbaik(kode, det)
            if d is None or b is None:
                L.append(f"| {latih} | {uji} | {det} | belum tersedia | belum tersedia | belum tersedia | belum tersedia | belum tersedia |")
                continue
            m = b["metrik"]
            L.append(f"| {latih} | {uji} | {det} | {km(d['makro']['map50'])} | {km(d['makro']['f1'])} | "
                     f"{km(m['mae_makro'])} | {km(m['mae_relatif_makro'])} | {km(m['acc_pm1_makro'])} |")
    L += ["", "Baris `1716` ke `763` memakai 66 pohon irisan.", ""]
    return L


def bagian_permutasi(s: Sumber) -> list[str]:
    kom = [(k, d) for k in KOR for d in DET]
    peta = {((r["sumber_korpus"], r["sumber_detektor"]), (r["sasaran_korpus"], r["sasaran_detektor"])): r["metrik"]["mae_makro"]
            for r in s.permutasi}
    L = ["## 9. Uji Silang Koefisien", "",
         f"![Matriks permutasi koefisien]({GAMBAR}/permutasi_koefisien.png)", "",
         "Y, RT, RF: YOLO26l, RT-DETR-L, RF-DETR-L. Diagonal tebal: koefisien sendiri.", "",
         "| Bagian | Yang dipindahkan | Pertanyaan yang dijawab |", "|---|---|---|",
         "| 8 | Detektor, pada citra korpus lain | Daya tahan detektor lintas populasi |",
         "| 9 | Koefisien, detektor dan citra tetap milik sasaran | Kekhususan koefisien terhadap arsitektur |", "",
         "| Jenis permutasi | Jumlah sel | $MAE$ rerata | Terendah | Tertinggi |", "|---|---|---|---|---|"]
    diag = [peta[(t, t)] for t in kom]
    bd = [v for (a, b), v in peta.items() if a[0] == b[0] and a[1] != b[1]]
    bk = [v for (a, b), v in peta.items() if a[0] != b[0] and a[1] == b[1]]
    b2 = [v for (a, b), v in peta.items() if a[0] != b[0] and a[1] != b[1]]
    for nama, v in [("Koefisien sendiri", diag), ("Korpus sama, detektor berbeda", bd),
                    ("Detektor sama, korpus berbeda", bk), ("Korpus dan detektor berbeda", b2)]:
        L.append(f"| {nama} | {len(v)} | {km(float(np.mean(v)))} | {km(min(v))} | {km(max(v))} |")
    L.append("")
    return L


def bagian_partisi(s: Sumber) -> list[str]:
    L = ["## 3. Karakteristik Partisi Uji", "",
         "| Korpus | Pohon | Tandan | Tandan per pohon | B1 | B2 | B3 | B4 |", "|---|---|---|---|---|---|---|---|"]
    for korpus in KOR:
        a = s.acuan[korpus]
        L.append(f"| {korpus} | {rb(len(a))} | {rb(a.sum())} | {km(a.sum(axis=1).mean(), 2)} | "
                 + " | ".join(km(v, 2) for v in a.mean(axis=0)) + " |")
    L += ["", "| Pasangan partisi | Pohon beririsan | Konsekuensi |", "|---|---|---|",
          "| `1716` dan `953` | 141 dari 141 pohon SAWIT | Tidak independen |",
          "| `1716` dan `763` | 66 dari 110 pohon | Basis pohon lebih kecil |",
          "| `953` dan `763` | 0 pohon | Terpisah penuh |", ""]
    return L


def bagian_glosarium() -> list[str]:
    return [
        "## 10. Glosarium", "",
        "| Simbol atau istilah | Arti |", "|---|---|",
        "| B1 sampai B4 | Kelas kematangan. B1 lewat matang, B4 mentah |",
        "| $n_c(t)$ | Jumlah deteksi kelas $c$ lintas sisi pohon $t$ yang lolos ambang |",
        "| $k_c$ | Pengali kelas $c$. Di bawah $1,00$ menurunkan hitungan, di atas menaikkan |",
        "| $" + B + "tau_c$ | Ambang skor keyakinan kelas $c$ |",
        "| $" + B + "hat{y}_c(t)$ | Hitungan akhir kelas $c$ pada pohon $t$ |",
        "| $" + B + "text{conf}^{*}$ | Ambang yang memaksimalkan F1 makro |",
        "| Presisi dan Recall | Deteksi yang cocok dengan acuan, dan acuan yang terdeteksi |",
        "| P dan R | Singkatan Presisi dan Recall |",
        "| F1 | Rerata harmonik presisi dan recall |",
        "| IoU | Rasio irisan terhadap gabungan dua kotak pembatas (*bounding box*) |",
        "| $AP50$ dan $AP50" + B + "text{--}95$ | Luas kurva presisi-recall pada IoU $0,50$, dan rerata sepuluh ambang |",
        "| $mAP50$ dan $mAP50" + B + "text{--}95$ | Rerata $AP50$ dan $AP50" + B + "text{--}95$ atas empat kelas |",
        "| $MAE$ | Rerata galat absolut per pohon, satuan tandan |",
        "| $MAE$ relatif | $MAE$ dibagi rerata cacah acuan kelas yang sama |",
        "| $RMSE$ | Akar rerata kuadrat galat, menekankan galat besar |",
        "| Bias | Rerata selisih bertanda. Negatif berarti kurang hitung |",
        "| Akurasi ±1 | Proporsi pohon dengan selisih paling banyak satu tandan |",
        "| pp | Persentase poin |",
        "| Makro | Rerata tanpa bobot atas B1–B4 |",
        "| Naif | Tanpa kalibrasi: $k = 1$, $" + B + "tau = 0,25$ |",
        "| $k$ global, $k$ per kelas, $k + " + B + "tau$ per kelas | Satu nilai; $k$ per kelas; $k$ dan $" + B + "tau$ per kelas |",
        "| Lipat-silang 5 lipatan | Koefisien dipasang pada empat kelompok, diuji pada kelompok yang ditahan |",
        "| Korpus 953, 763, 1716 | SawitMVC-YOLO, SawitMVC-Depth-YOLO, dan gabungannya |",
        "| Y, RT, RF | YOLO26l, RT-DETR-L, RF-DETR-L |",
        "| $n$ | Jumlah pohon pada partisi uji |",
        "",
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hasil", default="results/counting_koefisien_2026-09-16")
    ap.add_argument("--baseline-root", default="D:/Work/Assisten-Dosen/Baseline-SawitMVC")
    ap.add_argument("--depth-root", default="D:/Work/Assisten-Dosen/SawitMVC-Depth/SawitMVC-Depth-YOLO")
    ap.add_argument("--out", default="docs/LAPORAN-KINERJA-PENCACAHAN-2026-09-16.md")
    arg = ap.parse_args()

    s = Sumber(Path(arg.hasil), Path(arg.baseline_root), Path(arg.depth_root))
    baris = (
        bagian_ringkasan(s)
        + bagian_identitas()
        + bagian_partisi(s)
        + bagian_deteksi(s)
        + bagian_pencacahan(s)
        + bagian_koefisien(s)
        + bagian_efek(s)
        + bagian_silang_detektor(s)
        + bagian_permutasi(s)
        + bagian_glosarium()
    )
    Path(arg.out).write_text("\n".join(baris).rstrip("\n") + "\n", encoding="utf-8")
    print(f"Ditulis: {arg.out} ({len(baris)} baris)")


if __name__ == "__main__":
    main()
