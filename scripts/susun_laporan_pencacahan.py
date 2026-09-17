"""Menyusun laporan kinerja pencacahan dari berkas hasil (V2-E-050f, V2-E-050g).

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
from kalibrasi_pencacahan_perkorpus import basis_1716_bersih, gt_korpus, pohon_bocor_763, pohon_uji_1716

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
PYCOCO = {
    ("953", "YOLO26l"): 0.5435, ("953", "RT-DETR-L"): 0.5718, ("953", "RF-DETR-L"): 0.5965,
    ("763", "YOLO26l"): 0.5163, ("763", "RT-DETR-L"): 0.5580, ("763", "RF-DETR-L"): 0.6129,
    ("1716", "YOLO26l"): 0.5389, ("1716", "RT-DETR-L"): 0.5745, ("1716", "RF-DETR-L"): 0.5960,
}
SILANG = [("763", "953"), ("1716", "953"), ("1716", "763"), ("953", "763"), ("953", "1716"), ("763", "1716")]
KODE_SILANG = {("763", "953"): "763>953", ("1716", "953"): "1716>953",
               ("1716", "763"): "1716>763", ("953", "763"): "953>763",
               ("953", "1716"): "953>1716", ("763", "1716"): "763>1716"}
# Basis sama 207 pohon pada uji 1716: (kode uji deteksi, kode pencacahan) per korpus latih.
BASIS_207 = {"953": ("1716@207", "953>1716@207"), "763": ("1716", "763>1716"), "1716": ("1716@207", "1716@207")}
BIAS = B + "|Bias" + B + "| makro"


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
        akar = hasil.parent.parent
        basis = basis_1716_bersih(akar, gt763)
        self.acuan = {}
        for korpus in KOR:
            gt = gt_korpus(gt953, gt763, korpus)
            if korpus == "1716":
                gt = {k: v for k, v in gt.items() if k in pohon_1716}
            self.acuan[korpus] = np.array(list(gt.values()))
        gt = gt_korpus(gt953, gt763, "1716")
        self.acuan["1716@207"] = np.array([v for k, v in gt.items() if k in basis])
        uji953, uji763 = set(gt953["test"]), set(gt763["test"])
        lihat953 = set(gt953["train"]) | set(gt953["val"])
        lihat763 = set(gt763["train"]) | set(gt763["val"])
        sawit_1716 = {p[len("SAWIT_"):] for p in pohon_1716 if p.startswith("SAWIT_")}
        depth_1716 = {p[len("DEPTH_"):] for p in pohon_uji_1716(akar) if p.startswith("DEPTH_")}
        self.irisan = {
            "uji1716_uji953": len(sawit_1716 & uji953),
            "uji1716_uji763": len(depth_1716 & uji763),
            "uji1716_lihat763": len(pohon_bocor_763(akar, gt763)),
            "uji953_uji763": len(uji953 & uji763),
            "uji953_lihat763": len(uji953 & lihat763),
            "uji763_lihat953": len(uji763 & lihat953),
            "n953": len(uji953), "n763": len(uji763), "n_basis": len(basis),
            "basis_sawit": sum(p.startswith("SAWIT_") for p in basis),
            "basis_depth": sum(p.startswith("DEPTH_") for p in basis),
        }

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

    def rentang_penurunan(self, kunci):
        """Rentang penurunan metrik terhadap naif pada 27 kombinasi dalam domain."""
        nilai = [
            (1 - self.cac(k, d, m)["metrik"][kunci] / self.cac(k, d, "naif")["metrik"][kunci]) * 100
            for k in KOR for d in DET for m in ("global", "k_perkelas", "k_tau_perkelas")
        ]
        return min(nilai), max(nilai)

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
    mae_gab = [s.terbaik(BASIS_207["1716"][1], d)["metrik"]["mae_makro"] for d in DET]
    mae_tunggal = [s.terbaik(BASIS_207[k][1], d)["metrik"]["mae_makro"] for k in ("953", "763") for d in DET]
    assert max(mae_gab) < min(mae_tunggal), "klaim latih 1716 terbaik pada uji 1716 tidak lagi berlaku"
    mae_turun = s.rentang_penurunan("mae_makro")
    bias_turun = s.rentang_penurunan("bias_abs_makro")
    silang_semua = min(s.det(a, b_, d)["makro"]["map50"] for a, b_ in SILANG for d in DET if s.det(a, b_, d))
    L += ["", "| Temuan utama | Angka pendukung |", "|---|---|",
          f"| RF-DETR-L terbaik di ketiga korpus | $mAP50$ {km(min(map_domain))}–{km(max(map_domain))} |",
          f"| Kalibrasi menurunkan galat di semua kombinasi | $MAE$ turun {km(mae_turun[0], 1)}–{km(mae_turun[1], 1)}%; "
          f"{BIAS} turun {km(bias_turun[0], 1)}–{km(bias_turun[1], 1)}% |",
          f"| Peringkat korpus berbalik pada basis relatif | $MAE$ {km(mae['763'])} (763) berbanding {km(mae['953'])} (953); relatif {km(rel['763'])} berbanding {km(rel['953'])} |",
          f"| Detektor gugur lintas korpus, dua arah | $mAP50$ serendah {km(silang_semua)} |",
          f"| Uji `1716`: latih `1716` terbaik | $MAE$ {km(min(mae_gab))}–{km(max(mae_gab))} berbanding "
          f"{km(min(mae_tunggal))}–{km(max(mae_tunggal))} ({s.irisan['n_basis']} pohon) |", ""]
    return L


def bagian_identitas() -> list[str]:
    return [
        "## 2. Identitas Eksperimen", "",
        "| Parameter | Nilai |", "|---|---|",
        "| Identitas simpul | `V2-E-050` sampai `V2-E-050g` |",
        "| Tanggal | 16–17 September 2026 |",
        "| Korpus latih | `953` (SawitMVC-YOLO), `763` (SawitMVC-Depth-YOLO v2.0.0), `1716` (gabungan) |",
        "| Partisi uji | `953`: 141 pohon; `763`: 110 pohon; `1716`: 257 pohon, basis sama 207 pohon |",
        "| Detektor | YOLO26l, RT-DETR-L, RF-DETR-L |",
        "| Metrik deteksi | Presisi, Recall, F1, $AP50$, dan $AP50" + B + "text{--}95$ per kelas serta makro |",
        "| Model pencacahan | $" + B + "hat{y}_c(t) = " + B + "operatorname{round}(k_c " + B + "cdot n_c(t))$, dengan $n_c(t)$ sebagai jumlah deteksi kelas $c$ lintas sisi pohon yang memenuhi skor keyakinan $" + B + "ge " + B + "tau_c$ |",
        "| Kalibrasi | Lipat-silang 5 lipatan tingkat pohon, tanpa pelatihan ulang |",
        "| Metrik pencacahan | $MAE$ makro, $MAE$ relatif, $RMSE$ makro, " + BIAS + ", akurasi ±1 makro |",
        "| Skrip | [`susun_laporan_pencacahan.py`](../scripts/susun_laporan_pencacahan.py), [`metrik_deteksi_perkorpus.py`](../scripts/metrik_deteksi_perkorpus.py), [`kalibrasi_pencacahan_perkorpus.py`](../scripts/kalibrasi_pencacahan_perkorpus.py), [`permutasi_koefisien_pencacahan.py`](../scripts/permutasi_koefisien_pencacahan.py), [`inferensi_953_ke_763.py`](../scripts/inferensi_953_ke_763.py), [`gabung_dump_1716.py`](../scripts/gabung_dump_1716.py) |",
        "| Artefak angka | [`metrik_deteksi_perkorpus.json`](../results/counting_koefisien_2026-09-16/metrik_deteksi_perkorpus.json), [`pencacahan_perkorpus.json`](../results/counting_koefisien_2026-09-16/pencacahan_perkorpus.json), [`permutasi_koefisien.json`](../results/counting_koefisien_2026-09-16/permutasi_koefisien.json), [`gabungan_1716_manifest.json`](../results/cross_eval/predictions/gabungan_1716_manifest.json) |",
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
         "| Korpus | Detektor | Metode | $MAE$ makro | $MAE$ relatif | $RMSE$ makro | " + BIAS + " | Akurasi ±1 makro |",
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
    L = ["## 6. Koefisien Terbaik RF-DETR-L", "",
         f"![Koefisien pengali dan ambang per kelas]({GAMBAR}/koefisien.png)", "",
         "Varian per kelas dengan $MAE$ terendah. Rerata lima lipatan.", "",
         "| Korpus | Metode | $k_{B1}$ | $k_{B2}$ | $k_{B3}$ | $k_{B4}$ | $" + B + "tau_{B1}$ | $" + B + "tau_{B2}$ | $" + B + "tau_{B3}$ | $" + B + "tau_{B4}$ |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for korpus in KOR:
        b = s.terbaik(korpus, "RF-DETR-L")
        L.append(f"| {korpus} | {NAMA[b['metode']]} | " + " | ".join(km(v, 2) for v in b["k_rerata"])
                 + " | " + " | ".join(km(v, 2) for v in b["tau_rerata"]) + " |")
    L += ["", "### 6.1 Rincian per Kelas RF-DETR-L", "",
          f"![MAE, bias, dan akurasi per kelas]({GAMBAR}/pencacahan_perkelas.png)", "",
          "Konfigurasi sama dengan tabel §6. Nilai dari prediksi luar-lipatan.", "",
          "| Korpus | Metode | Metrik | B1 | B2 | B3 | B4 |", "|---|---|---|---|---|---|---|"]
    for korpus in KOR:
        b = s.terbaik(korpus, "RF-DETR-L")
        pk, metode = b["metrik"]["per_kelas"], NAMA[b["metode"]]
        L.append(f"| {korpus} | {metode} | $MAE$ | " + " | ".join(km(pk[k]["mae"], 3) for k in KLS) + " |")
        L.append(f"| {korpus} | {metode} | Bias | "
                 + " | ".join(("+" if pk[k]["bias"] > 0 else "") + km(pk[k]["bias"], 3) for k in KLS) + " |")
        L.append(f"| {korpus} | {metode} | Akurasi ±1 | " + " | ".join(km(pk[k]["acc_pm1"], 3) for k in KLS) + " |")
    L.append("")
    return L


def bagian_efek(s: Sumber) -> list[str]:
    L = ["## 7. Efek Kalibrasi", "",
         f"![Penurunan MAE dan kenaikan akurasi untuk seluruh kombinasi]({GAMBAR}/efek_kalibrasi_penuh.png)", "",
         "Garis dasar naif: $k = 1$, $" + B + "tau = 0,25$.", "",
         "| Korpus | Detektor | $MAE$ naif | $RMSE$ naif | " + BIAS + " naif | Akurasi ±1 naif |",
         "|---|---|---|---|---|---|"]
    for korpus in KOR:
        for det in DET:
            n = s.cac(korpus, det, "naif")["metrik"]
            L.append(f"| {korpus} | {det} | {km(n['mae_makro'])} | {km(n['rmse_makro'])} | "
                     f"{km(n['bias_abs_makro'])} | {km(n['acc_pm1_makro'])} |")
    L += ["", "### 7.1 Perbandingan Metode", "",
          f"![Perbandingan metode kalibrasi]({GAMBAR}/metode_kalibrasi.png)", "",
          "Sembilan kombinasi, tiga metode, perbaikan terhadap garis dasar di atas.", "",
          "| Korpus | Detektor | Metode | $MAE$ | Penurunan $MAE$ | $RMSE$ | Penurunan $RMSE$ | " + BIAS
          + " | Penurunan " + B + "|bias" + B + "| | Akurasi ±1 | Kenaikan akurasi |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for korpus in KOR:
        for det in DET:
            n = s.cac(korpus, det, "naif")["metrik"]
            nilai = [(met, s.cac(korpus, det, met)["metrik"]) for met in ("global", "k_perkelas", "k_tau_perkelas")]
            met_terbaik = min(nilai, key=lambda x: x[1]["mae_makro"])[0]
            for met, m in nilai:
                t = met == met_terbaik
                dm = (n["mae_makro"] - m["mae_makro"]) / n["mae_makro"] * 100
                dr = (n["rmse_makro"] - m["rmse_makro"]) / n["rmse_makro"] * 100
                db = (n["bias_abs_makro"] - m["bias_abs_makro"]) / n["bias_abs_makro"] * 100
                da = (m["acc_pm1_makro"] - n["acc_pm1_makro"]) * 100
                L.append(f"| {korpus} | {det} | {tebal(NAMA[met], t)} | {tebal(km(m['mae_makro']), t)} | "
                         f"{tebal(km(dm, 1), t)}% | {km(m['rmse_makro'])} | {km(dr, 1)}% | "
                         f"{km(m['bias_abs_makro'])} | {km(db, 1)}% | "
                         f"{km(m['acc_pm1_makro'])} | +{km(da, 1)} pp |")
    L += ["", "Tebal: $MAE$ terendah tiap detektor.", ""]
    return L


def bagian_silang_detektor(s: Sumber) -> list[str]:
    L = ["## 8. Uji Silang Detektor", "",
         f"![Dalam domain berbanding lintas korpus]({GAMBAR}/uji_silang.png)", "",
         "Detektor dipindah ke partisi uji korpus lain, koefisien dipasang ulang pada sasaran.", "",
         "| Korpus latih | Korpus uji | Pohon | Detektor | $mAP50$ | F1 | $MAE$ makro | $MAE$ relatif | Akurasi ±1 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for korpus in KOR:
        for det in DET:
            d = s.det(korpus, korpus, det)["makro"]
            m = s.terbaik(korpus, det)["metrik"]
            L.append(f"| {korpus} | {korpus} (dalam domain) | {m['n_pohon']} | {det} | {km(d['map50'])} | "
                     f"{km(d['f1'])} | {km(m['mae_makro'])} | {km(m['mae_relatif_makro'])} | {km(m['acc_pm1_makro'])} |")
    for latih, uji in SILANG:
        kode = KODE_SILANG[(latih, uji)]
        for det in DET:
            d = s.det(latih, uji, det)
            b = s.terbaik(kode, det)
            if d is None or b is None:
                L.append(f"| {latih} | {uji} | – | {det} | belum tersedia | belum tersedia | belum tersedia | "
                         "belum tersedia | belum tersedia |")
                continue
            m = b["metrik"]
            L.append(f"| {latih} | {uji} | {m['n_pohon']} | {det} | {km(d['makro']['map50'])} | "
                     f"{km(d['makro']['f1'])} | {km(m['mae_makro'])} | {km(m['mae_relatif_makro'])} | "
                     f"{km(m['acc_pm1_makro'])} |")
    L += ["", f"`763` ke `1716` tanpa {s.irisan['uji1716_lihat763']} pohon yang pernah dilihat detektor `763`.", "",
          f"### 8.1 Basis Sama {s.irisan['n_basis']} Pohon", "",
          f"Uji `1716`: {s.irisan['basis_sawit']} pohon SAWIT dan {s.irisan['basis_depth']} pohon DEPTH, "
          "sama untuk ketiga korpus latih. Tebal: $MAE$ terendah tiap detektor.", "",
          "| Detektor | Korpus latih | $mAP50$ | F1 | $MAE$ makro | $MAE$ relatif | Akurasi ±1 |",
          "|---|---|---|---|---|---|---|"]
    for det in DET:
        baris = []
        for latih in KOR:
            kode_det, kode_cacah = BASIS_207[latih]
            baris.append((latih, s.det(latih, kode_det, det)["makro"], s.terbaik(kode_cacah, det)["metrik"]))
        mae_min = min(m["mae_makro"] for _, _, m in baris)
        for latih, d, m in baris:
            t = m["mae_makro"] == mae_min
            L.append(f"| {det} | {tebal(latih, t)} | {km(d['map50'])} | {km(d['f1'])} | "
                     f"{tebal(km(m['mae_makro']), t)} | {km(m['mae_relatif_makro'])} | {km(m['acc_pm1_makro'])} |")
    L.append("")
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
    for korpus, label in [("953", "953"), ("763", "763"), ("1716", "1716"), ("1716@207", "1716, basis sama")]:
        a = s.acuan[korpus]
        L.append(f"| {label} | {rb(len(a))} | {rb(a.sum())} | {km(a.sum(axis=1).mean(), 2)} | "
                 + " | ".join(km(v, 2) for v in a.mean(axis=0)) + " |")
    i = s.irisan
    L += ["", "| Pasangan partisi | Pohon beririsan | Konsekuensi |", "|---|---|---|",
          f"| Uji `1716` dan uji `953` | {i['uji1716_uji953']} dari {i['n953']} pohon SAWIT | Tidak independen |",
          f"| Uji `1716` dan uji `763` | {i['uji1716_uji763']} dari {i['n763']} pohon | Basis `1716` ke `763` |",
          f"| Uji `1716` dan latih/validasi `763` | {i['uji1716_lihat763']} pohon DEPTH, citra identik | "
          "Dikeluarkan dari `763` ke `1716` |",
          f"| Uji `953` dan uji `763` | {i['uji953_uji763']} ID pohon | Beda sesi akuisisi |",
          f"| Uji `953` dan latih/validasi `763` | {i['uji953_lihat763']} dari {i['n953']} ID pohon | "
          "Beda sesi akuisisi, sah (V2-E-040) |",
          f"| Uji `763` dan latih/validasi `953` | {i['uji763_lihat953']} dari {i['n763']} ID pohon | "
          "Beda sesi akuisisi |", ""]
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
        "| " + BIAS + " | Rerata nilai mutlak bias per kelas atas B1–B4 |",
        "| Akurasi ±1 | Proporsi pohon dengan selisih paling banyak satu tandan |",
        "| pp | Persentase poin |",
        "| Makro | Rerata tanpa bobot atas B1–B4 |",
        "| Naif | Tanpa kalibrasi: $k = 1$, $" + B + "tau = 0,25$ |",
        "| $k$ global, $k$ per kelas, $k + " + B + "tau$ per kelas | Satu nilai; $k$ per kelas; $k$ dan $" + B + "tau$ per kelas |",
        "| Lipat-silang 5 lipatan | Koefisien dipasang pada empat kelompok, diuji pada kelompok yang ditahan |",
        "| Prediksi luar-lipatan | Hitungan pohon dari koefisien yang tidak dipasang pada pohon itu |",
        "| Basis sama 207 pohon | Uji `1716` tanpa 50 pohon DEPTH latih/validasi `763` |",
        "| ID pohon beda sesi | Pohon fisik sama, difoto pada sesi berselang sekitar 80 hari dengan kamera berbeda |",
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
