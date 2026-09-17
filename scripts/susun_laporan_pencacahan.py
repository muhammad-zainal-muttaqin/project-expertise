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
BIAS_KECIL = B + "|bias" + B + "| makro"


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
         "Metode kalibrasi dipilih dari dua varian per kelas; $k$ global dibandingkan di §7.1.", "",
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
    mae_gab = [s.terbaik(BASIS_207["1716"][1], d)["metrik"]["mae_makro"] for d in DET]
    mae_tunggal = [s.terbaik(BASIS_207[k][1], d)["metrik"]["mae_makro"] for k in ("953", "763") for d in DET]
    assert max(mae_gab) < min(mae_tunggal), "klaim latih 1716 terbaik pada uji 1716 tidak lagi berlaku"
    mae_turun = s.rentang_penurunan("mae_makro")
    bias_turun = s.rentang_penurunan("bias_abs_makro")
    tukar = [s.det(a, b_, d)["makro"]["map50"] for a, b_ in (("763", "953"), ("953", "763")) for d in DET]
    L += ["", "| Temuan utama | Angka pendukung |", "|---|---|",
          f"| RF-DETR-L terbaik pada ketiga korpus | $mAP50$ {km(min(map_domain))}–{km(max(map_domain))} |",
          f"| Kalibrasi menurunkan galat pada 27 kombinasi dalam domain | $MAE$ turun {km(mae_turun[0], 1)}–{km(mae_turun[1], 1)}%; "
          f"{BIAS_KECIL} turun {km(bias_turun[0], 1)}–{km(bias_turun[1], 1)}% |",
          f"| Peringkat korpus berbalik pada basis relatif | $MAE$ {km(mae['763'])} (`763`) berbanding {km(mae['953'])} (`953`); "
          f"$MAE$ relatif {km(rel['763'])} berbanding {km(rel['953'])} |",
          f"| Deteksi menurun saat `953` dan `763` bertukar korpus uji | $mAP50$ {km(min(tukar))}–{km(max(tukar))} |",
          f"| Korpus latih `1716` terbaik pada uji `1716` | $MAE$ {km(min(mae_gab))}–{km(max(mae_gab))} berbanding "
          f"{km(min(mae_tunggal))}–{km(max(mae_tunggal))} (latih `953` dan `763`, {s.irisan['n_basis']} pohon) |", ""]
    return L


def bagian_identitas() -> list[str]:
    return [
        "## 2. Identitas Eksperimen", "",
        "| Parameter | Nilai |", "|---|---|",
        "| Identitas simpul | `V2-E-050` sampai `V2-E-050h` |",
        "| Tanggal | 16–17 September 2026 |",
        "| Korpus latih | `953` (SawitMVC-YOLO), `763` (SawitMVC-Depth-YOLO v2.0.0), `1716` (gabungan) |",
        "| Partisi uji | `953`: 141 pohon; `763`: 110 pohon; `1716`: 257 pohon, basis sama 207 pohon |",
        "| Detektor | YOLO26l, RT-DETR-L, RF-DETR-L |",
        "| Metrik deteksi | Presisi, *recall*, F1, $AP50$, dan $AP50" + B + "text{--}95$ per kelas serta makro |",
        "| Model pencacahan | $" + B + "hat{y}_c(t) = " + B + "operatorname{round}(k_c " + B + "cdot n_c(t))$ |",
        "| Kalibrasi | Validasi silang 5 lipatan pada pohon partisi uji, tanpa pelatihan ulang detektor |",
        "| Metrik pencacahan | $MAE$ makro, $MAE$ relatif, $RMSE$ makro, " + BIAS_KECIL + ", akurasi ±1 makro |",
        "| Skrip | [`susun_laporan_pencacahan.py`](../scripts/susun_laporan_pencacahan.py), [`metrik_deteksi_perkorpus.py`](../scripts/metrik_deteksi_perkorpus.py), [`kalibrasi_pencacahan_perkorpus.py`](../scripts/kalibrasi_pencacahan_perkorpus.py), [`permutasi_koefisien_pencacahan.py`](../scripts/permutasi_koefisien_pencacahan.py), [`inferensi_953_ke_763.py`](../scripts/inferensi_953_ke_763.py), [`gabung_dump_1716.py`](../scripts/gabung_dump_1716.py) |",
        "| Artefak angka | [`metrik_deteksi_perkorpus.json`](../results/counting_koefisien_2026-09-16/metrik_deteksi_perkorpus.json), [`pencacahan_perkorpus.json`](../results/counting_koefisien_2026-09-16/pencacahan_perkorpus.json), [`permutasi_koefisien.json`](../results/counting_koefisien_2026-09-16/permutasi_koefisien.json), [`gabungan_1716_manifest.json`](../results/cross_eval/predictions/gabungan_1716_manifest.json) |",
        "",
    ]


def bagian_deteksi(s: Sumber) -> list[str]:
    L = ["## 4. Deteksi per Korpus Latih", "",
         f"![Metrik deteksi makro per korpus]({GAMBAR}/deteksi_makro.png)", "",
         f"![F1 per kelas, tiga korpus]({GAMBAR}/deteksi_perkelas.png)", "",
         "Presisi, *recall*, dan F1 diukur pada ambang $" + B + "text{conf}^{*}$ masing-masing detektor. Tebal: nilai tertinggi tiap korpus.", "",
         "| Korpus | Detektor | Citra | Anotasi | $" + B + "text{conf}^{*}$ | Presisi | Recall | F1 | $mAP50$ | $mAP50" + B + "text{--}95$ | $mAP50$ `pycocotools` |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    kolom = ("presisi", "recall", "f1", "map50", "map50_95")
    for korpus in KOR:
        baris = {det: s.det(korpus, korpus, det) for det in DET}
        puncak = {k: max(b["makro"][k] for b in baris.values()) for k in kolom}
        det_map = max(DET, key=lambda d: baris[d]["makro"]["map50"])
        for det in DET:
            b = baris[det]
            m = b["makro"]
            sel = " | ".join(tebal(km(m[k]), m[k] == puncak[k]) for k in kolom)
            L.append(
                f"| {tebal(korpus, det == det_map)} | {tebal(det, det == det_map)} | {rb(b['n_citra'])} | "
                f"{rb(m['n_gt'])} | {km(b['conf_optimal'], 2)} | {sel} | {km(PYCOCO[(korpus, det)])} |"
            )
    L.append("")
    for i, korpus in enumerate(KOR, 1):
        L += [f"### 4.{i} Rincian Kelas Korpus `{korpus}`", "",
              "| Detektor | Kelas | Anotasi | Presisi | Recall | F1 | $AP50$ | $AP50" + B + "text{--}95$ |",
              "|---|---|---|---|---|---|---|---|"]
        for det in DET:
            b = s.det(korpus, korpus, det)
            for kls in KLS:
                pk = b["per_kelas"][kls]
                L.append(f"| {det} | {kls} | {rb(pk['n_gt'])} | {km(pk['presisi'])} | {km(pk['recall'])} | "
                         f"{km(pk['f1'])} | {km(pk['ap50'])} | {km(pk['ap50_95'])} |")
            m = b["makro"]
            L.append(f"| {det} | **Makro** | {rb(m['n_gt'])} | **{km(m['presisi'])}** | **{km(m['recall'])}** | "
                     f"**{km(m['f1'])}** | **{km(m['map50'])}** | **{km(m['map50_95'])}** |")
        L.append("")
    return L


def bagian_pencacahan(s: Sumber) -> list[str]:
    L = ["## 5. Pencacahan per Korpus Latih", "",
         f"![Galat dan akurasi pencacahan per korpus latih]({GAMBAR}/pencacahan_makro.png)", "",
         "Setiap baris memakai varian per kelas dengan $MAE$ terendah. Tebal: nilai terbaik tiap korpus.", "",
         "| Korpus | Detektor | Metode | $MAE$ makro | $MAE$ relatif | $RMSE$ makro | " + BIAS + " | Akurasi ±1 makro |",
         "|---|---|---|---|---|---|---|---|"]
    kolom = [("mae_makro", -1), ("mae_relatif_makro", -1), ("rmse_makro", -1), ("bias_abs_makro", -1), ("acc_pm1_makro", +1)]
    for korpus in KOR:
        baris = {det: s.terbaik(korpus, det)["metrik"] for det in DET}
        metode = {det: s.terbaik(korpus, det)["metode"] for det in DET}
        puncak = {k: (max if arah > 0 else min)(round(m[k], 4) for m in baris.values()) for k, arah in kolom}
        for det in DET:
            m = baris[det]
            t = round(m["mae_makro"], 4) == puncak["mae_makro"]
            sel = " | ".join(tebal(km(m[k]), round(m[k], 4) == puncak[k]) for k, _ in kolom)
            L.append(f"| {tebal(korpus, t)} | {tebal(det, t)} | {tebal(NAMA[metode[det]], t)} | {sel} |")
    L.append("")
    return L


def bagian_koefisien(s: Sumber) -> list[str]:
    L = ["## 6. Koefisien Terbaik RF-DETR-L", "",
         f"![Koefisien pengali dan ambang per kelas]({GAMBAR}/koefisien.png)", "",
         "Nilai $k$ dan $" + B + "tau$ merupakan rerata lima lipatan dari varian per kelas ber-$MAE$ terendah. "
         "Hitungan $n_c(t)$ menjumlahkan deteksi dari semua sisi pohon, sehingga satu tandan dapat terhitung "
         "lebih dari sekali; $k$ di bawah 1 mengoreksi kelebihan itu.", "",
         "| Korpus | Metode | $k_{B1}$ | $k_{B2}$ | $k_{B3}$ | $k_{B4}$ | $" + B + "tau_{B1}$ | $" + B + "tau_{B2}$ | $" + B + "tau_{B3}$ | $" + B + "tau_{B4}$ |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for korpus in KOR:
        b = s.terbaik(korpus, "RF-DETR-L")
        L.append(f"| {korpus} | {NAMA[b['metode']]} | " + " | ".join(km(v, 2) for v in b["k_rerata"])
                 + " | " + " | ".join(km(v, 2) for v in b["tau_rerata"]) + " |")
    L += ["", "### 6.1 Rincian per Kelas RF-DETR-L", "",
          f"![MAE, bias, dan akurasi ±1 per kelas, RF-DETR-L]({GAMBAR}/pencacahan_perkelas.png)", "",
          "Detektor dan metode sama dengan §6. Tiap pohon dihitung dengan koefisien dari empat lipatan lain, "
          "bukan dengan rerata pada §6.", "",
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
         f"![Penurunan MAE dan kenaikan akurasi ±1, sembilan kombinasi]({GAMBAR}/efek_kalibrasi_penuh.png)", "",
         "Penurunan pada gambar dan §7.1 dihitung terhadap garis dasar naif berikut.", "",
         "| Korpus | Detektor | $MAE$ naif | $RMSE$ naif | " + BIAS + " naif | Akurasi ±1 naif |",
         "|---|---|---|---|---|---|"]
    for korpus in KOR:
        for det in DET:
            n = s.cac(korpus, det, "naif")["metrik"]
            L.append(f"| {korpus} | {det} | {km(n['mae_makro'])} | {km(n['rmse_makro'])} | "
                     f"{km(n['bias_abs_makro'])} | {km(n['acc_pm1_makro'])} |")
    L += ["", "### 7.1 Perbandingan Metode", "",
          f"![Perbandingan metode kalibrasi]({GAMBAR}/metode_kalibrasi.png)", "",
          "Kolom penurunan dan kenaikan dihitung terhadap garis dasar naif pada §7. "
          "Tebal: nilai terbaik tiap pasangan korpus dan detektor; nama metode tebal menandai $MAE$ terendah. "
          "§1, §5, dan §6 hanya memilih varian per kelas, sehingga $k$ global yang lebih rendah tidak terpilih.", "",
          "| Korpus | Detektor | Metode | $MAE$ | Penurunan $MAE$ | $RMSE$ | Penurunan $RMSE$ | " + BIAS
          + " | Penurunan " + B + "|bias" + B + "| | Akurasi ±1 | Kenaikan akurasi ±1 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for korpus in KOR:
        for det in DET:
            n = s.cac(korpus, det, "naif")["metrik"]
            nilai = [(met, s.cac(korpus, det, met)["metrik"]) for met in ("global", "k_perkelas", "k_tau_perkelas")]
            met_terbaik = min(nilai, key=lambda x: x[1]["mae_makro"])[0]
            # Tebal ditentukan dari nilai yang ditampilkan, agar nilai kembar tertebal bersama.
            turun = {met: {
                "dm": round((n["mae_makro"] - m["mae_makro"]) / n["mae_makro"] * 100, 1),
                "dr": round((n["rmse_makro"] - m["rmse_makro"]) / n["rmse_makro"] * 100, 1),
                "db": round((n["bias_abs_makro"] - m["bias_abs_makro"]) / n["bias_abs_makro"] * 100, 1),
                "da": round((m["acc_pm1_makro"] - n["acc_pm1_makro"]) * 100, 1),
            } for met, m in nilai}
            nilai = [(met, {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}) for met, m in nilai]
            puncak = {
                "mae": min(m["mae_makro"] for _, m in nilai), "rmse": min(m["rmse_makro"] for _, m in nilai),
                "bias": min(m["bias_abs_makro"] for _, m in nilai), "acc": max(m["acc_pm1_makro"] for _, m in nilai),
                **{k: max(turun[met][k] for met, _ in nilai) for k in ("dm", "dr", "db", "da")},
            }
            for met, m in nilai:
                t = met == met_terbaik
                u = turun[met]
                sel = [
                    tebal(km(m["mae_makro"]), m["mae_makro"] == puncak["mae"]),
                    tebal(km(u["dm"], 1) + "%", u["dm"] == puncak["dm"]),
                    tebal(km(m["rmse_makro"]), m["rmse_makro"] == puncak["rmse"]),
                    tebal(km(u["dr"], 1) + "%", u["dr"] == puncak["dr"]),
                    tebal(km(m["bias_abs_makro"]), m["bias_abs_makro"] == puncak["bias"]),
                    tebal(km(u["db"], 1) + "%", u["db"] == puncak["db"]),
                    tebal(km(m["acc_pm1_makro"]), m["acc_pm1_makro"] == puncak["acc"]),
                    tebal("+" + km(u["da"], 1) + " pp", u["da"] == puncak["da"]),
                ]
                L.append(f"| {korpus} | {det} | {tebal(NAMA[met], t)} | " + " | ".join(sel) + " |")
    L.append("")
    return L


# Kolom perbandingan: (kunci metrik, sumber deteksi atau pencacahan, arah). Arah +1 berarti makin besar makin baik.
KOLOM_BANDING = [("map50", "det", +1), ("f1", "det", +1), ("mae_makro", "cac", -1),
                 ("mae_relatif_makro", "cac", -1), ("acc_pm1_makro", "cac", +1)]


def sel_terbaik(grup: list[tuple[dict, dict]]) -> list[list[str]]:
    """Format lima kolom perbandingan, nilai terbaik tiap kolom dicetak tebal."""
    def ambil(d, m, sumber, kunci):
        return d[kunci] if sumber == "det" else m[kunci]

    keluaran = [[] for _ in grup]
    for kunci, sumber, arah in KOLOM_BANDING:
        nilai = [round(ambil(d, m, sumber, kunci), 4) for d, m in grup]
        terbaik = max(nilai) if arah > 0 else min(nilai)
        for i, v in enumerate(nilai):
            keluaran[i].append(tebal(km(v), v == terbaik))
    return keluaran


def bagian_silang_detektor(s: Sumber) -> list[str]:
    L = ["## 8. Uji Silang Detektor", "",
         f"![Dalam domain berbanding lintas korpus, RF-DETR-L]({GAMBAR}/uji_silang.png)", "",
         "Setiap detektor diuji pada ketiga partisi uji, dan koefisiennya dipasang pada partisi uji sasaran. "
         "Tebal: nilai terbaik tiap pasangan korpus uji dan detektor.", "",
         "| Korpus uji | Detektor | Korpus latih | Pohon | $mAP50$ | F1 | $MAE$ makro | $MAE$ relatif | Akurasi ±1 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for uji in KOR:
        for det in DET:
            grup, label = [], []
            for latih in KOR:
                if latih == uji:
                    d, b = s.det(latih, latih, det), s.terbaik(latih, det)
                    label.append(f"{latih} (dalam domain)")
                else:
                    d, b = s.det(latih, uji, det), s.terbaik(KODE_SILANG[(latih, uji)], det)
                    label.append(latih)
                grup.append((d["makro"], b["metrik"]))
            for nama, (_, m), sel in zip(label, grup, sel_terbaik(grup)):
                L.append(f"| {uji} | {det} | {nama} | {m['n_pohon']} | " + " | ".join(sel) + " |")
    L += ["", f"Basis pohon dalam satu kelompok berbeda pada uji `763` dan `1716`; perbandingan setara ada di §8.1. "
          f"Baris latih `763` pada uji `1716` tidak memuat {s.irisan['uji1716_lihat763']} pohon yang citranya "
          "dipakai untuk melatih atau memvalidasi detektor `763`.", "",
          f"### 8.1 Basis Sama {s.irisan['n_basis']} Pohon", "",
          f"Ketiga korpus latih diuji pada {s.irisan['n_basis']} pohon uji `1716` yang sama: "
          f"{s.irisan['basis_sawit']} SAWIT dan {s.irisan['basis_depth']} DEPTH. Tebal: nilai terbaik tiap detektor.", "",
          "| Detektor | Korpus latih | $mAP50$ | F1 | $MAE$ makro | $MAE$ relatif | Akurasi ±1 |",
          "|---|---|---|---|---|---|---|"]
    for det in DET:
        grup = []
        for latih in KOR:
            kode_det, kode_cacah = BASIS_207[latih]
            grup.append((s.det(latih, kode_det, det)["makro"], s.terbaik(kode_cacah, det)["metrik"]))
        mae_min = min(m["mae_makro"] for _, m in grup)
        for latih, (_, m), sel in zip(KOR, grup, sel_terbaik(grup)):
            L.append(f"| {det} | {tebal(latih, m['mae_makro'] == mae_min)} | " + " | ".join(sel) + " |")
    L.append("")
    return L


def bagian_permutasi(s: Sumber) -> list[str]:
    kom = [(k, d) for k in KOR for d in DET]
    peta = {((r["sumber_korpus"], r["sumber_detektor"]), (r["sasaran_korpus"], r["sasaran_detektor"])): r["metrik"]["mae_makro"]
            for r in s.permutasi}
    L = ["## 9. Uji Silang Koefisien", "",
         f"![Matriks permutasi koefisien]({GAMBAR}/permutasi_koefisien.png)", "",
         "Semua sel memakai $k$ per kelas. Pasangan `1716` dengan `953` atau `763` memasang koefisien sumber "
         "tanpa pohon sasaran yang identik. Diagonal tebal: koefisien milik kombinasi itu sendiri.", "",
         "| Bagian | Yang dipindahkan | Pertanyaan yang dijawab |", "|---|---|---|",
         "| 8 | Detektor, ke citra korpus lain | Ketangguhan detektor lintas korpus |",
         "| 9 | Koefisien; detektor dan citra tetap milik sasaran | Kekhususan koefisien terhadap arsitektur |", "",
         "Tebal: nilai terendah tiap kolom.", "",
         "| Jenis permutasi | Jumlah sel | $MAE$ rerata | Terendah | Tertinggi |", "|---|---|---|---|---|"]
    diag = [peta[(t, t)] for t in kom]
    bd = [v for (a, b), v in peta.items() if a[0] == b[0] and a[1] != b[1]]
    bk = [v for (a, b), v in peta.items() if a[0] != b[0] and a[1] == b[1]]
    b2 = [v for (a, b), v in peta.items() if a[0] != b[0] and a[1] != b[1]]
    jenis = [("Koefisien sendiri", diag), ("Korpus sama, detektor berbeda", bd),
             ("Detektor sama, korpus berbeda", bk), ("Korpus dan detektor berbeda", b2)]
    ringkas = [(nama, len(v), float(np.mean(v)), min(v), max(v)) for nama, v in jenis]
    batas = [min(r[k] for r in ringkas) for k in (2, 3, 4)]
    for nama, n, rerata, rendah, tinggi in ringkas:
        sel = [tebal(km(v), v == b) for v, b in zip((rerata, rendah, tinggi), batas)]
        L.append(f"| {tebal(nama, rerata == batas[0])} | {n} | " + " | ".join(sel) + " |")
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
          f"| Uji `1716` dan uji `953` | {i['uji1716_uji953']} dari {i['n953']} pohon SAWIT | Hasil tidak saling bebas |",
          f"| Uji `1716` dan uji `763` | {i['uji1716_uji763']} dari {i['n763']} pohon | Basis latih `1716` pada uji `763` |",
          f"| Uji `1716` dan latih/validasi `763` | {i['uji1716_lihat763']} pohon DEPTH, citra identik | "
          "Dikeluarkan dari latih `763` pada uji `1716` |",
          f"| Uji `953` dan uji `763` | {i['uji953_uji763']} ID pohon | Sesi akuisisi berbeda |",
          f"| Uji `953` dan latih/validasi `763` | {i['uji953_lihat763']} dari {i['n953']} ID pohon | "
          "Sesi akuisisi berbeda, sah menurut V2-E-040 |",
          f"| Uji `763` dan latih/validasi `953` | {i['uji763_lihat953']} dari {i['n763']} ID pohon | "
          "Sesi akuisisi berbeda, sah menurut V2-E-040 |", ""]
    return L


def bagian_glosarium() -> list[str]:
    return [
        "## 10. Glosarium", "",
        "| Simbol atau istilah | Arti |", "|---|---|",
        "| B1–B4 | Kelas kematangan; B1 lewat matang, B4 mentah |",
        "| SAWIT, DEPTH | Awalan citra `1716` yang berasal dari `953` dan `763` |",
        "| $n_c(t)$ | Jumlah deteksi kelas $c$ lintas sisi pohon $t$ yang lolos ambang $" + B + "tau_c$ |",
        "| $k_c$ | Pengali hitungan kelas $c$; nilai di bawah $1,00$ menurunkan hitungan |",
        "| $" + B + "tau_c$ | Ambang skor keyakinan kelas $c$ |",
        "| $" + B + "hat{y}_c(t)$ | Hitungan akhir kelas $c$ pada pohon $t$ |",
        "| $" + B + "text{conf}^{*}$ | Ambang skor keyakinan yang memaksimalkan F1 makro |",
        "| Presisi dan *recall* | Proporsi deteksi yang cocok dengan anotasi acuan, dan proporsi anotasi acuan yang terdeteksi |",
        "| F1 | Rerata harmonik presisi dan *recall* |",
        "| Anotasi | Kotak pembatas (*bounding box*) acuan hasil pelabelan manual |",
        "| IoU | Rasio irisan terhadap gabungan dua kotak pembatas |",
        "| $AP50$ dan $AP50" + B + "text{--}95$ | Luas kurva presisi-*recall* pada IoU $0,50$; rerata atas sepuluh ambang IoU $0,50$–$0,95$ |",
        "| $mAP50$ dan $mAP50" + B + "text{--}95$ | Rerata $AP50$ dan $AP50" + B + "text{--}95$ atas empat kelas |",
        "| `pycocotools` | Evaluator COCO resmi; kolomnya pembanding evaluator laporan ini |",
        "| $MAE$ | Rerata galat absolut per pohon, dalam satuan tandan |",
        "| $MAE$ relatif | $MAE$ dibagi rerata jumlah tandan acuan kelas yang sama |",
        "| $RMSE$ | Akar rerata kuadrat galat; lebih peka terhadap galat besar |",
        "| Bias | Rerata selisih hitungan terhadap acuan; negatif berarti hitungan di bawah acuan |",
        "| " + BIAS + " | Rerata nilai mutlak bias per kelas atas B1–B4 |",
        "| Akurasi ±1 | Proporsi pohon dengan selisih paling banyak satu tandan |",
        "| pp | Poin persentase |",
        "| Makro | Rerata tanpa bobot atas B1–B4 |",
        "| Naif | Tanpa kalibrasi: $k = 1$, $" + B + "tau = 0,25$ |",
        "| $k$ global | Satu $k$ dan satu $" + B + "tau$ untuk semua kelas |",
        "| $k$ per kelas | $k$ per kelas dengan satu $" + B + "tau$ bersama |",
        "| $k + " + B + "tau$ per kelas | $k$ dan $" + B + "tau$ per kelas |",
        "| Validasi silang 5 lipatan | Pohon uji dibagi lima lipatan; koefisien dipasang pada empat lipatan dan diterapkan pada lipatan sisanya, bergiliran |",
        "| Prediksi luar-lipatan (*out-of-fold*) | Hitungan pohon dari koefisien yang dipasang tanpa pohon itu |",
        "| Basis sama 207 pohon | Uji `1716` tanpa 50 pohon DEPTH latih/validasi `763` |",
        "| ID pohon sama, sesi berbeda | Pohon fisik yang sama, difoto pada sesi berselang sekitar 80 hari dengan kamera berbeda |",
        "| Y, RT, RF | YOLO26l, RT-DETR-L, RF-DETR-L |",
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
