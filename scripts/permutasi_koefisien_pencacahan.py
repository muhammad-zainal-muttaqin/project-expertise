"""Matriks permutasi koefisien pencacahan (V2-E-050d).

Koefisien pengali per kelas yang dipasang pada satu kombinasi korpus latih dan
detektor diterapkan ke seluruh kombinasi lain, menghasilkan matriks 9 x 9.

Aturan pemisahan data:

* Sumber dan sasaran pada korpus yang sama memakai lipat-silang lima lipatan
  dengan pembagian pohon yang identik, sehingga koefisien tidak pernah dipasang
  pada pohon yang sedang dinilai.
* Sumber dan sasaran pada korpus berbeda tidak memiliki irisan pohon, sehingga
  koefisien dipasang pada seluruh partisi uji sumber.

Pemakaian:
    python scripts/permutasi_koefisien_pencacahan.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kalibrasi_koefisien_pencacahan import (  # noqa: E402
    CLASSES,
    TAU_GRID,
    matriks_hitung,
    muat_gt_763,
    muat_gt_953,
    muat_prediksi,
    pasang_koefisien,
)
from kalibrasi_pencacahan_perkorpus import DETEKTOR, N_LIPATAN, SEED, gt_korpus, sumber_dump

METODE = "k_perkelas"
KORPUS = ("953", "763", "1716")


def metrik(prediksi: np.ndarray, y: np.ndarray) -> dict:
    selisih = prediksi - y
    per_kelas = {
        nama: {
            "mae": float(np.abs(selisih[:, c]).mean()),
            "rmse": float(np.sqrt((selisih[:, c] ** 2).mean())),
            "bias": float(selisih[:, c].mean()),
            "acc_pm1": float((np.abs(selisih[:, c]) <= 1).mean()),
        }
        for c, nama in enumerate(CLASSES)
    }
    return {
        "mae_makro": float(np.mean([per_kelas[c]["mae"] for c in CLASSES])),
        "rmse_makro": float(np.mean([per_kelas[c]["rmse"] for c in CLASSES])),
        "acc_pm1_makro": float(np.mean([per_kelas[c]["acc_pm1"] for c in CLASSES])),
        "bias_abs_makro": float(np.mean([abs(per_kelas[c]["bias"]) for c in CLASSES])),
    }


def terapkan(n: np.ndarray, tau: list[float], k: list[float], indeks: np.ndarray | None = None) -> np.ndarray:
    pilih = slice(None) if indeks is None else indeks
    keluaran = np.zeros((n.shape[1] if indeks is None else len(indeks), 4))
    for c in range(4):
        j = int(np.argmin(np.abs(TAU_GRID - tau[c])))
        keluaran[:, c] = np.rint(k[c] * n[j, pilih, c])
    return keluaran


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-root", default="D:/Work/Assisten-Dosen/Baseline-SawitMVC")
    ap.add_argument("--depth-root", default="D:/Work/Assisten-Dosen/SawitMVC-Depth/SawitMVC-Depth-YOLO")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--out", default="results/counting_koefisien_2026-09-16/permutasi_koefisien.json")
    arg = ap.parse_args()

    akar = Path(arg.project_root).resolve()
    gt953 = muat_gt_953(Path(arg.baseline_root))
    gt763 = muat_gt_763(Path(arg.depth_root))

    # Matriks hitungan per kombinasi, dengan urutan pohon yang identik per korpus.
    data: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    lipatan_korpus: dict[str, list[np.ndarray]] = {}
    for korpus in KORPUS:
        gt = gt_korpus(gt953, gt763, korpus)
        pohon_acuan = None
        for slug, label in DETEKTOR:
            path, _ = sumber_dump(akar, korpus, slug)
            n, y, pohon = matriks_hitung(muat_prediksi(path, None), gt, TAU_GRID)
            if pohon_acuan is None:
                pohon_acuan = pohon
                acak = np.random.default_rng(SEED)
                lipatan_korpus[korpus] = np.array_split(acak.permutation(len(pohon)), N_LIPATAN)
            elif pohon != pohon_acuan:
                raise RuntimeError(f"urutan pohon berbeda pada {korpus}/{label}")
            data[(korpus, label)] = (n, y)
        print(f"{korpus}: {len(pohon_acuan)} pohon dimuat")

    # Koefisien sumber: versi penuh untuk permutasi lintas korpus, versi per
    # lipatan untuk permutasi di dalam korpus yang sama.
    koef_penuh = {}
    koef_lipatan = {}
    for kunci, (n, y) in data.items():
        koef_penuh[kunci] = pasang_koefisien(n, y, METODE)
        korpus = kunci[0]
        daftar = []
        for tahan in lipatan_korpus[korpus]:
            latih = np.setdiff1d(np.arange(y.shape[0]), tahan)
            daftar.append(pasang_koefisien(n[:, latih, :], y[latih], METODE))
        koef_lipatan[kunci] = daftar
        print(f"koefisien terpasang: {kunci[0]:>4} {kunci[1]}")

    baris = []
    for sumber in data:
        for sasaran in data:
            n_t, y_t = data[sasaran]
            if sumber[0] == sasaran[0]:
                prediksi = np.zeros_like(y_t)
                for idx, tahan in enumerate(lipatan_korpus[sasaran[0]]):
                    tau, k = koef_lipatan[sumber][idx]
                    prediksi[tahan] = terapkan(n_t, tau, k, tahan)
                tau_lapor, k_lapor = koef_lipatan[sumber][0]
            else:
                tau_lapor, k_lapor = koef_penuh[sumber]
                prediksi = terapkan(n_t, tau_lapor, k_lapor)
            hasil = metrik(prediksi, y_t)
            baris.append(
                {
                    "sumber_korpus": sumber[0],
                    "sumber_detektor": sumber[1],
                    "sasaran_korpus": sasaran[0],
                    "sasaran_detektor": sasaran[1],
                    "sama_korpus": sumber[0] == sasaran[0],
                    "sama_detektor": sumber[1] == sasaran[1],
                    "n_pohon": int(y_t.shape[0]),
                    "tau": tau_lapor,
                    "k": k_lapor,
                    "metrik": hasil,
                }
            )
        print(f"selesai sumber {sumber[0]:>4} {sumber[1]}")

    keluaran = Path(arg.out)
    keluaran.parent.mkdir(parents=True, exist_ok=True)
    keluaran.write_text(
        json.dumps(
            {
                "_meta": {
                    "eksperimen": "V2-E-050d",
                    "tanggal": "2026-09-16",
                    "metode": "k per kelas dengan satu ambang bersama",
                    "protokol": (
                        "permutasi di dalam korpus memakai lipat-silang lima lipatan; "
                        "permutasi lintas korpus memasang koefisien pada seluruh partisi uji sumber"
                    ),
                },
                "baris": baris,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nDitulis: {keluaran} ({len(baris)} baris)")


if __name__ == "__main__":
    main()
