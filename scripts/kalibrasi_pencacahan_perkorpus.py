"""Pencacahan per korpus latih dengan kalibrasi lipat-silang (V2-E-050b).

Tiga korpus latih dievaluasi pada partisi ujinya masing-masing:

    953  -> SawitMVC-YOLO test          (141 pohon)
    763  -> SawitMVC-Depth-YOLO test    (110 pohon)
    1716 -> Combined-1716 test          (257 pohon: 141 SAWIT + 116 DEPTH)

Koefisien pengali per kelas dipasang dengan lipat-silang lima lipatan pada
tingkat pohon: koefisien untuk setiap lipatan dipasang pada empat lipatan lain,
lalu diterapkan pada lipatan yang ditahan. Prosedur ini tidak memakai partisi
validasi, sehingga korpus 953 yang tidak memiliki *dump* validasi tetap dapat
dibandingkan setara dengan dua korpus lain.

Pemakaian:
    python scripts/kalibrasi_pencacahan_perkorpus.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kalibrasi_koefisien_pencacahan import (  # noqa: E402
    CLASSES,
    TAU_GRID,
    hitung_metrik,
    matriks_hitung,
    muat_gt_763,
    muat_gt_953,
    muat_prediksi,
    pasang_koefisien,
)

DETEKTOR = [("yolo26l", "YOLO26l"), ("rtdetr_l", "RT-DETR-L"), ("rfdetr_l", "RF-DETR-L")]
METODE = ["naif", "global", "k_perkelas", "k_tau_perkelas"]
N_LIPATAN = 5
SEED = 42


def sumber_dump(akar: Path, korpus: str, slug: str) -> tuple[Path, str | None]:
    if korpus == "763>953":
        return akar / f"results/cross_eval/predictions/new763_{slug}__on_953__test.npz", None
    if korpus == "953":
        return akar / f"results/pred_{slug}_v2repro_953_test.npz", None
    if korpus == "763":
        return akar / f"results/new763/predictions/{slug}_rgb_s42_i1280__test.npz", None
    return akar / f"results/combined1716/predictions/combined1716_{slug}_rgb_s42_i1280__test.npz", None


def gt_korpus(gt953: dict, gt763: dict, korpus: str) -> dict[str, np.ndarray]:
    if korpus in ("953", "763>953"):
        return dict(gt953["test"])
    if korpus == "763":
        return dict(gt763["test"])
    gabungan: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        for pohon, nilai in gt953[split].items():
            gabungan[f"SAWIT_{pohon}"] = nilai
        for pohon, nilai in gt763[split].items():
            gabungan[f"DEPTH_{pohon}"] = nilai
    return gabungan


def muat_prediksi_korpus(path: Path, korpus: str) -> dict[str, list[np.ndarray]]:
    """Untuk korpus 1716 awalan SAWIT_/DEPTH_ dipertahankan sebagai kunci pohon."""
    prediksi = muat_prediksi(path, None)
    if korpus != "1716":
        return prediksi
    return prediksi


def metrik_lipat_silang(n: np.ndarray, y: np.ndarray, metode: str) -> tuple[dict, list[float], list[float]]:
    """Kalibrasi lipat-silang lima lipatan pada tingkat pohon."""
    acak = np.random.default_rng(SEED)
    urutan = acak.permutation(y.shape[0])
    lipatan = np.array_split(urutan, N_LIPATAN)

    prediksi = np.zeros_like(y)
    tau_kumpul, k_kumpul = [], []
    for tahan in lipatan:
        latih = np.setdiff1d(urutan, tahan)
        tau, k = pasang_koefisien(n[:, latih, :], y[latih], metode)
        tau_kumpul.append(tau)
        k_kumpul.append(k)
        for c in range(4):
            j = int(np.argmin(np.abs(TAU_GRID - tau[c])))
            prediksi[tahan, c] = np.rint(k[c] * n[j, tahan, c])

    selisih = prediksi - y
    per_kelas = {}
    for c, nama in enumerate(CLASSES):
        per_kelas[nama] = {
            "mae": float(np.abs(selisih[:, c]).mean()),
            "rmse": float(np.sqrt((selisih[:, c] ** 2).mean())),
            "bias": float(selisih[:, c].mean()),
            "acc_pm1": float((np.abs(selisih[:, c]) <= 1).mean()),
        }
    total_pred, total_gt = prediksi.sum(axis=1), y.sum(axis=1)
    metrik = {
        "n_pohon": int(y.shape[0]),
        "mae_makro": float(np.mean([per_kelas[c]["mae"] for c in CLASSES])),
        "rmse_makro": float(np.mean([per_kelas[c]["rmse"] for c in CLASSES])),
        "acc_pm1_makro": float(np.mean([per_kelas[c]["acc_pm1"] for c in CLASSES])),
        "bias_abs_makro": float(np.mean([abs(per_kelas[c]["bias"]) for c in CLASSES])),
        "mae_total_pohon": float(np.abs(total_pred - total_gt).mean()),
        "rmse_total_pohon": float(np.sqrt(((total_pred - total_gt) ** 2).mean())),
        "bias_total_pohon": float((total_pred - total_gt).mean()),
        "per_kelas": per_kelas,
    }
    tau_rerata = [float(np.mean([t[c] for t in tau_kumpul])) for c in range(4)]
    k_rerata = [float(np.mean([kk[c] for kk in k_kumpul])) for c in range(4)]
    return metrik, tau_rerata, k_rerata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-root", default="D:/Work/Assisten-Dosen/Baseline-SawitMVC")
    ap.add_argument("--depth-root", default="D:/Work/Assisten-Dosen/SawitMVC-Depth/SawitMVC-Depth-YOLO")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--out", default="results/counting_koefisien_2026-09-16/pencacahan_perkorpus.json")
    arg = ap.parse_args()

    akar = Path(arg.project_root).resolve()
    gt953 = muat_gt_953(Path(arg.baseline_root))
    gt763 = muat_gt_763(Path(arg.depth_root))

    baris = []
    for korpus in ("953", "763", "1716", "763>953"):
        gt = gt_korpus(gt953, gt763, korpus)
        for slug, label in DETEKTOR:
            path, awalan = sumber_dump(akar, korpus, slug)
            if not path.exists():
                print(f"[lewat] {path}")
                continue
            prediksi = muat_prediksi_korpus(path, korpus)
            n, y, pohon = matriks_hitung(prediksi, gt, TAU_GRID)
            for metode in METODE:
                metrik, tau, k = metrik_lipat_silang(n, y, metode)
                baris.append(
                    {
                        "korpus_latih": korpus,
                        "detektor": label,
                        "metode": metode,
                        "n_pohon": len(pohon),
                        "tau_rerata": tau,
                        "k_rerata": k,
                        "metrik": metrik,
                        "sumber_dump": str(path.relative_to(akar)).replace("\\", "/"),
                    }
                )
                print(
                    f"{korpus:>4} {label:>10} {metode:>15} n={len(pohon):>3} "
                    f"MAE={metrik['mae_makro']:.4f} RMSE={metrik['rmse_makro']:.4f} "
                    f"|bias|={metrik['bias_abs_makro']:.4f} Acc±1={metrik['acc_pm1_makro']:.4f}"
                )

    keluaran = Path(arg.out)
    keluaran.parent.mkdir(parents=True, exist_ok=True)
    keluaran.write_text(
        json.dumps(
            {
                "_meta": {
                    "eksperimen": "V2-E-050b",
                    "tanggal": "2026-09-16",
                    "protokol": f"kalibrasi lipat-silang {N_LIPATAN} lipatan pada tingkat pohon, seed {SEED}",
                    "korpus": {
                        "953": "SawitMVC-YOLO test (141 pohon)",
                        "763": "SawitMVC-Depth-YOLO test (110 pohon)",
                        "1716": "Combined-1716 test (257 pohon)",
                        "763>953": "detektor latih 763 diuji pada SawitMVC-YOLO test (141 pohon)",
                    },
                    "gt_953": "Baseline-SawitMVC/ground_truth/split_manifest.csv",
                    "gt_763": "SawitMVC-Depth-YOLO/{train,valid,test}/linked/*.json",
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
