"""Kalibrasi koefisien pengali per kelas untuk pencacahan tandan (V2-E-050).

Koefisien pengali mengoreksi bias sistematis pencacahan: kelas yang
*under-count* dinaikkan, kelas yang *over-count* diturunkan. Prosedur ini tidak
memerlukan pelatihan ulang detektor; koefisien dipasang dari dump prediksi
`.npz` yang sudah terlacak.

Model pencacahan per pohon t dan kelas c:

    n_c(t)   = jumlah deteksi kelas c pada seluruh sisi pohon dengan conf >= tau_c
    y_hat(t) = round(k_c * n_c(t))

Empat metode dibandingkan:
    naif          : tau = 0,25 dan k = 1 (tanpa kalibrasi)
    global        : satu tau dan satu k untuk seluruh kelas
    k_perkelas    : satu tau bersama, k_c per kelas
    k_tau_perkelas: tau_c dan k_c per kelas

Koefisien dipasang pada partisi kalibrasi (val), lalu diterapkan apa adanya ke
partisi uji (test), termasuk skenario silang antar korpus 953 dan 763.

Pemakaian:
    python scripts/kalibrasi_koefisien_pencacahan.py \
        --baseline-root "D:/Work/Assisten-Dosen/Baseline-SawitMVC" \
        --depth-root "D:/Work/Assisten-Dosen/SawitMVC-Depth/SawitMVC-Depth-YOLO"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CLASSES = ["B1", "B2", "B3", "B4"]
TAU_GRID = np.round(np.arange(0.05, 0.91, 0.05), 2)
K_GRID = np.round(np.arange(0.20, 3.001, 0.01), 2)
TAU_NAIF = 0.25

DETECTORS = [
    ("yolo26l", "YOLO26l"),
    ("rtdetr_l", "RT-DETR-L"),
    ("rfdetr_l", "RF-DETR-L"),
]


# --------------------------------------------------------------------------
# Nilai acuan kebenaran (ground truth) per pohon
# --------------------------------------------------------------------------
def muat_gt_953(baseline_root: Path) -> dict[str, dict[str, np.ndarray]]:
    """split_manifest.csv -> {split: {tree_id: array([B1..B4])}}."""
    import csv

    keluaran: dict[str, dict[str, np.ndarray]] = {"train": {}, "val": {}, "test": {}}
    berkas = baseline_root / "ground_truth" / "split_manifest.csv"
    with berkas.open(encoding="utf-8-sig", newline="") as fh:
        for baris in csv.DictReader(fh):
            split = baris["new_split"].strip()
            if split not in keluaran:
                continue
            keluaran[split][baris["tree_id"].strip()] = np.array(
                [float(baris[c]) for c in CLASSES], dtype=float
            )
    return keluaran


def muat_gt_763(depth_root: Path) -> dict[str, dict[str, np.ndarray]]:
    """linked/*.json -> {split: {tree_id: array([B1..B4])}} dari summary.by_class."""
    peta_split = {"train": "train", "valid": "val", "test": "test"}
    keluaran: dict[str, dict[str, np.ndarray]] = {"train": {}, "val": {}, "test": {}}
    for folder, split in peta_split.items():
        for berkas in sorted((depth_root / folder / "linked").glob("*.json")):
            data = json.loads(berkas.read_text(encoding="utf-8-sig"))
            by_class = data.get("summary", {}).get("by_class", {})
            keluaran[split][data["tree_id"]] = np.array(
                [float(by_class.get(c, 0)) for c in CLASSES], dtype=float
            )
    return keluaran


# --------------------------------------------------------------------------
# Dump prediksi
# --------------------------------------------------------------------------
def muat_prediksi(path: Path, awalan: str | None = None) -> dict[str, list[np.ndarray]]:
    """npz {image_id: (N,6)} -> {tree_id: [array conf per kelas]}.

    Mengembalikan, untuk setiap pohon, daftar 4 array berisi confidence seluruh
    deteksi kelas tersebut (digabung lintas sisi). Kelas di luar 0..3 dibuang
    (RF-DETR memancarkan indeks 4 pada confidence sangat rendah).
    """
    data = np.load(path, allow_pickle=True)
    kumpulan: dict[str, list[list[float]]] = {}
    for kunci in data.files:
        nama = kunci
        if awalan is not None:
            if not nama.startswith(awalan):
                continue
            nama = nama[len(awalan):]
        pohon = nama.rsplit("_", 1)[0]
        wadah = kumpulan.setdefault(pohon, [[], [], [], []])
        kotak = data[kunci]
        if kotak.size == 0:
            continue
        kelas = kotak[:, 5].astype(int)
        conf = kotak[:, 4].astype(float)
        sah = (kelas >= 0) & (kelas <= 3)
        for idx, cf in zip(kelas[sah], conf[sah]):
            wadah[int(idx)].append(float(cf))
    return {
        pohon: [np.sort(np.array(v, dtype=float))[::-1] for v in wadah]
        for pohon, wadah in kumpulan.items()
    }


def matriks_hitung(
    prediksi: dict[str, list[np.ndarray]],
    gt: dict[str, np.ndarray],
    tau_grid: np.ndarray,
    batas_pohon: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """-> (n[tau, pohon, kelas], y[pohon, kelas], daftar pohon)."""
    pohon_dipakai = sorted(
        p for p in prediksi if p in gt and (batas_pohon is None or p in batas_pohon)
    )
    n = np.zeros((len(tau_grid), len(pohon_dipakai), 4), dtype=float)
    y = np.zeros((len(pohon_dipakai), 4), dtype=float)
    for i, pohon in enumerate(pohon_dipakai):
        y[i] = gt[pohon]
        for c in range(4):
            conf = prediksi[pohon][c]
            for j, tau in enumerate(tau_grid):
                n[j, i, c] = float((conf >= tau).sum())
    return n, y, pohon_dipakai


# --------------------------------------------------------------------------
# Pemasangan koefisien dan metrik
# --------------------------------------------------------------------------
def mae_kelas(n_kelas: np.ndarray, y_kelas: np.ndarray, k: float) -> float:
    return float(np.abs(np.rint(k * n_kelas) - y_kelas).mean())


def pasang_koefisien(
    n: np.ndarray, y: np.ndarray, metode: str
) -> tuple[list[float], list[float]]:
    """-> (tau per kelas, k per kelas)."""
    if metode == "naif":
        j = int(np.argmin(np.abs(TAU_GRID - TAU_NAIF)))
        return [float(TAU_GRID[j])] * 4, [1.0] * 4

    if metode == "global":
        terbaik = (np.inf, 0, 1.0)
        for j in range(len(TAU_GRID)):
            for k in K_GRID:
                nilai = float(
                    np.mean([mae_kelas(n[j, :, c], y[:, c], float(k)) for c in range(4)])
                )
                if nilai < terbaik[0]:
                    terbaik = (nilai, j, float(k))
        _, j, k = terbaik
        return [float(TAU_GRID[j])] * 4, [k] * 4

    if metode == "k_perkelas":
        terbaik = (np.inf, 0, [1.0] * 4)
        for j in range(len(TAU_GRID)):
            k_terpilih, total = [], 0.0
            for c in range(4):
                nilai = [mae_kelas(n[j, :, c], y[:, c], float(k)) for k in K_GRID]
                idx = int(np.argmin(nilai))
                k_terpilih.append(float(K_GRID[idx]))
                total += nilai[idx]
            total /= 4.0
            if total < terbaik[0]:
                terbaik = (total, j, k_terpilih)
        _, j, k_terpilih = terbaik
        return [float(TAU_GRID[j])] * 4, k_terpilih

    if metode == "k_tau_perkelas":
        tau_terpilih, k_terpilih = [], []
        for c in range(4):
            terbaik = (np.inf, float(TAU_GRID[0]), 1.0)
            for j in range(len(TAU_GRID)):
                nilai = [mae_kelas(n[j, :, c], y[:, c], float(k)) for k in K_GRID]
                idx = int(np.argmin(nilai))
                if nilai[idx] < terbaik[0]:
                    terbaik = (nilai[idx], float(TAU_GRID[j]), float(K_GRID[idx]))
            tau_terpilih.append(terbaik[1])
            k_terpilih.append(terbaik[2])
        return tau_terpilih, k_terpilih

    raise ValueError(f"metode tidak dikenal: {metode}")


def hitung_metrik(
    n: np.ndarray, y: np.ndarray, tau: list[float], k: list[float]
) -> dict:
    prediksi = np.zeros_like(y)
    for c in range(4):
        j = int(np.argmin(np.abs(TAU_GRID - tau[c])))
        prediksi[:, c] = np.rint(k[c] * n[j, :, c])
    selisih = prediksi - y
    per_kelas = {}
    for c, nama in enumerate(CLASSES):
        per_kelas[nama] = {
            "mae": float(np.abs(selisih[:, c]).mean()),
            "rmse": float(np.sqrt((selisih[:, c] ** 2).mean())),
            "bias": float(selisih[:, c].mean()),
            "acc_pm1": float((np.abs(selisih[:, c]) <= 1).mean()),
            "tau": tau[c],
            "k": k[c],
        }
    total_pred = prediksi.sum(axis=1)
    total_gt = y.sum(axis=1)
    return {
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


# --------------------------------------------------------------------------
# Definisi skenario
# --------------------------------------------------------------------------
def bangun_skenario(akar: Path) -> list[dict]:
    """Daftar (korpus latih, dump kalibrasi, dump uji) per detektor."""
    skenario = []
    for slug, label in DETECTORS:
        skenario.append(
            {
                "detektor": label,
                "slug": slug,
                "korpus_latih": "763",
                "kalibrasi": ("763", "val", akar / f"results/new763/predictions/{slug}_rgb_s42_i1280__val.npz", None),
                "uji": ("763", "test", akar / f"results/new763/predictions/{slug}_rgb_s42_i1280__test.npz", None),
            }
        )
        skenario.append(
            {
                "detektor": label,
                "slug": slug,
                "korpus_latih": "763",
                "kalibrasi": ("763", "val", akar / f"results/new763/predictions/{slug}_rgb_s42_i1280__val.npz", None),
                "uji": ("953", "test", akar / f"results/cross_eval/predictions/new763_{slug}__on_953__test.npz", None),
            }
        )
        skenario.append(
            {
                "detektor": label,
                "slug": slug,
                "korpus_latih": "1716",
                "kalibrasi": ("953", "val", akar / f"results/pred_combined1716_{slug}_953_val.npz", None),
                "uji": ("953", "test", akar / f"results/cross_eval/predictions/combined1716_{slug}__on_953__test.npz", None),
            }
        )
        skenario.append(
            {
                "detektor": label,
                "slug": slug,
                "korpus_latih": "1716",
                "kalibrasi": ("763", "val", akar / f"results/pred_combined1716_{slug}_depth_val.npz", None),
                "uji": ("953", "test", akar / f"results/cross_eval/predictions/combined1716_{slug}__on_953__test.npz", None),
            }
        )
        skenario.append(
            {
                "detektor": label,
                "slug": slug,
                "korpus_latih": "1716",
                "kalibrasi": ("763", "val", akar / f"results/pred_combined1716_{slug}_depth_val.npz", None),
                "uji": ("763", "test", akar / f"results/combined1716/predictions/combined1716_{slug}_rgb_s42_i1280__test.npz", "DEPTH_"),
            }
        )
        skenario.append(
            {
                "detektor": label,
                "slug": slug,
                "korpus_latih": "1716",
                "kalibrasi": ("953", "val", akar / f"results/pred_combined1716_{slug}_953_val.npz", None),
                "uji": ("763", "test", akar / f"results/combined1716/predictions/combined1716_{slug}_rgb_s42_i1280__test.npz", "DEPTH_"),
            }
        )
    return skenario


def peta_map50(akar: Path) -> dict[tuple[str, str, str], float]:
    """(korpus_latih, detektor_slug, korpus_uji) -> mAP50 test dari JSON terlacak."""
    sumber = {}
    for slug, _ in DETECTORS:
        sumber[("763", slug, "763")] = (akar / f"results/new763/{slug}_rgb_s42_i1280.json", "test")
        sumber[("763", slug, "953")] = (akar / f"results/cross_eval/new763_{slug}__on_953.json", "test")
        sumber[("1716", slug, "953")] = (akar / f"results/cross_eval/combined1716_{slug}__on_953.json", "test")
        sumber[("1716", slug, "763")] = (
            akar / f"results/combined1716/combined1716_{slug}_rgb_s42_i1280.json",
            "test",
        )
    keluaran = {}
    for kunci, (path, split) in sumber.items():
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        nilai = data.get("splits", {}).get(split, {}).get("mAP50")
        if nilai is not None:
            keluaran[kunci] = float(nilai)
    return keluaran


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-root", default="D:/Work/Assisten-Dosen/Baseline-SawitMVC")
    ap.add_argument("--depth-root", default="D:/Work/Assisten-Dosen/SawitMVC-Depth/SawitMVC-Depth-YOLO")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--out", default="results/counting_koefisien_2026-09-16/koefisien_pencacahan.json")
    argumen = ap.parse_args()

    akar = Path(argumen.project_root).resolve()
    gt = {
        "953": muat_gt_953(Path(argumen.baseline_root)),
        "763": muat_gt_763(Path(argumen.depth_root)),
    }

    cache: dict[tuple[str, str | None], dict[str, list[np.ndarray]]] = {}

    def ambil(path: Path, awalan: str | None):
        kunci = (str(path), awalan)
        if kunci not in cache:
            cache[kunci] = muat_prediksi(path, awalan)
        return cache[kunci]

    map50 = peta_map50(akar)
    skenario = bangun_skenario(akar)

    # Irisan pohon uji 763: partisi test new763 (110 pohon) dan subset DEPTH pada
    # partisi test combined1716 memakai skema split berbeda. Irisan keduanya
    # dipakai sebagai basis perbandingan setara antar korpus latih.
    irisan_763: set[str] | None = None
    for sk in skenario:
        korpus_uji, split_uji, path_uji, awalan_uji = sk["uji"]
        if korpus_uji != "763" or not path_uji.exists():
            continue
        pohon = {
            p_ for p_ in ambil(path_uji, awalan_uji) if p_ in gt["763"][split_uji]
        }
        irisan_763 = pohon if irisan_763 is None else (irisan_763 & pohon)

    baris_keluaran = []
    for sk in skenario:
        korpus_kal, split_kal, path_kal, awalan_kal = sk["kalibrasi"]
        korpus_uji, split_uji, path_uji, awalan_uji = sk["uji"]
        if not path_kal.exists() or not path_uji.exists():
            print(f"[lewat] dump hilang: {path_kal.name} / {path_uji.name}")
            continue

        n_kal, y_kal, pohon_kal = matriks_hitung(
            ambil(path_kal, awalan_kal), gt[korpus_kal][split_kal], TAU_GRID
        )
        n_uji, y_uji, pohon_uji = matriks_hitung(
            ambil(path_uji, awalan_uji), gt[korpus_uji][split_uji], TAU_GRID
        )

        n_iris = y_iris = None
        if korpus_uji == "763" and irisan_763:
            n_iris, y_iris, pohon_iris = matriks_hitung(
                ambil(path_uji, awalan_uji), gt[korpus_uji][split_uji], TAU_GRID, irisan_763
            )

        for metode in ["naif", "global", "k_perkelas", "k_tau_perkelas"]:
            tau, k = pasang_koefisien(n_kal, y_kal, metode)
            metrik_uji = hitung_metrik(n_uji, y_uji, tau, k)
            metrik_kal = hitung_metrik(n_kal, y_kal, tau, k)
            metrik_iris = (
                hitung_metrik(n_iris, y_iris, tau, k) if n_iris is not None else None
            )
            baris_keluaran.append(
                {
                    "detektor": sk["detektor"],
                    "korpus_latih": sk["korpus_latih"],
                    "map50_uji": map50.get((sk["korpus_latih"], sk["slug"], korpus_uji)),
                    "uji_metrik_irisan763": metrik_iris,
                    "kalibrasi": f"{korpus_kal}-{split_kal}",
                    "n_pohon_kalibrasi": len(pohon_kal),
                    "uji": f"{korpus_uji}-{split_uji}",
                    "n_pohon_uji": len(pohon_uji),
                    "silang": korpus_kal != korpus_uji,
                    "metode": metode,
                    "tau": tau,
                    "k": k,
                    "uji_metrik": metrik_uji,
                    "kalibrasi_metrik": metrik_kal,
                    "sumber_kalibrasi": str(path_kal.relative_to(akar)).replace("\\", "/"),
                    "sumber_uji": str(path_uji.relative_to(akar)).replace("\\", "/"),
                }
            )
            print(
                f"{sk['detektor']:>10} latih={sk['korpus_latih']:>4} "
                f"kal={korpus_kal}-{split_kal} uji={korpus_uji}-{split_uji} "
                f"{metode:>15} MAE={metrik_uji['mae_makro']:.4f} "
                f"RMSE={metrik_uji['rmse_makro']:.4f} Acc±1={metrik_uji['acc_pm1_makro']:.4f}"
            )

    keluaran = Path(argumen.out)
    keluaran.parent.mkdir(parents=True, exist_ok=True)
    keluaran.write_text(
        json.dumps(
            {
                "_meta": {
                    "eksperimen": "V2-E-050",
                    "tanggal": "2026-09-16",
                    "metrik_peringkat": "MAE makro (rerata MAE per kelas B1-B4)",
                    "grid_tau": TAU_GRID.tolist(),
                    "grid_k": [float(K_GRID[0]), float(K_GRID[-1]), 0.01],
                    "gt_953": "Baseline-SawitMVC/ground_truth/split_manifest.csv",
                    "gt_763": "SawitMVC-Depth-YOLO/{train,valid,test}/linked/*.json (summary.by_class)",
                    "catatan": "Koefisien dipasang pada partisi val; test tidak pernah dipakai untuk memilih tau atau k.",
                },
                "baris": baris_keluaran,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nDitulis: {keluaran}  ({len(baris_keluaran)} baris)")


if __name__ == "__main__":
    main()
