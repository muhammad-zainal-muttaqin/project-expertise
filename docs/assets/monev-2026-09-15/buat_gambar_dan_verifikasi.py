"""Buat gambar laporan monev dari artefak tetap; tanpa inferensi atau pelatihan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SUMBER = {
    "953": "results/remote_eval_2026-08-28/gsp_artifacts/953/results_test_locked.json",
    "depth": "results/remote_eval_2026-08-28/gsp_artifacts/depth/results_test_locked.json",
}


def baca(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def angka(nilai, digit=2):
    return f"{nilai:.{digit}f}".replace(".", ",")


def main():
    data = {k: baca(v) for k, v in SUMBER.items()}
    bukti = {"tanggal": "2026-09-15", "cakupan": "Pembacaan artefak dan pemeriksaan aritmetika; tanpa evaluasi model ulang", "profil": {}}
    for key, d in data.items():
        m = d["metrics"]
        fisik = m["physical_detection"]
        kelas = m["classification"]
        cacah = m["counting"]
        n = d["n_trees"]
        assert abs(fisik["precision"] - fisik["tp"] / fisik["pred_clusters"]) < 1e-12
        assert abs(fisik["recall"] - fisik["tp"] / fisik["gt_bunches"]) < 1e-12
        assert abs(fisik["f1"] - 2 * fisik["tp"] / (fisik["pred_clusters"] + fisik["gt_bunches"])) < 1e-12
        cm = kelas["confusion_prediction_rows"]
        pred = [sum(row) for row in cm[:4]]
        gt = [sum(row[c] for row in cm) for c in range(4)]
        bias = [(p - g) / g for p, g in zip(pred, gt)]
        hitung = {}
        for field in ("exact_accuracy", "plus_minus_1_accuracy", "vector_exact_accuracy"):
            raw = cacah[field] * n
            assert abs(raw - round(raw)) < 1e-10
            hitung[field] = round(raw)
        bukti["profil"][key] = {
            "sumber": SUMBER[key],
            "sha256": hashlib.sha256((ROOT / SUMBER[key]).read_bytes()).hexdigest(),
            "pohon": n,
            "metrik_asli": m,
            "jumlah_pohon_memenuhi_kriteria": hitung,
            "kelas_benar_dari_akurasi": round(kelas["matched"] * kelas["matched_class_accuracy"]),
            "prediksi_per_kelas_dari_matriks": pred,
            "acuan_per_kelas_dari_matriks": gt,
            "bias_relatif_dari_matriks": bias,
            "rerata_mutlak_bias": sum(abs(b) for b in bias) / 4,
            "audit_konservasi": {
                "prediksi_fisik": fisik["pred_clusters"],
                "prediksi_matriks": sum(pred),
                "acuan_fisik": fisik["gt_bunches"],
                "acuan_matriks": sum(gt),
                "pasangan_fisik": fisik["tp"],
                "pasangan_matriks": sum(sum(row[:4]) for row in cm[:4]),
            },
        }
    (OUT / "verifikasi_angka.json").write_text(json.dumps(bukti, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    warna = ["#176B72", "#B36A2E"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharex=True)
    kategori = ["Jumlah tepat", "Galat ≤ 1", "B1–B4 tepat"]
    fields = ["exact_accuracy", "plus_minus_1_accuracy", "vector_exact_accuracy"]
    for ax, (key, d), color, nama in zip(axes, data.items(), warna, ["RGB 953 · 135 pohon uji", "Depth 763 · 110 pohon uji"]):
        vals = [d["metrics"]["counting"][f] * 100 for f in fields]
        ax.barh(range(3), vals, height=.5, color=color)
        ax.set_yticks(range(3), kategori)
        ax.invert_yaxis()
        ax.set_xlim(0, 105)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}%"))
        ax.grid(axis="x", alpha=.17)
        ax.set_axisbelow(True)
        ax.set_title(nama, loc="left", fontweight="bold", pad=16)
        for y, (v, field) in enumerate(zip(vals, fields)):
            n_ok = bukti["profil"][key]["jumlah_pohon_memenuhi_kriteria"][field]
            ax.text(v + 1, y, f"{angka(v)}%\n({n_ok}/{d['n_trees']})", va="center", fontsize=10)
    fig.suptitle("Ketepatan pencacahan per pohon", x=.02, ha="left", fontsize=19, fontweight="bold")
    fig.text(.02, .035, "Konfigurasi dan pohon uji berbeda; selisih ini belum menunjukkan efek sensor.\nSumber: hasil evaluasi 28 Agustus 2026.", fontsize=10, color="#444444")
    fig.tight_layout(rect=[0, .13, 1, .9], w_pad=2.5)
    fig.savefig(OUT / "01_ketepatan_pencacahan.png", dpi=180, facecolor="white")
    plt.close(fig)

    p = bukti["profil"]["953"]
    vals = [100 * b for b in p["bias_relatif_dari_matriks"]]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.barh(["B1", "B2", "B3", "B4"], vals, color=[warna[0] if x > 0 else warna[1] for x in vals], height=.55)
    ax.invert_yaxis()
    ax.axvline(0, color="#444444", linewidth=1)
    ax.set_xlim(-50, 30)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: angka(x, 0).replace("-", "−") + "%"))
    ax.grid(axis="x", alpha=.17)
    ax.set_axisbelow(True)
    for y, v in enumerate(vals):
        label = ("+" if v > 0 else "−") + angka(abs(v)) + "%"
        ax.text(v + (1 if v > 0 else -1), y, label, va="center", ha="left" if v > 0 else "right", fontweight="bold")
    ax.set_title("Bias jumlah per kelas, RGB 953", loc="left", fontsize=18, fontweight="bold", pad=22)
    ax.set_xlabel("Bias terhadap acuan (negatif = kurang terhitung)")
    fig.text(.08, .065, "Total 1.324 prediksi / 1.342 acuan (−1,34%); rerata mutlak bias 18,78%.\nHungarian Anchor A, 135 pohon uji.", fontsize=11)
    fig.tight_layout(rect=[0, .17, 1, 1])
    fig.savefig(OUT / "02_bias_kelas_rgb.png", dpi=180, facecolor="white")
    plt.close(fig)
    print(json.dumps({k: v["audit_konservasi"] for k, v in bukti["profil"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
