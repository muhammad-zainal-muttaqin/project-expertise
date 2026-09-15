"""Gambar tambahan laporan monev dari artefak tetap; tanpa inferensi atau pelatihan.

Melengkapi buat_gambar_dan_verifikasi.py dengan lima gambar: rangkaian sistem,
alur kehilangan tandan, matriks konfusi per tandan, efek informasi kedalaman,
dan mAP50 model terhadap konstruksi diagnostik.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TEAL, ORANYE, ABU = "#176B72", "#B36A2E", "#8C8C8C"
KELAS = ["B1", "B2", "B3", "B4"]
SUMBER_UJI = "Sumber: hasil evaluasi 28 Agustus 2026."
PROFIL = {
    "953": ("results/remote_eval_2026-08-28/gsp_artifacts/953/results_test_locked.json",
            "RGB 953 · Hungarian Anchor A · 135 pohon"),
    "depth": ("results/remote_eval_2026-08-28/gsp_artifacts/depth/results_test_locked.json",
              "Depth 763 · GSP MILP · 110 pohon"),
}


def baca(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def angka(nilai, digit=2, tanda=False):
    teks = f"{abs(nilai):.{digit}f}".replace(".", ",")
    if nilai < 0 and round(abs(nilai), digit) > 0:
        return "−" + teks
    return ("+" if tanda else "") + teks


def ribuan(n):
    return f"{n:,}".replace(",", ".")


def judul(fig, teks):
    fig.suptitle(teks, x=.02, ha="left", fontsize=19, fontweight="bold")


def catatan(fig, teks, y=.03):
    fig.text(.02, y, teks, fontsize=10, color="#444444")


def gambar_rangkaian():
    tahap = [
        ("Empat foto sisi pohon", "RGB, opsional kedalaman"),
        ("Deteksi tandan", "YOLO26l · RT-DETR-L\nRF-DETR-L"),
        ("Penggabungan deteksi", "WBF + re-ranker"),
        ("Pencocokan antarfoto", "urutan pengambilan foto\nHungarian / GSP MILP"),
        ("Kelas kematangan", "per tandan fisik"),
        ("Jumlah tandan", "per pohon dan\nper kelas B1–B4"),
    ]
    lebar, tinggi = 3.3, 1.7
    pusat = [(2, 3.75), (6, 3.75), (10, 3.75), (2, 1.15), (6, 1.15), (10, 1.15)]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, ((nama, isi), (x, y)) in enumerate(zip(tahap, pusat)):
        ujung = i in (0, len(tahap) - 1)
        ax.add_patch(FancyBboxPatch((x - lebar / 2, y - tinggi / 2), lebar, tinggi,
                                    boxstyle="round,pad=0.02,rounding_size=0.15", linewidth=1.8,
                                    facecolor="#EFEFEF" if ujung else "#E6F0F1",
                                    edgecolor=ABU if ujung else TEAL))
        ax.text(x, y + .38, f"{i + 1}. {nama}", ha="center", va="center", fontsize=15, fontweight="bold")
        ax.text(x, y - .3, isi, ha="center", va="center", fontsize=12.5, color="#333333")
    panah = dict(arrowstyle="-|>", mutation_scale=22, color="#444444", linewidth=1.6)
    for a, b in ((0, 1), (1, 2), (3, 4), (4, 5)):
        (xa, ya), (xb, yb) = pusat[a], pusat[b]
        ax.add_patch(FancyArrowPatch((xa + lebar / 2, ya), (xb - lebar / 2, yb), **panah))
    tengah = (pusat[2][1] - tinggi / 2 + pusat[3][1] + tinggi / 2) / 2
    ax.plot([pusat[2][0], pusat[2][0], pusat[3][0]],
            [pusat[2][1] - tinggi / 2, tengah, tengah], color="#444444", linewidth=1.6)
    ax.add_patch(FancyArrowPatch((pusat[3][0], tengah), (pusat[3][0], pusat[3][1] + tinggi / 2), **panah))
    judul(fig, "Rangkaian sistem")
    fig.tight_layout(rect=[0, 0, 1, .92])
    fig.savefig(OUT / "00_rangkaian_sistem.png", dpi=180, facecolor="white")
    plt.close(fig)


def gambar_alur(data):
    label = ["Tandan acuan", "Ditemukan", "Kelas benar", "Prediksi berlebih"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharex=True)
    ringkas = {}
    for ax, (key, (_, nama)) in zip(axes, PROFIL.items()):
        m = data[key]["metrics"]
        fisik, kelas = m["physical_detection"], m["classification"]
        acuan, tertaut = fisik["gt_bunches"], fisik["tp"]
        benar = round(kelas["matched"] * kelas["matched_class_accuracy"])
        berlebih = fisik["pred_clusters"] - tertaut
        nilai = [acuan, tertaut, benar, berlebih]
        pct = [100 * v / acuan for v in nilai]
        ax.barh(range(4), pct, height=.55, color=[ABU, TEAL, TEAL, ORANYE])
        ax.set_yticks(range(4), label)
        ax.invert_yaxis()
        ax.set_xlim(0, 120)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}%"))
        ax.grid(axis="x", alpha=.17)
        ax.set_axisbelow(True)
        ax.set_title(nama, loc="left", fontweight="bold", pad=14)
        for y, (v, p) in enumerate(zip(nilai, pct)):
            ax.text(p + 1, y, f"{ribuan(v)} ({angka(p, 1)}%)", va="center", fontsize=10.5)
        ringkas[key] = {"acuan": acuan, "tertaut": tertaut, "kelas_benar": benar,
                        "terlewat": acuan - tertaut, "kelas_salah": tertaut - benar,
                        "prediksi_berlebih": berlebih}
    judul(fig, "Hasil deteksi dan klasifikasi tandan")
    catatan(fig, "Persen terhadap tandan acuan. " + SUMBER_UJI)
    fig.tight_layout(rect=[0, .06, 1, .92], w_pad=3)
    fig.savefig(OUT / "03_alur_kehilangan_tandan.png", dpi=180, facecolor="white")
    plt.close(fig)
    return ringkas


def gambar_konfusi(data):
    peta = LinearSegmentedColormap.from_list("teal", ["#FFFFFF", TEAL])
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4))
    ringkas = {}
    for ax, (key, (_, nama)) in zip(axes, PROFIL.items()):
        cm = data[key]["metrics"]["classification"]["confusion_prediction_rows"]
        # Artefak: baris = prediksi B1–B4 lalu "tidak terdeteksi"; kolom = acuan B1–B4 lalu "tanpa acuan".
        mat = np.array([[cm[p][g] for p in range(5)] for g in range(4)], dtype=float)
        n_acuan = mat.sum(axis=1)
        frac = mat / n_acuan[:, None]
        ax.imshow(frac, cmap=peta, vmin=0, vmax=1, aspect="auto")
        for g in range(4):
            for p in range(5):
                warna = "white" if frac[g, p] > .55 else "#222222"
                ax.text(p, g, f"{int(mat[g, p])}\n{angka(100 * frac[g, p], 1)}%",
                        ha="center", va="center", fontsize=10, color=warna)
        ax.set_xticks(range(5), KELAS + ["Terlewat"])
        ax.set_yticks(range(4), [f"{k} (n={int(n)})" for k, n in zip(KELAS, n_acuan)])
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Acuan")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(nama, loc="left", fontweight="bold", pad=12)
        berlebih = [int(cm[p][4]) for p in range(4)]
        ax.text(0, -.2, "Prediksi berlebih: " + " · ".join(f"{k} {n}" for k, n in zip(KELAS, berlebih)),
                transform=ax.transAxes, fontsize=10)
        salah = sum(mat[g, p] for g in range(4) for p in range(4) if g != p)
        jauh = sum(mat[g, p] for g in range(4) for p in range(4) if abs(g - p) >= 2)
        ringkas[key] = {"kesalahan_kelas": int(salah), "kesalahan_tidak_bertetangga": int(jauh),
                        "benar_per_kelas": {k: angka(100 * frac[i, i], 2) + "%" for i, k in enumerate(KELAS)},
                        "prediksi_berlebih": berlebih}
    judul(fig, "Kesalahan klasifikasi per tandan")
    catatan(fig, "Persen per baris terhadap tandan acuan kelas tersebut. " + SUMBER_UJI, y=.02)
    fig.tight_layout(rect=[0, .06, 1, .92], w_pad=4)
    fig.savefig(OUT / "04_matriks_konfusi.png", dpi=180, facecolor="white")
    plt.close(fig)
    return ringkas


def gambar_depth():
    lok = baca("results/bootstrap_lokalisasi.json")["selisih_berpasangan"]["agn352_4ch - agn352_ft3_rgb"]
    edge = baca("results/bootstrap_map_awal.json")["selisih_berpasangan"]["edge_rgbd - yolo_rgb"]
    mono = {k: baca(f"results/{k}.json")["selisih"]
            for k in ("boot_sel6_vs_sel5", "boot_sel3_vs_sel1", "boot_sel4_vs_sel2")}
    baris = [
        ("Sensor kedalaman", None, None),
        ("352 · AP50 lokalisasi · RGB+D edge · uji", (lok["delta_titik"], *lok["CI95_delta"]), TEAL),
        ("352 · mAP50 · RGB+D edge · uji", (edge["delta_titik"], *edge["CI95_delta"]), TEAL),
        # Baris 763 disalin dari tabel docs/NEW763_RGBD4_RESULTS.md (bootstrap tingkat citra, VAL).
        ("763 · YOLO26l RGB+D4 · VAL", (0.000166, -0.024195, 0.028892), TEAL),
        ("763 · RT-DETR-L RGB+D4 · VAL", (0.006322, -0.026770, 0.039670), TEAL),
        ("763 · RF-DETR-L RGB+D4 · VAL", (-0.011163, -0.037049, 0.018074), TEAL),
        ("763 · YOLO26l, fusi prediksi · VAL*", (0.037912, 0.016060, 0.059120), TEAL),
        ("763 · RT-DETR-L, fusi prediksi · VAL*", (0.028492, 0.009231, 0.047236), TEAL),
        ("Depth monokular", None, None),
        ("953 · RGB+mono · uji", (mono["boot_sel6_vs_sel5"]["titik"], *mono["boot_sel6_vs_sel5"]["CI95"]), ORANYE),
        ("352 · RGB+mono · uji", (mono["boot_sel3_vs_sel1"]["titik"], *mono["boot_sel3_vs_sel1"]["CI95"]), ORANYE),
        ("352 · edge+mono terhadap edge · uji", (mono["boot_sel4_vs_sel2"]["titik"], *mono["boot_sel4_vs_sel2"]["CI95"]), ORANYE),
    ]
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ticks, labels = [], []
    for y, (label, nilai, warna) in enumerate(baris):
        if nilai is None:
            ax.text(-.01, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontweight="bold", fontsize=11.5)
            continue
        d, lo, hi = nilai
        signifikan = lo > 0 or hi < 0
        ax.hlines(y, lo, hi, color=warna, linewidth=2.2)
        ax.plot(d, y, "o", markersize=9, color=warna, markerfacecolor=warna if signifikan else "white",
                markeredgewidth=2)
        ax.text(1.02, y, f"{angka(d, 4, True)} [{angka(lo, 4, True)}; {angka(hi, 4, True)}]",
                transform=ax.get_yaxis_transform(), va="center", fontsize=10.5)
        ticks.append(y)
        labels.append(label)
    ax.axvline(0, color="#444444", linewidth=1)
    ax.set_yticks(ticks, labels)
    ax.invert_yaxis()
    ax.set_xlim(-.12, .13)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: angka(x, 2, True) if x else "0"))
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=.17)
    ax.set_axisbelow(True)
    ax.set_xlabel("Selisih terhadap kontrol")
    ax.legend(handles=[
        Line2D([], [], marker="o", color=ABU, markerfacecolor=ABU, linestyle="-", label="signifikan"),
        Line2D([], [], marker="o", color=ABU, markerfacecolor="white", markeredgewidth=2, linestyle="-",
               label="belum signifikan"),
    ], loc="lower left", frameon=False, bbox_to_anchor=(0, -.2), ncol=2)
    judul(fig, "Pengaruh penambahan kedalaman")
    catatan(fig, "Garis = selang kepercayaan 95%. mAP50 empat kelas, kecuali baris lokalisasi. * Pengaturan dipilih pada data validasi.\n"
            "Sumber: bootstrap_lokalisasi.json, bootstrap_map_awal.json, boot_sel*.json, NEW763_RGBD4_RESULTS.md.")
    fig.subplots_adjust(left=.28, right=.78, top=.85, bottom=.2)
    fig.savefig(OUT / "05_bukti_depth.png", dpi=180, facecolor="white")
    plt.close(fig)


def gambar_target():
    matriks = baca("results/audit_forensik_2026-09-06/detector_matrix.json")
    baris = [
        # Nilai tanpa JSON terbaca langsung: V2-E-001 (metrics/recap.md), CI_SUMMARY.md,
        # docs/BUKTI-BATAS-KOREKSI-MAP50.md, dan logs_ringkas/.../exp_ceiling.log.
        ("RF-DETR-L (953)", .6012, "model"),
        ("WBF + re-ranker", .5970, "model"),
        ("RF-DETR-L combined1716", .5890, "model"),
        ("YOLO26s empat kelas", matriks["may4"]["may4/test"]["map50"], "model"),
        ("YOLO26s dua kelas", matriks["may2"]["may2/test"]["map50"], "tugas"),
        ("Kotak acuan + ConvNeXt", .6569, "diag"),
        ("Prediksi combined1716, kelas diperbaiki", .7944, "diag"),
        ("Prediksi combined1716, dipilih yang benar", .9752, "diag"),
    ]
    gaya = {"model": (TEAL, None), "tugas": (ORANYE, None), "diag": ("#D9D9D9", "//")}
    fig, ax = plt.subplots(figsize=(13, 6.4))
    for y, (label, nilai, jenis) in enumerate(baris):
        warna, arsir = gaya[jenis]
        ax.barh(y, nilai, height=.6, color=warna, hatch=arsir, edgecolor="#6F6F6F" if arsir else warna)
        ax.text(nilai + .01, y, angka(nilai, 4), va="center", fontsize=10.5, fontweight="bold")
    for target, rata, geser in ((.75, "right", -.008), (.85, "left", .008)):
        ax.axvline(target, color="#444444", linestyle="--", linewidth=1)
        ax.text(target + geser, -.75, f"target {angka(target, 2)}", ha=rata, fontsize=10)
    ax.set_yticks(range(len(baris)), [b[0] for b in baris])
    ax.invert_yaxis()
    ax.set_ylim(len(baris) - .4, -1.1)
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: angka(x, 2)))
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=.17)
    ax.set_axisbelow(True)
    ax.set_xlabel("mAP50, uji 953 (588 citra)")
    ax.legend(handles=[Patch(color=TEAL, label="model"),
                       Patch(color=ORANYE, label="dua kelas"),
                       Patch(facecolor="#D9D9D9", hatch="//", edgecolor="#6F6F6F", label="simulasi dengan label acuan")],
              loc="upper center", bbox_to_anchor=(.5, -.13), ncol=3, frameon=False)
    judul(fig, "Hasil model dan simulasi perbaikan")
    catatan(fig, "Sumber: recap.md, CI_SUMMARY.md, detector_matrix.json, exp_ceiling.log, "
            "audit_batas_koreksi_map50_2026-09-08.json.", y=.025)
    fig.tight_layout(rect=[0, .05, 1, .93])
    fig.savefig(OUT / "06_capaian_dan_diagnostik.png", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    gambar_rangkaian()
    data = {key: baca(rel) for key, (rel, _) in PROFIL.items()}
    ringkas = {"alur": gambar_alur(data), "konfusi": gambar_konfusi(data)}
    gambar_depth()
    gambar_target()
    print(json.dumps(ringkas, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
