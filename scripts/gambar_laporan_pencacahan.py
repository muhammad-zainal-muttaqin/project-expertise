"""Gambar untuk laporan kinerja pencacahan (V2-E-050e, V2-E-050g).

Bentuk gambar mengikuti tugas datanya. Perbandingan antararsitektur memakai
diagram batang berkelompok, bias yang memiliki arah memakai batang divergen di
sekitar nol, dan hanya matriks permutasi 9 x 9 yang memakai peta panas. Palet
kategorikal memakai tiga warna Okabe-Ito yang lolos pemeriksaan keterpisahan
buta warna.

Pemakaian:
    python scripts/gambar_laporan_pencacahan.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

KELAS = ["B1", "B2", "B3", "B4"]
DETEKTOR = ["YOLO26l", "RT-DETR-L", "RF-DETR-L"]
KORPUS = ["953", "763", "1716"]
SINGKAT = {"YOLO26l": "Y", "RT-DETR-L": "RT", "RF-DETR-L": "RF"}
WARNA = ["#0072B2", "#E69F00", "#009E73"]
ABU = "#E4E4E4"
TINTA = "#2A2A2A"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "text.color": TINTA,
        "axes.labelcolor": TINTA,
        "xtick.color": TINTA,
        "ytick.color": TINTA,
        "figure.dpi": 200,
        "savefig.dpi": 230,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    }
)


def koma(nilai: float, desimal: int = 4) -> str:
    return f"{nilai:.{desimal}f}".replace(".", ",").replace("-", "−")


def format_koma(nilai, _pos) -> str:
    """Formatter sumbu dengan koma desimal."""
    teks = f"{nilai:g}" if float(nilai).is_integer() else f"{nilai:.2f}"
    return teks.replace(".", ",").replace("-", "−")


def rapikan(ax, *, sumbu_y: bool = True) -> None:
    for tepi in ("top", "right", "left"):
        ax.spines[tepi].set_visible(False)
    ax.spines["bottom"].set_color("#B8B8B8")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(length=0)
    if sumbu_y:
        ax.grid(axis="y", color=ABU, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_koma))


def batang_berkelompok(
    ax,
    kategori: list[str],
    seri: dict[str, list[float]],
    *,
    desimal: int = 3,
    judul: str = "",
    label_y: str = "",
    label_nilai: bool | None = None,
    batas_y: float | None = None,
) -> None:
    x = np.arange(len(kategori))
    if label_nilai is None:
        label_nilai = len(seri) <= 2
    lebar = 0.8 / len(seri)
    for idx, (nama, nilai) in enumerate(seri.items()):
        posisi = x - 0.4 + lebar * (idx + 0.5)
        balok = ax.bar(
            posisi, nilai, lebar * 0.88, label=nama, color=WARNA[idx % len(WARNA)],
            edgecolor="white", linewidth=1.0,
        )
        if label_nilai:
            ax.bar_label(balok, labels=[koma(v, desimal) for v in nilai], fontsize=6.6, padding=1.5)
    ax.set_xticks(x, kategori, fontsize=7.5)
    puncak = batas_y if batas_y is not None else max(max(v) for v in seri.values())
    ax.set_ylim(0, puncak * (1.18 if label_nilai else 1.08))
    ax.tick_params(labelleft=True)
    if judul:
        ax.set_title(judul, fontsize=8.5, pad=6)
    if label_y:
        ax.set_ylabel(label_y, fontsize=7.5)
    rapikan(ax)


def batang_divergen(ax, kategori: list[str], seri: dict[str, list[float]], *, judul: str = "") -> None:
    x = np.arange(len(kategori))
    lebar = 0.8 / len(seri)
    batas = max(abs(v) for nilai in seri.values() for v in nilai) * 1.25
    for idx, (nama, nilai) in enumerate(seri.items()):
        posisi = x - 0.4 + lebar * (idx + 0.5)
        balok = ax.bar(
            posisi, nilai, lebar * 0.88, label=nama, color=WARNA[idx % len(WARNA)],
            edgecolor="white", linewidth=1.0,
        )
    ax.axhline(0, color="#8A8A8A", linewidth=0.8)
    ax.set_xticks(x, kategori, fontsize=7.5)
    ax.set_ylim(-batas, batas)
    if judul:
        ax.set_title(judul, fontsize=8.5, pad=6)
    rapikan(ax)


NAMA_METODE = {"k_perkelas": "$k$ per kelas", "k_tau_perkelas": "$k + \\tau$ per kelas"}


def terbaik_perkelas(pencacahan: dict, korpus: str, detektor: str) -> dict:
    """Varian koefisien per kelas dengan MAE makro terendah."""
    kandidat = [
        x for x in pencacahan["baris"]
        if x["korpus_latih"] == korpus and x["detektor"] == detektor
        and x["metode"] in NAMA_METODE
    ]
    return min(kandidat, key=lambda x: x["metrik"]["mae_makro"])


def simpan(fig, keluaran: Path, nama: str) -> None:
    keluaran.mkdir(parents=True, exist_ok=True)
    jalur = keluaran / nama
    # Legenda dikeluarkan dari tight_layout agar jarak antarpanel tidak melebar,
    # lalu dikembalikan agar bbox_inches="tight" tetap memuatnya.
    for ax in fig.axes:
        if ax.get_legend() is not None:
            ax.get_legend().set_in_layout(True)
    fig.savefig(jalur)
    plt.close(fig)
    print(f"ditulis: {jalur}")


# --------------------------------------------------------------------------
def gambar_deteksi(deteksi: dict, keluaran: Path) -> None:
    metrik = [("presisi", "Presisi"), ("recall", "Recall"), ("f1", "F1"), ("map50", "mAP50"), ("map50_95", "mAP50–95")]
    fig, sumbu = plt.subplots(3, 1, figsize=(6.5, 6.31), sharex=True)
    for baris, korpus in enumerate(KORPUS):
        seri = {}
        for det in DETEKTOR:
            b = next(x for x in deteksi["baris"]
                     if x["korpus_latih"] == korpus and x.get("korpus_uji", korpus) == korpus and x["detektor"] == det)
            seri[det] = [b["makro"][kunci] for kunci, _ in metrik]
        batang_berkelompok(
            sumbu[baris], [nama for _, nama in metrik], seri, desimal=4,
            judul=f"Korpus {korpus}", label_y="nilai metrik",
        )
    sumbu[0].legend(frameon=False, fontsize=7.5, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.42)).set_in_layout(False)
    fig.tight_layout(h_pad=1.6)
    simpan(fig, keluaran, "deteksi_makro.png")


def gambar_deteksi_perkelas(deteksi: dict, keluaran: Path) -> None:
    """F1 per kelas untuk tiga korpus dalam satu gambar."""
    kumpulan = {}
    for korpus in KORPUS:
        kumpulan[korpus] = {}
        for det in DETEKTOR:
            b = next(x for x in deteksi["baris"]
                     if x["korpus_latih"] == korpus and x.get("korpus_uji", korpus) == korpus and x["detektor"] == det)
            kumpulan[korpus][det] = [b["per_kelas"][c]["f1"] for c in KELAS]
    batas = max(v for korpus in kumpulan.values() for seri in korpus.values() for v in seri)
    fig, sumbu = plt.subplots(1, 3, figsize=(6.5, 2.7), sharey=True)
    for kolom, korpus in enumerate(KORPUS):
        batang_berkelompok(sumbu[kolom], KELAS, kumpulan[korpus], desimal=3,
                           judul=f"Korpus {korpus}", batas_y=batas)
    sumbu[0].set_ylabel("F1", fontsize=7.5)
    sumbu[1].legend(frameon=False, fontsize=7.5, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.30)).set_in_layout(False)
    fig.tight_layout()
    simpan(fig, keluaran, "deteksi_perkelas.png")


def gambar_uji_silang(deteksi: dict, pencacahan: dict, keluaran: Path) -> None:
    """Dalam domain berbanding lintas korpus pada detektor RF-DETR-L.

    Kelompok uji 1716 memakai basis 207 pohon untuk ketiga korpus latih.
    """
    sasaran = [("953", "uji 953"), ("763", "uji 763"), ("1716", "uji 1716" + chr(10) + "(207 pohon)")]
    kode_deteksi = {("953", "1716"): "1716@207", ("1716", "1716"): "1716@207"}
    kode_cacah = {("763", "953"): "763>953", ("1716", "953"): "1716>953",
                  ("1716", "763"): "1716>763", ("953", "763"): "953>763",
                  ("953", "1716"): "953>1716@207", ("763", "1716"): "763>1716",
                  ("1716", "1716"): "1716@207"}

    def map50(latih, uji):
        uji = kode_deteksi.get((latih, uji), uji)
        b = next((x for x in deteksi["baris"]
                  if x["korpus_latih"] == latih and x.get("korpus_uji", latih) == uji
                  and x["detektor"] == "RF-DETR-L"), None)
        return b["makro"]["map50"] if b else float("nan")

    def mae_rel(latih, uji):
        korpus = kode_cacah.get((latih, uji), latih)
        return terbaik_perkelas(pencacahan, korpus, "RF-DETR-L")["metrik"]["mae_relatif_makro"]

    fig, sumbu = plt.subplots(1, 2, figsize=(6.5, 2.9))
    for kolom, (fungsi, judul, label_y) in enumerate(
        [(map50, "Kualitas deteksi", "mAP50"), (mae_rel, "Galat pencacahan", "MAE relatif")]
    ):
        seri = {f"latih {latih}": [fungsi(latih, uji) for uji, _ in sasaran] for latih in KORPUS}
        batang_berkelompok(sumbu[kolom], [label for _, label in sasaran], seri, desimal=3, judul=judul)
        sumbu[kolom].set_ylabel(label_y, fontsize=7.5)
    sumbu[0].legend(frameon=False, fontsize=7.5, ncols=3, loc="upper center", bbox_to_anchor=(1.05, 1.32)).set_in_layout(False)
    fig.tight_layout()
    simpan(fig, keluaran, "uji_silang.png")


def gambar_pencacahan(pencacahan: dict, keluaran: Path) -> None:
    terbaik = {(korpus, det): terbaik_perkelas(pencacahan, korpus, det)
               for korpus in KORPUS for det in DETEKTOR}
    fig, sumbu = plt.subplots(1, 3, figsize=(6.5, 2.79))
    for kolom, (kunci, nama) in enumerate(
        [("mae_makro", "MAE makro"), ("rmse_makro", "RMSE makro"), ("acc_pm1_makro", "Akurasi ±1 makro")]
    ):
        seri = {det: [terbaik[(k, det)]["metrik"][kunci] for k in KORPUS] for det in DETEKTOR}
        batang_berkelompok(sumbu[kolom], KORPUS, seri, desimal=4, judul=nama)
        sumbu[kolom].set_xlabel("korpus latih", fontsize=7.5)
    sumbu[1].legend(frameon=False, fontsize=7.5, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.30)).set_in_layout(False)
    fig.tight_layout()
    simpan(fig, keluaran, "pencacahan_makro.png")


def gambar_koefisien(pencacahan: dict, keluaran: Path) -> None:
    k_seri, tau_seri = {}, {}
    for korpus in KORPUS:
        b = terbaik_perkelas(pencacahan, korpus, "RF-DETR-L")
        nama = f"{korpus}: {NAMA_METODE[b['metode']]}"
        k_seri[nama] = b["k_rerata"]
        tau_seri[nama] = b["tau_rerata"]
    fig, sumbu = plt.subplots(1, 2, figsize=(6.5, 2.49))
    batang_berkelompok(sumbu[0], KELAS, k_seri, desimal=2, judul="Koefisien pengali $k$")
    batang_berkelompok(sumbu[1], KELAS, tau_seri, desimal=2, judul="Ambang skor keyakinan $\\tau$")
    sumbu[0].legend(frameon=False, fontsize=7.5, ncols=3, loc="upper center", bbox_to_anchor=(1.0, 1.30)).set_in_layout(False)
    fig.tight_layout()
    simpan(fig, keluaran, "koefisien.png")


def gambar_perkelas_pencacahan(pencacahan: dict, keluaran: Path) -> None:
    mae, bias, akurasi = {}, {}, {}
    for korpus in KORPUS:
        b = terbaik_perkelas(pencacahan, korpus, "RF-DETR-L")
        nama = f"{korpus}: {NAMA_METODE[b['metode']]}"
        mae[nama] = [b["metrik"]["per_kelas"][c]["mae"] for c in KELAS]
        bias[nama] = [b["metrik"]["per_kelas"][c]["bias"] for c in KELAS]
        akurasi[nama] = [b["metrik"]["per_kelas"][c]["acc_pm1"] for c in KELAS]
    fig, sumbu = plt.subplots(1, 3, figsize=(6.5, 2.71))
    batang_berkelompok(sumbu[0], KELAS, mae, desimal=3, judul="MAE per kelas")
    batang_divergen(sumbu[1], KELAS, bias, judul="Bias; negatif berarti kurang hitung")
    batang_berkelompok(sumbu[2], KELAS, akurasi, desimal=3, judul="Akurasi ±1 per kelas")
    sumbu[1].legend(frameon=False, fontsize=7.5, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.32)).set_in_layout(False)
    fig.tight_layout()
    simpan(fig, keluaran, "pencacahan_perkelas.png")


def gambar_efek_kalibrasi_penuh(pencacahan: dict, keluaran: Path) -> None:
    """Perbaikan relatif seluruh detektor dan seluruh metode kalibrasi."""
    metode = [("global", "$k$ global"), ("k_perkelas", "$k$ per kelas"), ("k_tau_perkelas", "$k + \\tau$ per kelas")]

    def ambil(kor, det, met):
        return next(
            x["metrik"] for x in pencacahan["baris"]
            if x["korpus_latih"] == kor and x["detektor"] == det and x["metode"] == met
        )

    data_turun, data_naik = {}, {}
    for kor in KORPUS:
        data_turun[kor], data_naik[kor] = {}, {}
        for kunci, nama in metode:
            data_turun[kor][nama] = []
            data_naik[kor][nama] = []
            for det in DETEKTOR:
                naif = ambil(kor, det, "naif")
                kal = ambil(kor, det, kunci)
                data_turun[kor][nama].append((naif["mae_makro"] - kal["mae_makro"]) / naif["mae_makro"] * 100)
                data_naik[kor][nama].append((kal["acc_pm1_makro"] - naif["acc_pm1_makro"]) * 100)
    batas_turun = max(v for kor in data_turun.values() for seri in kor.values() for v in seri)
    batas_naik = max(v for kor in data_naik.values() for seri in kor.values() for v in seri)
    fig, sumbu = plt.subplots(2, 3, figsize=(6.5, 4.15), sharey="row")
    for kolom, kor in enumerate(KORPUS):
        label = [SINGKAT[d] for d in DETEKTOR]
        batang_berkelompok(sumbu[0][kolom], label, data_turun[kor], desimal=1,
                           judul=f"Korpus {kor}", batas_y=batas_turun)
        batang_berkelompok(sumbu[1][kolom], label, data_naik[kor], desimal=1, batas_y=batas_naik)
    sumbu[0][0].set_ylabel("penurunan MAE (%)", fontsize=7.5)
    sumbu[1][0].set_ylabel("kenaikan akurasi ±1 (pp)", fontsize=7.5)
    sumbu[0][1].legend(frameon=False, fontsize=7.5, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.40)).set_in_layout(False)
    fig.tight_layout(h_pad=1.2)
    simpan(fig, keluaran, "efek_kalibrasi_penuh.png")


def gambar_efek_kalibrasi(keluaran: Path) -> None:
    seri = {"Naif": [6.7039, 1.7864, 4.1858], "Terkalibrasi": [1.0408, 0.6091, 0.8473]}
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    batang_berkelompok(ax, KORPUS, seri, desimal=4, label_y="MAE makro")
    ax.set_xlabel("korpus latih", fontsize=7.5)
    ax.legend(frameon=False, fontsize=7.5, ncols=2, loc="upper right")
    ax.set_title("Efek kalibrasi pada RF-DETR-L", fontsize=8.5, pad=6)
    simpan(fig, keluaran, "efek_kalibrasi.png")


def gambar_metode(pencacahan: dict, keluaran: Path) -> None:
    """MAE makro tiap metode kalibrasi untuk sembilan kombinasi."""
    metode = [("global", "$k$ global"), ("k_perkelas", "$k$ per kelas"),
              ("k_tau_perkelas", "$k + \\tau$ per kelas")]

    def ambil(kor, det, met):
        return next(
            x["metrik"]["mae_makro"] for x in pencacahan["baris"]
            if x["korpus_latih"] == kor and x["detektor"] == det and x["metode"] == met
        )

    kumpulan = {kor: {nama: [ambil(kor, det, kunci) for det in DETEKTOR] for kunci, nama in metode}
                for kor in KORPUS}
    batas = max(v for kor in kumpulan.values() for seri in kor.values() for v in seri)
    fig, sumbu = plt.subplots(1, 3, figsize=(6.5, 2.53), sharey=True)
    for kolom, kor in enumerate(KORPUS):
        batang_berkelompok(sumbu[kolom], [SINGKAT[d] for d in DETEKTOR], kumpulan[kor], desimal=3,
                           judul=f"Korpus {kor}", batas_y=batas)
    sumbu[0].set_ylabel("MAE makro", fontsize=7.5)
    sumbu[1].legend(frameon=False, fontsize=7.5, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.32)).set_in_layout(False)
    fig.tight_layout()
    simpan(fig, keluaran, "metode_kalibrasi.png")


def gambar_permutasi(permutasi: dict, keluaran: Path) -> None:
    kom = [(k, d) for k in KORPUS for d in DETEKTOR]
    peta = {
        ((r["sumber_korpus"], r["sumber_detektor"]), (r["sasaran_korpus"], r["sasaran_detektor"])): r["metrik"]["mae_makro"]
        for r in permutasi["baris"]
    }
    nilai = np.array([[peta[(s, t)] for t in kom] for s in kom])
    label = [f"{k} {SINGKAT[d]}" for k, d in kom]
    fig, ax = plt.subplots(figsize=(6.5, 4.27))
    gambar = ax.imshow(nilai, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(label)), label, fontsize=7.5)
    ax.set_yticks(range(len(label)), label, fontsize=7.5)
    ax.set_xticks(np.arange(-0.5, len(label), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(label), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(length=0)
    for tepi in ("top", "right", "bottom", "left"):
        ax.spines[tepi].set_visible(False)
    peta_warna = plt.get_cmap("Reds")
    rendah, tinggi = float(nilai.min()), float(nilai.max())
    for i in range(len(kom)):
        for j in range(len(kom)):
            posisi = (nilai[i, j] - rendah) / (tinggi - rendah)
            r, g, b, _ = peta_warna(posisi)
            terang = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(
                j, i, koma(nilai[i, j]), ha="center", va="center", fontsize=7,
                color="white" if terang < 0.55 else TINTA,
                fontweight="bold" if i == j else "normal",
            )
    bar = fig.colorbar(gambar, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("MAE makro", fontsize=7.5)
    bar.ax.tick_params(labelsize=7, length=0)
    bar.outline.set_visible(False)
    ax.set_xlabel("Kombinasi sasaran", fontsize=8)
    ax.set_ylabel("Sumber koefisien", fontsize=8)
    ax.set_title("Permutasi koefisien, diagonal bercetak tebal", fontsize=8.5, pad=8)
    simpan(fig, keluaran, "permutasi_koefisien.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hasil", default="results/counting_koefisien_2026-09-16")
    ap.add_argument("--keluaran", default="docs/assets/laporan-pencacahan-2026-09-16")
    arg = ap.parse_args()

    hasil = Path(arg.hasil)
    keluaran = Path(arg.keluaran)
    deteksi = json.loads((hasil / "metrik_deteksi_perkorpus.json").read_text(encoding="utf-8"))
    pencacahan = json.loads((hasil / "pencacahan_perkorpus.json").read_text(encoding="utf-8"))
    permutasi = json.loads((hasil / "permutasi_koefisien.json").read_text(encoding="utf-8"))

    gambar_deteksi(deteksi, keluaran)
    gambar_deteksi_perkelas(deteksi, keluaran)
    gambar_uji_silang(deteksi, pencacahan, keluaran)
    gambar_pencacahan(pencacahan, keluaran)
    gambar_koefisien(pencacahan, keluaran)
    gambar_perkelas_pencacahan(pencacahan, keluaran)
    gambar_efek_kalibrasi_penuh(pencacahan, keluaran)
    gambar_metode(pencacahan, keluaran)
    gambar_permutasi(permutasi, keluaran)


if __name__ == "__main__":
    main()
