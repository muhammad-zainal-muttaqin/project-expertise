"""Pengelompokan otomatis dan regresi kematangan pada citra terpotong tandan.

Korpus 953 (SawitMVC-YOLO), kotak acuan, partisi latih/validasi/uji kanonik.
Tiga pertanyaan:
1. Jika tandan dikelompokkan otomatis tanpa label (fitur warna atau DINOv2
   swa-latih), apakah kelompoknya sesuai dengan B1-B4, terutama B2 dan B3?
2. Apakah regresi ordinal lebih baik daripada klasifikasi pada fitur yang sama?
3. Apakah model mengurutkan kematangan lebih tepat di dalam satu citra daripada
   antarpohon (indikasi batas kelas bergeser karena pencahayaan)?

Fitur disimpan per bagian citra sebagai cache, sehingga ekstraksi yang terhenti
dapat dilanjutkan dan analisis dapat diulang tanpa ekstraksi ulang.

Pemakaian:
    python scripts/eksplorasi_klaster_regresi.py --dataset /workspace/SawitMVC-YOLO
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

KELAS = ["B1", "B2", "B3", "B4"]
SPLIT = ["train", "val", "test"]
MODEL_DINO = "vit_small_patch14_dinov2.lvd142m"
RERATA_IMAGENET = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SIMPANGAN_IMAGENET = np.array([0.229, 0.224, 0.225], dtype=np.float32)
AKURASI_ACUAN = 0.6612  # ConvNeXt terlatih, per tampak, kotak acuan uji 953
SUMBER_ACUAN = "logs_ringkas/audit_forensik_2026-09-06/exp_ceiling.log"

# Warna gambar (palet acuan dataviz, mode terang)
PERMUKAAN = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
LATAR_TITIK = "#d4d3cf"
SOROT = "#2a78d6"
RAMP_BIRU = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# --------------------------------------------------------------------------
# Data dan fitur
# --------------------------------------------------------------------------
def daftar_citra(root: Path, batas: int = 0) -> list[tuple[str, Path, str, list[tuple[int, float, float, float, float]]]]:
    """Citra beserta kotak acuannya: (split, jalur, pohon, [(kelas, cx, cy, w, h)])."""
    hasil = []
    for split in SPLIT:
        semua = sorted((root / "images" / split).glob("*.jpg"))
        for jalur in semua[:batas] if batas else semua:
            label = root / "labels" / split / f"{jalur.stem}.txt"
            if not label.exists():
                continue
            kotak = []
            for baris in label.read_text().splitlines():
                bagian = baris.split()
                if len(bagian) == 5:
                    kotak.append((int(bagian[0]), *map(float, bagian[1:])))
            if kotak:
                hasil.append((split, jalur, jalur.stem.rsplit("_", 1)[0], kotak))
    return hasil


def rgb_ke_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB [0, 1] ke CIELAB (iluminan D65)."""
    linier = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    matriks = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
    xyz = linier @ matriks.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def fitur_warna(piksel: np.ndarray) -> np.ndarray:
    """27 fitur warna: statistik Lab, kroma, arah rona, dan histogram rona."""
    from matplotlib.colors import rgb_to_hsv

    rgb = piksel.reshape(-1, 3)
    lab = rgb_ke_lab(rgb)
    kroma = np.hypot(lab[:, 1], lab[:, 2])
    sudut = np.arctan2(lab[:, 2], lab[:, 1])
    bobot_kroma = kroma / max(kroma.sum(), 1e-6)
    hsv = rgb_to_hsv(rgb)
    bobot_rona = hsv[:, 1] * hsv[:, 2]
    histogram = np.histogram(hsv[:, 0], bins=12, range=(0, 1), weights=bobot_rona)[0]
    histogram = histogram / max(histogram.sum(), 1e-6)
    return np.concatenate([
        lab.mean(0), lab.std(0),
        np.percentile(lab[:, 1], [10, 50, 90]), np.percentile(lab[:, 2], [10, 50, 90]),
        [kroma.mean(), (bobot_kroma * np.cos(sudut)).sum(), (bobot_kroma * np.sin(sudut)).sum()],
        histogram,
    ]).astype(np.float32)


def muat_model(ukuran: int, threads: int, perangkat: str):
    import torch
    import timm

    torch.set_num_threads(threads)
    return timm.create_model(MODEL_DINO, pretrained=True, num_classes=0, img_size=ukuran).eval().to(perangkat)


def ekstraksi(citra: list, model, ukuran: int, margin: float, batch: int, perangkat: str) -> dict[str, np.ndarray]:
    import torch

    awal = model.num_prefix_tokens
    meta, warna, konteks, dino, antre = [], [], [], [], []

    def jalankan() -> None:
        x = torch.from_numpy(np.stack(antre)).permute(0, 3, 1, 2).contiguous().to(perangkat)
        with torch.inference_mode():
            token = model.forward_features(x)
        # Token CLS digabung dengan rerata token petak, mengikuti linear probe DINOv2
        dino.append(torch.cat([token[:, 0], token[:, awal:].mean(1)], 1).float().cpu().numpy())
        antre.clear()

    for split, jalur, pohon, kotak in citra:
        gambar = Image.open(jalur).convert("RGB")
        rgb = np.asarray(gambar)
        tinggi, lebar = rgb.shape[:2]
        kecil = np.asarray(gambar.resize((lebar // 4, tinggi // 4), Image.BILINEAR), dtype=np.float32) / 255
        lab_citra = rgb_ke_lab(kecil.reshape(-1, 3))
        ringkas_citra = np.concatenate([lab_citra.mean(0), lab_citra.std(0)]).astype(np.float32)
        for kelas, cx, cy, w, h in kotak:
            px, py, pw, ph = cx * lebar, cy * tinggi, w * lebar, h * tinggi
            # Warna diukur pada 70% bagian tengah kotak untuk mengurangi latar pelepah
            x0, x1 = int(max(px - 0.35 * pw, 0)), int(min(px + 0.35 * pw, lebar))
            y0, y1 = int(max(py - 0.35 * ph, 0)), int(min(py + 0.35 * ph, tinggi))
            tengah = rgb[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)].astype(np.float32) / 255
            warna.append(fitur_warna(tengah))
            # DINOv2 menerima potongan persegi dengan konteks di setiap sisi
            sisi = max(pw, ph) * (1 + 2 * margin)
            potongan = gambar.crop((
                int(max(px - sisi / 2, 0)), int(max(py - sisi / 2, 0)),
                int(min(px + sisi / 2, lebar)), int(min(py + sisi / 2, tinggi)),
            )).resize((ukuran, ukuran), Image.BICUBIC)
            antre.append((np.asarray(potongan, dtype=np.float32) / 255 - RERATA_IMAGENET) / SIMPANGAN_IMAGENET)
            meta.append((split, jalur.name, pohon, kelas, cx, cy, w, h))
            konteks.append(ringkas_citra)
            if len(antre) == batch:
                jalankan()
    if antre:
        jalankan()

    return {
        "split": np.array([m[0] for m in meta]),
        "citra": np.array([m[1] for m in meta]),
        "pohon": np.array([m[2] for m in meta]),
        "kelas": np.array([m[3] for m in meta], dtype=np.int64),
        "kotak": np.array([m[4:] for m in meta], dtype=np.float32),
        "warna": np.stack(warna),
        "konteks": np.stack(konteks),
        "dino": np.concatenate(dino).astype(np.float32),
    }


def ekstraksi_bertahap(args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Ekstraksi per bagian; bagian yang sudah tersimpan tidak diulang jika proses terhenti."""
    citra = daftar_citra(args.dataset, args.batas)
    folder = args.cache / f"dinov2s_{args.ukuran}_m{round(args.margin * 100):02d}_batas{args.batas}_bagian{args.bagian}"
    folder.mkdir(parents=True, exist_ok=True)
    if args.ulang:
        for lama in folder.glob("bagian_*.npz"):
            lama.unlink()
    model, bagian, mulai = None, [], time.time()
    for awal in range(0, len(citra), args.bagian):
        jalur = folder / f"bagian_{awal // args.bagian:02d}.npz"
        if jalur.exists():
            bagian.append(dict(np.load(jalur)))
            continue
        if model is None:
            model = muat_model(args.ukuran, args.threads, args.perangkat)
        data = ekstraksi(citra[awal:awal + args.bagian], model, args.ukuran, args.margin, args.batch, args.perangkat)
        np.savez_compressed(jalur, **data)
        bagian.append(data)
        selesai = min(awal + args.bagian, len(citra))
        print(f"[ekstraksi] {selesai}/{len(citra)} citra, {(time.time() - mulai) / 60:.1f} menit", flush=True)
    return {kunci: np.concatenate([b[kunci] for b in bagian]) for kunci in bagian[0]}


def kurangi_rerata_citra(x: np.ndarray, citra: np.ndarray) -> np.ndarray:
    """Fitur relatif: selisih terhadap rerata tandan lain pada citra yang sama."""
    _, indeks = np.unique(citra, return_inverse=True)
    jumlah = np.zeros((indeks.max() + 1, x.shape[1]))
    np.add.at(jumlah, indeks, x)
    return (x - (jumlah / np.bincount(indeks)[:, None])[indeks]).astype(np.float32)


# --------------------------------------------------------------------------
# Metrik
# --------------------------------------------------------------------------
def metrik(y: np.ndarray, pred: np.ndarray) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score

    cm = confusion_matrix(y, pred, labels=range(4))
    baris = np.maximum(cm.sum(1), 1)
    return {
        "akurasi": round(float((y == pred).mean()), 4),
        "f1_makro": round(float(f1_score(y, pred, average="macro", labels=range(4))), 4),
        "akurasi_pm1": round(float((np.abs(y - pred) <= 1).mean()), 4),
        "mae_kelas": round(float(np.abs(y - pred).mean()), 4),
        "recall_per_kelas": dict(zip(KELAS, np.round(np.diag(cm) / baris, 4).tolist())),
        "b2_menjadi_b3": round(float(cm[1, 2] / baris[1]), 4),
        "b3_menjadi_b2": round(float(cm[2, 1] / baris[2]), 4),
        "matriks_konfusi": cm.tolist(),
    }


def akurasi_hungarian(y: np.ndarray, klaster: np.ndarray) -> float:
    """Akurasi terbaik jika setiap klaster dipasangkan satu-satu dengan satu kelas."""
    from scipy.optimize import linear_sum_assignment

    tabel = np.zeros((4, klaster.max() + 1))
    np.add.at(tabel, (y, klaster), 1)
    baris, kolom = linear_sum_assignment(-tabel)
    return float(tabel[baris, kolom].sum() / len(y))


# --------------------------------------------------------------------------
# Tanpa label: pengelompokan otomatis
# --------------------------------------------------------------------------
def pengelompokan(z_latih, z_uji, y_latih, y_uji, terang_uji, seed: int) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score, silhouette_score

    rng = np.random.default_rng(seed)
    sampel = rng.choice(len(z_latih), size=min(4000, len(z_latih)), replace=False)
    kuartil_terang = np.digitize(terang_uji, np.quantile(terang_uji, [0.25, 0.5, 0.75]))
    hasil: dict = {
        "silhouette_kelas_manusia": round(float(silhouette_score(z_latih[sampel], y_latih[sampel])), 4),
        "silhouette_per_k": {},
        "kelompok_lalu_beri_nama": {},
    }
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z_latih)
        hasil["silhouette_per_k"][k] = round(float(silhouette_score(z_latih[sampel], km.labels_[sampel])), 4)
        if k == 4:
            klaster = km.predict(z_uji)
            # Urutkan klaster menurut rerata kelas anggotanya agar diagonal terbaca
            urutan = np.argsort([y_uji[klaster == c].mean() if (klaster == c).any() else 9 for c in range(4)])
            klaster = np.argsort(urutan)[klaster]
            tabel = np.zeros((4, 4), dtype=int)
            np.add.at(tabel, (y_uji, klaster), 1)
            hasil["k4"] = {
                "ari": round(float(adjusted_rand_score(y_uji, klaster)), 4),
                "nmi_kelas": round(float(normalized_mutual_info_score(y_uji, klaster)), 4),
                "nmi_kuartil_terang_citra": round(float(normalized_mutual_info_score(kuartil_terang, klaster)), 4),
                "akurasi_hungarian": round(akurasi_hungarian(y_uji, klaster), 4),
                "kontingensi": tabel.tolist(),
            }
    for k in [4, 8, 16, 32, 64]:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z_latih)
        # Setiap klaster diberi nama kelas mayoritas di data latih, lalu diuji
        nama = np.array([np.bincount(y_latih[km.labels_ == c], minlength=4).argmax() for c in range(k)])
        pred = nama[km.predict(z_uji)]
        hasil["kelompok_lalu_beri_nama"][k] = {
            "akurasi": round(float((pred == y_uji).mean()), 4),
            "f1_makro": round(float(f1_score(y_uji, pred, average="macro", labels=range(4))), 4),
            "b2_menjadi_b3": round(float(((y_uji == 1) & (pred == 2)).sum() / max((y_uji == 1).sum(), 1)), 4),
        }
    return hasil


# --------------------------------------------------------------------------
# Dengan label: klasifikasi dan regresi ordinal
# --------------------------------------------------------------------------
def ambang_terbaik(skor: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tiga ambang ordinal yang memaksimalkan akurasi validasi (penurunan koordinat)."""
    ambang = np.array([0.5, 1.5, 2.5])
    kandidat = np.quantile(skor, np.linspace(0.005, 0.995, 199))
    for _ in range(4):
        for i in range(3):
            bawah = ambang[i - 1] if i > 0 else -np.inf
            atas = ambang[i + 1] if i < 2 else np.inf
            terbaik = (-1.0, ambang[i])
            for t in kandidat[(kandidat > bawah) & (kandidat < atas)]:
                uji = ambang.copy()
                uji[i] = t
                akurasi = float((np.digitize(skor, uji) == y).mean())
                if akurasi > terbaik[0]:
                    terbaik = (akurasi, t)
            ambang[i] = terbaik[1]
    return ambang


def probe(x_latih, y_latih, x_val, y_val, x_uji, y_uji) -> tuple[dict, np.ndarray]:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    skala = StandardScaler().fit(x_latih)
    x_latih, x_val, x_uji = skala.transform(x_latih), skala.transform(x_val), skala.transform(x_uji)

    terbaik = None
    for c in [0.001, 0.01, 0.1]:
        model = LogisticRegression(C=c, max_iter=1000).fit(x_latih, y_latih)
        akurasi = float((model.predict(x_val) == y_val).mean())
        if terbaik is None or akurasi > terbaik[0]:
            terbaik = (akurasi, c, model)
    hasil = {"klasifikasi": {"C": terbaik[1], **metrik(y_uji, terbaik[2].predict(x_uji))}}

    terbaik = None
    for alpha in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
        model = Ridge(alpha=alpha).fit(x_latih, y_latih)
        mae = float(np.abs(model.predict(x_val) - y_val).mean())
        if terbaik is None or mae < terbaik[0]:
            terbaik = (mae, alpha, model)
    skor_val, skor_uji = terbaik[2].predict(x_val), terbaik[2].predict(x_uji)
    ambang = ambang_terbaik(skor_val, y_val)
    hasil["regresi_dibulatkan"] = {"alpha": terbaik[1], **metrik(y_uji, np.clip(np.rint(skor_uji), 0, 3).astype(int))}
    hasil["regresi_ambang_validasi"] = {
        "alpha": terbaik[1], "ambang": np.round(ambang, 4).tolist(), **metrik(y_uji, np.digitize(skor_uji, ambang)),
    }
    return hasil, skor_uji


def urutan_pasangan(skor: np.ndarray, y: np.ndarray, citra: np.ndarray, pohon: np.ndarray) -> dict:
    """Peluang tandan yang lebih matang mendapat skor lebih kecil, per jenis pasangan."""
    hasil = {}
    for a in range(3):
        ia, ib = np.where(y == a)[0], np.where(y == a + 1)[0]
        benar = (skor[ia][:, None] < skor[ib][None, :]) + 0.5 * (skor[ia][:, None] == skor[ib][None, :])
        satu_citra = citra[ia][:, None] == citra[ib][None, :]
        satu_pohon = pohon[ia][:, None] == pohon[ib][None, :]
        jenis = {
            "satu_citra": satu_citra,
            "satu_pohon_beda_citra": satu_pohon & ~satu_citra,
            "beda_pohon": ~satu_pohon,
        }
        hasil[f"{KELAS[a]}-{KELAS[a + 1]}"] = {
            nama: {"benar": round(float(benar[topeng].mean()), 4) if topeng.any() else None, "n": int(topeng.sum())}
            for nama, topeng in jenis.items()
        }
    return hasil


# --------------------------------------------------------------------------
# Gambar
# --------------------------------------------------------------------------
def gaya_sumbu(ax) -> None:
    ax.set_facecolor(PERMUKAAN)
    ax.set_xticks([])
    ax.set_yticks([])
    for sisi in ax.spines.values():
        sisi.set_visible(False)


def gambar_peta(peta: dict[str, np.ndarray], y: np.ndarray, jalur: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, sumbu = plt.subplots(len(peta), 4, figsize=(12, 3.2 * len(peta)), facecolor=PERMUKAAN)
    for baris, (judul, titik) in enumerate(peta.items()):
        for kolom, kelas in enumerate(KELAS):
            ax = sumbu[baris, kolom]
            gaya_sumbu(ax)
            topeng = y == kolom
            ax.scatter(titik[~topeng, 0], titik[~topeng, 1], s=3, color=LATAR_TITIK, linewidths=0, rasterized=True)
            ax.scatter(titik[topeng, 0], titik[topeng, 1], s=5, color=SOROT, linewidths=0, rasterized=True)
            if baris == 0:
                ax.set_title(f"{kelas} (n = {topeng.sum():,})".replace(",", "."), color=TINTA, fontsize=11)
            if kolom == 0:
                ax.set_ylabel(judul, color=TINTA, fontsize=11)
    fig.tight_layout()
    fig.savefig(jalur, dpi=150, facecolor=PERMUKAAN)
    plt.close(fig)


def gambar_kontingensi(tabel: dict[str, list[list[int]]], jalur: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    peta_warna = LinearSegmentedColormap.from_list("biru", RAMP_BIRU)
    fig, sumbu = plt.subplots(1, len(tabel), figsize=(5.2 * len(tabel), 4.2), facecolor=PERMUKAAN)
    for ax, (judul, isi) in zip(np.atleast_1d(sumbu), tabel.items()):
        isi = np.array(isi, dtype=float)
        persen = 100 * isi / np.maximum(isi.sum(1, keepdims=True), 1)
        ax.imshow(persen, cmap=peta_warna, vmin=0, vmax=100)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{persen[i, j]:.0f}%", ha="center", va="center", fontsize=10,
                        color="#ffffff" if persen[i, j] >= 50 else TINTA)
        ax.set_xticks(range(4), [f"K{j + 1}" for j in range(4)], color=TINTA_2)
        ax.set_yticks(range(4), KELAS, color=TINTA_2)
        ax.set_title(judul, color=TINTA, fontsize=11)
        ax.tick_params(length=0)
        for sisi in ax.spines.values():
            sisi.set_visible(False)
    fig.tight_layout()
    fig.savefig(jalur, dpi=150, facecolor=PERMUKAAN)
    plt.close(fig)


# --------------------------------------------------------------------------
# Utama
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/SawitMVC-YOLO"))
    parser.add_argument("--keluaran", type=Path, default=Path("results/klaster_regresi_2026-09-21"))
    parser.add_argument("--cache", type=Path, default=Path("cache_fitur"))
    parser.add_argument("--ukuran", type=int, default=224)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--perangkat", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ulang", action="store_true", help="ekstraksi ulang walaupun cache tersedia")
    parser.add_argument("--batas", type=int, default=0, help="batasi jumlah citra per split untuk uji cepat")
    parser.add_argument("--bagian", type=int, default=500, help="jumlah citra per berkas cache")
    args = parser.parse_args()

    data = ekstraksi_bertahap(args)
    print(f"[fitur] {len(data['kelas'])} kotak", flush=True)

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler, normalize

    split, y, citra, pohon = data["split"], data["kelas"], data["citra"], data["pohon"]
    latih, val, uji = (split == "train"), (split == "val"), (split == "test")
    terang = data["konteks"][:, 0]
    dino = normalize(data["dino"])
    warna = data["warna"]
    himpunan = {
        "warna": warna,
        "warna_relatif": kurangi_rerata_citra(warna, citra),
        "dinov2": dino,
        "dinov2_relatif": kurangi_rerata_citra(dino, citra),
    }

    ringkasan: dict = {
        "dataset": str(args.dataset),
        "model_fitur": MODEL_DINO,
        "ukuran_potongan": args.ukuran,
        "margin_konteks": args.margin,
        "seed": args.seed,
        "jumlah_kotak": {s: dict(zip(KELAS, np.bincount(y[split == s], minlength=4).tolist())) for s in SPLIT},
        "acuan_akurasi_per_tampak": {"nilai": AKURASI_ACUAN, "sumber": SUMBER_ACUAN},
        "tanpa_label": {},
        "dengan_label": {},
        "urutan_pasangan_uji": {},
    }

    # Tanpa label: fitur warna distandarkan, DINOv2 diproyeksikan ke 64 dimensi
    ruang: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for nama, x in himpunan.items():
        if nama.startswith("warna"):
            skala = StandardScaler().fit(x[latih])
            z_latih, z_uji = skala.transform(x[latih]), skala.transform(x[uji])
        else:
            pca = PCA(n_components=64, random_state=args.seed).fit(x[latih])
            z_latih, z_uji = pca.transform(x[latih]), pca.transform(x[uji])
        ruang[nama] = (z_latih, z_uji)
        print(f"[tanpa label] {nama}", flush=True)
        ringkasan["tanpa_label"][nama] = pengelompokan(z_latih, z_uji, y[latih], y[uji], terang[uji], args.seed)

    # Dengan label: fitur mutlak, lalu mutlak + relatif
    kombinasi = {
        "warna": warna,
        "warna+relatif": np.hstack([warna, himpunan["warna_relatif"]]),
        "dinov2": dino,
        "dinov2+relatif": np.hstack([dino, himpunan["dinov2_relatif"]]),
    }
    for nama, x in kombinasi.items():
        print(f"[dengan label] {nama}", flush=True)
        hasil, skor_uji = probe(x[latih], y[latih], x[val], y[val], x[uji], y[uji])
        ringkasan["dengan_label"][nama] = hasil
        ringkasan["urutan_pasangan_uji"][nama] = urutan_pasangan(skor_uji, y[uji], citra[uji], pohon[uji])

    args.keluaran.mkdir(parents=True, exist_ok=True)
    (args.keluaran / "ringkasan.json").write_text(json.dumps(ringkasan, indent=2, ensure_ascii=False), encoding="utf-8")

    peta = {
        judul: TSNE(n_components=2, perplexity=30, init="pca", random_state=args.seed).fit_transform(ruang[nama][1])
        for judul, nama in [("Fitur warna", "warna"), ("DINOv2 tanpa label", "dinov2")]
    }
    gambar_peta(peta, y[uji], args.keluaran / "peta_tsne_uji.png")
    gambar_kontingensi(
        {
            "Fitur warna, 4 klaster": ringkasan["tanpa_label"]["warna"]["k4"]["kontingensi"],
            "DINOv2, 4 klaster": ringkasan["tanpa_label"]["dinov2"]["k4"]["kontingensi"],
        },
        args.keluaran / "kontingensi_k4_uji.png",
    )
    print(f"[selesai] {args.keluaran / 'ringkasan.json'}")


if __name__ == "__main__":
    main()
