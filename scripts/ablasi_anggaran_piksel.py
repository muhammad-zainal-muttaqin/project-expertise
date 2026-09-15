"""Ablasi anggaran piksel pada klasifikasi kematangan tingkat objek.

Pertanyaan: apakah kesalahan klasifikasi B1-B4 menurun ketika informasi
piksel pada objek bertambah? Bukti tandingan yang memicu ablasi ini adalah
dua titik data lama yang arahnya berlawanan dengan hipotesis "crop lebih
besar lebih baik": `ftS` (crop 176 px) = 0,6837 dan `ftH` (crop 256 px @224)
= 0,6569 pada korpus 352.

Dua titik itu tidak dapat memutuskan apa pun karena resolusi crop dan
resolusi masukan model berubah bersamaan. Ablasi ini memisahkan keduanya:

  Percobaan A (bottleneck anggaran piksel, kapasitas model DIKUNCI 224):
      crop native -> INTER_AREA ke S x S -> INTER_CUBIC ke 224 x 224
      S dalam {32, 48, 64, 96, 128, 176, 224}
      Hanya jumlah informasi piksel yang berubah. Backbone, jumlah token,
      dan biaya komputasi identik di seluruh kondisi.

  Percobaan B (plafon resolusi masukan, anggaran piksel DIBUKA PENUH):
      crop native -> langsung ke R x R, R dalam {224, 288, 320}
      Menguji apakah menaikkan resolusi masukan di atas 224 masih membayar.

Pengklasifikasi adalah linear probe di atas fitur ConvNeXt-Tiny ImageNet
yang dibekukan. Alasannya bukan kepraktisan semata: probe linear mengukur
keterpisahan linear informasi yang tersedia pada representasi, tanpa
mencampurkan efek resep penyesuaian terarah (*fine-tuning*), penjadwalan
laju belajar, atau seed. Angka absolutnya karena itu berada di bawah
pengklasifikasi yang disesuaikan terarah, dan memang tidak dimaksudkan
sebagai pembanding langsung terhadap 0,6837. Yang dibaca adalah BENTUK
kurva terhadap S, bukan ketinggiannya.

Korpus: SawitMVC-Depth-YOLO v2.0.0 (763 pohon), split kanonik 2026-08-21,
irisan pohon antar split = 0. Partisi TEST tidak disentuh sama sekali;
seluruh pelaporan memakai VALID.

Penggunaan:
    python scripts/ablasi_anggaran_piksel.py crop
    python scripts/ablasi_anggaran_piksel.py fitur
    python scripts/ablasi_anggaran_piksel.py probe
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

AKAR = Path("D:/Work/Assisten-Dosen/SawitMVC-Depth/SawitMVC-Depth-YOLO")
REPO = Path("D:/Work/Assisten-Dosen/project-expertise")
CACHE = Path(
    "C:/Users/Zainal/AppData/Local/Temp/claude/"
    "D--Work-Assisten-Dosen-project-expertise/"
    "27e30ba2-d1d9-42e9-a6e2-5b2f3ac8239e/scratchpad/ablasi_piksel"
)
KELUARAN = REPO / "results" / "ablasi_piksel_2026-09-10"

CTX = 1.6
KELAS = ["B1", "B2", "B3", "B4"]
SPLIT_PAKAI = ("train", "valid")

# Percobaan A: anggaran piksel dengan masukan model dikunci 224.
ANGGARAN_A = [32, 48, 64, 96, 128, 176, 224]
# Percobaan B: resolusi masukan model, anggaran piksel penuh.
RESOLUSI_B = [224, 288, 320]

ACUAN = "A176"  # meniru konfigurasi `ftS` historis (crop 176 px)


# --------------------------------------------------------------------------
# Tahap 1 - ekstraksi crop native
# --------------------------------------------------------------------------

def kotak_persegi(cx, cy, w, h, W, H):
    """Kotak YOLO ternormalisasi -> jendela persegi diperluas CTX.

    Identik dengan `scripts/build_crop_dataset.py` supaya geometri crop
    dapat dibandingkan dengan eksperimen crop terdahulu.
    """
    sisi = CTX * max(w * W, h * H)
    x0 = int(round(cx * W - sisi / 2))
    y0 = int(round(cy * H - sisi / 2))
    return x0, y0, x0 + int(round(sisi)), y0 + int(round(sisi))


def ambil(img, x0, y0, x1, y1, isi=0):
    """Crop dengan padding tepi; jendela boleh keluar batas citra."""
    H, W = img.shape[:2]
    bentuk = (y1 - y0, x1 - x0) + img.shape[2:]
    buf = np.full(bentuk, isi, dtype=img.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x1), min(H, y1)
    if sx1 > sx0 and sy1 > sy0:
        buf[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return buf


def id_pohon(stem: str) -> str:
    """DAMIMAS_A21B_0004_3 -> DAMIMAS_A21B_0004 (buang indeks sisi)."""
    bagian = stem.rsplit("_", 1)
    return bagian[0] if len(bagian) == 2 and bagian[1].isdigit() else stem


def tahap_crop() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    manifes = []
    t0 = time.time()
    for split in SPLIT_PAKAI:
        keluar = CACHE / split
        keluar.mkdir(exist_ok=True)
        dir_lbl, dir_img = AKAR / split / "labels", AKAR / split / "images"
        n_split = 0
        for p in sorted(dir_lbl.glob("*.txt")):
            teks = p.read_text().strip()
            if not teks:
                continue
            citra = next((dir_img / (p.stem + e) for e in (".jpg", ".jpeg", ".png")
                          if (dir_img / (p.stem + e)).exists()), None)
            if citra is None:
                continue
            bgr = cv2.imread(str(citra), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            H, W = bgr.shape[:2]
            for i, baris in enumerate(teks.splitlines()):
                f = baris.split()
                if len(f) < 5:
                    continue
                k = int(f[0])
                if k < 0:  # anotasi keluaran-saja kelas -1
                    continue
                cx, cy, w, h = (float(x) for x in f[1:5])
                x0, y0, x1, y1 = kotak_persegi(cx, cy, w, h, W, H)
                if x1 - x0 < 8:
                    continue
                crop = ambil(bgr, x0, y0, x1, y1)
                nama = f"{p.stem}__{i:02d}.png"
                cv2.imwrite(str(keluar / nama), crop)
                manifes.append({
                    "berkas": f"{split}/{nama}",
                    "split": split,
                    "kelas": k,
                    "pohon": id_pohon(p.stem),
                    "citra": p.stem,
                    "sisi_native": int(x1 - x0),
                })
                n_split += 1
        print(f"{split}: {n_split} crop  ({time.time() - t0:.0f}s)")
    (CACHE / "manifes.json").write_text(json.dumps(manifes, ensure_ascii=False))
    print(f"total {len(manifes)} crop -> {CACHE / 'manifes.json'}")


# --------------------------------------------------------------------------
# Tahap 2 - ekstraksi fitur per kondisi
# --------------------------------------------------------------------------

def kondisi_semua() -> dict[str, dict]:
    kond = {f"A{s}": {"jenis": "A", "anggaran": s, "masukan": 224} for s in ANGGARAN_A}
    for r in RESOLUSI_B:
        kond[f"B{r}"] = {"jenis": "B", "anggaran": None, "masukan": r}
    return kond


def siapkan_batch(crop_bgr, spec, mean, std):
    """BGR native -> tensor ternormalisasi mengikuti spesifikasi kondisi."""
    masukan = spec["masukan"]
    if spec["jenis"] == "A":
        s = spec["anggaran"]
        kecil = cv2.resize(crop_bgr, (s, s), interpolation=cv2.INTER_AREA)
        img = cv2.resize(kecil, (masukan, masukan), interpolation=cv2.INTER_CUBIC)
    else:
        img = cv2.resize(crop_bgr, (masukan, masukan), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (rgb - mean) / std


def tahap_fitur(hanya: list[str] | None) -> None:
    import torch
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

    torch.set_num_threads(12)
    manifes = json.loads((CACHE / "manifes.json").read_text())
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).eval()
    tubuh, pool = model.features, torch.nn.AdaptiveAvgPool2d(1)

    kond = kondisi_semua()
    if hanya:
        kond = {k: v for k, v in kond.items() if k in hanya}
    fdir = CACHE / "fitur"
    fdir.mkdir(exist_ok=True)

    for nama, spec in kond.items():
        target = fdir / f"{nama}.npy"
        if target.exists():
            print(f"{nama}: sudah ada, dilewati")
            continue
        t0 = time.time()
        batch, out = [], []
        bs = 16 if spec["masukan"] <= 256 else 8
        for rec in manifes:
            crop = cv2.imread(str(CACHE / rec["berkas"]), cv2.IMREAD_COLOR)
            batch.append(siapkan_batch(crop, spec, mean, std))
            if len(batch) == bs:
                x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2)
                with torch.no_grad():
                    out.append(pool(tubuh(x)).flatten(1).numpy())
                batch = []
        if batch:
            x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2)
            with torch.no_grad():
                out.append(pool(tubuh(x)).flatten(1).numpy())
        F = np.concatenate(out).astype(np.float32)
        np.save(target, F)
        print(f"{nama}: {F.shape} dalam {time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------
# Tahap 3 - linear probe dan metrik
# --------------------------------------------------------------------------

def metrik(y, p) -> dict:
    from sklearn.metrics import f1_score
    return {
        "akurasi": float((y == p).mean()),
        "macro_f1": float(f1_score(y, p, average="macro")),
        "mae_ordinal": float(np.abs(y - p).mean()),
        "akurasi_pm1": float((np.abs(y - p) <= 1).mean()),
    }


def tahap_probe(ulangan: int = 2000, seed: int = 42) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    manifes = json.loads((CACHE / "manifes.json").read_text())
    split = np.array([r["split"] for r in manifes])
    y = np.array([r["kelas"] for r in manifes])
    pohon = np.array([r["pohon"] for r in manifes])
    itr, iva = split == "train", split == "valid"

    kond = kondisi_semua()
    # Grid sengaja diperlebar ke bawah: pada putaran pertama seluruh kondisi
    # memilih C = 0,003 yang saat itu merupakan batas bawah grid, sehingga
    # optimumnya belum tentu tercakup.
    grid_C = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    hasil, pred_simpan = {}, {}

    for nama in kond:
        f = CACHE / "fitur" / f"{nama}.npy"
        if not f.exists():
            continue
        F = np.load(f)
        sc = StandardScaler().fit(F[itr])
        Xtr, Xva = sc.transform(F[itr]), sc.transform(F[iva])
        ytr, yva = y[itr], y[iva]

        # Pemilihan C murni di dalam TRAIN, dikelompokkan per pohon.
        gkf, skor = GroupKFold(n_splits=5), {}
        for C in grid_C:
            akur = []
            for i_fit, i_uji in gkf.split(Xtr, ytr, groups=pohon[itr]):
                m = LogisticRegression(C=C, max_iter=3000).fit(Xtr[i_fit], ytr[i_fit])
                akur.append((m.predict(Xtr[i_uji]) == ytr[i_uji]).mean())
            skor[C] = float(np.mean(akur))
        C_pilih = max(skor, key=skor.get)

        m = LogisticRegression(C=C_pilih, max_iter=5000).fit(Xtr, ytr)
        pva = m.predict(Xva)
        pred_simpan[nama] = pva
        hasil[nama] = {
            **kond[nama], "C": C_pilih, "cv_train": skor,
            "valid": metrik(yva, pva),
        }
        print(f"{nama:6s} C={C_pilih:<6g} akurasi={hasil[nama]['valid']['akurasi']:.4f} "
              f"macroF1={hasil[nama]['valid']['macro_f1']:.4f} "
              f"MAE={hasil[nama]['valid']['mae_ordinal']:.4f}", flush=True)

    # Bootstrap berpasangan, resampling pada tingkat POHON (bukan objek),
    # karena empat sisi satu pohon tidak saling bebas.
    yva, pohon_va = y[iva], pohon[iva]
    unik = np.unique(pohon_va)
    indeks = {t: np.where(pohon_va == t)[0] for t in unik}
    rng = np.random.default_rng(seed)
    sampel = [np.concatenate([indeks[t] for t in rng.choice(unik, unik.size, replace=True)])
              for _ in range(ulangan)]

    if ACUAN in pred_simpan:
        from sklearn.metrics import f1_score

        def ukur(pred, idx, nama_metrik):
            yy, pp = yva[idx], pred[idx]
            if nama_metrik == "akurasi":
                return float((yy == pp).mean())
            if nama_metrik == "macro_f1":
                return float(f1_score(yy, pp, average="macro"))
            return float(np.abs(yy - pp).mean())  # mae_ordinal, arah terbalik

        base = pred_simpan[ACUAN]
        for nama, pva in pred_simpan.items():
            if nama == ACUAN:
                continue
            blok = {}
            for nm in ("akurasi", "macro_f1", "mae_ordinal"):
                d = np.array([ukur(pva, s, nm) - ukur(base, s, nm) for s in sampel])
                penuh = np.arange(yva.size)
                lebih_baik = (d < 0) if nm == "mae_ordinal" else (d > 0)
                blok[nm] = {
                    "titik": ukur(pva, penuh, nm) - ukur(base, penuh, nm),
                    "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                    "P_lebih_baik": float(lebih_baik.mean()),
                }
            hasil[nama]["delta_vs_" + ACUAN] = blok

    KELUARAN.mkdir(parents=True, exist_ok=True)
    meta = {
        "korpus": "SawitMVC-Depth-YOLO v2.0.0 (763 pohon)",
        "partisi": {"train_objek": int(itr.sum()), "valid_objek": int(iva.sum()),
                    "valid_pohon": int(unik.size)},
        "backbone": "convnext_tiny IMAGENET1K_V1, dibekukan, global average pool 768-d",
        "pengklasifikasi": "LogisticRegression multinomial, C dipilih via GroupKFold-5 di TRAIN",
        "acuan_delta": ACUAN,
        "ulangan_bootstrap": ulangan,
        "seed": seed,
        "catatan": "TEST tidak disentuh. Crop RGB tanpa kanal mask kotak.",
    }
    (KELUARAN / "ablasi_anggaran_piksel.json").write_text(
        json.dumps({"meta": meta, "hasil": hasil}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\n-> {KELUARAN / 'ablasi_anggaran_piksel.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tahap", choices=["crop", "fitur", "probe"])
    ap.add_argument("--hanya", nargs="*", default=None)
    a = ap.parse_args()
    if a.tahap == "crop":
        tahap_crop()
    elif a.tahap == "fitur":
        tahap_fitur(a.hanya)
    else:
        tahap_probe()
