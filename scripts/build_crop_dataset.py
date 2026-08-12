"""Ekstrak crop tandan (RGB + relief depth) untuk classifier kematangan — Fase 6.

Kenapa crop, bukan citra penuh: diagnostik Fase 6 menunjukkan 44,5% kemampuan
detektor hangus karena SALAH KELAS (AP50 class-agnostic 0,6677 vs mAP50
class-aware 0,3707), dan sinyal depth yang terukur adalah **relief lokal** yang
hanya muncul setelah pooling wilayah (AUC B1-vs-B4: 0,592 per-piksel ->
0,724 setelah pooling 16 piksel). Classifier crop mengkonsumsi keduanya pada
skala yang benar.

Kanal depth yang dipakai BUKAN inverse-depth absolut (itu nuisance: standoff
per citra std 0,82 m), melainkan:

    R = Z - median(Z valid pada jendela crop)        [meter]
    v = 128 + clip(R, -0.10, +0.10) / 0.10 * 127     [uint8]

Step kuantisasi jadi 0,08 cm/level, vs 2,91 cm/level pada encoding lama di
Z=2,5 m (~36x lebih halus pada sinyal yang relevan). Sinyal yang mau ditangkap:
relief per kelas B1 +2,8 cm / B2 0,0 / B3 -1,5 / B4 -5,1 cm
(Kruskal-Wallis H=99,8, p=1,7e-21).

Crop sengaja diperluas (ctx=1.6x sisi box) supaya CINCIN sekitar objek ikut
masuk — relief adalah kontras box-vs-cincin, jadi tanpa cincin sinyalnya hilang.

Usage:
    .venv/bin/python scripts/build_crop_dataset.py --src 953
    .venv/bin/python scripts/build_crop_dataset.py --src 352
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

D953 = Path("/workspace/SawitMVC-YOLO")
D352 = Path("/workspace/SawitMVC-Depth")
DEPTH352 = Path("/workspace/depth_png_352")
SPLIT352 = D352 / "splits" / "canonical_70_15_15"
SPLIT6 = Path("/workspace/project-expertise/splits_fase6")
OUT = Path("/workspace/crops_fase6")

S = 176           # sisi crop tersimpan (training pakai random-resized-crop ke 160)
CTX = 1.6         # faktor perluasan box -> cincin ikut masuk
RELIEF_M = 0.10   # rentang relief yang dipetakan ke 0..255 (meter)
ZN, ZF = 0.8, 15.0


def dekode_z(v: np.ndarray) -> np.ndarray:
    """uint8 inverse-depth kanonik -> meter. 0 = tidak ada data (jadi NaN)."""
    z = v.astype(np.float32)
    inv = (z - 1.0) / 254.0
    out = 1.0 / (inv * (1.0 / ZN - 1.0 / ZF) + 1.0 / ZF)
    out[v == 0] = np.nan
    return out


def kotak_persegi(cx, cy, w, h, W, H):
    """Box YOLO ternormalisasi -> jendela persegi (x0,y0,x1,y1) diperluas CTX."""
    sisi = CTX * max(w * W, h * H)
    x0 = int(round(cx * W - sisi / 2))
    y0 = int(round(cy * H - sisi / 2))
    return x0, y0, x0 + int(round(sisi)), y0 + int(round(sisi))


def ambil(img, x0, y0, x1, y1, isi):
    """Crop dengan padding tepi (jendela boleh keluar citra)."""
    H, W = img.shape[:2]
    bentuk = (y1 - y0, x1 - x0) + img.shape[2:]
    buf = np.full(bentuk, isi, dtype=img.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x1), min(H, y1)
    if sx1 > sx0 and sy1 > sy0:
        buf[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return buf


def _kerja(tugas):
    img_path, lbl_path, dep_path = tugas
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return []
    H, W = bgr.shape[:2]

    Z = None
    if dep_path is not None:
        d = cv2.imread(str(dep_path), cv2.IMREAD_UNCHANGED)
        if d is not None:
            Z = dekode_z(d)

    hasil = []
    for baris in Path(lbl_path).read_text().splitlines():
        p = baris.split()
        if len(p) < 5:
            continue
        k = int(p[0])
        if k < 0:                      # anotasi output-only kelas -1 ("U")
            continue
        cx, cy, w, h = (float(x) for x in p[1:5])
        x0, y0, x1, y1 = kotak_persegi(cx, cy, w, h, W, H)
        if x1 - x0 < 8:
            continue

        rgb = cv2.resize(ambil(bgr, x0, y0, x1, y1, 0), (S, S), interpolation=cv2.INTER_AREA)

        # Kanal mask box: crop diperluas CTX=1.6 sehingga di kanopi yang padat
        # SERING ada lebih dari satu tandan di dalam satu crop. Tanpa penanda,
        # model tidak tahu tandan mana yang harus dinilai — ini bukan detail
        # kosmetik, ini ambiguitas target. Mask menandai footprint box GT.
        sisi_win = x1 - x0
        mw = (w * W) / sisi_win
        mh = (h * H) / sisi_win
        msk = np.zeros((S, S), np.uint8)
        mx0 = int(round((0.5 - mw / 2) * S)); mx1 = int(round((0.5 + mw / 2) * S))
        my0 = int(round((0.5 - mh / 2) * S)); my1 = int(round((0.5 + mh / 2) * S))
        msk[max(0, my0):min(S, my1), max(0, mx0):min(S, mx1)] = 255

        if Z is None:
            dep = np.zeros((S, S, 2), np.uint8)
        else:
            zc = ambil(Z, x0, y0, x1, y1, np.nan)
            valid = np.isfinite(zc)
            if valid.sum() < 20:
                dep = np.zeros((S, S, 2), np.uint8)
            else:
                ref = float(np.median(zc[valid]))          # referensi lokal -> buang standoff
                rel = np.clip(zc - ref, -RELIEF_M, RELIEF_M)
                v = np.where(valid, 128.0 + rel / RELIEF_M * 127.0, 128.0)
                dep = np.stack([
                    cv2.resize(v.astype(np.uint8), (S, S), interpolation=cv2.INTER_AREA),
                    cv2.resize(valid.astype(np.uint8) * 255, (S, S), interpolation=cv2.INTER_AREA),
                ], axis=-1)

        hasil.append((rgb, dep, msk, k, Path(img_path).stem))
    return hasil


def kumpulkan_tugas(src: str):
    tugas, split_per_citra = [], {}
    if src == "953":
        for baris in (SPLIT6 / "pretrain953_images.txt").read_text().splitlines():
            f = Path(baris.strip())
            if not f.name:
                continue
            lbl = f.parent.parent.parent / "labels" / f.parent.name / f"{f.stem}.txt"
            if lbl.exists():
                tugas.append((f, lbl, None))
                split_per_citra[f.stem] = "pretrain"
    else:
        for sp in ("train", "val", "test"):
            for baris in (SPLIT352 / f"{sp}.txt").read_text().splitlines():
                baris = baris.strip()
                if not baris:
                    continue
                stem = Path(baris).stem
                img = D352 / "images" / f"{stem}.jpg"
                lbl = D352 / "labels" / f"{stem}.txt"
                dep = DEPTH352 / f"{stem}.png"
                if img.exists() and lbl.exists():
                    tugas.append((img, lbl, dep if dep.exists() else None))
                    split_per_citra[stem] = sp
    return tugas, split_per_citra


def main() -> int:
    global S, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", choices=["953", "352"], required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sisi", type=int, default=S,
                    help="sisi crop tersimpan; 176 default, 256 untuk detail lebih")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    S = args.sisi
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    tugas, split_per_citra = kumpulkan_tugas(args.src)
    print(f"src={args.src}  citra={len(tugas)}")

    rgbs, deps, msks, ys, stems = [], [], [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, batch in enumerate(ex.map(_kerja, tugas, chunksize=16)):
            for rgb, dep, msk, k, stem in batch:
                rgbs.append(rgb); deps.append(dep); msks.append(msk)
                ys.append(k); stems.append(stem)
            if (i + 1) % 400 == 0:
                print(f"  {i+1}/{len(tugas)} citra -> {len(ys)} crop", flush=True)

    rgb = np.stack(rgbs); dep = np.stack(deps); msk = np.stack(msks)
    y = np.array(ys, np.int64)
    stem = np.array(stems)
    split = np.array([split_per_citra[s] for s in stems])
    tree = np.array([s.rsplit("_", 1)[0] for s in stems])

    pre = OUT / f"crops{args.src}"
    np.save(f"{pre}_rgb.npy", rgb)
    np.save(f"{pre}_dep.npy", dep)
    np.save(f"{pre}_msk.npy", msk)
    np.savez(f"{pre}_meta.npz", y=y, split=split, tree=tree, stem=stem)

    ringkas = {
        "src": args.src, "n_crop": int(len(y)), "sisi": S, "ctx": CTX,
        "relief_m": RELIEF_M,
        "per_kelas": {f"B{i+1}": int((y == i).sum()) for i in range(4)},
        "per_split": {s: int((split == s).sum()) for s in sorted(set(split.tolist()))},
        "n_pohon": int(len(set(tree.tolist()))),
    }
    (OUT / f"crops{args.src}_ringkas.json").write_text(json.dumps(ringkas, indent=2))
    print(json.dumps(ringkas, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
