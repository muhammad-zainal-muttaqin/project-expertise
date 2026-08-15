"""Seberapa sepakat monocular-depth dengan sensor Orbbec? (read-only, tanpa training)

Dijalankan SEBELUM melatih apa pun. Alasannya: seluruh nilai sel `953 RGB+Mono`
bertumpu pada asumsi bahwa peta monocular membawa struktur kedalaman yang nyata
— tapi 953 tidak punya sensor untuk mengujinya. Satu-satunya tempat asumsi itu
bisa diuji adalah 352, di mana kedua peta ada untuk citra yang sama persis.

Tiga hal yang diukur, dari yang paling longgar ke yang paling mengikat:

1. KORELASI GLOBAL (Spearman, hanya piksel sensor valid). Menjawab "apakah
   urutan dekat-jauh sama?" Kebal terhadap kesalahan skala absolut.
2. KORELASI DI DALAM KOTAK GT. Lebih ketat: kesepakatan pada tandan, bukan pada
   latar. Latar (langit, tanah) mudah ditebak model mana pun dan bisa
   menggelembungkan angka global.
3. RELIEF ORDINAL PER KELAS. Uji paling mengikat. V2-E-014 menemukan sinyal
   kematangan pada sensor berupa relief lokal yang monoton: B1 +2,8 cm turun ke
   B4 -5,1 cm (Kruskal-Wallis p=1,7e-21). Kalau monocular hanya menghasilkan
   permukaan halus yang masuk akal, ia TIDAK akan mereproduksi urutan itu.
   Kalau ia mereproduksinya, peta itu membawa struktur skala-tandan, bukan cuma
   gradien kedalaman kasar.

Relief dihitung sama seperti V2-E-014: median depth di dalam kotak dikurangi
median depth pada cincin latar di sekitarnya (positif = menonjol ke kamera).

Usage:
    .venv/bin/python scripts/probe_mono_vs_sensor.py --n 200
    .venv/bin/python scripts/probe_mono_vs_sensor.py --n 0 --out results/probe_mono_vs_sensor.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import stats

D352 = Path("/workspace/SawitMVC-Depth")
SENSOR = Path("/workspace/depth_png_352")
MONO = Path("/workspace/mono_png_352")
Z_NEAR, Z_FAR = 0.8, 15.0
W, H = 1280, 800
NAMA = ["B1", "B2", "B3", "B4"]


def ke_meter(png: np.ndarray) -> np.ndarray:
    """Balikkan encode_inverse: uint8 1..255 -> meter. 0 (invalid) -> NaN."""
    v = png.astype(np.float32)
    inv = (v - 1.0) / 254.0
    satu_per_z = inv * (1.0 / Z_NEAR - 1.0 / Z_FAR) + 1.0 / Z_FAR
    z = 1.0 / np.maximum(satu_per_z, 1e-9)
    z[png == 0] = np.nan
    return z


def kotak_gt(stem: str):
    fp = D352 / "labels" / f"{stem}.txt"
    out = []
    if not fp.exists():
        return out
    for ln in fp.read_text().splitlines():
        p = ln.split()
        if len(p) < 5 or int(p[0]) < 0:
            continue
        c = int(p[0])
        cx, cy, w, h = (float(x) for x in p[1:5])
        x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
        x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
        out.append((c, max(0, x1), max(0, y1), min(W, x2), min(H, y2)))
    return out


def relief(z: np.ndarray, b) -> float:
    """Median di dalam kotak - median cincin latar. Positif = menonjol."""
    _, x1, y1, x2, y2 = b
    if x2 - x1 < 4 or y2 - y1 < 4:
        return np.nan
    dalam = z[y1:y2, x1:x2]
    m = max(4, int(0.5 * min(x2 - x1, y2 - y1)))
    X1, Y1 = max(0, x1 - m), max(0, y1 - m)
    X2, Y2 = min(W, x2 + m), min(H, y2 + m)
    luas = z[Y1:Y2, X1:X2].copy()
    luas[y1 - Y1: y2 - Y1, x1 - X1: x2 - X1] = np.nan
    a, b_ = dalam[np.isfinite(dalam)], luas[np.isfinite(luas)]
    if a.size < 20 or b_.size < 20:
        return np.nan
    return float(np.median(b_) - np.median(a))  # latar - objek; + = objek lebih dekat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="jumlah citra (0 = semua)")
    ap.add_argument("--sub", type=int, default=20000, help="piksel disubsampel per citra")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for d in (SENSOR, MONO):
        if not d.is_dir():
            sys.exit(f"FATAL: {d} tidak ada — jalankan buat_mono_depth.py dulu")

    stems = sorted(p.stem for p in SENSOR.glob("*.png"))
    stems = [s for s in stems if (MONO / f"{s}.png").exists()]
    if not stems:
        sys.exit("FATAL: tidak ada stem yang punya kedua peta")
    rng = np.random.default_rng(args.seed)
    if args.n and args.n < len(stems):
        stems = [stems[i] for i in sorted(rng.choice(len(stems), args.n, replace=False))]
    print(f"{len(stems)} citra dengan sensor + mono")

    rho_glob, rho_kotak, skala = [], [], []
    rel = {k: {"sensor": [], "mono": []} for k in range(4)}

    for i, s in enumerate(stems, 1):
        zs = ke_meter(cv2.imread(str(SENSOR / f"{s}.png"), cv2.IMREAD_UNCHANGED))
        zm = ke_meter(cv2.imread(str(MONO / f"{s}.png"), cv2.IMREAD_UNCHANGED))
        if zs.shape != zm.shape:
            zm = cv2.resize(zm, (zs.shape[1], zs.shape[0]), interpolation=cv2.INTER_NEAREST)

        v = np.isfinite(zs) & np.isfinite(zm)
        if v.sum() > 100:
            idx = np.flatnonzero(v.ravel())
            if idx.size > args.sub:
                idx = rng.choice(idx, args.sub, replace=False)
            a, b = zs.ravel()[idx], zm.ravel()[idx]
            rho_glob.append(stats.spearmanr(a, b)[0])  # [0], bukan .statistic — kompatibel scipy lama
            skala.append(float(np.median(b) / max(np.median(a), 1e-6)))

        for kb in kotak_gt(s):
            c, x1, y1, x2, y2 = kb
            sa, sb = zs[y1:y2, x1:x2], zm[y1:y2, x1:x2]
            vv = np.isfinite(sa) & np.isfinite(sb)
            if vv.sum() > 50:
                r = stats.spearmanr(sa[vv], sb[vv])[0]
                if np.isfinite(r):
                    rho_kotak.append(r)
            for nm, z in (("sensor", zs), ("mono", zm)):
                rr = relief(z, kb)
                if np.isfinite(rr):
                    rel[c][nm].append(rr)
        if i % 100 == 0:
            print(f"  {i}/{len(stems)}")

    hasil = {
        "n_citra": len(stems),
        "spearman_global": {"median": round(float(np.median(rho_glob)), 4),
                            "p25": round(float(np.percentile(rho_glob, 25)), 4),
                            "p75": round(float(np.percentile(rho_glob, 75)), 4),
                            "n": len(rho_glob)},
        "spearman_dalam_kotak": {"median": round(float(np.median(rho_kotak)), 4),
                                 "p25": round(float(np.percentile(rho_kotak, 25)), 4),
                                 "p75": round(float(np.percentile(rho_kotak, 75)), 4),
                                 "n": len(rho_kotak)},
        "rasio_skala_mono_per_sensor": {"median": round(float(np.median(skala)), 4)},
        "relief_cm": {}, "kruskal": {},
    }
    for nm in ("sensor", "mono"):
        per = [np.array(rel[c][nm]) * 100.0 for c in range(4)]
        hasil["relief_cm"][nm] = {NAMA[c]: {"median": round(float(np.median(per[c])), 2),
                                            "n": int(per[c].size)}
                                  for c in range(4) if per[c].size}
        ada = [p for p in per if p.size > 5]
        if len(ada) >= 2:
            k = stats.kruskal(*ada)
            hasil["kruskal"][nm] = {"H": round(float(k[0]), 2), "p": float(k[1])}
            # urutan dihitung dari kelompok yang SAMA dengan yang diuji Kruskal
            urut = [float(np.median(p)) for p in ada]
            hasil["kruskal"][nm]["monoton_turun_B1_ke_B4"] = bool(
                all(urut[i] >= urut[i + 1] for i in range(len(urut) - 1)))

    print(json.dumps(hasil, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(hasil, indent=2))
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
