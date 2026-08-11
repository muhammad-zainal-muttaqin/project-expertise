"""Probe diagnostik: apa sebenarnya yang dibawa kanal depth? — Fase 6.

Skrip ini merapikan lima diagnostik yang dipakai untuk membongkar kenapa RGB+D
tidak menaikkan mAP. Semuanya read-only (tidak melatih apa pun) dan selesai
dalam hitungan menit. Jalankan ulang untuk memverifikasi setiap angka yang
dikutip di `docs/DIAGNOSIS-DEPTH.md`.

    .venv/bin/python scripts/probe_depth_signal.py --probe semua

Probe:
  1 distribusi   - bandingkan komposisi kelas 953 vs 352 (kenapa mAP-nya beda)
  2 cakupan      - berapa persen depth valid DI DALAM box (bukan di seluruh citra)
  3 relief       - relief lokal per kelas + uji Kruskal-Wallis
  4 kuantisasi   - besar 1 level uint8 dalam cm vs amplitudo sinyal
  5 pooling      - AUC B1-vs-B4 sebagai fungsi jumlah piksel yang di-pool
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from scipy import stats

D953 = Path("/workspace/SawitMVC-YOLO")
D352 = Path("/workspace/SawitMVC-Depth")
DEPTH = Path("/workspace/depth_png_352")
SPLIT = D352 / "splits" / "canonical_70_15_15"
W, H = 1280, 800
ZN, ZF = 0.8, 15.0
A = 1.0 / ZN - 1.0 / ZF


def dekode(v):
    inv = (v.astype(np.float32) - 1.0) / 254.0
    return 1.0 / (inv * A + 1.0 / ZF)


def step_cm(Z):
    """Besar satu level uint8 dalam cm pada jarak Z (turunan dari encoding)."""
    return Z ** 2 * (A / 254.0) * 100


def probe_distribusi():
    print("\n=== 1. Distribusi kelas: 953 vs 352 ===")
    import glob
    for tag, berkas in [
        ("953", {sp: sorted(glob.glob(str(D953 / f"labels/{sp}/*.txt"))) for sp in ("train", "val", "test")}),
        ("352", {sp: [str(D352 / "labels" / f"{Path(l).stem}.txt")
                      for l in (SPLIT / f"{sp}.txt").read_text().split() if l.strip()]
                 for sp in ("train", "val", "test")}),
    ]:
        for sp, fs in berkas.items():
            per = np.zeros(4, int); n = 0
            for f in fs:
                if not os.path.exists(f):
                    continue
                n += 1
                for ln in open(f):
                    p = ln.split()
                    if len(p) >= 5 and int(p[0]) >= 0:
                        per[int(p[0])] += 1
            tot = max(per.sum(), 1)
            print(f"  {tag}-{sp:<5s} citra={n:5d} instance={per.sum():6d} /citra={per.sum()/max(n,1):.2f}  "
                  + " ".join(f"B{i+1}={per[i]:5d}({100*per[i]/tot:4.1f}%)" for i in range(4)))


def kumpul_box():
    """Untuk tiap box GT: kelas, ukuran piksel, Z dalam box, Z cincin, cakupan valid."""
    baris = []
    for sp in ("train", "val", "test"):
        for l in (SPLIT / f"{sp}.txt").read_text().split():
            stem = Path(l.strip()).stem
            lp, dp = D352 / "labels" / f"{stem}.txt", DEPTH / f"{stem}.png"
            if not (lp.exists() and dp.exists()):
                continue
            d = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
            if d is None:
                continue
            Z = dekode(d); Z[d == 0] = np.nan
            for ln in lp.read_text().splitlines():
                p = ln.split()
                if len(p) < 5 or int(p[0]) < 0:
                    continue
                k = int(p[0]); cx, cy, w, h = (float(x) for x in p[1:5])
                x0, x1 = int((cx - w / 2) * W), int((cx + w / 2) * W)
                y0, y1 = int((cy - h / 2) * H), int((cy + h / 2) * H)
                m = int(0.5 * max(x1 - x0, y1 - y0))
                X0, X1 = max(0, x0 - m), min(W, x1 + m)
                Y0, Y1 = max(0, y0 - m), min(H, y1 + m)
                x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
                if x1 <= x0 or y1 <= y0:
                    continue
                din = Z[y0:y1, x0:x1]
                cincin = Z[Y0:Y1, X0:X1].copy()
                cincin[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = np.nan
                v_in = din[np.isfinite(din)]; v_ring = cincin[np.isfinite(cincin)]
                if v_in.size < 20 or v_ring.size < 20:
                    continue
                baris.append((sp, k, np.sqrt((x1 - x0) * (y1 - y0)),
                              float(np.median(v_in)), float(np.median(v_ring)),
                              v_in.size / din.size, v_in, v_ring))
    return baris


def probe_cakupan_relief_kuantisasi(baris):
    k = np.array([b[1] for b in baris])
    z_in = np.array([b[3] for b in baris]); z_ring = np.array([b[4] for b in baris])
    frac = np.array([b[5] for b in baris])

    print("\n=== 2. Cakupan depth DI DALAM box (bukan seluruh citra) ===")
    print(f"  median fraksi piksel valid di dalam box: {100*np.median(frac):.1f}%")
    print("  -> 29% invalid yang selama ini dikutip itu LATAR, bukan objek.")

    print("\n=== 3. Relief lokal (Z cincin - Z box) per kelas ===")
    rel = z_ring - z_in
    for c in range(4):
        m = k == c
        print(f"  B{c+1} n={m.sum():5d}  relief median={100*np.median(rel[m]):+6.1f} cm   "
              f"lebih dekat dari sekitar={100*np.mean(rel[m] > 0):5.1f}%   Z median={np.median(z_in[m]):.2f} m")
    g = [rel[k == c] for c in range(4)]
    hasil = stats.kruskal(*g)
    print(f"  Kruskal-Wallis 4 kelas: H={hasil.statistic:.1f}  p={hasil.pvalue:.2e}")

    print("\n=== 4. Kuantisasi encoding lama vs amplitudo sinyal ===")
    print(f"  {'Z (m)':>7} {'1 level uint8':>15}")
    for Z in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        print(f"  {Z:7.1f} {step_cm(Z):12.2f} cm")
    s = step_cm(2.5)
    print(f"  Sinyal relief median |{100*np.median(np.abs(rel)):.1f}| cm = {np.median(np.abs(rel))*100/s:.2f} level @Z=2,5 m")
    print("  -> sinyal yang dicari lebih kecil dari satu langkah kuantisasi.")


def probe_pooling(baris):
    print("\n=== 5. AUC B1-vs-B4 sebagai fungsi pooling ===")
    rng = np.random.RandomState(0)

    def auc(a, b):
        if len(a) < 3 or len(b) < 3:
            return float("nan")
        return stats.mannwhitneyu(a, b, alternative="two-sided").statistic / (len(a) * len(b))

    print(f"  {'piksel di-pool':>15} {'AUC train+val':>15} {'AUC test':>10}")
    for n in (1, 16, 256, 4096):
        skor = {"tv": ([], []), "te": ([], [])}
        for sp, kk, _, _, _, _, v_in, v_ring in baris:
            if kk not in (0, 3) or v_in.size < n or v_ring.size < n:
                continue
            d = (np.median(v_ring[rng.choice(v_ring.size, n, False)])
                 - np.median(v_in[rng.choice(v_in.size, n, False)]))
            kunci = "te" if sp == "test" else "tv"
            skor[kunci][0 if kk == 0 else 1].append(d)
        print(f"  {n:15d} {auc(*skor['tv']):15.3f} {auc(*skor['te']):10.3f}")
    print("  -> sinyal nyaris tak terbaca per-piksel, baru muncul setelah pooling wilayah.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="semua",
                    choices=["semua", "distribusi", "depth"])
    args = ap.parse_args()

    if args.probe in ("semua", "distribusi"):
        probe_distribusi()
    if args.probe in ("semua", "depth"):
        baris = kumpul_box()
        print(f"\nbox GT dengan depth terpakai: {len(baris)}")
        probe_cakupan_relief_kuantisasi(baris)
        probe_pooling(baris)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
