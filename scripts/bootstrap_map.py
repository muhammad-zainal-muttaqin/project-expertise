"""Selang kepercayaan bootstrap untuk mAP50 deteksi — koreksi metodologis Fase 6.

Fase 5 (V2-E-011) memakai bootstrap CI dan berani menyimpulkan INCONCLUSIVE.
Fase 6 tidak menghitung CI satu kali pun, lalu mengurutkan konfigurasi
berdasarkan titik estimasi (0,4102 -> 0,4395 -> 0,4500) tanpa pernah tahu
apakah selisihnya lebih besar dari deraunya. Skrip ini menutup lubang itu.

Resampling dilakukan pada tingkat CITRA (bukan kotak): citra test diambil
dengan pengembalian, lalu AP50 dihitung ulang dari nol pada sampel itu.
Untuk perbandingan antar-sumber dipakai bootstrap BERPASANGAN — sampel citra
yang sama dipakai untuk kedua sumber, sehingga korelasi antar-model tidak
menggelembungkan selang selisihnya.

Usage:
    .venv/bin/python scripts/bootstrap_map.py --split test --n-boot 2000 \
        --sumber results/pred_edge_test.npz results/pred_rgb352_test.npz \
        --nama edge_rgbd yolo_rgb --out results/bootstrap_map.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_twostage import ap50  # noqa: E402

D352 = Path("/workspace/SawitMVC-Depth")
SPLIT = D352 / "splits" / "canonical_70_15_15"
W, H = 1280, 800
K = 4
NAMA = ["B1", "B2", "B3", "B4"]


def muat_gt(split: str):
    stems = [Path(l.strip()).stem
             for l in (SPLIT / f"{split}.txt").read_text().splitlines() if l.strip()]
    gt = {}
    for s in stems:
        g = []
        for ln in (D352 / "labels" / f"{s}.txt").read_text().splitlines():
            p = ln.split()
            if len(p) < 5 or int(p[0]) < 0:
                continue
            c = int(p[0]); cx, cy, w, h = (float(x) for x in p[1:5])
            g.append([c, (cx - w / 2) * W, (cy - h / 2) * H,
                      (cx + w / 2) * W, (cy + h / 2) * H])
        gt[s] = np.array(g, float) if g else np.zeros((0, 5))
    return stems, gt


def mAP_pada(stems_sampel, gt, pred, agnostik=False):
    """AP50 pada satu sampel citra. Nama dibuat unik supaya citra yang terambil
    berkali-kali benar-benar dihitung berkali-kali.

    agnostik=True -> AP50 lokalisasi murni (1 angka), bukan rata-rata makro.
    """
    g2, p2 = {}, {}
    for i, s in enumerate(stems_sampel):
        kunci = f"{s}#{i}"
        g2[kunci] = gt[s]
        v = pred.get(s, np.zeros((0, 6)))
        p2[kunci] = (v[np.argsort(-v[:, 4])] if (agnostik and len(v)) else v)
    if agnostik:
        a = ap50(g2, p2, None)
        return float(a), [float(a)]
    per = [ap50(g2, p2, c) for c in range(K)]
    return float(np.mean(per)), per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumber", nargs="+", required=True, help="npz dump prediksi")
    ap.add_argument("--nama", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--agnostik", action="store_true",
                    help="AP50 lokalisasi murni, bukan rata-rata makro 4 kelas")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if len(args.sumber) != len(args.nama):
        print("FATAL: jumlah --sumber dan --nama harus sama"); return 1

    stems, gt = muat_gt(args.split)
    KK = 1 if args.agnostik else K
    LABEL = ["AP50_agnostik"] if args.agnostik else NAMA
    preds = []
    for j in args.sumber:
        z = np.load(j)
        preds.append({k: np.asarray(z[k], float) for k in z.files})
    n = len(stems)
    print(f"split={args.split}  {n} citra  {sum(len(g) for g in gt.values())} kotak GT")

    rng = np.random.default_rng(args.seed)
    idx = [rng.integers(0, n, n) for _ in range(args.n_boot)]

    titik, sebaran = {}, {}
    for nm, pr in zip(args.nama, preds):
        m0, per0 = mAP_pada(stems, gt, pr, args.agnostik)
        titik[nm] = {"mAP50": round(m0, 4),
                     "per_kelas": {LABEL[c]: round(float(per0[c]), 4) for c in range(KK)}}
        boot = np.empty((args.n_boot, KK + 1))
        for b, ii in enumerate(idx):
            m, per = mAP_pada([stems[i] for i in ii], gt, pr, args.agnostik)
            boot[b, 0] = m; boot[b, 1:] = per
        sebaran[nm] = boot
        lo, hi = np.percentile(boot[:, 0], [2.5, 97.5])
        titik[nm]["CI95_mAP50"] = [round(float(lo), 4), round(float(hi), 4)]
        titik[nm]["lebar_CI"] = round(float(hi - lo), 4)
        titik[nm]["CI95_per_kelas"] = {
            LABEL[c]: [round(float(x), 4) for x in np.percentile(boot[:, c + 1], [2.5, 97.5])]
            for c in range(KK)}
        print(f"  {nm:14s} {'AP50' if args.agnostik else 'mAP50'}={m0:.4f}  "
              f"CI95=[{lo:.4f}, {hi:.4f}]  lebar={hi-lo:.4f}")

    # --- selisih berpasangan -------------------------------------------------
    pasangan = {}
    for a, b in itertools.combinations(args.nama, 2):
        d = sebaran[a][:, 0] - sebaran[b][:, 0]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p_pos = float((d > 0).mean())
        pasangan[f"{a} - {b}"] = {
            "delta_titik": round(titik[a]["mAP50"] - titik[b]["mAP50"], 4),
            "CI95_delta": [round(float(lo), 4), round(float(hi), 4)],
            "P(delta>0)": round(p_pos, 3),
            "signifikan_95": bool(lo > 0 or hi < 0)}
        tanda = "SIGNIFIKAN" if (lo > 0 or hi < 0) else "tidak signifikan (CI memuat nol)"
        print(f"  {a} - {b}: delta={pasangan[f'{a} - {b}']['delta_titik']:+.4f} "
              f"CI95=[{lo:+.4f}, {hi:+.4f}]  P(>0)={p_pos:.3f}  -> {tanda}")

    hasil = {"split": args.split, "n_citra": n, "n_boot": args.n_boot,
             "seed": args.seed, "sumber": dict(zip(args.nama, args.sumber)),
             "per_sumber": titik, "selisih_berpasangan": pasangan}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(hasil, indent=2))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
