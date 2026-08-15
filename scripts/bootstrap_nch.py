"""Bootstrap CI berpasangan untuk selisih mAP50 antar-sel matriks monocular.

Kenapa bukan `bootstrap_map.py`: skrip itu meng-hardcode dataset 352
(`/workspace/SawitMVC-Depth`, split kanonik, 1280x800). Dipakai untuk 953 ia
akan memuat GT yang salah tanpa error. Skrip itu juga sudah menghasilkan angka
yang dikutip di laporan, jadi tidak boleh diubah. Ini versi yang menerima
dataset dan tata letak apa pun; ukuran citra dibaca per berkas, bukan
diasumsikan (352 = 1280x800, 953 = 960x1280).

Fungsi AP-nya sendiri diimpor dari `eval_twostage.ap50` — sama persis dengan
yang dipakai `bootstrap_map.py`, supaya angkanya sebanding.

Resampling di tingkat CITRA, berpasangan (sampel yang sama untuk kedua model),
seed tetap. Yang dilaporkan: titik estimasi, CI 95% tiap model, dan CI selisih
berpasangan + P(delta>0).

Usage:
    .venv/bin/python scripts/bootstrap_nch.py \
        --gt-root /workspace/SawitMVC-YOLO --tata-letak images_split --split test \
        --sumber results/pred_sel6_953_rgbmono_test.npz results/pred_sel5_953_rgb_test.npz \
        --nama sel6_mono sel5_rgb --out results/boot_sel6_vs_sel5.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_twostage import ap50  # noqa: E402  (satu sumber kebenaran, sama dengan bootstrap_map.py)

KELAS = ["B1", "B2", "B3", "B4"]
EKS = (".jpg", ".jpeg", ".png", ".tiff", ".tif")


def dir_split(akar: Path, split: str, tata: str) -> tuple[Path, Path]:
    if tata == "images_split":
        return akar / "images" / split, akar / "labels" / split
    return akar / split / "images", akar / split / "labels"


def muat_gt(akar: Path, split: str, tata: str):
    idir, ldir = dir_split(akar, split, tata)
    if not idir.is_dir():
        sys.exit(f"FATAL: {idir} tidak ada")
    stems, gt = [], {}
    for p in sorted(q for q in idir.iterdir() if q.suffix.lower() in EKS):
        w, h = Image.open(p).size  # per citra — jangan diasumsikan
        g = []
        lf = ldir / f"{p.stem}.txt"
        if lf.is_file():
            for ln in lf.read_text().splitlines():
                q = ln.split()
                if len(q) < 5 or int(float(q[0])) < 0:
                    continue
                c = int(float(q[0]))
                cx, cy, bw, bh = (float(x) for x in q[1:5])
                g.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h])
        stems.append(p.stem)
        gt[p.stem] = np.array(g, float) if g else np.zeros((0, 5))
    return stems, gt


def mAP_pada(sampel, gt, pred, agnostik=False):
    """mAP50 pada satu sampel citra. Kunci dibuat unik supaya citra yang
    terambil berkali-kali benar-benar dihitung berkali-kali."""
    g2, p2 = {}, {}
    for i, s in enumerate(sampel):
        k = f"{s}#{i}"
        g2[k] = gt[s]
        v = pred.get(s, np.zeros((0, 6)))
        p2[k] = v[np.argsort(-v[:, 4])] if (agnostik and len(v)) else v
    if agnostik:
        a = float(ap50(g2, p2, None))
        return a, [a]
    per = [float(ap50(g2, p2, c)) for c in range(len(KELAS))]
    return float(np.mean(per)), per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--tata-letak", choices=["images_split", "split_images"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--sumber", nargs=2, required=True, help="dua npz: [uji, pembanding]")
    ap.add_argument("--nama", nargs=2, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--agnostik", action="store_true",
                    help="AP50 lokalisasi dari prediksi class-aware yang DILIPAT — "
                         "sah sebagai pembacaan berpasangan, TAPI bukan detektor agnostik")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stems, gt = muat_gt(Path(args.gt_root), args.split, args.tata_letak)
    preds = []
    for j in args.sumber:
        if not Path(j).exists():
            sys.exit(f"FATAL: {j} tidak ada")
        z = np.load(j)
        preds.append({k: np.asarray(z[k], float) for k in z.files})
        hilang = [s for s in stems if s not in preds[-1]]
        if len(hilang) > len(stems) * 0.02:
            sys.exit(f"FATAL: {j} kehilangan {len(hilang)}/{len(stems)} citra — "
                     "dump tidak cocok dengan split ini")

    n = len(stems)
    n_gt = sum(len(g) for g in gt.values())
    print(f"split={args.split}  {n} citra  {n_gt} kotak GT")

    rng = np.random.default_rng(args.seed)
    idx = [rng.integers(0, n, n) for _ in range(args.n_boot)]
    LABEL = ["AP50_dilipat"] if args.agnostik else KELAS

    titik, sebaran = {}, {}
    for nm, pr in zip(args.nama, preds):
        m0, per0 = mAP_pada(stems, gt, pr, args.agnostik)
        titik[nm] = {"mAP50": round(m0, 4),
                     "per_kelas": {LABEL[c]: round(per0[c], 4) for c in range(len(LABEL))}}
        boot = np.empty(args.n_boot)
        for b, ii in enumerate(idx):
            boot[b] = mAP_pada([stems[i] for i in ii], gt, pr, args.agnostik)[0]
        sebaran[nm] = boot
        lo, hi = np.percentile(boot, [2.5, 97.5])
        titik[nm]["CI95"] = [round(float(lo), 4), round(float(hi), 4)]
        titik[nm]["lebar_CI"] = round(float(hi - lo), 4)
        print(f"  {nm}: mAP50={m0:.4f} CI[{lo:.4f}; {hi:.4f}] lebar={hi-lo:.4f}")

    a, b = args.nama
    d = sebaran[a] - sebaran[b]  # berpasangan: sampel citra yang sama
    lo, hi = np.percentile(d, [2.5, 97.5])
    p_pos = float((d > 0).mean())
    selisih = {"a_minus_b": f"{a} - {b}",
               "titik": round(titik[a]["mAP50"] - titik[b]["mAP50"], 4),
               "CI95": [round(float(lo), 4), round(float(hi), 4)],
               "P_delta_positif": round(p_pos, 4),
               "signifikan_95": bool(lo > 0 or hi < 0)}
    print(f"  selisih {a}-{b}: {selisih['titik']:+.4f} "
          f"CI[{lo:+.4f}; {hi:+.4f}] P(d>0)={p_pos:.3f} "
          f"signifikan={selisih['signifikan_95']}")

    hasil = {"gt_root": args.gt_root, "split": args.split, "n_citra": n,
             "n_kotak_gt": n_gt, "n_boot": args.n_boot, "seed": args.seed,
             "agnostik_dilipat": args.agnostik, "model": titik, "selisih": selisih}
    Path(args.out).write_text(json.dumps(hasil, indent=2))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
