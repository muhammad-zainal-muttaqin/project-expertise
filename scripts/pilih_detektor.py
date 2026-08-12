"""Pilih kombinasi detektor terbaik untuk stage-1 — Fase 6.

Mengevaluasi tiap detektor yang tersedia (dan gabungan WBF-nya) pada split
**val**, lalu menulis kombinasi terbaik ke JSON. Pemilihan sengaja di val,
bukan test: memilih di test sama saja mengepaskan model ke angka yang mau
dilaporkan.

Usage:
    .venv/bin/python scripts/pilih_detektor.py --kandidat A.pt B.pt \
        --out results/detektor_pilihan.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_twostage import D352, SPLIT, H, W, ap50, muat_detektor, wbf  # noqa: E402


def muat_gt(stems):
    gt = {}
    for s in stems:
        g = []
        for ln in (D352 / "labels" / f"{s}.txt").read_text().splitlines():
            p = ln.split()
            if len(p) < 5 or int(p[0]) < 0:
                continue
            cx, cy, w, h = (float(x) for x in p[1:5])
            g.append([0, (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
        gt[s] = np.array(g, float) if g else np.zeros((0, 5))
    return gt


def skor(gt, kotak):
    return ap50(gt, {s: np.concatenate([v, np.zeros((len(v), 1))], 1) if len(v) else np.zeros((0, 6))
                     for s, v in kotak.items()}, None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kandidat", nargs="+", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.005)
    ap.add_argument("--wbf-iou", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ada = [k for k in args.kandidat if Path(k).exists()]
    if not ada:
        print("tidak ada kandidat detektor yang bisa dipakai")
        return 1
    print(f"kandidat tersedia: {len(ada)}")

    stems = [Path(l.strip()).stem for l in (SPLIT / f"{args.split}.txt").read_text().splitlines() if l.strip()]
    gt = muat_gt(stems)

    per_det = {}
    for j in ada:
        try:
            det = muat_detektor(j)
            pd_ = {}
            for i in range(0, len(stems), 8):
                blok = stems[i:i + 8]
                hasil = det.predict([str(D352 / "images" / f"{s}.jpg") for s in blok],
                                    imgsz=1280, conf=args.conf, iou=0.7, max_det=100,
                                    verbose=False, save=False)
                for s, r in zip(blok, hasil):
                    b = r.boxes
                    pd_[s] = (np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]], 1)
                              if len(b) else np.zeros((0, 5)))
            per_det[j] = pd_
        except Exception as e:                       # detektor rusak tidak boleh menjatuhkan seluruh langkah
            print(f"  LEWAT {j}: {e}")

    if not per_det:
        print("semua kandidat gagal di-inference")
        return 1

    hasil = {}
    for j, pd_ in per_det.items():
        hasil[j] = float(skor(gt, pd_))
        print(f"  {j:<52} AP50={hasil[j]:.4f}")

    terbaik = (max(hasil, key=hasil.get),)
    nilai_terbaik = hasil[terbaik[0]]
    kunci = list(per_det)
    for n in range(2, len(kunci) + 1):
        for kombo in itertools.combinations(kunci, n):
            gab = {s: wbf(np.concatenate([per_det[j][s] for j in kombo], 0), args.wbf_iou, n)
                   for s in stems}
            v = float(skor(gt, gab))
            hasil[" + ".join(Path(k).parts[-3] for k in kombo)] = v
            print(f"  WBF({n}) {', '.join(Path(k).parts[-3] for k in kombo):<40} AP50={v:.4f}")
            if v > nilai_terbaik:
                nilai_terbaik, terbaik = v, kombo

    keluar = {"split_pemilihan": args.split, "semua_skor": hasil,
              "terpilih": list(terbaik), "AP50_val_terpilih": nilai_terbaik,
              "wbf_iou": args.wbf_iou}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(keluar, indent=2))
    print(f"\nTERPILIH ({len(terbaik)} detektor, AP50 val {nilai_terbaik:.4f}):")
    for t in terbaik:
        print(f"  {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
