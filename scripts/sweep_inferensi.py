"""Cari konfigurasi inference terbaik untuk stage-1 — Fase 6 (eksplorasi 0,80).

Bobot yang sama bisa memberi AP50 berbeda tergantung resolusi inference dan
ambang NMS. Ini gratis (tanpa training), tapi biasanya menyumbang 1-3 poin.
Disweep di **val**; test tidak pernah disentuh untuk pemilihan.

Usage:
    .venv/bin/python scripts/sweep_inferensi.py --detektor A.pt [B.pt ...] \
        --out results/sweep_inferensi.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_twostage import D352, SPLIT, H, W, ap50, muat_detektor, wbf  # noqa: E402
from pilih_detektor import muat_gt, skor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detektor", nargs="+", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.005)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[1280, 1536])
    ap.add_argument("--iou", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--wbf-iou", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ada = [d for d in args.detektor if Path(d).exists()]
    if not ada:
        print("tidak ada detektor")
        return 1

    stems = [Path(l.strip()).stem for l in (SPLIT / f"{args.split}.txt").read_text().splitlines() if l.strip()]
    gt = muat_gt(stems)
    jalur_citra = [str(D352 / "images" / f"{s}.jpg") for s in stems]

    model = {d: muat_detektor(d) for d in ada}
    hasil, terbaik = {}, None

    for sz in args.imgsz:
        for iou in args.iou:
            try:
                per_det = {}
                for d, m in model.items():
                    pd_ = {}
                    for i in range(0, len(stems), 8):
                        r = m.predict(jalur_citra[i:i + 8], imgsz=sz, conf=args.conf,
                                      iou=iou, max_det=100, verbose=False, save=False)
                        for s, rr in zip(stems[i:i + 8], r):
                            b = rr.boxes
                            pd_[s] = (np.concatenate([b.xyxy.cpu().numpy(),
                                                      b.conf.cpu().numpy()[:, None]], 1)
                                      if len(b) else np.zeros((0, 5)))
                    per_det[d] = pd_
                if len(ada) > 1:
                    kotak = {s: wbf(np.concatenate([per_det[d][s] for d in ada], 0),
                                    args.wbf_iou, len(ada)) for s in stems}
                else:
                    kotak = per_det[ada[0]]
                v = float(skor(gt, kotak))
            except Exception as e:                    # satu kombinasi gagal tidak boleh
                print(f"  imgsz={sz} iou={iou}: GAGAL {str(e)[:80]}")   # menjatuhkan sweep
                continue
            hasil[f"imgsz{sz}_iou{iou}"] = v
            tanda = ""
            if terbaik is None or v > terbaik[2]:
                terbaik, tanda = (sz, iou, v), "  *"
            print(f"  imgsz={sz} iou={iou}  AP50={v:.4f}{tanda}", flush=True)

    if terbaik is None:
        print("semua kombinasi gagal")
        return 1
    keluar = {"detektor": ada, "split_sweep": args.split, "semua": hasil,
              "terbaik": {"imgsz": terbaik[0], "iou": terbaik[1], "AP50_val": terbaik[2]}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(keluar, indent=2))
    print(f"\nTERBAIK: imgsz={terbaik[0]} iou={terbaik[1]} AP50_val={terbaik[2]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
