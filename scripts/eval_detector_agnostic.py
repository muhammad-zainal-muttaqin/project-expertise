"""Ukur AP50 lokalisasi murni (class-agnostic) sebuah detektor — Fase 6.

Plafon mAP50 pipeline dua-tahap = AP50 lokalisasi. Skrip ini mengukurnya
langsung, terpisah dari klasifikasi, supaya dorongan stage-1 bisa dinilai tanpa
harus menjalankan seluruh pipeline.

Mendukung penggabungan beberapa detektor (WBF sederhana: kotak yang saling
tumpang tindih di atas ambang IoU digabung dengan rata-rata berbobot skor).

Usage:
    .venv/bin/python scripts/eval_detector_agnostic.py \
        --detektor runs/agn352_ft/weights/best.pt --split test --tta
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_twostage import D352, SPLIT, H, W, ap50, wbf  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detektor", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.005)
    ap.add_argument("--tta", action="store_true", help="ultralytics augment=True")
    ap.add_argument("--wbf-iou", type=float, default=0.6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stems = [Path(l.strip()).stem for l in (SPLIT / f"{args.split}.txt").read_text().splitlines() if l.strip()]
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

    from ultralytics import YOLO
    per_det = {}
    for jalur in args.detektor:
        det = YOLO(jalur)
        pred = {}
        for i in range(0, len(stems), 8):
            blok = stems[i:i + 8]
            hasil = det.predict([str(D352 / "images" / f"{s}.jpg") for s in blok],
                                imgsz=1280, conf=args.conf, iou=0.7, max_det=100,
                                augment=args.tta, verbose=False, save=False)
            for s, r in zip(blok, hasil):
                b = r.boxes
                pred[s] = (np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]], 1)
                           if len(b) else np.zeros((0, 5)))
        per_det[jalur] = pred
        satu = {s: np.concatenate([v, np.zeros((len(v), 1))], 1) for s, v in pred.items()}
        print(f"  {Path(jalur).parts[-3]:<28} AP50_agnostic = {ap50(gt, satu, None):.4f}")

    if len(args.detektor) > 1:
        gab = {}
        for s in stems:
            semua = np.concatenate([per_det[j][s] for j in args.detektor], 0)
            f = wbf(semua, args.wbf_iou, len(args.detektor))
            gab[s] = np.concatenate([f, np.zeros((len(f), 1))], 1) if len(f) else np.zeros((0, 6))
        skor = ap50(gt, gab, None)
        print(f"  {'WBF gabungan':<28} AP50_agnostic = {skor:.4f}")
    else:
        skor = ap50(gt, {s: np.concatenate([v, np.zeros((len(v), 1))], 1)
                         for s, v in per_det[args.detektor[0]].items()}, None)

    if args.out:
        Path(args.out).write_text(json.dumps({
            "detektor": args.detektor, "split": args.split, "tta": bool(args.tta),
            "AP50_class_agnostic": float(skor)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
