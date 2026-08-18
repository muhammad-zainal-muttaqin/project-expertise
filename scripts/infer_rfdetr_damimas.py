"""Dump prediksi RF-DETR DAMIMAS ke format NPZ evaluator bersama."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DS = Path("/workspace/SawitMVC-YOLO-Damimas")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", nargs="+", default=["val", "test"])
    ap.add_argument("--resolution", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=.001)
    args = ap.parse_args()

    # Import ini memang lambat pada filesystem jaringan; ditunda sampai argumen
    # tervalidasi agar kesalahan CLI tidak membuang waktu satu menit.
    from rfdetr import RFDETRLarge
    model = RFDETRLarge(pretrain_weights=args.weights, resolution=args.resolution)

    for split in args.split:
        paths = sorted((DS / "images" / split).glob("*.jpg"))
        out: dict[str, np.ndarray] = {}
        t0 = time.time()
        for awal in range(0, len(paths), args.batch):
            blok = paths[awal:awal + args.batch]
            hasil = model.predict([str(p) for p in blok], threshold=args.threshold)
            if not isinstance(hasil, list):
                hasil = [hasil]
            if len(hasil) != len(blok):
                raise RuntimeError(
                    f"RF-DETR mengembalikan {len(hasil)} hasil untuk {len(blok)} citra")
            for p, d in zip(blok, hasil):
                if len(d.xyxy):
                    out[p.stem] = np.c_[
                        np.asarray(d.xyxy, np.float32),
                        np.asarray(d.confidence, np.float32),
                        np.asarray(d.class_id, np.float32),
                    ].astype(np.float32)
                else:
                    out[p.stem] = np.zeros((0, 6), np.float32)
            n = min(awal + args.batch, len(paths))
            if n % 100 < args.batch or n == len(paths):
                print(f"{split}: {n}/{len(paths)} ({time.time()-t0:.0f}s)", flush=True)
        tujuan = ROOT / "results" / f"pred_{args.tag}_{split}.npz"
        np.savez_compressed(tujuan, **out)
        print(f"-> {tujuan}")


if __name__ == "__main__":
    main()
