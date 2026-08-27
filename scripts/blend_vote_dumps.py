"""Blend detector WBF probabilities with an optional crop-classifier vote."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", type=Path, required=True)
    ap.add_argument("--classifier", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--classifier-weight", type=float, required=True,
                    help="0=detector only, 1=classifier only")
    args = ap.parse_args()
    w = float(args.classifier_weight)
    if not 0.0 <= w <= 1.0:
        raise ValueError("classifier-weight harus berada pada [0,1]")
    with np.load(args.detector) as zd, np.load(args.classifier) as zc:
        out = {}
        for stem in zd.files:
            d = np.asarray(zd[stem], float).copy()
            c = np.asarray(zc[stem], float)
            if d.shape != c.shape:
                raise ValueError(f"shape berbeda untuk {stem}: {d.shape} vs {c.shape}")
            p = (1.0 - w) * d[:, 5:9] + w * c[:, 5:9]
            p = np.maximum(p, 0.)
            p /= np.maximum(p.sum(1, keepdims=True), 1e-9)
            d[:, 5:9] = p
            out[stem] = d
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)
    print(f"wrote {args.output} rows={sum(len(v) for v in out.values())} weight={w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
