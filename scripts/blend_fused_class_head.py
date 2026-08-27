"""Blend a learned proposal-class head with detector WBF probabilities.

Both inputs must contain the same proposal rows in the same NPZ keys.  Geometry
and confidence remain untouched; only the B1--B4 probability columns change.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


K = 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--head-weight", type=float, required=True,
                    help="0=detector only; 1=learned head only")
    args = ap.parse_args()
    if not 0.0 <= args.head_weight <= 1.0:
        ap.error("--head-weight harus berada di [0,1]")
    with np.load(args.baseline) as b, np.load(args.head) as h:
        if set(b.files) != set(h.files):
            raise ValueError("key NPZ baseline dan head berbeda")
        arrays = {}
        for key in b.files:
            base = np.asarray(b[key], np.float32).copy()
            learned = np.asarray(h[key], np.float32)
            if base.shape != learned.shape or base.shape[1] < 5 + K:
                raise ValueError(f"shape berbeda pada {key}: {base.shape} vs {learned.shape}")
            p = ((1.0 - args.head_weight) * base[:, 5:5 + K] +
                 args.head_weight * learned[:, 5:5 + K])
            p = np.maximum(p, 0.)
            p /= np.maximum(p.sum(1, keepdims=True), 1e-9)
            base[:, 5:5 + K] = p
            arrays[key] = base
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"-> {args.output} (head_weight={args.head_weight})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
