"""Latih RF-DETR-L pada DAMIMAS-only untuk bank detektor berlapis."""
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/workspace/SawitMVC-YOLO-Damimas-RFDETR")
    ap.add_argument("--output", default=str(ROOT / "runs" / "rfdetr_l_damimas_s42"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--resolution", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from rfdetr import RFDETRLarge

    model = RFDETRLarge(resolution=args.resolution, gradient_checkpointing=True)
    model.train(
        dataset_dir=args.dataset, output_dir=args.output,
        epochs=args.epochs, batch_size=args.batch,
        grad_accum_steps=args.grad_accum, lr=1e-4,
        seed=args.seed, early_stopping=True, early_stopping_patience=15,
        checkpoint_interval=5, run_test=False,
    )


if __name__ == "__main__":
    main()
