"""Latih RF-DETR-L pada DAMIMAS-only untuk bank detektor berlapis."""
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/workspace/SawitMVC-YOLO-Damimas-RFDETR")
    ap.add_argument("--output", default=str(
        ROOT / "pipeline-pertandan" / "runs" / "rfdetr_l_damimas_s42"))
    ap.add_argument("--pretrain", default=
                    "/root/.roboflow/models/rf-detr-large-2026.pth")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--resolution", type=int, default=1280)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from rfdetr import RFDETRLarge

    model = RFDETRLarge(pretrain_weights=args.pretrain,
                        resolution=args.resolution, gradient_checkpointing=True)
    model.train(
        dataset_dir=args.dataset, output_dir=args.output,
        epochs=args.epochs, batch_size=args.batch,
        grad_accum_steps=args.grad_accum, lr=1e-4,
        lr_scheduler="cosine", lr_min_factor=.01, warmup_epochs=1.,
        seed=args.seed, early_stopping=False,
        multi_scale=False, expanded_scales=False,
        checkpoint_interval=5, run_test=False, tensorboard=False,
        num_workers=args.workers, progress_bar="tqdm", device="cuda",
        notes={"dataset": "SawitMVC-YOLO-Damimas",
               "scope": "641 train / 86 val; test tidak dijalankan saat training",
               "seed": args.seed, "tujuan": "bank detektor greedy DAMIMAS",
               "pretrain": str(Path(args.pretrain).resolve())},
    )


if __name__ == "__main__":
    main()
