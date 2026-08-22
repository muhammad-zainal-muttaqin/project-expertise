"""Latih satu anggota bank baseline pada SawitMVC-Depth-YOLO v2.0.0.

Tiga arsitektur yang dipakai untuk matriks ini adalah YOLO26l, RT-DETR-L, dan
RF-DETR-L. Resep RGB sengaja dibuat paralel dengan V2-E-001/V2-E-003:
COCO-pretrained, 1280 px, cosine LR, dan split bawaan per pohon.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path("/workspace/SawitMVC-Depth-YOLO-RGB")
DEFAULT_PROJECT = ROOT / "runs_new763"
DATA_CONFIG = ROOT / "configs" / "new763_rgb_abs.yaml"


def train_ultralytics(args: argparse.Namespace) -> None:
    if args.arch == "yolo26l":
        from ultralytics import YOLO
        model = YOLO(args.weights or "yolo26l.pt")
    elif args.arch == "rtdetr_l":
        from ultralytics import RTDETR
        model = RTDETR(args.weights or "rtdetr-l.pt")
    else:  # pragma: no cover - dilindungi parser
        raise ValueError(args.arch)
    dataset_config = args.data / "data.yaml"
    if not dataset_config.is_file():
        dataset_config = DATA_CONFIG
    model.train(
        data=str(dataset_config),
        project=str(args.project), name=args.name, exist_ok=True,
        epochs=args.epochs, patience=args.patience, imgsz=args.imgsz,
        batch=args.batch, workers=args.workers, seed=args.seed,
        deterministic=True, cos_lr=True, optimizer="auto",
        warmup_epochs=3.0, close_mosaic=7,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        translate=0.10, scale=0.50, fliplr=0.5,
        val=True, plots=False, save=True, save_period=5,
    )


def train_rfdetr(args: argparse.Namespace) -> None:
    from rfdetr import RFDETRLarge

    out = args.project / args.name
    model_kwargs = {"resolution": args.imgsz}
    if args.weights:
        model_kwargs["pretrain_weights"] = args.weights
    model = RFDETRLarge(**model_kwargs)
    model.train(
        dataset_dir=str(args.data), output_dir=str(out),
        epochs=args.epochs, batch_size=args.batch,
        grad_accum_steps=args.grad_accum, lr=1e-4,
        lr_scheduler="cosine", lr_min_factor=0.01, warmup_epochs=1.0,
        seed=args.seed, early_stopping=True,
        early_stopping_patience=args.patience, early_stopping_min_delta=0.001,
        multi_scale=False, expanded_scales=False,
        checkpoint_interval=1, run_test=False, tensorboard=False,
        num_workers=args.workers, progress_bar="tqdm", device="cuda",
        notes={
            "dataset": "SawitMVC-Depth-YOLO-v2.0.0",
            "scope": "763 pohon; split train/valid/test bawaan",
            "seed": args.seed, "imgsz": args.imgsz,
            "recipe": "top3_rgb_baseline",
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=("yolo26l", "rtdetr_l", "rfdetr_l"), required=True)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.project.mkdir(parents=True, exist_ok=True)
    if not (args.data / "data.yaml").is_file():
        raise FileNotFoundError(args.data / "data.yaml")
    if args.arch == "rfdetr_l":
        train_rfdetr(args)
    else:
        train_ultralytics(args)
    meta = vars(args).copy()
    meta.update({"data": str(args.data), "project": str(args.project)})
    (args.project / args.name / "baseline_args.json").write_text(
        json.dumps(meta, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
