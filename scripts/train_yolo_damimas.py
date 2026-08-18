"""Fine-tune detektor YOLO26l hanya pada SawitMVC-YOLO-Damimas.

Bobot awal boleh berasal dari detektor 953 yang sudah ada, tetapi setiap batch,
augmentasi, pemilihan epoch, dan validasi pada run ini hanya memakai pohon
DAMIMAS dalam split kanonik dataset baru.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(
        ROOT / "models" / "yolo26l_e60_i1280_v2repro" / "best.pt"))
    ap.add_argument("--data", default="/workspace/SawitMVC-YOLO-Damimas/data.yaml")
    ap.add_argument("--name", default="yolo26l_damimas_ft_s42")
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr0", type=float, default=5e-4)
    ap.add_argument("--freeze", type=int, default=None,
                    help="bekukan N lapisan awal; 22 menyisakan kepala Detect")
    args = ap.parse_args()

    data = Path(args.data)
    if data.resolve() != Path("/workspace/SawitMVC-YOLO-Damimas/data.yaml").resolve():
        raise RuntimeError("Run DAMIMAS dikunci ke data.yaml dataset DAMIMAS-only")
    model = YOLO(args.weights)
    model.train(
        data=str(data), project=str(ROOT / "runs"), name=args.name,
        exist_ok=False, epochs=args.epochs, patience=args.patience,
        imgsz=args.imgsz, batch=args.batch, workers=8, seed=args.seed,
        deterministic=True, optimizer="AdamW", lr0=args.lr0, lrf=0.05,
        freeze=args.freeze,
        cos_lr=True, warmup_epochs=2.0, close_mosaic=7,
        hsv_h=0.005, hsv_s=0.15, hsv_v=0.25,
        translate=0.10, scale=0.40, fliplr=0.5,
        val=True, plots=False, save=True, save_period=5,
    )


if __name__ == "__main__":
    main()
