"""Run a small, explicit RF-DETR test-time augmentation bank.

Only photometric transforms are used, so boxes stay in the original image
coordinate system and can be fused directly with the existing predictions.
The purpose is localization: class IDs are retained only for compatibility
with the raw-prediction format and are ignored by the agnostic WBF evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def transform(image: np.ndarray, name: str) -> np.ndarray:
    """RGB uint8 -> RGB uint8, same HxW."""
    if name == "identity":
        return image
    if name == "hflip":
        return np.ascontiguousarray(image[:, ::-1])
    if name == "vflip":
        return np.ascontiguousarray(image[::-1, :])
    if name == "rot180":
        return np.ascontiguousarray(image[::-1, ::-1])
    if name == "clahe":
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    if name == "unsharp":
        blur = cv2.GaussianBlur(image, (0, 0), 1.2)
        return np.clip(image.astype(np.float32) * 1.35 -
                       blur.astype(np.float32) * .35, 0, 255).astype(np.uint8)
    if name == "gamma095":
        lut = np.array([((i / 255.) ** .95) * 255 for i in range(256)],
                       dtype=np.uint8)
        return cv2.LUT(image, lut)
    if name == "gamma105":
        lut = np.array([((i / 255.) ** 1.05) * 255 for i in range(256)],
                       dtype=np.uint8)
        return cv2.LUT(image, lut)
    if name == "hue2":
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + 2) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    raise ValueError(name)


def image_paths(root: Path, split: str) -> list[Path]:
    if split == "val" and (root / "valid" / "images").is_dir():
        folder = root / "valid" / "images"
    elif (root / split / "images").is_dir():
        folder = root / split / "images"
    else:
        folder = root / "images" / split
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})


def run_variant(model, paths: list[Path], variant: str, batch: int) -> dict[str, np.ndarray]:
    out = {}
    for start in range(0, len(paths), batch):
        chunk = paths[start:start + batch]
        images = []
        for path in chunk:
            image = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR),
                                 cv2.COLOR_BGR2RGB)
            images.append(transform(image, variant))
        results = model.predict(images, threshold=.001,
                                include_source_image=False)
        if not isinstance(results, list):
            results = [results]
        for path, image, det in zip(chunk, images, results):
            rows = []
            height, width = image.shape[:2]
            for xyxy, score, klass in zip(det.xyxy, det.confidence, det.class_id):
                x1, y1, x2, y2 = map(float, xyxy)
                # RF-DETR predicts in the transformed image frame.  Bring
                # boxes back before storing them so the variant can be WBF'd
                # with the original source without changing geometry.
                if variant == "hflip":
                    x1, x2 = width - x2, width - x1
                elif variant == "vflip":
                    y1, y2 = height - y2, height - y1
                elif variant == "rot180":
                    x1, x2 = width - x2, width - x1
                    y1, y2 = height - y2, height - y1
                rows.append([x1, y1, x2, y2, float(score), float(int(klass))])
            out[path.stem] = np.asarray(rows, np.float32).reshape(-1, 6)
        print(json.dumps({"variant": variant, "done": min(start + batch, len(paths)),
                          "total": len(paths)}, ensure_ascii=False), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("depth", "953"), required=True)
    ap.add_argument("--split", choices=("train", "val", "test"), default="val")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--variants", nargs="+",
                    choices=("identity", "hflip", "vflip", "rot180", "clahe",
                             "unsharp", "gamma095", "gamma105", "hue2"),
                    default=("clahe", "unsharp"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()
    if args.data_root is None:
        args.data_root = (Path("/workspace/SawitMVC-Depth-YOLO") if args.dataset == "depth"
                          else Path("/workspace/SawitMVC-YOLO"))
    paths = image_paths(args.data_root, args.split)
    if args.max_images:
        paths = paths[:args.max_images]
    from rfdetr import RFDETRLarge
    model = RFDETRLarge(pretrain_weights=str(args.weights), resolution=args.imgsz)
    # The downloaded checkpoint is a training-form model.  FP16 inference is
    # safe for this detector and materially improves throughput on the 3090;
    # avoid torch.compile here because the input image sizes are heterogeneous.
    try:
        model.optimize_for_inference(compile=False, batch_size=args.batch,
                                     dtype="float16", inplace=True)
    except Exception as exc:  # keep the TTA path usable across RF-DETR versions
        print(json.dumps({"optimize_for_inference": "skipped",
                          "reason": str(exc)}), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        rows = run_variant(model, paths, variant, args.batch)
        path = args.output_dir / f"{args.dataset}_{args.split}_{variant}.npz"
        np.savez_compressed(path, **rows)
        print(json.dumps({"variant": variant, "images": len(rows),
                          "predictions": sum(len(x) for x in rows.values()),
                          "path": str(path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
