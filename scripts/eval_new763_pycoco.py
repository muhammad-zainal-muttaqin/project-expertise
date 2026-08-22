"""Evaluasi satu run baseline dengan COCOeval dan dump prediksi.

Prediksi validation dan test disimpan saat evaluasi dalam ``.npz`` sehingga
bootstrap/ensemble dapat diulang tanpa membutuhkan bobot training lagi.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


NAMES = ["B1", "B2", "B3", "B4"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def build_gt(root: Path, split: str):
    base = root / ("valid" if split == "val" else split)
    paths = sorted(p for p in (base / "images").iterdir()
                   if p.suffix.lower() in IMAGE_EXTS)
    images, anns, ann_id = [], [], 1
    for image_id, p in enumerate(paths, 1):
        with Image.open(p) as im:
            width, height = im.size
        images.append({"id": image_id, "file_name": p.name,
                       "width": width, "height": height})
        lf = base / "labels" / f"{p.stem}.txt"
        if lf.is_file():
            for line in lf.read_text().splitlines():
                q = line.split()
                if len(q) < 5:
                    continue
                c, cx, cy, bw, bh = int(q[0]), *(float(v) for v in q[1:5])
                x, y = (cx - bw / 2) * width, (cy - bh / 2) * height
                aw, ah = bw * width, bh * height
                anns.append({"id": ann_id, "image_id": image_id,
                             "category_id": c + 1, "bbox": [x, y, aw, ah],
                             "area": aw * ah, "iscrowd": 0})
                ann_id += 1
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns,
                  "categories": [{"id": i + 1, "name": n}
                                 for i, n in enumerate(NAMES)]}
    gt.createIndex()
    return gt, paths


def predict_ultra(model, paths: list[Path], imgsz: int, batch: int):
    predictions, dump = [], {}
    for start in range(0, len(paths), batch):
        chunk = paths[start:start + batch]
        results = model.predict([str(p) for p in chunk], imgsz=imgsz,
                                conf=0.001, iou=0.7, max_det=300,
                                verbose=False, save=False)
        for offset, (path, result) in enumerate(zip(chunk, results)):
            rows = []
            boxes = result.boxes
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                for (x1, y1, x2, y2), score, klass in zip(xyxy, conf, cls):
                    row = [float(x1), float(y1), float(x2), float(y2),
                           float(score), float(klass)]
                    rows.append(row)
                    predictions.append({
                        "image_id": start + offset + 1,
                        "category_id": int(klass) + 1,
                        "bbox": [float(x1), float(y1), float(x2 - x1),
                                 float(y2 - y1)],
                        "score": float(score),
                    })
            dump[path.stem] = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
    return predictions, dump


def predict_rfdetr(model, paths: list[Path], batch: int):
    predictions, dump = [], {}
    for start in range(0, len(paths), batch):
        chunk = paths[start:start + batch]
        results = model.predict([str(p) for p in chunk], threshold=0.001)
        if not isinstance(results, list):
            results = [results]
        for offset, (path, det) in enumerate(zip(chunk, results)):
            rows = []
            for xyxy, score, klass in zip(det.xyxy, det.confidence, det.class_id):
                x1, y1, x2, y2 = map(float, xyxy)
                score, klass = float(score), int(klass)
                rows.append([x1, y1, x2, y2, score, float(klass)])
                predictions.append({
                    "image_id": start + offset + 1,
                    "category_id": klass + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score,
                })
            dump[path.stem] = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
    return predictions, dump


def evaluate(gt: COCO, dt_list: list[dict]) -> dict:
    dt = gt.loadRes(dt_list) if dt_list else gt.loadRes([])
    ev = COCOeval(gt, dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    precision = ev.eval["precision"]
    per_class = {}
    for idx, name in enumerate(NAMES):
        values = precision[0, :, idx, 0, 2]
        values = values[values > -1]
        per_class[name] = round(float(values.mean()) if len(values) else 0.0, 6)
    return {"mAP50": round(float(ev.stats[1]), 6),
            "mAP50_95": round(float(ev.stats[0]), 6),
            "per_kelas_AP50": per_class}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("yolo", "rtdetr", "rfdetr"), required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--dataset", type=Path,
                    default=Path("/workspace/SawitMVC-Depth-YOLO"))
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=("val", "test"))
    args = ap.parse_args()

    if args.kind == "yolo":
        from ultralytics import YOLO
        model = YOLO(str(args.weights))
    elif args.kind == "rtdetr":
        from ultralytics import RTDETR
        model = RTDETR(str(args.weights))
    else:
        from rfdetr import RFDETRLarge
        model = RFDETRLarge(pretrain_weights=str(args.weights),
                            resolution=args.imgsz)

    result = {"run_name": args.run_name, "kind": args.kind,
              "weights": str(args.weights.resolve()),
              "dataset": str(args.dataset.resolve()), "imgsz": args.imgsz,
              "evaluator": "pycocotools.COCOeval", "splits": {}}
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        gt, paths = build_gt(args.dataset, split)
        if args.kind == "rfdetr":
            dt_list, dump = predict_rfdetr(model, paths, args.batch)
        else:
            dt_list, dump = predict_ultra(model, paths, args.imgsz, args.batch)
        metrics = evaluate(gt, dt_list)
        pred_path = args.pred_dir / f"{args.run_name}__{split}.npz"
        np.savez_compressed(pred_path, **dump)
        result["splits"][split] = {**metrics, "predictions": str(pred_path)}
        print(f"{args.run_name} {split}: mAP50={metrics['mAP50']:.6f} "
              f"mAP50-95={metrics['mAP50_95']:.6f}", flush=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2))
    print(f"-> {args.out_json}")
    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
