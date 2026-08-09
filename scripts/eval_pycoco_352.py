"""Eval pycocotools tiga model pada 352 pohon SawitMVC-Depth-YOLO (RGB).

Usage:
    python eval_pycoco_352.py --project-root /workspace/project-expertise \
        --dataset-root /workspace/SawitMVC-Depth-YOLO --splits val test
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

NAMES = ["B1", "B2", "B3", "B4"]
CHUNK = 8

MODELS_352 = [
    ("YOLO26l",   "yolo",   "runs/yolo26l_e60_i1280_rgb352/weights/best.pt",            1280, 26.3),
    ("RT-DETR-L", "rtdetr", "runs/rtdetr_l_e60_i1280_rgb352/weights/best.pt",           1280, 33.0),
    ("RF-DETR-L", "rfdetr", "runs/rfdetr_l_e60_i1280_rgb352/checkpoint_best_ema.pth",   1280, 35.7),
]


def build_gt(ds_root, split_dir):
    images, anns, ann_id = [], [], 1
    idir = ds_root / split_dir / "images"
    ldir = ds_root / split_dir / "labels"
    paths = sorted(p for p in idir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tiff', '.tif'))
    for img_id, p in enumerate(paths, 1):
        w, h = Image.open(p).size
        images.append({"id": img_id, "file_name": p.name, "width": w, "height": h})
        lf = ldir / (p.stem + ".txt")
        if lf.is_file():
            for line in lf.read_text().splitlines():
                if not line.strip():
                    continue
                c, cx, cy, bw, bh = map(float, line.split())
                x, y, aw, ah = (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h
                anns.append({
                    "id": ann_id, "image_id": img_id, "category_id": int(c) + 1,
                    "bbox": [x, y, aw, ah], "area": aw * ah, "iscrowd": 0,
                })
                ann_id += 1
    gt = COCO()
    gt.dataset = {
        "images": images, "annotations": anns,
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(NAMES)],
    }
    gt.createIndex()
    return gt, paths


def predict_ultra(model, paths, imgsz):
    res = []
    for i, p in enumerate(paths, 1):
        r = model.predict(str(p), imgsz=imgsz, conf=0.001, verbose=False)[0]
        b = r.boxes
        if b is None:
            continue
        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy()
        for k in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[k]
            res.append({
                "image_id": i, "category_id": int(cls[k]) + 1,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(conf[k]),
            })
    return res


def predict_rfdetr(model, paths):
    res = []
    for i in range(0, len(paths), CHUNK):
        chunk = [str(p) for p in paths[i:i + CHUNK]]
        dl = model.predict(chunk, threshold=0.001)
        if not isinstance(dl, list):
            dl = [dl]
        for j, d in enumerate(dl):
            img_id = i + j + 1
            for k in range(len(d.xyxy)):
                x1, y1, x2, y2 = d.xyxy[k]
                res.append({
                    "image_id": img_id, "category_id": int(d.class_id[k]) + 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(d.confidence[k]),
                })
    return res


def per_class_ap(ev):
    p = ev.eval["precision"]
    ap50, ap95 = {}, {}
    for k, n in enumerate(NAMES):
        s95, s50 = p[:, :, k, 0, 2], p[0, :, k, 0, 2]
        ap95[n] = round(float(s95[s95 > -1].mean()) if (s95 > -1).any() else 0.0, 4)
        ap50[n] = round(float(s50[s50 > -1].mean()) if (s50 > -1).any() else 0.0, 4)
    return ap50, ap95


def load_model(kind, weights, imgsz):
    if kind == "yolo":
        from ultralytics import YOLO
        return YOLO(weights)
    if kind == "rtdetr":
        from ultralytics import RTDETR
        return RTDETR(weights)
    from rfdetr import RFDETRLarge
    return RFDETRLarge(pretrain_weights=weights, resolution=imgsz)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/workspace/project-expertise")
    parser.add_argument("--dataset-root", default="/workspace/SawitMVC-Depth-YOLO")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    args = parser.parse_args()

    proj = Path(args.project_root)
    ds = Path(args.dataset_root)
    out = proj / "results" / "perkelas_pycoco_rgb352.json"

    gts = {s: build_gt(ds, s) for s in args.splits}

    data = json.loads(out.read_text()) if out.exists() else {}
    for key, kind, rel_weights, imgsz, params in MODELS_352:
        weights = str(proj / rel_weights)
        if not Path(weights).exists():
            print(f"SKIP {key}: bobot belum ada ({weights})")
            continue
        print(f"\n===== {key} ({params}jt, imgsz {imgsz}, {kind}) =====")
        model = load_model(kind, weights, imgsz)
        entry = {"params_juta": params, "imgsz": imgsz, "evaluator": "pycocotools"}
        for split in args.splits:
            gt, paths = gts[split]
            dt_list = predict_rfdetr(model, paths) if kind == "rfdetr" else predict_ultra(model, paths, imgsz)
            ev = COCOeval(gt, gt.loadRes(dt_list), "bbox")
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            ap50, ap95 = per_class_ap(ev)
            entry[split] = {
                "mAP50": round(float(ev.stats[1]), 4),
                "mAP50_95": round(float(ev.stats[0]), 4),
                "per_kelas_AP50": ap50,
                "per_kelas_AP50_95": ap95,
            }
            print(f"  {split}: mAP50={entry[split]['mAP50']} mAP50-95={entry[split]['mAP50_95']}")
        data[key] = entry
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2))
        del model
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
