"""Evaluate saved predictions by acquisition campaign.

The main evaluator stores predictions keyed by image stem.  This script joins
those dumps to the authoritative ``linked/*.json`` metadata and recomputes
COCO metrics for DAMIMAS, MARIHAT, and TOPAZ without running inference again.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


NAMES = ["B1", "B2", "B3", "B4"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
CAMPAIGNS = ("DAMIMAS", "MARIHAT", "TOPAZ")


def metadata_index(source: Path, split: str) -> dict[str, dict]:
    """Map image filename/stem to campaign and tree for one split."""
    out = {}
    for linked in sorted((source / ("valid" if split == "val" else split) / "linked").glob("*.json")):
        data = json.loads(linked.read_text())
        meta = data.get("metadata", {})
        session = str(meta.get("session_id", ""))
        campaign = session.split("-")[1] if "-" in session else "UNKNOWN"
        tree = data.get("tree_id", linked.stem)
        for image in data.get("images", {}).values():
            filename = image.get("filename")
            if filename:
                out[filename] = {"campaign": campaign, "tree_id": tree}
                out[Path(filename).stem] = {"campaign": campaign, "tree_id": tree}
    return out


def ground_truth(data_root: Path, source_root: Path, split: str):
    base = data_root / ("valid" if split == "val" else split)
    meta = metadata_index(source_root, split)
    images, annotations = [], []
    campaign_by_id, tree_by_id = {}, {}
    ann_id = 1
    paths = sorted(p for p in (base / "images").iterdir()
                   if p.suffix.lower() in IMAGE_EXTS)
    for image_id, path in enumerate(paths, 1):
        with Image.open(path) as im:
            width, height = im.size
        info = meta.get(path.name, meta.get(path.stem, {}))
        campaign = info.get("campaign", "UNKNOWN")
        tree_id = info.get("tree_id", path.stem.rsplit("_", 1)[0])
        images.append({"id": image_id, "file_name": path.name,
                       "width": width, "height": height})
        campaign_by_id[image_id] = campaign
        tree_by_id[image_id] = tree_id
        label = base / "labels" / f"{path.stem}.txt"
        if label.is_file():
            for line in label.read_text().splitlines():
                values = line.split()
                if len(values) < 5:
                    continue
                cls, cx, cy, bw, bh = int(values[0]), *(float(v) for v in values[1:5])
                x, y = (cx - bw / 2) * width, (cy - bh / 2) * height
                box_w, box_h = bw * width, bh * height
                annotations.append({
                    "id": ann_id, "image_id": image_id,
                    "category_id": cls + 1, "bbox": [x, y, box_w, box_h],
                    "area": box_w * box_h, "iscrowd": 0,
                })
                ann_id += 1
    return images, annotations, campaign_by_id, tree_by_id


def metric(images: list[dict], annotations: list[dict], detections: list[dict]) -> dict:
    gt = COCO()
    gt.dataset = {"images": images, "annotations": annotations,
                  "categories": [{"id": i + 1, "name": n}
                                 for i, n in enumerate(NAMES)]}
    gt.createIndex()
    dt = gt.loadRes(detections) if detections else gt.loadRes([])
    ev = COCOeval(gt, dt, "bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {"mAP50": round(float(ev.stats[1]), 6),
            "mAP50_95": round(float(ev.stats[0]), 6),
            "n_images": len(images),
            "n_trees": len({image.get("_tree_id", image["file_name"])
                             for image in images}),
            "n_annotations": len(annotations),
            "n_detections": len(detections)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path,
                    default=Path("/workspace/SawitMVC-Depth-YOLO-RGB"))
    ap.add_argument("--source-dataset", type=Path,
                    default=Path("/workspace/SawitMVC-Depth-YOLO"))
    ap.add_argument("--results-dir", type=Path,
                    default=Path("/workspace/project-expertise/results/new763"))
    ap.add_argument("--output", type=Path,
                    default=Path("/workspace/project-expertise/results/new763_campaigns.json"))
    args = ap.parse_args()

    output = {"dataset": str(args.dataset.resolve()), "source_dataset": str(args.source_dataset.resolve()),
              "campaigns": list(CAMPAIGNS), "runs": {}}
    for result_path in sorted(args.results_dir.glob("*_rgb_s*_i1280.json")):
        result = json.loads(result_path.read_text())
        run_name = result["run_name"]
        output["runs"][run_name] = {"kind": result.get("kind"), "splits": {}}
        for split, split_result in result.get("splits", {}).items():
            pred_path = Path(split_result["predictions"])
            if not pred_path.is_file():
                continue
            images, annotations, campaign_by_id, tree_by_id = ground_truth(
                args.dataset, args.source_dataset, split)
            predictions = np.load(pred_path, allow_pickle=False)
            detections = []
            for image in images:
                rows = predictions.get(Path(image["file_name"]).stem)
                if rows is None:
                    rows = np.empty((0, 6), dtype=np.float32)
                for x1, y1, x2, y2, score, cls in rows:
                    detections.append({
                        "image_id": image["id"], "category_id": int(cls) + 1,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score),
                    })
            by_campaign = {}
            for campaign in CAMPAIGNS:
                selected = {image["id"] for image in images
                            if campaign_by_id[image["id"]] == campaign}
                sub_images = [image for image in images if image["id"] in selected]
                sub_anns = [ann for ann in annotations if ann["image_id"] in selected]
                sub_dets = [det for det in detections if det["image_id"] in selected]
                # The metric helper needs tree counts; retain the IDs in the image dicts.
                for image in sub_images:
                    image["_tree_id"] = tree_by_id[image["id"]]
                metrics = metric(sub_images, sub_anns, sub_dets)
                metrics["n_trees"] = len({tree_by_id[i] for i in selected})
                by_campaign[campaign] = metrics
            output["runs"][run_name]["splits"][split] = by_campaign
            predictions.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
