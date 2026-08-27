"""Apply a short-run crop classifier to remote WBF proposals.

The output keeps the WBF geometry and confidence, replacing the detector
class vote with classifier probabilities.  It is intentionally a separate
artifact: the detector-only soft vote remains the reproducible baseline and
the classifier experiment must earn its place through the downstream sweep.

Input/output row format: ``x1,y1,x2,y2,score,p_B1,p_B2,p_B3,p_B4``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import build_crop_dataset as crop  # noqa: E402
import eval_remote_pipeline_postprocess as base  # noqa: E402
from train_crop_classifier import IMG, Model, prob_head  # noqa: E402


def make_crop(bgr: np.ndarray, box: np.ndarray, width: int, height: int,
              side: int = crop.S):
    """Create the same context crop and target-box mask used in training."""
    x1f, y1f, x2f, y2f = [float(x) for x in box]
    cx = (x1f + x2f) / 2.0 / max(width, 1)
    cy = (y1f + y2f) / 2.0 / max(height, 1)
    bw = (x2f - x1f) / max(width, 1)
    bh = (y2f - y1f) / max(height, 1)
    x0, y0, x1, y1 = crop.kotak_persegi(cx, cy, bw, bh, width, height)
    if x1 - x0 < 8:
        return None
    rgb = cv2.resize(crop.ambil(bgr, x0, y0, x1, y1, 0),
                     (side, side), interpolation=cv2.INTER_AREA)
    win = x1 - x0
    mw, mh = (x2f - x1f) / win, (y2f - y1f) / win
    mask = np.zeros((side, side), np.uint8)
    mx0 = int(round((0.5 - mw / 2) * side))
    mx1 = int(round((0.5 + mw / 2) * side))
    my0 = int(round((0.5 - mh / 2) * side))
    my1 = int(round((0.5 + mh / 2) * side))
    mask[max(0, my0):min(side, my1), max(0, mx0):min(side, mx1)] = 255
    return rgb, mask


def tensorize(rgb: np.ndarray, mask: np.ndarray, device: str):
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(0, 3, 1, 2)
    x = x.float().div_(255.)
    m = torch.from_numpy(np.ascontiguousarray(mask))[:, None].float().div_(255.)
    x = F.interpolate(x, (IMG, IMG), mode="bilinear", align_corners=False)
    m = F.interpolate(m, (IMG, IMG), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    x = (x - mean) / std
    x = torch.cat([x, m * 2. - 1.], 1)
    return x.to(device, non_blocking=True)


@torch.inference_mode()
def predict(model, head: str, rgb: list[np.ndarray], masks: list[np.ndarray],
            batch: int, device: str) -> np.ndarray:
    out = []
    for start in range(0, len(rgb), batch):
        x = tensorize(np.stack(rgb[start:start + batch]),
                      np.stack(masks[start:start + batch]), device)
        # The model was trained in RGB-only mode; the depth tensor is unused,
        # but passing a correctly shaped zero tensor keeps the model contract.
        d = torch.zeros((len(x), 2, IMG, IMG), device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits, _ = model(x, d)
            p = prob_head(logits, head)
        out.append(p.float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, base.K))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True,
                    help="WBF softvote NPZ")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path,
                    default=Path("/workspace/SawitMVC-YOLO"))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4,
                    help="OpenCV worker hint; image decoding is kept ordered")
    ap.add_argument("--proposal-min", type=float, default=0.0,
                    help="only classify rows at/above this score; 0 keeps all")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA diperlukan untuk eksperimen classifier ini")
    # OpenCV decoding/cropping is light; reserve a small CPU pool for it and
    # let the 3090 handle large, contiguous classifier batches.
    torch.set_num_threads(max(1, min(args.workers, os.cpu_count() or 1)))
    torch.backends.cudnn.benchmark = True
    device = "cuda"

    ck = torch.load(args.checkpoint, map_location="cpu")
    cargs = ck["args"]
    model = Model(cargs["backbone"], cargs["mode"] == "rgbd",
                  cargs.get("gate_init", 0.1), cargs["head"])
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    if cargs["mode"] != "rgb":
        raise ValueError("Eksperimen ini mengharapkan checkpoint mode=rgb")

    records = base.load_records(base.CONFIGS["SawitMVC-YOLO"], "test")
    with np.load(args.input) as archive:
        arrays = {stem: np.asarray(archive[stem], float).copy()
                  for stem in archive.files}

    # Group proposals by image so each JPEG is decoded once.  Collecting the
    # uint8 crops in host RAM lets the CNN see large contiguous batches.  This
    # is substantially faster than launching one GPU batch per image; the test
    # corpus is small enough for the available 30+ GiB RAM.
    by_stem = {}
    for rec in records.values():
        for view in rec["views"].values():
            by_stem[view["stem"]] = view

    total = 0
    pending_rgb, pending_masks, locations = [], [], []
    for num, (stem, rows) in enumerate(arrays.items(), 1):
        view = by_stem.get(stem)
        if view is None or len(rows) == 0:
            continue
        path = args.dataset_root / "images" / "test" / view["filename"]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        for i, row in enumerate(rows):
            total += 1
            if row[4] < args.proposal_min:
                continue
            item = make_crop(bgr, row[:4], view["width"], view["height"])
            if item is None:
                continue
            r, m = item
            pending_rgb.append(r); pending_masks.append(m)
            locations.append((stem, i))
        if num % 80 == 0:
            print(f"  {num}/{len(arrays)} citra; {len(locations)}/{total} proposal",
                  flush=True)

    p = predict(model, cargs["head"], pending_rgb, pending_masks,
                args.batch, device)
    for (stem, index), probs in zip(locations, p):
        arrays[stem][index, 5:9] = probs
    classified = len(p)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    meta = {
        "checkpoint": str(args.checkpoint),
        "input": str(args.input),
        "output": str(args.output),
        "dataset": "SawitMVC-YOLO test",
        "classifier": cargs,
        "proposal_min_classified": args.proposal_min,
        "n_images": len(arrays),
        "n_rows": total,
        "n_classified": classified,
    }
    meta_path = args.output.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
