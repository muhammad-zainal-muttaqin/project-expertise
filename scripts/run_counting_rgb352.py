"""Counting evaluation untuk tiga detektor RGB pada 352 pohon SawitMVC-Depth.

Langkah:
  1. Inference per-tree untuk YOLO26l, RT-DETR-L, RF-DETR-L (models RGB 352)
  2. Ridge + F_all counting (train+val → test)

Usage:
    python run_counting_rgb352.py \
        --project-root /workspace/project-expertise \
        --image-dir /workspace/SawitMVC-Depth/images \
        --gt-dir /workspace/SawitMVC-Depth/json \
        --yolo-dir /workspace/SawitMVC-Depth-YOLO
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = ["B1", "B2", "B3", "B4"]

MODELS = [
    ("YOLO26l",   "yolo",   "runs/yolo26l_e60_i1280_rgb352/weights/best.pt",          1280),
    ("RT-DETR-L", "rtdetr", "runs/rtdetr_l_e60_i1280_rgb352/weights/best.pt",         1280),
    ("RF-DETR-L", "rfdetr", "runs/rfdetr_l_e60_i1280_rgb352/checkpoint_best_ema.pth", 1280),
]


def derive_splits(yolo_dir: Path) -> dict[str, str]:
    """Derive tree_name → split from SawitMVC-Depth-YOLO directory structure."""
    splits = {}
    for sp in ["train", "val", "test"]:
        imgd = yolo_dir / sp / "images"
        if not imgd.exists():
            continue
        for p in imgd.iterdir():
            parts = p.stem.rsplit("_", 1)
            if len(parts) == 2:
                tree_name = parts[0]
                if tree_name not in splits:
                    splits[tree_name] = sp
    return splits


def group_images_by_tree(image_dir: Path) -> dict[str, list[Path]]:
    trees: dict[str, list[Path]] = defaultdict(list)
    for img in sorted(image_dir.glob("*.jpg")):
        parts = img.stem.rsplit("_", 1)
        if len(parts) == 2:
            trees[parts[0]].append(img)
    return trees


def load_gt(gt_dir: Path) -> dict[str, dict[str, int]]:
    gt = {}
    for fp in sorted(gt_dir.glob("*.json")):
        with open(fp, encoding="utf-8-sig") as fh:
            d = json.load(fh)
        tid = d.get("tree_name") or d.get("tree_id") or fp.stem
        summary = d.get("summary", {})
        by_class = summary.get("by_class", summary)
        gt[tid] = {c: by_class.get(c, 0) for c in CLASSES}
    return gt


def infer_yolo(model, img_paths: list[Path], imgsz: int) -> list[dict]:
    result = model.predict(source=[str(p) for p in img_paths], imgsz=imgsz,
                           conf=0.25, verbose=False)
    all_anns = []
    for r in result:
        anns = []
        for box in r.boxes:
            cls_id = int(box.cls.item())
            cx, cy, w, h = box.xywhn[0].tolist()
            anns.append({
                "class_name": CLASSES[cls_id] if cls_id < len(CLASSES) else f"cls{cls_id}",
                "bbox_yolo": [cx, cy, w, h],
                "conf": float(box.conf.item()),
            })
        all_anns.append(anns)
    return all_anns


def infer_rfdetr(model, img_paths: list[Path]) -> list[dict]:
    all_anns = []
    for p in img_paths:
        det = model.predict(str(p), threshold=0.25)
        anns = []
        for k in range(len(det.xyxy)):
            x1, y1, x2, y2 = det.xyxy[k]
            cls_id = int(det.class_id[k])
            w_img = det.image.shape[1] if hasattr(det, 'image') else 1280
            h_img = det.image.shape[0] if hasattr(det, 'image') else 800
            cx = float((x1 + x2) / 2) / w_img
            cy = float((y1 + y2) / 2) / h_img
            bw = float(x2 - x1) / w_img
            bh = float(y2 - y1) / h_img
            anns.append({
                "class_name": CLASSES[cls_id] if cls_id < len(CLASSES) else f"cls{cls_id}",
                "bbox_yolo": [cx, cy, bw, bh],
                "conf": float(det.confidence[k]),
            })
        all_anns.append(anns)
    return all_anns


def run_inference(kind: str, weights: str, imgsz: int,
                  trees: dict[str, list[Path]], tree_splits: dict[str, str],
                  out_dir: Path) -> int:
    """Run inference and write per-tree JSONs. Returns count."""
    import torch
    if kind == "yolo":
        from ultralytics import YOLO
        model = YOLO(weights)
    elif kind == "rtdetr":
        from ultralytics import RTDETR
        model = RTDETR(weights)
    else:
        from rfdetr import RFDETRLarge
        model = RFDETRLarge(pretrain_weights=weights, resolution=imgsz)

    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for tree_name in sorted(trees):
        imgs = sorted(trees[tree_name])
        if kind in ("yolo", "rtdetr"):
            all_anns = infer_yolo(model, imgs, imgsz)
        else:
            all_anns = infer_rfdetr(model, imgs)

        images_payload = {}
        for img_path, anns in zip(imgs, all_anns):
            side_str = img_path.stem.rsplit("_", 1)[1]
            side = int(side_str)
            images_payload[f"side_{side}"] = {"side_index": side - 1, "annotations": anns}

        split = tree_splits.get(tree_name, "train")
        payload = {
            "tree_name": tree_name,
            "split": split,
            "detector": kind,
            "images": images_payload,
        }
        (out_dir / f"{tree_name}.json").write_text(json.dumps(payload, indent=2))
        n += 1

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return n


def extract_all_features(tree_json: dict) -> dict[str, float]:
    sides = tree_json.get("images", {})
    n_sides = max(len(sides), 1)
    psc = {c: [] for c in CLASSES}
    cf_ = {c: [] for c in CLASSES}
    ar_ = {c: [] for c in CLASSES}
    cy_ = {c: [] for c in CLASSES}

    for sd in sides.values():
        cnt = {c: 0 for c in CLASSES}
        for ann in sd.get("annotations", []):
            cls = ann.get("class_name", "")
            if cls not in CLASSES:
                continue
            conf = float(ann.get("conf", 1.0))
            bbox = ann.get("bbox_yolo", [0, 0, 0, 0])
            cf_[cls].append(conf)
            ar_[cls].append(float(bbox[2]) * float(bbox[3]))
            cy_[cls].append(float(bbox[1]))
            cnt[cls] += 1
        for c in CLASSES:
            psc[c].append(cnt[c])

    f: dict[str, float] = {}
    for c in CLASSES:
        ps = np.array(psc[c], dtype=float)
        cf = np.array(cf_[c])
        ar = np.array(ar_[c])
        cy = np.array(cy_[c])
        n = len(cf)
        f[f"naive_sum_{c}"] = float(ps.sum())
        f[f"max_per_side_{c}"] = float(ps.max())
        f[f"mean_per_side_{c}"] = float(ps.mean())
        f[f"std_per_side_{c}"] = float(ps.std())
        f[f"min_per_side_{c}"] = float(ps.min())
        f[f"cv_per_side_{c}"] = float(ps.std() / (ps.mean() + 1e-6))
        f[f"n_sides_det_{c}"] = float((ps > 0).sum())
        f[f"consistency_{c}"] = float(1.0 / (1.0 + ps.std()))
        f[f"conf_sum_{c}"] = float(cf.sum())
        f[f"conf_mean_{c}"] = float(cf.mean()) if n > 0 else 0.0
        f[f"conf_max_{c}"] = float(cf.max()) if n > 0 else 0.0
        f[f"high_conf_{c}"] = float((cf >= 0.5).sum())
        f[f"vhigh_conf_{c}"] = float((cf >= 0.6).sum())
        f[f"mean_cy_{c}"] = float(cy.mean()) if n > 0 else 0.5
        f[f"mean_area_{c}"] = float(ar.mean()) if n > 0 else 0.0

    total = sum(f[f"naive_sum_{c}"] for c in CLASSES)
    f["n_sides"] = float(n_sides)
    f["total_naive"] = float(total)
    for c in CLASSES:
        f[f"frac_{c}"] = f[f"naive_sum_{c}"] / (total + 1e-6)
    f["b3_b23_frac"] = f["naive_sum_B3"] / (f["naive_sum_B2"] + f["naive_sum_B3"] + 1e-6)
    return f


def load_pertree_dataset(inference_dir: Path, gt_map: dict[str, dict[str, int]]):
    rows, labels, tree_ids, tree_splits = [], [], [], []
    for fp in sorted(inference_dir.glob("*.json")):
        with open(fp, encoding="utf-8-sig") as fh:
            d = json.load(fh)
        tid = d.get("tree_name") or d.get("tree_id") or fp.stem
        if tid not in gt_map:
            continue
        rows.append(extract_all_features(d))
        labels.append([gt_map[tid].get(c, 0) for c in CLASSES])
        tree_ids.append(tid)
        tree_splits.append(d.get("split", "train"))
    df = pd.DataFrame(rows)
    return df, np.array(labels, dtype=float), tree_ids, np.array(tree_splits)


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yr = np.clip(np.round(y_pred), 0, None).astype(int)
    yt = y_true.astype(int)
    r: dict = {}
    for j, c in enumerate(CLASSES):
        err = np.abs(yr[:, j] - yt[:, j])
        r[f"acc_{c}"] = float(np.mean(err <= 1))
        r[f"mae_{c}"] = float(np.mean(err))
        r[f"bias_{c}"] = float(np.mean(yr[:, j] - yt[:, j]))
    r["macro_acc"] = float(np.mean([r[f"acc_{c}"] for c in CLASSES]))
    r["macro_mae"] = float(np.mean([r[f"mae_{c}"] for c in CLASSES]))
    r["joint_acc"] = float(np.mean(
        np.all(np.array([np.abs(yr[:, j] - yt[:, j]) <= 1 for j in range(len(CLASSES))]), axis=0)
    ))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default="/workspace/project-expertise")
    ap.add_argument("--image-dir", default="/workspace/SawitMVC-Depth/images")
    ap.add_argument("--gt-dir", default="/workspace/SawitMVC-Depth/json")
    ap.add_argument("--yolo-dir", default="/workspace/SawitMVC-Depth-YOLO")
    args = ap.parse_args()

    proj = Path(args.project_root)
    image_dir = Path(args.image_dir)
    gt_dir = Path(args.gt_dir)
    yolo_dir = Path(args.yolo_dir)

    tree_splits = derive_splits(yolo_dir)
    trees = group_images_by_tree(image_dir)
    gt_map = load_gt(gt_dir)

    print(f"Trees: {len(trees)}, GT: {len(gt_map)}, Splits: train={sum(1 for v in tree_splits.values() if v=='train')}"
          f" val={sum(1 for v in tree_splits.values() if v=='val')}"
          f" test={sum(1 for v in tree_splits.values() if v=='test')}")

    out_file = proj / "results" / "counting_rgb352.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_file.read_text()) if out_file.exists() else {}

    for label, kind, rel_weights, imgsz in MODELS:
        weights = str(proj / rel_weights)
        if not Path(weights).exists():
            print(f"SKIP {label}: {weights} not found")
            continue

        pertree_dir = proj / "runs" / "pertree_rgb352" / f"{kind}_{label.lower().replace('-','')}"
        print(f"\n===== {label} ({kind}) =====")
        print(f"  Inference on {len(trees)} trees...")
        n = run_inference(kind, weights, imgsz, trees, tree_splits, pertree_dir)
        print(f"  Wrote {n} per-tree JSONs to {pertree_dir}")

        df, y, tree_ids, splits = load_pertree_dataset(pertree_dir, gt_map)
        tr = splits == "train"
        va = splits == "val"
        te = splits == "test"
        print(f"  Dataset: Train={tr.sum()} | Val={va.sum()} | Test={te.sum()}")

        train_mask = tr | va
        X_all = df.values.astype(float)
        X_tr, y_tr = X_all[train_mask], y[train_mask]
        X_te, y_te = X_all[te], y[te]

        model = Pipeline([
            ("sc", StandardScaler()),
            ("rid", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500]))
        ])
        model.fit(X_tr, y_tr)
        m = score(y_te, model.predict(X_te))

        print(f"  Ridge+F_all (train+val → test, {int(te.sum())} trees):")
        print(f"    Class ±1 Acc: {m['macro_acc']*100:.2f}%")
        print(f"    Tree  ±1 Acc: {m['joint_acc']*100:.2f}%")
        print(f"    Macro MAE:    {m['macro_mae']:.4f}")
        for c in CLASSES:
            print(f"    {c}: acc={m[f'acc_{c}']*100:.1f}% mae={m[f'mae_{c}']:.3f} bias={m[f'bias_{c}']:+.3f}")

        results[label] = {
            "detector": f"{kind}_rgb352",
            "n_trees_test": int(te.sum()),
            "feature_set": "F_all",
            "n_dim": int(df.shape[1]),
            "model": "Ridge",
            "strategy": "train+val",
            **m,
        }
        out_file.write_text(json.dumps(results, indent=2))

    print(f"\n-> {out_file}")


if __name__ == "__main__":
    main()
