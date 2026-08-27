"""Train a small train-only colour/context head for fused proposals.

The detector already produces a class probability vector for every WBF
proposal.  This head adds local RGB/HSV/Lab statistics, proposal geometry,
and the detector probabilities, then learns a calibrated B1--B4 decision
from TRAIN proposals that overlap a ground-truth bunch.  It deliberately
does not use validation/test annotations while fitting.

The output NPZ files keep the original proposal geometry and score and replace
only columns 5:9 with the learned probabilities, so they can be consumed by
the existing validation-locked four-view evaluator.

Example:
    python scripts/train_fused_color_head.py \
      --dataset depth \
      --fused-root /workspace/model_artifacts/project-expertise/eval_2026-08-27 \
      --output-root /workspace/model_artifacts/project-expertise/color_head_depth \
      --epochs 80 --batch 2048 --workers 32
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_remote_pipeline_postprocess as base  # noqa: E402


K = len(base.NAMES)
EPS = 1e-6
_COLLECT_STATE = None


def fused_path(root: Path, dataset: str, split: str) -> Path:
    safe = "SawitMVC_Depth_YOLO" if dataset == "depth" else "SawitMVC_YOLO"
    folder = root / ("fused_combined1716" if split == "test"
                     else f"fused_combined1716_{split}")
    path = folder / f"{safe}__wbf_softvote.npz"
    if path.exists():
        return path
    # Test soft-vote dumps are compactly tracked in the repository; the
    # larger train/validation staging files remain under model_artifacts.
    if split == "test":
        repo = Path(__file__).resolve().parents[1] / "results" / \
            "remote_eval_2026-08-27" / "fused_combined1716"
        return repo / f"{safe}__wbf_softvote.npz"
    return path


def load_vote(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {stem: np.asarray(archive[stem], np.float32) for stem in archive.files}


def image_path(cfg: dict, split: str, filename: str) -> Path:
    if cfg["kind"] == "depth":
        folder = "valid" if split == "val" else split
        return cfg["data_root"] / folder / "images" / filename
    return cfg["data_root"] / "images" / split / filename


def iou_box(box: np.ndarray, gt: np.ndarray) -> float:
    x1 = max(float(box[0]), float(gt[0]))
    y1 = max(float(box[1]), float(gt[1]))
    x2 = min(float(box[2]), float(gt[2]))
    y2 = min(float(box[3]), float(gt[3]))
    inter = max(x2 - x1, 0.) * max(y2 - y1, 0.)
    aa = max(float(box[2] - box[0]), 0.) * max(float(box[3] - box[1]), 0.)
    bb = max(float(gt[2] - gt[0]), 0.) * max(float(gt[3] - gt[1]), 0.)
    return inter / (aa + bb - inter + EPS)


def gt_boxes(view: dict) -> tuple[np.ndarray, np.ndarray]:
    boxes, labels = [], []
    for ann in view.get("annotations", []):
        box = ann.get("bbox_pixel")
        if box is None:
            continue
        # ``bbox_pixel`` in the linked metadata is xyxy (unlike the YOLO
        # label files, which are cx/cy/w/h).  Keep the representation aligned
        # with the WBF proposal rows.
        x1, y1, x2, y2 = [float(v) for v in box]
        boxes.append([x1, y1, x2, y2])
        labels.append(int(ann.get("class_id", -1)))
    # The linked metadata is normally complete, but the bunch list is the
    # authoritative four-view label source when image annotations are absent.
    if not boxes:
        for bunch in view.get("_record_bunches", []):
            for app in bunch.get("appearances", []):
                if int(app.get("side", -1)) != int(view["side"]):
                    continue
                x, y, x2, y2 = [float(v) for v in app["box"]]
                boxes.append([x, y, x2, y2])
                labels.append(int(bunch.get("cls", -1)))
    return np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(labels, np.int64)


def quantiles(a: np.ndarray) -> list[float]:
    if a.size == 0:
        return [0.] * 5
    return [float(x) for x in np.percentile(a, [10, 25, 50, 75, 90])]


def region_stats(rgb: np.ndarray, hsv: np.ndarray, lab: np.ndarray,
                 mask: np.ndarray) -> list[float]:
    """Colour statistics for a proposal region.

    H is represented by sin/cos and is weighted by saturation, avoiding the
    artificial discontinuity at red (0/179).  Lab a/b retains the red-green
    and blue-yellow axes that often separate ripeness under illumination
    changes better than raw RGB alone.
    """
    if not np.any(mask):
        return [0.] * 54
    r = rgb[mask].astype(np.float32) / 255.
    h = hsv[..., 0][mask].astype(np.float32) * (2. * np.pi / 180.)
    s = hsv[..., 1][mask].astype(np.float32) / 255.
    v = hsv[..., 2][mask].astype(np.float32) / 255.
    l = lab[mask].astype(np.float32)
    sat_weight = np.maximum(s, .05)
    sin_h = np.sum(np.sin(h) * sat_weight) / np.sum(sat_weight)
    cos_h = np.sum(np.cos(h) * sat_weight) / np.sum(sat_weight)
    out: list[float] = [
        *r.mean(0).tolist(), *r.std(0).tolist(),
        *quantiles(r[:, 0]), *quantiles(r[:, 1]), *quantiles(r[:, 2]),
        float(sin_h), float(cos_h), float(s.mean()), float(s.std()),
        *quantiles(s), float(v.mean()), float(v.std()), *quantiles(v),
        float(l[:, 0].mean() / 255.), float(l[:, 1].mean() / 255.),
        float(l[:, 2].mean() / 255.), float(l[:, 1].std() / 255.),
        float(l[:, 2].std() / 255.),
    ]
    # A compact hue histogram, weighted by saturation, gives the head a
    # stable signal even when a proposal contains a little background.
    hist, _ = np.histogram(hsv[..., 0][mask], bins=12, range=(0, 180),
                           weights=sat_weight)
    hist = hist.astype(np.float32)
    hist /= max(float(hist.sum()), EPS)
    out.extend(hist.tolist())
    return out


def extract_row_features(image: np.ndarray, row: np.ndarray,
                         rgb: np.ndarray | None = None,
                         hsv: np.ndarray | None = None,
                         lab: np.ndarray | None = None) -> np.ndarray:
    """Extract geometry + detector probabilities + colour/context features."""
    h_img, w_img = image.shape[:2]
    bgr = image
    if rgb is None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if hsv is None:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if lab is None:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    x1, y1, x2, y2 = [float(v) for v in row[:4]]
    x1, x2 = sorted((max(0., x1), min(float(w_img), x2)))
    y1, y2 = sorted((max(0., y1), min(float(h_img), y2)))
    bw, bh = max(x2 - x1, 1.), max(y2 - y1, 1.)
    ix0, iy0, ix1, iy1 = int(x1), int(y1), max(int(np.ceil(x2)), int(x1) + 1), max(int(np.ceil(y2)), int(y1) + 1)

    # The box and a surrounding ring are separate evidence.  The ring helps
    # distinguish fruit colour from the neighbouring frond/background.
    cx, cy = (x1 + x2) / 2., (y1 + y2) / 2.
    side = 1.6 * max(bw, bh)
    qx0, qy0 = int(max(0., cx - side / 2.)), int(max(0., cy - side / 2.))
    qx1, qy1 = int(min(w_img, cx + side / 2.)), int(min(h_img, cy + side / 2.))
    # Work in the small context crop instead of allocating full-resolution
    # masks for every proposal.  This matters on the 953 corpus, which has
    # tens of thousands of WBF rows.
    rgb_q = rgb[qy0:qy1, qx0:qx1]
    hsv_q = hsv[qy0:qy1, qx0:qx1]
    lab_q = lab[qy0:qy1, qx0:qx1]
    box_mask = np.zeros(rgb_q.shape[:2], bool)
    bx0, by0 = max(ix0 - qx0, 0), max(iy0 - qy0, 0)
    bx1 = min(ix1 - qx0, box_mask.shape[1])
    by1 = min(iy1 - qy0, box_mask.shape[0])
    if bx1 > bx0 and by1 > by0:
        box_mask[by0:by1, bx0:bx1] = True
    if not np.any(box_mask) and box_mask.size:
        box_mask[box_mask.shape[0] // 2, box_mask.shape[1] // 2] = True
    ring_mask = np.ones(box_mask.shape, bool) & ~box_mask
    if not np.any(ring_mask):
        ring_mask = box_mask.copy()

    geom = [
        float(row[4]), cx / max(w_img, 1), cy / max(h_img, 1),
        bw / max(w_img, 1), bh / max(h_img, 1),
        float(np.log((bw * bh) / max(w_img * h_img, 1))),
        float(bw / bh),
    ]
    p = np.asarray(row[5:5 + K], np.float32)
    if len(p) != K:
        p = np.zeros(K, np.float32)
    p = np.maximum(p, 0.)
    p /= max(float(p.sum()), EPS)
    logits = np.log(p + 1e-4).tolist()
    return np.asarray(geom + logits + region_stats(rgb_q, hsv_q, lab_q,
                                                    box_mask)
                      + region_stats(rgb_q, hsv_q, lab_q, ring_mask),
                      np.float32)


def prepare_views(rec: dict) -> dict[int, dict]:
    out = {}
    for side, view in rec["views"].items():
        item = dict(view)
        item["side"] = side
        item["_record_bunches"] = rec["bunches"]
        out[int(side)] = item
    return out


def _init_collect_worker() -> None:
    """Use one OpenCV thread per worker; the process pool supplies parallelism."""
    cv2.setNumThreads(1)


def _collect_record(item: tuple[str, dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract one tree in a worker; the parent only concatenates results."""
    if _COLLECT_STATE is None:
        raise RuntimeError("collector worker state is not initialized")
    cfg, split, vote = _COLLECT_STATE
    _tree_id, rec = item
    features, labels = [], []
    n_rows = n_matched = 0
    for _side, view in prepare_views(rec).items():
        rows = np.asarray(vote.get(view["stem"], np.zeros((0, 5 + K))),
                          np.float32)
        if not len(rows):
            continue
        path = image_path(cfg, split, view["filename"])
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gt, gt_cls = gt_boxes(view)
        for row in rows:
            feat = extract_row_features(image, row, rgb, hsv, lab)
            best, best_cls = 0., -1
            for box, cls in zip(gt, gt_cls):
                ov = iou_box(row[:4], box)
                if ov > best:
                    best, best_cls = ov, int(cls)
            features.append(feat)
            labels.append(best_cls if best >= .5 and 0 <= best_cls < K else -1)
            n_rows += 1
            n_matched += int(best >= .5 and 0 <= best_cls < K)
    X = np.stack(features) if features else np.zeros((0, 119), np.float32)
    y = np.asarray(labels, np.int64)
    return X, y, {"n_rows": n_rows, "n_matched": n_matched}


def collect_split(cfg: dict, split: str, vote: dict[str, np.ndarray],
                  workers: int = 1) -> tuple[np.ndarray, np.ndarray, dict]:
    """Collect all proposal features and labels; -1 means unmatched."""
    records = base.load_records(cfg, split)
    features, labels = [], []
    n_rows = n_matched = 0
    # The feature extractor is independent per tree.  Fork after loading the
    # vote bank so large NumPy arrays are shared read-only through COW.
    global _COLLECT_STATE
    _COLLECT_STATE = (cfg, split, vote)
    items = list(records.items())
    n_workers = max(1, min(int(workers), os.cpu_count() or 1, len(items)))
    if n_workers == 1:
        outputs = (_collect_record(item) for item in items)
        pool = None
    else:
        context = mp.get_context("fork")
        pool = ProcessPoolExecutor(max_workers=n_workers,
                                   mp_context=context,
                                   initializer=_init_collect_worker)
        outputs = pool.map(_collect_record, items, chunksize=1)
    try:
        for n, (X_tree, y_tree, stats) in enumerate(outputs, 1):
            if len(X_tree):
                features.append(X_tree)
                labels.append(y_tree)
            n_rows += stats["n_rows"]
            n_matched += stats["n_matched"]
            if n % 100 == 0 or n == len(items):
                print(f"  {split}: {n}/{len(items)} pohon, {n_rows} proposal",
                      flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    X = np.concatenate(features, axis=0) if features else np.zeros((0, 119), np.float32)
    y = np.concatenate(labels, axis=0) if labels else np.zeros(0, np.int64)
    meta = {"split": split, "n_trees": len(records), "n_rows": n_rows,
            "n_matched": n_matched,
            "matched_fraction": n_matched / max(n_rows, 1)}
    return X, y, meta


class ColorHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 192, dropout: float = .12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, K),
        )

    def forward(self, x):
        return self.net(x)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p.argmax(1)
    f1s, recalls = [], []
    for c in range(K):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
        recalls.append(tp / max(int((y == c).sum()), 1))
    return {"accuracy": float((pred == y).mean()),
            "macro_f1": float(np.mean(f1s)),
            "macro_recall": float(np.mean(recalls)),
            "f1_per_class": dict(zip(base.NAMES, f1s)),
            "recall_per_class": dict(zip(base.NAMES, recalls)),
            "n": int(len(y))}


def predict(model: nn.Module, X: np.ndarray, mean: np.ndarray,
            scale: np.ndarray, device: str, batch: int) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(X), batch):
            z = (X[start:start + batch] - mean) / scale
            t = torch.from_numpy(z).to(device, non_blocking=True)
            out.append(torch.softmax(model(t), 1).cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, K), np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("depth", "953"), required=True)
    ap.add_argument("--fused-root", type=Path,
                    default=Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27"))
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight-decay", type=float, default=2e-4)
    ap.add_argument("--loss", choices=("balanced", "plain"), default="balanced")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, min(args.workers, os.cpu_count() or 1)))
    cfg = base.CONFIGS["SawitMVC-Depth-YOLO" if args.dataset == "depth"
                       else "SawitMVC-YOLO"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA diperlukan untuk melatih color head")
    device = "cuda"
    args.output_root.mkdir(parents=True, exist_ok=True)

    votes = {}
    collected = {}
    for split in ("train", "val", "test"):
        path = fused_path(args.fused_root, args.dataset, split)
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"memuat {split}: {path}", flush=True)
        vote = load_vote(path)
        votes[split] = vote
        cache = args.output_root / f"features_{split}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as archive:
                X_cached = np.asarray(archive["X"], np.float32)
                y_cached = np.asarray(archive["y"], np.int64)
                meta_cached = json.loads(str(archive["meta"]))
            collected[split] = (X_cached, y_cached, meta_cached)
            print(f"  cache: {cache} ({len(y_cached)} proposal)", flush=True)
        else:
            collected[split] = collect_split(cfg, split, vote, args.workers)
            X_new, y_new, meta_new = collected[split]
            np.savez_compressed(cache, X=X_new, y=y_new,
                                meta=json.dumps(meta_new))
        print(json.dumps(collected[split][2], indent=2), flush=True)

    X_train, y_train_all, train_meta = collected["train"]
    keep = (y_train_all >= 0) & (y_train_all < K)
    X_train, y_train = X_train[keep], y_train_all[keep]
    if len(X_train) < 100 or len(np.unique(y_train)) < K:
        raise RuntimeError("proposal train tidak cukup untuk empat kelas")
    mean = X_train.mean(0).astype(np.float32)
    scale = X_train.std(0).astype(np.float32)
    scale[scale < 1e-5] = 1.
    dim = X_train.shape[1]
    model = ColorHead(dim, args.hidden).to(device)
    counts = np.bincount(y_train, minlength=K).astype(np.float32)
    if args.loss == "balanced":
        # Inverse square-root avoids letting rare B4 dominate every proposal,
        # while still stopping B1/B2/B3 from swallowing the minority class.
        weights = 1. / np.sqrt(np.maximum(counts, 1.))
        weights = weights / weights.mean()
        class_weight = torch.from_numpy(weights).float().to(device)
    else:
        class_weight = None
    ds = TensorDataset(torch.from_numpy((X_train - mean) / scale),
                       torch.from_numpy(y_train))
    sampler = WeightedRandomSampler(
        torch.from_numpy((1. / np.sqrt(np.maximum(counts, 1.)))[y_train]).double(),
        len(y_train), replacement=True)
    dl = DataLoader(ds, batch_size=args.batch, sampler=sampler,
                    num_workers=min(args.workers, 8), pin_memory=True,
                    persistent_workers=min(args.workers, 8) > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    best_state, best_val = None, -1.
    X_val, y_val_all, _ = collected["val"]
    val_keep = (y_val_all >= 0) & (y_val_all < K)
    y_val = y_val_all[val_keep]
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            loss = F.cross_entropy(model(xb), yb, weight=class_weight,
                                   label_smoothing=.02)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        p_val = predict(model, X_val[val_keep], mean, scale, device, args.batch)
        m_val = metrics(y_val, p_val)
        if m_val["macro_f1"] > best_val:
            best_val = m_val["macro_f1"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({"epoch": epoch, "loss": float(np.mean(losses)),
                              "val": m_val}, ensure_ascii=False), flush=True)
    if best_state is None:
        raise RuntimeError("tidak ada checkpoint valid")
    model.load_state_dict(best_state)

    output = {
        "dataset": cfg["kind"], "source_dataset": args.dataset,
        "fused_root": str(args.fused_root), "features": dim,
        "hidden": args.hidden, "loss": args.loss, "seed": args.seed,
        "train_rows": train_meta, "matched_train": int(len(y_train)),
        "class_counts": dict(zip(base.NAMES, counts.astype(int).tolist())),
        "val_matched_metrics": metrics(y_val, predict(model, X_val[val_keep], mean, scale, device, args.batch)),
    }
    checkpoint = args.output_root / "color_head.pt"
    torch.save({"model": model.state_dict(), "mean": mean, "scale": scale,
                "dim": dim, "hidden": args.hidden, "meta": output}, checkpoint)
    for split in ("train", "val", "test"):
        X, y_all, _ = collected[split]
        p_head = predict(model, X, mean, scale, device, args.batch)
        vote = votes[split]
        out_arrays = {}
        cursor = 0
        for stem, rows in vote.items():
            n = len(rows)
            out_rows = np.asarray(rows, np.float32).copy()
            if n:
                out_rows[:, 5:5 + K] = p_head[cursor:cursor + n]
            out_arrays[stem] = out_rows
            cursor += n
        np.savez_compressed(args.output_root / f"fused_{split}__wbf_softvote.npz",
                            **out_arrays)
    (args.output_root / "metadata.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint), **output}, indent=2,
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
