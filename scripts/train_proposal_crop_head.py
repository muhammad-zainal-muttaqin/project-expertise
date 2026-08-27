"""Train a proposal-aware visual B1--B4 head for the four-view pipeline.

Unlike a classifier trained on perfect GT crops, this head is trained on the
actual WBF proposal boxes.  Each proposal is labelled from TRAIN metadata when
IoU >= 0.5; validation is used only for checkpoint selection.  The model sees
RGB, a target-box mask, circular hue (sin/cos), saturation/value, and Lab
chromatic channels.  The detector/linker remains unchanged; the resulting
probabilities are consumed by ``evaluate_remote_class_head.py`` after a
physical cluster has already been formed.

No test labels are used by the training or selection path.  Test proposals are
only transformed and written for the final locked evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_remote_pipeline_postprocess as base  # noqa: E402


K = len(base.NAMES)
CTX = 1.6
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


@dataclass(frozen=True)
class Sample:
    stem: str
    row_index: int
    image_path: str
    box: tuple[float, float, float, float]
    label: int


def vote_path(root: Path, dataset: str, split: str) -> Path:
    safe = "SawitMVC_Depth_YOLO" if dataset == "depth" else "SawitMVC_YOLO"
    folder = root / ("fused_combined1716" if split == "test"
                     else f"fused_combined1716_{split}")
    path = folder / f"{safe}__wbf_softvote.npz"
    if path.exists():
        return path
    if split == "test":
        return (Path(__file__).resolve().parents[1] /
                "results" / "remote_eval_2026-08-27" /
                "fused_combined1716" / f"{safe}__wbf_softvote.npz")
    return path


def load_vote(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key], np.float32)
                for key in archive.files}


def image_path(cfg: dict, split: str, filename: str) -> Path:
    if cfg["kind"] == "depth":
        folder = "valid" if split == "val" else split
        return cfg["data_root"] / folder / "images" / filename
    return cfg["data_root"] / "images" / split / filename


def annotations(view: dict) -> tuple[np.ndarray, np.ndarray]:
    boxes, labels = [], []
    for ann in view.get("annotations", []):
        box = ann.get("bbox_pixel")
        if box is None:
            continue
        boxes.append([float(v) for v in box])  # metadata uses xyxy
        labels.append(int(ann.get("class_id", -1)))
    return np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(labels, np.int64)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(x2 - x1, 0.) * max(y2 - y1, 0.)
    aa = max(float(a[2] - a[0]), 0.) * max(float(a[3] - a[1]), 0.)
    bb = max(float(b[2] - b[0]), 0.) * max(float(b[3] - b[1]), 0.)
    return inter / (aa + bb - inter + 1e-9)


def build_samples(cfg: dict, dataset: str, split: str,
                  vote: dict[str, np.ndarray], label_split: bool,
                  match_iou: float = .5) -> tuple[list[Sample], dict[str, int]]:
    samples: list[Sample] = []
    records = base.load_records(cfg, split)
    counts = {name: 0 for name in base.NAMES}
    for rec in records.values():
        for view in rec["views"].values():
            rows = np.asarray(vote.get(view["stem"], np.zeros((0, 9))), np.float32)
            gt, gt_cls = annotations(view) if label_split else (
                np.zeros((0, 4), np.float32), np.zeros(0, np.int64))
            path = image_path(cfg, split, view["filename"])
            for index, row in enumerate(rows):
                label = -1
                if label_split and len(gt):
                    scores = np.asarray([iou(row[:4], box) for box in gt])
                    j = int(scores.argmax())
                    if scores[j] >= match_iou and 0 <= int(gt_cls[j]) < K:
                        label = int(gt_cls[j])
                        counts[base.NAMES[label]] += 1
                samples.append(Sample(
                    stem=view["stem"], row_index=index,
                    image_path=str(path),
                    box=tuple(float(v) for v in row[:4]), label=label))
    return samples, counts


@lru_cache(maxsize=96)
def read_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


@lru_cache(maxsize=96)
def read_color_corrected(path: str, mode: str) -> np.ndarray:
    """Cache frame-level white balance once per worker.

    A frame commonly contributes many proposals.  Performing the channel
    statistics inside every crop would scan the full image thousands of
    times, starving the GPU during TTA/evaluation.
    """
    image = read_image(path)
    source = image.astype(np.float32)
    channel_mean = source.mean(axis=(0, 1))
    target_mean = float(channel_mean.mean())
    gain = np.clip(target_mean / np.maximum(channel_mean, 1.), .70, 1.30)
    strength = .5 if mode == "rgb_mildwb" else 1.
    gain = 1. + strength * (gain - 1.)
    return np.clip(source * gain[None, None, :], 0, 255).astype(np.uint8)


def crop_features(image: np.ndarray, box: tuple[float, float, float, float],
                  side: int, augment: bool,
                  feature_mode: str = "rich") -> torch.Tensor:
    if feature_mode in ("rgb_grayworld", "rgb_mildwb"):
        # Correct frame-level illumination before cropping.  The operation is
        # deliberately a channel gain (not a hue rotation), so maturity
        # ordering is preserved while camera white-balance differences are
        # reduced.  A mild variant is useful when the background dominates the
        # global mean.
        source = image.astype(np.float32)
        channel_mean = source.mean(axis=(0, 1))
        target_mean = float(channel_mean.mean())
        gain = target_mean / np.maximum(channel_mean, 1.)
        gain = np.clip(gain, .70, 1.30)
        strength = .5 if feature_mode == "rgb_mildwb" else 1.
        gain = 1. + strength * (gain - 1.)
        image = np.clip(source * gain[None, None, :], 0, 255).astype(np.uint8)
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 2.), max(y2 - y1, 2.)
    cx, cy = (x1 + x2) / 2., (y1 + y2) / 2.
    window = max(bw, bh) * CTX
    x0, y0 = int(round(cx - window / 2)), int(round(cy - window / 2))
    x3, y3 = int(round(cx + window / 2)), int(round(cy + window / 2))
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x3 - w), max(0, y3 - h)
    if pad_l or pad_t or pad_r or pad_b:
        image = cv2.copyMakeBorder(image, pad_t, pad_b, pad_l, pad_r,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
    x0, x3, y0, y3 = x0 + pad_l, x3 + pad_l, y0 + pad_t, y3 + pad_t
    crop = image[max(0, y0):max(y0 + 1, y3),
                 max(0, x0):max(x0 + 1, x3)]
    if crop.size == 0:
        crop = np.zeros((side, side, 3), np.uint8)
    # Box coordinates are in the padded crop frame.
    mx0, mx1 = int(round(x1 + pad_l - x0)), int(round(x2 + pad_l - x0))
    my0, my1 = int(round(y1 + pad_t - y0)), int(round(y2 + pad_t - y0))
    mask = np.zeros(crop.shape[:2], np.uint8)
    mask[max(0, my0):min(mask.shape[0], max(my0 + 1, my1)),
         max(0, mx0):min(mask.shape[1], max(mx0 + 1, mx1))] = 255
    crop = cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (side, side), interpolation=cv2.INTER_NEAREST)

    # Keep a 3-channel ImageNet-compatible representation while exposing two
    # optional photometric variants.  CLAHE changes local contrast but keeps
    # chroma/hue in the LAB transform; unsharp is intentionally mild.  These
    # variants are used as separate heads, never as detector geometry input.
    if feature_mode == "rgb_clahe":
        lab_local = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_local[..., 0] = clahe.apply(lab_local[..., 0])
        crop = cv2.cvtColor(lab_local, cv2.COLOR_LAB2BGR)
    elif feature_mode == "rgb_sharp":
        blur = cv2.GaussianBlur(crop, (0, 0), 1.0)
        crop = cv2.addWeighted(crop, 1.35, blur, -0.35, 0)

    if augment:
        if np.random.rand() < .5:
            crop, mask = crop[:, ::-1], mask[:, ::-1]
        if np.random.rand() < .25:
            k = int(np.random.randint(1, 4))
            crop, mask = np.rot90(crop, k), np.rot90(mask, k)
        # Mild exposure jitter only.  Hue is deliberately not shifted because
        # hue is part of the maturity label definition.
        if np.random.rand() < .7:
            gain = float(np.random.uniform(.94, 1.06))
            crop = np.clip(crop.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue = hsv[..., 0] * (2. * np.pi / 180.)
    sat = hsv[..., 1] / 255.
    val = hsv[..., 2] / 255.
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_a = (lab[..., 1] - 128.) / 128.
    lab_b = (lab[..., 2] - 128.) / 128.
    if feature_mode in ("rgb", "rgb_clahe", "rgb_sharp", "rgb_grayworld",
                        "rgb_mildwb"):
        channels = np.stack([rgb[..., 0], rgb[..., 1], rgb[..., 2]], axis=0)
    else:
        # 3 RGB + mask + circular hue + S/V + Lab a/b = 10 channels.
        channels = np.stack([
            rgb[..., 0], rgb[..., 1], rgb[..., 2],
            mask.astype(np.float32) / 255. * 2. - 1.,
            np.sin(hue), np.cos(hue), sat, val, lab_a, lab_b,
        ], axis=0)
    channels[:3] = ((channels[:3] -
                     np.asarray([.485, .456, .406], np.float32)[:, None, None]) /
                    np.asarray([.229, .224, .225], np.float32)[:, None, None])
    return torch.from_numpy(np.ascontiguousarray(channels)).float()


class ProposalDS(Dataset):
    def __init__(self, samples: list[Sample], side: int, augment: bool,
                 feature_mode: str = "rich"):
        self.samples, self.side = samples, side
        self.augment, self.feature_mode = augment, feature_mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.feature_mode in ("rgb_grayworld", "rgb_mildwb"):
            image = read_color_corrected(sample.image_path, self.feature_mode)
            feature_mode = "rgb"
        else:
            image = read_image(sample.image_path)
            feature_mode = self.feature_mode
        x = crop_features(image, sample.box, self.side, self.augment,
                          feature_mode)
        return x, int(sample.label), index


class CachedProposalDS(Dataset):
    """Memory-mapped crop tensors with cheap tensor-side augmentation."""
    def __init__(self, path: Path, labels: np.ndarray, training: bool):
        self.x = np.load(path, mmap_mode="r")
        self.labels = np.asarray(labels, np.int64)
        self.training = training

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # A read-only memmap slice cannot safely be wrapped by torch when the
        # tensor-side augmentations below may mutate it.  Copying one sample
        # also avoids the undefined-behaviour warning emitted by torch.
        x = torch.from_numpy(np.array(self.x[index], copy=True)).float()
        if self.training:
            if torch.rand(()) < .5:
                x = torch.flip(x, (-1,))
            if torch.rand(()) < .25:
                x = torch.rot90(x, int(torch.randint(1, 4, ()).item()), (-2, -1))
            if torch.rand(()) < .7:
                # RGB channels are ImageNet-normalized; this small shift is
                # intentionally weaker than a hue rotation.
                gain = float(torch.empty(()).uniform_(.96, 1.04))
                x[:3] = x[:3] * gain
        return x, int(self.labels[index]), index


def materialize(samples: list[Sample], side: int, path: Path,
                workers: int, batch: int, feature_mode: str) -> np.ndarray:
    """Cache CPU crop conversion once so subsequent epochs are GPU-bound."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_channels = 3 if feature_mode in ("rgb", "rgb_clahe", "rgb_sharp",
                                       "rgb_grayworld", "rgb_mildwb") else 10
    cache = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float16,
        shape=(len(samples), n_channels, side, side))
    ds = ProposalDS(samples, side, False, feature_mode)
    dl = DataLoader(ds, batch_size=batch, shuffle=False,
                    num_workers=min(workers, 8), pin_memory=True,
                    persistent_workers=min(workers, 8) > 0)
    labels = np.empty(len(samples), np.int64)
    done = 0
    for x, y, index in dl:
        idx = index.numpy()
        cache[idx] = x.numpy().astype(np.float16)
        labels[idx] = y.numpy()
        done += len(idx)
        if done % max(batch * 10, 1) == 0 or done == len(samples):
            print(f"  cache {path.name}: {done}/{len(samples)}", flush=True)
    cache.flush()
    np.save(path.with_suffix(".labels.npy"), labels)
    return labels


class ProposalModel(nn.Module):
    def __init__(self, backbone: str, channels: int = 10,
                 freeze_backbone: bool = False):
        super().__init__()
        self.bb = timm.create_model(backbone, pretrained=True, num_classes=0,
                                    in_chans=channels)
        if freeze_backbone:
            for parameter in self.bb.parameters():
                parameter.requires_grad_(False)
        self.fc = nn.Linear(self.bb.num_features, K)

    def forward(self, x):
        return self.fc(self.bb(x))


def f1_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p.argmax(1)
    f1, rec = [], []
    for c in range(K):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
        rec.append(tp / max(int((y == c).sum()), 1))
    return {"n": int(len(y)), "accuracy": float((pred == y).mean()),
            "macro_f1": float(np.mean(f1)),
            "macro_recall": float(np.mean(rec)),
            "f1_per_class": dict(zip(base.NAMES, f1)),
            "recall_per_class": dict(zip(base.NAMES, rec))}


@torch.inference_mode()
def infer(model: nn.Module, ds: Dataset, device: str, batch: int,
          workers: int) -> tuple[np.ndarray, np.ndarray]:
    dl = DataLoader(ds, batch_size=batch, shuffle=False,
                    num_workers=min(workers, 8), pin_memory=True,
                    persistent_workers=min(workers, 8) > 0)
    probs, indices = [], []
    model.eval()
    for x, y, idx in dl:
        with torch.autocast("cuda"):
            p = torch.softmax(model(x.to(device, non_blocking=True)), 1)
        probs.append(p.float().cpu().numpy())
        indices.append(idx.numpy())
    return (np.concatenate(probs, 0) if probs else np.zeros((0, K)),
            np.concatenate(indices, 0) if indices else np.zeros(0, np.int64))


def save_probability_npz(path: Path, samples: list[Sample], probs: np.ndarray,
                         vote: dict[str, np.ndarray]) -> None:
    arrays = {key: np.asarray(rows, np.float32).copy()
              for key, rows in vote.items()}
    for sample, p in zip(samples, probs):
        arrays[sample.stem][sample.row_index, 5:5 + K] = p
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def main() -> int:
    global CTX
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("depth", "953"), required=True)
    ap.add_argument("--fused-root", type=Path,
                    default=Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27"))
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--backbone", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--feature-mode",
                    choices=("rich", "rgb", "rgb_clahe", "rgb_sharp",
                             "rgb_grayworld", "rgb_mildwb"),
                    default="rich")
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--sampling", choices=("balanced", "natural"),
                    default="balanced")
    ap.add_argument("--class-weight-power", type=float, default=0.0,
                    help="inverse-frequency loss weight power; 0 disables it")
    ap.add_argument("--img", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=40)
    # 160px ConvNeXt-Tiny with ten input channels fits comfortably on a 24GB
    # RTX 3090 at this batch size.  Larger batches (e.g. 768) OOM before the
    # first optimizer step.
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-backbone", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label-smoothing", type=float, default=.02)
    ap.add_argument("--focal-gamma", type=float, default=0.,
                    help="focal modulation gamma; 0 keeps cross-entropy")
    ap.add_argument("--init-checkpoint", type=Path, default=None,
                    help="optional crop-head checkpoint for a short fine-tune")
    ap.add_argument("--fit-val", action="store_true",
                    help="fit final model on TRAIN+VAL; VAL is diagnostic only")
    ap.add_argument("--match-iou", type=float, default=.5,
                    help="minimum proposal/GT IoU for a supervised crop label")
    ap.add_argument("--context", type=float, default=CTX,
                    help="crop window side / proposal side; larger adds context")
    ap.add_argument("--skip-export", action="store_true",
                    help="simpan checkpoint terbaik tanpa inferensi seluruh train/val/test; "
                         "gunakan evaluate_crop_tta.py untuk screening")
    ap.add_argument("--cache-root", type=Path, default=None,
                    help="reuse compatible materialized crop caches from another run")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, min(args.workers, os.cpu_count() or 1)))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA diperlukan untuk proposal crop head")
    if not .5 <= args.match_iou <= .95:
        raise ValueError("match-iou harus berada pada [.5,.95]")
    CTX = float(args.context)
    device = "cuda"
    cfg = base.CONFIGS["SawitMVC-Depth-YOLO" if args.dataset == "depth"
                       else "SawitMVC-YOLO"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    votes, samples, counts = {}, {}, {}
    for split in ("train", "val", "test"):
        path = vote_path(args.fused_root, args.dataset, split)
        if not path.exists():
            raise FileNotFoundError(path)
        votes[split] = load_vote(path)
        samples[split], counts[split] = build_samples(
            cfg, args.dataset, split, votes[split], split != "test",
            args.match_iou)
        print(json.dumps({"split": split, "samples": len(samples[split]),
                          "label_counts": counts[split]}, ensure_ascii=False),
              flush=True)

    train = [s for s in samples["train"] if s.label >= 0]
    val = [s for s in samples["val"] if s.label >= 0]
    if len(train) < 100 or len(val) < 20:
        raise RuntimeError("proposal labelled train/validation terlalu kecil")
    fit_samples = train + val if args.fit_val else train
    class_counts = np.bincount([s.label for s in fit_samples], minlength=K)
    sample_weights = (1. / np.maximum(class_counts, 1))[
        [s.label for s in fit_samples]]
    cache_tag = f"{args.img}_{args.feature_mode}_ctx{args.context:g}_iou{args.match_iou:g}"
    cache_root = args.cache_root or args.output_root
    cache_root.mkdir(parents=True, exist_ok=True)
    train_cache = cache_root / f"cache_train_{cache_tag}.npy"
    val_cache = cache_root / f"cache_val_{cache_tag}.npy"
    # Preserve/reuse caches produced before context/IoU became explicit CLI
    # knobs.  This is especially important for parallel sweeps: four workers
    # must not materialize four identical 5--6 GB RGB memmaps.
    if (args.context == 1.6 and args.match_iou == .5 and
            args.feature_mode in ("rgb", "rgb_clahe", "rgb_sharp")):
        legacy_train = cache_root / f"cache_train_{args.img}_{args.feature_mode}.npy"
        legacy_val = cache_root / f"cache_val_{args.img}_{args.feature_mode}.npy"
        if legacy_train.exists() and legacy_train.with_suffix(".labels.npy").exists():
            train_cache, val_cache = legacy_train, legacy_val
    train_labels_path = train_cache.with_suffix(".labels.npy")
    val_labels_path = val_cache.with_suffix(".labels.npy")
    if train_cache.exists() and train_labels_path.exists():
        train_labels = np.load(train_labels_path)
    else:
        train_labels = materialize(train, args.img, train_cache, args.workers,
                                   args.batch, args.feature_mode)
    if val_cache.exists() and val_labels_path.exists():
        val_labels = np.load(val_labels_path)
    else:
        val_labels = materialize(val, args.img, val_cache, args.workers,
                                 args.batch, args.feature_mode)
    sampler = None
    if args.fit_val:
        train_dataset = ConcatDataset([
            CachedProposalDS(train_cache, train_labels, True),
            CachedProposalDS(val_cache, val_labels, True),
        ])
        fit_labels = np.concatenate([train_labels, val_labels])
    else:
        train_dataset = CachedProposalDS(train_cache, train_labels, True)
        fit_labels = train_labels
    if args.sampling == "balanced":
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            len(fit_labels), replacement=True)
    dl_train = DataLoader(
        train_dataset, batch_size=args.batch,
        sampler=sampler, shuffle=sampler is None,
        num_workers=min(args.workers, 8), pin_memory=True, drop_last=True,
        persistent_workers=min(args.workers, 8) > 0)
    dl_val = DataLoader(CachedProposalDS(val_cache, val_labels, False),
                        batch_size=args.batch, shuffle=False,
                        num_workers=min(args.workers, 8), pin_memory=True,
                        persistent_workers=min(args.workers, 8) > 0)
    channels = 3 if args.feature_mode in ("rgb", "rgb_clahe", "rgb_sharp",
                                          "rgb_grayworld", "rgb_mildwb") else 10
    model = ProposalModel(args.backbone, channels, args.freeze_backbone).to(device)
    if args.init_checkpoint is not None:
        init = torch.load(args.init_checkpoint, map_location="cpu",
                          weights_only=False)
        state = init.get("model", init)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(json.dumps({"init_checkpoint": str(args.init_checkpoint),
                          "missing": len(missing),
                          "unexpected": len(unexpected)}), flush=True)
    parameter_groups = [{"params": model.fc.parameters(), "lr": args.lr}]
    if not args.freeze_backbone:
        parameter_groups.insert(0, {"params": model.bb.parameters(),
                                    "lr": args.lr_backbone})
    opt = torch.optim.AdamW(parameter_groups, weight_decay=.05)
    max_lr = [args.lr] if args.freeze_backbone else [args.lr_backbone, args.lr]
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lr,
        total_steps=args.epochs * max(len(dl_train), 1), pct_start=.2)
    scaler = torch.amp.GradScaler("cuda")
    loss_weight = None
    if args.class_weight_power:
        weights = np.power(
            np.maximum(class_counts, 1.) / max(class_counts.mean(), 1.),
            -args.class_weight_power)
        loss_weight = torch.as_tensor(weights, dtype=torch.float32,
                                      device=device)
    best, best_state, history = -1., None, []
    val_y = np.asarray([s.label for s in val], np.int64)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y, _idx in dl_train:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                logits = model(x)
                if args.focal_gamma > 0.:
                    ce = F.cross_entropy(logits, y, weight=loss_weight,
                                         label_smoothing=args.label_smoothing,
                                         reduction="none")
                    pt = torch.softmax(logits, 1).gather(1, y[:, None]).squeeze(1)
                    loss = (((1. - pt).clamp_min(1e-5) ** args.focal_gamma) * ce).mean()
                else:
                    loss = F.cross_entropy(logits, y, weight=loss_weight,
                                           label_smoothing=args.label_smoothing)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(float(loss.detach().cpu()))
        model.eval(); pv = []
        with torch.inference_mode():
            for x, _y, _idx in dl_val:
                with torch.autocast("cuda"):
                    pv.append(torch.softmax(model(x.to(device, non_blocking=True)), 1).float().cpu().numpy())
        p_val = np.concatenate(pv, 0)
        mv = f1_metrics(val_y, p_val)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **mv})
        if args.fit_val:
            # The validation labels are part of the final fit in this mode;
            # never use them to choose an epoch.  The last state is the
            # deterministic final-fit state, while mv remains diagnostic.
            best = mv["macro_f1"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        elif mv["macro_f1"] > best:
            best = mv["macro_f1"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({"epoch": epoch, "loss": float(np.mean(losses)),
                              "val": mv}, ensure_ascii=False), flush=True)
    if best_state is None:
        raise RuntimeError("proposal head tidak menghasilkan checkpoint")
    model.load_state_dict(best_state); model.to(device).eval()
    output = {
        "dataset": cfg["kind"], "source_dataset": args.dataset,
        "fit_split": "train+val" if args.fit_val else "train",
        "selection_split": "diagnostic_only" if args.fit_val else "val",
        "backbone": args.backbone, "img": args.img, "epochs": args.epochs,
        "batch": args.batch, "seed": args.seed, "channels": channels,
        "feature_mode": args.feature_mode,
        "freeze_backbone": args.freeze_backbone,
        "sampling": args.sampling,
        "class_weight_power": args.class_weight_power,
        "label_smoothing": args.label_smoothing,
        "match_iou": args.match_iou,
        "context": args.context,
        "focal_gamma": args.focal_gamma,
        "init_checkpoint": (str(args.init_checkpoint)
                             if args.init_checkpoint is not None else None),
        "fit_val": args.fit_val,
        "counts": counts, "best_val_macro_f1": best,
        "history": history,
    }
    checkpoint = args.output_root / "proposal_crop_head.pt"
    if args.skip_export:
        # The best state is already selected from validation above.  Keeping
        # this branch before the all-split export avoids an expensive pass over
        # 74k train proposals when only a validation screen is needed.
        output["val_metrics"] = max(history, key=lambda x: x["macro_f1"])
        torch.save({"model": model.state_dict(), "args": vars(args),
                    "meta": output}, checkpoint)
        (args.output_root / "metadata.json").write_text(
            json.dumps(output, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(json.dumps({"checkpoint": str(checkpoint), **output},
                         ensure_ascii=False), flush=True)
        return 0
    output["val_metrics"] = f1_metrics(
        val_y, infer(model, ProposalDS(val, args.img, False,
                                       args.feature_mode), device,
                     args.batch, args.workers)[0])
    torch.save({"model": model.state_dict(), "args": vars(args),
                "meta": output}, checkpoint)
    for split in ("train", "val", "test"):
        ds = ProposalDS(samples[split], args.img, False, args.feature_mode)
        probs, indices = infer(model, ds, device, args.batch, args.workers)
        ordered = np.zeros_like(probs)
        ordered[indices] = probs
        save_probability_npz(args.output_root / f"fused_{split}__wbf_softvote.npz",
                             samples[split], ordered, votes[split])
    (args.output_root / "metadata.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint), **output},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
