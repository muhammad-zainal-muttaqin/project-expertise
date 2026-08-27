"""Train a late-fusion RGB + explicit colour/statistics proposal head.

The RGB backbone keeps ImageNet pretrained weights.  A small branch consumes
the existing HSV/Lab/histogram/geometry/detector-probability features from the
colour head and is fused only before the B1--B4 classifier.  This avoids
replacing the pretrained first convolution with random extra channels.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import train_proposal_crop_head as rgb_head
import train_fused_color_head as colour


class CachedMultiDS(Dataset):
    def __init__(self, rgb_path: Path, labels: np.ndarray, colour_x: np.ndarray):
        self.rgb = np.load(rgb_path, mmap_mode="r")
        self.labels = np.asarray(labels, np.int64)
        self.colour = np.asarray(colour_x, np.float32)
        if len(self.rgb) != len(self.labels) or len(self.colour) != len(self.labels):
            raise ValueError("RGB cache, labels, dan colour features tidak sejajar")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = torch.from_numpy(np.array(self.rgb[index], copy=True)).float()
        if torch.rand(()) < .5:
            x = torch.flip(x, (-1,))
        if torch.rand(()) < .25:
            x = torch.rot90(x, int(torch.randint(1, 4, ()).item()), (-2, -1))
        if torch.rand(()) < .7:
            x[:3] = x[:3] * float(torch.empty(()).uniform_(.96, 1.04))
        return x, torch.from_numpy(self.colour[index]), int(self.labels[index]), index


class AllMultiDS(Dataset):
    def __init__(self, samples, colour_x: np.ndarray, side: int,
                 feature_mode: str = "rgb"):
        self.samples = samples
        self.colour = np.asarray(colour_x, np.float32)
        self.side = side
        self.feature_mode = feature_mode
        if len(self.samples) != len(self.colour):
            raise ValueError("samples dan colour features tidak sejajar")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = rgb_head.read_image(sample.image_path)
        x = rgb_head.crop_features(image, sample.box, self.side, False,
                                   self.feature_mode)
        return x, torch.from_numpy(self.colour[index]), int(sample.label), index


class MultiModalModel(nn.Module):
    def __init__(self, backbone: str, colour_dim: int,
                 visual_checkpoint: Path | None = None):
        super().__init__()
        self.bb = rgb_head.timm.create_model(
            backbone, pretrained=True, num_classes=0, in_chans=3)
        colour_dim_hidden = 128
        self.colour = nn.Sequential(
            nn.Linear(colour_dim, colour_dim_hidden), nn.LayerNorm(colour_dim_hidden),
            nn.GELU(), nn.Dropout(.15), nn.Linear(colour_dim_hidden, 64),
            nn.LayerNorm(64), nn.GELU())
        self.fc = nn.Linear(self.bb.num_features + 64, rgb_head.K)
        if visual_checkpoint is not None and visual_checkpoint.exists():
            ckpt = torch.load(visual_checkpoint, map_location="cpu",
                              weights_only=False)
            state = ckpt.get("model", ckpt)
            bb_state = {k[3:]: v for k, v in state.items()
                        if k.startswith("bb.")}
            missing, _unexpected = self.bb.load_state_dict(bb_state, strict=False)
            if missing:
                print(json.dumps({"visual_init_missing": len(missing)}), flush=True)

    def forward(self, x, c):
        v = self.bb(x)
        z = self.colour(c)
        return self.fc(torch.cat([v, z], 1))


@torch.inference_mode()
def predict(model, ds, device: str, batch: int, workers: int):
    dl = DataLoader(ds, batch_size=batch, shuffle=False,
                    num_workers=min(workers, 8), pin_memory=True,
                    persistent_workers=min(workers, 8) > 0)
    out, ids = [], []
    model.eval()
    for x, c, _y, idx in dl:
        with torch.autocast("cuda"):
            out.append(torch.softmax(model(x.to(device, non_blocking=True),
                                          c.to(device, non_blocking=True)), 1)
                        .float().cpu().numpy())
        ids.append(idx.numpy())
    return np.concatenate(out), np.concatenate(ids)


def evaluate(model, ds, y, device: str, batch: int, workers: int):
    p, ids = predict(model, ds, device, batch, workers)
    full = np.zeros((len(y), rgb_head.K), np.float32)
    full[ids] = p
    return rgb_head.f1_metrics(y, full), full


def collect_or_load(cfg, dataset, split, vote, cache_path, workers):
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            return np.asarray(z["X"], np.float32), np.asarray(z["y"], np.int64)
    X, y, meta = colour.collect_split(cfg, split, vote, workers)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, meta=json.dumps(meta))
    return X, y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("depth", "953"), required=True)
    ap.add_argument("--fused-root", type=Path,
                    default=Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27"))
    ap.add_argument("--rgb-cache-root", type=Path, required=True)
    ap.add_argument("--feature-cache-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--visual-checkpoint", type=Path, default=None)
    ap.add_argument("--backbone", default="convnext_small.fb_in22k_ft_in1k")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA diperlukan")
    cfg = rgb_head.base.CONFIGS["SawitMVC-Depth-YOLO" if args.dataset == "depth"
                                else "SawitMVC-YOLO"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    votes, samples = {}, {}
    features = {}
    for split in ("train", "val"):
        vote = rgb_head.load_vote(rgb_head.vote_path(args.fused_root,
                                                      args.dataset, split))
        votes[split] = vote
        samples[split], _ = rgb_head.build_samples(
            cfg, args.dataset, split, vote, True)
        features[split] = collect_or_load(
            cfg, args.dataset, split, vote,
            args.feature_cache_root / f"features_{split}.npz", args.workers)
        print(json.dumps({"split": split, "rows": len(features[split][1]),
                          "samples": len(samples[split])}, ensure_ascii=False),
              flush=True)

    # The colour collector follows the exact proposal-row order.  RGB caches
    # contain only labelled rows, matching y>=0 filtering in the crop trainer.
    X_train_all, y_train_all = features["train"]
    X_val_all, y_val_all = features["val"]
    train_keep = (y_train_all >= 0) & (y_train_all < rgb_head.K)
    val_keep = (y_val_all >= 0) & (y_val_all < rgb_head.K)
    proposal_train = [s for s in samples["train"] if s.label >= 0]
    proposal_val = [s for s in samples["val"] if s.label >= 0]
    if len(proposal_train) != int(train_keep.sum()) or len(proposal_val) != int(val_keep.sum()):
        raise ValueError("colour feature label alignment berbeda dengan proposal cache")
    train_labels = np.load(
        (args.rgb_cache_root / f"cache_train_{args.img}_rgb.npy")
        .with_suffix(".labels.npy"))
    val_labels = np.load(
        (args.rgb_cache_root / f"cache_val_{args.img}_rgb.npy")
        .with_suffix(".labels.npy"))
    if not np.array_equal(train_labels, y_train_all[train_keep]):
        raise ValueError("train RGB/colour labels mismatch")
    if not np.array_equal(val_labels, y_val_all[val_keep]):
        raise ValueError("val RGB/colour labels mismatch")
    mean = X_train_all[train_keep].mean(0).astype(np.float32)
    scale = X_train_all[train_keep].std(0).astype(np.float32)
    scale[scale < 1e-5] = 1.
    X_train = ((X_train_all[train_keep] - mean) / scale).astype(np.float32)
    X_val = ((X_val_all - mean) / scale).astype(np.float32)
    train_cache = args.rgb_cache_root / f"cache_train_{args.img}_rgb.npy"
    val_cache = args.rgb_cache_root / f"cache_val_{args.img}_rgb.npy"
    train_ds = CachedMultiDS(train_cache, train_labels, X_train)
    val_ds = CachedMultiDS(val_cache, val_labels, X_val[val_keep])
    all_val_ds = AllMultiDS(samples["val"], X_val, args.img, "rgb")
    model = MultiModalModel(args.backbone, X_train.shape[1],
                            args.visual_checkpoint).cuda()
    dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                    num_workers=min(args.workers, 8), pin_memory=True,
                    drop_last=True, persistent_workers=min(args.workers, 8) > 0)
    opt = torch.optim.AdamW([
        {"params": model.bb.parameters(), "lr": args.lr_backbone},
        {"params": model.colour.parameters(), "lr": args.lr},
        {"params": model.fc.parameters(), "lr": args.lr},
    ], weight_decay=.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr_backbone, args.lr, args.lr],
        total_steps=max(args.epochs * len(dl), 1), pct_start=.2)
    scaler = torch.amp.GradScaler("cuda")
    best, best_state, history = -1., None, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for x, c, y, _idx in dl:
            x = x.cuda(non_blocking=True); c = c.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                loss = F.cross_entropy(model(x, c), y, label_smoothing=.02)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            losses.append(float(loss.detach().cpu()))
        mv, _ = evaluate(model, val_ds, val_labels, "cuda",
                         args.batch, args.workers)
        item = {"epoch": epoch, "loss": float(np.mean(losses)), **mv}
        history.append(item)
        if mv["macro_f1"] > best:
            best = mv["macro_f1"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        print(json.dumps(item, ensure_ascii=False), flush=True)
    if best_state is None:
        raise RuntimeError("multimodal head tidak menghasilkan checkpoint")
    model.load_state_dict(best_state); model.cuda().eval()
    # Produce a complete validation proposal file; train file is only needed
    # as an existence/compatibility input for the E2E evaluator.
    p_val, ids = predict(model, all_val_ds, "cuda", args.batch, args.workers)
    full = np.zeros((len(samples["val"]), rgb_head.K), np.float32); full[ids] = p_val
    rgb_head.save_probability_npz(
        args.output_root / "fused_val__wbf_softvote.npz",
        samples["val"], full, votes["val"])
    shutil.copy2(rgb_head.vote_path(args.fused_root, args.dataset, "train"),
                 args.output_root / "fused_train__wbf_softvote.npz")
    meta = {"dataset": cfg["kind"], "source_dataset": args.dataset,
            "backbone": args.backbone, "img": args.img,
            "colour_features": int(X_train.shape[1]),
            "best_val_macro_f1": best, "history": history,
            "visual_checkpoint": (str(args.visual_checkpoint)
                                   if args.visual_checkpoint else None)}
    torch.save({"model": model.state_dict(), "mean": mean, "scale": scale,
                "args": {"backbone": args.backbone, "img": args.img,
                         "feature_mode": "rgb", "channels": 3},
                "meta": meta}, args.output_root / "multimodal_crop_head.pt")
    (args.output_root / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_root), **meta},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
