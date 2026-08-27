"""Train an ordinal RGB crop head for the B1--B4 maturity classes.

The backbone and proposal samples are the same as the regular crop head, but
the classifier learns the three ordered boundaries (B1|B2, B2|B3, B3|B4).
This is a train-only experiment; output probabilities keep the original
proposal geometry and are compatible with ``evaluate_remote_class_head.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

import train_proposal_crop_head as head


def ordinal_prob(logits: torch.Tensor) -> torch.Tensor:
    q = torch.sigmoid(logits)
    # Valid cumulative ordinal probabilities must decrease with the threshold.
    q = torch.cummin(q, dim=1).values
    p = torch.cat([1. - q[:, :1], q[:, :1] - q[:, 1:2],
                   q[:, 1:2] - q[:, 2:3], q[:, 2:3]], dim=1)
    return p.clamp_min(0.) / p.clamp_min(0.).sum(1, keepdim=True).clamp_min(1e-8)


class OrdinalModel(nn.Module):
    def __init__(self, backbone: str):
        super().__init__()
        self.bb = head.timm.create_model(backbone, pretrained=True,
                                         num_classes=0, in_chans=3)
        self.fc = nn.Linear(self.bb.num_features, 3)

    def forward(self, x):
        return self.fc(self.bb(x))


@torch.inference_mode()
def predict(model, ds, device: str, batch: int, workers: int):
    dl = DataLoader(ds, batch_size=batch, shuffle=False,
                    num_workers=min(workers, 8), pin_memory=True,
                    persistent_workers=min(workers, 8) > 0)
    out, ids = [], []
    model.eval()
    for x, _y, idx in dl:
        with torch.autocast("cuda"):
            p = ordinal_prob(model(x.to(device, non_blocking=True)))
        out.append(p.float().cpu().numpy()); ids.append(idx.numpy())
    if not out:
        return np.zeros((0, head.K), np.float32), np.zeros(0, np.int64)
    return np.concatenate(out), np.concatenate(ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("depth", "953"), required=True)
    ap.add_argument("--fused-root", type=Path,
                    default=Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27"))
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--backbone", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--sampling", choices=("balanced", "natural"), default="natural")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-backbone", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA diperlukan")
    device = "cuda"
    cfg = head.base.CONFIGS["SawitMVC-Depth-YOLO" if args.dataset == "depth"
                            else "SawitMVC-YOLO"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    votes, samples, counts = {}, {}, {}
    for split in ("train", "val", "test"):
        vote_path = head.vote_path(args.fused_root, args.dataset, split)
        votes[split] = head.load_vote(vote_path)
        samples[split], counts[split] = head.build_samples(
            cfg, args.dataset, split, votes[split], split != "test")
        print(json.dumps({"split": split, "samples": len(samples[split]),
                          "label_counts": counts[split]}, ensure_ascii=False),
              flush=True)
    train = [s for s in samples["train"] if s.label >= 0]
    val = [s for s in samples["val"] if s.label >= 0]
    class_counts = np.bincount([s.label for s in train], minlength=head.K)
    tag = f"{args.img}_rgb"
    train_cache = args.cache_root / f"cache_train_{tag}.npy"
    val_cache = args.cache_root / f"cache_val_{tag}.npy"
    if not (train_cache.exists() and train_cache.with_suffix(".labels.npy").exists()):
        head.materialize(train, args.img, train_cache, args.workers,
                         args.batch, "rgb")
    if not (val_cache.exists() and val_cache.with_suffix(".labels.npy").exists()):
        head.materialize(val, args.img, val_cache, args.workers,
                         args.batch, "rgb")
    train_labels = np.load(train_cache.with_suffix(".labels.npy"))
    val_labels = np.load(val_cache.with_suffix(".labels.npy"))
    train_ds = head.CachedProposalDS(train_cache, train_labels, True)
    val_ds = head.CachedProposalDS(val_cache, val_labels, False)
    sampler = None
    if args.sampling == "balanced":
        weights = (1. / np.maximum(class_counts, 1))[train_labels]
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                        len(train_labels), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch,
                           sampler=sampler, shuffle=sampler is None,
                           num_workers=min(args.workers, 8), pin_memory=True,
                           drop_last=True, persistent_workers=min(args.workers, 8) > 0)
    model = OrdinalModel(args.backbone).to(device)
    opt = torch.optim.AdamW([
        {"params": model.bb.parameters(), "lr": args.lr_backbone},
        {"params": model.fc.parameters(), "lr": args.lr},
    ], weight_decay=.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr_backbone, args.lr],
        total_steps=args.epochs * max(len(train_dl), 1), pct_start=.2)
    scaler = torch.amp.GradScaler("cuda")
    best, best_state, history = -1., None, []
    val_y = val_labels.astype(np.int64)
    thresholds = torch.arange(1, head.K, device=device).float()[None, :]
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for x, y, _idx in train_dl:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            target = (y[:, None].float() >= thresholds).float()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                loss = F.binary_cross_entropy_with_logits(model(x), target)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            losses.append(float(loss.detach().cpu()))
        pv, _ = predict(model, val_ds, device, args.batch, args.workers)
        mv = head.f1_metrics(val_y, pv)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **mv})
        if mv["macro_f1"] > best:
            best = mv["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps({"epoch": epoch, "loss": float(np.mean(losses)), "val": mv},
                         ensure_ascii=False), flush=True)
    if best_state is None:
        raise RuntimeError("ordinal head tidak menghasilkan checkpoint")
    model.load_state_dict(best_state); model.to(device).eval()
    meta = {"dataset": cfg["kind"], "source_dataset": args.dataset,
            "fit_split": "train", "selection_split": "val",
            "backbone": args.backbone, "img": args.img, "epochs": args.epochs,
            "batch": args.batch, "seed": args.seed, "sampling": args.sampling,
            "objective": "three ordinal BCE thresholds", "counts": counts,
            "best_val_macro_f1": best,
            "val_metrics": head.f1_metrics(val_y, predict(model, val_ds, device,
                                                             args.batch, args.workers)[0]),
            "history": history}
    checkpoint = args.output_root / "ordinal_crop_head.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "meta": meta}, checkpoint)
    for split in ("train", "val", "test"):
        ds = head.ProposalDS(samples[split], args.img, False, "rgb")
        probs, indices = predict(model, ds, device, args.batch, args.workers)
        ordered = np.zeros_like(probs); ordered[indices] = probs
        head.save_probability_npz(args.output_root / f"fused_{split}__wbf_softvote.npz",
                                  samples[split], ordered, votes[split])
    (args.output_root / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint), **meta}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
