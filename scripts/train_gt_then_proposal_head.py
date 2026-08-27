"""Pretrain a crop head on clean GT boxes, then adapt it to detector proposals.

The proposal head remains the deployed model.  GT crops are used only as a
short train-only warm start; validation is always measured on labelled WBF
proposals, and no validation/test annotations enter either optimizer stage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

import train_proposal_crop_head as head


def gt_samples(cfg: dict, dataset: str, split: str):
    samples = []
    records = head.base.load_records(cfg, split)
    for tree_id, rec in records.items():
        for side, view in rec["views"].items():
            boxes, labels = head.annotations(view)
            path = head.image_path(cfg, split, view["filename"])
            for i, (box, label) in enumerate(zip(boxes, labels)):
                if 0 <= int(label) < head.K:
                    samples.append(head.Sample(
                        stem=f"{tree_id}_{side}_{i}", row_index=i,
                        image_path=str(path),
                        box=tuple(float(v) for v in box), label=int(label)))
    return samples


@torch.inference_mode()
def predict(model, ds, device: str, batch: int, workers: int):
    dl = DataLoader(ds, batch_size=batch, shuffle=False,
                    num_workers=min(workers, 8), pin_memory=True,
                    persistent_workers=min(workers, 8) > 0)
    out, ids = [], []
    model.eval()
    for x, _y, idx in dl:
        with torch.autocast("cuda"):
            out.append(torch.softmax(model(x.to(device, non_blocking=True)), 1)
                        .float().cpu().numpy())
        ids.append(idx.numpy())
    return np.concatenate(out), np.concatenate(ids)


def evaluate(model, ds, y, device: str, batch: int, workers: int):
    p, order = predict(model, ds, device, batch, workers)
    full = np.zeros((len(y), head.K), np.float32)
    full[order] = p
    return head.f1_metrics(y, full), full


def run_stage(model, ds, labels, val_ds, val_y, device, args,
              epochs, balanced, lr, lr_backbone):
    sampler = None
    counts = np.bincount(labels, minlength=head.K)
    if balanced:
        weights = (1. / np.maximum(counts, 1))[labels]
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double), len(labels),
            replacement=True)
    dl = DataLoader(ds, batch_size=args.batch, sampler=sampler,
                    shuffle=sampler is None, num_workers=min(args.workers, 8),
                    pin_memory=True, drop_last=True,
                    persistent_workers=min(args.workers, 8) > 0)
    opt = torch.optim.AdamW([
        {"params": model.bb.parameters(), "lr": lr_backbone},
        {"params": model.fc.parameters(), "lr": lr},
    ], weight_decay=.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[lr_backbone, lr],
        total_steps=max(epochs * len(dl), 1), pct_start=.2)
    scaler = torch.amp.GradScaler("cuda")
    history, best, best_state = [], -1., None
    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        for x, y, _idx in dl:
            x = x.to("cuda", non_blocking=True)
            y = y.to("cuda", non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                loss = F.cross_entropy(model(x), y, label_smoothing=.02)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(float(loss.detach().cpu()))
        mv, _ = evaluate(model, val_ds, val_y, "cuda",
                         args.batch, args.workers)
        item = {"epoch": epoch, "loss": float(np.mean(losses)), **mv}
        history.append(item)
        if mv["macro_f1"] > best:
            best = mv["macro_f1"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        print(json.dumps({"stage": args._stage, **item}, ensure_ascii=False),
              flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("depth", "953"), required=True)
    ap.add_argument("--fused-root", type=Path,
                    default=Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27"))
    ap.add_argument("--proposal-cache-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--backbone", default="convnext_small.fb_in22k_ft_in1k")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--gt-epochs", type=int, default=3)
    ap.add_argument("--proposal-epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA diperlukan")
    args.output_root.mkdir(parents=True, exist_ok=True)
    cfg = head.base.CONFIGS["SawitMVC-Depth-YOLO" if args.dataset == "depth"
                            else "SawitMVC-YOLO"]
    safe = "SawitMVC_Depth_YOLO" if args.dataset == "depth" else "SawitMVC_YOLO"
    val_vote = head.load_vote(head.vote_path(args.fused_root, args.dataset, "val"))
    val_samples, _ = head.build_samples(cfg, args.dataset, "val", val_vote, True)
    proposal_train_vote = head.load_vote(
        head.vote_path(args.fused_root, args.dataset, "train"))
    proposal_samples, _ = head.build_samples(
        cfg, args.dataset, "train", proposal_train_vote, True)
    proposal_train = [s for s in proposal_samples if s.label >= 0]
    val = [s for s in val_samples if s.label >= 0]
    gt = gt_samples(cfg, args.dataset, "train")
    print(json.dumps({"gt_samples": len(gt), "proposal_train": len(proposal_train),
                      "proposal_val": len(val)}, ensure_ascii=False), flush=True)

    args.proposal_cache_root.mkdir(parents=True, exist_ok=True)
    gt_cache = args.proposal_cache_root / f"cache_gt_{args.img}_rgb.npy"
    gt_labels_path = gt_cache.with_suffix(".labels.npy")
    if gt_cache.exists() and gt_labels_path.exists():
        gt_labels = np.load(gt_labels_path)
    else:
        gt_labels = head.materialize(gt, args.img, gt_cache, args.workers,
                                     args.batch, "rgb")
    tag = f"{args.img}_rgb"
    train_cache = args.proposal_cache_root / f"cache_train_{tag}.npy"
    val_cache = args.proposal_cache_root / f"cache_val_{tag}.npy"
    for path in (train_cache, val_cache):
        if not path.exists() or not path.with_suffix(".labels.npy").exists():
            raise FileNotFoundError(path)
    train_labels = np.load(train_cache.with_suffix(".labels.npy"))
    val_labels = np.load(val_cache.with_suffix(".labels.npy"))
    gt_ds = head.CachedProposalDS(gt_cache, gt_labels, True)
    proposal_ds = head.CachedProposalDS(train_cache, train_labels, True)
    val_ds = head.CachedProposalDS(val_cache, val_labels, False)
    model = head.ProposalModel(args.backbone, 3, False).cuda()
    histories = {}
    args._stage = "gt_pretrain"
    histories["gt_pretrain"], _ = run_stage(
        model, gt_ds, gt_labels, val_ds, val_labels, "cuda", args,
        args.gt_epochs, True, args.lr, args.lr_backbone)
    args._stage = "proposal_finetune"
    histories["proposal_finetune"], best = run_stage(
        model, proposal_ds, train_labels, val_ds, val_labels, "cuda", args,
        args.proposal_epochs, False, args.lr, args.lr_backbone)
    meta = {"dataset": cfg["kind"], "source_dataset": args.dataset,
            "backbone": args.backbone, "img": args.img,
            "gt_epochs": args.gt_epochs, "proposal_epochs": args.proposal_epochs,
            "gt_samples": len(gt), "proposal_train": len(proposal_train),
            "proposal_val": len(val), "best_val_macro_f1": best,
            "history": histories,
            "args": {k: str(v) if isinstance(v, Path) else v
                     for k, v in vars(args).items() if not k.startswith("_")}}
    checkpoint = args.output_root / "proposal_crop_head.pt"
    torch.save({"model": model.state_dict(),
                "args": {"backbone": args.backbone, "img": args.img,
                         "feature_mode": "rgb", "channels": 3,
                         "freeze_backbone": False},
                "meta": meta}, checkpoint)
    (args.output_root / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint), **meta},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
