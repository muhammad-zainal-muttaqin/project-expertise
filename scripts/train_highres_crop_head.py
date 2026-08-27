"""Fine-tune the proposal crop head at a higher spatial resolution.

This is an on-the-fly variant of ``train_proposal_crop_head.py``.  It avoids
materializing multi-gigabyte high-resolution memmaps, initializes from the
validation-selected 224px head, and writes only compact probability dumps.
The detector geometry/linker are untouched.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_proposal_crop_head as crop  # noqa: E402
import eval_remote_pipeline_postprocess as base  # noqa: E402


K = len(base.NAMES)


class LabeledSubset(Dataset):
    def __init__(self, samples, indices, side, feature_mode):
        self.samples = samples
        self.indices = np.asarray(indices, int)
        self.ds = crop.ProposalDS(samples, side, True, feature_mode)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x, y, _ = self.ds[int(self.indices[index])]
        return x, y


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p.argmax(1)
    f1, rec = [], []
    for c in range(K):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
        rec.append(tp / max(int((y == c).sum()), 1))
    return {"n": int(len(y)), "accuracy": float((pred == y).mean()),
            "macro_f1": float(np.mean(f1)), "macro_recall": float(np.mean(rec)),
            "f1_per_class": dict(zip(base.NAMES, f1)),
            "recall_per_class": dict(zip(base.NAMES, rec))}


@torch.inference_mode()
def infer(model, samples, side, feature_mode, batch, workers):
    ds = crop.ProposalDS(samples, side, False, feature_mode)
    dl = DataLoader(ds, batch_size=batch, shuffle=False,
                    num_workers=min(workers, 8), pin_memory=True,
                    persistent_workers=min(workers, 8) > 0)
    out = []
    model.eval()
    for x, _y, _idx in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.softmax(model(x.cuda(non_blocking=True)), 1).float().cpu().numpy())
    return np.concatenate(out, 0)


def load_vote(path: Path):
    with np.load(path) as z:
        return {k: np.asarray(z[k], np.float32) for k in z.files}


def vote_path(root: Path, split: str) -> Path:
    safe = "SawitMVC_YOLO"
    if split == "test":
        normal = root / "fused_combined1716" / f"{safe}__wbf_softvote.npz"
        rebuilt = root / "fused_combined1716_test_rebuilt" / f"{safe}__wbf_softvote.npz"
        return normal if normal.exists() else rebuilt
    return root / f"fused_combined1716_{split}" / f"{safe}__wbf_softvote.npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-checkpoint", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--fused-root", type=Path,
                    default=Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27"))
    ap.add_argument("--img", type=int, default=320)
    ap.add_argument("--context", type=float, default=1.6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--lr-backbone", type=float, default=8e-6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA diperlukan")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(args.workers, os.cpu_count() or 1))
    crop.CTX = float(args.context)
    args.output_root.mkdir(parents=True, exist_ok=True)
    cfg = base.CONFIGS["SawitMVC-YOLO"]
    votes, samples = {}, {}
    for split in ("train", "val", "test"):
        path = vote_path(args.fused_root, split)
        if not path.exists(): raise FileNotFoundError(path)
        votes[split] = load_vote(path)
        samples[split], _ = crop.build_samples(
            cfg, "953", split, votes[split], split != "test")
        print(json.dumps({"split": split, "samples": len(samples[split])},
                         ensure_ascii=False), flush=True)
    train_idx = np.asarray([i for i, s in enumerate(samples["train"])
                            if s.label >= 0], int)
    val_idx = np.asarray([i for i, s in enumerate(samples["val"])
                          if s.label >= 0], int)
    y_train = np.asarray([samples["train"][i].label for i in train_idx], int)
    y_val = np.asarray([samples["val"][i].label for i in val_idx], int)
    freq = np.bincount(y_train, minlength=K).astype(float)
    class_w = np.sqrt(freq.max() / np.maximum(freq, 1.))
    sampler_w = class_w[y_train]
    ds_train = LabeledSubset(samples["train"], train_idx, args.img, "rgb")
    dl = DataLoader(ds_train, batch_size=args.batch, sampler=WeightedRandomSampler(
        torch.from_numpy(sampler_w).double(), len(sampler_w), replacement=True),
        num_workers=min(args.workers, 8), pin_memory=True, drop_last=True,
        persistent_workers=min(args.workers, 8) > 0)
    ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    ck_args = ckpt["args"]
    model = crop.ProposalModel(ck_args["backbone"], ck_args.get("channels", 3),
                               ck_args.get("freeze_backbone", False)).cuda()
    model.load_state_dict(ckpt["model"])
    backbone = [p for p in model.bb.parameters() if p.requires_grad]
    head = list(model.fc.parameters())
    opt = torch.optim.AdamW([
        {"params": backbone, "lr": args.lr_backbone},
        {"params": head, "lr": args.lr},
    ], weight_decay=2e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs,
                                                      eta_min=args.lr / 20.)
    cw = torch.tensor(class_w, device="cuda", dtype=torch.float32)
    # Validation is streamed every epoch; no high-resolution cache is written.
    val_ds = crop.ProposalDS(samples["val"], args.img, False, "rgb")
    val_dl = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False,
                        num_workers=min(args.workers, 8), pin_memory=True,
                        persistent_workers=min(args.workers, 8) > 0)
    best, best_state, stale, history = -math.inf, None, 0, []
    for ep in range(1, args.epochs + 1):
        model.train(); total = 0.; nb = 0
        for x, y in dl:
            x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.float(), y, weight=cw,
                                       label_smoothing=.02)
                p = torch.softmax(logits.float(), 1)
                loss = loss + .10 * F.smooth_l1_loss(
                    p @ torch.arange(K, device="cuda", dtype=torch.float32), y.float())
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
            total += float(loss.detach()); nb += 1
        sch.step()
        model.eval(); vp = []
        for x, _y, idx in val_dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vp.append(torch.softmax(model(x.cuda(non_blocking=True)), 1).float().detach().cpu().numpy())
        all_v = np.concatenate(vp, 0)
        met = metrics(y_val, all_v[val_idx])
        row = {"epoch": ep, "loss": total / max(nb, 1), **met}
        history.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
        score = met["macro_f1"]
        if score > best + 1e-8:
            best, stale = score, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience: break
    if best_state is None: raise RuntimeError("tidak ada checkpoint valid")
    model.load_state_dict(best_state); torch.save(
        {"model": model.state_dict(), "args": vars(args),
         "backbone_args": ck_args, "best_val_macro_f1": best,
         "val_metrics": history[-1], "history": history},
        args.output_root / "proposal_crop_head.pt")
    # Export all proposal rows for the fixed post-cluster evaluator.
    for split in ("train", "val", "test"):
        p = infer(model, samples[split], args.img, "rgb", args.batch * 2, args.workers)
        crop.save_probability_npz(args.output_root / f"fused_{split}__wbf_softvote.npz",
                                  samples[split], p, votes[split])
    meta = {"dataset": "yolo953_adapter", "fit_split": "train",
            "selection_split": "val", "img": args.img, "context": args.context,
            "batch": args.batch, "epochs": args.epochs, "backbone": ck_args["backbone"],
            "init_checkpoint": str(args.init_checkpoint), "best_val_macro_f1": best,
            "val_metrics": history[-1], "history": history}
    (args.output_root / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"best_val_macro_f1": best, "output": str(args.output_root)},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
