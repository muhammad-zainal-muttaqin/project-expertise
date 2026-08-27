"""Run rotation/flip TTA for a trained proposal crop classifier.

The proposal geometry and scores are copied unchanged; only B1--B4
probabilities are replaced by an average over a selected D4 transform set.
This keeps the experiment isolated from localization, linking, and counting.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_proposal_crop_head as regular
import train_ordinal_crop_head as ordinal


def transforms(x: torch.Tensor, mode: str) -> list[torch.Tensor]:
    if mode in ("gamma095", "gamma105", "brightness095", "brightness105",
                "contrast105", "contrast095"):
        mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
        std = x.new_tensor([.229, .224, .225])[None, :, None, None]
        rgb = (x[:, :3] * std + mean).clamp(0, 1)
        if mode.startswith("gamma"):
            gamma = .95 if mode == "gamma095" else 1.05
            rgb = rgb.pow(gamma)
        elif mode.startswith("brightness"):
            gain = .95 if mode == "brightness095" else 1.05
            rgb = (rgb * gain).clamp(0, 1)
        else:
            gain = 1.05 if mode == "contrast105" else .95
            rgb = ((rgb - .5) * gain + .5).clamp(0, 1)
        out = x.clone()
        out[:, :3] = (rgb - mean) / std
        return [out]
    if mode == "none":
        return [x]
    if mode == "hflip":
        return [x, torch.flip(x, (-1,))]
    if mode == "rot4":
        return [torch.rot90(x, k, (-2, -1)) for k in range(4)]
    # Square crops permit the eight dihedral symmetries.  This includes
    # rotations and reflected rotations without changing proposal geometry.
    return [
        x, torch.flip(x, (-1,)), torch.flip(x, (-2,)),
        torch.rot90(x, 2, (-2, -1)),
        torch.rot90(x, 1, (-2, -1)),
        torch.rot90(torch.flip(x, (-1,)), 1, (-2, -1)),
        torch.rot90(x, 3, (-2, -1)),
        torch.rot90(torch.flip(x, (-1,)), 3, (-2, -1)),
    ]


@torch.inference_mode()
def predict(model, ds, device: str, batch: int, workers: int,
            mode: str, ordinal_mode: bool):
    dl = DataLoader(ds, batch_size=batch, shuffle=False,
                    num_workers=min(workers, 8), pin_memory=True,
                    persistent_workers=min(workers, 8) > 0)
    values, indices = [], []
    model.eval()
    for x, _y, idx in dl:
        votes = []
        for tx in transforms(x, mode):
            with torch.autocast("cuda"):
                raw = model(tx.to(device, non_blocking=True))
                p = (ordinal.ordinal_prob(raw) if ordinal_mode
                     else torch.softmax(raw, 1))
            votes.append(p.float())
        values.append(torch.stack(votes).mean(0).cpu().numpy())
        indices.append(idx.numpy())
    if not values:
        return np.zeros((0, regular.K), np.float32), np.zeros(0, np.int64)
    return np.concatenate(values), np.concatenate(indices)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("depth", "953"), required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--fused-root", type=Path,
                    default=Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27"))
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--split", choices=("val", "test"), default="val",
                     help="split receiving TTA; train probabilities are copied with no TTA")
    ap.add_argument("--feature-mode", default=None,
                    choices=("rgb", "rgb_grayworld", "rgb_mildwb",
                             "rgb_clahe", "rgb_sharp"),
                     help="optional input transform override for a 3-channel checkpoint")
    ap.add_argument("--context", type=float, default=None,
                    help="crop context; defaults to the checkpoint setting")
    ap.add_argument("--match-iou", type=float, default=None,
                    help="supervised match threshold; defaults to checkpoint setting")
    ap.add_argument("--tta", choices=("none", "hflip", "rot4", "d4", "gamma095",
                                       "gamma105", "brightness095", "brightness105",
                                       "contrast095", "contrast105"), default="d4")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA diperlukan")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ck_args = ckpt["args"]
    model_args = ck_args if "backbone" in ck_args else ckpt.get("backbone_args", ck_args)
    regular.CTX = float(args.context if args.context is not None
                        else ck_args.get("context", regular.CTX))
    match_iou = float(args.match_iou if args.match_iou is not None
                      else ck_args.get("match_iou", .5))
    ordinal_mode = "ordinal" in args.checkpoint.name or "ordinal" in str(args.checkpoint.parent)
    if ordinal_mode:
        model = ordinal.OrdinalModel(model_args["backbone"]).cuda()
    else:
        model = regular.ProposalModel(
            model_args["backbone"], model_args.get("channels", 3),
            model_args.get("freeze_backbone", False)).cuda()
    model.load_state_dict(ckpt["model"])
    model.eval()
    cfg = regular.base.CONFIGS[
        "SawitMVC-Depth-YOLO" if args.dataset == "depth" else "SawitMVC-YOLO"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    votes, samples = {}, {}
    for split in ("train", "val", "test"):
        path = regular.vote_path(args.fused_root, args.dataset, split)
        votes[split] = regular.load_vote(path)
        samples[split], _ = regular.build_samples(
            cfg, args.dataset, split, votes[split], split != "test", match_iou)

    # The training cache contains only labelled proposals.  For TTA we infer
    # every proposal in the selected split so unmatched proposals retain a
    # valid head probability too; this also preserves exact NPZ row alignment.
    safe = "SawitMVC_Depth_YOLO" if args.dataset == "depth" else "SawitMVC_YOLO"
    train_src = regular.vote_path(args.fused_root, args.dataset, "train")
    shutil.copy2(train_src,
                 args.output_root / "fused_train__wbf_softvote.npz")
    split = args.split
    feature_mode = args.feature_mode or ck_args.get("feature_mode", "rgb")
    ds = regular.ProposalDS(samples[split], ck_args["img"], False,
                             feature_mode)
    p, order = predict(model, ds, "cuda", args.batch, args.workers,
                       args.tta, ordinal_mode)
    full = np.zeros((len(samples[split]), regular.K), np.float32)
    full[order] = p
    regular.save_probability_npz(
        args.output_root / f"fused_{split}__wbf_softvote.npz",
        samples[split], full, votes[split])
    print(json.dumps({"split": split, "tta": args.tta,
                      "n_inferred": int(len(p)), "safe": safe},
                     ensure_ascii=False), flush=True)
    meta = {"dataset": args.dataset, "checkpoint": str(args.checkpoint),
            "tta": args.tta, "split": args.split,
            "feature_mode": feature_mode,
            "context": regular.CTX, "match_iou": match_iou,
            "train_no_tta": True, "ordinal": ordinal_mode}
    (args.output_root / "metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
