"""GPU group-level residual class head, TRAIN/VAL only.

Earlier heads classified individual members and pooled their opinions.  This
module instead learns directly from a frozen physical cluster, allowing a
small attention block to model cross-view agreement while keeping the
detector probability as an explicit residual/skip anchor.

Only matched TRAIN clusters provide labels.  The frozen linker, target count,
and cluster selection are not changed, and this module refuses any split other
than TRAIN or VAL.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, "/workspace/cluster_head")
import harness  # noqa: E402


ROOT = Path("/workspace/cluster_head")
OUT = ROOT / "artifacts"
K = harness.K
MAX_MEMBERS = 4


def group_key(group: dict) -> tuple:
    return tuple(sorted((int(m["side"]), str(m["stem"]), int(m["row_index"]))
                        for m in group["members"]))


def load_map(dataset: str, split: str):
    with np.load(f"/workspace/dino_head/crops/{dataset}/{split}_index.npz",
                 allow_pickle=True) as z:
        stems = z["stem"].astype(str)
        rows = np.asarray(z["row_index"], dtype=np.int64)
    feat = np.load(f"/workspace/dino_head/features_large/{dataset}/{split}_dinolargefeat.npy",
                   mmap_mode="r")
    if len(stems) != len(feat):
        raise RuntimeError(f"DINO-Large index mismatch {dataset}/{split}")
    return {(str(s), int(r)): np.asarray(feat[i], dtype=np.float32)
            for i, (s, r) in enumerate(zip(stems, rows))}, int(feat.shape[1])


def member_vector(member: dict, fmap: dict, dim: int):
    f = fmap.get((str(member["stem"]), int(member["row_index"])),
                 np.zeros(dim, dtype=np.float32))
    p = np.asarray(member["p"], dtype=np.float32)
    side = np.zeros(4, dtype=np.float32)
    if 0 <= int(member["side"]) < 4:
        side[int(member["side"])] = 1.0
    scalars = np.asarray([
        float(member["score"]), float(member["cx"]), float(member["cy"]),
        float(member["w"]), float(member["h"]),
        float(member.get("rank_cx", 0.)), float(member.get("rank_cy", 0.)),
        float(member.get("z_side_x", 0.)), float(member.get("z_side_y", 0.)),
        float(member.get("z_side_area", 0.)), float(member.get("side_count", 1.)),
    ], dtype=np.float32)
    return np.concatenate([f, p, side, scalars]), int(member["side"])


def collect(dataset: str, split: str) -> dict:
    if split not in ("train", "val"):
        raise ValueError("this experiment accepts only train or val")
    records, payload, targets, _prior = harness.build_payload(dataset, split)
    groups = harness.make_groups(payload, targets, harness.PROFILES[dataset])
    fmap, dim = load_map(dataset, split)
    feature_dim = dim + K + 4 + 11
    xs, masks, side_ids, score_weights = [], [], [], []
    detectors, contexts, labels, keys, flat = [], [], [], [], []
    for rec, tree_groups in groups:
        matches = dict(harness.count.tree_matches(rec, tree_groups))
        for gi, group in enumerate(tree_groups):
            x = np.zeros((MAX_MEMBERS, feature_dim), dtype=np.float32)
            mask = np.zeros(MAX_MEMBERS, dtype=bool)
            sides = np.zeros(MAX_MEMBERS, dtype=np.int64)
            weights = np.zeros(MAX_MEMBERS, dtype=np.float32)
            for mi, member in enumerate(group["members"][:MAX_MEMBERS]):
                x[mi], sides[mi] = member_vector(member, fmap, dim)
                mask[mi] = True
                weights[mi] = max(float(member["score"]), 1e-6)
            weights /= max(float(weights.sum()), 1e-8)
            p = np.asarray(group["p"], dtype=np.float32)
            p = np.maximum(p, 1e-8)
            p /= max(float(p.sum()), 1e-8)
            entropy = float(-(p * np.log(p)).sum())
            sp = np.sort(p)
            contexts.append(np.asarray([
                float(group["score"]), float(len(group["members"])),
                float(weights.max()), float(weights[mask].mean()) if mask.any() else 0.,
                float(entropy), float(sp[-1] - sp[-2]),
            ], dtype=np.float32))
            xs.append(x); masks.append(mask); side_ids.append(sides)
            score_weights.append(weights); detectors.append(p)
            labels.append(int(rec["bunches"][matches[gi]]["cls"]) if gi in matches else -1)
            keys.append(group_key(group)); flat.append(group)
    return {
        "dataset": dataset, "split": split, "records": records,
        "payload": payload, "targets": targets, "groups": groups,
        "X": np.asarray(xs, dtype=np.float32),
        "mask": np.asarray(masks, dtype=bool),
        "side": np.asarray(side_ids, dtype=np.int64),
        "member_weights": np.asarray(score_weights, dtype=np.float32),
        "detector": np.asarray(detectors, dtype=np.float32),
        "context": np.asarray(contexts, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64), "keys": keys, "flat": flat,
        "feature_dim": feature_dim, "dino_dim": dim,
    }


class GroupResidualHead(nn.Module):
    def __init__(self, feature_dim: int, context_dim: int = 6,
                 hidden: int = 256):
        super().__init__()
        self.member = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden),
            nn.GELU(), nn.Dropout(.10),
        )
        # Match the member projection width so the view identity can be added
        # as a residual signal without a broadcast or dimensionality shortcut.
        self.side = nn.Embedding(4, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
            dropout=.10, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.context = nn.Sequential(
            nn.LayerNorm(context_dim + K), nn.Linear(context_dim + K, 64),
            nn.GELU(),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden * 2 + 64), nn.Linear(hidden * 2 + 64, 128),
            nn.GELU(), nn.Dropout(.10), nn.Linear(128, K),
        )

    def forward(self, x, mask, sides, weights, detector, context):
        h = self.member(x) + self.side(sides)
        h = self.encoder(h, src_key_padding_mask=~mask)
        w = weights.unsqueeze(-1) * mask.unsqueeze(-1)
        mean = (h * w).sum(1) / w.sum(1).clamp_min(1e-6)
        hmax = h.masked_fill(~mask.unsqueeze(-1), -1e4).amax(1)
        ctx = self.context(torch.cat([context, torch.log(detector.clamp_min(1e-8))], 1))
        residual = self.out(torch.cat([mean, hmax, ctx], 1))
        logits = torch.log(detector.clamp_min(1e-8)) + residual
        return logits, residual


def make_loader(data: dict, indices: np.ndarray, batch_size: int,
                shuffle: bool) -> DataLoader:
    tensors = [
        torch.from_numpy(data["X"][indices]),
        torch.from_numpy(data["mask"][indices]),
        torch.from_numpy(data["side"][indices]),
        torch.from_numpy(data["member_weights"][indices]),
        torch.from_numpy(data["detector"][indices]),
        torch.from_numpy(data["context"][indices]),
        torch.from_numpy(data["labels"][indices]),
    ]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, pin_memory=True)


def train_model(train: dict, seed: int, epochs: int, batch_size: int,
                device: torch.device, progress_path: Path):
    torch.manual_seed(seed)
    np.random.seed(seed)
    matched = np.flatnonzero(train["labels"] >= 0)
    if len(matched) < 20:
        raise RuntimeError("not enough matched TRAIN groups")
    counts = np.bincount(train["labels"][matched], minlength=K).astype(np.float32)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.))
    weights *= K / max(float(weights.sum()), 1e-8)
    model = GroupResidualHead(train["feature_dim"]).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-4)
    loader = make_loader(train, matched, batch_size, True)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses, correct = [], 0
        for batch in loader:
            x, mask, sides, mw, det, context, y = [z.to(device, non_blocking=True)
                                                    for z in batch]
            optim.zero_grad(set_to_none=True)
            logits, _residual = model(x, mask, sides, mw, det, context)
            loss = F.cross_entropy(logits, y, weight=torch.as_tensor(
                weights, device=device), label_smoothing=.025)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optim.step()
            losses.append(float(loss.detach().cpu()))
            correct += int((logits.argmax(1) == y).sum().detach().cpu())
        row = {"epoch": epoch, "loss": float(np.mean(losses)),
               "train_accuracy": correct / max(len(matched), 1)}
        history.append(row)
        progress_path.write_text(json.dumps({"status": "running", "history": history},
                                            indent=2) + "\n", encoding="utf-8")
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(json.dumps(row), flush=True)
    return model, history, counts.tolist()


@torch.no_grad()
def predict(model, data: dict, device: torch.device, batch_size: int = 512):
    model.eval()
    all_logits = []
    indices = np.arange(len(data["labels"]))
    loader = make_loader(data, indices, batch_size, False)
    for batch in loader:
        x, mask, sides, mw, det, context, _y = [z.to(device, non_blocking=True)
                                                 for z in batch]
        logits, _residual = model(x, mask, sides, mw, det, context)
        all_logits.append(logits.float().cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def short(m: dict) -> dict:
    return {
        "physical_f1": m["physical_detection"]["f1"],
        "mae": m["counting"]["mae"],
        "pm1": m["counting"]["plus_minus_1_accuracy"],
        "matched_class_accuracy": m["classification"]["matched_class_accuracy"],
        "matched": m["classification"]["matched"],
        "macro_f1": m["classification"]["macro_f1_end_to_end"],
        "per_class_f1": m["classification"]["per_class_f1_end_to_end"],
    }


def evaluate(data: dict, logits: np.ndarray, mode: str, alpha: float) -> dict:
    detector = np.maximum(data["detector"], 1e-8)
    if mode == "head":
        z = logits
    elif mode == "blend":
        z = np.log(detector) + alpha * (logits - np.log(detector))
    else:
        raise ValueError(mode)
    pred = np.argmax(z, axis=1).astype(int)
    pmap = {key: int(cls) for key, cls in zip(data["keys"], pred)}
    m = harness.evaluate_clusters(
        data["payload"], data["targets"], harness.PROFILES[data["dataset"]],
        lambda group, pmap=pmap: pmap[group_key(group)])
    return short(m)


def run(dataset: str, seed: int, epochs: int, batch_size: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    started = time.time()
    torch.set_num_threads(2)
    device = torch.device("cuda")
    train, val = collect(dataset, "train"), collect(dataset, "val")
    progress = OUT / f"{dataset}_gpu_group_head_progress.json"
    model, history, class_counts = train_model(
        train, seed, epochs, batch_size, device, progress)
    state_path = OUT / f"{dataset}_gpu_group_head.pt"
    torch.save({"state_dict": model.state_dict(), "feature_dim": train["feature_dim"],
                "seed": seed, "epochs": epochs, "class_counts": class_counts}, state_path)
    val_logits = predict(model, val, device)
    baseline = short(harness.evaluate_clusters(
        val["payload"], val["targets"], harness.PROFILES[dataset]))
    rows = []
    for mode, alpha in (("head", 1.0), ("blend", .15), ("blend", .25),
                        ("blend", .50), ("blend", .75), ("blend", 1.0)):
        rows.append({"mode": mode, "alpha": alpha,
                     "metrics": evaluate(val, val_logits, mode, alpha)})
    eligible = [r for r in rows if (
        abs(r["metrics"]["physical_f1"] - baseline["physical_f1"]) < 1e-10
        and abs(r["metrics"]["mae"] - baseline["mae"]) < 1e-10
        and abs(r["metrics"]["pm1"] - baseline["pm1"]) < 1e-10)]
    best = max(eligible, key=lambda r: (r["metrics"]["matched_class_accuracy"],
                                        r["metrics"]["macro_f1"]))
    report = {
        "dataset": dataset,
        "protocol": "group-level residual attention head; fit TRAIN; evaluate VAL; no TEST",
        "seed": seed, "epochs": epochs, "batch_size": batch_size,
        "feature_dim": train["feature_dim"], "dino_dim": train["dino_dim"],
        "train_groups": int(len(train["labels"])),
        "train_matched_groups": int((train["labels"] >= 0).sum()),
        "val_groups": int(len(val["labels"])),
        "class_counts_train": class_counts, "history": history,
        "baseline_val": baseline, "results": rows,
        "selected_validation": best, "state_path": str(state_path),
        "elapsed_sec": time.time() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{dataset}_gpu_group_head_results.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    progress.write_text(json.dumps({"status": "done", "report": str(path),
                                    "history": history}, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({"dataset": dataset, "selected_validation": best,
                      "report": str(path), "seconds": report["elapsed_sec"]},
                     ensure_ascii=False), flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("953", "depth"), required=True)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    run(args.dataset, args.seed, args.epochs, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
