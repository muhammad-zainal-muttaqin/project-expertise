"""GPU residual MLP member classifier over DINOv2-Large features.

TRAIN/VAL only.  The model keeps detector probabilities and proposal
metadata as a direct skip connection while a nonlinear residual bottleneck
learns visual corrections.  Physical linking/counting remain frozen.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import harness
import large_member_head as lm


OUT = Path("/workspace/cluster_head/artifacts")
K = harness.K


class ResidualMemberNet(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.visual = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.skip = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden))
        self.refine = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, hidden),
                                    nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden, K)

    def forward(self, x):
        z = self.visual(x) + self.skip(x)
        return self.head(z + self.refine(z))


def short(m):
    return {"physical_f1": m["physical_detection"]["f1"],
            "mae": m["counting"]["mae"],
            "pm1": m["counting"]["plus_minus_1_accuracy"],
            "matched_class_accuracy": m["classification"]["matched_class_accuracy"],
            "matched": m["classification"]["matched"],
            "macro_f1": m["classification"]["macro_f1_end_to_end"],
            "per_class_f1": m["classification"]["per_class_f1_end_to_end"]}


def pool(q, data, pooling):
    flat = [g for _rec, gs in data["groups"] for g in gs]
    out = []
    for gi, rows in enumerate(data["group_rows"]):
        z = q[rows]
        members = flat[gi]["members"]
        if pooling == "mean":
            w = np.asarray([float(m["score"]) for m in members], dtype=np.float32)
            w /= max(float(w.sum()), 1e-8)
            out.append((z * w[:, None]).sum(0))
        elif pooling == "max":
            out.append(z.max(0))
        else:
            j = int(np.argmax([float(m["score"]) for m in members]))
            out.append(z[j])
    return np.asarray(out, dtype=np.float32)


def evaluate_predictions(data, q_member, detector, baseline, dataset):
    rows = []
    for pooling in ("mean", "max", "top"):
        q = pool(q_member, data, pooling)
        for alpha in (.05, .10, .15, .20, .30, .45, .60, .80, 1.0):
            z = np.log(np.maximum(detector, 1e-8)) + alpha * np.log(np.maximum(q, 1e-8))
            pred = np.argmax(z, axis=1).astype(int)
            pmap = {k: int(c) for k, c in zip(data["keys"], pred)}
            m = harness.evaluate_clusters(
                data["payload"], data["targets"], harness.PROFILES[dataset],
                lambda g, pmap=pmap: pmap[lm.key(g)])
            s = short(m)
            s["physical_count_invariant"] = bool(
                abs(s["physical_f1"] - baseline["physical_f1"]) < 1e-10
                and abs(s["mae"] - baseline["mae"]) < 1e-10
                and abs(s["pm1"] - baseline["pm1"]) < 1e-10)
            rows.append({"pooling": pooling, "alpha": alpha, "metrics": s})
    return rows


def run(dataset: str, seed: int, device: str, epochs: int):
    started = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    train, val = lm.collect(dataset, "train"), lm.collect(dataset, "val")
    mask = train["y"] >= 0
    x_train = torch.as_tensor(train["X"], dtype=torch.float32)
    y_train = torch.as_tensor(train["y"], dtype=torch.long)
    x_val = torch.as_tensor(val["X"], dtype=torch.float32)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)
    detector = np.asarray([np.asarray(g["p"], dtype=np.float32)
                           for _rec, gs in val["groups"] for g in gs])
    detector = np.maximum(detector, 1e-8)
    detector /= detector.sum(axis=1, keepdims=True)
    baseline = short(harness.evaluate_clusters(
        val["payload"], val["targets"], harness.PROFILES[dataset]))
    counts = np.bincount(train["y"][mask], minlength=K).astype(np.float32)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights /= weights.mean()
    dataset_tensor = TensorDataset(x_train[mask], y_train[mask])
    loader = DataLoader(dataset_tensor, batch_size=512, shuffle=True,
                        num_workers=0, pin_memory=dev.type == "cuda")
    configs = [("residual_mlp", 512, .15), ("residual_mlp_compact", 320, .10)]
    rows = []
    best_checkpoint = None
    for name, hidden, dropout in configs:
        model = ResidualMemberNet(x_train.shape[1], hidden, dropout).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
        loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, device=dev))
        model.train()
        for epoch in range(1, epochs + 1):
            running = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                running += float(loss.detach()) * len(yb)
            if epoch in {epochs // 2, epochs}:
                model.eval()
                with torch.no_grad():
                    logits = []
                    for start in range(0, len(x_val), 1024):
                        logits.append(model(x_val[start:start + 1024].to(dev)).cpu())
                q_member = torch.softmax(torch.cat(logits), dim=1).numpy()
                model_rows = evaluate_predictions(val, q_member, detector, baseline, dataset)
                rows.extend([{"model": name, "epoch": epoch, **r} for r in model_rows])
                print(json.dumps({"dataset": dataset, "model": name, "epoch": epoch,
                                  "loss": running / max(len(dataset_tensor), 1),
                                  "best": max(rows, key=lambda r:(r["metrics"]["matched_class_accuracy"],
                                                                     r["metrics"]["macro_f1"]))},
                                 ensure_ascii=False), flush=True)
                model.train()
        eligible = [r for r in rows if r["metrics"]["physical_count_invariant"]]
        current = max(eligible, key=lambda r:(r["metrics"]["matched_class_accuracy"],
                                               r["metrics"]["macro_f1"]))
        if best_checkpoint is None or (current["metrics"]["matched_class_accuracy"],
                                       current["metrics"]["macro_f1"]) > (
                                           best_checkpoint["row"]["metrics"]["matched_class_accuracy"],
                                           best_checkpoint["row"]["metrics"]["macro_f1"]):
            best_checkpoint = {"row": current, "state": model.state_dict(),
                               "hidden": hidden, "dropout": dropout}
    eligible = [r for r in rows if r["metrics"]["physical_count_invariant"]]
    best_match = max(eligible, key=lambda r:(r["metrics"]["matched_class_accuracy"],
                                              r["metrics"]["macro_f1"]))
    best_macro = max(eligible, key=lambda r:(r["metrics"]["macro_f1"],
                                               r["metrics"]["matched_class_accuracy"]))
    if best_checkpoint is not None:
        torch.save({"state_dict": best_checkpoint["state"],
                    "input_dim": int(x_train.shape[1]),
                    "hidden": best_checkpoint["hidden"],
                    "dropout": best_checkpoint["dropout"],
                    "dataset": dataset}, OUT / f"{dataset}_residual_mlp.pt")
    report = {"dataset": dataset,
              "protocol": "fit residual MLP on matched TRAIN members; epoch/opinion selected VAL; no TEST",
              "device": str(dev), "epochs": epochs,
              "train_members": int(mask.sum()), "val_members": int(len(val["X"])),
              "input_dim": int(x_train.shape[1]), "class_counts": counts.tolist(),
              "baseline_val": baseline, "best_by_matched": best_match,
              "best_by_macro": best_macro, "results": rows,
              "elapsed_sec": time.time() - started}
    out = OUT / f"{dataset}_residual_mlp_results.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({"dataset": dataset, "best_by_matched": best_match,
                      "best_by_macro": best_macro, "report": str(out)},
                     ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("953", "depth"), required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    run(args.dataset, args.seed, args.device, args.epochs)


if __name__ == "__main__":
    main()
