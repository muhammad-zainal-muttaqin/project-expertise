"""Fit a train-only group-level class decision layer.

The four-view linker already turns several per-image proposals into one
physical bunch candidate.  This script learns the final B1--B4 decision from
that candidate's detector vote, optional visual crop-head vote, view
consistency, confidence, and normalized geometry.  Linking and count
reconciliation remain fixed, so a classification gain is isolated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_remote_pipeline_postprocess as base  # noqa: E402
import evaluate_remote_count_reconciled as count  # noqa: E402
import evaluate_remote_class_head as post  # noqa: E402
import sweep_remote_pipeline as sweep  # noqa: E402


K = len(base.NAMES)


def load_vote(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key], np.float32) for key in archive.files}


def vote_path(folder: Path, dataset: str) -> Path:
    safe = "SawitMVC_Depth_YOLO" if dataset == "depth" else "SawitMVC_YOLO"
    return folder / f"{safe}__wbf_softvote.npz"


def member_stats(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return [0.] * (4 * K)
    return np.concatenate([
        values.mean(0), values.max(0), values.min(0), values.std(0),
    ]).astype(float).tolist()


def group_features(group: dict, rec: dict) -> np.ndarray:
    members = group["members"]
    bp = np.stack([post.normalise(x["p"]) for x in members])
    hp = np.stack([post.normalise(x["head_p"]) for x in members])
    scores = np.asarray([x["score"] for x in members], float)
    geom = np.asarray([[x["cx"], x["cy"], x["w"], x["h"]]
                       for x in members], float)
    sides = sorted({int(x["side"]) for x in members})
    feat = [
        *post.normalise(group["p"]).tolist(),
        *np.average(hp, axis=0, weights=np.maximum(scores, 1e-6)).tolist(),
        *np.log(np.maximum(post.normalise(group["p"]), 1e-6)).tolist(),
        *np.log(np.maximum(np.average(hp, axis=0,
                                      weights=np.maximum(scores, 1e-6)), 1e-6)).tolist(),
        *member_stats(bp), *member_stats(hp),
        float(scores.mean()), float(scores.max()), float(scores.min()),
        float(scores.std()), float(len(members)),
        *[float(i in sides) for i in range(rec["n_sides"])],
        *geom.mean(0).tolist(), *geom.std(0).tolist(),
    ]
    return np.asarray(feat, np.float32)


def make_payload(cfg: dict, dataset: str, split: str, base_vote: dict,
                 head_vote: dict, prior: dict, proposal_min: float,
                 pair_mode: str, link_threshold: float,
                 singleton_min: float, max_size: int):
    records = count.four_side(base.load_records(cfg, split))
    payload = []
    for rec in records.values():
        dets = post.make_detections(rec, base_vote, head_vote, proposal_min)
        edges = sweep.build_edges(dets, rec["n_sides"], prior, pair_mode)
        groups = sweep.clusters(dets, edges, link_threshold, singleton_min, max_size)
        payload.append((rec, dets, edges, groups))
    return payload


def matched_training_data(payload: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for rec, _dets, _edges, groups in payload:
        matches = count.tree_matches(rec, groups)
        for i, j in matches:
            cls = int(rec["bunches"][j]["cls"])
            if 0 <= cls < K:
                features.append(group_features(groups[i], rec))
                labels.append(cls)
    return np.asarray(features, np.float32), np.asarray(labels, np.int64)


class LinearMeta(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.fc = nn.Linear(n_features, K)

    def forward(self, x):
        return self.fc(x)


def fit_model(x: np.ndarray, y: np.ndarray, power: float,
              epochs: int, seed: int) -> tuple[LinearMeta, np.ndarray, np.ndarray]:
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1.
    z = torch.from_numpy((x - mean) / scale)
    target = torch.from_numpy(y)
    counts = np.bincount(y, minlength=K).astype(np.float32)
    weights = np.power(np.maximum(counts, 1.) / max(counts.mean(), 1.), -power)
    weight = torch.from_numpy(weights)
    torch.manual_seed(seed)
    model = LinearMeta(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=.03, weight_decay=.03)
    best_loss = float("inf")
    best_state = None
    for _ in range(epochs):
        model.train()
        loss = F.cross_entropy(model(z), target, weight=weight)
        opt.zero_grad(set_to_none=True)
        loss.backward(); opt.step()
        if float(loss) < best_loss:
            best_loss = float(loss)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, mean, scale


@torch.inference_mode()
def predict(model: LinearMeta, mean: np.ndarray, scale: np.ndarray,
            x: np.ndarray) -> np.ndarray:
    model.eval()
    return torch.softmax(model(torch.from_numpy((x - mean) / scale)), 1).numpy()


def evaluate_meta(payload: list[tuple], target_counts: dict[str, int],
                  model: LinearMeta, mean: np.ndarray, scale: np.ndarray,
                  link_threshold: float, singleton_min: float, max_size: int,
                  rank_mode: str, class_prior: np.ndarray,
                  prior_exponent: float) -> dict:
    cm = np.zeros((K + 1, K + 1), int)
    total_pred = total_gt = total_tp = 0
    abs_count = exact = pm1 = vector_exact = 0
    correct = matched = 0
    for rec, dets, edges, _groups in payload:
        target = target_counts[rec["tree_id"]]
        groups = count.selected_clusters(dets, edges, link_threshold,
                                         singleton_min, max_size, target,
                                         rank_mode)
        if groups:
            probs = predict(model, mean, scale,
                            np.stack([group_features(g, rec) for g in groups]))
        else:
            probs = np.zeros((0, K), np.float32)
        for group, p in zip(groups, probs):
            if prior_exponent:
                p = post.normalise(p * np.power(np.maximum(class_prior, 1e-9),
                                                prior_exponent))
            group["cls"] = int(np.argmax(p))
        matches = count.tree_matches(rec, groups)
        bunches = rec["bunches"]
        total_pred += len(groups); total_gt += len(bunches); total_tp += len(matches)
        delta = len(groups) - len(bunches)
        abs_count += abs(delta); exact += int(delta == 0); pm1 += int(abs(delta) <= 1)
        pred_count = np.bincount([g["cls"] for g in groups], minlength=K)
        gt_count = np.bincount([b["cls"] for b in bunches if 0 <= b["cls"] < K],
                               minlength=K)
        vector_exact += int(np.array_equal(pred_count, gt_count))
        matched += len(matches)
        matched_pred = {i for i, _ in matches}; matched_gt = {j for _, j in matches}
        for i, j in matches:
            pc, gc = groups[i]["cls"], bunches[j]["cls"]
            if 0 <= pc < K and 0 <= gc < K:
                cm[pc, gc] += 1; correct += int(pc == gc)
        for i, g in enumerate(groups):
            if i not in matched_pred and 0 <= g["cls"] < K:
                cm[g["cls"], K] += 1
        for j, b in enumerate(bunches):
            if j not in matched_gt and 0 <= b["cls"] < K:
                cm[K, b["cls"]] += 1
    precision = total_tp / max(total_pred, 1); recall = total_tp / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    f1s = []
    for c in range(K):
        tp = cm[c, c]; fp = int(cm[c].sum() - tp); fn = int(cm[:, c].sum() - tp)
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {"physical_detection": {"precision": precision, "recall": recall,
                                    "f1": f1, "tp": total_tp,
                                    "pred_clusters": total_pred,
                                    "gt_bunches": total_gt},
            "counting": {"mae": abs_count / max(len(payload), 1),
                         "exact_accuracy": exact / max(len(payload), 1),
                         "plus_minus_1_accuracy": pm1 / max(len(payload), 1),
                         "vector_exact_accuracy": vector_exact / max(len(payload), 1)},
            "classification": {
                "matched_class_accuracy": correct / max(matched, 1),
                "matched": matched, "macro_f1_end_to_end": float(np.mean(f1s)),
                "per_class_f1_end_to_end": dict(zip(base.NAMES, f1s)),
                "confusion_prediction_rows": cm.tolist(),
            }}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("depth", "953"), required=True)
    ap.add_argument("--base-train-dir", type=Path, required=True)
    ap.add_argument("--base-val-dir", type=Path, required=True)
    ap.add_argument("--head-dir", type=Path, required=True)
    ap.add_argument("--proposal-min", type=float, default=.075)
    ap.add_argument("--link-threshold", type=float, default=.25)
    ap.add_argument("--singleton-min", type=float, default=.15)
    ap.add_argument("--max-size", type=int, default=3)
    ap.add_argument("--pair-mode", choices=("all", "adjacent"), default="adjacent")
    ap.add_argument("--rank-mode", choices=("score", "support", "max_member", "class_conf"),
                    default="support")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--class-weight-powers", nargs="+", type=float, default=[0., .5, 1.])
    ap.add_argument("--prior-exponents", nargs="+", type=float, default=[-.25, 0.])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    name = "SawitMVC-Depth-YOLO" if args.dataset == "depth" else "SawitMVC-YOLO"
    cfg = base.CONFIGS[name]
    train_vote = load_vote(vote_path(args.base_train_dir, args.dataset))
    val_vote = load_vote(vote_path(args.base_val_dir, args.dataset))
    head_train = load_vote(args.head_dir / "fused_train__wbf_softvote.npz")
    head_val = load_vote(args.head_dir / "fused_val__wbf_softvote.npz")
    train_records = count.four_side(base.load_records(cfg, "train"))
    prior_rotation = base.build_rotation_prior(base.load_records(cfg, "train"))
    train_payload = make_payload(cfg, args.dataset, "train", train_vote, head_train,
                                 prior_rotation, args.proposal_min, args.pair_mode,
                                 args.link_threshold, args.singleton_min, args.max_size)
    val_payload = make_payload(cfg, args.dataset, "val", val_vote, head_val,
                               prior_rotation, args.proposal_min, args.pair_mode,
                               args.link_threshold, args.singleton_min, args.max_size)
    x_train, y_train = matched_training_data(train_payload)
    if len(x_train) < 100:
        raise RuntimeError("terlalu sedikit group train yang match")
    # Count reconciliation uses detector-only features and train-only fit.
    y_count = np.asarray([count.target_count(r) for r in train_records.values()], float)
    x_count = np.stack([count.feature_vector(r, train_vote, args.proposal_min)
                        for r in train_records.values()])
    alpha, cv = count.choose_alpha(x_count, y_count)
    count_model = count.fit_ridge(x_count, y_count, alpha)
    val_records = count.four_side(base.load_records(cfg, "val"))
    x_val_count = np.stack([count.feature_vector(r, val_vote, args.proposal_min)
                            for r in val_records.values()])
    target_counts = {tree_id: int(n) for tree_id, n in zip(
        val_records, count.predict_count(count_model, x_val_count))}
    class_prior = np.bincount(y_train, minlength=K).astype(float)
    class_prior /= max(class_prior.sum(), 1.)
    trials = []
    best = None
    for power in args.class_weight_powers:
        model, mean, scale = fit_model(x_train, y_train, power, args.epochs, 42)
        for exponent in args.prior_exponents:
            metrics = evaluate_meta(
                val_payload, target_counts, model, mean, scale,
                args.link_threshold, args.singleton_min, args.max_size,
                args.rank_mode, class_prior, exponent)
            item = {"class_weight_power": power, "prior_exponent": exponent,
                    "metrics": metrics}
            trials.append(item)
            print(json.dumps({"class_weight_power": power,
                              "prior_exponent": exponent,
                              "physical_f1": metrics["physical_detection"]["f1"],
                              "count_mae": metrics["counting"]["mae"],
                              "matched_class_accuracy": metrics["classification"]["matched_class_accuracy"],
                              "macro_f1": metrics["classification"]["macro_f1_end_to_end"]},
                             ensure_ascii=False), flush=True)
            key = (metrics["classification"]["macro_f1_end_to_end"],
                   metrics["classification"]["matched_class_accuracy"])
            if best is None or key > best["key"]:
                best = {"key": key, "item": item, "model": model,
                        "mean": mean, "scale": scale}
    if best is None:
        raise RuntimeError("meta sweep kosong")
    checkpoint = args.output.with_suffix(".pt")
    torch.save({"model": best["model"].state_dict(), "mean": best["mean"],
                "scale": best["scale"], "n_features": x_train.shape[1],
                "dataset": name}, checkpoint)
    output = {"protocol": "train-only group-level class meta-head",
              "dataset": name, "fit_split": "train", "selection_split": "val",
              "proposal_min": args.proposal_min, "link_threshold": args.link_threshold,
              "singleton_min": args.singleton_min, "max_size": args.max_size,
              "pair_mode": args.pair_mode, "rank_mode": args.rank_mode,
              "train_groups_matched": int(len(y_train)),
              "train_class_counts": np.bincount(y_train, minlength=K).tolist(),
              "count_model_alpha": alpha, "count_cv": cv,
              "class_prior": class_prior.tolist(), "n_features": int(x_train.shape[1]),
              "trials": trials, "best": best["item"], "checkpoint": str(checkpoint)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
