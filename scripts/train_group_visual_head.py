"""Train a group-level visual classifier from multi-view crop embeddings.

The ordinary crop head classifies each proposal independently and then averages
its probabilities.  This experiment keeps the detector, WBF, linker, and
count-reconciliation path fixed, but pools the *pre-classifier visual
embeddings* from all views in one linked group.  A train-only classifier can
therefore learn which view is reliable and how disagreement between views
should be resolved.  Validation is used only for model/profile selection.

This is intentionally an isolated post-cluster layer: no box or score is
changed, so a failed classifier cannot improve its score by damaging
localisation or counting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_remote_pipeline_postprocess as base  # noqa: E402
import evaluate_remote_class_head as post  # noqa: E402
import evaluate_remote_count_reconciled as count  # noqa: E402
import sweep_remote_pipeline as sweep  # noqa: E402
import train_proposal_crop_head as crop  # noqa: E402


K = len(base.NAMES)


def load_vote(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key], np.float32)
                for key in archive.files}


def vote_path(root: Path, dataset: str, split: str) -> Path:
    safe = "SawitMVC_Depth_YOLO" if dataset == "depth" else "SawitMVC_YOLO"
    folder = root / ("fused_combined1716" if split == "test"
                     else f"fused_combined1716_{split}")
    path = folder / f"{safe}__wbf_softvote.npz"
    if path.exists():
        return path
    if split == "test":
        return (Path(__file__).resolve().parents[1] / "results" /
                "remote_eval_2026-08-27" / "fused_combined1716" /
                f"{safe}__wbf_softvote.npz")
    return path


def make_payload(cfg: dict, split: str, vote: dict[str, np.ndarray],
                 prior: dict, proposal_min: float, link_threshold: float,
                 singleton_min: float, max_size: int, pair_mode: str):
    records = count.four_side(base.load_records(cfg, split))
    payload = []
    for rec in records.values():
        dets = post.make_detections(rec, vote, vote, proposal_min)
        edges = sweep.build_edges(dets, rec["n_sides"], prior, pair_mode)
        groups = sweep.clusters(dets, edges, link_threshold,
                                singleton_min, max_size)
        payload.append((rec, dets, edges, groups))
    return records, payload


@torch.inference_mode()
def extract_embeddings(model: torch.nn.Module, samples: list[crop.Sample],
                       indices: list[int], side: int, workers: int,
                       batch: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract backbone embeddings and proposal-head probabilities by index."""
    ds = Subset(crop.ProposalDS(samples, side, False, "rgb"), indices)
    dl = DataLoader(ds, batch_size=batch, shuffle=False,
                    num_workers=min(workers, 8), pin_memory=True,
                    persistent_workers=min(workers, 8) > 0)
    embeddings, probabilities, orders = [], [], []
    model.eval()
    for x, _y, original_index in dl:
        with torch.autocast("cuda"):
            feature = model.bb(x.cuda(non_blocking=True))
            logits = model.fc(feature)
        embeddings.append(feature.float().cpu().numpy())
        probabilities.append(torch.softmax(logits, 1).float().cpu().numpy())
        orders.append(original_index.numpy())
    if not orders:
        return (np.zeros((0, model.bb.num_features), np.float32),
                np.zeros((0, K), np.float32))
    order = np.concatenate(orders)
    emb = np.concatenate(embeddings)[np.argsort(order)]
    prob = np.concatenate(probabilities)[np.argsort(order)]
    return emb.astype(np.float32), prob.astype(np.float32)


def group_feature(group: dict, embedding_map: dict[tuple[str, int], np.ndarray],
                  probability_map: dict[tuple[str, int], np.ndarray],
                  mode: str) -> np.ndarray:
    members = group["members"]
    keys = [(str(x["stem"]), int(x["row_index"])) for x in members]
    emb = np.stack([embedding_map[k] for k in keys]).astype(np.float32)
    hp = np.stack([probability_map[k] for k in keys]).astype(np.float32)
    scores = np.asarray([max(float(x["score"]), 1e-6) for x in members],
                        np.float32)
    weights = scores / max(float(scores.sum()), 1e-6)
    mean = (emb * weights[:, None]).sum(0)
    pmean = (hp * weights[:, None]).sum(0)
    base_p = post.normalise(group["p"])
    geom = np.asarray([[x["cx"], x["cy"], x["w"], x["h"]]
                       for x in members], np.float32)
    compact = np.asarray([
        *base_p.tolist(), *pmean.tolist(),
        *np.maximum(emb, 0).max(0).tolist(),
        *np.minimum(emb, 0).min(0).tolist(),
        float(scores.mean()), float(scores.max()), float(scores.min()),
        float(scores.std()), float(len(members)),
        *geom.mean(0).tolist(), *geom.std(0).tolist(),
    ], np.float32)
    if mode == "mean":
        return np.r_[mean, base_p, pmean, compact[-10:]].astype(np.float32)
    # Rich mode exposes view disagreement while retaining the score/geometry
    # context.  The signed positive/negative extrema are cheaper and more
    # stable than a full 3x embedding concatenation.
    return np.r_[mean, emb.max(0), emb.min(0), emb.std(0), compact].astype(np.float32)


def group_signature(group: dict) -> tuple[tuple[str, int], ...]:
    """Stable identity for a linked group across independently rebuilt graphs."""
    return tuple(sorted((str(x["stem"]), int(x["row_index"]))
                        for x in group["members"]))


def matched_group_data(records: dict[str, dict], payload: list[tuple],
                       features: dict[int, np.ndarray]):
    x, y = [], []
    for rec_index, (rec, _dets, _edges, groups) in enumerate(payload):
        matches = count.tree_matches(rec, groups)
        for i, j in matches:
            cls = int(rec["bunches"][j]["cls"])
            if 0 <= cls < K:
                x.append(features[rec_index][i]); y.append(cls)
    return np.asarray(x, np.float32), np.asarray(y, np.int64)


def classifier(name: str, seed: int):
    if name.startswith("logreg"):
        c = float(name.split("_")[-1]) if "_" in name else .1
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=c, max_iter=800, solver="lbfgs", random_state=seed))
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(256, 64), alpha=3e-4,
            learning_rate_init=8e-4, batch_size=128, max_iter=250,
            early_stopping=True, validation_fraction=.15,
            n_iter_no_change=20, random_state=seed))
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=600, max_features="sqrt",
                                    min_samples_leaf=5, class_weight=None,
                                    n_jobs=8, random_state=seed)
    raise ValueError(name)


def eval_model(model, feature_rows: list[np.ndarray], payload: list[tuple],
               target_counts: dict[str, int], prior_exp: float,
               class_prior: np.ndarray, link_threshold: float,
               singleton_min: float, max_size: int, rank_mode: str) -> dict:
    cm = np.zeros((K + 1, K + 1), int)
    total_tp = total_gt = total_pred = abs_count = 0
    for ri, (rec, dets, edges, _raw) in enumerate(payload):
        groups = count.selected_clusters(
            dets, edges, link_threshold, singleton_min, max_size,
            target_counts[rec["tree_id"]], rank_mode)
        if groups:
            raw_groups = payload[ri][3]
            by_signature = {
                group_signature(g): feature_rows[ri][i]
                for i, g in enumerate(raw_groups)
            }
            p = model.predict_proba(np.stack([
                by_signature[group_signature(g)] for g in groups]))
        else:
            p = np.zeros((0, K), float)
        for group, prob in zip(groups, p):
            if prior_exp:
                prob = prob * np.power(np.maximum(class_prior, 1e-9), prior_exp)
                prob /= max(float(prob.sum()), 1e-9)
            group["cls"] = int(np.argmax(prob))
        matches = count.tree_matches(rec, groups)
        total_tp += len(matches); total_gt += len(rec["bunches"])
        total_pred += len(groups); abs_count += abs(len(groups) - len(rec["bunches"]))
        matched_pred = {i for i, _ in matches}; matched_gt = {j for _, j in matches}
        for i, j in matches:
            pc, gc = groups[i]["cls"], rec["bunches"][j]["cls"]
            if 0 <= pc < K and 0 <= gc < K:
                cm[pc, gc] += 1
        for i, group in enumerate(groups):
            if i not in matched_pred:
                cm[group["cls"], K] += 1
        for j, bunch in enumerate(rec["bunches"]):
            if j not in matched_gt and 0 <= bunch["cls"] < K:
                cm[K, bunch["cls"]] += 1
    f1 = []
    for c in range(K):
        tp = cm[c, c]
        f1.append(2 * tp / max(2 * tp + cm[c].sum() - tp +
                                cm[:, c].sum() - tp, 1))
    matched = int(cm[:K, :K].sum())
    return {
        "physical_detection": {
            "precision": total_tp / max(total_pred, 1),
            "recall": total_tp / max(total_gt, 1),
            "f1": 2 * (total_tp / max(total_pred, 1)) *
                   (total_tp / max(total_gt, 1)) /
                   max(total_tp / max(total_pred, 1) +
                       total_tp / max(total_gt, 1), 1e-12),
            "tp": total_tp, "pred_clusters": total_pred,
            "gt_bunches": total_gt},
        "counting": {"mae": abs_count / max(len(payload), 1)},
        "classification": {
            "matched_class_accuracy": float(np.trace(cm[:K, :K]) /
                                              max(matched, 1)),
            "matched": matched, "macro_f1_end_to_end": float(np.mean(f1)),
            "per_class_f1_end_to_end": dict(zip(base.NAMES, f1)),
            "confusion_prediction_rows": cm.tolist()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("depth", "953"), required=True)
    ap.add_argument("--fused-root", type=Path, default=Path(
        "/workspace/model_artifacts/project-expertise/eval_2026-08-27"))
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--proposal-min", type=float, default=.125)
    ap.add_argument("--link-threshold", type=float, default=.30)
    ap.add_argument("--singleton-min", type=float, default=.15)
    ap.add_argument("--max-size", type=int, default=3)
    ap.add_argument("--pair-mode", choices=("all", "adjacent"), default="adjacent")
    ap.add_argument("--rank-mode", choices=("score", "support", "max_member", "class_conf"),
                    default="max_member")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--models", nargs="+",
                    default=["logreg_0.01", "logreg_0.1", "logreg_1.0", "mlp", "extra_trees"])
    ap.add_argument("--prior-exponents", nargs="+", type=float, default=[0., -.25])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA diperlukan")
    name = "SawitMVC-Depth-YOLO" if args.dataset == "depth" else "SawitMVC-YOLO"
    cfg = base.CONFIGS[name]
    args.output_root.mkdir(parents=True, exist_ok=True)
    prior = base.build_rotation_prior(base.load_records(cfg, "train"))
    votes = {s: load_vote(vote_path(args.fused_root, args.dataset, s))
             for s in ("train", "val")}
    samples = {}
    payloads = {}
    records = {}
    for split in ("train", "val"):
        samples[split], _ = crop.build_samples(
            cfg, args.dataset, split, votes[split], True)
        records[split], payloads[split] = make_payload(
            cfg, split, votes[split], prior, args.proposal_min,
            args.link_threshold, args.singleton_min, args.max_size,
            args.pair_mode)
    model_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ck_args = model_ckpt["args"]
    visual = crop.ProposalModel(ck_args["backbone"], ck_args.get("channels", 3),
                                ck_args.get("freeze_backbone", False)).cuda()
    visual.load_state_dict(model_ckpt["model"]); visual.eval()
    maps = {}
    for split in ("train", "val"):
        keys = {(str(x["stem"]), int(x["row_index"]))
                for _rec, _dets, _edges, groups in payloads[split]
                for group in groups for x in group["members"]}
        key_to_index = {(s.stem, int(s.row_index)): i
                        for i, s in enumerate(samples[split])}
        missing = keys.difference(key_to_index)
        if missing:
            raise RuntimeError(f"proposal map missing {len(missing)} rows")
        ordered_keys = [k for _index, k in sorted(
            (key_to_index[k], k) for k in keys)]
        selected = [key_to_index[k] for k in ordered_keys]
        emb, prob = extract_embeddings(visual, samples[split], selected,
                                       args.img, args.workers, args.batch)
        emb_map = {k: emb[i] for i, k in enumerate(ordered_keys)}
        prob_map = {k: prob[i] for i, k in enumerate(ordered_keys)}
        maps[split] = (emb_map, prob_map)
        print(json.dumps({"split": split, "groups": len(payloads[split]),
                          "unique_proposals": len(keys),
                          "embedding_dim": int(emb.shape[1])}, ensure_ascii=False),
              flush=True)
    feature_sets = {}
    for mode in ("mean", "rich"):
        train_rows = []
        val_rows = []
        for split, destination in (("train", train_rows), ("val", val_rows)):
            em, pm = maps[split]
            for _rec, _dets, _edges, groups in payloads[split]:
                destination.append(np.stack([group_feature(g, em, pm, mode)
                                             for g in groups])
                                   if groups else np.zeros((0, 1), np.float32))
        x_train, y_train = matched_group_data(records["train"], payloads["train"],
                                               {i: x for i, x in enumerate(train_rows)})
        feature_sets[mode] = (x_train, y_train, val_rows)
        print(json.dumps({"feature_mode": mode, "train_groups_matched": int(len(y_train)),
                          "feature_dim": int(x_train.shape[1])}, ensure_ascii=False),
              flush=True)
    # Count target is fit only on train, exactly as the ordinary post-cluster
    # evaluator.  It is shared across all class-head trials.
    train_records = records["train"]; val_records = records["val"]
    x_count = np.stack([count.feature_vector(r, votes["train"], args.proposal_min)
                        for r in train_records.values()])
    y_count = np.asarray([count.target_count(r) for r in train_records.values()], float)
    alpha, cv = count.choose_alpha(x_count, y_count)
    count_model = count.fit_ridge(x_count, y_count, alpha)
    val_targets = {k: int(v) for k, v in zip(
        val_records, count.predict_count(count_model, np.stack([
            count.feature_vector(r, votes["val"], args.proposal_min)
            for r in val_records.values()]))) }
    class_prior = np.bincount(y_train, minlength=K).astype(float)
    class_prior /= max(float(class_prior.sum()), 1.)
    results = []
    for mode, (x_train, y_train, val_rows) in feature_sets.items():
        for model_name in args.models:
            clf = classifier(model_name, args.seed)
            clf.fit(x_train, y_train)
            for exponent in args.prior_exponents:
                metrics = eval_model(
                    clf, val_rows, payloads["val"], val_targets, exponent,
                    class_prior, args.link_threshold, args.singleton_min,
                    args.max_size, args.rank_mode)
                item = {"feature_mode": mode, "model": model_name,
                        "prior_exponent": exponent, "metrics": metrics}
                results.append(item)
                print(json.dumps({"feature_mode": mode, "model": model_name,
                                  "prior_exponent": exponent,
                                  "physical_f1": metrics["physical_detection"]["f1"],
                                  "count_mae": metrics["counting"]["mae"],
                                  "matched_class_accuracy": metrics["classification"]["matched_class_accuracy"],
                                  "macro_f1": metrics["classification"]["macro_f1_end_to_end"]},
                                 ensure_ascii=False), flush=True)
    best = max(results, key=lambda x: (
        x["metrics"]["classification"]["matched_class_accuracy"],
        x["metrics"]["classification"]["macro_f1_end_to_end"]))
    output = {"protocol": "train-only group visual embedding classifier",
              "dataset": name, "fit_split": "train", "selection_split": "val",
              "checkpoint": str(args.checkpoint), "proposal_min": args.proposal_min,
              "link_threshold": args.link_threshold, "singleton_min": args.singleton_min,
              "max_size": args.max_size, "pair_mode": args.pair_mode,
              "rank_mode": args.rank_mode, "count_model_alpha": alpha,
              "count_cv": cv, "class_prior": class_prior.tolist(),
              "feature_modes": {k: int(v[0].shape[1]) for k, v in feature_sets.items()},
              "results": results, "best": best}
    (args.output_root / "results.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"best": best, "output": str(args.output_root / "results.json")},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
