"""Validation-locked count reconciliation for the remote four-view pipeline.

The detector/WBF/linker produces a variable number of physical-cluster
proposals.  This module adds a small, train-only ridge layer that predicts the
number of bunches in a four-side tree from proposal statistics.  At inference
time the highest-confidence linked clusters are kept up to that predicted
count.  The layer is deliberately separate from the detector and linker so
that its contribution can be ablated and audited.

Protocol:

* fit the count model on four-side TRAIN trees only;
* choose the pipeline profile on VAL only;
* evaluate TEST once after the profile is locked.

No ground-truth count is used by the inference path.  The ``--split`` option
only chooses which prediction dump is evaluated; ``--fit-split`` is kept
explicit in the CLI to make accidental test fitting difficult.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import eval_remote_pipeline_postprocess as base
import sweep_remote_pipeline as sweep


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = Path(
    "/workspace/model_artifacts/project-expertise/eval_2026-08-27"
)
REPO_TEST_ROOT = ROOT / "results" / "remote_eval_2026-08-27"
K = len(base.NAMES)
_WORKER_STATE = None


def four_side(records: dict[str, dict]) -> dict[str, dict]:
    return {key: rec for key, rec in records.items() if rec["n_sides"] == 4}


def vote_file(root: Path, dataset: str, split: str,
              fused_dir: Path | None = None) -> Path:
    safe = dataset.replace("/", "_").replace("-", "_")
    if fused_dir is not None:
        return fused_dir / f"{safe}__wbf_softvote.npz"
    if split == "test" and not (root / "fused_combined1716" / f"{safe}__wbf_softvote.npz").exists():
        folder = REPO_TEST_ROOT / "fused_combined1716"
    else:
        folder = root / ("fused_combined1716" if split == "test"
                         else f"fused_combined1716_{split}")
    return folder / f"{safe}__wbf_softvote.npz"


def load_vote(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {stem: np.asarray(archive[stem], float) for stem in archive.files}


def feature_vector(rec: dict, vote: dict[str, np.ndarray], proposal_min: float) -> np.ndarray:
    """Compact count features from proposals, independent of GT annotations."""
    features: list[float] = []
    side_counts: list[int] = []
    for _side, view in sorted(rec["views"].items()):
        rows = np.asarray(vote.get(view["stem"], np.zeros((0, 9))), float)
        rows = rows[rows[:, 4] >= proposal_min] if len(rows) else rows
        side_counts.append(len(rows))
        if len(rows) and rows.shape[1] >= 5 + K:
            probs = np.maximum(rows[:, 5:5 + K], 0.)
            probs /= np.maximum(probs.sum(1, keepdims=True), 1e-9)
        elif len(rows):
            probs = np.eye(K)[rows[:, 5].astype(int)]
        else:
            probs = np.zeros((0, K), float)
        scores = rows[:, 4] if len(rows) else np.zeros(0)
        area = ((rows[:, 2] - rows[:, 0]) * (rows[:, 3] - rows[:, 1]) /
                max(view["width"] * view["height"], 1)) if len(rows) else np.zeros(0)
        cy = ((rows[:, 1] + rows[:, 3]) / (2 * max(view["height"], 1))
              if len(rows) else np.zeros(0))
        features.extend([
            float(len(rows)), float(scores.sum()),
            float(scores.mean()) if len(scores) else 0.,
            float(scores.std()) if len(scores) else 0.,
            float(scores.max()) if len(scores) else 0.,
        ])
        features.extend((probs * scores[:, None]).sum(0).tolist()
                        if len(rows) else [0.] * K)
        features.extend([
            float(area.sum()) if len(area) else 0.,
            float(area.mean()) if len(area) else 0.,
            float(cy.mean()) if len(cy) else .5,
        ])
    side = np.asarray(side_counts, float)
    features.extend([
        float(side.sum()), float(side.max()), float(side.min()),
        float(side.mean()), float(side.std()), float(np.median(side)),
        float(np.percentile(side, 75)), float(np.sort(side)[-2:].sum()),
    ])
    return np.asarray(features, float)


def target_count(rec: dict) -> float:
    return float(len(rec["bunches"]))


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    mean = X.mean(0)
    scale = X.std(0)
    scale[scale < 1e-8] = 1.
    z = (X - mean) / scale
    y_mean = float(y.mean())
    weights = np.linalg.solve(
        z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - y_mean)
    )
    return {"feature_mean": mean, "feature_scale": scale,
            "target_mean": y_mean, "weights": weights, "alpha": alpha}


def predict_count(model: dict, X: np.ndarray) -> np.ndarray:
    z = (X - model["feature_mean"]) / model["feature_scale"]
    return np.maximum(0, np.rint(model["target_mean"] + z @ model["weights"])).astype(int)


def choose_alpha(X: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple[float, dict]:
    """Choose regularization by deterministic grouped CV inside TRAIN."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(y))
    folds = np.array_split(order, 5)
    candidates = (.1, 1., 10., 100., 1000.)
    scores = []
    for alpha in candidates:
        fold_mae = []
        for holdout in folds:
            fit_idx = np.setdiff1d(order, holdout, assume_unique=False)
            model = fit_ridge(X[fit_idx], y[fit_idx], alpha)
            fold_mae.append(float(np.abs(predict_count(model, X[holdout]) -
                                         y[holdout]).mean()))
        scores.append({"alpha": alpha, "cv_mae": float(np.mean(fold_mae)),
                       "fold_mae": fold_mae})
    best = min(scores, key=lambda row: (row["cv_mae"], row["alpha"]))
    return float(best["alpha"]), {"candidates": scores, "selected": best}


def prepare_payload(records: dict[str, dict], vote: dict[str, np.ndarray],
                    prior: dict, proposal_min: float, pair_mode: str,
                    count_model: dict):
    payload = []
    count_features = {}
    for tree_id, rec in records.items():
        dets = sweep.make_detections(rec, vote, proposal_min)
        edges = sweep.build_edges(dets, rec["n_sides"], prior, pair_mode)
        count_features[tree_id] = feature_vector(rec, vote, proposal_min)
        payload.append((rec, dets, edges))
    X = np.stack([count_features[tree_id] for tree_id in records])
    counts = predict_count(count_model, X)
    return payload, {tree_id: int(n) for tree_id, n in zip(records, counts)}


def selected_clusters(dets: list[dict], edges, link_threshold: float,
                      singleton_min: float, max_size: int, target: int,
                      rank_mode: str = "score"):
    groups = sweep.clusters(dets, edges, link_threshold, singleton_min, max_size)
    if rank_mode == "support":
        key = lambda item: (item["score"] * np.sqrt(len(item["members"])),
                            item["score"], len(item["members"]))
    elif rank_mode == "max_member":
        key = lambda item: (max(x["score"] for x in item["members"]),
                            item["score"], len(item["members"]))
    elif rank_mode == "class_conf":
        key = lambda item: (item["score"] * float(item["p"].max()),
                            item["score"], len(item["members"]))
    else:
        key = lambda item: (item["score"], len(item["members"]))
    groups.sort(key=key, reverse=True)
    return groups[:max(int(target), 0)]


def tree_matches(rec: dict, groups: list[dict]) -> list[tuple[int, int]]:
    bunches = rec["bunches"]
    matrix = np.zeros((len(groups), len(bunches)), float)
    for i, group in enumerate(groups):
        for j, bunch in enumerate(bunches):
            for member in group["members"]:
                for appearance in bunch["appearances"]:
                    if member["side"] == appearance["side"]:
                        matrix[i, j] = max(
                            matrix[i, j],
                            float(base.iou_one(member["box"],
                                               np.asarray([appearance["box"]]))[0]),
                        )
    if not matrix.size:
        return []
    return [(int(i), int(j)) for i, j in zip(*linear_sum_assignment(-matrix))
            if matrix[i, j] >= .5]


def evaluate_payload(payload, target_counts: dict[str, int]) -> dict:
    cm = np.zeros((K + 1, K + 1), int)
    total_pred = total_gt = total_tp = 0
    abs_count = exact = pm1 = vector_exact = 0
    class_correct = matched = 0
    per_tree = []
    for rec, dets, edges in payload:
        model_target = target_counts[rec["tree_id"]]
        raw_count = len(sweep.clusters(
            dets, edges, _WORKER_STATE["link_threshold"],
            _WORKER_STATE["singleton_min"], _WORKER_STATE["max_size"]))
        blend = _WORKER_STATE.get("count_blend", 0.0)
        target = int(round((1.0 - blend) * model_target + blend * raw_count))
        # The requested count is a reconciliation cap, never a source of new
        # detections.  If the cap is above the available proposals, all groups
        # survive and the deficit remains visible in the metrics.
        groups = selected_clusters(dets, edges, _WORKER_STATE["link_threshold"],
                                   _WORKER_STATE["singleton_min"],
                                   _WORKER_STATE["max_size"], target,
                                   _WORKER_STATE["rank_mode"])
        exponent = _WORKER_STATE["class_prior_exponent"]
        if exponent:
            prior = np.maximum(_WORKER_STATE["class_prior"], 1e-9)
            for group in groups:
                group["cls"] = int(np.argmax(
                    group["p"] * np.power(prior, exponent)))
        matches = tree_matches(rec, groups)
        bunches = rec["bunches"]
        total_pred += len(groups); total_gt += len(bunches); total_tp += len(matches)
        delta = len(groups) - len(bunches)
        abs_count += abs(delta); exact += int(delta == 0); pm1 += int(abs(delta) <= 1)
        pred_count = np.bincount([x["cls"] for x in groups], minlength=K)
        gt_count = np.bincount([x["cls"] for x in bunches if x["cls"] >= 0],
                                minlength=K)
        vector_exact += int(np.array_equal(pred_count, gt_count))
        matched += len(matches)
        matched_pred = {i for i, _ in matches}
        matched_gt = {j for _, j in matches}
        for i, j in matches:
            pred_cls, gt_cls = groups[i]["cls"], bunches[j]["cls"]
            if 0 <= pred_cls < K and 0 <= gt_cls < K:
                cm[pred_cls, gt_cls] += 1
                class_correct += int(pred_cls == gt_cls)
        for i, group in enumerate(groups):
            if i not in matched_pred and 0 <= group["cls"] < K:
                cm[group["cls"], K] += 1
        for j, bunch in enumerate(bunches):
            if j not in matched_gt and 0 <= bunch["cls"] < K:
                cm[K, bunch["cls"]] += 1
        per_tree.append({"tree_id": rec["tree_id"], "gt_count": len(bunches),
                         "pred_count": len(groups), "count_delta": delta,
                         "matched": len(matches), "predicted_target": target})
    precision = total_tp / max(total_pred, 1)
    recall = total_tp / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    f1s = []
    for c in range(K):
        tp = cm[c, c]
        fp = int(cm[c, :].sum() - tp)
        fn = int(cm[:, c].sum() - tp)
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    n_trees = len(payload)
    return {
        "physical_detection": {
            "precision": precision, "recall": recall, "f1": f1,
            "tp": total_tp, "pred_clusters": total_pred,
            "gt_bunches": total_gt,
        },
        "counting": {
            "mae": abs_count / max(n_trees, 1),
            "exact_accuracy": exact / max(n_trees, 1),
            "plus_minus_1_accuracy": pm1 / max(n_trees, 1),
            "vector_exact_accuracy": vector_exact / max(n_trees, 1),
        },
        "classification": {
            "matched_class_accuracy": class_correct / max(matched, 1),
            "matched": matched, "macro_f1_end_to_end": float(np.mean(f1s)),
            "per_class_f1_end_to_end": dict(zip(base.NAMES, f1s)),
        },
        "per_tree": per_tree,
    }


def init_worker(state):
    global _WORKER_STATE
    _WORKER_STATE = state


def evaluate_task(task):
    (link, singleton, max_size, rank_mode, class_prior_exponent,
     count_blend) = task
    _WORKER_STATE["link_threshold"] = link
    _WORKER_STATE["singleton_min"] = singleton
    _WORKER_STATE["max_size"] = max_size
    _WORKER_STATE["rank_mode"] = rank_mode
    _WORKER_STATE["class_prior_exponent"] = class_prior_exponent
    _WORKER_STATE["count_blend"] = count_blend
    return {
        "link_threshold": link, "singleton_min": singleton,
        "max_cluster_size": max_size,
        "rank_mode": rank_mode,
        "class_prior_exponent": class_prior_exponent,
        "count_blend": count_blend,
        "metrics": evaluate_payload(_WORKER_STATE["payload"],
                                     _WORKER_STATE["target_counts"]),
    }


def rank(item):
    metrics = item["metrics"]
    physical = metrics["physical_detection"]
    counting = metrics["counting"]
    classification = metrics["classification"]
    return (physical["f1"], -counting["mae"],
            counting["plus_minus_1_accuracy"],
            classification["macro_f1_end_to_end"])


def run_grid(cfg: dict, split: str, fit_split: str, vote: dict,
             train_vote: dict, proposal_mins: list[float],
             link_thresholds: list[float], singleton_mins: list[float],
             max_sizes: list[int], pair_modes: list[str], rank_modes: list[str],
             class_prior_exponents: list[float], workers: int,
             count_blends: list[float] = (0.0,)):
    records = four_side(base.load_records(cfg, split))
    train_records = four_side(base.load_records(cfg, fit_split))
    prior = base.build_rotation_prior(base.load_records(cfg, "train"))
    results = []
    count_models = {}
    count_cv = {}
    y_train = np.asarray([target_count(r) for r in train_records.values()], float)
    class_prior = np.bincount(
        [bunch["cls"] for rec in train_records.values()
         for bunch in rec["bunches"] if 0 <= bunch["cls"] < K],
        minlength=K).astype(float)
    class_prior /= max(float(class_prior.sum()), 1.)
    for proposal_min in proposal_mins:
        X_train = np.stack([feature_vector(r, train_vote, proposal_min)
                            for r in train_records.values()])
        alpha, cv = choose_alpha(X_train, y_train)
        model = fit_ridge(X_train, y_train, alpha)
        count_models[proposal_min] = model
        count_cv[proposal_min] = cv
        for pair_mode in pair_modes:
            payload, target_counts = prepare_payload(
                records, vote, prior, proposal_min, pair_mode, model)
            state = {"payload": payload, "target_counts": target_counts,
                     "class_prior": class_prior}
            tasks = [(link, singleton, max_size)
                     for link in link_thresholds
                     for singleton in singleton_mins
                     for max_size in max_sizes]
            tasks = [(*task[:3], rank_mode, class_prior_exponent, count_blend)
                     for task in tasks for rank_mode in rank_modes
                     for class_prior_exponent in class_prior_exponents
                     for count_blend in count_blends]
            if workers > 1 and len(tasks) > 1:
                context = mp.get_context("fork")
                with ProcessPoolExecutor(
                        max_workers=min(workers, len(tasks)),
                        mp_context=context, initializer=init_worker,
                        initargs=(state,)) as pool:
                    evaluated = pool.map(evaluate_task, tasks, chunksize=1)
            else:
                init_worker(state)
                evaluated = map(evaluate_task, tasks)
            for item in evaluated:
                item.update({"proposal_min": proposal_min,
                             "pair_mode": pair_mode,
                             "count_model_alpha": model["alpha"]})
                results.append(item)
    return {
        "dataset": cfg["kind"], "split": split, "fit_split": fit_split,
        "n_trees": len(records), "n_8_side_excluded": len(base.load_records(cfg, split)) - len(records),
        "proposal_mins": proposal_mins, "link_thresholds": link_thresholds,
        "singleton_mins": singleton_mins, "max_sizes": max_sizes,
        "pair_modes": pair_modes, "rank_modes": rank_modes,
        "class_prior_exponents": class_prior_exponents,
        "count_blends": list(count_blends),
        "class_prior_train": class_prior.tolist(), "count_model_cv": count_cv,
        "results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("depth", "953"), required=True)
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--fit-split", choices=("train",), default="train")
    ap.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    ap.add_argument("--fused-dir", type=Path, default=None,
                    help="optional directory containing softvote NPZ files")
    ap.add_argument("--fit-fused-dir", type=Path, default=None,
                    help="optional separate softvote directory for fit-split")
    ap.add_argument("--proposal-mins", nargs="+", type=float,
                    default=[.05, .075, .10, .125, .15, .20, .25, .30])
    ap.add_argument("--link-thresholds", nargs="+", type=float,
                    default=[.20, .25, .30, .35, .40, .45, .50, .55, .60])
    ap.add_argument("--singleton-mins", nargs="+", type=float,
                    default=[.10, .15, .20, .25, .30, .40])
    ap.add_argument("--max-sizes", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--pair-modes", nargs="+", choices=("all", "adjacent"),
                    default=["all", "adjacent"])
    ap.add_argument("--rank-modes", nargs="+",
                    choices=("score", "support", "max_member", "class_conf"),
                    default=["score"])
    ap.add_argument("--class-prior-exponents", nargs="+", type=float,
                    default=[0.0])
    ap.add_argument("--count-blends", nargs="+", type=float,
                    default=[0.0],
                    help="blend model count with raw linked count: 0=model, 1=raw")
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 32))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    dataset_name = "SawitMVC-YOLO" if args.dataset == "953" else "SawitMVC-Depth-YOLO"
    cfg = base.CONFIGS[dataset_name]
    path = vote_file(args.artifact_root, dataset_name, args.split, args.fused_dir)
    train_path = vote_file(
        args.artifact_root, dataset_name, "train",
        args.fit_fused_dir if args.fit_fused_dir is not None else args.fused_dir)
    if not path.exists():
        raise FileNotFoundError(path)
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    print(f"[{dataset_name}] split={args.split} vote={path}", flush=True)
    result = run_grid(
        cfg, args.split, args.fit_split, load_vote(path), load_vote(train_path),
        args.proposal_mins, args.link_thresholds, args.singleton_mins,
        args.max_sizes, args.pair_modes, args.rank_modes,
        args.class_prior_exponents, max(args.workers, 1), args.count_blends,
    )
    result["workers"] = max(args.workers, 1)
    ranked = sorted(result["results"], key=rank, reverse=True)
    result["top"] = ranked[:50]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False,
                                      default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x)) + "\n")
    print(json.dumps(result["top"][:10], indent=2, ensure_ascii=False), flush=True)
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
