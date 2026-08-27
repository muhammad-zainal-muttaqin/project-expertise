"""Evaluate a train-fitted pair linker on the remote four-view detector bank.

The original linker uses a robust rotation prior and a hand-written geometric
cost.  This experiment fits a small logistic pair classifier on TRAIN WBF
proposals.  A proposal is assigned to a ground-truth bunch only when its
same-side IoU is at least 0.5; proposal pairs assigned to the same bunch are
positive links.  At inference the classifier score replaces the hand-written
edge score before the same one-to-one matching and disjoint clustering.

No validation or test annotations are used while fitting the pair model or
choosing its threshold.  This is intended as a generalization experiment,
not as a test-set optimizer.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).parent))
import eval_remote_pipeline_postprocess as base  # noqa: E402
import evaluate_remote_count_reconciled as count  # noqa: E402
import sweep_remote_pipeline as sweep  # noqa: E402


K = len(base.NAMES)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = Path(
    "/workspace/model_artifacts/project-expertise/eval_2026-08-27"
)


def four_side(records: dict[str, dict]) -> dict[str, dict]:
    return {key: rec for key, rec in records.items() if rec["n_sides"] == 4}


def pair_features(a: dict, b: dict, n: int,
                  prior: dict[tuple[int, int], tuple[float, ...]]) -> np.ndarray:
    if a["side"] > b["side"]:
        a, b = b, a
    d = (b["side"] - a["side"]) % n
    mux, muy, sx, sy, sarea, _ = prior.get(
        (n, d), (0.0, 0.0, 0.20, 0.15, 0.70, 0)
    )
    zdx = ((b["cx"] - a["cx"]) - mux) / max(sx, 0.025)
    zdy = ((b["cy"] - a["cy"]) - muy) / max(sy, 0.025)
    zarea = np.log(max(b["w"] * b["h"], 1e-8) /
                   max(a["w"] * a["h"], 1e-8)) / max(sarea, 0.15)
    zshape = np.log(max(b["w"] / max(b["h"], 1e-8), 1e-8) /
                    max(a["w"] / max(a["h"], 1e-8), 1e-8)) / 0.85
    class_sim = float(np.sqrt(np.maximum(a["p"], 0.0) *
                              np.maximum(b["p"], 0.0)).sum())
    score_a, score_b = float(a["score"]), float(b["score"])
    # Signed and squared geometry let the small linear model learn both
    # direction-specific offsets and the robust cost-like behavior.
    return np.asarray([
        zdx, zdy, zarea, zshape,
        zdx * zdx, zdy * zdy, zarea * zarea, zshape * zshape,
        class_sim, score_a, score_b, min(score_a, score_b),
        abs(score_a - score_b),
    ], dtype=float)


def proposal_bunch_ids(rec: dict, dets: list[dict]) -> list[int]:
    ids = []
    for det in dets:
        best_iou, best = 0.0, -1
        for index, bunch in enumerate(rec["bunches"]):
            overlap = 0.0
            for appearance in bunch["appearances"]:
                if int(appearance["side"]) != int(det["side"]):
                    continue
                overlap = max(overlap, sweep.iou(
                    det["box"], np.asarray(appearance["box"], dtype=float)))
            if overlap > best_iou:
                best_iou, best = overlap, index
        ids.append(best if best_iou >= 0.5 else -1)
    return ids


def pair_dataset(records: dict[str, dict], vote: dict[str, np.ndarray],
                 proposal_min: float,
                 prior: dict[tuple[int, int], tuple[float, ...]],
                 pair_mode: str) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for rec in records.values():
        dets = sweep.make_detections(rec, vote, proposal_min)
        assignments = proposal_bunch_ids(rec, dets)
        by_side = defaultdict(list)
        for index, det in enumerate(dets):
            by_side[int(det["side"])].append(index)
        sides = sorted(by_side)
        n = rec["n_sides"]
        for pos, side_a in enumerate(sides):
            for side_b in sides[pos + 1:]:
                if pair_mode == "adjacent" and (side_b - side_a) % n not in (1, n - 1):
                    continue
                for ia in by_side[side_a]:
                    for ib in by_side[side_b]:
                        features.append(pair_features(
                            dets[ia], dets[ib], n, prior))
                        labels.append(int(
                            assignments[ia] >= 0 and
                            assignments[ia] == assignments[ib]))
    if not features:
        return np.zeros((0, 13), float), np.zeros(0, int)
    return np.stack(features), np.asarray(labels, int)


class LogisticPairModel:
    """Small standardized, class-balanced logistic model without sklearn."""

    def __init__(self, c: float = 1.0):
        self.c = float(c)
        self.mean = None
        self.scale = None
        self.coef = None
        self.intercept = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.mean = x.mean(0)
        self.scale = x.std(0)
        self.scale[self.scale < 1e-6] = 1.0
        z = (x - self.mean) / self.scale
        n_pos = max(int(y.sum()), 1)
        n_neg = max(int(len(y) - y.sum()), 1)
        sample_weight = np.where(y == 1, len(y) / (2 * n_pos),
                                 len(y) / (2 * n_neg))

        def objective(theta):
            logits = theta[0] + z @ theta[1:]
            loss = np.sum(sample_weight *
                          (np.logaddexp(0.0, logits) - y * logits)) / len(y)
            loss += 0.5 * np.sum(theta[1:] ** 2) / self.c
            residual = sample_weight * (expit(logits) - y) / len(y)
            grad = np.empty_like(theta)
            grad[0] = residual.sum()
            grad[1:] = z.T @ residual + theta[1:] / self.c
            return float(loss), grad

        initial = np.zeros(z.shape[1] + 1, dtype=float)
        result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                          options={"maxiter": 500, "ftol": 1e-10})
        if not result.success:
            raise RuntimeError(f"pair logistic optimization failed: {result.message}")
        self.intercept = float(result.x[0])
        self.coef = result.x[1:]
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.scale
        positive = expit(self.intercept + z @ self.coef)
        return np.column_stack([1.0 - positive, positive])


def fit_linker(records: dict[str, dict], vote: dict[str, np.ndarray],
               proposal_min: float,
               prior: dict[tuple[int, int], tuple[float, ...]],
               pair_mode: str) -> tuple[LogisticPairModel, float, dict]:
    x, y = pair_dataset(records, vote, proposal_min, prior, pair_mode)
    if len(np.unique(y)) < 2:
        raise RuntimeError("train pair dataset does not contain both link classes")
    model = LogisticPairModel(c=1.0).fit(x, y)
    scores = model.predict_proba(x)[:, 1]
    best = (0.0, 0.5, 0, 0, 0)
    for threshold in np.arange(0.05, 0.96, 0.01):
        pred = scores >= threshold
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best[0]:
            best = (float(f1), float(threshold), tp, fp, fn)
    info = {
        "n_pairs": int(len(y)),
        "positive_pairs": int(y.sum()),
        "positive_rate": float(y.mean()),
        "train_pair_f1": best[0],
        "threshold": best[1],
        "tp": best[2], "fp": best[3], "fn": best[4],
        "coefficients": model.coef.tolist(),
        "intercept": float(model.intercept),
    }
    return model, best[1], info


def learned_edges(dets: list[dict], rec: dict,
                  prior: dict[tuple[int, int], tuple[float, ...]],
                  model: LogisticPairModel, pair_mode: str):
    by_side = defaultdict(list)
    for index, det in enumerate(dets):
        by_side[int(det["side"])].append(index)
    edges = []
    sides = sorted(by_side)
    n = rec["n_sides"]
    for pos, side_a in enumerate(sides):
        for side_b in sides[pos + 1:]:
            if pair_mode == "adjacent" and (side_b - side_a) % n not in (1, n - 1):
                continue
            left, right = by_side[side_a], by_side[side_b]
            if not left or not right:
                continue
            x = np.stack([pair_features(dets[ia], dets[ib], n, prior)
                          for ia in left for ib in right])
            scores = model.predict_proba(x)[:, 1].reshape(len(left), len(right))
            rows, cols = linear_sum_assignment(-scores)
            edges.extend((float(scores[row, col]), left[row], right[col])
                         for row, col in zip(rows, cols))
    edges.sort(reverse=True)
    return edges


def dynamic_cfg(dataset: str) -> dict:
    return base.CONFIGS[dataset]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("depth", "953"), required=True)
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    ap.add_argument("--fused-dir", type=Path, default=None)
    ap.add_argument("--fit-fused-dir", type=Path, default=None)
    ap.add_argument("--proposal-min", type=float, required=True)
    ap.add_argument("--singleton-min", type=float, default=.15)
    ap.add_argument("--link-threshold", type=float, default=None)
    ap.add_argument("--max-size", type=int, default=3)
    ap.add_argument("--pair-mode", choices=("all", "adjacent"), default="adjacent")
    ap.add_argument("--rank-mode", choices=("score", "support", "max_member", "class_conf"), default="support")
    ap.add_argument("--class-prior-exponent", type=float, default=-.25)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    dataset = "SawitMVC-YOLO" if args.dataset == "953" else "SawitMVC-Depth-YOLO"
    cfg = dynamic_cfg(dataset)
    eval_vote_path = count.vote_file(
        args.artifact_root, dataset, args.split, args.fused_dir)
    fit_dir = args.fit_fused_dir if args.fit_fused_dir is not None else args.fused_dir
    fit_vote_path = count.vote_file(
        args.artifact_root, dataset, "train", fit_dir)
    if not eval_vote_path.exists() or not fit_vote_path.exists():
        raise FileNotFoundError(f"vote files: {fit_vote_path}, {eval_vote_path}")
    eval_vote = count.load_vote(eval_vote_path)
    fit_vote = count.load_vote(fit_vote_path)
    train_records = four_side(base.load_records(cfg, "train"))
    eval_records = four_side(base.load_records(cfg, args.split))
    prior = base.build_rotation_prior(base.load_records(cfg, "train"))

    model, learned_threshold, link_info = fit_linker(
        train_records, fit_vote, args.proposal_min, prior, args.pair_mode)
    link_threshold = (args.link_threshold if args.link_threshold is not None
                      else learned_threshold)

    y_train = np.asarray([count.target_count(rec)
                          for rec in train_records.values()], float)
    x_train = np.stack([count.feature_vector(rec, fit_vote, args.proposal_min)
                        for rec in train_records.values()])
    alpha, cv = count.choose_alpha(x_train, y_train)
    count_model = count.fit_ridge(x_train, y_train, alpha)
    x_eval = np.stack([count.feature_vector(rec, eval_vote, args.proposal_min)
                       for rec in eval_records.values()])
    target_counts = {
        tree_id: int(n) for tree_id, n in zip(
            eval_records, count.predict_count(count_model, x_eval))
    }
    payload = []
    for rec in eval_records.values():
        dets = sweep.make_detections(rec, eval_vote, args.proposal_min)
        edges = learned_edges(dets, rec, prior, model, args.pair_mode)
        payload.append((rec, dets, edges))
    count.init_worker({
        "link_threshold": link_threshold,
        "singleton_min": args.singleton_min,
        "max_size": args.max_size,
        "rank_mode": args.rank_mode,
        "class_prior_exponent": args.class_prior_exponent,
        "class_prior": np.bincount(
            [b["cls"] for rec in train_records.values() for b in rec["bunches"]
             if 0 <= b["cls"] < K], minlength=K).astype(float),
    })
    state = count._WORKER_STATE
    state["class_prior"] /= max(float(state["class_prior"].sum()), 1.0)
    metrics = count.evaluate_payload(payload, target_counts)
    result = {
        "dataset": dataset, "split": args.split, "fit_split": "train",
        "proposal_min": args.proposal_min, "pair_mode": args.pair_mode,
        "link_threshold": link_threshold,
        "singleton_min": args.singleton_min, "max_cluster_size": args.max_size,
        "rank_mode": args.rank_mode,
        "class_prior_exponent": args.class_prior_exponent,
        "link_model": "standardized logistic regression (class-balanced, C=1.0)",
        "link_training": link_info,
        "count_model_alpha": alpha, "count_model_cv": cv,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "metrics"}, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_tree"}, indent=2))
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
