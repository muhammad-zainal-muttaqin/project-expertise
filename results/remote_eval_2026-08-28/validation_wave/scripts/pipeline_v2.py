"""PIPELINE V2 for the oil-palm four-view project.

Integrates the trained per-box TP re-ranker (``/workspace/map_boost``) into the
end-to-end four-view pipeline, retrains the cross-view edge linker on the
resulting improved proposals, adds a stronger counting layer, and selects a
profile on VALIDATION ONLY.

Everything under ``/workspace/project-expertise`` and ``/workspace/model_artifacts``
is read-only. ``/workspace/map_boost`` and ``/workspace/gsp_linker`` belong to
sibling agents and are only imported/read here, never written to. All new
artifacts produced by this script live under ``/workspace/pipeline_v2``.

Stages (see task spec for the exact, declared grids -- nothing here is a free
knob beyond what is written down):

  0. Sanity anchor: reproduce one already-locked GSP row (depth, tau_prob=0.10,
     singleton=0.20, max_size=3, rank=support) with the ORIGINAL v1 fused
     dumps and the ORIGINAL depth edge model. Must equal F1=0.8526 (+/-0.003).
  1. Build V2 proposal dumps (per dataset, per split) by scoring every cached
     floor-0.02 box with the trained floor-0.02 TP re-ranker, in two score
     modes (``ptp`` and ``geo``).
  2. Retrain the cross-view edge linker (ExtraTreesClassifier) on those V2
     proposals, per (dataset, mode).
  3. Run the GSP linker (imported from ``link_global_setpartition.py``) with
     the retrained edge model over the declared tau_prob/max_size grid.
  4. Fit a stronger counting layer on TRAIN only (ridge + HistGradientBoosting
     regressor) using proposal statistics plus GSP-derived cluster counts.
  5. Stage-1 / stage-2 grid selection on VAL, as declared in the task spec.

No test file is opened anywhere in this script.
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, "/workspace/project-expertise/scripts")
sys.path.insert(0, "/workspace/map_boost")
sys.path.insert(0, "/workspace/gsp_linker")

import eval_remote_pipeline_postprocess as base  # noqa: E402
import evaluate_remote_class_head as head_eval  # noqa: E402
import evaluate_remote_count_reconciled as count  # noqa: E402
import train_detection_edge_linker as edge  # noqa: E402
import link_global_setpartition as gsp  # noqa: E402


K = len(base.NAMES)

ORIG_FUSED_ROOT = Path("/workspace/model_artifacts/project-expertise/eval_2026-08-27")
MAP_BOOST_CACHE = Path("/workspace/map_boost/cache")
MAP_BOOST_ARTIFACTS = Path("/workspace/map_boost/artifacts")

OUT_ROOT = Path("/workspace/pipeline_v2")
ARTIFACT_ROOT = OUT_ROOT / "artifacts"

DATASETS = ["953", "depth"]
MODES = ["ptp", "geo"]

RANKER_FLOOR = 0.02
PROPOSAL_MIN_V2 = 0.15
PAIR_MODE = "adjacent"
P_FLOOR = 0.02
ENUM_MAX_SIZE = 4
COUNT_FEATURE_TAUS = (0.15, 0.25, 0.35)
COUNT_FEATURE_MAX_SIZE = 4

GSP_TAU_PROBS = [.05, .10, .15, .20, .25, .35, .50]
GSP_MAX_SIZES = [3, 4]

STAGE1_SINGLETON = 0.15
STAGE1_RANK = "score"

STAGE2_SINGLETONS = [.10, .15, .20, .25]
STAGE2_RANKS = ["score", "support", "max_member"]
STAGE2_BLENDS = [0., .25, .5]
STAGE2_COUNT_MODELS = ["ridge", "hgb"]

# Sanity-anchor target (see task spec): depth GSP row, ORIGINAL v1 dumps +
# ORIGINAL depth edge model, tau_prob=0.10 / singleton=0.20 / max_size=3 /
# rank=support. Independently confirmed against
# /workspace/gsp_linker/artifacts/depth/results_val.json (F1=0.8526413345690453).
ANCHOR_EXPECTED_F1 = 0.8526
ANCHOR_TOLERANCE = 0.003

# Current locked E2E val bests to beat (task "Why" section; exact values cross
# -checked against /workspace/gsp_linker/artifacts/{953,depth}/results_val.json).
CURRENT_BEST = {
    "953": {
        "f1": 0.8232161874334399, "mae": 1.2527472527472527,
        "pm1": 0.6703296703296703, "matched": 0.7542043984476067,
        "macro": 0.601393979256198, "n_trees": 91, "linker": "hungarian",
        "gsp_variant": {"matched": 0.755464480874317, "f1": 0.8313458262350937,
                        "mae": 1.7472527472527473, "pm1": 0.5054945054945055,
                        "macro": 0.6079169570435579},
    },
    "depth": {
        "f1": 0.8526413345690453, "mae": 0.9316239316239316,
        "pm1": 0.7863247863247863, "matched": 0.8456521739130435,
        "macro": 0.6806848973454227, "n_trees": 117, "linker": "gsp",
    },
}

# Old (v1, floor-0.125 proposals) edge-linker diagnostics, read from
# /workspace/model_artifacts/project-expertise/detection_edge_linker_{953_v2,depth_v1}/results.json
# (extra model), used only as a fixed comparison reference for the retrained V2
# edge linker. Not recomputed here to avoid re-deriving numbers already on disk.
OLD_EDGE_DIAG = {
    "953": {"train_auc": 0.9999996063102731, "val_auc": 0.9484599773550072,
            "train_ap": 0.9999926978673317, "val_ap": 0.5963600472722672,
            "proposal_min": 0.125, "pair_mode": "adjacent",
            "source": str(Path("/workspace/model_artifacts/project-expertise/"
                               "detection_edge_linker_953_v2/results.json"))},
    "depth": {"train_auc": 0.9999999999999999, "val_auc": 0.9554436557231588,
              "train_ap": 1.0, "val_ap": 0.7400223121883881,
              "proposal_min": 0.125, "pair_mode": "adjacent",
              "source": str(Path("/workspace/model_artifacts/project-expertise/"
                                 "detection_edge_linker_depth_v1/results.json"))},
}

# "Materially below" tolerance for the depth no-regression guardrail (task
# text does not pin an exact number; declared explicitly here and reported).
DEPTH_NO_REGRESSION_TOLERANCE = {"f1": 0.01, "mae": 0.05, "pm1": 0.02, "macro": 0.01}


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                default=_json_default) + "\n", encoding="utf-8")


def summary(m: dict) -> dict:
    return {
        "f1": m["physical_detection"]["f1"],
        "precision": m["physical_detection"]["precision"],
        "recall": m["physical_detection"]["recall"],
        "mae": m["counting"]["mae"],
        "pm1": m["counting"]["plus_minus_1_accuracy"],
        "vector_exact": m["counting"]["vector_exact_accuracy"],
        "matched": m["classification"]["matched_class_accuracy"],
        "macro": m["classification"]["macro_f1_end_to_end"],
    }


def rank_key(m: dict) -> tuple:
    """(matched_class_acc, F1, -MAE, macro) -- the declared selection order."""
    return (m["classification"]["matched_class_accuracy"],
            m["physical_detection"]["f1"],
            -m["counting"]["mae"],
            m["classification"]["macro_f1_end_to_end"])


# ---------------------------------------------------------------------------
# Stage 0: sanity anchor
# ---------------------------------------------------------------------------

def run_sanity_anchor() -> dict:
    """Reproduce the depth GSP anchor row with ORIGINAL v1 inputs.

    Uses fused_combined1716_val/_train (proposal_min=0.125) and
    detection_edge_linker_depth_v1/extra.joblib, imported straight from
    ``link_global_setpartition`` (gsp) -- no reimplementation.
    """
    t0 = time.time()
    dataset = "depth"
    ctx = gsp.load_context(dataset, gsp.FUSED_ROOT, "val", gsp.PROPOSAL_MIN)
    model = joblib.load(gsp.MODEL_PATHS[(dataset, "extra")])
    tree_ids = list(ctx["split_records"].keys())
    tau_prob, singleton, max_size, rank_mode = 0.10, 0.20, 3, "support"
    tau_logit = math.log(tau_prob / (1. - tau_prob))
    payload = []
    tags = defaultdict(int)
    for tree_id in tree_ids:
        rec = ctx["split_records"][tree_id]
        dets = ctx["dets_per_tree"][tree_id]
        probs = gsp.tree_pair_probs(dets, rec["n_sides"], ctx["prior"], model, gsp.PAIR_MODE)
        candidates, _floor = gsp.enumerate_candidates(dets, probs, gsp.P_FLOOR, gsp.ENUM_MAX_SIZE)
        chosen, tag = gsp.solve_partition(len(dets), candidates, tau_logit, max_size)
        tags[tag] += 1
        payload.append((rec, dets, gsp.decided_edges(chosen)))
    metrics = head_eval.evaluate_payload(
        payload, ctx["targets"], 0.5, singleton, max_size, rank_mode,
        0.0, ctx["class_prior"], 0.0, None, "mean", 0.0)
    actual_f1 = metrics["physical_detection"]["f1"]
    diff = abs(actual_f1 - ANCHOR_EXPECTED_F1)
    passed = diff <= ANCHOR_TOLERANCE
    result = {
        "dataset": dataset, "model": "extra", "linker": "gsp",
        "tau_prob": tau_prob, "singleton_min": singleton, "max_size": max_size,
        "rank_mode": rank_mode, "proposal_min": gsp.PROPOSAL_MIN,
        "n_trees": len(tree_ids), "solver_tag_counts": dict(tags),
        "model_path": str(gsp.MODEL_PATHS[(dataset, "extra")]),
        "train_vote_path": str(ctx["train_vote_path"]),
        "split_vote_path": str(ctx["split_vote_path"]),
        "expected_f1": ANCHOR_EXPECTED_F1, "tolerance": ANCHOR_TOLERANCE,
        "actual_f1": actual_f1, "diff": diff, "passed": passed,
        "full_metrics": metrics, "seconds": time.time() - t0,
    }
    print(json.dumps({"stage": "sanity_anchor", "actual_f1": actual_f1,
                      "expected_f1": ANCHOR_EXPECTED_F1, "diff": diff,
                      "passed": passed}, default=_json_default), flush=True)
    return result


# ---------------------------------------------------------------------------
# Stage 1: V2 proposal dumps
# ---------------------------------------------------------------------------

def load_features_and_ranker(dataset: str) -> tuple[dict, object, Path, dict]:
    feat = {}
    feat_paths = {}
    for split in ("train", "val"):
        p = MAP_BOOST_CACHE / f"features_{dataset}_{split}_floor{RANKER_FLOOR:.2f}.joblib"
        if not p.exists():
            raise FileNotFoundError(
                f"missing cached feature matrix (spec says rebuild via "
                f"rank_and_emit.py if this happens): {p}")
        feat[split] = joblib.load(p)
        feat_paths[split] = p
    ranker_p = MAP_BOOST_ARTIFACTS / dataset / f"ranker_floor{RANKER_FLOOR:.2f}.joblib"
    if not ranker_p.exists():
        raise FileNotFoundError(f"missing trained ranker: {ranker_p}")
    model = joblib.load(ranker_p)
    for split, d in feat.items():
        n = len(d["y"])
        assert model.n_features_in_ == d["X"].shape[1], (
            f"ranker/feature dim mismatch for {dataset}/{split}: "
            f"{model.n_features_in_} vs {d['X'].shape[1]}")
        for key in ("stems", "row_index", "box", "score", "p"):
            assert len(d[key]) == n, f"cache misalignment {dataset}/{split}/{key}"
    return feat, model, ranker_p, feat_paths


def build_v2_vote_dict(feat_split: dict, p_tp: np.ndarray, mode: str) -> dict[str, np.ndarray]:
    score = np.asarray(feat_split["score"], float)
    p = np.asarray(feat_split["p"], float)
    box = np.asarray(feat_split["box"], float)
    stems = feat_split["stems"]
    row_index = np.asarray(feat_split["row_index"], int)
    if mode == "ptp":
        sprime = p_tp
    elif mode == "geo":
        sprime = np.sqrt(np.clip(score, 0., None) * np.clip(p_tp, 0., None))
    else:
        raise ValueError(mode)
    by_stem: dict[str, list[int]] = defaultdict(list)
    for i, stem in enumerate(stems):
        by_stem[str(stem)].append(i)
    out = {}
    for stem, idxs in by_stem.items():
        idxs_sorted = sorted(idxs, key=lambda i: int(row_index[i]))
        rows = np.zeros((len(idxs_sorted), 5 + K), dtype=np.float32)
        for r, i in enumerate(idxs_sorted):
            rows[r, :4] = box[i]
            rows[r, 4] = sprime[i]
            pp = np.maximum(p[i], 0.)
            s = float(pp.sum())
            rows[r, 5:5 + K] = (pp / s) if s > 0 else (1. / K)
        out[stem] = rows
    return out


def build_all_v2_votes() -> tuple[dict, dict]:
    votes: dict[str, dict] = {}
    diag: dict[str, dict] = {}
    for dataset in DATASETS:
        t0 = time.time()
        feat, model, ranker_p, feat_paths = load_features_and_ranker(dataset)
        votes[dataset] = {}
        diag[dataset] = {
            "ranker_path": str(ranker_p),
            "feature_cache": {s: str(p) for s, p in feat_paths.items()},
            "n_boxes": {s: int(len(feat[s]["y"])) for s in feat},
            "n_features": int(feat["train"]["X"].shape[1]),
        }
        for mode in MODES:
            votes[dataset][mode] = {}
            for split in ("train", "val"):
                d = feat[split]
                p_tp = model.predict_proba(d["X"])[:, 1]
                votes[dataset][mode][split] = build_v2_vote_dict(d, p_tp, mode)
        diag[dataset]["seconds"] = time.time() - t0
        print(json.dumps({"stage": "v2_votes", "dataset": dataset,
                          "seconds": diag[dataset]["seconds"],
                          "n_boxes": diag[dataset]["n_boxes"]},
                         default=_json_default), flush=True)
    return votes, diag


def save_votes(votes: dict) -> dict:
    paths = {}
    for dataset in DATASETS:
        out_dir = ARTIFACT_ROOT / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        for mode in MODES:
            for split in ("train", "val"):
                d = votes[dataset][mode][split]
                p = out_dir / f"vote_v2_{mode}_{split}.npz"
                np.savez_compressed(p, **d)
                paths[f"{dataset}_{mode}_{split}"] = str(p)
    return paths


# ---------------------------------------------------------------------------
# Stage 2: edge linker retrain
# ---------------------------------------------------------------------------

def train_edge_v2(dataset: str, mode: str, votes: dict, prior: dict) -> tuple[object, dict, str]:
    cfg = edge.cfg_for(dataset)
    train_records = count.four_side(base.load_records(cfg, "train"))
    val_records = count.four_side(base.load_records(cfg, "val"))
    v2_train_vote = votes[dataset][mode]["train"]
    v2_val_vote = votes[dataset][mode]["val"]
    t0 = time.time()
    x_train, y_train, stats_train = edge.build_pair_data(
        train_records, v2_train_vote, prior, PROPOSAL_MIN_V2, PAIR_MODE)
    x_val, y_val, stats_val = edge.build_pair_data(
        val_records, v2_val_vote, prior, PROPOSAL_MIN_V2, PAIR_MODE)
    if y_train.min() == y_train.max():
        raise RuntimeError(f"edge pair labels single-class for {dataset}/{mode}")
    pos_weight = (len(y_train) - int(y_train.sum())) / max(int(y_train.sum()), 1)
    sample_weight = np.where(y_train == 1, min(pos_weight, 30.), 1.).astype(np.float32)
    model = ExtraTreesClassifier(
        n_estimators=180, min_samples_leaf=3, max_features=.8,
        class_weight="balanced", n_jobs=16, random_state=42)
    model.fit(x_train, y_train, sample_weight=sample_weight)
    fit_seconds = time.time() - t0
    train_prob = model.predict_proba(x_train)[:, 1]
    val_prob = model.predict_proba(x_val)[:, 1]
    diag = {
        "train_auc": float(roc_auc_score(y_train, train_prob)),
        "val_auc": float(roc_auc_score(y_val, val_prob)),
        "train_ap": float(average_precision_score(y_train, train_prob)),
        "val_ap": float(average_precision_score(y_val, val_prob)),
        "train_pair_stats": stats_train, "val_pair_stats": stats_val,
        "fit_seconds": fit_seconds, "n_features": int(x_train.shape[1]),
        "proposal_min": PROPOSAL_MIN_V2, "pair_mode": PAIR_MODE,
        "pos_weight_uncapped": pos_weight, "pos_weight_used": min(pos_weight, 30.),
    }
    out_dir = ARTIFACT_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"edge_v2_{mode}.joblib"
    joblib.dump(model, model_path, compress=3)
    print(json.dumps({"stage": "edge_retrain", "dataset": dataset, "mode": mode,
                      **{k: v for k, v in diag.items()
                         if k not in ("train_pair_stats", "val_pair_stats")}},
                     default=_json_default), flush=True)
    return model, diag, str(model_path)


# ---------------------------------------------------------------------------
# GSP candidate/dets cache + count-feature construction
# ---------------------------------------------------------------------------

def build_dets_and_candidates(records: dict, vote: dict, prior: dict, model,
                              proposal_min: float = PROPOSAL_MIN_V2,
                              pair_mode: str = PAIR_MODE, p_floor: float = P_FLOOR,
                              enum_max_size: int = ENUM_MAX_SIZE) -> dict:
    per_tree = {}
    for tree_id, rec in records.items():
        dets = edge.make_detections(rec, vote, proposal_min)
        probs = gsp.tree_pair_probs(dets, rec["n_sides"], prior, model, pair_mode)
        candidates, used_floor = gsp.enumerate_candidates(dets, probs, p_floor, enum_max_size)
        per_tree[tree_id] = {"dets": dets, "candidates": candidates, "used_floor": used_floor}
    return per_tree


def count_extra_features(dets: list, candidates: list,
                         taus=COUNT_FEATURE_TAUS, max_size=COUNT_FEATURE_MAX_SIZE) -> np.ndarray:
    n = len(dets)
    scores = np.asarray([d["score"] for d in dets], float) if n else np.zeros(0)
    extra = [float(n), float(scores.sum()),
             float(int(np.sum(scores >= 0.3))), float(int(np.sum(scores >= 0.5)))]
    for tau_prob in taus:
        tau_logit = math.log(tau_prob / (1. - tau_prob))
        chosen, _tag = gsp.solve_partition(n, candidates, tau_logit, max_size)
        n_multi = len(chosen)
        n_in_clusters = sum(len(m) for m in chosen)
        n_total = (n - n_in_clusters) + n_multi
        extra.extend([float(n_total), float(n_multi)])
    return np.asarray(extra, float)


def build_count_features(records: dict, vote: dict, per_tree: dict,
                         proposal_min: float = PROPOSAL_MIN_V2):
    feats, ys, tree_ids = [], [], []
    for tree_id, rec in records.items():
        base_feat = count.feature_vector(rec, vote, proposal_min)
        extra_feat = count_extra_features(per_tree[tree_id]["dets"], per_tree[tree_id]["candidates"])
        feats.append(np.concatenate([base_feat, extra_feat]))
        ys.append(count.target_count(rec))
        tree_ids.append(tree_id)
    return np.stack(feats), np.asarray(ys, float), tree_ids


def hgb_cv_mae(X: np.ndarray, y: np.ndarray, seed: int = 42, folds: int = 5) -> dict:
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(y))
    fold_idx = np.array_split(order, folds)
    fold_mae = []
    for holdout in fold_idx:
        fit_idx = np.setdiff1d(order, holdout, assume_unique=False)
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)
        m.fit(X[fit_idx], y[fit_idx])
        pred = np.maximum(0, np.rint(m.predict(X[holdout]))).astype(int)
        fold_mae.append(float(np.abs(pred - y[holdout]).mean()))
    return {"cv_mae": float(np.mean(fold_mae)), "fold_mae": fold_mae, "folds": folds, "seed": seed}


def fit_count_models(x_train: np.ndarray, y_train: np.ndarray) -> dict:
    alpha, cv = count.choose_alpha(x_train, y_train)
    ridge = count.fit_ridge(x_train, y_train, alpha)
    hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)
    hgb.fit(x_train, y_train)
    hgb_cv = hgb_cv_mae(x_train, y_train)
    return {"ridge": ridge, "ridge_alpha": alpha, "ridge_cv": cv,
            "hgb": hgb, "hgb_cv": hgb_cv}


def predict_counts(model_kind: str, models: dict, X: np.ndarray) -> np.ndarray:
    if model_kind == "ridge":
        return count.predict_count(models["ridge"], X)
    if model_kind == "hgb":
        return np.maximum(0, np.rint(models["hgb"].predict(X))).astype(int)
    raise ValueError(model_kind)


# ---------------------------------------------------------------------------
# GSP payload construction for the declared grid
# ---------------------------------------------------------------------------

def payload_for_tau(records: dict, per_tree: dict, tau_prob: float, max_size: int):
    tau_logit = math.log(tau_prob / (1. - tau_prob))
    payload = []
    tags: dict[str, int] = defaultdict(int)
    for tree_id, rec in records.items():
        info = per_tree[tree_id]
        chosen, tag = gsp.solve_partition(len(info["dets"]), info["candidates"], tau_logit, max_size)
        tags[tag] += 1
        payload.append((rec, info["dets"], gsp.decided_edges(chosen)))
    return payload, dict(tags)


def train_class_prior(cfg: dict) -> np.ndarray:
    train_records = count.four_side(base.load_records(cfg, "train"))
    prior = np.bincount(
        [b["cls"] for r in train_records.values() for b in r["bunches"] if 0 <= b["cls"] < K],
        minlength=K).astype(float)
    prior /= max(float(prior.sum()), 1.)
    return prior


# ---------------------------------------------------------------------------
# Final selection helpers
# ---------------------------------------------------------------------------

def select_allrounder(dataset: str, rows: list[dict]) -> dict | None:
    current = CURRENT_BEST[dataset]
    candidates = []
    if dataset == "953":
        for r in rows:
            m = r["metrics"]
            if (m["counting"]["mae"] <= 1.35
                    and m["counting"]["plus_minus_1_accuracy"] >= 0.65
                    and m["classification"]["matched_class_accuracy"] > current["matched"]):
                candidates.append(r)
    else:
        tol = DEPTH_NO_REGRESSION_TOLERANCE
        for r in rows:
            m = r["metrics"]
            no_regress = (
                m["physical_detection"]["f1"] >= current["f1"] - tol["f1"]
                and m["counting"]["mae"] <= current["mae"] + tol["mae"]
                and m["counting"]["plus_minus_1_accuracy"] >= current["pm1"] - tol["pm1"]
                and m["classification"]["macro_f1_end_to_end"] >= current["macro"] - tol["macro"])
            improves = (m["classification"]["matched_class_accuracy"] > current["matched"]
                        or m["physical_detection"]["f1"] > current["f1"])
            if no_regress and improves:
                candidates.append(r)
    if not candidates:
        return None
    return max(candidates, key=lambda r: rank_key(r["metrics"]))


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------

def run_dataset(dataset: str, votes: dict, wall_times: dict) -> dict:
    print(f"\n########## DATASET {dataset} ##########", flush=True)
    cfg = edge.cfg_for(dataset)
    prior = base.build_rotation_prior(base.load_records(cfg, "train"))
    class_prior_vec = train_class_prior(cfg)
    train_records = count.four_side(base.load_records(cfg, "train"))
    val_records = count.four_side(base.load_records(cfg, "val"))

    # --- edge retrain, per mode ---
    edge_models, edge_diag = {}, {}
    for mode in MODES:
        t0 = time.time()
        model, diag, path = train_edge_v2(dataset, mode, votes, prior)
        edge_models[mode] = model
        edge_diag[mode] = {**diag, "model_path": path,
                           "old_reference": OLD_EDGE_DIAG[dataset]}
        wall_times[f"{dataset}_edge_{mode}"] = time.time() - t0

    # --- dets/candidates cache + count features, per mode ---
    per_tree_train, per_tree_val, count_bundle = {}, {}, {}
    for mode in MODES:
        t0 = time.time()
        per_tree_train[mode] = build_dets_and_candidates(
            train_records, votes[dataset][mode]["train"], prior, edge_models[mode])
        per_tree_val[mode] = build_dets_and_candidates(
            val_records, votes[dataset][mode]["val"], prior, edge_models[mode])
        x_train, y_train_cnt, train_ids = build_count_features(
            train_records, votes[dataset][mode]["train"], per_tree_train[mode])
        x_val, y_val_cnt, val_ids = build_count_features(
            val_records, votes[dataset][mode]["val"], per_tree_val[mode])
        models = fit_count_models(x_train, y_train_cnt)
        out_dir = ARTIFACT_ROOT / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(models["ridge"], out_dir / f"count_ridge_{mode}.joblib", compress=3)
        joblib.dump(models["hgb"], out_dir / f"count_hgb_{mode}.joblib", compress=3)
        count_bundle[mode] = {
            "models": models, "x_val": x_val, "val_ids": val_ids,
            "n_features": int(x_train.shape[1]), "n_train_trees": len(train_ids),
            "n_val_trees": len(val_ids),
            "ridge_alpha": models["ridge_alpha"], "ridge_cv": models["ridge_cv"],
            "hgb_cv": models["hgb_cv"],
            "ridge_path": str(out_dir / f"count_ridge_{mode}.joblib"),
            "hgb_path": str(out_dir / f"count_hgb_{mode}.joblib"),
        }
        wall_times[f"{dataset}_count_{mode}"] = time.time() - t0
        print(json.dumps({"stage": "count_fit", "dataset": dataset, "mode": mode,
                          "n_features": count_bundle[mode]["n_features"],
                          "ridge_alpha": models["ridge_alpha"],
                          "ridge_cv_mae": models["ridge_cv"]["selected"]["cv_mae"],
                          "hgb_cv_mae": models["hgb_cv"]["cv_mae"]},
                         default=_json_default), flush=True)

    # --- stage 1: fixed singleton/rank/ridge/blend=0, sweep mode x tau x max_size ---
    t0 = time.time()
    target_counts_ridge = {}
    for mode in MODES:
        pred = predict_counts("ridge", count_bundle[mode]["models"], count_bundle[mode]["x_val"])
        target_counts_ridge[mode] = {tid: int(n) for tid, n in zip(count_bundle[mode]["val_ids"], pred)}

    stage1_rows = []
    for mode in MODES:
        for max_size in GSP_MAX_SIZES:
            for tau_prob in GSP_TAU_PROBS:
                payload, tags = payload_for_tau(val_records, per_tree_val[mode], tau_prob, max_size)
                metrics = head_eval.evaluate_payload(
                    payload, target_counts_ridge[mode], 0.5, STAGE1_SINGLETON, max_size,
                    STAGE1_RANK, 0.0, class_prior_vec, 0.0, None, "mean", 0.0)
                stage1_rows.append({
                    "mode": mode, "tau_prob": tau_prob, "max_size": max_size,
                    "singleton_min": STAGE1_SINGLETON, "rank_mode": STAGE1_RANK,
                    "count_model": "ridge", "count_blend": 0.0,
                    "solver_tag_counts": tags, "summary": summary(metrics),
                    "metrics": metrics,
                })
    stage1_sorted = sorted(stage1_rows, key=lambda r: rank_key(r["metrics"]), reverse=True)
    top2 = stage1_sorted[:2]
    wall_times[f"{dataset}_stage1"] = time.time() - t0
    print(json.dumps({"stage": "stage1_done", "dataset": dataset, "n_rows": len(stage1_rows),
                      "top2": [{"mode": r["mode"], "tau_prob": r["tau_prob"],
                                "max_size": r["max_size"], "summary": r["summary"]}
                               for r in top2]}, default=_json_default), flush=True)

    # --- stage 2: grid singleton x rank x blend x count-model, per top-2 triple ---
    t0 = time.time()
    stage2_blocks = []
    for triple in top2:
        mode, tau_prob, max_size = triple["mode"], triple["tau_prob"], triple["max_size"]
        payload, tags = payload_for_tau(val_records, per_tree_val[mode], tau_prob, max_size)
        target_counts = {}
        for cm in STAGE2_COUNT_MODELS:
            pred = predict_counts(cm, count_bundle[mode]["models"], count_bundle[mode]["x_val"])
            target_counts[cm] = {tid: int(n) for tid, n in zip(count_bundle[mode]["val_ids"], pred)}
        rows = []
        for singleton in STAGE2_SINGLETONS:
            for rank_mode in STAGE2_RANKS:
                for blend in STAGE2_BLENDS:
                    for cm in STAGE2_COUNT_MODELS:
                        metrics = head_eval.evaluate_payload(
                            payload, target_counts[cm], 0.5, singleton, max_size, rank_mode,
                            0.0, class_prior_vec, 0.0, None, "mean", blend)
                        rows.append({
                            "mode": mode, "tau_prob": tau_prob, "max_size": max_size,
                            "singleton_min": singleton, "rank_mode": rank_mode,
                            "count_blend": blend, "count_model": cm,
                            "solver_tag_counts": tags, "summary": summary(metrics),
                            "metrics": metrics,
                        })
        stage2_blocks.append({"triple": {"mode": mode, "tau_prob": tau_prob, "max_size": max_size},
                              "rows": rows})
    wall_times[f"{dataset}_stage2"] = time.time() - t0
    flat_stage2 = [r for block in stage2_blocks for r in block["rows"]]
    print(json.dumps({"stage": "stage2_done", "dataset": dataset,
                      "n_rows": len(flat_stage2)}, default=_json_default), flush=True)

    # --- final selection ---
    best_matched = max(flat_stage2, key=lambda r: (
        r["metrics"]["classification"]["matched_class_accuracy"],
        r["metrics"]["physical_detection"]["f1"]))
    best_f1 = max(flat_stage2, key=lambda r: (
        r["metrics"]["physical_detection"]["f1"],
        r["metrics"]["classification"]["matched_class_accuracy"]))
    best_allrounder = select_allrounder(dataset, flat_stage2)

    print(json.dumps({
        "stage": "final_selection", "dataset": dataset,
        "best_matched": {k: best_matched[k] for k in ("mode", "tau_prob", "max_size",
                          "singleton_min", "rank_mode", "count_blend", "count_model")},
        "best_matched_summary": best_matched["summary"],
        "best_f1_summary": best_f1["summary"],
        "best_allrounder_summary": (best_allrounder["summary"] if best_allrounder else None),
    }, default=_json_default), flush=True)

    counting_compromise_fixed = False
    if dataset == "953":
        counting_compromise_fixed = any(
            r["metrics"]["counting"]["mae"] <= 1.35
            and r["metrics"]["classification"]["matched_class_accuracy"] > CURRENT_BEST["953"]["matched"]
            for r in flat_stage2)

    return {
        "dataset": dataset, "n_train_trees": len(train_records), "n_val_trees": len(val_records),
        "class_prior_train": class_prior_vec.tolist(),
        "current_best_reference": CURRENT_BEST[dataset],
        "edge_diagnostics": edge_diag,
        "count_layer": {
            mode: {k: v for k, v in count_bundle[mode].items() if k != "models"}
            for mode in MODES
        },
        "stage1_grid": stage1_rows,
        "stage1_top2": top2,
        "stage2_blocks": stage2_blocks,
        "selected_profiles": {
            "best_by_matched": best_matched, "best_by_f1": best_f1,
            "best_allrounder": best_allrounder,
        },
        "counting_compromise_fixed_953": counting_compromise_fixed if dataset == "953" else None,
        "depth_no_regression_tolerance": DEPTH_NO_REGRESSION_TOLERANCE if dataset == "depth" else None,
        "wall_times": {k: v for k, v in wall_times.items() if k.startswith(dataset)},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t_start = time.time()
    wall_times: dict[str, float] = {}
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    anchor = run_sanity_anchor()
    wall_times["sanity_anchor"] = anchor["seconds"]
    if not anchor["passed"]:
        for dataset in DATASETS:
            dump_json(ARTIFACT_ROOT / dataset / "results_val.json", {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "status": "ABORTED: sanity anchor failed",
                "sanity_anchor": anchor,
            })
        print(json.dumps({"ANCHOR_GATE": "FAILED", "anchor": anchor},
                         default=_json_default, indent=2))
        return 1

    t0 = time.time()
    votes, vote_diag = build_all_v2_votes()
    vote_paths = save_votes(votes)
    wall_times["v2_votes"] = time.time() - t0

    per_dataset_results = {}
    for dataset in DATASETS:
        per_dataset_results[dataset] = run_dataset(dataset, votes, wall_times)

    wall_times["total_seconds"] = time.time() - t_start

    for dataset in DATASETS:
        out = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "OK",
            "sanity_anchor": anchor,
            "v2_vote_diagnostics": vote_diag,
            "v2_vote_files": {k: v for k, v in vote_paths.items() if k.startswith(dataset)},
            "constants": {
                "ranker_floor": RANKER_FLOOR, "proposal_min_v2": PROPOSAL_MIN_V2,
                "pair_mode": PAIR_MODE, "p_floor": P_FLOOR, "enum_max_size": ENUM_MAX_SIZE,
                "count_feature_taus": list(COUNT_FEATURE_TAUS),
                "count_feature_max_size": COUNT_FEATURE_MAX_SIZE,
                "gsp_tau_probs": GSP_TAU_PROBS, "gsp_max_sizes": GSP_MAX_SIZES,
                "stage1_singleton": STAGE1_SINGLETON, "stage1_rank": STAGE1_RANK,
                "stage2_singletons": STAGE2_SINGLETONS, "stage2_ranks": STAGE2_RANKS,
                "stage2_blends": STAGE2_BLENDS, "stage2_count_models": STAGE2_COUNT_MODELS,
            },
            **per_dataset_results[dataset],
            "wall_times": {k: v for k, v in wall_times.items()
                          if k.startswith(dataset) or k in ("sanity_anchor", "v2_votes", "total_seconds")},
        }
        out_path = ARTIFACT_ROOT / dataset / "results_val.json"
        dump_json(out_path, out)
        print(f"-> wrote {out_path}", flush=True)

    print(json.dumps({"wall_times": wall_times}, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
