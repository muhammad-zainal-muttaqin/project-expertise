"""Validation-only ensemble sweep for the V2 count reconciliation layer.

The V2 proposal/link topology is reused from its saved TRAIN/VAL artifacts;
only the train-fitted tree-count regressor changes.  This targets the known
953 compromise (better matching but slightly worse count MAE) without any
test read or test inference.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor

sys.path.insert(0, "/workspace/pipeline_v2")
import pipeline_v2 as v2  # noqa: E402


OUT = Path("/workspace/pipeline_v3_count/artifacts")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {str(k): np.asarray(z[k]) for k in z.files}


def fit_ridge(X, y, alpha):
    return v2.count.fit_ridge(X, y, alpha)


def predict(kind, model, X):
    if kind == "ridge":
        return v2.count.predict_count(model, X)
    return np.maximum(0, np.rint(model.predict(X))).astype(int)


def model_bank(seed):
    yield "ridge_a10", "ridge", fit_ridge
    yield "ridge_a100", "ridge", fit_ridge
    yield "hgb_small", "sklearn", HistGradientBoostingRegressor(
        max_iter=260, learning_rate=.045, max_leaf_nodes=7,
        min_samples_leaf=10, l2_regularization=2., random_state=seed)
    yield "hgb_medium", "sklearn", HistGradientBoostingRegressor(
        max_iter=320, learning_rate=.05, max_leaf_nodes=15,
        min_samples_leaf=8, l2_regularization=3., random_state=seed)
    yield "extra", "sklearn", ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=3, max_features=.8,
        n_jobs=8, random_state=seed)
    yield "random_forest", "sklearn", RandomForestRegressor(
        n_estimators=500, min_samples_leaf=3, max_features=.8,
        n_jobs=8, random_state=seed)
    yield "gradient_boost", "sklearn", GradientBoostingRegressor(
        n_estimators=220, learning_rate=.04, max_depth=2,
        min_samples_leaf=8, loss="huber", random_state=seed)


def fit_model(name, kind, factory, X, y, seed):
    if name == "ridge_a10":
        return factory(X, y, 10.)
    if name == "ridge_a100":
        return factory(X, y, 100.)
    model = factory
    model.fit(X, y)
    return model


def summary(m):
    return {
        "f1": float(m["physical_detection"]["f1"]),
        "mae": float(m["counting"]["mae"]),
        "exact": float(m["counting"]["exact_accuracy"]),
        "pm1": float(m["counting"]["plus_minus_1_accuracy"]),
        "matched": float(m["classification"]["matched_class_accuracy"]),
        "macro": float(m["classification"]["macro_f1_end_to_end"]),
        "pred_clusters": int(m["physical_detection"]["pred_clusters"]),
    }


def cv_mae(model_name, kind, factory, X, y, seed):
    # Deterministic tree-level folds, fit only on TRAIN.  This is a diagnostic
    # to expose overfitting; VAL remains the sole selection split below.
    order = np.random.RandomState(seed).permutation(len(y))
    folds = np.array_split(order, 5)
    scores = []
    for holdout in folds:
        fit_idx = np.setdiff1d(order, holdout, assume_unique=False)
        if kind == "ridge":
            alpha = 10. if model_name == "ridge_a10" else 100.
            m = fit_ridge(X[fit_idx], y[fit_idx], alpha)
        else:
            # Clone by reconstructing the estimator from its parameters.
            m = factory.__class__(**factory.get_params())
            m.fit(X[fit_idx], y[fit_idx])
        pred = predict(kind, m, X[holdout])
        scores.append(float(np.abs(pred - y[holdout]).mean()))
    return {"cv_mae": float(np.mean(scores)), "fold_mae": scores, "folds": 5}


def run(dataset: str, seed: int):
    if dataset != "953":
        raise ValueError("this targeted run is for the 953 count compromise")
    started = time.time()
    cfg = v2.edge.cfg_for(dataset)
    train_records = v2.count.four_side(v2.base.load_records(cfg, "train"))
    val_records = v2.count.four_side(v2.base.load_records(cfg, "val"))
    prior = v2.base.build_rotation_prior(v2.base.load_records(cfg, "train"))
    class_prior = v2.train_class_prior(cfg)
    modes = {}
    for mode in v2.MODES:
        train_vote = load_npz(v2.ARTIFACT_ROOT / dataset / f"vote_v2_{mode}_train.npz")
        val_vote = load_npz(v2.ARTIFACT_ROOT / dataset / f"vote_v2_{mode}_val.npz")
        edge_model = joblib.load(v2.ARTIFACT_ROOT / dataset / f"edge_v2_{mode}.joblib")
        if hasattr(edge_model, "n_jobs"):
            edge_model.n_jobs = 1
        ptrain = v2.build_dets_and_candidates(train_records, train_vote, prior, edge_model)
        pval = v2.build_dets_and_candidates(val_records, val_vote, prior, edge_model)
        Xtr, ytr, _ = v2.build_count_features(train_records, train_vote, ptrain)
        Xva, yva, val_ids = v2.build_count_features(val_records, val_vote, pval)
        modes[mode] = {"train_vote": train_vote, "val_vote": val_vote,
                       "ptrain": ptrain, "pval": pval, "Xtr": Xtr,
                       "Xva": Xva, "ytr": ytr, "yva": yva, "val_ids": val_ids}

    rows = []
    topologies = [(tau, max_size) for max_size in v2.GSP_MAX_SIZES
                  for tau in v2.GSP_TAU_PROBS]
    rank_modes = ["score", "support", "max_member"]
    singletons = [.10, .15, .20, .25]
    blends = [0., .25, .5]
    for mode, info in modes.items():
        fitted = []
        for name, kind, factory in model_bank(seed):
            t0 = time.time()
            model = fit_model(name, kind, factory, info["Xtr"], info["ytr"], seed)
            cv = cv_mae(name, kind, factory, info["Xtr"], info["ytr"], seed)
            pred = predict(kind, model, info["Xva"])
            fitted.append((name, kind, model, pred, cv, time.time() - t0))
            print(json.dumps({"stage": "count_model", "dataset": dataset,
                              "mode": mode, "model": name,
                              "cv_mae": cv["cv_mae"],
                              "elapsed_sec": fitted[-1][-1]}, ensure_ascii=False),
                  flush=True)
        for name, kind, model, pred, cv, fit_sec in fitted:
            target_counts = {tid: int(n) for tid, n in zip(info["val_ids"], pred)}
            for tau, max_size in topologies:
                payload, tags = v2.payload_for_tau(val_records, info["pval"], tau, max_size)
                for singleton in singletons:
                    for rank_mode in rank_modes:
                        for blend in blends:
                            metrics = v2.head_eval.evaluate_payload(
                                payload, target_counts, .5, singleton, max_size,
                                rank_mode, 0., class_prior, 0., None, "mean", blend)
                            s = summary(metrics)
                            rows.append({"mode": mode, "model": name,
                                         "cv": cv, "tau_prob": tau,
                                         "max_size": max_size,
                                         "singleton_min": singleton,
                                         "rank_mode": rank_mode,
                                         "count_blend": blend,
                                         "solver_tag_counts": tags,
                                         "metrics": s,
                                         "fit_elapsed_sec": fit_sec})
            OUT.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, OUT / f"{dataset}_{mode}_{name}.joblib", compress=3)
        print(json.dumps({"stage": "mode_done", "dataset": dataset,
                          "mode": mode, "rows": len(rows)}, ensure_ascii=False),
              flush=True)

    current = v2.CURRENT_BEST[dataset]
    eligible = [r for r in rows if r["metrics"]["mae"] <= 1.35
                and r["metrics"]["matched"] > current["matched"]]
    best_allrounder = max(eligible, key=lambda r: (r["metrics"]["matched"],
                                                    r["metrics"]["f1"],
                                                    -r["metrics"]["mae"])) if eligible else None
    best_mae = min(rows, key=lambda r: (r["metrics"]["mae"],
                                         -r["metrics"]["matched"]))
    best_match = max(rows, key=lambda r: (r["metrics"]["matched"],
                                           r["metrics"]["f1"],
                                           -r["metrics"]["mae"]))
    report = {"dataset": dataset,
              "protocol": "count regressor fit TRAIN; topology/profile selected VAL; no TEST",
              "n_train_trees": len(train_records), "n_val_trees": len(val_records),
              "feature_dimensions": {mode: int(info["Xtr"].shape[1])
                                     for mode, info in modes.items()},
              "current_reference": current,
              "topologies": topologies, "rows": rows,
              "best_allrounder_953": best_allrounder,
              "best_by_mae": best_mae, "best_by_matched": best_match,
              "elapsed_sec": time.time() - started}
    out = OUT / f"{dataset}_results_val.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({"best_allrounder_953": best_allrounder,
                      "best_by_mae": best_mae, "best_by_matched": best_match,
                      "report": str(out)}, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="953", choices=("953",))
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()
    run(args.dataset, args.seed)


if __name__ == "__main__":
    main()
