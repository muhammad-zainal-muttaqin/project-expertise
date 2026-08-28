"""Cluster-equal weighted member heads, validation-only."""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import member_head as mh

warnings.filterwarnings(
    "ignore", message=r"`sklearn\.utils\.parallel\.delayed` should be used.*",
    category=UserWarning,
)

OUT = Path("/workspace/cluster_head/artifacts")


def short(m):
    return {"physical_f1": m["physical_detection"]["f1"],
            "mae": m["counting"]["mae"],
            "pm1": m["counting"]["plus_minus_1_accuracy"],
            "matched_class_accuracy": m["classification"]["matched_class_accuracy"],
            "matched": m["classification"]["matched"],
            "macro_f1": m["classification"]["macro_f1_end_to_end"],
            "per_class_f1": m["classification"]["per_class_f1_end_to_end"]}


def pool(q, data, kind):
    flat = [g for _rec, gs in data["groups"] for g in gs]
    out = []
    for gi, rows in enumerate(data["group_rows"]):
        z = q[rows]
        if kind == "mean":
            w = np.asarray([float(m["score"]) for m in flat[gi]["members"]], dtype=np.float32)
            w /= max(float(w.sum()), 1e-8)
            out.append((z * w[:, None]).sum(axis=0))
        elif kind == "max":
            out.append(z.max(axis=0))
        else:
            j = int(np.argmax([float(m["score"]) for m in flat[gi]["members"]]))
            out.append(z[j])
    return np.asarray(out, dtype=np.float32)


def make_model(kind, seed):
    if kind == "logistic":
        return Pipeline([("scale", StandardScaler()),
                         ("pca", PCA(n_components=128, whiten=True, random_state=seed)),
                         ("clf", LogisticRegression(C=.15, max_iter=350, solver="lbfgs",
                                                     class_weight=None, random_state=seed))])
    return Pipeline([("scale", StandardScaler()),
                     ("pca", PCA(n_components=96, whiten=True, random_state=seed)),
                     ("clf", ExtraTreesClassifier(n_estimators=320, min_samples_leaf=3,
                                                  max_features="sqrt", class_weight=None,
                                                  n_jobs=8, random_state=seed))])


def run(dataset, seed):
    started = time.time()
    train, val = mh.collect(dataset, "train"), mh.collect(dataset, "val")
    mask = train["y"] >= 0
    if mask.sum() < 20 or len(np.unique(train["y"][mask])) < mh.harness.K:
        raise RuntimeError("insufficient matched member labels")
    # Equalize cluster contribution, then optionally equalize class totals.
    w_equal = np.zeros(len(train["y"]), dtype=np.float32)
    for rows in train["group_rows"]:
        if rows:
            w_equal[rows] = 1. / len(rows)
    w_equal *= float(mask.sum()) / max(float(w_equal[mask].sum()), 1e-8)
    class_totals = np.bincount(train["y"][mask], weights=w_equal[mask], minlength=mh.harness.K)
    w_bal = w_equal.copy()
    for c in range(mh.harness.K):
        if class_totals[c] > 0:
            w_bal[train["y"] == c] *= float(class_totals.sum()) / (mh.harness.K * class_totals[c])
    baseline = mh.harness.evaluate_clusters(val["payload"], val["targets"],
                                            mh.harness.PROFILES[dataset])
    detector = np.asarray([np.asarray(g["p"], dtype=np.float32)
                           for _rec, gs in val["groups"] for g in gs])
    detector = np.maximum(detector, 1e-8)
    detector /= detector.sum(axis=1, keepdims=True)
    results = []
    for weight_name, weights in (("cluster_equal", w_equal),
                                 ("cluster_equal_class_balanced", w_bal)):
        for model_name in ("logistic", "extra"):
            model = make_model(model_name, seed)
            fit_start = time.time()
            # Pipeline routes sample weights to its final classifier.
            model.fit(train["X"][mask], train["y"][mask],
                      clf__sample_weight=weights[mask])
            q_member = np.asarray(model.predict_proba(val["X"]), dtype=np.float32)
            for pooling in ("mean", "max", "top"):
                q = pool(q_member, val, pooling)
                for alpha in (.10, .20, .30, .45, .60, .80, 1.0):
                    z = np.log(detector) + alpha * np.log(np.maximum(q, 1e-8))
                    pred = np.argmax(z, axis=1).astype(int)
                    pmap = {key: int(cls) for key, cls in zip(val["keys"], pred)}
                    metrics = mh.harness.evaluate_clusters(
                        val["payload"], val["targets"], mh.harness.PROFILES[dataset],
                        lambda g, pmap=pmap: pmap[mh.harness_group_key(g)])
                    results.append({"weighting": weight_name, "model": model_name,
                                    "pooling": pooling, "alpha": alpha,
                                    "metrics": short(metrics),
                                    "fit_elapsed_sec": time.time() - fit_start})
            joblib.dump(model, OUT / f"{dataset}_{weight_name}_{model_name}.joblib", compress=3)
            top = sorted(results, key=lambda r: (r["metrics"]["matched_class_accuracy"],
                                                 r["metrics"]["macro_f1"]), reverse=True)[:3]
            print(json.dumps({"dataset": dataset, "weighting": weight_name,
                              "model": model_name, "top": top}, ensure_ascii=False), flush=True)
    eligible = [r for r in results if (
        abs(r["metrics"]["physical_f1"] - baseline["physical_detection"]["f1"]) < 1e-10
        and abs(r["metrics"]["mae"] - baseline["counting"]["mae"]) < 1e-10
        and abs(r["metrics"]["pm1"] - baseline["counting"]["plus_minus_1_accuracy"]) < 1e-10)]
    best = max(eligible, key=lambda r: (r["metrics"]["matched_class_accuracy"],
                                        r["metrics"]["macro_f1"]))
    report = {"dataset": dataset,
              "protocol": "fit weighted matched TRAIN members; select VAL; no TEST",
              "train": {"members": len(train["X"]), "matched_members": train["matched_members"],
                        "missing": train["missing"]},
              "val": {"members": len(val["X"]), "matched_members": val["matched_members"],
                      "missing": val["missing"]},
              "baseline_val": short(baseline), "results": results,
              "selected_validation": best, "elapsed_sec": time.time() - started}
    path = OUT / f"{dataset}_weighted_member_head_results.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"dataset": dataset, "selected_validation": best,
                      "report": str(path)}, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("953", "depth"), required=True)
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()
    run(args.dataset, args.seed)


if __name__ == "__main__":
    main()
