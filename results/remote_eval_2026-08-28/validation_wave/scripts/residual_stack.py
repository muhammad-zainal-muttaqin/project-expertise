"""Dynamic residual meta-stack for frozen TRAIN/VAL cluster topology.

The detector distribution is an explicit skip path.  Additional opinions
come from the existing member, multi-scale, cluster, and auxiliary heads;
two small meta-classifiers learn how to gate those opinions on matched TRAIN
clusters and are evaluated on VAL.  No test split or test artifact is loaded.

This experiment is intentionally limited to inexpensive heads over already
extracted features.  It is a targeted attempt to reduce the known class
confusions without changing physical linking or count reconciliation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import harness
import member_head as mh
import multiscale_member_head as ms

sys.path.insert(0, "/workspace/aux_modal")
import multimodal_cluster_head as mm  # noqa: E402


OUT = Path("/workspace/cluster_head/artifacts")
K = harness.K


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


def _softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-8)


def _pool(q_member: np.ndarray, data: dict, pooling: str) -> np.ndarray:
    flat = [g for _rec, gs in data["groups"] for g in gs]
    out = []
    for gi, rows in enumerate(data["group_rows"]):
        q = q_member[rows]
        members = flat[gi]["members"]
        if pooling == "max":
            out.append(q.max(axis=0))
        elif pooling == "mean":
            w = np.asarray([float(m["score"]) for m in members], dtype=np.float32)
            w /= max(float(w.sum()), 1e-8)
            out.append((q * w[:, None]).sum(axis=0))
        elif pooling == "top":
            out.append(q[int(np.argmax([float(m["score"]) for m in members]))])
        else:
            raise ValueError(pooling)
    return np.asarray(out, dtype=np.float32)


def _member_experts(dataset: str, split: str, data: dict) -> dict[str, np.ndarray]:
    out = {}
    for stem, model_name in (("member_extra", "member_extra"),
                             ("member_logistic", "member_logistic")):
        model = joblib.load(OUT / f"{dataset}_{model_name}.joblib")
        q = np.asarray(model.predict_proba(data["X"]), dtype=np.float32)
        for pooling in ("mean", "max", "top"):
            out[f"{stem}_{pooling}"] = _pool(q, data, pooling)
    return out


def _multiscale_experts(dataset: str, split: str) -> dict[str, np.ndarray]:
    data = ms.collect(dataset, split)
    out = {}
    for stem, model_name in (("ms_extra", "ms_extra"),
                             ("ms_logistic", "ms_logistic")):
        model = joblib.load(OUT / f"{dataset}_{model_name}.joblib")
        q = np.asarray(model.predict_proba(data["X"]), dtype=np.float32)
        for pooling in ("mean", "max", "top"):
            out[f"{stem}_{pooling}"] = _pool(q, data, pooling)
    return out, data


def _cluster_experts(dataset: str, split: str, data: dict) -> dict[str, np.ndarray]:
    out = {}
    for model_name in ("extra_compact", "hist_compact", "linear_pca128"):
        path = OUT / f"{dataset}_{model_name}.joblib"
        if path.exists():
            model = joblib.load(path)
            out[f"cluster_{model_name}"] = np.asarray(
                model.predict_proba(data["X"]), dtype=np.float32)
    return out


def _aux_expert(dataset: str, split: str, data: dict) -> dict[str, np.ndarray]:
    path = Path(f"/workspace/aux_modal/artifacts/{dataset}_base_aux_extra_pca128.joblib")
    if not path.exists():
        return {}
    x, _diag = mm.modal_matrix(data, dataset, split, "base_aux")
    model = joblib.load(path)
    return {"aux_base_extra": np.asarray(model.predict_proba(x), dtype=np.float32)}


def _base_features(data: dict) -> np.ndarray:
    # The final block of group_features is the low-dimensional detector and
    # geometry summary.  Keep it as a residual side channel; the classifier
    # never needs to see the 3*1536 raw embedding blocks again.
    x = np.asarray(data["X"], dtype=np.float32)
    tail = x[:, -48:] if x.shape[1] >= 48 else x
    detector = []
    for _rec, groups in data["groups"]:
        for group in groups:
            p = np.asarray(group["p"], dtype=np.float32)
            p = np.maximum(p, 1e-8)
            p /= max(float(p.sum()), 1e-8)
            detector.append(p)
    detector = np.asarray(detector, dtype=np.float32)
    return np.concatenate([np.log(np.maximum(detector, 1e-8)), tail], axis=1)


def build_split(dataset: str, split: str) -> dict:
    data = __import__("cluster_head_experiment").collect_groups(dataset, split)
    member = mh.collect(dataset, split)
    if data["keys"] != member["keys"]:
        raise RuntimeError(f"{dataset}/{split}: member/group topology mismatch")
    ms_experts, ms_data = _multiscale_experts(dataset, split)
    if data["keys"] != ms_data["keys"]:
        raise RuntimeError(f"{dataset}/{split}: multiscale topology mismatch")
    experts = {}
    experts.update(_member_experts(dataset, split, member))
    experts.update(ms_experts)
    experts.update(_cluster_experts(dataset, split, data))
    experts.update(_aux_expert(dataset, split, data))
    base = _base_features(data)
    opinion_blocks = [base]
    names = ["detector_skip_and_geometry"]
    for name in sorted(experts):
        q = np.maximum(np.asarray(experts[name], dtype=np.float32), 1e-8)
        q /= np.maximum(q.sum(axis=1, keepdims=True), 1e-8)
        opinion_blocks.append(np.log(q))
        names.append(name)
    x = np.concatenate(opinion_blocks, axis=1).astype(np.float32)
    return {"data": data, "X": x, "y": data["y"],
            "experts": experts, "feature_names": names}


def model_bank(seed: int):
    yield "meta_logistic", Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=.15, max_iter=500, solver="lbfgs",
                                    class_weight="balanced", random_state=seed)),
    ])
    yield "meta_extra", ExtraTreesClassifier(
        n_estimators=600, min_samples_leaf=4, max_features="sqrt",
        class_weight="balanced", n_jobs=8, random_state=seed,
    )
    yield "meta_hist", HistGradientBoostingClassifier(
        max_iter=260, learning_rate=.045, max_leaf_nodes=15,
        min_samples_leaf=18, l2_regularization=3., random_state=seed,
    )


def evaluate(dataset: str, val: dict, probs: np.ndarray, detector: np.ndarray,
             alpha: float, mode: str) -> dict:
    if mode == "head":
        fused = probs
    elif mode == "blend":
        fused = _softmax(np.log(np.maximum(detector, 1e-8))
                         + alpha * np.log(np.maximum(probs, 1e-8)))
    elif mode == "residual":
        residual = np.log(np.maximum(probs, 1e-8)) - np.log(1. / K)
        fused = _softmax(np.log(np.maximum(detector, 1e-8)) + alpha * residual)
    else:
        raise ValueError(mode)
    pred = np.argmax(fused, axis=1).astype(int)
    pmap = {key: int(cls) for key, cls in zip(val["data"]["keys"], pred)}
    m = harness.evaluate_clusters(
        val["data"]["payload"], val["data"]["targets"],
        harness.PROFILES[dataset],
        lambda g, pmap=pmap: pmap[__import__("member_head").harness_group_key(g)],
    )
    return short(m)


def run(dataset: str, seed: int) -> dict:
    started = time.time()
    train, val = build_split(dataset, "train"), build_split(dataset, "val")
    mask = train["y"] >= 0
    if mask.sum() < 20 or len(np.unique(train["y"][mask])) < K:
        raise RuntimeError(f"{dataset}: insufficient matched training groups")
    baseline_m = harness.evaluate_clusters(
        val["data"]["payload"], val["data"]["targets"], harness.PROFILES[dataset])
    baseline = short(baseline_m)
    detector = val["experts"].get("detector")
    if detector is None:
        detector = []
        for _rec, groups in val["data"]["groups"]:
            for group in groups:
                p = np.asarray(group["p"], dtype=np.float32)
                p = np.maximum(p, 1e-8)
                detector.append(p / max(float(p.sum()), 1e-8))
        detector = np.asarray(detector, dtype=np.float32)
    rows = []
    xtr, ytr = train["X"][mask], train["y"][mask]
    for name, model in model_bank(seed):
        fit_start = time.time()
        model.fit(xtr, ytr)
        q = np.asarray(model.predict_proba(val["X"]), dtype=np.float32)
        for mode, alpha in [("head", 1.0), ("blend", .05), ("blend", .10),
                            ("blend", .15), ("blend", .25), ("blend", .40),
                            ("blend", .60), ("blend", .85), ("blend", 1.0),
                            ("residual", .15), ("residual", .25),
                            ("residual", .40), ("residual", .60)]:
            metrics = evaluate(dataset, val, q, detector, alpha, mode)
            invariant = bool(
                abs(metrics["physical_f1"] - baseline["physical_f1"]) < 1e-10
                and abs(metrics["mae"] - baseline["mae"]) < 1e-10
                and abs(metrics["pm1"] - baseline["pm1"]) < 1e-10)
            metrics["physical_count_invariant"] = invariant
            rows.append({"model": name, "mode": mode, "alpha": alpha,
                         "metrics": metrics,
                         "fit_elapsed_sec": time.time() - fit_start})
        joblib.dump(model, OUT / f"{dataset}_{name}.joblib", compress=3)
        print(json.dumps({"dataset": dataset, "model": name,
                          "top": sorted(rows,
                              key=lambda r: (r["metrics"]["matched_class_accuracy"],
                                             r["metrics"]["macro_f1"]),
                              reverse=True)[:3]}, ensure_ascii=False), flush=True)
    eligible = [r for r in rows if r["metrics"]["physical_count_invariant"]]
    best_match = max(eligible, key=lambda r: (
        r["metrics"]["matched_class_accuracy"], r["metrics"]["macro_f1"]))
    best_macro = max(eligible, key=lambda r: (
        r["metrics"]["macro_f1"], r["metrics"]["matched_class_accuracy"]))
    report = {"dataset": dataset,
              "protocol": "fit dynamic residual meta-head on matched TRAIN; select VAL; no TEST",
              "feature_names": train["feature_names"],
              "train": {"groups": int(len(train["y"])),
                        "matched_groups": int(mask.sum()),
                        "features": int(train["X"].shape[1])},
              "val": {"groups": int(len(val["y"])),
                      "matched_groups": int((val["y"] >= 0).sum()),
                      "features": int(val["X"].shape[1])},
              "baseline_val": baseline, "best_by_matched": best_match,
              "best_by_macro": best_macro, "results": rows,
              "elapsed_sec": time.time() - started}
    out = OUT / f"{dataset}_residual_stack_results.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({"dataset": dataset, "best_by_matched": best_match,
                      "best_by_macro": best_macro, "report": str(out)},
                     ensure_ascii=False), flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("953", "depth"), required=True)
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()
    run(args.dataset, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
