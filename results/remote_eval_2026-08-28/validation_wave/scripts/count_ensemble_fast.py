"""Small checkpointed count-regressor follow-up for V2 (953, VAL only)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

sys.path.insert(0, "/workspace/pipeline_v2")
import pipeline_v2 as v2  # noqa: E402


OUT = Path("/workspace/pipeline_v3_count/artifacts")


def load_npz(path):
    with np.load(path, allow_pickle=True) as z:
        return {str(k): np.asarray(z[k]) for k in z.files}


def predict(model, X):
    return np.maximum(0, np.rint(model.predict(X))).astype(int)


def short(m):
    return {"f1": float(m["physical_detection"]["f1"]),
            "mae": float(m["counting"]["mae"]),
            "exact": float(m["counting"]["exact_accuracy"]),
            "pm1": float(m["counting"]["plus_minus_1_accuracy"]),
            "matched": float(m["classification"]["matched_class_accuracy"]),
            "macro": float(m["classification"]["macro_f1_end_to_end"]),
            "pred_clusters": int(m["physical_detection"]["pred_clusters"])}


def chosen_profiles():
    d = json.load(open("/workspace/pipeline_v2/artifacts/953/results_val.json"))
    rows = []
    rows.extend(d.get("stage1_top2", []))
    all_stage2 = [r for block in d.get("stage2_blocks", []) for r in block["rows"]]
    # Keep only a small, auditable frontier from the already-computed V2
    # validation sweep: high matching and low MAE.  These are profile
    # settings, not test-derived choices.
    all_stage2.sort(key=lambda r: (-r["summary"]["matched"], r["summary"]["mae"]))
    rows.extend(all_stage2[:8])
    rows.extend(sorted(all_stage2, key=lambda r: (r["summary"]["mae"],
                                                    -r["summary"]["matched"]))[:8])
    out, seen = [], set()
    for r in rows:
        key = tuple(r[k] for k in ("mode", "tau_prob", "max_size",
                                   "singleton_min", "rank_mode", "count_blend"))
        if key not in seen:
            seen.add(key)
            out.append({k: r[k] for k in ("mode", "tau_prob", "max_size",
                                           "singleton_min", "rank_mode", "count_blend")})
    return out


def model_bank(seed):
    return [
        ("extra", ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3,
                                       max_features=.8, n_jobs=8,
                                       random_state=seed)),
        ("gradient_boost", GradientBoostingRegressor(n_estimators=220,
                                                       learning_rate=.04,
                                                       max_depth=2,
                                                       min_samples_leaf=8,
                                                       loss="huber",
                                                       random_state=seed)),
    ]


def run(seed):
    started = time.time()
    dataset = "953"
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
        Xva, _yva, ids = v2.build_count_features(val_records, val_vote, pval)
        modes[mode] = {"pval": pval, "Xtr": Xtr, "Xva": Xva,
                       "ytr": ytr, "ids": ids}
    profiles = chosen_profiles()
    rows = []
    for mode, info in modes.items():
        for name, model in model_bank(seed):
            t0 = time.time()
            model.fit(info["Xtr"], info["ytr"])
            pred = predict(model, info["Xva"])
            targets = {tid: int(n) for tid, n in zip(info["ids"], pred)}
            for prof in profiles:
                if prof["mode"] != mode:
                    continue
                payload, tags = v2.payload_for_tau(
                    val_records, info["pval"], prof["tau_prob"], prof["max_size"])
                m = v2.head_eval.evaluate_payload(
                    payload, targets, .5, prof["singleton_min"], prof["max_size"],
                    prof["rank_mode"], 0., class_prior, 0., None, "mean",
                    prof["count_blend"])
                rows.append({"mode": mode, "model": name, **prof,
                             "solver_tag_counts": tags, "metrics": short(m),
                             "elapsed_model_sec": time.time() - t0})
            OUT.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, OUT / f"953_{mode}_{name}.joblib", compress=3)
            checkpoint = {"dataset": dataset,
                          "protocol": "fit count model TRAIN; evaluate selected V2 profile frontier VAL; no TEST",
                          "profiles": profiles, "rows": rows,
                          "checkpoint": {"mode": mode, "model": name},
                          "elapsed_sec": time.time() - started}
            (OUT / "953_fast_checkpoint.json").write_text(
                json.dumps(checkpoint, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")
            print(json.dumps({"mode": mode, "model": name,
                              "rows": len(rows), "elapsed_sec": time.time() - t0},
                             ensure_ascii=False), flush=True)
    current = v2.CURRENT_BEST[dataset]
    eligible = [r for r in rows if r["metrics"]["mae"] <= 1.35
                and r["metrics"]["matched"] > current["matched"]]
    best_all = max(eligible, key=lambda r:(r["metrics"]["matched"],
                                            r["metrics"]["f1"],
                                            -r["metrics"]["mae"])) if eligible else None
    best = max(rows, key=lambda r:(r["metrics"]["matched"],
                                   r["metrics"]["f1"],
                                   -r["metrics"]["mae"]))
    report = {"dataset": dataset,
              "protocol": "fit count models TRAIN; choose within predeclared V2 frontier VAL; no TEST",
              "profiles": profiles, "rows": rows,
              "best_allrounder_953": best_all, "best_by_matched": best,
              "elapsed_sec": time.time() - started}
    out = OUT / "953_fast_results_val.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({"best_allrounder_953": best_all,
                      "best_by_matched": best, "report": str(out)},
                     ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()
    run(args.seed)


if __name__ == "__main__":
    main()
