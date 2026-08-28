"""Paired VAL bootstrap for selected member-head stackers; never TEST."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

import member_head as mh


OUT = Path("/workspace/cluster_head/artifacts")


def pool(q, data, pooling):
    flat = [g for _rec, gs in data["groups"] for g in gs]
    out = []
    for gi, rows in enumerate(data["group_rows"]):
        z = q[rows]
        if pooling == "mean":
            w = np.asarray([float(m["score"]) for m in flat[gi]["members"]], dtype=np.float32)
            w /= max(float(w.sum()), 1e-8)
            out.append((z * w[:, None]).sum(axis=0))
        elif pooling == "max":
            out.append(z.max(axis=0))
        elif pooling == "top":
            j = int(np.argmax([float(m["score"]) for m in flat[gi]["members"]]))
            out.append(z[j])
    return np.asarray(out, dtype=np.float32)


def collect_arrays(data, qe, ql, pooling, we, wl):
    pe, pl = pool(qe, data, pooling), pool(ql, data, pooling)
    detector = np.asarray([np.asarray(g["p"], dtype=np.float32)
                           for _rec, gs in data["groups"] for g in gs])
    detector = np.maximum(detector, 1e-8)
    detector /= detector.sum(axis=1, keepdims=True)
    z = np.log(detector) + we * np.log(np.maximum(pe, 1e-8)) \
        + wl * np.log(np.maximum(pl, 1e-8))
    z -= z.max(axis=1, keepdims=True)
    candidate = np.argmax(z, axis=1)
    baseline = np.argmax(detector, axis=1)
    base_correct, cand_correct, matched = [], [], []
    base_by_class, cand_by_class, gt_by_class = [], [], []
    offset = 0
    for rec, groups in data["groups"]:
        b = c = 0
        bb = np.zeros(mh.harness.K, dtype=np.int32)
        cc = np.zeros(mh.harness.K, dtype=np.int32)
        gg = np.zeros(mh.harness.K, dtype=np.int32)
        matches = mh.harness.count.tree_matches(rec, groups)
        for gi, gj in matches:
            gt = int(rec["bunches"][gj]["cls"])
            bp, cp = int(baseline[offset + gi]), int(candidate[offset + gi])
            if 0 <= gt < mh.harness.K:
                gg[gt] += 1
                bb[gt] += int(bp == gt)
                cc[gt] += int(cp == gt)
                b += int(bp == gt)
                c += int(cp == gt)
        matched.append(len(matches)); base_correct.append(b); cand_correct.append(c)
        base_by_class.append(bb); cand_by_class.append(cc); gt_by_class.append(gg)
        offset += len(groups)
    return {"base": np.asarray(base_correct), "cand": np.asarray(cand_correct),
            "matched": np.asarray(matched), "base_by_class": np.asarray(base_by_class),
            "cand_by_class": np.asarray(cand_by_class), "gt_by_class": np.asarray(gt_by_class)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("953", "depth"), required=True)
    ap.add_argument("--pooling", default="max", choices=("mean", "max", "top"))
    ap.add_argument("--extra-weight", type=float, default=None)
    ap.add_argument("--logistic-weight", type=float, default=None)
    ap.add_argument("--n-resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()
    if args.extra_weight is None:
        args.extra_weight = .45 if args.dataset == "953" else .30
    if args.logistic_weight is None:
        args.logistic_weight = .10 if args.dataset == "953" else .30
    data = mh.collect(args.dataset, "val")
    extra = joblib.load(OUT / f"{args.dataset}_member_extra.joblib")
    logistic = joblib.load(OUT / f"{args.dataset}_member_logistic.joblib")
    qe = np.asarray(extra.predict_proba(data["X"]), dtype=np.float32)
    ql = np.asarray(logistic.predict_proba(data["X"]), dtype=np.float32)
    a = collect_arrays(data, qe, ql, args.pooling, args.extra_weight, args.logistic_weight)
    n = len(data["groups"])
    rng = np.random.RandomState(args.seed)
    idx = rng.randint(0, n, size=(args.n_resamples, n))
    b = a["base"][idx].sum(axis=1); c = a["cand"][idx].sum(axis=1)
    m = a["matched"][idx].sum(axis=1)
    delta = c / np.maximum(m, 1) - b / np.maximum(m, 1)
    lo, hi = np.percentile(delta, [2.5, 97.5])
    report = {
        "dataset": args.dataset,
        "protocol": "paired tree-level bootstrap on VAL; member experts fit TRAIN; no TEST",
        "seed": args.seed, "n_resamples": args.n_resamples, "n_val_trees": n,
        "pooling": args.pooling, "extra_weight": args.extra_weight,
        "logistic_weight": args.logistic_weight,
        "point_baseline": float(a["base"].sum() / max(a["matched"].sum(), 1)),
        "point_candidate": float(a["cand"].sum() / max(a["matched"].sum(), 1)),
        "point_delta": float((a["cand"].sum() - a["base"].sum()) /
                              max(a["matched"].sum(), 1)),
        "ci95_delta": [float(lo), float(hi)], "delta_mean": float(delta.mean()),
        "frac_delta_gt_zero": float((delta > 0).mean()),
        "excludes_zero": bool(lo > 0 or hi < 0), "matched_total": int(a["matched"].sum()),
        "class_point_delta_correct": (a["cand_by_class"].sum(0) -
                                       a["base_by_class"].sum(0)).tolist(),
        "class_matched_total": a["gt_by_class"].sum(0).tolist(),
        "interpretation": "VAL robustness only; it does not replace the locked test result",
    }
    path = OUT / f"{args.dataset}_member_head_val_bootstrap.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
