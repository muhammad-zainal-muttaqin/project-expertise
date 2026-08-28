"""Paired VAL bootstrap for the selected DINOv2-Large/Base stack."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

import harness
import large_member_head as large
import member_head as mh
import multiscale_member_head as ms


OUT = Path("/workspace/cluster_head/artifacts")
K = harness.K


def pool(q, data, pooling):
    flat = [g for _rec, gs in data["groups"] for g in gs]
    out = []
    for gi, rows in enumerate(data["group_rows"]):
        z = q[rows]
        members = flat[gi]["members"]
        if pooling == "mean":
            w = np.asarray([float(m["score"]) for m in members], dtype=np.float32)
            w /= max(float(w.sum()), 1e-8)
            out.append((z * w[:, None]).sum(0))
        elif pooling == "max":
            out.append(z.max(0))
        else:
            j = int(np.argmax([float(m["score"]) for m in members]))
            out.append(z[j])
    return np.asarray(out, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("953",), default="953")
    ap.add_argument("--n-resamples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--weights", nargs=4, type=float,
                    default=[.10, .20, .05, .10],
                    metavar=("LARGE", "EXTRA", "LOGISTIC", "MULTISCALE"))
    ap.add_argument("--bias", nargs=4, type=float, default=[0., 0., 0., 0.],
                    metavar=("B1", "B2", "B3", "B4"))
    args = ap.parse_args()
    data = mh.collect(args.dataset, "val")
    larg = large.collect(args.dataset, "val")
    msc = ms.collect(args.dataset, "val")
    if data["keys"] != larg["keys"] or data["keys"] != msc["keys"]:
        raise RuntimeError("frozen group topology mismatch")
    extra = joblib.load(OUT / f"{args.dataset}_member_extra.joblib")
    logistic = joblib.load(OUT / f"{args.dataset}_member_logistic.joblib")
    large_hist = joblib.load(OUT / f"{args.dataset}_large_hist.joblib")
    ms_extra = joblib.load(OUT / f"{args.dataset}_ms_extra.joblib")
    qe = pool(np.asarray(extra.predict_proba(data["X"]), dtype=np.float32), data, "max")
    ql = pool(np.asarray(logistic.predict_proba(data["X"]), dtype=np.float32), data, "max")
    qlarge = pool(np.asarray(large_hist.predict_proba(larg["X"]), dtype=np.float32), larg, "mean")
    qms = pool(np.asarray(ms_extra.predict_proba(msc["X"]), dtype=np.float32), msc, "mean")
    detector = np.asarray([np.asarray(g["p"], dtype=np.float32)
                           for _rec, gs in data["groups"] for g in gs])
    detector = np.maximum(detector, 1e-8)
    detector /= detector.sum(axis=1, keepdims=True)
    wl, we, wx, wm = args.weights
    z = (np.log(detector) + wl * np.log(np.maximum(qlarge, 1e-8))
         + we * np.log(np.maximum(qe, 1e-8))
         + wx * np.log(np.maximum(ql, 1e-8))
         + wm * np.log(np.maximum(qms, 1e-8))
         + np.asarray(args.bias, dtype=np.float32))
    candidate = np.argmax(z, axis=1).astype(int)
    baseline = np.argmax(detector, axis=1).astype(int)

    base_correct, cand_correct, matched = [], [], []
    base_cm, cand_cm = [], []
    offset = 0
    for rec, groups in data["groups"]:
        matches = harness.count.tree_matches(rec, groups)
        matched_pred = {i for i, _ in matches}
        matched_gt = {j for _, j in matches}
        bcorrect = ccorrect = 0
        bcm = np.zeros((K + 1, K + 1), dtype=np.int32)
        ccm = np.zeros((K + 1, K + 1), dtype=np.int32)
        for gi, gj in matches:
            gt = int(rec["bunches"][gj]["cls"])
            bp, cp = int(baseline[offset + gi]), int(candidate[offset + gi])
            if 0 <= gt < K:
                bcm[bp, gt] += 1
                ccm[cp, gt] += 1
                bcorrect += int(bp == gt)
                ccorrect += int(cp == gt)
        for gi, group in enumerate(groups):
            if gi not in matched_pred:
                p = int(baseline[offset + gi]); q = int(candidate[offset + gi])
                if 0 <= p < K: bcm[p, K] += 1
                if 0 <= q < K: ccm[q, K] += 1
        for gj, bunch in enumerate(rec["bunches"]):
            if gj not in matched_gt and 0 <= int(bunch["cls"]) < K:
                bcm[K, int(bunch["cls"])] += 1
                ccm[K, int(bunch["cls"])] += 1
        base_correct.append(bcorrect); cand_correct.append(ccorrect)
        matched.append(len(matches)); base_cm.append(bcm); cand_cm.append(ccm)
        offset += len(groups)

    base_correct = np.asarray(base_correct, dtype=np.float64)
    cand_correct = np.asarray(cand_correct, dtype=np.float64)
    matched = np.asarray(matched, dtype=np.float64)
    base_cm = np.asarray(base_cm, dtype=np.float64)
    cand_cm = np.asarray(cand_cm, dtype=np.float64)
    n = len(matched)
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.n_resamples, n), dtype=np.int64)
    denom = np.maximum(matched[idx].sum(axis=1), 1.)
    bacc = base_correct[idx].sum(axis=1) / denom
    cacc = cand_correct[idx].sum(axis=1) / denom
    delta_acc = cacc - bacc

    def macro(cms):
        sums = cms[idx].sum(axis=1)
        f1s = []
        for c in range(K):
            tp = sums[:, c, c]
            fp = sums[:, c, :].sum(axis=1) - tp
            fn = sums[:, :, c].sum(axis=1) - tp
            f1s.append(2 * tp / np.maximum(2 * tp + fp + fn, 1.))
        return np.mean(np.asarray(f1s), axis=0)

    def macro_point(cms):
        sums = cms.sum(axis=0)
        f1s = []
        for c in range(K):
            tp = sums[c, c]
            fp = sums[c, :].sum() - tp
            fn = sums[:, c].sum() - tp
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1.))
        return float(np.mean(f1s))

    bmacro = macro(base_cm); cmacro = macro(cand_cm); delta_macro = cmacro - bmacro

    def ci(arr):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return {"ci95": [float(lo), float(hi)],
                "p_positive": float(np.mean(arr > 0)),
                "excludes_zero": bool(lo > 0 or hi < 0)}
    report = {
        "dataset": args.dataset,
        "protocol": "paired tree bootstrap on VAL; DINOv2-Large/Base experts fit TRAIN; no TEST",
        "seed": args.seed, "n_resamples": args.n_resamples, "n_val_trees": n,
        "weights": {"large_mean": wl, "base_extra_max": we,
                    "base_logistic_max": wx, "multiscale_extra_mean": wm},
        "bias": list(args.bias),
        "matched_accuracy": {"baseline": float(base_correct.sum() / max(matched.sum(), 1)),
                              "candidate": float(cand_correct.sum() / max(matched.sum(), 1)),
                              "delta": float(delta_acc.mean()), **ci(delta_acc)},
        "macro_f1": {"baseline": macro_point(base_cm), "candidate": macro_point(cand_cm),
                     "delta": float(delta_macro.mean()), **ci(delta_macro)},
        "class_correct_delta": (cand_correct.sum() - base_correct.sum()).item(),
        "interpretation": "VAL robustness only; does not replace locked test results",
    }
    if any(abs(float(x)) > 0. for x in args.bias) or list(args.weights) != [.10, .20, .05, .10]:
        out = OUT / f"{args.dataset}_large_stacker_bias_val_bootstrap.json"
    else:
        out = OUT / f"{args.dataset}_large_stacker_val_bootstrap.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
