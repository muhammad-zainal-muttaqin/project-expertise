"""Empirical ceiling: what mAP50 is reachable if localization were PERFECT
(every predicted box == a ground-truth box) and only the maturity class is predicted?"""
import json
import numpy as np
from collections import defaultdict

OUT = "/workspace/crops953"
idx = json.load(open(f"{OUT}/index.json"))
te = [r for r in idx if r["split"] == "test"]
P = np.load(f"{OUT}/P_test.npy")
NAMES = ["B1", "B2", "B3", "B4"]


def ap50(gt_by_img, preds, cls):
    """COCO-style AP50, 101-point, perfect IoU (predictions are the GT boxes themselves)."""
    npos = sum(sum(1 for c in v.values() if c == cls) for v in gt_by_img.values())
    if npos == 0:
        return float("nan")
    p = sorted([x for x in preds if x[2] == cls], key=lambda z: -z[3])
    used = defaultdict(set)
    tp = np.zeros(len(p)); fp = np.zeros(len(p))
    for i, (img, bi, c, s) in enumerate(p):
        if gt_by_img[img][bi] == cls and bi not in used[img]:
            tp[i] = 1; used[img].add(bi)
        else:
            fp[i] = 1
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec, prec = tp / npos, tp / np.maximum(tp + fp, 1e-9)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        m = prec[rec >= t]
        ap += (m.max() if m.size else 0.0) / 101
    return ap


# ground truth per image (crop index -> its own box, one per record)
gt = defaultdict(dict)
for i, r in enumerate(te):
    gt[(r["tree"], r["side"])][i] = r["cls"]

print("=== CEILING: perfect boxes + learned maturity classifier (953 test) ===")
for mode in ["argmax-only", "all-classes"]:
    preds = []
    for i, r in enumerate(te):
        k = (r["tree"], r["side"])
        if mode == "argmax-only":
            c = int(P[i].argmax()); preds.append((k, i, c, float(P[i][c])))
        else:
            for c in range(4):
                preds.append((k, i, c, float(P[i][c])))
    aps = [ap50(gt, preds, c) for c in range(4)]
    print(f"  {mode:<12} " + "  ".join(f"{NAMES[c]}={aps[c]:.4f}" for c in range(4)) +
          f"   ->  mAP50 = {np.nanmean(aps):.4f}")

# perfect-class oracle for calibration of the metric itself
preds = [((r["tree"], r["side"]), i, r["cls"], 1.0) for i, r in enumerate(te)]
aps = [ap50(gt, preds, c) for c in range(4)]
print(f"  {'oracle-class':<12} " + "  ".join(f"{NAMES[c]}={aps[c]:.4f}" for c in range(4)) +
      f"   ->  mAP50 = {np.nanmean(aps):.4f}   (sanity: must be 1.0)")

# what class accuracy would be needed for mAP50 = 0.85?
acc = (P.argmax(1) == np.array([r["cls"] for r in te])).mean()
print(f"\n  per-view class accuracy of this model: {acc:.4f}")
print(f"  project's best reported class-aware mAP50 on 953 test: 0.5970 (with real detectors)")

# 2-class and 3-class collapses of the same predictions
for name, mapping in [("B1 vs rest (harvest decision)", {0: 0, 1: 1, 2: 1, 3: 1}),
                      ("B1+B2 vs B3+B4", {0: 0, 1: 0, 2: 1, 3: 1}),
                      ("B1 / B2 / B3+B4", {0: 0, 1: 1, 2: 2, 3: 2})]:
    k = max(mapping.values()) + 1
    gt2 = defaultdict(dict)
    for i, r in enumerate(te):
        gt2[(r["tree"], r["side"])][i] = mapping[r["cls"]]
    Pm = np.zeros((len(te), k))
    for c in range(4):
        Pm[:, mapping[c]] += P[:, c]
    preds = [((r["tree"], r["side"]), i, c, float(Pm[i][c])) for i, r in enumerate(te) for c in range(k)]
    aps = [ap50(gt2, preds, c) for c in range(k)]
    print(f"  [{name:<30}] mAP50 = {np.nanmean(aps):.4f}   per-class " +
          " ".join(f"{a:.3f}" for a in aps))
