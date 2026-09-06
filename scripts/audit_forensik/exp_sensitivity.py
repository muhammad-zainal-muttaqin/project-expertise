"""How good would the maturity classifier have to be for 4-class mAP50 = 0.85,
assuming localization is already PERFECT?  Errors are drawn to ordinal neighbours,
matching the confusion structure this project measures everywhere."""
import json
import numpy as np
from collections import defaultdict

OUT = "/workspace/crops953"
te = [r for r in json.load(open(f"{OUT}/index.json")) if r["split"] == "test"]
y = np.array([r["cls"] for r in te])
rng = np.random.default_rng(0)


def ap50(gt, preds, cls):
    npos = sum(sum(1 for c in v.values() if c == cls) for v in gt.values())
    if npos == 0:
        return float("nan")
    p = sorted([x for x in preds if x[2] == cls], key=lambda z: -z[3])
    used = defaultdict(set); tp = np.zeros(len(p)); fp = np.zeros(len(p))
    for i, (img, bi, c, s) in enumerate(p):
        if gt[img][bi] == cls and bi not in used[img]:
            tp[i] = 1; used[img].add(bi)
        else:
            fp[i] = 1
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec, prec = tp / npos, tp / np.maximum(tp + fp, 1e-9)
    return sum((prec[rec >= t].max() if (rec >= t).any() else 0.) for t in np.linspace(0, 1, 101)) / 101


gt = defaultdict(dict)
for i, r in enumerate(te):
    gt[(r["tree"], r["side"])][i] = r["cls"]

print("  target acc | simulated 4-class mAP50 (perfect boxes)")
for acc in [0.661, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]:
    m = []
    for rep in range(3):
        P = np.zeros((len(y), 4))
        for i, c in enumerate(y):
            nb = [x for x in (c - 1, c + 1) if 0 <= x < 4]
            # the model's DECISION is wrong (1-acc) of the time, to an ordinal neighbour
            chat = c if rng.random() < acc else nb[rng.integers(len(nb))]
            P[i, chat] = 0.6
            for x in [z for z in (chat - 1, chat + 1) if 0 <= z < 4]:
                P[i, x] = 0.35 / len([z for z in (chat - 1, chat + 1) if 0 <= z < 4])
        P += rng.normal(0, 0.05, P.shape); P = np.clip(P, 1e-6, None)
        P /= P.sum(1, keepdims=True)
        preds = [((r["tree"], r["side"]), i, c, float(P[i, c])) for i, r in enumerate(te) for c in range(4)]
        m.append(np.nanmean([ap50(gt, preds, c) for c in range(4)]))
    print(f"     {acc:.3f}   ->   mAP50 = {np.mean(m):.4f}")
print("\n  observed per-view class accuracy of every backbone tried in this project: 0.62-0.70")
