import os, re, glob
import numpy as np
from collections import defaultdict, Counter

ROOT953 = "/workspace/SawitMVC-YOLO"
ROOT763 = "/workspace/SawitMVC-Depth-YOLO"
NAMES = ["B1(ripe)", "B2(trans)", "B3(unripe)", "B4(v.unripe)"]


def load(p):
    out = []
    for line in open(p):
        q = line.split()
        if len(q) >= 5:
            out.append((int(q[0]), *map(float, q[1:5])))
    return out


def collect(root, splits, per_split_layout):
    trees = defaultdict(list)
    for s in splits:
        pat = f"{root}/labels/{s}/*.txt" if not per_split_layout else f"{root}/{s}/labels/*.txt"
        for f in glob.glob(pat):
            b = os.path.basename(f)[:-4]
            m = re.match(r"(.+)_(\d+)$", b)
            trees[m.group(1)].append((int(m.group(2)), load(f)))
    return trees


t953 = collect(ROOT953, ["train", "val", "test"], False)
t763 = collect(ROOT763, ["train", "valid", "test"], True)
shared = sorted(set(t953) & set(t763))
july = [t for t in t763 if t.startswith("DAMIMAS")]
aug = [t for t in t763 if not t.startswith("DAMIMAS")]


def vstats(trees, keys, label, H):
    box = [b for t in keys for _, bx in trees[t] for b in bx]
    cy = np.array([b[2] for b in box])
    h = np.array([b[4] for b in box])
    cls = np.array([b[0] for b in box])
    top = cy - h / 2
    print(f"\n=== {label} (n={len(box)}) ===")
    print(f"  cy (0=top,1=bottom): p10={np.percentile(cy,10):.3f} p50={np.median(cy):.3f} p90={np.percentile(cy,90):.3f}")
    print(f"  boxes touching TOP edge (top<0.02): {100*(top<0.02).mean():.1f}%")
    print(f"  boxes in upper third (cy<0.33): {100*(cy<0.33).mean():.1f}%   lower third(cy>0.67): {100*(cy>0.67).mean():.1f}%")
    print("  median cy per class:  " + "  ".join(
        f"{NAMES[c]}={np.median(cy[cls==c]):.3f}(n={ (cls==c).sum() })" for c in range(4) if (cls == c).any()))
    return cy, cls


vstats(t953, shared, "MAY 953 - the 352 shared trees", 1280)
vstats(t763, july, "JULY Depth - same 352 trees", 800)
vstats(t763, aug, "AUGUST Depth - 411 new trees", 800)

# --- empty-side and per-side breakdown ---
print("\n=== empty sides & per-side counts (shared 352 trees) ===")
for label, trees, keys in [("MAY", t953, shared), ("JULY", t763, july)]:
    per_side = defaultdict(list)
    for t in keys:
        for side, bx in trees[t]:
            per_side[side].append(len(bx))
    row = []
    for s in sorted(per_side):
        v = np.array(per_side[s])
        row.append(f"side{s}: n={len(v)} mean={v.mean():.2f} empty={100*(v==0).mean():.0f}%")
    print(f"  {label}: " + " | ".join(row))

# --- how many trees have zero bunches at all ---
print("\n=== trees with ZERO annotated boxes across all sides ===")
for label, trees, keys in [("MAY 953(shared)", t953, shared), ("JULY", t763, july), ("AUG", t763, aug)]:
    z = sum(1 for t in keys if sum(len(bx) for _, bx in trees[t]) == 0)
    tot = sum(sum(len(bx) for _, bx in trees[t]) for t in keys)
    print(f"  {label}: {z}/{len(keys)} trees empty; total boxes={tot}; boxes/tree={tot/len(keys):.2f}")

# --- vertical class ordering test: is B1 lowest / B4 highest? ---
print("\n=== vertical ordering of classes (median cy, 0=top) ===")
for label, trees, keys in [("MAY 953", t953, sorted(t953)), ("JULY", t763, july), ("AUG", t763, aug)]:
    box = [b for t in keys for _, bx in trees[t] for b in bx]
    cy = np.array([b[2] for b in box]); cls = np.array([b[0] for b in box])
    med = [np.median(cy[cls == c]) if (cls == c).any() else np.nan for c in range(4)]
    order = "B1<B2<B3<B4 (ripe lower)" if all(med[i] > med[i+1] for i in range(3)) else "not monotone"
    print(f"  {label}: " + "  ".join(f"{NAMES[c]}={med[c]:.3f}" for c in range(4)) + f"   -> {order}")
