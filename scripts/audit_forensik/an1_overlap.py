import os, re, json, glob
from collections import defaultdict, Counter

ROOT953 = "/workspace/SawitMVC-YOLO"
ROOT763 = "/workspace/SawitMVC-Depth-YOLO"

def load_labels(path):
    out = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                out.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return out

# --- index 953 ---
t953 = defaultdict(list)   # tree -> list of (side, boxes)
for split in ["train", "val", "test"]:
    for f in glob.glob(f"{ROOT953}/labels/{split}/*.txt"):
        base = os.path.basename(f)[:-4]
        m = re.match(r"(.+)_(\d+)$", base)
        tree, side = m.group(1), int(m.group(2))
        t953[tree].append((side, split, load_labels(f)))

# --- index 763 ---
t763 = defaultdict(list)
for split in ["train", "valid", "test"]:
    for f in glob.glob(f"{ROOT763}/{split}/labels/*.txt"):
        base = os.path.basename(f)[:-4]
        m = re.match(r"(.+)_(\d+)$", base)
        tree, side = m.group(1), int(m.group(2))
        t763[tree].append((side, split, load_labels(f)))

print(f"953 trees: {len(t953)}   763 trees: {len(t763)}")

# campaign split within 763
july = [t for t in t763 if t.startswith("DAMIMAS")]
aug  = [t for t in t763 if not t.startswith("DAMIMAS")]
print(f"763 breakdown: DAMIMAS(July)={len(july)}  other(Aug)={len(aug)}")
print("  Aug prefixes:", Counter(t.split('_')[0] for t in aug))

shared = sorted(set(t953) & set(t763))
print(f"\nTrees present in BOTH 953(May) and 763(Jul): {len(shared)}")

# --- per-tree comparison on shared trees ---
def tree_stats(entries):
    n_side = len(entries)
    boxes = [b for _, _, bx in entries for b in bx]
    cls = Counter(b[0] for b in boxes)
    empty_sides = sum(1 for _, _, bx in entries if len(bx) == 0)
    return n_side, len(boxes), cls, empty_sides

tot_may = Counter(); tot_jul = Counter()
n_may = n_jul = 0
sides_may = sides_jul = 0
empty_may = empty_jul = 0
per_tree = []
for t in shared:
    sm, bm, cm, em = tree_stats(t953[t])
    sj, bj, cj, ej = tree_stats(t763[t])
    tot_may += cm; tot_jul += cj
    n_may += bm; n_jul += bj
    sides_may += sm; sides_jul += sj
    empty_may += em; empty_jul += ej
    per_tree.append((t, bm, bj))

print(f"\n=== SAME {len(shared)} TREES: May(953) vs July(Depth) ===")
print(f"sides: May={sides_may}  Jul={sides_jul}")
print(f"boxes: May={n_may} ({n_may/sides_may:.2f}/img)   Jul={n_jul} ({n_jul/sides_jul:.2f}/img)   ratio={n_may/max(n_jul,1):.2f}x")
print(f"empty sides: May={empty_may} ({100*empty_may/sides_may:.1f}%)   Jul={empty_jul} ({100*empty_jul/sides_jul:.1f}%)")
names = ["B1(ripe)", "B2(trans)", "B3(unripe)", "B4(v.unripe)"]
print(f"{'class':<14}{'May':>8}{'May%':>8}{'Jul':>8}{'Jul%':>8}{'drop':>9}")
for c in range(4):
    m, j = tot_may[c], tot_jul[c]
    print(f"{names[c]:<14}{m:>8}{100*m/n_may:>7.1f}%{j:>8}{100*j/max(n_jul,1):>7.1f}%{(m/j if j else float('inf')):>8.1f}x")

# distribution of per-tree deltas
drops = [(bm - bj) for _, bm, bj in per_tree]
import numpy as np
d = np.array(drops)
print(f"\nper-tree box delta (May-Jul): mean={d.mean():.2f} median={np.median(d):.1f} min={d.min()} max={d.max()}")
print(f"trees where Jul >= May: {(d<=0).sum()} / {len(d)}")

# --- class distribution per corpus/campaign ---
def corpus_stats(idx, keys, label):
    boxes = [b for t in keys for _, _, bx in idx[t] for b in bx]
    sides = sum(len(idx[t]) for t in keys)
    cls = Counter(b[0] for b in boxes)
    n = len(boxes)
    print(f"\n{label}: {len(keys)} trees, {sides} images, {n} boxes ({n/sides:.2f}/img)")
    print("   " + "  ".join(f"{names[c]}={100*cls[c]/n:.1f}%" for c in range(4)))
    return boxes

b953 = corpus_stats(t953, sorted(t953), "953 / SawitMVC-YOLO (May)")
bjul = corpus_stats(t763, july, "763 subset: July DAMIMAS")
baug = corpus_stats(t763, aug,  "763 subset: August MARIHAT/TOPAZ")

# --- box geometry (normalized + absolute px) ---
def geom(boxes, W, H, label):
    a = np.array([[b[3], b[4]] for b in boxes])
    area_n = a[:, 0] * a[:, 1]
    wpx, hpx = a[:, 0] * W, a[:, 1] * H
    diag = np.sqrt(wpx * hpx)
    ar = a[:, 0] / np.maximum(a[:, 1], 1e-9)
    print(f"\n{label} (img {W}x{H}), n={len(boxes)}")
    print(f"  norm area   : p5={np.percentile(area_n,5):.5f} p50={np.median(area_n):.5f} p95={np.percentile(area_n,95):.5f}")
    print(f"  sqrt(px)    : p5={np.percentile(diag,5):.1f} p50={np.median(diag):.1f} p95={np.percentile(diag,95):.1f}")
    print(f"  aspect w/h  : p5={np.percentile(ar,5):.2f} p50={np.median(ar):.2f} p95={np.percentile(ar,95):.2f}")
    print(f"  tiny (<16px): {100*(diag<16).mean():.2f}%   small(<32px): {100*(diag<32).mean():.2f}%")
    return diag

d953 = geom(b953, 960, 1280, "953 (portrait phone)")
djul = geom(bjul, 1280, 800, "763-July (Orbbec landscape)")
daug = geom(baug, 1280, 800, "763-August (Orbbec landscape)")

# at letterbox 1280 the 953 images are scaled by 1280/1280=1.0 (H) -> w 960
# depth imgs 1280x800 scaled by 1280/1280=1 -> same
print("\n=== per-class box size (sqrt px) ===")
for label, bx, W, H in [("953", b953, 960, 1280), ("Jul", bjul, 1280, 800), ("Aug", baug, 1280, 800)]:
    row = []
    for c in range(4):
        s = [np.sqrt(b[3]*W*b[4]*H) for b in bx if b[0] == c]
        row.append(f"{names[c]}={np.median(s):.0f}" if s else f"{names[c]}=NA")
    print(f"  {label}: " + "  ".join(row))
