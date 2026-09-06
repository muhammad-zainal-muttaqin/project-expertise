import json, glob, os
import numpy as np
from collections import Counter, defaultdict

R953 = "/workspace/SawitMVC-YOLO/json"
R763 = "/workspace/SawitMVC-Depth-YOLO"
NAMES = ["B1(ripe)", "B2(trans)", "B3(unripe)", "B4(v.unripe)"]


def read_tree(p):
    d = json.load(open(p))
    bun = d.get("bunches", []) or []
    app = [b.get("appearance_count", len(b.get("appearances", []))) for b in bun]
    cls = [b.get("class") for b in bun]
    nimg = len(d.get("images", {}))
    nbox = sum(v.get("bbox_count", 0) for v in d.get("images", {}).values())
    return dict(n_bunch=len(bun), app=app, cls=cls, nimg=nimg, nbox=nbox, tree=d.get("tree_id"))


t953 = {}
for p in glob.glob(f"{R953}/*.json"):
    r = read_tree(p)
    t953[r["tree"]] = r
t763 = {}
for s in ["train", "valid", "test"]:
    for p in glob.glob(f"{R763}/{s}/linked/*.json"):
        r = read_tree(p)
        t763[r["tree"]] = r

shared = sorted(set(t953) & set(t763))
july = [t for t in t763 if t.startswith("DAMIMAS")]
aug = [t for t in t763 if not t.startswith("DAMIMAS")]


def summarize(idx, keys, label):
    nb = np.array([idx[t]["n_bunch"] for t in keys])
    nbox = np.array([idx[t]["nbox"] for t in keys])
    app = [a for t in keys for a in idx[t]["app"]]
    cls = Counter(c for t in keys for c in idx[t]["cls"])
    ac = Counter(app)
    tot = sum(ac.values()) or 1
    print(f"\n=== {label}: {len(keys)} trees ===")
    print(f"  unique bunches/tree: mean={nb.mean():.2f} median={np.median(nb):.0f} max={nb.max()}  total={nb.sum()}")
    print(f"  raw boxes/tree     : mean={nbox.mean():.2f}   duplication factor k={nbox.sum()/max(nb.sum(),1):.3f}")
    print(f"  trees with 0 bunches: {(nb==0).sum()}")
    print("  appearances/bunch: " + "  ".join(f"{k}x:{100*ac[k]/tot:.1f}%" for k in sorted(ac)))
    n = sum(cls.values()) or 1
    print("  class mix: " + "  ".join(f"{k}={100*v/n:.1f}%" for k, v in sorted(cls.items())))
    return nb


a = summarize(t953, shared, "MAY 953 (the 352 shared trees)")
b = summarize(t763, july, "JULY Depth (same 352 trees)")
summarize(t763, aug, "AUGUST Depth (411 new trees)")
summarize(t953, sorted(t953), "MAY 953 (all 953 trees)")

print("\n=== paired per-tree bunch count, SAME 352 trees ===")
d = a - b
print(f"  May mean={a.mean():.2f}  July mean={b.mean():.2f}  ratio={a.mean()/b.mean():.2f}x")
print(f"  per-tree diff: median={np.median(d):.1f}  IQR=[{np.percentile(d,25):.0f};{np.percentile(d,75):.0f}]  "
      f"trees where July>=May: {(d<=0).sum()}/{len(d)}")

# what an agronomist would expect: standing bunches per palm should be roughly stable
# check whether the loss is concentrated in unripe classes
cm = Counter(c for t in shared for c in t953[t]["cls"])
cj = Counter(c for t in july for c in t763[t]["cls"])
print("\n  unique-bunch class counts (same trees):")
for k in ["B1", "B2", "B3", "B4"]:
    print(f"    {k}: May={cm[k]:5d}  July={cj[k]:5d}  ratio={cm[k]/max(cj[k],1):.2f}x")
