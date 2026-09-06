import json, glob
import numpy as np
from collections import Counter, defaultdict

R953 = "/workspace/SawitMVC-YOLO/json"
R763 = "/workspace/SawitMVC-Depth-YOLO"
CID = {"B1": 0, "B2": 1, "B3": 2, "B4": 3}


def per_view_classes(path):
    """For each unique bunch, return the list of class_ids annotated in each view it appears in."""
    d = json.load(open(path))
    imgs = d.get("images", {})
    ann = {}
    for side, v in imgs.items():
        for a in v.get("annotations", []):
            ann[(side, a["box_index"])] = a.get("class_id", CID.get(a.get("class_name"), -1))
    out = []
    for b in d.get("bunches", []) or []:
        cs = []
        for ap in b.get("appearances", []):
            k = (ap.get("side"), ap.get("box_index"))
            if k in ann:
                cs.append(ann[k])
        if cs:
            out.append((b.get("class"), cs))
    return out


def analyze(paths, label):
    multi = 0
    inconsistent = 0
    pairs = Counter()
    spread = Counter()
    declared_mismatch = 0
    for p in paths:
        for declared, cs in per_view_classes(p):
            if len(cs) < 2:
                continue
            multi += 1
            u = sorted(set(cs))
            spread[max(u) - min(u)] += 1
            if len(u) > 1:
                inconsistent += 1
                for i in range(len(cs)):
                    for j in range(i + 1, len(cs)):
                        if cs[i] != cs[j]:
                            pairs[tuple(sorted((cs[i], cs[j])))] += 1
            dc = CID.get(declared, -1)
            if dc >= 0 and dc not in u:
                declared_mismatch += 1
    print(f"\n=== {label} ===")
    print(f"  multi-view bunches: {multi}")
    if multi:
        print(f"  bunches whose OWN views disagree on class: {inconsistent} ({100*inconsistent/multi:.2f}%)")
        print("  disagreement spread: " + "  ".join(f"|d|={k}:{v}" for k, v in sorted(spread.items()) if k))
        print("  most common disagreeing pairs: " + "  ".join(
            f"B{a+1}/B{b+1}:{n}" for (a, b), n in pairs.most_common(6)))
        print(f"  declared bunch class not among its view labels: {declared_mismatch}")
    return multi, inconsistent


p953 = sorted(glob.glob(f"{R953}/*.json"))
p763 = [p for s in ["train", "valid", "test"] for p in glob.glob(f"{R763}/{s}/linked/*.json")]
pjul = [p for p in p763 if "DAMIMAS" in p]
paug = [p for p in p763 if "DAMIMAS" not in p]

analyze(p953, "MAY 953 (all trees)")
analyze(pjul, "JULY Depth (352 trees)")
analyze(paug, "AUGUST Depth (411 trees)")

# --- unknown class codes ---
print("\n=== class codes present in bunch records ===")
for label, ps in [("953", p953), ("July", pjul), ("Aug", paug)]:
    c = Counter()
    for p in ps:
        d = json.load(open(p))
        for b in d.get("bunches", []) or []:
            c[b.get("class")] += 1
    print(f"  {label}: {dict(c)}")

# --- geometric consistency: does the same bunch keep its size across views? ---
print("\n=== same-bunch box-size variability across views (log2 ratio max/min) ===")
for label, ps in [("953", p953[:400]), ("July", pjul), ("Aug", paug)]:
    ratios = []
    for p in ps:
        d = json.load(open(p))
        imgs = d.get("images", {})
        area = {}
        for side, v in imgs.items():
            for a in v.get("annotations", []):
                bb = a["bbox_yolo"]
                area[(side, a["box_index"])] = bb[2] * bb[3]
        for b in d.get("bunches", []) or []:
            aa = [area[(ap["side"], ap["box_index"])] for ap in b.get("appearances", [])
                  if (ap["side"], ap["box_index"]) in area]
            if len(aa) >= 2 and min(aa) > 0:
                ratios.append(np.log2(max(aa) / min(aa)) / 2)  # linear-scale log2 ratio
    r = np.array(ratios)
    print(f"  {label}: n={len(r)} median={np.median(r):.2f} p90={np.percentile(r,90):.2f} "
          f"(fraction >1 octave in linear size: {100*(r>1).mean():.1f}%)")
