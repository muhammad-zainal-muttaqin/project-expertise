"""Build bunch crops from SawitMVC-YOLO 953 (tree-level split) and cache to disk."""
import json, glob, os, sys
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

ROOT = "/workspace/SawitMVC-YOLO"
OUT = "/workspace/crops953"
CID = {"B1": 0, "B2": 1, "B3": 2, "B4": 3}
CTX = 1.6      # context ring, matches PROPOSAL-Pipeline.md
SZ = 160

split = {}
for i, line in enumerate(open(f"{ROOT}/split_manifest.csv", encoding="utf-8-sig")):
    if i:
        f = line.strip().split(",")
        split[f[0]] = f[-1]

img_dir = {}
for s in ["train", "val", "test"]:
    for p in glob.glob(f"{ROOT}/images/{s}/*.jpg"):
        img_dir[os.path.basename(p)] = p


def do_tree(path):
    d = json.load(open(path))
    tree = d["tree_id"]
    sp = split.get(tree, "train")
    imgs = d.get("images", {})
    ann = {}
    for side, v in imgs.items():
        for a in v.get("annotations", []):
            ann[(side, a["box_index"])] = (v["filename"], a["bbox_yolo"])
    recs = []
    cache = {}
    for b in d.get("bunches", []) or []:
        c = CID.get(b.get("class"), -1)
        if c < 0:
            continue
        for ap in b.get("appearances", []):
            k = (ap.get("side"), ap.get("box_index"))
            if k not in ann:
                continue
            fn, bb = ann[k]
            p = img_dir.get(fn)
            if p is None:
                continue
            if p not in cache:
                cache[p] = Image.open(p).convert("RGB")
            im = cache[p]
            W, H = im.size
            cx, cy, w, h = bb
            side_px = max(w * W, h * H) * CTX
            x0, y0 = cx * W - side_px / 2, cy * H - side_px / 2
            crop = im.crop((int(x0), int(y0), int(x0 + side_px), int(y0 + side_px))).resize((SZ, SZ))
            name = f"{tree}__{k[0]}__{k[1]}.jpg"
            crop.save(f"{OUT}/{sp}/{name}", quality=90)
            recs.append(dict(f=name, split=sp, tree=tree, cls=c, bunch=b.get("bunch_id"),
                             cx=cx, cy=cy, w=w, h=h, napp=len(b.get("appearances", [])),
                             side=k[0], W=W, H=H))
    return recs


if __name__ == "__main__":
    for s in ["train", "val", "test"]:
        os.makedirs(f"{OUT}/{s}", exist_ok=True)
    paths = sorted(glob.glob(f"{ROOT}/json/*.json"))
    allr = []
    with ProcessPoolExecutor(max_workers=32) as ex:
        for i, r in enumerate(ex.map(do_tree, paths, chunksize=4)):
            allr += r
            if i % 200 == 0:
                print(f"  {i}/{len(paths)} trees, {len(allr)} crops", flush=True)
    json.dump(allr, open(f"{OUT}/index.json", "w"))
    from collections import Counter
    print("crops:", len(allr), Counter(r["split"] for r in allr))
