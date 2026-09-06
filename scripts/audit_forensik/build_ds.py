"""Build YOLO dataset variants (symlinked images, rewritten labels) for the audit experiments.

Taxonomies:  4  = B1..B4 as-is
             2  = 0:B1 (harvest-ready)  1:rest
             1  = 0:bunch (class-agnostic)
Corpora:     may = SawitMVC-YOLO 953 (May, phone)
             dep = SawitMVC-Depth-YOLO 763 (July DAMIMAS + August MARIHAT/TOPAZ)
"""
import os, glob, shutil, json

MAY = "/workspace/SawitMVC-YOLO"
DEP = "/workspace/SawitMVC-Depth-YOLO"
OUT = "/workspace/ds"
MAP = {4: {0: 0, 1: 1, 2: 2, 3: 3}, 2: {0: 0, 1: 1, 2: 1, 3: 1}, 1: {0: 0, 1: 0, 2: 0, 3: 0}}
NAMES = {4: ["B1", "B2", "B3", "B4"], 2: ["siap_panen", "belum"], 1: ["tandan"]}


def write(dst_img, dst_lbl, img, lbl, tax):
    os.makedirs(dst_img, exist_ok=True); os.makedirs(dst_lbl, exist_ok=True)
    b = os.path.basename(img)
    link = f"{dst_img}/{b}"
    if not os.path.islink(link) and not os.path.exists(link):
        os.symlink(img, link)
    rows = []
    if os.path.exists(lbl):
        for line in open(lbl):
            p = line.split()
            if len(p) >= 5:
                rows.append(f"{MAP[tax][int(p[0])]} {' '.join(p[1:5])}")
    open(f"{dst_lbl}/{b[:-4]}.txt", "w").write("\n".join(rows) + ("\n" if rows else ""))


def yaml(path, splits, tax):
    n = NAMES[tax]
    body = f"path: {path}\n"
    for k, v in splits.items():
        body += f"{k}: {v}\n"
    body += f"nc: {len(n)}\nnames:\n" + "".join(f"  {i}: {x}\n" for i, x in enumerate(n))
    open(f"{path}/data.yaml", "w").write(body)


def build_may(tax):
    root = f"{OUT}/may{tax}"
    for sp in ["train", "val", "test"]:
        for img in sorted(glob.glob(f"{MAY}/images/{sp}/*.jpg")):
            lbl = f"{MAY}/labels/{sp}/{os.path.basename(img)[:-4]}.txt"
            write(f"{root}/images/{sp}", f"{root}/labels/{sp}", img, lbl, tax)
    yaml(root, {"train": "images/train", "val": "images/val", "test": "images/test"}, tax)
    return root


def build_dep(tax):
    """763 with its own split, plus per-campaign eval folders."""
    root = f"{OUT}/dep{tax}"
    for sp, alias in [("train", "train"), ("valid", "val"), ("test", "test")]:
        for img in sorted(glob.glob(f"{DEP}/{sp}/images/*.jpg")):
            lbl = f"{DEP}/{sp}/labels/{os.path.basename(img)[:-4]}.txt"
            write(f"{root}/images/{alias}", f"{root}/labels/{alias}", img, lbl, tax)
            camp = "jul" if os.path.basename(img).startswith("DAMIMAS") else "aug"
            # every image of a campaign (for evaluating a model that never saw 763)
            write(f"{root}/images/{camp}_all", f"{root}/labels/{camp}_all", img, lbl, tax)
            if sp == "test":   # clean per-campaign test for models trained on 763
                write(f"{root}/images/{camp}_test", f"{root}/labels/{camp}_test", img, lbl, tax)
    yaml(root, {"train": "images/train", "val": "images/val", "test": "images/test",
                "jul_all": "images/jul_all", "aug_all": "images/aug_all",
                "jul_test": "images/jul_test", "aug_test": "images/aug_test"}, tax)
    return root


if __name__ == "__main__":
    shutil.rmtree(OUT, ignore_errors=True)
    made = {}
    for tax in [4, 2, 1]:
        made[f"may{tax}"] = build_may(tax)
        made[f"dep{tax}"] = build_dep(tax)
    for k, v in sorted(made.items()):
        c = {sp: len(glob.glob(f"{v}/images/{sp}/*.jpg")) for sp in
             os.listdir(f"{v}/images")}
        print(f"{k:6} {v}  " + "  ".join(f"{a}={b}" for a, b in sorted(c.items())))
