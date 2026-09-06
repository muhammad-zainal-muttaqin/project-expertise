"""E1 (protocol gap) + E2 (taxonomy) — train real detectors and cross-evaluate."""
import os, json, glob, sys
os.environ["YOLO_VERBOSE"] = "false"
from ultralytics import YOLO

OUT = "/workspace/ds"
RUNS = "/workspace/runs_audit"
RES = "/workspace/results_audit"
os.makedirs(RES, exist_ok=True)
IMGSZ, EPOCHS, BATCH, WORKERS = 960, 30, 16, 32
BASE = sys.argv[1] if len(sys.argv) > 1 else "yolo11s.pt"


def eval_yaml(root, folder, tax):
    """A yaml whose `val` points at an arbitrary image folder."""
    names = {4: ["B1", "B2", "B3", "B4"], 2: ["siap_panen", "belum"], 1: ["tandan"]}[tax]
    p = f"{root}/eval_{folder}.yaml"
    body = (f"path: {root}\ntrain: images/{folder}\nval: images/{folder}\n"
            f"nc: {len(names)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    open(p, "w").write(body)
    return p


def train(tag, data_root, tax):
    w = f"{RUNS}/{tag}/weights/best.pt"
    if os.path.exists(w):
        print(f"[skip] {tag} already trained", flush=True)
        return w
    m = YOLO(BASE)
    m.train(data=f"{data_root}/data.yaml", imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH,
            workers=WORKERS, cache="ram", device=0, seed=42, deterministic=True,
            project=RUNS, name=tag, exist_ok=True, patience=8, val=True, plots=False,
            verbose=False)
    return w


def evaluate(weights, root, folder, tax, tag):
    m = YOLO(weights)
    r = m.val(data=eval_yaml(root, folder, tax), imgsz=IMGSZ, batch=BATCH, workers=WORKERS,
              device=0, verbose=False, plots=False, split="val",
              project=RUNS, name=f"val_{tag}_{folder}", exist_ok=True)
    b = r.box
    out = dict(map50=float(b.map50), map=float(b.map), precision=float(b.mp), recall=float(b.mr),
               per_class_map50={n: float(v) for n, v in zip(
                   [r.names[i] for i in b.ap_class_index], b.ap50)})
    print(f"    {tag} -> {folder}: mAP50={out['map50']:.4f} P={out['precision']:.4f} "
          f"R={out['recall']:.4f} {out['per_class_map50']}", flush=True)
    return out


results = {}
JOBS = [
    # tag,        data root,    taxonomy, eval targets (root, folder)
    ("may4", f"{OUT}/may4", 4, [(f"{OUT}/may4", "test")]),
    ("may2", f"{OUT}/may2", 2, [(f"{OUT}/may2", "test")]),
    ("may1", f"{OUT}/may1", 1, [(f"{OUT}/may1", "test"), (f"{OUT}/dep1", "jul_all"),
                                (f"{OUT}/dep1", "aug_all")]),
    ("dep1", f"{OUT}/dep1", 1, [(f"{OUT}/dep1", "test"), (f"{OUT}/dep1", "jul_test"),
                                (f"{OUT}/dep1", "aug_test"), (f"{OUT}/may1", "test")]),
    ("dep4", f"{OUT}/dep4", 4, [(f"{OUT}/dep4", "test"), (f"{OUT}/may4", "test")]),
]

for tag, root, tax, targets in JOBS:
    print(f"\n=== TRAIN {tag} ({tax}-class) ===", flush=True)
    w = train(tag, root, tax)
    results[tag] = {}
    for er, folder in targets:
        results[tag][f"{os.path.basename(er)}/{folder}"] = evaluate(w, er, folder, tax, tag)
    json.dump(results, open(f"{RES}/detector_matrix.json", "w"), indent=1)

print("\nDONE")
print(json.dumps(results, indent=1))
