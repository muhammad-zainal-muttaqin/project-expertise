"""Decompose the May-model's false positives on July into:
   (a) NESTED   - centre falls inside a July GT box, model box much smaller  -> boxing-convention mismatch
   (b) ORPHAN   - nowhere near any July GT box                               -> object July did not label
   (c) SHIFTED  - partial overlap 0.2-0.5                                    -> localisation slop
Also measures the box-size convention gap directly on matched pairs."""
import os, glob, json
import numpy as np
os.environ["YOLO_VERBOSE"] = "false"
from ultralytics import YOLO

DS = "/workspace/ds"
m = YOLO("/workspace/runs_audit/may1/weights/best.pt")


def gt_of(lbl, W, H):
    b = []
    if os.path.exists(lbl):
        for line in open(lbl):
            p = line.split()
            if len(p) >= 5:
                cx, cy, w, h = map(float, p[1:5])
                b.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    return np.array(b, float).reshape(-1, 4)


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    it = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return it / (ar(a)[:, None] + ar(b)[None, :] - it + 1e-9)


def run(idir, ldir, label, conf_min=0.50):
    imgs = sorted(glob.glob(f"{idir}/*.jpg"))
    kind = dict(nested=0, orphan=0, shifted=0, tp=0)
    ratios, ious = [], []
    for i in range(0, len(imgs), 32):
        ch = imgs[i:i + 32]
        for p, r in zip(ch, m.predict(ch, imgsz=960, conf=conf_min, iou=0.7, max_det=300,
                                      device=0, verbose=False)):
            H, W = r.orig_shape
            g = gt_of(f"{ldir}/{os.path.basename(p)[:-4]}.txt", W, H)
            d = r.boxes.xyxy.cpu().numpy()
            if len(d) == 0:
                continue
            M = iou_mat(d, g)
            best = M.max(1) if M.shape[1] else np.zeros(len(d))
            arg = M.argmax(1) if M.shape[1] else np.zeros(len(d), int)
            for k in range(len(d)):
                if best[k] >= 0.5:
                    kind["tp"] += 1
                    gb = g[arg[k]]
                    ad = (d[k, 2] - d[k, 0]) * (d[k, 3] - d[k, 1])
                    ag = (gb[2] - gb[0]) * (gb[3] - gb[1])
                    ratios.append(np.sqrt(ad / ag)); ious.append(best[k])
                    continue
                cx, cy = (d[k, 0] + d[k, 2]) / 2, (d[k, 1] + d[k, 3]) / 2
                inside = False
                if len(g):
                    inside = bool(((g[:, 0] <= cx) & (cx <= g[:, 2]) &
                                   (g[:, 1] <= cy) & (cy <= g[:, 3])).any())
                if inside:
                    kind["nested"] += 1
                elif best[k] >= 0.2:
                    kind["shifted"] += 1
                else:
                    kind["orphan"] += 1
    n_fp = kind["nested"] + kind["shifted"] + kind["orphan"]
    print(f"\n{label}  (conf>={conf_min})")
    print(f"  TP (IoU>=0,5)                         : {kind['tp']}")
    print(f"  FP total                              : {n_fp}")
    if n_fp:
        for k in ["nested", "shifted", "orphan"]:
            print(f"    {k:8}: {kind[k]:4d}  ({100*kind[k]/n_fp:5.1f}% dari FP)")
    if ratios:
        print(f"  pada pasangan yang COCOK: sisi kotak model / kotak acuan "
              f"= {np.median(ratios):.3f}  (IoU median {np.median(ious):.3f})")
    return dict(kind=kind, size_ratio_median=float(np.median(ratios)) if ratios else None,
                iou_median=float(np.median(ious)) if ious else None)


out = {}
out["mei_test"] = run(f"{DS}/may1/images/test", f"{DS}/may1/labels/test", "MEI 953 test (protokol sendiri)")
out["juli"] = run(f"{DS}/dep1/images/jul_all", f"{DS}/dep1/labels/jul_all", "JULI 352 (pohon sama persis)")
out["agustus"] = run(f"{DS}/dep1/images/aug_all", f"{DS}/dep1/labels/aug_all", "AGUSTUS 411")
json.dump(out, open("/workspace/results_audit/e1c_fp_kind.json", "w"), indent=1)

# what does the AP look like if the box-convention gap is removed? (IoU 0.3)
print("\n--- AP50 vs AP30: seberapa besar bagian yang murni konvensi kotak ---")
for tag, idir, ldir in [("Mei test", f"{DS}/may1/images/test", f"{DS}/may1/labels/test"),
                        ("Juli", f"{DS}/dep1/images/jul_all", f"{DS}/dep1/labels/jul_all")]:
    for thr in [0.5, 0.3]:
        imgs = sorted(glob.glob(f"{idir}/*.jpg"))
        tp = fp = npos = 0
        for i in range(0, len(imgs), 32):
            ch = imgs[i:i + 32]
            for p, r in zip(ch, m.predict(ch, imgsz=960, conf=0.25, iou=0.7, max_det=300,
                                          device=0, verbose=False)):
                H, W = r.orig_shape
                g = gt_of(f"{ldir}/{os.path.basename(p)[:-4]}.txt", W, H)
                npos += len(g)
                d = r.boxes.xyxy.cpu().numpy()
                M = iou_mat(d, g)
                used = set()
                for k in range(len(d)):
                    j = int(M[k].argmax()) if M.shape[1] else -1
                    if j >= 0 and M[k, j] >= thr and j not in used:
                        tp += 1; used.add(j)
                    else:
                        fp += 1
        print(f"  {tag:10} IoU>={thr}: presisi={tp/max(tp+fp,1):.4f}  recall={tp/max(npos,1):.4f}")
