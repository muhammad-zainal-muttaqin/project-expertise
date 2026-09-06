"""Are the May-model's 'false positives' on July real bunches or genuine errors?
Genuine errors are low-confidence.  Unlabelled true objects stay confident."""
import os, glob, json, re
import numpy as np
from collections import defaultdict
os.environ["YOLO_VERBOSE"] = "false"
from ultralytics import YOLO
from PIL import Image, ImageDraw

DS = "/workspace/ds"
W = "/workspace/runs_audit/may1/weights/best.pt"
OUTDIR = "/tmp/claude-1001/-workspace/ebcbd941-6775-4113-b727-404085458263/scratchpad"


def load_gt(lbl, W_, H_):
    b = []
    if os.path.exists(lbl):
        for line in open(lbl):
            p = line.split()
            if len(p) >= 5:
                cx, cy, w, h = map(float, p[1:5])
                b.append([(cx - w / 2) * W_, (cy - h / 2) * H_, (cx + w / 2) * W_, (cy + h / 2) * H_])
    return np.array(b, float).reshape(-1, 4)


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    it = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return it / (ar(a)[:, None] + ar(b)[None, :] - it + 1e-9)


m = YOLO(W)
sets = {"Juli 352": (f"{DS}/dep1/images/jul_all", f"{DS}/dep1/labels/jul_all")}
THR = [0.25, 0.40, 0.50, 0.60, 0.70, 0.80]
report, fp_gallery = {}, []

for name, (idir, ldir) in sets.items():
    imgs = sorted(glob.glob(f"{idir}/*.jpg"))
    tp = {t: 0 for t in THR}; fp = {t: 0 for t in THR}
    for i in range(0, len(imgs), 32):
        chunk = imgs[i:i + 32]
        for p, r in zip(chunk, m.predict(chunk, imgsz=960, conf=0.20, iou=0.7,
                                         max_det=300, device=0, verbose=False)):
            H_, W_ = r.orig_shape
            g = load_gt(f"{ldir}/{os.path.basename(p)[:-4]}.txt", W_, H_)
            d = r.boxes.xyxy.cpu().numpy(); c = r.boxes.conf.cpu().numpy()
            M = iou_mat(d, g)
            best = M.max(1) if M.shape[1] else np.zeros(len(d))
            for t in THR:
                k = c >= t
                tp[t] += int((best[k] >= 0.5).sum()); fp[t] += int((best[k] < 0.5).sum())
            if "Juli" in name and len(fp_gallery) < 6:
                sel = np.where((c >= 0.60) & (best < 0.3))[0]
                if len(sel) >= 1:
                    fp_gallery.append((p, d[sel], c[sel], g))
    report[name] = {f"{t:.2f}": dict(precision=round(tp[t] / max(tp[t] + fp[t], 1), 4),
                                     tp=tp[t], fp=fp[t]) for t in THR}
    print(f"\n{name}: presisi menurut ambang keyakinan")
    for t in THR:
        r_ = report[name][f"{t:.2f}"]
        print(f"   conf>={t:.2f}  presisi={r_['precision']:.4f}  TP={r_['tp']:5d}  FP={r_['fp']:5d}")

json.dump(report, open("/workspace/results_audit/e1b_precision_vs_conf.json", "w"), indent=1)

# render a contact sheet of confident "false positives" on July
if fp_gallery:
    tiles = []
    for p, boxes, confs, g in fp_gallery[:4]:
        im = Image.open(p).convert("RGB"); dr = ImageDraw.Draw(im)
        for b in g:
            dr.rectangle(list(b), outline=(90, 210, 120), width=5)
        for b, cf in zip(boxes, confs):
            dr.rectangle(list(b), outline=(235, 90, 40), width=6)
            dr.text((b[0] + 4, max(0, b[1] - 16)), f"{cf:.2f}", fill=(235, 90, 40))
        dr.rectangle([0, 0, im.width, 40], fill=(0, 0, 0))
        dr.text((8, 12), f"{os.path.basename(p)}  hijau=label Juli  jingga=deteksi model Mei conf>=0.60",
                fill=(255, 255, 255))
        tiles.append(im.resize((760, int(760 * im.height / im.width))))
    Wt = 760 * 2 + 12
    Ht = tiles[0].height * 2 + 12
    sheet = Image.new("RGB", (Wt, Ht), (25, 25, 25))
    for i, t in enumerate(tiles[:4]):
        sheet.paste(t, ((i % 2) * (760 + 12), (i // 2) * (t.height + 12)))
    sheet.save(f"{OUTDIR}/fp_july.jpg", quality=76)
    print(f"\nsaved {OUTDIR}/fp_july.jpg")
