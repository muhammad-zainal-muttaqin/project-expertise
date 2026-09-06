"""Does the July/August annotation draw ONE loose box around a CLUSTER of bunches
where May draws several tight ones?  If so, one GT box should contain several
confident detections from the May-trained model."""
import os, glob, json
import numpy as np
from collections import Counter
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


def run(idir, ldir, label, conf=0.5):
    imgs = sorted(glob.glob(f"{idir}/*.jpg"))
    occupancy = Counter()      # how many confident detections sit inside one GT box
    gt_area, det_area = [], []
    for i in range(0, len(imgs), 32):
        ch = imgs[i:i + 32]
        for p, r in zip(ch, m.predict(ch, imgsz=960, conf=conf, iou=0.7, max_det=300,
                                      device=0, verbose=False)):
            H, W = r.orig_shape
            g = gt_of(f"{ldir}/{os.path.basename(p)[:-4]}.txt", W, H)
            d = r.boxes.xyxy.cpu().numpy()
            if len(g) == 0:
                continue
            cx = (d[:, 0] + d[:, 2]) / 2 if len(d) else np.zeros(0)
            cy = (d[:, 1] + d[:, 3]) / 2 if len(d) else np.zeros(0)
            for j, gb in enumerate(g):
                k = ((gb[0] <= cx) & (cx <= gb[2]) & (gb[1] <= cy) & (cy <= gb[3])) if len(d) else []
                occupancy[int(np.sum(k))] += 1
                gt_area.append(((gb[2] - gb[0]) * (gb[3] - gb[1])) / (W * H))
            for b in d:
                det_area.append(((b[2] - b[0]) * (b[3] - b[1])) / (W * H))
    tot = sum(occupancy.values())
    multi = sum(v for k, v in occupancy.items() if k >= 2)
    print(f"\n{label}")
    print(f"  kotak acuan                       : {tot}")
    print("  deteksi yakin di dalam satu kotak : " +
          "  ".join(f"{k}→{100*occupancy[k]/tot:.1f}%" for k in sorted(occupancy) if k <= 4))
    print(f"  kotak acuan berisi >=2 deteksi    : {multi} ({100*multi/tot:.1f}%)")
    print(f"  luas kotak acuan (median, ternorm): {np.median(gt_area):.5f}")
    print(f"  luas kotak model (median, ternorm): {np.median(det_area):.5f}"
          f"   rasio acuan/model = {np.median(gt_area)/max(np.median(det_area),1e-9):.2f}x")
    return dict(total=tot, multi=multi, frac_multi=multi / max(tot, 1),
                gt_area=float(np.median(gt_area)), det_area=float(np.median(det_area)))


out = {}
out["mei"] = run(f"{DS}/may1/images/test", f"{DS}/may1/labels/test", "MEI 953 test (protokol sendiri)")
out["juli"] = run(f"{DS}/dep1/images/jul_all", f"{DS}/dep1/labels/jul_all", "JULI 352")
out["agustus"] = run(f"{DS}/dep1/images/aug_all", f"{DS}/dep1/labels/aug_all", "AGUSTUS 411")
json.dump(out, open("/workspace/results_audit/e1d_merge.json", "w"), indent=1)
