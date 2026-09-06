"""E4 redone with a clean protocol:
   structure model fitted on TRAIN detections · fusion weight tuned on VAL · TEST opened once."""
import os, json, glob, re
import numpy as np
from collections import defaultdict
os.environ["YOLO_VERBOSE"] = "false"
from ultralytics import YOLO
import torch, torch.nn as nn
from torchvision import transforms as T, models
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

MAY, DS, RUNS = "/workspace/SawitMVC-YOLO", "/workspace/ds", "/workspace/runs_audit"
det_model = YOLO(f"{RUNS}/may1/weights/best.pt")
cnn = models.convnext_tiny(); cnn.classifier[2] = nn.Linear(768, 4)
cnn.load_state_dict(torch.load("/workspace/crops953/convnext.pt", map_location="cpu"))
cnn = cnn.cuda().eval()
tf = T.Compose([T.Resize((144, 144)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
GT = {}


def gt_boxes(tree, side, W, H):
    if tree not in GT:
        GT[tree] = json.load(open(f"{MAY}/json/{tree}.json"))
    v = GT[tree]["images"].get(f"side_{side}")
    if not v:
        return np.zeros((0, 4)), []
    bb, cc = [], []
    for a in v.get("annotations", []):
        cx, cy, w, h = a["bbox_yolo"]
        bb.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
        cc.append(a["class_id"])
    return np.array(bb, float).reshape(-1, 4), cc


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    it = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return it / (ar(a)[:, None] + ar(b)[None, :] - it + 1e-9)


def collect(sp):
    rows, imgs = [], sorted(glob.glob(f"{DS}/may1/images/{sp}/*.jpg"))
    for i in range(0, len(imgs), 32):
        ch = imgs[i:i + 32]
        for p, r in zip(ch, det_model.predict(ch, imgsz=960, conf=0.20, iou=0.7,
                                              max_det=300, device=0, verbose=False)):
            fn = os.path.basename(p); mm = re.match(r"(.+)_(\d+)\.jpg$", fn)
            t, s = mm.group(1), int(mm.group(2))
            H, W = r.orig_shape
            g, gc = gt_boxes(t, s, W, H)
            d = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
            M = iou_mat(d, g)
            img = Image.open(p).convert("RGB")
            crops, meta = [], []
            for k in range(len(d)):
                j = int(M[k].argmax()) if M.shape[1] else -1
                if j < 0 or M[k, j] < 0.5:
                    continue
                x0, y0, x1, y1 = d[k]
                sp_px = max(x1 - x0, y1 - y0) * 1.6
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                crops.append(tf(img.crop((int(cx - sp_px / 2), int(cy - sp_px / 2),
                                          int(cx + sp_px / 2), int(cy + sp_px / 2)))))
                meta.append(dict(tree=t, y=gc[j], cy=cy / H, conf=float(cf[k]),
                                 ar=max(((x1 - x0) * (y1 - y0)) / (W * H), 1e-7)))
            if crops:
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                    P = torch.softmax(cnn(torch.stack(crops).cuda()).float(), 1).cpu().numpy()
                for m_, p_ in zip(meta, P):
                    m_["p"] = p_; rows.append(m_)
    bt = defaultdict(list)
    for r in rows:
        bt[r["tree"]].append(r)
    for t, rs in bt.items():
        n = len(rs)
        cy = np.array([r["cy"] for r in rs]); ar = np.array([r["ar"] for r in rs])
        rcy = cy.argsort().argsort() / max(n - 1, 1)
        rar = (-ar).argsort().argsort() / max(n - 1, 1)
        for i, r in enumerate(rs):
            r.update(rcy=rcy[i], rar=rar[i], n=n, dcy=cy[i] - cy.mean(),
                     dar=np.log(ar[i]) - np.log(ar).mean())
    return rows


R = {sp: collect(sp) for sp in ["train", "val", "test"]}
print({k: len(v) for k, v in R.items()}, flush=True)
G = lambda rs: np.array([[r["rcy"], r["rar"], r["cy"], np.log(r["ar"]), r["n"],
                          r["conf"], r["dcy"], r["dar"]] for r in rs])
Y = lambda rs: np.array([r["y"] for r in rs])

S = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, l2_regularization=1.0,
                                   random_state=42).fit(G(R["train"]), Y(R["train"]))
LP = np.log(np.bincount(Y(R["train"]), minlength=4) / len(R["train"]))


def fuse(rs, w):
    A = np.log(np.clip(np.array([r["p"] for r in rs]), 1e-9, 1))
    Sp = np.log(np.clip(S.predict_proba(G(rs)), 1e-9, 1))
    z = A + w * (Sp - LP); z -= z.max(1, keepdims=True)
    e = np.exp(z); return e / e.sum(1, keepdims=True)


print("\n--- tala w di VAL ---")
best, bw = -1, 0
for w in [0, .1, .2, .3, .4, .6, .8, 1.0]:
    p = fuse(R["val"], w).argmax(1)
    a, f = accuracy_score(Y(R["val"]), p), f1_score(Y(R["val"]), p, average="macro")
    print(f"   w={w:<4} acc={a:.4f} macroF1={f:.4f}")
    if a + f > best:
        best, bw = a + f, w
print(f"  -> w terpilih = {bw}")

yt = Y(R["test"])
res = {}
for name, pred in [("penampilan saja", np.array([r["p"].argmax() for r in R["test"]])),
                   ("struktur saja", S.predict(G(R["test"]))),
                   (f"penampilan + struktur (w={bw})", fuse(R["test"], bw).argmax(1))]:
    res[name] = dict(acc=float(accuracy_score(yt, pred)),
                     macro_f1=float(f1_score(yt, pred, average="macro")),
                     within1=float(np.mean(np.abs(pred - yt) <= 1)))
    print(f"  {name:34} acc={res[name]['acc']:.4f} macroF1={res[name]['macro_f1']:.4f} "
          f"±1={res[name]['within1']:.4f}")
json.dump(dict(n=len(yt), w=bw, results=res),
          open("/workspace/results_audit/e4b.json", "w"), indent=1)
