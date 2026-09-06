"""E3 harvest counting · E4 structure fusion on real detections · E5 UF side-constraint bug."""
import os, json, glob, re, sys
import numpy as np
from collections import defaultdict, Counter
os.environ["YOLO_VERBOSE"] = "false"
from ultralytics import YOLO

MAY = "/workspace/SawitMVC-YOLO"
DS = "/workspace/ds"
RUNS = "/workspace/runs_audit"
RES = "/workspace/results_audit"
CID = {"B1": 0, "B2": 1, "B3": 2, "B4": 3}
IMGSZ = 960
out = {}

split = {}
for i, line in enumerate(open(f"{MAY}/split_manifest.csv", encoding="utf-8-sig")):
    if i:
        f = line.strip().split(","); split[f[0]] = f[-1]

# ---- ground truth per tree -------------------------------------------------
gt = {}
for p in glob.glob(f"{MAY}/json/*.json"):
    d = json.load(open(p))
    imgs = d.get("images", {})
    ann = {}
    for side, v in imgs.items():
        for a in v.get("annotations", []):
            ann[(side, a["box_index"])] = a["bbox_yolo"]
    B = []
    for b in d.get("bunches", []) or []:
        c = CID.get(b.get("class"), -1)
        bb = [ann[(x["side"], x["box_index"])] for x in b.get("appearances", [])
              if (x["side"], x["box_index"]) in ann]
        if c >= 0 and bb:
            B.append(dict(c=c, cy=float(np.median([z[1] for z in bb])),
                          ar=float(np.median([z[2] * z[3] for z in bb])), n=len(bb)))
    gt[d["tree_id"]] = B

# ---- run a detector over a split, keep raw boxes ---------------------------
def detect(weights, imgs, conf=0.01):
    m = YOLO(weights)
    res = {}
    B = 32
    for i in range(0, len(imgs), B):
        for p, r in zip(imgs[i:i + B], m.predict(imgs[i:i + B], imgsz=IMGSZ, conf=conf,
                                                 iou=0.7, max_det=300, device=0,
                                                 verbose=False)):
            b = r.boxes
            res[os.path.basename(p)] = dict(
                xyxy=b.xyxy.cpu().numpy(), conf=b.conf.cpu().numpy(),
                cls=b.cls.cpu().numpy().astype(int),
                wh=(r.orig_shape[1], r.orig_shape[0]))
    return res


def tree_side(fn):
    m = re.match(r"(.+)_(\d+)\.jpg$", fn)
    return m.group(1), int(m.group(2))


# ============================ E3 : harvest counting =========================
print("=== E3 harvest counting (2-class detector -> Ridge F_all) ===", flush=True)
W2 = f"{RUNS}/may2/weights/best.pt"
det = {}
for sp in ["train", "val", "test"]:
    det[sp] = detect(W2, sorted(glob.glob(f"{DS}/may2/images/{sp}/*.jpg")))
    print(f"  detected {sp}: {len(det[sp])} images", flush=True)

THR = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]


def features(dets):
    """F_all-style: per-class detection counts at many thresholds + geometry stats."""
    per_tree = defaultdict(list)
    for fn, d in dets.items():
        t, s = tree_side(fn)
        per_tree[t].append((s, d))
    X, keys = {}, []
    for t, sides in per_tree.items():
        f = []
        for c in [0, 1]:
            for th in THR:
                f.append(sum(int(((d["cls"] == c) & (d["conf"] >= th)).sum()) for _, d in sides))
        allc = np.concatenate([d["conf"] for _, d in sides]) if sides else np.zeros(1)
        ripe = np.concatenate([d["conf"][d["cls"] == 0] for _, d in sides]) if sides else np.zeros(1)
        f += [len(sides), float(allc.sum()), float(ripe.sum()) if ripe.size else 0.0,
              float(allc.mean()) if allc.size else 0.0,
              float((allc >= 0.25).sum()), float((ripe >= 0.25).sum()) if ripe.size else 0.0]
        X[t] = f
    return X


F = {sp: features(det[sp]) for sp in det}
from sklearn.linear_model import RidgeCV


def target(t, kind):
    B = gt.get(t, [])
    return sum(1 for b in B if b["c"] == 0) if kind == "ripe" else len(B)


def fit_eval(fit_split, kind):
    tr = [t for t in F[fit_split] if t in gt]
    te = [t for t in F["test"] if t in gt]
    Xtr = np.array([F[fit_split][t] for t in tr], float)
    ytr = np.array([target(t, kind) for t in tr], float)
    Xte = np.array([F["test"][t] for t in te], float)
    yte = np.array([target(t, kind) for t in te], float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    R = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit((Xtr - mu) / sd, ytr)
    p = np.clip(np.round(R.predict((Xte - mu) / sd)), 0, None)
    e = np.abs(p - yte)
    return dict(n_fit=len(tr), n_test=len(te), mae=float(e.mean()),
                exact=float((e == 0).mean()), within1=float((e <= 1).mean()),
                mean_true=float(yte.mean()))


out["E3"] = {}
for kind in ["ripe", "total"]:
    for fs in ["train", "val"]:
        k = f"{kind}_fit_on_{fs}"
        out["E3"][k] = fit_eval(fs, kind)
        r = out["E3"][k]
        print(f"  {k:22} MAE={r['mae']:.3f} exact={r['exact']:.3f} ±1={r['within1']:.3f} "
              f"(rerata acuan {r['mean_true']:.2f}/pohon)", flush=True)
json.dump(out, open(f"{RES}/e345.json", "w"), indent=1)

# ================= E4 : structure fusion on REAL detections =================
print("\n=== E4 structure fusion on real detections (agnostic detector + crop head) ===", flush=True)
import torch, torch.nn as nn
from torchvision import transforms as T, models
from PIL import Image

W1 = f"{RUNS}/may1/weights/best.pt"
det1 = {sp: detect(W1, sorted(glob.glob(f"{DS}/may1/images/{sp}/*.jpg")), conf=0.20)
        for sp in ["val", "test"]}

cnn = models.convnext_tiny()
cnn.classifier[2] = nn.Linear(768, 4)
cnn.load_state_dict(torch.load("/workspace/crops953/convnext.pt", map_location="cpu"))
cnn = cnn.cuda().eval()
tf = T.Compose([T.Resize((144, 144)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


def gt_boxes(tree, side, W, H):
    d = json.load(open(f"{MAY}/json/{tree}.json"))
    v = d["images"].get(f"side_{side}")
    if not v:
        return np.zeros((0, 4)), []
    bb, cc = [], []
    for a in v.get("annotations", []):
        cx, cy, w, h = a["bbox_yolo"]
        bb.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
        cc.append(a["class_id"])
    return np.array(bb, float).reshape(-1, 4), cc


def collect(sp):
    rows = []
    for fn, d in det1[sp].items():
        t, s = tree_side(fn)
        W, H = d["wh"]
        g, gc = gt_boxes(t, s, W, H)
        M = iou_mat(d["xyxy"], g)
        img = Image.open(f"{DS}/may1/images/{sp}/{fn}").convert("RGB")
        crops, meta = [], []
        for i, box in enumerate(d["xyxy"]):
            j = int(M[i].argmax()) if M.shape[1] else -1
            if j < 0 or M[i, j] < 0.5:
                continue
            x0, y0, x1, y1 = box
            side_px = max(x1 - x0, y1 - y0) * 1.6
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            crops.append(tf(img.crop((int(cx - side_px / 2), int(cy - side_px / 2),
                                      int(cx + side_px / 2), int(cy + side_px / 2)))))
            meta.append(dict(tree=t, side=s, y=gc[j], gtidx=j, cy=cy / H,
                             ar=((x1 - x0) * (y1 - y0)) / (W * H), conf=float(d["conf"][i])))
        if crops:
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                P = torch.softmax(cnn(torch.stack(crops).cuda()).float(), 1).cpu().numpy()
            for m, p in zip(meta, P):
                m["p"] = p; rows.append(m)
    return rows


rv, rt = collect("val"), collect("test")
print(f"  matched detections: val={len(rv)} test={len(rt)}", flush=True)


def add_struct(rows):
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


rv, rt = add_struct(rv), add_struct(rt)
G = lambda rows: np.array([[r["rcy"], r["rar"], r["cy"], np.log(r["ar"]), r["n"],
                            r["conf"], r["dcy"], r["dar"]] for r in rows])
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

yv = np.array([r["y"] for r in rv]); yt = np.array([r["y"] for r in rt])
S = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, l2_regularization=1.0,
                                   random_state=42).fit(G(rv), yv)
prior = np.bincount(yv, minlength=4) / len(yv)
LP = np.log(prior)


def fuse(rows, w):
    A = np.log(np.clip(np.array([r["p"] for r in rows]), 1e-9, 1))
    Sp = np.log(np.clip(S.predict_proba(G(rows)), 1e-9, 1))
    z = A + w * (Sp - LP); z -= z.max(1, keepdims=True)
    e = np.exp(z); return e / e.sum(1, keepdims=True)


best, bw = -1, 0
for w in [0, .2, .4, .6, .8, 1.0, 1.3]:
    p = fuse(rv, w).argmax(1)
    s = accuracy_score(yv, p) + f1_score(yv, p, average="macro")
    if s > best:
        best, bw = s, w
base = np.array([r["p"].argmax() for r in rt])
fus = fuse(rt, bw).argmax(1)
out["E4"] = dict(w=bw, n_test=len(rt),
                 appearance=dict(acc=float(accuracy_score(yt, base)),
                                 macro_f1=float(f1_score(yt, base, average="macro"))),
                 fused=dict(acc=float(accuracy_score(yt, fus)),
                            macro_f1=float(f1_score(yt, fus, average="macro"))))
print(f"  w={bw}  appearance acc={out['E4']['appearance']['acc']:.4f} "
      f"macroF1={out['E4']['appearance']['macro_f1']:.4f}", flush=True)
print(f"        fused      acc={out['E4']['fused']['acc']:.4f} "
      f"macroF1={out['E4']['fused']['macro_f1']:.4f}", flush=True)
json.dump(out, open(f"{RES}/e345.json", "w"), indent=1)

# ==================== E5 : UF side-constraint bug ==========================
print("\n=== E5 UF side-constraint bug, measured on real proposals ===", flush=True)


class UF_buggy:
    def __init__(self, dets, max_size):
        n = len(dets)
        self.p = list(range(n)); self.sz = [1] * n
        self.sides = [{i} for i in range(n)]        # <-- repo code: proposal INDEX
        self.max = max_size
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b or (self.sides[a] & self.sides[b]) or self.sz[a] + self.sz[b] > self.max:
            return False
        self.p[b] = a; self.sz[a] += self.sz[b]; self.sides[a] |= self.sides[b]; return True


class UF_fixed(UF_buggy):
    def __init__(self, dets, max_size):
        super().__init__(dets, max_size)
        self.sides = [{d} for d in dets]            # <-- physical side


def pair_edges(items):
    """cheap geometric pair score between detections on different sides"""
    E = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if a["side"] == b["side"]:
                continue
            if (b["side"] - a["side"]) % 4 not in (1, 3):
                continue
            s = (1 - abs(a["cy"] - b["cy"]) * 2) * 0.6 + \
                (1 - abs(np.log(a["ar"] / b["ar"])) / 2) * 0.4
            if s > 0:
                E.append((s, i, j))
    E.sort(reverse=True)
    return E


stats = {"buggy": Counter(), "fixed": Counter()}
for sp in ["test"]:
    bytree = defaultdict(list)
    for fn, d in det1[sp].items():
        t, s = tree_side(fn)
        W, H = d["wh"]
        for k in range(len(d["conf"])):
            x0, y0, x1, y1 = d["xyxy"][k]
            bytree[t].append(dict(side=s, cy=(y0 + y1) / 2 / H,
                                  ar=max(((x1 - x0) * (y1 - y0)) / (W * H), 1e-6)))
    for t, items in bytree.items():
        E = pair_edges(items)
        for name, K in [("buggy", UF_buggy), ("fixed", UF_fixed)]:
            uf = K([it["side"] for it in items], 3)
            for s, i, j in E:
                if s < 0.30:
                    break
                uf.union(i, j)
            grp = defaultdict(list)
            for i, it in enumerate(items):
                grp[uf.find(i)].append(it["side"])
            stats[name]["clusters"] += len(grp)
            stats[name]["violating"] += sum(1 for g in grp.values() if len(g) != len(set(g)))
    ntree = len(bytree)
    out["E5"] = {k: dict(clusters=v["clusters"], clusters_with_same_side_duplicate=v["violating"],
                         trees=ntree) for k, v in stats.items()}
    for k, v in out["E5"].items():
        print(f"  {k:6}: {v['clusters']} klaster · "
              f"{v['clusters_with_same_side_duplicate']} memuat >=2 deteksi dari sisi yang sama "
              f"({100*v['clusters_with_same_side_duplicate']/max(v['clusters'],1):.1f}%)", flush=True)

json.dump(out, open(f"{RES}/e345.json", "w"), indent=1)
print("\nE345 DONE")
