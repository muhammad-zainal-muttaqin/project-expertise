"""Pipeline Panen — deteksi agnostik -> penaut lintas-sisi -> skor kematangan
ordinal tingkat tandan -> pencacahan per kelas kasar.

Rancangan mengikuti bukti audit:
  * kelas diputuskan di tingkat tandan fisik (AF-E-002), bukan per tampak;
  * keputusan matang/belum adalah AMBANG pada skor kontinu, bukan akar hierarki
    keras, sehingga galat di batas B2|B3 masih dapat dipulihkan;
  * singleton ditangani sebagai keputusan terpisah karena lajunya bergantung
    kelas (AF-E-004);
  * kendala satu-proposal-per-sisi memakai UF yang sudah diperbaiki (AF-E-010).

Seluruh ambang ditala pada VALIDATION; TEST dibuka satu kali di akhir.
"""
import json, os, re, glob, sys
import numpy as np
import torch, torch.nn as nn
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
from torchvision import transforms as T, models
from PIL import Image
os.environ["YOLO_VERBOSE"] = "false"
from ultralytics import YOLO

MAY, DS = "/workspace/SawitMVC-YOLO", "/workspace/ds"
DET_W = "/workspace/runs_panen/agnostik_m1280/weights/best.pt"
CORN_W = "/workspace/crops953/corn_best.pt"
RES = "/workspace/results_panen"
os.makedirs(RES, exist_ok=True)
NSIDE, IMGSZ = 4, 1280
DEV = "cuda"

# ---------------------------------------------------------------- ground truth
GT = {}
for p in glob.glob(f"{MAY}/json/*.json"):
    d = json.load(open(p))
    ann = {}
    for side, v in d.get("images", {}).items():
        for a in v.get("annotations", []):
            ann[(side, a["box_index"])] = a["bbox_yolo"]
    bun = []
    for b in d.get("bunches", []) or []:
        c = {"B1": 0, "B2": 1, "B3": 2, "B4": 3}.get(b.get("class"), -1)
        app = [(int(x["side"].split("_")[1]), ann[(x["side"], x["box_index"])])
               for x in b.get("appearances", []) if (x["side"], x["box_index"]) in ann]
        if c >= 0 and app:
            bun.append(dict(c=c, app=app))
    GT[d["tree_id"]] = bun

split = {}
for i, line in enumerate(open(f"{MAY}/split_manifest.csv", encoding="utf-8-sig")):
    if i:
        f = line.strip().split(","); split[f[0]] = f[-1]

# Metrik pipeline hanya sah pada pohon empat sisi; 45 pohon berkamera 8 sisi
# dikeluarkan, sejalan dengan "135 pohon empat sisi" pada metrik proyek.
EMPAT_SISI = {t for t, b in GT.items()
              if b and max((s for x in b for s, _ in x["app"]), default=0) <= NSIDE}
print(f"pohon empat sisi: {len(EMPAT_SISI)} dari {len(GT)}")

# ---------------------------------------------------------------- deteksi
def detect(sp, conf=0.10):
    m = YOLO(DET_W)
    imgs = sorted(glob.glob(f"{DS}/may1/images/{sp}/*.jpg"))
    out = defaultdict(list)
    for i in range(0, len(imgs), 24):
        ch = imgs[i:i + 24]
        for p, r in zip(ch, m.predict(ch, imgsz=IMGSZ, conf=conf, iou=0.7, max_det=300,
                                      device=0, verbose=False)):
            fn = os.path.basename(p); mm = re.match(r"(.+)_(\d+)\.jpg$", fn)
            t, s = mm.group(1), int(mm.group(2))
            H, W = r.orig_shape
            for b, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                if s > NSIDE:
                    continue
                out[t].append(dict(side=s, conf=float(cf),
                                   box=[b[0] / W, b[1] / H, b[2] / W, b[3] / H],
                                   px=[float(x) for x in b], img=p, W=W, H=H))
    return out


def iou1(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    it = max(0., x2 - x1) * max(0., y2 - y1)
    ar = lambda z: (z[2] - z[0]) * (z[3] - z[1])
    return it / (ar(a) + ar(b) - it + 1e-9)


def yolo_to_xyxy(bb):
    cx, cy, w, h = bb
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


# ---------------------------------------------------------------- skor ordinal
class CORN(nn.Module):
    def __init__(self):
        super().__init__()
        b = models.convnext_small()
        self.feat = nn.Sequential(b.features, b.avgpool, b.classifier[0], b.classifier[1])
        self.ord = nn.Linear(768, 3); self.cls = nn.Linear(768, 4)
    def forward(self, x):
        f = self.feat(x); return self.ord(f), self.cls(f)


corn = CORN(); corn.load_state_dict(torch.load(CORN_W, map_location="cpu"))
corn = corn.to(DEV).eval()
tf = T.Compose([T.Resize((176, 176)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


@torch.no_grad()
def score_dets(dets):
    """Skor kematangan kontinu untuk setiap deteksi."""
    byimg = defaultdict(list)
    for d in dets:
        byimg[d["img"]].append(d)
    for img, ds in byimg.items():
        im = Image.open(img).convert("RGB")
        crops = []
        for d in ds:
            x0, y0, x1, y1 = d["px"]
            side = max(x1 - x0, y1 - y0) * 1.6
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            crops.append(tf(im.crop((int(cx - side / 2), int(cy - side / 2),
                                     int(cx + side / 2), int(cy + side / 2)))))
        for i in range(0, len(crops), 64):
            with torch.autocast("cuda", torch.bfloat16):
                lo, _ = corn(torch.stack(crops[i:i + 64]).to(DEV))
            cum = torch.cumprod(torch.sigmoid(lo.float()), 1)
            for d, s in zip(ds[i:i + 64], cum.sum(1).cpu().numpy()):
                d["score"] = float(s)


# ---------------------------------------------------------------- penaut
def pair_feats(a, b, n=NSIDE):
    off = (b["side"] - a["side"]) % n
    ax, ay = (a["box"][0] + a["box"][2]) / 2, (a["box"][1] + a["box"][3]) / 2
    bx, by = (b["box"][0] + b["box"][2]) / 2, (b["box"][1] + b["box"][3]) / 2
    aa = max((a["box"][2] - a["box"][0]) * (a["box"][3] - a["box"][1]), 1e-8)
    ab = max((b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1]), 1e-8)
    sgn = 1.0 if off == 1 else -1.0
    return [sgn * (bx - ax), by - ay, abs(by - ay), np.log(ab / aa),
            ax, bx, ay, by, a["conf"], b["conf"], float(off == 1),
            abs(a.get("score", 0) - b.get("score", 0)), a.get("score", 0) + b.get("score", 0)]


def gt_bunch_of(det, tree):
    """Indeks tandan acuan yang cocok dengan deteksi ini (IoU>=0.5 pada sisi sama)."""
    best, bi = 0.5, -1
    for i, b in enumerate(GT.get(tree, [])):
        for s, bb in b["app"]:
            if s == det["side"]:
                v = iou1(det["box"], yolo_to_xyxy(bb))
                if v > best:
                    best, bi = v, i
    return bi


def build_pairs(det_by_tree, trees, labelled=True):
    X, y, keys = [], [], []
    for t in trees:
        ds = det_by_tree.get(t, [])
        gi = [gt_bunch_of(d, t) for d in ds] if labelled else [None] * len(ds)
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                if ds[i]["side"] == ds[j]["side"]:
                    continue
                if (ds[j]["side"] - ds[i]["side"]) % NSIDE not in (1, NSIDE - 1):
                    continue
                a, b = (ds[i], ds[j]) if ds[i]["side"] < ds[j]["side"] else (ds[j], ds[i])
                X.append(pair_feats(a, b))
                keys.append((t, i, j))
                if labelled:
                    y.append(int(gi[i] >= 0 and gi[i] == gi[j]))
    return np.array(X, float), np.array(y, int), keys


class UF:
    """Salinan versi yang sudah diperbaiki (AF-E-010)."""
    def __init__(self, sides, max_size):
        n = len(sides)
        self.p = list(range(n)); self.sz = [1] * n
        self.sides = [{s} for s in sides]; self.max = max_size
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b or (self.sides[a] & self.sides[b]) or self.sz[a] + self.sz[b] > self.max:
            return False
        self.p[b] = a; self.sz[a] += self.sz[b]; self.sides[a] |= self.sides[b]
        return True


def cluster_tree(ds, edge_p, link_thr, max_size):
    """Hungarian per pasangan sisi -> UF berkendala sisi."""
    by_side = defaultdict(list)
    for i, d in enumerate(ds):
        by_side[d["side"]].append(i)
    cand = []
    for sa in sorted(by_side):
        for sb in sorted(by_side):
            if sb <= sa or (sb - sa) % NSIDE not in (1, NSIDE - 1):
                continue
            aa, bb = by_side[sa], by_side[sb]
            M = np.zeros((len(aa), len(bb)))
            for x, i in enumerate(aa):
                for yy, j in enumerate(bb):
                    M[x, yy] = edge_p.get((min(i, j), max(i, j)), 0.0)
            if M.size == 0:
                continue
            for r, c in zip(*linear_sum_assignment(-M)):
                if M[r, c] >= link_thr:
                    cand.append((M[r, c], aa[r], bb[c]))
    cand.sort(reverse=True)
    uf = UF([d["side"] for d in ds], max_size)
    for s, i, j in cand:
        uf.union(i, j)
    g = defaultdict(list)
    for i in range(len(ds)):
        g[uf.find(i)].append(i)
    return list(g.values())


# ---------------------------------------------------------------- evaluasi
def evaluate(det_by_tree, trees, edge_model, link_thr, max_size, single_thr,
             t_coarse, t_b1b2, t_b3b4, det_conf):
    rows, per_tree = [], []
    for t in trees:
        ds = [d for d in det_by_tree.get(t, []) if d["conf"] >= det_conf]
        if ds:
            X, _, keys = build_pairs({t: ds}, [t], labelled=False)
            pr = edge_model.predict_proba(X)[:, 1] if len(X) else []
            ep = {(min(i, j), max(i, j)): float(p) for (_, i, j), p in zip(keys, pr)}
            groups = cluster_tree(ds, ep, link_thr, max_size)
        else:
            groups = []
        clus = []
        for g in groups:
            mem = [ds[i] for i in g]
            w = np.array([m["conf"] for m in mem])
            if len(mem) == 1 and w[0] < single_thr:
                continue
            clus.append(dict(members=mem, score=float(np.average(
                [m["score"] for m in mem], weights=w)), conf=float(w.mean())))
        # pencocokan klaster <-> tandan acuan (serakah menurut conf)
        gt = GT.get(t, [])
        used, match = set(), {}
        for ci, c in enumerate(sorted(range(len(clus)), key=lambda k: -clus[k]["conf"])):
            best, bi = 0.5, -1
            for gi, b in enumerate(gt):
                if gi in used:
                    continue
                v = 0.
                for m in clus[c]["members"]:
                    for s, bb in b["app"]:
                        if s == m["side"]:
                            v = max(v, iou1(m["box"], yolo_to_xyxy(bb)))
                if v > best:
                    best, bi = v, gi
            if bi >= 0:
                used.add(bi); match[c] = bi
        tp, fp, fn = len(match), len(clus) - len(match), len(gt) - len(match)
        def coarse(s):  return 0 if s < t_coarse else 1
        def fine(s):
            return (0 if s < t_b1b2 else 1) if s < t_coarse else (2 if s < t_b3b4 else 3)
        pred_c = np.array([coarse(c["score"]) for c in clus])
        gt_c = np.array([0 if b["c"] <= 1 else 1 for b in gt])
        per_tree.append(dict(
            tree=t, n_pred=len(clus), n_gt=len(gt), tp=tp, fp=fp, fn=fn,
            pred_matang=int((pred_c == 0).sum()), gt_matang=int((gt_c == 0).sum()),
            pred_belum=int((pred_c == 1).sum()), gt_belum=int((gt_c == 1).sum())))
        for c, gi in match.items():
            rows.append(dict(y=gt[gi]["c"], s=clus[c]["score"],
                             p4=fine(clus[c]["score"]), p2=coarse(clus[c]["score"])))
    TP = sum(r["tp"] for r in per_tree); FP = sum(r["fp"] for r in per_tree)
    FN = sum(r["fn"] for r in per_tree)
    f1 = 2 * TP / max(2 * TP + FP + FN, 1)
    def cnt(a, b):
        e = np.array([abs(r[a] - r[b]) for r in per_tree], float)
        return dict(mae=float(e.mean()), exact=float((e == 0).mean()),
                    within1=float((e <= 1).mean()))
    y4 = np.array([r["y"] for r in rows]); p4 = np.array([r["p4"] for r in rows])
    y2 = (y4 > 1).astype(int); p2 = np.array([r["p2"] for r in rows])
    from sklearn.metrics import f1_score
    return dict(
        physical_f1=f1, precision=TP / max(TP + FP, 1), recall=TP / max(TP + FN, 1),
        n_matched=len(rows),
        count_total=cnt("n_pred", "n_gt"), count_matang=cnt("pred_matang", "gt_matang"),
        count_belum=cnt("pred_belum", "gt_belum"),
        class2_acc=float((p2 == y2).mean()) if len(rows) else 0.,
        class2_f1=float(f1_score(y2, p2)) if len(rows) else 0.,
        class4_acc=float((p4 == y4).mean()) if len(rows) else 0.,
        class4_within1=float((np.abs(p4 - y4) <= 1).mean()) if len(rows) else 0.,
        class4_macro_f1=float(f1_score(y4, p4, average="macro")) if len(rows) else 0.,
    ), per_tree


if __name__ == "__main__":
    print("=== deteksi ===", flush=True)
    D = {}
    for sp in ["train", "val", "test"]:
        D[sp] = detect(sp)
        for t in D[sp]:
            score_dets(D[sp][t])
        print(f"  {sp}: {len(D[sp])} pohon, {sum(len(v) for v in D[sp].values())} deteksi", flush=True)

    print("=== melatih penaut tepi pada TRAIN ===", flush=True)
    Xtr, ytr, _ = build_pairs(D["train"], [t for t in D["train"] if t in EMPAT_SISI])
    print(f"  pasangan latih {len(ytr)}, positif {int(ytr.sum())}", flush=True)
    E = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                       l2_regularization=1.0, random_state=42).fit(Xtr, ytr)
    Xv, yv, _ = build_pairs(D["val"], [t for t in D["val"] if t in EMPAT_SISI])
    from sklearn.metrics import roc_auc_score, average_precision_score
    pv = E.predict_proba(Xv)[:, 1]
    print(f"  VAL tepi AUC={roc_auc_score(yv, pv):.4f} AP={average_precision_score(yv, pv):.4f}",
          flush=True)

    json.dump(dict(n_pairs=int(len(ytr)), n_pos=int(ytr.sum()),
                   val_auc=float(roc_auc_score(yv, pv)),
                   val_ap=float(average_precision_score(yv, pv))),
              open(f"{RES}/linker.json", "w"), indent=1)
    import pickle
    pickle.dump(E, open(f"{RES}/edge_model.pkl", "wb"))
    pickle.dump({sp: D[sp] for sp in D}, open(f"{RES}/dets.pkl", "wb"))
    print("PIPELINE STAGE-1 DONE", flush=True)
