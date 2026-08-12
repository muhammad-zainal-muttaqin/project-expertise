"""Evaluasi pipeline dua-tahap: detektor class-agnostic + classifier crop — Fase 6.

Tahap 1 mencari tandan (1 kelas), tahap 2 memberi kematangan pada tiap box.
Kelas + confidence tahap 2 ditempel kembali ke box tahap 1, lalu dihitung mAP50
yang LANGSUNG sebanding dengan seluruh angka Fase 1-5 (split test 352 yang sama,
IoU 0,5, rata-rata makro 4 kelas).

Juga melaporkan AP50 class-agnostic (plafon teoretis pipeline ini) supaya
terlihat berapa banyak yang masih hilang di tahap klasifikasi.

Usage:
    .venv/bin/python scripts/eval_twostage.py \
        --detektor runs/agn352_ft/weights/best.pt \
        --classifier runs_fase6/ft_rgbd/best.pt \
        --split test --out results/twostage_rgbd.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from build_crop_dataset import CTX, RELIEF_M, S, ambil, dekode_z, kotak_persegi  # noqa: E402
from train_crop_classifier import IMG, K, Model, prob_head  # noqa: E402

D352 = Path("/workspace/SawitMVC-Depth")
DEPTH = Path("/workspace/depth_png_352")
SPLIT = D352 / "splits" / "canonical_70_15_15"
W, H = 1280, 800


# ------------------------------------------------------------------ metrik AP50

def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def ap50(gt: dict, pred: dict, kelas: int | None):
    """AP50 gaya COCO (interpolasi 101 titik). kelas=None -> class-agnostic."""
    rekam, npos = [], 0
    for stem, g in gt.items():
        gg = g if kelas is None else g[g[:, 0] == kelas]
        npos += len(gg)
        pr = pred.get(stem, np.zeros((0, 6)))
        if kelas is not None:
            pr = pr[pr[:, 5] == kelas]
        pr = pr[np.argsort(-pr[:, 4])]
        M = iou_mat(pr[:, :4], gg[:, 1:5])
        dipakai = np.zeros(len(gg), bool)
        for k in range(len(pr)):
            kol = np.where(dipakai, -1.0, M[k]) if len(gg) else np.zeros(0)
            j = int(np.argmax(kol)) if len(gg) else -1
            if j >= 0 and kol[j] >= 0.5:
                dipakai[j] = True
                rekam.append((pr[k, 4], 1))
            else:
                rekam.append((pr[k, 4], 0))
    if npos == 0:
        return float("nan")
    if not rekam:
        return 0.0
    rekam.sort(key=lambda x: -x[0])
    tp = np.cumsum([r[1] for r in rekam]); fp = np.cumsum([1 - r[1] for r in rekam])
    rec, prec = tp / npos, tp / (tp + fp)
    return float(np.mean([prec[rec >= t].max() if (rec >= t).any() else 0.0
                          for t in np.linspace(0, 1, 101)]))


def wbf(kotak: np.ndarray, iou_th: float = 0.6, n_model: int = 2) -> np.ndarray:
    """Weighted box fusion sederhana pada satu citra (kotak: N x 5 = xyxy+skor)."""
    if len(kotak) == 0:
        return kotak
    kotak = kotak[np.argsort(-kotak[:, 4])]
    gugus: list[list[np.ndarray]] = []
    for k in kotak:
        for g in gugus:
            if iou_mat(k[None, :4], g[0][None, :4])[0, 0] >= iou_th:
                g.append(k)
                break
        else:
            gugus.append([k])
    keluar = []
    for g in gugus:
        a = np.stack(g)
        bobot = a[:, 4:5]
        xy = (a[:, :4] * bobot).sum(0) / bobot.sum()
        # Skor = rata-rata skor anggota, diredam kalau hanya SATU detektor yang
        # menemukannya (n_model dipakai sebagai jumlah detektor yang ideal
        # menyetujui). Kotak yang disepakati banyak detektor naik peringkat.
        keluar.append([*xy, float(a[:, 4].mean() * min(len(g), n_model) / n_model)])
    return np.array(keluar, float)


# --------------------------------------------------------------------- pipeline

def siapkan_crop(bgr, Z, box, sisi=None):
    """Crop satu box persis seperti saat training classifier (harus identik).

    `sisi` = resolusi crop tersimpan saat training. Kalau model dilatih pada
    crop 256 tapi di sini di-crop 176 lalu diperbesar ke IMG, detail halus
    (yang justru mendefinisikan kematangan) hilang.
    """
    global S
    S_lama = S
    if sisi:
        S = sisi
    x0f, y0f, x1f, y1f = box
    cx = (x0f + x1f) / 2 / W; cy = (y0f + y1f) / 2 / H
    w = (x1f - x0f) / W; h = (y1f - y0f) / H
    x0, y0, x1, y1 = kotak_persegi(cx, cy, w, h, W, H)
    if x1 - x0 < 8:
        return None
    rgb = cv2.resize(ambil(bgr, x0, y0, x1, y1, 0), (S, S), interpolation=cv2.INTER_AREA)

    sisi = x1 - x0
    mw, mh = (x1f - x0f) / sisi, (y1f - y0f) / sisi
    msk = np.zeros((S, S), np.uint8)
    mx0 = int(round((0.5 - mw / 2) * S)); mx1 = int(round((0.5 + mw / 2) * S))
    my0 = int(round((0.5 - mh / 2) * S)); my1 = int(round((0.5 + mh / 2) * S))
    msk[max(0, my0):min(S, my1), max(0, mx0):min(S, mx1)] = 255

    if Z is None:
        dep = np.zeros((S, S, 2), np.uint8)
    else:
        zc = ambil(Z, x0, y0, x1, y1, np.nan)
        valid = np.isfinite(zc)
        if valid.sum() < 20:
            dep = np.zeros((S, S, 2), np.uint8)
        else:
            ref = float(np.median(zc[valid]))
            rel = np.clip(zc - ref, -RELIEF_M, RELIEF_M)
            v = np.where(valid, 128.0 + rel / RELIEF_M * 127.0, 128.0)
            dep = np.stack([
                cv2.resize(v.astype(np.uint8), (S, S), interpolation=cv2.INTER_AREA),
                cv2.resize((valid * 255).astype(np.uint8), (S, S), interpolation=cv2.INTER_AREA),
            ], -1)
    S = S_lama
    return rgb, dep, msk


def muat_detektor(jalur: str):
    """Muat detektor dengan kelas ultralytics yang benar.

    `YOLO(...)` menerima bobot RT-DETR tanpa error tapi membangunnya sebagai
    DetectionModel biasa — head-nya beda, jadi hasilnya tidak bisa dipercaya.
    Pemilihan kelas dilakukan eksplisit dari nama bobot, dengan fallback.
    """
    from ultralytics import RTDETR, YOLO
    nama = str(jalur).lower()
    if "rtdetr" in nama or "rt-detr" in nama:
        try:
            return RTDETR(jalur)
        except Exception as e:
            print(f"  RTDETR() gagal untuk {jalur} ({e}); fallback ke YOLO()")
    return YOLO(jalur)


def muat_classifier(jalur: str, dev: str):
    ck = torch.load(jalur, map_location="cpu")
    ca = ck["args"]
    m = Model(ca["backbone"], ca["mode"] == "rgbd", ca.get("gate_init", 0.1), ca["head"])
    m.load_state_dict(ck["model"]); m.to(dev).eval()
    return m, ca


@torch.no_grad()
def prediksi(models, xs, ds, tta: bool):
    """Rata-rata probabilitas lintas model dan lintas transformasi TTA.

    TTA memakai grup dihedral (4 rotasi x 2 flip) — sama persis dengan
    augmentasi geometrik saat training, jadi tidak memasukkan transformasi
    yang belum pernah dilihat model.
    """
    X, D = torch.stack(xs), torch.stack(ds)
    total = None
    varian = [(k, f) for k in range(4) for f in (False, True)] if tta else [(0, False)]
    for m, ca in models:
        for k, f in varian:
            xv, dv = X, D
            if k:
                xv, dv = torch.rot90(xv, k, (2, 3)), torch.rot90(dv, k, (2, 3))
            if f:
                xv, dv = torch.flip(xv, (3,)), torch.flip(dv, (3,))
            with torch.amp.autocast("cuda"):
                keluar, _ = m(xv, dv)
            p = prob_head(keluar, ca["head"]).float()
            total = p if total is None else total + p
    return (total / (len(models) * len(varian))).cpu().numpy()


def ke_tensor(rgb, dep, msk, dev, img=None):
    import torch.nn.functional as Fn
    S_in = img or IMG
    r = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    m = torch.from_numpy(msk[..., None]).permute(2, 0, 1).float() / 255.0
    d = torch.from_numpy(dep).permute(2, 0, 1).float() / 255.0
    r = Fn.interpolate(r[None], (S_in, S_in), mode="bilinear", align_corners=False)[0]
    m = Fn.interpolate(m[None], (S_in, S_in), mode="bilinear", align_corners=False)[0]
    d = Fn.interpolate(d[None], (S_in, S_in), mode="bilinear", align_corners=False)[0]
    r = (r - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / \
        torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return torch.cat([r, m * 2 - 1], 0).to(dev), ((d - 0.5) / 0.5).to(dev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detektor", nargs="+", required=True,
                    help="satu atau beberapa detektor; kalau >1 digabung dengan WBF")
    ap.add_argument("--wbf-iou", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--det-iou", type=float, default=0.7)
    ap.add_argument("--classifier", nargs="+", required=True,
                    help="satu atau beberapa bobot; kalau >1 probabilitasnya dirata-rata (ensemble)")
    ap.add_argument("--tta", action="store_true", help="rata-rata 8 transformasi dihedral")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--multi-kelas", action="store_true",
                    help="pancarkan 4 deteksi per box (skor = conf x P(kelas)) "
                         "alih-alih hanya kelas terbaik")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda"

    stems = [Path(l.strip()).stem for l in (SPLIT / f"{args.split}.txt").read_text().splitlines() if l.strip()]

    gt = {}
    for s in stems:
        g = []
        for ln in (D352 / "labels" / f"{s}.txt").read_text().splitlines():
            p = ln.split()
            if len(p) < 5 or int(p[0]) < 0:
                continue
            c = int(p[0]); cx, cy, w, h = (float(x) for x in p[1:5])
            g.append([c, (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
        gt[s] = np.array(g, float) if g else np.zeros((0, 5))

    models = [muat_classifier(j, dev) for j in args.classifier]
    cargs = models[0][1]
    pakai_depth = any(ca["mode"] == "rgbd" for _, ca in models)
    img_in = cargs.get("img", IMG)
    sisi_crop = max(S, img_in)      # jangan pernah memperbesar dari crop kecil
    print(f"classifier: {len(models)} model, tta={args.tta}, "
          f"mode={[ca['mode'] for _, ca in models]}, img={img_in}")

    # Deteksi seluruh detektor dulu, lalu digabung WBF -> satu himpunan kotak.
    per_det = {}
    for jd in args.detektor:
        det = muat_detektor(jd)
        pd_ = {}
        for i in range(0, len(stems), 8):
            blok = stems[i:i + 8]
            jalur = [str(D352 / "images" / f"{s}.jpg") for s in blok]
            hasil = det.predict(jalur, imgsz=args.imgsz, conf=args.conf,
                                iou=args.det_iou, max_det=100,
                                verbose=False, save=False)
            for s, r in zip(blok, hasil):
                b = r.boxes
                pd_[s] = (np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]], 1)
                          if len(b) else np.zeros((0, 5)))
        per_det[jd] = pd_
        print(f"  deteksi selesai: {jd}", flush=True)

    kotak_akhir = {}
    for s in stems:
        semua = np.concatenate([per_det[j][s] for j in args.detektor], 0)
        kotak_akhir[s] = (wbf(semua, args.wbf_iou, len(args.detektor))
                          if len(args.detektor) > 1 else semua)

    pred = {}
    for i in range(0, len(stems), 8):
        blok = stems[i:i + 8]
        for s in blok:
            kk = kotak_akhir[s]
            if len(kk) == 0:
                pred[s] = np.zeros((0, 6)); continue
            xyxy = kk[:, :4]; conf = kk[:, 4]
            bgr = cv2.imread(str(D352 / "images" / f"{s}.jpg"), cv2.IMREAD_COLOR)
            Z = None
            if pakai_depth and (DEPTH / f"{s}.png").exists():
                d = cv2.imread(str(DEPTH / f"{s}.png"), cv2.IMREAD_UNCHANGED)
                if d is not None:
                    Z = dekode_z(d)
            xs, ds, sah = [], [], []
            for j, bb in enumerate(xyxy):
                c = siapkan_crop(bgr, Z, bb, sisi_crop)
                if c is None:
                    continue
                t_x, t_d = ke_tensor(*c, dev, img_in)
                xs.append(t_x); ds.append(t_d); sah.append(j)
            if not xs:
                pred[s] = np.zeros((0, 6)); continue
            P = prediksi(models, xs, ds, args.tta)
            baris = []
            for n, j in enumerate(sah):
                if args.multi_kelas:
                    for c in range(K):
                        baris.append([*xyxy[j], conf[j] * P[n, c], c])
                else:
                    c = int(P[n].argmax())
                    baris.append([*xyxy[j], conf[j] * P[n, c], c])
            pred[s] = np.array(baris, float)
        if (i + 8) % 80 == 0:
            print(f"  {min(i+8, len(stems))}/{len(stems)} citra", flush=True)

    per = [ap50(gt, pred, c) for c in range(K)]
    agn = ap50(gt, {k: v[np.argsort(-v[:, 4])] for k, v in pred.items()}, None)
    hasil = {
        "detektor": args.detektor, "classifier": args.classifier, "tta": bool(args.tta),
        "split": args.split, "multi_kelas": bool(args.multi_kelas),
        "imgsz": args.imgsz, "det_iou": args.det_iou,
        "mAP50": float(np.mean(per)),
        "AP50_per_kelas": {f"B{i+1}": round(float(per[i]), 4) for i in range(4)},
        "AP50_class_agnostic": round(float(agn), 4),
        "n_gt": int(sum(len(g) for g in gt.values())),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(hasil, indent=2))
    print(json.dumps(hasil, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
