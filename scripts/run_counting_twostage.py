"""Counting Ridge+F_all untuk pipeline dua-tahap — Fase 6.

Tahap 1 (detektor class-agnostic) mencari tandan, tahap 2 (classifier crop)
memberi kematangan; hasilnya ditulis dalam skema per-pohon yang SAMA dengan
Fase 1-5, lalu dihitung dengan fungsi counting yang SAMA
(`extract_all_features`, `load_pertree_dataset`, `score` diimpor langsung dari
`run_counting_rgbd352.py`). Ini disengaja: kalau counting-nya diimplementasi
ulang, angkanya tidak lagi sebanding dengan V2-E-002/004/007/011.

Usage:
    .venv/bin/python scripts/run_counting_twostage.py \
        --detektor runs/agn352_ft/weights/best.pt \
        --classifier runs_fase6/sd202_rgb/best.pt \
        --label TwoStage-RGB --out results/counting_twostage.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from build_crop_dataset import dekode_z  # noqa: E402
from eval_twostage import (S, ke_tensor, muat_classifier,  # noqa: E402
                           muat_detektor, prediksi, siapkan_crop, wbf)
from run_counting_rgbd352 import (CLASSES, load_gt, load_pertree_dataset,  # noqa: E402
                                  score)


D352 = Path("/workspace/SawitMVC-Depth")
DEPTH = Path("/workspace/depth_png_352")
SPLIT = D352 / "splits" / "canonical_70_15_15"


def split_per_pohon() -> dict[str, str]:
    out = {}
    for sp in ("train", "val", "test"):
        for t in (SPLIT / f"{sp}_trees.txt").read_text().split():
            if t.strip():
                out[t.strip()] = sp
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detektor", nargs="+", required=True,
                    help="satu atau beberapa detektor; kalau >1 digabung dengan WBF")
    ap.add_argument("--classifier", nargs="+", required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--label", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--wbf-iou", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--det-iou", type=float, default=0.7)
    ap.add_argument("--out", default="results/counting_twostage.json")
    args = ap.parse_args()
    dev = "cuda"
    proj = Path(__file__).resolve().parent.parent

    models = [muat_classifier(j, dev) for j in args.classifier]
    ca = models[0][1]
    pakai_depth = any(c["mode"] == "rgbd" for _, c in models)
    img_in = ca.get("img") or 160
    sisi_crop = max(S, img_in)
    print(f"classifier: {len(models)} model, tta={args.tta}")

    tree_splits = split_per_pohon()
    semua = sorted(D352.glob("images/*.jpg"))
    per_pohon: dict[str, list[Path]] = {}
    for p in semua:
        per_pohon.setdefault(p.stem.rsplit("_", 1)[0], []).append(p)

    out_dir = proj / "runs" / "pertree_twostage" / args.label.lower().replace("-", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"pohon={len(per_pohon)}  citra={len(semua)}  -> {out_dir}")

    # Deteksi di-BATCH lebih dulu. Versi awal memanggil det.predict() sekali per
    # citra dan memproyeksikan ~4 jam untuk 1.408 citra — yang dominan overhead
    # per-panggilan, bukan komputasinya.
    B = 8
    per_det = {}
    for jd in args.detektor:
        det = muat_detektor(jd)
        pd_ = {}
        for i in range(0, len(semua), B):
            blok = semua[i:i + B]
            hasil = det.predict([str(x) for x in blok], imgsz=args.imgsz,
                                conf=args.conf, iou=args.det_iou, max_det=100,
                                verbose=False, save=False)
            for x, r in zip(blok, hasil):
                b = r.boxes
                pd_[x.stem] = (np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]], 1)
                               if len(b) else np.zeros((0, 5)))
            if (i + B) % 400 == 0:
                print(f"  deteksi {min(i + B, len(semua))}/{len(semua)} citra", flush=True)
        per_det[jd] = pd_
        print(f"  detektor selesai: {jd}", flush=True)

    kotak = {}
    for x in semua:
        gab = np.concatenate([per_det[j][x.stem] for j in args.detektor], 0)
        f = wbf(gab, args.wbf_iou, len(args.detektor)) if len(args.detektor) > 1 else gab
        kotak[x.stem] = ((f[:, :4], f[:, 4]) if len(f) else (np.zeros((0, 4)), np.zeros(0)))

    n = 0
    for tid, jalur in sorted(per_pohon.items()):
        citra = {}
        for p in sorted(jalur):
            xyxy, conf = kotak[p.stem]
            anot = []
            if len(xyxy):
                bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                H, W = bgr.shape[:2]
                Z = None
                if pakai_depth and (DEPTH / f"{p.stem}.png").exists():
                    d = cv2.imread(str(DEPTH / f"{p.stem}.png"), cv2.IMREAD_UNCHANGED)
                    if d is not None:
                        Z = dekode_z(d)
                xs, ds, sah = [], [], []
                for j, bb in enumerate(xyxy):
                    c = siapkan_crop(bgr, Z, bb, sisi_crop)
                    if c is None:
                        continue
                    tx, td = ke_tensor(*c, dev, img_in)
                    xs.append(tx); ds.append(td); sah.append(j)
                if xs:
                    P = prediksi(models, xs, ds, args.tta)
                    for k, j in enumerate(sah):
                        x0, y0, x1, y1 = xyxy[j]
                        anot.append({
                            "class_name": CLASSES[int(P[k].argmax())],
                            "bbox_yolo": [float((x0 + x1) / 2 / W), float((y0 + y1) / 2 / H),
                                          float((x1 - x0) / W), float((y1 - y0) / H)],
                            "conf": float(conf[j] * P[k].max()),
                        })
            sisi = p.stem.rsplit("_", 1)[1]
            citra[f"side_{sisi}"] = {"side_index": int(sisi) - 1, "annotations": anot}
        (out_dir / f"{tid}.json").write_text(json.dumps({
            "tree_name": tid, "split": tree_splits.get(tid, "train"),
            "detector": "twostage", "images": citra}, indent=2))
        n += 1
        if n % 50 == 0:
            print(f"  {n}/{len(per_pohon)} pohon", flush=True)

    gt_map = load_gt(D352 / "json")
    df, y, tree_ids, splits = load_pertree_dataset(out_dir, gt_map)
    tr, va, te = splits == "train", splits == "val", splits == "test"
    X = df.values
    model = Pipeline([("sc", StandardScaler()),
                      ("rid", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500]))])
    model.fit(X[tr | va], y[tr | va])
    hasil = score(y[te], model.predict(X[te]))
    hasil.update({"label": args.label, "detektor": args.detektor,
                  "classifier": args.classifier, "tta": bool(args.tta),
                  "n_pohon_test": int(te.sum())})
    print(json.dumps({k: v for k, v in hasil.items()
                      if k in ("label", "class_acc", "tree_acc", "macro_mae", "n_pohon_test")}, indent=2))

    p_out = proj / args.out
    lama = json.loads(p_out.read_text()) if p_out.exists() else {}
    lama[args.label] = hasil                       # merge, jangan timpa isi lama
    p_out.write_text(json.dumps(lama, indent=2))
    print(f"ditulis {p_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
