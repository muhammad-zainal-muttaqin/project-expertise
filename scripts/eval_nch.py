"""Evaluasi pycocotools untuk sel matriks monocular-depth + dump prediksi .npz.

Protokolnya SENGAJA dibuat identik dengan evaluator yang menghasilkan angka sel
pembanding, bukan evaluator baru:
  - GT dibangun dari label YOLO, kategori = kelas+1, bbox xywh piksel
    (sama persis dengan `eval_all_pycoco_v2repro.py` dan `eval_pycoco_352.py`).
  - Prediksi ultralytics dengan `conf=0.001`, imgsz sama dengan saat latih.
  - mAP50 = COCOeval.stats[1], mAP50-95 = stats[0].

Dua tata letak direktori dipakai proyek ini dan keduanya didukung lewat
`--tata-letak`, karena sel pembandingnya memang dievaluasi begitu:
  images_split : ds_root/images/{split}   <- SawitMVC-YOLO (sel 5)
  split_images : ds_root/{split}/images   <- rak 352 (sel 1, 2)

TIFF WAJIB dibaca dengan IMREAD_UNCHANGED lalu dioper sebagai array. Kalau
jalurnya yang dioper, ultralytics memuatnya sebagai 3 kanal dan kanal
depth/mono hilang DIAM-DIAM — tidak ada error, hanya angka yang salah. Ini
mengikuti `eval_pycoco_rgbd352.py`.

Dump `.npz` ditulis PADA SAAT evaluasi (aturan repo): kunci = stem citra,
nilai = array (N,6) [x1,y1,x2,y2,skor,kelas], format sama dengan
`results/pred_*.npz` yang sudah ada sehingga langsung bisa dipakai
`bootstrap_map.py`.

Usage:
    # gerbang: reproduksi angka sel 5 sebelum apa pun dibandingkan dengannya
    .venv/bin/python scripts/eval_nch.py \
        --bobot models/yolo26l_e60_i1280_v2repro/best.pt \
        --ds-root /workspace/SawitMVC-YOLO --tata-letak images_split \
        --split test --nama sel5_953_rgb --harap-map50 0.5435

    # sel baru
    .venv/bin/python scripts/eval_nch.py \
        --bobot runs/sel6_953_rgbmono/weights/best.pt \
        --ds-root /workspace/d953_rgbmono --tata-letak images_split \
        --split test --nama sel6_953_rgbmono
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.utils.patches import imread as ul_imread

KELAS = ["B1", "B2", "B3", "B4"]
CITRA_EKS = (".jpg", ".jpeg", ".png", ".tiff", ".tif")


def dir_split(akar: Path, split: str, tata: str) -> tuple[Path, Path]:
    if tata == "images_split":
        return akar / "images" / split, akar / "labels" / split
    return akar / split / "images", akar / split / "labels"


def bangun_gt(akar: Path, split: str, tata: str):
    idir, ldir = dir_split(akar, split, tata)
    if not idir.is_dir():
        sys.exit(f"FATAL: {idir} tidak ada — tata letak salah?")
    paths = sorted(p for p in idir.iterdir() if p.suffix.lower() in CITRA_EKS)
    if not paths:
        sys.exit(f"FATAL: tidak ada citra di {idir}")

    images, anns, ann_id = [], [], 1
    for img_id, p in enumerate(paths, 1):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None:
            sys.exit(f"FATAL: gagal membaca {p}")
        h, w = im.shape[:2]
        images.append({"id": img_id, "file_name": p.name, "width": w, "height": h})
        lf = ldir / f"{p.stem}.txt"
        if lf.is_file():
            for ln in lf.read_text().splitlines():
                q = ln.split()
                if len(q) < 5:
                    continue
                c, cx, cy, bw, bh = (float(x) for x in q[:5])
                if c < 0:
                    continue
                x, y = (cx - bw / 2) * w, (cy - bh / 2) * h
                aw, ah = bw * w, bh * h
                anns.append({"id": ann_id, "image_id": img_id, "category_id": int(c) + 1,
                             "bbox": [x, y, aw, ah], "area": aw * ah, "iscrowd": 0})
                ann_id += 1
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns,
                  "categories": [{"id": i + 1, "name": n} for i, n in enumerate(KELAS)]}
    gt.createIndex()
    print(f"{split}: {len(paths)} citra, {len(anns)} kotak GT")
    return gt, paths


def prediksi(model, paths, imgsz):
    dt, dump = [], {}
    for i, p in enumerate(paths, 1):
        if p.suffix.lower() in (".tiff", ".tif"):
            # Array, bukan jalur: kalau jalurnya yang dioper ke predict(), kanal
            # ke-4/5 hilang tanpa peringatan apa pun.
            #
            # Dan pembacanya WAJIB `ultralytics.utils.patches.imread`, bukan
            # cv2.imread — bahkan dengan IMREAD_UNCHANGED, cv2 hanya
            # mengembalikan HALAMAN PERTAMA dari TIFF multi-page. Sel 4 (5
            # kanal) disimpan sebagai 5 halaman satu-kanal karena cv2.imwrite
            # menolak menulis 5 kanal sekaligus, jadi cv2.imread memberi
            # (H, W) grayscale, bukan (H, W, 5). Diverifikasi 2026-08-14:
            # cv2 -> (800, 1280), ultralytics -> (800, 1280, 5).
            # Memakai fungsi yang sama dengan yang dipakai dataloader training
            # juga menjamin eval membaca citra persis seperti saat dilatih.
            im = ul_imread(str(p))
            if im is None:
                sys.exit(f"FATAL: gagal membaca {p}")
            r = model.predict(source=im, imgsz=imgsz, conf=0.001, verbose=False)[0]
        else:
            r = model.predict(str(p), imgsz=imgsz, conf=0.001, verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            dump[p.stem] = np.zeros((0, 6))
            continue
        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy()
        dump[p.stem] = np.column_stack([xyxy, conf, cls]).astype(np.float64)
        for k in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[k]
            dt.append({"image_id": i, "category_id": int(cls[k]) + 1,
                       "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                       "score": float(conf[k])})
        if i % 100 == 0:
            print(f"  {i}/{len(paths)}")
    return dt, dump


def per_kelas(ev):
    p = ev.eval["precision"]  # [T,R,K,A,M]
    ap50, ap95 = {}, {}
    for k, n in enumerate(KELAS):
        s95, s50 = p[:, :, k, 0, 2], p[0, :, k, 0, 2]
        ap95[n] = round(float(s95[s95 > -1].mean()) if (s95 > -1).any() else 0.0, 4)
        ap50[n] = round(float(s50[s50 > -1].mean()) if (s50 > -1).any() else 0.0, 4)
    return ap50, ap95


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bobot", required=True)
    ap.add_argument("--ds-root", required=True)
    ap.add_argument("--tata-letak", choices=["images_split", "split_images"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--nama", required=True, help="dipakai untuk nama berkas keluaran")
    ap.add_argument("--harap-map50", type=float, default=None,
                    help="gerbang reproduksi: gagal keras kalau selisihnya > --toleransi")
    ap.add_argument("--toleransi", type=float, default=0.002)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    if not Path(args.bobot).exists():
        sys.exit(f"FATAL: bobot {args.bobot} tidak ada")
    from ultralytics import YOLO

    gt, paths = bangun_gt(Path(args.ds_root), args.split, args.tata_letak)
    model = YOLO(args.bobot)
    dt, dump = prediksi(model, paths, args.imgsz)
    if not dt:
        sys.exit("FATAL: nol deteksi — hampir pasti salah kanal atau salah bobot")

    ev = COCOeval(gt, gt.loadRes(dt), "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    map95, map50 = float(ev.stats[0]), float(ev.stats[1])
    ap50, ap95 = per_kelas(ev)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz = out / f"pred_{args.nama}_{args.split}.npz"
    np.savez_compressed(npz, **dump)

    hasil = {"nama": args.nama, "bobot": args.bobot, "ds_root": args.ds_root,
             "split": args.split, "imgsz": args.imgsz, "evaluator": "pycocotools",
             "n_citra": len(paths), "n_kotak_gt": len(gt.dataset["annotations"]),
             "mAP50": round(map50, 4), "mAP50_95": round(map95, 4),
             "per_kelas_AP50": ap50, "per_kelas_AP50_95": ap95,
             "dump_prediksi": str(npz)}
    (out / f"eval_{args.nama}_{args.split}.json").write_text(json.dumps(hasil, indent=2))
    print(json.dumps({k: hasil[k] for k in
                      ("nama", "split", "n_citra", "n_kotak_gt", "mAP50", "mAP50_95",
                       "per_kelas_AP50")}, indent=2))
    print(f"prediksi -> {npz}")

    if args.harap_map50 is not None:
        d = abs(map50 - args.harap_map50)
        if d > args.toleransi:
            sys.exit(f"FATAL: gerbang reproduksi GAGAL — mAP50 {map50:.4f} vs harapan "
                     f"{args.harap_map50:.4f} (selisih {d:.4f} > toleransi {args.toleransi}). "
                     "Split/protokol/bobot tidak cocok. JANGAN bandingkan angka apa pun "
                     "dengan sel ini sampai sebabnya ketemu.")
        print(f"gerbang reproduksi LULUS: {map50:.4f} vs {args.harap_map50:.4f} "
              f"(selisih {d:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
