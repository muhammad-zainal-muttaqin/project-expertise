"""Dump prediksi detektor CLASS-AWARE (4 kelas) ke format yang bisa difusikan.

Detektor Fase 1-5 memprediksi kelas sendiri end-to-end. Detektor Fase 6
class-agnostic + classifier crop memprediksi kelas lewat jalur berbeda.
Galat keduanya tidak berkorelasi penuh, jadi bisa difusikan — dan `edge`
adalah RGB+D 4-kanal, sehingga fusi ini jalur pertama di proyek ini yang
membuat DEPTH ikut menyumbang ke mAP50 utama.

Keluaran: npz {stem: array Nx6 = [x1,y1,x2,y2,skor,kelas]}, identik dengan
dump `eval_twostage.py`, supaya `fuse_final.py` bisa memuat keduanya.

Usage:
    .venv/bin/python scripts/dump_classaware.py \
        --bobot runs/yolo26l_e60_i1280_rgbd352_edge/weights/best.pt \
        --data /workspace/SawitMVC-Depth-4ch-edge-YOLO \
        --split test --out results/pred_edge_test.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_twostage import ap50, muat_detektor  # noqa: E402

D352 = Path("/workspace/SawitMVC-Depth")
SPLIT = D352 / "splits" / "canonical_70_15_15"
W, H = 1280, 800
K = 4


def muat_gt(split: str, daftar: str | None = None,
            labels_dir: str | None = None) -> dict:
    """GT default dari split kanonik 352 (1280x800).

    `daftar` + `labels_dir` dipakai untuk dataset lain (mis. test-953 bersih,
    yang resolusinya 960x1280) — ukuran citra dibaca per berkas, tidak
    diasumsikan.
    """
    if daftar:
        jalur = [Path(l.strip()) for l in Path(daftar).read_text().splitlines() if l.strip()]
        ldir = Path(labels_dir)
        gt, stems = {}, []
        for p in jalur:
            s = p.stem; stems.append(s)
            im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if im is None:
                gt[s] = np.zeros((0, 5)); continue
            hh, ww = im.shape[:2]
            g = []
            lf = ldir / f"{s}.txt"
            if lf.is_file():
                for ln in lf.read_text().splitlines():
                    q = ln.split()
                    if len(q) < 5 or int(q[0]) < 0:
                        continue
                    c = int(q[0]); cx, cy, w, h = (float(x) for x in q[1:5])
                    g.append([c, (cx - w / 2) * ww, (cy - h / 2) * hh,
                              (cx + w / 2) * ww, (cy + h / 2) * hh])
            gt[s] = np.array(g, float) if g else np.zeros((0, 5))
        return stems, gt

    stems = [Path(l.strip()).stem
             for l in (SPLIT / f"{split}.txt").read_text().splitlines() if l.strip()]
    gt = {}
    for s in stems:
        g = []
        for ln in (D352 / "labels" / f"{s}.txt").read_text().splitlines():
            p = ln.split()
            if len(p) < 5 or int(p[0]) < 0:
                continue
            c = int(p[0]); cx, cy, w, h = (float(x) for x in p[1:5])
            g.append([c, (cx - w / 2) * W, (cy - h / 2) * H,
                      (cx + w / 2) * W, (cy + h / 2) * H])
        gt[s] = np.array(g, float) if g else np.zeros((0, 5))
    return stems, gt


def cari_citra(root: Path, split: str, stem: str) -> Path | None:
    """Dataset 4-kanal memakai .tiff, dataset RGB memakai .jpg."""
    for sub in (root / split / "images", root / "images", root):
        if not sub.is_dir():
            continue
        for ext in (".tiff", ".tif", ".jpg", ".png"):
            p = sub / f"{stem}{ext}"
            if p.is_file():
                return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bobot", required=True)
    ap.add_argument("--data", required=True,
                    help="root dataset ({split}/images) — 4ch TIFF atau RGB jpg")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.005)
    ap.add_argument("--det-iou", type=float, default=0.7)
    ap.add_argument("--agnostik", action="store_true",
                    help="detektor 1 kelas: laporkan AP50 lokalisasi murni, "
                         "bukan rata-rata makro 4 kelas")
    ap.add_argument("--daftar", help="txt berisi jalur citra (ganti split kanonik 352)")
    ap.add_argument("--labels-dir", help="direktori label untuk --daftar")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stems, gt = muat_gt(args.split, args.daftar, args.labels_dir)
    root = Path(args.data)

    if args.daftar:
        peta = {Path(l.strip()).stem: Path(l.strip())
                for l in Path(args.daftar).read_text().splitlines() if l.strip()}
        cari = lambda r, sp, s: peta.get(s)          # noqa: E731
    else:
        cari = cari_citra

    hilang = [s for s in stems if cari(root, args.split, s) is None]
    if hilang:
        print(f"FATAL: {len(hilang)} citra tidak ditemukan di {root}, contoh: {hilang[:3]}")
        return 1

    det = muat_detektor(args.bobot)
    pred = {}
    for i in range(0, len(stems), 8):
        blok = stems[i:i + 8]
        jalur = [cari(root, args.split, s) for s in blok]
        if jalur[0].suffix.lower() in (".tiff", ".tif"):
            src = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in jalur]
        else:
            src = [str(p) for p in jalur]
        hasil = det.predict(src, imgsz=args.imgsz, conf=args.conf,
                            iou=args.det_iou, max_det=100, verbose=False, save=False)
        for s, r in zip(blok, hasil):
            b = r.boxes
            if b is None or len(b) == 0:
                pred[s] = np.zeros((0, 6)); continue
            pred[s] = np.concatenate([
                b.xyxy.cpu().numpy(),
                b.conf.cpu().numpy()[:, None],
                b.cls.cpu().numpy()[:, None],
            ], 1).astype(float)
        if (i + 8) % 80 == 0:
            print(f"  {min(i + 8, len(stems))}/{len(stems)} citra", flush=True)

    nama_run = Path(args.bobot).parts[-3]
    if args.agnostik:
        # kelas=None -> AP50 lokalisasi murni; prediksi harus urut skor menurun
        urut = {k: (v[np.argsort(-v[:, 4])] if len(v) else v) for k, v in pred.items()}
        a = ap50(gt, urut, None)
        print(f"{nama_run} split={args.split} AP50_agnostik={a:.4f} "
              f"(n_pred={sum(len(v) for v in pred.values())}, "
              f"n_gt={sum(len(g) for g in gt.values())})")
    else:
        per = [ap50(gt, pred, c) for c in range(K)]
        print(f"{nama_run} split={args.split} "
              f"mAP50={np.mean(per):.4f} per_kelas="
              f"{ {f'B{i+1}': round(float(per[i]), 4) for i in range(K)} }")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out), **pred)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
