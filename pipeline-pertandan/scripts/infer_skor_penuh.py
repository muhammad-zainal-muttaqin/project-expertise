"""Inferensi detektor + dump VEKTOR SKOR 4-KELAS PENUH (bukan cuma top-1).

Kenapa perlu skrip sendiri: dump lama (`results/pred_sel5_953_rgb_test.npz`)
menyimpan `[x1,y1,x2,y2,skor,kelas]` — hanya kelas pemenang. Aturan agregasi
R2/R3/R4 di `docs/PROPOSAL.md` §5.4 butuh distribusi atas keempat kelas, jadi
dump itu tidak cukup.

Cara memperolehnya. YOLO26 memakai kepala end-to-end (bebas-NMS): keluaran
resminya hanya memuat kelas pemenang. Namun cabang `one2one` di keluaran
tambahan menyimpan `scores (1, 4, A)` — logit untuk keempat kelas di seluruh A
anchor. `sigmoid(logit)` persis sama dengan skor di keluaran resmi.

Kotak diambil dari `model.predict()`, **bukan** dari tensor mentah. Ini penting
dan sudah terbukti: memakai tensor mentah `(1,300,6)` apa adanya menghasilkan
mAP50 test **0,1342** — jauh di bawah 0,5436 yang tercatat — karena top-300 itu
memuat banyak baris duplikat yang, tanpa penyaringan, jadi positif palsu.
`predict()` adalah jalur yang memang menghasilkan angka resmi, jadi itu yang
dipakai.

Vektor kelas dipulihkan lewat **forward hook** pada model: hook menangkap
`one2one.scores` dari forward pass yang SAMA dengan yang dipakai `predict()`,
lalu tiap deteksi dicocokkan ke anchor-nya berdasarkan pasangan (kelas, skor).
Pasangan itu praktis unik untuk deteksi berkeyakinan wajar; yang ambigu
dilaporkan jumlahnya.

Satu jebakan lagi yang sudah memakan korban: seleksi YOLO26 dilakukan atas
matriks (anchor x kelas) yang diratakan, jadi **satu anchor bisa memancarkan
beberapa deteksi dengan kelas berbeda**, dan kelas yang dipancarkan itu belum
tentu argmax anchor-nya (41% deteksi, diperiksa langsung). Karena itu dump ini
menyimpan DUA-DUANYA: `conf`/`cls` resmi (dipakai untuk mAP, harus mereproduksi
0,5436) dan vektor 4-kelas (dipakai untuk agregasi). Kolom `anchor` ada supaya
hilirnya bisa menyatukan deteksi yang berasal dari anchor yang sama — satu
anchor = satu kotak = satu distribusi.

Keluaran: npz {stem: array (N, 11)} dengan kolom
    [x1, y1, x2, y2, conf, cls, p_B1, p_B2, p_B3, p_B4, anchor]
kotak dalam koordinat citra ASLI (960x1280), bukan koordinat letterbox.

Pemakaian:
    .venv/bin/python pipeline-pertandan/scripts/infer_skor_penuh.py --split val test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops

DS = Path("/workspace/SawitMVC-YOLO")
SUB = Path(__file__).resolve().parents[1]
REPO = SUB.parent
BOBOT = REPO / "models" / "yolo26l_e60_i1280_v2repro" / "best.pt"
IMGSZ = 1280
KELAS = ["B1", "B2", "B3", "B4"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="+", default=["val", "test"])
    ap.add_argument("--ds", default=str(DS), help="akar dataset")
    ap.add_argument("--daftar", default=None,
                    help="pola berkas daftar pohon per split, mis. "
                         "'splits/canonical_70_15_15/{split}_trees.txt'. Kalau "
                         "diberikan, citra dicari di <ds>/images/ datar")
    ap.add_argument("--tag", default="", help="akhiran nama berkas keluaran")
    ap.add_argument("--bobot", default=str(BOBOT))
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--conf-lapor", type=float, default=0.10,
                    help="ambang untuk melaporkan ambiguitas pemulihan vektor")
    args = ap.parse_args()

    m = YOLO(args.bobot)
    m.model.eval().cuda()

    tangkap = {}

    def hook(_mod, _inp, out):
        """Tangkap logit 4-kelas dari forward pass yang sama dengan predict()."""
        if isinstance(out, (list, tuple)) and len(out) > 1 and isinstance(out[1], dict):
            tangkap["P"] = torch.sigmoid(out[1]["one2one"]["scores"][0]).float().cpu().numpy()

    h = m.model.register_forward_hook(hook)

    ds = Path(args.ds)
    for split in args.split:
        if args.daftar:
            pohon = [x.strip() for x in
                     (ds / args.daftar.format(split=split)).read_text().splitlines()
                     if x.strip()]
            paths = sorted(q for t in pohon
                           for q in (ds / "images").glob(f"{t}_*.jpg"))
        else:
            paths = sorted(q for q in (ds / "images" / split).iterdir()
                           if q.suffix.lower() == ".jpg")
        print(f"{split}: {len(paths)} citra")
        out, n_amb, n_tinggi, t0 = {}, 0, 0, time.time()

        for i, p in enumerate(paths, 1):
            tangkap.clear()
            r = m.predict(str(p), imgsz=args.imgsz, conf=0.001, verbose=False)[0]
            b = r.boxes
            if b is None or len(b) == 0 or "P" not in tangkap:
                out[p.stem] = np.zeros((0, 11), np.float32)
                continue
            box = b.xyxy.cpu().numpy()          # sudah di koordinat citra ASLI
            cf = b.conf.cpu().numpy()
            cl = b.cls.cpu().numpy().astype(int)
            P = tangkap["P"]                     # (4, A)

            vec = np.zeros((len(box), 4), np.float32)
            anc = np.full(len(box), -1.0, np.float32)
            for j in range(len(box)):
                c, v = int(cl[j]), float(cf[j])
                idx = np.where(np.abs(P[c] - v) < 1e-6)[0]
                if len(idx) == 0:                # tak mungkin terjadi; jaga-jaga
                    vec[j, c] = v
                    continue
                if len(idx) > 1:
                    n_amb += v >= args.conf_lapor
                    idx = idx[np.argmax(P[:, idx].max(0))][None]
                vec[j] = P[:, idx[0]]
                anc[j] = idx[0]
                n_tinggi += v >= args.conf_lapor

            # kolom: x1 y1 x2 y2 | conf cls (resmi, untuk mAP) | p_B1..p_B4 | anchor
            out[p.stem] = np.concatenate(
                [box, cf[:, None], cl[:, None].astype(float), vec, anc[:, None]],
                1).astype(np.float32)

            if i % 100 == 0 or i == len(paths):
                print(f"  {split}: {i}/{len(paths)}  ({time.time()-t0:.0f}s)", flush=True)

        f = SUB / "results" / f"pred_skorpenuh{args.tag}_{split}.npz"
        np.savez_compressed(f, **out)
        print(f"{split}: {len(out)} citra -> {f}")
        print(f"  deteksi conf>={args.conf_lapor}: {n_tinggi}, "
              f"ambigu saat pemulihan vektor: {n_amb} "
              f"({100*n_amb/max(n_tinggi,1):.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
