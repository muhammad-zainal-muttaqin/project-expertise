"""Uji penyaring depth monokular untuk kandidat low-conf (V2-E-049).

Pertanyaan: apakah tepi/variansi peta depth artifisial dapat membedakan
kandidat low-conf yang benar (IoU >= 0,5 terhadap acuan) dari alarm palsu
(IoU < 0,2)? Jika AUC ≈ 0,5, depth-edge ditolak sebagai penyeimbang ulang
dan run depth generatif yang berat tidak perlu dijalankan.

Model: `depth-anything/Depth-Anything-V2-Small-hf` sebagai proksi murah
sebelum Marigold V2. Himpunan uji: 150 citra uji dengan miss terbanyak
menurut dump V2-E-049 agar penyaringan fokus pada kasus sulit.

Contoh:
    .venv/bin/python scripts/probe_depth_v2e049.py \
        --dataset /workspace/SawitMVC \
        --dump results/pred_v2e049_test_conf005.npz \
        --keluaran results/v2e049_penyaringan_recall_depth.json
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
LEBAR, TINGGI = 960, 1280


class CitraDS(Dataset):
    def __init__(self, daftar: list[str]):
        self.daftar = daftar

    def __len__(self) -> int:
        return len(self.daftar)

    def __getitem__(self, i: int):
        jalur = self.daftar[i]
        return jalur, Image.open(jalur).convert("RGB")


def muat_label(nama: str, direktori: Path) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            data = np.loadtxt(direktori / (Path(nama).stem + ".txt"), ndmin=2)
        except OSError:
            return np.zeros((0, 5))
    return data if data.size else np.zeros((0, 5))


def ke_xyxy(anotasi: np.ndarray) -> np.ndarray:
    kotak = []
    for baris in anotasi:
        if len(baris) < 5:
            continue
        _, cx, cy, w, h = baris[:5]
        kotak.append([(cx - w / 2) * LEBAR, (cy - h / 2) * TINGGI,
                      (cx + w / 2) * LEBAR, (cy + h / 2) * TINGGI])
    return np.array(kotak, dtype=float) if kotak else np.zeros((0, 4))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iris = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    gabung = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iris
    return iris / gabung if gabung > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--keluaran", required=True)
    parser.add_argument("--n_sulit", type=int, default=150)
    parser.add_argument("--tumpuk", type=int, default=16)
    args = parser.parse_args()

    import json

    dataset = Path(args.dataset)
    pred = np.load(args.dump)
    citra_uji = [b.strip() for b in open(dataset / "test.txt") if b.strip()]

    # Pilih citra tersulit agar uji hemat tetapi tajam.
    jumlah_lolos: dict[str, int] = {}
    for relatif in citra_uji:
        nama = Path(relatif).name
        kunci = nama.replace(".jpg", "")
        acuan = ke_xyxy(muat_label(nama, dataset / "labels"))
        cacah = 0
        for g in acuan:
            if max([iou(g, p[:4]) for p in pred[kunci] if p[4] >= 0.25], default=0.0) < 0.5:
                cacah += 1
        jumlah_lolos[nama] = cacah
    sulit = sorted(jumlah_lolos, key=jumlah_lolos.get, reverse=True)[: args.n_sulit]
    print(f"subset sulit: {len(sulit)} citra", flush=True)

    prosesor = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(
        MODEL, dtype=torch.float16).to("cuda").eval()

    def gabung(tumpukan):
        nama = [b[0] for b in tumpukan]
        citra = [b[1] for b in tumpukan]
        enk = prosesor(images=citra, return_tensors="pt")
        return nama, enk["pixel_values"]

    peta: dict[str, np.ndarray] = {}
    pemuat = DataLoader(CitraDS([str(dataset / "images" / h) for h in sulit]),
                        batch_size=args.tumpuk, shuffle=False,
                        num_workers=12, collate_fn=gabung, pin_memory=True)
    mulai = time.time()
    with torch.inference_mode():
        for nama_berkas, piksel in pemuat:
            piksel = piksel.to("cuda", non_blocking=True)
            keluar = model(pixel_values=piksel).predicted_depth
            for nm, d in zip(nama_berkas, keluar):
                peta[Path(nm).stem] = d.float().cpu().numpy()
    print(f"{len(peta)} peta depth dalam {time.time() - mulai:.1f} detik", flush=True)

    benar, palsu = [], []
    for nama in sulit:
        kunci = nama.replace(".jpg", "")
        d = peta[kunci].astype(np.float32)
        d = (d - d.min()) / (d.max() - d.min() + 1e-6)
        tinggi, lebar = d.shape
        sx, sy = lebar / LEBAR, tinggi / TINGGI
        gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
        tepi = np.sqrt(gx ** 2 + gy ** 2)
        acuan = ke_xyxy(muat_label(nama, dataset / "labels"))
        for p in pred[kunci]:
            if not 0.05 <= p[4] < 0.25:
                continue
            x1, y1, x2, y2 = (p[:4] * np.array([sx, sy, sx, sy])).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(lebar - 1, x2), min(tinggi - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            terbaik = max([iou(p[:4], g) for g in acuan], default=0.0)
            ciri = (float(tepi[y1:y2, x1:x2].mean()), float(d[y1:y2, x1:x2].std()))
            if terbaik >= 0.5:
                benar.append(ciri)
            elif terbaik < 0.2:
                palsu.append(ciri)

    benar = np.array(benar)
    palsu = np.array(palsu)
    label = np.array([1] * len(benar) + [0] * len(palsu))
    auc_tepi = float(roc_auc_score(label, np.concatenate([benar[:, 0], palsu[:, 0]])))
    auc_var = float(roc_auc_score(label, np.concatenate([benar[:, 1], palsu[:, 1]])))
    print(f"benar:{len(benar)} palsu:{len(palsu)} AUC tepi:{auc_tepi:.3f} var:{auc_var:.3f}")

    json.dump(dict(
        model_proksi=MODEL, n_sulit=len(sulit),
        n_benar=int(len(benar)), n_palsu=int(len(palsu)),
        auc_tepi=auc_tepi, auc_varians=auc_var,
        rerata_tepi_benar=float(benar[:, 0].mean()),
        rerata_tepi_palsu=float(palsu[:, 0].mean()),
    ), open(args.keluaran, "w"), indent=2)
    print(f"tersimpan: {args.keluaran}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
