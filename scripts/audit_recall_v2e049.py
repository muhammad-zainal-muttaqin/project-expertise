"""Audit recall detektor pada split uji SawitMVC (V2-E-049).

Menjalankan bobot detektor di seluruh citra uji satu per satu (hemat VRAM),
menyimpan dump prediksi `.npz` pada ambang keyakinan rendah, lalu menghitung
recall lokalisasi (IoU >= 0,5, sembarang kelas) pada ambang operasi 0,25.

Keluaran:
- `pred_v2e049_test_conf005.npz`: satu array per citra berisi
  [x1, y1, x2, y2, conf, cls] untuk seluruh prediksi conf >= 0,05.
- `v2e049_penyaringan_recall_depth.json`: ringkasan metrik (ditulis manual
  dari keluaran skrip ini, lihat EKSPERIMEN.md V2-E-049).

Contoh:
    .venv/bin/python scripts/audit_recall_v2e049.py \
        --dataset /workspace/SawitMVC \
        --bobot models/yolo26l_e60_i1280_v2repro/best.pt \
        --keluaran results/pred_v2e049_test_conf005.npz
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

LEBAR, TINGGI = 960, 1280  # resolusi kanonik SawitMVC (potret)


def muat_label(nama_citra: str, direktori_label: Path) -> np.ndarray:
    berkas = direktori_label / (Path(nama_citra).stem + ".txt")
    if not berkas.exists():
        return np.zeros((0, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            data = np.loadtxt(berkas, ndmin=2)
        except OSError:
            return np.zeros((0, 5))
    return data if data.size else np.zeros((0, 5))


def yolo_ke_xyxy(anotasi: np.ndarray) -> np.ndarray:
    kotak = []
    for baris in anotasi:
        if len(baris) < 5:
            continue
        cls, cx, cy, w, h = baris[:5]
        kotak.append([
            (cx - w / 2) * LEBAR, (cy - h / 2) * TINGGI,
            (cx + w / 2) * LEBAR, (cy + h / 2) * TINGGI, int(cls),
        ])
    return np.array(kotak, dtype=float) if kotak else np.zeros((0, 5))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iris = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    gabung = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iris
    return iris / gabung if gabung > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--bobot", required=True)
    parser.add_argument("--keluaran", required=True)
    parser.add_argument("--ambang_simpan", type=float, default=0.05)
    parser.add_argument("--ambang_operasi", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    citra_uji = [b.strip() for b in open(dataset / "test.txt") if b.strip()]
    print(f"citra uji: {len(citra_uji)}", flush=True)

    model = YOLO(args.bobot)
    prediksi: dict = {}
    mulai = time.time()
    for i, relatif in enumerate(citra_uji):
        hasil = model.predict(
            source=str(dataset / relatif), imgsz=args.imgsz,
            conf=args.ambang_simpan, iou=args.iou, device=0, verbose=False,
        )[0]
        kotak = hasil.boxes
        nama = Path(relatif).name
        if kotak is not None and len(kotak):
            prediksi[nama] = dict(
                xyxy=kotak.xyxy.cpu().numpy().astype(np.float32),
                conf=kotak.conf.cpu().numpy().astype(np.float32),
                cls=kotak.cls.cpu().numpy().astype(np.int16),
            )
        else:
            prediksi[nama] = dict(
                xyxy=np.zeros((0, 4), np.float32),
                conf=np.zeros((0,), np.float32),
                cls=np.zeros((0,), np.int16),
            )
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(citra_uji)}", flush=True)
            torch.cuda.empty_cache()
    print(f"inferensi {len(prediksi)} citra dalam {time.time() - mulai:.1f} detik", flush=True)

    np.savez_compressed(
        args.keluaran,
        **{k.replace(".jpg", ""): (
            np.concatenate([v["xyxy"], v["conf"][:, None],
                            v["cls"][:, None].astype(np.float32)], axis=1)
            if len(v["conf"]) else np.zeros((0, 6), np.float32))
            for k, v in prediksi.items()},
    )
    print(f"tersimpan: {args.keluaran}")

    # Ringkasan recall pada ambang operasi (diagnostik layar, bukan klaim).
    n_acuan = n_kena = 0
    luas_kena, luas_lolos = [], []
    kecil_lolos = b4_lolos = 0
    recall_citra = []
    for relatif in citra_uji:
        nama = Path(relatif).name
        acuan = yolo_ke_xyxy(muat_label(nama, dataset / "labels"))
        det = prediksi[nama]
        lolos = det["conf"] >= args.ambang_operasi
        dete = det["xyxy"][lolos]
        kena = 0
        for g in acuan:
            n_acuan += 1
            luas = (g[2] - g[0]) * (g[3] - g[1])
            terbaik = max([iou(g[:4], d) for d in dete], default=0.0)
            if terbaik >= 0.5:
                n_kena += 1
                kena += 1
                luas_kena.append(luas)
            else:
                luas_lolos.append(luas)
                if luas < 6400:
                    kecil_lolos += 1
                if int(g[4]) == 3:
                    b4_lolos += 1
        recall_citra.append(kena / max(1, len(acuan)))
    n_lolos = n_acuan - n_kena
    print(f"acuan:{n_acuan} kena:{n_kena} lolos:{n_lolos}")
    print(f"median luas kena:{np.median(luas_kena):.0f} lolos:{np.median(luas_lolos):.0f}")
    print(f"rerata recall per citra:{np.mean(recall_citra) * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
