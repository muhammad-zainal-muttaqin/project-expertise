"""Inferensi detektor korpus 953 pada citra korpus lain (V2-E-050f, V2-E-050g).

Sasaran baku adalah partisi uji korpus 763. Opsi `--daftar-citra` menerima
berkas teks berisi satu lintasan citra per baris, dipakai V2-E-050g untuk 200
citra DEPTH pada partisi uji 1716 yang berasal dari latih dan validasi 763.

Menghasilkan dump prediksi dengan format yang sama seperti `results/cross_eval`,
yakni satu kunci per citra berisi larik (N, 6) bertata letak
[x1, y1, x2, y2, skor, kelas] pada ruang piksel citra asli.

Protokol mengikuti `V2-E-001`: imgsz 1280, conf 0,001, iou 0,7, max_det 300.

Pemakaian:
    python scripts/inferensi_953_ke_763.py --detektor yolo26l
    python scripts/inferensi_953_ke_763.py --detektor yolo26l --daftar-citra daftar.txt         --keluaran results/cross_eval/predictions/v2repro953_yolo26l__on_1716_depth50__test.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

IMGSZ = 1280
CONF = 0.001
IOU = 0.7
MAX_DET = 300


def inferensi_ultralytics(bobot: Path, citra: list[Path], perangkat: str) -> dict[str, np.ndarray]:
    from ultralytics import RTDETR, YOLO

    model = RTDETR(str(bobot)) if "rtdetr" in bobot.as_posix().lower() else YOLO(str(bobot))
    keluaran = {}
    mulai = time.time()
    for i, p in enumerate(citra, 1):
        hasil = model.predict(
            str(p), imgsz=IMGSZ, conf=CONF, iou=IOU, max_det=MAX_DET,
            device=perangkat, verbose=False,
        )[0]
        kotak = hasil.boxes
        if kotak is None or len(kotak) == 0:
            keluaran[p.stem] = np.zeros((0, 6), dtype=np.float32)
            continue
        keluaran[p.stem] = np.concatenate(
            [
                kotak.xyxy.cpu().numpy(),
                kotak.conf.cpu().numpy()[:, None],
                kotak.cls.cpu().numpy()[:, None],
            ],
            axis=1,
        ).astype(np.float32)
        if i % 50 == 0:
            laju = (time.time() - mulai) / i
            print(f"  {i}/{len(citra)} citra, {laju:.2f} detik per citra", flush=True)
    return keluaran


def inferensi_rfdetr(bobot: Path, citra: list[Path], perangkat: str) -> dict[str, np.ndarray]:
    """Mengikuti pola `scripts/eval_rfdetr_v2repro_953_test.py`, prediksi per kelompok 8 citra."""
    from rfdetr import RFDETRLarge

    model = RFDETRLarge(pretrain_weights=str(bobot), resolution=IMGSZ)
    keluaran = {}
    mulai = time.time()
    for awal in range(0, len(citra), 8):
        kelompok = citra[awal:awal + 8]
        hasil = model.predict([str(p) for p in kelompok], threshold=CONF)
        if not isinstance(hasil, list):
            hasil = [hasil]
        for p, det in zip(kelompok, hasil):
            if det.xyxy is None or len(det.xyxy) == 0:
                keluaran[p.stem] = np.zeros((0, 6), dtype=np.float32)
                continue
            keluaran[p.stem] = np.concatenate(
                [
                    np.asarray(det.xyxy, dtype=np.float32),
                    np.asarray(det.confidence, dtype=np.float32)[:, None],
                    np.asarray(det.class_id, dtype=np.float32)[:, None],
                ],
                axis=1,
            ).astype(np.float32)
        selesai = awal + len(kelompok)
        if selesai % 80 == 0:
            laju = (time.time() - mulai) / selesai
            print(f"  {selesai}/{len(citra)} citra, {laju:.2f} detik per citra", flush=True)
    return keluaran


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detektor", required=True, choices=["yolo26l", "rtdetr_l", "rfdetr_l"])
    ap.add_argument("--bobot", required=True)
    ap.add_argument("--citra", default="D:/Work/Assisten-Dosen/SawitMVC-Depth/SawitMVC-Depth-YOLO/test/images")
    ap.add_argument("--daftar-citra", default=None, help="berkas teks, satu lintasan citra per baris")
    ap.add_argument("--perangkat", default="cpu")
    ap.add_argument("--keluaran", default=None)
    arg = ap.parse_args()

    if arg.daftar_citra:
        baris = Path(arg.daftar_citra).read_text(encoding="utf-8").splitlines()
        citra = sorted((Path(b.strip()) for b in baris if b.strip()), key=lambda p: p.stem)
        hilang = [p for p in citra if not p.is_file()]
        if hilang:
            raise SystemExit(f"{len(hilang)} citra tidak ditemukan, contoh {hilang[0]}")
    else:
        citra = sorted(Path(arg.citra).glob("*.jpg"))
    if not citra:
        raise SystemExit(f"tidak ada citra pada {arg.daftar_citra or arg.citra}")
    print(f"detektor {arg.detektor}, {len(citra)} citra, perangkat {arg.perangkat}", flush=True)

    bobot = Path(arg.bobot)
    mulai = time.time()
    if arg.detektor == "rfdetr_l":
        hasil = inferensi_rfdetr(bobot, citra, arg.perangkat)
    else:
        hasil = inferensi_ultralytics(bobot, citra, arg.perangkat)

    keluaran = Path(arg.keluaran or f"results/cross_eval/predictions/v2repro953_{arg.detektor}__on_763__test.npz")
    keluaran.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(keluaran, **hasil)
    total = sum(len(v) for v in hasil.values())
    print(f"selesai dalam {time.time() - mulai:.0f} detik, {total} deteksi, ditulis ke {keluaran}")


if __name__ == "__main__":
    main()
