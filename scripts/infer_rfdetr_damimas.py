"""Dump prediksi RF-DETR DAMIMAS ke format NPZ evaluator bersama.

Enam kolom pertama selalu ``xyxy, score, class``. RF-DETR memilih pasangan
query-kelas, sehingga satu query dapat muncul beberapa kali dengan box identik.
Baris itu dipakai untuk merekonstruksi empat skor kelas dan ID query pada kolom
6:11. Evaluator deteksi tetap membaca enam kolom pertama; kepala hilir mendapat
bukti kelas lunak tanpa forward kedua.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DS = Path("/workspace/SawitMVC-YOLO-Damimas")
NAMA_KELAS = ("B1", "B2", "B3", "B4")


def muat_model(weights: str | Path, resolution: int):
    """Muat checkpoint custom dan verifikasi kontrak kelasnya."""
    from rfdetr import RFDETRLarge

    # ``from_checkpoint`` membaca jumlah kelas dan konfigurasi yang benar dari
    # checkpoint fine-tune. Konstruktor ``pretrain_weights=...`` dapat memakai
    # kepala COCO 90-kelas dan tidak aman untuk checkpoint custom 4-kelas.
    model = RFDETRLarge.from_checkpoint(str(weights), resolution=resolution)
    names = tuple(getattr(model, "class_names", ()) or ())
    if names and names != NAMA_KELAS:
        raise RuntimeError(
            f"Urutan kelas checkpoint tidak cocok: {names} != {NAMA_KELAS}")
    return model


def infer_split(model, dataset: Path, split: str, batch: int,
                threshold: float) -> dict[str, np.ndarray]:
    paths = sorted((dataset / "images" / split).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"Tidak ada citra untuk split {split}: {dataset}")
    out: dict[str, np.ndarray] = {}
    t0 = time.time()
    for awal in range(0, len(paths), batch):
        blok = paths[awal:awal + batch]
        hasil = model.predict([str(p) for p in blok], threshold=threshold)
        if not isinstance(hasil, list):
            hasil = [hasil]
        if len(hasil) != len(blok):
            raise RuntimeError(
                f"RF-DETR mengembalikan {len(hasil)} hasil untuk {len(blok)} citra")
        for p, d in zip(blok, hasil):
            if len(d.xyxy):
                boxes = np.asarray(d.xyxy, np.float32)
                conf = np.asarray(d.confidence, np.float32)
                kelas = np.asarray(d.class_id, np.int64)
                if boxes.shape != (len(conf), 4) or len(kelas) != len(conf):
                    raise RuntimeError(f"Shape prediksi tidak konsisten: {p}")
                if not np.isfinite(boxes).all() or not np.isfinite(conf).all():
                    raise RuntimeError(f"Prediksi non-finite: {p}")
                if ((kelas < 0) | (kelas >= len(NAMA_KELAS))).any():
                    raise RuntimeError(
                        f"class_id di luar 0..3 pada {p}: {np.unique(kelas)}")
                if ((conf < 0) | (conf > 1)).any():
                    raise RuntimeError(f"Confidence di luar [0,1]: {p}")
                # Baris dengan box bit-identik berasal dari query DETR yang
                # sama tetapi kelas berbeda. Gabungkan skornya menjadi vektor
                # empat kelas, lalu tempelkan kembali pada setiap hipotesis.
                # Skor kelas yang tidak masuk top-300 tetap nol (bukti absen),
                # bukan probabilitas karangan.
                _unik, anchor = np.unique(boxes, axis=0, return_inverse=True)
                pkelas = np.zeros((len(_unik), len(NAMA_KELAS)), np.float32)
                np.maximum.at(pkelas, (anchor, kelas), conf)
                out[p.stem] = np.c_[
                    boxes, conf, kelas.astype(np.float32), pkelas[anchor],
                    anchor.astype(np.float32),
                ].astype(np.float32)
            else:
                out[p.stem] = np.zeros((0, 11), np.float32)
        n = min(awal + batch, len(paths))
        if n % 100 < batch or n == len(paths):
            print(f"{split}: {n}/{len(paths)} ({time.time()-t0:.0f}s)", flush=True)
    if set(out) != {p.stem for p in paths}:
        raise RuntimeError(f"Cakupan stem dump {split} tidak lengkap")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", nargs="+", default=["val", "test"])
    ap.add_argument("--dataset", type=Path, default=DS)
    ap.add_argument("--resolution", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=.001)
    args = ap.parse_args()

    # Import/model loading sengaja ditunda sampai argumen tervalidasi.
    model = muat_model(args.weights, args.resolution)

    for split in args.split:
        out = infer_split(model, args.dataset, split, args.batch, args.threshold)
        tujuan = ROOT / "results" / f"pred_{args.tag}_{split}.npz"
        np.savez_compressed(tujuan, **out)
        print(f"-> {tujuan}")


if __name__ == "__main__":
    main()
