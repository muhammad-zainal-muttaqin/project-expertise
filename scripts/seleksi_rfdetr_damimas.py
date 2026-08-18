"""Pilih checkpoint RF-DETR di VAL, lalu baru infer TRAIN dan TEST.

Checkpoint EMA dan regular dapat memiliki ranking berbeda. Skrip ini membuat
dump VAL untuk seluruh kandidat, memilih objective COCO tanpa melihat TEST,
mem-print lock, lalu membuka split TRAIN/TEST hanya untuk kandidat terpilih.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import eval_dump_damimas as ED  # noqa: E402
import fusi_detektor_damimas as FD  # noqa: E402
import infer_rfdetr_damimas as IR  # noqa: E402


def parse_checkpoint(items: list[str]) -> dict[str, Path]:
    out = {}
    for item in items:
        nama, sep, path = item.partition("=")
        if not sep or not nama or not path:
            raise ValueError("--checkpoint harus NAMA=PATH")
        p = Path(path).resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        if nama in out:
            raise ValueError(f"Nama checkpoint duplikat: {nama}")
        out[nama] = p
    if not out:
        raise ValueError("Sedikitnya satu --checkpoint wajib diberikan")
    return out


def bersihkan_memori_model() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True)
    ap.add_argument("--dataset", type=Path, default=IR.DS)
    ap.add_argument("--tag", default="rfdetr_l_damimas")
    ap.add_argument("--resolution", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=.001)
    ap.add_argument("--output", type=Path, default=ROOT / "results" /
                    "damimas_rfdetr_l.json")
    args = ap.parse_args()
    checkpoints = parse_checkpoint(args.checkpoint)

    # Hanya VAL dibuka selama checkpoint dibandingkan.
    coco_v, paths_v, gt_v = ED.bangun_gt(args.dataset, "val")
    ranking = []
    pred_val = {}
    for nama, path in checkpoints.items():
        print(f"VAL checkpoint {nama}: {path}", flush=True)
        model = IR.muat_model(path, args.resolution)
        pred = IR.infer_split(model, args.dataset, "val", args.batch,
                              args.threshold)
        del model
        bersihkan_memori_model()
        tujuan = ROOT / "results" / f"pred_{args.tag}_{nama}_val.npz"
        np.savez_compressed(tujuan, **pred)
        met = FD.coco_detail(coco_v, paths_v, pred)
        row = {"nama": nama, "checkpoint": str(path),
               "pred_val": str(tujuan), "metrik": met,
               "objective": FD.objektif(met)}
        ranking.append(row); pred_val[nama] = pred
        print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)

    ranking.sort(key=lambda x: x["objective"], reverse=True)
    lock = ranking[0]
    nama = lock["nama"]
    print("TERKUNCI DI VAL", json.dumps(lock, indent=2,
                                        ensure_ascii=False), flush=True)

    # Setelah lock, baru muat model terpilih dan buka TRAIN/TEST.
    model = IR.muat_model(checkpoints[nama], args.resolution)
    pred_paths = {"val": lock["pred_val"]}
    pred_test = None
    for split in ("train", "test"):
        pred = IR.infer_split(model, args.dataset, split, args.batch,
                              args.threshold)
        tujuan = ROOT / "results" / f"pred_{args.tag}_{split}.npz"
        np.savez_compressed(tujuan, **pred)
        pred_paths[split] = str(tujuan)
        if split == "test":
            pred_test = pred
    del model
    bersihkan_memori_model()
    assert pred_test is not None

    coco_t, paths_t, gt_t = ED.bangun_gt(args.dataset, "test")
    test = FD.coco_detail(coco_t, paths_t, pred_test)
    amb_info = ED.pilih_ambang(gt_v, pred_val[nama])
    amb = {n: amb_info[n]["ambang"] for n in ED.NAMA}
    operasi = {
        "val": ED.nilai_ambang(gt_v, pred_val[nama], amb),
        "test": ED.nilai_ambang(gt_t, pred_test, amb),
    }
    hasil = {
        "dataset": "SawitMVC-YOLO-Damimas",
        "protokol": ("checkpoint dipilih dan dikunci di VAL; TRAIN/TEST hanya "
                     "diinfer dengan checkpoint terpilih"),
        "threshold_dump": args.threshold,
        "ranking_val": ranking,
        "terkunci_di_val": lock,
        "test": test,
        "ambang_dipilih_di_val": amb_info,
        "titik_operasi": operasi,
        "prediksi": pred_paths,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
    print(json.dumps({"terkunci": lock, "test": test,
                      "operasi_test": operasi["test"]}, indent=2,
                     ensure_ascii=False), flush=True)
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
