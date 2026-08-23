"""Hitung AP50 class-agnostic (lokalisasi murni) dari dump .npz yang sudah ada.

Tidak menjalankan inferensi ulang -- cukup melipat GT dan prediksi jadi satu
kelas lalu menghitung ulang lewat pycocotools. Dipakai untuk model sesi
2026-08-22/23 (new763 dan combined1716) supaya plafon lokalisasi bisa
dibandingkan tanpa GPU.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).parent))
from eval_new763_pycoco import build_gt  # noqa: E402


def agnostic_ap50(dataset: Path, split: str, npz_path: Path) -> dict:
    gt, paths = build_gt(dataset, split)
    for ann in gt.dataset["annotations"]:
        ann["category_id"] = 1
    gt.dataset["categories"] = [{"id": 1, "name": "tandan"}]
    gt.createIndex()

    dump = np.load(npz_path)
    stem_to_id = {p.stem: i for i, p in enumerate(paths, 1)}
    dt_list = []
    for stem, rows in dump.items():
        image_id = stem_to_id.get(stem)
        if image_id is None:
            continue
        for x1, y1, x2, y2, score, _cls in rows:
            dt_list.append({
                "image_id": image_id, "category_id": 1,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            })
    dt = gt.loadRes(dt_list) if dt_list else gt.loadRes([])
    ev = COCOeval(gt, dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {"AP50_agnostic": round(float(ev.stats[1]), 6),
            "AP50_95_agnostic": round(float(ev.stats[0]), 6),
            "n_images": len(paths)}


RUNS = [
    ("new763_yolo26l", Path("/workspace/SawitMVC-Depth-YOLO-RGB"),
     Path("/workspace/project-expertise/results/new763/predictions/yolo26l_rgb_s42_i1280__test.npz")),
    ("new763_rtdetr_l", Path("/workspace/SawitMVC-Depth-YOLO-RGB"),
     Path("/workspace/project-expertise/results/new763/predictions/rtdetr_l_rgb_s42_i1280__test.npz")),
    ("new763_rfdetr_l", Path("/workspace/SawitMVC-Depth-YOLO-RGB"),
     Path("/workspace/project-expertise/results/new763/predictions/rfdetr_l_rgb_s42_i1280__test.npz")),
    ("combined1716_yolo26l", Path("/workspace/SawitMVC-Combined-1716-RGB"),
     Path("/workspace/project-expertise/results/combined1716/predictions/combined1716_yolo26l_rgb_s42_i1280__test.npz")),
    ("combined1716_rtdetr_l", Path("/workspace/SawitMVC-Combined-1716-RGB"),
     Path("/workspace/project-expertise/results/combined1716/predictions/combined1716_rtdetr_l_rgb_s42_i1280__test.npz")),
    ("combined1716_rfdetr_l", Path("/workspace/SawitMVC-Combined-1716-RGB"),
     Path("/workspace/project-expertise/results/combined1716/predictions/combined1716_rfdetr_l_rgb_s42_i1280__test.npz")),
]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-json", type=Path,
        default=Path("/workspace/project-expertise/results/agnostic_ap50_sesi2026-08.json"),
    )
    args = ap.parse_args()

    results = []
    for name, dataset, npz_path in RUNS:
        metrics = agnostic_ap50(dataset, "test", npz_path)
        results.append((name, {**metrics, "dataset": str(dataset),
                                "predictions": str(npz_path)}))
        print(f"{name:<24} AP50_agnostic={metrics['AP50_agnostic']:.4f} "
              f"AP50-95_agnostic={metrics['AP50_95_agnostic']:.4f} "
              f"n_images={metrics['n_images']}", flush=True)
    ranking = sorted(results, key=lambda kv: -kv[1]["AP50_agnostic"])
    print("\n=== Ranking AP50_agnostic (test) ===")
    for rank, (name, metrics) in enumerate(ranking, 1):
        print(f"{rank}. {name:<24} {metrics['AP50_agnostic']:.4f}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": "AP50 class-agnostic dihitung ulang dari dump .npz prediksi "
                    "test yang sudah ada (eval_new763_pycoco.py), tanpa "
                    "inferensi ulang. GT dan prediksi dilipat jadi 1 kelas.",
            "runs": {name: metrics for name, metrics in results},
            "ranking": [name for name, _ in ranking],
        }, indent=2) + "\n")
        print(f"\n-> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
