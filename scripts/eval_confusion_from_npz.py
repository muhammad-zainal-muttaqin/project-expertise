"""Confusion analysis dari dump .npz -- replikasi metodologi V2-E-013.

Untuk tiap GT box, cari prediksi dengan skor >= --conf yang IoU-nya >= 0.5
(pencocokan class-agnostic, greedy per skor tertinggi), lalu tabulasi kelas
GT vs kelas prediksi. Menghasilkan confusion matrix, recall per kelas
(bersyarat pada box yang berhasil dideteksi), akurasi klasifikasi bersyarat
dan akurasi atas SELURUH GT (termasuk yang tidak terdeteksi).

Tidak re-infer -- pakai .npz yang sudah ada dari eval_new763_pycoco.py.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_new763_pycoco import NAMES, build_gt  # noqa: E402


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def confusion_for_run(dataset: Path, split: str, npz_path: Path, conf: float = 0.25) -> dict:
    gt_coco, paths = build_gt(dataset, split)
    stem_to_id = {p.stem: i for i, p in enumerate(paths, 1)}
    id_to_stem = {i: p.stem for i, p in enumerate(paths, 1)}

    gt_by_image: dict[str, list[tuple[float, float, float, float, int]]] = {
        stem: [] for stem in stem_to_id
    }
    for ann in gt_coco.dataset["annotations"]:
        stem = id_to_stem[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        gt_by_image[stem].append((x, y, x + w, y + h, ann["category_id"] - 1))

    dump = np.load(npz_path)
    n_class = len(NAMES)
    confusion = np.zeros((n_class, n_class), dtype=int)
    n_gt_total = 0
    n_matched = 0
    n_correct_matched = 0

    for stem, gt_boxes in gt_by_image.items():
        n_gt_total += len(gt_boxes)
        if not gt_boxes:
            continue
        gt_arr = np.array([b[:4] for b in gt_boxes], dtype=float)
        gt_cls = np.array([b[4] for b in gt_boxes], dtype=int)

        rows = dump[stem] if stem in dump.files else np.zeros((0, 6))
        rows = rows[rows[:, 4] >= conf]
        if len(rows) == 0:
            continue
        order = np.argsort(-rows[:, 4])
        rows = rows[order]
        pred_boxes = rows[:, :4]
        pred_cls = rows[:, 5].astype(int)

        ious = iou_mat(pred_boxes, gt_arr)
        claimed = np.zeros(len(gt_boxes), dtype=bool)
        for k in range(len(rows)):
            col = np.where(claimed, -1.0, ious[k])
            j = int(np.argmax(col))
            if col[j] >= 0.5:
                claimed[j] = True
                confusion[gt_cls[j], pred_cls[k]] += 1
                n_matched += 1
                if pred_cls[k] == gt_cls[j]:
                    n_correct_matched += 1

    per_class_recall = {}
    for c, name in enumerate(NAMES):
        row_total = int(confusion[c].sum())
        per_class_recall[name] = {
            "n_detected": row_total,
            "recall_conditional": round(confusion[c, c] / row_total, 4) if row_total else None,
        }

    return {
        "n_gt_total": n_gt_total,
        "n_matched_iou50": n_matched,
        "confusion_matrix": confusion.tolist(),
        "class_names": NAMES,
        "per_class_recall_conditional": per_class_recall,
        "accuracy_conditional": round(n_correct_matched / n_matched, 4) if n_matched else None,
        "accuracy_over_all_gt": round(n_correct_matched / n_gt_total, 4) if n_gt_total else None,
        "conf_threshold": conf,
    }


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
    out = {}
    for name, dataset, npz_path in RUNS:
        metrics = confusion_for_run(dataset, "test", npz_path)
        out[name] = metrics
        print(f"\n=== {name} ===")
        print(f"n_gt_total={metrics['n_gt_total']} n_matched_iou50={metrics['n_matched_iou50']} "
              f"(conf>=0.25)")
        print(f"akurasi klasifikasi bersyarat (di antara yang terdeteksi): "
              f"{metrics['accuracy_conditional']:.4f}")
        print(f"akurasi atas SELURUH GT (termasuk gagal deteksi): "
              f"{metrics['accuracy_over_all_gt']:.4f}")
        for cname, r in metrics["per_class_recall_conditional"].items():
            rc = r["recall_conditional"]
            print(f"  {cname}: n_detected={r['n_detected']} "
                  f"recall_conditional={rc if rc is None else f'{rc:.4f}'}")

    out_path = Path("/workspace/project-expertise/results/confusion_analysis_sesi2026-08.json")
    out_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "Confusion analysis dari dump .npz test (V2-E-034/035), replikasi "
                "metodologi V2-E-013. Match GT<->prediksi class-agnostic IoU>=0.5, "
                "conf>=0.25, greedy per skor tertinggi.",
        "runs": out,
    }, indent=2) + "\n")
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
