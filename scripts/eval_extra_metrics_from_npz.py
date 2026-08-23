"""Precision/Recall/F1 standar, sweep threshold, dan WBF ensemble -- dari .npz.

Tiga hal, semuanya reuse dump .npz test yang sudah ada (V2-E-034/035), tanpa
re-infer / GPU:

1. Precision/Recall/F1 per kelas pada satu ambang confidence tetap (default
   0,25, sama seperti V2-E-013): pencocokan IoU>=0,5 di DALAM kelas yang sama
   (bukan class-agnostic seperti confusion analysis V2-E-037).
2. Sweep ambang confidence untuk cari titik macro-F1 terbaik per model.
3. WBF ensemble 3 detektor per korpus (V2-E-019 style): fusi per-kelas untuk
   mAP50 class-aware, fusi lintas-kelas untuk AP50 class-agnostic (plafon).
"""
from __future__ import annotations

import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_new763_pycoco import NAMES, build_gt  # noqa: E402

K = len(NAMES)


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def ap50(gt: dict, pred: dict, kelas: int | None):
    rekam, npos = [], 0
    for stem, g in gt.items():
        gg = g if kelas is None else g[g[:, 0] == kelas]
        npos += len(gg)
        pr = pred.get(stem, np.zeros((0, 6)))
        if kelas is not None:
            pr = pr[pr[:, 5] == kelas]
        pr = pr[np.argsort(-pr[:, 4])] if len(pr) else pr
        M = iou_mat(pr[:, :4], gg[:, 1:5])
        dipakai = np.zeros(len(gg), bool)
        for k in range(len(pr)):
            kol = np.where(dipakai, -1.0, M[k]) if len(gg) else np.zeros(0)
            j = int(np.argmax(kol)) if len(gg) else -1
            if j >= 0 and kol[j] >= 0.5:
                dipakai[j] = True
                rekam.append((pr[k, 4], 1))
            else:
                rekam.append((pr[k, 4], 0))
    if npos == 0:
        return float("nan")
    if not rekam:
        return 0.0
    rekam.sort(key=lambda x: -x[0])
    tp = np.cumsum([r[1] for r in rekam]); fp = np.cumsum([1 - r[1] for r in rekam])
    rec, prec = tp / npos, tp / (tp + fp)
    return float(np.mean([prec[rec >= t].max() if (rec >= t).any() else 0.0
                          for t in np.linspace(0, 1, 101)]))


def wbf(kotak: np.ndarray, iou_th: float = 0.6, n_model: int = 2) -> np.ndarray:
    if len(kotak) == 0:
        return kotak
    kotak = kotak[np.argsort(-kotak[:, 4])]
    gugus: list[list[np.ndarray]] = []
    for k in kotak:
        for g in gugus:
            if iou_mat(k[None, :4], g[0][None, :4])[0, 0] >= iou_th:
                g.append(k)
                break
        else:
            gugus.append([k])
    keluar = []
    for g in gugus:
        a = np.stack(g)
        bobot = a[:, 4:5]
        xy = (a[:, :4] * bobot).sum(0) / bobot.sum()
        keluar.append([*xy, float(a[:, 4].mean() * min(len(g), n_model) / n_model)])
    return np.array(keluar, float)


def load_gt_dict(dataset: Path, split: str):
    gt_coco, paths = build_gt(dataset, split)
    stem_to_id = {p.stem: i for i, p in enumerate(paths, 1)}
    id_to_stem = {i: p.stem for i, p in enumerate(paths, 1)}
    gt = {stem: [] for stem in stem_to_id}
    for ann in gt_coco.dataset["annotations"]:
        stem = id_to_stem[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        gt[stem].append([ann["category_id"] - 1, x, y, x + w, y + h])
    return {s: (np.array(v, float) if v else np.zeros((0, 5))) for s, v in gt.items()}, paths


def prf_at_threshold(gt: dict, pred: dict, conf: float):
    """Precision/Recall/F1 per kelas, pencocokan IoU>=0.5 DI DALAM kelas yang sama."""
    per_class = {}
    for c, name in enumerate(NAMES):
        tp = fp = fn = 0
        for stem, g in gt.items():
            gg = g[g[:, 0] == c]
            pr = pred.get(stem, np.zeros((0, 6)))
            pr = pr[(pr[:, 5] == c) & (pr[:, 4] >= conf)]
            pr = pr[np.argsort(-pr[:, 4])] if len(pr) else pr
            M = iou_mat(pr[:, :4], gg[:, 1:5])
            claimed = np.zeros(len(gg), bool)
            for k in range(len(pr)):
                col = np.where(claimed, -1.0, M[k]) if len(gg) else np.zeros(0)
                j = int(np.argmax(col)) if len(gg) else -1
                if j >= 0 and col[j] >= 0.5:
                    claimed[j] = True
                    tp += 1
                else:
                    fp += 1
            fn += int((~claimed).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[name] = {"tp": tp, "fp": fp, "fn": fn,
                            "precision": round(precision, 4), "recall": round(recall, 4),
                            "f1": round(f1, 4)}
    macro_p = float(np.mean([v["precision"] for v in per_class.values()]))
    macro_r = float(np.mean([v["recall"] for v in per_class.values()]))
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))
    return {"conf_threshold": conf, "per_class": per_class,
            "macro_precision": round(macro_p, 4), "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4)}


def sweep_threshold(gt: dict, pred: dict, thresholds):
    best = None
    curve = []
    for t in thresholds:
        m = prf_at_threshold(gt, pred, t)
        curve.append({"conf": t, "macro_f1": m["macro_f1"],
                      "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"]})
        if best is None or m["macro_f1"] > best["macro_f1"]:
            best = {**m, "conf": t}
    return best, curve


def _wbf_one_image(args):
    stem, boxes_per_model, n_model, iou_th = args
    per_class_boxes = {c: [] for c in range(K)}
    all_boxes = []
    for rows in boxes_per_model:
        for r in rows:
            c = int(r[5])
            if c >= K:
                # Artefak kelas "background"/no-object (mis. RF-DETR kadang
                # keluar kelas ke-K) -- tidak berkorespondensi ke kategori GT
                # manapun, sudah diabaikan pycocotools juga di
                # eval_new763_pycoco.py. Diabaikan di sini juga.
                continue
            per_class_boxes[c].append(r[:5])
            all_boxes.append(r[:5])
    rows_out = []
    for c in range(K):
        arr = np.array(per_class_boxes[c], float) if per_class_boxes[c] else np.zeros((0, 5))
        fused = wbf(arr, iou_th, n_model)
        for x1, y1, x2, y2, s in fused:
            rows_out.append([x1, y1, x2, y2, s, c])
    ca = np.array(rows_out, float) if rows_out else np.zeros((0, 6))

    arr_all = np.array(all_boxes, float) if all_boxes else np.zeros((0, 5))
    fused_agn = wbf(arr_all, iou_th, n_model)
    agn = (np.concatenate([fused_agn, np.zeros((len(fused_agn), 1))], 1)
           if len(fused_agn) else np.zeros((0, 6)))
    return stem, ca, agn


def wbf_ensemble_corpus(dataset: Path, paths, preds: dict, iou_th=0.6, executor=None):
    """preds: {model_name: {stem: (N,6) array}}. Paralel per-gambar kalau
    executor (ProcessPoolExecutor) diberikan -- tiap gambar independen."""
    n_model = len(preds)
    stems = [p.stem for p in paths]
    tasks = [(stem, [pr.get(stem, np.zeros((0, 6))) for pr in preds.values()], n_model, iou_th)
             for stem in stems]
    fused_classaware, fused_agnostic = {}, {}
    mapper = executor.map(_wbf_one_image, tasks) if executor else map(_wbf_one_image, tasks)
    for stem, ca, agn in mapper:
        fused_classaware[stem] = ca
        fused_agnostic[stem] = agn
    return fused_classaware, fused_agnostic


CORPORA = {
    "new763": {
        "dataset": Path("/workspace/SawitMVC-Depth-YOLO-RGB"),
        "sources": {
            "yolo26l": Path("/workspace/project-expertise/results/new763/predictions/yolo26l_rgb_s42_i1280__test.npz"),
            "rtdetr_l": Path("/workspace/project-expertise/results/new763/predictions/rtdetr_l_rgb_s42_i1280__test.npz"),
            "rfdetr_l": Path("/workspace/project-expertise/results/new763/predictions/rfdetr_l_rgb_s42_i1280__test.npz"),
        },
    },
    "combined1716": {
        "dataset": Path("/workspace/SawitMVC-Combined-1716-RGB"),
        "sources": {
            "yolo26l": Path("/workspace/project-expertise/results/combined1716/predictions/combined1716_yolo26l_rgb_s42_i1280__test.npz"),
            "rtdetr_l": Path("/workspace/project-expertise/results/combined1716/predictions/combined1716_rtdetr_l_rgb_s42_i1280__test.npz"),
            "rfdetr_l": Path("/workspace/project-expertise/results/combined1716/predictions/combined1716_rfdetr_l_rgb_s42_i1280__test.npz"),
        },
    },
}


def _prf_task(args):
    corpus_name, name, gt, pr, thresholds = args
    m025 = prf_at_threshold(gt, pr, 0.25)
    best, curve = sweep_threshold(gt, pr, thresholds)
    return ("prf", corpus_name, name, m025, best, curve)


def main() -> int:
    result = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "note": "P/R/F1 standar + sweep threshold + WBF ensemble, dari dump "
                      ".npz test (V2-E-034/035), tanpa re-infer.",
              "prf_at_025": {}, "sweep_best_f1": {}, "wbf_ensemble": {}}

    thresholds = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]

    prf_tasks = []
    corpus_data = {}
    for corpus_name, cfg in CORPORA.items():
        gt, paths = load_gt_dict(cfg["dataset"], "test")
        preds_raw = {name: {k: np.asarray(v, float) for k, v in np.load(npz_path).items()}
                     for name, npz_path in cfg["sources"].items()}
        corpus_data[corpus_name] = (cfg["dataset"], paths, gt, preds_raw)
        for name, pr in preds_raw.items():
            prf_tasks.append((corpus_name, name, gt, pr, thresholds))

    with ProcessPoolExecutor(max_workers=27) as executor:
        prf_futures = [executor.submit(_prf_task, t) for t in prf_tasks]
        for fut in prf_futures:
            _, corpus_name, name, m025, best, curve = fut.result()
            key = f"{corpus_name}_{name}"
            result["prf_at_025"][key] = m025
            result["sweep_best_f1"][key] = {"best": best, "curve": curve}
            print(f"=== {key} ===", flush=True)
            for cname, v in m025["per_class"].items():
                print(f"  {cname}: P={v['precision']:.4f} R={v['recall']:.4f} F1={v['f1']:.4f} "
                      f"(tp={v['tp']} fp={v['fp']} fn={v['fn']})")
            print(f"  macro@0.25: P={m025['macro_precision']:.4f} R={m025['macro_recall']:.4f} "
                  f"F1={m025['macro_f1']:.4f}")
            print(f"  best conf={best['conf']:.2f} macro_F1={best['macro_f1']:.4f} "
                  f"(P={best['macro_precision']:.4f} R={best['macro_recall']:.4f})", flush=True)

        for corpus_name, (dataset, paths, gt, preds_raw) in corpus_data.items():
            print(f"=== WBF ensemble {corpus_name} ({len(paths)} citra, paralel) ===", flush=True)
            fused_ca, fused_agn = wbf_ensemble_corpus(dataset, paths, preds_raw, executor=executor)
            per_class_ap = [ap50(gt, fused_ca, c) for c in range(K)]
            map50_ensemble = float(np.mean(per_class_ap))
            ap50_agnostic_ensemble = ap50(gt, fused_agn, None)
            ensemble = {
                "mAP50_classaware": round(map50_ensemble, 4),
                "AP50_agnostic": round(ap50_agnostic_ensemble, 4),
                "per_class_AP50": {NAMES[c]: round(per_class_ap[c], 4) for c in range(K)},
            }
            result["wbf_ensemble"][corpus_name] = ensemble
            print(f"  mAP50 class-aware = {ensemble['mAP50_classaware']:.4f}")
            print(f"  AP50 agnostic     = {ensemble['AP50_agnostic']:.4f}", flush=True)

    out_path = Path("/workspace/project-expertise/results/extra_metrics_sesi2026-08.json")
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
