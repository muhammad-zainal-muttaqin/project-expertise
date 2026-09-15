"""Audit CPU atas peluang koreksi mAP50 dari prediksi tetap.

Tidak melatih model, mengubah anotasi, atau menjalankan inferensi. Koreksi
berbantuan anotasi hanya merupakan diagnosis bersyarat, bukan model inferensi.
Evaluator menghitung AP50 dengan 101 titik daya tangkap, pencocokan serakah
satu-ke-satu, dan maksimum 100 prediksi per citra per kelas. Cakupan evaluator
dibatasi pada kotak tanpa anotasi crowd/ignore, sebagaimana sumber SawitMVC ini.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
RECALL_GRID = np.linspace(0, 1, 101)
NAMES = ["B1", "B2", "B3", "B4"]
BANKS = [
    ("remote_rfdetr_val", "val",
     "results/pred_combined1716_rfdetr_l_953_val.npz", None),
    ("remote_rfdetr_test", "test",
     "results/remote_eval_2026-08-27/predictions/"
     "remote_combined1716_rfdetr_l_953_test__test.npz",
     "results/remote_eval_2026-08-27/metrics/remote_combined1716_rfdetr_l_953_test.json"),
    ("retrained_rfdetr_test", "test",
     "results/pred_rfdetr_l_v2repro_953_test.npz",
     "results/rfdetr_l_v2repro_953_retrain_2026-09-07.json"),
]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_ground_truth(root: Path):
    manifest_path = root / "split_manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = {r["tree_id"]: r["new_split"] for r in csv.DictReader(handle)}
    truth, stems = {}, {"val": [], "test": []}
    digest = hashlib.sha256(manifest_path.read_bytes())
    for tree, split in sorted(manifest.items()):
        if split not in stems:
            continue
        raw = (root / "annotations" / f"{tree}.json").read_bytes()
        digest.update(tree.encode())
        digest.update(raw)
        record = json.loads(raw.decode("utf-8-sig"))
        for view in record["images"].values():
            stem = Path(view["filename"]).stem
            if stem in truth:
                raise ValueError(f"Identitas citra berulang: {stem}")
            w, h = view["width"], view["height"]
            rows = []
            for annotation in view["annotations"]:
                if annotation.get("iscrowd") or annotation.get("ignore"):
                    raise ValueError("Anotasi crowd/ignore di luar cakupan audit ini.")
                cx, cy, bw, bh = annotation["bbox_yolo"]
                label = int(annotation["class_id"])
                if label not in range(4) or bw <= 0 or bh <= 0:
                    raise ValueError(f"Anotasi tidak sah: {stem}")
                rows.append([(cx-bw/2)*w, (cy-bh/2)*h,
                             (cx+bw/2)*w, (cy+bh/2)*h, label])
            truth[stem] = np.asarray(rows, dtype=float).reshape(-1, 5)
            stems[split].append(stem)
    return truth, {s: sorted(v) for s, v in stems.items()}, digest.hexdigest()


def iou_matrix(a, b):
    area_a = np.prod(np.maximum(a[:, 2:4] - a[:, :2], 0), axis=1)
    area_b = np.prod(np.maximum(b[:, 2:4] - b[:, :2], 0), axis=1)
    low = np.maximum(a[:, None, :2], b[None, :, :2])
    high = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    intersection = np.prod(np.maximum(high - low, 0), axis=2)
    return intersection / np.maximum(
        area_a[:, None] + area_b[None, :] - intersection, 1e-15)


def recall_ap_bound(matched, count):
    """Batas AP diskret jika semua prediksi yang dipertahankan adalah TP."""
    return float(np.mean(RECALL_GRID <= matched / count)) if matched and count else 0.


def evaluate(predictions, truth, stems):
    aps, recalls, counts, true_positives, rectangles = [], [], [], [], []
    selected = {stem: set() for stem in stems}
    for label in range(4):
        scores, flags, count = [], [], 0
        for stem in stems:
            target = truth[stem][truth[stem][:, 4] == label, :4]
            count += len(target)
            rows = predictions.get(stem, np.empty((0, 6)))
            indices = np.flatnonzero(rows[:, 5] == label)
            order = np.argsort(-rows[indices, 4], kind="stable")
            indices = indices[order][:100]
            overlap = iou_matrix(rows[indices, :4], target)
            used = set()
            for j, index in enumerate(indices):
                available = overlap[j].copy()
                if used:
                    available[list(used)] = -1
                # COCO memilih GT terakhir jika beberapa IoU maksimum sama.
                best = (len(target) - 1 - int(np.argmax(available[::-1]))
                        if len(target) else -1)
                valid = best >= 0 and available[best] >= .5
                if valid:
                    used.add(best)
                    selected[stem].add(int(index))
                flags.append(valid)
                scores.append(rows[index, 4])
        if not count:
            raise ValueError("Audit ini mensyaratkan dukungan keempat kelas.")
        order = np.argsort(-np.asarray(scores), kind="stable")
        flags = np.asarray(flags, dtype=float)
        tp = np.cumsum(flags[order])
        fp = np.cumsum(1 - flags[order])
        recall = tp / count
        precision = tp / np.maximum(tp + fp, 1e-15)
        grid_counts = np.searchsorted(RECALL_GRID, recall, side="right")
        rectangle_values = precision * grid_counts / len(RECALL_GRID)
        rectangles.append(float(rectangle_values.max()) if len(tp) else 0.)
        if len(precision):
            precision = np.maximum.accumulate(precision[::-1])[::-1]
        sampled = np.zeros(101)
        indices = np.searchsorted(recall, RECALL_GRID, side="left")
        valid = indices < len(precision)
        sampled[valid] = precision[indices[valid]]
        aps.append(float(sampled.mean()))
        recalls.append(float(recall[-1]) if len(recall) else 0.)
        counts.append(count)
        true_positives.append(int(tp[-1]) if len(tp) else 0)
    metrics = {
        "mAP50": float(np.mean(aps)), "AP50_by_class": dict(zip(NAMES, aps)),
        "max_recall_by_class": dict(zip(NAMES, recalls)),
        "GT_by_class": dict(zip(NAMES, counts)),
        "TP_by_class": dict(zip(NAMES, true_positives)),
        "precision_recall_rectangle_lower_bound": float(np.mean(rectangles)),
    }
    return metrics, selected


def retain_matched(predictions, matched):
    return {k: rows[sorted(matched.get(k, set()))] for k, rows in predictions.items()}


def audit_bank(name, split, relative_path, reference_path, truth, stems):
    path = ROOT / relative_path
    with np.load(path, allow_pickle=False) as archive:
        predictions = {k: archive[k].astype(float) for k in archive.files}
    if set(predictions) != set(stems):
        raise ValueError(f"Populasi dump tidak sama dengan manifest: {name}")
    ignored_counts, ignored_max_score = {}, None
    original_rows = sum(map(len, predictions.values()))
    for stem, rows in predictions.items():
        if (rows.ndim != 2 or rows.shape[1] != 6 or not np.isfinite(rows).all()
                or not (rows[:, 5] == np.floor(rows[:, 5])).all()
                or (rows[:, 5] < 0).any()):
            raise ValueError(f"Skema prediksi tidak sah: {name}")
        included = np.isin(rows[:, 5], np.arange(4))
        for label in np.unique(rows[~included, 5]):
            key = str(int(label))
            ignored_counts[key] = ignored_counts.get(key, 0) + int(
                np.sum(rows[:, 5] == label))
        if (~included).any():
            score = float(rows[~included, 4].max())
            ignored_max_score = max(ignored_max_score or score, score)
        # COCOeval empat kelas mengabaikan kategori di luar B1--B4.
        # Kategori tersebut juga dikeluarkan dari seluruh konstruksi oracle.
        predictions[stem] = rows[included]
    actual, matched = evaluate(predictions, truth, stems)
    reference, reference_errors = None, []
    if reference_path is not None:
        record = json.loads((ROOT / reference_path).read_text(encoding="utf-8-sig"))
        if "splits" in record:
            anchor = record["splits"][split]
            reference = anchor["mAP50"]
            reference_errors = [abs(actual["AP50_by_class"][c] - value)
                                for c, value in anchor["per_kelas_AP50"].items()]
        else:
            reference = record["mAP50_classaware"]
        reference_errors.append(abs(actual["mAP50"] - reference))
        if max(reference_errors) > 1e-6:
            raise ValueError(f"Acuan COCOeval tidak tereproduksi: {name}, {actual}")
    filtered, _ = evaluate(retain_matched(predictions, matched), truth, stems)
    corrected, maximum = {}, np.zeros(4, dtype=int)
    changed = 0
    for stem in stems:
        rows = predictions[stem].copy()
        target = truth[stem]
        overlap = iou_matrix(rows[:, :4], target[:, :4])
        if len(rows) and len(target):
            nearest = overlap.argmax(axis=1)
            valid = overlap[np.arange(len(rows)), nearest] >= .5
            changed += int(np.sum(rows[valid, 5] != target[nearest[valid], 4]))
            rows[valid, 5] = target[nearest[valid], 4]
        corrected[stem] = rows
        for label in range(4):
            columns = np.flatnonzero(target[:, 4] == label)
            if len(columns) and len(rows):
                edges = overlap[:, columns] >= .5
                ri, ci = linear_sum_assignment(edges.astype(int), maximize=True)
                maximum[label] += int(edges[ri, ci].sum())
    corrected_metrics, corrected_matches = evaluate(corrected, truth, stems)
    ideal, _ = evaluate(retain_matched(corrected, corrected_matches), truth, stems)
    upper = {c: recall_ap_bound(int(m), actual["GT_by_class"][c])
             for c, m in zip(NAMES, maximum)}
    for metrics in [actual, filtered, corrected_metrics, ideal]:
        if metrics["precision_recall_rectangle_lower_bound"] > metrics["mAP50"] + 1e-12:
            raise AssertionError("Pelanggaran batas bawah AP.")
    return {
        "bank": name, "split": split, "images": len(stems),
        "prediction_path": relative_path, "prediction_sha256": sha256(path),
        "original_prediction_rows": original_rows,
        "evaluated_prediction_rows": sum(map(len, predictions.values())),
        "ignored_category_row_counts": ignored_counts,
        "ignored_category_max_score": ignored_max_score,
        "reference_path": reference_path,
        "reference_sha256": sha256(ROOT / reference_path) if reference_path else None,
        "reference_mAP50": reference, "reference_reproduced": reference is not None,
        "reference_max_absolute_error": max(reference_errors) if reference_errors else None,
        "actual": actual,
        "oracle_filter_only_keep_original_classes_boxes_scores": filtered,
        "oracle_relabel_overlaps_keep_boxes_scores_and_all_predictions": corrected_metrics,
        "oracle_relabel_and_filter": ideal,
        "optimistic_fixed_geometry_AP_bound_by_class": upper,
        "optimistic_fixed_geometry_mAP_bound": float(np.mean(list(upper.values()))),
        "class_changes_before_evaluator_limit": changed,
    }


def self_check():
    truth = {"a": np.array([[30*c, 0, 30*c+10, 10, c] for c in range(4)], float)}
    perfect = {"a": np.array([[30*c, 0, 30*c+10, 10, .9, c] for c in range(4)], float)}
    assert evaluate(perfect, truth, ["a"])[0]["mAP50"] == 1.
    assert evaluate({}, truth, ["a"])[0]["mAP50"] == 0.
    duplicate = {"a": np.concatenate([perfect["a"], perfect["a"]])}
    metrics, matched = evaluate(duplicate, truth, ["a"])
    assert sum(metrics["TP_by_class"].values()) == 4
    assert evaluate(retain_matched(duplicate, matched), truth, ["a"])[0]["mAP50"] == 1.
    swapped = {"a": perfect["a"].copy()}
    swapped["a"][:, 5] = (swapped["a"][:, 5] + 1) % 4
    assert evaluate(swapped, truth, ["a"])[0]["mAP50"] == 0.
    leading_false = perfect["a"].copy()
    leading_false[:, [1, 3]] += 50
    leading_false[:, 4] = .95
    mixed = {"a": np.concatenate([leading_false, perfect["a"]])}
    assert evaluate(mixed, truth, ["a"])[0]["mAP50"] == .5
    boundary = {"a": perfect["a"].copy()}
    boundary["a"][:, 2] += 10  # IoU tepat 0,50 harus diterima.
    assert evaluate(boundary, truth, ["a"])[0]["mAP50"] == 1.
    blocked = {"a": np.concatenate([np.tile(leading_false, (101, 1)), perfect["a"]])}
    assert evaluate(blocked, truth, ["a"])[0]["mAP50"] == 0.
    extra_truth = truth["a"].copy()
    extra_truth[:, [0, 2]] += 1000
    doubled = {"a": np.concatenate([truth["a"], extra_truth])}
    assert evaluate(perfect, doubled, ["a"])[0]["mAP50"] == 51 / 101


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-root", type=Path,
                        default=ROOT.parent / "Baseline-SawitMVC" / "ground_truth")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    self_check()
    truth, stems, digest = load_ground_truth(args.ground_truth_root)
    results = [audit_bank(*bank, truth, stems[bank[1]]) for bank in BANKS]
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "Diagnosis berbantuan anotasi; bukan pipeline inferensi.",
        "synthetic_checks_passed": True,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__},
        "training": False, "gpu_inference": False, "model_selection": False,
        "script_sha256": sha256(Path(__file__)),
        "ground_truth_root": str(args.ground_truth_root.resolve()),
        "ground_truth_manifest_and_selected_json_sha256": digest,
        "protocol": {
            "IoU": .5, "recall_points": 101,
            "max_predictions_per_image_per_class": 100,
            "evaluator": "AP50 kotak tanpa crowd/ignore; acuan historis COCOeval diverifikasi.",
            "category_policy": "ID kelas di luar 0--3 diabaikan sebelum evaluasi aktual maupun semua koreksi oracle; sama dengan cakupan empat kategori COCOeval.",
            "class_correction": "Kelas setiap kotak dengan IoU >= 0.5 diganti kelas GT ber-IoU tertinggi. Kotak lainnya dipertahankan.",
            "filter_correction": "Hanya TP yang ditemukan evaluator dipertahankan; operasi ini memakai GT.",
            "upper_bound": "Pencocokan maksimum per kelas dengan geometri tetap; penggunaan proposal lintas kelas dilonggarkan. Bukan batas dataset atau batas semua algoritme.",
            "test_usage": "TEST historis dibaca kembali hanya untuk diagnosis, tanpa memilih parameter.",
        },
        "sufficient_conditions_examples": {
            "each_class_precision_090_recall_085_AP_lower_bound": .9 * 86 / 101,
            "each_class_precision_095_recall_090_AP_lower_bound": .95 * 91 / 101,
            "note": "Syarat cukup pada keluaran yang telah dievaluasi; bukan syarat wajib, jaminan pelatihan, atau jaminan generalisasi.",
        },
        "runs": results,
    }
    text = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
