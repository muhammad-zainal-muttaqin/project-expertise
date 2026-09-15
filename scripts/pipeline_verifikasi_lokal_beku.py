"""Verifikasi kandidat YOLO26 melalui pemeriksaan lokal dengan bobot tetap.

Inferensi hanya membaca citra dan bobot. Anotasi dibaca terpisah oleh tahap
evaluasi. Tidak ada optimizer, backward, adaptasi bobot, atau pemilihan resep
dari TEST. Skor gabungan adalah skor konsistensi, bukan probabilitas TP
terkalibrasi. Pencapaian mAP50 harus diukur; tidak diasumsikan dari rumus skor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

from audit_batas_koreksi_map50 import evaluate, iou_matrix, recall_ap_bound, self_check


ROOT = Path(__file__).resolve().parents[1]
RECIPE = {
    "version": "local_frozen_v1",
    "native_long_edge": 1280,
    "candidate_min_score": 0.01,
    "candidate_nms_iou": 0.75,
    "candidate_limit": 48,
    "local_side": 384,
    "local_context": 1.6,
    "verification_iou": 0.5,
    "local_head": "one2many",
    "score_rule": "native_class_score * max(local_class_score * IoU**2)",
    "coordinate_rule": "retain_native_candidate_box",
}


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def save_cache(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def nms_indices(boxes, scores, threshold, limit):
    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order) and len(keep) < limit:
        first, order = int(order[0]), order[1:]
        keep.append(first)
        if len(order):
            overlap = iou_matrix(boxes[first:first + 1], boxes[order])[0]
            order = order[overlap <= threshold]
    return np.asarray(keep, dtype=int)


def emit_classes(boxes, probabilities):
    """Keluarkan skor empat kelas; evaluator melakukan pencocokan per kelas."""
    if not len(boxes):
        return np.empty((0, 6), np.float32)
    return np.column_stack((np.repeat(boxes, 4, axis=0),
                            probabilities.reshape(-1), np.tile(np.arange(4), len(boxes))))


def local_crop(image, box):
    """Transformasi afin seragam; kotak target tetap dalam koordinat potongan."""
    x1, y1, x2, y2 = map(float, box)
    side = max(x2 - x1, y2 - y1, 2.) * RECIPE["local_context"]
    scale = RECIPE["local_side"] / side
    left = (x1 + x2) / 2 - side / 2
    top = (y1 + y2) / 2 - side / 2
    matrix = np.array([[scale, 0, -left * scale],
                       [0, scale, -top * scale]], np.float32)
    crop = cv2.warpAffine(image, matrix, (RECIPE["local_side"], RECIPE["local_side"]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(114, 114, 114))
    target = (np.asarray(box, float) - [left, top, left, top]) * scale
    return crop, target


class FrozenDetector:
    def __init__(self, weights, threads):
        torch.set_num_threads(threads)
        self.model = YOLO(str(weights)).model.float().cpu().eval()
        self.model.requires_grad_(False)
        self.head = self.model.model[-1]
        if self.head.nc != 4 or not self.head.end2end:
            raise ValueError("Pipeline ini mensyaratkan YOLO empat kelas dengan kepala ganda.")
        if self.head.one2many["cls_head"] is None:
            raise ValueError("Kepala one-to-many telah dihapus dari bobot ini.")
        assert not any(p.requires_grad for p in self.model.parameters())

    @staticmethod
    def tensor(images):
        rgb = np.stack([im[:, :, ::-1].transpose(2, 0, 1) for im in images])
        return torch.from_numpy(np.ascontiguousarray(rgb)).float().div_(255.)

    @torch.inference_mode()
    def native(self, image):
        h, w = image.shape[:2]
        side = RECIPE["native_long_edge"]
        padded = LetterBox((side, side), auto=True, stride=32)(image=image)
        ratio = min(side / h, side / w)
        pad_x = round((padded.shape[1] - round(w * ratio)) / 2 - .1)
        pad_y = round((padded.shape[0] - round(h * ratio)) / 2 - .1)
        out, raw = self.model(self.tensor([padded]))
        baseline = out[0].cpu().numpy().copy()
        decoded = self.head._inference(raw["one2one"])[0].T.cpu().numpy()
        boxes = decoded[:, :4].copy()
        probabilities = decoded[:, 4:8].copy()
        boxes = (boxes - [pad_x, pad_y, pad_x, pad_y]) / ratio
        baseline[:, :4] = (baseline[:, :4] - [pad_x, pad_y, pad_x, pad_y]) / ratio
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        baseline[:, [0, 2]] = baseline[:, [0, 2]].clip(0, w)
        baseline[:, [1, 3]] = baseline[:, [1, 3]].clip(0, h)
        valid = ((probabilities.max(1) >= RECIPE["candidate_min_score"])
                 & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
        boxes, probabilities = boxes[valid], probabilities[valid]
        keep = nms_indices(boxes, probabilities.max(1), RECIPE["candidate_nms_iou"],
                           RECIPE["candidate_limit"])
        return {"baseline": baseline.astype(np.float32),
                "boxes": boxes[keep].astype(np.float32),
                "native_probs": probabilities[keep].astype(np.float32),
                "image_hw": np.array([h, w]),
                "candidate_count_before_nms": np.array(len(boxes))}

    @torch.inference_mode()
    def verify(self, image, boxes, batch):
        result = np.zeros((len(boxes), 4), np.float32)
        for start in range(0, len(boxes), batch):
            samples = [local_crop(image, box) for box in boxes[start:start + batch]]
            _, raw = self.model(self.tensor([item[0] for item in samples]))
            dense = self.head._inference(raw["one2many"]).permute(0, 2, 1).cpu().numpy()
            for j, (_, target) in enumerate(samples):
                rows = dense[j]
                overlap = iou_matrix(rows[:, :4], target[None])[..., 0]
                valid = overlap >= RECIPE["verification_iou"]
                if valid.any():
                    # Maksimum ini hanya memakai prediksi dan kotak kandidat.
                    # Target di sini adalah transformasi kotak prediksi, bukan GT.
                    evidence = rows[valid, 4:8] * overlap[valid, None] ** 2
                    result[start + j] = evidence.max(axis=0)
        return result


def load_cache(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def run_inference(args, images):
    model = None
    started = time.perf_counter()
    candidates, processed = 0, 0
    for number, path in enumerate(images, 1):
        native_path = args.cache / args.split / "native" / f"{path.stem}.npz"
        local_path = args.cache / args.split / "local" / f"{path.stem}.npz"
        image = None
        if native_path.exists():
            native = load_cache(native_path)
            if str(native["image_sha256"]) != file_hash(path):
                raise ValueError(f"Citra berubah sejak cache dibuat: {path}")
        else:
            model = model or FrozenDetector(args.weights, args.threads)
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Citra tidak dapat dibaca: {path}")
            native = model.native(image)
            native["image_sha256"] = np.array(file_hash(path))
            save_cache(native_path, **native)
            processed += 1
        candidates += len(native["boxes"])
        if args.stage in ("verify", "all") and not local_path.exists():
            model = model or FrozenDetector(args.weights, args.threads)
            image = image if image is not None else cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Citra tidak dapat dibaca: {path}")
            local = model.verify(image, native["boxes"], args.batch)
            save_cache(local_path, local_probs=local,
                       native_sha256=np.array(file_hash(native_path)))
        if number % 10 == 0 or number == len(images):
            print(json.dumps({"stage": args.stage, "split": args.split,
                              "done": number, "total": len(images),
                              "native_computed": processed, "candidates": candidates,
                              "seconds": round(time.perf_counter() - started, 2)}), flush=True)


def run_evaluation(args, images):
    truth, baseline, native_preds, verified, boxes_by_image = {}, {}, {}, {}, {}
    all_local = True
    label_digest = hashlib.sha256()
    for path in images:
        stem = path.stem
        native_path = args.cache / args.split / "native" / f"{stem}.npz"
        native = load_cache(native_path)
        h, w = native["image_hw"]
        label_path = args.dataset / "labels" / args.split / f"{stem}.txt"
        raw = label_path.read_bytes()
        label_digest.update(stem.encode())
        label_digest.update(raw)
        labels = np.loadtxt(label_path, ndmin=2) if raw.strip() else np.empty((0, 5))
        if labels.shape[1] != 5 or not np.isin(labels[:, 0], np.arange(4)).all():
            raise ValueError(f"Label tidak sah: {label_path}")
        xy = labels[:, 1:3] * [w, h]
        wh = labels[:, 3:5] * [w, h]
        truth[stem] = np.column_stack((xy - wh / 2, xy + wh / 2, labels[:, 0]))
        baseline[stem] = native["baseline"]
        boxes = native["boxes"]
        boxes_by_image[stem] = boxes
        native_preds[stem] = emit_classes(boxes, native["native_probs"])
        local_path = args.cache / args.split / "local" / f"{stem}.npz"
        if local_path.exists():
            local = load_cache(local_path)
            if str(local["native_sha256"]) != file_hash(native_path):
                raise ValueError(f"Cache lokal berasal dari kandidat berbeda: {stem}")
            verified[stem] = emit_classes(boxes, native["native_probs"] * local["local_probs"])
        else:
            all_local = False
    stems = sorted(truth)
    base_metrics, _ = evaluate(baseline, truth, stems)
    native_metrics, _ = evaluate(native_preds, truth, stems)
    counts = base_metrics["GT_by_class"]
    maximum = np.zeros(4, dtype=int)
    for stem in stems:
        target = truth[stem]
        overlap = iou_matrix(boxes_by_image[stem], target[:, :4])
        for label in range(4):
            edges = overlap[:, target[:, 4] == label] >= .5
            if all(edges.shape):
                ri, ci = linear_sum_assignment(edges.astype(int), maximize=True)
                maximum[label] += int(edges[ri, ci].sum())
    upper = {f"B{i + 1}": recall_ap_bound(int(m), counts[f"B{i + 1}"])
             for i, m in enumerate(maximum)}
    metrics = evaluate(verified, truth, stems)[0] if all_local else None
    output = {"status": "OBSERVED_MAP50_GE_075" if metrics and metrics["mAP50"] >= .75 else "TARGET_NOT_VERIFIED",
              "split": args.split, "images": len(stems),
              "complete_split": args.limit_images is None,
              "recipe": RECIPE, "weights_sha256": file_hash(args.weights),
              "labels_sha256": label_digest.hexdigest(),
              "script_sha256": file_hash(Path(__file__)),
              "training": False, "gpu": False,
              "baseline": base_metrics, "candidate_native": native_metrics,
              "optimistic_candidate_geometry_AP_bound": upper,
              "optimistic_candidate_geometry_mAP_bound": float(np.mean(list(upper.values()))),
              "verified_local": metrics}
    suffix = f"_{args.limit_images}images" if args.limit_images else ""
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / f"{args.split}{suffix}.json", output)
    if all_local:
        np.savez_compressed(args.output / f"pred_{args.split}{suffix}.npz", **verified)
    np.savez_compressed(args.output / f"baseline_{args.split}{suffix}.npz", **baseline)
    np.savez_compressed(args.output / f"candidate_native_{args.split}{suffix}.npz", **native_preds)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("native", "verify", "evaluate", "all"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--dataset", type=Path,
                        default=ROOT.parent / "Baseline-SawitMVC" / "SawitMVC-YOLO")
    parser.add_argument("--weights", type=Path,
                        default=ROOT / "models" / "yolo26l_e60_i1280_v2repro" / "best.pt")
    parser.add_argument("--cache", type=Path,
                        default=ROOT / "runs" / "verifikasi_lokal_beku_20260908")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "verifikasi_lokal_beku_2026-09-08")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--limit-images", type=int)
    args = parser.parse_args()
    self_check()
    signature = {"recipe": RECIPE, "weights_sha256": file_hash(args.weights),
                 "dataset": str(args.dataset.resolve())}
    config = args.cache / "configuration.json"
    if config.exists() and json.loads(config.read_text(encoding="utf-8")) != signature:
        raise ValueError("Konfigurasi cache berbeda; gunakan lokasi cache baru.")
    write_json(config, signature)
    images = sorted((args.dataset / "images" / args.split).glob("*.jpg"))
    if args.limit_images:
        images = images[:args.limit_images]
    if not images:
        raise ValueError("Tidak ditemukan citra pada partisi yang diminta.")
    if args.stage != "evaluate":
        run_inference(args, images)
    if args.stage in ("all", "evaluate"):
        run_evaluation(args, images)
    if file_hash(args.weights) != signature["weights_sha256"]:
        raise AssertionError("Berkas bobot berubah selama eksekusi.")


if __name__ == "__main__":
    main()
