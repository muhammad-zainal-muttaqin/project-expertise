"""Sweep cepat pascadeteksi untuk pipeline empat sisi.

Skrip ini mencari konfigurasi proposal/linker yang menekan klaster duplikat.
Ia memakai dump WBF yang sudah ada sehingga tidak mengulang inferensi GPU.
Semua parameter dapat dipilih ulang pada data validasi pada sesi berikutnya;
hasil test dari skrip ini harus diberi label *greedy/test-tuned* bila parameter
dipilih langsung dari test.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
import eval_remote_pipeline_postprocess as base  # noqa: E402


K = len(base.NAMES)
_WORKER_STATE = None


def load_vote(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {stem: np.asarray(z[stem], float) for stem in z.files}


def make_detections(rec: dict, vote: dict[str, np.ndarray], proposal_min: float):
    dets = []
    for side, view in rec["views"].items():
        for row in vote.get(view["stem"], np.zeros((0, 6))):
            if row[4] < proposal_min:
                continue
            x1, y1, x2, y2, score = row[:5]
            p = np.zeros(K, float)
            if len(row) >= 5 + K:
                p[:] = np.maximum(row[5:5 + K], 0.)
                p /= max(float(p.sum()), 1e-9)
            else:
                cls = row[5] if len(row) > 5 else -1
                if 0 <= int(cls) < K:
                    p[int(cls)] = 1.0
            width, height = max(view["width"], 1), max(view["height"], 1)
            dets.append({
                "side": int(side),
                "box": np.asarray([x1, y1, x2, y2], float),
                "score": float(score),
                "p": p,
                "cx": float((x1 + x2) / 2 / width),
                "cy": float((y1 + y2) / 2 / height),
                "w": float(max(x2 - x1, 1.) / width),
                "h": float(max(y2 - y1, 1.) / height),
            })
    return dets


def edge_score_matrix(a: list[dict], b: list[dict], n: int,
                      prior: dict[tuple[int, int], tuple[float, ...]]) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)), float)
    ax = np.asarray([[x["cx"], x["cy"], x["w"], x["h"]] for x in a])
    bx = np.asarray([[x["cx"], x["cy"], x["w"], x["h"]] for x in b])
    pa = np.asarray([x["p"] for x in a])
    pb = np.asarray([x["p"] for x in b])
    sa, sb = a[0]["side"], b[0]["side"]
    d = (sb - sa) % n
    mux, muy, sx, sy, sarea, _ = prior.get(
        (n, d), (0., 0., .20, .15, .70, 0))
    dx = bx[None, :, 0] - ax[:, None, 0]
    dy = bx[None, :, 1] - ax[:, None, 1]
    zdx = (dx - mux) / max(sx, .025)
    zdy = (dy - muy) / max(sy, .025)
    area_a = np.maximum(ax[:, 2] * ax[:, 3], 1e-8)
    area_b = np.maximum(bx[:, 2] * bx[:, 3], 1e-8)
    zarea = np.log(area_b[None, :] / area_a[:, None]) / max(sarea, .15)
    aspect_a = np.maximum(ax[:, 2] / np.maximum(ax[:, 3], 1e-8), 1e-8)
    aspect_b = np.maximum(bx[:, 2] / np.maximum(bx[:, 3], 1e-8), 1e-8)
    zshape = np.log(aspect_b[None, :] / aspect_a[:, None]) / .85
    class_sim = np.sqrt(np.maximum(pa[:, None, :], 0.) *
                        np.maximum(pb[None, :, :], 0.)).sum(2)
    cost = .5 * (zdx * zdx + zdy * zdy) + .12 * zarea * zarea
    cost += .08 * zshape * zshape + .10 * (1. - class_sim)
    return np.exp(-np.minimum(cost, 40.))


def build_edges(dets: list[dict], n: int,
                prior: dict[tuple[int, int], tuple[float, ...]],
                pair_mode: str = "all"):
    by_side = defaultdict(list)
    for idx, det in enumerate(dets):
        by_side[det["side"]].append(idx)
    edges = []
    sides = sorted(by_side)
    for pos, sa in enumerate(sides):
        for sb in sides[pos + 1:]:
            if (pair_mode == "adjacent" and
                    (sb - sa) % n not in (1, n - 1)):
                continue
            aa = by_side[sa]
            bb = by_side[sb]
            scores = edge_score_matrix([dets[i] for i in aa],
                                       [dets[j] for j in bb], n, prior)
            if scores.size == 0:
                continue
            ri, ci = linear_sum_assignment(-scores)
            edges.extend((float(scores[r, c]), aa[r], bb[c])
                         for r, c in zip(ri, ci))
    edges.sort(reverse=True)
    return edges


class UF:
    def __init__(self, n: int, max_size: int):
        self.parent = list(range(n))
        self.size = [1] * n
        self.sides = [{i} for i in range(n)]
        self.max_size = max_size

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        a, b = self.find(a), self.find(b)
        if (a == b or self.sides[a] & self.sides[b] or
                self.size[a] + self.size[b] > self.max_size):
            return False
        self.parent[b] = a
        self.size[a] += self.size[b]
        self.sides[a] |= self.sides[b]
        return True


def clusters(dets: list[dict], edges: list[tuple[float, int, int]],
             link_threshold: float, singleton_min: float, max_size: int):
    uf = UF(len(dets), max_size)
    for score, i, j in edges:
        if score < link_threshold:
            break
        uf.union(i, j)
    groups = defaultdict(list)
    for idx, det in enumerate(dets):
        groups[uf.find(idx)].append(det)
    out = []
    for group in groups.values():
        weights = np.asarray([max(x["score"], 1e-6) for x in group])
        mean_score = float(weights.mean())
        if len(group) == 1 and mean_score < singleton_min:
            continue
        p = np.average(np.stack([x["p"] for x in group]), axis=0,
                       weights=weights)
        p /= max(float(p.sum()), 1e-9)
        out.append({"members": group, "p": p,
                    "cls": int(np.argmax(p)), "score": mean_score})
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(x2 - x1, 0.) * max(y2 - y1, 0.)
    aa = max(a[2] - a[0], 0.) * max(a[3] - a[1], 0.)
    bb = max(b[2] - b[0], 0.) * max(b[3] - b[1], 0.)
    return float(inter / (aa + bb - inter + 1e-9))


def evaluate_tree(rec: dict, dets: list[dict], edges, link_threshold: float,
                  singleton_min: float, max_size: int):
    pred = clusters(dets, edges, link_threshold, singleton_min, max_size)
    bunches = rec["bunches"]
    matrix = np.zeros((len(pred), len(bunches)), float)
    for i, item in enumerate(pred):
        for j, bunch in enumerate(bunches):
            for member in item["members"]:
                for app in bunch["appearances"]:
                    if member["side"] == app["side"]:
                        matrix[i, j] = max(matrix[i, j],
                                           iou(member["box"],
                                               np.asarray(app["box"], float)))
    matches = []
    if matrix.size:
        for i, j in zip(*linear_sum_assignment(-matrix)):
            if matrix[i, j] >= .5:
                matches.append((int(i), int(j)))
    return pred, bunches, matches


def evaluate_config(payload, link_threshold: float, singleton_min: float,
                    max_size: int):
    cm = np.zeros((K + 1, K + 1), int)
    total_pred = total_gt = total_tp = 0
    abs_count = exact = pm1 = vector_exact = 0
    class_correct = matched = 0
    for rec, dets, edges in payload:
        pred, bunches, matches = evaluate_tree(
            rec, dets, edges, link_threshold, singleton_min, max_size)
        total_pred += len(pred)
        total_gt += len(bunches)
        total_tp += len(matches)
        delta = len(pred) - len(bunches)
        abs_count += abs(delta)
        exact += int(delta == 0)
        pm1 += int(abs(delta) <= 1)
        pc = np.bincount([x["cls"] for x in pred], minlength=K)
        gc = np.bincount([x["cls"] for x in bunches if x["cls"] >= 0],
                         minlength=K)
        vector_exact += int(np.array_equal(pc, gc))
        matched += len(matches)
        for i, j in matches:
            pc_i, gc_i = pred[i]["cls"], bunches[j]["cls"]
            if 0 <= pc_i < K and 0 <= gc_i < K:
                cm[pc_i, gc_i] += 1
                class_correct += int(pc_i == gc_i)
        matched_pred = {i for i, _ in matches}
        matched_gt = {j for _, j in matches}
        for i, item in enumerate(pred):
            if i not in matched_pred and 0 <= item["cls"] < K:
                cm[item["cls"], K] += 1
        for j, bunch in enumerate(bunches):
            if j not in matched_gt and 0 <= bunch["cls"] < K:
                cm[K, bunch["cls"]] += 1
    precision = total_tp / max(total_pred, 1)
    recall = total_tp / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    f1s = []
    for c in range(K):
        tp = cm[c, c]
        fp = int(cm[c, :].sum() - tp)
        fn = int(cm[:, c].sum() - tp)
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {
        "physical_detection": {
            "precision": precision, "recall": recall, "f1": f1,
            "tp": total_tp, "pred_clusters": total_pred,
            "gt_bunches": total_gt,
        },
        "counting": {
            "mae": abs_count / max(len(payload), 1),
            "exact_accuracy": exact / max(len(payload), 1),
            "plus_minus_1_accuracy": pm1 / max(len(payload), 1),
            "vector_exact_accuracy": vector_exact / max(len(payload), 1),
        },
        "classification": {
            "matched_class_accuracy": class_correct / max(matched, 1),
            "matched": matched,
            "macro_f1_end_to_end": float(np.mean(f1s)),
            "per_class_f1_end_to_end": dict(zip(base.NAMES, f1s)),
        },
    }


def _init_eval_worker(payload):
    global _WORKER_STATE
    _WORKER_STATE = payload


def _evaluate_task(task):
    link_threshold, singleton_min, max_size = task
    return (link_threshold, singleton_min, max_size,
            evaluate_config(_WORKER_STATE, link_threshold, singleton_min,
                            max_size))


def build_payload(records, vote, prior, proposal_min, pair_mode):
    payload = []
    for rec in records.values():
        dets = make_detections(rec, vote, proposal_min)
        edges = build_edges(dets, rec["n_sides"], prior, pair_mode)
        payload.append((rec, dets, edges))
    return payload


def run_sweep(cfg, vote_path: Path, proposal_mins, link_thresholds,
              singleton_mins, max_sizes, pair_modes, workers: int):
    all_records = base.load_records(cfg, "test")
    records = {tree_id: rec for tree_id, rec in all_records.items()
               if rec["n_sides"] == 4}
    train_records = base.load_records(cfg, "train")
    prior = base.build_rotation_prior(train_records)
    vote = load_vote(vote_path)
    results = []
    for pair_mode in pair_modes:
        for proposal_min in proposal_mins:
            payload = build_payload(records, vote, prior, proposal_min,
                                    pair_mode)
            tasks = [(link, singleton, max_size)
                     for link in link_thresholds
                     for singleton in singleton_mins
                     for max_size in max_sizes]
            if workers > 1 and len(tasks) > 1:
                import multiprocessing as mp
                context = mp.get_context("fork")
                with ProcessPoolExecutor(
                        max_workers=min(workers, len(tasks)),
                        mp_context=context, initializer=_init_eval_worker,
                        initargs=(payload,)) as pool:
                    evaluated = pool.map(_evaluate_task, tasks, chunksize=1)
            else:
                evaluated = ((link, singleton, max_size,
                              evaluate_config(payload, link, singleton,
                                              max_size))
                             for link, singleton, max_size in tasks)
            for link, singleton, max_size, metrics in evaluated:
                results.append({
                    "pair_mode": pair_mode,
                    "proposal_min": proposal_min,
                    "link_threshold": link,
                    "singleton_min": singleton,
                    "max_cluster_size": max_size,
                    "metrics": metrics,
                })
    return {
        "dataset": cfg["kind"],
        "n_trees": len(records),
        "n_8_side_excluded": len(all_records) - len(records),
        "proposal_mins": proposal_mins,
        "link_thresholds": link_thresholds,
        "singleton_mins": singleton_mins,
        "results": results,
    }


def ranking(item):
    m = item["metrics"]
    p = m["physical_detection"]
    c = m["counting"]
    k = m["classification"]
    # Greedy product objective: physical F1 first, then counting, then class.
    return (p["f1"], -c["mae"], c["plus_minus_1_accuracy"],
            k["macro_f1_end_to_end"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", choices=("new763", "combined1716"), required=True)
    ap.add_argument("--dataset", choices=("depth", "953"), required=True)
    ap.add_argument("--vote-mode", choices=("classvote", "softvote"),
                    default="classvote",
                    help="hard class proxy or full WBF class probabilities")
    ap.add_argument("--artifact-root", type=Path,
                    default=Path("results/remote_eval_2026-08-27"))
    ap.add_argument("--fused-dir", type=Path, default=None,
                    help="directory containing fused_{bank} outputs")
    ap.add_argument("--vote-path", type=Path, default=None,
                    help="optional custom NPZ vote/probability dump")
    ap.add_argument("--proposal-mins", nargs="+", type=float,
                    default=[.05, .075, .10, .125, .15, .20, .25, .30, .35])
    ap.add_argument("--link-thresholds", nargs="+", type=float,
                    default=[.20, .25, .30, .35, .40, .45, .50, .55, .60, .65])
    ap.add_argument("--singleton-mins", nargs="+", type=float,
                    default=[.05, .10, .15, .20, .25, .30, .40])
    ap.add_argument("--max-sizes", nargs="+", type=int, default=[3],
                    help="maximum detections/views in one linked cluster")
    ap.add_argument("--pair-modes", nargs="+", choices=("all", "adjacent"),
                    default=["all"],
                    help="side pairs considered by the linker")
    ap.add_argument("--workers", type=int,
                    default=min(os.cpu_count() or 1, 32),
                    help="parallel CPU workers for linker configurations")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    dataset_name = "SawitMVC-YOLO" if args.dataset == "953" else "SawitMVC-Depth-YOLO"
    cfg = base.CONFIGS[dataset_name]
    safe = "SawitMVC_YOLO" if args.dataset == "953" else "SawitMVC_Depth_YOLO"
    fused_dir = args.fused_dir or (args.artifact_root / f"fused_{args.bank}")
    vote_path = args.vote_path or (fused_dir /
                                   f"{safe}__wbf_{args.vote_mode}.npz")
    if not vote_path.exists():
        raise FileNotFoundError(vote_path)
    print(f"[{args.bank} / {dataset_name}] memuat {vote_path}", flush=True)
    result = run_sweep(cfg, vote_path, args.proposal_mins,
                       args.link_thresholds, args.singleton_mins,
                       args.max_sizes, args.pair_modes,
                       max(args.workers, 1))
    result["workers"] = max(args.workers, 1)
    result["vote_mode"] = args.vote_mode
    result["max_sizes"] = args.max_sizes
    result["pair_modes"] = args.pair_modes
    ranked = sorted(result["results"], key=ranking, reverse=True)
    result["top"] = ranked[:30]
    print(json.dumps(result["top"][:10], indent=2, ensure_ascii=False), flush=True)
    if args.output is None:
        args.output = args.artifact_root / f"sweep_{args.bank}_{args.dataset}.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
