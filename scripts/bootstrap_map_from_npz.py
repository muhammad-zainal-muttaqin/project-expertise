"""Bootstrap CI untuk mAP50, generalisasi bootstrap_map.py ke new763/combined1716.

Resampling citra dengan pengembalian (bukan kotak), berpasangan (sampel citra
sama dipakai untuk ketiga arsitektur), replikasi metodologi V2-E-023.
Dipakai untuk menguji apakah urutan RF-DETR-L > RT-DETR-L > YOLO26l pada
V2-E-034/035 signifikan atau masih di dalam derau.

Tidak re-infer -- pakai .npz yang sudah ada.
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


def load_gt_dict(dataset: Path, split: str):
    gt_coco, paths = build_gt(dataset, split)
    stem_to_id = {p.stem: i for i, p in enumerate(paths, 1)}
    id_to_stem = {i: p.stem for i, p in enumerate(paths, 1)}
    gt = {stem: [] for stem in stem_to_id}
    for ann in gt_coco.dataset["annotations"]:
        stem = id_to_stem[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        gt[stem].append([ann["category_id"] - 1, x, y, x + w, y + h])
    return {s: (np.array(v, float) if v else np.zeros((0, 5))) for s, v in gt.items()}


def mAP_pada(stems_sampel, gt, pred):
    g2, p2 = {}, {}
    for i, s in enumerate(stems_sampel):
        kunci = f"{s}#{i}"
        g2[kunci] = gt[s]
        p2[kunci] = pred.get(s, np.zeros((0, 6)))
    per = [ap50(g2, p2, c) for c in range(K)]
    return float(np.mean(per)), per


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


def _boot_chunk(args):
    corpus_name, name, gt, pr, stems, idx_chunk, chunk_id = args
    boot = np.empty(len(idx_chunk))
    for b, ii in enumerate(idx_chunk):
        boot[b], _ = mAP_pada([stems[i] for i in ii], gt, pr)
    return corpus_name, name, chunk_id, boot


def _point_estimate(args):
    corpus_name, name, gt, pr, stems = args
    m0, _ = mAP_pada(stems, gt, pr)
    return corpus_name, name, m0


def main() -> int:
    n_boot, seed, n_chunks = 500, 42, 8
    prepared = {}
    point_tasks = []
    chunk_tasks = []
    for corpus_name, cfg in CORPORA.items():
        gt = load_gt_dict(cfg["dataset"], "test")
        stems = list(gt.keys())
        n = len(stems)
        n_boxes = sum(len(v) for v in gt.values())
        print(f"=== {corpus_name}: {n} citra, {n_boxes} kotak GT ===", flush=True)
        preds = {name: {k: np.asarray(v, float) for k, v in np.load(path).items()}
                 for name, path in cfg["sources"].items()}
        rng = np.random.default_rng(seed)
        idx = [rng.integers(0, n, n) for _ in range(n_boot)]
        idx_chunks = np.array_split(np.arange(n_boot), n_chunks)
        prepared[corpus_name] = {"stems": stems, "n": n, "n_boxes": n_boxes, "preds": preds}
        for name, pr in preds.items():
            point_tasks.append((corpus_name, name, gt, pr, stems))
            for chunk_id, positions in enumerate(idx_chunks):
                idx_chunk = [idx[p] for p in positions]
                chunk_tasks.append((corpus_name, name, gt, pr, stems, idx_chunk, chunk_id))

    n_workers = min(27, len(chunk_tasks))
    print(f"Menjalankan {len(chunk_tasks)} chunk bootstrap + {len(point_tasks)} titik estimasi "
          f"dengan {n_workers} worker paralel...", flush=True)

    hasil = {c: {"titik": {}, "sebaran": {}} for c in CORPORA}
    chunks_by_model: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        point_futures = [executor.submit(_point_estimate, t) for t in point_tasks]
        chunk_futures = [executor.submit(_boot_chunk, t) for t in chunk_tasks]

        points = {}
        for fut in point_futures:
            corpus_name, name, m0 = fut.result()
            points[(corpus_name, name)] = m0

        for fut in chunk_futures:
            corpus_name, name, chunk_id, boot_chunk = fut.result()
            key = (corpus_name, name)
            chunks_by_model.setdefault(key, {})[chunk_id] = boot_chunk
            print(f"  [{corpus_name}/{name}] chunk {chunk_id} selesai "
                  f"({len(boot_chunk)} replikasi)", flush=True)

    for (corpus_name, name), chunks in chunks_by_model.items():
        boot = np.concatenate([chunks[i] for i in sorted(chunks)])
        m0 = points[(corpus_name, name)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        hasil[corpus_name]["titik"][name] = {
            "mAP50": round(m0, 4), "CI95_mAP50": [round(float(lo), 4), round(float(hi), 4)],
            "lebar_CI": round(float(hi - lo), 4)}
        hasil[corpus_name]["sebaran"][name] = boot
        print(f"  [{corpus_name}] {name:10s} mAP50={m0:.4f}  CI95=[{lo:.4f}, {hi:.4f}]  "
              f"lebar={hi-lo:.4f}", flush=True)

    hasil_final = {}
    for corpus_name, cfg in CORPORA.items():
        titik = hasil[corpus_name]["titik"]
        sebaran = hasil[corpus_name]["sebaran"]
        pasangan = {}
        for a, b in itertools.combinations(cfg["sources"].keys(), 2):
            d = sebaran[a] - sebaran[b]
            lo, hi = np.percentile(d, [2.5, 97.5])
            p_pos = float((d > 0).mean())
            sig = bool(lo > 0 or hi < 0)
            pasangan[f"{a} - {b}"] = {
                "delta_titik": round(titik[a]["mAP50"] - titik[b]["mAP50"], 4),
                "CI95_delta": [round(float(lo), 4), round(float(hi), 4)],
                "P(delta>0)": round(p_pos, 3), "signifikan_95": sig}
            tanda = "SIGNIFIKAN" if sig else "tidak signifikan (CI memuat nol)"
            print(f"  [{corpus_name}] {a} - {b}: delta={pasangan[f'{a} - {b}']['delta_titik']:+.4f} "
                  f"CI95=[{lo:+.4f}, {hi:+.4f}]  P(>0)={p_pos:.3f}  -> {tanda}", flush=True)
        hasil_final[corpus_name] = {
            "n_citra": prepared[corpus_name]["n"], "n_boxes": prepared[corpus_name]["n_boxes"],
            "n_boot": n_boot, "seed": seed, "per_model": titik, "selisih_berpasangan": pasangan}
    hasil = hasil_final
    out_path = Path("/workspace/project-expertise/results/bootstrap_map_sesi2026-08.json")
    out_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "Bootstrap CI mAP50 (500 replikasi, seed 42, resampling citra "
                "berpasangan), replikasi metodologi V2-E-023, dari dump .npz "
                "test yang sudah ada (V2-E-034/035). Tidak re-infer.",
        "corpora": hasil,
    }, indent=2) + "\n")
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
