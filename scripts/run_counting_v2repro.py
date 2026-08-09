"""Counting evaluation untuk ketiga detektor v2repro (953 pohon).

Mengikuti pola exp_counting_v3.py dari Baseline-SawitMVC:
- Feature set: F_all (67-dim)
- Model: Ridge (RidgeCV)
- Strategy: train+val (812 pohon)
- Eval: test (141 pohon)

Usage:
    python run_counting_v2repro.py \
        --baseline-root /workspace/Baseline-SawitMVC \
        --project-root /workspace/project-expertise
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

CLASSES = ["B1", "B2", "B3", "B4"]

DETECTORS = [
    ("yolo26l_v2repro",  "YOLO26l"),
    ("rtdetr_l_v2repro", "RT-DETR-L"),
    ("rfdetr_l_v2repro", "RF-DETR-L"),
]


def extract_all_features(tree_json: dict) -> dict[str, float]:
    sides = tree_json.get("images", {})
    n_sides = max(len(sides), 1)
    psc = {c: [] for c in CLASSES}
    cf_ = {c: [] for c in CLASSES}
    ar_ = {c: [] for c in CLASSES}
    cy_ = {c: [] for c in CLASSES}

    for sd in sides.values():
        cnt = {c: 0 for c in CLASSES}
        for ann in sd.get("annotations", []):
            cls = ann.get("class_name", "")
            if cls not in CLASSES:
                continue
            conf = float(ann.get("conf", 1.0))
            bbox = ann.get("bbox_yolo", [0, 0, 0, 0])
            cf_[cls].append(conf)
            ar_[cls].append(float(bbox[2]) * float(bbox[3]))
            cy_[cls].append(float(bbox[1]))
            cnt[cls] += 1
        for c in CLASSES:
            psc[c].append(cnt[c])

    f: dict[str, float] = {}
    for c in CLASSES:
        ps = np.array(psc[c], dtype=float)
        cf = np.array(cf_[c])
        ar = np.array(ar_[c])
        cy = np.array(cy_[c])
        n = len(cf)
        f[f"naive_sum_{c}"] = float(ps.sum())
        f[f"max_per_side_{c}"] = float(ps.max())
        f[f"mean_per_side_{c}"] = float(ps.mean())
        f[f"std_per_side_{c}"] = float(ps.std())
        f[f"min_per_side_{c}"] = float(ps.min())
        f[f"cv_per_side_{c}"] = float(ps.std() / (ps.mean() + 1e-6))
        f[f"n_sides_det_{c}"] = float((ps > 0).sum())
        f[f"consistency_{c}"] = float(1.0 / (1.0 + ps.std()))
        f[f"conf_sum_{c}"] = float(cf.sum())
        f[f"conf_mean_{c}"] = float(cf.mean()) if n > 0 else 0.0
        f[f"conf_max_{c}"] = float(cf.max()) if n > 0 else 0.0
        f[f"high_conf_{c}"] = float((cf >= 0.5).sum())
        f[f"vhigh_conf_{c}"] = float((cf >= 0.6).sum())
        f[f"mean_cy_{c}"] = float(cy.mean()) if n > 0 else 0.5
        f[f"mean_area_{c}"] = float(ar.mean()) if n > 0 else 0.0

    total = sum(f[f"naive_sum_{c}"] for c in CLASSES)
    f["n_sides"] = float(n_sides)
    f["total_naive"] = float(total)
    for c in CLASSES:
        f[f"frac_{c}"] = f[f"naive_sum_{c}"] / (total + 1e-6)
    f["b3_b23_frac"] = f["naive_sum_B3"] / (f["naive_sum_B2"] + f["naive_sum_B3"] + 1e-6)
    return f


def load_gt(gt_dir: Path) -> dict[str, dict[str, int]]:
    gt = {}
    for fp in sorted(gt_dir.glob("*.json")):
        with open(fp, encoding="utf-8-sig") as fh:
            d = json.load(fh)
        tid = d.get("tree_name") or d.get("tree_id") or fp.stem
        summary = d.get("summary", {})
        by_class = summary.get("by_class", summary)
        gt[tid] = {c: by_class.get(c, 0) for c in CLASSES}
    return gt


def load_splits(data_dir: Path) -> dict[str, str]:
    manifest = data_dir / "split_manifest.csv"
    splits = {}
    with open(manifest) as fh:
        import csv
        reader = csv.DictReader(fh)
        for row in reader:
            tid = row.get("tree_id") or row.get("tree_name")
            sp = row.get("new_split") or row.get("split")
            if tid and sp:
                splits[tid] = sp
    return splits


def load_dataset(inference_dir: Path, gt_dir: Path, splits_map: dict[str, str]):
    gt_map = load_gt(gt_dir)
    rows, labels, tree_ids, tree_splits = [], [], [], []
    for fp in sorted(inference_dir.glob("*.json")):
        with open(fp, encoding="utf-8-sig") as fh:
            d = json.load(fh)
        tid = d.get("tree_name") or d.get("tree_id") or fp.stem
        if tid not in gt_map:
            continue
        rows.append(extract_all_features(d))
        labels.append([gt_map[tid].get(c, 0) for c in CLASSES])
        tree_ids.append(tid)
        tree_splits.append(splits_map.get(tid, d.get("split", "train")))
    df = pd.DataFrame(rows)
    return df, np.array(labels, dtype=float), tree_ids, np.array(tree_splits)


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yr = np.clip(np.round(y_pred), 0, None).astype(int)
    yt = y_true.astype(int)
    r: dict = {}
    for j, c in enumerate(CLASSES):
        err = np.abs(yr[:, j] - yt[:, j])
        r[f"acc_{c}"] = float(np.mean(err <= 1))
        r[f"mae_{c}"] = float(np.mean(err))
        r[f"bias_{c}"] = float(np.mean(yr[:, j] - yt[:, j]))
    r["macro_acc"] = float(np.mean([r[f"acc_{c}"] for c in CLASSES]))
    r["macro_mae"] = float(np.mean([r[f"mae_{c}"] for c in CLASSES]))
    r["joint_acc"] = float(np.mean(
        np.all(np.array([np.abs(yr[:, j] - yt[:, j]) <= 1 for j in range(len(CLASSES))]), axis=0)
    ))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-root", default="/workspace/Baseline-SawitMVC")
    ap.add_argument("--project-root", default="/workspace/project-expertise")
    args = ap.parse_args()

    baseline = Path(args.baseline_root)
    proj = Path(args.project_root)
    gt_dir = baseline / "ground_truth" / "annotations"
    splits_map = load_splits(baseline / "ground_truth")

    out_file = proj / "results" / "counting_v2repro.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_file.read_text()) if out_file.exists() else {}

    for det_dir_name, det_label in DETECTORS:
        inference_dir = proj / "runs" / "pertree" / det_dir_name
        if not inference_dir.exists():
            print(f"SKIP {det_label}: {inference_dir} not found")
            continue

        n_jsons = len(list(inference_dir.glob("*.json")))
        print(f"\n===== {det_label} ({n_jsons} trees) =====")

        df, y, tree_ids, splits = load_dataset(inference_dir, gt_dir, splits_map)
        tr = splits == "train"
        va = splits == "val"
        te = splits == "test"
        print(f"  Train={tr.sum()} | Val={va.sum()} | Test={te.sum()}")

        train_mask = tr | va
        X_all = df.values.astype(float)
        X_tr, y_tr = X_all[train_mask], y[train_mask]
        X_te, y_te = X_all[te], y[te]

        model = Pipeline([
            ("sc", StandardScaler()),
            ("rid", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500]))
        ])
        model.fit(X_tr, y_tr)
        m = score(y_te, model.predict(X_te))

        print(f"  Ridge+F_all (train+val → test):")
        print(f"    Class ±1 Acc: {m['macro_acc']*100:.2f}%")
        print(f"    Tree  ±1 Acc: {m['joint_acc']*100:.2f}%")
        print(f"    Macro MAE:    {m['macro_mae']:.4f}")
        for c in CLASSES:
            print(f"    {c}: acc={m[f'acc_{c}']*100:.1f}% mae={m[f'mae_{c}']:.3f} bias={m[f'bias_{c}']:+.3f}")

        results[det_label] = {
            "detector": det_dir_name,
            "n_trees_test": int(te.sum()),
            "feature_set": "F_all",
            "n_dim": int(df.shape[1]),
            "model": "Ridge",
            "strategy": "train+val",
            **m,
        }
        out_file.write_text(json.dumps(results, indent=2))

    print(f"\n-> {out_file}")


if __name__ == "__main__":
    main()
