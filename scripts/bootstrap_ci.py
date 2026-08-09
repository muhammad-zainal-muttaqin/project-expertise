"""Bootstrap confidence intervals untuk perbandingan RGB vs RGBD (Fase 4.2).

Menghitung per-pohon bootstrap CI untuk detection dan counting metrics.

Usage:
    python bootstrap_ci.py --project-root /workspace/project-expertise
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

CLASSES = ["B1", "B2", "B3", "B4"]
N_BOOT = 10_000
SEED = 42


def load_pertree_detection(pred_dir: Path, split: str = "test") -> dict[str, dict]:
    """Load per-tree detection results (per-image annotations) for a split."""
    trees = {}
    for fp in sorted(pred_dir.glob("*.json")):
        d = json.loads(fp.read_text())
        if d.get("split") != split:
            continue
        tid = d.get("tree_name") or fp.stem
        all_dets = []
        for side_data in d.get("images", {}).values():
            all_dets.extend(side_data.get("annotations", []))
        trees[tid] = all_dets
    return trees


def load_pertree_counting(pred_dir: Path, gt_dir: Path,
                          split_map: dict[str, str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load per-tree counting predictions and GT for test split."""
    from collections import defaultdict
    import pandas as pd
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    gt_map = {}
    for fp in sorted(gt_dir.glob("*.json")):
        d = json.loads(fp.read_text())
        tid = d.get("tree_name") or d.get("tree_id") or fp.stem
        by_class = d.get("summary", {}).get("by_class", d.get("summary", {}))
        gt_map[tid] = {c: by_class.get(c, 0) for c in CLASSES}

    rows, labels, tids, splits = [], [], [], []
    for fp in sorted(pred_dir.glob("*.json")):
        d = json.loads(fp.read_text())
        tid = d.get("tree_name") or fp.stem
        if tid not in gt_map:
            continue
        rows.append(_extract_features(d))
        labels.append([gt_map[tid].get(c, 0) for c in CLASSES])
        tids.append(tid)
        sp = split_map.get(tid, d.get("split", "train"))
        splits.append(sp)

    df = pd.DataFrame(rows)
    y = np.array(labels, dtype=float)
    sp_arr = np.array(splits)
    tr = (sp_arr == "train") | (sp_arr == "val")
    te = sp_arr == "test"

    X_tr, y_tr = df.values[tr], y[tr]
    X_te, y_te = df.values[te], y[te]

    ridge = Pipeline([("sc", StandardScaler()),
                      ("rid", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500]))])
    ridge.fit(X_tr, y_tr)
    y_pred = ridge.predict(X_te)

    test_tids = [t for t, s in zip(tids, splits) if s == "test"]
    return test_tids, y_te, y_pred


def _extract_features(tree_json: dict) -> dict[str, float]:
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


def bootstrap_counting_ci(y_true: np.ndarray, y_pred: np.ndarray,
                           n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    yr = np.clip(np.round(y_pred), 0, None).astype(int)
    yt = y_true.astype(int)

    boot_class_acc = []
    boot_tree_acc = []
    boot_mae = []

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt_b = yt[idx]
        yr_b = yr[idx]
        accs = []
        maes = []
        for j in range(4):
            err = np.abs(yr_b[:, j] - yt_b[:, j])
            accs.append(np.mean(err <= 1))
            maes.append(np.mean(err))
        boot_class_acc.append(np.mean(accs))
        boot_tree_acc.append(np.mean(
            np.all(np.array([np.abs(yr_b[:, j] - yt_b[:, j]) <= 1 for j in range(4)]), axis=0)
        ))
        boot_mae.append(np.mean(maes))

    return {
        "class_acc_ci": (float(np.percentile(boot_class_acc, 2.5)),
                         float(np.percentile(boot_class_acc, 97.5))),
        "tree_acc_ci": (float(np.percentile(boot_tree_acc, 2.5)),
                        float(np.percentile(boot_tree_acc, 97.5))),
        "mae_ci": (float(np.percentile(boot_mae, 2.5)),
                   float(np.percentile(boot_mae, 97.5))),
    }


def bootstrap_paired_delta(y_true_rgb: np.ndarray, y_pred_rgb: np.ndarray,
                            y_true_rgbd: np.ndarray, y_pred_rgbd: np.ndarray,
                            n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Paired bootstrap: delta = RGBD metric - RGB metric, per tree."""
    rng = np.random.RandomState(seed)
    n = len(y_true_rgb)
    assert n == len(y_true_rgbd)

    yr_rgb = np.clip(np.round(y_pred_rgb), 0, None).astype(int)
    yt_rgb = y_true_rgb.astype(int)
    yr_rgbd = np.clip(np.round(y_pred_rgbd), 0, None).astype(int)
    yt_rgbd = y_true_rgbd.astype(int)

    deltas_class = []
    deltas_tree = []

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)

        def macro_acc(yt, yr, idx):
            accs = []
            for j in range(4):
                err = np.abs(yr[idx, j] - yt[idx, j])
                accs.append(np.mean(err <= 1))
            return np.mean(accs)

        def tree_acc(yt, yr, idx):
            return np.mean(
                np.all(np.array([np.abs(yr[idx, j] - yt[idx, j]) <= 1 for j in range(4)]), axis=0)
            )

        d_class = macro_acc(yt_rgbd, yr_rgbd, idx) - macro_acc(yt_rgb, yr_rgb, idx)
        d_tree = tree_acc(yt_rgbd, yr_rgbd, idx) - tree_acc(yt_rgb, yr_rgb, idx)
        deltas_class.append(d_class)
        deltas_tree.append(d_tree)

    deltas_class = np.array(deltas_class)
    deltas_tree = np.array(deltas_tree)
    return {
        "delta_class_acc": float(np.mean(deltas_class)),
        "delta_class_acc_ci": (float(np.percentile(deltas_class, 2.5)),
                                float(np.percentile(deltas_class, 97.5))),
        "delta_class_acc_p_positive": float(np.mean(deltas_class > 0)),
        "delta_tree_acc": float(np.mean(deltas_tree)),
        "delta_tree_acc_ci": (float(np.percentile(deltas_tree, 2.5)),
                               float(np.percentile(deltas_tree, 97.5))),
        "delta_tree_acc_p_positive": float(np.mean(deltas_tree > 0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default="/workspace/project-expertise")
    ap.add_argument("--gt-dir", default="/workspace/SawitMVC-Depth/json")
    ap.add_argument("--yolo-dir", default="/workspace/SawitMVC-Depth-YOLO")
    args = ap.parse_args()
    proj = Path(args.project_root)
    gt_dir = Path(args.gt_dir)

    split_map = {}
    yolo_dir = Path(args.yolo_dir)
    for sp in ["train", "val", "test"]:
        imgd = yolo_dir / sp / "images"
        if not imgd.exists():
            continue
        for p in imgd.iterdir():
            parts = p.stem.rsplit("_", 1)
            if len(parts) == 2 and parts[0] not in split_map:
                split_map[parts[0]] = sp

    MODELS = [
        ("YOLO26l", "pertree_rgb352/yolo_yolo26l", "pertree_rgbd352/yolo_yolo26lrgbd"),
        ("RT-DETR-L", "pertree_rgb352/rtdetr_rtdetrl", "pertree_rgbd352/rtdetr_rtdetrlrgbd"),
        ("RF-DETR-L", "pertree_rgb352/rfdetr_rfdetrl", "pertree_rgbd352/rfdetr_rfdetrlrgbd"),
    ]

    results = {}
    for arch, rgb_dir_rel, rgbd_dir_rel in MODELS:
        rgb_dir = proj / "runs" / rgb_dir_rel
        rgbd_dir = proj / "runs" / rgbd_dir_rel
        if not rgb_dir.exists() or not rgbd_dir.exists():
            print(f"SKIP {arch}: missing pertree dir")
            continue

        print(f"\n===== {arch} =====")
        tids_rgb, yt_rgb, yp_rgb = load_pertree_counting(rgb_dir, gt_dir, split_map)
        tids_rgbd, yt_rgbd, yp_rgbd = load_pertree_counting(rgbd_dir, gt_dir, split_map)

        ci_rgb = bootstrap_counting_ci(yt_rgb, yp_rgb)
        ci_rgbd = bootstrap_counting_ci(yt_rgbd, yp_rgbd)

        print(f"  RGB   Class ±1: [{ci_rgb['class_acc_ci'][0]*100:.1f}%, {ci_rgb['class_acc_ci'][1]*100:.1f}%]")
        print(f"  RGBD  Class ±1: [{ci_rgbd['class_acc_ci'][0]*100:.1f}%, {ci_rgbd['class_acc_ci'][1]*100:.1f}%]")

        if set(tids_rgb) == set(tids_rgbd):
            paired = bootstrap_paired_delta(yt_rgb, yp_rgb, yt_rgbd, yp_rgbd)
            print(f"  Δ Class ±1: {paired['delta_class_acc']*100:.2f}pp "
                  f"CI=[{paired['delta_class_acc_ci'][0]*100:.1f}pp, {paired['delta_class_acc_ci'][1]*100:.1f}pp]")
            print(f"  P(RGBD > RGB): {paired['delta_class_acc_p_positive']*100:.1f}%")
            results[arch] = {"rgb_ci": ci_rgb, "rgbd_ci": ci_rgbd, "paired": paired}
        else:
            print(f"  (trees differ, no paired test)")
            results[arch] = {"rgb_ci": ci_rgb, "rgbd_ci": ci_rgbd}

    out = proj / "results" / "bootstrap_ci_352.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
