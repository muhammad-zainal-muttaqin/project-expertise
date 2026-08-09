"""Kompilasi matriks 9 sel (3 arsitektur × 3 dataset) untuk Fase 4.

Usage:
    python compile_matrix.py --project-root /workspace/project-expertise
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ARCHS = ["YOLO26l", "RT-DETR-L", "RF-DETR-L"]
CLASSES = ["B1", "B2", "B3", "B4"]

DATASETS = [
    ("953-RGB",  "perkelas_pycoco_v2repro.json",  "counting_v2repro.json",  None),
    ("352-RGB",  "perkelas_pycoco_rgb352.json",    "counting_rgb352.json",   None),
    ("352-RGBD", "perkelas_pycoco_rgbd352.json",   "counting_rgbd352.json",  "-RGBD"),
]


def load(proj: Path, fname: str) -> dict:
    fp = proj / "results" / fname
    if not fp.exists():
        return {}
    return json.loads(fp.read_text())


def get_det_key(arch: str, suffix: str | None) -> str:
    if suffix:
        return arch + suffix
    return arch


def get_count_key(arch: str, suffix: str | None) -> str:
    if suffix:
        return arch + suffix
    return arch


def compile_detection(proj: Path) -> list[dict]:
    rows = []
    for ds_name, det_file, _, suffix in DATASETS:
        det = load(proj, det_file)
        for arch in ARCHS:
            key = get_det_key(arch, suffix)
            entry = det.get(key, {})
            test = entry.get("test", {})
            row = {
                "dataset": ds_name,
                "architecture": arch,
                "params_juta": entry.get("params_juta", "?"),
                "test_mAP50": test.get("mAP50", None),
                "test_mAP50_95": test.get("mAP50_95", None),
            }
            ap50 = test.get("per_kelas_AP50", {})
            for c in CLASSES:
                row[f"AP50_{c}"] = ap50.get(c, None)
            rows.append(row)
    return rows


def compile_counting(proj: Path) -> list[dict]:
    rows = []
    for ds_name, _, cnt_file, suffix in DATASETS:
        cnt = load(proj, cnt_file)
        for arch in ARCHS:
            key = get_count_key(arch, suffix)
            entry = cnt.get(key, {})
            row = {
                "dataset": ds_name,
                "architecture": arch,
                "n_trees_test": entry.get("n_trees_test", None),
                "class_acc": entry.get("macro_acc", None),
                "tree_acc": entry.get("joint_acc", None),
                "macro_mae": entry.get("macro_mae", None),
            }
            for c in CLASSES:
                row[f"acc_{c}"] = entry.get(f"acc_{c}", None)
                row[f"mae_{c}"] = entry.get(f"mae_{c}", None)
                row[f"bias_{c}"] = entry.get(f"bias_{c}", None)
            rows.append(row)
    return rows


def fmt(v, precision=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    return str(v)


def pct(v, precision=2):
    if v is None:
        return "—"
    return f"{v*100:.{precision}f}%"


def delta(rgbd_val, rgb_val):
    if rgbd_val is None or rgb_val is None:
        return "—"
    d = rgbd_val - rgb_val
    return f"{d:+.4f}"


def delta_pct(rgbd_val, rgb_val):
    if rgbd_val is None or rgb_val is None:
        return "—"
    d = (rgbd_val - rgb_val) * 100
    return f"{d:+.2f}pp"


def print_detection_matrix(rows: list[dict]):
    print("\n## Matriks Deteksi (test split, pycocotools)")
    print()
    print("| Dataset | Arsitektur | mAP50 | mAP50-95 | B1 | B2 | B3 | B4 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['dataset']} | {r['architecture']} | "
              f"{fmt(r['test_mAP50'])} | {fmt(r['test_mAP50_95'])} | "
              f"{fmt(r.get('AP50_B1'))} | {fmt(r.get('AP50_B2'))} | "
              f"{fmt(r.get('AP50_B3'))} | {fmt(r.get('AP50_B4'))} |")

    print("\n### Delta RGBD vs RGB (352 pohon)")
    print()
    print("| Arsitektur | Δ mAP50 | Δ mAP50-95 | Δ B1 | Δ B2 | Δ B3 | Δ B4 |")
    print("|---|---|---|---|---|---|---|")
    rgb_rows = {r['architecture']: r for r in rows if r['dataset'] == '352-RGB'}
    rgbd_rows = {r['architecture']: r for r in rows if r['dataset'] == '352-RGBD'}
    for arch in ARCHS:
        rgb = rgb_rows.get(arch, {})
        rgbd = rgbd_rows.get(arch, {})
        print(f"| {arch} | "
              f"{delta(rgbd.get('test_mAP50'), rgb.get('test_mAP50'))} | "
              f"{delta(rgbd.get('test_mAP50_95'), rgb.get('test_mAP50_95'))} | "
              f"{delta(rgbd.get('AP50_B1'), rgb.get('AP50_B1'))} | "
              f"{delta(rgbd.get('AP50_B2'), rgb.get('AP50_B2'))} | "
              f"{delta(rgbd.get('AP50_B3'), rgb.get('AP50_B3'))} | "
              f"{delta(rgbd.get('AP50_B4'), rgb.get('AP50_B4'))} |")


def print_counting_matrix(rows: list[dict]):
    print("\n## Matriks Counting (test split, Ridge + F_all)")
    print()
    print("| Dataset | Arsitektur | N test | Class ±1 | Tree ±1 | MAE |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['dataset']} | {r['architecture']} | "
              f"{r.get('n_trees_test', '—')} | "
              f"{pct(r.get('class_acc'))} | {pct(r.get('tree_acc'))} | "
              f"{fmt(r.get('macro_mae'))} |")

    print("\n### Delta RGBD vs RGB (352 pohon)")
    print()
    print("| Arsitektur | Δ Class ±1 | Δ Tree ±1 | Δ MAE |")
    print("|---|---|---|---|")
    rgb_rows = {r['architecture']: r for r in rows if r['dataset'] == '352-RGB'}
    rgbd_rows = {r['architecture']: r for r in rows if r['dataset'] == '352-RGBD'}
    for arch in ARCHS:
        rgb = rgb_rows.get(arch, {})
        rgbd = rgbd_rows.get(arch, {})
        print(f"| {arch} | "
              f"{delta_pct(rgbd.get('class_acc'), rgb.get('class_acc'))} | "
              f"{delta_pct(rgbd.get('tree_acc'), rgb.get('tree_acc'))} | "
              f"{delta(rgbd.get('macro_mae'), rgb.get('macro_mae'))} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default="/workspace/project-expertise")
    args = ap.parse_args()
    proj = Path(args.project_root)

    det_rows = compile_detection(proj)
    cnt_rows = compile_counting(proj)

    out = proj / "results" / "matrix_compiled.json"
    out.write_text(json.dumps({"detection": det_rows, "counting": cnt_rows}, indent=2))
    print(f"Compiled matrix → {out}")

    print_detection_matrix(det_rows)
    print_counting_matrix(cnt_rows)

    missing = []
    for r in det_rows:
        if r["test_mAP50"] is None:
            missing.append(f"  DET: {r['dataset']} × {r['architecture']}")
    for r in cnt_rows:
        if r["class_acc"] is None:
            missing.append(f"  CNT: {r['dataset']} × {r['architecture']}")
    if missing:
        print(f"\n⚠ Missing cells ({len(missing)}):")
        for m in missing:
            print(m)
    else:
        print("\nAll 9 cells complete for both detection and counting.")


if __name__ == "__main__":
    main()
