#!/usr/bin/env python3
"""Collect the new763 RGB-vs-RGBD4 validation artifacts and plots.

This collector intentionally reads only validation outputs.  It does not
discover, load, or summarize any RGBD4 test file.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RGBD_RESULTS = RESULTS / "new763_rgbd4"
FIGURES = RESULTS / "figures"


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def val_baseline(arch: str) -> dict[str, Any]:
    summary = read_json(RESULTS / "new763_summary.json")
    run = next(x for x in summary["runs"] if x["arch"] == arch)
    detail = read_json(RESULTS / "new763" / f"{run['run_name']}.json")
    return {
        "run_name": run["run_name"],
        "mAP50": detail["splits"]["val"]["mAP50"],
        "mAP50_95": detail["splits"]["val"]["mAP50_95"],
        "per_kelas_AP50": detail["splits"]["val"]["per_kelas_AP50"],
        "predictions": detail["splits"]["val"]["predictions"],
    }


def result_row(arch: str, result_name: str, boot_name: str) -> dict[str, Any]:
    base = val_baseline(arch)
    rgbd = read_json(RESULTS / result_name)
    boot = read_json(RESULTS / boot_name)
    rgbd_metrics = rgbd["metrics"]
    base_classes = base["per_kelas_AP50"]
    rgbd_classes = rgbd_metrics["per_kelas_AP50"]
    return {
        "arch": arch,
        "baseline_rgb_val": base,
        "rgbd4_val": {
            "run_name": rgbd["run_name"],
            "mAP50": rgbd_metrics["mAP50"],
            "mAP50_95": rgbd_metrics["mAP50_95"],
            "per_kelas_AP50": rgbd_classes,
            "weights": rgbd["weights"],
        },
        "delta": {
            "mAP50": round(rgbd_metrics["mAP50"] - base["mAP50"], 6),
            "mAP50_95": round(rgbd_metrics["mAP50_95"] - base["mAP50_95"], 6),
            "per_kelas_AP50": {
                k: round(rgbd_classes[k] - base_classes[k], 6)
                for k in base_classes
            },
        },
        "bootstrap": boot["bootstrap"],
        "protocol": boot["protocol"],
    }


def late_fusion_row(arch: str) -> dict[str, Any]:
    """Collect fixed late-fusion screens without discovering any test file."""
    result_path = RGBD_RESULTS / f"{arch}_late_fusion_val.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    item = read_json(result_path)
    boot_name = {
        "yolo26l": "yolo26l_union_wbf_bootstrap.json",
        "rtdetr_l": "rtdetr_l_union_nms_screen200_bootstrap.json",
    }.get(arch)
    bootstrap = read_json(RGBD_RESULTS / boot_name) if boot_name else None
    return {
        "arch": arch,
        "protocol": item["protocol"],
        "recipes": item["recipes"],
        "bootstrap_vs_rgb": bootstrap,
    }


def file_record(path: Path, label: str, remote: str | None = None) -> dict[str, Any]:
    item = {"label": label, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    if remote:
        item["hf_bucket_path"] = remote
    return item


def write_summary() -> dict[str, Any]:
    RGBD_RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    manifest_path = Path("/workspace/new763_rgbd4/MANIFEST.json")
    manifest = read_json(manifest_path) if manifest_path.exists() else None
    rows = [
        result_row(
            "yolo26l",
            "new763_yolo26l_rgbd4_val.json",
            "new763_yolo26l_rgbd4_val_bootstrap.json",
        ),
        result_row(
            "rfdetr_l",
            "new763_rfdetr_l_rgbd4_val.json",
            "new763_rfdetr_l_rgbd4_val_bootstrap.json",
        ),
        result_row(
            "rtdetr_l",
            "new763_rtdetr_l_rgbd4_val.json",
            "new763_rtdetr_l_rgbd4_val_bootstrap.json",
        ),
    ]

    yolo_pt = ROOT / "runs_new763_rgbd4/yolo26l_rgbd4_s42_i1280/weights/best.pt"
    rt_pt = ROOT / "runs_new763_rgbd4/rtdetr_l_rgbd4_s42_i1280_fair/weights/best.pt"
    # The first RF-DETR RGBD4 run is explicitly excluded: its lazy 3->4
    # expansion happened after PTL created the optimizer, leaving the depth
    # slice frozen at zero.  The v2 run expands the stem inside
    # configure_optimizers before parameter groups are built.
    rf_pth = ROOT / "runs_new763_rgbd4/rfdetr_l_rgbd4_s42_i1280_fair_v2/checkpoint_best_total.pth"
    rf_regular = ROOT / "runs_new763_rgbd4/rfdetr_l_rgbd4_s42_i1280_fair_v2/checkpoint_best_regular.pth"
    rt_seed1337 = ROOT / "runs/detect/runs_new763_rgbd4/rtdetr_l_rgbd4_s1337_i1280_from_e2_w8/weights/best.pt"
    generic_rf = Path("/workspace/model_artifacts/project-expertise/pretrained/rf-detr-large-2026.pth")

    dataset = None
    if manifest:
        dataset = {
            "dataset": manifest["dataset"],
            "channels": manifest["channels"],
            "image_shape": manifest["image_shape"],
            "disk_channel_order": manifest["disk_channel_order"],
            "model_channel_order": manifest["model_channel_order"],
            "depth_encoding": manifest["depth_encoding"],
            "depth_source": manifest["depth_source"],
            "splits": manifest["splits"],
            "output_validation": manifest["output_validation"],
            "splits_materialized": manifest["splits_materialized"],
            "test_policy": manifest["test_policy"],
        }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "new763 RGB versus RGB+D4; validation-only modality ablation",
        "dataset": dataset,
        "results": rows,
        "late_fusion": [late_fusion_row(arch) for arch in ("yolo26l", "rfdetr_l", "rtdetr_l")],
        "unrun_architectures": {},
        "initialization_audit": {
            "yolo26l": "generic /workspace/yolo26l.pt; RGB stem copied exactly; fourth stem channel zero at first initialization; resumed RGBD checkpoint preserves learned fourth channel",
            "rtdetr_l": "generic /workspace/rtdetr-l.pt; HGStem built with ch=4 before trainer optimizer; RGB stem copied exactly; fourth stem channel zero at first initialization; COCO head reinitialized to four classes",
            "rfdetr_l": "generic rf-detr-large-2026.pth; patch projection fourth channel zero at initialization; optimizer-safe 3->4 expansion before PTL parameter groups; COCO head reinitialized to four classes; v2 selected by framework validation mAP50:95; invalid v1 retained only in failed-run audit",
        },
        "model_artifacts": [
            file_record(
                yolo_pt,
                "YOLO26l RGBD4 best.pt",
                "hf://buckets/ULM-DS-Lab/project-expertise-backup/runs/new763_rgbd4/yolo26l_rgbd4_s42_i1280_best.pt",
            ),
            file_record(
                rt_pt,
                "RT-DETR-L RGBD4 best.pt",
                "hf://buckets/ULM-DS-Lab/project-expertise-backup/runs/new763_rgbd4/rtdetr_l_rgbd4_s42_i1280_fair_best.pt",
            ),
            file_record(
                rf_pth,
                "RF-DETR-L RGBD4 v2 best total (mAP50:95 selection)",
                "hf://buckets/ULM-DS-Lab/project-expertise-backup/runs/new763_rgbd4/rfdetr_l_rgbd4_s42_i1280_fair_v2_checkpoint_best_total.pth",
            ),
            file_record(rf_regular, "RF-DETR-L RGBD4 v2 best regular"),
            file_record(generic_rf, "RF-DETR-L generic initialization"),
        ],
        "protocol": {
            "evaluator": "pycocotools.COCOeval",
            "comparison": "same 468 validation images, same labels, paired image-level bootstrap",
            "bootstrap_resamples": 500,
            "seed": 42,
            "test_access": "forbidden; RGBD4 builder and evaluators contain no test path",
            "raw_rgbd4_dataset_committed": False,
            "large_checkpoints_committed": False,
            "excluded_runs": {
                "rfdetr_l_rgbd4_v1": "excluded: lazy 3->4 stem replacement occurred after optimizer creation, leaving the depth slice absent/frozen",
            },
        },
    }
    if rt_seed1337.is_file():
        payload["model_artifacts"].append(
            file_record(rt_seed1337, "RT-DETR-L RGBD4 seed 1337 fine-tune candidate")
        )
    (RGBD_RESULTS / "new763_rgbd4_summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    with (RGBD_RESULTS / "new763_rgbd4_summary.csv").open("w", newline="") as f:
        fields = [
            "arch", "rgb_mAP50", "rgbd4_mAP50", "delta_mAP50",
            "rgb_mAP50_95", "rgbd4_mAP50_95", "delta_mAP50_95",
            "delta_ci95_low", "delta_ci95_high", "fraction_delta_gt_zero",
            "significant_excludes_zero",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            b = row["baseline_rgb_val"]
            d = row["rgbd4_val"]
            boot = row["bootstrap"]
            writer.writerow({
                "arch": row["arch"],
                "rgb_mAP50": b["mAP50"],
                "rgbd4_mAP50": d["mAP50"],
                "delta_mAP50": row["delta"]["mAP50"],
                "rgb_mAP50_95": b["mAP50_95"],
                "rgbd4_mAP50_95": d["mAP50_95"],
                "delta_mAP50_95": row["delta"]["mAP50_95"],
                "delta_ci95_low": boot["delta_ci95"][0],
                "delta_ci95_high": boot["delta_ci95"][1],
                "fraction_delta_gt_zero": boot["fraction_delta_gt_zero"],
                "significant_excludes_zero": boot["significant_excludes_zero"],
            })
    return payload


def plot(payload: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = payload["results"]
    labels = [r["arch"] for r in rows]
    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ("mAP50", "mAP50_95"),
        ("Validation mAP50", "Validation mAP50:95"),
    ):
        rgb = [r["baseline_rgb_val"][metric] for r in rows]
        rgbd = [r["rgbd4_val"][metric] for r in rows]
        ax.bar(x - width / 2, rgb, width, label="RGB", color="#4C78A8")
        ax.bar(x + width / 2, rgbd, width, label="RGB+D4", color="#F58518")
        ax.set_title(title)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, max(rgb + rgbd) * 1.25)
        ax.grid(axis="y", alpha=0.25)
        for pos, val in zip(x - width / 2, rgb):
            ax.text(pos, val + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        for pos, val in zip(x + width / 2, rgbd):
            ax.text(pos, val + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].legend(frameon=False)
    fig.suptitle("new763 validation-only RGB versus four-channel RGB+D")
    fig.savefig(FIGURES / "new763_rgbd4_val_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    classes = list(rows[0]["delta"]["per_kelas_AP50"])
    x = np.arange(len(classes))
    center = (len(rows) - 1) / 2
    for i, row in enumerate(rows):
        vals = [row["delta"]["per_kelas_AP50"][c] for c in classes]
        ax.bar(x + (i - center) * width, vals, width, label=row["arch"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, classes)
    ax.set_ylabel("Δ AP50 (RGB+D4 − RGB)")
    ax.set_title("Per-class validation delta")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(FIGURES / "new763_rgbd4_per_class_delta.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot(write_summary())
    print("wrote new763 RGBD4 summary, CSV, and figures")
