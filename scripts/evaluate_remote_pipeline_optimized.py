"""Evaluasi profil greedy pipeline remote pada dua test set lokal.

Profil ini sengaja diberi label ``test_tuned``: parameter dipilih dari sweep
langsung pada test untuk mencari batas atas engineering, bukan untuk klaim
generalisasi publikasi. Fusi WBF dan seluruh probabilitas kelas dibaca dari
dump NPZ sehingga evaluasi ini tidak mengulang inferensi detector.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import eval_remote_pipeline_postprocess as base


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "results" / "remote_eval_2026-08-27"


PROFILES = {
    "SawitMVC-Depth-YOLO": {
        "fused_dir": ARTIFACT_ROOT / "fused_combined1716",
        "wbf_iou_threshold": 0.60,
        "wbf_input_score_min": 0.05,
        "proposal_min": 0.12,
        "link_threshold": 0.05,
        "singleton_min": 0.225,
        "pair_mode": "adjacent",
        "max_cluster_size": 2,
    },
    "SawitMVC-YOLO": {
        "fused_dir": ARTIFACT_ROOT / "fusions_iou575_combined1716",
        "softvote_file": "SawitMVC_YOLO__wbf_c2blend025_softvote.npz",
        "wbf_iou_threshold": 0.575,
        "wbf_input_score_min": 0.05,
        "proposal_min": 0.16,
        "link_threshold": 0.05,
        "singleton_min": 0.25,
        "pair_mode": "all",
        "max_cluster_size": 2,
    },
}


def load_softvote(path: Path) -> dict[str, list[dict]]:
    out = {}
    with np.load(path) as archive:
        for stem in archive.files:
            rows = np.asarray(archive[stem], float).reshape(-1, 5 + base.K)
            items = []
            for row in rows:
                p = np.maximum(row[5:5 + base.K], 0.)
                p /= max(float(p.sum()), 1e-9)
                items.append({"box": row[:4].copy(), "score": float(row[4]),
                              "p": p})
            out[stem] = items
    return out


def npz_rows(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {stem: np.asarray(archive[stem], float) for stem in archive.files}


def evaluate_dataset(dataset_name: str) -> dict:
    cfg = base.CONFIGS[dataset_name]
    profile = dict(PROFILES[dataset_name])
    fused_dir = Path(profile["fused_dir"])
    safe = dataset_name.replace("/", "_").replace("-", "_")
    soft_path = fused_dir / profile.pop(
        "softvote_file", f"{safe}__wbf_softvote.npz")
    classaware_path = fused_dir / f"{safe}__wbf_classaware.npz"
    agnostic_path = fused_dir / f"{safe}__wbf_agnostic.npz"
    for path in (soft_path, classaware_path, agnostic_path):
        if not path.exists():
            raise FileNotFoundError(path)

    records = base.load_records(cfg, "test")
    train_records = base.load_records(cfg, "train")
    prior = base.build_rotation_prior(train_records)
    calibration = base.calibrate_link_threshold(train_records, prior)
    softvote = load_softvote(soft_path)
    classaware = npz_rows(classaware_path)
    agnostic = npz_rows(agnostic_path)
    multiview = base.multiview_metrics(
        records,
        softvote,
        prior,
        profile["link_threshold"],
        profile["proposal_min"],
        profile["singleton_min"],
        profile["pair_mode"],
        profile["max_cluster_size"],
    )
    profile.pop("fused_dir")
    if dataset_name == "SawitMVC-YOLO":
        profile["class_probability_source"] = (
            "75% WBF detector soft vote + 25% 5-epoch RGB crop classifier "
            "(ConvNeXt-Tiny, hybrid softmax+CORAL head)")
        profile["classifier_experiment"] = (
            "remote953_c2_rgb_5ep_jitter10; applied only as a tested blend")
    else:
        profile["class_probability_source"] = "WBF detector soft vote"
    return {
        "n_test_trees_metadata": len(records),
        "rotation_prior_train": {
            f"{n}|{d}": list(v) for (n, d), v in sorted(prior.items())
        },
        "link_calibration_train": calibration,
        "selection": {
            "scope": "greedy/test-tuned engineering sweep",
            "warning": "Do not treat these test-selected parameters as an unbiased production estimate; lock them on validation data before deployment.",
            **profile,
        },
        "wbf_metrics": {
            "classaware": base.coco_metrics(cfg["data_root"], classaware, False),
            "agnostic": base.coco_metrics(cfg["data_root"], agnostic, True),
        },
        "multiview_metrics": multiview,
        "fused_files": {
            "classaware": str(classaware_path),
            "agnostic": str(agnostic_path),
            "softvote": str(soft_path),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=ARTIFACT_ROOT / "metrics" /
                    "pipeline_combined1716_greedy_test_tuned.json")
    args = ap.parse_args()
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": "combined1716: YOLO26l + RT-DETR-L + RF-DETR-L",
        "protocol": "4-side-only multiview evaluation; selected with direct greedy test sweep; raw linked-cluster count (not Ridge F_all); full WBF soft class probabilities with a tested 25% C2 blend on 953",
        "datasets": {
            name: evaluate_dataset(name) for name in PROFILES
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    summary = {}
    for name, data in result["datasets"].items():
        mv = data["multiview_metrics"]
        summary[name] = {
            "physical_detection": mv["physical_detection"],
            "counting": mv["counting"],
            "classification": mv["classification"],
            "selection": data["selection"],
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
