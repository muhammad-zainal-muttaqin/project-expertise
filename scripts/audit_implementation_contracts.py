"""Reproduce audit findings on synthetic inputs, without datasets or training.

This is a diagnostic of the current source, not a replacement evaluator.
Run with .venv-audit/bin/python scripts/audit_implementation_contracts.py.
No pretrained weights are downloaded. Historical metrics are left untouched.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torchvision
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment
from sklearn.utils.class_weight import compute_sample_weight

import eval_new763_pycoco as coco_eval
import eval_remote_pipeline_postprocess as base
import evaluate_remote_class_head as head
import evaluate_remote_count_reconciled as count
import sweep_remote_pipeline as sweep
import train_crop_classifier as crop

ROOT = Path(__file__).resolve().parents[1]


def detection(side=0, box=None, p=None, hp=None, identity="synthetic"):
    return {"side": side, "box": np.asarray(box or [0, 0, 10, 10], float),
            "score": .9, "p": np.asarray(p or [1, 0, 0, 0], float),
            "head_p": np.asarray(hp or [1, 0, 0, 0], float),
            "audit_identity": identity}


def association_probes():
    dets = [detection(side=s) for s in [0, 1, 2, 0]]
    groups = sweep.clusters(dets, [(.9, 0, 1), (.8, 1, 2), (.7, 2, 3)], .5, 0., 4)
    side_lists = [[d["side"] for d in g["members"]] for g in groups]
    assert all(len(sides) == len(set(sides)) for sides in side_lists)

    # Valid boxes, not an arbitrary IoU matrix that might be unrealizable.
    preds = [[1.5510445158382746, 0, 9.81466858266568, 1],
             [2.5588945280532913, 0, 14.876801775651957, 1]]
    truth = [[5.203047068022556, 0, 9.715917095505924, 1],
             [.34205806544546247, 0, 9.863732372221252, 1]]
    matrix = np.stack([base.iou_one(np.asarray(p), np.asarray(truth)) for p in preds])
    rec = {"bunches": [{"appearances": [{"side": 0, "box": b}]} for b in truth]}
    predicted = [{"members": [detection(box=b)]} for b in preds]
    legacy = count.tree_matches(rec, predicted)
    # Threshold cardinality first, IoU second; diagnostic reference only.
    valid = matrix >= .5
    utility = valid * (min(matrix.shape) + 1 + matrix)
    r, c = linear_sum_assignment(utility, maximize=True)
    reference = [(int(i), int(j)) for i, j in zip(r, c) if valid[i, j]]
    assert len(legacy) == 1 and len(reference) == 2

    # Swapping identity between views leaves this presence metric perfect.
    a, b = [0, 0, 10, 10], [20, 0, 30, 10]
    rec = {"bunches": [{"appearances": [{"side": s, "box": box} for s in [0, 1]]}
                       for box in [a, b]]}
    mixed = [{"members": [detection(0, a, identity="A"), detection(1, b, identity="B")]},
             {"members": [detection(0, b, identity="B"), detection(1, a, identity="A")]}]
    mixed_matches = count.tree_matches(rec, mixed)
    assert len(mixed_matches) == 2
    return {"side_constraint": {"groups": side_lists, "status": "fixed in e6bddc9"},
            "hungarian": {"pred_boxes": preds,
            "gt_boxes": truth, "iou": matrix.tolist(), "legacy": legacy,
            "cardinality_first_reference": reference},
            "mixed_identity": {"matched": len(mixed_matches), "predicted": 2,
                               "gt": 2, "identity_pure_clusters": 0}}


def confidence_probes():
    rows = np.asarray([[0, 0, 10, 10, .9, 0, 0],
                       [0, 0, 10, 10, .9, 0, 0],
                       [0, 0, 10, 10, .9, 1, 1]], float)
    regular = base.fuse_groups(rows, .5, 0., 2)[0]
    weighted = base.fuse_groups(rows, .5, 0., 2, np.ones(2))[0]
    assert np.isclose(regular["score"], .9) and np.isclose(weighted["score"], 1.35)
    y = np.r_[np.zeros(90, int), np.ones(10, int)]
    balanced = compute_sample_weight("balanced", y)
    combined = balanced * np.where(y == 1, 9., 1.)

    dets = [detection(p=[.9, .1, 0, 0], hp=[.25] * 4, identity="A"),
            detection(box=[20, 0, 30, 10], p=[.6, .4, 0, 0],
                      hp=[1, 0, 0, 0], identity="B")]
    selection = {}
    for mode in ["class_conf", "head_conf", "joint_conf", "head_conf_power_1.0"]:
        selected = count.selected_clusters(dets, [], .5, 0., 4, 1, mode)
        selection[mode] = selected[0]["members"][0]["audit_identity"]
    assert set(selection.values()) == {"A"}  # The actual head score prefers B.
    rec = {"views": {0: {"stem": "synthetic", "width": 40, "height": 20}}}
    bank = np.asarray([[0, 0, 10, 10, .9, 1, 0, 0, 0],
                       [20, 0, 30, 10, .9, 0, 1, 0, 0]], np.float32)
    mismatched = head.make_detections(rec, {"synthetic": bank},
                                     {"synthetic": bank[::-1].copy()}, 0.)
    assert np.argmax(mismatched[0]["p"]) != np.argmax(mismatched[0]["head_p"])
    return {"wbf": {"implicit_equal_weights_score": regular["score"],
                    "explicit_equal_weights_score": weighted["score"],
                    "class_vote": regular["p"].tolist()},
            "positive_negative_weight_ratio": {"balanced_only": balanced[-1] / balanced[0],
                                               "combined": combined[-1] / combined[0]},
            "rank_modes_selected": selection, "head_rank_reference": "B",
            "permuted_head_rows_accepted": True}


def empty_coco_probe():
    gt = COCO()
    gt.dataset = {"images": [{"id": 1, "width": 10, "height": 10}],
                  "categories": [{"id": 1, "name": "B1"}],
                  "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                                   "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0}]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        try:
            coco_eval.evaluate(gt, [])
        except IndexError as exc:
            return {"exception": type(exc).__name__, "message": str(exc)}
    raise AssertionError("Empty-prediction behavior changed; review this finding")


def c3_padding_probe():
    # Extract the two actual classes without importing scripts that load
    # project datasets. Replace only pretrained construction to avoid download.
    path = ROOT / "pipeline-pertandan/scripts/c3_multitampak.py"
    parsed = ast.parse(path.read_text())
    classes = [n for n in parsed.body if isinstance(n, ast.ClassDef)
               and n.name in {"Batang", "C3"}]
    proxy = SimpleNamespace(models=SimpleNamespace(
        resnet18=lambda **kw: torchvision.models.resnet18(weights=None),
        ResNet18_Weights=torchvision.models.ResNet18_Weights))
    namespace = {"torch": torch, "nn": nn, "torchvision": proxy}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    original = namespace["C3"]()
    short, padded = copy.deepcopy(original).train(), copy.deepcopy(original).train()
    for m in [short, padded]:
        for module in m.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.
    real = torch.randn(2, 1, 3, 32, 32)
    extra = torch.full((2, 5, 3, 32, 32), -2.)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, 0] = True
    old_mean = short.batang.b.bn1.running_mean.clone()
    with torch.no_grad():
        a = short(real, torch.ones(2, 1, dtype=torch.bool))
        b = padded(torch.cat([real, extra], dim=1), mask)
    delta = float((a - b).abs().max())
    bn_delta = float((short.batang.b.bn1.running_mean - old_mean).abs().max())
    assert delta > 1e-5 and bn_delta > 0
    assert not short.batang.b.bn1.weight.requires_grad
    # At eval, padding is masked and batch statistics no longer change.
    original.eval()
    with torch.no_grad():
        eval_delta = float((original(real, torch.ones(2, 1, dtype=torch.bool)) -
                            original(torch.cat([real, extra], dim=1), mask)).abs().max())
    return {"random_weights_no_training": True, "train_logit_max_abs_change": delta,
            "frozen_bn_running_mean_change": bn_delta,
            "eval_logit_max_abs_change": eval_delta,
            "scope": "ResNet C3 path; does not invalidate later cached-feature Set Transformer"}


def crop_channel_probe():
    # A red pixel read by OpenCV is [B,G,R]=[0,0,255]. Same byte order is
    # written by build_crop_dataset._kerja and passed through CropDS.
    bgr = np.zeros((1, 8, 8, 3), np.uint8)
    bgr[..., 2] = 255
    ds = crop.CropDS(bgr, None, np.full((1, 8, 8), 255, np.uint8),
                     np.array([0]), latih=False, pakai_depth=False)
    x, _, _ = ds[0]
    restored = x[:3, 0, 0] * torch.tensor([.229, .224, .225]) + torch.tensor([.485, .456, .406])
    assert torch.allclose(restored, torch.tensor([0., 0., 1.]), atol=1e-6)
    return {"red_pixel_channels_before_normalization": restored.tolist(),
            "expected_pretrained_rgb": [1., 0., 0.],
            "scope": "Fase 6 crop path; training/inference agree on BGR"}


def main():
    result = {"schema_version": 1, "data_access": "synthetic only",
              "association": association_probes(), "confidence": confidence_probes(),
              "empty_coco": empty_coco_probe(), "c3_padding": c3_padding_probe(),
              "crop_channels": crop_channel_probe()}
    paths = [Path(m.__file__) for m in [coco_eval, base, head, count, sweep, crop]]
    paths += [ROOT / "pipeline-pertandan/scripts/c3_multitampak.py",
              ROOT / "scripts/build_crop_dataset.py", Path(__file__)]
    result["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in paths}
    result["environment"] = {"torch": torch.__version__, "numpy": np.__version__,
                             "torchvision": torchvision.__version__}
    out = ROOT / "results/audit_2026-09-06/implementation_probes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
