#!/usr/bin/env python3
"""Train/evaluate a new763 four-channel RGB+D detector on TRAIN/VALID only.

The corresponding RGB baselines are the existing new763 RGB runs.  This
script keeps the data split and recipe fixed, and refuses a dataset YAML that
contains a ``test`` entry.  It is deliberately a detector-only experiment:
the 4th channel is an aligned inverse-depth image, not a second label source.

Ultralytics boundary contract
-----------------------------
TIFFs are read by OpenCV as BGRD.  The stock Ultralytics code does not reverse
four-channel arrays (it only reverses 3-channel arrays), so this script patches
the *training* format transform to reverse exactly channels 0..2 and leave
channel 3 untouched.  It also keeps RGB color augmentation safe for RGBD and
uses zero depth for geometric padding.  The inference evaluator passes RGBD
arrays explicitly.

RF-DETR boundary contract
-------------------------
Its lazy YOLO reader is patched to return RGBD arrays, Normalize receives a
TRAIN-only depth mean/std, and its DINO patch embedding is expanded from 3->4
channels before PTL creates the optimizer (with a first-forward fallback).
The new channel starts at zero, preserving the trained RGB checkpoint exactly
at initialization while allowing depth to learn.

No test split is opened by this script.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_DATA = Path("/workspace/new763_rgbd4")
DEFAULT_PROJECT = Path("/workspace/project-expertise/runs_new763_rgbd4")
DEFAULT_WEIGHTS = {
    "yolo26l": "/workspace/yolo26l.pt",
    "rtdetr_l": "/workspace/rtdetr-l.pt",
    "rfdetr_l": "/workspace/model_artifacts/project-expertise/pretrained/rf-detr-large-2026.pth",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    if int(data.get("channels", 3)) != 4:
        raise ValueError(f"{path} must declare channels: 4")
    # A test key, even if unused, makes accidental test access too easy.
    if "test" in data:
        raise ValueError(f"{path} contains forbidden test entry for this validation-only run")
    return data


def validate_dataset(root: Path) -> Path:
    yaml_path = root / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(yaml_path)
    data = _read_yaml(yaml_path)
    for split in ("train", "valid"):
        split_dir = root / split
        image_dir = split_dir / "images"
        label_dir = split_dir / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"missing {split}/images or {split}/labels in {root}")
        images = sorted(image_dir.glob("*.tiff")) + sorted(image_dir.glob("*.tif"))
        if not images:
            raise FileNotFoundError(f"no 4-channel TIFFs in {image_dir}")
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in label_dir.glob("*.txt")}
        if image_stems != label_stems:
            raise ValueError(f"{split}: image/label stem mismatch ({len(image_stems)} vs {len(label_stems)})")
    return yaml_path


def _source_state_dict(source: Any) -> dict[str, Any]:
    """Extract a state dict from a trainer weight path or loaded module."""
    import torch
    import torch.nn as nn

    if isinstance(source, nn.Module):
        return source.state_dict()
    if isinstance(source, dict):
        if isinstance(source.get("model"), nn.Module):
            return source["model"].state_dict()
        if isinstance(source.get("state_dict"), dict):
            return source["state_dict"]
        if isinstance(source.get("model"), dict):
            return source["model"]
    path = Path(os.fspath(source))
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # compatibility with older torch
        checkpoint = torch.load(path, map_location="cpu")
    return _source_state_dict(checkpoint)


def _first_four_channel_conv(model):
    import torch.nn as nn

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and int(module.in_channels) == 4:
            return name, module
    raise RuntimeError("4-channel input convolution was not found in target model")


def _copy_rgb_into_four_channel_conv(model, source: Any) -> dict[str, Any]:
    """Initialize a 4-channel stem from RGB, or preserve it when resuming."""
    import torch

    name, target = _first_four_channel_conv(model)
    state = _source_state_dict(source)
    key = f"{name}.weight"
    if key not in state:
        candidates = [k for k, value in state.items() if k.endswith(key) and getattr(value, "ndim", 0) == 4]
        if not candidates:
            raise KeyError(f"cannot find RGB source stem {key}; candidates={list(state)[:5]}")
        key = candidates[0]
    source_weight = state[key]
    if tuple(source_weight.shape[0:1]) != tuple(target.weight.shape[0:1]) or tuple(source_weight.shape[2:]) != tuple(target.weight.shape[2:]):
        raise ValueError(f"stem shape mismatch source={tuple(source_weight.shape)} target={tuple(target.weight.shape)}")
    if source_weight.shape[1] < 3:
        raise ValueError(f"source stem has only {source_weight.shape[1]} input channels")
    with torch.no_grad():
        target.weight.zero_()
        target.weight[:, :3].copy_(source_weight[:, :3].to(target.weight.device, target.weight.dtype))
        # A resumed RGB+D checkpoint already contains a learned fourth
        # channel.  Preserve it; only a genuine 3-channel RGB source gets a
        # zero-initialized depth channel.
        depth_initialization = "zero"
        if int(source_weight.shape[1]) >= 4:
            target.weight[:, 3].copy_(source_weight[:, 3].to(target.weight.device, target.weight.dtype))
            depth_initialization = "checkpoint_preserved"
    return {
        "target_module": name,
        "source_key": key,
        "source_shape": list(source_weight.shape),
        "target_shape": list(target.weight.shape),
        "depth_initialization": depth_initialization,
    }


def patch_ultralytics_rgbd() -> None:
    """Make stock Ultralytics geometry/color transforms channel-safe."""
    import torch
    import ultralytics.data.augment as augment

    if getattr(augment, "_project_expertise_rgbd4_patched", False):
        return

    original_format = augment.Format._format_img

    def format_rgbd(self, img):
        if img.ndim >= 3 and img.shape[-1] == 4:
            chw = np.ascontiguousarray(img.transpose(2, 0, 1))
            # OpenCV disk order BGRD -> model order RGBD.
            chw = np.ascontiguousarray(chw[[2, 1, 0, 3]])
            return torch.from_numpy(chw)
        return original_format(self, img)

    augment.Format._format_img = format_rgbd

    original_hsv = augment.RandomHSV.apply_image

    def hsv_rgbd(self, labels, params=None):
        img = labels["img"]
        if img.ndim == 3 and img.shape[-1] == 4:
            color = np.ascontiguousarray(img[..., :3])
            temp = {"img": color}
            original_hsv(self, temp, params)
            img[..., :3] = temp["img"]
            labels["img"] = img
            return labels
        return original_hsv(self, labels, params)

    augment.RandomHSV.apply_image = hsv_rgbd

    original_letterbox = augment.LetterBox.apply_image

    def letterbox_rgbd(self, labels, params):
        result = original_letterbox(self, labels, params)
        img = result["img"]
        if img.ndim == 3 and img.shape[-1] == 4:
            top, bottom = int(params["top"]), int(params["bottom"])
            left, right = int(params["left"]), int(params["right"])
            # Padding RGB with 114 matches baseline; padding depth with 0
            # preserves the invalid-depth convention.
            if top:
                img[:top, :, 3] = 0
            if bottom:
                img[img.shape[0] - bottom :, :, 3] = 0
            if left:
                img[:, :left, 3] = 0
            if right:
                img[:, img.shape[1] - right :, 3] = 0
        return result

    augment.LetterBox.apply_image = letterbox_rgbd

    original_perspective = augment.RandomPerspective.apply_image

    def perspective_rgbd(self, labels, params=None):
        img = labels["img"]
        if img.ndim != 3 or img.shape[-1] != 4:
            return original_perspective(self, labels, params)
        M, size = params["M"], params["size"]
        if (size[0] != img.shape[1] or size[1] != img.shape[0]) or (M != np.eye(3)).any():
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=size, borderValue=(114, 114, 114, 0))
            else:
                img = cv2.warpAffine(img, M[:2], dsize=size, borderValue=(114, 114, 114, 0))
            if img.ndim == 2:
                img = img[..., None]
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels

    augment.RandomPerspective.apply_image = perspective_rgbd
    augment._project_expertise_rgbd4_patched = True
    print("ultralytics RGBD patch: Format BGRD->RGBD; safe HSV; invalid depth padding=0", flush=True)


def patch_ultralytics_trainers() -> None:
    """Force both large Ultralytics backbones to start from RGB weights + zero D."""
    from ultralytics.models.rtdetr.train import RTDETRTrainer
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    if not getattr(DetectionTrainer, "_project_expertise_rgbd4_patched", False):
        original_detection = DetectionTrainer.get_model

        def get_detection(self, cfg=None, weights=None, verbose=True):
            model = original_detection(self, cfg=cfg, weights=weights, verbose=verbose)
            report = _copy_rgb_into_four_channel_conv(model, weights)
            model._project_expertise_rgbd4_stem_report = report
            return model

        DetectionTrainer.get_model = get_detection
        DetectionTrainer._project_expertise_rgbd4_patched = True

    if not getattr(RTDETRTrainer, "_project_expertise_rgbd4_patched", False):
        original_rtdetr = RTDETRTrainer.get_model

        def get_rtdetr(self, cfg=None, weights=None, verbose=True):
            model = original_rtdetr(self, cfg=cfg, weights=weights, verbose=verbose)
            report = _copy_rgb_into_four_channel_conv(model, weights)
            model._project_expertise_rgbd4_stem_report = report
            return model

        RTDETRTrainer.get_model = get_rtdetr
        RTDETRTrainer._project_expertise_rgbd4_patched = True
    print("ultralytics RGBD patch: trainer stem initialization RGB exact + depth zero", flush=True)


def _depth_train_stats(dataset: Path, seed: int = 42, n_images: int = 200, n_pixels: int = 20_000) -> tuple[float, float, dict]:
    """Estimate Normalize statistics from a deterministic TRAIN-only sample."""
    files = sorted((dataset / "train" / "images").glob("*.tiff")) + sorted((dataset / "train" / "images").glob("*.tif"))
    if not files:
        raise FileNotFoundError("no train TIFFs for depth statistics")
    rng = np.random.default_rng(seed)
    chosen = files if len(files) <= n_images else [files[i] for i in rng.choice(len(files), n_images, replace=False)]
    samples = []
    valid_fraction = []
    for path in chosen:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3 or img.shape[2] != 4:
            raise ValueError(f"bad TRAIN TIFF {path}: {None if img is None else img.shape}")
        depth = img[..., 3].astype(np.float32) / 255.0
        valid_fraction.append(float((img[..., 3] > 0).mean()))
        flat = depth.reshape(-1)
        take = min(n_pixels, flat.size)
        samples.append(flat if take == flat.size else flat[rng.choice(flat.size, take, replace=False)])
    values = np.concatenate(samples)
    mean, std = float(values.mean()), float(max(values.std(), 1e-3))
    return mean, std, {
        "source_split": "train",
        "seed": seed,
        "n_images": len(chosen),
        "pixels_per_image_max": n_pixels,
        "sampled_values": int(values.size),
        "mean": mean,
        "std": std,
        "valid_fraction_mean": float(np.mean(valid_fraction)),
    }


def patch_rfdetr_loader() -> None:
    """Read BGRD TIFF and hand RF-DETR RGBD arrays."""
    import rfdetr.datasets.yolo as yolo_mod

    if getattr(yolo_mod._LazyYoloDetectionDataset, "_project_expertise_rgbd4_patched", False):
        return

    def getitem_4ch(self, idx):
        sample = self._samples[idx]
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] != 4:
            raise ValueError(f"expected BGRD TIFF at {sample.image_path}, got {None if image is None else image.shape}")
        rgbd = np.ascontiguousarray(image[..., [2, 1, 0, 3]])
        return sample.image_path, rgbd, sample.to_detections()

    yolo_mod._LazyYoloDetectionDataset.__getitem__ = getitem_4ch
    yolo_mod._LazyYoloDetectionDataset._project_expertise_rgbd4_patched = True
    print("rfdetr RGBD patch: lazy TIFF BGRD -> RGBD", flush=True)


def patch_rfdetr_normalize(mean_d: float, std_d: float) -> None:
    import rfdetr.datasets.transforms as transforms

    if getattr(transforms.Normalize, "_project_expertise_rgbd4_patched", False):
        return
    original = transforms.Normalize.__init__

    def init_4ch(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        if len(mean) == 3:
            mean = (*mean, mean_d)
            std = (*std, std_d)
        original(self, mean, std)

    transforms.Normalize.__init__ = init_4ch
    transforms.Normalize._project_expertise_rgbd4_patched = True
    print(f"rfdetr RGBD patch: Normalize mean_D={mean_d:.6f} std_D={std_d:.6f}", flush=True)


def _ensure_rfdetr_train_stem(model) -> dict[str, Any]:
    """Materialize RF-DETR's RGBD patch projection before optimizer creation.

    RF-DETR's public ``num_channels`` setting adapts the inference model, while
    the PTL training builder constructs the DINO patch projection as 3-channel
    and the first-forward adapter would replace that module lazily.  Replacing
    a parameter after ``configure_optimizers`` means the new depth weights are
    absent from the optimizer forever.  Expand the already-loaded RGB module
    in-place before the optimizer sees it, preserving the RGB weights exactly
    and adding a trainable zero-initialized depth slice.
    """
    import torch
    import torch.nn as nn

    root = getattr(model, "_orig_mod", model)
    try:
        patch_embeddings = root.backbone[0].encoder.encoder.embeddings.patch_embeddings
        projection = patch_embeddings.projection
    except AttributeError as exc:
        raise RuntimeError("RF-DETR DINO patch projection not found in the training model") from exc

    channels = int(projection.in_channels)
    if channels == 4:
        return {
            "module": "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection",
            "shape": list(projection.weight.shape),
            "depth_initialization": "already_4ch",
            "optimizer_safe": True,
        }
    if channels != 3:
        raise RuntimeError(f"RF-DETR training patch projection has unsupported input channels: {channels}")

    expanded = nn.Conv2d(
        4,
        projection.out_channels,
        kernel_size=projection.kernel_size,
        stride=projection.stride,
        padding=projection.padding,
        dilation=projection.dilation,
        groups=projection.groups,
        bias=projection.bias is not None,
        padding_mode=projection.padding_mode,
    ).to(device=projection.weight.device, dtype=projection.weight.dtype)
    with torch.no_grad():
        expanded.weight.zero_()
        expanded.weight[:, :3].copy_(projection.weight)
        if projection.bias is not None and expanded.bias is not None:
            expanded.bias.copy_(projection.bias)
    expanded.weight.requires_grad = projection.weight.requires_grad
    if projection.bias is not None and expanded.bias is not None:
        expanded.bias.requires_grad = projection.bias.requires_grad
    patch_embeddings.projection = expanded
    patch_embeddings.num_channels = 4
    return {
        "module": "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection",
        "shape": list(expanded.weight.shape),
        "depth_initialization": "zero_trainable",
        "optimizer_safe": True,
    }


def patch_rfdetr_train_stem() -> None:
    """Ensure the learned RF-DETR depth slice exists before PTL builds AdamW."""
    from rfdetr.training.module_model import RFDETRModelModule

    if getattr(RFDETRModelModule, "_project_expertise_rgbd4_train_stem_patched", False):
        return
    original = RFDETRModelModule.configure_optimizers

    def configure_with_rgbd_stem(self):
        report = _ensure_rfdetr_train_stem(self.model)
        self._project_expertise_rgbd4_train_stem_report = report
        return original(self)

    RFDETRModelModule.configure_optimizers = configure_with_rgbd_stem
    RFDETRModelModule._project_expertise_rgbd4_train_stem_patched = True
    print("rfdetr RGBD patch: materialize 4-channel stem before optimizer creation", flush=True)


def patch_rfdetr_backbone() -> None:
    """Adapt RF-DETR's 3-channel patch projection if a forward sees RGBD first."""
    import torch
    import torch.nn as nn
    import rfdetr.models.backbone.dinov2_with_windowed_attn as dino

    if getattr(dino, "_project_expertise_rgbd4_patched", False):
        return
    cls = dino.Dinov2WithRegistersPatchEmbeddings
    original = cls.forward

    def forward_rgbd(self, pixel_values):
        projection = self.projection
        channels = int(pixel_values.shape[1])
        if channels == 4 and int(projection.in_channels) == 3:
            expanded = nn.Conv2d(
                4,
                projection.out_channels,
                kernel_size=projection.kernel_size,
                stride=projection.stride,
                padding=projection.padding,
                bias=projection.bias is not None,
            ).to(projection.weight.device, projection.weight.dtype)
            with torch.no_grad():
                expanded.weight[:, :3].copy_(projection.weight)
                expanded.weight[:, 3].zero_()
                if projection.bias is not None:
                    expanded.bias.copy_(projection.bias)
            self.projection = expanded
            self.num_channels = 4
        elif channels != int(self.projection.in_channels):
            raise ValueError(f"RF-DETR patch embed expects {self.projection.in_channels}, got {channels}")
        return original(self, pixel_values)

    cls.forward = forward_rgbd
    dino._project_expertise_rgbd4_patched = True
    print("rfdetr RGBD patch: patch embedding expands 3->4 with depth=0", flush=True)


def _patch_rfdetr_inference_stem(model) -> dict[str, Any]:
    """Undo RF-DETR's default 3/4 weight tiling for inference."""
    import torch
    import torch.nn as nn

    network = model.model.model if hasattr(model.model, "model") else model.model
    for name, module in network.named_modules():
        if isinstance(module, nn.Conv2d) and int(module.in_channels) == 4:
            with torch.no_grad():
                # rfdetr._adapt_input_conv scales tiled RGB weights by 3/4.
                module.weight[:, :3].div_(0.75)
                module.weight[:, 3].zero_()
            return {"module": name, "shape": list(module.weight.shape), "depth_initialization": "zero"}
    raise RuntimeError("RF-DETR inference 4-channel patch projection not found")


def _rgbd_from_tiff(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"bad RGBD TIFF {path}: {None if image is None else image.shape}")
    return np.ascontiguousarray(image[..., [2, 1, 0, 3]])


def smoke_ultralytics(args: argparse.Namespace, data_yaml: Path) -> dict[str, Any]:
    """Build a real 4-channel target and run one GPU forward pass."""
    import torch
    from ultralytics.data.dataset import YOLODataset
    from ultralytics import RTDETR, YOLO
    from ultralytics.nn.tasks import DetectionModel, RTDETRDetectionModel
    import yaml

    patch_ultralytics_rgbd()
    raw_data = yaml.safe_load(data_yaml.read_text())
    ds = YOLODataset(
        img_path=str(args.data / "valid" / "images"),
        data=raw_data,
        imgsz=min(args.imgsz, 256),
        augment=False,
        batch_size=1,
        stride=32,
        pad=0.5,
    )
    sample = ds[0]
    image = sample["img"]
    if tuple(image.shape[:1]) != (4,):
        raise RuntimeError(f"Ultralytics loader returned {tuple(image.shape)}, not 4 channels")
    wrapper = YOLO(args.weights) if args.arch == "yolo26l" else RTDETR(args.weights)
    source = wrapper.model
    target_cls = DetectionModel if args.arch == "yolo26l" else RTDETRDetectionModel
    target = target_cls(source.yaml, nc=4, ch=4, verbose=False)
    init = _copy_rgb_into_four_channel_conv(target, source)
    target.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target.to(device)
    x = image.unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
    with torch.inference_mode():
        output = target(x)
    return {
        "arch": args.arch,
        "loader_shape_chw": list(image.shape),
        "loader_dtype": str(image.dtype),
        "target_device": str(device),
        "forward": "pass",
        "stem": init,
        "output_type": type(output).__name__,
    }


def smoke_rfdetr(args: argparse.Namespace, data_yaml: Path) -> dict[str, Any]:
    mean_d, std_d, stats = _depth_train_stats(args.data, args.seed)
    patch_rfdetr_loader()
    patch_rfdetr_normalize(mean_d, std_d)
    patch_rfdetr_backbone()
    from rfdetr import RFDETRLarge

    model = RFDETRLarge(
        gradient_checkpointing=False,
        resolution=args.imgsz,
        num_channels=4,
        num_classes=4,
        pretrain_weights=str(args.weights),
    )
    stem = _patch_rfdetr_inference_stem(model)
    model.means = [0.485, 0.456, 0.406, mean_d]
    model.stds = [0.229, 0.224, 0.225, std_d]
    path = sorted((args.data / "valid" / "images").glob("*.tiff"))[0]
    det = model.predict(_rgbd_from_tiff(path), threshold=0.001)
    return {"arch": args.arch, "forward": "pass", "prediction_type": type(det).__name__, "stem": stem, "depth_stats": stats}


def train_ultralytics(args: argparse.Namespace, data_yaml: Path) -> dict[str, Any]:
    patch_ultralytics_rgbd()
    patch_ultralytics_trainers()
    if args.arch == "yolo26l":
        from ultralytics import YOLO

        model = YOLO(str(args.weights))
    else:
        from ultralytics import RTDETR

        model = RTDETR(str(args.weights))
    epochs = args.epochs if args.epochs is not None else 60
    patience = args.patience if args.patience is not None else 15
    started = time.time()
    train_kwargs = dict(
        data=str(data_yaml),
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        epochs=epochs,
        patience=patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        cos_lr=True,
        optimizer="auto",
        warmup_epochs=3.0,
        close_mosaic=7,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.10,
        scale=0.50,
        fliplr=0.5,
        val=True,
        plots=False,
        save=True,
        save_period=5,
        cache=False,
        device=0,
    )
    if args.resume:
        # Ultralytics restores optimizer, EMA, scheduler, and the start epoch
        # from last.pt.  This is intentionally supported only for the
        # Ultralytics runs; RF-DETR has a separate checkpoint format/API.
        train_kwargs["resume"] = True
    model.train(**train_kwargs)
    return {
        "framework": "ultralytics",
        "arch": args.arch,
        "epochs_requested": epochs,
        "patience": patience,
        "elapsed_seconds": round(time.time() - started, 1),
        "initialization": (
            "resume from RGB+D checkpoint; fourth stem channel preserved"
            if args.resume
            else "same generic RGB pretrained weights as RGB baseline; first depth stem channel zero; COCO head reinitialized"
        ),
        "resume": bool(args.resume),
    }


def train_rfdetr(args: argparse.Namespace, data_yaml: Path) -> dict[str, Any]:
    mean_d, std_d, stats = _depth_train_stats(args.data, args.seed)
    patch_rfdetr_loader()
    patch_rfdetr_normalize(mean_d, std_d)
    patch_rfdetr_backbone()
    patch_rfdetr_train_stem()
    from rfdetr import RFDETRLarge

    out = args.project / args.name
    epochs = args.epochs if args.epochs is not None else 20
    patience = args.patience if args.patience is not None else 5
    model = RFDETRLarge(
        gradient_checkpointing=True,
        resolution=args.imgsz,
        num_channels=4,
        num_classes=4,
        pretrain_weights=str(args.weights),
    )
    inference_stem = _patch_rfdetr_inference_stem(model)
    model.means = [0.485, 0.456, 0.406, mean_d]
    model.stds = [0.229, 0.224, 0.225, std_d]
    started = time.time()
    model.train(
        dataset_dir=str(args.data),
        output_dir=str(out),
        epochs=epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        lr=1e-4,
        lr_scheduler="cosine",
        lr_min_factor=0.01,
        warmup_epochs=1.0,
        seed=args.seed,
        early_stopping=True,
        early_stopping_patience=patience,
        early_stopping_min_delta=0.001,
        multi_scale=False,
        expanded_scales=False,
        checkpoint_interval=1,
        run_test=False,
        tensorboard=False,
        num_workers=args.workers,
        progress_bar="tqdm",
        device="cuda",
        notes={
            "dataset": "new763 RGB+D; train/valid only",
            "depth_normalization": stats,
            "rgb_initialization": (
                "resume from RGB+D checkpoint; patch embed depth preserved"
                if args.resume
                else "same generic RF-DETR pretrain as RGB baseline; patch embed depth=0; COCO head reinitialized"
            ),
            "inference_stem": inference_stem,
            "train_stem_optimizer_guard": "configure_optimizers expands 3->4 before parameter groups",
        },
    )
    return {
        "framework": "rfdetr",
        "arch": args.arch,
        "epochs_requested": epochs,
        "patience": patience,
        "elapsed_seconds": round(time.time() - started, 1),
        "depth_stats": stats,
        "inference_stem": inference_stem,
        "initialization": (
            "resume from RGB+D checkpoint; patch embed depth preserved"
            if args.resume
            else "same generic RF-DETR pretrain as RGB baseline; patch embed depth channel zero; COCO head reinitialized"
        ),
    }


def main() -> int:
    # OpenCV emits one harmless ExtraSamples warning per TIFF.  Silence that
    # metadata warning so dataloader workers do not flood the training log.
    if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
        cv2.utils.logging.setLogLevel(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=("yolo26l", "rtdetr_l", "rfdetr_l"), required=True)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="resume an Ultralytics RGB+D checkpoint")
    ap.add_argument("--smoke", action="store_true", help="loader + one forward only; no training")
    args = ap.parse_args()
    if args.resume and args.arch == "rfdetr_l":
        raise ValueError("--resume is currently supported only for yolo26l and rtdetr_l")
    args.project.mkdir(parents=True, exist_ok=True)
    data_yaml = validate_dataset(args.data)
    if not args.weights:
        args.weights = DEFAULT_WEIGHTS[args.arch]
    if not Path(args.weights).is_file():
        raise FileNotFoundError(f"weights not found: {args.weights}")

    if args.smoke:
        result = smoke_rfdetr(args, data_yaml) if args.arch == "rfdetr_l" else smoke_ultralytics(args, data_yaml)
        path = args.project / f"{args.name}_smoke.json"
    else:
        result = train_rfdetr(args, data_yaml) if args.arch == "rfdetr_l" else train_ultralytics(args, data_yaml)
        path = args.project / args.name / "rgbd4_train_meta.json"
        path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(args.data.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "test_access": "forbidden; data.yaml has no test entry",
        "weights": str(Path(args.weights).resolve()),
        "seed": args.seed,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "args": vars(args) | {"data": str(args.data), "project": str(args.project)},
        "result": result,
    }
    path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(meta, indent=2, sort_keys=True, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
