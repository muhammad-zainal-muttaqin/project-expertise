"""Latih RF-DETR-L 4-kanal (RGB+D) pada dataset SawitMVC-Depth.

Diadaptasi dari research-pipeline/experiments/code/train/train_rfdetr_4ch.py (E-022).
Tiga patch wajib:
  A — pemuat data: baca TIFF 4-kanal dari dataset 4ch
  B — normalisasi: tambah mean/std kanal ke-4
  C — conv patch-embed: inflasi 3→4 kanal, depth=0

Usage:
    python train_rfdetr_4ch.py \
        --dataset /workspace/rfdetr_ds_352_4ch \
        --depth-dir /workspace/depth_png_352 \
        --epochs 60 --resolution 1280 --batch 4 --grad-accum 4 \
        --output /workspace/project-expertise/runs/rfdetr_l_e60_i1280_rgbd352
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def statistik_depth_train(dataset_dir: Path) -> tuple[float, float]:
    """mean/std kanal depth (skala 0..1) dari split TRAIN saja."""
    train_img_dir = dataset_dir / "train" / "images"
    files = sorted(train_img_dir.glob("*.tiff")) + sorted(train_img_dir.glob("*.tif"))
    rng = np.random.default_rng(42)
    contoh = [files[i] for i in rng.choice(len(files), min(200, len(files)), replace=False)]
    nilai = []
    for p in contoh:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None and img.shape[2] >= 4:
            d = img[:, :, 3].astype(np.float32) / 255.0
            nilai.append(rng.choice(d.ravel(), 20000, replace=False))
    v = np.concatenate(nilai)
    return float(v.mean()), float(max(v.std(), 1e-3))


def patch_a_pemuat_tiff() -> None:
    """Pemuat data membaca TIFF 4-kanal [B,G,R,D] dan mengembalikan [R,G,B,D]."""
    from PIL import Image

    import rfdetr.datasets.yolo as yolo_mod

    def getitem_4ch(self, idx):
        sample = self._samples[idx]
        img = cv2.imread(str(sample.image_path), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3 or img.shape[2] < 4:
            with Image.open(sample.image_path) as pil_img:
                rgb = np.array(pil_img.convert("RGB"))
            d8 = np.zeros(rgb.shape[:2], np.uint8)
            bgrd = np.dstack([rgb[:, :, ::-1], d8])
        else:
            bgrd = img
        rgbd = np.dstack([bgrd[:, :, 2], bgrd[:, :, 1], bgrd[:, :, 0], bgrd[:, :, 3]])
        return sample.image_path, rgbd, sample.to_detections()

    yolo_mod._LazyYoloDetectionDataset.__getitem__ = getitem_4ch
    print("patch A: pemuat TIFF 4-kanal aktif (BGRD → RGBD)")


def patch_b_normalisasi(mean_d: float, std_d: float) -> None:
    import rfdetr.datasets.transforms as T

    asli_init = T.Normalize.__init__

    def init_4ch(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        if len(mean) == 3:
            mean = (*mean, mean_d)
            std = (*std, std_d)
        asli_init(self, mean, std)

    T.Normalize.__init__ = init_4ch
    print(f"patch B: normalisasi 4-kanal, mean_D={mean_d:.4f} std_D={std_d:.4f}")


def patch_c0_validasi_kanal() -> None:
    """Validasi kanal di PatchEmbeddings: inflasi 3→4 saat forward."""
    import rfdetr.models.backbone.dinov2_with_windowed_attn as dino

    if getattr(dino, "_4ch_patched", False):
        return
    dino._4ch_patched = True

    Kelas = dino.Dinov2WithRegistersPatchEmbeddings
    asli = Kelas.forward

    def forward_selaras(self, pixel_values):
        import torch.nn as nn
        c = pixel_values.shape[1]
        proj = self.projection
        if c == 4 and proj.in_channels == 3:
            import torch
            baru_conv = nn.Conv2d(4, proj.out_channels, kernel_size=proj.kernel_size,
                                  stride=proj.stride, padding=proj.padding,
                                  bias=proj.bias is not None)
            baru_conv = baru_conv.to(proj.weight.device, proj.weight.dtype)
            with torch.no_grad():
                baru_conv.weight[:, :3] = proj.weight
                baru_conv.weight[:, 3] = 0.0
                if proj.bias is not None:
                    baru_conv.bias.copy_(proj.bias)
            self.projection = baru_conv
            self.num_channels = 4
            print("patch C0: conv patch-embed 3→4 (depth kanal = 0)")
        self.num_channels = self.projection.in_channels
        return asli(self, pixel_values)

    Kelas.forward = forward_selaras
    print("patch C0: validasi kanal PatchEmbeddings aktif")


def patch_c_conv(model) -> None:
    """Timpa heuristik ubin-lalu-skala 0,75 dengan inflasi nol-kanal-4."""
    import torch
    import torch.nn as nn

    net = model.model.model if hasattr(model.model, "model") else model.model

    target = None
    for m in net.modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == 4:
            target = m
            break
    if target is None:
        print("patch C: PERINGATAN — conv 4-kanal tidak ditemukan (OK jika inflate via forward)")
        return
    with torch.no_grad():
        w = target.weight.detach().clone()
        pratlatih = w[:, :3] / 0.75
        target.weight[:, :3] = pratlatih
        target.weight[:, 3] = 0.0
    print(f"patch C: conv patch-embed {tuple(target.weight.shape)} dipulihkan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/workspace/rfdetr_ds_352_4ch")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--resolution", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="/workspace/project-expertise/runs/rfdetr_l_e60_i1280_rgbd352")
    args = ap.parse_args()

    patch_a_pemuat_tiff()
    patch_c0_validasi_kanal()
    mean_d, std_d = statistik_depth_train(Path(args.dataset))
    patch_b_normalisasi(mean_d, std_d)

    from rfdetr import RFDETRLarge
    model = RFDETRLarge(
        gradient_checkpointing=True,
        resolution=args.resolution,
        num_channels=4,
    )
    patch_c_conv(model)

    mulai = time.time()
    model.train(
        dataset_dir=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        seed=args.seed,
        multi_scale=False,
        expanded_scales=False,
        run_test=True,
    )
    durasi = time.time() - mulai

    meta = {
        "modal": "rgbd",
        "epochs": args.epochs,
        "resolution": args.resolution,
        "batch": args.batch,
        "grad_accum": args.grad_accum,
        "durasi_detik": round(durasi, 1),
    }
    (Path(args.output) / "hasil.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
