"""Konversi checkpoint Lightning RF-DETR menjadi payload inference ``.pth``.

Run training menyimpan ``checkpoint_*.ckpt`` dengan state dict Lightning
(``model.<nama>``), sedangkan ``RFDETR.from_checkpoint`` mengharapkan payload
inference berisi state dict tanpa prefix. Metadata model dan urutan kelas
diambil dari checkpoint best yang dibuat library; bobot periodik tetap berasal
sepenuhnya dari file sumbernya.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def muat(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def ubah_state(d: dict) -> dict[str, object]:
    src = d.get("state_dict")
    if not isinstance(src, dict):
        raise ValueError("Checkpoint tidak memiliki state_dict Lightning")
    out = {k[len("model."):]: v for k, v in src.items()
           if k.startswith("model.")}
    if not out or "class_embed.weight" not in out:
        raise ValueError("State dict tidak tampak seperti model RF-DETR")
    return out


def konversi(src: Path, template: dict, dst: Path) -> None:
    d = muat(src)
    state = ubah_state(d)
    model = dict(template["model_config"])
    model["num_classes"] = int(state["class_embed.weight"].shape[0] - 1)
    model_state = {k: v for k, v in state.items()}
    payload = {
        "model": model_state,
        "state_dict": {"model." + k: v for k, v in model_state.items()},
        "args": template["args"],
        "model_config": model,
        "model_name": template.get("model_name", "RFDETRLarge"),
        "epoch": d.get("epoch", -1),
        "global_step": d.get("global_step", -1),
        "converted_from": str(src.resolve()),
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--template", type=Path, required=True,
                    help="checkpoint_best_ema.pth atau checkpoint_best_regular.pth")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    template = muat(args.template)
    if not {"args", "model_config", "model"} <= set(template):
        raise ValueError("Template bukan payload inference RF-DETR lengkap")
    sources = sorted(set(args.run.glob("checkpoint_*.ckpt")) |
                     ({args.run / "last.ckpt"} if (args.run / "last.ckpt").is_file()
                      else set()))
    if not sources:
        raise FileNotFoundError(f"Tidak ada checkpoint_*.ckpt di {args.run}")
    for src in sources:
        dst = src.with_name(src.stem + "_infer.pth")
        if dst.exists() and not args.overwrite:
            print(f"skip {dst}", flush=True)
            continue
        konversi(src, template, dst)
        print(f"{src.name} -> {dst.name}", flush=True)


if __name__ == "__main__":
    main()
