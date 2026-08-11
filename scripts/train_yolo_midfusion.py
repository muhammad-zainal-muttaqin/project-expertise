"""Latih YOLO26l dengan mid-fusion depth + gate non-zero-init — Fase 5 lever arsitektur.

BEDA MENDASAR dari early fusion (V2-E-005/E-022/E-027, semua GAGAL): kanal
depth TIDAK di-concat ke input conv pertama. Stem RGB tetap PERSIS bobot
pratlatih COCO 3-kanal, tidak pernah disentuh. Kanal depth dialihkan ke
cabang terpisah kecil, lalu fitur-nya dijumlahkan (gated) ke feature map
backbone di titik menengah (P3/8, layer index 4 pada yolo26.yaml — level
resolusi yang jadi input head "small object", relevan untuk B4).

Kenapa desain ini, bukan yang lain (lihat docs/RENCANA.md Fase 5 + CLAUDE.md
"Hal yang sudah dicoba dan GAGAL"):
  - Early fusion naif (E-022/E-027/V2-E-005) rusak karena conv pertama harus
    belajar dari nol memakai kanal depth dalam 352 pohon/60 epoch, sambil
    ikut mengganggu bobot RGB pratlatih yang sudah bagus. Desain ini
    membiarkan bobot RGB pratlatih 100% utuh; depth murni aditif.
  - Gate init-nol (F-007) bikin CABANG SAMPING tidak pernah dapat gradien
    (dikali gate=0) DAN gradien gate sendiri jadi derau murni (fitur depth
    dari cabang yang juga tidak belajar) -- deadlock. Gate di sini
    diinisialisasi kecil-taknol (default 0.02): cabang depth dapat gradien
    riil sejak awal, dan gradien gate sendiri jadi sinyal informatif
    (apakah menambah fitur depth membantu atau merugikan loss), bukan derau.
  - Mitigasi tambahan (menu F-007, "inisialisasi kecil-taknol"): conv
    terakhir cabang depth diinisialisasi dengan skala kecil (0.1x default)
    supaya fitur depth mulai berorde-kecil relatif ke feature map RGB yang
    dijumlahkan -- fusi mulai konservatif, bukan warmup freeze/unfreeze
    terpisah (lebih sederhana, menghindari risiko cabang beku menyuntik
    derau tetap ke backbone selama warmup).
  - Titik fusi menengah (bukan input, bukan paling akhir) selaras sinyal
    indikatif E-032 (mid-fusion 3/3 seed positif, CI masih memuat nol) dan
    sapuan 28 titik fusi Ophoff dkk. (dikutip Research-Pipeline/CLAUDE.md:
    early fusion konsisten lebih buruk dari mid).

Mekanisme (diverifikasi langsung dari source ultralytics==8.4.103 terpasang):
  - `DetectionTrainer.get_model` (ultralytics/models/yolo/detect/train.py)
    membangun `DetectionModel(cfg, nc=.., ch=self.data["channels"])` -- yaml
    data kita channels:4, jadi TANPA patch ini ultralytics akan mencoba
    membangun/menyesuaikan conv pertama ke 4 kanal (pola early-fusion yang
    justru ingin dihindari). Di-patch supaya model SELALU dibangun ch=3
    (stem asli, load bobot pratlatih bersih tanpa mismatch shape sama
    sekali), lalu cabang depth+gate ditempel sebagai submodul terpisah.
  - `BaseModel.forward -> predict -> _predict_once` (ultralytics/nn/tasks.py)
    adalah SATU-SATUNYA jalur forward yang dipakai baik untuk loss training
    maupun inference/validasi (diverifikasi: `loss()` memanggil
    `self.forward(batch["img"])`). Di-patch di LEVEL CLASS (bukan per-instance
    via `types.MethodType` -- sudah dicoba, GAGAL: `Trainer.final_eval()`
    me-reload model dari checkpoint lewat `AutoBackend`, dan method yang
    di-bind ke satu instance tidak ikut "nempel" ke objek hasil reload,
    walau `depth_branch`/`gate` submodul/parameter biasa yang memang selalu
    ikut tersimpan). Versi class-level cek `hasattr(self, "depth_branch")`
    sehingga aman dipasang sekali per proses tanpa mengganggu model lain.
  - Loader dataset TIDAK diubah -- dataset 4-kanal TIFF yang sama dengan
    early fusion (`data_rgbd_352.yaml`) dipakai apa adanya; kanal ke-4 hanya
    "dibaca beda" oleh model, bukan oleh data pipeline.

Usage:
    .venv/bin/python train_yolo_midfusion.py \
        --data /workspace/SawitMVC-Depth-4ch/data_rgbd_352.yaml \
        --fuse-at 4 --gate-init 0.02 \
        --epochs 15 --patience 3 \
        --name yolo26l_screening_midfusion352
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn


class DepthBranch(nn.Module):
    """Cabang kecil yang HANYA memproses kanal depth (1 kanal), stride-8.

    3 conv stride-2 (1->16->32->C_fuse). Conv terakhir diinisialisasi skala
    kecil (0.1x default) supaya fitur depth mulai berorde-kecil -- fusi
    konservatif di awal, selaras pelajaran F-007 ("inisialisasi kecil-taknol").
    """

    def __init__(self, c_fuse: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.SiLU(inplace=True),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.SiLU(inplace=True),
            nn.Conv2d(32, c_fuse, 3, 2, 1), nn.BatchNorm2d(c_fuse),
        )
        last_conv = self.net[6]
        with torch.no_grad():
            last_conv.weight.mul_(0.1)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.net(depth)


@torch.no_grad()
def discover_fuse_shape(model, fuse_at: int, imgsz: int = 256) -> tuple[int, int]:
    """Dummy forward (ch=3, ukuran kecil) untuk menemukan (channels, stride)
    riil di layer `fuse_at` -- robust terhadap scale (n/s/m/l/x) tanpa hardcode.
    """
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, imgsz, imgsz, device=device)
    y = []
    out_shape = None
    for m in model.model:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x if m.i in model.save else None)
        if m.i == fuse_at:
            out_shape = x.shape
            break
    if out_shape is None:
        raise ValueError(f"fuse_at={fuse_at} tidak ditemukan di model.model")
    c_fuse = out_shape[1]
    stride = imgsz // out_shape[-1]
    return c_fuse, stride


def _predict_once_midfusion(self, x, profile=False, visualize=False, embed=None):
    """Pengganti BaseModel._predict_once di LEVEL CLASS (bukan instance).

    Kenapa level class, bukan `types.MethodType` per-instance (percobaan
    pertama, GAGAL): ultralytics me-reload model dari checkpoint di beberapa
    titik terpisah dari training (mis. `Trainer.final_eval()` lewat
    `AutoBackend`) -- direkonstruksi ulang, bukan objek Python yang sama
    persis. `depth_branch`/`gate`/`_fuse_at` (submodul & parameter
    ter-registrasi biasa) IKUT tersimpan di checkpoint dengan benar, tapi
    method yang di-bind ke SATU instance via `types.MethodType` tidak ikut
    "nempel" ke objek hasil reload itu -- terverifikasi lewat crash nyata:
    validasi-akhir sempat menabrak conv 3-kanal dengan input 4-kanal karena
    forward yang dipakai kembali ke default class, bukan yang di-patch.
    Patch di level class + cek `hasattr(self, "depth_branch")` di sini
    bekerja utuh berapa pun kali model itu dikonstruksi ulang, selama
    checkpoint-nya membawa depth_branch/gate/_fuse_at (yang memang selalu
    ikut, bagian normal dari state model).

    Replika persis BaseModel._predict_once asli (ultralytics/nn/tasks.py:173-201)
    dengan satu tambahan: fusi aditif ber-gate setelah layer `self._fuse_at`.
    Kanal 0-2 (RGB) jalan lewat backbone/head ASLI tanpa modifikasi apa pun;
    kanal 3 (depth) dipisah sebelum layer 0, tidak pernah masuk stem asli.
    """
    if not (hasattr(self, "depth_branch") and x.shape[1] == 4):
        return self._predict_once_orig(x, profile, visualize, embed)

    rgb = x[:, :3]
    depth_feat = self.depth_branch(x[:, 3:4])
    fuse_at = self._fuse_at
    gate = self.gate

    y, dt, embeddings = [], [], []
    embed_set = frozenset(embed) if embed else {-1}
    max_idx = max(embed_set)
    xi = rgb
    for m in self.model:
        if m.f != -1:
            xi = y[m.f] if isinstance(m.f, int) else [xi if j == -1 else y[j] for j in m.f]
        if profile:
            self._profile_one_layer(m, xi, dt)
        xi = m(xi)
        if m.i == fuse_at:
            xi = xi + gate * depth_feat
        y.append(xi if m.i in self.save else None)
        if m.i in embed_set:
            embeddings.append(torch.nn.functional.adaptive_avg_pool2d(xi, (1, 1)).squeeze(-1).squeeze(-1))
            if m.i == max_idx:
                return torch.unbind(torch.cat(embeddings, 1), dim=0)
    return xi


def patch_midfusion_class() -> None:
    """Tempel _predict_once_midfusion ke BaseModel SEKALI per proses (idempoten)."""
    from ultralytics.nn.tasks import BaseModel

    if getattr(BaseModel, "_midfusion_class_patched", False):
        return
    BaseModel._predict_once_orig = BaseModel._predict_once
    BaseModel._predict_once = _predict_once_midfusion
    BaseModel._midfusion_class_patched = True


def patch_midfusion_trainer(fuse_at: int, gate_init: float) -> None:
    import ultralytics.models.yolo.detect.train as train_mod
    from ultralytics.nn.tasks import DetectionModel

    patch_midfusion_class()

    if getattr(train_mod, "_midfusion_patched", False):
        return

    def get_model_midfusion(self, cfg=None, weights=None, verbose=True):
        # ch=3 dipaksa (BUKAN self.data["channels"]=4) -- stem tetap 3-kanal,
        # bobot pratlatih load bersih tanpa shape mismatch/inflasi sama sekali.
        model = self.set_model_names_for_load(
            DetectionModel(cfg, nc=self.data["nc"], ch=3, verbose=verbose)
        )
        if weights:
            model.load(weights)

        c_fuse, stride = discover_fuse_shape(model, fuse_at)
        depth_branch = DepthBranch(c_fuse)
        gate = nn.Parameter(torch.tensor(float(gate_init)))
        model.depth_branch = depth_branch
        model.gate = gate
        model._fuse_at = fuse_at
        print(f"mid-fusion: fuse_at={fuse_at} (stride {stride}, {c_fuse} kanal), "
              f"gate init={gate_init}, depth_branch {sum(p.numel() for p in depth_branch.parameters())} param")
        return model

    train_mod.DetectionTrainer.get_model = get_model_midfusion
    train_mod._midfusion_patched = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/SawitMVC-Depth-4ch/data_rgbd_352.yaml")
    ap.add_argument("--fuse-at", type=int, default=4, help="index layer backbone tempat fusi (P3/8 default)")
    ap.add_argument("--gate-init", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="yolo26l_midfusion352")
    ap.add_argument("--project", default="/workspace/project-expertise/runs")
    ap.add_argument("--weights", default="yolo26l.pt")
    args = ap.parse_args()

    patch_midfusion_trainer(args.fuse_at, args.gate_init)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    mulai = time.time()
    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        cos_lr=True,
        project=args.project,
        name=args.name,
    )
    durasi = time.time() - mulai

    gate_final = float(model.model.gate.detach().cpu())
    meta = {
        "modal": "rgbd_midfusion",
        "fuse_at": args.fuse_at,
        "gate_init": args.gate_init,
        "gate_final": gate_final,
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "durasi_detik": round(durasi, 1),
    }
    out_dir = Path(args.project) / args.name
    (out_dir / "hasil.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
