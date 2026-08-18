"""Classifier visual residual khusus DAMIMAS: crop + fitur tangan + skor C1.

Kepala residual diinisialisasi nol sehingga prediksi awal identik dengan C1;
model hanya perlu belajar koreksi yang didukung piksel crop. Backbone ImageNet
dilatih dengan learning-rate kecil, sedangkan kepala memakai learning-rate
lebih besar. Checkpoint dipilih hanya dari metrik per-tandan VAL.

Kotak dan tautan yang digunakan di sini adalah GT. Karena itu hasilnya adalah
mutu modul classifier/oracle, bukan end-to-end. Dump TEST baru dibuat sesudah
checkpoint dan aturan agregasi terkunci di VAL.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


SUB = Path(__file__).resolve().parents[1]
KELAS = ("B1", "B2", "B3", "B4")
sys.path.insert(0, str(Path(__file__).parent))
import classifier_klasik_damimas as CK  # noqa: E402
import stacker_damimas as SD  # noqa: E402


class DataCrop(Dataset):
    def __init__(self, img, crop_idx, aux, p_c1, y):
        self.img, self.crop_idx = img, np.asarray(crop_idx, int)
        self.aux, self.p_c1, self.y = aux, p_c1, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # Cache dibuat OpenCV (BGR); tensor dibalik ke RGB di GPU.
        return (torch.from_numpy(self.img[self.crop_idx[i]].copy()).permute(2, 0, 1),
                torch.from_numpy(self.aux[i]), torch.from_numpy(self.p_c1[i]),
                int(self.y[i]), i)


class Hibrida(nn.Module):
    def __init__(self, dim_aux: int, backbone: str = "convnext_tiny",
                 mode_c1: str = "residual"):
        super().__init__()
        if backbone == "convnext_tiny":
            b = torchvision.models.convnext_tiny(
                weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            b.classifier[2] = nn.Identity(); d_backbone = 768
            beku = (0, 1)
        elif backbone == "convnext_base":
            b = torchvision.models.convnext_base(
                weights=torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
            b.classifier[2] = nn.Identity(); d_backbone = 1024
            beku = (0, 1)
        elif backbone == "efficientnet_v2_s":
            b = torchvision.models.efficientnet_v2_s(
                weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            b.classifier[1] = nn.Identity(); d_backbone = 1280
            beku = (0, 1)
        elif backbone == "swin_t":
            b = torchvision.models.swin_t(
                weights=torchvision.models.Swin_T_Weights.IMAGENET1K_V1)
            b.head = nn.Identity(); d_backbone = 768
            beku = (0, 1)
        else:
            raise ValueError(f"Backbone tidak dikenal: {backbone}")
        # Bekukan stem + stage pertama; warna domain tetap boleh mengadaptasi
        # seluruh stage lebih tinggi.
        for i in beku:
            for p in b.features[i].parameters():
                p.requires_grad = False
        self.backbone = b
        self.nama_backbone = backbone
        self.mode_c1 = mode_c1
        self.d_fitur = d_backbone + 128
        self.aux = nn.Sequential(nn.LayerNorm(dim_aux), nn.Linear(dim_aux, 128),
                                 nn.GELU(), nn.Dropout(.15))
        self.residual = nn.Sequential(
            nn.LayerNorm(self.d_fitur), nn.Dropout(.25),
            nn.Linear(self.d_fitur, 256), nn.GELU(), nn.Dropout(.20),
            nn.Linear(256, 4),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.log_skala_c1 = nn.Parameter(torch.zeros(()))

    def ekstrak(self, x, aux):
        return torch.cat([self.backbone(x), self.aux(aux)], 1)

    def logit_dari_fitur(self, z, p_c1):
        if self.mode_c1 == "bebas":
            return self.residual(z)
        skala = self.log_skala_c1.exp().clamp(.25, 4.)
        return skala * torch.log(p_c1.clamp_min(1e-6)) + self.residual(z)

    def forward(self, x, aux, p_c1):
        return self.logit_dari_fitur(self.ekstrak(x, aux), p_c1)


MEAN = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
STD = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)


def olah_img(x: torch.Tensor, latih: bool, ukuran: int) -> torch.Tensor:
    x = x.cuda(non_blocking=True).float()[:, [2, 1, 0]] / 255.
    if latih:
        flip = torch.rand(len(x), device=x.device) < .5
        x[flip] = torch.flip(x[flip], (3,))
        terang = .90 + .20 * torch.rand(len(x), 1, 1, 1, device=x.device)
        kontras = .90 + .20 * torch.rand(len(x), 1, 1, 1, device=x.device)
        rerata = x.mean((2, 3), keepdim=True)
        x = ((x - rerata) * kontras + rerata) * terang
        x = (x + .008 * torch.randn_like(x)).clamp(0, 1)
        # Occlusion kecil meniru pelepah tanpa menghapus sinyal warna utama.
        for i in torch.where(torch.rand(len(x), device=x.device) < .10)[0].tolist():
            h, w = x.shape[-2:]
            eh, ew = random.randint(h // 12, h // 5), random.randint(w // 12, w // 5)
            y0, x0 = random.randint(0, h - eh), random.randint(0, w - ew)
            x[i, :, y0:y0 + eh, x0:x0 + ew] = x[i].mean((1, 2), keepdim=True)
    if x.shape[-1] != ukuran:
        x = F.interpolate(x, (ukuran, ukuran), mode="bilinear",
                          align_corners=False, antialias=True)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


@torch.inference_mode()
def infer(m, loader, ukuran, ambil_fitur=False):
    m.eval()
    out = np.zeros((len(loader.dataset), 4), np.float32)
    fitur = (np.zeros((len(loader.dataset), m.d_fitur), np.float32)
             if ambil_fitur else None)
    for img, aux, pc1, _y, idx in loader:
        x = olah_img(img, False, ukuran)
        with torch.autocast("cuda", torch.bfloat16):
            z = m.ekstrak(x, aux.cuda(non_blocking=True))
            p = torch.softmax(m.logit_dari_fitur(
                z, pc1.cuda(non_blocking=True)).float(), 1)
        out[idx.numpy()] = p.cpu().numpy()
        if fitur is not None:
            fitur[idx.numpy()] = z.float().cpu().numpy()
    return (out, fitur) if ambil_fitur else out


def agregasi(Pview: np.ndarray, data_split: list[dict], mode: str) -> np.ndarray:
    out, awal = [], 0
    for b in data_split:
        q = Pview[awal:awal + len(b["idx"])]
        awal += len(b["idx"])
        if mode == "mean":
            out.append(q.mean(0))
        else:
            out.append(np.average(q, axis=0, weights=np.maximum(q.max(1), 1e-6)))
    return np.asarray(out)


def kandidat_agregasi(p_h, p_c1, data_split):
    return {f"{nama}_{mode}": agregasi(p, data_split, mode)
            for nama, p in (("hibrida", p_h), ("C1", p_c1))
            for mode in ("mean", "conf")}


def nilai_val(p_h, p_c1, y_view, y_bunch, data_split):
    wv, qv, pv, _ = CK.pilih_blend({"hibrida": p_h, "C1": p_c1}, y_view)
    wb, qb, pb, _ = CK.pilih_blend(
        kandidat_agregasi(p_h, p_c1, data_split), y_bunch)
    return {
        "view": {"bobot": wv, "aturan": qv[2], "tau": qv[3],
                 "metrik": qv[4], "prob": pv},
        "bunch": {"bobot": wb, "aturan": qb[2], "tau": qb[3],
                  "metrik": qb[4], "prob": pb, "objective": qb[0]},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="convnext_tiny_s42")
    ap.add_argument("--backbone", choices=("convnext_tiny", "convnext_base",
                                             "efficientnet_v2_s", "swin_t"),
                    default="convnext_tiny")
    ap.add_argument("--mode-c1", choices=("residual", "bebas"), default="residual",
                    help="residual mengoreksi log-prob C1; bebas menjadi classifier visual mandiri")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="akumulasi gradien; batch efektif = batch x nilai ini")
    ap.add_argument("--ukuran", type=int, default=160)
    ap.add_argument("--crop-img", type=Path, default=None,
                    help="cache NPY crop resolusi tinggi; default memakai cache re-ID 128")
    ap.add_argument("--crop-meta", type=Path, default=None,
                    help="metadata NPZ pasangan --crop-img; default <stem>_meta.npz")
    ap.add_argument("--lr-backbone", type=float, default=3e-5)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.grad_accum < 1:
        raise ValueError("--grad-accum harus >= 1")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    X, P, keys, data = CK.muat_fitur(
        SUB / "results" / "damimas_deskriptor_matrix.npz")
    dv = {s: CK.data_view(data[s]) for s in data}
    yb = {s: np.asarray([b["y"] for b in data[s]], int) for s in data}
    if args.crop_img is None:
        zc = np.load(SUB / "results" / "potongan_reid.npz", allow_pickle=True)
        img = zc["img"]
        crop_keys = zc["kunci"].astype(str)
    else:
        meta_path = args.crop_meta or args.crop_img.with_name(
            args.crop_img.stem + "_meta.npz")
        if not args.crop_img.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Cache crop tidak lengkap: {args.crop_img} / {meta_path}")
        img = np.load(args.crop_img, mmap_mode="r")
        zc = np.load(meta_path, allow_pickle=True)
        crop_keys = zc["kunci"].astype(str)
        if len(img) != len(crop_keys):
            raise RuntimeError("Tensor crop dan metadata berbeda panjang")
    pos = {k: i for i, k in enumerate(crop_keys)}
    crop_idx_lokal = np.asarray([pos[k] for k in keys], int)

    tr_idx = dv["train"][0]
    mu, sd = X[tr_idx].mean(0), X[tr_idx].std(0)
    sd[sd < 1e-5] = 1.
    Xa = ((X - mu) / sd).astype(np.float32)
    datasets, loaders = {}, {}
    for s in data:
        idx, y, _tree, _bunch = dv[s]
        datasets[s] = DataCrop(img, crop_idx_lokal[idx], Xa[idx], P[idx], y)
        loaders[s] = DataLoader(datasets[s], batch_size=args.batch,
                                shuffle=False, num_workers=0, pin_memory=True)

    ytr = dv["train"][1]
    frek = np.bincount(ytr, minlength=4).astype(float)
    # Satu tandan yang punya banyak tampak tidak boleh mendominasi sampler.
    inv_view = np.empty(len(ytr), float)
    awal = 0
    for b in data["train"]:
        inv_view[awal:awal + len(b["idx"])] = 1 / len(b["idx"])
        awal += len(b["idx"])
    sw = inv_view * np.sqrt(frek.max() / frek[ytr])
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(datasets["train"], batch_size=args.batch,
                              sampler=sampler, num_workers=0, pin_memory=True)

    m = Hibrida(Xa.shape[1], args.backbone, args.mode_c1).cuda()
    p_backbone = [p for p in m.backbone.parameters() if p.requires_grad]
    p_head = list(m.aux.parameters()) + list(m.residual.parameters()) + [m.log_skala_c1]
    opt = torch.optim.AdamW([
        {"params": p_backbone, "lr": args.lr_backbone},
        {"params": p_head, "lr": args.lr_head},
    ], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=1e-6)
    kelas_w = torch.tensor(np.sqrt(frek.max() / frek), device="cuda", dtype=torch.float32)

    run = SUB / "runs" / f"classifier_hibrida_damimas_{args.tag}"
    run.mkdir(parents=True, exist_ok=False)
    history, best, best_epoch, tanpa_naik = [], -math.inf, 0, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        m.train(); jumlah = n_batch = 0.
        opt.zero_grad(set_to_none=True)
        for ib, (im, aux, pc1, y, _idx) in enumerate(train_loader):
            x = olah_img(im, True, args.ukuran)
            aux = aux.cuda(non_blocking=True); pc1 = pc1.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            with torch.autocast("cuda", torch.bfloat16):
                logit = m(x, aux, pc1)
                ce = F.cross_entropy(logit.float(), y, weight=kelas_w,
                                     label_smoothing=.02)
                prob = torch.softmax(logit.float(), 1)
                ordinal = F.smooth_l1_loss(
                    prob @ torch.arange(4, device="cuda", dtype=torch.float32),
                    y.float())
                loss = ce + .10 * ordinal
            (loss / args.grad_accum).backward()
            if ((ib + 1) % args.grad_accum == 0 or
                    ib + 1 == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5.)
                opt.step(); opt.zero_grad(set_to_none=True)
            jumlah += float(loss.detach()); n_batch += 1
        sch.step()

        phv = infer(m, loaders["val"], args.ukuran)
        pcv = P[dv["val"][0]]
        nilai = nilai_val(phv, pcv, dv["val"][1], yb["val"], data["val"])
        skor = nilai["bunch"]["objective"]
        row = {"epoch": ep, "loss": jumlah / max(n_batch, 1),
               "val_view_acc": nilai["view"]["metrik"]["akurasi"],
               "val_view_macro_f1": nilai["view"]["metrik"]["macro_f1"],
               "val_bunch_acc": nilai["bunch"]["metrik"]["akurasi"],
               "val_bunch_macro_f1": nilai["bunch"]["metrik"]["macro_f1"],
               "objective": skor, "skala_c1": float(m.log_skala_c1.exp().detach()),
               "detik": time.time() - t0}
        history.append(row)
        print(f"ep {ep:02d} loss={row['loss']:.4f} view={row['val_view_acc']:.4f}/"
              f"{row['val_view_macro_f1']:.4f} bunch={row['val_bunch_acc']:.4f}/"
              f"{row['val_bunch_macro_f1']:.4f} obj={skor:.5f}", flush=True)
        state = {"state_dict": m.state_dict(), "epoch": ep, "args": vars(args),
                 "aux_mean": mu, "aux_std": sd, "nilai_val": nilai,
                 "history": history}
        torch.save(state, run / "last.pt")
        if ep % 5 == 0:
            torch.save(state, run / f"epoch_{ep:03d}.pt")
        if skor > best + 1e-9:
            best, best_epoch, tanpa_naik = skor, ep, 0
            torch.save(state, run / "best.pt")
        else:
            tanpa_naik += 1
        if tanpa_naik >= args.patience:
            print(f"early stop: {tanpa_naik} epoch tanpa kenaikan", flush=True)
            break

    ck = torch.load(run / "best.pt", map_location="cuda", weights_only=False)
    m.load_state_dict(ck["state_dict"])
    cfgv, cfgb = ck["nilai_val"]["view"], ck["nilai_val"]["bunch"]
    probs_h, fitur_view, probs_final_view, probs_final_bunch = {}, {}, {}, {}
    for s in ("train", "val", "test"):
        probs_h[s], fitur_view[s] = infer(
            m, loaders[s], args.ukuran, ambil_fitur=True)
        pc = P[dv[s][0]]
        kv = {"hibrida": probs_h[s], "C1": pc}
        probs_final_view[s] = CK.gabung_prob(cfgv["bobot"], kv)
        kb = kandidat_agregasi(probs_h[s], pc, data[s])
        probs_final_bunch[s] = CK.gabung_prob(cfgb["bobot"], kb)

    # TEST baru disentuh setelah checkpoint, blend, dan tau berasal dari VAL.
    yh_view_test = SD.prediksi_prob(
        probs_final_view["test"], cfgv["aturan"], cfgv["tau"])
    yh_bunch_test = SD.prediksi_prob(
        probs_final_bunch["test"], cfgb["aturan"], cfgb["tau"])
    hasil = {
        "dataset": "SawitMVC-YOLO-Damimas",
        "protokol": "train DAMIMAS; pilih checkpoint/blend/tau VAL; TEST sekali",
        "kaveat": "crop GT + tautan GT; metrik modul classifier/oracle",
        "arsitektur": f"{args.backbone} mode-{args.mode_c1} + descriptor warna/geometri",
        "sumber_crop": (str(args.crop_img) if args.crop_img is not None
                         else "results/potongan_reid.npz (128x128)"),
        "ukuran_input": args.ukuran,
        "best_epoch": best_epoch, "best_objective": best,
        "n": {s: {"view": len(dv[s][1]), "tandan": len(yb[s]),
                    "pohon": len(set(dv[s][2]))} for s in data},
        "per_view": {"bobot": cfgv["bobot"], "aturan": cfgv["aturan"],
                     "tau": cfgv["tau"], "val": cfgv["metrik"],
                     "test": SD.metrik(dv["test"][1], yh_view_test)},
        "per_tandan": {"bobot": cfgb["bobot"], "aturan": cfgb["aturan"],
                       "tau": cfgb["tau"], "val": cfgb["metrik"],
                       "test": SD.metrik(yb["test"], yh_bunch_test),
                       "test_subgrup": SD.metrik_subgrup(
                           yb["test"], yh_bunch_test,
                           np.r_[0, np.cumsum([len(b["idx"]) for b in data["test"]])])},
        "history": history,
        "bobot": str((run / "best.pt").relative_to(SUB)),
    }
    out = SUB / "results" / f"damimas_classifier_hibrida_{args.tag}.json"
    pred = SUB / "results" / f"damimas_classifier_hibrida_{args.tag}_pred.npz"
    payload = {}
    for s in ("train", "val", "test"):
        payload[f"{s}_view_hibrida"] = probs_h[s]
        payload[f"{s}_view_fitur"] = fitur_view[s]
        payload[f"{s}_view_prob"] = probs_final_view[s]
        payload[f"{s}_view_y"] = dv[s][1]
        payload[f"{s}_view_tree"] = dv[s][2]
        payload[f"{s}_bunch_prob"] = probs_final_bunch[s]
        payload[f"{s}_bunch_y"] = yb[s]
        payload[f"{s}_bunch_tree"] = np.asarray([b["tree"] for b in data[s]])
        payload[f"{s}_bunch_nview"] = np.asarray([len(b["idx"]) for b in data[s]])
    # Fitur 896/1408-dim berukuran >50 MB dan diperlukan hanya untuk kepala set;
    # simpan bersama bobot di runs/ (ikut backup artefak model). Results/ tetap
    # membawa dump probabilitas ringkas yang cukup untuk ensemble dan audit angka.
    fitur_pred = run / "fitur_dan_prediksi.npz"
    np.savez_compressed(fitur_pred, **payload)
    ringkas = {}
    for s in ("val", "test"):
        for unit in ("view", "bunch"):
            for field in ("prob", "y", "tree"):
                ringkas[f"{s}_{unit}_{field}"] = payload[f"{s}_{unit}_{field}"]
        ringkas[f"{s}_bunch_nview"] = payload[f"{s}_bunch_nview"]
    np.savez_compressed(pred, **ringkas)
    hasil["fitur_prediksi"] = str(fitur_pred.relative_to(SUB))
    hasil["prediksi_ringkas"] = str(pred.relative_to(SUB))
    out.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
    with (SUB / "results" / "riwayat_epoch" /
          f"classifier_hibrida_damimas_{args.tag}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
    print(json.dumps({"best_epoch": best_epoch, "per_view": hasil["per_view"],
                      "per_tandan": hasil["per_tandan"]}, indent=2, ensure_ascii=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
