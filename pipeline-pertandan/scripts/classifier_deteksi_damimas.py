"""Classifier crop pada domain proposal nyata, khusus DAMIMAS.

Classifier visual sebelumnya dilatih pada kotak GT tetapi dipakai untuk
merelabel kotak prediksi. Skrip ini menutup domain shift itu dengan melatih
langsung pada proposal TRAIN. Proposal ber-IoU tinggi diberi kelas B1--B4,
hard false positive diberi kelas kelima ``background``, dan proposal ambigu
diabaikan saat loss tetapi tetap diinferensi untuk evaluasi mAP VAL.

Kontrak kebersihan split sengaja sempit: skrip ini hanya menerima TRAIN dan
VAL. Tidak ada argumen, cache, label, atau dump TEST yang dibuka. Checkpoint
dipilih dari objective COCO VAL; penerapan ke TEST dilakukan oleh skrip lain
setelah aturan scoring terkunci.
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

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


SUB = Path(__file__).resolve().parents[1]
ROOT = SUB.parent
DS = Path("/workspace/SawitMVC-YOLO-Damimas")
KELAS = ("B1", "B2", "B3", "B4", "background")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))
import eval_dump_damimas as ED  # noqa: E402
import fusi_detektor_damimas as FD  # noqa: E402
import classifier_hibrida_damimas as CH  # noqa: E402
import penaut_pertandan as PP  # noqa: E402


def fingerprint(path: Path) -> dict:
    q = path.resolve()
    st = q.stat()
    return {"path": str(q), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def iou_matriks(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    aa = np.maximum(0, a[:, 2] - a[:, 0]) * np.maximum(0, a[:, 3] - a[:, 1])
    bb = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    return (inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)).astype(
        np.float32)


def baca_gt(dataset: Path, split: str, path: Path) -> np.ndarray:
    w, h = Image.open(path).size
    rows = []
    label = dataset / "labels" / split / f"{path.stem}.txt"
    for line in label.read_text().splitlines():
        if not line.strip():
            continue
        c, cx, cy, bw, bh = map(float, line.split())
        rows.append([int(c), (cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.asarray(rows, np.float32).reshape(-1, 5)


def prob_empat(D: np.ndarray) -> np.ndarray:
    """Distribusi kelas kondisional dari dump 6/11 kolom."""
    out = np.zeros((len(D), 4), np.float32)
    if D.shape[1] >= 10:
        out[:] = np.clip(D[:, 6:10], 0, None)
    kosong = out.sum(1) <= 1e-12
    if kosong.any():
        kelas = np.clip(D[kosong, 5].astype(int), 0, 3)
        out[kosong] = np.eye(4, dtype=np.float32)[kelas]
    return out / np.maximum(out.sum(1, keepdims=True), 1e-9)


def prior_lima(D: np.ndarray) -> np.ndarray:
    p4 = prob_empat(D)
    obj = np.clip(D[:, 4:5], 1e-5, 1 - 1e-5).astype(np.float32)
    # Jumlah lima komponen tepat satu: objectness membagi massa foreground vs
    # background, sementara empat skor hanya membagi massa foreground.
    return np.c_[obj * p4, 1 - obj].astype(np.float32)


def crop_box(img: np.ndarray, box: np.ndarray, ukuran: int, pad: float) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    a1, b1 = max(0, int(math.floor(x1 - dx))), max(0, int(math.floor(y1 - dy)))
    a2, b2 = min(w, int(math.ceil(x2 + dx))), min(h, int(math.ceil(y2 + dy)))
    if a2 - a1 <= 3 or b2 - b1 <= 3:
        return np.zeros((ukuran, ukuran, 3), np.uint8)
    return cv2.resize(img[b1:b2, a1:a2], (ukuran, ukuran),
                      interpolation=cv2.INTER_AREA)


def fitur_aux(img: np.ndarray, box: np.ndarray, p: np.ndarray,
              side: int, nv: int) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1 + x2) / (2 * w), (y1 + y2) / (2 * h)
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    desc = PP.deskriptor(img, box)
    eks = float(p @ np.arange(4))
    ent = float(-(p * np.log(np.clip(p, 1e-9, 1))).sum())
    q = np.sort(p)
    geo = [side / max(nv - 1, 1), nv / 8., cx, cy, bw, bh, bw * bh,
           bw / max(bh, 1e-6), min(cx, 1 - cx, cy, 1 - cy)]
    return np.r_[desc, p, eks, ent, p.max(), q[-1] - q[-2], geo].astype(np.float32)


def jalur_cache(cache_dir: Path, tag: str, split: str) -> dict[str, Path]:
    dasar = cache_dir / f"crop_proposal_{tag}_{split}"
    return {"img": dasar.with_suffix(".npy"),
            "partial": dasar.with_name(dasar.name + ".partial.npy"),
            "meta": dasar.with_name(dasar.name + "_meta.npz"),
            "manifest": dasar.with_name(dasar.name + "_manifest.json")}


def bangun_cache(dataset: Path, split: str, pred_path: Path, cache_dir: Path,
                 tag: str, ukuran: int, pad: float, pos_iou: float,
                 neg_iou: float, rebuild: bool) -> dict[str, Path]:
    paths = sorted((dataset / "images" / split).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"Split kosong: {dataset}/images/{split}")
    tujuan = jalur_cache(cache_dir, tag, split)
    sig = {"versi": 1, "dataset": str(dataset.resolve()), "split": split,
           "prediksi": fingerprint(pred_path), "n_citra": len(paths),
           "ukuran": ukuran, "pad": pad, "pos_iou": pos_iou,
           "neg_iou": neg_iou}
    lengkap = all(tujuan[k].exists() for k in ("img", "meta", "manifest"))
    if lengkap and not rebuild:
        lama = json.loads(tujuan["manifest"].read_text())
        if lama == sig:
            print(f"cache {split} valid -> {tujuan['img']}", flush=True)
            return tujuan
        raise RuntimeError(
            f"Cache {split} ada tetapi fingerprint berubah; gunakan --rebuild-cache")
    if not rebuild and any(tujuan[k].exists() for k in tujuan):
        raise RuntimeError(f"Cache {split} tidak lengkap; gunakan --rebuild-cache")

    cache_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(pred_path, allow_pickle=True)
    n = sum(len(z[p.stem]) if p.stem in z.files else 0 for p in paths)
    if n == 0:
        z.close()
        raise RuntimeError(f"Dump proposal tidak berisi baris untuk {split}")
    if tujuan["partial"].exists():
        tujuan["partial"].unlink()
    mm = np.lib.format.open_memmap(tujuan["partial"], mode="w+",
                                   dtype=np.uint8,
                                   shape=(n, ukuran, ukuran, 3))
    max_stem = max(map(lambda p: len(p.stem), paths))
    stems = np.empty(n, dtype=f"<U{max_stem}")
    row_idx = np.empty(n, np.int32)
    target = np.empty(n, np.int8)
    best_iou = np.empty(n, np.float32)
    prior = np.empty((n, 5), np.float32)
    aux_list: list[np.ndarray] = []
    n_sisi = {}
    for p in paths:
        tree, _sep, _s = p.stem.rpartition("_")
        n_sisi[tree] = n_sisi.get(tree, 0) + 1

    awal = 0
    for ip, path in enumerate(paths, 1):
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        D = (np.asarray(z[path.stem], np.float32)
             if path.stem in z.files else np.zeros((0, 11), np.float32))
        if D.ndim != 2 or (len(D) and D.shape[1] < 6):
            raise RuntimeError(f"Format proposal tidak sah: {path.stem} {D.shape}")
        G = baca_gt(dataset, split, path)
        M = iou_matriks(D[:, :4], G[:, 1:])
        if len(G) and len(D):
            cocok = M.argmax(1)
            bi = M[np.arange(len(D)), cocok]
            y = G[cocok, 0].astype(np.int8)
        else:
            cocok = np.zeros(len(D), int)
            bi = np.zeros(len(D), np.float32)
            y = np.zeros(len(D), np.int8)
        y[(bi > neg_iou) & (bi < pos_iou)] = -1
        y[bi <= neg_iou] = 4
        p5 = prior_lima(D)
        p4 = p5[:, :4] / np.maximum(p5[:, :4].sum(1, keepdims=True), 1e-9)
        tree, _sep, sisi_s = path.stem.rpartition("_")
        side, nv = int(sisi_s) - 1, n_sisi[tree]
        for j, r in enumerate(D):
            k = awal + j
            mm[k] = crop_box(img, r[:4], ukuran, pad)
            stems[k] = path.stem
            row_idx[k] = j
            target[k] = y[j]
            best_iou[k] = bi[j]
            prior[k] = p5[j]
            aux_list.append(fitur_aux(img, r[:4], p4[j], side, nv))
        awal += len(D)
        if ip % 100 == 0 or ip == len(paths):
            print(f"cache {split}: {ip}/{len(paths)} citra, {awal}/{n} proposal",
                  flush=True)
    z.close()
    if awal != n:
        raise RuntimeError(f"Jumlah proposal berubah saat cache dibuat: {awal} != {n}")
    mm.flush()
    del mm
    tujuan["partial"].replace(tujuan["img"])
    aux = np.stack(aux_list).astype(np.float32)
    np.savez_compressed(tujuan["meta"], stem=stems, row_idx=row_idx,
                        target=target, iou=best_iou, prior=prior, aux=aux)
    tujuan["manifest"].write_text(json.dumps(sig, indent=2, ensure_ascii=False))
    print(f"cache {split}: target="
          f"{dict(zip(KELAS, np.bincount(target[target >= 0], minlength=5).tolist()))} "
          f"ambigu={int((target < 0).sum())}", flush=True)
    return tujuan


class DataProposal(Dataset):
    def __init__(self, img_path: Path, meta_path: Path, indeks=None):
        self.img = np.load(img_path, mmap_mode="r")
        z = np.load(meta_path, allow_pickle=False)
        self.aux = z["aux"].astype(np.float32)
        self.prior = z["prior"].astype(np.float32)
        self.target = z["target"].astype(np.int64)
        z.close()
        self.indeks = (np.arange(len(self.target), dtype=np.int64)
                       if indeks is None else np.asarray(indeks, np.int64))

    def __len__(self):
        return len(self.indeks)

    def __getitem__(self, i):
        j = int(self.indeks[i])
        return (torch.from_numpy(self.img[j].copy()).permute(2, 0, 1),
                torch.from_numpy(self.aux[j]), torch.from_numpy(self.prior[j]),
                int(self.target[j]), j)


class HibridaDeteksi(nn.Module):
    def __init__(self, dim_aux: int, backbone: str):
        super().__init__()
        dasar = CH.Hibrida(dim_aux, backbone, mode_c1="bebas")
        self.backbone = dasar.backbone
        self.aux = dasar.aux
        self.d_fitur = dasar.d_fitur
        self.nama_backbone = backbone
        self.residual = nn.Sequential(
            nn.LayerNorm(self.d_fitur), nn.Dropout(.25),
            nn.Linear(self.d_fitur, 256), nn.GELU(), nn.Dropout(.20),
            nn.Linear(256, 5),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.log_skala_prior = nn.Parameter(torch.zeros(()))

    def ekstrak(self, x, aux):
        return torch.cat([self.backbone(x), self.aux(aux)], 1)

    def logit_dari_fitur(self, z, prior):
        skala = self.log_skala_prior.exp().clamp(.25, 4.)
        return skala * torch.log(prior.clamp_min(1e-6)) + self.residual(z)

    def forward(self, x, aux, prior):
        return self.logit_dari_fitur(self.ekstrak(x, aux), prior)


def warmstart_gt(model: HibridaDeteksi, path: Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    args = ck.get("args", {})
    if args.get("backbone", model.nama_backbone) != model.nama_backbone:
        raise RuntimeError("Backbone warm-start berbeda dari --backbone")
    src = ck["state_dict"]
    dst = model.state_dict()
    dipakai = []
    for k in list(dst):
        q = "log_skala_c1" if k == "log_skala_prior" else k
        if q in src and src[q].shape == dst[k].shape:
            dst[k] = src[q].detach().clone()
            dipakai.append(k)
    # Kepala GT punya empat keluaran. Salin empat barisnya dan biarkan baris
    # background nol agar prior objectness tetap menjadi titik awal aman.
    for k in ("residual.5.weight", "residual.5.bias"):
        if k in src and k in dst and src[k].shape[0] == 4 and dst[k].shape[0] == 5:
            dst[k][:4] = src[k]
            dipakai.append(k + "[:4]")
    model.load_state_dict(dst)
    return {"path": str(path.resolve()), "n_tensor": len(dipakai),
            "best_epoch_gt": ck.get("epoch")}


@torch.inference_mode()
def infer(model, loader, ukuran: int) -> np.ndarray:
    model.eval()
    out = np.zeros((len(loader.dataset.target), 5), np.float32)
    for img, aux, prior, _y, idx in loader:
        x = CH.olah_img(img, False, ukuran)
        with torch.autocast("cuda", torch.bfloat16):
            p = torch.softmax(model(x, aux.cuda(non_blocking=True),
                                    prior.cuda(non_blocking=True)).float(), 1)
        out[idx.numpy()] = p.cpu().numpy()
    return out


def prediksi_top1(pred_path: Path, meta_path: Path,
                  prob: np.ndarray) -> dict[str, np.ndarray]:
    zmeta = np.load(meta_path, allow_pickle=False)
    stems = zmeta["stem"].astype(str)
    rows = zmeta["row_idx"].astype(int)
    zmeta.close()
    zpred = np.load(pred_path, allow_pickle=True)
    out = {s: [] for s in zpred.files}
    for i, (stem, j) in enumerate(zip(stems, rows)):
        r = np.asarray(zpred[stem][j], np.float32)
        p4 = prob[i, :4]
        exist = float(np.clip(1 - prob[i, 4], 1e-8, 1))
        q = p4 / max(float(p4.sum()), 1e-9)
        k = int(np.argmax(q))
        skor = float(np.clip(r[4], 1e-8, 1) ** .75 * exist ** 1.5 *
                     max(float(q[k]), 1e-8))
        out[stem].append(np.r_[r[:4], skor, float(k)])
    zpred.close()
    return {s: (np.asarray(v, np.float32).reshape(-1, 6)
                if v else np.zeros((0, 6), np.float32)) for s, v in out.items()}


def metrik_klasifikasi(y: np.ndarray, p: np.ndarray) -> dict:
    m = y >= 0
    yh = p[m].argmax(1)
    yt = y[m]
    pos = yt < 4
    return {"n": int(m.sum()), "akurasi_5kelas": float((yh == yt).mean()),
            "macro_f1_5kelas": float(f1_score(yt, yh, labels=range(5),
                                               average="macro", zero_division=0)),
            "akurasi_positif": float((yh[pos] == yt[pos]).mean()),
            "macro_f1_positif": float(f1_score(
                yt[pos], yh[pos], labels=range(4), average="macro", zero_division=0)),
            "confusion": confusion_matrix(yt, yh, labels=range(5)).tolist()}


def tulis_history(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-train", type=Path, required=True)
    ap.add_argument("--pred-val", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=DS)
    ap.add_argument("--tag", default="detected_convnext_base_s42")
    ap.add_argument("--cache-tag", default="proposal_final")
    ap.add_argument("--cache-dir", type=Path, default=SUB / "results")
    ap.add_argument("--backbone", choices=("convnext_tiny", "convnext_base",
                                             "efficientnet_v2_s", "swin_t"),
                    default="convnext_base")
    ap.add_argument("--warmstart", type=Path)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--crop-size", type=int, default=128)
    ap.add_argument("--ukuran", type=int, default=224)
    ap.add_argument("--pad", type=float, default=.12)
    ap.add_argument("--pos-iou", type=float, default=.40)
    ap.add_argument("--neg-iou", type=float, default=.15)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()
    if not (0 <= args.neg_iou < args.pos_iou <= 1):
        raise ValueError("Harus 0 <= neg-iou < pos-iou <= 1")
    if min(args.epochs, args.patience, args.batch, args.grad_accum) < 1:
        raise ValueError("epochs/patience/batch/grad-accum harus positif")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cache = {
        "train": bangun_cache(args.dataset, "train", args.pred_train,
                              args.cache_dir, args.cache_tag, args.crop_size,
                              args.pad, args.pos_iou, args.neg_iou,
                              args.rebuild_cache),
        "val": bangun_cache(args.dataset, "val", args.pred_val,
                            args.cache_dir, args.cache_tag, args.crop_size,
                            args.pad, args.pos_iou, args.neg_iou,
                            args.rebuild_cache),
    }
    meta = {}
    for s in cache:
        z = np.load(cache[s]["meta"], allow_pickle=False)
        meta[s] = {k: z[k] for k in z.files}
        z.close()
    train_idx = np.flatnonzero(meta["train"]["target"] >= 0)
    ds_tr = DataProposal(cache["train"]["img"], cache["train"]["meta"], train_idx)
    ds_va = DataProposal(cache["val"]["img"], cache["val"]["meta"])
    mu = ds_tr.aux[train_idx].mean(0)
    sd = ds_tr.aux[train_idx].std(0)
    sd[sd < 1e-5] = 1.
    # Normalisasi ditaruh pada dataset agar checkpoint menyimpan statistik yang
    # sama untuk inferensi proposal final.
    ds_tr.aux = ((ds_tr.aux - mu) / sd).astype(np.float32)
    ds_va.aux = ((ds_va.aux - mu) / sd).astype(np.float32)
    ytr = ds_tr.target[train_idx]
    frek = np.bincount(ytr, minlength=5).astype(float)
    sw = np.sqrt(frek.max() / np.maximum(frek[ytr], 1))
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    tr_loader = DataLoader(ds_tr, batch_size=args.batch, sampler=sampler,
                           num_workers=args.workers, pin_memory=True,
                           persistent_workers=args.workers > 0)
    va_loader = DataLoader(ds_va, batch_size=args.batch, shuffle=False,
                           num_workers=args.workers, pin_memory=True,
                           persistent_workers=args.workers > 0)

    model = HibridaDeteksi(ds_tr.aux.shape[1], args.backbone).cuda()
    warm = warmstart_gt(model, args.warmstart) if args.warmstart else None
    pb = [p for p in model.backbone.parameters() if p.requires_grad]
    ph = list(model.aux.parameters()) + list(model.residual.parameters()) + [
        model.log_skala_prior]
    opt = torch.optim.AdamW([{"params": pb, "lr": args.lr_backbone},
                             {"params": ph, "lr": args.lr_head}],
                            weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, args.epochs, eta_min=1e-6)
    run = SUB / "runs" / f"classifier_deteksi_damimas_{args.tag}"
    run.mkdir(parents=True, exist_ok=False)
    coco_v, paths_v, _gt_v = ED.bangun_gt(args.dataset, "val")
    history, best, best_epoch, tanpa_naik = [], -math.inf, 0, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); opt.zero_grad(set_to_none=True)
        loss_sum = n_batch = 0.
        for ib, (img, aux, prior, y, _idx) in enumerate(tr_loader):
            x = CH.olah_img(img, True, args.ukuran)
            aux = aux.cuda(non_blocking=True); prior = prior.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            with torch.autocast("cuda", torch.bfloat16):
                logit = model(x, aux, prior).float()
                ce = F.cross_entropy(logit, y, label_smoothing=.02)
                positif = y < 4
                if positif.any():
                    p4 = torch.softmax(logit[positif, :4], 1)
                    ordinal = F.smooth_l1_loss(
                        p4 @ torch.arange(4, device="cuda", dtype=torch.float32),
                        y[positif].float())
                else:
                    ordinal = ce.new_zeros(())
                loss = ce + .10 * ordinal
            (loss / args.grad_accum).backward()
            if ((ib + 1) % args.grad_accum == 0 or ib + 1 == len(tr_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                opt.step(); opt.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach()); n_batch += 1
        sch.step()

        pval = infer(model, va_loader, args.ukuran)
        det_val = prediksi_top1(args.pred_val, cache["val"]["meta"], pval)
        md = FD.coco_detail(coco_v, paths_v, det_val)
        mc = metrik_klasifikasi(meta["val"]["target"], pval)
        obj = FD.objektif(md)
        row = {"epoch": ep, "loss": loss_sum / max(n_batch, 1),
               "val_mAP50": md["mAP50"], "val_mAP50_95": md["mAP50_95"],
               "val_objective": obj, "val_acc5": mc["akurasi_5kelas"],
               "val_f1_5": mc["macro_f1_5kelas"],
               "val_acc_pos": mc["akurasi_positif"],
               "val_f1_pos": mc["macro_f1_positif"],
               "skala_prior": float(model.log_skala_prior.exp().detach()),
               "detik": time.time() - t0}
        history.append(row)
        print(f"ep {ep:02d} loss={row['loss']:.4f} "
              f"mAP={row['val_mAP50']:.4f}/{row['val_mAP50_95']:.4f} "
              f"acc5={row['val_acc5']:.4f} pos={row['val_acc_pos']:.4f} "
              f"obj={obj:.6f}", flush=True)
        state = {"state_dict": model.state_dict(), "epoch": ep,
                 "args": vars(args), "aux_mean": mu, "aux_std": sd,
                 "warmstart": warm, "val_deteksi": md,
                 "val_klasifikasi": mc, "history": history,
                 "kelas": KELAS}
        torch.save(state, run / "last.pt")
        if ep % 5 == 0:
            torch.save(state, run / f"epoch_{ep:03d}.pt")
        if obj > best + 1e-12:
            best, best_epoch, tanpa_naik = obj, ep, 0
            torch.save(state, run / "best.pt")
        else:
            tanpa_naik += 1
        if tanpa_naik >= args.patience:
            print(f"early stop: {tanpa_naik} epoch tanpa gain mAP VAL", flush=True)
            break

    ck = torch.load(run / "best.pt", map_location="cuda", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    pval = infer(model, va_loader, args.ukuran)
    np.savez_compressed(SUB / "results" / f"damimas_classifier_deteksi_{args.tag}_val.npz",
                        prob=pval, stem=meta["val"]["stem"],
                        row_idx=meta["val"]["row_idx"])
    hist_path = SUB / "results" / "riwayat_epoch" / f"classifier_deteksi_{args.tag}.csv"
    tulis_history(hist_path, history)
    hasil = {"dataset": "SawitMVC-YOLO-Damimas",
             "protokol": "fit TRAIN proposal; checkpoint dipilih COCO VAL; TEST tidak dibuka",
             "proposal": {"train": fingerprint(args.pred_train),
                          "val": fingerprint(args.pred_val)},
             "ambang_label": {"positif_iou": args.pos_iou,
                               "background_iou": args.neg_iou},
             "n": {s: {"semua": int(len(meta[s]["target"])),
                        "dipakai_loss": int((meta[s]["target"] >= 0).sum()),
                        "ambigu": int((meta[s]["target"] < 0).sum()),
                        "per_target": dict(zip(
                            KELAS, np.bincount(meta[s]["target"][meta[s]["target"] >= 0],
                                              minlength=5).tolist()))}
                   for s in meta},
             "arsitektur": f"{args.backbone} residual prior 5-kelas",
             "warmstart": warm, "best_epoch": best_epoch,
             "best_objective": best, "val_deteksi": ck["val_deteksi"],
             "val_klasifikasi": ck["val_klasifikasi"],
             "checkpoint": str(run / "best.pt"),
             "probabilitas_val": str(SUB / "results" /
                                      f"damimas_classifier_deteksi_{args.tag}_val.npz"),
             "riwayat_epoch": str(hist_path)}
    out = SUB / "results" / f"damimas_classifier_deteksi_{args.tag}.json"
    out.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
    print(json.dumps(hasil, indent=2, ensure_ascii=False), flush=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
