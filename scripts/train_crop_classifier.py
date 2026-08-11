"""Classifier kematangan tandan pada crop, RGB vs RGB+relief-depth — Fase 6.

Dasar (diagnostik Fase 6, semua terukur):
- AP50 class-agnostic 0,6677 vs mAP50 class-aware 0,3707 -> 44,5% kemampuan
  detektor hangus MURNI karena salah kelas. Akurasi klasifikasi pada box yang
  sudah benar lokasinya cuma 70,5%, dan B3 cuma 30,6%.
- Konfusi SELALU antar-kelas-tetangga (nol B1->B3/B4) -> masalahnya ORDINAL,
  jadi head-nya CORAL (Cao dkk. 2020), bukan softmax 4-arah. Karena bobot `w`
  dipakai bersama untuk semua ambang, selisih logit antar-ambang konstan
  lintas-sampel -> urutan rank dijamin monoton (itu jaminan CORAL).
- Sinyal depth = relief lokal, monoton terhadap kematangan (B1 +2,8 cm ->
  B4 -5,1 cm, Kruskal-Wallis p=1,7e-21) tapi SNR per-piksel ~0,3; baru terbaca
  setelah pooling wilayah (AUC 0,592 -> 0,724). Maka cabang depth di sini
  difusikan SETELAH global pooling, bukan di stem.
- Gate depth diinisialisasi TAKNOL (pelajaran F-007 Volume 1: gate init-nol
  tidak pernah terbuka).
- Loss auxiliary RGB-only: jalur RGB dilatih juga tanpa depth, sehingga jalur
  RGB tidak bisa dirusak jalur depth ("do no harm" secara konstruksi).
- Kelangkaan B3/B4 di 352 (215/98 instance) ditambal dengan pretraining pada
  846 pohon dataset 953 yang sudah dibersihkan dari kebocoran (7.333 B3,
  2.513 B4 tersedia di sana).

Usage:
    # tahap A - pretrain RGB pada 953
    .venv/bin/python scripts/train_crop_classifier.py --tahap pretrain \
        --mode rgb --epochs 20 --name pre953

    # tahap B - finetune pada 352 (ablasi: --mode rgb vs --mode rgbd)
    .venv/bin/python scripts/train_crop_classifier.py --tahap finetune \
        --mode rgbd --init runs_fase6/pre953/best.pt --epochs 40 --name ft_rgbd
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

CROPS = Path("/workspace/crops_fase6")
RUNS = Path("/workspace/project-expertise/runs_fase6")
K = 4          # jumlah kelas B1..B4
IMG = 160      # ukuran masuk jaringan (crop tersimpan 176)


# --------------------------------------------------------------------------- data

class CropDS(Dataset):
    """Augmentasi geometrik diterapkan IDENTIK pada RGB dan depth; augmentasi
    fotometrik (jitter warna) HANYA pada RGB — relief depth itu besaran metrik,
    menggeser nilainya sama saja merusak sinyal yang mau dipakai."""

    def __init__(self, rgb, dep, msk, y, latih: bool, pakai_depth: bool):
        self.rgb, self.dep, self.msk, self.y = rgb, dep, msk, y
        self.latih, self.pakai_depth = latih, pakai_depth

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        rgb = self.rgb[i]
        msk = self.msk[i][..., None]
        dep = self.dep[i] if self.pakai_depth else None
        S = rgb.shape[0]

        if self.latih:
            skala = np.random.uniform(0.85, 1.0)
            sisi = max(8, int(S * skala))
            x0 = np.random.randint(0, S - sisi + 1)
            y0 = np.random.randint(0, S - sisi + 1)
            pot = (slice(y0, y0 + sisi), slice(x0, x0 + sisi))
            rgb, msk = rgb[pot], msk[pot]
            if dep is not None:
                dep = dep[pot]
            if np.random.rand() < 0.5:
                rgb, msk = rgb[:, ::-1], msk[:, ::-1]
                if dep is not None:
                    dep = dep[:, ::-1]
            kali = np.random.randint(0, 4)
            if kali:
                rgb, msk = np.rot90(rgb, kali), np.rot90(msk, kali)
                if dep is not None:
                    dep = np.rot90(dep, kali)

        def ke_tensor(a, kanal):
            t = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0
            return F.interpolate(t[None], (IMG, IMG), mode="bilinear", align_corners=False)[0]

        rgb = ke_tensor(rgb, 3)
        msk = ke_tensor(msk, 1)

        # Augmentasi fotometrik sengaja SANGAT ringan: kematangan tandan
        # DIDEFINISIKAN oleh warna, jadi jitter kuat (versi awal: brightness
        # +-25%, saturasi 0,6-1,4) justru menghapus label. Yang disisakan cuma
        # kompensasi pencahayaan lapangan seadanya.
        if self.latih:
            rgb = (rgb * np.random.uniform(0.93, 1.07)).clamp(0, 1)

        rgb = (rgb - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / \
              torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        x = torch.cat([rgb, msk * 2 - 1], 0)             # 4 kanal: RGB + mask box

        if dep is None:
            d = torch.zeros(2, IMG, IMG)
        else:
            d = ke_tensor(dep, 2)
            d = (d - 0.5) / 0.5
        return x, d, int(self.y[i])


def muat(src: str, pakai_depth: bool):
    rgb = np.load(CROPS / f"crops{src}_rgb.npy", mmap_mode="r")
    msk = np.load(CROPS / f"crops{src}_msk.npy", mmap_mode="r")
    dep = np.load(CROPS / f"crops{src}_dep.npy", mmap_mode="r") if pakai_depth else None
    m = np.load(CROPS / f"crops{src}_meta.npz", allow_pickle=True)
    return (np.asarray(rgb), (np.asarray(dep) if dep is not None else None),
            np.asarray(msk), m["y"], m["split"], m["tree"])


# -------------------------------------------------------------------------- model

class CabangDepth(nn.Module):
    """CNN kecil untuk 2 kanal (relief, mask valid) -> vektor terpool.

    Sengaja dangkal dan diakhiri global pooling: sinyal relief ber-SNR ~0,3 per
    piksel, jadi yang berguna adalah rata-rata wilayah, bukan tekstur halus.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 5, 2, 2), nn.BatchNorm2d(32), nn.SiLU(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.SiLU(True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.SiLU(True),
            nn.Conv2d(128, dim, 3, 2, 1), nn.BatchNorm2d(dim), nn.SiLU(True),
        )

    def forward(self, x):
        return self.net(x).mean((2, 3))


class CoralHead(nn.Module):
    """Head ordinal CORAL: satu vektor bobot dipakai bersama, K-1 bias terpisah."""

    def __init__(self, dim: int, k: int = K):
        super().__init__()
        self.w = nn.Linear(dim, 1, bias=False)
        self.b = nn.Parameter(torch.zeros(k - 1))

    def forward(self, f):
        return self.w(f) + self.b            # (N, K-1) logit P(y > j)


class Head(nn.Module):
    """Head gabungan.

    `coral`   - ordinal murni (rank-consistent, tapi memaksa semua pembedaan
                kelas ke SATU sumbu 1-D; terbukti menggencet B2/B3 saat
                pretraining 953: recall B1/B4 93-96% tapi B2/B3 ~30%).
    `softmax` - 4-arah bebas, tidak tahu urutan kelas.
    `hybrid`  - keduanya: prediksi dari softmax (ekspresif, plus probabilitas
                per kelas yang dipakai untuk rekomposisi mAP), diregularisasi
                suku CORAL supaya struktur ordinal tetap terpakai.
    """

    def __init__(self, dim: int, jenis: str):
        super().__init__()
        self.jenis = jenis
        if jenis in ("coral", "hybrid"):
            self.coral = CoralHead(dim)
        if jenis in ("softmax", "hybrid"):
            self.fc = nn.Linear(dim, K)

    def forward(self, f):
        return (self.coral(f) if self.jenis != "softmax" else None,
                self.fc(f) if self.jenis != "coral" else None)


class Model(nn.Module):
    def __init__(self, backbone: str, pakai_depth: bool, gate_init: float = 0.1,
                 head: str = "hybrid"):
        super().__init__()
        import timm
        self.bb = timm.create_model(backbone, pretrained=True, num_classes=0, in_chans=4)
        dim = self.bb.num_features
        self.pakai_depth = pakai_depth
        self.head = Head(dim, head)
        if pakai_depth:
            self.dep = CabangDepth(dim)
            self.gate = nn.Parameter(torch.tensor(float(gate_init)))   # TAKNOL (F-007)

    def forward(self, rgb, dep):
        f_rgb = self.bb(rgb)
        if not self.pakai_depth:
            keluar = self.head(f_rgb)
            return keluar, keluar
        f = f_rgb + self.gate * self.dep(dep)
        return self.head(f), self.head(f_rgb)          # (gabungan, RGB-only auxiliary)


# ------------------------------------------------------------------------ ordinal

def coral_target(y, k=K):
    """y -> matriks biner 1{y > j} untuk j=0..K-2."""
    j = torch.arange(k - 1, device=y.device)[None]
    return (y[:, None] > j).float()


def coral_loss(logit, y):
    return F.binary_cross_entropy_with_logits(logit, coral_target(y))


def rugi_head(keluar, y, jenis: str, ls: float = 0.05):
    lc, lf = keluar
    if jenis == "coral":
        return coral_loss(lc, y)
    if jenis == "softmax":
        return F.cross_entropy(lf, y, label_smoothing=ls)
    return F.cross_entropy(lf, y, label_smoothing=ls) + 0.5 * coral_loss(lc, y)


def prob_head(keluar, jenis: str):
    """Probabilitas per kelas -> dipakai untuk prediksi DAN sebagai confidence
    saat kelas ditempel kembali ke box detektor (rekomposisi mAP)."""
    lc, lf = keluar
    if jenis == "coral":
        return coral_prob(lc)
    return F.softmax(lf, dim=1)


def coral_prob(logit):
    """logit P(y>j) -> probabilitas per kelas (di-clamp supaya tidak negatif)."""
    p_gt = torch.sigmoid(logit)                                  # (N, K-1)
    satu = torch.ones(len(logit), 1, device=logit.device)
    nol = torch.zeros(len(logit), 1, device=logit.device)
    p_ge = torch.cat([satu, p_gt], 1)                            # P(y >= k)
    p_gt_full = torch.cat([p_gt, nol], 1)                        # P(y > k)
    return (p_ge - p_gt_full).clamp_min(1e-6)


# ------------------------------------------------------------------------ metrik

def metrik(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    out = {
        "akurasi": float((y == pred).mean()),
        "akurasi_pm1": float((np.abs(y - pred) <= 1).mean()),
        "mae": float(np.abs(y - pred).mean()),
    }
    f1s, recs = [], {}
    for k in range(K):
        tp = int(((pred == k) & (y == k)).sum())
        fp = int(((pred == k) & (y != k)).sum())
        fn = int(((pred != k) & (y == k)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-9))
        recs[f"B{k+1}"] = rec
    out["macro_f1"] = float(np.mean(f1s))
    out["recall_per_kelas"] = recs
    out["macro_recall"] = float(np.mean(list(recs.values())))
    return out


@torch.no_grad()
def evaluasi(model, dl, dev, jenis: str):
    model.eval()
    ys, ps, probs = [], [], []
    for rgb, dep, y in dl:
        keluar, _ = model(rgb.to(dev, non_blocking=True), dep.to(dev, non_blocking=True))
        p = prob_head(keluar, jenis)
        probs.append(p.float().cpu().numpy())
        ps.append(p.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(probs)


# -------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tahap", choices=["pretrain", "finetune"], required=True)
    ap.add_argument("--mode", choices=["rgb", "rgbd"], required=True)
    ap.add_argument("--backbone", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-backbone", type=float, default=None)
    ap.add_argument("--aux", type=float, default=0.5)
    ap.add_argument("--gate-init", type=float, default=0.1)
    ap.add_argument("--head", choices=["coral", "softmax", "hybrid"], default="hybrid")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init", default=None)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    pakai_depth = args.mode == "rgbd"
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)

    if args.tahap == "pretrain":
        rgb, dep, msk, y, split, tree = muat("953", pakai_depth)
        pohon = np.array(sorted(set(tree.tolist())))
        rng = np.random.RandomState(args.seed)
        rng.shuffle(pohon)
        pohon_val = set(pohon[:max(1, len(pohon) // 10)].tolist())   # split per-POHON
        m_val = np.array([t in pohon_val for t in tree])
        idx_tr, idx_va, idx_te = np.where(~m_val)[0], np.where(m_val)[0], np.where(m_val)[0]
    else:
        rgb, dep, msk, y, split, tree = muat("352", pakai_depth)
        idx_tr = np.where(split == "train")[0]
        idx_va = np.where(split == "val")[0]
        idx_te = np.where(split == "test")[0]

    if dep is None:
        dep = np.zeros((len(y), 1, 1, 2), np.uint8)

    def buat(idx, latih):
        ds = CropDS(rgb[idx], dep[idx] if pakai_depth else None, msk[idx], y[idx], latih, pakai_depth)
        if latih:
            cnt = np.bincount(y[idx], minlength=K).astype(np.float64)
            w = (1.0 / np.maximum(cnt, 1))[y[idx]]               # sampling seimbang kelas
            smp = WeightedRandomSampler(torch.as_tensor(w), len(idx), replacement=True)
            return DataLoader(ds, args.batch, sampler=smp, num_workers=6,
                              pin_memory=True, drop_last=True, persistent_workers=True)
        return DataLoader(ds, args.batch, shuffle=False, num_workers=4, pin_memory=True)

    dl_tr, dl_va, dl_te = buat(idx_tr, True), buat(idx_va, False), buat(idx_te, False)
    print(f"[{args.name}] tahap={args.tahap} mode={args.mode} "
          f"train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)} "
          f"dist_train={np.bincount(y[idx_tr], minlength=K).tolist()}")

    model = Model(args.backbone, pakai_depth, args.gate_init, args.head).to(dev)
    if args.init:
        sd = torch.load(args.init, map_location="cpu")["model"]
        hilang = model.load_state_dict(sd, strict=False)
        print(f"  init dari {args.init}: missing={len(hilang.missing_keys)} "
              f"unexpected={len(hilang.unexpected_keys)}")

    lr_bb = args.lr_backbone if args.lr_backbone is not None else args.lr * 0.1
    baru = [p for n, p in model.named_parameters() if not n.startswith("bb.")]
    opt = torch.optim.AdamW([
        {"params": model.bb.parameters(), "lr": lr_bb},
        {"params": baru, "lr": args.lr},
    ], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[lr_bb, args.lr], total_steps=args.epochs * len(dl_tr), pct_start=0.25)
    scaler = torch.amp.GradScaler("cuda")

    riwayat, terbaik, mulai = [], -1.0, time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        rugi_tot = 0.0
        for rgb_b, dep_b, y_b in dl_tr:
            rgb_b = rgb_b.to(dev, non_blocking=True)
            dep_b = dep_b.to(dev, non_blocking=True)
            y_b = y_b.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                keluar, keluar_rgb = model(rgb_b, dep_b)
                rugi = rugi_head(keluar, y_b, args.head)
                if pakai_depth and args.aux > 0:
                    rugi = rugi + args.aux * rugi_head(keluar_rgb, y_b, args.head)
            scaler.scale(rugi).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            rugi_tot += float(rugi)

        yv, pv, _ = evaluasi(model, dl_va, dev, args.head)
        mv = metrik(yv, pv)
        skor = mv["macro_f1"]
        baris = {"epoch": ep, "rugi": rugi_tot / max(len(dl_tr), 1), **mv}
        if pakai_depth:
            baris["gate"] = float(model.gate.detach())
        riwayat.append(baris)
        tanda = ""
        if skor > terbaik:
            terbaik = skor
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep}, out / "best.pt")
            tanda = "  *"
        g = f" gate={baris['gate']:.3f}" if pakai_depth else ""
        print(f"  ep{ep:3d} rugi={baris['rugi']:.4f} val_akur={mv['akurasi']:.4f} "
              f"macroF1={mv['macro_f1']:.4f} mR={mv['macro_recall']:.4f}{g}{tanda}", flush=True)

    model.load_state_dict(torch.load(out / "best.pt", map_location="cpu")["model"])
    model.to(dev)
    yv, pv, _ = evaluasi(model, dl_va, dev, args.head)
    yt, pt, probt = evaluasi(model, dl_te, dev, args.head)
    hasil = {
        "name": args.name, "tahap": args.tahap, "mode": args.mode, "head": args.head,
        "backbone": args.backbone, "epochs": args.epochs, "seed": args.seed,
        "durasi_detik": round(time.time() - mulai, 1),
        "val": metrik(yv, pv), "test": metrik(yt, pt),
        "konfusi_test": [[int(((yt == a) & (pt == b)).sum()) for b in range(K)] for a in range(K)],
        "riwayat": riwayat,
    }
    (out / "hasil.json").write_text(json.dumps(hasil, indent=2))
    np.savez(out / "pred_test.npz", y=yt, pred=pt, prob=probt)
    print(json.dumps({k: hasil[k] for k in ("val", "test")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
