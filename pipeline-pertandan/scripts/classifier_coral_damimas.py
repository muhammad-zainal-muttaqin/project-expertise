"""PT-E-030 — Anggota classifier ORDINAL (CORAL) untuk DAMIMAS.

## Kenapa ini yang dikerjakan berikutnya

Tidak satu pun anggota classifier DAMIMAS memakai loss ordinal -- seluruhnya
cross-entropy. Padahal B1<B2<B3<B4 berurutan, dan PT-E-015 (korpus 953) sudah
mengukur efeknya: CORAL menaikkan C2 sebesar +2,35 pp di ResNet-18 dan menaikkan
C3 di kedua backbone, konsisten arahnya di keempat pasangan yang diuji.

Ini juga anggota yang paling mungkin TERDEKORELASI dari yang sudah ada. Seluruh
anggota DAMIMAS berjangkar pada C1 (`mode_c1="residual"`: keluarannya
`skala*log(p_C1) + residual(z)`), jadi mereka berbagi tulang punggung yang sama.
Model CORAL di sini sengaja TIDAK memakai jangkar C1: ia melihat potongan saja.
Untuk ensemble, anggota yang salah karena alasan berbeda lebih berharga daripada
anggota yang sedikit lebih akurat tetapi salah di tempat yang sama.

## Kenapa CORAL, bukan sekadar "loss ordinal"

CORAL memodelkan K-1 ambang kumulatif `P(y>k)` dengan bobot BERSAMA dan bias
dipaksa menurun (`b_k = b0 - cumsum(softplus(delta))`). Monotonisitas itu wajib:
tanpanya selisih kumulatifnya bisa negatif, dan vektor kelas yang dihasilkan
tidak sah dipakai sebagai anggota ensemble probabilitas.

## Keluaran

Bank probabilitas per-TANDAN yang selaras urutannya dengan bank DAMIMAS yang
sudah ada (919 val / 1316 test), diverifikasi dengan mencocokkan label `y`.
Tanpa jaminan itu, rata-rata ensemble menjumlahkan tandan yang berbeda.

Pemakaian:
    .venv/bin/python pipeline-pertandan/scripts/classifier_coral_damimas.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

SUB = Path(__file__).resolve().parents[1]
R = SUB / "results"
K = 4


class KepalaCORAL(nn.Module):
    """K-1 ambang kumulatif, bobot bersama, bias dipaksa menurun."""

    def __init__(self, dim, p_drop=0.3):
        super().__init__()
        self.drop = nn.Dropout(p_drop)
        self.w = nn.Linear(dim, 1, bias=False)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.delta = nn.Parameter(torch.full((K - 2,), 0.5413))

    def forward(self, z):
        b = self.b0 - torch.cat([torch.zeros(1, device=self.b0.device),
                                 torch.cumsum(F.softplus(self.delta), 0)])
        return self.w(self.drop(z)) + b

    @staticmethod
    def rugi(o, y):
        t = (y.unsqueeze(1) > torch.arange(K - 1, device=y.device)).float()
        return F.binary_cross_entropy_with_logits(o.float(), t)

    @staticmethod
    def prob(o):
        c = torch.sigmoid(o.float())
        satu = torch.ones(len(c), 1, device=c.device)
        nol = torch.zeros(len(c), 1, device=c.device)
        return (torch.cat([satu, c], 1) - torch.cat([c, nol], 1)).clamp_min(0)


class KepalaCORN(nn.Module):
    """CORN (Shi, Cao & Raschka 2023, arXiv:2111.08851). K-1 task biner dengan
    bobot SENDIRI-SENDIRI -- tidak ada weight-sharing seperti CORAL.

    Task k menaksir peluang BERSYARAT `P(y > r_k | y > r_{k-1})`, dilatih hanya
    pada subset `S_k = {y > r_{k-1}}`. Konsistensi rank datang dari aturan rantai
    saat inferensi, `P(y > r_k) = prod_{j<=k} f_j`, yang otomatis monoton karena
    tiap faktor ada di [0,1]. Jadi monotonisitas tidak lagi dibayar dengan
    mengekang ekspresivitas.

    Kenapa ini penting di sini: CORAL dengan bobot bersama membuat
    `P(y=tengah) = sigma(s+b0) - sigma(s+b1)` terkurung oleh jarak bias, dan di
    DAMIMAS ia runtuh -- maks P(B2)=0,291, maks P(B3)=0,301, test 0,3305.
    Struktur bersyarat CORN juga kebetulan menjawab analisis galat kita: task 3
    dilatih HANYA di {B3,B4}, yaitu batas yang menyumbang 45% galat.
    """

    def __init__(self, dim, p_drop=0.3):
        super().__init__()
        self.f = nn.Sequential(nn.Dropout(p_drop), nn.Linear(dim, K - 1))

    def forward(self, z):
        return self.f(z)

    @staticmethod
    def rugi(o, y):
        # subset bersarang: S_1 = semua, S_k = {y > r_{k-1}}
        tot, n = 0.0, 0
        for k in range(K - 1):
            sel = y > (k - 1) if k > 0 else torch.ones_like(y, dtype=torch.bool)
            if sel.sum() == 0:
                continue
            t = (y[sel] > k).float()
            tot = tot + F.binary_cross_entropy_with_logits(
                o[sel, k].float(), t, reduction="sum")
            n += int(sel.sum())
        return tot / max(n, 1)

    @staticmethod
    def prob(o):
        f = torch.sigmoid(o.float())                    # P(y>r_k | y>r_{k-1})
        kum = torch.cumprod(f, dim=1)                   # P(y>r_k), monoton
        satu = torch.ones(len(kum), 1, device=kum.device)
        nol = torch.zeros(len(kum), 1, device=kum.device)
        return (torch.cat([satu, kum], 1) - torch.cat([kum, nol], 1)).clamp_min(0)


KEPALA = {"coral": KepalaCORAL, "corn": KepalaCORN}


class Model(nn.Module):
    def __init__(self, loss="corn"):
        super().__init__()
        b = torchvision.models.convnext_tiny(
            weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        b.classifier[2] = nn.Identity()
        for i in range(0, 4):                    # bekukan stem..stage2
            for p in b.features[i].parameters():
                p.requires_grad = False
        self.b, self.kepala = b, KEPALA[loss](768)

    def forward(self, x):
        return self.kepala(self.b(x))


MEAN = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
STD = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)


def ke_tensor(batch, latih):
    x = torch.from_numpy(np.ascontiguousarray(batch[:, :, :, ::-1])).permute(0, 3, 1, 2).float() / 255
    if latih:
        if random.random() < .5:
            x = torch.flip(x, [3])
        x = (x * (.8 + .4 * torch.rand(len(x), 1, 1, 1))).clamp(0, 1)
    return (x - MEAN) / STD


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epoch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--loss", choices=("coral", "corn"), default="corn")
    args = ap.parse_args()
    KLS = KEPALA[args.loss]
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dev = "cuda"
    t0 = time.time()

    img = np.load(R / "crop_damimas_224.npy", mmap_mode="r")
    m = np.load(R / "crop_damimas_224_meta.npz", allow_pickle=True)
    split = np.asarray(m["split"]); tree = np.asarray(m["tree"])
    bunch = np.asarray(m["bunch"], int); yv = np.asarray(m["y"], int)
    print(f"potongan {img.shape}, split {dict(zip(*np.unique(split, return_counts=True)))}")

    idx = {s: np.where(split == s)[0] for s in ("train", "val", "test")}
    model = Model(args.loss).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epoch)
    tr = idx["train"].copy()
    riwayat = []
    for ep in range(args.epoch):
        model.train(); np.random.shuffle(tr); tot = nb = 0
        for s in range(0, len(tr), args.batch):
            sel = np.sort(tr[s:s + args.batch])
            x = ke_tensor(np.asarray(img[sel]), True).to(dev, non_blocking=True)
            y = torch.from_numpy(yv[sel]).to(dev)
            with torch.autocast("cuda", torch.bfloat16):
                L = KLS.rugi(model(x), y)
            opt.zero_grad(set_to_none=True); L.backward(); opt.step()
            tot += float(L.detach()); nb += 1
        sch.step()
        riwayat.append({"epoch": ep + 1, "loss": round(tot / max(nb, 1), 6)})
        print(f"  epoch {ep+1}/{args.epoch} loss {tot/max(nb,1):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    @torch.no_grad()
    def prob(ii):
        model.eval(); out = []
        for s in range(0, len(ii), 128):
            x = ke_tensor(np.asarray(img[np.sort(ii[s:s + 128])]), False).to(dev)
            with torch.autocast("cuda", torch.bfloat16):
                out.append(KLS.prob(model(x)).cpu().numpy())
        return np.concatenate(out)

    runs = SUB / "runs" / f"classifier_{args.loss}_damimas_s{args.seed}"
    runs.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "seed": args.seed,
                "epoch": args.epoch}, runs / "best.pt")

    # ---- agregasi per-tandan, urutan diselaraskan dengan bank yang sudah ada ----
    ref = np.load(R / "damimas_classifier_hibrida_convnext224_s42_pred.npz",
                  allow_pickle=True)
    simpan = {}
    for s in ("val", "test"):
        ii = np.sort(idx[s])
        P = prob(ii)
        per = defaultdict(list)
        for j, k in enumerate(ii):
            per[(str(tree[k]), int(bunch[k]))].append(j)
        # urutan kanonik: kemunculan pertama di dalam split, sama seperti bank lain
        urut, lihat = [], set()
        for k in ii:
            key = (str(tree[k]), int(bunch[k]))
            if key not in lihat:
                lihat.add(key); urut.append(key)
        Pb = np.stack([P[per[k]].mean(0) for k in urut])
        Pb = Pb / np.clip(Pb.sum(1, keepdims=True), 1e-9, None)
        yb = np.array([yv[ii[per[k][0]]] for k in urut], int)
        y_ref = np.asarray(ref[f"{s}_bunch_y"], int)
        cocok = len(yb) == len(y_ref) and np.array_equal(yb, y_ref)
        print(f"  {s}: {len(urut)} tandan | selaras dengan bank lain: {cocok}")
        if not cocok:
            raise SystemExit(
                f"URUTAN TIDAK SELARAS di {s} ({len(yb)} vs {len(y_ref)}). "
                "Menyimpan bank yang tidak selaras akan membuat ensemble "
                "menjumlahkan tandan berbeda -- dihentikan.")
        simpan[f"{s}_bunch_prob"] = Pb.astype(np.float32)
        simpan[f"{s}_bunch_y"] = yb
        simpan[f"{s}_bunch_tree"] = np.array([k[0] for k in urut])
        simpan[f"{s}_bunch_nview"] = np.array([len(per[k]) for k in urut], int)
        simpan[f"{s}_view_prob"] = P.astype(np.float32)
        akur = float((Pb.argmax(1) == yb).mean())
        print(f"  {s} akurasi per-tandan (argmax): {akur:.4f}")
        simpan[f"{s}_akurasi"] = np.array(akur)

    f = R / f"damimas_classifier_{args.loss}_s{args.seed}_pred.npz"
    np.savez_compressed(f, **simpan)
    (R / f"damimas_classifier_{args.loss}_s{args.seed}.json").write_text(json.dumps(
        {"pt_e": "030", "seed": args.seed, "epoch": args.epoch, "lr": args.lr,
         "backbone": "convnext_tiny", "loss": args.loss, "ukuran": 224,
         "riwayat_epoch": riwayat,
         "val_akurasi": float(simpan["val_akurasi"]),
         "test_akurasi": float(simpan["test_akurasi"]),
         "detik": round(time.time() - t0, 1)}, indent=1, ensure_ascii=False))
    print(f"-> {f}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
