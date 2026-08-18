"""PT-E-014 / PT-E-015 — Modul C diulang dengan backbone lain dan ordinal loss.

PT-E-012 memalsukan C3 (dan menutup jalur modul C) memakai SATU backbone
(ResNet-18) dan SATU loss (cross-entropy). Kesimpulannya sah untuk kombinasi
itu, tapi `IDEA.md` sec.4 mengusulkan dua hal yang belum pernah diuji persis:
backbone yang lebih kuat (ConvNeXt) dan ordinal regression loss. Skrip ini
menjalankan keduanya sebagai dua faktor terpisah supaya efeknya tidak tercampur.

  PT-E-014  faktor `--backbone`  resnet18 vs convnext_tiny   (loss dipatok)
  PT-E-015  faktor `--loss`      ce vs coral                 (backbone dipatok)

Protokolnya SENGAJA identik dengan PT-E-012 supaya angkanya sebanding baris per
baris: potongan GT, tautan oracle, himpunan tandan yang sama, seed sama, epoch
sama, `tau` dipas di val per jalur. Yang berubah hanya backbone dan loss.

## Kenapa ordinal loss, dan kenapa CORAL

Kelas B1<B2<B3<B4 berurutan. Cross-entropy memperlakukan B1-vs-B2 sama mahalnya
dengan B1-vs-B4, padahal galat ke kelas tetangga jauh lebih ringan akibatnya.
CORAL memodelkan K-1 ambang kumulatif P(y>k) dengan bobot BERSAMA dan bias yang
dipaksa menurun, jadi P(y>0) >= P(y>1) >= P(y>2) dijamin secara konstruksi --
tanpa itu, selisih kumulatifnya bisa negatif dan vektor kelasnya tidak sah untuk
disuap ke R4.

Aturan agregasi R4 sendiri sudah ordinal (ekspektasi ordinal), jadi ini
menyelaraskan loss dengan aturan penggabungan yang dipakai di hilir.

Pemakaian:
    .venv/bin/python pipeline-pertandan/scripts/c_backbone_ordinal.py \
        --backbone convnext_tiny --loss coral --epoch 25
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

sys.path.insert(0, str(Path(__file__).parent))
import penaut_pertandan as PP           # noqa: E402
import eval_pertandan as EP             # noqa: E402
import reid_pertandan as RD             # noqa: E402
import c3_multitampak as C3M            # noqa: E402

SUB = PP.SUB
KELAS = PP.KELAS
SEED = 0
K = 4


# --------------------------------------------------------------------------
# backbone
# --------------------------------------------------------------------------
def buat_batang(nama: str):
    """Kembalikan (modul, dim_fitur). Paruh pertama dibekukan, sama seperti
    PT-E-012 membekukan conv1/bn1/layer1/layer2 di ResNet-18."""
    if nama == "resnet18":
        b = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        b.fc = nn.Identity()
        beku = ("conv1", "bn1", "layer1", "layer2")
        for n_, p in b.named_parameters():
            if n_.startswith(beku):
                p.requires_grad = False
        return b, 512

    if nama == "convnext_tiny":
        b = torchvision.models.convnext_tiny(
            weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        # classifier = [LayerNorm2d, Flatten, Linear(768,1000)] -> buang Linear-nya
        b.classifier[2] = nn.Identity()
        # features: 0 stem, 1 stage1, 2 down, 3 stage2, 4 down, 5 stage3, 6 down, 7 stage4
        # bekukan stem..stage2 -> analog dengan conv1..layer2 di ResNet-18
        for i in range(0, 4):
            for p in b.features[i].parameters():
                p.requires_grad = False
        return b, 768

    raise SystemExit(f"backbone tidak dikenal: {nama}")


# --------------------------------------------------------------------------
# kepala: cross-entropy biasa vs CORAL ordinal
# --------------------------------------------------------------------------
class KepalaCE(nn.Module):
    def __init__(self, dim, p_drop=0.3):
        super().__init__()
        self.f = nn.Sequential(nn.Dropout(p_drop), nn.Linear(dim, K))

    def forward(self, z):
        return self.f(z)                               # logit (N,4)

    @staticmethod
    def rugi(keluaran, y):
        return F.cross_entropy(keluaran.float(), y)

    @staticmethod
    def prob(keluaran):
        return torch.softmax(keluaran.float(), 1)


class KepalaCORAL(nn.Module):
    """K-1 ambang kumulatif, bobot BERSAMA, bias dipaksa menurun.

    b_k = b0 - cumsum(softplus(delta))  =>  b0 > b1 > b2, sehingga
    P(y>0) >= P(y>1) >= P(y>2) untuk setiap z. Tanpa jaminan ini selisih
    kumulatifnya bisa negatif dan vektor kelas jadi tidak sah.
    """

    def __init__(self, dim, p_drop=0.3):
        super().__init__()
        self.drop = nn.Dropout(p_drop)
        self.w = nn.Linear(dim, 1, bias=False)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.delta = nn.Parameter(torch.full((K - 2,), 0.5413))   # softplus(.)≈1.0

    def bias(self):
        return self.b0 - torch.cat([torch.zeros(1, device=self.b0.device),
                                    torch.cumsum(F.softplus(self.delta), 0)])

    def forward(self, z):
        return self.w(self.drop(z)) + self.bias()      # logit kumulatif (N,3)

    @staticmethod
    def rugi(keluaran, y):
        # target kumulatif: t_k = 1 kalau y > k
        t = (y.unsqueeze(1) > torch.arange(K - 1, device=y.device)).float()
        return F.binary_cross_entropy_with_logits(keluaran.float(), t)

    @staticmethod
    def prob(keluaran):
        c = torch.sigmoid(keluaran.float())            # P(y>0),P(y>1),P(y>2)
        satu = torch.ones(len(c), 1, device=c.device)
        nol = torch.zeros(len(c), 1, device=c.device)
        atas = torch.cat([satu, c], 1)                 # P(y>=k)
        bawah = torch.cat([c, nol], 1)                 # P(y>=k+1)
        return (atas - bawah).clamp_min(0)             # P(y=k), sudah >= 0


KEPALA = {"ce": KepalaCE, "coral": KepalaCORAL}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class C2(nn.Module):
    """Satu tampak masuk, satu distribusi keluar. Penggabungan di luar model."""

    def __init__(self, backbone, loss):
        super().__init__()
        self.batang, dim = buat_batang(backbone)
        self.kepala = KEPALA[loss](dim)

    def forward(self, x):
        return self.kepala(self.batang(x))


class C3(nn.Module):
    """Seluruh tampak satu tandan masuk bersama, attention antar-tampak."""

    def __init__(self, backbone, loss, dim=256):
        super().__init__()
        self.batang, d_in = buat_batang(backbone)
        self.proj = nn.Linear(d_in, dim)
        self.skor = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1))
        self.norm = nn.LayerNorm(dim)
        self.kepala = KEPALA[loss](dim)

    def forward(self, x, mask):
        B, T = x.shape[:2]
        f = self.proj(self.batang(x.flatten(0, 1))).view(B, T, -1)
        a = self.skor(f).squeeze(-1).masked_fill(~mask, -1e4)
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        return self.kepala(self.norm((f * w).sum(1)))


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def latih_c2(img, data, epoch, lr, dev, backbone, loss, log):
    m = C2(backbone, loss).to(dev)
    kls = KEPALA[loss]
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                            lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epoch)
    contoh = [(i, b["y"]) for b in data["train"] for i in b["idx"]]
    for ep in range(epoch):
        m.train(); random.shuffle(contoh); tot = n = 0
        for s in range(0, len(contoh), 64):
            c = contoh[s:s + 64]
            x = C3M.ke_tensor(img[[i for i, _ in c]], True).to(dev)
            y = torch.tensor([v for _, v in c], device=dev)
            with torch.autocast("cuda", torch.bfloat16):
                L = kls.rugi(m(x), y)
            opt.zero_grad(set_to_none=True); L.backward(); opt.step()
            tot += float(L.detach()); n += 1
        sch.step()
        log.append({"model": "C2", "epoch": ep + 1, "loss": round(tot / max(n, 1), 6)})
        print(f"  C2 epoch {ep+1}/{epoch} loss {tot/max(n,1):.4f}", flush=True)
    return m


def latih_c3(img, data, epoch, lr, dev, backbone, loss, log, maks_t=6):
    m = C3(backbone, loss).to(dev)
    kls = KEPALA[loss]
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                            lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epoch)
    tr = data["train"]
    for ep in range(epoch):
        m.train(); random.shuffle(tr); tot = n = 0
        for s in range(0, len(tr), 16):
            grup = tr[s:s + 16]; B = len(grup)
            X = np.zeros((B, maks_t) + img.shape[1:], img.dtype)
            mask = torch.zeros(B, maks_t, dtype=torch.bool)
            for bi, g in enumerate(grup):
                sel = g["idx"][:maks_t]
                X[bi, :len(sel)] = img[sel]; mask[bi, :len(sel)] = True
            x = C3M.ke_tensor(X.reshape((-1,) + img.shape[1:]), True)
            x = x.view(B, maks_t, *x.shape[1:]).to(dev)
            y = torch.tensor([g["y"] for g in grup], device=dev)
            with torch.autocast("cuda", torch.bfloat16):
                L = kls.rugi(m(x, mask.to(dev)), y)
            opt.zero_grad(set_to_none=True); L.backward(); opt.step()
            tot += float(L.detach()); n += 1
        sch.step()
        log.append({"model": "C3", "epoch": ep + 1, "loss": round(tot / max(n, 1), 6)})
        print(f"  C3 epoch {ep+1}/{epoch} loss {tot/max(n,1):.4f}", flush=True)
    return m


@torch.no_grad()
def prob_c2(m, img, idx_all, dev, loss):
    m.eval(); kls = KEPALA[loss]
    out = np.zeros((len(idx_all), K), np.float32)
    for s in range(0, len(idx_all), 256):
        x = C3M.ke_tensor(img[idx_all[s:s + 256]], False).to(dev)
        with torch.autocast("cuda", torch.bfloat16):
            out[s:s + 256] = kls.prob(m(x)).cpu().numpy()
    return out


@torch.no_grad()
def prob_c3(m, img, data, dev, loss, maks_t=6):
    m.eval(); kls = KEPALA[loss]; out = []
    for s in range(0, len(data), 16):
        grup = data[s:s + 16]; B = len(grup)
        X = np.zeros((B, maks_t) + img.shape[1:], img.dtype)
        mask = torch.zeros(B, maks_t, dtype=torch.bool)
        for bi, g in enumerate(grup):
            sel = g["idx"][:maks_t]
            X[bi, :len(sel)] = img[sel]; mask[bi, :len(sel)] = True
        x = C3M.ke_tensor(X.reshape((-1,) + img.shape[1:]), False)
        x = x.view(B, maks_t, *x.shape[1:]).to(dev)
        with torch.autocast("cuda", torch.bfloat16):
            out.append(kls.prob(m(x, mask.to(dev))).cpu().numpy())
    return np.concatenate(out)


def metrik_langsung(P, data):
    """Metrik untuk keluaran yang TIDAK lewat aturan agregasi (C3)."""
    y = np.array([b["y"] for b in data]); yh = P.argmax(1)
    multi = np.array([len(b["idx"]) >= 2 for b in data])
    f1 = []
    for k in range(K):
        tp = int(((yh == k) & (y == k)).sum()); fp = int(((yh == k) & (y != k)).sum())
        fn = int(((yh != k) & (y == k)).sum())
        pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
        f1.append(2 * pr * rc / (pr + rc + 1e-9))
    return {"akurasi": round(float((yh == y).mean()), 4),
            "akurasi_multi": round(float((yh[multi] == y[multi]).mean()), 4),
            "akurasi_pm1": round(float((np.abs(y - yh) <= 1).mean()), 4),
            "macro_f1": round(float(np.mean(f1)), 4),
            "mae_ordinal": round(float(np.abs(yh - y).mean()), 4),
            "recall_per_kelas": {KELAS[k]: round(float(((yh == k) & (y == k)).sum() /
                                                       max((y == k).sum(), 1)), 4)
                                 for k in range(K)},
            "n": len(y), "n_multi": int(multi.sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="convnext_tiny",
                    choices=["resnet18", "convnext_tiny"])
    ap.add_argument("--loss", default="coral", choices=["ce", "coral"])
    ap.add_argument("--epoch", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keluaran", default=None)
    args = ap.parse_args()
    # seed 0 memakai nama lama supaya sel yang sudah jalan tidak bertabrakan
    tag = f"{args.backbone}_{args.loss}" + ("" if args.seed == 0 else f"_s{args.seed}")
    global SEED
    SEED = args.seed
    keluaran = Path(args.keluaran or SUB / "results" / f"pt_e_014_c_{tag}.json")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    dev = "cuda"
    t0 = time.time()

    man = PP.muat_manifest()
    ids = {s: [t for t, v in man.items() if v == s] for s in ("train", "val", "test")}
    img, kunci, _, _ = RD.bangun_potongan(
        ids["train"] + ids["val"] + ids["test"],
        SUB / "results" / "potongan_reid.npz")
    kunci = list(kunci)
    data = C3M.siapkan(ids, img, kunci, man)
    for s in data:
        n_multi = sum(1 for b in data[s] if len(b["idx"]) >= 2)
        print(f"{s}: {len(data[s])} tandan ({n_multi} multi-tampak)")

    prob = PP.bangun_prob_prediksi({k: ids[k] for k in ("train", "val", "test")})
    pos = {k: i for i, k in enumerate(kunci)}
    kunci_dari_idx = {i: k for k, i in pos.items()}

    log_epoch = []
    print(f"\nbackbone={args.backbone} loss={args.loss}")
    print("melatih C2 (per-tampak)...")
    m2 = latih_c2(img, data, args.epoch, args.lr, dev, args.backbone, args.loss, log_epoch)
    print("melatih C3 (multi-tampak)...")
    m3 = latih_c3(img, data, args.epoch, args.lr, dev, args.backbone, args.loss, log_epoch)

    skema = "conf_luas"
    hasil = {"pt_e": "014/015", "backbone": args.backbone, "loss": args.loss,
             "epoch": args.epoch, "lr": args.lr, "seed": args.seed,
             "n_tandan": {s: len(data[s]) for s in data},
             "riwayat_epoch": log_epoch, "split": {}}

    # cache probabilitas C2 sekali per split (dipakai untuk tau dan evaluasi)
    cache_c2 = {}

    def pf_c1(b):
        return np.stack([prob[kunci_dari_idx[i]] for i in b["idx"]])

    def pf_c2(b):
        return cache_c2["P"][np.array(b["idx"])]

    cache_c2["P"] = prob_c2(m2, img, np.arange(len(img)), dev, args.loss)

    dump: dict = {}

    tau_c = {}
    for nama, pf in (("C1", pf_c1), ("C2", pf_c2)):
        pv = C3M.nilai_r4(data["val"], pf, skema, (0.5, 1.5, 2.5))
        tau_c[nama] = EP.cari_tau(pv, skema)
    print(f"tau val: {tau_c}")
    hasil["tau"] = tau_c

    for s in ("val", "test"):
        blok = {}
        for nama, pf in (("C1", pf_c1), ("C2", pf_c2)):
            pools = C3M.nilai_r4(data[s], pf, skema, tau_c[nama])
            multi = [q for q in pools if len(q["pool"]) >= 2]
            blok[nama] = {"R0": EP.nilai(pools, "R0", skema, tau_c[nama]),
                          "R4": EP.nilai(pools, "R4", skema, tau_c[nama]),
                          "R4_multi": EP.nilai(multi, "R4", skema, tau_c[nama])}
            if s == "test":
                blok[nama]["ci_R4_vs_R0"] = EP.bootstrap_pohon(
                    pools, "R4", "R0", skema, tau_c[nama])
        P3 = prob_c3(m3, img, data[s], dev, args.loss)
        blok["C3"] = metrik_langsung(P3, data[s])
        # DUMP saat evaluasi, bukan belakangan (aturan ../CLAUDE.md). Tanpa ini
        # ensemble C1+C2+C3 tidak bisa dihitung tanpa melatih ulang semuanya.
        dump[f"{s}__C3"] = P3
        dump[f"{s}__y"] = np.array([b["y"] for b in data[s]], np.int8)
        dump[f"{s}__n_tampak"] = np.array([len(b["idx"]) for b in data[s]], np.int16)
        dump[f"{s}__tree"] = np.array([b["tree"] for b in data[s]])
        # C1 dan C2 per-TAMPAK, diratakan dengan penunjuk offset supaya pool
        # bisa dibentuk ulang persis (panjang variabel 1-6)
        off, c1, c2 = [0], [], []
        for b in data[s]:
            c1.append(pf_c1(b)); c2.append(pf_c2(b)); off.append(off[-1] + len(b["idx"]))
        dump[f"{s}__C1_rata"] = np.concatenate(c1).astype(np.float32)
        dump[f"{s}__C2_rata"] = np.concatenate(c2).astype(np.float32)
        dump[f"{s}__offset"] = np.array(off, np.int32)
        hasil["split"][s] = blok
        print(f"\n--- {s} ---")
        for nama in ("C1", "C2"):
            b = blok[nama]
            print(f"  {nama}: R0 {b['R0']['akurasi']}  R4 {b['R4']['akurasi']}  "
                  f"R4 multi {b['R4_multi']['akurasi']}")
        b = blok["C3"]
        print(f"  C3: {b['akurasi']}  multi {b['akurasi_multi']}  macroF1 {b['macro_f1']}")

    # bobot WAJIB disimpan (ATURAN #1) -- tanpa ini probabilitas tidak bisa
    # dihitung ulang kalau dump-nya hilang, dan sel ini harus dilatih dari nol
    runs = SUB / "runs" / f"c_{tag}"
    runs.mkdir(parents=True, exist_ok=True)
    torch.save({"C2": m2.state_dict(), "C3": m3.state_dict(),
                "backbone": args.backbone, "loss": args.loss,
                "epoch": args.epoch, "seed": SEED}, runs / "best.pt")
    f_dump = SUB / "results" / f"pt_e_014_prob_{tag}.npz"
    np.savez_compressed(f_dump, **dump)
    hasil["dump"] = str(f_dump.name)
    hasil["bobot"] = str((runs / "best.pt").relative_to(SUB))

    t = hasil["split"]["test"]
    hasil["putusan"] = {
        "C2_vs_C1_pp": round((t["C2"]["R4"]["akurasi"] - t["C1"]["R4"]["akurasi"]) * 100, 2),
        "C3_vs_C2_pp": round((t["C3"]["akurasi"] - t["C2"]["R4"]["akurasi"]) * 100, 2),
        "C3_vs_C1_pp": round((t["C3"]["akurasi"] - t["C1"]["R4"]["akurasi"]) * 100, 2),
        "C3_vs_C2_multi_pp": round((t["C3"]["akurasi_multi"] -
                                    t["C2"]["R4_multi"]["akurasi"]) * 100, 2),
        "acuan_pt_e_012": {"C2_vs_C1_pp": -1.21, "C3_vs_C2_pp": -3.06,
                           "C3_vs_C1_pp": -4.27, "C3_vs_C2_multi_pp": -3.91},
        "arti": ("positif = mengalahkan. PT-E-012 (resnet18+ce) negatif di semua "
                 "kolom; kalau tetap negatif di sini, jalur modul C bertahan tertutup"),
    }
    hasil["detik"] = round(time.time() - t0, 1)
    keluaran.write_text(json.dumps(hasil, indent=1, ensure_ascii=False))
    print("\n" + json.dumps(hasil["putusan"], indent=1, ensure_ascii=False))
    print(f"-> {keluaran}  ({hasil['detik']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
