"""PT-E-012 — Modul C3: classifier MULTI-TAMPAK, bentuk paling setia ke sketsa.

Sketsa asal menunjukkan pool `T1` berisi potongan V1 dan V2 masuk **ke dalam**
model, lalu keluar satu label. Yang dibangun sejauh ini bukan itu: tiap potongan
dinilai sendiri-sendiri, lalu digabung **rumus di luar model** (R4).

Rumus itu buta konteks. Ia tidak bisa tahu bahwa satu foto buram, bahwa tandan
setengah tertutup pelepah, atau bahwa dua tampak berbeda kelas karena yang satu
dari sisi bayangan. Model yang melihat seluruh tampak sekaligus bisa
mempelajarinya dari data.

## Tiga jalur dibandingkan pada POTONGAN GT dan TAUTAN ORACLE

Sengaja di kotak GT: yang diuji modul C saja, terlepas dari galat deteksi dan
galat penautan. Ketiganya dinilai atas himpunan tandan yang SAMA.

  C1  skor detektor          distribusi kelas detektor dipetakan ke kotak GT,
                             digabung R4                    (tanpa training)
  C2  classifier per-tampak  ResNet-18 dilatih di potongan, tiap tampak dinilai
                             sendiri, digabung R4           (~10 mnt GPU)
  C3  classifier multi-tampak satu model menerima SELURUH tampak tandan itu
                             sekaligus, attention antar-tampak, keluar satu
                             distribusi                     (~15 mnt GPU)

C2 ada supaya kontribusi C3 terisolasi. Tanpa C2, selisih C3 vs C1 bercampur
antara "punya classifier khusus" dan "melihat banyak tampak sekaligus" — dua hal
yang berbeda.

## Kaveat yang sudah diketahui sebelum dijalankan

Data latihnya tipis: 7.427 tandan train (5.546 multi-sisi). Backbone dibekukan
sebagian supaya tidak menghafal. Dan C3 **tidak menyentuh masalah kepadatan**
(PT-E-011) — ia hanya memperbesar nilai dari tandan yang berhasil disatukan.

Pemakaian:
    .venv/bin/python pipeline-pertandan/scripts/c3_multitampak.py --epoch 25
"""

from __future__ import annotations

import argparse
import json
import random
import sys
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

SUB = PP.SUB
KELAS = PP.KELAS
SEED = 0


class Batang(nn.Module):
    """Backbone bersama: ResNet-18 ImageNet, dua blok pertama dibekukan."""

    def __init__(self):
        super().__init__()
        b = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        b.fc = nn.Identity()
        for nama, p in b.named_parameters():
            if nama.startswith(("conv1", "bn1", "layer1", "layer2")):
                p.requires_grad = False
        self.b = b

    def forward(self, x):
        return self.b(x)                       # (N, 512)


class C2(nn.Module):
    """Satu tampak masuk, satu distribusi keluar. Penggabungan di luar model."""

    def __init__(self):
        super().__init__()
        self.batang = Batang()
        self.kepala = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 4))

    def forward(self, x):
        return self.kepala(self.batang(x))


class C3(nn.Module):
    """SELURUH tampak satu tandan masuk bersama, satu distribusi keluar.

    Attention antar-tampak: model menghitung sendiri bobot tiap tampak, jadi ia
    bisa belajar mengabaikan tampak yang buram/terpotong. Panjangnya variabel
    (1-6 tampak), ditangani dengan mask.
    """

    def __init__(self, dim=256):
        super().__init__()
        self.batang = Batang()
        self.proj = nn.Linear(512, dim)
        self.skor = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1))
        self.kepala = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.3),
                                    nn.Linear(dim, 4))

    def forward(self, x, mask):
        """x: (B, T, 3, H, W) — B tandan, maksimal T tampak. mask: (B, T) bool."""
        B, T = x.shape[:2]
        f = self.proj(self.batang(x.flatten(0, 1))).view(B, T, -1)
        a = self.skor(f).squeeze(-1)                       # (B, T)
        a = a.masked_fill(~mask, -1e4)
        w = torch.softmax(a, dim=1).unsqueeze(-1)          # (B, T, 1)
        return self.kepala((f * w).sum(1))


def siapkan(ids_split, img, kunci, man):
    """tandan -> daftar indeks potongan + kelas GT + split."""
    pos = {k: i for i, k in enumerate(kunci)}
    per_split = defaultdict(list)
    for split, daftar in ids_split.items():
        for t in daftar:
            d = json.loads((PP.DS / "json" / f"{t}.json").read_text(encoding="utf-8-sig"))
            for b in d["bunches"]:
                idx = [pos[f"{t}|{ap['side_index']}|{ap['box_index']}"]
                       for ap in b["appearances"]
                       if f"{t}|{ap['side_index']}|{ap['box_index']}" in pos]
                if idx:
                    per_split[split].append(
                        {"tree": t, "idx": idx, "y": KELAS.index(b["class"])})
    return per_split


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def ke_tensor(batch, latih):
    x = torch.from_numpy(batch[:, :, :, ::-1].copy()).permute(0, 3, 1, 2).float() / 255
    if latih:
        if random.random() < 0.5:
            x = torch.flip(x, [3])
        x = (x * (0.8 + 0.4 * torch.rand(len(x), 1, 1, 1))).clamp(0, 1)
    return ((x - MEAN) / STD)


def latih_c2(img, data, epoch, lr, dev):
    m = C2().to(dev)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                            lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epoch)
    # satu contoh = satu TAMPAK (bukan tandan)
    contoh = [(i, b["y"]) for b in data["train"] for i in b["idx"]]
    for ep in range(epoch):
        m.train(); random.shuffle(contoh); tot = n = 0
        for s in range(0, len(contoh), 64):
            c = contoh[s:s + 64]
            x = ke_tensor(img[[i for i, _ in c]], True).to(dev)
            y = torch.tensor([v for _, v in c], device=dev)
            with torch.autocast("cuda", torch.bfloat16):
                L = F.cross_entropy(m(x).float(), y)
            opt.zero_grad(set_to_none=True); L.backward(); opt.step()
            tot += float(L.detach()); n += 1
        sch.step()
        print(f"  C2 epoch {ep+1}/{epoch} loss {tot/max(n,1):.4f}", flush=True)
    return m


def latih_c3(img, data, epoch, lr, dev, maks_t=6):
    m = C3().to(dev)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                            lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epoch)
    tr = data["train"]
    for ep in range(epoch):
        m.train(); random.shuffle(tr); tot = n = 0
        for s in range(0, len(tr), 16):
            grup = tr[s:s + 16]
            B = len(grup)
            X = np.zeros((B, maks_t) + img.shape[1:], img.dtype)
            mask = torch.zeros(B, maks_t, dtype=torch.bool)
            for bi, g in enumerate(grup):
                sel = g["idx"][:maks_t]
                X[bi, :len(sel)] = img[sel]
                mask[bi, :len(sel)] = True
            x = ke_tensor(X.reshape((-1,) + img.shape[1:]), True)
            x = x.view(B, maks_t, *x.shape[1:]).to(dev)
            y = torch.tensor([g["y"] for g in grup], device=dev)
            with torch.autocast("cuda", torch.bfloat16):
                L = F.cross_entropy(m(x, mask.to(dev)).float(), y)
            opt.zero_grad(set_to_none=True); L.backward(); opt.step()
            tot += float(L.detach()); n += 1
        sch.step()
        print(f"  C3 epoch {ep+1}/{epoch} loss {tot/max(n,1):.4f}", flush=True)
    return m


@torch.no_grad()
def prob_c2(m, img, idx_all, dev):
    m.eval(); out = np.zeros((len(idx_all), 4), np.float32)
    for s in range(0, len(idx_all), 256):
        x = ke_tensor(img[idx_all[s:s + 256]], False).to(dev)
        with torch.autocast("cuda", torch.bfloat16):
            out[s:s + 256] = torch.softmax(m(x).float(), 1).cpu().numpy()
    return out


@torch.no_grad()
def prob_c3(m, img, data, dev, maks_t=6):
    m.eval(); out = []
    for s in range(0, len(data), 16):
        grup = data[s:s + 16]; B = len(grup)
        X = np.zeros((B, maks_t) + img.shape[1:], img.dtype)
        mask = torch.zeros(B, maks_t, dtype=torch.bool)
        for bi, g in enumerate(grup):
            sel = g["idx"][:maks_t]
            X[bi, :len(sel)] = img[sel]; mask[bi, :len(sel)] = True
        x = ke_tensor(X.reshape((-1,) + img.shape[1:]), False)
        x = x.view(B, maks_t, *x.shape[1:]).to(dev)
        with torch.autocast("cuda", torch.bfloat16):
            out.append(torch.softmax(m(x, mask.to(dev)).float(), 1).cpu().numpy())
    return np.concatenate(out)


def nilai_r4(data, probfn, skema, tau):
    """Bungkus tiap tandan jadi pool agar bisa dinilai aturan yang sama (R0/R4)."""
    pools = []
    for b in data:
        P = probfn(b)
        pools.append({"tree": b["tree"], "gt": b["y"],
                      "pool": [{"p": p / max(p.sum(), 1e-9), "conf": float(p.max()),
                                "luas": 1.0, "tepi": 0.5} for p in P]})
    return pools


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--keluaran", default=str(SUB / "results" / "pt_e_012_c3.json"))
    args = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    dev = "cuda"

    man = PP.muat_manifest()
    ids = {s: [t for t, v in man.items() if v == s] for s in ("train", "val", "test")}
    img, kunci, _, _ = RD.bangun_potongan(
        ids["train"] + ids["val"] + ids["test"],
        SUB / "results" / "potongan_reid.npz")
    data = siapkan(ids, img, list(kunci), man)
    for s in data:
        n_multi = sum(1 for b in data[s] if len(b["idx"]) >= 2)
        print(f"{s}: {len(data[s])} tandan ({n_multi} multi-tampak)")

    # C1: distribusi detektor dipetakan ke kotak GT
    prob = PP.bangun_prob_prediksi({k: ids[k] for k in ("train", "val", "test")})
    pos = {k: i for i, k in enumerate(kunci)}
    kunci_dari_idx = {i: k for k, i in pos.items()}

    print("\nmelatih C2 (per-tampak)...")
    m2 = latih_c2(img, data, args.epoch, args.lr, dev)
    print("melatih C3 (multi-tampak)...")
    m3 = latih_c3(img, data, args.epoch, args.lr, dev)

    skema = "conf_luas"
    hasil = {"epoch": args.epoch, "lr": args.lr, "seed": SEED,
             "n_tandan": {s: len(data[s]) for s in data}, "split": {}}

    # tau dipas di val untuk tiap jalur, terpisah
    tau_c = {}
    for nama in ("C1", "C2"):
        def pf(b, nama=nama):
            if nama == "C1":
                return np.stack([prob[kunci_dari_idx[i]] for i in b["idx"]])
            return prob_c2(m2, img, np.array(b["idx"]), dev)
        pv = nilai_r4(data["val"], pf, skema, (0.5, 1.5, 2.5))
        tau_c[nama] = EP.cari_tau(pv, skema)
    print(f"tau val: {tau_c}")

    for s in ("val", "test"):
        blok = {}
        for nama in ("C1", "C2"):
            def pf(b, nama=nama):
                if nama == "C1":
                    return np.stack([prob[kunci_dari_idx[i]] for i in b["idx"]])
                return prob_c2(m2, img, np.array(b["idx"]), dev)
            pools = nilai_r4(data[s], pf, skema, tau_c[nama])
            multi = [q for q in pools if len(q["pool"]) >= 2]
            blok[nama] = {
                "R0": EP.nilai(pools, "R0", skema, tau_c[nama]),
                "R4": EP.nilai(pools, "R4", skema, tau_c[nama]),
                "R4_multi": EP.nilai(multi, "R4", skema, tau_c[nama]),
            }
        # C3: langsung, tanpa aturan agregasi
        P3 = prob_c3(m3, img, data[s], dev)
        y = np.array([b["y"] for b in data[s]])
        yh = P3.argmax(1)
        multi_m = np.array([len(b["idx"]) >= 2 for b in data[s]])
        f1 = []
        for k in range(4):
            tp = int(((yh == k) & (y == k)).sum()); fp = int(((yh == k) & (y != k)).sum())
            fn = int(((yh != k) & (y == k)).sum())
            pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
            f1.append(2 * pr * rc / (pr + rc + 1e-9))
        blok["C3"] = {
            "akurasi": round(float((yh == y).mean()), 4),
            "akurasi_multi": round(float((yh[multi_m] == y[multi_m]).mean()), 4),
            "macro_f1": round(float(np.mean(f1)), 4),
            "mae_ordinal": round(float(np.abs(yh - y).mean()), 4),
            "recall_per_kelas": {KELAS[k]: round(float(((yh == k) & (y == k)).sum() /
                                                       max((y == k).sum(), 1)), 4)
                                 for k in range(4)},
            "n": len(y), "n_multi": int(multi_m.sum()),
        }
        hasil["split"][s] = blok
        print(f"\n--- {s} ---")
        for nama in ("C1", "C2"):
            b = blok[nama]
            print(f"  {nama}: R0 {b['R0']['akurasi']}  R4 {b['R4']['akurasi']}  "
                  f"R4 multi {b['R4_multi']['akurasi']}")
        b = blok["C3"]
        print(f"  C3: {b['akurasi']}  multi {b['akurasi_multi']}  "
              f"macroF1 {b['macro_f1']}")

    t = hasil["split"]["test"]
    hasil["putusan"] = {
        "C3_vs_C2_pp": round((t["C3"]["akurasi"] - t["C2"]["R4"]["akurasi"]) * 100, 2),
        "C3_vs_C1_pp": round((t["C3"]["akurasi"] - t["C1"]["R4"]["akurasi"]) * 100, 2),
        "C3_vs_C2_multi_pp": round((t["C3"]["akurasi_multi"] -
                                    t["C2"]["R4_multi"]["akurasi"]) * 100, 2),
        "arti": ("C3 vs C2 mengisolasi 'melihat banyak tampak sekaligus'; "
                 "C3 vs C1 mencampurnya dengan 'punya classifier khusus'"),
    }
    Path(args.keluaran).write_text(json.dumps(hasil, indent=1, ensure_ascii=False))
    print("\n" + json.dumps(hasil["putusan"], indent=1, ensure_ascii=False))
    print(f"-> {args.keluaran}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
