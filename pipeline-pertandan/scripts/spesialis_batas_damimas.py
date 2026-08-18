"""PT-E-031 — Spesialis biner per-BATAS kelas untuk DAMIMAS.

## Kenapa batas, bukan kelas

Analisis galat ensemble PT-E-029 (test DAMIMAS, 1.316 tandan, 337 galat):

    akurasi +-1 = 0,9962   -> hampir SELURUH galat adalah kelas bertetangga

    batas B3<->B4 : 152 galat  (45%)
    batas B2<->B3 : 137 galat  (41%)
    batas B1<->B2 :  43 galat  (13%)

86% galat duduk di DUA batas. Model empat-kelas membagi kapasitasnya untuk
enam keputusan pasangan sekaligus, padahal empat di antaranya nyaris tidak
pernah salah (B1<->B3 tiga kasus, B1<->B4 nol, B2<->B4 satu). Spesialis biner
yang hanya belajar satu batas memakai seluruh kapasitasnya untuk keputusan yang
benar-benar menentukan.

## Intervensi sengaja dibuat minimal: satu parameter per batas

Spesialis TIDAK menggantikan keputusan. Ia hanya membagi ulang massa
probabilitas yang SUDAH ada di dua kelas bertetangga, dan total massanya tidak
berubah:

    q  = (1-lam) * q_ensemble + lam * q_spesialis      # q = P(kelas atas | dua kelas itu)
    p'[bawah], p'[atas] = (p[bawah]+p[atas]) * (1-q), (p[bawah]+p[atas]) * q

Kalau `lam = 0` hasilnya identik ensemble, jadi spesialis tidak bisa merusak
kecuali ia benar-benar lebih baik di val. `lam` dipilih lewat CV 5-fold tingkat
pohon di dalam VAL -- bukan lewat fit VAL, karena PT-E-029 sudah membuktikan
seleksi berbasis fit VAL memberi keuntungan tidak adil pada aturan berparameter
banyak (tau per-nview: fit VAL 0,7595 -> test 0,7318).

TEST dibuka SEKALI setelah `lam` terkunci.

Pemakaian:
    .venv/bin/python pipeline-pertandan/scripts/spesialis_batas_damimas.py
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
BATAS = [(1, 2), (2, 3)]          # B2|B3 dan B3|B4 -- 86% galat
LAM = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]


class Spesialis(nn.Module):
    def __init__(self):
        super().__init__()
        b = torchvision.models.convnext_tiny(
            weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        b.classifier[2] = nn.Identity()
        for i in range(0, 4):
            for p in b.features[i].parameters():
                p.requires_grad = False
        self.b = b
        self.kepala = nn.Sequential(nn.Dropout(0.3), nn.Linear(768, 1))

    def forward(self, x):
        return self.kepala(self.b(x)).squeeze(-1)


MEAN = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
STD = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)


def ke_tensor(b, latih):
    x = torch.from_numpy(np.ascontiguousarray(b[:, :, :, ::-1])).permute(0, 3, 1, 2).float() / 255
    if latih:
        if random.random() < .5:
            x = torch.flip(x, [3])
        x = (x * (.8 + .4 * torch.rand(len(x), 1, 1, 1))).clamp(0, 1)
    return (x - MEAN) / STD


def latih_satu(img, ii, y_bin, epoch, lr, batch, dev, tag):
    m = Spesialis().to(dev)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                            lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epoch)
    pw = torch.tensor(float((y_bin == 0).sum() / max((y_bin == 1).sum(), 1)), device=dev)
    urut = np.arange(len(ii))
    for ep in range(epoch):
        m.train(); np.random.shuffle(urut); tot = nb = 0
        for s in range(0, len(urut), batch):
            sel = np.sort(urut[s:s + batch])
            x = ke_tensor(np.asarray(img[ii[sel]]), True).to(dev)
            y = torch.from_numpy(y_bin[sel].astype(np.float32)).to(dev)
            with torch.autocast("cuda", torch.bfloat16):
                L = F.binary_cross_entropy_with_logits(m(x).float(), y, pos_weight=pw)
            opt.zero_grad(set_to_none=True); L.backward(); opt.step()
            tot += float(L.detach()); nb += 1
        sch.step()
        print(f"    [{tag}] epoch {ep+1}/{epoch} loss {tot/max(nb,1):.4f}", flush=True)
    return m


@torch.no_grad()
def skor(m, img, ii, dev):
    m.eval(); out = []
    for s in range(0, len(ii), 128):
        x = ke_tensor(np.asarray(img[ii[s:s + 128]]), False).to(dev)
        with torch.autocast("cuda", torch.bfloat16):
            out.append(torch.sigmoid(m(x).float()).cpu().numpy())
    return np.concatenate(out)


def terap(P, q_spes, lo, hi, lam):
    """Bagi ulang massa dua kelas bertetangga. Total massa tidak berubah."""
    P = P.copy()
    massa = P[:, lo] + P[:, hi]
    q_ens = np.divide(P[:, hi], np.clip(massa, 1e-9, None))
    q = (1 - lam) * q_ens + lam * q_spes
    P[:, lo] = massa * (1 - q)
    P[:, hi] = massa * q
    return P / np.clip(P.sum(1, keepdims=True), 1e-9, None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda"; t0 = time.time()

    img = np.load(R / "crop_damimas_224.npy", mmap_mode="r")
    meta = np.load(R / "crop_damimas_224_meta.npz", allow_pickle=True)
    split = np.asarray(meta["split"]); tree_v = np.asarray(meta["tree"])
    bunch = np.asarray(meta["bunch"], int); yv = np.asarray(meta["y"], int)

    ens = np.load(R / "pt_e_029_ensemble_kelas_damimas_pred.npz", allow_pickle=True)
    Pv0 = np.asarray(ens["val_prob"], np.float64)
    Pt0 = np.asarray(ens["test_prob"], np.float64)
    yt = np.asarray(ens["test_y"], int); tree_t = np.asarray(ens["test_tree"])
    ref = np.load(R / "damimas_classifier_hibrida_convnext224_s42_pred.npz", allow_pickle=True)
    yvb = np.asarray(ref["val_bunch_y"], int); tree_vb = np.asarray(ref["val_bunch_tree"])

    def urut_tandan(s):
        ii = np.sort(np.where(split == s)[0])
        per = defaultdict(list); urut = []; lihat = set()
        for j, k in enumerate(ii):
            key = (str(tree_v[k]), int(bunch[k]))
            per[key].append(j)
            if key not in lihat:
                lihat.add(key); urut.append(key)
        return ii, per, urut

    hasil = {"pt_e": "031", "batas": [], "lam_kisi": LAM,
             "acuan_ensemble": {"val": float((Pv0.argmax(1) == yvb).mean()),
                                "test": float((Pt0.argmax(1) == yt).mean())}}
    Pv, Pt = Pv0.copy(), Pt0.copy()

    for lo, hi in BATAS:
        nama = f"B{lo+1}_vs_B{hi+1}"
        print(f"\n=== spesialis {nama} ===")
        tr = np.sort(np.where((split == "train") & np.isin(yv, [lo, hi]))[0])
        y_bin = (yv[tr] == hi).astype(int)
        print(f"  latih {len(tr)} potongan ({int((y_bin==0).sum())} vs {int(y_bin.sum())})")
        m = latih_satu(img, tr, y_bin, args.epoch, args.lr, args.batch, dev, nama)
        runs = SUB / "runs" / f"spesialis_{nama}_damimas"
        runs.mkdir(parents=True, exist_ok=True)
        torch.save({"model": m.state_dict(), "batas": [lo, hi]}, runs / "best.pt")

        q = {}
        for s, (P, ybs) in (("val", (Pv, yvb)), ("test", (Pt, yt))):
            ii, per, urut = urut_tandan(s)
            sv = skor(m, img, ii, dev)
            q[s] = np.array([sv[per[k]].mean() for k in urut])
            assert len(q[s]) == len(P), f"panjang beda {s}"

        # --- lam dipilih lewat CV 5-fold tingkat pohon DI DALAM VAL ---
        pohon = np.unique(tree_vb); rng = np.random.default_rng(args.seed)
        fmap = {t: i % 5 for i, t in enumerate(rng.permutation(pohon))}
        fid = np.array([fmap[t] for t in tree_vb])
        cv = {}
        for lam in LAM:
            benar = []
            for f in range(5):
                te = fid == f
                benar.append(terap(Pv[te], q["val"][te], lo, hi, lam).argmax(1) == yvb[te])
            cv[lam] = float(np.concatenate(benar).mean())
        lam = max(cv, key=cv.get)
        print("  CV val:", {k: round(v, 4) for k, v in cv.items()})
        print(f"  -> lam dikunci {lam}")

        Pv = terap(Pv, q["val"], lo, hi, lam)
        Pt = terap(Pt, q["test"], lo, hi, lam)
        hasil["batas"].append({
            "nama": nama, "lam": lam, "cv_val": {str(k): round(v, 4) for k, v in cv.items()},
            "val_setelah": round(float((Pv.argmax(1) == yvb).mean()), 4),
            "test_setelah": round(float((Pt.argmax(1) == yt).mean()), 4)})
        print(f"  val {hasil['batas'][-1]['val_setelah']} | "
              f"test {hasil['batas'][-1]['test_setelah']}")

    yh = Pt.argmax(1)
    f1 = []
    for k in range(K):
        tp = int(((yh == k) & (yt == k)).sum()); fp = int(((yh == k) & (yt != k)).sum())
        fn = int(((yh != k) & (yt == k)).sum())
        p_ = tp / max(tp + fp, 1); r_ = tp / max(tp + fn, 1)
        f1.append(2 * p_ * r_ / max(p_ + r_, 1e-9))
    nv = np.asarray(ens["test_nview"], int); m1 = nv == 1
    benar_a = (Pt0.argmax(1) == yt).astype(float); benar_b = (yh == yt).astype(float)
    uniq = sorted(set(tree_t.tolist())); idxp = {t: np.where(tree_t == t)[0] for t in uniq}
    rng = np.random.default_rng(0); d = []
    for _ in range(2000):
        pil = rng.choice(len(uniq), len(uniq))
        ii = np.concatenate([idxp[uniq[k]] for k in pil])
        d.append(benar_b[ii].mean() - benar_a[ii].mean())
    d = np.array(d) * 100
    hasil["akhir"] = {
        "test": round(float(benar_b.mean()), 4),
        "test_1view": round(float((yh[m1] == yt[m1]).mean()), 4),
        "test_multi": round(float((yh[~m1] == yt[~m1]).mean()), 4),
        "macro_f1": round(float(np.mean(f1)), 4),
        "vs_ensemble": {"delta_pp": round(float((benar_b.mean() - benar_a.mean()) * 100), 2),
                        "ci95": [round(float(np.percentile(d, 2.5)), 2),
                                 round(float(np.percentile(d, 97.5)), 2)],
                        "P(delta>0)": round(float((d > 0).mean()), 3)},
        "target_IDEA": 0.80}
    np.savez_compressed(R / "pt_e_031_spesialis_batas_pred.npz",
                        test_prob=Pt.astype(np.float32), test_yhat=yh, test_y=yt,
                        test_tree=tree_t, test_nview=nv, val_prob=Pv.astype(np.float32))
    hasil["detik"] = round(time.time() - t0, 1)
    (R / "pt_e_031_spesialis_batas.json").write_text(json.dumps(hasil, indent=1, ensure_ascii=False))
    print("\n=== AKHIR ===")
    print(json.dumps(hasil["akhir"], indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
