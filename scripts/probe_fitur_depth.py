"""Uji langsung: apakah statistik depth TERPOOL menambah informasi di atas RGB?

Ablasi cabang-CNN-depth (V2-E-015) memberi delta -1,4pp val / -2,0pp test atas
3 seed — depth tidak menolong. Tapi desain cabang itu melanggar temuan
diagnostiknya sendiri: probe mengukur relief sebagai SATU SKALAR hasil pooling
wilayah (AUC B1-vs-B4 0,73), sedangkan cabang CNN menaruh global pooling di
paling akhir — empat conv ber-stride lebih dulu bekerja pada medan ber-SNR ~0,3
per piksel, memperkuat derau sebelum sempat dirata-rata.

Skrip ini menguji hipotesisnya tanpa melatih CNN apa pun: ambil fitur penultimate
dari classifier RGB yang sudah terlatih, tempelkan statistik depth yang SUDAH
terpool secara analitik, lalu bandingkan regresi logistik dengan dan tanpa
statistik itu. Kalau depth memang membawa informasi tambahan, ia harus muncul
di sini — ini kondisi paling menguntungkan yang bisa diberikan ke depth.

Usage:
    .venv/bin/python scripts/probe_fitur_depth.py --model runs_fase6/sd202_rgb/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from train_crop_classifier import IMG, K, Model  # noqa: E402

CROPS = Path("/workspace/crops_fase6")


def statistik_depth(dep, msk):
    """Statistik depth TERPOOL per crop — persis besaran yang probe ukur.

    dep[...,0] = relief (128 = 0 cm, skala +-10 cm), dep[...,1] = mask valid.
    Semua dikembalikan dalam cm supaya skalanya interpretable.
    """
    rel = (dep[..., 0].astype(np.float32) - 128.0) / 127.0 * 10.0     # cm
    valid = dep[..., 1] > 127
    di_dalam = (msk > 127) & valid
    di_luar = (msk <= 127) & valid
    if di_dalam.sum() < 20 or di_luar.sum() < 20:
        return np.zeros(8, np.float32)
    r_in, r_out = rel[di_dalam], rel[di_luar]
    return np.array([
        np.median(r_out) - np.median(r_in),          # relief: cincin - box (sinyal utama)
        np.mean(r_out) - np.mean(r_in),
        np.median(r_in), np.median(r_out),
        np.std(r_in), np.std(r_out),
        float(di_dalam.sum()) / max(float((msk > 127).sum()), 1.0),   # cakupan valid di box
        np.percentile(r_in, 90) - np.percentile(r_in, 10),            # rentang relief dalam box
    ], np.float32)


@torch.no_grad()
def fitur_rgb(model, rgb, msk, dev, batch=64):
    keluar = []
    for i in range(0, len(rgb), batch):
        r = torch.from_numpy(np.ascontiguousarray(rgb[i:i + batch])).permute(0, 3, 1, 2).float() / 255.0
        m = torch.from_numpy(np.ascontiguousarray(msk[i:i + batch]))[:, None].float() / 255.0
        r = F.interpolate(r, (IMG, IMG), mode="bilinear", align_corners=False)
        m = F.interpolate(m, (IMG, IMG), mode="bilinear", align_corners=False)
        r = (r - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / \
            torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        x = torch.cat([r, m * 2 - 1], 1).to(dev)
        with torch.amp.autocast("cuda"):
            keluar.append(model.bb(x).float().cpu().numpy())
    return np.concatenate(keluar)


def nilai(Xtr, ytr, Xva, yva, Xte, yte, tag):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=4000, C=0.05, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    hasil = {}
    for nm, X, y in (("val", Xva, yva), ("test", Xte, yte)):
        p = clf.predict(sc.transform(X))
        hasil[nm] = {
            "akurasi": float((p == y).mean()),
            "akurasi_pm1": float((np.abs(p - y) <= 1).mean()),
            "macro_recall": float(np.mean([((p == k) & (y == k)).sum() / max((y == k).sum(), 1)
                                           for k in range(K)])),
        }
    print(f"  {tag:<28} val={hasil['val']['akurasi']:.4f}  test={hasil['test']['akurasi']:.4f}  "
          f"test_mR={hasil['test']['macro_recall']:.4f}")
    return hasil


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="results/probe_fitur_depth.json")
    args = ap.parse_args()
    dev = "cuda"

    rgb = np.load(CROPS / "crops352_rgb.npy", mmap_mode="r")
    dep = np.load(CROPS / "crops352_dep.npy", mmap_mode="r")
    msk = np.load(CROPS / "crops352_msk.npy", mmap_mode="r")
    m = np.load(CROPS / "crops352_meta.npz", allow_pickle=True)
    y, split = m["y"], m["split"]

    ck = torch.load(args.model, map_location="cpu")
    a = ck["args"]
    model = Model(a["backbone"], a["mode"] == "rgbd", a.get("gate_init", 0.1), a["head"])
    model.load_state_dict(ck["model"]); model.to(dev).eval()

    print("ekstrak fitur RGB penultimate...")
    Fr = fitur_rgb(model, np.asarray(rgb), np.asarray(msk), dev)
    print("hitung statistik depth terpool...")
    Fd = np.stack([statistik_depth(np.asarray(dep[i]), np.asarray(msk[i])) for i in range(len(y))])

    tr, va, te = split == "train", split == "val", split == "test"
    print(f"fitur RGB {Fr.shape}, fitur depth {Fd.shape}\n")

    print("relief (cincin-box) rata-rata per kelas, cm — cek sinyalnya masih ada di crop:")
    for k in range(K):
        print(f"  B{k+1}: {np.median(Fd[y == k, 0]):+.2f} cm  (n={(y==k).sum()})")
    print()

    hasil = {
        "model": args.model,
        "rgb_saja": nilai(Fr[tr], y[tr], Fr[va], y[va], Fr[te], y[te], "RGB saja"),
        "depth_saja": nilai(Fd[tr], y[tr], Fd[va], y[va], Fd[te], y[te], "statistik depth saja"),
        "rgb_plus_depth": nilai(np.hstack([Fr, Fd])[tr], y[tr], np.hstack([Fr, Fd])[va], y[va],
                                np.hstack([Fr, Fd])[te], y[te], "RGB + statistik depth"),
    }
    d_val = hasil["rgb_plus_depth"]["val"]["akurasi"] - hasil["rgb_saja"]["val"]["akurasi"]
    d_te = hasil["rgb_plus_depth"]["test"]["akurasi"] - hasil["rgb_saja"]["test"]["akurasi"]
    hasil["delta_depth"] = {"val": d_val, "test": d_te}
    print(f"\n  kontribusi depth: val {d_val:+.4f}   test {d_te:+.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(hasil, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
