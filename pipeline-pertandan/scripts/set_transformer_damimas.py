"""Kepala multi-view berbasis himpunan untuk tandan DAMIMAS.

Masukan berasal dari ``classifier_hibrida_damimas.py``: satu fitur visual dan
probabilitas per tampak, dengan urutan tampak yang sudah dikelompokkan menurut
identitas tandan GT. Kepala ini belajar interaksi antar-tampak, lalu memberi
koreksi residual terhadap rerata probabilitas. Tidak ada posisi urutan yang
dipakai sehingga permutasi tampak tidak mengubah prediksi.

Kotak dan kelompok pada eksperimen ini masih GT. Karena itu hasilnya mengukur
plafon modul classifier multi-view, bukan angka end-to-end. Model dilatih hanya
pada TRAIN, checkpoint/blend/ambang ordinal dipilih hanya pada VAL, dan TEST
baru diinferensi setelah seluruh konfigurasi dikunci.
"""
from __future__ import annotations

import argparse
import copy
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


SUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import classifier_klasik_damimas as CK  # noqa: E402
import stacker_damimas as SD  # noqa: E402


class DataTandan(Dataset):
    def __init__(self, fitur, prob_h, prob_base, y, nview):
        self.fitur = fitur.astype(np.float32)
        self.prob_h = prob_h.astype(np.float32)
        self.prob_base = prob_base.astype(np.float32)
        self.y = y.astype(np.int64)
        self.nview = nview.astype(int)
        self.offset = np.r_[0, np.cumsum(self.nview)]
        if self.offset[-1] != len(self.fitur):
            raise RuntimeError("Jumlah nview tidak sama dengan panjang fitur tampak")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        a, b = self.offset[i:i + 2]
        return self.fitur[a:b], self.prob_h[a:b], self.prob_base[a:b], self.y[i], i


def collate(batch):
    n = np.asarray([len(x[0]) for x in batch], int)
    m = int(n.max())
    d = batch[0][0].shape[1]
    fitur = np.zeros((len(batch), m, d), np.float32)
    ph = np.zeros((len(batch), m, 4), np.float32)
    pb = np.zeros((len(batch), m, 4), np.float32)
    mask = np.ones((len(batch), m), bool)
    for j, (f, h, b, _y, _i) in enumerate(batch):
        fitur[j, :len(f)] = f
        ph[j, :len(f)] = h
        pb[j, :len(f)] = b
        mask[j, :len(f)] = False
    return (torch.from_numpy(fitur), torch.from_numpy(ph), torch.from_numpy(pb),
            torch.from_numpy(mask), torch.tensor([x[3] for x in batch]),
            torch.tensor([x[4] for x in batch]))


class KepalaSet(nn.Module):
    def __init__(self, d_in: int, d_model: int = 192, lapis: int = 2,
                 kepala: int = 6, dropout: float = .15):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_in + 8), nn.Linear(d_in + 8, d_model),
            nn.GELU(), nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=kepala, dim_feedforward=3 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=lapis,
                                             enable_nested_tensor=False)
        self.attn = nn.Linear(d_model, 1)
        self.residual = nn.Sequential(
            nn.LayerNorm(3 * d_model + 2), nn.Dropout(dropout),
            nn.Linear(3 * d_model + 2, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 4),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.log_skala_base = nn.Parameter(torch.zeros(()))

    @staticmethod
    def agregasi_prob(prob, mask):
        sah = (~mask).float()
        bobot = prob.max(-1).values * sah
        return (prob * bobot.unsqueeze(-1)).sum(1) / bobot.sum(1, keepdim=True).clamp_min(1e-6)

    def forward(self, fitur, prob_h, prob_base, mask, drop_view=False):
        if drop_view and self.training:
            jatuh = (torch.rand_like(mask.float()) < .12) & ~mask
            # Sedikitnya satu tampak selalu tersisa.
            semua = ((~mask) & ~jatuh).sum(1) == 0
            if semua.any():
                first = (~mask[semua]).float().argmax(1)
                jatuh[semua, first] = False
            mask = mask | jatuh
        x = self.proj(torch.cat([fitur, prob_h, prob_base], -1))
        x = self.encoder(x, src_key_padding_mask=mask)
        sah = (~mask).float()
        mean = (x * sah.unsqueeze(-1)).sum(1) / sah.sum(1, keepdim=True).clamp_min(1)
        xmax = x.masked_fill(mask.unsqueeze(-1), -1e4).max(1).values
        a = self.attn(x).squeeze(-1).masked_fill(mask, -1e4)
        attn = torch.softmax(a, 1)
        pool = (x * attn.unsqueeze(-1)).sum(1)
        n = sah.sum(1, keepdim=True)
        meta = torch.cat([n / 6., torch.log1p(n) / math.log(7.)], 1)
        base = self.agregasi_prob(prob_base, mask)
        skala = self.log_skala_base.exp().clamp(.25, 4.)
        logit = skala * torch.log(base.clamp_min(1e-6))
        return logit + self.residual(torch.cat([mean, xmax, pool, meta], 1))


def muat_split(z, split: str) -> dict:
    wajib = [f"{split}_view_fitur", f"{split}_view_hibrida",
             f"{split}_view_prob", f"{split}_bunch_y",
             f"{split}_bunch_tree", f"{split}_bunch_nview"]
    hilang = [k for k in wajib if k not in z.files]
    if hilang:
        raise RuntimeError(f"Dump classifier tidak lengkap untuk {split}: {hilang}")
    return {"fitur": z[f"{split}_view_fitur"],
            "prob_h": z[f"{split}_view_hibrida"],
            "prob_base": z[f"{split}_view_prob"],
            "y": z[f"{split}_bunch_y"],
            "tree": z[f"{split}_bunch_tree"].astype(str),
            "nview": z[f"{split}_bunch_nview"]}


@torch.inference_mode()
def infer(model, loader, n):
    model.eval()
    out = np.zeros((n, 4), np.float32)
    for fitur, ph, pb, mask, _y, idx in loader:
        with torch.autocast("cuda", torch.bfloat16):
            q = torch.softmax(model(fitur.cuda(non_blocking=True),
                                    ph.cuda(non_blocking=True),
                                    pb.cuda(non_blocking=True),
                                    mask.cuda(non_blocking=True)).float(), 1)
        out[idx.numpy()] = q.cpu().numpy()
    return out


def agregasi_langsung(prob, nview):
    out, a = [], 0
    for n in nview:
        q = prob[a:a + n]
        out.append(np.average(q, axis=0, weights=np.maximum(q.max(1), 1e-6)))
        a += n
    return np.asarray(out)


def nilai_val(p_set, data):
    dasar = agregasi_langsung(data["prob_base"], data["nview"])
    w, q, p, ranking = CK.pilih_blend({"set": p_set, "dasar": dasar}, data["y"])
    return {"bobot": w, "aturan": q[2], "tau": q[3], "metrik": q[4],
            "objective": q[0], "prob": p,
            "ranking": CK.serial_ranking(ranking)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=SUB / "runs" /
                    "classifier_hibrida_damimas_convnext_tiny_s42" /
                    "fitur_dan_prediksi.npz")
    ap.add_argument("--tag", default="convnext_tiny_s42")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    z = np.load(args.input, allow_pickle=True)
    data = {s: muat_split(z, s) for s in ("train", "val")}
    ds = {s: DataTandan(data[s]["fitur"], data[s]["prob_h"],
                        data[s]["prob_base"], data[s]["y"], data[s]["nview"])
          for s in data}
    loader_val = DataLoader(ds["val"], batch_size=args.batch, shuffle=False,
                            num_workers=0, pin_memory=True, collate_fn=collate)
    freq = np.bincount(data["train"]["y"].astype(int), minlength=4).astype(float)
    sw = np.sqrt(freq.max() / freq[data["train"]["y"].astype(int)])
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    loader_train = DataLoader(ds["train"], batch_size=args.batch, sampler=sampler,
                              num_workers=0, pin_memory=True, collate_fn=collate)

    model = KepalaSet(data["train"]["fitur"].shape[1]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=2e-6)
    cw = torch.tensor(np.sqrt(freq.max() / freq), device="cuda", dtype=torch.float32)
    run = SUB / "runs" / f"set_transformer_damimas_{args.tag}"
    run.mkdir(parents=True, exist_ok=False)
    best, best_epoch, stale, history = -math.inf, 0, 0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); total = nb = 0.
        for fitur, ph, pb, mask, y, _idx in loader_train:
            fitur = fitur.cuda(non_blocking=True); ph = ph.cuda(non_blocking=True)
            pb = pb.cuda(non_blocking=True); mask = mask.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            with torch.autocast("cuda", torch.bfloat16):
                logit = model(fitur, ph, pb, mask, drop_view=True)
                ce = F.cross_entropy(logit.float(), y, weight=cw,
                                     label_smoothing=.02)
                p = torch.softmax(logit.float(), 1)
                ord_loss = F.smooth_l1_loss(
                    p @ torch.arange(4, device="cuda", dtype=torch.float32), y.float())
                loss = ce + .10 * ord_loss
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step(); total += float(loss.detach()); nb += 1
        sch.step()
        pv = infer(model, loader_val, len(ds["val"]))
        q = nilai_val(pv, data["val"])
        row = {"epoch": ep, "loss": total / max(nb, 1),
               "val_acc": q["metrik"]["akurasi"],
               "val_macro_f1": q["metrik"]["macro_f1"],
               "objective": q["objective"],
               "skala_base": float(model.log_skala_base.exp().detach()),
               "detik": time.time() - t0}
        history.append(row)
        print(f"ep {ep:02d} loss={row['loss']:.4f} val={row['val_acc']:.4f}/"
              f"{row['val_macro_f1']:.4f} obj={row['objective']:.5f}", flush=True)
        state = {"state_dict": copy.deepcopy(model.state_dict()), "epoch": ep,
                 "args": vars(args), "nilai_val": q, "history": history}
        torch.save(state, run / "last.pt")
        if ep % 5 == 0:
            torch.save(state, run / f"epoch_{ep:03d}.pt")
        if q["objective"] > best + 1e-9:
            best, best_epoch, stale = q["objective"], ep, 0
            torch.save(state, run / "best.pt")
        else:
            stale += 1
        if stale >= args.patience:
            print(f"early stop: {stale} epoch tanpa kenaikan", flush=True)
            break

    ck = torch.load(run / "best.pt", map_location="cuda", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    cfg = ck["nilai_val"]
    # Baru sekarang baca label dan fitur TEST dari arsip.
    data["test"] = muat_split(z, "test")
    ds["test"] = DataTandan(data["test"]["fitur"], data["test"]["prob_h"],
                            data["test"]["prob_base"], data["test"]["y"],
                            data["test"]["nview"])
    loader_test = DataLoader(ds["test"], batch_size=args.batch, shuffle=False,
                             num_workers=0, pin_memory=True, collate_fn=collate)
    pset_test = infer(model, loader_test, len(ds["test"]))
    dasar_test = agregasi_langsung(data["test"]["prob_base"], data["test"]["nview"])
    kandidat_test = {"set": pset_test, "dasar": dasar_test}
    ptest = CK.gabung_prob(cfg["bobot"], kandidat_test)
    yh = SD.prediksi_prob(ptest, cfg["aturan"], cfg["tau"])
    mt = SD.metrik(data["test"]["y"], yh)
    hasil = {
        "dataset": "SawitMVC-YOLO-Damimas",
        "protokol": "TRAIN; checkpoint/blend/tau di VAL; TEST sekali setelah terkunci",
        "kaveat": "fitur crop GT + tautan GT; metrik modul/oracle",
        "arsitektur": "set transformer residual permutation-invariant",
        "input": str(args.input), "best_epoch": best_epoch,
        "best_objective": best,
        "n": {s: {"tandan": int(len(data[s]["y"])),
                    "pohon": int(len(set(data[s]["tree"]))) } for s in data},
        "terpilih": {k: cfg[k] for k in ("bobot", "aturan", "tau", "metrik",
                                          "objective", "ranking")},
        "test": mt,
        "test_subgrup": SD.metrik_subgrup(
            data["test"]["y"], yh, np.r_[0, np.cumsum(data["test"]["nview"])]),
        "history": history,
        "bobot": str((run / "best.pt").relative_to(SUB)),
    }
    out = SUB / "results" / f"damimas_set_transformer_{args.tag}.json"
    pred = SUB / "results" / f"damimas_set_transformer_{args.tag}_pred.npz"
    out.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
    np.savez_compressed(pred, val_prob=cfg["prob"], val_y=data["val"]["y"],
                        val_tree=data["val"]["tree"],
                        test_prob=ptest, test_y=data["test"]["y"], test_yhat=yh,
                        test_tree=data["test"]["tree"],
                        test_nview=data["test"]["nview"])
    hist = SUB / "results" / "riwayat_epoch" / f"set_transformer_damimas_{args.tag}.csv"
    hist.parent.mkdir(parents=True, exist_ok=True)
    with hist.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
    print(json.dumps({"best_epoch": best_epoch, "val": cfg["metrik"],
                      "test": mt}, indent=2, ensure_ascii=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
