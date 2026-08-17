"""PT-E-002b — Embedding re-ID tandan, dilatih dari graf identitas GT.

Dijalankan karena PT-E-002a gugur: deskriptor penampilan TANGAN (histogram HSV,
statistik warna, ketajaman) menambah nyaris nol di atas geometri — AUC val
0,9301 -> 0,9307, F1 val 0,4518 -> 0,4485. Dugaan penyebabnya masuk akal
secara fisik: negatif yang harus dikalahkan semuanya berasal dari POHON YANG
SAMA, jadi warnanya nyaris identik (pencahayaan sama, kematangan sama,
varietas sama), sementara tandan yang SAMA dilihat dari 90 derajat berbeda
justru berubah rupa. Warna global karena itu hampir tidak membawa informasi
diskriminatif di sini.

Yang dicoba sekarang: representasi yang DILATIH untuk tugas itu persis.

  backbone  ResNet-18 (init ImageNet), kepala proyeksi -> 128-d, dinormalkan L2
  loss      supervised contrastive; positif = `bunch_id` sama, negatif = seluruh
            potongan lain di dalam batch
  batch     disusun PER POHON (16 pohon sekaligus) supaya mayoritas negatifnya
            adalah tandan lain di pohon yang sama -- hard negative yang benar,
            bukan negatif lintas-pohon yang sepele

Hanya split train (716 pohon) yang dipakai melatih. Val/test hanya diinferensi.

Keluaran: `results/reid_embedding.npz` {"<tree>|<sisi>|<box>": vektor 128-d}
plus bobotnya di `runs/reid_resnet18/best.pt`.

Pemakaian:
    .venv/bin/python pipeline-pertandan/scripts/reid_pertandan.py --epoch 30
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

sys.path.insert(0, str(Path(__file__).parent))
import penaut_pertandan as PP           # noqa: E402

DS = PP.DS
SUB = PP.SUB
SISI = 128
PAD = 0.10
SEED = 0


# --------------------------------------------------------------------------
def bangun_potongan(ids: list[str], cache: Path):
    """Potongan 128x128 untuk seluruh kotak GT. Disimpan sekali, dipakai ulang."""
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["img"], list(z["kunci"]), list(z["tree"]), list(z["bunch"])
    img, kunci, tree_of, bunch_of = [], [], [], []
    for n, t in enumerate(ids, 1):
        _, kotak = PP.muat_pohon(t)
        per_stem = {}
        for b in kotak:
            per_stem.setdefault(b["stem"], []).append(b)
        for stem, bs in per_stem.items():
            f = PP.cari_citra(stem)
            im = cv2.imread(str(f)) if f else None
            for b in bs:
                if im is None:
                    c = np.zeros((SISI, SISI, 3), np.uint8)
                else:
                    h, w = im.shape[:2]
                    x1, y1, x2, y2 = b["px"]
                    dx, dy = (x2 - x1) * PAD, (y2 - y1) * PAD
                    x1 = max(0, int(x1 - dx)); y1 = max(0, int(y1 - dy))
                    x2 = min(w, int(x2 + dx)); y2 = min(h, int(y2 + dy))
                    c = (cv2.resize(im[y1:y2, x1:x2], (SISI, SISI))
                         if x2 - x1 > 3 and y2 - y1 > 3
                         else np.zeros((SISI, SISI, 3), np.uint8))
                img.append(c)
                kunci.append(f"{t}|{b['s']}|{b['i']}")
                tree_of.append(t)
                bunch_of.append(-1 if b["bid"] is None else b["bid"])
        if n % 100 == 0:
            print(f"  potongan: {n}/{len(ids)} pohon", flush=True)
    img = np.stack(img)
    np.savez(cache, img=img, kunci=np.array(kunci), tree=np.array(tree_of),
             bunch=np.array(bunch_of))
    return img, kunci, tree_of, bunch_of


class Reid(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        b = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        b.fc = nn.Identity()
        self.b = b
        self.p = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True),
                               nn.Linear(256, dim))

    def forward(self, x):
        return F.normalize(self.p(self.b(x)), dim=1)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def ke_tensor(batch: np.ndarray, latih: bool, dev) -> torch.Tensor:
    x = torch.from_numpy(batch[:, :, :, ::-1].copy()).permute(0, 3, 1, 2).float() / 255
    if latih:
        if random.random() < 0.5:
            x = torch.flip(x, [3])
        x = x * (0.8 + 0.4 * torch.rand(len(x), 1, 1, 1))       # jitter kecerahan
        x = x.clamp(0, 1)
    return ((x - MEAN) / STD).to(dev, non_blocking=True)


def supcon(z: torch.Tensor, label: torch.Tensor, suhu: float = 0.1):
    sim = z @ z.T / suhu
    n = len(z)
    diag = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(diag, -1e4)
    pos = (label[:, None] == label[None, :]) & ~diag
    ada = pos.any(1)
    if not ada.any():
        return None
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    return -( (logp * pos).sum(1)[ada] / pos.sum(1)[ada] ).mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=30)
    ap.add_argument("--fold", type=int, default=-1,
                    help="kalau >=0: latih HANYA di pohon train di luar fold ini "
                         "(untuk fitur embedding out-of-fold)")
    ap.add_argument("--nfold", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--pohon-per-batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    man = PP.muat_manifest()
    ids = {s: [t for t, v in man.items() if v == s] for s in ["train", "val", "test"]}
    semua = ids["train"] + ids["val"] + ids["test"]
    print("membangun potongan 128x128 (sekali, lalu di-cache)...")
    img, kunci, tree_of, bunch_of = bangun_potongan(
        semua, SUB / "results" / "potongan_reid.npz")
    print(f"  {len(img)} potongan")

    idx_pohon = {}
    for i, t in enumerate(tree_of):
        idx_pohon.setdefault(t, []).append(i)
    # id tandan global (unik lintas pohon)
    gid = {}
    label = np.zeros(len(img), np.int64)
    for i, (t, b) in enumerate(zip(tree_of, bunch_of)):
        k = (t, b)
        if b < 0:
            label[i] = -1 - i                      # tak tertaut: unik sendiri
        else:
            label[i] = gid.setdefault(k, len(gid))

    dev = "cuda"
    m = Reid().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epoch)
    tr = sorted(ids["train"])
    if args.fold >= 0:
        # pohon fold ini DITAHAN: embedding-nya jadi genuinely unseen bagi mereka
        tr = [t for i, t in enumerate(tr) if i % args.nfold != args.fold]
        print(f"fold {args.fold}/{args.nfold}: melatih di {len(tr)} pohon "
              f"(menahan {len(ids['train']) - len(tr)})")
    riwayat = []
    t0 = time.time()
    for ep in range(args.epoch):
        m.train()
        random.shuffle(tr)
        tot, nb = 0.0, 0
        for s in range(0, len(tr), args.pohon_per_batch):
            sel = tr[s:s + args.pohon_per_batch]
            # 7 pohon di korpus tidak punya kotak sama sekali -> tidak ada potongan
            ii = [i for t in sel for i in idx_pohon.get(t, [])]
            if len(ii) < 8:
                continue
            x = ke_tensor(img[ii], True, dev)
            y = torch.from_numpy(label[ii]).to(dev)
            with torch.autocast("cuda", torch.bfloat16):
                z = m(x)
            L = supcon(z.float(), y)
            if L is None:
                continue
            opt.zero_grad(set_to_none=True)
            L.backward()
            opt.step()
            tot += float(L); nb += 1
        sched.step()
        riwayat.append(round(tot / max(nb, 1), 4))
        print(f"  epoch {ep+1}/{args.epoch}  loss {riwayat[-1]:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    run = SUB / "runs" / f"reid_resnet18{args.tag}"
    run.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), run / "best.pt")
    (run / "riwayat.json").write_text(json.dumps(
        {"loss_per_epoch": riwayat, "epoch": args.epoch, "lr": args.lr,
         "pohon_per_batch": args.pohon_per_batch, "seed": SEED}, indent=1))

    m.eval()
    emb = np.zeros((len(img), 128), np.float32)
    with torch.no_grad():
        for s in range(0, len(img), 256):
            x = ke_tensor(img[s:s + 256], False, dev)
            with torch.autocast("cuda", torch.bfloat16):
                emb[s:s + 256] = m(x).float().cpu().numpy()
    f_out = SUB / "results" / f"reid_embedding{args.tag}.npz"
    np.savez_compressed(f_out, **{k: e for k, e in zip(kunci, emb)})
    print(f"-> {run/'best.pt'}  &  {f_out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
