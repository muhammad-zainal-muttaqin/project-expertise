"""Kepala ordinal CORN untuk kematangan tandan.

Keluarannya satu skor kontinu s in [0,3] (0 = paling matang / B1, 3 = paling
mentah / B4), bukan argmax empat kelas. Keputusan kasar matang/belum dan
keputusan halus di dalam tiap kelompok keduanya menjadi ambang pada skor yang
sama, sehingga tidak ada informasi yang dibuang di akar hierarki.
"""
import json, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, models
from PIL import Image

OUT = "/workspace/crops953"
DEV = "cuda"
EP, SZ = 18, 176
torch.manual_seed(42); np.random.seed(42)

idx = json.load(open(f"{OUT}/index.json"))
by = {s: [r for r in idx if r["split"] == s] for s in ["train", "val", "test"]}
print({k: len(v) for k, v in by.items()}, flush=True)

norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tr_tf = T.Compose([T.RandomResizedCrop(SZ, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
                   T.RandomHorizontalFlip(), T.ColorJitter(0.3, 0.3, 0.2, 0.04),
                   T.ToTensor(), norm, T.RandomErasing(p=0.25, scale=(0.02, 0.12))])
ev_tf = T.Compose([T.Resize((SZ, SZ)), T.ToTensor(), norm])


class DS(Dataset):
    def __init__(self, recs, tf, sp): self.r, self.tf, self.sp = recs, tf, sp
    def __len__(self): return len(self.r)
    def __getitem__(self, i):
        r = self.r[i]
        return self.tf(Image.open(f"{OUT}/{self.sp}/{r['f']}").convert("RGB")), r["cls"], i


def loader(sp, tf, bs, sh):
    return DataLoader(DS(by[sp], tf, sp), batch_size=bs, shuffle=sh, num_workers=10,
                      pin_memory=True, persistent_workers=True)


class CORN(nn.Module):
    """ConvNeXt-Small + 3 logit ordinal terkondisi (K-1 untuk K=4)."""
    def __init__(self):
        super().__init__()
        b = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        # classifier[0] = LayerNorm2d (butuh 4-D), classifier[1] = Flatten
        self.feat = nn.Sequential(b.features, b.avgpool, b.classifier[0], b.classifier[1])
        self.ord = nn.Linear(768, 3)
        self.cls = nn.Linear(768, 4)      # kepala tambahan untuk regularisasi
    def forward(self, x):
        f = self.feat(x)
        return self.ord(f), self.cls(f)


def corn_loss(logits, y):
    """Setiap tugas k dilatih hanya pada sampel dengan y >= k (terkondisi)."""
    loss, n = 0.0, 0
    for k in range(3):
        m = y >= k
        if m.sum() == 0:
            continue
        tgt = (y[m] > k).float()
        loss = loss + F.binary_cross_entropy_with_logits(logits[m, k], tgt, reduction="sum")
        n += int(m.sum())
    return loss / max(n, 1)


def corn_score(logits):
    """s = sum_k P(y > k), memakai probabilitas kumulatif terkondisi."""
    p = torch.sigmoid(logits)
    cum = torch.cumprod(p, dim=1)
    return cum.sum(1), cum          # skor kontinu, dan P(y>k)


m = CORN().to(DEV).to(memory_format=torch.channels_last)
tl = loader("train", tr_tf, 40, True)
opt = torch.optim.AdamW(m.parameters(), lr=2.5e-4, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2.5e-4, epochs=EP, steps_per_epoch=len(tl))
scaler = torch.amp.GradScaler()


@torch.no_grad()
def infer(sp):
    m.eval()
    dl = loader(sp, ev_tf, 64, False)
    S = np.zeros(len(by[sp]), np.float32)
    C = np.zeros((len(by[sp]), 3), np.float32)
    P = np.zeros((len(by[sp]), 4), np.float32)
    for x, y, i in dl:
        with torch.autocast("cuda", torch.bfloat16):
            lo, lc = m(x.to(DEV, non_blocking=True).to(memory_format=torch.channels_last))
        s, cum = corn_score(lo.float())
        S[i.numpy()] = s.cpu().numpy()
        C[i.numpy()] = cum.cpu().numpy()
        P[i.numpy()] = torch.softmax(lc.float(), 1).cpu().numpy()
    return S, C, P


best = 1e9
for ep in range(EP):
    m.train(); t0 = time.time(); tot = n = 0
    for x, y, _ in tl:
        x = x.to(DEV, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(DEV, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            lo, lc = m(x)
            loss = corn_loss(lo.float(), y) + 0.3 * F.cross_entropy(lc.float(), y,
                                                                    label_smoothing=0.05)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item() * len(y); n += len(y)
    S, C, P = infer("val")
    yv = np.array([r["cls"] for r in by["val"]])
    mae = np.abs(S - yv).mean()
    acc = (np.clip(np.round(S), 0, 3) == yv).mean()
    print(f"ep{ep+1}/{EP} loss={tot/n:.4f} val_MAE_skor={mae:.4f} val_acc_bulat={acc:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if mae < best:
        best = mae
        torch.save(m.state_dict(), f"{OUT}/corn_best.pt")

m.load_state_dict(torch.load(f"{OUT}/corn_best.pt"))
for sp in ["train", "val", "test"]:
    S, C, P = infer(sp)
    np.savez(f"{OUT}/corn_{sp}.npz", score=S, cum=C, prob=P,
             y=np.array([r["cls"] for r in by[sp]]))
print(f"terbaik val MAE skor = {best:.4f}")
print("ORDINAL DONE")
