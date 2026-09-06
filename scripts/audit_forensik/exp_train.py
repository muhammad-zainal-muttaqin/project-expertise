"""Appearance vs appearance+structure for bunch maturity, 953 corpus, tree-level split."""
import json, os, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, models
from PIL import Image

OUT = "/workspace/crops953"
DEV = "cuda"
torch.manual_seed(42); np.random.seed(42)

idx = json.load(open(f"{OUT}/index.json"))
by = {s: [r for r in idx if r["split"] == s] for s in ["train", "val", "test"]}
print({k: len(v) for k, v in by.items()})

norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tr_tf = T.Compose([T.RandomResizedCrop(144, scale=(0.75, 1.0)), T.RandomHorizontalFlip(),
                   T.ColorJitter(0.25, 0.25, 0.15, 0.03), T.ToTensor(), norm])
ev_tf = T.Compose([T.CenterCrop(144), T.ToTensor(), norm])


class DS(Dataset):
    def __init__(self, recs, tf, sp):
        self.r, self.tf, self.sp = recs, tf, sp
    def __len__(self):
        return len(self.r)
    def __getitem__(self, i):
        r = self.r[i]
        im = Image.open(f"{OUT}/{self.sp}/{r['f']}").convert("RGB")
        return self.tf(im), r["cls"], i


def loader(sp, tf, bs, sh):
    return DataLoader(DS(by[sp], tf, sp), batch_size=bs, shuffle=sh, num_workers=12,
                      pin_memory=True, drop_last=False, persistent_workers=True)


m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
m.classifier[2] = nn.Linear(768, 4)
m = m.to(DEV).to(memory_format=torch.channels_last)

cnt = np.bincount([r["cls"] for r in by["train"]], minlength=4)
w = torch.tensor((cnt.sum() / (4 * cnt)) ** 0.5, dtype=torch.float32, device=DEV)
crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
EP = 10
opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.05)
tl = loader("train", tr_tf, 128, True)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, epochs=EP, steps_per_epoch=len(tl))
scaler = torch.amp.GradScaler()


@torch.no_grad()
def infer(sp):
    m.eval()
    dl = loader(sp, ev_tf, 256, False)
    P = np.zeros((len(by[sp]), 4), np.float32)
    for x, y, i in dl:
        with torch.autocast("cuda", torch.bfloat16):
            o = m(x.to(DEV, non_blocking=True).to(memory_format=torch.channels_last))
        P[i.numpy()] = torch.softmax(o.float(), 1).cpu().numpy()
    return P


for ep in range(EP):
    m.train(); t0 = time.time(); tot = n = 0
    for x, y, _ in tl:
        x = x.to(DEV, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(DEV, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            loss = crit(m(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item() * len(y); n += len(y)
    P = infer("val")
    acc = (P.argmax(1) == np.array([r["cls"] for r in by["val"]])).mean()
    print(f"ep{ep+1}/{EP} loss={tot/n:.4f} val_crop_acc={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)

torch.save(m.state_dict(), f"{OUT}/convnext.pt")
np.save(f"{OUT}/P_val.npy", infer("val"))
np.save(f"{OUT}/P_test.npy", infer("test"))
np.save(f"{OUT}/P_train.npy", infer("train"))
print("saved probabilities")
