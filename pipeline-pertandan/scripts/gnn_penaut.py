"""PT-E-016 — Penaut GNN: penilaian pasangan yang melihat seluruh pohon sekaligus.

`IDEA.md` sec.4 butir 3 meminta "Visual Re-ID + GNN penaut" untuk menaikkan cakupan
penyatuan dari 29% ke >70%. Bagian Re-ID sudah ada (PT-E-002b) dan sudah masuk
fitur. Bagian GNN belum pernah dicoba. Ini menjalankannya.

## Kenapa GNN, dan kenapa ini BUKAN sekadar "penilai pasangan yang lebih pintar"

CLAUDE.md sec.6 menyimpulkan obat yang tepat adalah prior yang MEMANGKAS RUANG
KANDIDAT, bukan penilai pasangan yang lebih pintar. Itu benar untuk penilai yang
menilai tiap pasangan SENDIRI-SENDIRI, dan penaut sekarang persis begitu:
`HistGradientBoosting` melihat satu vektor fitur pasangan dan tidak tahu apa-apa
tentang 234 pasangan lain di pohon yang sama.

PT-E-007 mengukur akibatnya: memaksa menggabung lebih banyak justru MENURUNKAN
akurasi, jadi masalahnya bukan kapan berhenti melainkan **urutan skornya**.
Pasangan berskor tertinggi yang belum tergabung mayoritas keliru.

Urutan skor yang salah adalah gejala khas penilaian independen. Kalau kotak `a`
di sisi 1 sangat cocok dengan `b` di sisi 2, itu semestinya MENURUNKAN skor
`a`-dengan-`c` — tapi penilai independen tidak punya jalan untuk tahu. GNN
punya: tiap simpul mengumpulkan seluruh sisi yang menempel padanya lewat
attention, sehingga persaingan antar-kandidat masuk ke dalam skor, bukan cuma
ke dalam Hungarian di belakangnya.

Jadi ini menguji klaim CLAUDE.md sec.6 secara langsung, bukan mengabaikannya:
apakah penalaran BERSAMA (kombinatorik) menolong di tempat penilai independen
gagal.

## Perbandingan yang adil

Baseline dan GNN memakai **fitur pasangan yang sama persis** (`fitur_pasangan`
varian E: geometri + arah putar + penampilan + re-ID + prob prediksi) dan
**perakit klaster yang sama persis** (Hungarian per pasangan-sisi, lalu
union-find serakah dengan batasan sisi-unik dan ukuran maksimum). Yang berbeda
HANYA cara skor sisi dihitung. Kalau tidak begitu, selisihnya bercampur dengan
"punya fitur lebih banyak" atau "perakit berbeda".

Pemakaian:
    .venv/bin/python pipeline-pertandan/scripts/gnn_penaut.py --epoch 40
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import penaut_pertandan as PP           # noqa: E402

SUB = PP.SUB
SEED = 0
AMBANG_SAPU = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


# --------------------------------------------------------------------------
# graf per pohon
# --------------------------------------------------------------------------
def fitur_simpul(b, nv):
    ap = max(b["w"] * b["h"], 1e-9)
    return [b["cx"], b["cy"], b["w"], b["h"], float(np.log(ap)),
            b["w"] / max(b["h"], 1e-9), b["s"] / max(nv, 1), nv / 8.0]


def bangun_graf(tree, desk, emb, prob):
    """Simpul = kotak; sisi = seluruh pasangan LINTAS-SISI (kandidat yang sama
    persis dengan yang dinilai baseline)."""
    nv, kotak = PP.muat_pohon(tree)
    n = len(kotak)
    if n < 2:
        return None
    kunci = [f"{tree}|{b['s']}|{b['i']}" for b in kotak]
    V = []
    for b, k in zip(kotak, kunci):
        f = fitur_simpul(b, nv)
        f += list(prob[k]) if (prob is not None and k in prob) else [0.25] * 4
        V.append(f)
    ei, ej, EF, y = [], [], [], []
    for i, j in itertools.combinations(range(n), 2):
        if kotak[i]["s"] == kotak[j]["s"]:
            continue
        ei.append(i); ej.append(j)
        EF.append(PP.fitur_pasangan(kotak[i], kotak[j], nv, tree, desk, True,
                                    emb, False, prob))
        y.append(int(kotak[i]["bid"] is not None and
                     kotak[i]["bid"] == kotak[j]["bid"]))
    if not ei:
        return None
    return {"tree": tree, "nv": nv, "kotak": kotak,
            "V": np.array(V, np.float32), "ei": np.array(ei), "ej": np.array(ej),
            "E": np.array(EF, np.float32), "y": np.array(y, np.float32)}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def mlp(d_in, d_out, d_h=None):
    d_h = d_h or d_out
    return nn.Sequential(nn.Linear(d_in, d_h), nn.LayerNorm(d_h), nn.GELU(),
                         nn.Linear(d_h, d_out))


class LapisPesan(nn.Module):
    """Satu putaran: perbarui sisi dari kedua ujungnya, lalu perbarui simpul dari
    seluruh sisi yang menempel padanya lewat attention.

    Attention di sisi simpul itulah yang membawa PERSAINGAN antar-kandidat ke
    dalam skor: bobotnya dinormalkan atas seluruh sisi yang menempel, jadi satu
    kandidat kuat menekan sisanya."""

    def __init__(self, H):
        super().__init__()
        self.e = mlp(3 * H, H)
        self.a = nn.Linear(H, 1)
        self.v = mlp(2 * H, H)

    def forward(self, V, E, ei, ej):
        E = E + self.e(torch.cat([E, V[ei], V[ej]], 1))
        skor = self.a(E).squeeze(-1)
        # softmax per simpul, atas sisi yang menempel (dua arah)
        idx = torch.cat([ei, ej]); pesan = torch.cat([E, E], 0)
        s = torch.cat([skor, skor])
        m = torch.full((len(V),), -1e4, device=s.device).index_reduce_(
            0, idx, s, "amax", include_self=True)
        w = torch.exp(s - m[idx])
        Z = torch.zeros(len(V), device=s.device).index_add_(0, idx, w) + 1e-9
        agg = torch.zeros_like(V).index_add_(0, idx, pesan * (w / Z[idx]).unsqueeze(-1))
        V = V + self.v(torch.cat([V, agg], 1))
        return V, E


class PenautGNN(nn.Module):
    def __init__(self, d_v, d_e, H=96, L=3):
        super().__init__()
        self.enc_v = mlp(d_v, H)
        self.enc_e = mlp(d_e, H)
        self.lapis = nn.ModuleList([LapisPesan(H) for _ in range(L)])
        self.baca = nn.Sequential(nn.Linear(3 * H, H), nn.GELU(), nn.Linear(H, 1))

    def forward(self, V, E, ei, ej):
        V = self.enc_v(V); E = self.enc_e(E)
        for lp in self.lapis:
            V, E = lp(V, E, ei, ej)
        return self.baca(torch.cat([E, V[ei], V[ej]], 1)).squeeze(-1)


# --------------------------------------------------------------------------
# perakit klaster — SALINAN PERSIS logika PP.klaster, hanya sumber skor berbeda
# --------------------------------------------------------------------------
def rakit(g, S_flat, ambang):
    """S_flat: skor per sisi, urutannya sama dengan g['ei']/g['ej']."""
    kotak, nv, n = g["kotak"], g["nv"], len(g["kotak"])
    S = {}
    for e, (i, j) in enumerate(zip(g["ei"], g["ej"])):
        S[(int(i), int(j))] = float(S_flat[e])
    per_sisi = defaultdict(list)
    for k, b in enumerate(kotak):
        per_sisi[b["s"]].append(k)
    kandidat = []
    for a, b in itertools.combinations(sorted(per_sisi), 2):
        A, B = per_sisi[a], per_sisi[b]
        if not A or not B:
            continue
        M = np.zeros((len(A), len(B)), float)
        for x, i in enumerate(A):
            for yy, j in enumerate(B):
                M[x, yy] = S.get((i, j), S.get((j, i), 0.0))
        for x, yy in zip(*linear_sum_assignment(-M)):
            if M[x, yy] >= ambang:
                kandidat.append((float(M[x, yy]), A[x], B[yy]))
    kandidat.sort(reverse=True)
    uf = PP.UF(n)
    ukuran = Counter({k: 1 for k in range(n)})
    sisi = {k: {b["s"]} for k, b in enumerate(kotak)}
    maks = 3 if nv == 4 else 6
    for _, i, j in kandidat:
        ri, rj = uf.cari(i), uf.cari(j)
        if ri == rj or (sisi[ri] & sisi[rj]) or ukuran[ri] + ukuran[rj] > maks:
            continue
        uf.gabung(ri, rj)
        rn = uf.cari(ri)
        ukuran[rn] = ukuran[ri] + ukuran[rj]
        sisi[rn] = sisi[ri] | sisi[rj]
    return [uf.cari(k) for k in range(n)]


def nilai(graf, skor_per_graf, ambang):
    """F1 pasangan + ARI + cakupan. Definisi F1/ARI identik PP.nilai_klaster."""
    tp = fp = fn = 0
    aris, selisih = [], []
    n_multi_gt = n_multi_ketemu = 0
    n_kotak_multi = n_kotak_tersatukan = 0
    for g, S in zip(graf, skor_per_graf):
        lab = rakit(g, S, ambang)
        kotak = g["kotak"]
        gt = [b["bid"] if b["bid"] is not None else -1000 - k
              for k, b in enumerate(kotak)]
        for i, j in itertools.combinations(range(len(kotak)), 2):
            if kotak[i]["s"] == kotak[j]["s"]:
                continue
            P, G = lab[i] == lab[j], gt[i] == gt[j]
            tp += P and G; fp += P and not G; fn += G and not P
        aris.append(adjusted_rand_score(gt, lab))
        selisih.append(len(set(lab)) - len(set(gt)))
        # cakupan: tandan GT yang MEMANG multi-sisi, apakah tersatukan (>=2 anggota)
        per_gt = defaultdict(list)
        for k, bid in enumerate(gt):
            per_gt[bid].append(k)
        for bid, anggota in per_gt.items():
            if bid < -999 or len(anggota) < 2:
                continue
            n_multi_gt += 1
            n_kotak_multi += len(anggota)
            grup = Counter(lab[k] for k in anggota)
            terbesar = max(grup.values())
            if terbesar >= 2:
                n_multi_ketemu += 1
            n_kotak_tersatukan += sum(v for v in grup.values() if v >= 2)
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return dict(presisi=round(p, 4), recall=round(r, 4),
                f1=round(2 * p * r / (p + r + 1e-9), 4),
                ari=round(float(np.mean(aris)), 4),
                bias_jumlah=round(float(np.mean(selisih)), 3),
                mae_jumlah=round(float(np.mean(np.abs(selisih))), 3),
                cakupan_tandan=round(n_multi_ketemu / max(n_multi_gt, 1), 4),
                cakupan_kotak=round(n_kotak_tersatukan / max(n_kotak_multi, 1), 4),
                n_tandan_multi_gt=n_multi_gt)


# --------------------------------------------------------------------------
def ke_dev(g, dev):
    return (torch.from_numpy(g["V"]).to(dev), torch.from_numpy(g["E"]).to(dev),
            torch.from_numpy(g["ei"]).long().to(dev),
            torch.from_numpy(g["ej"]).long().to(dev))


@torch.no_grad()
def skor_semua(m, graf, dev):
    m.eval()
    return [torch.sigmoid(m(*ke_dev(g, dev))).cpu().numpy() for g in graf]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lapis", type=int, default=3)
    ap.add_argument("--lebar", type=int, default=96)
    ap.add_argument("--batch-pohon", type=int, default=8)
    ap.add_argument("--keluaran", default=str(SUB / "results" / "pt_e_016_gnn.json"))
    args = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    man = PP.muat_manifest()
    pohon = {s: [t for t, v in man.items() if v == s] for s in ("train", "val", "test")}
    semua = pohon["train"] + pohon["val"] + pohon["test"]

    print("deskriptor penampilan...")
    desk = PP.bangun_deskriptor(semua, SUB / "results" / "deskriptor_crop.npz")

    # embedding re-ID, out-of-fold untuk train (alasan di penaut_pertandan.main)
    emb = None
    f_emb = SUB / "results" / "reid_embedding.npz"
    if f_emb.exists():
        z = np.load(f_emb, allow_pickle=True)
        emb = {k: z[k] for k in z.files}
        oof = 0
        for fo in range(2):
            fz = SUB / "results" / f"reid_embedding_f{fo}.npz"
            if not fz.exists():
                continue
            zz = np.load(fz, allow_pickle=True)
            ditahan = {t for i, t in enumerate(sorted(pohon["train"])) if i % 2 == fo}
            for k in zz.files:
                if k.split("|")[0] in ditahan:
                    emb[k] = zz[k]; oof += 1
        print(f"  re-ID: {len(emb)} potongan, {oof} out-of-fold")
    else:
        print("  re-ID TIDAK ADA -> jalankan reid_pertandan.py dulu")

    PP.HARAP = PP.hitung_harapan_geser(pohon["train"])
    print(f"  arah putar: {len(PP.HARAP)} entri (n_sisi, offset)")
    prob = PP.bangun_prob_prediksi(pohon)

    print("membangun graf per pohon...")
    graf = {s: [g for g in (bangun_graf(t, desk, emb, prob) for t in pohon[s])
                if g is not None] for s in pohon}
    for s in graf:
        ne = sum(len(g["y"]) for g in graf[s]); npos = sum(int(g["y"].sum()) for g in graf[s])
        print(f"  {s}: {len(graf[s])} pohon, {ne} sisi, {npos} positif "
              f"({100*npos/max(ne,1):.1f}%)")

    d_v = graf["train"][0]["V"].shape[1]; d_e = graf["train"][0]["E"].shape[1]
    hasil = {"pt_e": "016", "seed": SEED, "d_simpul": d_v, "d_sisi": d_e,
             "lapis": args.lapis, "lebar": args.lebar, "epoch": args.epoch,
             "lr": args.lr, "riwayat_epoch": []}

    # ---------------- baseline: penilai pasangan independen ----------------
    print("\n=== BASELINE HistGradientBoosting (penilai independen) ===")
    Xtr = np.concatenate([g["E"] for g in graf["train"]])
    ytr = np.concatenate([g["y"] for g in graf["train"]])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         random_state=SEED).fit(Xtr, ytr)
    skor_base = {s: [clf.predict_proba(g["E"])[:, 1] for g in graf[s]] for s in graf}
    for s in ("val", "test"):
        yv = np.concatenate([g["y"] for g in graf[s]])
        pv = np.concatenate(skor_base[s])
        print(f"  {s} AUC per-pasangan: {roc_auc_score(yv, pv):.4f}")

    base_sapu = {}
    for a in AMBANG_SAPU:
        base_sapu[f"{a:.2f}"] = nilai(graf["val"], skor_base["val"], a)
        print(f"  ambang {a:.2f}: F1 {base_sapu[f'{a:.2f}']['f1']} "
              f"ARI {base_sapu[f'{a:.2f}']['ari']} "
              f"cakupan {base_sapu[f'{a:.2f}']['cakupan_tandan']}")
    a_base = max(AMBANG_SAPU, key=lambda a: base_sapu[f"{a:.2f}"]["f1"])
    hasil["baseline"] = {
        "auc_val": round(float(roc_auc_score(
            np.concatenate([g["y"] for g in graf["val"]]),
            np.concatenate(skor_base["val"]))), 4),
        "sapuan_val": base_sapu, "ambang_dikunci_dari_val": a_base,
        "val": base_sapu[f"{a_base:.2f}"],
        "test_sekali": nilai(graf["test"], skor_base["test"], a_base)}
    print(f"  TEST baseline: {hasil['baseline']['test_sekali']}")

    # ---------------- GNN ----------------
    print("\n=== GNN (penalaran bersama) ===")
    m = PenautGNN(d_v, d_e, args.lebar, args.lapis).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epoch)
    prev = ytr.mean()
    pw = torch.tensor(float((1 - prev) / max(prev, 1e-9)), device=dev)
    print(f"  prevalensi positif {prev:.4f} -> pos_weight {float(pw):.1f}")

    tr = list(graf["train"])
    auc_terbaik, sd_terbaik = -1.0, None
    for ep in range(args.epoch):
        m.train(); random.shuffle(tr); tot = n = 0
        opt.zero_grad(set_to_none=True)
        for bi, g in enumerate(tr):
            V, E, ei, ej = ke_dev(g, dev)
            y = torch.from_numpy(g["y"]).to(dev)
            L = F.binary_cross_entropy_with_logits(m(V, E, ei, ej), y, pos_weight=pw)
            (L / args.batch_pohon).backward()
            tot += float(L.detach()); n += 1
            if (bi + 1) % args.batch_pohon == 0 or bi == len(tr) - 1:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step(); opt.zero_grad(set_to_none=True)
        sch.step()
        sv = skor_semua(m, graf["val"], dev)
        auc = float(roc_auc_score(np.concatenate([g["y"] for g in graf["val"]]),
                                  np.concatenate(sv)))
        hasil["riwayat_epoch"].append({"epoch": ep + 1, "loss": round(tot / max(n, 1), 6),
                                       "auc_val": round(auc, 4)})
        if auc > auc_terbaik:
            auc_terbaik = auc
            sd_terbaik = {k: v.detach().clone() for k, v in m.state_dict().items()}
        print(f"  epoch {ep+1}/{args.epoch} loss {tot/max(n,1):.4f} AUC val {auc:.4f}",
              flush=True)

    m.load_state_dict(sd_terbaik)
    runs = SUB / "runs" / "gnn_penaut"
    runs.mkdir(parents=True, exist_ok=True)
    torch.save({"model": m.state_dict(), "d_v": d_v, "d_e": d_e,
                "lebar": args.lebar, "lapis": args.lapis}, runs / "best.pt")

    skor_g = {s: skor_semua(m, graf[s], dev) for s in graf}
    gnn_sapu = {}
    for a in AMBANG_SAPU:
        gnn_sapu[f"{a:.2f}"] = nilai(graf["val"], skor_g["val"], a)
        print(f"  ambang {a:.2f}: F1 {gnn_sapu[f'{a:.2f}']['f1']} "
              f"ARI {gnn_sapu[f'{a:.2f}']['ari']} "
              f"cakupan {gnn_sapu[f'{a:.2f}']['cakupan_tandan']}")
    a_gnn = max(AMBANG_SAPU, key=lambda a: gnn_sapu[f"{a:.2f}"]["f1"])
    hasil["gnn"] = {"auc_val": round(auc_terbaik, 4), "sapuan_val": gnn_sapu,
                    "ambang_dikunci_dari_val": a_gnn,
                    "val": gnn_sapu[f"{a_gnn:.2f}"],
                    "test_sekali": nilai(graf["test"], skor_g["test"], a_gnn)}
    print(f"  TEST GNN: {hasil['gnn']['test_sekali']}")

    # dump skor test supaya tidak perlu inferensi ulang (aturan repo)
    np.savez_compressed(
        SUB / "results" / "pt_e_016_skor_test.npz",
        **{g["tree"]: s for g, s in zip(graf["test"], skor_g["test"])})

    b, gg = hasil["baseline"], hasil["gnn"]
    hasil["putusan"] = {
        "G1_ambang": {"f1": 0.65, "ari": 0.55},
        "baseline_val": {"f1": b["val"]["f1"], "ari": b["val"]["ari"],
                         "cakupan": b["val"]["cakupan_tandan"]},
        "gnn_val": {"f1": gg["val"]["f1"], "ari": gg["val"]["ari"],
                    "cakupan": gg["val"]["cakupan_tandan"]},
        "delta_f1_test_pp": round((gg["test_sekali"]["f1"] - b["test_sekali"]["f1"]) * 100, 2),
        "delta_ari_test_pp": round((gg["test_sekali"]["ari"] - b["test_sekali"]["ari"]) * 100, 2),
        "delta_cakupan_test_pp": round((gg["test_sekali"]["cakupan_tandan"] -
                                        b["test_sekali"]["cakupan_tandan"]) * 100, 2),
        "target_IDEA_cakupan": 0.70,
        "arti": ("delta positif = penalaran bersama menolong di tempat penilai "
                 "independen gagal; target IDEA.md cakupan >70%"),
    }
    hasil["detik"] = round(time.time() - t0, 1)
    Path(args.keluaran).write_text(json.dumps(hasil, indent=1, ensure_ascii=False))
    print("\n" + json.dumps(hasil["putusan"], indent=1, ensure_ascii=False))
    print(f"-> {args.keluaran}  ({hasil['detik']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
