"""PT-E-017 — Penaut di RUANG DETEKSI: domain shift dulu, baru GNN.

## Yang memicu ini

`IDEA.md` sec.4 butir 3 menargetkan cakupan penaut 29% -> >70%. PT-E-016
menjalankan GNN di ruang KOTAK GT dan di sana baseline sudah 66% cakupan, jadi
targetnya salah alamat. Angka 29% hidup di ruang DETEKSI (PT-E-003: F1 0,1766,
recall pasangan 0,1200, 39,9% pool seluruhnya positif palsu).

## Hipotesis yang belum pernah diuji: penautnya dilatih di dunia yang salah

Sejak PT-E-002 sampai PT-E-010, penaut SELALU dilatih di pasangan kotak GT
(`eval_endtoend.py`: "melatih ulang penaut di pasangan kotak GT split train")
lalu dipakai di atas deteksi. Kotak GT bersih: tepat satu per tandan nyata, nol
positif palsu, kotak pas. Deteksi tidak: ada positif palsu, kotak bergeser,
skor berbeda distribusinya. Penaut yang tak pernah melihat positif palsu saat
latihan tidak punya cara belajar menolaknya -- ia hanya bisa menilai "seberapa
mirip dua kotak bagus", bukan "apakah salah satunya sampah".

Diagnosis yang berlaku (CLAUDE.md sec.6) menyebut hambatannya kombinatorik.
Itu bisa benar dan tetap tidak lengkap: kombinatorik menjelaskan kenapa tugasnya
sulit, domain shift menjelaskan kenapa penautnya tidak dilatih untuk kesulitan
itu. Keduanya bisa berlaku bersamaan, dan hanya satu yang murah diperbaiki.

## Tiga lengan, memisahkan dua efek

  A  HistGB dilatih di pasangan KOTAK GT   -> dipakai di deteksi   (repo saat ini)
  B  HistGB dilatih di pasangan DETEKSI    -> dipakai di deteksi   (isolasi domain shift)
  C  GNN    dilatih di pasangan DETEKSI    -> dipakai di deteksi   (isolasi penalaran bersama)

B-A = nilai melatih di domain yang benar. C-B = nilai penalaran bersama di atas
itu. Ketiganya memakai fitur, ambang-conf, dan perakit klaster yang sama persis.

## Penyebut dilaporkan DUA kali, sengaja

CLAUDE.md sec.8 mencatat empat klaim nyaris-palsu dari kesalahan penyebut. Cakupan
di ruang deteksi punya dua penyebut yang sah dan sangat berbeda:

  cakupan_atas_terdeteksi  tandan GT multi-sisi yang punya >=2 DETEKSI terpetakan
                           -- yang secara fisik BISA disatukan penaut
  cakupan_atas_semua       seluruh tandan GT multi-sisi, termasuk yang detektornya
                           lewatkan -- angka end-to-end yang jujur

Angka 29% di STATUS.md bertipe pertama. Menaruhnya bersebelahan dengan tipe kedua
tanpa label adalah persis jebakan yang dimaksud.

Pemakaian:
    .venv/bin/python pipeline-pertandan/scripts/gnn_deteksi.py --epoch 40
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
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import penaut_pertandan as PP           # noqa: E402
import eval_pertandan as EP             # noqa: E402
import reid_pertandan as RD             # noqa: E402
import eval_endtoend as EE              # noqa: E402
import gnn_penaut as GN                 # noqa: E402

SUB = PP.SUB
SEED = 0
GRID = [round(x, 3) for x in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                              0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
                              0.85, 0.90, 0.95]]


def buat_reid(dev):
    m = RD.Reid().to(dev).eval()
    m.load_state_dict(torch.load(SUB / "runs" / "reid_resnet18" / "best.pt",
                                 map_location=dev))

    def fn(crops):
        out = []
        with torch.no_grad():
            for i in range(0, len(crops), 256):
                out.append(m(RD.ke_tensor(crops[i:i + 256], False, dev))
                           .float().cpu().numpy())
        return np.concatenate(out)
    return fn


def graf_deteksi(P, z, conf, reid_fn):
    """Graf ruang deteksi. Simpul = deteksi (termasuk positif palsu)."""
    det = EE.deteksi_pohon(P, z, conf, reid_fn)
    n = len(det)
    if n < 2:
        return None
    nv = P["n_sisi"]
    V, ei, ej, EF, y = [], [], [], [], []
    for d in det:
        ap = max(d["w"] * d["h"], 1e-9)
        V.append([d["cx"], d["cy"], d["w"], d["h"], float(np.log(ap)),
                  d["w"] / max(d["h"], 1e-9), d["s"] / max(nv, 1), nv / 8.0]
                 + list(d["p"]) + [d["conf"], d["tepi"]])
    for i, j in itertools.combinations(range(n), 2):
        if det[i]["s"] == det[j]["s"]:
            continue
        ei.append(i); ej.append(j)
        EF.append(EE.fitur_det(det[i], det[j], nv, True, False, True))
        y.append(int(det[i]["bid"] is not None and det[i]["bid"] == det[j]["bid"]))
    if not ei:
        return None
    return {"tree": P["tree"], "nv": nv, "kotak": det,
            "V": np.array(V, np.float32), "ei": np.array(ei), "ej": np.array(ej),
            "E": np.array(EF, np.float32), "y": np.array(y, np.float32),
            "P": P}


def nilai_det(graf, skor, ambang):
    """F1 pasangan + ARI + DUA cakupan dengan penyebut berbeda (lihat docstring)."""
    tp = fp = fn = 0
    aris = []
    n_bisa = n_bisa_ok = 0          # penyebut: tandan yang punya >=2 deteksi
    n_semua = n_semua_ok = 0        # penyebut: seluruh tandan GT multi-sisi
    n_pool_palsu = n_pool = 0
    for g, S in zip(graf, skor):
        lab = GN.rakit(g, S, ambang)
        det = g["kotak"]
        gt = [d["bid"] if d["bid"] is not None else -1000 - k
              for k, d in enumerate(det)]
        for i, j in itertools.combinations(range(len(det)), 2):
            if det[i]["s"] == det[j]["s"]:
                continue
            Pp = lab[i] == lab[j]
            G = (det[i]["bid"] is not None and det[i]["bid"] == det[j]["bid"])
            tp += Pp and G; fp += Pp and not G; fn += G and not Pp
        aris.append(adjusted_rand_score(gt, lab))
        # pool yang seluruhnya positif palsu
        per_lab = defaultdict(list)
        for k, l in enumerate(lab):
            per_lab[l].append(k)
        for l, anggota in per_lab.items():
            if len(anggota) < 2:
                continue
            n_pool += 1
            if all(det[k]["bid"] is None for k in anggota):
                n_pool_palsu += 1
        # cakupan, penyebut 1: tandan GT yang punya >=2 deteksi terpetakan
        per_bid = defaultdict(list)
        for k, d in enumerate(det):
            if d["bid"] is not None:
                per_bid[d["bid"]].append(k)
        for bid, anggota in per_bid.items():
            if len(anggota) < 2:
                continue
            n_bisa += 1
            if max(Counter(lab[k] for k in anggota).values()) >= 2:
                n_bisa_ok += 1
        # cakupan, penyebut 2: seluruh tandan GT multi-sisi di pohon ini
        multi_gt = {b: c for b, c in Counter(
            ap_bid for s in g["P"]["sisi"] for ap_bid in
            [gg["bid"] for gg in s["gt"] if gg["bid"] is not None]).items() if c >= 2}
        n_semua += len(multi_gt)
        for bid in multi_gt:
            anggota = per_bid.get(bid, [])
            if len(anggota) >= 2 and max(Counter(lab[k] for k in anggota).values()) >= 2:
                n_semua_ok += 1
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return dict(presisi=round(p, 4), recall=round(r, 4),
                f1=round(2 * p * r / (p + r + 1e-9), 4),
                ari=round(float(np.mean(aris)), 4),
                cakupan_atas_terdeteksi=round(n_bisa_ok / max(n_bisa, 1), 4),
                cakupan_atas_semua=round(n_semua_ok / max(n_semua, 1), 4),
                frac_pool_palsu=round(n_pool_palsu / max(n_pool, 1), 4),
                n_bisa=n_bisa, n_semua=n_semua)


def sapu(graf, skor, tag):
    sap = {}
    for a in GRID:
        sap[f"{a:.3f}"] = nilai_det(graf, skor, a)
        r = sap[f"{a:.3f}"]
        print(f"  [{tag}] ambang {a:.3f}: F1 {r['f1']:.4f} ARI {r['ari']:.4f} "
              f"cakupan(terdeteksi) {r['cakupan_atas_terdeteksi']:.4f} "
              f"palsu {r['frac_pool_palsu']:.3f}", flush=True)
    best = max(GRID, key=lambda a: sap[f"{a:.3f}"]["f1"])
    return sap, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--keluaran", default=str(SUB / "results" / "pt_e_017_gnn_deteksi.json"))
    args = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    o = json.loads((SUB / "results" / "pt_e_001_oracle.json").read_text())
    conf = o["conf_dikunci"]
    print(f"conf deteksi dikunci dari PT-E-001: {conf}")

    man = PP.muat_manifest()
    ids = {s: [t for t, v in man.items() if v == s] for s in ("train", "val", "test")}
    semua = ids["train"] + ids["val"] + ids["test"]
    desk = PP.bangun_deskriptor(semua, SUB / "results" / "deskriptor_crop.npz")
    zemb = np.load(SUB / "results" / "reid_embedding.npz", allow_pickle=True)
    emb = {k: zemb[k] for k in zemb.files}
    PP.HARAP = PP.hitung_harapan_geser(ids["train"])
    prob_gt = PP.bangun_prob_prediksi(ids)
    reid_fn = buat_reid(dev)

    z = {s: np.load(SUB / "results" / f"pred_skorpenuh_{s}.npz", allow_pickle=True)
         for s in ("train", "val", "test")}

    print("membangun graf RUANG DETEKSI (baca citra, jadi lambat)...")
    graf = {}
    for s in ("train", "val", "test"):
        gs = []
        for n, t in enumerate(ids[s], 1):
            g = graf_deteksi(EP.muat_pohon(t), z[s], conf, reid_fn)
            if g is not None:
                gs.append(g)
            if n % 200 == 0:
                print(f"  {s}: {n}/{len(ids[s])}", flush=True)
        graf[s] = gs
        ne = sum(len(g["y"]) for g in gs); npos = sum(int(g["y"].sum()) for g in gs)
        print(f"  {s}: {len(gs)} pohon, {ne} sisi, {npos} positif "
              f"({100*npos/max(ne,1):.1f}%)", flush=True)

    hasil = {"pt_e": "017", "conf": conf, "seed": SEED, "grid": GRID, "lengan": {}}

    # ---------- lengan A: HistGB dilatih di KOTAK GT ----------
    print("\n=== A: HistGB dilatih di pasangan KOTAK GT (cara repo sekarang) ===")
    Xg, yg = PP.pasangan(ids["train"], desk, True, emb, False, prob_gt)
    clfA = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                          random_state=SEED).fit(Xg, yg)
    sA = {s: [clfA.predict_proba(g["E"])[:, 1] for g in graf[s]] for s in ("val", "test")}
    sapA, aA = sapu(graf["val"], sA["val"], "A")
    hasil["lengan"]["A_latih_kotakGT"] = {
        "n_pasangan_latih": int(len(yg)), "sapuan_val": sapA,
        "ambang_dikunci_dari_val": aA, "val": sapA[f"{aA:.3f}"],
        "test_sekali": nilai_det(graf["test"], sA["test"], aA)}
    print(f"  TEST A: {hasil['lengan']['A_latih_kotakGT']['test_sekali']}")

    # ---------- lengan B: HistGB dilatih di DETEKSI ----------
    print("\n=== B: HistGB dilatih di pasangan DETEKSI (domain yang benar) ===")
    Xd = np.concatenate([g["E"] for g in graf["train"]])
    yd = np.concatenate([g["y"] for g in graf["train"]])
    print(f"  pasangan latih: {len(yd)} ({int(yd.sum())} positif, "
          f"{100*yd.mean():.1f}%)")
    clfB = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                          random_state=SEED).fit(Xd, yd)
    sB = {s: [clfB.predict_proba(g["E"])[:, 1] for g in graf[s]] for s in ("val", "test")}
    aucB = roc_auc_score(np.concatenate([g["y"] for g in graf["val"]]),
                         np.concatenate(sB["val"]))
    aucA = roc_auc_score(np.concatenate([g["y"] for g in graf["val"]]),
                         np.concatenate(sA["val"]))
    print(f"  AUC val pasangan: A {aucA:.4f} -> B {aucB:.4f}")
    sapB, aB = sapu(graf["val"], sB["val"], "B")
    hasil["lengan"]["B_latih_deteksi"] = {
        "n_pasangan_latih": int(len(yd)), "auc_val": round(float(aucB), 4),
        "sapuan_val": sapB, "ambang_dikunci_dari_val": aB, "val": sapB[f"{aB:.3f}"],
        "test_sekali": nilai_det(graf["test"], sB["test"], aB)}
    hasil["lengan"]["A_latih_kotakGT"]["auc_val"] = round(float(aucA), 4)
    print(f"  TEST B: {hasil['lengan']['B_latih_deteksi']['test_sekali']}")

    # ---------- lengan C: GNN dilatih di DETEKSI ----------
    print("\n=== C: GNN dilatih di pasangan DETEKSI ===")
    d_v = graf["train"][0]["V"].shape[1]; d_e = graf["train"][0]["E"].shape[1]
    m = GN.PenautGNN(d_v, d_e, 96, 3).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epoch)
    prev = float(yd.mean())
    pw = torch.tensor((1 - prev) / max(prev, 1e-9), device=dev)
    print(f"  prevalensi {prev:.4f} -> pos_weight {float(pw):.1f}")
    tr = list(graf["train"])
    auc_best, sd_best = -1.0, None
    riwayat = []
    for ep in range(args.epoch):
        m.train(); random.shuffle(tr); tot = nb = 0
        opt.zero_grad(set_to_none=True)
        for bi, g in enumerate(tr):
            V, E, ei, ej = GN.ke_dev(g, dev)
            yy = torch.from_numpy(g["y"]).to(dev)
            L = F.binary_cross_entropy_with_logits(m(V, E, ei, ej), yy, pos_weight=pw)
            (L / 8).backward(); tot += float(L.detach()); nb += 1
            if (bi + 1) % 8 == 0 or bi == len(tr) - 1:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step(); opt.zero_grad(set_to_none=True)
        sch.step()
        sv = GN.skor_semua(m, graf["val"], dev)
        auc = float(roc_auc_score(np.concatenate([g["y"] for g in graf["val"]]),
                                  np.concatenate(sv)))
        riwayat.append({"epoch": ep + 1, "loss": round(tot / max(nb, 1), 6),
                        "auc_val": round(auc, 4)})
        if auc > auc_best:
            auc_best = auc
            sd_best = {k: v.detach().clone() for k, v in m.state_dict().items()}
        print(f"  epoch {ep+1}/{args.epoch} loss {tot/max(nb,1):.4f} "
              f"AUC val {auc:.4f}", flush=True)
    m.load_state_dict(sd_best)
    runs = SUB / "runs" / "gnn_deteksi"; runs.mkdir(parents=True, exist_ok=True)
    torch.save({"model": m.state_dict(), "d_v": d_v, "d_e": d_e,
                "lebar": 96, "lapis": 3, "conf": conf}, runs / "best.pt")
    sC = {s: GN.skor_semua(m, graf[s], dev) for s in ("val", "test")}
    sapC, aC = sapu(graf["val"], sC["val"], "C")
    hasil["lengan"]["C_gnn_deteksi"] = {
        "auc_val": round(auc_best, 4), "riwayat_epoch": riwayat,
        "sapuan_val": sapC, "ambang_dikunci_dari_val": aC, "val": sapC[f"{aC:.3f}"],
        "test_sekali": nilai_det(graf["test"], sC["test"], aC)}
    print(f"  TEST C: {hasil['lengan']['C_gnn_deteksi']['test_sekali']}")

    np.savez_compressed(SUB / "results" / "pt_e_017_skor_test.npz",
                        **{f"{arm}__{g['tree']}": s
                           for arm, S in (("A", sA), ("B", sB), ("C", sC))
                           for g, s in zip(graf["test"], S["test"])})

    A = hasil["lengan"]["A_latih_kotakGT"]["test_sekali"]
    B = hasil["lengan"]["B_latih_deteksi"]["test_sekali"]
    C = hasil["lengan"]["C_gnn_deteksi"]["test_sekali"]
    hasil["putusan"] = {
        "acuan_pt_e_003": {"f1": 0.1766, "recall": 0.1200, "cakupan": 0.29},
        "A_f1": A["f1"], "B_f1": B["f1"], "C_f1": C["f1"],
        "domain_shift_B_minus_A_f1_pp": round((B["f1"] - A["f1"]) * 100, 2),
        "gnn_C_minus_B_f1_pp": round((C["f1"] - B["f1"]) * 100, 2),
        "cakupan_terdeteksi": {"A": A["cakupan_atas_terdeteksi"],
                               "B": B["cakupan_atas_terdeteksi"],
                               "C": C["cakupan_atas_terdeteksi"]},
        "cakupan_semua": {"A": A["cakupan_atas_semua"], "B": B["cakupan_atas_semua"],
                          "C": C["cakupan_atas_semua"]},
        "target_IDEA_cakupan": 0.70,
        "arti": ("B-A memisahkan nilai melatih di domain yang benar; C-B memisahkan "
                 "nilai penalaran bersama di atasnya"),
    }
    hasil["detik"] = round(time.time() - t0, 1)
    Path(args.keluaran).write_text(json.dumps(hasil, indent=1, ensure_ascii=False))
    print("\n" + json.dumps(hasil["putusan"], indent=1, ensure_ascii=False))
    print(f"-> {args.keluaran}  ({hasil['detik']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
