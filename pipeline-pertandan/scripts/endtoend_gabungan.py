"""PT-E-019 — Pipeline UTUH dengan penaut PT-E-017 dan ensemble PT-E-018.

Dua kemenangan sesi 2026-08-18 sejauh ini diukur di dunia masing-masing:

  PT-E-017  penaut di ruang deteksi   F1 0,1492 -> 0,3788
  PT-E-018  ensemble kelas            0,7208 -> 0,7464   (potongan GT, tautan oracle)

Keduanya menyentuh pipeline utuh lewat jalur berbeda: penaut menentukan BERAPA
BANYAK tandan tersentuh agregasi, ensemble menentukan akurasi TIAP tandan yang
tersentuh. Kalau keduanya nyata, efeknya berlipat, bukan bertumpuk.

## Faktorial 2x2, supaya kontribusinya terpisah

           kelas C1              kelas ENSEMBLE
  penaut   (a) reproduksi        (c) isolasi kontribusi classifier
  lama         PT-E-003 0,7124
  penaut   (b) isolasi           (d) gabungan
  baru         kontribusi penaut

(a) harus mendekati 0,7124. Kalau tidak, ada yang salah sebelum kesimpulan apa pun.

## Risiko yang sengaja diuji di sini

Anggota C2 ensemble dilatih di potongan KOTAK GT, sedangkan pipeline utuh memberi
potongan DETEKSI. Itu domain shift sejenis dengan yang baru saja terbukti
menghancurkan penaut (AUC 0,9508 -> 0,5868 di PT-E-017). Kalau ensemble rapuh
terhadap shift itu, +2,56 pp-nya akan menguap end-to-end -- dan itu justru salah
satu hal terpenting yang perlu diketahui sebelum ada yang membangun di atasnya.

Ambang penaut DIKUNCI dari PT-E-017 (lama 0,05; baru 0,90), tidak dicari ulang
di sini: mencarinya terhadap akurasi kelas berarti menyetel penaut pada metrik
hilir, dan itu bocor.

Pemakaian:
    .venv/bin/python pipeline-pertandan/scripts/endtoend_gabungan.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).parent))
import penaut_pertandan as PP           # noqa: E402
import eval_pertandan as EP             # noqa: E402
import reid_pertandan as RD             # noqa: E402
import eval_endtoend as EE              # noqa: E402
import gnn_penaut as GN                 # noqa: E402
import gnn_deteksi as GD                # noqa: E402
import c_backbone_ordinal as CB         # noqa: E402
import c3_multitampak as C3M            # noqa: E402

SUB = PP.SUB
SEED = 0
SKEMA = "conf_luas"
# dikunci dari PT-E-018 (dipilih & dibobot di val, jangan disetel ulang di sini)
ENS = [("C1", 0.6), ("convnext_tiny_coral", 0.2), ("convnext_tiny_ce", 0.2)]


def muat_c2(tag, dev):
    ck = torch.load(SUB / "runs" / f"c_{tag}" / "best.pt", map_location=dev)
    m = CB.C2(ck["backbone"], ck["loss"]).to(dev).eval()
    m.load_state_dict(ck["C2"])
    return m, ck["loss"]


@torch.no_grad()
def prob_c2_crops(m, loss, crops, dev):
    kls = CB.KEPALA[loss]
    out = []
    for i in range(0, len(crops), 256):
        x = C3M.ke_tensor(crops[i:i + 256], False).to(dev)
        with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
            out.append(kls.prob(m(x)).float().cpu().numpy())
    return np.concatenate(out)


def bangun(P, z, conf, reid_model, c2s, dev):
    """Graf deteksi + probabilitas ensemble per deteksi.

    Potongan ditangkap lewat `reid_fn` karena `EE.deteksi_pohon` membuang kunci
    `crop` setelah memakainya. Indeks yang menerima embedding diidentifikasi dari
    norma embedding: keluaran re-ID dinormalkan L2 (norma 1), sedangkan deteksi
    yang citranya gagal dibaca diberi vektor nol.
    """
    tangkap = {}

    def reid_fn(crops):
        tangkap["crops"] = crops
        out = []
        with torch.no_grad():
            for i in range(0, len(crops), 256):
                out.append(reid_model(RD.ke_tensor(crops[i:i + 256], False, dev))
                           .float().cpu().numpy())
        return np.concatenate(out)

    g = GD.graf_deteksi(P, z, conf, reid_fn)
    if g is None:
        return None
    det = g["kotak"]
    idx = [k for k, d in enumerate(det) if np.linalg.norm(d["emb"]) > 0.5]
    crops = tangkap.get("crops")
    for d in det:
        d["p_C1"] = d["p"]
        d["p_ens"] = d["p"].copy()
    if crops is not None and len(idx) == len(crops):
        akum = np.zeros((len(crops), 4), np.float32)
        w_c1 = dict(ENS)["C1"]
        for k, dd in enumerate(idx):
            akum[k] = w_c1 * det[dd]["p_C1"]
        for tag, w in ENS:
            if tag == "C1":
                continue
            m, loss = c2s[tag]
            akum += w * prob_c2_crops(m, loss, crops, dev)
        for k, dd in enumerate(idx):
            s = akum[k].sum()
            det[dd]["p_ens"] = akum[k] / max(s, 1e-9)
    return g


def pools_dari(g, lab, kunci_p):
    """Pola PT-E-003: tiap tandan GT diwakili pool yang memuat anggota
    tercocokkannya terbanyak."""
    det = g["kotak"]
    per_pool = defaultdict(list)
    for k, d in enumerate(det):
        per_pool[lab[k]].append(d)
    milik = defaultdict(Counter)
    for k, d in enumerate(det):
        if d["bid"] is not None:
            milik[d["bid"]][lab[k]] += 1
    keluar = []
    for bid, c in milik.items():
        pid = c.most_common(1)[0][0]
        keluar.append({"tree": g["tree"], "gt": g["P"]["tandan"][bid],
                       "pool": [{"p": d[kunci_p], "conf": d["conf"],
                                 "luas": d["luas"], "tepi": d["tepi"]}
                                for d in per_pool[pid]]})
    return keluar


def evaluasi(graf, skor, ambang, kunci_p, tau=None):
    pools = []
    for g, S in zip(graf, skor):
        pools += pools_dari(g, GN.rakit(g, S, ambang), kunci_p)
    if tau is None:
        tau = EP.cari_tau(pools, SKEMA)
    multi = [q for q in pools if len(q["pool"]) >= 2]
    return {"R0": EP.nilai(pools, "R0", SKEMA, tau),
            "R4": EP.nilai(pools, "R4", SKEMA, tau),
            "R4_multi": EP.nilai(multi, "R4", SKEMA, tau),
            "n_tandan": len(pools), "n_multi": len(multi)}, tau, pools


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keluaran", default=str(SUB / "results" / "pt_e_019_gabungan.json"))
    args = ap.parse_args()
    np.random.seed(SEED); torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    o = json.loads((SUB / "results" / "pt_e_001_oracle.json").read_text())
    conf = o["conf_dikunci"]
    p17 = json.loads((SUB / "results" / "pt_e_017_gnn_deteksi.json").read_text())
    A_LAMA = p17["lengan"]["A_latih_kotakGT"]["ambang_dikunci_dari_val"]
    A_BARU = p17["lengan"]["C_gnn_deteksi"]["ambang_dikunci_dari_val"]
    print(f"conf {conf} | ambang penaut dikunci dari PT-E-017: lama {A_LAMA} baru {A_BARU}")

    man = PP.muat_manifest()
    ids = {s: [t for t, v in man.items() if v == s] for s in ("train", "val", "test")}
    desk = PP.bangun_deskriptor(ids["train"] + ids["val"] + ids["test"],
                                SUB / "results" / "deskriptor_crop.npz")
    ze = np.load(SUB / "results" / "reid_embedding.npz", allow_pickle=True)
    emb = {k: ze[k] for k in ze.files}
    PP.HARAP = PP.hitung_harapan_geser(ids["train"])
    prob_gt = PP.bangun_prob_prediksi(ids)

    reid_model = RD.Reid().to(dev).eval()
    reid_model.load_state_dict(torch.load(SUB / "runs" / "reid_resnet18" / "best.pt",
                                          map_location=dev))
    c2s = {tag: muat_c2(tag, dev) for tag, _ in ENS if tag != "C1"}
    print(f"anggota ensemble dimuat: {list(c2s)} + C1")

    z = {s: np.load(SUB / "results" / f"pred_skorpenuh_{s}.npz", allow_pickle=True)
         for s in ("val", "test")}
    print("membangun graf deteksi + probabilitas ensemble...")
    graf = {}
    for s in ("val", "test"):
        gs = []
        for n, t in enumerate(ids[s], 1):
            g = bangun(EP.muat_pohon(t), z[s], conf, reid_model, c2s, dev)
            if g is not None:
                gs.append(g)
            if n % 50 == 0:
                print(f"  {s}: {n}/{len(ids[s])}", flush=True)
        graf[s] = gs
        print(f"  {s}: {len(gs)} pohon", flush=True)

    # ---- dua penaut ----
    print("\npenaut LAMA (HistGB dilatih di pasangan kotak GT)...")
    Xg, yg = PP.pasangan(ids["train"], desk, True, emb, False, prob_gt)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         random_state=SEED).fit(Xg, yg)
    skor = {"lama": {s: [clf.predict_proba(g["E"])[:, 1] for g in graf[s]]
                     for s in graf}}
    print("penaut BARU (GNN dilatih di pasangan deteksi, PT-E-017 lengan C)...")
    ck = torch.load(SUB / "runs" / "gnn_deteksi" / "best.pt", map_location=dev)
    m = GN.PenautGNN(ck["d_v"], ck["d_e"], ck["lebar"], ck["lapis"]).to(dev)
    m.load_state_dict(ck["model"])
    skor["baru"] = {s: GN.skor_semua(m, graf[s], dev) for s in graf}

    hasil = {"pt_e": "019", "conf": conf, "seed": SEED,
             "ambang_penaut": {"lama": A_LAMA, "baru": A_BARU},
             "ensemble": [list(x) for x in ENS],
             "acuan": {"pt_e_003_pipeline_utuh": 0.7124,
                       "pipeline_lama_per_citra": 0.7203,
                       "plafon_oracle_C1": 0.7360},
             "sel": {}}

    print(f"\n{'sel':28}{'val R4':>9}{'test R4':>9}{'test multi':>12}{'n multi':>9}")
    tau_sel, pools_test = {}, {}
    for pn, amb in (("lama", A_LAMA), ("baru", A_BARU)):
        for pk, kunci in (("C1", "p_C1"), ("ensemble", "p_ens")):
            nama = f"penaut_{pn}__kelas_{pk}"
            mv, tau, _ = evaluasi(graf["val"], skor[pn]["val"], amb, kunci)
            mt, _, pt = evaluasi(graf["test"], skor[pn]["test"], amb, kunci, tau)
            tau_sel[nama] = tau; pools_test[nama] = pt
            hasil["sel"][nama] = {"tau": tau, "val": mv, "test": mt}
            print(f"{nama:28}{mv['R4']['akurasi']:>9}{mt['R4']['akurasi']:>9}"
                  f"{mt['R4_multi']['akurasi']:>12}{mt['n_multi']:>9}")

    # ---- CI tingkat pohon: gabungan vs reproduksi PT-E-003 ----
    dasar = "penaut_lama__kelas_C1"; puncak = "penaut_baru__kelas_ensemble"
    bd = pools_test[dasar]; bp = pools_test[puncak]
    cd = np.array([EP.benar(q["pool"], q["gt"], "R4", SKEMA, tau_sel[dasar]) for q in bd])
    cp = np.array([EP.benar(q["pool"], q["gt"], "R4", SKEMA, tau_sel[puncak]) for q in bp])
    td = np.array([q["tree"] for q in bd]); tp_ = np.array([q["tree"] for q in bp])
    uniq = sorted(set(td.tolist()) | set(tp_.tolist()))
    id_d = {t: np.where(td == t)[0] for t in uniq}
    id_p = {t: np.where(tp_ == t)[0] for t in uniq}
    rng = np.random.default_rng(SEED); d = []
    for _ in range(2000):
        pil = rng.choice(len(uniq), len(uniq))
        a = np.concatenate([cd[id_d[uniq[k]]] for k in pil if len(id_d[uniq[k]])])
        b = np.concatenate([cp[id_p[uniq[k]]] for k in pil if len(id_p[uniq[k]])])
        d.append(b.mean() - a.mean())
    d = np.array(d) * 100
    hasil["putusan"] = {
        "dasar": dasar, "puncak": puncak,
        "delta_pp": round(float((cp.mean() - cd.mean()) * 100), 2),
        "ci95": [round(float(np.percentile(d, 2.5)), 2),
                 round(float(np.percentile(d, 97.5)), 2)],
        "P(delta>0)": round(float((d > 0).mean()), 3), "n_pohon": len(uniq),
        "kontribusi_penaut_pp": round(
            (hasil["sel"]["penaut_baru__kelas_C1"]["test"]["R4"]["akurasi"] -
             hasil["sel"][dasar]["test"]["R4"]["akurasi"]) * 100, 2),
        "kontribusi_kelas_pp": round(
            (hasil["sel"]["penaut_lama__kelas_ensemble"]["test"]["R4"]["akurasi"] -
             hasil["sel"][dasar]["test"]["R4"]["akurasi"]) * 100, 2),
    }
    hasil["detik"] = round(time.time() - t0, 1)
    Path(args.keluaran).write_text(json.dumps(hasil, indent=1, ensure_ascii=False))
    print("\n" + json.dumps(hasil["putusan"], indent=1, ensure_ascii=False))
    print(f"-> {args.keluaran}  ({hasil['detik']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
