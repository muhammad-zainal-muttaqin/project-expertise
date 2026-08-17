"""PT-E-009 — Sapuan ulang ambang keyakinan DETEKSI (`conf`).

Kenapa. `conf = 0,10` dikunci di PT-E-001 ketika penggabungan masih lemah:
saat itu memperbanyak deteksi memang menguntungkan, karena makin banyak tampak
per tandan makin besar peluang di-pool. Setelah PT-E-008 memperbaiki penaut
(F1 kotak GT 0,365 -> 0,649), pilihan itu berbalik merugikan — di ruang deteksi
**40% kelompok seluruhnya berisi positif palsu**, dan sampah itu merusak
klasifikasi maupun cacah. Ambangnya belum pernah disapu ulang sejak penaut
diperbaiki.

## Cara supaya murah

Deteksi di ambang tinggi adalah HIMPUNAN BAGIAN dari deteksi di ambang rendah.
Jadi bagian mahalnya — memuat citra, deskriptor penampilan, embedding re-ID, dan
skor seluruh pasangan lintas-sisi — dihitung **sekali** pada `conf_min`, lalu
tiap titik sapuan cukup menyaring. Tanpa ini, 5 conf x 3 ambang penaut butuh
berjam-jam; dengan ini, satu lintasan mahal lalu sapuan hampir gratis.

## Protokol

Seluruh pemilihan (conf, ambang penaut, ambang ordinal tau) di **val**; test
dievaluasi **sekali** dengan konfigurasi terkunci. `tau` disetel ulang untuk
tiap conf, karena distribusi skor berubah saat ambang deteksi berubah.

Counting ikut dihitung di tiap titik: penghitung pembanding (k-per-kelas dan
Ridge+F_all) **dipas ulang di train pada conf yang sama**, supaya perbandingannya
adil dan bukan sekadar memberi keuntungan sepihak ke jalur pool.

Pemakaian:
    .venv/bin/python pipeline-pertandan/scripts/sapu_conf.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeCV
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import penaut_pertandan as PP           # noqa: E402
import eval_pertandan as EP             # noqa: E402
import eval_endtoend as EE              # noqa: E402
import eval_counting as EC              # noqa: E402
import reid_pertandan as RD             # noqa: E402

SUB = PP.SUB
KELAS = PP.KELAS


def siapkan_pohon(P, z, conf_min, reid_fn, clf, cfg):
    """Satu lintasan mahal: deteksi + deskriptor + embedding + SEMUA skor pasangan."""
    det = EE.deteksi_pohon(P, z, conf_min, reid_fn)
    if not det:
        return None
    nv = P["n_sisi"]
    per_sisi = defaultdict(list)
    for k, d in enumerate(det):
        per_sisi[d["s"]].append(k)
    pasang = []
    for a, b in itertools.combinations(sorted(per_sisi), 2):
        A, B = per_sisi[a], per_sisi[b]
        if not A or not B:
            continue
        F = [EE.fitur_det(det[i], det[j], nv, cfg["pakai_reid"], cfg["pakai_kelas"],
                          cfg["pakai_prob"]) for i in A for j in B]
        S = clf.predict_proba(np.array(F, float))[:, 1].reshape(len(A), len(B))
        for x, i in enumerate(A):
            for y, j in enumerate(B):
                pasang.append((float(S[x, y]), i, j))
    pasang.sort(reverse=True)
    return {"tree": P["tree"], "nv": nv, "det": det, "pasang": pasang,
            "tandan": P["tandan"]}


def klaster_tersaring(pak, conf, ambang):
    """Klaster hanya atas deteksi yang lolos `conf`. Indeks asli dipertahankan."""
    det = pak["det"]
    hidup = [k for k, d in enumerate(det) if d["conf"] >= conf]
    if not hidup:
        return {}, []
    ada = set(hidup)
    uf = PP.UF(len(det))
    ukuran = Counter({k: 1 for k in hidup})
    sisi = {k: {det[k]["s"]} for k in hidup}
    maks = 3 if pak["nv"] == 4 else 6
    for s, i, j in pak["pasang"]:
        if s < ambang:
            break
        if i not in ada or j not in ada:
            continue
        ri, rj = uf.cari(i), uf.cari(j)
        if ri == rj or (sisi[ri] & sisi[rj]) or ukuran[ri] + ukuran[rj] > maks:
            continue
        uf.gabung(ri, rj)
        rn = uf.cari(ri)
        ukuran[rn] = ukuran[ri] + ukuran[rj]
        sisi[rn] = sisi[ri] | sisi[rj]
    return {k: uf.cari(k) for k in hidup}, hidup


def nilai_titik(paket, conf, ambang, skema, tau, hitung_counting=True):
    """JEBAKAN yang sudah terjadi sekali: menilai akurasi hanya pada tandan yang
    TERDETEKSI membuat angka antar `conf` tidak sebanding. Menaikkan conf
    membuang tandan yang sulit lebih dulu, jadi penyebutnya menyusut dan
    akurasinya naik semu — terukur: conf 0,10 dinilai atas 890 tandan (R4 0,709),
    conf 0,60 atas 243 tandan (R4 0,811). Itu bukan pipeline yang membaik,
    melainkan soal ujian yang dipermudah.

    Karena itu metrik pemilih adalah `R4_semua`, dihitung atas SELURUH tandan GT:
    tandan yang tidak terdeteksi sama sekali dihitung SALAH. Dengan begitu
    menaikkan conf ikut terasa ruginya."""
    pools, pools_oracle = [], []
    tp = fp = fn = 0
    aris = []
    cacah_pool, cacah_benar = [], []
    for pak in paket:
        lab, hidup = klaster_tersaring(pak, conf, ambang)
        det = pak["det"]
        benar = np.zeros(4)
        for kls in pak["tandan"].values():
            benar[kls] += 1
        cacah_benar.append(benar)
        if not hidup:
            cacah_pool.append(np.zeros(4))
            continue
        cocok = [k for k in hidup if det[k]["bid"] is not None]
        for i, j in itertools.combinations(cocok, 2):
            if det[i]["s"] == det[j]["s"]:
                continue
            Pr, G = lab[i] == lab[j], det[i]["bid"] == det[j]["bid"]
            tp += Pr and G; fp += Pr and not G; fn += G and not Pr
        if cocok:
            aris.append(adjusted_rand_score([det[k]["bid"] for k in cocok],
                                            [lab[k] for k in cocok]))
        per_pool = defaultdict(list)
        for k in hidup:
            per_pool[lab[k]].append(det[k])
        milik = defaultdict(Counter)
        for k in hidup:
            if det[k]["bid"] is not None:
                milik[det[k]["bid"]][lab[k]] += 1
        for bid, c in milik.items():
            pools.append({"tree": pak["tree"], "gt": pak["tandan"][bid],
                          "pool": per_pool[c.most_common(1)[0][0]]})
            pools_oracle.append({"tree": pak["tree"], "gt": pak["tandan"][bid],
                                 "pool": [det[k] for k in hidup
                                          if det[k]["bid"] == bid]})
        if hitung_counting:
            v = np.zeros(4)
            for anggota in per_pool.values():
                v[EP.prediksi(anggota, "R4", skema, tau)] += 1
            cacah_pool.append(v)
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    multi = [q for q in pools if len(q["pool"]) >= 2]
    n_gt = sum(len(pak["tandan"]) for pak in paket)
    benar_R4 = EP.nilai(pools, "R4", skema, tau)["akurasi"] * len(pools) if pools else 0.0
    benar_R0 = EP.nilai(pools, "R0", skema, tau)["akurasi"] * len(pools) if pools else 0.0
    out = {
        "n_tandan_GT": n_gt,
        "R4_semua": round(benar_R4 / max(n_gt, 1), 4),
        "R0_semua": round(benar_R0 / max(n_gt, 1), 4),
        "cakupan": round(len(pools) / max(n_gt, 1), 4),
        "n_tandan": len(pools), "n_multi": len(multi),
        "frac_multi": round(len(multi) / max(len(pools), 1), 4),
        "penautan": {"presisi": round(p, 4), "recall": round(r, 4),
                     "f1": round(2 * p * r / (p + r + 1e-9), 4),
                     "ari": round(float(np.mean(aris)), 4) if aris else None},
        "R0": EP.nilai(pools, "R0", skema, tau)["akurasi"],
        "R4": EP.nilai(pools, "R4", skema, tau)["akurasi"],
        "R4_multi": EP.nilai(multi, "R4", skema, tau)["akurasi"],
        "R0cal_multi": EP.nilai(multi, "R0cal", skema, tau)["akurasi"],
        "oracle_R4": EP.nilai(pools_oracle, "R4", skema, tau)["akurasi"],
    }
    if hitung_counting:
        out["_cacah"] = (np.stack(cacah_pool), np.stack(cacah_benar))
    return out, pools_oracle


def fitur_counting_murah(P, z, conf):
    """Fitur F_all + hitungan naif TANPA memuat citra — untuk C1/C2/C3/C5."""
    benar = np.zeros(4)
    for kls in P["tandan"].values():
        benar[kls] += 1
    naif = np.zeros(4)
    per_sisi = defaultdict(Counter)
    skor = {c: {"conf": [], "area": [], "cy": []} for c in KELAS}
    for s in P["sisi"]:
        D = z[s["stem"]] if s["stem"] in z.files else np.zeros((0, 11))
        if len(D):
            _, u = np.unique(D[:, 10], return_index=True)
            D = D[np.sort(u)]
            D = D[D[:, 6:10].max(1) >= conf]
        w, h = s["wh"]
        for rr in D:
            v = rr[6:10].astype(float)
            k = int(np.argmax(v))
            naif[k] += 1
            per_sisi[s["si"]][k] += 1
            skor[KELAS[k]]["conf"].append(float(v.max()))
            skor[KELAS[k]]["area"].append(float((rr[2]-rr[0])*(rr[3]-rr[1])/(w*h)))
            skor[KELAS[k]]["cy"].append(float((rr[1]+rr[3])/2/h))
    urut = [per_sisi[k] for k in sorted(per_sisi)] or [Counter()]
    return benar, naif, EC.fitur_fall(urut, skor)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", nargs="+", type=float,
                    default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
    ap.add_argument("--ambang", nargs="+", type=float, default=[0.25, 0.45, 0.65])
    ap.add_argument("--keluaran", default=str(SUB / "results" / "pt_e_009_sapu_conf.json"))
    args = ap.parse_args()
    conf_min = min(args.conf)

    o = json.loads((SUB / "results" / "pt_e_001_oracle.json").read_text())
    skema = o["skema_bobot_dikunci"]
    pen = json.loads((SUB / "results" / "pt_e_002_penaut.json").read_text())
    nama_var = pen.get("varian_dipakai_endtoend", pen["varian_terbaik_di_val"])
    V = pen["varian"][nama_var]
    cfg = {"pakai_kelas": V.get("pakai_kelas_sama", True),
           "pakai_prob": V.get("pakai_prob_prediksi", False),
           "pakai_reid": V.get("pakai_reid", False)}
    print(f"penaut {nama_var} | cfg {cfg} | conf_min {conf_min}")

    man = PP.muat_manifest()
    ids = {s: [t for t, v in man.items() if v == s] for s in ["train", "val", "test"]}
    PP.HARAP = PP.hitung_harapan_geser(ids["train"])
    print(f"konstanta arah-putar: {len(PP.HARAP)} entri")
    desk = PP.bangun_deskriptor(ids["train"] + ids["val"] + ids["test"],
                                SUB / "results" / "deskriptor_crop.npz")
    emb = None
    reid_fn = None
    if cfg["pakai_reid"]:
        zz = np.load(SUB / "results" / "reid_embedding.npz", allow_pickle=True)
        emb = {k: zz[k] for k in zz.files}
        import torch
        model = RD.Reid().cuda().eval()
        model.load_state_dict(torch.load(SUB / "runs" / "reid_resnet18" / "best.pt"))

        def reid_fn(crops):
            keluar = []
            with torch.no_grad():
                for i in range(0, len(crops), 256):
                    keluar.append(model(RD.ke_tensor(crops[i:i+256], False, "cuda"))
                                  .float().cpu().numpy())
            return np.concatenate(keluar)

    prob = PP.bangun_prob_prediksi({k: ids[k] for k in ["train", "val", "test"]}) \
        if cfg["pakai_prob"] else None
    print("melatih penaut...")
    Xtr, ytr = PP.pasangan(ids["train"], desk, True, emb, cfg["pakai_kelas"], prob)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         random_state=PP.SEED).fit(Xtr, ytr)

    pohon = {s: [EP.muat_pohon(t) for t in ids[s]] for s in ["train", "val", "test"]}
    z = {s: np.load(SUB / "results" / f"pred_skorpenuh_{s}.npz", allow_pickle=True)
         for s in ["train", "val", "test"]}

    paket = {}
    for s in ["val", "test"]:
        print(f"lintasan mahal {s} (sekali)...", flush=True)
        paket[s] = [q for q in (siapkan_pohon(P, z[s], conf_min, reid_fn, clf, cfg)
                                for P in pohon[s]) if q]

    hasil = {"penaut": nama_var, "skema": skema, "conf_diuji": args.conf,
             "ambang_diuji": args.ambang, "sapuan_val": {}, "acuan_lama": {
                 "conf": o["conf_dikunci"], "R4_test": 0.7179,
                 "counting_C5_ridge_test": 1.0542, "pipeline_lama_kelas": 0.7203}}

    terbaik = (None, None, None, -1.0)
    for c in args.conf:
        # tau disetel ulang per conf, di val, memakai tautan ORACLE
        _, po = nilai_titik(paket["val"], c, 0.0, skema, (0.5, 1.5, 2.5), False)
        tau = EP.cari_tau(po, skema)
        for a in args.ambang:
            r, _ = nilai_titik(paket["val"], c, a, skema, tau, False)
            kunci = f"conf{c:.2f}_ambang{a:.2f}"
            hasil["sapuan_val"][kunci] = {
                "tau": tau, "R4_semua": r["R4_semua"], "R0_semua": r["R0_semua"],
                "cakupan": r["cakupan"], "R4_terdeteksi": r["R4"], "R0_terdeteksi": r["R0"],
                "R4_multi": r["R4_multi"],
                "n_tandan": r["n_tandan"], "n_tandan_GT": r["n_tandan_GT"],
                "frac_multi": r["frac_multi"],
                "f1_penaut": r["penautan"]["f1"], "oracle_R4": r["oracle_R4"]}
            print(f"  val conf={c:.2f} ambang={a:.2f} -> R4_semua {r['R4_semua']:.4f} "
                  f"(cakupan {r['cakupan']:.0%}, R4_terdeteksi {r['R4']:.4f}, "
                  f"F1 {r['penautan']['f1']:.4f})", flush=True)
            if r["R4_semua"] > terbaik[3]:
                terbaik = (c, a, tau, r["R4_semua"])
    conf, ambang, tau, _ = terbaik
    print(f"\nTERKUNCI dari val: conf={conf} ambang={ambang} tau={tau}")
    hasil["terkunci"] = {"conf": conf, "ambang": ambang, "tau": list(tau)}

    for s in ["val", "test"]:
        r, _ = nilai_titik(paket[s], conf, ambang, skema, tau, True)
        cp, cb = r.pop("_cacah")
        # penghitung pembanding dipas ulang di TRAIN pada conf yang sama
        tr = [fitur_counting_murah(P, z["train"], conf) for P in pohon["train"]]
        ev = [fitur_counting_murah(P, z[s], conf) for P in pohon[s]]
        Btr = np.stack([x[0] for x in tr]); Ntr = np.stack([x[1] for x in tr])
        Ftr = np.array([x[2] for x in tr], float)
        Bs = np.stack([x[0] for x in ev]); Ns = np.stack([x[1] for x in ev])
        Fs = np.array([x[2] for x in ev], float)
        kc = Ntr.sum(0) / np.maximum(Btr.sum(0), 1)
        ridge = Pipeline([("sc", StandardScaler()),
                          ("rg", RidgeCV(alphas=np.logspace(-3, 3, 25)))]).fit(Ftr, Btr)
        r["counting"] = {
            "C1_naif": EC.metrik(Ns, Bs),
            "C2_k_global": EC.metrik(Ns / 1.8905, Bs),
            "C3_k_per_kelas": EC.metrik(Ns / kc, Bs),
            "C4_hitung_pool": EC.metrik(cp, cb),
            "C5_ridge_fall": EC.metrik(np.clip(ridge.predict(Fs), 0, None), Bs),
        }
        hasil[s] = r
        print(f"\n--- {s} (conf {conf}, ambang {ambang}) ---")
        print(f"  kelas atas SELURUH {r['n_tandan_GT']} tandan GT: R0 {r['R0_semua']} "
              f"R4 {r['R4_semua']} (cakupan {r['cakupan']:.0%})")
        print(f"  kelas atas tandan terdeteksi saja: R0 {r['R0']} R4 {r['R4']} "
              f"(oracle {r['oracle_R4']}, multi {r['frac_multi']:.0%}, "
              f"F1 penaut {r['penautan']['f1']})")
        for k, v in r["counting"].items():
            print(f"  {k:16s} macroMAE {v['macro_mae']:.4f}  class±1 {v['class_pm1_acc']:.4f}"
                  f"  tree±1 {v['tree_pm1_acc']:.4f}  bias {v['bias_total']:+.2f}")

    t = hasil["test"]
    hasil["putusan"] = {
        "CATATAN": ("R4_semua dihitung atas SELURUH tandan GT (tak terdeteksi = "
                    "salah) supaya sebanding antar conf; R4_terdeteksi tidak "
                    "sebanding karena penyebutnya berubah"),
        "R4_semua_test": t["R4_semua"], "cakupan_test": t["cakupan"],
        "kelas_terdeteksi_vs_pipeline_lama_pp": round((t["R4"] - 0.7203) * 100, 2),
        "kelas_vs_conf010_pp": round((t["R4"] - 0.7179) * 100, 2),
        "G2_selisih_ke_oracle_pp": round((t["R4"] - t["oracle_R4"]) * 100, 2),
        "G2": "LOLOS" if (t["R4"] - t["oracle_R4"]) * 100 >= -2.0 else "GUGUR",
        "G3_C4": t["counting"]["C4_hitung_pool"]["macro_mae"],
        "G3_pembanding_terbaik": min(v["macro_mae"] for k, v in t["counting"].items()
                                     if k != "C4_hitung_pool"),
        "G3": ("LOLOS" if t["counting"]["C4_hitung_pool"]["macro_mae"] <
               min(v["macro_mae"] for k, v in t["counting"].items()
                   if k != "C4_hitung_pool") else "GUGUR"),
    }
    Path(args.keluaran).write_text(json.dumps(hasil, indent=1, ensure_ascii=False,
                                              default=float))
    print("\n" + json.dumps(hasil["putusan"], indent=1, ensure_ascii=False))
    print(f"-> {args.keluaran}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
