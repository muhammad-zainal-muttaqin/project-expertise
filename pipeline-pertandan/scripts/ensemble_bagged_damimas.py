"""PT-E-033 — Bagged ensemble selection (Caruana et al., ICML 2004) untuk DAMIMAS.

## Masalah yang diserang

PT-E-029 memakai seleksi maju serakah di VAL. Itu persis algoritma Caruana, dan
paper yang sama mendokumentasikan kelemahannya: **seleksi maju serakah overfit
himpunan hillclimb**. Di sini himpunan itu cuma 86 pohon / 919 tandan, dan
gejalanya sudah terukur empat kali dalam satu sesi -- naik di VAL, tidak
bertransfer ke TEST:

    tau ordinal per-nview        VAL 0,7595  ->  TEST 0,7318
    menambah corn224             VAL +1,31   ->  TEST -0,30
    stacking seluruh model       VAL 0,7312  ->  TEST 0,7226

Caruana et al. memberi tiga penawar, dan ketiganya dipakai di sini:

1. **Selection WITH REPLACEMENT** -- satu model boleh dipilih berkali-kali,
   sehingga bobotnya pecahan alih-alih 1/n keras. Ini melunakkan keputusan
   "masuk atau tidak" yang paling rentan derau.
2. **Bagged selection** -- seluruh prosedur seleksi diulang di banyak bootstrap
   himpunan hillclimb, lalu bobotnya dijumlahkan. Ini meredam varians PROSES
   SELEKSI, bukan varians model.
3. **Sorted initialization** -- mulai dari model tunggal terbaik pada bag itu,
   bukan dari himpunan kosong.

Bootstrap dilakukan di tingkat POHON, bukan tandan: tandan dalam satu pohon jauh
dari independen (berbagi pencahayaan, varietas, sesi pemotretan), jadi bootstrap
per-tandan akan melaporkan stabilitas yang terlalu optimistis.

Nol training baru; hanya bank probabilitas yang sudah ada.

TEST dibuka SEKALI setelah bobot dan aturan keputusan terkunci dari VAL.

Acuan: Caruana, Niculescu-Mizil, Crew & Ksikes (2004), "Ensemble Selection from
Libraries of Models", ICML.

Pemakaian:
    .venv/bin/python pipeline-pertandan/scripts/ensemble_bagged_damimas.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SUB = Path(__file__).resolve().parents[1]
R = SUB / "results"
K = 4
SEED = 0
N_BAG = 200
N_ITER = 25          # langkah seleksi per bag (dengan pengembalian)
FRAC_LIB = 0.8       # porsi library yang ditawarkan tiap bag

ANGGOTA = {
    "convnext224":     ("damimas_classifier_hibrida_convnext224_s42_pred.npz", "bunch_prob"),
    "convnext128":     ("damimas_classifier_hibrida_convnext_tiny_s42_pred.npz", "bunch_prob"),
    "klasik":          ("damimas_classifier_klasik_pred.npz", "bunch_prob"),
    "set_transformer": ("damimas_set_transformer_convnext_tiny_s42_pred.npz", "prob"),
    "corn224":         ("damimas_classifier_corn_s42_pred.npz", "bunch_prob"),
}


def muat(f, k):
    z = np.load(R / f, allow_pickle=True)
    out = {}
    for s in ("val", "test"):
        P = np.asarray(z[f"{s}_{k}"], np.float64)
        out[s] = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)
    return out


def cari_tau(P, y):
    g = P @ np.arange(K)
    kisi = np.round(np.arange(0.20, 3.00, 0.10), 2)
    best, sc = None, -1.0
    for t1 in kisi:
        for t2 in kisi[kisi > t1]:
            for t3 in kisi[kisi > t2]:
                a = float((np.searchsorted(np.array([t1, t2, t3]), g) == y).mean())
                if a > sc:
                    sc, best = a, (float(t1), float(t2), float(t3))
    return best, sc


def main() -> int:
    bank = {n: muat(f, k) for n, (f, k) in ANGGOTA.items()}
    ref = np.load(R / ANGGOTA["convnext224"][0], allow_pickle=True)
    yv = np.asarray(ref["val_bunch_y"], int); yt = np.asarray(ref["test_bunch_y"], int)
    tv = np.asarray(ref["val_bunch_tree"]); tt = np.asarray(ref["test_bunch_tree"])
    nvt = np.asarray(ref["test_bunch_nview"], int)

    nama = list(bank)
    buang = {b for i, a in enumerate(nama) for b in nama[i + 1:]
             if np.allclose(bank[a]["val"], bank[b]["val"], atol=1e-6)}
    for b in buang:
        del bank[b]
    lib = list(bank)
    print(f"library {lib}" + (f" (duplikat dibuang: {sorted(buang)})" if buang else ""))

    pohon = np.unique(tv)
    idx_p = {t: np.where(tv == t)[0] for t in pohon}
    rng = np.random.default_rng(SEED)
    hitung = {n: 0 for n in lib}

    for b in range(N_BAG):
        pil = rng.choice(len(pohon), len(pohon))                 # bootstrap POHON
        ii = np.concatenate([idx_p[pohon[k]] for k in pil])
        sub = list(rng.choice(lib, max(2, int(len(lib) * FRAC_LIB)), replace=False))
        y_b = yv[ii]

        # sorted initialization: model tunggal terbaik di bag ini
        awal = max(sub, key=lambda n: (bank[n]["val"][ii].argmax(1) == y_b).mean())
        jml = {n: 0 for n in sub}; jml[awal] = 1
        akum = bank[awal]["val"][ii].copy(); tot = 1
        skor = float((akum.argmax(1) == y_b).mean())

        for _ in range(N_ITER - 1):                              # DENGAN pengembalian
            kand = []
            for n in sub:
                cand = (akum + bank[n]["val"][ii]) / (tot + 1)
                kand.append((float((cand.argmax(1) == y_b).mean()), n))
            s_baru, n_baru = max(kand)
            if s_baru < skor - 1e-9:
                break
            akum = akum + bank[n_baru]["val"][ii]; tot += 1
            jml[n_baru] += 1; skor = s_baru
        for n, c in jml.items():
            hitung[n] += c

    total = sum(hitung.values())
    bobot = {n: hitung[n] / total for n in lib}
    print("\nbobot hasil bagging (200 bag, seleksi dengan pengembalian):")
    for n in sorted(bobot, key=lambda x: -bobot[x]):
        print(f"  {n:20} {bobot[n]:.4f}")

    Pv = sum(bobot[n] * bank[n]["val"] for n in lib)
    Pt = sum(bobot[n] * bank[n]["test"] for n in lib)

    # aturan keputusan tetap dipilih lewat CV di dalam VAL (pelajaran PT-E-029)
    fmap = {t: i % 5 for i, t in enumerate(rng.permutation(pohon))}
    fid = np.array([fmap[t] for t in tv])
    cv = {}
    for aturan in ("argmax", "ordinal"):
        ok = []
        for f in range(5):
            tr, te = fid != f, fid == f
            if aturan == "argmax":
                yh = Pv[te].argmax(1)
            else:
                t_, _ = cari_tau(Pv[tr], yv[tr])
                yh = np.searchsorted(np.array(t_), Pv[te] @ np.arange(K))
            ok.append(yh == yv[te])
        cv[aturan] = float(np.concatenate(ok).mean())
    mode = max(cv, key=cv.get)
    tau, _ = cari_tau(Pv, yv)
    print(f"\nCV dalam VAL: {  {k: round(v,4) for k,v in cv.items()} } -> {mode}")

    def putus(P):
        return P.argmax(1) if mode == "argmax" else np.searchsorted(
            np.array(tau), P @ np.arange(K))

    yhv, yh = putus(Pv), putus(Pt)
    m1 = nvt == 1
    benar_b = (yh == yt).astype(float)

    # pembanding: PT-E-029 (konfigurasi terkunci pertama)
    p29 = np.load(R / "pt_e_029_ensemble_kelas_damimas_pred.npz", allow_pickle=True)
    benar_a = (np.asarray(p29["test_yhat"], int) == yt).astype(float)
    uniq = sorted(set(tt.tolist())); ip = {t: np.where(tt == t)[0] for t in uniq}
    rr = np.random.default_rng(0); d = []
    for _ in range(2000):
        s = rr.choice(len(uniq), len(uniq))
        jj = np.concatenate([ip[uniq[k]] for k in s])
        d.append(benar_b[jj].mean() - benar_a[jj].mean())
    d = np.array(d) * 100

    hasil = {"pt_e": "033", "acuan": "Caruana et al. ICML 2004",
             "n_bag": N_BAG, "n_iter": N_ITER, "frac_lib": FRAC_LIB,
             "library": lib, "bobot": {n: round(bobot[n], 4) for n in lib},
             "aturan": mode, "tau": tau if mode == "ordinal" else None,
             "cv_val": {k: round(v, 4) for k, v in cv.items()},
             "val": round(float((yhv == yv).mean()), 4),
             "test": round(float(benar_b.mean()), 4),
             "test_1view": round(float((yh[m1] == yt[m1]).mean()), 4),
             "test_multi": round(float((yh[~m1] == yt[~m1]).mean()), 4),
             "vs_pt_e_029": {"pt_e_029_test": round(float(benar_a.mean()), 4),
                             "delta_pp": round(float((benar_b.mean() - benar_a.mean()) * 100), 2),
                             "ci95": [round(float(np.percentile(d, 2.5)), 2),
                                      round(float(np.percentile(d, 97.5)), 2)],
                             "P(delta>0)": round(float((d > 0).mean()), 3)},
             "target_IDEA": 0.80}
    np.savez_compressed(R / "pt_e_033_bagged_pred.npz", test_prob=Pt.astype(np.float32),
                        test_yhat=yh, test_y=yt, test_tree=tt, test_nview=nvt)
    (R / "pt_e_033_bagged.json").write_text(json.dumps(hasil, indent=1, ensure_ascii=False))
    print("\n=== TEST (dibuka sekali) ===")
    print(json.dumps({k: v for k, v in hasil.items()
                      if k in ("val", "test", "test_1view", "test_multi",
                               "vs_pt_e_029", "target_IDEA")}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
