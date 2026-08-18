"""PT-E-029 — Ensemble kelas per-tandan DAMIMAS: rata-rata berbobot, bukan stacker.

## Kenapa ini dijalankan padahal "stacking seluruh model strict" sudah dicoba

`PIPELINE_DAMIMAS.md` mencatat stacking seluruh model strict berhenti di 0,7272,
KALAH dari champion tunggalnya sendiri (ConvNeXt residual 128 = 0,7378). Stacker
yang kalah dari anggota terbaiknya bukan bukti bahwa anggotanya tidak saling
melengkapi -- itu gejala khas meta-learner yang overfit. Dengan VAL cuma 86 pohon
(919 tandan), meta-learner punya jauh lebih banyak derajat kebebasan daripada
yang bisa ditopang data seleksinya.

Metode di sini sengaja jauh lebih kaku, dan itu intinya:

  - rata-rata probabilitas berbobot, BUKAN model yang dilatih di atas prediksi
  - subset dipilih SERAKAH MAJU, berhenti begitu VAL tidak naik lagi
  - bobot cuma satu parameter (seberapa berat champion), dipilih dari 4 opsi
  - `tau` ordinal dipatok selama pencarian subset, baru dipas untuk pemenang

Derajat kebebasan totalnya bisa dihitung jari, jadi 919 tandan VAL cukup untuk
memilihnya tanpa menghafal. Pendekatan yang sama menaikkan +2,56 pp (CI95
[+0,52; +4,53]) di korpus 953 pada PT-E-018, di situasi yang mirip: tiap anggota
KALAH dari C1, tetapi gabungannya menang karena galatnya terdekorelasi.

## Yang dilaporkan terpisah, dan kenapa

`PIPELINE_DAMIMAS.md` menunjukkan bottleneck kelas ada di tandan SATU-tampak
(0,6329) lawan multi-tampak (0,7753). Karena itu akurasi dipecah menurut
`nview`. PT-E-019 di korpus 953 menemukan ensemble menolong paling besar justru
di tandan satu-tampak -- di sana tidak ada agregasi multi-view yang bisa
menutupi galat, jadi mutu probabilitas menentukan segalanya. Kalau pola itu
berulang di DAMIMAS, ia menyerang bottleneck yang tepat.

TEST dibuka SEKALI setelah subset, bobot, dan `tau` terkunci dari VAL.

Pemakaian:
    .venv/bin/python pipeline-pertandan/scripts/ensemble_kelas_damimas.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SUB = Path(__file__).resolve().parents[1]
R = SUB / "results"
K = 4
SEED = 0

# model DASAR saja. `ensemble_classifier_all` dan `stacker` adalah turunan dari
# anggota-anggota ini -- memasukkannya berarti menghitung sinyal yang sama dua kali.
ANGGOTA = {
    "convnext224":     ("damimas_classifier_hibrida_convnext224_s42_pred.npz", "bunch_prob"),
    "convnext128":     ("damimas_classifier_hibrida_convnext_tiny_s42_pred.npz", "bunch_prob"),
    "klasik":          ("damimas_classifier_klasik_pred.npz", "bunch_prob"),
    "moe":             ("damimas_moe_classifier_pred.npz", "prob"),
    "set_transformer": ("damimas_set_transformer_convnext_tiny_s42_pred.npz", "prob"),
    # PT-E-030: satu-satunya anggota berloss ORDINAL, dan satu-satunya yang
    # TIDAK berjangkar pada C1 -- dua alasan galatnya bisa terdekorelasi.
    # CORAL DIBUANG (test 0,3305): weight-sharing-nya mengurung kelas tengah,
    # maks P(B2)=0,291. CORN dengan resep identik memberi 0,6983.
    "corn224":         ("damimas_classifier_corn_s42_pred.npz", "bunch_prob"),
}
PEMBANDING = {
    "stacking_semua(referensi)": ("damimas_ensemble_classifier_all_pred.npz", "bunch_prob"),
}


def muat(berkas, kunci):
    z = np.load(R / berkas, allow_pickle=True)
    out = {}
    for s in ("val", "test"):
        out[s] = np.asarray(z[f"{s}_{kunci}"], np.float64)
        out[s] = out[s] / np.clip(out[s].sum(1, keepdims=True), 1e-9, None)
    return out


def akurasi(P, y, tau=None):
    if tau is None:
        yh = P.argmax(1)
    else:
        yh = np.searchsorted(np.asarray(tau), P @ np.arange(K))
    return float((yh == y).mean()), yh


def cari_tau(P, y):
    """Ambang ordinal di ekspektasi kelas. Kisi kasar -- sengaja, supaya tidak
    ada ruang menghafal 919 titik dengan tiga parameter."""
    g = P @ np.arange(K)
    kisi = np.round(np.arange(0.20, 3.00, 0.10), 2)
    terbaik, skor = None, -1.0
    for t1 in kisi:
        for t2 in kisi[kisi > t1]:
            for t3 in kisi[kisi > t2]:
                a = float((np.searchsorted(np.array([t1, t2, t3]), g) == y).mean())
                if a > skor:
                    skor, terbaik = a, (float(t1), float(t2), float(t3))
    return terbaik, skor


def main() -> int:
    bank = {n: muat(f, k) for n, (f, k) in ANGGOTA.items()}
    ref = np.load(R / ANGGOTA["convnext224"][0], allow_pickle=True)
    y = {s: np.asarray(ref[f"{s}_bunch_y"], int) for s in ("val", "test")}
    tree = {s: np.asarray(ref[f"{s}_bunch_tree"]) for s in ("val", "test")}
    nview = {s: np.asarray(ref[f"{s}_bunch_nview"], int) for s in ("val", "test")}

    # kewarasan: seluruh bank harus bicara tentang tandan yang SAMA dan urutan sama
    for n, (f, k) in ANGGOTA.items():
        z = np.load(R / f, allow_pickle=True)
        for s in ("val", "test"):
            ky = f"{s}_bunch_y" if f"{s}_bunch_y" in z.files else f"{s}_y"
            assert np.array_equal(np.asarray(z[ky], int), y[s]), f"label beda: {n}/{s}"
    print(f"selaras: val {len(y['val'])} tandan, test {len(y['test'])} tandan")
    print(f"  satu-tampak val {int((nview['val']==1).sum())} / test {int((nview['test']==1).sum())}")

    # `moe` memilih `klasik` saja (PIPELINE_DAMIMAS.md), jadi keduanya sinyal yang
    # sama. Seleksi serakah akan memungut duplikat sebagai anggota "baru" dan
    # menghitung satu sinyal dua kali; buang lebih dulu.
    nama = list(bank)
    buang = set()
    for i, a in enumerate(nama):
        for b in nama[i + 1:]:
            if b not in buang and np.allclose(bank[a]["val"], bank[b]["val"], atol=1e-6):
                buang.add(b); print(f"duplikat: {b} == {a} -> dibuang")
    for b in buang:
        del bank[b]

    hasil = {"pt_e": "029", "anggota": list(bank), "duplikat_dibuang": sorted(buang),
             "sendiri": {}}
    print(f"\n{'model':22}{'val':>8}{'test':>8}{'test 1-view':>13}{'test multi':>12}")
    for n, P in bank.items():
        av, _ = akurasi(P["val"], y["val"])
        at, yh = akurasi(P["test"], y["test"])
        m1 = nview["test"] == 1
        a1 = float((yh[m1] == y["test"][m1]).mean())
        am = float((yh[~m1] == y["test"][~m1]).mean())
        hasil["sendiri"][n] = {"val": round(av, 4), "test": round(at, 4),
                               "test_1view": round(a1, 4), "test_multi": round(am, 4)}
        print(f"{n:22}{av:>8.4f}{at:>8.4f}{a1:>13.4f}{am:>12.4f}")
    for n, (f, k) in PEMBANDING.items():
        P = muat(f, k)
        hasil["sendiri"][n] = {"val": round(akurasi(P["val"], y["val"])[0], 4),
                               "test": round(akurasi(P["test"], y["test"])[0], 4)}
        print(f"{n:22}{hasil['sendiri'][n]['val']:>8.4f}{hasil['sendiri'][n]['test']:>8.4f}")

    # ---- seleksi maju serakah di VAL, argmax (tau dipatok = tidak dipakai) ----
    print("\nseleksi maju serakah di VAL...")
    terpilih, sisa, jalan = [], list(bank), -1.0
    while sisa:
        kand = sorted(((akurasi(np.mean([bank[m]["val"] for m in terpilih + [n]], 0),
                                y["val"])[0], n) for n in sisa), reverse=True)
        if kand[0][0] <= jalan + 1e-12:
            break
        jalan, pilih = kand[0]
        terpilih.append(pilih); sisa.remove(pilih)
        print(f"  + {pilih:20} val {jalan:.4f}  ({len(terpilih)} anggota)")

    # ---- bobot: seberapa berat anggota pertama (yang terkuat di val) ----
    terbaik = None
    for w in (1.0, 1.5, 2.0, 3.0):
        bobot = np.array([w if i == 0 else 1.0 for i in range(len(terpilih))])
        bobot = bobot / bobot.sum()
        Pv = sum(b * bank[n]["val"] for b, n in zip(bobot, terpilih))
        a, _ = akurasi(Pv, y["val"])
        if terbaik is None or a > terbaik[0]:
            terbaik = (a, w, bobot)
    av, w, bobot = terbaik
    Pv = sum(b * bank[n]["val"] for b, n in zip(bobot, terpilih))

    # ---- memilih ATURAN KEPUTUSAN lewat CV di dalam VAL, bukan lewat fit VAL ----
    #
    # Kenapa tidak boleh dipilih dari fit VAL langsung: aturan dengan parameter
    # lebih banyak selalu memenangkan fit di data yang sama tempat parameternya
    # dipas. Terukur di sini -- `tau` per-nview mendapat VAL 0,7595 (tertinggi)
    # padahal 3 ambang tambahannya dipas pada 212 tandan satu-tampak saja.
    # CV memberi tiap aturan biaya kompleksitasnya sendiri, dan TEST tidak
    # tersentuh sama sekali dalam pemilihan ini.
    m1v = nview["val"] == 1
    pohon_val = np.unique(tree["val"])
    rng_cv = np.random.default_rng(SEED)
    fold = {t: i % 5 for i, t in enumerate(rng_cv.permutation(pohon_val))}
    fid = np.array([fold[t] for t in tree["val"]])

    def cv_skor(aturan):
        benar = []
        for f in range(5):
            tr, te = fid != f, fid == f
            if aturan == "argmax":
                yh = Pv[te].argmax(1)
            elif aturan == "ordinal":
                t_, _ = cari_tau(Pv[tr], y["val"][tr])
                yh = np.searchsorted(np.array(t_), Pv[te] @ np.arange(K))
            else:
                a1_, _ = cari_tau(Pv[tr & m1v], y["val"][tr & m1v])
                am_, _ = cari_tau(Pv[tr & ~m1v], y["val"][tr & ~m1v])
                g_ = Pv[te] @ np.arange(K)
                yh = np.where(m1v[te], np.searchsorted(np.array(a1_), g_),
                              np.searchsorted(np.array(am_), g_))
            benar.append(yh == y["val"][te])
        return float(np.concatenate(benar).mean())

    cv = {a: cv_skor(a) for a in ("argmax", "ordinal", "ordinal_per_nview")}
    mode = max(cv, key=cv.get)
    print("  CV-dalam-VAL (5 fold, tingkat pohon):")
    for a, v in cv.items():
        print(f"    {a:20} {v:.4f}{'   <- terpilih' if a == mode else ''}")

    # parameter aturan terpilih baru dipas di SELURUH val
    tau, av_tau = cari_tau(Pv, y["val"])
    tau1, _ = cari_tau(Pv[m1v], y["val"][m1v])
    taum, _ = cari_tau(Pv[~m1v], y["val"][~m1v])
    g = Pv @ np.arange(K)
    av_split = float((np.where(m1v, np.searchsorted(np.array(tau1), g),
                               np.searchsorted(np.array(taum), g)) == y["val"]).mean())
    print(f"  fit VAL penuh: argmax {av:.4f} | ordinal {av_tau:.4f} | per-nview {av_split:.4f}")
    print(f"  -> aturan dikunci: {mode}")
    pakai_tau = mode != "argmax"

    # ---- TEST dibuka SEKALI ----
    Pt = sum(b * bank[n]["test"] for b, n in zip(bobot, terpilih))
    if mode == "ordinal_per_nview":
        m1t = nview["test"] == 1; gt = Pt @ np.arange(K)
        yh = np.where(m1t, np.searchsorted(np.array(tau1), gt),
                      np.searchsorted(np.array(taum), gt))
        at = float((yh == y["test"]).mean())
    else:
        at, yh = akurasi(Pt, y["test"], tau if mode == "ordinal" else None)
    m1 = nview["test"] == 1
    a1 = float((yh[m1] == y["test"][m1]).mean())
    am = float((yh[~m1] == y["test"][~m1]).mean())
    f1 = []
    for k in range(K):
        tp = int(((yh == k) & (y["test"] == k)).sum()); fp = int(((yh == k) & (y["test"] != k)).sum())
        fn = int(((yh != k) & (y["test"] == k)).sum())
        pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
        f1.append(2 * pr * rc / (pr + rc + 1e-9))

    # ---- CI tingkat pohon vs champion tunggal ----
    champ = max(hasil["sendiri"], key=lambda n: hasil["sendiri"][n]["val"]
                if n in ANGGOTA else -1)
    _, yh_c = akurasi(bank[champ]["test"], y["test"])
    benar_e = (yh == y["test"]).astype(float)
    benar_c = (yh_c == y["test"]).astype(float)
    uniq = sorted(set(tree["test"].tolist()))
    idx = {t: np.where(tree["test"] == t)[0] for t in uniq}
    rng = np.random.default_rng(SEED); d = []
    for _ in range(2000):
        pil = rng.choice(len(uniq), len(uniq))
        ii = np.concatenate([idx[uniq[k]] for k in pil])
        d.append(benar_e[ii].mean() - benar_c[ii].mean())
    d = np.array(d) * 100

    hasil["ensemble"] = {
        "subset": terpilih, "bobot": bobot.tolist(), "bobot_pertama": w,
        "aturan": mode, "tau": tau, "tau_1view": tau1, "tau_multi": taum,
        "val": round({"argmax": av, "ordinal": av_tau, "ordinal_per_nview": av_split}[mode], 4),
        "test": round(at, 4), "test_1view": round(a1, 4), "test_multi": round(am, 4),
        "macro_f1_test": round(float(np.mean(f1)), 4),
        "vs_champion": {
            "champion": champ, "champion_test": hasil["sendiri"][champ]["test"],
            "delta_pp": round(float((benar_e.mean() - benar_c.mean()) * 100), 2),
            "ci95": [round(float(np.percentile(d, 2.5)), 2),
                     round(float(np.percentile(d, 97.5)), 2)],
            "P(delta>0)": round(float((d > 0).mean()), 3), "n_pohon": len(uniq)},
        "acuan": {"champion_strict_pipeline_md": 0.7378, "stacking_semua": 0.7272,
                  "target_IDEA": 0.80},
    }
    np.savez_compressed(R / "pt_e_029_ensemble_kelas_damimas_pred.npz",
                        test_prob=Pt.astype(np.float32), test_yhat=yh,
                        test_y=y["test"], test_tree=tree["test"],
                        test_nview=nview["test"], val_prob=Pv.astype(np.float32))
    (R / "pt_e_029_ensemble_kelas_damimas.json").write_text(
        json.dumps(hasil, indent=1, ensure_ascii=False))
    print("\n=== TEST (dibuka sekali) ===")
    print(json.dumps(hasil["ensemble"], indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
