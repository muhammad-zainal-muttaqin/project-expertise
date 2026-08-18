"""Greedy stacker per-tandan khusus varietas DAMIMAS.

Masukan adalah dump C1 dan delapan C2 PT-E-014/015. Setiap tandan diubah menjadi
statistik lintas-view dan lintas-model, lalu beberapa meta-classifier ringan
dinilai dengan GroupKFold berdasarkan pohon. Pemilihan tidak melihat test.

Meta-classifier dipasang pada seluruh val (base model dilatih pada train), lalu
test dibuka sekali dengan aturan prediksi yang sudah dipilih dari OOF.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import ensemble_c as EC  # noqa: E402


SUB = Path(__file__).resolve().parents[1]
KELAS = ("B1", "B2", "B3", "B4")


def potong_flat(flat: np.ndarray, offset: np.ndarray, indeks: np.ndarray):
    bagian = [flat[offset[i]:offset[i + 1]] for i in indeks]
    off = np.r_[0, np.cumsum([len(x) for x in bagian])].astype(np.int32)
    return np.concatenate(bagian), off


def muat_split(dumps, ref, split: str):
    tree_semua = ref[f"{split}__tree"].astype(str)
    indeks = np.flatnonzero(np.char.startswith(tree_semua, "DAMIMAS_"))
    offset_asli = ref[f"{split}__offset"]
    nama = ["C1"] + [f"C2_{tag}" for tag in sorted(dumps)]
    flat = []
    p, offset = potong_flat(ref[f"{split}__C1_rata"], offset_asli, indeks)
    flat.append(p)
    for tag in sorted(dumps):
        p, off = potong_flat(dumps[tag][f"{split}__C2_rata"], offset_asli, indeks)
        if not np.array_equal(off, offset):
            raise RuntimeError(f"Offset tidak sejajar: {tag}/{split}")
        flat.append(p)
    return {"nama": nama, "flat": np.stack(flat, 0), "offset": offset,
            "y": ref[f"{split}__y"][indeks].astype(int),
            "tree": tree_semua[indeks]}


def ringkas_prob(p: np.ndarray) -> np.ndarray:
    """Statistik satu model untuk satu tandan; p berbentuk (view, 4)."""
    eps = 1e-9
    eks = p @ np.arange(4)
    ent = -(p * np.log(np.clip(p, eps, 1))).sum(1)
    return np.r_[p.mean(0), p.std(0), p.min(0), p.max(0),
                 eks.mean(), eks.std(), eks.min(), eks.max(),
                 ent.mean(), ent.std()]


def fitur(data):
    M, _, _ = data["flat"].shape
    X = []
    for i in range(len(data["y"])):
        a, b = data["offset"][i:i + 2]
        per_model = data["flat"][:, a:b, :]
        f = np.concatenate([ringkas_prob(per_model[m]) for m in range(M)])
        rerata_model = per_model.mean(0)
        f = np.r_[f, ringkas_prob(rerata_model),
                  per_model.mean(1).std(0),
                  per_model.std(1).mean(0),
                  b - a, *[int(b - a == n) for n in range(1, 7)]]
        X.append(f)
    return np.asarray(X, np.float32)


def kandidat_model():
    out = {}
    for c in (0.01, 0.1, 1.0, 10.0):
        for cw in (None, "balanced"):
            tag = f"logreg_C{c:g}_{'bal' if cw else 'plain'}"
            out[tag] = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c, class_weight=cw, max_iter=3000,
                                   random_state=0))
    for leaf, l2 in ((7, 1.0), (15, 1.0), (15, 10.0), (31, 10.0)):
        out[f"hist_leaf{leaf}_l2{l2:g}"] = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=250, max_leaf_nodes=leaf,
            l2_regularization=l2, random_state=0)
    for leaf, mf in ((2, 0.5), (4, 0.5), (8, 0.7), (12, 1.0)):
        out[f"extra_leaf{leaf}_mf{mf:g}"] = ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=leaf, max_features=mf,
            class_weight="balanced", n_jobs=-1, random_state=0)
    return out


def metrik(y, yh):
    p, r, f, dukung = precision_recall_fscore_support(
        y, yh, labels=np.arange(4), zero_division=0)
    return {"akurasi": float(accuracy_score(y, yh)),
            "macro_f1": float(f1_score(y, yh, average="macro")),
            "mae_ordinal": float(np.abs(y - yh).mean()),
            "precision_per_kelas": {KELAS[i]: float(p[i]) for i in range(4)},
            "recall_per_kelas": {KELAS[i]: float(r[i]) for i in range(4)},
            "f1_per_kelas": {KELAS[i]: float(f[i]) for i in range(4)},
            "support_per_kelas": {KELAS[i]: int(dukung[i]) for i in range(4)},
            "confusion": confusion_matrix(y, yh, labels=np.arange(4)).tolist()}


def metrik_subgrup(y, yh, offset):
    n = np.diff(offset)
    grup = {"satu_tampak": n == 1, "multi_tampak": n >= 2}
    grup.update({f"{k}_tampak": n == k for k in sorted(set(n.tolist()))})
    return {nama: {"n": int(mask.sum()), **metrik(y[mask], yh[mask])}
            for nama, mask in grup.items() if mask.any()}


BOBOT_AKURASI = 0.55


def skor_obj(m):
    """Sasaran tunggal yang tidak boleh membeli akurasi dengan merusak kelas kecil."""
    return BOBOT_AKURASI * m["akurasi"] + (1 - BOBOT_AKURASI) * m["macro_f1"]


def oof_prob(model, X, y, groups):
    out = np.zeros((len(y), 4), float)
    for tr, va in GroupKFold(5).split(X, y, groups):
        m = clone(model).fit(X[tr], y[tr])
        q = m.predict_proba(X[va])
        for j, c in enumerate(m.classes_):
            out[va, int(c)] = q[:, j]
    return out


def prediksi_prob(prob, aturan, tau=None):
    if aturan == "argmax":
        return prob.argmax(1)
    return np.searchsorted(np.asarray(tau), prob @ np.arange(4))


def cari_tau_cepat(prob, y):
    """Cari tiga ambang ordinal persis, tanpa matriks (triplet x sampel).

    Objective dapat diuraikan menjadi jumlah kontribusi empat interval kelas.
    Karena itu setiap interval cukup menyimpan TP dan banyak prediksi; seluruh
    triplet lalu dinilai lewat lookup array kecil. Hasilnya identik dengan
    enumerasi prediksi, tetapi beberapa ribu kali lebih hemat alokasi.
    """
    g = prob @ np.arange(4)
    kisi = np.round(np.arange(0.10, 3.00, 0.05), 2)
    urut = np.argsort(g, kind="stable")
    gs, ys = g[urut], y[urut]
    posisi = np.searchsorted(gs, kisi, side="left")
    pref = np.zeros((len(y) + 1, 4), int)
    for k in range(4):
        pref[1:, k] = np.cumsum(ys == k)
    support = pref[-1].astype(float)

    def kontribusi(k, awal, akhir):
        awal = np.asarray(awal); akhir = np.asarray(akhir)
        tp = pref[akhir, k] - pref[awal, k]
        npred = akhir - awal
        acc = tp / len(y)
        f1 = 2 * tp / np.maximum(npred + support[k], 1e-9)
        return BOBOT_AKURASI * acc + (1 - BOBOT_AKURASI) * f1 / 4, acc, f1

    nol = np.zeros(len(kisi), int)
    ujung = np.full(len(kisi), len(y), int)
    s0, a0, _ = kontribusi(0, nol, posisi)
    s3, a3, _ = kontribusi(3, posisi, ujung)
    s1 = np.full((len(kisi), len(kisi)), -np.inf)
    s2 = np.full_like(s1, -np.inf)
    a1 = np.zeros_like(s1); a2 = np.zeros_like(s1)
    for i in range(len(kisi)):
        q1, qa1, _ = kontribusi(1, posisi[i], posisi)
        q2, qa2, _ = kontribusi(2, posisi[i], posisi)
        s1[i] = q1; a1[i] = qa1
        s2[i] = q2; a2[i] = qa2

    ijk = np.array(list(itertools.combinations(range(len(kisi)), 3)), int)
    i, j, k = ijk.T
    obj = s0[i] + s1[i, j] + s2[j, k] + s3[k]
    ak = a0[i] + a1[i, j] + a2[j, k] + a3[k]
    # Objective utama, accuracy hanya pemecah seri numerik.
    q = int(np.argmax(obj + 1e-10 * ak))
    return tuple(float(x) for x in kisi[ijk[q]])


def pilih_aturan(prob, y):
    kandidat = [("argmax", None)]
    kandidat.append(("ordinal", cari_tau_cepat(prob, y)))
    hasil = []
    for aturan, tau in kandidat:
        m = metrik(y, prediksi_prob(prob, aturan, tau))
        hasil.append((skor_obj(m), m["akurasi"], aturan, tau, m))
    return max(hasil, key=lambda x: (x[0], x[1]))


def prob_langsung(data):
    """Kepala tanpa meta-training; berguna sebagai jangkar kelas minoritas."""
    out = {"langsung_C1_mean": [], "langsung_C1_conf": [],
           "langsung_semua_mean": [], "langsung_semua_conf": []}
    for i in range(len(data["y"])):
        a, b = data["offset"][i:i + 2]
        pm = data["flat"][:, a:b, :].astype(float)
        c1 = pm[0]
        rerata = pm.mean(0)
        out["langsung_C1_mean"].append(c1.mean(0))
        out["langsung_C1_conf"].append(
            np.average(c1, axis=0, weights=np.maximum(c1.max(1), 1e-6)))
        out["langsung_semua_mean"].append(rerata.mean(0))
        out["langsung_semua_conf"].append(
            np.average(rerata, axis=0, weights=np.maximum(rerata.max(1), 1e-6)))
    return {k: np.asarray(v) for k, v in out.items()}


def baseline_c1(data_val, data_test):
    Pv, offv = data_val["flat"][0], data_val["offset"]
    Pt, offt = data_test["flat"][0], data_test["offset"]
    mv, _, tau = EC.nilai_anggota(Pv, offv, data_val["y"], cari=True)
    mt, mtm, _ = EC.nilai_anggota(Pt, offt, data_test["y"], tau=tau)
    return {"tau": tau, "val": mv, "test": mt, "test_multi": mtm}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keluaran", type=Path,
                    default=SUB / "results" / "damimas_stacker_pertandan.json")
    ap.add_argument("--model-out", type=Path,
                    default=SUB / "runs" / "stacker_damimas" / "stacker.joblib")
    ap.add_argument("--pred-out", type=Path,
                    default=SUB / "results" / "damimas_stacker_pred.npz")
    args = ap.parse_args()

    dumps = EC.muat_dump()
    if not dumps:
        raise RuntimeError("Dump PT-E-014 tidak ditemukan")
    ref = dumps[sorted(dumps)[0]]
    data = {s: muat_split(dumps, ref, s) for s in ("val", "test")}
    X = {s: fitur(data[s]) for s in data}
    yv, groups = data["val"]["y"], data["val"]["tree"]
    langsung = {s: prob_langsung(data[s]) for s in data}

    models = kandidat_model()
    oof, ranking = {}, []
    for nama, model in models.items():
        p = oof_prob(model, X["val"], yv, groups)
        oof[nama] = p
        pilih = pilih_aturan(p, yv)
        ranking.append((pilih[0], pilih[1], nama, pilih[2], pilih[3], pilih[4]))
        print(f"{nama:28s} OOF acc={pilih[4]['akurasi']:.4f} "
              f"macroF1={pilih[4]['macro_f1']:.4f} rule={pilih[2]}", flush=True)
    for nama, p in langsung["val"].items():
        oof[nama] = p
        pilih = pilih_aturan(p, yv)
        ranking.append((pilih[0], pilih[1], nama, pilih[2], pilih[3], pilih[4]))
        print(f"{nama:28s} direct acc={pilih[4]['akurasi']:.4f} "
              f"macroF1={pilih[4]['macro_f1']:.4f} rule={pilih[2]}", flush=True)
    ranking.sort(reverse=True)

    # Forward selection berbobot atas probabilitas OOF/direct. Kepala C1
    # langsung ikut sebagai jangkar agar kelas B2 tidak dikorbankan demi
    # accuracy mayoritas. Test sama sekali belum disentuh di sini.
    bobot = {ranking[0][2]: 1.0}
    prob_best = oof[ranking[0][2]]
    best = pilih_aturan(prob_best, yv)
    while len(bobot) < 5:
        calon = []
        for nama in oof:
            if nama in bobot:
                continue
            for w in (0.10, 0.20, 0.30, 0.40, 0.50):
                p = (1 - w) * prob_best + w * oof[nama]
                q = pilih_aturan(p, yv)
                calon.append((q[0], q[1], nama, w, q, p))
        if not calon:
            break
        c = max(calon, key=lambda x: (x[0], x[1]))
        if c[0] <= best[0] + 1e-9:
            break
        bobot = {n: (1 - c[3]) * w for n, w in bobot.items()}
        bobot[c[2]] = c[3]
        best, prob_best = c[4], c[5]

    _, _, aturan, tau, m_oof = best
    print(f"TERPILIH OOF: {bobot} rule={aturan} tau={tau} "
          f"acc={m_oof['akurasi']:.4f} macroF1={m_oof['macro_f1']:.4f}")

    fitted, test_prob = {}, {}
    for nama, w in bobot.items():
        if nama in models:
            m = clone(models[nama]).fit(X["val"], yv)
            fitted[nama] = m
            test_prob[nama] = m.predict_proba(X["test"])
        else:
            test_prob[nama] = langsung["test"][nama]
    pt = sum(bobot[n] * test_prob[n] for n in bobot)
    yht = prediksi_prob(pt, aturan, tau)
    mt = metrik(data["test"]["y"], yht)
    base = baseline_c1(data["val"], data["test"])
    yhb = prediksi_prob(langsung["test"]["langsung_C1_conf"], "ordinal",
                        base["tau"])

    hasil = {
        "dataset": "SawitMVC-YOLO-Damimas",
        "protokol": "base model=train; stacker dipilih dengan GroupKFold pohon di val; test sekali",
        "n": {s: {"tandan": int(len(data[s]["y"])),
                    "pohon": int(len(set(data[s]["tree"]))) } for s in data},
        "fitur_dim": int(X["val"].shape[1]), "anggota_base": data["val"]["nama"],
        "baseline_C1_R4": base,
        "ranking_oof": [{"model": r[2], "aturan": r[3], "tau": r[4],
                         "metrik": r[5]} for r in ranking],
        "stacker": {"bobot": bobot, "objective_bobot_akurasi": BOBOT_AKURASI,
                    "aturan": aturan, "tau": tau,
                    "oof_val": m_oof, "test": mt,
                    "test_subgrup": metrik_subgrup(
                        data["test"]["y"], yht, data["test"]["offset"])},
        "baseline_test_subgrup": metrik_subgrup(
            data["test"]["y"], yhb, data["test"]["offset"]),
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": fitted, "bobot": bobot, "aturan": aturan, "tau": tau,
                 "anggota_base": data["val"]["nama"]}, args.model_out)
    np.savez_compressed(args.pred_out, test_prob=pt, test_y=data["test"]["y"],
                        test_yhat=yht, test_tree=data["test"]["tree"],
                        test_n_tampak=np.diff(data["test"]["offset"]),
                        val_prob=prob_best, val_y=yv,
                        val_yhat=prediksi_prob(prob_best, aturan, tau),
                        val_tree=groups,
                        val_n_tampak=np.diff(data["val"]["offset"]))
    hasil["model_out"] = str(args.model_out.relative_to(SUB))
    args.keluaran.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
    print(json.dumps({"baseline": hasil["baseline_C1_R4"],
                      "stacker": hasil["stacker"]}, indent=2, ensure_ascii=False))
    print(f"-> {args.keluaran}")


if __name__ == "__main__":
    main()
