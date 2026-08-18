"""Kepala mixture-of-experts per-view strict DAMIMAS.

Berbeda dari ``moe_classifier_damimas.py`` yang bekerja setelah beberapa view
ditautkan, kepala ini mengoptimalkan klasifikasi satu kotak/satu citra. Seleksi
meta-model, blend, dan ambang ordinal memakai OOF GroupKFold per pohon pada VAL;
TEST baru dibaca setelah konfigurasi terkunci.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import classifier_klasik_damimas as CK  # noqa: E402
import stacker_damimas as SD  # noqa: E402


def muat(paths, split):
    bank, ref, key = {}, None, None
    handles = []
    for nama, path in paths.items():
        z = np.load(path, allow_pickle=True); handles.append(z)
        p = np.asarray(z[f"{split}_view_prob"], float)
        y = np.asarray(z[f"{split}_view_y"], int)
        tree = z[f"{split}_view_tree"].astype(str)
        if f"{split}_view_key" in z.files:
            key = z[f"{split}_view_key"].astype(str)
        if ref is None:
            ref = (y, tree)
        elif not np.array_equal(y, ref[0]) or not np.array_equal(tree, ref[1]):
            raise RuntimeError(f"Urutan anggota tidak sejajar: {nama}")
        bank[nama] = np.clip(p, 1e-8, None) / np.clip(p.sum(1, keepdims=True), 1e-9, None)
    for z in handles:
        z.close()
    if key is None:
        raise RuntimeError("Kunci view dari classifier klasik tidak ditemukan")
    return bank, ref[0], ref[1], key


def fitur(bank, tree, key):
    p = np.stack([bank[n] for n in sorted(bank)], 1)
    eks = p @ np.arange(4)
    ent = -(p * np.log(np.clip(p, 1e-8, 1))).sum(2)
    urut = np.sort(p, axis=2)
    vote = np.eye(4)[p.argmax(2)].mean(1)
    side = np.asarray([int(x.split("|")[1]) for x in key])
    nv = np.zeros(len(key), int)
    for t in np.unique(tree):
        idx = np.flatnonzero(tree == t); nv[idx] = side[idx].max() + 1
    sudut = 2 * np.pi * side / np.maximum(nv, 1)
    ctx = []
    for m in range(p.shape[1]):
        f = np.zeros((len(p), 8), np.float32)
        for t in np.unique(tree):
            idx = np.flatnonzero(tree == t); e = eks[idx, m]
            rank = np.argsort(np.argsort(e, kind="stable"), kind="stable")
            f[idx, 0] = rank / max(len(idx) - 1, 1)
            f[idx, 1] = e.mean(); f[idx, 2] = e.std()
            f[idx, 3:7] = p[idx, m].mean(0)
            f[idx, 7] = (e - e.mean()) / max(float(e.std()), .15)
        ctx.append(f)
    return np.c_[
        p.reshape(len(p), -1), np.log(np.clip(p, 1e-8, 1)).reshape(len(p), -1),
        p.mean(1), p.std(1), p.min(1), p.max(1),
        eks, eks.mean(1), eks.std(1), ent,
        urut[:, :, -1] - urut[:, :, -2], vote,
        side / np.maximum(nv - 1, 1), nv / 8., np.sin(sudut), np.cos(sudut),
        *ctx,
    ].astype(np.float32)


def specs():
    return {
        "lr_c01": ("mc", make_pipeline(StandardScaler(), LogisticRegression(
            C=.1, class_weight="balanced", max_iter=4000, random_state=42))),
        "lr_c1": ("mc", make_pipeline(StandardScaler(), LogisticRegression(
            C=1., class_weight="balanced", max_iter=4000, random_state=42))),
        "hist_l15": ("mc", HistGradientBoostingClassifier(
            learning_rate=.04, max_iter=350, max_leaf_nodes=15,
            l2_regularization=10., class_weight="balanced", random_state=42)),
        "ord_hist_l15": ("ordinal", HistGradientBoostingClassifier(
            learning_rate=.04, max_iter=350, max_leaf_nodes=15,
            l2_regularization=10., class_weight="balanced", random_state=43)),
        "extra_l8": ("mc", ExtraTreesClassifier(
            n_estimators=700, min_samples_leaf=8, max_features=.75,
            class_weight="balanced", n_jobs=8, random_state=42)),
    }


def oof(spec, X, y, group):
    out = np.zeros((len(y), 4), float)
    for tr, va in GroupKFold(5).split(X, y, group):
        out[va] = CK.prob_model(CK.pasang(spec, X[tr], y[tr]), X[va])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--klasik", type=Path, default=SUB / "results" /
                    "damimas_classifier_klasik_pred.npz")
    ap.add_argument("--conv128", type=Path, default=SUB / "results" /
                    "damimas_classifier_hibrida_convnext_tiny_s42_pred.npz")
    ap.add_argument("--conv224", type=Path, default=SUB / "results" /
                    "damimas_classifier_hibrida_convnext224_s42_pred.npz")
    ap.add_argument("--output", type=Path, default=SUB / "results" /
                    "damimas_moe_view.json")
    args = ap.parse_args()
    paths = {"klasik": args.klasik, "conv128": args.conv128,
             "conv224": args.conv224}
    bv, yv, gv, kv = muat(paths, "val")
    Xv = fitur(bv, gv, kv); model_specs = specs()
    kandidat = dict(bv); kandidat["mean_strict"] = np.mean(list(bv.values()), 0)
    for nama, spec in model_specs.items():
        print(f"OOF {nama}", flush=True); kandidat[nama] = oof(spec, Xv, yv, gv)
    bobot, aturan, pval, ranking = CK.pilih_blend(kandidat, yv)
    dipakai = sorted(n for n in bobot if n in model_specs)
    lock = {"bobot": bobot, "aturan": aturan[2],
            "tau": None if aturan[3] is None else list(aturan[3]),
            "val": aturan[4], "fitur_dim": int(Xv.shape[1]), "meta": dipakai}
    print("TERKUNCI DI VAL", json.dumps(lock, indent=2), flush=True)

    bt, yt, gt, kt = muat(paths, "test")
    Xt = fitur(bt, gt, kt); pred = dict(bt)
    pred["mean_strict"] = np.mean(list(bt.values()), 0); fitted = {}
    for nama in dipakai:
        fitted[nama] = CK.pasang(model_specs[nama], Xv, yv)
        pred[nama] = CK.prob_model(fitted[nama], Xt)
    ptest = CK.gabung_prob(bobot, pred)
    yh_val = SD.prediksi_prob(pval, aturan[2], aturan[3])
    yh_test = SD.prediksi_prob(ptest, aturan[2], aturan[3])
    hasil = {"dataset": "SawitMVC-YOLO-Damimas",
             "protokol": "meta OOF GroupKFold-pohon pada VAL; TEST sekali",
             "kaveat": "crop GT; klasifikasi per-view modular",
             "anggota_strict": {n: str(p) for n, p in paths.items()},
             "terkunci_di_val": lock,
             "val_oof": SD.metrik(yv, yh_val), "test": SD.metrik(yt, yh_test),
             "ranking_val": [{"nama": r[2], "objective": float(r[3][0]),
                               "aturan": r[3][2],
                               "tau": None if r[3][3] is None else list(r[3][3]),
                               "metrik": r[3][4]} for r in ranking]}
    args.output.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
    np.savez_compressed(args.output.with_name(args.output.stem + "_pred.npz"),
                        val_prob=pval, val_y=yv, val_yhat=yh_val, val_tree=gv,
                        val_key=kv, test_prob=ptest, test_y=yt,
                        test_yhat=yh_test, test_tree=gt, test_key=kt)
    run = SUB / "runs" / "moe_view_damimas"; run.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": fitted, "bobot": bobot, "aturan": aturan[2],
                 "tau": aturan[3], "fitur_dim": Xv.shape[1]},
                run / "model.joblib", compress=3)
    print(json.dumps({"val": hasil["val_oof"], "test": hasil["test"]},
                     indent=2, ensure_ascii=False))
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
