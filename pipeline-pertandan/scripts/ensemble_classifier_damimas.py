"""Stacking DAMIMAS-only untuk kepala kelas per-view dan per-tandan.

Anggota default adalah classifier klasik dan ConvNeXt residual yang keduanya
dipasang hanya pada DAMIMAS; kepala set ditambahkan bila artefaknya tersedia.
Meta-model dibuat dengan GroupKFold berdasarkan pohon pada VAL. TEST tidak
dibaca sampai model, blend, aturan keputusan, dan ambang ordinal terkunci.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import classifier_klasik_damimas as CK  # noqa: E402
import stacker_damimas as SD  # noqa: E402


def prob_aligned(model, X):
    q = np.zeros((len(X), 4), float)
    p = model.predict_proba(X)
    for j, c in enumerate(model.classes_):
        q[:, int(c)] = p[:, j]
    return q


def fitur_meta(bank: dict[str, np.ndarray]) -> np.ndarray:
    p = np.stack([bank[k] for k in sorted(bank)], 1).astype(float)
    eps = 1e-7
    eks = p @ np.arange(4)
    ent = -(p * np.log(np.clip(p, eps, 1))).sum(2)
    urut = np.sort(p, axis=2)
    return np.c_[p.reshape(len(p), -1), np.log(np.clip(p, eps, 1)).reshape(len(p), -1),
                 p.mean(1), p.std(1), p.min(1), p.max(1),
                 eks, eks.mean(1), eks.std(1), ent,
                 urut[:, :, -1] - urut[:, :, -2]]


def model_meta():
    return {
        "meta_lr_c01": make_pipeline(
            StandardScaler(), LogisticRegression(C=.1, class_weight="balanced",
                                                   max_iter=3000, random_state=42)),
        "meta_lr_c1": make_pipeline(
            StandardScaler(), LogisticRegression(C=1., class_weight="balanced",
                                                   max_iter=3000, random_state=42)),
        "meta_hist_l15": HistGradientBoostingClassifier(
            learning_rate=.05, max_iter=250, max_leaf_nodes=15,
            l2_regularization=8., class_weight="balanced", random_state=42),
        "meta_extra_l6": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=6, max_features=.8,
            class_weight="balanced", n_jobs=-1, random_state=42),
    }


def oof(model, X, y, groups):
    out = np.zeros((len(y), 4), float)
    for tr, va in GroupKFold(5).split(X, y, groups):
        m = clone(model).fit(X[tr], y[tr])
        out[va] = prob_aligned(m, X[va])
    return out


def pilih_kepala(bank_val, y, groups):
    X = fitur_meta(bank_val)
    specs = model_meta()
    kandidat = dict(bank_val)
    kandidat["langsung_mean"] = np.mean(list(bank_val.values()), axis=0)
    for nama, model in specs.items():
        print(f"  OOF {nama}", flush=True)
        kandidat[nama] = oof(model, X, y, groups)
    bobot, aturan, pbest, ranking = CK.pilih_blend(kandidat, y)
    return {"X": X, "specs": specs, "kandidat": kandidat,
            "bobot": bobot, "aturan": aturan, "prob": pbest,
            "ranking": ranking}


def infer_kepala(cfg, bank_val, bank_test, y_val):
    Xt = fitur_meta(bank_test)
    pred = dict(bank_test)
    pred["langsung_mean"] = np.mean(list(bank_test.values()), axis=0)
    fitted = {}
    for nama in cfg["bobot"]:
        if nama not in cfg["specs"]:
            continue
        m = clone(cfg["specs"][nama]).fit(cfg["X"], y_val)
        fitted[nama] = m
        pred[nama] = prob_aligned(m, Xt)
    return CK.gabung_prob(cfg["bobot"], pred), fitted


def ambil(z, prefix, split):
    if prefix == "bunch":
        return (z[f"{split}_bunch_prob"], z[f"{split}_bunch_y"],
                z[f"{split}_bunch_tree"].astype(str))
    return (z[f"{split}_view_prob"], z[f"{split}_view_y"],
            z[f"{split}_view_tree"].astype(str))


def cek_sama(ref, cur, nama):
    for a, b, kolom in zip(ref[1:], cur[1:], ("label", "pohon")):
        if not np.array_equal(a, b):
            raise RuntimeError(f"Urutan {kolom} tidak sejajar pada {nama}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--klasik", type=Path, default=SUB / "results" /
                    "damimas_classifier_klasik_pred.npz")
    ap.add_argument("--hibrida", type=Path, default=SUB / "results" /
                    "damimas_classifier_hibrida_convnext_tiny_s42_pred.npz")
    ap.add_argument("--set", dest="set_path", type=Path, default=SUB / "results" /
                    "damimas_set_transformer_convnext_tiny_s42_pred.npz")
    ap.add_argument("--tambahan", action="append", default=[],
                    help="anggota DAMIMAS tambahan dalam bentuk NAMA=PATH.npz")
    ap.add_argument("--output", type=Path, default=SUB / "results" /
                    "damimas_ensemble_classifier.json")
    args = ap.parse_args()
    for p in (args.klasik, args.hibrida):
        if not p.exists():
            raise FileNotFoundError(p)
    arsip = {"klasik": np.load(args.klasik, allow_pickle=True),
             "hibrida": np.load(args.hibrida, allow_pickle=True)}
    if args.set_path.exists():
        arsip["set"] = np.load(args.set_path, allow_pickle=True)
    for item in args.tambahan:
        if "=" not in item:
            raise ValueError("--tambahan harus NAMA=PATH.npz")
        nama, path = item.split("=", 1)
        if nama in arsip:
            raise ValueError(f"Nama anggota duplikat: {nama}")
        arsip[nama] = np.load(Path(path), allow_pickle=True)

    hasil, model_out, pred_out = {}, {}, {}
    for kepala in ("view", "bunch"):
        # Set Transformer hanya menghasilkan distribusi per-tandan.
        anggota = {k: v for k, v in arsip.items() if not (kepala == "view" and k == "set")}
        ref_val = ambil(arsip["hibrida"], kepala, "val")
        bank_val = {}
        for nama, z in anggota.items():
            if nama == "set":
                cur = (z["val_prob"], z["val_y"], z["val_tree"].astype(str))
            else:
                cur = ambil(z, kepala, "val")
            cek_sama(ref_val, cur, nama)
            bank_val[nama] = cur[0]
        yv, gv = ref_val[1], ref_val[2]
        print(f"=== kepala {kepala}: {list(bank_val)} ===", flush=True)
        cfg = pilih_kepala(bank_val, yv, gv)

        # Semua pilihan kini terkunci; baru ambil array TEST.
        ref_test = ambil(arsip["hibrida"], kepala, "test")
        bank_test = {}
        for nama, z in anggota.items():
            if nama == "set":
                cur = (z["test_prob"], z["test_y"], z["test_tree"].astype(str))
            else:
                cur = ambil(z, kepala, "test")
            cek_sama(ref_test, cur, nama)
            bank_test[nama] = cur[0]
        pt, fitted = infer_kepala(cfg, bank_val, bank_test, yv)
        q = cfg["aturan"]
        yh_val = SD.prediksi_prob(cfg["prob"], q[2], q[3])
        yh_test = SD.prediksi_prob(pt, q[2], q[3])
        hasil[kepala] = {
            "anggota": sorted(bank_val), "bobot": cfg["bobot"],
            "aturan": q[2], "tau": q[3],
            "val_oof": SD.metrik(yv, yh_val),
            "test": SD.metrik(ref_test[1], yh_test),
            "ranking_val": CK.serial_ranking(cfg["ranking"]),
        }
        if kepala == "bunch":
            nview = arsip["hibrida"]["test_bunch_nview"]
            hasil[kepala]["test_subgrup"] = SD.metrik_subgrup(
                ref_test[1], yh_test, np.r_[0, np.cumsum(nview)])
            pred_out["test_nview"] = nview
        model_out[kepala] = {"fitted": fitted, "bobot": cfg["bobot"],
                             "aturan": q[2], "tau": q[3]}
        pred_out.update({f"val_{kepala}_prob": cfg["prob"],
                         f"val_{kepala}_y": yv, f"val_{kepala}_tree": gv,
                         f"test_{kepala}_prob": pt,
                         f"test_{kepala}_y": ref_test[1],
                         f"test_{kepala}_tree": ref_test[2]})

    payload = {
        "dataset": "SawitMVC-YOLO-Damimas",
        "protokol": "meta OOF GroupKFold pohon pada VAL; TEST sekali sesudah terkunci",
        "kaveat": "crop GT + tautan GT; metrik modul/oracle",
        "sumber": {"klasik": str(args.klasik), "hibrida": str(args.hibrida),
                   "set": str(args.set_path) if args.set_path.exists() else None,
                   "tambahan": args.tambahan},
        **hasil,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    np.savez_compressed(args.output.with_name(args.output.stem + "_pred.npz"), **pred_out)
    model_path = SUB / "runs" / "ensemble_classifier_damimas" / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_out, model_path)
    print(json.dumps({k: {"val": v["val_oof"], "test": v["test"]}
                      for k, v in hasil.items()}, indent=2, ensure_ascii=False))
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
