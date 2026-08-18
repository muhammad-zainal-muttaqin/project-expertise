"""Mixture-of-experts strict DAMIMAS untuk klasifikasi per-tandan.

Empat anggota yang dipakai di sini seluruhnya dilatih pada DAMIMAS: classifier
klasik, dua ConvNeXt residual, dan Set Transformer. Kepala meta tidak hanya
melihat probabilitas mereka, tetapi juga ketidaksetujuan, jumlah tampak, serta
konteks prediksi tandan lain pada pohon yang sama.

Protokolnya sengaja ketat:

1. kandidat meta dibandingkan dengan prediksi OOF GroupKFold pada VAL;
2. aturan ordinal dan blend dipilih hanya dari prediksi OOF itu;
3. kandidat terpilih dipasang pada seluruh VAL;
4. TEST baru dibaca dan diprediksi setelah konfigurasi terkunci.

Kotak dan tautan tetap GT. Jadi angka keluaran adalah mutu modul classifier,
bukan klaim end-to-end.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import classifier_klasik_damimas as CK  # noqa: E402
import stacker_damimas as SD  # noqa: E402


def muat_anggota(paths: dict[str, Path], split: str):
    bank = {}
    ref = None
    nview = None
    arsip = []
    for nama, path in paths.items():
        z = np.load(path, allow_pickle=True)
        arsip.append(z)
        if nama == "set":
            p = z[f"{split}_prob"]
            y = z[f"{split}_y"]
            tree = z[f"{split}_tree"].astype(str)
            nv = z[f"{split}_nview"] if f"{split}_nview" in z.files else None
        else:
            p = z[f"{split}_bunch_prob"]
            y = z[f"{split}_bunch_y"]
            tree = z[f"{split}_bunch_tree"].astype(str)
            key_nv = f"{split}_bunch_nview"
            nv = z[key_nv] if key_nv in z.files else None
        cur = (np.asarray(y, int), np.asarray(tree, str))
        if ref is None:
            ref = cur
        elif not np.array_equal(ref[0], cur[0]) or not np.array_equal(ref[1], cur[1]):
            raise RuntimeError(f"Label/urutan pohon tidak sejajar pada anggota {nama}")
        if nv is not None:
            if nview is None:
                nview = np.asarray(nv, int)
            elif not np.array_equal(nview, nv):
                raise RuntimeError(f"Jumlah view tidak sejajar pada anggota {nama}")
        q = np.clip(np.asarray(p, float), 1e-8, None)
        bank[nama] = q / q.sum(1, keepdims=True)
    if nview is None:
        raise RuntimeError("Tidak ada anggota yang menyimpan jumlah view")
    # Array sudah dimuat ke memori; tutup handle NPZ sebelum fitting paralel.
    for z in arsip:
        z.close()
    return bank, ref[0], ref[1], nview


def konteks_pohon(bank: dict[str, np.ndarray], tree: np.ndarray) -> np.ndarray:
    """Fitur konteks tanpa label: rank dan distribusi prediksi dalam pohon."""
    nama = sorted(bank)
    n = len(tree)
    out = []
    for model in nama:
        p = bank[model]
        eks = p @ np.arange(4)
        f = np.zeros((n, 11), np.float32)
        for t in np.unique(tree):
            idx = np.flatnonzero(tree == t)
            e = eks[idx]
            urut = np.argsort(np.argsort(e, kind="stable"), kind="stable")
            f[idx, 0] = urut / max(len(idx) - 1, 1)
            f[idx, 1] = e.mean()
            f[idx, 2] = e.std()
            f[idx, 3] = e.min()
            f[idx, 4] = e.max()
            f[idx, 5:9] = p[idx].mean(0)
            f[idx, 9] = len(idx) / 20.0
            f[idx, 10] = (e - e.mean()) / max(float(e.std()), .15)
        out.append(f)
    return np.concatenate(out, axis=1)


def fitur_meta(bank: dict[str, np.ndarray], tree: np.ndarray,
               nview: np.ndarray) -> np.ndarray:
    nama = sorted(bank)
    p = np.stack([bank[k] for k in nama], axis=1)
    eps = 1e-8
    eks = p @ np.arange(4)
    ent = -(p * np.log(np.clip(p, eps, 1))).sum(2)
    urut = np.sort(p, axis=2)
    vote = np.eye(4, dtype=float)[p.argmax(2)].mean(1)
    base = np.c_[
        p.reshape(len(p), -1),
        np.log(np.clip(p, eps, 1)).reshape(len(p), -1),
        p.mean(1), p.std(1), p.min(1), p.max(1),
        eks, eks.mean(1), eks.std(1), eks.min(1), eks.max(1),
        ent, ent.mean(1), ent.std(1),
        urut[:, :, -1] - urut[:, :, -2],
        vote,
        (vote.max(1) == 1).astype(float),
        nview, np.log1p(nview), (nview == 1), (nview >= 2), (nview >= 3),
    ]
    return np.c_[base, konteks_pohon(bank, tree)].astype(np.float32)


def spesifikasi():
    out = {}
    for c in (.01, .1, 1.0):
        for cw in (None, "balanced"):
            tag = f"lr_c{c:g}_{'bal' if cw else 'plain'}"
            out[tag] = ("mc", make_pipeline(
                StandardScaler(), LogisticRegression(
                    C=c, class_weight=cw, max_iter=4000, random_state=42)))
    for leaf, l2 in ((7, 5.), (15, 10.), (31, 20.)):
        dasar = HistGradientBoostingClassifier(
            learning_rate=.04, max_iter=350, max_leaf_nodes=leaf,
            l2_regularization=l2, class_weight="balanced", random_state=42)
        out[f"hist_l{leaf}"] = ("mc", dasar)
        out[f"ord_hist_l{leaf}"] = ("ordinal", dasar)
    for leaf in (4, 8, 16):
        out[f"extra_l{leaf}"] = ("mc", ExtraTreesClassifier(
            n_estimators=700, min_samples_leaf=leaf, max_features=.75,
            class_weight="balanced", n_jobs=8, random_state=42 + leaf))
    for leaf in (5, 10):
        out[f"rf_l{leaf}"] = ("mc", RandomForestClassifier(
            n_estimators=700, min_samples_leaf=leaf, max_features=.65,
            class_weight="balanced", n_jobs=8, random_state=52 + leaf))
    return out


def oof(spec, X, y, groups):
    q = np.zeros((len(y), 4), float)
    for tr, va in GroupKFold(5).split(X, y, groups):
        m = CK.pasang(spec, X[tr], y[tr])
        q[va] = CK.prob_model(m, X[va])
    return q


def serialize_aturan(q):
    return {"objective": float(q[0]), "tie_break": float(q[1]),
            "aturan": q[2], "tau": None if q[3] is None else list(q[3]),
            "metrik": q[4]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--klasik", type=Path, default=SUB / "results" /
                    "damimas_classifier_klasik_pred.npz")
    ap.add_argument("--conv128", type=Path, default=SUB / "results" /
                    "damimas_classifier_hibrida_convnext_tiny_s42_pred.npz")
    ap.add_argument("--conv224", type=Path, default=SUB / "results" /
                    "damimas_classifier_hibrida_convnext224_s42_pred.npz")
    ap.add_argument("--set", dest="set_path", type=Path, default=SUB / "results" /
                    "damimas_set_transformer_convnext_tiny_s42_pred.npz")
    ap.add_argument("--output", type=Path, default=SUB / "results" /
                    "damimas_moe_classifier.json")
    args = ap.parse_args()
    paths = {"klasik": args.klasik, "conv128": args.conv128,
             "conv224": args.conv224, "set": args.set_path}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    # Hanya VAL yang dibuka selama seluruh seleksi.
    bv, yv, gv, nv = muat_anggota(paths, "val")
    Xv = fitur_meta(bv, gv, nv)
    specs = spesifikasi()
    kandidat = dict(bv)
    kandidat["mean_strict"] = np.mean(list(bv.values()), axis=0)
    for nama, spec in specs.items():
        print(f"OOF {nama}", flush=True)
        kandidat[nama] = oof(spec, Xv, yv, gv)
    bobot, aturan, pval, ranking = CK.pilih_blend(kandidat, yv)
    dipakai_meta = sorted(n for n in bobot if n in specs)
    terkunci = {"bobot": bobot, **serialize_aturan(aturan),
                "meta_dipasang": dipakai_meta, "fitur_dim": int(Xv.shape[1])}
    print("TERKUNCI DI VAL", json.dumps(terkunci, indent=2), flush=True)

    # TEST baru dibuka setelah kandidat, blend, dan aturan keputusan terkunci.
    bt, yt, gt, nt = muat_anggota(paths, "test")
    Xt = fitur_meta(bt, gt, nt)
    pred_t = dict(bt)
    pred_t["mean_strict"] = np.mean(list(bt.values()), axis=0)
    fitted = {}
    for nama in dipakai_meta:
        fitted[nama] = CK.pasang(specs[nama], Xv, yv)
        pred_t[nama] = CK.prob_model(fitted[nama], Xt)
    ptest = CK.gabung_prob(bobot, pred_t)
    yh_val = SD.prediksi_prob(pval, aturan[2], aturan[3])
    yh_test = SD.prediksi_prob(ptest, aturan[2], aturan[3])
    hasil = {
        "dataset": "SawitMVC-YOLO-Damimas",
        "protokol": "meta OOF GroupKFold-pohon pada VAL; fit VAL; TEST sekali",
        "kaveat": "crop GT + tautan GT; metrik modul classifier/oracle",
        "anggota_strict": {k: str(v) for k, v in paths.items()},
        "n": {"val": {"tandan": len(yv), "pohon": len(np.unique(gv))},
              "test": {"tandan": len(yt), "pohon": len(np.unique(gt))}},
        "terkunci_di_val": terkunci,
        "val_oof": SD.metrik(yv, yh_val),
        "test": SD.metrik(yt, yh_test),
        "test_subgrup": SD.metrik_subgrup(
            yt, yh_test, np.r_[0, np.cumsum(nt)]),
        "ranking_val": [
            {"nama": r[2], **serialize_aturan(r[3])} for r in ranking
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
    np.savez_compressed(
        args.output.with_name(args.output.stem + "_pred.npz"),
        val_prob=pval, val_y=yv, val_yhat=yh_val, val_tree=gv, val_nview=nv,
        test_prob=ptest, test_y=yt, test_yhat=yh_test, test_tree=gt,
        test_nview=nt,
    )
    model_path = SUB / "runs" / "moe_classifier_damimas" / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": fitted, "bobot": bobot, "aturan": aturan[2],
                 "tau": aturan[3], "fitur_dim": Xv.shape[1],
                 "anggota": sorted(paths)}, model_path, compress=3)
    print(json.dumps({"val_oof": hasil["val_oof"], "test": hasil["test"]},
                     indent=2, ensure_ascii=False))
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
