"""Counting CatBoost multi-bank dengan guard OOF + validation.

PT-E-026 menunjukkan learner klasik dapat memilih fitur multi-bank yang tampak
kuat di 86 pohon VAL tetapi tidak bertransfer ke TEST. Skrip ini memakai satu
resep CatBoost regularized yang tetap, membuat prediksi out-of-fold pada TRAIN,
dan menggabungkan OOF-TRAIN dengan prediksi VAL bersih untuk memilih kepala per
kelas, kalibrasi, dan rekonsiliasi. TEST baru dimuat setelah lock tercetak.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold


SUB = Path(__file__).resolve().parents[1]
ROOT = SUB.parent
sys.path.insert(0, str(Path(__file__).parent))
import counting_damimas as CD  # noqa: E402
import counting_multibank_damimas as CM  # noqa: E402


def model_cat(seed: int) -> CatBoostRegressor:
    # Satu resep tetap: pohon dangkal, feature subsampling, dan L2 kuat untuk
    # n=641 dengan ribuan fitur. Tidak ada grid yang dipilih dari TEST.
    return CatBoostRegressor(
        iterations=500, depth=5, learning_rate=.035,
        loss_function="MultiRMSE", l2_leaf_reg=20., random_strength=1.,
        rsm=.25, random_seed=seed, thread_count=8,
        verbose=False, allow_writing_files=False,
    )


def oof_dan_val(Xtr, ytr, Xv, folds: int, seed: int):
    oof = np.zeros_like(ytr, dtype=float)
    kf = KFold(folds, shuffle=True, random_state=seed)
    for fold, (a, b) in enumerate(kf.split(Xtr), 1):
        print(f"  fold {fold}/{folds}", flush=True)
        m = model_cat(seed + fold).fit(Xtr[a], ytr[a])
        oof[b] = m.predict(Xtr[b])
    full = model_cat(seed).fit(Xtr, ytr)
    return oof, full.predict(Xv)


def kepala_total(y, pred_kelas):
    ranking = []
    for nama, p in pred_kelas.items():
        kal = CD.cari_kalibrasi_total(y.sum(1), p.sum(1))
        ranking.append((kal[0], nama, kal[1], kal[2], kal[3]))
    ranking.sort()
    return ranking


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", action="append", default=[])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--linker-config", type=Path, default=SUB / "results" /
                    "damimas_linker_global_proposal_yolo_lock.json")
    ap.add_argument("--linker-model", type=Path, default=SUB / "runs" /
                    "linker_global_damimas_proposal_yolo" / "model.joblib")
    ap.add_argument("--cache-train", type=Path, default=SUB / "results" /
                    "cache_linker_damimas_damimas_damimas_proposal_base_train_train.joblib")
    ap.add_argument("--cache-val", type=Path, default=SUB / "results" /
                    "cache_linker_damimas_damimas_damimas_proposal_yolo_val_val.joblib")
    ap.add_argument("--cache-test", type=Path, default=SUB / "results" /
                    "cache_linker_damimas_damimas_damimas_proposal_yolo_test_test.joblib")
    ap.add_argument("--output", type=Path, default=SUB / "results" /
                    "damimas_counting_catboost.json")
    ap.add_argument("--pred-out", type=Path, default=SUB / "results" /
                    "damimas_counting_catboost_pred.npz")
    ap.add_argument("--model-out", type=Path, default=SUB / "runs" /
                    "counting_catboost_damimas" / "ensemble.joblib")
    args = ap.parse_args()
    if args.folds < 2:
        raise ValueError("--folds harus >= 2")

    banks = CM.parse_bank(args.bank) if args.bank else {
        "anchor": {s: SUB / "results" / f"pred_skorpenuh_{s}.npz"
                   for s in ("train", "val", "test")},
        "proposal": {
            "train": ROOT / "results" / "pred_damimas_proposal_base_train.npz",
            "val": ROOT / "results" / "pred_damimas_proposal_yolo_val.npz",
            "test": ROOT / "results" / "pred_damimas_proposal_yolo_test.npz",
        },
    }
    caches = {"train": args.cache_train, "val": args.cache_val,
              "test": args.cache_test}

    # TRAIN dan VAL saja sampai lock.
    ids, y, X = {}, {}, {}
    for split in ("train", "val"):
        ids[split], y[split], X[split] = CM.muat_split(
            split, banks, caches, args.linker_model, args.linker_config)
    views = tuple(X["train"])
    if tuple(X["val"]) != views:
        raise RuntimeError("Ruang fitur TRAIN/VAL berbeda")

    pred_train, pred_val = {}, {}
    for i, view in enumerate(views):
        nama = f"cat_{view}"
        print(f"CatBoost {nama}: {X['train'][view].shape[1]} fitur", flush=True)
        pred_train[nama], pred_val[nama] = oof_dan_val(
            X["train"][view], y["train"], X["val"][view],
            args.folds, args.seed + 100 * i)

    ysel = np.concatenate([y["train"], y["val"]])
    pred_sel = {n: np.concatenate([pred_train[n], pred_val[n]])
                for n in pred_train}
    kepala = CD.pilih_per_kelas(ysel, pred_sel)
    psel = CD.prediksi_kepala(kepala, pred_sel)
    pval = CD.prediksi_kepala(kepala, pred_val)

    rank_total = kepala_total(ysel, {**pred_sel, "kepala": psel})
    _, nama_total, a_total, b_total, met_total = rank_total[0]
    total_sel_raw = psel.sum(1) if nama_total == "kepala" \
        else pred_sel[nama_total].sum(1)
    total_sel = a_total * total_sel_raw + b_total
    rek = CD.pilih_rekonsiliasi(ysel, psel, total_sel)
    pfinal_sel = CD.terapkan_rekonsiliasi(psel, total_sel, rek)

    total_val_raw = pval.sum(1) if nama_total == "kepala" \
        else pred_val[nama_total].sum(1)
    total_val = a_total * total_val_raw + b_total
    pfinal_val = CD.terapkan_rekonsiliasi(pval, total_val, rek)
    met_total_val_saja = CD.metrik_total(y["val"].sum(1), total_val)
    lock = {
        "resep": {"iterations": 500, "depth": 5, "learning_rate": .035,
                  "l2_leaf_reg": 20., "rsm": .25, "folds": args.folds},
        "kepala_per_kelas": CD.serial_kepala(kepala),
        "kepala_total": {"nama": nama_total, "skala": a_total,
                          "bias": b_total, "metrik_seleksi": met_total,
                          "metrik_val_saja": met_total_val_saja},
        "rekonsiliasi": {"mode": rek[1], "beta": rek[2],
                          "alokasi": None if rek[3] is None else rek[3].tolist(),
                          "proyeksi": rek[4]},
        "metrik_oof_train_plus_val": CD.metrik(ysel, pfinal_sel),
        "metrik_val_saja": CD.metrik(y["val"], pfinal_val),
    }
    print("TERKUNCI OOF+VAL", json.dumps(lock, indent=2,
                                          ensure_ascii=False), flush=True)

    # TEST baru dibuka setelah seluruh keputusan tetap.
    ids["test"], y["test"], X["test"] = CM.muat_split(
        "test", banks, caches, args.linker_model, args.linker_config)
    Xtv = {v: np.concatenate([X["train"][v], X["val"][v]]) for v in views}
    ytv = np.concatenate([y["train"], y["val"]])
    dipakai = sorted({n for h in kepala for n in h.anggota} |
                     ({nama_total} if nama_total != "kepala" else set()))
    fitted, pred_test = {}, {}
    for j, nama in enumerate(dipakai):
        view = nama.removeprefix("cat_")
        print(f"refit {nama} TRAIN+VAL", flush=True)
        fitted[nama] = model_cat(args.seed + 1000 + j).fit(Xtv[view], ytv)
        pred_test[nama] = fitted[nama].predict(X["test"][view])
    ptest = CD.prediksi_kepala(kepala, pred_test)
    total_test_raw = ptest.sum(1) if nama_total == "kepala" \
        else pred_test[nama_total].sum(1)
    total_test = a_total * total_test_raw + b_total
    pfinal_test = CD.terapkan_rekonsiliasi(ptest, total_test, rek)

    hasil = {
        "dataset": "SawitMVC-YOLO-Damimas",
        "protokol": ("resep tetap; kepala dipilih pada OOF TRAIN + VAL bersih; "
                     "TEST dibuka setelah lock"),
        "bank": {n: {s: str(p) for s, p in q.items()} for n, q in banks.items()},
        "fitur": {s: {n: int(v.shape[1]) for n, v in X[s].items()} for s in X},
        "terkunci": lock,
        "test": CD.metrik(y["test"], pfinal_test),
        "test_sebelum_rekonsiliasi": CD.metrik(y["test"], ptest),
        "kepala_total_terpisah": {
            "definisi": "regresor langsung untuk jumlah B1+B2+B3+B4",
            "val": met_total_val_saja,
            "test": CD.metrik_total(y["test"].sum(1), total_test),
        },
        "catatan_total_mae": (
            "test.total_mae adalah MAE jumlah empat kepala kelas; ketika "
            "rekonsiliasi mode raw, ia bukan metrik kepala total terpisah"),
        "ranking_total_seleksi": [
            {"nama": r[1], "skala": r[2], "bias": r[3], "metrik": r[4]}
            for r in rank_total],
    }
    args.pred_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.pred_out,
        val_id=np.asarray(ids["val"]), val_y=y["val"],
        val_pred_raw=pval, val_pred_final=pfinal_val, val_total=total_val,
        test_id=np.asarray(ids["test"]), test_y=y["test"],
        test_pred_raw=ptest, test_pred_final=pfinal_test,
        test_total=total_test,
    )
    hasil["prediksi_per_pohon"] = str(args.pred_out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": fitted, "kepala": CD.serial_kepala(kepala),
                 "total": (nama_total, a_total, b_total),
                 "rekonsiliasi": lock["rekonsiliasi"], "views": views,
                 "bank": hasil["bank"]}, args.model_out, compress=3)
    print(json.dumps({"val": lock["metrik_val_saja"],
                      "test": hasil["test"]}, indent=2,
                     ensure_ascii=False), flush=True)
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
