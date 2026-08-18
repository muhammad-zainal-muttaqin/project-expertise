"""Counting DAMIMAS dari multi-bank deteksi dan statistik linker.

Kepala lama hanya melihat dump anchor YOLO. Versi ini membandingkan empat ruang
fitur: anchor penuh, proposal fisik, gabungannya, dan gabungan + ringkasan
klaster linker. Ringkasan linker tidak pernah membaca ``bid``/kelas GT yang ada
di cache evaluasi.

Urutan aksesnya ketat: fitur/label TRAIN+VAL dibangun, seluruh model, kalibrasi,
dan rekonsiliasi dikunci di VAL, baru cache/prediksi/label TEST dibuka.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression


SUB = Path(__file__).resolve().parents[1]
ROOT = SUB.parent
sys.path.insert(0, str(Path(__file__).parent))
import counting_damimas as CD  # noqa: E402
import linker_global_damimas as LG  # noqa: E402


def parse_bank(items):
    out = {}
    for item in items:
        p = item.split("=")
        if len(p) != 4:
            raise ValueError("--bank harus NAMA=TRAIN.npz=VAL.npz=TEST.npz")
        nama, tr, va, te = p
        out[nama] = {"train": Path(tr), "val": Path(va), "test": Path(te)}
    return out


def ids_y(split):
    ids, y = [], []
    with (CD.DS / "split_manifest.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["new_split"] != split:
                continue
            if r["variety"] != "DAMIMAS":
                raise RuntimeError(f"Split {split} tercemar: {r['tree_id']}")
            ids.append(r["tree_id"])
            y.append([int(r[c]) for c in CD.KELAS])
    order = np.argsort(ids)
    return [ids[i] for i in order], np.asarray(y, float)[order]


def stats(x):
    return CD.statistik(np.asarray(x, float))


def rakit(g, score, q):
    if q["assembler"] == "hungarian":
        return LG.rakit_hungarian(g, score, q["ambang"], q["max_mode"])
    if q["assembler"] == "ilp":
        return LG.rakit_ilp(g, score, q["ambang"], q["metode"], q["max_mode"])
    return LG.rakit_aglom(g, score, q["ambang"], q["metode"], q["max_mode"])


def fitur_graf(g, bundle, heads):
    model_names = sorted(bundle["models"])
    # Panjang dibuat eksplisit agar pohon tanpa graf mendapat vektor identik.
    # Tiap head: 3 statistik jumlah, 4x8 statistik struktur, lalu 24 fitur
    # kelas cluster (hard/soft count + confidence/entropy). Seluruhnya berasal
    # dari probabilitas prediksi, tidak pernah dari ``bid``/kelas GT cache.
    dim = (5 + 4 * 8 + 8 + len(model_names) * 8
           + len(heads) * (3 + 4 * 8 + 24))
    if g is None:
        return np.zeros(dim, np.float32)
    det = g["kotak"]
    conf = np.asarray([d["conf"] for d in det], float)
    p = np.stack([d["p"] for d in det])
    eks = p @ np.arange(4)
    ent = -(p * np.log(np.clip(p, 1e-9, 1))).sum(1)
    sisi = np.asarray([d["s"] for d in det], int)
    f = [len(det), len(np.unique(sisi)), g["nv"],
         len(g["pairs"]), len(det) / max(g["nv"], 1)]
    for x in (conf, eks, ent, np.asarray([d["luas"] for d in det])):
        f += stats(x)
    f += stats(np.bincount(sisi, minlength=g["nv"]))
    score_model = {}
    for nama in model_names:
        s = bundle["models"][nama].predict_proba(g["E"])[:, 1]
        score_model[nama] = s; f += stats(s)
    for q in heads.values():
        s = sum(w * score_model[n] for n, w in q["bobot_skor"].items())
        lab = rakit(g, s, q)
        grup = defaultdict(list)
        for i, k in enumerate(lab):
            grup[int(k)].append(i)
        ukuran = np.asarray([len(v) for v in grup.values()], float)
        f += [len(grup), float((ukuran >= 2).sum()),
              float((ukuran >= 2).mean()) if len(ukuran) else 0.]
        f += stats(ukuran)
        f += stats([conf[v].mean() for v in grup.values()])
        f += stats([eks[v].mean() for v in grup.values()])
        f += stats([len(np.unique(sisi[v])) for v in grup.values()])
        Q = []
        for v in grup.values():
            ww = np.maximum(conf[v], 1e-6)
            qq = np.average(p[v], axis=0, weights=ww)
            Q.append(qq / max(float(qq.sum()), 1e-9))
        Q = np.stack(Q) if Q else np.zeros((0, 4), float)
        if len(Q):
            f += np.bincount(Q.argmax(1), minlength=4).astype(float).tolist()
            f += Q.sum(0).tolist()
            f += stats(Q.max(1))
            f += stats(-(Q * np.log(np.clip(Q, 1e-9, 1))).sum(1))
        else:
            f += [0.] * 24
    out = np.asarray(f, np.float32)
    if len(out) != dim:
        raise RuntimeError(f"Dimensi fitur linker berubah: {len(out)} != {dim}")
    return out


def linker_map(cache_path, model_path, config_path):
    config = json.loads(config_path.read_text())
    bundle = joblib.load(model_path)
    heads = config["heads"]
    graphs = joblib.load(cache_path)["graf"]
    return {g["tree"]: fitur_graf(g, bundle, heads) for g in graphs}, (
        len(fitur_graf(None, bundle, heads)))


def muat_split(split, banks, cache, model_linker, hasil_linker):
    ids, y = ids_y(split)
    bagian = {}
    for nama, path in banks.items():
        z = np.load(path[split], allow_pickle=True)
        bagian[nama] = np.stack([CD.fitur_pohon(t, z) for t in ids])
        z.close()
    lmap, ldim = linker_map(cache[split], model_linker, hasil_linker)
    L = np.stack([lmap.get(t, np.zeros(ldim, np.float32)) for t in ids])
    names = list(banks)
    if not names:
        raise RuntimeError("Sedikitnya satu bank prediksi diperlukan")
    gabung = np.concatenate([bagian[n] for n in names], axis=1)
    # Setiap bank tetap kandidat mandiri. ``concat`` memuat seluruh bank
    # berurutan dan ``concat_linker`` menambahkan statistik cluster. Dengan
    # demikian RF-DETR/RT-DETR dapat ditambahkan tanpa membuang baseline anchor.
    views = {n: bagian[n] for n in names}
    views.update({"concat": gabung, "concat_linker": np.c_[gabung, L]})
    print(f"{split}: pohon={len(ids)} dim=" +
          ", ".join(f"{k}:{v.shape[1]}" for k, v in views.items()), flush=True)
    return ids, y, views


def spesifikasi_kelas(views):
    base = CD.kandidat()
    out = {}
    # Pencarian penuh: setiap keluarga baseline dijalankan pada setiap ruang
    # fitur. Ini menjamin anchor lama menjadi kandidat persis, bukan baseline
    # yang diam-diam hilang karena penghematan waktu.
    for view in views:
        for nama in base:
            out[f"{view}__{nama}"] = (view, base[nama])
        for ncomp in (8, 16):
            out[f"{view}__pls{ncomp}"] = (
                view, PLSRegression(n_components=ncomp, scale=True,
                                    max_iter=1000, tol=1e-6))
    return out


def spesifikasi_total(views):
    base = CD.kandidat_total()
    out = {}
    for view in views:
        for nama in base:
            out[f"{view}__{nama}"] = (view, base[nama])
    return out


def prediksi_kepala(kepala, pred):
    return CD.prediksi_kepala(kepala, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", action="append", default=[])
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
                    "damimas_counting_multibank.json")
    ap.add_argument("--model-out", type=Path, default=SUB / "runs" /
                    "counting_multibank_damimas" / "ensemble.joblib")
    ap.add_argument("--pred-out", type=Path, default=SUB / "results" /
                    "damimas_counting_multibank_pred.npz")
    args = ap.parse_args()
    banks = parse_bank(args.bank) if args.bank else {
        "anchor": {s: SUB / "results" / f"pred_skorpenuh_{s}.npz"
                   for s in ("train", "val", "test")},
        "proposal": {"train": ROOT / "results" / "pred_damimas_proposal_base_train.npz",
                     "val": ROOT / "results" / "pred_damimas_proposal_yolo_val.npz",
                     "test": ROOT / "results" / "pred_damimas_proposal_yolo_test.npz"},
    }
    caches = {"train": args.cache_train, "val": args.cache_val,
              "test": args.cache_test}

    # Fase seleksi: tidak ada path/label TEST yang dibuka di sini.
    ids, y, X = {}, {}, {}
    for split in ("train", "val"):
        ids[split], y[split], X[split] = muat_split(
            split, banks, caches, args.linker_model, args.linker_config)
    views = tuple(X["train"])
    if tuple(X["val"]) != views:
        raise RuntimeError("Ruang fitur TRAIN dan VAL tidak identik")
    specs = spesifikasi_kelas(views); pred_val = {}; ranking = []
    for nama, (view, model) in specs.items():
        m = clone(model).fit(X["train"][view], y["train"])
        pred_val[nama] = m.predict(X["val"][view])
        mm = CD.metrik(y["val"], pred_val[nama])
        ranking.append((mm["macro_mae"], -mm["class_pm1_acc"], nama, mm))
        print(f"{nama:40s} val MAE={mm['macro_mae']:.4f} "
              f"tree+-1={mm['tree_pm1_acc']:.4f}", flush=True)
    ranking.sort()
    kepala = CD.pilih_per_kelas(y["val"], pred_val)
    pval = prediksi_kepala(kepala, pred_val)

    total_specs = spesifikasi_total(views); rank_total = []
    for nama, (view, model) in total_specs.items():
        m = clone(model).fit(X["train"][view], y["train"].sum(1))
        pv = m.predict(X["val"][view])
        kal = CD.cari_kalibrasi_total(y["val"].sum(1), pv)
        rank_total.append((kal[0], nama, kal[1], kal[2], kal[3]))
        print(f"{nama:40s} val total MAE={kal[3]['mae']:.4f} "
              f"+-1={kal[3]['pm1_acc']:.4f}", flush=True)
    rank_total.sort()
    _, nama_total, a_total, b_total, met_total_val = rank_total[0]
    view_total, spec_total = total_specs[nama_total]
    mtv = clone(spec_total).fit(X["train"][view_total], y["train"].sum(1))
    total_val = a_total * mtv.predict(X["val"][view_total]) + b_total
    rek = CD.pilih_rekonsiliasi(y["val"], pval, total_val)
    pfinal_val = CD.terapkan_rekonsiliasi(pval, total_val, rek)
    lock = {"kepala_per_kelas": CD.serial_kepala(kepala),
            "kepala_total": {"nama": nama_total, "skala": a_total,
                              "bias": b_total,
                              "metrik_val_saja": met_total_val},
            "rekonsiliasi": {"mode": rek[1], "beta": rek[2],
                              "alokasi": None if rek[3] is None else rek[3].tolist(),
                              "proyeksi": rek[4]},
            "metrik_val": CD.metrik(y["val"], pfinal_val)}
    print("TERKUNCI DI VAL", json.dumps(lock, indent=2), flush=True)

    # Baru sekarang buka TEST dan refit TRAIN+VAL.
    ids["test"], y["test"], X["test"] = muat_split(
        "test", banks, caches, args.linker_model, args.linker_config)
    dipakai = sorted({n for h in kepala for n in h.anggota})
    fitted, pred_test = {}, {}
    for nama in dipakai:
        view, spec = specs[nama]
        print(f"refit {nama} pada TRAIN+VAL", flush=True)
        fitted[nama] = clone(spec).fit(
            np.concatenate([X["train"][view], X["val"][view]]),
            np.concatenate([y["train"], y["val"]]))
        pred_test[nama] = fitted[nama].predict(X["test"][view])
    ptest = prediksi_kepala(kepala, pred_test)
    model_total = clone(spec_total).fit(
        np.concatenate([X["train"][view_total], X["val"][view_total]]),
        np.concatenate([y["train"], y["val"]]).sum(1))
    total_test = a_total * model_total.predict(X["test"][view_total]) + b_total
    pfinal_test = CD.terapkan_rekonsiliasi(ptest, total_test, rek)
    hasil = {"dataset": "SawitMVC-YOLO-Damimas",
             "protokol": "fit TRAIN; semua seleksi VAL; TEST dibuka setelah lock",
             "bank": {n: {s: str(p) for s, p in q.items()} for n, q in banks.items()},
             "fitur": {s: {n: int(v.shape[1]) for n, v in X[s].items()} for s in X},
             "terkunci_di_val": lock,
             "test": CD.metrik(y["test"], pfinal_test),
             "test_sebelum_rekonsiliasi": CD.metrik(y["test"], ptest),
             "kepala_total_terpisah": {
                 "definisi": "regresor langsung untuk jumlah B1+B2+B3+B4",
                 "val": met_total_val,
                 "test": CD.metrik_total(y["test"].sum(1), total_test)},
             "catatan_total_mae": (
                 "test.total_mae adalah MAE jumlah empat kepala kelas; ketika "
                 "rekonsiliasi mode raw, ia bukan metrik kepala total terpisah"),
             "ranking_val": [{"model": r[2], "metrik": r[3]} for r in ranking],
             "ranking_total_val": [{"model": r[1], "skala": r[2],
                                     "bias": r[3], "metrik": r[4]}
                                    for r in rank_total]}
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
    args.output.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": fitted, "kepala": CD.serial_kepala(kepala),
                 "model_total": model_total, "total": (a_total, b_total),
                 "rekonsiliasi": lock["rekonsiliasi"],
                 "feature_views": {n: specs[n][0] for n in dipakai},
                 "fitur_dim": {n: X["train"][n].shape[1] for n in X["train"]},
                 "bank": {n: {s: str(p) for s, p in q.items()}
                          for n, q in banks.items()}},
                args.model_out, compress=3)
    print(json.dumps({"val": lock["metrik_val"], "test": hasil["test"]},
                     indent=2, ensure_ascii=False))
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
