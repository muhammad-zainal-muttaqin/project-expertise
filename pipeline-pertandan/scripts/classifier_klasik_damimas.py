"""Classifier piksel + C1 yang benar-benar dilatih hanya pada DAMIMAS.

Eksperimen C2/C3 lama dilatih pada dua varietas lalu baru difilter di stacker.
Skrip ini memanfaatkan crop GT dan tautan GT yang sudah ada, tetapi seluruh
proses fitting hanya memakai pohon DAMIMAS. Tujuannya adalah membuat anggota
ensemble yang murah dan komplementer sebelum classifier visual besar dilatih.

Protokol ketatnya adalah TRAIN -> pilih model/blend/ambang di VAL -> refit
TRAIN+VAL -> buka TEST satu kali. Metrik per-view dan per-tandan sama-sama
disimpan. Karena memakai kotak dan tautan GT, hasil per-tandan adalah metrik
modul/oracle, bukan klaim end-to-end.
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SUB = Path(__file__).resolve().parents[1]
KELAS = ("B1", "B2", "B3", "B4")
sys.path.insert(0, str(Path(__file__).parent))
import c3_multitampak as C3M  # noqa: E402
import penaut_pertandan as PP  # noqa: E402
import stacker_damimas as SD  # noqa: E402


def spesifikasi_model() -> dict[str, tuple[str, object]]:
    return {
        "mc_logreg_c01": ("mc", make_pipeline(
            StandardScaler(), LogisticRegression(
                C=.1, class_weight="balanced", max_iter=3000, random_state=0))),
        "mc_logreg_c1": ("mc", make_pipeline(
            StandardScaler(), LogisticRegression(
                C=1., class_weight="balanced", max_iter=3000, random_state=0))),
        "mc_extra_l2": ("mc", ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, max_features=.6,
            class_weight="balanced", n_jobs=-1, random_state=0)),
        "mc_extra_l6": ("mc", ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=6, max_features=.8,
            class_weight="balanced", n_jobs=-1, random_state=1)),
        "mc_hist_l15": ("mc", HistGradientBoostingClassifier(
            learning_rate=.05, max_iter=250, max_leaf_nodes=15,
            l2_regularization=5., class_weight="balanced", random_state=0)),
        "mc_hist_l31": ("mc", HistGradientBoostingClassifier(
            learning_rate=.04, max_iter=300, max_leaf_nodes=31,
            l2_regularization=10., class_weight="balanced", random_state=1)),
        "ord_extra_l4": ("ordinal", ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=4, max_features=.7,
            class_weight="balanced", n_jobs=-1, random_state=2)),
        "ord_hist_l15": ("ordinal", HistGradientBoostingClassifier(
            learning_rate=.05, max_iter=250, max_leaf_nodes=15,
            l2_regularization=8., class_weight="balanced", random_state=2)),
    }


def pasang(spec: tuple[str, object], X: np.ndarray, y: np.ndarray) -> dict:
    jenis, dasar = spec
    if jenis == "mc":
        return {"jenis": jenis, "model": clone(dasar).fit(X, y)}
    return {"jenis": jenis,
            "model": [clone(dasar).fit(X, y > k) for k in range(3)]}


def prob_model(model: dict, X: np.ndarray) -> np.ndarray:
    if model["jenis"] == "mc":
        m = model["model"]
        q = np.zeros((len(X), 4), float)
        p = m.predict_proba(X)
        for j, c in enumerate(m.classes_):
            q[:, int(c)] = p[:, j]
        return q
    kum = []
    for m in model["model"]:
        p = m.predict_proba(X)
        j = int(np.flatnonzero(m.classes_ == 1)[0])
        kum.append(p[:, j])
    c = np.minimum.accumulate(np.stack(kum, 1), axis=1)
    q = np.c_[1 - c[:, 0], c[:, 0] - c[:, 1], c[:, 1] - c[:, 2], c[:, 2]]
    return np.clip(q, 0, None) / np.maximum(q.sum(1, keepdims=True), 1e-9)


def pilih_blend(kandidat: dict[str, np.ndarray], y: np.ndarray):
    ranking = []
    for nama, p in kandidat.items():
        q = SD.pilih_aturan(p, y)
        ranking.append((q[0], q[1], nama, q, p))
    ranking.sort(key=lambda x: (x[0], x[1]), reverse=True)
    bobot = {ranking[0][2]: 1.0}
    best, pbest = ranking[0][3], ranking[0][4]
    while len(bobot) < 5:
        calon = []
        for nama, p in kandidat.items():
            if nama in bobot:
                continue
            for w in (.10, .20, .30, .40, .50):
                pb = (1 - w) * pbest + w * p
                q = SD.pilih_aturan(pb, y)
                calon.append((q[0], q[1], nama, w, q, pb))
        if not calon:
            break
        c = max(calon, key=lambda x: (x[0], x[1]))
        if c[0] <= best[0] + 1e-9:
            break
        bobot = {n: (1 - c[3]) * w for n, w in bobot.items()}
        bobot[c[2]] = c[3]
        best, pbest = c[4], c[5]
    return bobot, best, pbest, ranking


def gabung_prob(bobot: dict[str, float], kandidat: dict[str, np.ndarray]) -> np.ndarray:
    return sum(w * kandidat[n] for n, w in bobot.items())


def metadata(ids: dict[str, list[str]]) -> dict[str, np.ndarray]:
    out = {}
    for daftar in ids.values():
        for tree in daftar:
            nv, kotak = PP.muat_pohon(tree)
            for b in kotak:
                area = b["w"] * b["h"]
                out[f"{tree}|{b['s']}|{b['i']}"] = np.array([
                    b["s"] / max(nv - 1, 1), nv / 8., b["cx"], b["cy"],
                    b["w"], b["h"], area, b["w"] / max(b["h"], 1e-6),
                    min(b["cx"], 1 - b["cx"], b["cy"], 1 - b["cy"]),
                ], np.float32)
    return out


def muat_fitur(cache: Path):
    man = PP.muat_manifest()
    ids = {s: sorted(t for t, v in man.items()
                     if v == s and t.startswith("DAMIMAS_"))
           for s in ("train", "val", "test")}
    zc = np.load(SUB / "results" / "potongan_reid.npz", allow_pickle=True)
    kunci_semua = zc["kunci"].astype(str)
    zc.close()
    pos = {k: i for i, k in enumerate(kunci_semua)}
    data = C3M.siapkan(ids, None, list(kunci_semua), man)
    perlu = np.array(sorted({i for s in data for b in data[s] for i in b["idx"]}), int)
    keys = kunci_semua[perlu]

    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if not np.array_equal(z["kunci"].astype(str), keys):
            raise RuntimeError("Cache descriptor DAMIMAS tidak sejajar dengan crop")
        desc = z["descriptor"].astype(np.float32)
        z.close()
    else:
        zd = np.load(SUB / "results" / "deskriptor_crop.npz", allow_pickle=True)
        desc = np.stack([zd[k] for k in keys]).astype(np.float32)
        zd.close()
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, kunci=keys, descriptor=desc)

    PP.TAG = ""
    peta_p = PP.bangun_prob_prediksi(ids)
    geo = metadata(ids)
    p = np.stack([peta_p.get(k, np.full(4, .25)) for k in keys]).astype(np.float32)
    p /= np.maximum(p.sum(1, keepdims=True), 1e-9)
    eks = p @ np.arange(4)
    ent = -(p * np.log(np.clip(p, 1e-9, 1))).sum(1)
    urut = np.sort(p, axis=1)
    g = np.stack([geo[k] for k in keys])
    X = np.c_[desc, p, eks, ent, p.max(1), urut[:, -1] - urut[:, -2], g]

    global_ke_lokal = np.full(len(kunci_semua), -1, int)
    global_ke_lokal[perlu] = np.arange(len(perlu))
    for s in data:
        for b in data[s]:
            b["idx"] = global_ke_lokal[np.asarray(b["idx"], int)].tolist()
            if min(b["idx"], default=0) < 0:
                raise RuntimeError("Indeks crop hilang saat pemetaan DAMIMAS")
    return X.astype(np.float32), p, keys, data


def data_view(data_split: list[dict]):
    idx, y, tree, bunch = [], [], [], []
    for bi, b in enumerate(data_split):
        idx.extend(b["idx"]); y.extend([b["y"]] * len(b["idx"]))
        tree.extend([b["tree"]] * len(b["idx"])); bunch.extend([bi] * len(b["idx"]))
    return (np.asarray(idx, int), np.asarray(y, int), np.asarray(tree),
            np.asarray(bunch, int))


def fitur_bunch(X: np.ndarray, P: np.ndarray, data_split: list[dict]) -> np.ndarray:
    out = []
    for b in data_split:
        v = X[np.asarray(b["idx"], int)]
        p = P[np.asarray(b["idx"], int)]
        n = len(v)
        conf = p.max(1)
        pc = np.average(p, axis=0, weights=np.maximum(conf, 1e-6))
        out.append(np.r_[v.mean(0), v.std(0), v.min(0), v.max(0),
                         np.quantile(v, .25, axis=0), np.quantile(v, .75, axis=0),
                         p.mean(0), pc, p.max(0), p.min(0),
                         n, np.log1p(n), *[int(n == q) for q in range(1, 7)]])
    return np.asarray(out, np.float32)


def agregasi_view(Pview: np.ndarray, data_split: list[dict]) -> np.ndarray:
    out, awal = [], 0
    for b in data_split:
        q = Pview[awal:awal + len(b["idx"])]
        awal += len(b["idx"])
        out.append(np.average(q, axis=0, weights=np.maximum(q.max(1), 1e-6)))
    return np.asarray(out)


def serial_ranking(ranking):
    return [{"nama": r[2], "aturan": r[3][2], "tau": r[3][3],
             "metrik": r[3][4]} for r in ranking]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path,
                    default=SUB / "results" / "damimas_deskriptor_matrix.npz")
    ap.add_argument("--keluaran", type=Path,
                    default=SUB / "results" / "damimas_classifier_klasik.json")
    ap.add_argument("--pred-out", type=Path,
                    default=SUB / "results" / "damimas_classifier_klasik_pred.npz")
    ap.add_argument("--model-out", type=Path,
                    default=SUB / "runs" / "classifier_klasik_damimas" / "model.joblib")
    args = ap.parse_args()

    X, P, keys, data = muat_fitur(args.cache)
    dv = {s: data_view(data[s]) for s in data}
    Xb = {s: fitur_bunch(X, P, data[s]) for s in data}
    yb = {s: np.asarray([b["y"] for b in data[s]], int) for s in data}
    specs = spesifikasi_model()

    # Tahap seleksi murni TRAIN -> VAL.
    model_view, pv_val = {}, {"C1": P[dv["val"][0]]}
    for nama, spec in specs.items():
        print(f"fit view {nama}", flush=True)
        model_view[nama] = pasang(spec, X[dv["train"][0]], dv["train"][1])
        pv_val[nama] = prob_model(model_view[nama], X[dv["val"][0]])
    wv, pilih_v, pview_val, rank_v = pilih_blend(pv_val, dv["val"][1])
    print(f"view terpilih {wv} {pilih_v[2:4]}", flush=True)

    model_bunch, pb_val = {
    }, {"C1_mean": agregasi_view(P[dv["val"][0]], data["val"]),
        "view_ensemble": agregasi_view(pview_val, data["val"])}
    for nama, spec in specs.items():
        print(f"fit bunch {nama}", flush=True)
        model_bunch[nama] = pasang(spec, Xb["train"], yb["train"])
        pb_val[nama] = prob_model(model_bunch[nama], Xb["val"])
    wb, pilih_b, pbunch_val, rank_b = pilih_blend(pb_val, yb["val"])
    print(f"bunch terpilih {wb} {pilih_b[2:4]}", flush=True)

    # Konfigurasi terkunci. Refit hanya anggota yang terpakai pada TRAIN+VAL.
    Xtv = np.concatenate([X[dv["train"][0]], X[dv["val"][0]]])
    ytv = np.concatenate([dv["train"][1], dv["val"][1]])
    final_view = {}
    pv_test = {"C1": P[dv["test"][0]]}
    for nama in wv:
        if nama == "C1":
            continue
        final_view[nama] = pasang(specs[nama], Xtv, ytv)
        pv_test[nama] = prob_model(final_view[nama], X[dv["test"][0]])
    pview_test = gabung_prob(wv, pv_test)

    Xbtv = np.concatenate([Xb["train"], Xb["val"]])
    ybtv = np.concatenate([yb["train"], yb["val"]])
    final_bunch = {}
    pb_test = {
        "C1_mean": agregasi_view(P[dv["test"][0]], data["test"]),
        "view_ensemble": agregasi_view(pview_test, data["test"]),
    }
    for nama in wb:
        if nama in pb_test:
            continue
        final_bunch[nama] = pasang(specs[nama], Xbtv, ybtv)
        pb_test[nama] = prob_model(final_bunch[nama], Xb["test"])
    pbunch_test = gabung_prob(wb, pb_test)

    aturan_v, tau_v = pilih_v[2], pilih_v[3]
    aturan_b, tau_b = pilih_b[2], pilih_b[3]
    yhv_val = SD.prediksi_prob(pview_val, aturan_v, tau_v)
    yhv_test = SD.prediksi_prob(pview_test, aturan_v, tau_v)
    yhb_val = SD.prediksi_prob(pbunch_val, aturan_b, tau_b)
    yhb_test = SD.prediksi_prob(pbunch_test, aturan_b, tau_b)
    hasil = {
        "dataset": "SawitMVC-YOLO-Damimas",
        "protokol": "fit TRAIN; pilih model/blend/tau VAL; refit TRAIN+VAL; TEST sekali",
        "kaveat": "crop GT dan tautan GT: metrik classifier modul, bukan end-to-end",
        "n": {s: {"view": int(len(dv[s][1])), "tandan": int(len(yb[s])),
                    "pohon": int(len(set(dv[s][2])))} for s in data},
        "fitur": {"per_view": int(X.shape[1]), "per_tandan": int(Xb["train"].shape[1])},
        "per_view": {
            "bobot": wv, "aturan": aturan_v, "tau": tau_v,
            "val": SD.metrik(dv["val"][1], yhv_val),
            "test": SD.metrik(dv["test"][1], yhv_test),
            "ranking_val": serial_ranking(rank_v),
        },
        "per_tandan": {
            "bobot": wb, "aturan": aturan_b, "tau": tau_b,
            "val": SD.metrik(yb["val"], yhb_val),
            "test": SD.metrik(yb["test"], yhb_test),
            "test_subgrup": SD.metrik_subgrup(
                yb["test"], yhb_test,
                np.r_[0, np.cumsum([len(b["idx"]) for b in data["test"]])]),
            "ranking_val": serial_ranking(rank_b),
        },
    }
    args.keluaran.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
    args.pred_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.pred_out,
        val_view_prob=pview_val, val_view_y=dv["val"][1],
        val_view_tree=dv["val"][2], val_view_key=keys[dv["val"][0]],
        test_view_prob=pview_test, test_view_y=dv["test"][1],
        test_view_tree=dv["test"][2], test_view_key=keys[dv["test"][0]],
        val_bunch_prob=pbunch_val, val_bunch_y=yb["val"],
        val_bunch_tree=np.asarray([b["tree"] for b in data["val"]]),
        test_bunch_prob=pbunch_test, test_bunch_y=yb["test"],
        test_bunch_tree=np.asarray([b["tree"] for b in data["test"]]),
        test_bunch_nview=np.asarray([len(b["idx"]) for b in data["test"]]),
    )
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"view": final_view, "bunch": final_bunch,
                 "view_bobot": wv, "view_aturan": aturan_v, "view_tau": tau_v,
                 "bunch_bobot": wb, "bunch_aturan": aturan_b, "bunch_tau": tau_b,
                 "fitur_view": X.shape[1], "fitur_bunch": Xb["train"].shape[1]},
                args.model_out)
    print(json.dumps({"per_view": hasil["per_view"]["test"],
                      "per_tandan": hasil["per_tandan"]["test"]},
                     indent=2, ensure_ascii=False))
    print(f"-> {args.keluaran}")


if __name__ == "__main__":
    main()
