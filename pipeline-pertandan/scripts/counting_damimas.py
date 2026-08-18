"""Penghitung per-pohon khusus DAMIMAS dari dump skor detektor penuh.

Skrip ini sengaja memisahkan tiga tahap data:

1. model kandidat dipasang pada TRAIN DAMIMAS;
2. pemilihan model, ensemble, dan kalibrasi bilangan bulat hanya memakai VAL;
3. konfigurasi yang sudah terkunci dipasang ulang pada TRAIN+VAL dan TEST
   dibuka satu kali.

Fitur tidak bergantung pada satu ambang deteksi. Setiap pohon diringkas pada
beberapa ambang keyakinan, per sisi dan per kelas, ditambah statistik softmax,
margin, entropi, posisi, dan luas. Ini membuat counting menjadi kepala khusus;
jumlah pool hasil linker tidak dipaksa menjadi estimasi jumlah tandan.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


SUB = Path(__file__).resolve().parents[1]
ROOT = SUB.parent
DS = Path("/workspace/SawitMVC-YOLO-Damimas")
DS_ASLI = Path("/workspace/SawitMVC-YOLO")
KELAS = ("B1", "B2", "B3", "B4")
AMBANG = (0.01, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25,
          0.30, 0.40, 0.50, 0.60)


def statistik(x: np.ndarray) -> list[float]:
    if len(x) == 0:
        return [0.0] * 8
    return [float(x.sum()), float(x.mean()), float(x.std()), float(x.min()),
            float(np.quantile(x, 0.25)), float(np.median(x)),
            float(np.quantile(x, 0.75)), float(x.max())]


def baca_manifest() -> tuple[dict[str, str], dict[str, np.ndarray]]:
    split, gt = {}, {}
    with (DS / "split_manifest.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            t = r["tree_id"]
            if r["variety"] != "DAMIMAS":
                raise RuntimeError(f"Dataset tercemar varietas lain: {t}")
            split[t] = r["new_split"]
            gt[t] = np.array([int(r[c]) for c in KELAS], float)
    return split, gt


def metadata_pohon(tree: str) -> list[dict]:
    p = DS_ASLI / "json" / f"{tree}.json"
    d = json.loads(p.read_text(encoding="utf-8-sig"))
    sisi = []
    for im in sorted(d["images"].values(), key=lambda x: x["side_index"]):
        sisi.append({"stem": Path(im["filename"]).stem,
                     "w": float(im["width"]), "h": float(im["height"])})
    return sisi


def dedup_anchor(D: np.ndarray) -> np.ndarray:
    if len(D) == 0:
        return np.zeros((0, 11), np.float32)
    # infer_skor_penuh dapat menyimpan satu baris per kelas untuk anchor yang
    # sama. Keempat skor kelas pada baris-baris itu identik.
    _, u = np.unique(D[:, 10], return_index=True)
    return D[np.sort(u)]


def fitur_pohon(tree: str, z) -> np.ndarray:
    semua = []
    for nomor, s in enumerate(metadata_pohon(tree)):
        D = dedup_anchor(z[s["stem"]] if s["stem"] in z.files
                         else np.zeros((0, 11), np.float32))
        if len(D):
            p_raw = np.clip(D[:, 6:10].astype(float), 1e-8, 1.0)
            p = p_raw / np.maximum(p_raw.sum(1, keepdims=True), 1e-9)
            box = D[:, :4].astype(float)
            geo = np.c_[((box[:, 0] + box[:, 2]) * .5 / s["w"]),
                        ((box[:, 1] + box[:, 3]) * .5 / s["h"]),
                        ((box[:, 2] - box[:, 0]) / s["w"]),
                        ((box[:, 3] - box[:, 1]) / s["h"])]
            geo = np.c_[geo, geo[:, 2] * geo[:, 3]]
        else:
            p_raw = p = np.zeros((0, 4), float)
            geo = np.zeros((0, 5), float)
        semua.append({"p": p, "raw": p_raw, "geo": geo, "side": nomor})

    n_sisi = len(semua)
    f: list[float] = [float(n_sisi), float(n_sisi == 4), float(n_sisi == 8)]

    # Fitur multi-ambang. Hard count menangkap perilaku NMS operasional,
    # sementara soft count menjaga informasi kelas kedua yang sering penting
    # pada transisi B2/B3.
    for t in AMBANG:
        hard_sisi = np.zeros((n_sisi, 4), float)
        soft_sisi = np.zeros((n_sisi, 4), float)
        raw_sisi = np.zeros((n_sisi, 4), float)
        total_sisi = np.zeros(n_sisi, float)
        for i, d in enumerate(semua):
            if not len(d["p"]):
                continue
            conf = d["raw"].max(1)
            m = conf >= t
            if not m.any():
                continue
            hard_sisi[i] = np.bincount(d["p"][m].argmax(1), minlength=4)
            soft_sisi[i] = d["p"][m].sum(0)
            raw_sisi[i] = d["raw"][m].sum(0)
            total_sisi[i] = float(m.sum())
        for k in range(4):
            f += statistik(hard_sisi[:, k])
            f += statistik(soft_sisi[:, k])
            f += [float(raw_sisi[:, k].sum()), float(raw_sisi[:, k].max())]
        f += statistik(total_sisi)
        tot = hard_sisi.sum()
        f += [float(hard_sisi[:, k].sum() / max(tot, 1e-9)) for k in range(4)]

    # Statistik kandidat pada tiga ambang kunci. Posisi dan luas membantu
    # membedakan banyak deteksi kecil/tepi dari tandan fisik yang konsisten.
    for t in (0.03, 0.10, 0.25):
        for k in range(4):
            conf, margin, ent, cx, cy, luas, rasio = ([] for _ in range(7))
            for d in semua:
                if not len(d["p"]):
                    continue
                c = d["raw"].max(1)
                lab = d["p"].argmax(1)
                m = (c >= t) & (lab == k)
                if not m.any():
                    continue
                q = d["p"][m]
                urut = np.sort(q, axis=1)
                conf.extend(c[m]); margin.extend(urut[:, -1] - urut[:, -2])
                ent.extend(-(q * np.log(np.clip(q, 1e-9, 1))).sum(1))
                cx.extend(d["geo"][m, 0]); cy.extend(d["geo"][m, 1])
                luas.extend(d["geo"][m, 4])
                rasio.extend(d["geo"][m, 2] / np.maximum(d["geo"][m, 3], 1e-6))
            for x in (conf, margin, ent, cx, cy, luas, rasio):
                f += statistik(np.asarray(x, float))

    return np.asarray(f, np.float32)


def muat_data() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[str]]]:
    split, gt = baca_manifest()
    ids = {s: sorted(t for t, v in split.items() if v == s)
           for s in ("train", "val", "test")}
    X, Y = {}, {}
    for s in ids:
        z = np.load(SUB / "results" / f"pred_skorpenuh_{s}.npz", allow_pickle=True)
        X[s] = np.stack([fitur_pohon(t, z) for t in ids[s]])
        Y[s] = np.stack([gt[t] for t in ids[s]])
        z.close()
        print(f"{s:5s}: pohon={len(ids[s])} fitur={X[s].shape[1]}", flush=True)
    return X, Y, ids


def kandidat() -> dict[str, object]:
    return {
        "ridge_a1": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "ridge_a10": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "ridge_a100": make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        "ridge_robust_a10": make_pipeline(RobustScaler(), Ridge(alpha=10.0)),
        "extra_l2_f05": ExtraTreesRegressor(
            n_estimators=600, min_samples_leaf=2, max_features=.5,
            n_jobs=-1, random_state=0),
        "extra_l4_f07": ExtraTreesRegressor(
            n_estimators=600, min_samples_leaf=4, max_features=.7,
            n_jobs=-1, random_state=0),
        "extra_l8_f10": ExtraTreesRegressor(
            n_estimators=600, min_samples_leaf=8, max_features=1.0,
            n_jobs=-1, random_state=0),
        "rf_l3_f05": RandomForestRegressor(
            n_estimators=600, min_samples_leaf=3, max_features=.5,
            n_jobs=-1, random_state=0),
        "rf_l6_f08": RandomForestRegressor(
            n_estimators=600, min_samples_leaf=6, max_features=.8,
            n_jobs=-1, random_state=0),
        "hist_l15_l2": MultiOutputRegressor(HistGradientBoostingRegressor(
            learning_rate=.05, max_iter=300, max_leaf_nodes=15,
            l2_regularization=2.0, random_state=0)),
        "hist_l7_l10": MultiOutputRegressor(HistGradientBoostingRegressor(
            learning_rate=.05, max_iter=300, max_leaf_nodes=7,
            l2_regularization=10.0, random_state=0)),
        "gbr_huber_d1": MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=300, learning_rate=.03, max_depth=1, loss="huber",
            random_state=0)),
        "gbr_huber_d2": MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=300, learning_rate=.03, max_depth=2, loss="huber",
            random_state=0)),
    }


def kandidat_total() -> dict[str, object]:
    """Kepala class-agnostic untuk jumlah seluruh tandan fisik."""
    return {
        "total_ridge10": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "total_ridge100": make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        "total_extra_l2": ExtraTreesRegressor(
            n_estimators=600, min_samples_leaf=2, max_features=.5,
            n_jobs=-1, random_state=11),
        "total_extra_l6": ExtraTreesRegressor(
            n_estimators=600, min_samples_leaf=6, max_features=.8,
            n_jobs=-1, random_state=11),
        "total_rf_l4": RandomForestRegressor(
            n_estimators=600, min_samples_leaf=4, max_features=.6,
            n_jobs=-1, random_state=11),
        "total_hist_l7": HistGradientBoostingRegressor(
            learning_rate=.05, max_iter=300, max_leaf_nodes=7,
            l2_regularization=10.0, random_state=11),
        "total_gbr_huber": GradientBoostingRegressor(
            n_estimators=300, learning_rate=.03, max_depth=2, loss="huber",
            random_state=11),
    }


def bulat(p: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(np.asarray(p) + .5), 0, None).astype(int)


def metrik(y: np.ndarray, p: np.ndarray) -> dict:
    q = bulat(p)
    e = np.abs(q - y.astype(int))
    return {
        "macro_mae": float(e.mean()),
        "class_pm1_acc": float((e <= 1).mean()),
        "tree_pm1_acc": float((e <= 1).all(1).mean()),
        "total_mae": float(np.abs(q.sum(1) - y.sum(1)).mean()),
        "bias_total": float((q - y).sum(1).mean()),
        "per_kelas": {
            KELAS[k]: {"mae": float(e[:, k].mean()),
                       "pm1_acc": float((e[:, k] <= 1).mean()),
                       "bias": float((q[:, k] - y[:, k]).mean())}
            for k in range(4)},
    }


def nilai_kelas(y: np.ndarray, p: np.ndarray) -> tuple:
    q = bulat(p)
    e = np.abs(q - y)
    # MAE adalah sasaran pertama. Toleransi +-1 dan bias memecahkan seri.
    return (float(e.mean()), -float((e <= 1).mean()),
            abs(float((q - y).mean())))


@dataclass
class KepalaKelas:
    anggota: list[str]
    bobot: list[float]
    skala: float
    bias: float


def kalibrasi(y: np.ndarray, p: np.ndarray) -> tuple[float, float, tuple]:
    terbaik = None
    for a in np.arange(.80, 1.201, .025):
        for b in np.arange(-1.50, 1.501, .05):
            v = nilai_kelas(y, a * p + b)
            kunci = (*v, abs(a - 1), abs(b))
            if terbaik is None or kunci < terbaik[0]:
                terbaik = (kunci, float(a), float(b))
    return terbaik[1], terbaik[2], terbaik[0]


def pilih_per_kelas(y: np.ndarray, pv: dict[str, np.ndarray]) -> list[KepalaKelas]:
    kepala = []
    for k, kelas in enumerate(KELAS):
        tunggal = []
        for nama, p in pv.items():
            a, b, skor = kalibrasi(y[:, k], p[:, k])
            tunggal.append((skor, nama, a, b))
        tunggal.sort()
        _, awal, a, b = tunggal[0]
        anggota, bobot, best = [awal], [1.0], tunggal[0][0]

        # Forward blend. Bobot kandidat baru disapu halus dan selalu dinilai
        # setelah kalibrasi ulang di val.
        while len(anggota) < 4:
            calon = []
            p_lama = sum(w * pv[n][:, k] for n, w in zip(anggota, bobot))
            for nama in pv:
                if nama in anggota:
                    continue
                for w_baru in np.arange(.10, .501, .05):
                    p = (1 - w_baru) * p_lama + w_baru * pv[nama][:, k]
                    ca, cb, skor = kalibrasi(y[:, k], p)
                    calon.append((skor, nama, float(w_baru), ca, cb))
            if not calon:
                break
            c = min(calon)
            if c[0] >= best:
                break
            w_baru = c[2]
            bobot = [(1 - w_baru) * w for w in bobot] + [w_baru]
            anggota.append(c[1]); a, b, best = c[3], c[4], c[0]
        print(f"  {kelas}: {anggota} w={np.round(bobot, 3).tolist()} "
              f"kal=({a:.3f},{b:+.3f}) val={best}", flush=True)
        kepala.append(KepalaKelas(anggota, bobot, a, b))
    return kepala


def prediksi_kepala(kepala: list[KepalaKelas], pred: dict[str, np.ndarray]) -> np.ndarray:
    out = np.zeros((len(next(iter(pred.values()))), 4), float)
    for k, h in enumerate(kepala):
        p = sum(w * pred[n][:, k] for n, w in zip(h.anggota, h.bobot))
        out[:, k] = h.skala * p + h.bias
    return out


def serial_kepala(kepala):
    return [{"kelas": KELAS[k], "anggota": h.anggota, "bobot": h.bobot,
             "skala": h.skala, "bias": h.bias} for k, h in enumerate(kepala)]


def metrik_total(y: np.ndarray, p: np.ndarray) -> dict:
    q = bulat(p)
    e = np.abs(q - y.astype(int))
    return {"mae": float(e.mean()), "pm1_acc": float((e <= 1).mean()),
            "bias": float((q - y).mean())}


def cari_kalibrasi_total(y: np.ndarray, p: np.ndarray):
    terbaik = None
    for a in np.arange(.8, 1.201, .025):
        for b in np.arange(-2.0, 2.001, .05):
            m = metrik_total(y, a * p + b)
            k = (m["mae"], -m["pm1_acc"], abs(m["bias"]), abs(a - 1), abs(b))
            if terbaik is None or k < terbaik[0]:
                terbaik = (k, float(a), float(b), m)
    return terbaik


def pilih_total(X, y):
    yt = y["train"].sum(1); yv = y["val"].sum(1)
    ranking = []
    for nama, model in kandidat_total().items():
        m = clone(model).fit(X["train"], yt)
        pv = m.predict(X["val"])
        kal = cari_kalibrasi_total(yv, pv)
        ranking.append((kal[0], nama, kal[1], kal[2], kal[3]))
        print(f"{nama:18s} val total MAE={kal[3]['mae']:.4f} "
              f"+-1={kal[3]['pm1_acc']:.4f} kal=({kal[1]:.3f},{kal[2]:+.2f})",
              flush=True)
    ranking.sort()
    return ranking


def proyeksi_jumlah(p: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Proyeksi integer konveks: sum empat kelas tepat sama dengan total."""
    out = np.zeros_like(p, int)
    for i, (v, tt) in enumerate(zip(p, bulat(total))):
        q = np.zeros(4, int)
        for _ in range(int(tt)):
            biaya = (q + 1 - v) ** 2 - (q - v) ** 2
            q[int(np.argmin(biaya))] += 1
        out[i] = q
    return out


def simplex_langkah_10():
    for a in range(11):
        for b in range(11 - a):
            for c in range(11 - a - b):
                d = 10 - a - b - c
                yield np.array([a, b, c, d], float) / 10


def pilih_rekonsiliasi(y, p, total):
    """Cari pembagian koreksi total di val; raw selalu menjadi kandidat."""
    p = np.clip(p, 0, None)
    kandidat_rek = [("raw", 0.0, None, False, p)]
    delta = total - p.sum(1)
    prop = p / np.maximum(p.sum(1, keepdims=True), 1e-9)
    for beta in np.arange(.25, 1.501, .25):
        q = p + beta * delta[:, None] * prop
        kandidat_rek += [("prop", float(beta), None, False, q),
                         ("prop", float(beta), None, True,
                          proyeksi_jumlah(q, total))]
        for alokasi in simplex_langkah_10():
            q = p + beta * delta[:, None] * alokasi[None, :]
            kandidat_rek.append(("tetap", float(beta), alokasi, False, q))
            kandidat_rek.append(("tetap", float(beta), alokasi, True,
                                  proyeksi_jumlah(q, total)))
    terbaik = None
    for mode, beta, alokasi, proj, q in kandidat_rek:
        m = metrik(y, q)
        k = (m["macro_mae"], -m["class_pm1_acc"], -m["tree_pm1_acc"],
             m["total_mae"], abs(m["bias_total"]))
        if terbaik is None or k < terbaik[0]:
            terbaik = (k, mode, beta, alokasi, proj, m)
    return terbaik


def terapkan_rekonsiliasi(p, total, cfg):
    _, mode, beta, alokasi, proj, _ = cfg
    p = np.clip(p, 0, None)
    if mode == "raw":
        q = p
    else:
        delta = total - p.sum(1)
        if mode == "prop":
            a = p / np.maximum(p.sum(1, keepdims=True), 1e-9)
        else:
            a = np.asarray(alokasi)[None, :]
        q = p + beta * delta[:, None] * a
    return proyeksi_jumlah(q, total) if proj else q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keluaran", type=Path,
                    default=SUB / "results" / "damimas_counting.json")
    ap.add_argument("--model-out", type=Path,
                    default=SUB / "runs" / "counting_damimas" / "ensemble.joblib")
    args = ap.parse_args()
    X, y, ids = muat_data()
    models = kandidat()

    # Seleksi murni train -> val.
    pred_val, ranking = {}, []
    for nama, model in models.items():
        m = clone(model).fit(X["train"], y["train"])
        pred_val[nama] = m.predict(X["val"])
        mm = metrik(y["val"], pred_val[nama])
        ranking.append((mm["macro_mae"], -mm["class_pm1_acc"], nama, mm))
        print(f"{nama:18s} val MAE={mm['macro_mae']:.4f} "
              f"class+-1={mm['class_pm1_acc']:.4f} tree+-1={mm['tree_pm1_acc']:.4f}",
              flush=True)
    ranking.sort()
    kepala = pilih_per_kelas(y["val"], pred_val)
    pred_ens_val = prediksi_kepala(kepala, pred_val)
    met_val = metrik(y["val"], pred_ens_val)
    print("ENSEMBLE VAL", met_val, flush=True)

    # Kepala total dipilih terpisah, lalu dipakai untuk merekonsiliasi empat
    # keluaran kelas. Seluruh konfigurasi rekonsiliasi tetap hanya melihat val.
    rank_total = pilih_total(X, y)
    _, nama_total, skala_total, bias_total, met_total_val = rank_total[0]
    model_total_val = clone(kandidat_total()[nama_total]).fit(
        X["train"], y["train"].sum(1))
    total_val = skala_total * model_total_val.predict(X["val"]) + bias_total
    rek = pilih_rekonsiliasi(y["val"], pred_ens_val, total_val)
    pred_final_val = terapkan_rekonsiliasi(pred_ens_val, total_val, rek)
    met_final_val = metrik(y["val"], pred_final_val)
    print(f"REKONSILIASI VAL mode={rek[1]} beta={rek[2]} "
          f"alokasi={None if rek[3] is None else rek[3].tolist()} "
          f"proyeksi={rek[4]} -> {met_final_val}", flush=True)

    # Konfigurasi sudah terkunci. Refit hanya anggota yang terpakai pada
    # train+val, baru kemudian inferensi test.
    dipakai = sorted({n for h in kepala for n in h.anggota})
    Xtv = np.concatenate([X["train"], X["val"]])
    ytv = np.concatenate([y["train"], y["val"]])
    fitted, pred_test = {}, {}
    for nama in dipakai:
        fitted[nama] = clone(models[nama]).fit(Xtv, ytv)
        pred_test[nama] = fitted[nama].predict(X["test"])
    pred_ens_test = prediksi_kepala(kepala, pred_test)
    met_test = metrik(y["test"], pred_ens_test)
    model_total = clone(kandidat_total()[nama_total]).fit(Xtv, ytv.sum(1))
    total_test = skala_total * model_total.predict(X["test"]) + bias_total
    pred_final_test = terapkan_rekonsiliasi(pred_ens_test, total_test, rek)
    met_final_test = metrik(y["test"], pred_final_test)

    hasil = {
        "dataset": "SawitMVC-YOLO-Damimas",
        "protokol": "fit train; pilih+kalibrasi val; refit train+val; test sekali",
        "n": {s: {"pohon": len(ids[s]), "fitur": int(X[s].shape[1])} for s in ids},
        "ambang_fitur": list(AMBANG),
        "ranking_val": [{"model": r[2], "metrik": r[3]} for r in ranking],
        "ensemble": {"kepala_per_kelas": serial_kepala(kepala),
                     "val": met_val, "test": met_test},
        "kepala_total": {
            "ranking_val": [{"model": r[1], "skala": r[2], "bias": r[3],
                              "metrik": r[4]} for r in rank_total],
            "terpilih": nama_total, "skala": skala_total, "bias": bias_total,
            "val": met_total_val,
            "test": metrik_total(y["test"].sum(1), total_test),
            "definisi": "regresor langsung untuk jumlah B1+B2+B3+B4",
        },
        "final_rekonsiliasi": {
            "mode": rek[1], "beta": rek[2],
            "alokasi": None if rek[3] is None else rek[3].tolist(),
            "proyeksi_integer": rek[4], "val": met_final_val,
            "test": met_final_test,
        },
        "catatan_total_mae": (
            "final_rekonsiliasi.test.total_mae adalah MAE jumlah empat kepala "
            "kelas; ketika mode raw, ia bukan metrik kepala_total.test"),
    }
    args.keluaran.parent.mkdir(parents=True, exist_ok=True)
    args.keluaran.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    # Simpan kepala sebagai dict primitif. Menyimpan dataclass saat skrip
    # dieksekusi langsung akan merekam tipe ``__main__.KepalaKelas`` dan model
    # tidak bisa dimuat dari proses inferensi lain.
    joblib.dump({"models": fitted, "kepala": serial_kepala(kepala),
                 "model_total": model_total,
                 "total_kalibrasi": (skala_total, bias_total),
                 "rekonsiliasi": {"mode": rek[1], "beta": rek[2],
                                   "alokasi": rek[3], "proyeksi": rek[4]},
                 "ambang": AMBANG,
                 "fitur_dim": X["train"].shape[1]}, args.model_out)
    print(json.dumps({"ensemble": hasil["ensemble"],
                      "final_rekonsiliasi": hasil["final_rekonsiliasi"]},
                     indent=2, ensure_ascii=False))
    print(f"-> {args.keluaran}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
