"""Rekonsiliasi label tandan dengan kepala counting pada tingkat pohon.

Classifier per-tandan dan penghitung per-pohon melihat sinyal yang berbeda.
Classifier mengetahui tandan mana yang ambigu, sedangkan counting memberi
perkiraan komposisi B1--B4 seluruh pohon. Skrip ini menggabungkan keduanya
sebagai inferensi global, tanpa memakai label test untuk memilih aturan.

Protokol:

* probabilitas classifier VAL adalah OOF berbasis grup pohon;
* model counting untuk VAL hanya dipasang pada TRAIN;
* keluarga aturan dan kekuatannya dipilih di VAL;
* model counting TEST adalah artefak yang sudah di-refit pada TRAIN+VAL;
* TEST dinilai sekali setelah aturan terkunci.

Jumlah pool yang masuk dianggap diketahui oleh tahap penaut. Vektor count
diproyeksikan agar jumlahnya sama dengan banyak pool, lalu dipakai sebagai
prior lunak atau kuota penugasan global. Ini tidak memakai jumlah/kelas GT saat
inferensi; ``y`` hanya dipakai oleh fungsi metrik setelah prediksi selesai.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


SUB = Path(__file__).resolve().parents[1]
KELAS = ("B1", "B2", "B3", "B4")
sys.path.insert(0, str(Path(__file__).parent))
import counting_damimas as CD  # noqa: E402
import stacker_damimas as SD  # noqa: E402


def metrik(y: np.ndarray, yh: np.ndarray) -> dict:
    p, r, f, n = precision_recall_fscore_support(
        y, yh, labels=np.arange(4), zero_division=0)
    return {
        "akurasi": float(accuracy_score(y, yh)),
        "macro_f1": float(f.mean()),
        "mae_ordinal": float(np.abs(y - yh).mean()),
        "precision_per_kelas": {KELAS[k]: float(p[k]) for k in range(4)},
        "recall_per_kelas": {KELAS[k]: float(r[k]) for k in range(4)},
        "f1_per_kelas": {KELAS[k]: float(f[k]) for k in range(4)},
        "support_per_kelas": {KELAS[k]: int(n[k]) for k in range(4)},
        "confusion": confusion_matrix(y, yh, labels=np.arange(4)).tolist(),
    }


def objektif(m: dict) -> float:
    # Akurasi tetap sasaran utama, macro-F1 mencegah keuntungan dibeli dengan
    # mengorbankan kelas minoritas, dan kelas terlemah menjadi pemecah seri.
    return (0.55 * m["akurasi"] + 0.40 * m["macro_f1"]
            + 0.05 * min(m["f1_per_kelas"].values()))


def prediksi_kepala_serial(kepala: list[dict], pred: dict[str, np.ndarray]) -> np.ndarray:
    out = np.zeros((len(next(iter(pred.values()))), 4), float)
    for k, h in enumerate(kepala):
        q = sum(w * pred[nama][:, k]
                for nama, w in zip(h["anggota"], h["bobot"]))
        out[:, k] = h["skala"] * q + h["bias"]
    return np.clip(out, 0, None)


def prediksi_count_val_test() -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Rekonstruksi prediksi counting sesuai protokol eksperimen asal."""
    X, y, ids = CD.muat_data()
    artefak = joblib.load(SUB / "runs" / "counting_damimas" / "ensemble.joblib")
    kepala = artefak["kepala"]
    dipakai = sorted({n for h in kepala for n in h["anggota"]})

    # VAL: fit TRAIN saja. Kalibrasi kepala memang dipilih di VAL oleh
    # counting_damimas.py dan menjadi bagian konfigurasi yang diwarisi.
    pv = {}
    semua = CD.kandidat()
    for nama in dipakai:
        pv[nama] = clone(semua[nama]).fit(X["train"], y["train"]).predict(X["val"])

    # TEST: gunakan model final yang sudah di-refit TRAIN+VAL.
    pt = {nama: artefak["models"][nama].predict(X["test"])
          for nama in dipakai}
    return ({"val": prediksi_kepala_serial(kepala, pv),
             "test": prediksi_kepala_serial(kepala, pt)}, ids)


def proyeksi_kuota(target: np.ndarray, total: int) -> np.ndarray:
    """Proyeksi integer konveks nonnegatif dengan jumlah tepat ``total``."""
    target = np.clip(np.asarray(target, float), 0, None)
    q = np.zeros(4, int)
    for _ in range(total):
        biaya = (q + 1 - target) ** 2 - (q - target) ** 2
        q[int(np.argmin(biaya))] += 1
    return q


def skala_ke_total(v: np.ndarray, total: int) -> np.ndarray:
    v = np.clip(np.asarray(v, float), 0, None)
    if v.sum() <= 1e-9:
        return np.full(4, total / 4)
    return v * (total / v.sum())


def tugas_kuota(prob: np.ndarray, kuota: np.ndarray) -> np.ndarray:
    slot = np.repeat(np.arange(4), kuota)
    if len(slot) != len(prob):
        raise RuntimeError("Jumlah slot kuota tidak sama dengan jumlah tandan")
    biaya = -np.log(np.clip(prob[:, slot], 1e-9, 1.0))
    baris, kolom = linear_sum_assignment(biaya)
    out = np.empty(len(prob), int)
    out[baris] = slot[kolom]
    return out


def prediksi_global(prob: np.ndarray, trees: np.ndarray,
                    count_by_tree: dict[str, np.ndarray], cfg: dict,
                    aturan_dasar: str, tau_dasar) -> np.ndarray:
    out = SD.prediksi_prob(prob, aturan_dasar, tau_dasar)
    if cfg["mode"] == "identity":
        return out
    for tree in np.unique(trees):
        idx = np.flatnonzero(trees == tree)
        p = np.clip(prob[idx], 1e-9, 1.0)
        n = len(idx)
        dasar = p.sum(0)
        hitung = skala_ke_total(count_by_tree[tree], n)
        target = (1 - cfg["lambda_count"]) * dasar + cfg["lambda_count"] * hitung
        if cfg["mode"] == "kuota":
            out[idx] = tugas_kuota(p, proyeksi_kuota(target, n))
        elif cfg["mode"] == "prior_lunak":
            rasio = (target + cfg["alpha"]) / (dasar + cfg["alpha"])
            skor = np.log(p) + cfg["gamma"] * np.log(np.clip(rasio, 1e-6, 1e6))
            skor -= skor.max(1, keepdims=True)
            q = np.exp(skor)
            q /= np.maximum(q.sum(1, keepdims=True), 1e-9)
            out[idx] = SD.prediksi_prob(q, aturan_dasar, tau_dasar)
        else:
            raise ValueError(cfg["mode"])
    return out


def kandidat_cfg() -> list[dict]:
    out = [{"mode": "identity", "lambda_count": 0.0,
            "gamma": 0.0, "alpha": 0.0}]
    for lam in np.arange(0.0, 1.501, 0.10):
        out.append({"mode": "kuota", "lambda_count": float(round(lam, 2)),
                    "gamma": 0.0, "alpha": 0.0})
        for gamma in (0.25, 0.50, 0.75, 1.0, 1.5, 2.0):
            for alpha in (0.1, 0.5, 1.0):
                out.append({"mode": "prior_lunak",
                            "lambda_count": float(round(lam, 2)),
                            "gamma": gamma, "alpha": alpha})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", type=Path,
                    default=SUB / "results" / "damimas_stacker_pred.npz")
    ap.add_argument("--keluaran", type=Path,
                    default=SUB / "results" / "damimas_rekonsiliasi_pohon.json")
    ap.add_argument("--pred-out", type=Path,
                    default=SUB / "results" / "damimas_rekonsiliasi_pohon_pred.npz")
    args = ap.parse_args()

    z = np.load(args.classifier, allow_pickle=True)
    info_stacker = json.loads(
        (SUB / "results" / "damimas_stacker_pertandan.json").read_text())
    aturan_dasar = info_stacker["stacker"]["aturan"]
    tau_dasar = info_stacker["stacker"]["tau"]
    data = {
        s: {"prob": z[f"{s}_prob"].astype(float),
            "y": z[f"{s}_y"].astype(int),
            "tree": z[f"{s}_tree"].astype(str)}
        for s in ("val", "test")
    }
    count, ids = prediksi_count_val_test()
    count_map = {s: {t: count[s][i] for i, t in enumerate(ids[s])}
                 for s in ("val", "test")}
    for s in data:
        hilang = sorted(set(data[s]["tree"]) - set(count_map[s]))
        if hilang:
            raise RuntimeError(f"{s}: tree tanpa prediksi counting: {hilang[:3]}")

    ranking = []
    for cfg in kandidat_cfg():
        yh = prediksi_global(data["val"]["prob"], data["val"]["tree"],
                             count_map["val"], cfg, aturan_dasar, tau_dasar)
        m = metrik(data["val"]["y"], yh)
        ranking.append((objektif(m), m["akurasi"], m["macro_f1"], cfg, m))
    ranking.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    _, _, _, cfg, mv = ranking[0]

    yh_test = prediksi_global(data["test"]["prob"], data["test"]["tree"],
                              count_map["test"], cfg, aturan_dasar, tau_dasar)
    mt = metrik(data["test"]["y"], yh_test)
    base = {s: metrik(data[s]["y"], SD.prediksi_prob(
        data[s]["prob"], aturan_dasar, tau_dasar))
            for s in data}
    hasil = {
        "dataset": "SawitMVC-YOLO-Damimas",
        "protokol": ("classifier VAL OOF + counting VAL fit TRAIN; pilih aturan di VAL; "
                     "TEST memakai classifier terkunci dan counting refit TRAIN+VAL"),
        "objective": "0.55 accuracy + 0.40 macro-F1 + 0.05 worst-class F1",
        "aturan_stacker_dasar": {"aturan": aturan_dasar, "tau": tau_dasar},
        "baseline_argmax": base,
        "terpilih_di_val": {"config": cfg, "metrik": mv},
        "test": mt,
        "top10_val": [{"config": r[3], "objective": r[0], "metrik": r[4]}
                      for r in ranking[:10]],
    }
    args.keluaran.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
    np.savez_compressed(
        args.pred_out,
        val_y=data["val"]["y"], val_yhat=prediksi_global(
            data["val"]["prob"], data["val"]["tree"], count_map["val"], cfg,
            aturan_dasar, tau_dasar),
        val_tree=data["val"]["tree"], test_y=data["test"]["y"],
        test_yhat=yh_test, test_tree=data["test"]["tree"],
    )
    print(json.dumps({"baseline": base, "terpilih_di_val": hasil["terpilih_di_val"],
                      "test": mt}, indent=2, ensure_ascii=False))
    print(f"-> {args.keluaran}")


if __name__ == "__main__":
    main()
