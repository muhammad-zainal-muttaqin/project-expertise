"""Audit kepala counting total yang sudah terkunci pada DAMIMAS.

Eksperimen counting menyimpan dua keluaran berbeda: empat kepala per kelas dan
satu regresor khusus jumlah total. Jika rekonsiliasi yang dipilih adalah
``raw``, ``test.total_mae`` pada laporan lama berasal dari penjumlahan empat
kepala kelas; ia bukan metrik regresor total. Skrip ini hanya memuat model yang
sudah dipasang dan konfigurasi yang sudah terkunci. Tidak ada fitting,
kalibrasi, pemilihan model, atau perubahan prediksi berdasarkan TEST.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np


SUB = Path(__file__).resolve().parents[1]
ROOT = SUB.parent
sys.path.insert(0, str(Path(__file__).parent))
import counting_damimas as CD  # noqa: E402
import counting_multibank_damimas as CM  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blok in iter(lambda: f.read(1024 * 1024), b""):
            h.update(blok)
    return h.hexdigest()


def fitur_graf_legacy(g, bundle, heads):
    """Fitur linker 225-dim persis seperti source run compact PT-E-026."""
    model_names = sorted(bundle["models"])
    dim = 5 + 4 * 8 + 8 + len(model_names) * 8 + len(heads) * (3 + 4 * 8)
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
        f += CM.stats(x)
    f += CM.stats(np.bincount(sisi, minlength=g["nv"]))
    score_model = {}
    for nama in model_names:
        s = bundle["models"][nama].predict_proba(g["E"])[:, 1]
        score_model[nama] = s
        f += CM.stats(s)
    for q in heads.values():
        s = sum(w * score_model[n] for n, w in q["bobot_skor"].items())
        lab = CM.rakit(g, s, q)
        grup = defaultdict(list)
        for i, k in enumerate(lab):
            grup[int(k)].append(i)
        ukuran = np.asarray([len(v) for v in grup.values()], float)
        f += [len(grup), float((ukuran >= 2).sum()),
              float((ukuran >= 2).mean()) if len(ukuran) else 0.]
        f += CM.stats(ukuran)
        f += CM.stats([conf[v].mean() for v in grup.values()])
        f += CM.stats([eks[v].mean() for v in grup.values()])
        f += CM.stats([len(np.unique(sisi[v])) for v in grup.values()])
    out = np.asarray(f, np.float32)
    if len(out) != dim:
        raise RuntimeError(f"Dimensi fitur legacy berubah: {len(out)} != {dim}")
    return out


def metrik(y_total, pred):
    return CD.metrik_total(np.asarray(y_total), np.asarray(pred))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=SUB / "results" /
                    "damimas_counting_total_head_audit.json")
    ap.add_argument("--pred-out", type=Path, default=SUB / "results" /
                    "damimas_counting_total_head_audit_pred.npz")
    ap.add_argument("--baseline-model", type=Path, default=SUB / "runs" /
                    "counting_damimas" / "ensemble.joblib")
    ap.add_argument("--compact-model", type=Path, default=SUB / "runs" /
                    "counting_multibank_damimas" / "ensemble_compact.joblib")
    ap.add_argument("--full-model", type=Path, default=SUB / "runs" /
                    "counting_multibank_damimas" / "ensemble_full.joblib")
    ap.add_argument("--catboost-model", type=Path, default=SUB / "runs" /
                    "counting_catboost_damimas" / "ensemble.joblib")
    ap.add_argument("--catboost-pred", type=Path, default=SUB / "results" /
                    "damimas_counting_catboost_pred.npz")
    ap.add_argument("--linker-config", type=Path, default=SUB / "results" /
                    "damimas_linker_global_proposal_yolo_lock.json")
    ap.add_argument("--linker-model", type=Path, default=SUB / "runs" /
                    "linker_global_damimas_proposal_yolo" / "model.joblib")
    ap.add_argument("--cache-test", type=Path, default=SUB / "results" /
                    "cache_linker_damimas_damimas_damimas_proposal_yolo_test_test.joblib")
    args = ap.parse_args()

    ids, y = CM.ids_y("test")
    anchor_path = SUB / "results" / "pred_skorpenuh_test.npz"
    proposal_path = ROOT / "results" / "pred_damimas_proposal_yolo_test.npz"
    A_z = np.load(anchor_path, allow_pickle=True)
    P_z = np.load(proposal_path, allow_pickle=True)
    A = np.stack([CD.fitur_pohon(t, A_z) for t in ids])
    P = np.stack([CD.fitur_pohon(t, P_z) for t in ids])
    A_z.close(); P_z.close()

    cfg = json.loads(args.linker_config.read_text())
    linker = joblib.load(args.linker_model)
    heads = cfg["heads"]
    graf = joblib.load(args.cache_test)["graf"]
    old = {g["tree"]: fitur_graf_legacy(g, linker, heads) for g in graf}
    new = {g["tree"]: CM.fitur_graf(g, linker, heads) for g in graf}
    old0 = fitur_graf_legacy(None, linker, heads)
    new0 = CM.fitur_graf(None, linker, heads)
    L_old = np.stack([old.get(t, old0) for t in ids])
    L_new = np.stack([new.get(t, new0) for t in ids])

    b0 = joblib.load(args.baseline_model)
    a0, c0 = b0["total_kalibrasi"]
    pred_baseline = a0 * b0["model_total"].predict(A) + c0

    bc = joblib.load(args.compact_model)
    ac, cc = bc["total"]
    X_compact = np.c_[A, P, L_old]
    if X_compact.shape[1] != bc["fitur_dim"]["concat_linker"]:
        raise RuntimeError("Dimensi compact tidak cocok dengan model tersimpan")
    pred_compact = ac * bc["model_total"].predict(X_compact) + cc

    bf = joblib.load(args.full_model)
    af, cf = bf["total"]
    X_full = np.c_[A, P, L_new]
    if X_full.shape[1] != bf["fitur_dim"]["concat_linker"]:
        raise RuntimeError("Dimensi full tidak cocok dengan model tersimpan")
    pred_full = af * bf["model_total"].predict(X_full) + cf

    zcat = np.load(args.catboost_pred, allow_pickle=True)
    if list(zcat["test_id"].astype(str)) != ids:
        raise RuntimeError("Urutan ID CatBoost tidak identik")
    if not np.array_equal(zcat["test_y"], y):
        raise RuntimeError("Label audit dan dump CatBoost tidak identik")
    pred_cat = np.asarray(zcat["test_total"], float)
    zcat.close()

    pred = {"baseline_anchor": pred_baseline,
            "compact_multibank": pred_compact,
            "full_multibank": pred_full,
            "catboost": pred_cat}
    hasil_test = {n: metrik(y.sum(1), p) for n, p in pred.items()}
    urut = sorted(hasil_test, key=lambda n: (
        hasil_test[n]["mae"], -hasil_test[n]["pm1_acc"],
        abs(hasil_test[n]["bias"])))

    sumber_model = {"baseline_anchor": args.baseline_model,
                    "compact_multibank": args.compact_model,
                    "full_multibank": args.full_model,
                    "catboost": args.catboost_model}
    hasil = {
        "dataset": "SawitMVC-YOLO-Damimas",
        "protokol": ("audit inference-only atas kepala total yang sebelumnya "
                     "sudah dikunci; tanpa fit, kalibrasi, atau seleksi di TEST"),
        "definisi": {
            "kepala_total_terpisah": "regresor langsung untuk jumlah B1+B2+B3+B4",
            "total_mae_laporan_lama": ("MAE penjumlahan empat kepala kelas ketika "
                                       "rekonsiliasi terkunci pada mode raw"),
        },
        "n_test": len(ids),
        "dimensi": {"anchor": int(A.shape[1]), "proposal": int(P.shape[1]),
                    "linker_legacy": int(L_old.shape[1]),
                    "linker_full": int(L_new.shape[1]),
                    "compact": int(X_compact.shape[1]),
                    "full": int(X_full.shape[1])},
        "test_kepala_total_terpisah": hasil_test,
        "ranking_test_deskriptif_bukan_kunci_seleksi": urut,
        "catatan_seleksi": ("Ranking TEST ini hanya audit deskriptif. Kepala final "
                            "paper harus dipilih kembali dari OOF/VAL sebelum "
                            "evaluasi konfirmatori."),
        "artefak": {n: {"path": str(p), "sha256": sha256(p)}
                     for n, p in sumber_model.items()},
    }
    args.pred_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.pred_out, test_id=np.asarray(ids),
                        test_y_total=y.sum(1), **pred)
    hasil["prediksi_per_pohon"] = str(args.pred_out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
    print(json.dumps(hasil, indent=2, ensure_ascii=False), flush=True)
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
