"""Evaluasi end-to-end proposal -> linker -> kelas fisik DAMIMAS.

Berbeda dari evaluasi classifier strict, seluruh pool di sini dibentuk oleh
linker prediksi. Pool dipasangkan satu-ke-satu dengan tandan GT *sesudah*
inferensi; pool palsu menjadi false positive dan tandan tanpa pool menjadi
false negative. Identitas/kelas GT dalam cache tidak pernah menjadi fitur.

Urutan akses data dijaga ketat: seluruh sumber probabilitas, head linker,
ambang pool, aturan agregasi, skema bobot, dan tau ordinal dipilih di VAL.
Cache/prediksi/label TEST baru dibuka setelah konfigurasi final tercetak.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_recall_fscore_support


SUB = Path(__file__).resolve().parents[1]
ROOT = SUB.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))
import eval_pertandan as EP  # noqa: E402
import linker_global_damimas as LG  # noqa: E402
import propagasi_multiview_damimas as PM  # noqa: E402

KELAS = tuple(EP.KELAS)


def parse_sumber(items):
    out = {}
    for item in items:
        p = item.split("=")
        if len(p) != 3:
            raise ValueError("--sumber harus NAMA=VAL.npz=TEST.npz")
        nama, val, test = p
        out[nama] = {"val": Path(val), "test": Path(test)}
    return out


def score_head(g, bundle, head):
    return sum(w * bundle["models"][n].predict_proba(g["E"])[:, 1]
               for n, w in head["bobot_skor"].items())


def label_head(graphs, bundle, head):
    return [PM.rakit(g, score_head(g, bundle, head), head) for g in graphs]


def sumber_prob(local_path, phys_path, stems):
    """Peta ``(stem, indeks proposal)`` ke distribusi kelas deployable."""
    local = PM.ED.muat_prediksi(local_path, stems)
    phys = PM.load_full(phys_path, stems)
    _row, prob, _obj = PM.siapkan_lokal(local, phys)
    return prob, phys


def prob_deteksi(g, d, prob, phys):
    """Ganti p proposal cache dengan p sumber lokal melalui IoU eksak."""
    if prob is None:
        return np.asarray(d["p"], float)
    stem = g["P"]["sisi"][d["s"]]["stem"]
    D = phys.get(stem, np.zeros((0, 11), np.float32))
    if not len(D):
        return np.asarray(d["p"], float)
    ov = PM.iou(d["px"], D[:, :4]); j = int(np.argmax(ov))
    return np.asarray(prob.get((stem, j), d["p"]), float) if ov[j] >= .80 \
        else np.asarray(d["p"], float)


def grup_prediksi(graphs, labels, prob=None, phys=None):
    """Bangun pool prediksi tanpa membaca label untuk keputusan inferensi."""
    keluar = []
    for g, lab in zip(graphs, labels):
        grup = defaultdict(list)
        for i, d in enumerate(g["kotak"]):
            q = dict(d)
            q["p"] = prob_deteksi(g, d, prob, phys)
            q["p"] = q["p"] / max(float(q["p"].sum()), 1e-9)
            grup[int(lab[i])].append(q)
        for pid, anggota in grup.items():
            keluar.append({"tree": g["tree"], "pid": pid,
                            "pool": anggota,
                            "pool_conf": max(d["conf"] for d in anggota)})
    return keluar


def pasangkan_satu_ke_satu(ids, pred, ambang):
    """Pasangkan pool dan GT dengan Hungarian atas jumlah anggota cocok.

    ``bid`` hanya dibaca di sini, yaitu lapisan evaluator setelah pool sudah
    final. Satu pool tidak boleh mengklaim dua tandan dan satu tandan tidak
    boleh menerima dua pool.
    """
    by_tree = defaultdict(list)
    for q in pred:
        if q["pool_conf"] >= ambang:
            by_tree[q["tree"]].append(q)
    semua, total_gt, gt_per_tree = [], Counter(), {}
    for tree in ids:
        P = EP.muat_pohon(tree)
        bids = list(P["tandan"])
        gt_per_tree[tree] = Counter(P["tandan"].values())
        total_gt.update(P["tandan"].values())
        pools = by_tree.get(tree, [])
        pasangan = {}
        if pools and bids:
            W = np.zeros((len(pools), len(bids)), float)
            pos = {b: j for j, b in enumerate(bids)}
            for i, q in enumerate(pools):
                for d in q["pool"]:
                    if d.get("bid") in pos:
                        W[i, pos[d["bid"]]] += 1
            rr, cc = linear_sum_assignment(-W)
            pasangan = {int(i): bids[int(j)] for i, j in zip(rr, cc)
                        if W[i, j] > 0}
        for i, q in enumerate(pools):
            bid = pasangan.get(i)
            semua.append({**q, "gt": None if bid is None else P["tandan"][bid],
                           "bid_eval": bid})
    return semua, total_gt, gt_per_tree


def prediksi_pool(q, aturan, skema, tau):
    return EP.prediksi(q["pool"], aturan, skema, tau)


def metrik_fisik(records, total_gt, gt_per_tree, aturan, skema, tau):
    yhat = np.asarray([prediksi_pool(q, aturan, skema, tau) for q in records], int)
    matched = np.asarray([q["gt"] is not None for q in records], bool)
    y = np.asarray([q["gt"] if q["gt"] is not None else -1 for q in records], int)
    total = int(sum(total_gt.values()))
    tp = np.asarray([int(((yhat == k) & (y == k)).sum()) for k in range(4)])
    pred_n = np.asarray([(yhat == k).sum() for k in range(4)])
    gt_n = np.asarray([total_gt[k] for k in range(4)])
    fp, fn = pred_n - tp, gt_n - tp
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)

    if matched.any():
        pm, rm, fm, _ = precision_recall_fscore_support(
            y[matched], yhat[matched], labels=np.arange(4), zero_division=0)
        acc = float((y[matched] == yhat[matched]).mean())
        mae = float(np.abs(y[matched] - yhat[matched]).mean())
        macro_matched = float(fm.mean())
    else:
        pm = rm = fm = np.zeros(4); acc = mae = macro_matched = 0.

    subgrup = {}
    nview = np.asarray([len(q["pool"]) for q in records], int)
    for nama, mask in (("satu_tampak", matched & (nview == 1)),
                       ("multi_tampak", matched & (nview >= 2))):
        if mask.any():
            _p, _r, _f, _ = precision_recall_fscore_support(
                y[mask], yhat[mask], labels=np.arange(4), zero_division=0)
            subgrup[nama] = {
                "n": int(mask.sum()),
                "akurasi": float((y[mask] == yhat[mask]).mean()),
                "macro_f1": float(_f.mean()),
                "mae_ordinal": float(np.abs(y[mask] - yhat[mask]).mean()),
            }
        else:
            subgrup[nama] = {"n": 0, "akurasi": 0., "macro_f1": 0.,
                             "mae_ordinal": 0.}

    pred_tree = Counter(q["tree"] for q in records)
    mae_pool = float(np.mean([
        abs(pred_tree[t] - sum(gt_per_tree[t].values())) for t in gt_per_tree
    ]))
    out = {
        "n_gt": total, "n_pool_pred": len(records),
        "n_gt_terpasang": int(matched.sum()),
        "recall_pool_fisik": float(matched.sum() / max(total, 1)),
        "precision_pool_fisik": float(matched.sum() / max(len(records), 1)),
        "akurasi_kelas_pada_pool_terpasang": acc,
        "macro_f1_kelas_pada_pool_terpasang": macro_matched,
        "mae_ordinal_pada_pool_terpasang": mae,
        "correct_class_recall_endtoend": float(tp.sum() / max(total, 1)),
        "macro_f1_endtoend": float(f1.mean()),
        "mae_jumlah_pool_per_pohon": mae_pool,
        "menurut_jumlah_tampak": subgrup,
        "precision_per_kelas_endtoend": {KELAS[k]: float(precision[k]) for k in range(4)},
        "recall_per_kelas_endtoend": {KELAS[k]: float(recall[k]) for k in range(4)},
        "f1_per_kelas_endtoend": {KELAS[k]: float(f1[k]) for k in range(4)},
        "support_gt": {KELAS[k]: int(gt_n[k]) for k in range(4)},
        "matched_classifier": {
            "precision": {KELAS[k]: float(pm[k]) for k in range(4)},
            "recall": {KELAS[k]: float(rm[k]) for k in range(4)},
            "f1": {KELAS[k]: float(fm[k]) for k in range(4)},
        },
    }
    out["objective"] = (.45 * out["correct_class_recall_endtoend"]
                        + .25 * out["macro_f1_endtoend"]
                        + .15 * out["precision_pool_fisik"]
                        + .15 * out["akurasi_kelas_pada_pool_terpasang"])
    return out


def pools_matched(records):
    return [{"tree": q["tree"], "gt": q["gt"], "pool": q["pool"]}
            for q in records if q["gt"] is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumber", action="append", default=[])
    ap.add_argument("--phys-val", type=Path, default=ROOT / "results" /
                    "pred_damimas_proposal_yolo_val.npz")
    ap.add_argument("--phys-test", type=Path, default=ROOT / "results" /
                    "pred_damimas_proposal_yolo_test.npz")
    ap.add_argument("--cache-val", type=Path, default=SUB / "results" /
                    "cache_linker_damimas_damimas_damimas_proposal_yolo_val_val.joblib")
    ap.add_argument("--cache-test", type=Path, default=SUB / "results" /
                    "cache_linker_damimas_damimas_damimas_proposal_yolo_test_test.joblib")
    ap.add_argument("--linker-model", type=Path, default=SUB / "runs" /
                    "linker_global_damimas_proposal_yolo" / "model.joblib")
    ap.add_argument("--linker-config", type=Path, default=SUB / "results" /
                    "damimas_linker_global_proposal_yolo_lock.json")
    ap.add_argument("--output", type=Path, default=SUB / "results" /
                    "damimas_endtoend_global.json")
    args = ap.parse_args()
    sumber = parse_sumber(args.sumber) if args.sumber else {
        "routing": {
            "val": ROOT / "results" / "pred_damimas_fusi_yolo_relabel_val.npz",
            "test": ROOT / "results" / "pred_damimas_fusi_yolo_relabel_test.npz",
        },
        "propagasi": {
            "val": ROOT / "results" / "pred_damimas_propagasi_multiview_val.npz",
            "test": ROOT / "results" / "pred_damimas_propagasi_multiview_test.npz",
        },
    }

    # ----------------------------- hanya VAL -----------------------------
    cache_v = joblib.load(args.cache_val)
    graphs_v, ids_v = cache_v["graf"], cache_v["ids"]
    bundle = joblib.load(args.linker_model)
    heads = json.loads(args.linker_config.read_text())["heads"]
    labels_v = {nama: label_head(graphs_v, bundle, q)
                for nama, q in heads.items()}
    stems_v = {s["stem"] for g in graphs_v for s in g["P"]["sisi"]}
    bank_v = {"proposal": (None, None)}
    for nama, path in sumber.items():
        bank_v[nama] = sumber_prob(path["val"], args.phys_val, stems_v)

    ranking = []
    # Ambang berada pada objectness proposal yang sudah menjadi input linker.
    # 0,10 selalu disertakan sebagai kontrol persis cache.
    for head, lab in labels_v.items():
        for src, (prob, phys) in bank_v.items():
            pred = grup_prediksi(graphs_v, lab, prob, phys)
            for ambang in (.10, .15, .20, .25, .30, .40):
                records, total_gt, gt_tree = pasangkan_satu_ke_satu(
                    ids_v, pred, ambang)
                matched = pools_matched(records)
                kandidat = [("R1", "conf", (0.5, 1.5, 2.5)),
                             ("R2", "seragam", (0.5, 1.5, 2.5))]
                for skema in ("seragam", "conf", "luas", "conf_luas",
                               "conf_luas_tepi"):
                    kandidat.append(("R3", skema, (0.5, 1.5, 2.5)))
                    tau = EP.cari_tau(matched, skema) if matched else (0.5, 1.5, 2.5)
                    kandidat.append(("R4", skema, tau))
                for aturan, skema, tau in kandidat:
                    m = metrik_fisik(records, total_gt, gt_tree,
                                      aturan, skema, tau)
                    ranking.append({
                        "config": {"head": head, "sumber_prob": src,
                                   "ambang_pool": ambang, "aturan": aturan,
                                   "skema": skema, "tau": list(tau)},
                        "metrik": m, "objective": m["objective"],
                    })
        print(f"VAL head={head}: kandidat={len(ranking)} "
              f"best={max(x['objective'] for x in ranking):.6f}", flush=True)
    best = max(ranking, key=lambda x: x["objective"])
    lock = {"config": best["config"], "metrik": best["metrik"],
            "objective": best["objective"]}
    print("TERKUNCI DI VAL", json.dumps(lock, indent=2), flush=True)

    # ----------------------- TEST baru dibuka di sini --------------------
    cfg = best["config"]
    cache_t = joblib.load(args.cache_test)
    graphs_t, ids_t = cache_t["graf"], cache_t["ids"]
    labels_t = label_head(graphs_t, bundle, heads[cfg["head"]])
    if cfg["sumber_prob"] == "proposal":
        prob_t = phys_t = None
    else:
        stems_t = {s["stem"] for g in graphs_t for s in g["P"]["sisi"]}
        prob_t, phys_t = sumber_prob(
            sumber[cfg["sumber_prob"]]["test"], args.phys_test, stems_t)
    pred_t = grup_prediksi(graphs_t, labels_t, prob_t, phys_t)
    rec_t, gt_t, gt_tree_t = pasangkan_satu_ke_satu(
        ids_t, pred_t, cfg["ambang_pool"])
    mt = metrik_fisik(rec_t, gt_t, gt_tree_t, cfg["aturan"], cfg["skema"],
                       tuple(cfg["tau"]))

    hasil = {
        "dataset": "SawitMVC-YOLO-Damimas",
        "protokol": ("seluruh source/head/threshold/agregasi/tau dipilih VAL; "
                     "TEST dibuka sekali setelah lock"),
        "definisi": ("pool-GT Hungarian satu-ke-satu; pool tak terpasang=FP; "
                      "GT tak terpasang=FN"),
        "sumber_prob": {n: {s: str(p) for s, p in q.items()}
                         for n, q in sumber.items()},
        "terkunci_di_val": lock,
        "test": mt,
        "ranking_val": sorted(ranking, key=lambda x: x["objective"],
                              reverse=True)[:50],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
    print(json.dumps({"val": lock["metrik"], "test": mt}, indent=2,
                     ensure_ascii=False), flush=True)
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
