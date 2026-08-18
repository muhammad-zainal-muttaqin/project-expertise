"""Propagasi kelas lintas-view melalui linker proposal DAMIMAS.

Baris deteksi dan koordinat dari kepala class-aware dipertahankan. Setiap baris
dipetakan ke proposal fisik unik; proposal yang ditautkan pada tandan yang sama
menggabungkan probabilitas kelas, lalu hanya skornya yang diperbarui. Dengan
demikian konfigurasi kontrol identik bit-for-bit dan tidak ada objek baru.

Seluruh head linker, metode agregasi, kekuatan propagasi, dan rumus skor dipilih
di VAL. Cache/GT/prediksi TEST baru dibuka setelah konfigurasi terkunci.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "pipeline-pertandan"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(SUB / "scripts"))
import eval_dump_damimas as ED  # noqa: E402
import fusi_detektor_damimas as FD  # noqa: E402
import linker_global_damimas as LG  # noqa: E402


def iou(box, boxes):
    if not len(boxes):
        return np.zeros(0)
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    a = max(0, box[2]-box[0]) * max(0, box[3]-box[1])
    b = np.maximum(0, boxes[:, 2]-boxes[:, 0]) * np.maximum(0, boxes[:, 3]-boxes[:, 1])
    return inter / np.maximum(a+b-inter, 1e-12)


def load_full(path, stems):
    z = np.load(path, allow_pickle=True)
    out = {s: np.asarray(z[s], np.float32) if s in z.files
           else np.zeros((0, 11), np.float32) for s in stems}
    z.close(); return out


def rakit(g, score, q):
    if q["assembler"] == "hungarian":
        return LG.rakit_hungarian(g, score, q["ambang"], q["max_mode"])
    if q["assembler"] == "ilp":
        return LG.rakit_ilp(g, score, q["ambang"], q["metode"], q["max_mode"])
    return LG.rakit_aglom(g, score, q["ambang"], q["metode"], q["max_mode"])


def assignment(cache_path, bundle, head, phys):
    graphs = joblib.load(cache_path)["graf"]
    out = {}
    for g in graphs:
        score = sum(w * bundle["models"][n].predict_proba(g["E"])[:, 1]
                    for n, w in head["bobot_skor"].items())
        lab = rakit(g, score, head)
        for i, (d, k) in enumerate(zip(g["kotak"], lab)):
            stem = g["P"]["sisi"][d["s"]]["stem"]
            D = phys.get(stem, np.zeros((0, 11)))
            if not len(D):
                continue
            ov = iou(d["px"], D[:, :4]); j = int(np.argmax(ov))
            if ov[j] >= .80:
                out[(stem, j)] = f"{g['tree']}|{int(k)}"
    return out


def siapkan_lokal(local, phys):
    """Peta baris lokal -> proposal dan distribusi lokal per proposal."""
    rowmap, prob, obj = {}, {}, {}
    for stem, D in local.items():
        P = phys[stem]
        rm = np.full(len(D), -1, int)
        skor = np.zeros((len(P), 4), float)
        for i, r in enumerate(D):
            if not len(P):
                continue
            ov = iou(r[:4], P[:, :4]); j = int(np.argmax(ov))
            if ov[j] < .30:
                continue
            rm[i] = j; k = int(r[5])
            if 0 <= k < 4:
                skor[j, k] = max(skor[j, k], float(r[4]))
        rowmap[stem] = rm
        for j, p in enumerate(P):
            q = skor[j]
            if q.sum() <= 0:
                q = np.clip(p[6:10].astype(float), 1e-8, None)
            prob[(stem, j)] = q / max(q.sum(), 1e-9)
            obj[(stem, j)] = float(p[4])
    return rowmap, prob, obj


def agregat(prob, obj, assign, mode):
    grup = defaultdict(list)
    for key, cluster in assign.items():
        if key in prob:
            grup[cluster].append(key)
    out = {}
    for keys in grup.values():
        # Singleton sengaja tidak diubah; hanya bukti multi-view yang dihitung.
        if len(keys) < 2:
            continue
        P = np.stack([prob[k] for k in keys])
        w = np.asarray([max(obj[k], 1e-4) for k in keys])
        if mode == "mean":
            q = P.mean(0)
        elif mode == "conf":
            q = np.average(P, axis=0, weights=w)
        elif mode == "geom":
            q = np.exp(np.average(np.log(np.clip(P, 1e-8, 1)), axis=0,
                                  weights=w))
        else:
            # Robust mean: buang satu view paling tidak yakin jika >=3.
            keep = np.argsort(-P.max(1))[:max(2, len(P)-1)]
            q = np.average(P[keep], axis=0, weights=w[keep])
        q = q / max(q.sum(), 1e-9)
        for k in keys:
            out[k] = q
    return out


def prediksi(local, phys, rowmap, prob, obj, assign, cfg):
    if cfg["mode"] == "kontrol":
        return {s: d.copy() for s, d in local.items()}
    agg = agregat(prob, obj, assign, cfg["agregasi"])
    out = {}
    for stem, D in local.items():
        q = D.copy(); rm = rowmap[stem]
        for i, r in enumerate(q):
            j = int(rm[i])
            if j < 0 or (stem, j) not in agg:
                continue
            key = (stem, j); p0 = prob[key]; pg = agg[key]
            p = (1-cfg["alpha"]) * p0 + cfg["alpha"] * pg
            k = int(r[5])
            target = (max(obj[key], 1e-8) ** cfg["loc_power"] *
                      max(float(p[k]), 1e-8) ** cfg["gamma"])
            r[4] = ((1-cfg["score_blend"]) * float(r[4]) +
                    cfg["score_blend"] * target)
        out[stem] = q
    return out


def prediksi_route(local, phys, rowmap, prob, obj, assign_by_head,
                   per_kelas):
    """Rakit skor dari konfigurasi berbeda untuk setiap kelas COCO.

    AP setiap kategori dihitung independen. Karena koordinat, label, dan jumlah
    baris tidak berubah, mengambil konfigurasi terbaik per kategori di VAL
    mendominasi satu konfigurasi global tanpa mencampur evaluasi antar-kelas.
    """
    out = {s: d.copy() for s, d in local.items()}
    cache = {}
    for k, cfg in enumerate(per_kelas):
        key = nama(cfg)
        if key not in cache:
            cache[key] = prediksi(
                local, phys, rowmap, prob, obj,
                assign_by_head[cfg["head"]], cfg)
        for stem, d in out.items():
            m = d[:, 5].astype(int) == k
            d[m, 4] = cache[key][stem][m, 4]
    return out


def objektif_kelas(m, kelas):
    return (.65 * m["AP50_per_kelas"][kelas]
            + .35 * m["AP50_95_per_kelas"][kelas])


def nama(cfg):
    if cfg["mode"] == "kontrol":
        return "kontrol"
    return (f"{cfg['head']}_{cfg['agregasi']}_a{cfg['alpha']:g}_"
            f"g{cfg['gamma']:g}_lp{cfg['loc_power']:g}_sb{cfg['score_blend']:g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-val", type=Path, default=ROOT / "results" /
                    "pred_damimas_fusi_yolo_relabel_val.npz")
    ap.add_argument("--local-test", type=Path, default=ROOT / "results" /
                    "pred_damimas_fusi_yolo_relabel_test.npz")
    ap.add_argument("--phys-val", type=Path, default=ROOT / "results" /
                    "pred_damimas_proposal_yolo_val.npz")
    ap.add_argument("--phys-test", type=Path, default=ROOT / "results" /
                    "pred_damimas_proposal_yolo_test.npz")
    ap.add_argument("--linker-model", type=Path, default=SUB / "runs" /
                    "linker_global_damimas_proposal_yolo" / "model.joblib")
    ap.add_argument("--linker-config", type=Path, default=SUB / "results" /
                    "damimas_linker_global_proposal_yolo_lock.json")
    ap.add_argument("--cache-val", type=Path, default=SUB / "results" /
                    "cache_linker_damimas_damimas_damimas_proposal_yolo_val_val.joblib")
    ap.add_argument("--cache-test", type=Path, default=SUB / "results" /
                    "cache_linker_damimas_damimas_damimas_proposal_yolo_test_test.joblib")
    ap.add_argument("--dataset", type=Path,
                    default=Path("/workspace/SawitMVC-YOLO-Damimas"))
    ap.add_argument("--output", type=Path, default=ROOT / "results" /
                    "damimas_propagasi_multiview.json")
    ap.add_argument("--pred-val-out", type=Path, default=ROOT / "results" /
                    "pred_damimas_propagasi_multiview_val.npz")
    ap.add_argument("--pred-test-out", type=Path, default=ROOT / "results" /
                    "pred_damimas_propagasi_multiview_test.npz")
    args = ap.parse_args()

    # Tahap VAL. Berkas TEST belum dibuka.
    coco_v, paths_v, gt_v = ED.bangun_gt(args.dataset, "val")
    stems_v = set(gt_v)
    local_v = ED.muat_prediksi(args.local_val, stems_v)
    phys_v = load_full(args.phys_val, stems_v)
    bundle = joblib.load(args.linker_model)
    heads = json.loads(args.linker_config.read_text())["heads"]
    assign_v = {n: assignment(args.cache_val, bundle, q, phys_v)
                for n, q in heads.items() if n in ("utility", "cakupan_atas_terdeteksi")}
    row_v, prob_v, obj_v = siapkan_lokal(local_v, phys_v)

    configs = [{"mode": "kontrol", "head": "utility", "agregasi": "mean",
                "alpha": 0., "gamma": 1., "loc_power": 1., "score_blend": 0.}]
    for head in assign_v:
        for agg in ("mean", "conf", "geom", "robust"):
            for alpha in (.25, .50, .75, 1.):
                for blend in (.25, .50, .75):
                    configs.append({"mode": "propagasi", "head": head,
                                    "agregasi": agg, "alpha": alpha,
                                    "gamma": 2., "loc_power": 1.25,
                                    "score_blend": blend})
    ranking = []
    for i, cfg in enumerate(configs, 1):
        pv = prediksi(local_v, phys_v, row_v, prob_v, obj_v,
                      assign_v[cfg["head"]], cfg)
        m = FD.coco_detail(coco_v, paths_v, pv); o = FD.objektif(m)
        ranking.append({"config": cfg, "metrik": m, "objective": o})
        if i % 25 == 0:
            print(f"{i}/{len(configs)} best={max(x['objective'] for x in ranking):.6f}",
                  flush=True)
    best = max(ranking, key=lambda x: x["objective"])
    cfg = dict(best["config"]); obj_best = best["objective"]
    met_best = best["metrik"]
    # Coordinate refinement kontinu di VAL.
    ruang = {"alpha": (.1, .25, .4, .5, .6, .75, .9, 1.),
             "gamma": (.5, 1., 1.5, 2., 2.5, 3.),
             "loc_power": (.5, .75, 1., 1.25, 1.5),
             "score_blend": (.1, .25, .4, .5, .6, .75, .9),
             "agregasi": ("mean", "conf", "geom", "robust"),
             "head": tuple(assign_v)}
    if cfg["mode"] != "kontrol":
        for field, vals in ruang.items():
            # Pertahankan metrik dari konfigurasi terkini bila koordinat ini
            # tidak memberi perbaikan. Mengacu kembali ke ``best`` awal akan
            # membuat laporan VAL tidak sesuai dengan konfigurasi final.
            pilih = (obj_best, cfg, met_best)
            for value in vals:
                q = dict(cfg); q[field] = value
                pv = prediksi(local_v, phys_v, row_v, prob_v, obj_v,
                              assign_v[q["head"]], q)
                m = FD.coco_detail(coco_v, paths_v, pv); o = FD.objektif(m)
                if o > pilih[0] + 1e-12:
                    pilih = (o, q, m)
            obj_best, cfg, met_best = pilih
            print(f"refine {field}: {obj_best:.6f} {nama(cfg)}", flush=True)
    global_best = {"config": cfg, "metrik": met_best,
                   "objective": obj_best}

    # AP COCO bersifat separabel per kelas. Mulai dari seluruh grid ditambah
    # optimum global yang telah direfinement, lalu lakukan coordinate search
    # per kelas. Semua langkah ini masih hanya membuka VAL.
    seeds = ranking + [global_best]
    per_kelas = []
    jejak_kelas = {}
    for kelas in ED.NAMA:
        awal = max(seeds, key=lambda x: objektif_kelas(x["metrik"], kelas))
        qcfg = dict(awal["config"])
        qmet = awal["metrik"]
        qobj = objektif_kelas(qmet, kelas)
        jejak = [{"tahap": "seed", "config": qcfg,
                  "objective": qobj, "metrik": qmet}]
        if qcfg["mode"] != "kontrol":
            for field, vals in ruang.items():
                pilih = (qobj, qcfg, qmet)
                for value in vals:
                    cand = dict(qcfg); cand[field] = value
                    pv = prediksi(local_v, phys_v, row_v, prob_v, obj_v,
                                  assign_v[cand["head"]], cand)
                    mm = FD.coco_detail(coco_v, paths_v, pv)
                    oo = objektif_kelas(mm, kelas)
                    if oo > pilih[0] + 1e-12:
                        pilih = (oo, cand, mm)
                qobj, qcfg, qmet = pilih
                jejak.append({"tahap": field, "config": qcfg,
                              "objective": qobj, "metrik": qmet})
        per_kelas.append(qcfg)
        jejak_kelas[kelas] = jejak
        print(f"route {kelas}: {qobj:.6f} {nama(qcfg)}", flush=True)

    pred_val = prediksi_route(local_v, phys_v, row_v, prob_v, obj_v,
                              assign_v, per_kelas)
    met_route = FD.coco_detail(coco_v, paths_v, pred_val)
    obj_route = FD.objektif(met_route)
    if obj_route + 1e-12 < obj_best:
        raise RuntimeError(
            f"Router per-kelas merusak objective VAL: {obj_route} < {obj_best}")
    lock = {"config": {"mode": "route_per_kelas",
                        "per_kelas": dict(zip(ED.NAMA, per_kelas))},
            "metrik": met_route, "objective": obj_route,
            "global_best": global_best,
            "jejak_per_kelas": jejak_kelas}
    print("TERKUNCI DI VAL", json.dumps(lock, indent=2), flush=True)

    # TEST baru dibuka setelah seluruh konfigurasi tetap.
    coco_t, paths_t, gt_t = ED.bangun_gt(args.dataset, "test")
    stems_t = set(gt_t)
    local_t = ED.muat_prediksi(args.local_test, stems_t)
    phys_t = load_full(args.phys_test, stems_t)
    row_t, prob_t, obj_t = siapkan_lokal(local_t, phys_t)
    kepala_test = {q["head"] for q in per_kelas}
    assign_t = {h: assignment(args.cache_test, bundle, heads[h], phys_t)
                for h in kepala_test}
    pred_test = prediksi_route(local_t, phys_t, row_t, prob_t, obj_t,
                               assign_t, per_kelas)
    mt = FD.coco_detail(coco_t, paths_t, pred_test)
    amb_info = ED.pilih_ambang(gt_v, pred_val)
    amb = {n: amb_info[n]["ambang"] for n in ED.NAMA}
    operasi = {"val": ED.nilai_ambang(gt_v, pred_val, amb),
               "test": ED.nilai_ambang(gt_t, pred_test, amb)}
    np.savez_compressed(args.pred_val_out, **pred_val)
    np.savez_compressed(args.pred_test_out, **pred_test)
    hasil = {"dataset": "SawitMVC-YOLO-Damimas",
             "protokol": "seluruh propagasi dipilih VAL; TEST sekali",
             "input": {"local_val": str(args.local_val),
                       "local_test": str(args.local_test),
                       "phys_val": str(args.phys_val),
                       "phys_test": str(args.phys_test)},
             "terkunci_di_val": lock, "test": mt,
             "ambang_dipilih_di_val": amb_info, "titik_operasi": operasi,
             "ranking_val": sorted(ranking, key=lambda x: x["objective"],
                                   reverse=True)[:30],
             "prediksi": {"val": str(args.pred_val_out),
                           "test": str(args.pred_test_out)}}
    args.output.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
    print(json.dumps({"val": met_route, "test": mt,
                      "operasi_test": operasi["test"]}, indent=2), flush=True)
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
